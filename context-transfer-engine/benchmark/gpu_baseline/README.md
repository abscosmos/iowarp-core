# gpu_baseline Benchmarks

## Build

```bash
cmake -B build -DWRP_CORE_ENABLE_CUDA=ON -DWRP_CORE_ENABLE_BENCHMARKS=ON
cmake --build build --target gpu_baseline_pinned_host
```

libnuma is auto-detected. If present (`apt-get install libnuma-dev`), NUMA-node pinning
is enabled automatically.

## Benchmark 1: Pinned Host Memory (`gpu_baseline_pinned_host`)

Measures GPU-to-host (write) and host-to-GPU (read) PCIe bandwidth to pinned memory.
NUMA node selection lets you test DRAM bank locality on multi-GPU/multi-NUMA machines.

### Flags

| Flag | Description |
|------|-------------|
| `--blocks N` / `--threads N` | Grid dims; threads must be a multiple of 32 |
| `--bytes-per-warp N` | Bytes per warp; k/m/g suffix accepted (e.g. `1m`); multiple of 16 |
| `--gpu-id N` | CUDA device ID (default: 0) |
| `--numa-node N` | NUMA node for pinned allocation; requires libnuma; no-op on single-NUMA machines |
| `--mode write\|read` | Transfer direction (default: `write`) |
| `--iterations N` | Timed iterations (default: 100) |
| `--warmup N` | Warmup iterations (default: 5) |
| `--validate` | Write mode only: reads back buffer and checks pattern |
| `--no-fence` | Skip `__threadfence_system()` for comparison |

### Working Modes

- `write` (GPU → pinned host): ~25 GB/s on RTX 4090
- `read` (pinned host → GPU): ~20 GB/s on RTX 4090

### Known Limitation

`--mode readwrite` is implemented but shows ~0.3 GB/s due to PCIe load-store
serialization (each store waits for its load to complete before the write is issued).
Use `write` and `read` for bandwidth numbers.

### Example Runs

Sanity check:
```bash
gpu_baseline_pinned_host --blocks 1 --threads 32 --bytes-per-warp 4096 --iterations 10
```

Full benchmark:
```bash
gpu_baseline_pinned_host --blocks 4 --threads 256 --bytes-per-warp 1m --iterations 50 --mode write
gpu_baseline_pinned_host --blocks 4 --threads 256 --bytes-per-warp 1m --iterations 50 --mode read
```

NUMA sweep (Delta — 4 GPUs, 4 NUMA nodes):
```bash
for gpu in 0 1 2 3; do
  for numa in 0 1 2 3; do
    gpu_baseline_pinned_host --gpu-id $gpu --numa-node $numa \
      --blocks 8 --threads 256 --bytes-per-warp 1m --iterations 100 --mode write
  done
done
```

## Benchmark 2: Remote HBM (`gpu_baseline_remote_hbm`)

Measures GPU-to-GPU HBM bandwidth over NVLink/PCIe on a single node using NVSHMEM RMA
operations. Warp-collective `nvshmemx_{put,get}mem_nbi_warp` calls drive the transfer;
only PE 0 runs the kernel, PE 1 acts as the target.

### Build

Requires NVSHMEM (already installed in the devcontainer; `NVSHMEM_HOME` is set automatically).

```bash
cmake -B build \
  -DWRP_CORE_ENABLE_CUDA=ON \
  -DWRP_CORE_ENABLE_BENCHMARKS=ON \
  -DWRP_CORE_ENABLE_NVSHMEM=ON
cmake --build build --target gpu_baseline_remote_hbm
```

### Flags

| Flag | Description |
|------|-------------|
| `--blocks N` / `--threads N` | Grid dims; threads must be a multiple of 32 |
| `--bytes-per-warp N` | Bytes per warp; k/m/g suffix accepted; multiple of 16 |
| `--mode put\|get\|ping-pong` | Transfer direction (default: `put`) |
| `--src-numa N` / `--dst-numa N` | Informational: NUMA node of each GPU |
| `--warmup N` / `--iterations N` | Warmup and timed iteration counts |
| `--no-fence` | Skip per-iteration `nvshmem_quiet()` (pipeline mode) |
| `--validate` | Write pattern, quiet, read back and check |

### Example Runs

Loopback sanity (single GPU):
```bash
mpirun -n 1 gpu_baseline_remote_hbm \
  --blocks 1 --threads 32 --bytes-per-warp 4096 --iterations 5
```

Two GPUs, same node:
```bash
mpirun -n 2 gpu_baseline_remote_hbm \
  --blocks 4 --threads 256 --bytes-per-warp 1m --iterations 50 --mode put
mpirun -n 2 gpu_baseline_remote_hbm \
  --blocks 4 --threads 256 --bytes-per-warp 1m --iterations 50 --mode get
mpirun -n 2 gpu_baseline_remote_hbm \
  --blocks 4 --threads 256 --bytes-per-warp 1m --iterations 50 --mode ping-pong
```

NUMA sweep (4 GPUs, vary src/dst for documentation):
```bash
for src in 0 1 2 3; do
  for dst in 0 1 2 3; do
    mpirun -n 2 gpu_baseline_remote_hbm \
      --src-numa $src --dst-numa $dst \
      --blocks 8 --threads 256 --bytes-per-warp 1m --iterations 100 --mode put
  done
done
```

For large allocations (>512 MB total), set:
```bash
export NVSHMEM_SYMMETRIC_SIZE=<bytes>
```

## Benchmark 3: Bus Contention (`gpu_baseline_bus_contention`)

Measures PCIe/NVLink bus contention by running the pinned-host write (bench 1) and an
NVSHMEM HBM-to-HBM put (bench 2) simultaneously on separate GPUs. Both traffic flows
converge on PE 0's PCIe lanes. The benchmark runs solo baselines first, then the
contention phase, and reports bandwidth degradation for each workload.

Expected result on a multi-GPU node: each workload drops to roughly 50% of its solo
bandwidth when the bus saturates.

### Build

Built alongside bench 2 when NVSHMEM is enabled — no additional CMake flags needed
beyond what bench 2 already requires:

```bash
cmake -B build \
  -DWRP_CORE_ENABLE_CUDA=ON \
  -DWRP_CORE_ENABLE_BENCHMARKS=ON \
  -DWRP_CORE_ENABLE_NVSHMEM=ON
cmake --build build --target gpu_baseline_bus_contention
```

### Flags

| Flag | Description |
|------|-------------|
| `--A-blocks N` / `--A-threads N` / `--A-bytes-per-warp N` | Pinned-host write workload (PE 0); threads multiple of 32, bytes multiple of 16, k/m/g suffix accepted |
| `--B-blocks N` / `--B-threads N` / `--B-bytes-per-warp N` | NVSHMEM put workload (PE 1); same constraints as A |
| `--numa-node N` | NUMA node for PE 0's pinned host allocation (default: 0) |
| `--gpu-id N` | Override CUDA device ID (default: auto = `mype % num_gpus`) |
| `--warmup N` / `--iterations N` | Warmup and timed iteration counts |
| `--no-fence` | Disable fencing for both workloads |

### Example Runs

Loopback sanity (1 GPU — runs solo baselines only, skips contention phase):
```bash
mpirun -n 1 gpu_baseline_bus_contention \
  --A-blocks 1 --A-threads 32 --A-bytes-per-warp 4096 \
  --B-blocks 1 --B-threads 32 --B-bytes-per-warp 4096 \
  --warmup 2 --iterations 5
```

Two GPUs — full contention measurement:
```bash
mpirun -n 2 gpu_baseline_bus_contention \
  --A-blocks 4 --A-threads 256 --A-bytes-per-warp 1m \
  --B-blocks 4 --B-threads 256 --B-bytes-per-warp 1m \
  --iterations 50
```

### Example Output

```
Phase 1: Solo baselines
Workload          Warps   bytes/warp   Iters     Time(ms)   BW(GB/s)
----------------  -----   ----------  ------   ----------  ----------
A (pinned)         ...
B (nvshmem)        ...

Phase 2: Contention (both simultaneous)
Workload           Solo BW   Contention BW  Degradation
----------------  ----------  --------------  ------------
pinned-write       25.4 GB/s      12.8 GB/s      -49.6%
remote-hbm-put     18.2 GB/s       9.1 GB/s      -50.0%

Contention factor: A dropped 49.6%, B dropped 50.0%
```

The contention phase is automatically skipped when only one PE is present — requires
`mpirun -n 2` (or `nvshmrun -n 2`) for contention measurement.

For large transfer sizes (>512 MB), set:
```bash
export NVSHMEM_SYMMETRIC_SIZE=<bytes>
```

## Benchmark 4: Cross-Node HBM (`gpu_baseline_cross_node_hbm`)

Measures cross-node GPU-to-GPU HBM **bandwidth** (put/get) and **one-way send-notify latency**
over InfiniBand using kernel-initiated NVSHMEM RMA. PE 0 is the initiator; PE 1 is the target.
The binary is transport-agnostic — select IB via environment variables at launch time.

### Build

Requires NVSHMEM (same flags as Bench 2):

```bash
cmake -B build \
  -DWRP_CORE_ENABLE_CUDA=ON \
  -DWRP_CORE_ENABLE_BENCHMARKS=ON \
  -DWRP_CORE_ENABLE_NVSHMEM=ON
cmake --build build --target gpu_baseline_cross_node_hbm
```

### Flags

| Flag | Description |
|------|-------------|
| `--blocks N` / `--threads N` | Grid dims; threads must be a multiple of 32 |
| `--bytes-per-warp N` | Transfer size per warp; k/m/g suffix; multiple of 16 (default: 4096 for BW modes, 64 for latency mode) |
| `--mode put-bw\|get-bw\|latency` | Measurement mode (default: `put-bw`) |
| `--warmup N` / `--iterations N` | Warmup and timed iterations for BW modes |
| `--latency-iters N` | Number of timed latency samples (default: 1000) |
| `--gpu-id N` | CUDA device ID per node (default: 0) |
| `--src-numa N` / `--dst-numa N` | Informational: NUMA node of each GPU |
| `--no-fence` | Disable per-iteration `nvshmem_quiet()` in BW modes (pipeline mode) |
| `--validate` | BW modes only: readback and verify data |

### InfiniBand Transport Setup

```bash
export NVSHMEM_BOOTSTRAP=MPI
export NVSHMEM_REMOTE_TRANSPORT=ibrc   # or: ucx
```

No code change needed; the binary is transport-agnostic.

### Example Runs

Loopback sanity (single node):
```bash
mpirun -n 1 gpu_baseline_cross_node_hbm \
  --blocks 1 --threads 32 --bytes-per-warp 4096 --iterations 5
```

Cross-node bandwidth (2 nodes, 1 GPU each):
```bash
mpirun -n 2 --hostfile hosts gpu_baseline_cross_node_hbm \
  --blocks 4 --threads 256 --bytes-per-warp 1m --iterations 50 --mode put-bw
mpirun -n 2 --hostfile hosts gpu_baseline_cross_node_hbm \
  --blocks 4 --threads 256 --bytes-per-warp 1m --iterations 50 --mode get-bw
```

Cross-node one-way latency:
```bash
mpirun -n 2 --hostfile hosts gpu_baseline_cross_node_hbm \
  --mode latency --bytes-per-warp 64 --latency-iters 1000
mpirun -n 2 --hostfile hosts gpu_baseline_cross_node_hbm \
  --mode latency --bytes-per-warp 4096 --latency-iters 1000
```

**On Delta (Slurm):** use `srun` instead of `mpirun --hostfile`. Request 2 nodes, 1 GPU
task each, and set the IB transport:
```bash
export NVSHMEM_REMOTE_TRANSPORT=ibrc   # or: ucx
srun --ntasks=2 --nodes=2 --ntasks-per-node=1 --gpus-per-task=1 \
  gpu_baseline_cross_node_hbm \
  --blocks 4 --threads 256 --bytes-per-warp 1m --iterations 50 --mode put-bw
```

### Example Output

BW mode:
```
=== GPU Baseline Benchmark 4: Cross-Node HBM ===
Mode          PE  Target   Warps   bytes/warp   Iters   Time(ms)   BW(GB/s)
------------  ----  ------  ------  ------------  ------  ----------  ----------
put-bw           0       1      32       1048576      50      ...       12.3
```

Latency mode (printed by PE 1 — or PE 0 in loopback):
```
Cross-node one-way latency (send-notify, 64 B payload)
  Samples : 1000
  Min     :  2.41 us
  Median  :  2.53 us
  Mean    :  2.61 us
  Max     :  3.12 us
```

> **Note (loopback):** When run with a single PE (`mpirun -n 1`), the send and poll kernels
> execute sequentially on the same stream, so the reported latency reflects device-local
> NVSHMEM overhead (~0.1 µs), not real InfiniBand round-trip time. Use `mpirun -n 2`
> across two nodes for meaningful cross-node latency numbers.

For large transfer sizes (>512 MB total), set:
```bash
export NVSHMEM_SYMMETRIC_SIZE=<bytes>
```

## CTest

Both benchmarks have ctests:
```bash
ctest --test-dir build -R gpu_baseline -V
```

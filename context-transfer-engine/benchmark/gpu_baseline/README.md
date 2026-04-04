# gpu_baseline Benchmarks

## Build

```bash
cmake -B build -DWRP_CORE_ENABLE_CUDA=ON -DWRP_CORE_ENABLE_BENCHMARKS=ON
cmake --build build --target gpu_baseline_pinned_host
```

libnuma is auto-detected. If present (`apt-get install libnuma-dev`), NUMA-node pinning
is enabled automatically.

### Building on DeltaAI

DeltaAI requires loading the CUDA and NVSHMEM modules before configuring. The exact module
names vary — check with `module avail nvshmem` and `module avail cuda`. A typical setup:

```bash
module load cuda/12.6
module load nvshmem        # sets NVSHMEM_HOME automatically, or check: module show nvshmem

cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DWRP_CORE_ENABLE_CUDA=ON \
  -DWRP_CORE_ENABLE_BENCHMARKS=ON \
  -DWRP_CORE_ENABLE_NVSHMEM=ON \
  -DNVSHMEM_HOME=$NVSHMEM_HOME
cmake --build build -j$(nproc) --target \
  gpu_baseline_pinned_host \
  gpu_baseline_remote_hbm \
  gpu_baseline_bus_contention \
  gpu_baseline_cross_node_hbm \
  gpu_baseline_cross_node_pinned_host
```

The build output lands in `build/bin/` on the shared Lustre filesystem and is accessible
from all nodes in your allocation.

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

### Transport Selection

The binary uses NVSHMEM's MPI bootstrap (`nvshmemx_init_attr` with `NVSHMEMX_INIT_WITH_MPI_COMM`)
so no `NVSHMEM_BOOTSTRAP` env var is needed. Set only the network transport:

```bash
export NVSHMEM_REMOTE_TRANSPORT=ibrc   # InfiniBand RC — standard HPC clusters
# or: libfabric                        # HPE Slingshot (DeltaAI) — use this instead of ibrc
# or: ucx                              # UCX backend
```

> **DeltaAI note:** DeltaAI's inter-node fabric is HPE Slingshot, not InfiniBand RC.
> Use `NVSHMEM_REMOTE_TRANSPORT=libfabric` (or `ucx` if that's how your NVSHMEM module
> was built — check with `module show nvshmem` to see which transport backends are included).
> Using `ibrc` on Slingshot will cause NVSHMEM init to fail or silently fall back to loopback.

For large transfers (>512 MB total symmetric allocation):
```bash
export NVSHMEM_SYMMETRIC_SIZE=2147483648   # 2 GB example
```

### Loopback sanity (single node, no IB needed)

Verifies the binary runs correctly. Results reflect device-local NVSHMEM overhead, not
real IB bandwidth or latency.

```bash
mpirun -n 1 gpu_baseline_cross_node_hbm \
  --blocks 1 --threads 32 --bytes-per-warp 4096 --iterations 5 --mode put-bw
```

### Cross-Node Runs — Generic (`mpirun --hostfile`)

Each process runs on a separate node and uses one GPU (PE 0 = node 0, PE 1 = node 1).
The binary must be in a path accessible from both nodes (shared filesystem).

**Bandwidth:**
```bash
export NVSHMEM_REMOTE_TRANSPORT=ibrc
mpirun -n 2 --hostfile hosts gpu_baseline_cross_node_hbm \
  --blocks 4 --threads 256 --bytes-per-warp 1m --iterations 50 --mode put-bw
mpirun -n 2 --hostfile hosts gpu_baseline_cross_node_hbm \
  --blocks 4 --threads 256 --bytes-per-warp 1m --iterations 50 --mode get-bw
```

**Latency (sweep across message sizes to characterize IB latency curve):**
```bash
export NVSHMEM_REMOTE_TRANSPORT=ibrc
for size in 64 256 1k 4k 16k 64k 256k 1m; do
  mpirun -n 2 --hostfile hosts gpu_baseline_cross_node_hbm \
    --mode latency --bytes-per-warp $size --latency-iters 1000
done
```

### Cross-Node Runs — Delta (Slurm / `srun`)

Delta uses Slurm; use `srun` instead of `mpirun --hostfile`. Request 2 nodes with 1 GPU
task per node. The binary is in your build's `bin/` directory on the shared Lustre filesystem.

**Interactive / one-off run:**
```bash
export NVSHMEM_REMOTE_TRANSPORT=ibrc
srun --ntasks=2 --nodes=2 --ntasks-per-node=1 --gpus-per-task=1 \
  /path/to/build/bin/gpu_baseline_cross_node_hbm \
  --blocks 4 --threads 256 --bytes-per-warp 1m --iterations 50 --mode put-bw
```

**Batch script (`bench4.sb`):**
```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuGH200x4       # DeltaAI GH200 partition; check: sinfo -s
#SBATCH --time=00:10:00
#SBATCH --account=<your_account>

# DeltaAI uses HPE Slingshot — use libfabric transport, not ibrc
export NVSHMEM_REMOTE_TRANSPORT=libfabric
# If your NVSHMEM module was built with UCX instead: export NVSHMEM_REMOTE_TRANSPORT=ucx
BIN=/path/to/build/bin/gpu_baseline_cross_node_hbm

# Bandwidth
srun $BIN --blocks 4 --threads 256 --bytes-per-warp 1m --iterations 100 --mode put-bw
srun $BIN --blocks 4 --threads 256 --bytes-per-warp 1m --iterations 100 --mode get-bw

# Latency sweep
for size in 64 256 1k 4k 16k 64k 256k 1m; do
  srun $BIN --mode latency --bytes-per-warp $size --latency-iters 1000
done
```

Submit with `sbatch bench4.sb`. Output from PE 1 (latency) and PE 0 (bandwidth) will be
interleaved in the Slurm output file.

**Note on GPU selection on Delta:** Delta's A100 nodes have 4 GPUs. With `--ntasks-per-node=1
--gpus-per-task=1`, Slurm assigns GPU 0 to each task. To test a specific GPU, add
`--gpu-id N` to the benchmark flags, but note that `--gpu-id` must be set BEFORE NVSHMEM
init (which happens at startup) — the default (GPU 0) is correct for single-GPU-per-task jobs.

### Example Output

BW mode (PE 0 prints):
```
=== GPU Baseline Benchmark 4: Cross-node HBM (NVSHMEM) — BW mode ===
Device  : NVIDIA A100-SXM4-40GB (PE 0 -> GPU 0)
Target  : PE 1
PEs     : 2 total
...
Mode          PE  Target   Warps   bytes/warp   Iters   Time(ms)   BW(GB/s)
------------  ----  ------  ------  ------------  ------  ----------  ----------
put-bw           0       1     128       1048576     100      ...       ~12.5
```

Expected IB HDR bandwidth: ~12–15 GB/s unidirectional (200 Gb/s link, ~50% efficiency
for kernel-initiated RDMA). Compare with intra-node NVLink (bench 2): ~300–600 GB/s.

Latency mode (PE 1 prints):
```
Cross-node one-way latency (send-notify, 64 B payload)
  Samples : 1000
  Min     :  2.41 us
  Median  :  2.53 us
  Mean    :  2.61 us
  Max     :  3.12 us
```

Expected IB HDR one-way latency: ~2–4 µs for small messages (64–256 B).

> **Note (loopback):** When run with a single PE (`mpirun -n 1`), the send and poll kernels
> execute sequentially on the same stream, so the reported latency is ~0.1 µs (device-local
> NVSHMEM overhead), not real IB latency. Two nodes required for meaningful numbers.

## Benchmark 5: Cross-Node Remote Pinned Host Memory (`gpu_baseline_cross_node_pinned_host`)

Measures cross-node **EGM (Extended GPU Memory) bandwidth** (write/read) and **one-way
send-notify latency** to a remote node's NUMA-pinned host DRAM via NVLink fabric handles.

Each rank allocates host DRAM using `cuMemCreate` with `CU_MEM_LOCATION_TYPE_HOST_NUMA` +
`CU_MEM_HANDLE_TYPE_FABRIC`, exports the handle as a `CUmemFabricHandle`, exchanges it
with the partner rank over MPI, and maps the remote buffer into local GPU VA space via
`cuMemImportFromShareableHandle`. The GPU then issues direct `uint4` stores/loads to the
remote pinned host — no NVSHMEM, no IB verbs required.

The binary performs a runtime check for `CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED`
and exits with a clear error message if the GPU lacks fabric handle support.

### Build

No NVSHMEM required. Only MPI and the CUDA Driver API (`CUDA::cuda_driver`) are needed.

```bash
cmake -B build \
  -DWRP_CORE_ENABLE_CUDA=ON \
  -DWRP_CORE_ENABLE_BENCHMARKS=ON
cmake --build build --target gpu_baseline_cross_node_pinned_host
```

### Flags

| Flag | Description |
|------|-------------|
| `--blocks N` / `--threads N` | Grid dims; threads must be a multiple of 32 |
| `--bytes-per-warp N` | Transfer size per warp; k/m/g suffix; multiple of 16 (default: 4096 for BW modes, 64 for latency) |
| `--mode write-bw\|read-bw\|latency` | Measurement mode (default: `write-bw`) |
| `--warmup N` / `--iterations N` | Warmup and timed iterations for BW modes |
| `--latency-iters N` | Number of timed latency samples (default: 1000) |
| `--src-gpu-id N` | GPU device ID on the source node (default: 0) |
| `--src-numa-node N` | NUMA node for source EGM allocation (default: 0) |
| `--dst-gpu-id N` | GPU device ID on the destination node (default: 0) |
| `--dst-numa-node N` | NUMA node for destination EGM allocation (default: 0) |
| `--no-fence` | Skip `__threadfence_system()` in write kernel (may show inflated BW) |

### Hardware Requirements

- NVIDIA GH200 (NVLink-C2C) or GB200 NVL72 (NVSwitch fabric) — these expose fabric handle support
- `CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED` must be `1`; the binary checks at runtime

> **DeltaAI note:** DeltaAI quad-GH200 nodes connect their four Grace Hopper superchips via
> intra-node NVLink-C2C. The inter-node fabric is HPE Slingshot, **not** NVLink. As a result,
> `CU_MEM_HANDLE_TYPE_FABRIC` works within a single quad-GH200 node (cross-socket, different
> NUMA nodes) but **cross-node runs (2 separate DeltaAI nodes) will likely fail** because
> Slingshot does not provide NVLink fabric handles. The interesting and supported measurement
> on DeltaAI is **intra-node cross-NUMA**: two MPI ranks on the same node, each pinned to a
> different socket, with `--nodes=1 --ntasks=2`. Cross-node NVLink fabric requires hardware
> with inter-node NVSwitch (e.g. GB200 NVL72).

### Loopback sanity (single node)

Verifies the code path. Results reflect intra-node EGM overhead, not cross-node latency.
On hardware without fabric support the binary will exit with a clear diagnostic.

```bash
mpirun -n 1 gpu_baseline_cross_node_pinned_host \
  --blocks 1 --threads 32 --bytes-per-warp 4096 --iterations 5 --mode write-bw \
  --src-gpu-id 0 --src-numa-node 0 --dst-gpu-id 0 --dst-numa-node 0
```

### Intra-Node Cross-NUMA Runs — DeltaAI (recommended)

On a single DeltaAI quad-GH200 node, run two ranks pinned to different sockets. This
measures GPU-to-remote-NUMA-DRAM bandwidth routed over NVLink-C2C — the meaningful path
this benchmark was designed for on DeltaAI.

**Bandwidth:**
```bash
srun --ntasks=2 --nodes=1 --ntasks-per-node=2 --gpus-per-task=1 \
  /path/to/build/bin/gpu_baseline_cross_node_pinned_host \
  --blocks 4 --threads 256 --bytes-per-warp 1m --iterations 50 --mode write-bw \
  --src-gpu-id 0 --src-numa-node 0 --dst-gpu-id 1 --dst-numa-node 3
```

**Latency sweep:**
```bash
for size in 64 256 1k 4k 16k; do
  srun --ntasks=2 --nodes=1 --ntasks-per-node=2 --gpus-per-task=1 \
    /path/to/build/bin/gpu_baseline_cross_node_pinned_host \
    --mode latency --bytes-per-warp $size --latency-iters 1000 \
    --src-gpu-id 0 --src-numa-node 0 --dst-gpu-id 1 --dst-numa-node 3
done
```

**Batch script (`bench5.sb`):**
```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --partition=gpuGH200x4       # DeltaAI GH200 partition; check: sinfo -s
#SBATCH --time=00:10:00
#SBATCH --account=<your_account>

BIN=/path/to/build/bin/gpu_baseline_cross_node_pinned_host

# Bandwidth (GPU 0 / NUMA 0  →  GPU 1 / NUMA 3, routed over NVLink-C2C)
srun $BIN --blocks 4 --threads 256 --bytes-per-warp 1m --iterations 100 --mode write-bw \
  --src-gpu-id 0 --src-numa-node 0 --dst-gpu-id 1 --dst-numa-node 3
srun $BIN --blocks 4 --threads 256 --bytes-per-warp 1m --iterations 100 --mode read-bw \
  --src-gpu-id 0 --src-numa-node 0 --dst-gpu-id 1 --dst-numa-node 3

# Latency sweep
for size in 64 256 1k 4k 16k; do
  srun $BIN --mode latency --bytes-per-warp $size --latency-iters 1000 \
    --src-gpu-id 0 --src-numa-node 0 --dst-gpu-id 1 --dst-numa-node 3
done
```

### Example Output

BW mode (rank 0 prints):
```
=== GPU Baseline Benchmark 5: Cross-node Remote Pinned Host Memory (EGM) ===
Device     : NVIDIA GH200 480GB (rank 0 -> GPU 0)
Ranks      : 2 total
...
Mode          Rank  Target   Warps   bytes/warp   Iters   Time(ms)   BW(GB/s)
------------  ----  ------  ------  ------------  ------  ----------  ----------
write-bw         0       1     128       1048576     100      ...       ~200.0
```

Latency mode (rank 1 prints):
```
Cross-node EGM one-way latency (send-notify, 64 B payload)
  Samples : 1000
  Min     :  1.20 us
  Median  :  1.45 us
  Mean    :  1.52 us
  Max     :  2.10 us
```

### Expected Numbers

| Configuration | Write BW | Read BW | Latency |
|---------------|----------|---------|---------|
| GH200 intra-node (NVLink-C2C, local NUMA host) | ~400–500 GB/s | ~300–400 GB/s | ~0.5–1 µs |
| GH200 cross-node NVLink fabric (4-node quad) | ~200–400 GB/s | ~150–300 GB/s | ~1–5 µs |
| Generic cross-node (NVSwitch fabric, remote DRAM bottleneck) | ~50–200 GB/s | ~50–200 GB/s | ~1–5 µs |

The bandwidth ceiling is the remote node's host DRAM bandwidth (~500 GB/s LPDDR5 on GH200),
not the NVLink-C2C or NVLink fabric bandwidth (~900 GB/s bidirectional). Latency is
dominated by NVLink fabric hop count and PCIe-to-DRAM write combining on the remote side.

> **Note (loopback):** When run with a single rank (`mpirun -n 1`), two separate EGM
> allocations are made locally and the send/poll kernels execute sequentially on the same
> stream. Results reflect local EGM round-trip overhead, not cross-node fabric numbers.
> A `(loopback: no cross-node traffic)` warning is printed in this mode.

## CTest

All benchmarks have ctests:
```bash
ctest --test-dir build -R gpu_baseline -V
```

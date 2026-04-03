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

## CTest

Both benchmarks have ctests:
```bash
ctest --test-dir build -R gpu_baseline -V
```

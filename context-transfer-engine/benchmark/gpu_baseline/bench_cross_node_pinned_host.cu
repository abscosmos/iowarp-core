/**
 * bench_cross_node_pinned_host.cu — Benchmark 5: Cross-node remote pinned host
 *                                   memory bandwidth and latency via CUDA EGM
 *
 * Uses the CUDA Driver API VMM + CU_MEM_HANDLE_TYPE_FABRIC (EGM) to allocate
 * NUMA-pinned host memory that is shared between nodes via fabric handles. Each
 * rank exports its local EGM handle, exchanges it with its partner over MPI, and
 * maps the remote buffer into local GPU VA space. The GPU can then load/store
 * directly to the remote node's pinned host DRAM via NVLink fabric — no UCX,
 * no NVSHMEM, no IB verbs.
 *
 * Three modes:
 *   write-bw  — rank 0 GPU writes to rank 1's pinned host buffer
 *   read-bw   — rank 0 GPU reads from rank 1's pinned host buffer
 *   latency   — rank 0 sends payload+flag, rank 1 spin-polls and records delta
 *
 * Key design choices:
 *   - CU_MEM_LOCATION_TYPE_HOST_NUMA + CU_MEM_HANDLE_TYPE_FABRIC:
 *     allocates physically NUMA-local host DRAM that the GPU can access as EGM.
 *   - MPI_Sendrecv for handle exchange (all-to-all is overkill for 2 ranks).
 *   - __threadfence_system() after stores to ensure propagation to remote DRAM.
 *   - Runtime check of CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED before
 *     any allocation attempt; exits cleanly on unsupported hardware.
 *   - Loopback (n_pes==1): two separate EGM allocations on the same node.
 *
 * Launch:
 *   mpirun -n 1 ./gpu_baseline_cross_node_pinned_host [options]   # loopback
 *   mpirun -n 2 ./gpu_baseline_cross_node_pinned_host [options]   # cross-node
 *
 * Usage:
 *   gpu_baseline_cross_node_pinned_host
 *       [--blocks N] [--threads N] [--bytes-per-warp N]
 *       [--mode write-bw|read-bw|latency]
 *       [--warmup N] [--iterations N] [--latency-iters N]
 *       [--src-gpu-id N] [--src-numa-node N]
 *       [--dst-gpu-id N] [--dst-numa-node N]
 *       [--no-fence]
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include "utils.h"
#include "kernels_cross_node_pinned_host.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ============================================================
// CUDA Driver API error checking
// ============================================================

#define CU_CHECK(call)                                                    \
  do {                                                                    \
    CUresult _r = (call);                                                 \
    if (_r != CUDA_SUCCESS) {                                             \
      const char *_s = nullptr;                                           \
      cuGetErrorString(_r, &_s);                                          \
      fprintf(stderr, "CUDA Driver error at %s:%d: %s\n",                \
              __FILE__, __LINE__, _s ? _s : "unknown");                   \
      exit(1);                                                            \
    }                                                                     \
  } while (0)

// ============================================================
// Configuration
// ============================================================

enum class CrossNodePinnedMode { kWriteBw, kReadBw, kLatency };

struct CrossNodePinnedConfig {
  uint32_t            blocks         = 1;
  uint32_t            threads        = 32;
  uint64_t            bytes_per_warp = 4096;
  uint32_t            warmup_iters   = 5;
  uint32_t            iterations     = 100;
  uint32_t            latency_iters  = 1000;
  bool                no_fence       = false;
  int                 src_gpu_id     = 0;
  int                 src_numa_node  = 0;
  int                 dst_gpu_id     = 0;
  int                 dst_numa_node  = 0;
  CrossNodePinnedMode mode           = CrossNodePinnedMode::kWriteBw;
  bool                bpw_explicit   = false;

  uint32_t total_warps() const { return (blocks * threads) / 32; }
  uint64_t total_bytes() const { return (uint64_t)total_warps() * bytes_per_warp; }
};

static inline void print_cross_node_pinned_args_usage(const char *prog) {
  fprintf(stderr,
    "Usage: %s [options]\n"
    "  --blocks N              Grid blocks (default: 1)\n"
    "  --threads N             Threads per block, must be multiple of 32 (default: 32)\n"
    "  --bytes-per-warp N      Bytes per warp, multiple of 16 (default: 4096 for BW, 64 for latency)\n"
    "                          Accepts k/m/g suffix (e.g. 1m)\n"
    "  --mode write-bw|read-bw|latency  Measurement mode (default: write-bw)\n"
    "  --warmup N              Warmup iterations for BW modes (default: 5)\n"
    "  --iterations N          Timed iterations for BW modes (default: 100)\n"
    "  --latency-iters N       Timed samples for latency mode (default: 1000)\n"
    "  --src-gpu-id N          Source GPU device ID (default: 0)\n"
    "  --src-numa-node N       NUMA node for source EGM allocation (default: 0)\n"
    "  --dst-gpu-id N          Destination GPU device ID (loopback only; default: 0)\n"
    "  --dst-numa-node N       NUMA node for destination EGM alloc (loopback only; default: 0)\n"
    "  --no-fence              Skip __threadfence_system() in write kernel\n"
    "  --help                  Show this message\n"
    "\n"
    "  Launch with: mpirun -n 1 %s [options]  (loopback)\n"
    "               mpirun -n 2 %s [options]  (cross-node, one rank per node)\n",
    prog, prog, prog);
}

static inline CrossNodePinnedConfig parse_cross_node_pinned_args(int argc, char **argv) {
  CrossNodePinnedConfig cfg;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      print_cross_node_pinned_args_usage(argv[0]);
      exit(0);
    } else if (strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) {
      cfg.blocks = (uint32_t)atoi(argv[++i]);
    } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      cfg.threads = (uint32_t)atoi(argv[++i]);
    } else if (strcmp(argv[i], "--bytes-per-warp") == 0 && i + 1 < argc) {
      cfg.bytes_per_warp = parse_size(argv[++i]);
      cfg.bpw_explicit = true;
    } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      cfg.warmup_iters = (uint32_t)atoi(argv[++i]);
    } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
      cfg.iterations = (uint32_t)atoi(argv[++i]);
    } else if (strcmp(argv[i], "--latency-iters") == 0 && i + 1 < argc) {
      cfg.latency_iters = (uint32_t)atoi(argv[++i]);
    } else if (strcmp(argv[i], "--src-gpu-id") == 0 && i + 1 < argc) {
      cfg.src_gpu_id = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--src-numa-node") == 0 && i + 1 < argc) {
      cfg.src_numa_node = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--dst-gpu-id") == 0 && i + 1 < argc) {
      cfg.dst_gpu_id = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--dst-numa-node") == 0 && i + 1 < argc) {
      cfg.dst_numa_node = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--no-fence") == 0) {
      cfg.no_fence = true;
    } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
      ++i;
      if (strcmp(argv[i], "write-bw") == 0)      cfg.mode = CrossNodePinnedMode::kWriteBw;
      else if (strcmp(argv[i], "read-bw") == 0)  cfg.mode = CrossNodePinnedMode::kReadBw;
      else if (strcmp(argv[i], "latency") == 0)  cfg.mode = CrossNodePinnedMode::kLatency;
      else {
        fprintf(stderr, "Unknown mode '%s'. Use write-bw, read-bw, or latency.\n", argv[i]);
        exit(1);
      }
    } else {
      fprintf(stderr, "Unknown argument: %s\n", argv[i]);
      print_cross_node_pinned_args_usage(argv[0]);
      exit(1);
    }
  }

  if (cfg.mode == CrossNodePinnedMode::kLatency && !cfg.bpw_explicit) {
    cfg.bytes_per_warp = 64;
  }

  if (cfg.threads % 32 != 0) {
    fprintf(stderr, "Error: --threads must be a multiple of 32 (got %u)\n", cfg.threads);
    exit(1);
  }
  if (cfg.bytes_per_warp % 16 != 0) {
    fprintf(stderr, "Error: --bytes-per-warp must be a multiple of 16 (got %lu)\n",
            (unsigned long)cfg.bytes_per_warp);
    exit(1);
  }
  if (cfg.bytes_per_warp < 16) {
    fprintf(stderr, "Error: --bytes-per-warp must be at least 16\n");
    exit(1);
  }

  return cfg;
}

// ============================================================
// EGM allocation helpers
// ============================================================

// Computes the minimum granularity-aligned size >= requested_bytes.
static uint64_t align_up(uint64_t val, uint64_t align) {
  return (val + align - 1) & ~(align - 1);
}

struct EgmAlloc {
  CUmemGenericAllocationHandle handle = 0;
  CUdeviceptr                  va     = 0;
  size_t                       size   = 0;
};

// Allocates EGM (host NUMA) memory with a fabric-exportable handle, maps it
// into the GPU's VA space, and sets read/write access from gpu_device.
static EgmAlloc egm_alloc(uint64_t requested_bytes, int numa_node, int gpu_device,
                           uint64_t granularity) {
  EgmAlloc a;
  a.size = (size_t)align_up(requested_bytes, granularity);
  if (a.size == 0) a.size = (size_t)granularity;

  CUmemAllocationProp prop{};
  prop.type                  = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type         = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  prop.location.id           = numa_node;
  prop.requestedHandleTypes  = CU_MEM_HANDLE_TYPE_FABRIC;

  CU_CHECK(cuMemCreate(&a.handle, a.size, &prop, 0));
  CU_CHECK(cuMemAddressReserve(&a.va, a.size, 0, 0, 0));
  CU_CHECK(cuMemMap(a.va, a.size, 0, a.handle, 0));

  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id   = gpu_device;
  access.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CU_CHECK(cuMemSetAccess(a.va, a.size, &access, 1));

  return a;
}

static void egm_free(EgmAlloc &a) {
  if (a.va)     { cuMemUnmap(a.va, a.size); cuMemAddressFree(a.va, a.size); }
  if (a.handle) { cuMemRelease(a.handle); }
  a = {};
}

// Imports a remote fabric handle and maps it into local GPU VA space.
static EgmAlloc egm_import(const CUmemFabricHandle &fh, size_t size, int gpu_device) {
  EgmAlloc a;
  a.size = size;
  CU_CHECK(cuMemImportFromShareableHandle(
      &a.handle,
      (void *)const_cast<CUmemFabricHandle *>(&fh),
      CU_MEM_HANDLE_TYPE_FABRIC));
  CU_CHECK(cuMemAddressReserve(&a.va, a.size, 0, 0, 0));
  CU_CHECK(cuMemMap(a.va, a.size, 0, a.handle, 0));

  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id   = gpu_device;
  access.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CU_CHECK(cuMemSetAccess(a.va, a.size, &access, 1));

  return a;
}

// ============================================================
// main
// ============================================================

int main(int argc, char **argv) {
  MPI_CHECK(MPI_Init(&argc, &argv));

  CrossNodePinnedConfig cfg = parse_cross_node_pinned_args(argc, argv);

  // set device before cuInit so the primary context is created on the right GPU
  CUDA_CHECK(cudaSetDevice(cfg.src_gpu_id));

  CU_CHECK(cuInit(0));

  CUdevice  dev = 0;
  CUcontext ctx = nullptr;
  CU_CHECK(cuDeviceGet(&dev, cfg.src_gpu_id));
  CU_CHECK(cuCtxGetCurrent(&ctx));
  if (!ctx) {
    CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CU_CHECK(cuCtxSetCurrent(ctx));
  }

  int my_pe, n_pes;
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &my_pe));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &n_pes));

  if (n_pes > 2) {
    if (my_pe == 0) {
      fprintf(stderr, "bench_cross_node_pinned_host: only 1 or 2 ranks supported "
              "(got %d)\n", n_pes);
    }
    MPI_Finalize();
    return 1;
  }

  // runtime check: fabric handle support required
  {
    int fabric_supported = 0;
    CUresult r = cuDeviceGetAttribute(
        &fabric_supported,
        CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
        dev);
    if (r != CUDA_SUCCESS || fabric_supported == 0) {
      if (my_pe == 0) {
        fprintf(stderr,
          "bench_cross_node_pinned_host: CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED "
          "is 0 or unavailable on device %d.\n"
          "  EGM fabric handles require NVLink C2C or NVSwitch fabric hardware "
          "(e.g. GH200, GB200).\n",
          cfg.src_gpu_id);
      }
      MPI_Finalize();
      return 1;
    }
  }

  // each rank picks its local NUMA node from the config
  int local_numa = (my_pe == 0) ? cfg.src_numa_node : cfg.dst_numa_node;

  // compute granularity once — reused for all allocations below
  size_t gran = 0;
  {
    CUmemAllocationProp prop{};
    prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type        = CU_MEM_LOCATION_TYPE_HOST_NUMA;
    prop.location.id          = local_numa;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    CU_CHECK(cuMemGetAllocationGranularity(
        &gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  }

  cudaDeviceProp rt_prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&rt_prop, cfg.src_gpu_id));

  bool loopback = (n_pes == 1);

  if (my_pe == 0) {
    const char *mode_str = "write-bw";
    if (cfg.mode == CrossNodePinnedMode::kReadBw)  mode_str = "read-bw";
    if (cfg.mode == CrossNodePinnedMode::kLatency)  mode_str = "latency";

    printf("=== GPU Baseline Benchmark 5: Cross-node Remote Pinned Host Memory (EGM) ===\n");
    printf("Device     : %s (rank %d -> GPU %d)\n", rt_prop.name, my_pe, cfg.src_gpu_id);
    printf("Ranks      : %d total%s\n", n_pes, loopback ? " (loopback)" : "");
    if (cfg.mode != CrossNodePinnedMode::kLatency) {
      printf("Grid       : %u blocks x %u threads = %u warps\n",
             cfg.blocks, cfg.threads, cfg.total_warps());
      printf("Transfer   : %lu bytes/warp x %u warps = %.2f MB total\n",
             (unsigned long)cfg.bytes_per_warp,
             cfg.total_warps(),
             (double)cfg.total_bytes() / (1024.0 * 1024.0));
      printf("Timing     : %u warmup + %u timed iterations\n",
             cfg.warmup_iters, cfg.iterations);
      printf("Fence      : %s\n", cfg.no_fence ? "disabled (--no-fence)" : "enabled");
    } else {
      printf("Payload    : %lu bytes/sample\n", (unsigned long)cfg.bytes_per_warp);
      printf("Samples    : %u warmup + %u timed\n", cfg.warmup_iters, cfg.latency_iters);
    }
    printf("src NUMA   : %d   dst NUMA: %d\n", cfg.src_numa_node, cfg.dst_numa_node);
    printf("Mode       : %s\n", mode_str);
    if (loopback) {
      printf("WARNING    : loopback mode — no cross-node traffic; results reflect local "
             "EGM overhead only.\n");
    }
    printf("\n");
  }

  // ============================================================
  // BW path (write-bw or read-bw)
  // ============================================================
  if (cfg.mode == CrossNodePinnedMode::kWriteBw ||
      cfg.mode == CrossNodePinnedMode::kReadBw) {

    uint64_t alloc_bytes = cfg.total_bytes();
    EgmAlloc local_alloc = egm_alloc(alloc_bytes, local_numa, cfg.src_gpu_id, gran);

    // zero local buffer so reads return defined data
    CUDA_CHECK(cudaMemset((void *)local_alloc.va, 0, local_alloc.size));
    CUDA_CHECK(cudaDeviceSynchronize());

    // export local handle and exchange with partner
    CUmemFabricHandle local_fh{};
    CU_CHECK(cuMemExportToShareableHandle(
        &local_fh, local_alloc.handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));

    CUmemFabricHandle remote_fh{};
    EgmAlloc remote_alloc{};

    if (!loopback) {
      int partner = 1 - my_pe;
      MPI_CHECK(MPI_Sendrecv(
          &local_fh,  sizeof(CUmemFabricHandle), MPI_BYTE, partner, 0,
          &remote_fh, sizeof(CUmemFabricHandle), MPI_BYTE, partner, 0,
          MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      remote_alloc = egm_import(remote_fh, local_alloc.size, cfg.src_gpu_id);
    } else {
      // loopback: allocate a second buffer for the "remote" side
      int dst_numa = cfg.dst_numa_node;
      // granularity for dst numa
      size_t dst_gran = 0;
      {
        CUmemAllocationProp prop{};
        prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type        = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        prop.location.id          = dst_numa;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        CU_CHECK(cuMemGetAllocationGranularity(
            &dst_gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
      }
      remote_alloc = egm_alloc(alloc_bytes, dst_numa, cfg.dst_gpu_id, dst_gran);
      CUDA_CHECK(cudaMemset((void *)remote_alloc.va, 0, remote_alloc.size));
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // rank 0 is the active initiator; rank 1 waits at barriers
    bool is_active = (my_pe == 0) || loopback;

    char *write_dst = reinterpret_cast<char *>(remote_alloc.va);
    const char *read_src = reinterpret_cast<const char *>(remote_alloc.va);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    GpuTimer timer;

    bool use_fence = !cfg.no_fence;

    // warmup
    if (is_active) {
      if (cfg.mode == CrossNodePinnedMode::kWriteBw) {
        cross_node_pinned_write_kernel<<<cfg.blocks, cfg.threads, 0, stream>>>(
            write_dst, cfg.bytes_per_warp, cfg.warmup_iters, use_fence);
      } else {
        cross_node_pinned_read_kernel<<<cfg.blocks, cfg.threads, 0, stream>>>(
            read_src, cfg.bytes_per_warp, cfg.warmup_iters);
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // timed run
    float elapsed_ms = 0.f;
    if (is_active) {
      timer.record_start(stream);
      if (cfg.mode == CrossNodePinnedMode::kWriteBw) {
        cross_node_pinned_write_kernel<<<cfg.blocks, cfg.threads, 0, stream>>>(
            write_dst, cfg.bytes_per_warp, cfg.iterations, use_fence);
      } else {
        cross_node_pinned_read_kernel<<<cfg.blocks, cfg.threads, 0, stream>>>(
            read_src, cfg.bytes_per_warp, cfg.iterations);
      }
      timer.record_stop(stream);
      elapsed_ms = timer.elapsed_ms();
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (my_pe == 0) {
      const char *label = (cfg.mode == CrossNodePinnedMode::kWriteBw) ? "write-bw" : "read-bw";

      printf("%-12s  %4s  %6s  %6s  %12s  %6s  %10s  %10s\n",
             "Mode", "Rank", "Target", "Warps", "bytes/warp", "Iters",
             "Time(ms)", "BW(GB/s)");
      printf("%-12s  %4s  %6s  %6s  %12s  %6s  %10s  %10s\n",
             "------------", "----", "------", "------", "------------",
             "------", "----------", "----------");

      double bytes_moved = (double)cfg.total_bytes() * cfg.iterations;
      double gbps = bytes_moved / (elapsed_ms / 1000.0) / 1e9;
      int target_rank = loopback ? 0 : 1;
      printf("%-12s  %4d  %6d  %6u  %12lu  %6u  %10.2f  %10.3f\n",
             label, my_pe, target_rank, cfg.total_warps(),
             (unsigned long)cfg.bytes_per_warp, cfg.iterations,
             elapsed_ms, gbps);
      printf("\n");
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    egm_free(remote_alloc);
    egm_free(local_alloc);
  }

  // ============================================================
  // Latency path
  // ============================================================
  else {
    uint32_t total_lat   = cfg.warmup_iters + cfg.latency_iters;
    uint64_t payload_bytes = align_up(cfg.bytes_per_warp, gran);
    uint64_t flag_bytes    = align_up((uint64_t)total_lat * sizeof(uint32_t), gran);

    // each rank allocates local payload + flag EGM buffers
    EgmAlloc local_payload = egm_alloc(payload_bytes, local_numa, cfg.src_gpu_id, gran);
    EgmAlloc local_flags   = egm_alloc(flag_bytes,    local_numa, cfg.src_gpu_id, gran);

    CUDA_CHECK(cudaMemset((void *)local_payload.va, 0, local_payload.size));
    CUDA_CHECK(cudaMemset((void *)local_flags.va,   0, local_flags.size));
    CUDA_CHECK(cudaDeviceSynchronize());

    // export handles
    CUmemFabricHandle local_payload_fh{};
    CUmemFabricHandle local_flags_fh{};
    CU_CHECK(cuMemExportToShareableHandle(
        &local_payload_fh, local_payload.handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
    CU_CHECK(cuMemExportToShareableHandle(
        &local_flags_fh, local_flags.handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));

    // exchange handles
    EgmAlloc remote_payload{};
    EgmAlloc remote_flags{};

    if (!loopback) {
      int partner = 1 - my_pe;

      CUmemFabricHandle remote_payload_fh{};
      CUmemFabricHandle remote_flags_fh{};

      MPI_CHECK(MPI_Sendrecv(
          &local_payload_fh,  sizeof(CUmemFabricHandle), MPI_BYTE, partner, 0,
          &remote_payload_fh, sizeof(CUmemFabricHandle), MPI_BYTE, partner, 0,
          MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      MPI_CHECK(MPI_Sendrecv(
          &local_flags_fh,  sizeof(CUmemFabricHandle), MPI_BYTE, partner, 1,
          &remote_flags_fh, sizeof(CUmemFabricHandle), MPI_BYTE, partner, 1,
          MPI_COMM_WORLD, MPI_STATUS_IGNORE));

      remote_payload = egm_import(remote_payload_fh, local_payload.size, cfg.src_gpu_id);
      remote_flags   = egm_import(remote_flags_fh,   local_flags.size,   cfg.src_gpu_id);
    } else {
      // loopback: allocate second set on dst_numa
      int dst_numa = cfg.dst_numa_node;
      size_t dst_gran = 0;
      {
        CUmemAllocationProp prop{};
        prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type        = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        prop.location.id          = dst_numa;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        CU_CHECK(cuMemGetAllocationGranularity(
            &dst_gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
      }
      remote_payload = egm_alloc(payload_bytes, dst_numa, cfg.dst_gpu_id, dst_gran);
      remote_flags   = egm_alloc(flag_bytes,    dst_numa, cfg.dst_gpu_id, dst_gran);
      CUDA_CHECK(cudaMemset((void *)remote_payload.va, 0, remote_payload.size));
      CUDA_CHECK(cudaMemset((void *)remote_flags.va,   0, remote_flags.size));
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (loopback) {
      fprintf(stderr,
        "NOTE: latency loopback (1 rank) — send and poll run sequentially on same stream;\n"
        "      flags are pre-set when poll kernel starts. Results reflect local EGM overhead,\n"
        "      not cross-node fabric latency. Use mpirun -n 2 for real measurements.\n");
    }

    bool is_sender   = (my_pe == 0) || loopback;
    bool is_receiver = (my_pe == 1) || loopback;

    uint64_t *result_buf  = nullptr;
    uint64_t *result_host = nullptr;
    if (is_receiver) {
      CUDA_CHECK(cudaMalloc(&result_buf, (size_t)cfg.latency_iters * sizeof(uint64_t)));
      result_host = (uint64_t *)malloc((size_t)cfg.latency_iters * sizeof(uint64_t));
      if (!result_host) {
        fprintf(stderr, "[rank %d] malloc failed for latency result buffer\n", my_pe);
        egm_free(remote_payload);
        egm_free(remote_flags);
        egm_free(local_payload);
        egm_free(local_flags);
        if (result_buf) { CUDA_CHECK(cudaFree(result_buf)); }
        MPI_Finalize();
        return 1;
      }
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // sender targets remote; receiver polls local
    char              *remote_payload_ptr = reinterpret_cast<char *>(remote_payload.va);
    volatile uint32_t *remote_flag_ptr    = reinterpret_cast<volatile uint32_t *>(remote_flags.va);
    volatile uint32_t *local_flag_ptr     = reinterpret_cast<volatile uint32_t *>(local_flags.va);

    if (is_sender) {
      cross_node_pinned_latency_send_kernel<<<1, 32, 0, stream>>>(
          remote_payload_ptr,
          remote_flag_ptr,
          cfg.bytes_per_warp,
          cfg.warmup_iters,
          cfg.latency_iters);
    }
    if (is_receiver) {
      cross_node_pinned_latency_poll_kernel<<<1, 32, 0, stream>>>(
          local_flag_ptr,
          result_buf,
          cfg.warmup_iters,
          cfg.latency_iters);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (is_receiver && result_buf) {
      CUDA_CHECK(cudaMemcpy(result_host, result_buf,
                            (size_t)cfg.latency_iters * sizeof(uint64_t),
                            cudaMemcpyDeviceToHost));

      LatencyStats stats = compute_latency_stats(result_host, cfg.latency_iters, cfg.src_gpu_id);

      printf("Cross-node EGM one-way latency (send-notify, %lu B payload)\n",
             (unsigned long)cfg.bytes_per_warp);
      printf("  Samples : %u\n",           cfg.latency_iters);
      printf("  Min     : %6.2f us\n",     stats.min_us);
      printf("  Median  : %6.2f us\n",     stats.median_us);
      printf("  Mean    : %6.2f us\n",     stats.mean_us);
      printf("  Max     : %6.2f us\n",     stats.max_us);
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    if (result_buf)  { CUDA_CHECK(cudaFree(result_buf)); }
    if (result_host) { free(result_host); }
    egm_free(remote_payload);
    egm_free(remote_flags);
    egm_free(local_payload);
    egm_free(local_flags);
  }

  MPI_CHECK(MPI_Finalize());
  return 0;
}

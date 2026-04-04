/**
 * bench_bus_contention.cu — Benchmark 3: Bus Contention (HBM + pinned host)
 *
 * Measures PCIe bus contention when two workloads run concurrently on
 * separate GPUs within a single multi-GPU node:
 *
 *   Workload A (PE 0): GPU kernel writing to pinned host memory (PCIe write)
 *   Workload B (PE 1): NVSHMEM put from GPU 1 to GPU 0's symmetric memory
 *
 * Both traffic flows converge on GPU 0's PCIe lanes, creating contention.
 * The benchmark compares solo bandwidth (each workload alone) against
 * contention bandwidth (both simultaneous) to quantify degradation.
 *
 * Three-phase execution:
 *   Phase 1 — Solo baselines: each workload runs alone while the other idles
 *   Phase 2 — Contention: both workloads launch simultaneously
 *   Phase 3 — Report: PE 0 collects timings and prints comparison table
 *
 * Single-PE loopback (nvshmrun -n 1): runs solo baselines only, skips
 * contention phase.  Validates compilation and code paths on 1-GPU machines.
 *
 * Launch:
 *   nvshmrun -n 2 ./gpu_baseline_bus_contention [options]
 *   nvshmrun -n 1 ./gpu_baseline_bus_contention [options]   # loopback
 *
 * No HermesShm / Chimaera dependency — raw hardware measurement only.
 */

#ifdef HAVE_NVSHMEM

#include <mpi.h>
#include "utils.h"
#include "kernels_pinned_host.cuh"
#include "kernels_remote_hbm.cuh"
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ============================================================
// Configuration
// ============================================================

struct ContentionConfig {
  // Workload A (pinned-host write on PE 0)
  uint32_t a_blocks         = 8;
  uint32_t a_threads        = 256;
  uint64_t a_bytes_per_warp = 1048576;   // 1 MB

  // Workload B (NVSHMEM put on PE 1)
  uint32_t b_blocks         = 1;
  uint32_t b_threads        = 32;
  uint64_t b_bytes_per_warp = 1048576;   // 1 MB

  // Shared parameters
  uint32_t warmup_iters     = 5;
  uint32_t iterations       = 100;
  bool     no_fence         = false;
  int      numa_node        = 0;
  int      gpu_id           = -1;        // -1 = auto (mype % num_gpus)

  // Derived helpers
  uint32_t a_total_warps() const { return (a_blocks * a_threads) / 32; }
  uint64_t a_total_bytes() const { return (uint64_t)a_total_warps() * a_bytes_per_warp; }
  uint32_t b_total_warps() const { return (b_blocks * b_threads) / 32; }
  uint64_t b_total_bytes() const { return (uint64_t)b_total_warps() * b_bytes_per_warp; }
};

static inline void print_contention_usage(const char *prog) {
  fprintf(stderr,
    "Usage: %s [options]\n"
    "\n"
    "  Workload A (pinned-host write, PE 0):\n"
    "    --A-blocks N           Grid blocks (default: 8)\n"
    "    --A-threads N          Threads/block, multiple of 32 (default: 256)\n"
    "    --A-bytes-per-warp N   Bytes per warp, multiple of 16 (default: 1m)\n"
    "                           Accepts k/m/g suffix\n"
    "\n"
    "  Workload B (NVSHMEM put, PE 1):\n"
    "    --B-blocks N           Grid blocks (default: 1)\n"
    "    --B-threads N          Threads/block, multiple of 32 (default: 32)\n"
    "    --B-bytes-per-warp N   Bytes per warp, multiple of 16 (default: 1m)\n"
    "                           Accepts k/m/g suffix\n"
    "\n"
    "  Shared:\n"
    "    --warmup N             Warmup iterations (default: 5)\n"
    "    --iterations N         Timed iterations (default: 100)\n"
    "    --no-fence             Disable fencing for both workloads\n"
    "    --numa-node N          NUMA node for PE 0's pinned host alloc (default: 0)\n"
    "    --gpu-id N             Override CUDA device ID (default: auto = mype %% num_gpus)\n"
    "    --help                 Show this message\n"
    "\n"
    "  Launch with: nvshmrun -n 2 %s [options]\n"
    "               nvshmrun -n 1 %s [options]   (loopback — solo baselines only)\n"
    "\n"
    "  For large B-workload transfers, set NVSHMEM_SYMMETRIC_SIZE env var.\n",
    prog, prog, prog);
}

static inline ContentionConfig parse_contention_args(int argc, char **argv) {
  ContentionConfig cfg;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      print_contention_usage(argv[0]);
      exit(0);
    }
    // Workload A
    else if (strcmp(argv[i], "--A-blocks") == 0 && i + 1 < argc) {
      cfg.a_blocks = (uint32_t)atoi(argv[++i]);
    } else if (strcmp(argv[i], "--A-threads") == 0 && i + 1 < argc) {
      cfg.a_threads = (uint32_t)atoi(argv[++i]);
    } else if (strcmp(argv[i], "--A-bytes-per-warp") == 0 && i + 1 < argc) {
      cfg.a_bytes_per_warp = parse_size(argv[++i]);
    }
    // Workload B
    else if (strcmp(argv[i], "--B-blocks") == 0 && i + 1 < argc) {
      cfg.b_blocks = (uint32_t)atoi(argv[++i]);
    } else if (strcmp(argv[i], "--B-threads") == 0 && i + 1 < argc) {
      cfg.b_threads = (uint32_t)atoi(argv[++i]);
    } else if (strcmp(argv[i], "--B-bytes-per-warp") == 0 && i + 1 < argc) {
      cfg.b_bytes_per_warp = parse_size(argv[++i]);
    }
    // Shared
    else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      cfg.warmup_iters = (uint32_t)atoi(argv[++i]);
    } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
      cfg.iterations = (uint32_t)atoi(argv[++i]);
    } else if (strcmp(argv[i], "--no-fence") == 0) {
      cfg.no_fence = true;
    } else if (strcmp(argv[i], "--numa-node") == 0 && i + 1 < argc) {
      cfg.numa_node = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--gpu-id") == 0 && i + 1 < argc) {
      cfg.gpu_id = atoi(argv[++i]);
    } else {
      fprintf(stderr, "Unknown argument: %s\n", argv[i]);
      print_contention_usage(argv[0]);
      exit(1);
    }
  }

  // Validate workload A
  if (cfg.a_threads % 32 != 0) {
    fprintf(stderr, "Error: --A-threads must be a multiple of 32 (got %u)\n", cfg.a_threads);
    exit(1);
  }
  if (cfg.a_bytes_per_warp % 16 != 0 || cfg.a_bytes_per_warp < 16) {
    fprintf(stderr, "Error: --A-bytes-per-warp must be a multiple of 16, >= 16 (got %lu)\n",
            (unsigned long)cfg.a_bytes_per_warp);
    exit(1);
  }

  // Validate workload B
  if (cfg.b_threads % 32 != 0) {
    fprintf(stderr, "Error: --B-threads must be a multiple of 32 (got %u)\n", cfg.b_threads);
    exit(1);
  }
  if (cfg.b_bytes_per_warp % 16 != 0 || cfg.b_bytes_per_warp < 16) {
    fprintf(stderr, "Error: --B-bytes-per-warp must be a multiple of 16, >= 16 (got %lu)\n",
            (unsigned long)cfg.b_bytes_per_warp);
    exit(1);
  }

  return cfg;
}

// ============================================================
// main
// ============================================================

int main(int argc, char **argv) {
  // ---- MPI + NVSHMEM init (same pattern as bench_remote_hbm) ----
  MPI_CHECK(MPI_Init(&argc, &argv));

  nvshmemx_init_attr_t attr;
  MPI_Comm comm = MPI_COMM_WORLD;
  nvshmemx_set_attr_mpi_comm_args(&comm, &attr);
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

  int my_pe = nvshmem_my_pe();
  int n_pes = nvshmem_n_pes();

  ContentionConfig cfg = parse_contention_args(argc, argv);

  // ---- GPU assignment ----
  int num_gpus = 0;
  CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
  int device_id = (cfg.gpu_id >= 0) ? cfg.gpu_id : (my_pe % num_gpus);
  CUDA_CHECK(cudaSetDevice(device_id));

  bool use_fence = !cfg.no_fence;

  // ---- Gather device names for the report ----
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

  char dev_names[2][256];
  memset(dev_names, 0, sizeof(dev_names));
  snprintf(dev_names[my_pe < 2 ? my_pe : 0], 256, "%s", prop.name);

  if (n_pes >= 2) {
    // Gather device names to PE 0
    char my_name[256];
    memset(my_name, 0, sizeof(my_name));
    snprintf(my_name, 256, "%s", prop.name);
    MPI_CHECK(MPI_Gather(my_name, 256, MPI_CHAR,
                         dev_names, 256, MPI_CHAR,
                         0, MPI_COMM_WORLD));
  }

  // ---- Warn if symmetric heap might be too small ----
  uint64_t max_sym_bytes = (cfg.a_total_bytes() > cfg.b_total_bytes())
                           ? cfg.a_total_bytes() : cfg.b_total_bytes();
  if (my_pe == 0 && max_sym_bytes > 512ULL * 1024 * 1024) {
    fprintf(stderr, "WARNING: max workload transfer size is %.2f MB. "
            "Ensure NVSHMEM_SYMMETRIC_SIZE is set large enough.\n",
            (double)max_sym_bytes / (1024.0 * 1024.0));
  }

  // ---- Allocate buffers ----
  // Symmetric buffer: needed on all PEs (for NVSHMEM put target)
  char *sym_buf = (char *)nvshmem_malloc(max_sym_bytes);
  if (!sym_buf) {
    fprintf(stderr, "[PE %d] nvshmem_malloc(%lu) failed — "
            "try increasing NVSHMEM_SYMMETRIC_SIZE\n",
            my_pe, (unsigned long)max_sym_bytes);
    nvshmem_finalize();
    MPI_Finalize();
    return 1;
  }

  // Local device buffer for NVSHMEM source data
  char *local_buf = nullptr;
  CUDA_CHECK(cudaMalloc(&local_buf, cfg.b_total_bytes()));

  // Pinned host buffer for workload A (PE 0, or PE 0 in loopback)
  PinnedBuffer pinned_buf;
  if (my_pe == 0) {
    pinned_buf.allocate(cfg.a_total_bytes(), cfg.numa_node);
  }

  // ---- Initialize NVSHMEM buffers ----
  // Fill local_buf with a pattern (needed for put source)
  init_pattern_kernel<<<cfg.b_blocks, cfg.b_threads>>>(
      local_buf, cfg.b_bytes_per_warp, cfg.b_total_warps(), my_pe);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Initialize sym_buf (target for remote put)
  init_pattern_kernel<<<cfg.b_blocks, cfg.b_threads>>>(
      sym_buf, cfg.b_bytes_per_warp, cfg.b_total_warps(), my_pe);
  CUDA_CHECK(cudaDeviceSynchronize());

  nvshmem_barrier_all();

  // ---- NVSHMEM target: PE 1 puts to PE 0 ----
  int target_pe = 0;

  // ---- Create streams and timers ----
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // ---- Print configuration header (PE 0 only) ----
  if (my_pe == 0) {
    printf("=== GPU Baseline Benchmark 3: Bus Contention ===\n");
    if (n_pes == 1) {
      printf("NOTE: single PE — running solo baselines only (no contention phase).\n");
      printf("      Launch with nvshmrun -n 2 for contention measurement.\n\n");
      printf("Device: %s (PE 0 -> GPU %d)\n", dev_names[0], device_id);
    } else {
      printf("Device (PE 0): %s  GPU %d\n", dev_names[0], device_id);
      printf("Device (PE 1): %s  GPU %d\n", dev_names[1],
             (cfg.gpu_id >= 0) ? cfg.gpu_id : (1 % num_gpus));
    }
    printf("PEs     : %d\n", n_pes);
    printf("Grid A  : %u blocks x %u threads = %u warps, %lu bytes/warp = %.2f MB\n",
           cfg.a_blocks, cfg.a_threads, cfg.a_total_warps(),
           (unsigned long)cfg.a_bytes_per_warp,
           (double)cfg.a_total_bytes() / (1024.0 * 1024.0));
    printf("Grid B  : %u blocks x %u threads = %u warps, %lu bytes/warp = %.2f MB\n",
           cfg.b_blocks, cfg.b_threads, cfg.b_total_warps(),
           (unsigned long)cfg.b_bytes_per_warp,
           (double)cfg.b_total_bytes() / (1024.0 * 1024.0));
    printf("Timing  : %u warmup + %u timed iterations\n",
           cfg.warmup_iters, cfg.iterations);
    printf("Fence   : %s\n", cfg.no_fence ? "disabled (--no-fence)" : "enabled");
#ifdef HAVE_LIBNUMA
    printf("NUMA    : node %d (libnuma: enabled)\n", cfg.numa_node);
#else
    printf("NUMA    : node %d (libnuma: not compiled — using cudaMallocHost)\n",
           cfg.numa_node);
#endif
    printf("\n");
  }

  // ============================================================
  // Warmup phase (each PE warms up its own workload)
  // ============================================================

  if (my_pe == 0) {
    // Warmup workload A: pinned-host write
    pinned_host_write_kernel<<<cfg.a_blocks, cfg.a_threads, 0, stream>>>(
        pinned_buf.dev_get<char>(), cfg.a_bytes_per_warp,
        cfg.a_total_warps(), cfg.warmup_iters, use_fence);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  if (my_pe == 1 || n_pes == 1) {
    // Warmup workload B: NVSHMEM put
    remote_hbm_put_kernel<<<cfg.b_blocks, cfg.b_threads, 0, stream>>>(
        sym_buf, local_buf, cfg.b_bytes_per_warp,
        cfg.b_total_warps(), cfg.warmup_iters, target_pe, use_fence);
    nvshmemx_quiet_on_stream(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // ============================================================
  // Phase 1: Solo baselines
  // ============================================================

  float solo_ms_a = 0.f;
  float solo_ms_b = 0.f;

  // ---- Solo A: PE 0 runs pinned-host write, others idle ----

  if (my_pe == 0) {
    GpuTimer timer_a;
    timer_a.record_start(stream);
    pinned_host_write_kernel<<<cfg.a_blocks, cfg.a_threads, 0, stream>>>(
        pinned_buf.dev_get<char>(), cfg.a_bytes_per_warp,
        cfg.a_total_warps(), cfg.iterations, use_fence);
    timer_a.record_stop(stream);
    solo_ms_a = timer_a.elapsed_ms();
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // ---- Solo B: PE 1 (or PE 0 in loopback) runs NVSHMEM put, others idle ----

  if (my_pe == 1 || n_pes == 1) {
    GpuTimer timer_b;
    timer_b.record_start(stream);
    remote_hbm_put_kernel<<<cfg.b_blocks, cfg.b_threads, 0, stream>>>(
        sym_buf, local_buf, cfg.b_bytes_per_warp,
        cfg.b_total_warps(), cfg.iterations, target_pe, use_fence);
    nvshmemx_quiet_on_stream(stream);
    timer_b.record_stop(stream);
    solo_ms_b = timer_b.elapsed_ms();
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // PE 1 sends solo_ms_b to PE 0
  if (n_pes >= 2) {
    if (my_pe == 1) {
      MPI_CHECK(MPI_Send(&solo_ms_b, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD));
    } else if (my_pe == 0) {
      MPI_CHECK(MPI_Recv(&solo_ms_b, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE));
    }
  }

  // ---- Print Phase 1 results (PE 0) ----
  if (my_pe == 0) {
    double solo_bw_a = (double)cfg.a_total_bytes() * cfg.iterations
                       / (solo_ms_a / 1000.0) / 1e9;
    double solo_bw_b = (double)cfg.b_total_bytes() * cfg.iterations
                       / (solo_ms_b / 1000.0) / 1e9;

    printf("Phase 1: Solo baselines\n");
    printf("%-16s  %5s  %10s  %6s  %10s  %10s\n",
           "Workload", "Warps", "bytes/warp", "Iters", "Time(ms)", "BW(GB/s)");
    printf("%-16s  %5s  %10s  %6s  %10s  %10s\n",
           "----------------", "-----", "----------", "------",
           "----------", "----------");
    printf("%-16s  %5u  %10lu  %6u  %10.2f  %10.3f\n",
           "A (pinned)", cfg.a_total_warps(),
           (unsigned long)cfg.a_bytes_per_warp, cfg.iterations,
           solo_ms_a, solo_bw_a);
    printf("%-16s  %5u  %10lu  %6u  %10.2f  %10.3f\n",
           "B (nvshmem)", cfg.b_total_warps(),
           (unsigned long)cfg.b_bytes_per_warp, cfg.iterations,
           solo_ms_b, solo_bw_b);
    printf("\n");
  }

  // ============================================================
  // Phase 2: Contention run (both simultaneous)
  // ============================================================

  float contention_ms_a = 0.f;
  float contention_ms_b = 0.f;

  if (n_pes >= 2) {
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (my_pe == 0) {
      // PE 0: pinned-host write
      GpuTimer timer_a;
      timer_a.record_start(stream);
      pinned_host_write_kernel<<<cfg.a_blocks, cfg.a_threads, 0, stream>>>(
          pinned_buf.dev_get<char>(), cfg.a_bytes_per_warp,
          cfg.a_total_warps(), cfg.iterations, use_fence);
      timer_a.record_stop(stream);
      contention_ms_a = timer_a.elapsed_ms();
    } else if (my_pe == 1) {
      // PE 1: NVSHMEM put to PE 0
      GpuTimer timer_b;
      timer_b.record_start(stream);
      remote_hbm_put_kernel<<<cfg.b_blocks, cfg.b_threads, 0, stream>>>(
          sym_buf, local_buf, cfg.b_bytes_per_warp,
          cfg.b_total_warps(), cfg.iterations, target_pe, use_fence);
      nvshmemx_quiet_on_stream(stream);
      timer_b.record_stop(stream);
      contention_ms_b = timer_b.elapsed_ms();
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // PE 1 sends contention_ms_b to PE 0
    if (my_pe == 1) {
      MPI_CHECK(MPI_Send(&contention_ms_b, 1, MPI_FLOAT, 0, 1, MPI_COMM_WORLD));
    } else if (my_pe == 0) {
      MPI_CHECK(MPI_Recv(&contention_ms_b, 1, MPI_FLOAT, 1, 1, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE));
    }

    // ---- Print Phase 2 results + degradation (PE 0) ----
    if (my_pe == 0) {
      double solo_bw_a = (double)cfg.a_total_bytes() * cfg.iterations
                         / (solo_ms_a / 1000.0) / 1e9;
      double solo_bw_b = (double)cfg.b_total_bytes() * cfg.iterations
                         / (solo_ms_b / 1000.0) / 1e9;
      double cont_bw_a = (double)cfg.a_total_bytes() * cfg.iterations
                         / (contention_ms_a / 1000.0) / 1e9;
      double cont_bw_b = (double)cfg.b_total_bytes() * cfg.iterations
                         / (contention_ms_b / 1000.0) / 1e9;
      double degrad_a  = (1.0 - cont_bw_a / solo_bw_a) * 100.0;
      double degrad_b  = (1.0 - cont_bw_b / solo_bw_b) * 100.0;

      printf("Phase 2: Contention (both simultaneous)\n");
      printf("%-16s  %10s  %14s  %12s\n",
             "Workload", "Solo BW", "Contention BW", "Degradation");
      printf("%-16s  %10s  %14s  %12s\n",
             "----------------", "----------", "--------------",
             "------------");
      printf("%-16s  %8.1f GB/s  %10.1f GB/s  %10.1f%%\n",
             "pinned-write", solo_bw_a, cont_bw_a, degrad_a);
      printf("%-16s  %8.1f GB/s  %10.1f GB/s  %10.1f%%\n",
             "remote-hbm-put", solo_bw_b, cont_bw_b, degrad_b);
      printf("\nContention factor: A dropped %.1f%%, B dropped %.1f%%\n",
             degrad_a, degrad_b);
    }
  } else {
    // Single-PE mode: skip contention
    if (my_pe == 0) {
      printf("Contention phase skipped (need 2 PEs / 2 GPUs).\n");
    }
  }

  printf("\n");

  // ---- Cleanup ----
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(local_buf));
  nvshmem_free(sym_buf);
  // pinned_buf cleaned up by RAII destructor

  nvshmem_finalize();
  MPI_CHECK(MPI_Finalize());

  return 0;
}

#else  // !HAVE_NVSHMEM

#include <cstdio>

int main(int /*argc*/, char ** /*argv*/) {
  fprintf(stderr,
    "bench_bus_contention: not built with NVSHMEM support.\n"
    "Rebuild with -DWRP_CORE_ENABLE_NVSHMEM=ON and ensure nvshmem is installed.\n");
  return 1;
}

#endif  // HAVE_NVSHMEM

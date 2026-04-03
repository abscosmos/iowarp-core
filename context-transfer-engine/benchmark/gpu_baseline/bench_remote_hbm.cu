/**
 * bench_remote_hbm.cu — Benchmark 2: GPU-to-GPU HBM bandwidth via NVSHMEM
 *
 * Measures GPU-to-GPU HBM bandwidth on a single multi-GPU node using NVSHMEM.
 * A CUDA kernel running on PE 0 (GPU 0) uses nvshmemx_putmem_nbi_warp /
 * nvshmemx_getmem_nbi_warp to transfer data to/from a remote PE's symmetric
 * memory.
 *
 * Three modes:
 *   put       — PE 0 writes data into PE 1's symmetric buffer (put)
 *   get       — PE 0 reads data from PE 1's symmetric buffer  (get)
 *   ping-pong — PE 0 puts then gets back (round-trip)
 *
 * Key design choices:
 *   - Warp-collective nvshmem operations: all 32 lanes participate in each
 *     nvshmemx_{put,get}mem_nbi_warp call for optimal multi-lane data movement.
 *   - nvshmem_quiet() per iteration (fenced mode) or after all iterations
 *     (pipeline mode with --no-fence) for latency vs bandwidth measurement.
 *   - nvshmemx_quiet_on_stream() on host after kernel for belt-and-suspenders
 *     timing accuracy.
 *   - No HermesShm / Chimaera dependency — raw hardware measurement only.
 *
 * Launch:
 *   nvshmrun -n 2 ./gpu_baseline_remote_hbm [options]
 *   mpirun   -n 2 ./gpu_baseline_remote_hbm [options]
 *   nvshmrun -n 1 ./gpu_baseline_remote_hbm [options]   # loopback (PE 0 -> PE 0)
 *
 * Usage:
 *   gpu_baseline_remote_hbm [--blocks N] [--threads N] [--bytes-per-warp N]
 *                           [--mode put|get|ping-pong]
 *                           [--warmup N] [--iterations N]
 *                           [--no-fence] [--validate]
 *                           [--src-numa N] [--dst-numa N]
 */

#ifdef HAVE_NVSHMEM

#include "utils.h"
#include <nvshmem.h>
#include <nvshmemx.h>
#include <mpi.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ============================================================
// MPI error checking
// ============================================================

#define MPI_CHECK(call)                                                   \
  do {                                                                    \
    int _err = (call);                                                    \
    if (_err != MPI_SUCCESS) {                                            \
      char _errstr[MPI_MAX_ERROR_STRING];                                 \
      int _errlen;                                                        \
      MPI_Error_string(_err, _errstr, &_errlen);                          \
      fprintf(stderr, "MPI error at %s:%d: %s\n",                        \
              __FILE__, __LINE__, _errstr);                               \
      exit(1);                                                            \
    }                                                                     \
  } while (0)

// ============================================================
// Configuration
// ============================================================

enum class NvshmemMode { kPut, kGet, kPingPong };

struct RemoteHbmConfig {
  uint32_t    blocks         = 1;
  uint32_t    threads        = 32;
  uint64_t    bytes_per_warp = 4096;
  uint32_t    warmup_iters   = 5;
  uint32_t    iterations     = 100;
  bool        validate       = false;
  bool        no_fence       = false;
  NvshmemMode mode           = NvshmemMode::kPut;
  int         src_numa       = -1;    // informational only
  int         dst_numa       = -1;    // informational only

  uint32_t total_warps() const { return (blocks * threads) / 32; }
  uint64_t total_bytes() const { return (uint64_t)total_warps() * bytes_per_warp; }
};

static inline void print_remote_hbm_usage(const char *prog) {
  fprintf(stderr,
    "Usage: %s [options]\n"
    "  --blocks N           Grid blocks (default: 1)\n"
    "  --threads N          Threads per block, must be multiple of 32 (default: 32)\n"
    "  --bytes-per-warp N   Bytes each warp transfers, multiple of 16 (default: 4096)\n"
    "                       Accepts k/m/g suffix (e.g. 1m)\n"
    "  --mode put|get|ping-pong  Transfer direction (default: put)\n"
    "  --warmup N           Warmup iterations (default: 5)\n"
    "  --iterations N       Timed iterations (default: 100)\n"
    "  --no-fence           Skip per-iteration nvshmem_quiet() (pipeline mode)\n"
    "  --validate           Write pattern, quiet, read back and check\n"
    "  --src-numa N         Source NUMA node (informational only)\n"
    "  --dst-numa N         Destination NUMA node (informational only)\n"
    "  --help               Show this message\n"
    "\n"
    "  For large transfers, set NVSHMEM_SYMMETRIC_SIZE env var (e.g. 1073741824 for 1 GB).\n"
    "  Launch with: nvshmrun -n 2 %s [options]\n"
    "               nvshmrun -n 1 %s [options]   (loopback mode)\n",
    prog, prog, prog);
}

static inline RemoteHbmConfig parse_remote_hbm_args(int argc, char **argv) {
  RemoteHbmConfig cfg;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      print_remote_hbm_usage(argv[0]);
      exit(0);
    } else if (strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) {
      cfg.blocks = (uint32_t)atoi(argv[++i]);
    } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      cfg.threads = (uint32_t)atoi(argv[++i]);
    } else if (strcmp(argv[i], "--bytes-per-warp") == 0 && i + 1 < argc) {
      cfg.bytes_per_warp = parse_size(argv[++i]);
    } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      cfg.warmup_iters = (uint32_t)atoi(argv[++i]);
    } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
      cfg.iterations = (uint32_t)atoi(argv[++i]);
    } else if (strcmp(argv[i], "--validate") == 0) {
      cfg.validate = true;
    } else if (strcmp(argv[i], "--no-fence") == 0) {
      cfg.no_fence = true;
    } else if (strcmp(argv[i], "--src-numa") == 0 && i + 1 < argc) {
      cfg.src_numa = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--dst-numa") == 0 && i + 1 < argc) {
      cfg.dst_numa = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
      ++i;
      if (strcmp(argv[i], "put") == 0)             cfg.mode = NvshmemMode::kPut;
      else if (strcmp(argv[i], "get") == 0)         cfg.mode = NvshmemMode::kGet;
      else if (strcmp(argv[i], "ping-pong") == 0)   cfg.mode = NvshmemMode::kPingPong;
      else {
        fprintf(stderr, "Unknown mode '%s'. Use put, get, or ping-pong.\n", argv[i]);
        exit(1);
      }
    } else {
      fprintf(stderr, "Unknown argument: %s\n", argv[i]);
      print_remote_hbm_usage(argv[0]);
      exit(1);
    }
  }

  // Validate
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
// Kernel: fill local buffer with identifiable pattern
// ============================================================

__global__ void init_pattern_kernel(char *buf, uint64_t bytes_per_warp,
                                    uint32_t total_warps, int pe_id) {
  uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t warp_id    = global_tid / 32;
  uint32_t lane_id    = global_tid & 31;

  if (warp_id >= total_warps) return;

  uint4 *dst4     = reinterpret_cast<uint4 *>(buf + (uint64_t)warp_id * bytes_per_warp);
  uint32_t num_u4 = (uint32_t)(bytes_per_warp / sizeof(uint4));

  uint4 pattern;
  pattern.x = (uint32_t)pe_id;
  pattern.y = warp_id;
  pattern.z = 0xDEADBEEFu;
  pattern.w = 0xCAFEBABEu;

  for (uint32_t i = lane_id; i < num_u4; i += 32) {
    dst4[i] = pattern;
  }
}

// ============================================================
// Kernel: NVSHMEM put (PE -> target PE)
// ============================================================

__global__ void remote_hbm_put_kernel(
    char    *sym_buf,
    char    *local_src,
    uint64_t bytes_per_warp,
    uint32_t total_warps,
    uint32_t iterations,
    int      target_pe,
    bool     use_fence)
{
  uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t warp_id    = global_tid / 32;

  if (warp_id >= total_warps) return;

  char *remote_dst = sym_buf  + (uint64_t)warp_id * bytes_per_warp;
  char *local_data = local_src + (uint64_t)warp_id * bytes_per_warp;

  for (uint32_t iter = 0; iter < iterations; ++iter) {
    nvshmemx_putmem_nbi_warp(remote_dst, local_data, bytes_per_warp, target_pe);

    if (use_fence) {
      __syncwarp();
      nvshmem_quiet();
    }
  }

  if (!use_fence) {
    __syncwarp();
    nvshmem_quiet();
  }
}

// ============================================================
// Kernel: NVSHMEM get (target PE -> PE)
// ============================================================

__global__ void remote_hbm_get_kernel(
    char    *sym_buf,
    char    *local_dst,
    uint64_t bytes_per_warp,
    uint32_t total_warps,
    uint32_t iterations,
    int      target_pe,
    bool     use_fence)
{
  uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t warp_id    = global_tid / 32;

  if (warp_id >= total_warps) return;

  char *remote_src = sym_buf   + (uint64_t)warp_id * bytes_per_warp;
  char *local_data = local_dst + (uint64_t)warp_id * bytes_per_warp;

  for (uint32_t iter = 0; iter < iterations; ++iter) {
    nvshmemx_getmem_nbi_warp(local_data, remote_src, bytes_per_warp, target_pe);

    if (use_fence) {
      __syncwarp();
      nvshmem_quiet();
    }
  }

  if (!use_fence) {
    __syncwarp();
    nvshmem_quiet();
  }
}

// ============================================================
// Kernel: NVSHMEM ping-pong (put then get)
// ============================================================

__global__ void remote_hbm_pingpong_kernel(
    char    *sym_buf,
    char    *local_buf,
    uint64_t bytes_per_warp,
    uint32_t total_warps,
    uint32_t iterations,
    int      target_pe,
    bool     use_fence)
{
  uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t warp_id    = global_tid / 32;

  if (warp_id >= total_warps) return;

  char *remote_region = sym_buf   + (uint64_t)warp_id * bytes_per_warp;
  char *local_data    = local_buf + (uint64_t)warp_id * bytes_per_warp;

  for (uint32_t iter = 0; iter < iterations; ++iter) {
    // Put to remote
    nvshmemx_putmem_nbi_warp(remote_region, local_data, bytes_per_warp, target_pe);

    if (use_fence) {
      __syncwarp();
      nvshmem_quiet();
    }

    // Get from remote
    nvshmemx_getmem_nbi_warp(local_data, remote_region, bytes_per_warp, target_pe);

    if (use_fence) {
      __syncwarp();
      nvshmem_quiet();
    }
  }

  if (!use_fence) {
    __syncwarp();
    nvshmem_quiet();
  }
}

// ============================================================
// Validation (host-side): check pattern in buffer
// ============================================================

static int validate_buffer_on_host(const RemoteHbmConfig &cfg,
                                   const char *host_buf, int expected_pe) {
  uint32_t total_warps = cfg.total_warps();
  int errors = 0;
  for (uint32_t w = 0; w < total_warps && errors < 32; ++w) {
    const uint4 *row = reinterpret_cast<const uint4 *>(
                           host_buf + (uint64_t)w * cfg.bytes_per_warp);
    uint32_t num_u4 = (uint32_t)(cfg.bytes_per_warp / sizeof(uint4));
    for (uint32_t i = 0; i < num_u4 && errors < 32; ++i) {
      uint4 expected;
      expected.x = (uint32_t)expected_pe;
      expected.y = w;
      expected.z = 0xDEADBEEFu;
      expected.w = 0xCAFEBABEu;
      uint4 got = row[i];
      if (got.x != expected.x || got.y != expected.y ||
          got.z != expected.z || got.w != expected.w) {
        if (errors == 0) {
          fprintf(stderr, "[PE %d] VALIDATE FAIL: first mismatch at warp=%u elem=%u\n"
                          "  expected {%u,%u,%u,%u} got {%u,%u,%u,%u}\n",
                  expected_pe, w, i,
                  expected.x, expected.y, expected.z, expected.w,
                  got.x, got.y, got.z, got.w);
        }
        ++errors;
      }
    }
  }
  return errors;
}

// ============================================================
// Result reporting (matches bench_pinned_host format)
// ============================================================

static inline void print_remote_hbm_result_header() {
  printf("%-12s  %4s  %6s  %6s  %12s  %6s  %10s  %10s\n",
         "Mode", "PE", "Target", "Warps", "bytes/warp", "Iters",
         "Time(ms)", "BW(GB/s)");
  printf("%-12s  %4s  %6s  %6s  %12s  %6s  %10s  %10s\n",
         "------------", "----", "------", "------", "------------",
         "------", "----------", "----------");
}

static inline void print_remote_hbm_result(const char *label, int my_pe,
                                           int target_pe,
                                           const RemoteHbmConfig &cfg,
                                           float elapsed_ms) {
  double bytes_moved = (double)cfg.total_bytes() * cfg.iterations;
  // For ping-pong, count both put and get
  if (cfg.mode == NvshmemMode::kPingPong) {
    bytes_moved *= 2.0;
  }
  double gbps = bytes_moved / (elapsed_ms / 1000.0) / 1e9;
  printf("%-12s  %4d  %6d  %6u  %12lu  %6u  %10.2f  %10.3f\n",
         label, my_pe, target_pe, cfg.total_warps(),
         (unsigned long)cfg.bytes_per_warp, cfg.iterations,
         elapsed_ms, gbps);
}

// ============================================================
// main
// ============================================================

int main(int argc, char **argv) {
  // MPI init must come before nvshmem init
  MPI_CHECK(MPI_Init(&argc, &argv));

  // nvshmem init with MPI communicator
  // NVSHMEM 3.x: use nvshmemx_set_attr_mpi_comm_args() helper to populate
  // the attr struct correctly (works across Open MPI / MPICH).
  nvshmemx_init_attr_t attr;
  MPI_Comm comm = MPI_COMM_WORLD;
  nvshmemx_set_attr_mpi_comm_args(&comm, &attr);
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

  int my_pe = nvshmem_my_pe();
  int n_pes = nvshmem_n_pes();

  // PE i uses GPU i
  CUDA_CHECK(cudaSetDevice(my_pe));

  RemoteHbmConfig cfg = parse_remote_hbm_args(argc, argv);

  // Determine target PE: for n_pes == 1, loopback to self
  int target_pe;
  if (n_pes == 1) {
    target_pe = 0;
    if (my_pe == 0) {
      fprintf(stderr, "NOTE: single PE — running loopback mode (PE 0 -> PE 0)\n");
    }
  } else {
    // PE 0 targets PE 1; PE 1 idles (unless ping-pong)
    target_pe = (my_pe == 0) ? 1 : 0;
  }

  // Warn if symmetric heap might be too small
  if (my_pe == 0 && cfg.total_bytes() > 512ULL * 1024 * 1024) {
    fprintf(stderr, "WARNING: total transfer size is %.2f MB. "
            "Ensure NVSHMEM_SYMMETRIC_SIZE is set large enough.\n",
            (double)cfg.total_bytes() / (1024.0 * 1024.0));
  }

  // ---- Allocate symmetric buffer + local staging buffer ----
  char *sym_buf = (char *)nvshmem_malloc(cfg.total_bytes());
  if (!sym_buf) {
    fprintf(stderr, "[PE %d] nvshmem_malloc(%lu) failed — "
            "try increasing NVSHMEM_SYMMETRIC_SIZE\n",
            my_pe, (unsigned long)cfg.total_bytes());
    nvshmem_finalize();
    MPI_Finalize();
    return 1;
  }

  char *local_buf;
  CUDA_CHECK(cudaMalloc(&local_buf, cfg.total_bytes()));

  // Fill local_buf with identifiable pattern
  init_pattern_kernel<<<cfg.blocks, cfg.threads>>>(
      local_buf, cfg.bytes_per_warp, cfg.total_warps(), my_pe);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Also initialize sym_buf (needed for get mode — remote PE must have data)
  init_pattern_kernel<<<cfg.blocks, cfg.threads>>>(
      sym_buf, cfg.bytes_per_warp, cfg.total_warps(), my_pe);
  CUDA_CHECK(cudaDeviceSynchronize());

  nvshmem_barrier_all();

  // ---- Print configuration header (PE 0 only) ----
  if (my_pe == 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, my_pe));

    const char *mode_str = "put";
    if (cfg.mode == NvshmemMode::kGet) mode_str = "get";
    else if (cfg.mode == NvshmemMode::kPingPong) mode_str = "ping-pong";

    printf("=== GPU Baseline Benchmark 2: Remote HBM (NVSHMEM) ===\n");
    printf("Device  : %s (PE %d -> GPU %d)\n", prop.name, my_pe, my_pe);
    printf("Target  : PE %d (GPU %d)\n", target_pe, target_pe);
    printf("PEs     : %d total\n", n_pes);
    printf("Grid    : %u blocks x %u threads = %u warps\n",
           cfg.blocks, cfg.threads, cfg.total_warps());
    printf("Transfer: %lu bytes/warp x %u warps = %.2f MB total\n",
           (unsigned long)cfg.bytes_per_warp,
           cfg.total_warps(),
           (double)cfg.total_bytes() / (1024.0 * 1024.0));
    printf("Timing  : %u warmup + %u timed iterations\n",
           cfg.warmup_iters, cfg.iterations);
    printf("Fence   : %s\n", cfg.no_fence ? "disabled (--no-fence)" : "enabled");
    printf("Mode    : %s\n", mode_str);
    if (cfg.src_numa >= 0 || cfg.dst_numa >= 0) {
      printf("NUMA    : src=%d dst=%d (informational)\n", cfg.src_numa, cfg.dst_numa);
    }
    printf("\n");
  }

  bool use_fence = !cfg.no_fence;

  // Create stream for benchmarking
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  GpuTimer timer;

  // Only PE 0 runs the transfer kernel (PE 1 waits at barriers)
  // Exception: for ping-pong with >1 PE, only PE 0 drives the operation
  bool is_active_pe = (my_pe == 0) || (n_pes == 1);

  // ---- Warmup ----
  if (is_active_pe) {
    if (cfg.mode == NvshmemMode::kPut) {
      remote_hbm_put_kernel<<<cfg.blocks, cfg.threads, 0, stream>>>(
          sym_buf, local_buf, cfg.bytes_per_warp, cfg.total_warps(),
          cfg.warmup_iters, target_pe, use_fence);
    } else if (cfg.mode == NvshmemMode::kGet) {
      remote_hbm_get_kernel<<<cfg.blocks, cfg.threads, 0, stream>>>(
          sym_buf, local_buf, cfg.bytes_per_warp, cfg.total_warps(),
          cfg.warmup_iters, target_pe, use_fence);
    } else {  // ping-pong
      remote_hbm_pingpong_kernel<<<cfg.blocks, cfg.threads, 0, stream>>>(
          sym_buf, local_buf, cfg.bytes_per_warp, cfg.total_warps(),
          cfg.warmup_iters, target_pe, use_fence);
    }
    nvshmemx_quiet_on_stream(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  nvshmem_barrier_all();

  // ---- Timed run ----
  float elapsed_ms = 0.f;
  if (is_active_pe) {
    timer.record_start(stream);

    if (cfg.mode == NvshmemMode::kPut) {
      remote_hbm_put_kernel<<<cfg.blocks, cfg.threads, 0, stream>>>(
          sym_buf, local_buf, cfg.bytes_per_warp, cfg.total_warps(),
          cfg.iterations, target_pe, use_fence);
    } else if (cfg.mode == NvshmemMode::kGet) {
      remote_hbm_get_kernel<<<cfg.blocks, cfg.threads, 0, stream>>>(
          sym_buf, local_buf, cfg.bytes_per_warp, cfg.total_warps(),
          cfg.iterations, target_pe, use_fence);
    } else {  // ping-pong
      remote_hbm_pingpong_kernel<<<cfg.blocks, cfg.threads, 0, stream>>>(
          sym_buf, local_buf, cfg.bytes_per_warp, cfg.total_warps(),
          cfg.iterations, target_pe, use_fence);
    }

    nvshmemx_quiet_on_stream(stream);
    timer.record_stop(stream);
    elapsed_ms = timer.elapsed_ms();
  }

  nvshmem_barrier_all();

  // ---- Print results (PE 0 only) ----
  if (my_pe == 0) {
    print_remote_hbm_result_header();

    const char *label = "put";
    if (cfg.mode == NvshmemMode::kGet) label = "get";
    else if (cfg.mode == NvshmemMode::kPingPong) label = "ping-pong";

    print_remote_hbm_result(label, my_pe, target_pe, cfg, elapsed_ms);
    printf("\n");
  }

  // ---- Validation ----
  if (cfg.validate) {
    // Run a single fenced iteration for validation
    if (is_active_pe) {
      if (cfg.mode == NvshmemMode::kPut || cfg.mode == NvshmemMode::kPingPong) {
        remote_hbm_put_kernel<<<cfg.blocks, cfg.threads, 0, stream>>>(
            sym_buf, local_buf, cfg.bytes_per_warp, cfg.total_warps(),
            1, target_pe, /*use_fence=*/true);
      } else {  // get
        remote_hbm_get_kernel<<<cfg.blocks, cfg.threads, 0, stream>>>(
            sym_buf, local_buf, cfg.bytes_per_warp, cfg.total_warps(),
            1, target_pe, /*use_fence=*/true);
      }
      nvshmemx_quiet_on_stream(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    nvshmem_barrier_all();

    // Allocate host buffer for readback
    char *host_buf = (char *)malloc(cfg.total_bytes());
    if (!host_buf) {
      fprintf(stderr, "[PE %d] malloc failed for validation buffer\n", my_pe);
    } else {
      if (cfg.mode == NvshmemMode::kPut || cfg.mode == NvshmemMode::kPingPong) {
        // After put: target PE checks its sym_buf
        if (my_pe == target_pe || (n_pes == 1 && my_pe == 0)) {
          CUDA_CHECK(cudaMemcpy(host_buf, sym_buf, cfg.total_bytes(),
                                cudaMemcpyDeviceToHost));
          // Pattern was written by PE 0 (the source)
          int errs = validate_buffer_on_host(cfg, host_buf, 0);
          if (errs == 0) {
            printf("[PE %d] validate: PASS\n", my_pe);
          } else {
            printf("[PE %d] validate: FAIL — %d mismatches (first 32 shown)\n",
                   my_pe, errs);
          }
        }
      } else {  // get
        // After get: PE 0 checks its local_buf
        if (my_pe == 0) {
          CUDA_CHECK(cudaMemcpy(host_buf, local_buf, cfg.total_bytes(),
                                cudaMemcpyDeviceToHost));
          // Pattern was written by the target PE
          int errs = validate_buffer_on_host(cfg, host_buf, target_pe);
          if (errs == 0) {
            printf("[PE %d] validate: PASS\n", my_pe);
          } else {
            printf("[PE %d] validate: FAIL — %d mismatches (first 32 shown)\n",
                   my_pe, errs);
          }
        }
      }
      free(host_buf);
    }
  }

  // ---- Cleanup ----
  CUDA_CHECK(cudaStreamDestroy(stream));
  nvshmem_free(sym_buf);
  CUDA_CHECK(cudaFree(local_buf));
  nvshmem_finalize();
  MPI_CHECK(MPI_Finalize());

  return 0;
}

#else  // !HAVE_NVSHMEM

#include <cstdio>

int main(int /*argc*/, char ** /*argv*/) {
  fprintf(stderr,
    "bench_remote_hbm: not built with NVSHMEM support.\n"
    "Rebuild with -DWRP_CORE_ENABLE_NVSHMEM=ON and ensure nvshmem is installed.\n");
  return 1;
}

#endif  // HAVE_NVSHMEM

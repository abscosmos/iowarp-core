/**
 * bench_cross_node_hbm.cu — Benchmark 4: Cross-node HBM bandwidth and latency via NVSHMEM
 *
 * Measures GPU-to-GPU HBM bandwidth and one-way latency across nodes using NVSHMEM
 * over a high-speed interconnect (e.g. InfiniBand via NVSHMEM's ibgda/ibrc transport).
 *
 * Three modes:
 *   put-bw    — PE 0 writes data into PE 1's symmetric buffer (warp-collective NBI put)
 *   get-bw    — PE 0 reads data from PE 1's symmetric buffer (warp-collective NBI get)
 *   latency   — PE 0 sends payload+flag, PE 1 polls flag and records clock64() delta
 *
 * Key design choices:
 *   - BW modes reuse remote_hbm_put_kernel / remote_hbm_get_kernel from
 *     kernels_remote_hbm.cuh (same warp-collective NVSHMEM operations as Benchmark 2).
 *   - Latency mode uses dedicated cross_node_latency_send_kernel /
 *     cross_node_latency_poll_kernel from kernels_cross_node_hbm.cuh.
 *   - nvshmemx_quiet_on_stream() on host after BW kernels for belt-and-suspenders
 *     timing accuracy.
 *   - No HermesShm / Chimaera dependency — raw hardware measurement only.
 *
 * Launch:
 *   mpirun   -n 2 ./gpu_baseline_cross_node_hbm [options]   # cross-node (one rank per node)
 *   nvshmrun -n 2 ./gpu_baseline_cross_node_hbm [options]
 *   nvshmrun -n 1 ./gpu_baseline_cross_node_hbm [options]   # loopback (PE 0 -> PE 0)
 *
 * Usage:
 *   gpu_baseline_cross_node_hbm [--blocks N] [--threads N] [--bytes-per-warp N]
 *                               [--mode put-bw|get-bw|latency]
 *                               [--warmup N] [--iterations N] [--latency-iters N]
 *                               [--gpu-id N]
 *                               [--no-fence] [--validate]
 *                               [--src-numa N] [--dst-numa N]
 */

#ifdef HAVE_NVSHMEM

#include "utils.h"
#include "kernels_remote_hbm.cuh"
#include "kernels_cross_node_hbm.cuh"
#include <nvshmem.h>
#include <nvshmemx.h>
#include <mpi.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

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

// Warmup samples discarded in latency mode (not a CLI flag)
static constexpr uint32_t kLatencyWarmup = 10;

enum class CrossNodeMode { kPutBw, kGetBw, kLatency };

struct CrossNodeConfig {
  uint32_t      blocks         = 1;
  uint32_t      threads        = 32;
  uint64_t      bytes_per_warp = 4096;
  uint32_t      warmup_iters   = 5;
  uint32_t      iterations     = 100;
  uint32_t      latency_iters  = 1000;
  bool          validate       = false;
  bool          no_fence       = false;
  CrossNodeMode mode           = CrossNodeMode::kPutBw;
  int           gpu_id         = 0;
  int           src_numa       = -1;   // informational only
  int           dst_numa       = -1;   // informational only
  bool          bpw_explicit   = false;

  uint32_t total_warps() const { return (blocks * threads) / 32; }
  uint64_t total_bytes() const { return (uint64_t)total_warps() * bytes_per_warp; }
};

static inline void print_cross_node_usage(const char *prog) {
  fprintf(stderr,
    "Usage: %s [options]\n"
    "  --blocks N            Grid blocks (default: 1)\n"
    "  --threads N           Threads per block, must be multiple of 32 (default: 32)\n"
    "  --bytes-per-warp N    Bytes each warp transfers, multiple of 16 (default: 4096 for bw, 64 for latency)\n"
    "                        Accepts k/m/g suffix (e.g. 1m)\n"
    "  --mode put-bw|get-bw|latency  Transfer mode (default: put-bw)\n"
    "  --warmup N            Warmup iterations for BW modes (default: 5)\n"
    "  --iterations N        Timed iterations for BW modes (default: 100)\n"
    "  --latency-iters N     Timed samples for latency mode (default: 1000)\n"
    "  --gpu-id N            CUDA device ID (default: 0)\n"
    "  --no-fence            Skip per-iteration nvshmem_quiet() (BW pipeline mode)\n"
    "  --validate            Write pattern, quiet, read back and check (BW modes only)\n"
    "  --src-numa N          Source NUMA node (informational only)\n"
    "  --dst-numa N          Destination NUMA node (informational only)\n"
    "  --help                Show this message\n"
    "\n"
    "  For large transfers, set NVSHMEM_SYMMETRIC_SIZE env var (e.g. 1073741824 for 1 GB).\n"
    "  Launch with: mpirun -n 2 %s [options]   (one rank per node)\n"
    "               nvshmrun -n 1 %s [options]  (loopback mode)\n",
    prog, prog, prog);
}

static inline CrossNodeConfig parse_cross_node_args(int argc, char **argv) {
  CrossNodeConfig cfg;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      print_cross_node_usage(argv[0]);
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
    } else if (strcmp(argv[i], "--gpu-id") == 0 && i + 1 < argc) {
      cfg.gpu_id = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--src-numa") == 0 && i + 1 < argc) {
      cfg.src_numa = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--dst-numa") == 0 && i + 1 < argc) {
      cfg.dst_numa = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--validate") == 0) {
      cfg.validate = true;
    } else if (strcmp(argv[i], "--no-fence") == 0) {
      cfg.no_fence = true;
    } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
      ++i;
      if (strcmp(argv[i], "put-bw") == 0)        cfg.mode = CrossNodeMode::kPutBw;
      else if (strcmp(argv[i], "get-bw") == 0)   cfg.mode = CrossNodeMode::kGetBw;
      else if (strcmp(argv[i], "latency") == 0)  cfg.mode = CrossNodeMode::kLatency;
      else {
        fprintf(stderr, "Unknown mode '%s'. Use put-bw, get-bw, or latency.\n", argv[i]);
        exit(1);
      }
    } else {
      fprintf(stderr, "Unknown argument: %s\n", argv[i]);
      print_cross_node_usage(argv[0]);
      exit(1);
    }
  }

  // Default bytes_per_warp for latency mode if not explicitly set
  if (cfg.mode == CrossNodeMode::kLatency && !cfg.bpw_explicit) {
    cfg.bytes_per_warp = 64;
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
// Latency statistics
// ============================================================

struct LatencyStats {
  double min_us;
  double median_us;
  double mean_us;
  double max_us;
};

static LatencyStats compute_latency_stats(const uint64_t *cycle_deltas,
                                          uint32_t n_samples,
                                          int device_id) {
  int clock_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device_id));
  double us_per_cycle = 1000.0 / (double)clock_khz;

  std::vector<double> us_vals(n_samples);
  double sum = 0.0;
  for (uint32_t i = 0; i < n_samples; ++i) {
    us_vals[i] = (double)cycle_deltas[i] * us_per_cycle;
    sum += us_vals[i];
  }

  std::sort(us_vals.begin(), us_vals.end());

  LatencyStats stats;
  stats.min_us    = us_vals.front();
  stats.max_us    = us_vals.back();
  stats.mean_us   = sum / (double)n_samples;
  stats.median_us = (n_samples % 2 == 0)
                      ? (us_vals[n_samples / 2 - 1] + us_vals[n_samples / 2]) * 0.5
                      : us_vals[n_samples / 2];
  return stats;
}

// ============================================================
// Validation (host-side): check pattern in symmetric buffer
// ============================================================

static int validate_sym_buf(const CrossNodeConfig &cfg,
                             const char *host_buf, int pe_id) {
  uint32_t total_warps = cfg.total_warps();
  int errors = 0;
  for (uint32_t w = 0; w < total_warps && errors < 32; ++w) {
    const uint4 *row = reinterpret_cast<const uint4 *>(
                           host_buf + (uint64_t)w * cfg.bytes_per_warp);
    uint32_t num_u4 = (uint32_t)(cfg.bytes_per_warp / sizeof(uint4));
    for (uint32_t i = 0; i < num_u4 && errors < 32; ++i) {
      uint4 expected;
      expected.x = (uint32_t)pe_id;
      expected.y = w;
      expected.z = 0xDEADBEEFu;
      expected.w = 0xCAFEBABEu;
      uint4 got = row[i];
      if (got.x != expected.x || got.y != expected.y ||
          got.z != expected.z || got.w != expected.w) {
        if (errors == 0) {
          fprintf(stderr, "[PE %d] VALIDATE FAIL: first mismatch at warp=%u elem=%u\n"
                          "  expected {%u,%u,%u,%u} got {%u,%u,%u,%u}\n",
                  pe_id, w, i,
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
// main
// ============================================================

int main(int argc, char **argv) {
  // MPI init must come before nvshmem init
  MPI_CHECK(MPI_Init(&argc, &argv));

  CrossNodeConfig cfg = parse_cross_node_args(argc, argv);

  // Set device before NVSHMEM init so NVSHMEM runtime uses the correct GPU.
  // --gpu-id selects the device per node (default 0).
  CUDA_CHECK(cudaSetDevice(cfg.gpu_id));

  // nvshmem init with MPI communicator
  // NVSHMEM 3.x: use nvshmemx_set_attr_mpi_comm_args() helper to populate
  // the attr struct correctly (works across Open MPI / MPICH).
  nvshmemx_init_attr_t attr;
  MPI_Comm comm = MPI_COMM_WORLD;
  nvshmemx_set_attr_mpi_comm_args(&comm, &attr);
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

  int my_pe = nvshmem_my_pe();
  int n_pes = nvshmem_n_pes();

  // Determine target PE: for n_pes == 1, loopback to self
  if (n_pes == 1) {
    if (my_pe == 0) {
      fprintf(stderr, "NOTE: single PE — running loopback mode (PE 0 -> PE 0)\n");
    }
  }
  int target_pe = (n_pes == 1) ? 0 : ((my_pe == 0) ? 1 : 0);

  // Warn if symmetric heap might be too small
  {
    uint64_t sym_needed;
    if (cfg.mode == CrossNodeMode::kLatency) {
      uint32_t total_lat = kLatencyWarmup + cfg.latency_iters;
      sym_needed = cfg.bytes_per_warp * (uint64_t)total_lat   // payload slots
                 + sizeof(uint32_t) * total_lat;              // flag array
    } else {
      sym_needed = cfg.total_bytes();
    }
    if (my_pe == 0 && sym_needed > 512ULL * 1024 * 1024) {
      fprintf(stderr, "WARNING: symmetric allocation %.2f MB — "
              "set NVSHMEM_SYMMETRIC_SIZE if init fails.\n",
              (double)sym_needed / (1024.0 * 1024.0));
    }
  }

  // Create stream for benchmarking
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // ============================================================
  // BW path (put-bw or get-bw)
  // ============================================================
  if (cfg.mode == CrossNodeMode::kPutBw || cfg.mode == CrossNodeMode::kGetBw) {
    // Allocate symmetric and local buffers
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

    // Print configuration header (PE 0 only)
    if (my_pe == 0) {
      cudaDeviceProp prop;
      CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.gpu_id));

      const char *mode_str = (cfg.mode == CrossNodeMode::kPutBw) ? "put-bw" : "get-bw";

      printf("=== GPU Baseline Benchmark 4: Cross-node HBM (NVSHMEM) — BW mode ===\n");
      printf("Device  : %s (PE %d -> GPU %d)\n", prop.name, my_pe, cfg.gpu_id);
      printf("Target  : PE %d\n", target_pe);
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
    // Only PE 0 runs the transfer kernel (PE 1 waits at barriers)
    bool is_active = (my_pe == 0) || (n_pes == 1);

    // Warmup
    if (is_active) {
      if (cfg.mode == CrossNodeMode::kPutBw) {
        remote_hbm_put_kernel<<<cfg.blocks, cfg.threads, 0, stream>>>(
            sym_buf, local_buf, cfg.bytes_per_warp, cfg.total_warps(),
            cfg.warmup_iters, target_pe, use_fence);
      } else {
        remote_hbm_get_kernel<<<cfg.blocks, cfg.threads, 0, stream>>>(
            sym_buf, local_buf, cfg.bytes_per_warp, cfg.total_warps(),
            cfg.warmup_iters, target_pe, use_fence);
      }
      nvshmemx_quiet_on_stream(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    nvshmem_barrier_all();

    // Timed run
    GpuTimer timer;
    float elapsed_ms = 0.f;
    if (is_active) {
      timer.record_start(stream);

      if (cfg.mode == CrossNodeMode::kPutBw) {
        remote_hbm_put_kernel<<<cfg.blocks, cfg.threads, 0, stream>>>(
            sym_buf, local_buf, cfg.bytes_per_warp, cfg.total_warps(),
            cfg.iterations, target_pe, use_fence);
      } else {
        remote_hbm_get_kernel<<<cfg.blocks, cfg.threads, 0, stream>>>(
            sym_buf, local_buf, cfg.bytes_per_warp, cfg.total_warps(),
            cfg.iterations, target_pe, use_fence);
      }

      nvshmemx_quiet_on_stream(stream);
      timer.record_stop(stream);
      elapsed_ms = timer.elapsed_ms();
    }

    nvshmem_barrier_all();

    // Print results (PE 0 only)
    if (my_pe == 0) {
      const char *label = (cfg.mode == CrossNodeMode::kPutBw) ? "put-bw" : "get-bw";

      // Header
      printf("%-12s  %4s  %6s  %6s  %12s  %6s  %10s  %10s\n",
             "Mode", "PE", "Target", "Warps", "bytes/warp", "Iters",
             "Time(ms)", "BW(GB/s)");
      printf("%-12s  %4s  %6s  %6s  %12s  %6s  %10s  %10s\n",
             "------------", "----", "------", "------", "------------",
             "------", "----------", "----------");

      double bytes_moved = (double)cfg.total_bytes() * cfg.iterations;
      double gbps = bytes_moved / (elapsed_ms / 1000.0) / 1e9;
      printf("%-12s  %4d  %6d  %6u  %12lu  %6u  %10.2f  %10.3f\n",
             label, my_pe, target_pe, cfg.total_warps(),
             (unsigned long)cfg.bytes_per_warp, cfg.iterations,
             elapsed_ms, gbps);
      printf("\n");
    }

    // Validation
    if (cfg.validate) {
      // Run a single fenced iteration for validation
      if (is_active) {
        if (cfg.mode == CrossNodeMode::kPutBw) {
          remote_hbm_put_kernel<<<cfg.blocks, cfg.threads, 0, stream>>>(
              sym_buf, local_buf, cfg.bytes_per_warp, cfg.total_warps(),
              1, target_pe, /*use_fence=*/true);
        } else {
          remote_hbm_get_kernel<<<cfg.blocks, cfg.threads, 0, stream>>>(
              sym_buf, local_buf, cfg.bytes_per_warp, cfg.total_warps(),
              1, target_pe, /*use_fence=*/true);
        }
        nvshmemx_quiet_on_stream(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }

      nvshmem_barrier_all();

      char *host_buf = (char *)malloc(cfg.total_bytes());
      if (!host_buf) {
        fprintf(stderr, "[PE %d] malloc failed for validation buffer\n", my_pe);
      } else {
        if (cfg.mode == CrossNodeMode::kPutBw) {
          // After put: target PE checks its sym_buf
          if (my_pe == target_pe || (n_pes == 1 && my_pe == 0)) {
            CUDA_CHECK(cudaMemcpy(host_buf, sym_buf, cfg.total_bytes(),
                                  cudaMemcpyDeviceToHost));
            // Pattern was written by PE 0 (the source)
            int errs = validate_sym_buf(cfg, host_buf, 0);
            if (errs == 0) {
              printf("[PE %d] validate: PASS\n", my_pe);
            } else {
              printf("[PE %d] validate: FAIL — %d mismatches (first 32 shown)\n",
                     my_pe, errs);
            }
          }
        } else {  // get-bw
          // After get: PE 0 checks its local_buf
          if (my_pe == 0) {
            CUDA_CHECK(cudaMemcpy(host_buf, local_buf, cfg.total_bytes(),
                                  cudaMemcpyDeviceToHost));
            // Pattern was written by the target PE
            int errs = validate_sym_buf(cfg, host_buf, target_pe);
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

    nvshmem_free(sym_buf);
    CUDA_CHECK(cudaFree(local_buf));
  }

  // ============================================================
  // Latency path
  // ============================================================
  else {
    uint32_t total_lat = kLatencyWarmup + cfg.latency_iters;

    // Symmetric flag array: one slot per sample (sender writes, receiver polls)
    uint32_t *sym_flag = (uint32_t *)nvshmem_malloc(
        (size_t)total_lat * sizeof(uint32_t));
    if (!sym_flag) {
      fprintf(stderr, "[PE %d] nvshmem_malloc(flags, %lu) failed — "
              "try increasing NVSHMEM_SYMMETRIC_SIZE\n",
              my_pe, (unsigned long)(total_lat * sizeof(uint32_t)));
      nvshmem_finalize();
      MPI_Finalize();
      return 1;
    }

    // Symmetric payload buffer: one slot per sample
    char *sym_payload = (char *)nvshmem_malloc(
        (size_t)cfg.bytes_per_warp * total_lat);
    if (!sym_payload) {
      fprintf(stderr, "[PE %d] nvshmem_malloc(payload, %lu) failed — "
              "try increasing NVSHMEM_SYMMETRIC_SIZE\n",
              my_pe, (unsigned long)(cfg.bytes_per_warp * total_lat));
      nvshmem_free(sym_flag);
      nvshmem_finalize();
      MPI_Finalize();
      return 1;
    }

    // Local source buffer (sender side)
    char *local_src;
    CUDA_CHECK(cudaMalloc(&local_src, cfg.bytes_per_warp));

    // Result buffers (receiver side only)
    bool is_sender   = (my_pe == 0) || (n_pes == 1);
    bool is_receiver = (my_pe == 1) || (n_pes == 1);

    uint64_t *result_buf  = nullptr;
    uint64_t *result_host = nullptr;
    if (is_receiver) {
      CUDA_CHECK(cudaMalloc(&result_buf, (size_t)cfg.latency_iters * sizeof(uint64_t)));
      result_host = (uint64_t *)malloc((size_t)cfg.latency_iters * sizeof(uint64_t));
      if (!result_host) {
        fprintf(stderr, "[PE %d] malloc failed for latency result buffer\n", my_pe);
        nvshmem_free(sym_flag);
        nvshmem_free(sym_payload);
        CUDA_CHECK(cudaFree(local_src));
        CUDA_CHECK(cudaFree(result_buf));
        nvshmem_finalize();
        MPI_Finalize();
        return 1;
      }
    }

    // Zero flags before the experiment so receiver doesn't see stale data
    CUDA_CHECK(cudaMemset(sym_flag, 0, (size_t)total_lat * sizeof(uint32_t)));
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    // Print configuration header (PE 0 only)
    if (my_pe == 0) {
      cudaDeviceProp prop;
      CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.gpu_id));

      printf("=== GPU Baseline Benchmark 4: Cross-node HBM (NVSHMEM) — Latency mode ===\n");
      printf("Device  : %s (PE %d -> GPU %d)\n", prop.name, my_pe, cfg.gpu_id);
      printf("Target  : PE %d\n", target_pe);
      printf("PEs     : %d total\n", n_pes);
      printf("Payload : %lu bytes/sample\n", (unsigned long)cfg.bytes_per_warp);
      printf("Samples : %u warmup + %u timed\n", kLatencyWarmup, cfg.latency_iters);
      if (cfg.src_numa >= 0 || cfg.dst_numa >= 0) {
        printf("NUMA    : src=%d dst=%d (informational)\n", cfg.src_numa, cfg.dst_numa);
      }
      printf("\n");
    }

    if (my_pe == 0 && n_pes == 1) {
      fprintf(stderr,
        "NOTE: latency loopback (1 PE) — send and poll run sequentially on the same stream;\n"
        "      flags are pre-set when poll kernel starts. Latency results reflect device overhead,\n"
        "      not cross-node IB latency. Use mpirun -n 2 for real measurements.\n");
    }

    // MPI_Barrier ensures PE 1 poll kernel is ready before PE 0 sends
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (is_sender) {
      cross_node_latency_send_kernel<<<1, 32, 0, stream>>>(
          sym_payload, sym_flag, local_src,
          cfg.bytes_per_warp, kLatencyWarmup, cfg.latency_iters, target_pe);
    }
    if (is_receiver) {
      cross_node_latency_poll_kernel<<<1, 32, 0, stream>>>(
          (volatile uint32_t *)sym_flag, result_buf,
          kLatencyWarmup, cfg.latency_iters);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Print latency results (receiver side)
    if (is_receiver && result_buf != nullptr) {
      CUDA_CHECK(cudaMemcpy(result_host, result_buf,
                            (size_t)cfg.latency_iters * sizeof(uint64_t),
                            cudaMemcpyDeviceToHost));

      LatencyStats stats = compute_latency_stats(result_host, cfg.latency_iters, cfg.gpu_id);

      printf("Cross-node one-way latency (send-notify, %lu B payload)\n",
             (unsigned long)cfg.bytes_per_warp);
      printf("  Samples : %u\n",   cfg.latency_iters);
      printf("  Min     : %6.2f us\n", stats.min_us);
      printf("  Median  : %6.2f us\n", stats.median_us);
      printf("  Mean    : %6.2f us\n", stats.mean_us);
      printf("  Max     : %6.2f us\n", stats.max_us);
    }

    // Cleanup latency resources
    nvshmem_free(sym_flag);
    nvshmem_free(sym_payload);
    CUDA_CHECK(cudaFree(local_src));
    if (result_buf)  { CUDA_CHECK(cudaFree(result_buf)); }
    if (result_host) { free(result_host); }
  }

  // ============================================================
  // Shared cleanup
  // ============================================================
  CUDA_CHECK(cudaStreamDestroy(stream));
  nvshmem_finalize();
  MPI_CHECK(MPI_Finalize());

  return 0;
}

#else  // !HAVE_NVSHMEM

#include <cstdio>

int main(int /*argc*/, char ** /*argv*/) {
  fprintf(stderr,
    "bench_cross_node_hbm: not built with NVSHMEM support.\n"
    "Rebuild with -DWRP_CORE_ENABLE_NVSHMEM=ON and ensure nvshmem is installed.\n");
  return 1;
}

#endif  // HAVE_NVSHMEM

/**
 * utils.h — Shared utilities for gpu_baseline benchmarks
 *
 * Standalone header: no HermesShm, Chimaera, or other iowarp dependencies.
 * Include directly in benchmark .cu files.
 */
#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <vector>

#ifdef HAVE_LIBNUMA
#include <numa.h>
#endif

// ============================================================
// CUDA error checking
// ============================================================

#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t _err = (call);                                            \
    if (_err != cudaSuccess) {                                            \
      fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
              __FILE__, __LINE__, cudaGetErrorString(_err));              \
      exit(1);                                                            \
    }                                                                     \
  } while (0)

// ============================================================
// Size parser: accepts "4096", "4k", "1m", "2g" (case-insensitive)
// ============================================================

static inline uint64_t parse_size(const char *s) {
  double val = atof(s);
  const char *p = s;
  while (*p && (isdigit((unsigned char)*p) || *p == '.')) ++p;
  switch (tolower((unsigned char)*p)) {
    case 'k': return (uint64_t)(val * 1024ULL);
    case 'm': return (uint64_t)(val * 1024ULL * 1024ULL);
    case 'g': return (uint64_t)(val * 1024ULL * 1024ULL * 1024ULL);
    default:  return (uint64_t)val;
  }
}

// ============================================================
// Benchmark configuration
// ============================================================

enum class TransferMode { kWrite, kRead, kReadWrite };

struct BenchConfig {
  uint32_t     blocks        = 1;
  uint32_t     threads       = 32;      // must be a multiple of 32
  uint64_t     bytes_per_warp = 4096;   // must be a multiple of 16
  int          numa_node     = 0;
  int          gpu_id        = 0;
  uint32_t     warmup_iters  = 5;
  uint32_t     iterations    = 100;
  bool         validate      = false;
  bool         no_fence      = false;   // skip __threadfence_system() for comparison
  TransferMode mode          = TransferMode::kWrite;

  uint32_t total_warps() const { return (blocks * threads) / 32; }
  uint64_t total_bytes() const { return (uint64_t)total_warps() * bytes_per_warp; }
};

static inline void print_usage(const char *prog) {
  fprintf(stderr,
    "Usage: %s [options]\n"
    "  --blocks N           Grid blocks (default: 1)\n"
    "  --threads N          Threads per block, must be multiple of 32 (default: 32)\n"
    "  --bytes-per-warp N   Bytes each warp transfers, multiple of 16 (default: 4096)\n"
    "                       Accepts k/m/g suffix (e.g. 1m)\n"
    "  --numa-node N        NUMA node for pinned allocation (default: 0)\n"
    "  --gpu-id N           CUDA device ID (default: 0)\n"
    "  --warmup N           Warmup iterations (default: 5)\n"
    "  --iterations N       Timed iterations (default: 100)\n"
    "  --mode write|read|readwrite  Transfer direction (default: write)\n"
    "  --validate           Read back buffer and check pattern after write\n"
    "  --no-fence           Skip __threadfence_system() (may show inflated BW)\n"
    "  --help               Show this message\n",
    prog);
}

static inline BenchConfig parse_args(int argc, char **argv) {
  BenchConfig cfg;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      print_usage(argv[0]);
      exit(0);
    } else if (strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) {
      cfg.blocks = (uint32_t)atoi(argv[++i]);
    } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      cfg.threads = (uint32_t)atoi(argv[++i]);
    } else if (strcmp(argv[i], "--bytes-per-warp") == 0 && i + 1 < argc) {
      cfg.bytes_per_warp = parse_size(argv[++i]);
    } else if (strcmp(argv[i], "--numa-node") == 0 && i + 1 < argc) {
      cfg.numa_node = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--gpu-id") == 0 && i + 1 < argc) {
      cfg.gpu_id = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      cfg.warmup_iters = (uint32_t)atoi(argv[++i]);
    } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
      cfg.iterations = (uint32_t)atoi(argv[++i]);
    } else if (strcmp(argv[i], "--validate") == 0) {
      cfg.validate = true;
    } else if (strcmp(argv[i], "--no-fence") == 0) {
      cfg.no_fence = true;
    } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
      ++i;
      if (strcmp(argv[i], "write") == 0)          cfg.mode = TransferMode::kWrite;
      else if (strcmp(argv[i], "read") == 0)       cfg.mode = TransferMode::kRead;
      else if (strcmp(argv[i], "readwrite") == 0)  cfg.mode = TransferMode::kReadWrite;
      else {
        fprintf(stderr, "Unknown mode '%s'. Use write, read, or readwrite.\n", argv[i]);
        exit(1);
      }
    } else {
      fprintf(stderr, "Unknown argument: %s\n", argv[i]);
      print_usage(argv[0]);
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
// CUDA event timer (RAII)
// ============================================================

struct GpuTimer {
  cudaEvent_t start_, stop_;

  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  // Non-copyable
  GpuTimer(const GpuTimer &) = delete;
  GpuTimer &operator=(const GpuTimer &) = delete;

  void record_start(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(start_, s)); }
  void record_stop (cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(stop_,  s)); }

  float elapsed_ms() {
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }
};

// ============================================================
// NUMA-aware pinned memory buffer (RAII)
// ============================================================

class PinnedBuffer {
  void  *ptr_            = nullptr;
  void  *dev_ptr_        = nullptr;  // GPU-accessible pointer (may differ for registered mem)
  size_t size_           = 0;
  bool   numa_allocated_ = false;

public:
  PinnedBuffer() = default;
  ~PinnedBuffer() { free(); }

  // Non-copyable, non-movable for simplicity
  PinnedBuffer(const PinnedBuffer &) = delete;
  PinnedBuffer &operator=(const PinnedBuffer &) = delete;

  void allocate(size_t size, int numa_node) {
    size_ = size;
#ifdef HAVE_LIBNUMA
    if (numa_available() >= 0) {
      ptr_ = numa_alloc_onnode(size, numa_node);
      if (!ptr_) {
        fprintf(stderr, "numa_alloc_onnode(%zu, %d) failed\n", size, numa_node);
        exit(1);
      }
      cudaError_t err = cudaHostRegister(ptr_, size,
                                         cudaHostRegisterPortable |
                                         cudaHostRegisterMapped);
      if (err != cudaSuccess) {
        fprintf(stderr, "cudaHostRegister failed after numa_alloc_onnode: %s\n"
                        "  Hint: try a smaller --bytes-per-warp or fewer warps.\n",
                cudaGetErrorString(err));
        numa_free(ptr_, size);
        exit(1);
      }
      // Obtain the GPU-visible pointer for use inside kernels.
      // cudaHostRegister memory may have a different device VA than the host VA.
      CUDA_CHECK(cudaHostGetDevicePointer(&dev_ptr_, ptr_, 0));
      numa_allocated_ = true;
    } else {
      // NUMA lib present but no NUMA hardware — fall through to cudaMallocHost
      if (numa_node != 0) {
        fprintf(stderr, "WARNING: NUMA hardware not available, ignoring --numa-node %d\n",
                numa_node);
      }
      CUDA_CHECK(cudaMallocHost(&ptr_, size));
      dev_ptr_ = ptr_;  // cudaMallocHost: host and device VAs are identical
      numa_allocated_ = false;
    }
#else
    if (numa_node != 0) {
      fprintf(stderr, "WARNING: libnuma not compiled in — ignoring --numa-node %d\n",
              numa_node);
    }
    CUDA_CHECK(cudaMallocHost(&ptr_, size));
    dev_ptr_ = ptr_;  // cudaMallocHost: host and device VAs are identical
    numa_allocated_ = false;
#endif
    memset(ptr_, 0, size);
  }

  void free() {
    if (!ptr_) return;
#ifdef HAVE_LIBNUMA
    if (numa_allocated_) {
      cudaHostUnregister(ptr_);
      numa_free(ptr_, size_);
    } else {
      cudaFreeHost(ptr_);
    }
#else
    cudaFreeHost(ptr_);
#endif
    ptr_     = nullptr;
    dev_ptr_ = nullptr;
    size_    = 0;
    numa_allocated_ = false;
  }

  // CPU-side host pointer (use for memset, memcpy, host-side validation)
  template <typename T = void>
  T *get() { return static_cast<T *>(ptr_); }

  // GPU-side device pointer (use for kernel arguments)
  template <typename T = void>
  T *dev_get() { return static_cast<T *>(dev_ptr_); }

  size_t size() const { return size_; }
};

// ============================================================
// MPI error checking (only active when <mpi.h> has been included)
// ============================================================

#ifdef MPI_VERSION
#ifndef GPU_BASELINE_MPI_CHECK_DEFINED_
#define GPU_BASELINE_MPI_CHECK_DEFINED_

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

#endif  // GPU_BASELINE_MPI_CHECK_DEFINED_
#endif  // MPI_VERSION

// ============================================================
// Latency statistics (shared by bench 4 and bench 5)
// ============================================================

struct LatencyStats {
  double min_us;
  double median_us;
  double mean_us;
  double max_us;
};

static inline LatencyStats compute_latency_stats(const uint64_t *cycle_deltas,
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
// Result reporting
// ============================================================

static inline void print_result_header() {
  printf("%-12s  %4s  %4s  %6s  %12s  %6s  %10s  %10s\n",
         "Mode", "GPU", "NUMA", "Warps", "bytes/warp", "Iters",
         "Time(ms)", "BW(GB/s)");
  printf("%-12s  %4s  %4s  %6s  %12s  %6s  %10s  %10s\n",
         "------------", "----", "----", "------", "------------",
         "------", "----------", "----------");
}

static inline void print_result(const char *label, const BenchConfig &cfg,
                                float elapsed_ms) {
  double bytes_moved = (double)cfg.total_bytes() * cfg.iterations;
  double gbps = bytes_moved / (elapsed_ms / 1000.0) / 1e9;
  printf("%-12s  %4d  %4d  %6u  %12lu  %6u  %10.2f  %10.3f\n",
         label, cfg.gpu_id, cfg.numa_node, cfg.total_warps(),
         (unsigned long)cfg.bytes_per_warp, cfg.iterations,
         elapsed_ms, gbps);
}

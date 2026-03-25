/**
 * bench_common.h — Shared types for GPU workload benchmarks
 *
 * Each workload (PageRank, GNN, Gray-Scott, LLM KV cache) implements
 * three modes:
 *   - bam:    GPU reads/writes through BaM HBM page cache from DRAM
 *   - direct: GPU reads/writes pinned DRAM directly over PCIe
 *   - hbm:    GPU reads/writes HBM only (performance ceiling)
 *
 * CTE mode (AsyncGetBlob/AsyncPutBlob) requires the Chimaera runtime
 * and is added via the wrp_cte_gpu_bench infrastructure.
 */
#ifndef BENCH_GPU_COMMON_H
#define BENCH_GPU_COMMON_H

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <string>

#define CUDA_CHECK(call)                                              \
  do {                                                                \
    cudaError_t err = (call);                                        \
    if (err != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
              __FILE__, __LINE__, cudaGetErrorString(err));          \
      exit(1);                                                       \
    }                                                                \
  } while (0)

enum class BenchMode {
  kBam,     // BaM HBM page cache over DRAM
  kDirect,  // Direct pinned DRAM access
  kHbm,     // Full HBM (ceiling)
};

struct BenchResult {
  const char *workload;
  const char *mode;
  double elapsed_ms;
  double primary_metric;       // Workload-specific (edges/sec, nodes/sec, etc.)
  const char *metric_name;
  double bandwidth_gbps;
};

static inline void print_result(const BenchResult &r) {
  printf("  %-12s %-8s %10.2f ms  %10.3f GB/s  %10.2e %s\n",
         r.workload, r.mode, r.elapsed_ms, r.bandwidth_gbps,
         r.primary_metric, r.metric_name);
}

static inline uint64_t parse_size(const char *s) {
  double val = atof(s);
  const char *p = s;
  while (*p && (isdigit(*p) || *p == '.')) p++;
  switch (tolower(*p)) {
    case 'k': return (uint64_t)(val * 1024);
    case 'm': return (uint64_t)(val * 1024 * 1024);
    case 'g': return (uint64_t)(val * 1024ULL * 1024 * 1024);
    default:  return (uint64_t)val;
  }
}

#endif  // BENCH_GPU_COMMON_H

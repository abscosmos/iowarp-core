/**
 * bench_pinned_host.cu — Benchmark 1: GPU → pinned host memory bandwidth
 *
 * Measures the PCIe bandwidth (and NUMA sensitivity) of GPU kernels writing
 * to / reading from cudaMallocHost (or NUMA-pinned) memory.
 *
 * Three modes:
 *   write     — GPU stores data into pinned host buffer  (GPU→Host, PCIe write)
 *   read      — GPU loads data from pinned host buffer   (Host→GPU, PCIe read)
 *   readwrite — GPU reads from src buffer, writes to dst (bidirectional)
 *               NOTE: this mode does NOT measure bidirectional PCIe bandwidth.
 *               Each per-thread store depends on its load result (both over
 *               PCIe), creating a serialized dependency chain with ~2-3 µs
 *               round-trip latency per element.  Use write/read modes for
 *               meaningful bandwidth numbers; see kernel comment for details.
 *
 * Key design choices:
 *   • uint4 (128-bit) stores/loads — 32 lanes × 16 B = 512 B/warp/cycle,
 *     fully coalesced, optimal for PCIe write-combining.
 *   • __threadfence_system() once per iteration — flushes the GPU's L2
 *     write-back buffer to the PCIe bus so timing reflects real transfers.
 *     Skippable with --no-fence for comparison (may show inflated BW).
 *   • Pattern encodes warp_id + lane_id so --validate can catch corruption.
 *   • No HermesShm / Chimaera dependency — raw hardware measurement only.
 *
 * Usage:
 *   gpu_baseline_pinned_host [--blocks N] [--threads N] [--bytes-per-warp N]
 *                            [--numa-node N] [--gpu-id N]
 *                            [--warmup N] [--iterations N]
 *                            [--mode write|read|readwrite]
 *                            [--validate] [--no-fence]
 */

#include "utils.h"
#include "kernels_pinned_host.cuh"
#include <cstdint>
#include <cstdio>

// ============================================================
// Validation (host-side): check write-kernel pattern
// ============================================================

static int validate_write_buffer(const BenchConfig &cfg, const char *buf) {
  uint32_t total_warps = cfg.total_warps();
  int errors = 0;
  for (uint32_t w = 0; w < total_warps && errors < 32; ++w) {
    const uint4 *row = reinterpret_cast<const uint4 *>(
                           buf + (uint64_t)w * cfg.bytes_per_warp);
    uint32_t num_u4 = (uint32_t)(cfg.bytes_per_warp / sizeof(uint4));
    for (uint32_t i = 0; i < num_u4 && errors < 32; ++i) {
      uint32_t lane_id = i % 32;      // lane that wrote element i
      uint4 expected;
      expected.x = w;
      expected.y = lane_id;
      expected.z = 0xDEADBEEFu;
      expected.w = 0xCAFEBABEu;
      uint4 got = row[i];
      if (got.x != expected.x || got.y != expected.y ||
          got.z != expected.z || got.w != expected.w) {
        if (errors == 0) {
          fprintf(stderr, "VALIDATE FAIL: first mismatch at warp=%u elem=%u\n"
                          "  expected {%u,%u,%u,%u} got {%u,%u,%u,%u}\n",
                  w, i,
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
// Timed launch helper
// ============================================================

template <typename KernelFn>
static float timed_launch(KernelFn launch_fn, const BenchConfig &cfg,
                          uint32_t iters) {
  GpuTimer timer;
  timer.record_start();
  launch_fn(iters);
  timer.record_stop();
  CUDA_CHECK(cudaDeviceSynchronize());
  return timer.elapsed_ms();
}

// ============================================================
// main
// ============================================================

int main(int argc, char **argv) {
  BenchConfig cfg = parse_args(argc, argv);

  CUDA_CHECK(cudaSetDevice(cfg.gpu_id));

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.gpu_id));

  // ---- Print configuration header ----------------------------------------
  printf("=== GPU Baseline Benchmark 1: Pinned Host Memory ===\n");
  printf("Device  : %s (GPU %d)\n", prop.name, cfg.gpu_id);
#ifdef HAVE_LIBNUMA
  printf("NUMA    : node %d (libnuma: enabled)\n", cfg.numa_node);
#else
  printf("NUMA    : node %d (libnuma: not compiled — using cudaMallocHost)\n",
         cfg.numa_node);
#endif
  printf("Grid    : %u blocks x %u threads = %u warps\n",
         cfg.blocks, cfg.threads, cfg.total_warps());
  printf("Transfer: %lu bytes/warp  x  %u warps  =  %.2f MB total\n",
         (unsigned long)cfg.bytes_per_warp,
         cfg.total_warps(),
         (double)cfg.total_bytes() / (1024.0 * 1024.0));
  printf("Timing  : %u warmup + %u timed iterations\n",
         cfg.warmup_iters, cfg.iterations);
  printf("Fence   : %s\n", cfg.no_fence ? "disabled (--no-fence)" : "enabled");
  printf("\n");

  bool use_fence = !cfg.no_fence;

  // ---- Allocate buffers --------------------------------------------------
  PinnedBuffer write_buf, read_buf;
  write_buf.allocate(cfg.total_bytes(), cfg.numa_node);
  // read_buf only needed for kRead / kReadWrite — allocate lazily below

  // ---- Print result header -----------------------------------------------
  print_result_header();

  // ============================================================
  // Mode: write (GPU → pinned host)
  // ============================================================
  if (cfg.mode == TransferMode::kWrite) {

    // Warmup
    pinned_host_write_kernel<<<cfg.blocks, cfg.threads>>>(
        write_buf.dev_get<char>(), cfg.bytes_per_warp,
        cfg.total_warps(), cfg.warmup_iters, use_fence);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed
    float ms = timed_launch([&](uint32_t iters) {
      pinned_host_write_kernel<<<cfg.blocks, cfg.threads>>>(
          write_buf.dev_get<char>(), cfg.bytes_per_warp,
          cfg.total_warps(), iters, use_fence);
    }, cfg, cfg.iterations);

    print_result("write", cfg, ms);

    // Validate
    if (cfg.validate) {
      int errs = validate_write_buffer(cfg, write_buf.get<char>());
      if (errs == 0) {
        printf("  [validate: PASS]\n");
      } else {
        printf("  [validate: FAIL — %d mismatches (first 32 shown)]\n", errs);
      }
    }
  }

  // ============================================================
  // Mode: read (pinned host → GPU)
  // ============================================================
  if (cfg.mode == TransferMode::kRead) {
    // Seed the read buffer with a known pattern on the host
    {
      uint4 *p = write_buf.get<uint4>();
      uint32_t num_u4 = (uint32_t)(cfg.total_bytes() / sizeof(uint4));
      for (uint32_t i = 0; i < num_u4; ++i) {
        p[i].x = i; p[i].y = ~i; p[i].z = 0xA5A5A5A5u; p[i].w = 0x5A5A5A5Au;
      }
    }

    // Warmup
    pinned_host_read_kernel<<<cfg.blocks, cfg.threads>>>(
        write_buf.dev_get<char>(), cfg.bytes_per_warp,
        cfg.total_warps(), cfg.warmup_iters, use_fence);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed
    float ms = timed_launch([&](uint32_t iters) {
      pinned_host_read_kernel<<<cfg.blocks, cfg.threads>>>(
          write_buf.dev_get<char>(), cfg.bytes_per_warp,
          cfg.total_warps(), iters, use_fence);
    }, cfg, cfg.iterations);

    print_result("read", cfg, ms);
  }

  // ============================================================
  // Mode: readwrite (GPU reads src, writes dst — bidirectional)
  // ============================================================
  if (cfg.mode == TransferMode::kReadWrite) {
    read_buf.allocate(cfg.total_bytes(), cfg.numa_node);

    // Warmup
    pinned_host_readwrite_kernel<<<cfg.blocks, cfg.threads>>>(
        write_buf.dev_get<char>(), read_buf.dev_get<char>(),
        cfg.bytes_per_warp, cfg.total_warps(), cfg.warmup_iters, use_fence);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("NOTE: readwrite mode measures PCIe load-store serialization overhead, not\n"
           "      bidirectional bandwidth. See kernel comment for details. Use write/read\n"
           "      modes for meaningful bandwidth numbers.\n");

    // Timed — bytes_moved counts both read and write sides
    float ms = timed_launch([&](uint32_t iters) {
      pinned_host_readwrite_kernel<<<cfg.blocks, cfg.threads>>>(
          write_buf.dev_get<char>(), read_buf.dev_get<char>(),
          cfg.bytes_per_warp, cfg.total_warps(), iters, use_fence);
    }, cfg, cfg.iterations);

    // For readwrite, bytes_moved = 2x total_bytes (both directions)
    {
      double bytes_moved = 2.0 * (double)cfg.total_bytes() * cfg.iterations;
      double gbps = bytes_moved / (ms / 1000.0) / 1e9;
      printf("%-12s  %4d  %4d  %6u  %12lu  %6u  %10.2f  %10.3f  (bidir)\n",
             "readwrite", cfg.gpu_id, cfg.numa_node, cfg.total_warps(),
             (unsigned long)cfg.bytes_per_warp, cfg.iterations, ms, gbps);
    }
  }

  printf("\n");
  return 0;
}

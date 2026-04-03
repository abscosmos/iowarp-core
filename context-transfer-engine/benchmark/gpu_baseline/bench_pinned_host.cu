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
#include <cstdint>
#include <cstdio>

// ============================================================
// Kernel: GPU writes to pinned host memory
// ============================================================

__global__ void pinned_host_write_kernel(
    char    *dst,            // pinned host buffer
    uint64_t bytes_per_warp,
    uint32_t total_warps,
    uint32_t iterations,
    bool     use_fence)
{
  uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t warp_id    = global_tid / 32;
  uint32_t lane_id    = global_tid & 31;

  if (warp_id >= total_warps) return;

  uint4 *dst4      = reinterpret_cast<uint4 *>(dst + (uint64_t)warp_id * bytes_per_warp);
  uint32_t num_u4  = (uint32_t)(bytes_per_warp / sizeof(uint4));

  // Pattern encodes warp and lane for optional validation
  uint4 pattern;
  pattern.x = warp_id;
  pattern.y = lane_id;
  pattern.z = 0xDEADBEEFu;
  pattern.w = 0xCAFEBABEu;

  for (uint32_t iter = 0; iter < iterations; ++iter) {
    // Warp-strided 128-bit stores: each lane writes every 32nd uint4
    for (uint32_t i = lane_id; i < num_u4; i += 32) {
      dst4[i] = pattern;
    }
    __syncwarp();
    if (use_fence) {
      // Flush L2 write-back buffer so stores reach host-visible memory
      __threadfence_system();
    }
  }
}

// ============================================================
// Kernel: GPU reads from pinned host memory
// ============================================================

// Shared sink prevents the compiler from dead-code-eliminating the loads.
__shared__ volatile uint32_t s_read_sink;

__global__ void pinned_host_read_kernel(
    const char *src,         // pinned host buffer
    uint64_t    bytes_per_warp,
    uint32_t    total_warps,
    uint32_t    iterations,
    bool        use_fence)
{
  uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t warp_id    = global_tid / 32;
  uint32_t lane_id    = global_tid & 31;

  if (warp_id >= total_warps) return;

  const uint4 *src4   = reinterpret_cast<const uint4 *>(
                            src + (uint64_t)warp_id * bytes_per_warp);
  uint32_t num_u4     = (uint32_t)(bytes_per_warp / sizeof(uint4));
  uint32_t acc        = 0;

  for (uint32_t iter = 0; iter < iterations; ++iter) {
    for (uint32_t i = lane_id; i < num_u4; i += 32) {
      uint4 v = src4[i];
      acc ^= v.x ^ v.y ^ v.z ^ v.w;   // force dependency chain
    }
    __syncwarp();
    if (use_fence) {
      __threadfence_system();
    }
  }

  // Write sink so acc is not optimised away
  if (lane_id == 0) {
    s_read_sink = acc;
  }
}

// ============================================================
// Kernel: GPU reads from src and writes to dst (bidirectional)
// ============================================================

__global__ void pinned_host_readwrite_kernel(
    const char *src,         // pinned host read buffer
    char       *dst,         // pinned host write buffer
    uint64_t    bytes_per_warp,
    uint32_t    total_warps,
    uint32_t    iterations,
    bool        use_fence)
{
  uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t warp_id    = global_tid / 32;
  uint32_t lane_id    = global_tid & 31;

  if (warp_id >= total_warps) return;

  const uint4 *src4 = reinterpret_cast<const uint4 *>(
                          src + (uint64_t)warp_id * bytes_per_warp);
  uint4       *dst4 = reinterpret_cast<uint4 *>(
                          dst + (uint64_t)warp_id * bytes_per_warp);
  uint32_t num_u4   = (uint32_t)(bytes_per_warp / sizeof(uint4));

  for (uint32_t iter = 0; iter < iterations; ++iter) {
    /*
     * WHY READWRITE IS SLOW (~200-300 ms/iteration instead of ~1-2 ms):
     *
     * `dst4[i] = src4[i]` where both src and dst are pinned host memory
     * creates a serialized PCIe dependency chain per thread: each store must
     * wait for its load to complete over PCIe (~2-3 µs round trip) before it
     * can issue the write.  The GPU cannot pipeline the read and write because
     * the store depends on the load result.  With ~2048 such pairs per thread,
     * this results in ~200-300 ms/iteration instead of the ~1-2 ms expected
     * from raw PCIe bandwidth.
     *
     * Write-only and read-only modes are fast because:
     *   • Stores use write-combining (no blocking dependency).
     *   • Reads can have multiple outstanding requests in flight.
     *
     * A proper bidirectional measurement would need to stage through device
     * memory (shared or global) to decouple the two PCIe phases.
     */
    for (uint32_t i = lane_id; i < num_u4; i += 32) {
      dst4[i] = src4[i];
    }
    __syncwarp();
  }
  if (use_fence) {
    __threadfence_system();
  }
}

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
        write_buf.get<char>(), cfg.bytes_per_warp,
        cfg.total_warps(), cfg.warmup_iters, use_fence);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed
    float ms = timed_launch([&](uint32_t iters) {
      pinned_host_write_kernel<<<cfg.blocks, cfg.threads>>>(
          write_buf.get<char>(), cfg.bytes_per_warp,
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
        write_buf.get<char>(), cfg.bytes_per_warp,
        cfg.total_warps(), cfg.warmup_iters, use_fence);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed
    float ms = timed_launch([&](uint32_t iters) {
      pinned_host_read_kernel<<<cfg.blocks, cfg.threads>>>(
          write_buf.get<char>(), cfg.bytes_per_warp,
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
        write_buf.get<char>(), read_buf.get<char>(),
        cfg.bytes_per_warp, cfg.total_warps(), cfg.warmup_iters, use_fence);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("NOTE: readwrite mode measures PCIe load-store serialization overhead, not\n"
           "      bidirectional bandwidth. See kernel comment for details. Use write/read\n"
           "      modes for meaningful bandwidth numbers.\n");

    // Timed — bytes_moved counts both read and write sides
    float ms = timed_launch([&](uint32_t iters) {
      pinned_host_readwrite_kernel<<<cfg.blocks, cfg.threads>>>(
          write_buf.get<char>(), read_buf.get<char>(),
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

/**
 * kernels_pinned_host.cuh — GPU kernels for pinned host memory benchmarks
 *
 * Extracted from bench_pinned_host.cu so that bench_bus_contention.cu can
 * reuse the same kernel code without duplication.
 *
 * Three kernels:
 *   pinned_host_write_kernel     — GPU stores to pinned host (GPU->Host PCIe write)
 *   pinned_host_read_kernel      — GPU loads from pinned host (Host->GPU PCIe read)
 *   pinned_host_readwrite_kernel — GPU reads src, writes dst (bidirectional)
 */
#ifndef KERNELS_PINNED_HOST_CUH_
#define KERNELS_PINNED_HOST_CUH_

#include <cstdint>

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
     * wait for its load to complete over PCIe (~2-3 us round trip) before it
     * can issue the write.  The GPU cannot pipeline the read and write because
     * the store depends on the load result.  With ~2048 such pairs per thread,
     * this results in ~200-300 ms/iteration instead of the ~1-2 ms expected
     * from raw PCIe bandwidth.
     *
     * Write-only and read-only modes are fast because:
     *   - Stores use write-combining (no blocking dependency).
     *   - Reads can have multiple outstanding requests in flight.
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

#endif  // KERNELS_PINNED_HOST_CUH_

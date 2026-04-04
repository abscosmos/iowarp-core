/**
 * kernels_cross_node_pinned_host.cuh — GPU kernels for Benchmark 5
 *
 * Four kernels for cross-node EGM (Extended GPU Memory) pinned host benchmarks:
 *   cross_node_pinned_write_kernel        — GPU warp-strided uint4 writes to remote pinned host
 *   cross_node_pinned_read_kernel         — GPU warp-strided uint4 reads from remote pinned host
 *   cross_node_pinned_latency_send_kernel — single-warp: write payload + flag to remote
 *   cross_node_pinned_latency_poll_kernel — single-warp, lane 0: spin-poll flag, record clock64() delta
 *
 * All kernels operate on plain device pointers — the EGM fabric-imported VA is passed
 * directly as char* so the caller controls address computation.
 */
#ifndef KERNELS_CROSS_NODE_PINNED_HOST_CUH_
#define KERNELS_CROSS_NODE_PINNED_HOST_CUH_

#include <cstdint>

// ============================================================
// Kernel: GPU writes to remote EGM pinned host memory
// ============================================================

__global__ void cross_node_pinned_write_kernel(
    char    *dst,
    uint64_t bytes_per_warp,
    uint32_t iters,
    bool     use_fence)
{
  uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t warp_id    = global_tid / 32;
  uint32_t lane_id    = global_tid & 31;

  uint4 *dst4     = reinterpret_cast<uint4 *>(dst + (uint64_t)warp_id * bytes_per_warp);
  uint32_t num_u4 = (uint32_t)(bytes_per_warp / sizeof(uint4));

  uint4 pattern;
  pattern.x = warp_id;
  pattern.y = lane_id;
  pattern.z = 0xDEADBEEFu;
  pattern.w = 0xCAFEBABEu;

  for (uint32_t iter = 0; iter < iters; ++iter) {
    for (uint32_t i = lane_id; i < num_u4; i += 32) {
      dst4[i] = pattern;
    }
    __syncwarp();
    if (use_fence) {
      __threadfence_system();
    }
  }
}

// ============================================================
// Kernel: GPU reads from remote EGM pinned host memory
// ============================================================

// shared sink prevents the compiler from dead-code-eliminating the loads
__shared__ volatile uint32_t s_cn_pinned_read_sink;

__global__ void cross_node_pinned_read_kernel(
    const char *src,
    uint64_t    bytes_per_warp,
    uint32_t    iters)
{
  uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t warp_id    = global_tid / 32;
  uint32_t lane_id    = global_tid & 31;

  const uint4 *src4 = reinterpret_cast<const uint4 *>(
                          src + (uint64_t)warp_id * bytes_per_warp);
  uint32_t num_u4   = (uint32_t)(bytes_per_warp / sizeof(uint4));
  uint32_t acc      = 0;

  for (uint32_t iter = 0; iter < iters; ++iter) {
    for (uint32_t i = lane_id; i < num_u4; i += 32) {
      uint4 v = src4[i];
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    __syncwarp();
  }

  if (lane_id == 0) {
    s_cn_pinned_read_sink = acc;
  }
}

// ============================================================
// Kernel: single-warp sender — write payload + flag into remote EGM buffer
// ============================================================

__global__ void cross_node_pinned_latency_send_kernel(
    char             *remote_payload,   // EGM-mapped ptr to remote payload slots
    volatile uint32_t *remote_flag,     // EGM-mapped ptr to remote flag slots
    uint64_t          bytes_per_warp,
    uint32_t          warmup_iters,
    uint32_t          latency_iters)
{
  uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t warp_id    = global_tid / 32;
  uint32_t lane_id    = global_tid & 31;

  if (warp_id >= 1) return;  // single-warp kernel

  uint32_t total_iters = warmup_iters + latency_iters;

  for (uint32_t iter = 0; iter < total_iters; ++iter) {
    if (lane_id == 0) {
      // write a single uint4 payload into the iter-th slot
      uint4 *slot = reinterpret_cast<uint4 *>(remote_payload + (uint64_t)iter * bytes_per_warp);
      uint4 val;
      val.x = iter;
      val.y = 0xDEADBEEFu;
      val.z = 0xCAFEBABEu;
      val.w = iter;
      *slot = val;
    }
    __threadfence_system();
    __syncwarp();

    if (lane_id == 0) {
      const uint32_t flag_val = 1u;
      remote_flag[iter] = flag_val;
    }
    __threadfence_system();
    __syncwarp();
  }

}

// ============================================================
// Kernel: single-warp receiver — spin-poll flag, record clock64() delta
// ============================================================

__global__ void cross_node_pinned_latency_poll_kernel(
    volatile uint32_t *local_flag,
    uint64_t          *result_buf,
    uint32_t           warmup_iters,
    uint32_t           latency_iters)
{
  uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t warp_id    = global_tid / 32;
  uint32_t lane_id    = global_tid & 31;

  if (warp_id >= 1 || lane_id != 0) return;  // lane 0 only

  uint32_t total_iters = warmup_iters + latency_iters;

  for (uint32_t iter = 0; iter < total_iters; ++iter) {
    uint64_t t0 = clock64();

    while (local_flag[iter] == 0) { /* busy wait */ }

    uint64_t t1 = clock64();

    if (iter >= warmup_iters) {
      result_buf[iter - warmup_iters] = t1 - t0;
    }
  }
}

#endif  // KERNELS_CROSS_NODE_PINNED_HOST_CUH_

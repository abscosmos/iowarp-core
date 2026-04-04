/**
 * kernels_remote_hbm.cuh — GPU kernels for NVSHMEM remote HBM benchmarks
 *
 * Extracted from bench_remote_hbm.cu so that bench_bus_contention.cu can
 * reuse the same kernel code without duplication.
 *
 * Kernels:
 *   init_pattern_kernel        — fill buffer with identifiable pattern
 *   remote_hbm_put_kernel      — NVSHMEM warp-collective put
 *   remote_hbm_get_kernel      — NVSHMEM warp-collective get
 *   remote_hbm_pingpong_kernel — NVSHMEM put then get (round-trip)
 *
 * Guarded with HAVE_NVSHMEM — only available when NVSHMEM is compiled in.
 */
#ifndef KERNELS_REMOTE_HBM_CUH_
#define KERNELS_REMOTE_HBM_CUH_

#ifdef HAVE_NVSHMEM

#include <cassert>  // NVSHMEM 3.x device headers use assert(); must be included first
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cstdint>

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

#endif  // HAVE_NVSHMEM
#endif  // KERNELS_REMOTE_HBM_CUH_

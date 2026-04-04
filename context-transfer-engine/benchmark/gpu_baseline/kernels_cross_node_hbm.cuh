/**
 * kernels_cross_node_hbm.cuh — Latency kernels for Benchmark 4
 *
 * cross_node_latency_send_kernel  — PE 0: put payload + flag, nvshmem_quiet()
 * cross_node_latency_poll_kernel  — PE 1: spin-poll flag, record clock64() delta
 *
 * BW kernels (put/get) are reused from kernels_remote_hbm.cuh.
 * Guarded with HAVE_NVSHMEM.
 */
#ifndef KERNELS_CROSS_NODE_HBM_CUH_
#define KERNELS_CROSS_NODE_HBM_CUH_

#ifdef HAVE_NVSHMEM

#include <nvshmem.h>
#include <nvshmemx.h>
#include <cstdint>

// ============================================================
// PE 0: put payload + flag, then nvshmem_quiet()
//
// For each sample iter:
//   1. Warp-collective NBI put of bytes_per_warp bytes into target PE's payload buffer
//      at offset iter * bytes_per_warp (payload slots are pre-allocated).
//   2. Lane 0: NBI scalar put of flag=1 to target PE's flag slot [iter].
//   3. All lanes: nvshmem_quiet() — flushes both puts together before next iter.
//
// The single nvshmem_quiet() guarantees payload delivery happens before
// the flag is visible on the target PE (NVSHMEM ordering: quiet drains all pending NBI).
// There is NO quiet between the payload put and the flag put.
// ============================================================

__global__ void cross_node_latency_send_kernel(
    char      *sym_payload_buf,  // local symmetric payload addr (NVSHMEM resolves to target PE)
    uint32_t  *sym_flag,         // local symmetric flag addr (NVSHMEM resolves to target PE)
    char      *local_src,        // local source data (PE 0's device memory)
    uint64_t   bytes_per_warp,   // payload bytes per sample
    uint32_t   warmup_iters,     // warmup samples (not timed)
    uint32_t   latency_iters,    // timed samples
    int        target_pe)
{
    uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t warp_id    = global_tid / 32;
    uint32_t lane_id    = global_tid & 31;

    if (warp_id >= 1) return;  // single-warp kernel

    uint32_t total_iters = warmup_iters + latency_iters;

    for (uint32_t iter = 0; iter < total_iters; ++iter) {
        char *remote_payload = sym_payload_buf + (uint64_t)iter * bytes_per_warp;

        // 1. Warp-collective NBI payload put
        nvshmemx_putmem_nbi_warp(remote_payload, local_src, bytes_per_warp, target_pe);

        // 2. Lane 0: NBI scalar flag put
        __syncwarp();
        if (lane_id == 0) {
            const uint32_t flag_val = 1u;
            nvshmem_uint32_put_nbi(&sym_flag[iter], &flag_val, 1, target_pe);
        }
        // __syncwarp() here is load-bearing: ensures lane 0's flag NBI (step 2) is issued
        // before any lane calls nvshmem_quiet(). Without it, lanes 1-31 could drain the
        // queue before lane 0 has submitted the flag put, leaving it unordered.
        __syncwarp();

        // 3. Flush both puts (payload NBI + flag NBI) before next iteration
        nvshmem_quiet();
    }
}

// ============================================================
// PE 1: spin-poll flag, record clock64() delta per sample
//
// For each sample iter:
//   1. Record t0 = clock64() — taken just before the spin, so the delta
//      includes ~10-50 cycles of warp scheduling overhead before the first poll.
//      This overhead is fixed and small relative to cross-node IB latency (microseconds).
//   2. Spin on local_flag[iter] until it reads 1 (volatile, no cacheable load)
//   3. Record t1 = clock64()
//   4. Store t1 - t0 into result_buf[iter - warmup_iters] (skip warmup)
//
// Only lane 0 of the single warp drives polling and timing.
// Other lanes are idle (return after lane_id check).
// ============================================================

__global__ void cross_node_latency_poll_kernel(
    volatile uint32_t *local_flag,   // local NVSHMEM symmetric flag array (polled in-place)
    uint64_t          *result_buf,   // output: cycle deltas for timed samples only
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

        // Spin-poll until flag arrives
        while (local_flag[iter] == 0) { /* busy wait */ }

        uint64_t t1 = clock64();

        // Record only timed samples
        if (iter >= warmup_iters) {
            result_buf[iter - warmup_iters] = t1 - t0;
        }
    }
}

#endif  // HAVE_NVSHMEM
#endif  // KERNELS_CROSS_NODE_HBM_CUH_

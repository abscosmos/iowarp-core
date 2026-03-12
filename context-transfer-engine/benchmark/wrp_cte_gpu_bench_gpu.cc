/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * GPU benchmark kernels for CTE PutBlob/GetBlob.
 *
 * Each GPU block acts as an independent worker thread, issuing
 * AsyncPutBlob / AsyncGetBlob via the Client API (NewTask + Send ->
 * SendGpu -> gpu2gpu queue). The GPU orchestrator processes tasks
 * on separate blocks.
 *
 * Compiled via add_cuda_library (clang-cuda dual-pass).
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <wrp_cte/core/core_client.h>
#include <chimaera/chimaera.h>
#include <chimaera/singletons.h>
#include <hermes_shm/util/gpu_api.h>
#include <chimaera/gpu_work_orchestrator.h>
#include <thread>
#include <chrono>

/**
 * Benchmark parameters passed to the kernel.
 */
struct GpuBenchParams {
  chi::PoolId pool_id;
  wrp_cte::core::TagId tag_id;
  chi::u64 io_size;
  int io_count;      // Per-block I/O count
  int mode;          // 0=Put, 1=Get, 2=PutGet
};

/**
 * Per-block result written to pinned memory.
 */
struct GpuBenchResult {
  int status;              // 1=success, negative=error
  long long elapsed_ns;    // Nanoseconds elapsed for this block
};

/**
 * GPU kernel: each block runs io_count Put and/or Get operations.
 * Only thread 0 of each block does work (matches orchestrator pattern).
 */
__global__ void gpu_bench_kernel(
    chi::IpcManagerGpuInfo gpu_info,
    GpuBenchParams params,
    char **block_put_buffers,
    char **block_get_buffers,
    GpuBenchResult *results) {
  CHIMAERA_GPU_INIT(gpu_info);

  if (threadIdx.x != 0) return;

  int block_id = blockIdx.x;
  GpuBenchResult *my_result = &results[block_id];
  // Status convention: 0 = not started / in progress, 1 = success, <0 = error.
  // Do NOT set a non-zero value here — the host polls for status != 0.

  wrp_cte::core::Client client(params.pool_id);

  char *put_buf = block_put_buffers[block_id];
  char *get_buf = block_get_buffers[block_id];

  // Use GPU clock for timing
  long long start_clock = clock64();

  if (params.mode == 0 || params.mode == 2) {
    // Put phase
    for (int i = 0; i < params.io_count; ++i) {
      auto future = client.AsyncPutBlob(
          params.tag_id, "blob",
          chi::u64(0), params.io_size,
          hipc::ShmPtr<>::FromRaw(put_buf), 0.5f,
          wrp_cte::core::Context(),
          chi::u32(0),
          chi::PoolQuery::Local());
      if (future.IsNull()) {
        my_result->status = -2;
        return;
      }
      future.Wait();
    }
  }

  if (params.mode == 1 || params.mode == 2) {
    // Get phase
    for (int i = 0; i < params.io_count; ++i) {
      auto future = client.AsyncGetBlob(
          params.tag_id, "blob",
          chi::u64(0), params.io_size,
          chi::u32(0),
          hipc::ShmPtr<>::FromRaw(get_buf),
          chi::PoolQuery::Local());
      if (future.IsNull()) {
        my_result->status = -3;
        return;
      }
      future.Wait();
    }
  }

  long long end_clock = clock64();
  my_result->elapsed_ns = end_clock - start_clock;
  my_result->status = 1;
}

/**
 * Host entry point: sets up GPU backends, launches benchmark kernel,
 * and collects timing results.
 *
 * @param pool_id      CTE core pool ID
 * @param tag_id       Tag ID for blob operations
 * @param io_size      Size of each I/O in bytes
 * @param io_count     Number of I/Os per block
 * @param mode         0=Put, 1=Get, 2=PutGet
 * @param num_blocks   Number of GPU client blocks
 * @param num_threads  Threads per block (only thread 0 does work)
 * @param results      Output array (num_blocks entries, pinned memory)
 * @return 0 on success, negative on error
 */
extern "C" int run_gpu_bench(
    chi::PoolId pool_id,
    wrp_cte::core::TagId tag_id,
    chi::u64 io_size,
    int io_count,
    int mode,
    int num_blocks,
    int num_threads,
    GpuBenchResult *results) {

  // Create GPU memory backend for client kernel allocations
  hipc::MemoryBackendId backend_id(20, 0);
  hipc::GpuShmMmap gpu_backend;
  size_t backend_size = static_cast<size_t>(64) * 1024 * 1024;  // 64 MB
  if (!gpu_backend.shm_init(backend_id, backend_size,
                             "/cte_gpu_bench", 0))
    return -100;

  CHI_IPC->RegisterGpuAllocator(backend_id, gpu_backend.data_,
                                 gpu_backend.data_capacity_);

  // GPU heap for serialization
  hipc::MemoryBackendId heap_id(21, 0);
  hipc::GpuMalloc gpu_heap;
  size_t heap_size = static_cast<size_t>(16) * 1024 * 1024;  // 16 MB
  if (!gpu_heap.shm_init(heap_id, heap_size, "", 0))
    return -102;

  chi::IpcManagerGpuInfo gpu_info = CHI_IPC->GetClientGpuInfo(0);
  gpu_info.backend = gpu_backend;
  gpu_info.gpu_heap_backend = gpu_heap;

  // Allocate per-block pinned buffers
  char **h_put_ptrs = new char*[num_blocks];
  char **h_get_ptrs = new char*[num_blocks];
  for (int i = 0; i < num_blocks; ++i) {
    cudaMallocHost(&h_put_ptrs[i], io_size);
    cudaMallocHost(&h_get_ptrs[i], io_size);
    if (!h_put_ptrs[i] || !h_get_ptrs[i]) return -101;
    memset(h_put_ptrs[i], 0xAB, io_size);
    memset(h_get_ptrs[i], 0x00, io_size);
  }

  // Copy pointer arrays to pinned memory (GPU-accessible)
  char **d_put_ptrs, **d_get_ptrs;
  cudaMallocHost(&d_put_ptrs, num_blocks * sizeof(char*));
  cudaMallocHost(&d_get_ptrs, num_blocks * sizeof(char*));
  memcpy(d_put_ptrs, h_put_ptrs, num_blocks * sizeof(char*));
  memcpy(d_get_ptrs, h_get_ptrs, num_blocks * sizeof(char*));

  // Initialize results
  for (int i = 0; i < num_blocks; ++i) {
    results[i].status = 0;
    results[i].elapsed_ns = 0;
  }

  GpuBenchParams params;
  params.pool_id = pool_id;
  params.tag_id = tag_id;
  params.io_size = io_size;
  params.io_count = io_count;
  params.mode = mode;

  // Pause orchestrator, launch client kernel, resume
  CHI_IPC->PauseGpuOrchestrator();

  void *stream = hshm::GpuApi::CreateStream();
  gpu_bench_kernel<<<num_blocks, num_threads, 0,
      static_cast<cudaStream_t>(stream)>>>(
      gpu_info, params, d_put_ptrs, d_get_ptrs, results);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    CHI_IPC->ResumeGpuOrchestrator();
    hshm::GpuApi::DestroyStream(stream);
    delete[] h_put_ptrs;
    delete[] h_get_ptrs;
    return -201;
  }

  CHI_IPC->ResumeGpuOrchestrator();

  // Poll pinned memory for all blocks to complete
  int timeout_us = 30000000;  // 30 seconds
  int elapsed_us = 0;
  while (elapsed_us < timeout_us) {
    bool all_done = true;
    for (int i = 0; i < num_blocks; ++i) {
      if (results[i].status == 0) {
        all_done = false;
        break;
      }
    }
    if (all_done) break;
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    elapsed_us += 100;
  }

  hshm::GpuApi::DestroyStream(stream);
  delete[] h_put_ptrs;
  delete[] h_get_ptrs;
  // Intentional leak of pinned buffers (cudaFreeHost blocks on persistent kernel)

  // Check for errors
  for (int i = 0; i < num_blocks; ++i) {
    if (results[i].status != 1) {
      return results[i].status == 0 ? -4 : results[i].status;
    }
  }

  return 0;
}

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

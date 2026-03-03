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
 * GPU kernels for Part 3: GPU Task Submission tests
 * This file contains only GPU kernel code and is compiled as CUDA
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include <chimaera/MOD_NAME/MOD_NAME_client.h>
#include <chimaera/MOD_NAME/MOD_NAME_tasks.h>
#include <chimaera/chimaera.h>
#include <chimaera/pool_query.h>
#include <chimaera/singletons.h>
#include <chimaera/task.h>
#include <chimaera/types.h>
#include <hermes_shm/util/gpu_api.h>

/**
 * GPU kernel that submits a task from within the kernel
 * Tests Part 3: GPU kernel calling NewTask and Send
 */
__global__ void gpu_submit_task_kernel(chi::IpcManagerGpu gpu_info,
                                       chi::PoolId pool_id, chi::u32 test_value,
                                       int *result) {
  *result = 100;  // Kernel started

  // Step 1: Initialize IPC manager (no queue needed for NewTask-only test)
  CHIMAERA_GPU_INIT(gpu_info);

  *result = 200;  // After CHIMAERA_GPU_INIT

  // Step 2: Create task using NewTask
  chi::TaskId task_id = chi::CreateTaskId();
  chi::PoolQuery query = chi::PoolQuery::Local();

  *result = 300;  // Before NewTask
  hipc::FullPtr<chimaera::MOD_NAME::GpuSubmitTask> task;
  task = CHI_IPC->NewTask<chimaera::MOD_NAME::GpuSubmitTask>(
                      task_id, pool_id, query, 0, test_value);

  if (task.ptr_ == nullptr) {
    *result = -1;  // NewTask failed
    return;
  }

  *result = 1;  // Success - NewTask works
}

/**
 * C++ wrapper function to run the GPU kernel test
 * This allows the CPU test file to call this without needing CUDA headers
 */
extern "C" int run_gpu_kernel_task_submission_test(chi::PoolId pool_id,
                                                   chi::u32 test_value) {
  // Create GPU memory backend using GPU-registered shared memory
  hipc::MemoryBackendId backend_id(2, 0);
  size_t gpu_memory_size = 10 * 1024 * 1024;  // 10MB
  hipc::GpuShmMmap gpu_backend;
  if (!gpu_backend.shm_init(backend_id, gpu_memory_size, "/gpu_kernel_submit",
                            0)) {
    return -100;  // Backend init failed
  }

  // Allocate result on GPU
  int *d_result = hshm::GpuApi::Malloc<int>(sizeof(int));
  int h_result = 0;
  hshm::GpuApi::Memcpy(d_result, &h_result, sizeof(int));

  // Create IpcManagerGpu for kernel
  chi::IpcManagerGpu gpu_info(gpu_backend, nullptr);

  // Launch kernel with 1 thread, 1 block
  gpu_submit_task_kernel<<<1, 1>>>(gpu_info, pool_id, test_value, d_result);

  // Check for kernel launch errors
  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    hshm::GpuApi::Free(d_result);
    return -201;  // Kernel launch error
  }

  // Synchronize and check for errors
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    hshm::GpuApi::Free(d_result);
    return -200;  // CUDA error
  }

  // Get result
  hshm::GpuApi::Memcpy(&h_result, d_result, sizeof(int));

  // Cleanup
  hshm::GpuApi::Free(d_result);

  return h_result;
}

/**
 * GPU kernel that tests full end-to-end runtime roundtrip using client API:
 * GPU kernel calls AsyncGpuSubmit() -> worker processes -> Wait() -> verify
 */
__global__ void gpu_full_runtime_kernel(chi::IpcManagerGpu gpu_info,
                                         chi::PoolId pool_id,
                                         chi::u32 test_value,
                                         int *d_result,
                                         chi::u32 *d_result_value) {
  *d_result = 0;
  CHIMAERA_GPU_INIT(gpu_info);
  chimaera::MOD_NAME::Client client(pool_id);
  auto future = client.AsyncGpuSubmit(chi::PoolQuery::Local(), 0, test_value);
  future.Wait();
  *d_result_value = future->result_value_;
  *d_result = 1;  // success
}

/**
 * C++ wrapper to launch the full runtime roundtrip GPU kernel
 */
extern "C" int run_gpu_full_runtime_test(chi::PoolId pool_id,
                                          chi::u32 test_value,
                                          chi::u32 *out_result_value) {
  cudaDeviceSetLimit(cudaLimitStackSize, 131072);  // 128KB stack for deep template chains

  // Create GPU memory backend for kernel allocations
  hipc::MemoryBackendId backend_id(3, 0);
  hipc::GpuShmMmap gpu_backend;
  if (!gpu_backend.shm_init(backend_id, 10 * 1024 * 1024, "/gpu_rt_test", 0))
    return -100;

  // Create GPU queue in pinned shared memory for task submission
  hipc::MemoryBackendId queue_backend_id(4, 0);
  hipc::GpuShmMmap queue_backend;
  if (!queue_backend.shm_init(queue_backend_id, 2 * 1024 * 1024,
                               "/gpu_rt_queue", 0))
    return -101;

  // Create allocator in the queue backend's data region
  auto *queue_alloc = queue_backend.MakeAlloc<hipc::ArenaAllocator<false>>(
      queue_backend.data_capacity_);
  if (!queue_alloc)
    return -103;

  // Create TaskQueue (1 lane, 2 priorities, depth 1024)
  auto gpu_queue_ptr = queue_alloc->NewObj<chi::TaskQueue>(
      queue_alloc, 1, 2, 1024);
  if (gpu_queue_ptr.IsNull())
    return -102;

  // Register queue with runtime and assign to GPU worker
  CHI_IPC->RegisterGpuQueue(gpu_queue_ptr);
  CHI_IPC->AssignGpuLanesToWorker();

  // Register GPU backend memory for host-side ShmPtr resolution.
  // The GPU kernel allocates FutureShm in this pinned memory; the worker
  // needs to resolve ShmPtrs pointing into it via ToFullPtr.
  CHI_IPC->RegisterGpuAllocator(backend_id, gpu_backend.data_,
                                 gpu_backend.data_capacity_);

  chi::IpcManagerGpu gpu_info(gpu_backend, gpu_queue_ptr.ptr_);

  int *d_result = hshm::GpuApi::Malloc<int>(sizeof(int));
  chi::u32 *d_rv = hshm::GpuApi::Malloc<chi::u32>(sizeof(chi::u32));
  int h_result = 0;
  chi::u32 h_rv = 0;
  hshm::GpuApi::Memcpy(d_result, &h_result, sizeof(int));
  hshm::GpuApi::Memcpy(d_rv, &h_rv, sizeof(chi::u32));

  gpu_full_runtime_kernel<<<1, 1>>>(gpu_info, pool_id, test_value, d_result,
                                     d_rv);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    hshm::GpuApi::Free(d_result);
    hshm::GpuApi::Free(d_rv);
    return -201;
  }

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    hshm::GpuApi::Free(d_result);
    hshm::GpuApi::Free(d_rv);
    return -200;
  }

  hshm::GpuApi::Memcpy(&h_result, d_result, sizeof(int));
  hshm::GpuApi::Memcpy(&h_rv, d_rv, sizeof(chi::u32));

  *out_result_value = h_rv;
  hshm::GpuApi::Free(d_result);
  hshm::GpuApi::Free(d_rv);
  return h_result;
}

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

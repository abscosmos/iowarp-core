/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * See COPYING file in the top-level directory.
 */

/**
 * GPU-compiled wrapper for cross-warp parallelism validation test.
 * Must be compiled as CUDA so Send() detects ToLocalGpu routing
 * and redirects to SendToGpu().
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
 * Run the cross-warp parallelism test.
 * Allocates a device atomic counter, submits a GpuSubmitTask with the
 * given parallelism, waits for completion, and reads the counter.
 *
 * @param pool_id       Pool ID of the MOD_NAME container
 * @param parallelism   Number of GPU threads to use (e.g., 2048)
 * @param out_counter   Output: number of lanes that executed the handler
 * @return 1 on success, negative on error
 */
extern "C" int run_gpu_parallelism_test(
    chi::PoolId pool_id,
    chi::u32 parallelism,
    chi::u32 *out_counter) {

  // Allocate a device-side atomic counter, initialized to 0
  unsigned int *d_counter;
  cudaError_t err = cudaMalloc(&d_counter, sizeof(unsigned int));
  if (err != cudaSuccess) {
    return -100;  // cudaMalloc failed
  }
  err = cudaMemset(d_counter, 0, sizeof(unsigned int));
  if (err != cudaSuccess) {
    cudaFree(d_counter);
    return -101;  // cudaMemset failed
  }

  // Create client and submit task with parallelism via ToLocalGpu
  chimaera::MOD_NAME::Client client(pool_id);
  chi::u32 gpu_id = 0;
  chi::u32 test_value = 42;
  chi::u64 counter_addr = reinterpret_cast<chi::u64>(d_counter);

  auto future = client.AsyncGpuSubmit(
      chi::PoolQuery::ToLocalGpu(gpu_id, parallelism),
      gpu_id, test_value, counter_addr);

  fprintf(stderr, "[PARALLELISM] Submitted, parallelism=%u, waiting...\n", parallelism);

  // Wait with timeout
  bool completed = future.Wait(30.0f);
  fprintf(stderr, "[PARALLELISM] Wait done: completed=%d\n", (int)completed);
  if (!completed) {
    auto fshm_full = future.GetFutureShm();
    chi::FutureShm *fshm = fshm_full.ptr_;
    if (fshm) {
      fprintf(stderr, "[PARALLELISM] TIMEOUT: total_warps=%u cc=%u\n",
              fshm->total_warps_,
              fshm->completion_counter_.load());
    }
    cudaFree(d_counter);
    return -3;  // Timeout
  }

  // Read counter from device
  unsigned int h_counter = 0;
  err = cudaMemcpy(&h_counter, d_counter, sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    cudaFree(d_counter);
    return -102;  // cudaMemcpy failed
  }

  *out_counter = h_counter;

  // Verify result_value_ is correct (last warp's output)
  chi::u32 expected = (test_value * 3) + gpu_id;
  if (future->result_value_ != expected) {
    fprintf(stderr, "[PARALLELISM] result_value_=%u expected=%u\n",
            future->result_value_, expected);
    cudaFree(d_counter);
    return -4;  // Wrong result value
  }

  cudaFree(d_counter);
  return 1;  // Success
}

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

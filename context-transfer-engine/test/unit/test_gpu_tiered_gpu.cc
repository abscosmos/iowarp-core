/**
 * test_gpu_tiered_gpu.cc — GPU kernel for tiered storage CTE unit test
 *
 * Single kernel with CHIMAERA_GPU_ORCHESTRATOR_INIT (1 warp).
 * The warp:
 *   1. Fills 200MB of data with a known pattern (all lanes participate)
 *   2. Lane 0: AsyncPutBlob the 200MB as 4 x 50MB blobs
 *   3. Zeros the read buffer (all lanes)
 *   4. Lane 0: AsyncGetBlob the first 2 blobs (100MB) into the read buffer
 *   5. All lanes: verify data matches the original pattern
 *
 * This tests CTE's two-tier storage: 50MB HBM + 400MB pinned host.
 * The 200MB PutBlob must spill from HBM (50MB) to pinned (150MB).
 * The 100MB GetBlob reads back from whichever tier CTE placed it.
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <wrp_cte/core/core_client.h>
#include <chimaera/chimaera.h>
#include <chimaera/singletons.h>
#include <chimaera/gpu_work_orchestrator.h>
#include <chimaera/ipc_manager.h>
#include <hermes_shm/util/gpu_api.h>

static constexpr chi::u64 kBlobChunkSize = 50ULL * 1024 * 1024;  // 50MB per blob
static constexpr chi::u32 kPutBlobCount = 4;                      // 4 x 50MB = 200MB
static constexpr chi::u32 kGetBlobCount = 2;                      // 2 x 50MB = 100MB

__global__ void gpu_tiered_putget_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId cte_pool_id,
    wrp_cte::core::TagId tag_id,
    chi::u32 num_blocks,
    hipc::FullPtr<char> write_ptr,    // 200MB write buffer in data backend
    hipc::FullPtr<char> read_ptr,     // 100MB read buffer in data backend
    hipc::AllocatorId data_alloc_id,
    int *d_result,
    volatile int *d_progress) {
  CHIMAERA_GPU_ORCHESTRATOR_INIT(gpu_info, num_blocks);

  chi::u32 warp_id = chi::IpcManager::GetWarpId();
  chi::u32 lane_id = chi::IpcManager::GetLaneId();

  if (warp_id != 0) return;

  // ========== STEP 1: Fill 200MB write buffer with pattern ==========
  // Pattern: byte[i] = (i % 251) — a prime-modulo pattern for verification
  if (lane_id == 0) { d_progress[0] = 1; __threadfence_system(); }
  __syncwarp();

  chi::u64 total_write = kBlobChunkSize * kPutBlobCount;  // 200MB
  for (chi::u64 i = lane_id; i < total_write; i += 32) {
    write_ptr.ptr_[i] = static_cast<char>(i % 251);
  }
  __syncwarp();

  // ========== STEP 2: PutBlob 4 x 50MB blobs ==========
  if (lane_id == 0) { d_progress[0] = 2; __threadfence_system(); }
  __syncwarp();

  if (chi::IpcManager::IsWarpScheduler()) {
    wrp_cte::core::Client cte_client(cte_pool_id);
    using StrT = hshm::priv::basic_string<char, CHI_PRIV_ALLOC_T>;

    for (chi::u32 b = 0; b < kPutBlobCount; b++) {
      char name[32];
      int pos = 0;
      const char *pfx = "tiered_b";
      while (*pfx) name[pos++] = *pfx++;
      pos += StrT::NumberToStr(name + pos, 32 - pos, b);
      name[pos] = '\0';

      hipc::ShmPtr<> shm;
      shm.alloc_id_ = data_alloc_id;
      shm.off_.exchange(write_ptr.shm_.off_.load() + b * kBlobChunkSize);

      auto future = cte_client.AsyncPutBlob(
          tag_id, name,
          (chi::u64)0, kBlobChunkSize,
          shm, -1.0f,
          wrp_cte::core::Context(), (chi::u32)0,
          chi::PoolQuery::Local());

      if (future.GetFutureShmPtr().IsNull()) {
        *d_result = -(100 + b);  // PutBlob alloc failed for blob b
        __threadfence_system();
        return;
      }
      future.Wait();

      chi::u32 rc = future->GetReturnCode();
      if (rc != 0) {
        *d_result = -(200 + b * 10 + rc);  // PutBlob error
        __threadfence_system();
        return;
      }
    }
  }
  __syncwarp();

  if (lane_id == 0) { d_progress[0] = 3; __threadfence_system(); }

  // ========== STEP 3: Zero the 100MB read buffer ==========
  chi::u64 total_read = kBlobChunkSize * kGetBlobCount;  // 100MB
  for (chi::u64 i = lane_id; i < total_read; i += 32) {
    read_ptr.ptr_[i] = 0;
  }
  __syncwarp();

  // ========== STEP 4: GetBlob 2 x 50MB blobs (first 100MB) ==========
  if (lane_id == 0) { d_progress[0] = 4; __threadfence_system(); }
  __syncwarp();

  if (chi::IpcManager::IsWarpScheduler()) {
    wrp_cte::core::Client cte_client(cte_pool_id);
    using StrT = hshm::priv::basic_string<char, CHI_PRIV_ALLOC_T>;

    for (chi::u32 b = 0; b < kGetBlobCount; b++) {
      char name[32];
      int pos = 0;
      const char *pfx = "tiered_b";
      while (*pfx) name[pos++] = *pfx++;
      pos += StrT::NumberToStr(name + pos, 32 - pos, b);
      name[pos] = '\0';

      hipc::ShmPtr<> shm;
      shm.alloc_id_ = data_alloc_id;
      shm.off_.exchange(read_ptr.shm_.off_.load() + b * kBlobChunkSize);

      auto future = cte_client.AsyncGetBlob(
          tag_id, name,
          (chi::u64)0, kBlobChunkSize,
          (chi::u32)0, shm,
          chi::PoolQuery::Local());

      if (future.GetFutureShmPtr().IsNull()) {
        *d_result = -(300 + b);  // GetBlob alloc failed
        __threadfence_system();
        return;
      }
      future.Wait();

      chi::u32 rc = future->GetReturnCode();
      if (rc != 0) {
        *d_result = -(400 + b * 10 + rc);  // GetBlob error
        __threadfence_system();
        return;
      }
    }
  }
  __syncwarp();

  if (lane_id == 0) { d_progress[0] = 5; __threadfence_system(); }

  // ========== STEP 5: Verify 100MB read data matches write pattern ==========
  __threadfence_system();

  int mismatches = 0;
  for (chi::u64 i = lane_id; i < total_read; i += 32) {
    char expected = static_cast<char>(i % 251);
    if (read_ptr.ptr_[i] != expected) {
      mismatches++;
    }
  }

  // Warp-reduce mismatches
  for (int offset = 16; offset > 0; offset >>= 1) {
    mismatches += __shfl_down_sync(0xFFFFFFFF, mismatches, offset);
  }

  if (lane_id == 0) {
    if (mismatches > 0) {
      *d_result = -500 - mismatches;  // Verification failed
    } else {
      *d_result = 1;  // SUCCESS
    }
    d_progress[0] = 6;
    __threadfence_system();
  }
}

// Alloc kernel
__global__ void gpu_tiered_alloc_kernel(
    hipc::MemoryBackend data_backend,
    chi::u64 total_bytes,
    hipc::FullPtr<char> *d_out_ptr) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  using AllocT = hipc::PrivateBuddyAllocator;
  auto *alloc = data_backend.MakeAlloc<AllocT>(data_backend.data_capacity_);
  if (!alloc) { d_out_ptr->SetNull(); return; }
  *d_out_ptr = alloc->AllocateObjs<char>(total_bytes);
}

/**
 * Host wrapper: sets up data backends and launches the GPU kernel.
 */
#if HSHM_IS_HOST
#include <hermes_shm/lightbeam/transport_factory_impl.h>

extern "C" int run_gpu_tiered_test(
    chi::PoolId pool_id,
    wrp_cte::core::TagId tag_id,
    int timeout_sec) {

  chi::u64 write_bytes = kBlobChunkSize * kPutBlobCount;  // 200MB
  chi::u64 read_bytes  = kBlobChunkSize * kGetBlobCount;  // 100MB
  chi::u64 total_data  = write_bytes + read_bytes;        // 300MB

  CHI_IPC->SetGpuOrchestratorBlocks(1, 32);
  CHI_IPC->PauseGpuOrchestrator();

  // Data backend: holds write buffer (200MB) + read buffer (100MB)
  hipc::MemoryBackendId data_id(200, 0);
  hipc::GpuMalloc data_backend;
  if (!data_backend.shm_init(data_id, total_data + 8*1024*1024, "", 0)) {
    CHI_IPC->ResumeGpuOrchestrator();
    fprintf(stderr, "TEST: data backend shm_init failed\n");
    return -1;
  }

  // Scratch backend (1MB per warp)
  hipc::MemoryBackendId scratch_id(201, 0);
  hipc::GpuMalloc scratch_backend;
  if (!scratch_backend.shm_init(scratch_id, 4*1024*1024, "", 0)) {
    CHI_IPC->ResumeGpuOrchestrator();
    return -2;
  }

  // Heap backend
  hipc::MemoryBackendId heap_id(202, 0);
  hipc::GpuMalloc heap_backend;
  if (!heap_backend.shm_init(heap_id, 4*1024*1024, "", 0)) {
    CHI_IPC->ResumeGpuOrchestrator();
    return -3;
  }

  // Alloc write buffer (200MB)
  hipc::FullPtr<char> *d_write_ptr;
  cudaMallocHost(&d_write_ptr, sizeof(hipc::FullPtr<char>));
  d_write_ptr->SetNull();
  gpu_tiered_alloc_kernel<<<1,1>>>(
      static_cast<hipc::MemoryBackend&>(data_backend), write_bytes, d_write_ptr);
  cudaDeviceSynchronize();
  if (d_write_ptr->IsNull()) {
    cudaFreeHost(d_write_ptr);
    CHI_IPC->ResumeGpuOrchestrator();
    fprintf(stderr, "TEST: write alloc failed\n");
    return -4;
  }
  hipc::FullPtr<char> write_ptr = *d_write_ptr;

  // Alloc read buffer (100MB)
  hipc::FullPtr<char> *d_read_ptr;
  cudaMallocHost(&d_read_ptr, sizeof(hipc::FullPtr<char>));
  d_read_ptr->SetNull();
  gpu_tiered_alloc_kernel<<<1,1>>>(
      static_cast<hipc::MemoryBackend&>(data_backend), read_bytes, d_read_ptr);
  cudaDeviceSynchronize();
  if (d_read_ptr->IsNull()) {
    cudaFreeHost(d_write_ptr); cudaFreeHost(d_read_ptr);
    CHI_IPC->ResumeGpuOrchestrator();
    fprintf(stderr, "TEST: read alloc failed\n");
    return -5;
  }
  hipc::FullPtr<char> read_ptr = *d_read_ptr;
  cudaFreeHost(d_write_ptr); cudaFreeHost(d_read_ptr);

  // Register data backend
  hipc::AllocatorId data_alloc_id(data_id.major_, data_id.minor_);
  CHI_IPC->RegisterGpuAllocator(data_id, data_backend.data_,
                                 data_backend.data_capacity_);

  // Build GPU info
  chi::IpcManagerGpu gpu_info = CHI_IPC->GetClientGpuInfo(0);
  gpu_info.backend = scratch_backend;

  // Result and progress
  int *d_result;
  cudaMallocHost(&d_result, sizeof(int));
  *d_result = 0;
  volatile int *d_progress;
  cudaMallocHost((void**)&d_progress, sizeof(int));
  *d_progress = 0;

  // Zero scratch/heap
  if (scratch_backend.data_)
    cudaMemset(scratch_backend.data_, 0, sizeof(hipc::PartitionedAllocator));
  if (heap_backend.data_)
    cudaMemset(heap_backend.data_, 0, sizeof(hipc::PartitionedAllocator));
  cudaDeviceSynchronize();

  // Launch kernel (1 block x 32 threads = 1 warp)
  void *stream = hshm::GpuApi::CreateStream();
  gpu_tiered_putget_kernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
      gpu_info, pool_id, tag_id, 1,
      write_ptr, read_ptr, data_alloc_id,
      d_result, d_progress);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    fprintf(stderr, "TEST: kernel launch failed: %s\n",
            cudaGetErrorString(launch_err));
    CHI_IPC->ResumeGpuOrchestrator();
    hshm::GpuApi::DestroyStream(stream);
    return -6;
  }

  // Resume orchestrator and poll for completion
  CHI_IPC->ResumeGpuOrchestrator();
  auto *orch = static_cast<chi::gpu::WorkOrchestrator*>(CHI_IPC->gpu_orchestrator_);
  auto *ctrl = orch ? orch->control_ : nullptr;
  if (ctrl) {
    int w = 0;
    while (ctrl->running_flag == 0 && w < 5000) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1)); ++w;
    }
  }

  // Poll with progress reporting
  int64_t timeout_us = (int64_t)timeout_sec * 1000000;
  int64_t elapsed_us = 0;
  int last_progress = -1;
  while (__atomic_load_n(d_result, __ATOMIC_ACQUIRE) == 0 && elapsed_us < timeout_us) {
    std::this_thread::sleep_for(std::chrono::microseconds(500));
    elapsed_us += 500;
    int p = *d_progress;
    if (p != last_progress) {
      const char *step_names[] = {
        "not started", "filling write buffer", "putting blobs",
        "put complete", "getting blobs", "get complete", "verifying"
      };
      fprintf(stderr, "TEST: progress=%d (%s) elapsed=%.1fs\n",
              p, p < 7 ? step_names[p] : "unknown", elapsed_us / 1e6);
      fflush(stderr);
      last_progress = p;
    }
  }

  int result = __atomic_load_n(d_result, __ATOMIC_ACQUIRE);
  if (result == 0) {
    fprintf(stderr, "TEST: TIMEOUT after %ds (progress=%d)\n",
            timeout_sec, (int)*d_progress);
  } else if (result == 1) {
    fprintf(stderr, "TEST: PASSED — Put 200MB + Get 100MB + Verify OK\n");
  } else {
    fprintf(stderr, "TEST: FAILED with result=%d\n", result);
  }
  fflush(stderr);

  CHI_IPC->PauseGpuOrchestrator();
  hshm::GpuApi::Synchronize(stream);
  hshm::GpuApi::DestroyStream(stream);
  cudaFreeHost(d_result);
  cudaFreeHost((void*)d_progress);

  return result;
}

#endif  // HSHM_IS_HOST
#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

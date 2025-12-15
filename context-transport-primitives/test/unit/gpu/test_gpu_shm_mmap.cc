/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of Hermes. The full Hermes copyright notice, including  *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the top directory. If you do not  *
 * have access to the file, you may request a copy from help@hdfgroup.org.   *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <catch2/catch_all.hpp>

#include "hermes_shm/data_structures/ipc/ring_buffer.h"
#include "hermes_shm/memory/backend/gpu_shm_mmap.h"
#include "hermes_shm/memory/allocator/arena_allocator.h"
#include "hermes_shm/util/gpu_api.h"

using hshm::ipc::GpuShmMmap;
using hshm::ipc::MemoryBackendId;
using hshm::ipc::mpsc_ring_buffer;
using hshm::ipc::ArenaAllocator;

/**
 * GPU kernel to push elements onto ring buffer
 *
 * @tparam T The element type
 * @tparam AllocT The allocator type
 * @param ring Pointer to the ring buffer
 * @param values Array of values to push
 * @param count Number of elements to push
 */
template<typename T, typename AllocT>
__global__ void PushElementsKernel(mpsc_ring_buffer<T, AllocT> *ring, T *values, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    ring->Emplace(values[i]);
  }
}

/**
 * Test GpuShmMmap backend with ring buffer
 *
 * Steps:
 * 1. Create a GpuShmMmap backend
 * 2. Create an allocator on that backend
 * 3. Allocate a ring_buffer on that backend
 * 4. Pass the ring_buffer to the kernel
 * 5. Verify that we can place 10 elements on the ring buffer
 * 6. Verify the runtime can pop the 10 elements
 */
TEST_CASE("GpuShmMmap", "[gpu][backend]") {
  constexpr size_t kBackendSize = 64 * 1024 * 1024;  // 64MB
  constexpr size_t kNumElements = 10;
  constexpr int kGpuId = 0;
  const std::string kUrl = "/test_gpu_shm_mmap";

  SECTION("RingBufferGpuAccess") {
    // Step 1: Create a GpuShmMmap backend
    GpuShmMmap backend;
    MemoryBackendId backend_id(0, 0);
    bool init_success = backend.shm_init(backend_id, kBackendSize, kUrl, kGpuId);
    REQUIRE(init_success);

    // Step 2: Create an allocator on that backend (on the host)
    // Since GpuShmMmap provides unified memory, we can create the allocator on the host
    using AllocT = hipc::ArenaAllocator<false>;
    AllocT *alloc_ptr = backend.MakeAlloc<AllocT>();
    REQUIRE(alloc_ptr != nullptr);

    // Step 3: Allocate a ring_buffer on that backend (on the host)
    // The ring buffer is allocated in unified memory, accessible from both CPU and GPU
    using RingBuffer = mpsc_ring_buffer<int, AllocT>;
    RingBuffer *ring_ptr = alloc_ptr->NewObj<RingBuffer>(alloc_ptr, kNumElements).ptr_;
    REQUIRE(ring_ptr != nullptr);

    // Step 4 & 5: Pass the ring_buffer to the kernel and push 10 elements
    // Allocate GPU-accessible host memory for the values array
    int *host_values;
    cudaMallocHost(&host_values, kNumElements * sizeof(int));
    for (size_t i = 0; i < kNumElements; ++i) {
      host_values[i] = static_cast<int>(i);
    }

    // Launch kernel to push elements (host_values is GPU-accessible pinned memory)
    PushElementsKernel<int, AllocT><<<1, 1>>>(ring_ptr, host_values, kNumElements);
    cudaDeviceSynchronize();

    // Step 6: Verify the runtime (CPU) can pop the 10 elements
    // Since GpuShmMmap provides unified memory, CPU can directly access the ring buffer
    // But we still need to verify the values, so we'll store them in a regular array
    int host_output[kNumElements];
    bool all_popped = true;

    for (size_t i = 0; i < kNumElements; ++i) {
      int value;
      bool popped = ring_ptr->Pop(value);
      if (!popped) {
        all_popped = false;
        break;
      }
      host_output[i] = value;
    }

    // Verify all pops succeeded
    REQUIRE(all_popped);

    // Verify the popped values match what we pushed
    for (size_t i = 0; i < kNumElements; ++i) {
      REQUIRE(host_output[i] == host_values[i]);
    }

    // Free pinned host memory
    cudaFreeHost(host_values);

    // Cleanup handled automatically by destructor
  }
}

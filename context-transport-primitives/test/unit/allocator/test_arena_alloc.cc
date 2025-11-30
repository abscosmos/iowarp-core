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

#include <catch2/catch_test_macros.hpp>
#include "allocator_test.h"
#include "hermes_shm/memory/backend/malloc_backend.h"
#include "hermes_shm/memory/allocator/arena_allocator.h"

using hshm::testing::AllocatorTest;

/**
 * Helper function to create a MallocBackend and ArenaAllocator
 * Returns the allocator pointer (caller must manage backend lifetime)
 */
template<bool ATOMIC>
hipc::ArenaAllocator<ATOMIC>* CreateArenaAllocator(hipc::MallocBackend &backend, size_t arena_size) {
  // Initialize backend (MallocBackend doesn't enforce size limits)
  backend.shm_init(hipc::MemoryBackendId(0, 0), arena_size);

  // Create allocator on top of backend with explicit arena size
  hipc::ArenaAllocator<ATOMIC> *alloc = new hipc::ArenaAllocator<ATOMIC>();
  alloc->shm_init(hipc::AllocatorId(hipc::MemoryBackendId(0, 0), 0), 0, arena_size, backend);

  return alloc;
}

TEST_CASE("ArenaAllocator - Basic Allocation", "[ArenaAllocator]") {
  hipc::MallocBackend backend;
  size_t arena_size = 1024 * 1024;  // 1 MB
  hipc::ArenaAllocator<false> *alloc = CreateArenaAllocator<false>(backend, arena_size);

  SECTION("Single allocation") {
    hipc::CtxAllocator<hipc::ArenaAllocator<false>> ctx_alloc(HSHM_MCTX, alloc);

    auto ptr = ctx_alloc->AllocateOffset(ctx_alloc.ctx_, 100);
    REQUIRE_FALSE(ptr.IsNull());
    REQUIRE(ptr.off_.load() == 0);  // First allocation at offset 0
    REQUIRE(alloc->GetHeapOffset() == 100);
  }

  SECTION("Multiple allocations") {
    hipc::CtxAllocator<hipc::ArenaAllocator<false>> ctx_alloc(HSHM_MCTX, alloc);

    auto ptr1 = ctx_alloc->AllocateOffset(ctx_alloc.ctx_, 100);
    auto ptr2 = ctx_alloc->AllocateOffset(ctx_alloc.ctx_, 200);
    auto ptr3 = ctx_alloc->AllocateOffset(ctx_alloc.ctx_, 300);

    REQUIRE(ptr1.off_.load() == 0);
    REQUIRE(ptr2.off_.load() == 100);
    REQUIRE(ptr3.off_.load() == 300);
    REQUIRE(alloc->GetHeapOffset() == 600);
  }

  // Note: Allocation tracking requires HSHM_ALLOC_TRACK_SIZE to be defined

  delete alloc;
  backend.shm_destroy();
}

TEST_CASE("ArenaAllocator - Aligned Allocation", "[ArenaAllocator]") {
  hipc::MallocBackend backend;
  size_t arena_size = 1024 * 1024;
  hipc::ArenaAllocator<false> *alloc = CreateArenaAllocator<false>(backend, arena_size);

  SECTION("Aligned allocations") {
    hipc::CtxAllocator<hipc::ArenaAllocator<false>> ctx_alloc(HSHM_MCTX, alloc);

    // Allocate 100 bytes aligned to 64
    auto ptr1 = ctx_alloc->AllocateOffset(ctx_alloc.ctx_, 100, 64);
    REQUIRE(ptr1.off_.load() % 64 == 0);

    // Next allocation should also be 64-byte aligned
    auto ptr2 = ctx_alloc->AllocateOffset(ctx_alloc.ctx_, 50, 64);
    REQUIRE(ptr2.off_.load() % 64 == 0);
  }

  SECTION("Mixed alignment") {
    hipc::CtxAllocator<hipc::ArenaAllocator<false>> ctx_alloc(HSHM_MCTX, alloc);

    auto ptr1 = ctx_alloc->AllocateOffset(ctx_alloc.ctx_, 1);  // 1 byte
    auto ptr2 = ctx_alloc->AllocateOffset(ctx_alloc.ctx_, 1, 64);  // Align to 64

    REQUIRE(ptr1.off_.load() == 0);
    REQUIRE(ptr2.off_.load() == 64);  // Should skip to next 64-byte boundary
  }

  delete alloc;
  backend.shm_destroy();
}

TEST_CASE("ArenaAllocator - Reset", "[ArenaAllocator]") {
  hipc::MallocBackend backend;
  size_t arena_size = 1024 * 1024;
  hipc::ArenaAllocator<false> *alloc = CreateArenaAllocator<false>(backend, arena_size);

  hipc::CtxAllocator<hipc::ArenaAllocator<false>> ctx_alloc(HSHM_MCTX, alloc);

  // Allocate some memory
  ctx_alloc->AllocateOffset(ctx_alloc.ctx_, 100);
  ctx_alloc->AllocateOffset(ctx_alloc.ctx_, 200);
  ctx_alloc->AllocateOffset(ctx_alloc.ctx_, 300);

  REQUIRE(alloc->GetHeapOffset() == 600);

  // Reset the arena
  alloc->Reset();

  REQUIRE(alloc->GetHeapOffset() == 0);

  // Allocate again - should start from offset 0
  auto ptr = ctx_alloc->AllocateOffset(ctx_alloc.ctx_, 50);
  REQUIRE(ptr.off_.load() == 0);
  REQUIRE(alloc->GetHeapOffset() == 50);

  delete alloc;
  backend.shm_destroy();
}

TEST_CASE("ArenaAllocator - Out of Memory", "[ArenaAllocator]") {
  hipc::MallocBackend backend;
  size_t arena_size = 1024;  // Small arena - 1 KB
  hipc::ArenaAllocator<false> *alloc = CreateArenaAllocator<false>(backend, arena_size);

  hipc::CtxAllocator<hipc::ArenaAllocator<false>> ctx_alloc(HSHM_MCTX, alloc);

  // Allocate most of the arena
  ctx_alloc->AllocateOffset(ctx_alloc.ctx_, 512);
  ctx_alloc->AllocateOffset(ctx_alloc.ctx_, 256);

  // This allocation should succeed (768 + 200 = 968 < 1024)
  REQUIRE_NOTHROW(ctx_alloc->AllocateOffset(ctx_alloc.ctx_, 200));

  // This allocation should fail (968 + 100 = 1068 > 1024)
  REQUIRE_THROWS(ctx_alloc->AllocateOffset(ctx_alloc.ctx_, 100));

  delete alloc;
  backend.shm_destroy();
}

TEST_CASE("ArenaAllocator - Free is No-op", "[ArenaAllocator]") {
  hipc::MallocBackend backend;
  size_t arena_size = 1024 * 1024;
  hipc::ArenaAllocator<false> *alloc = CreateArenaAllocator<false>(backend, arena_size);

  hipc::CtxAllocator<hipc::ArenaAllocator<false>> ctx_alloc(HSHM_MCTX, alloc);

  auto ptr1 = ctx_alloc->Allocate<int>(ctx_alloc.ctx_, 10);
  auto ptr2 = ctx_alloc->Allocate<int>(ctx_alloc.ctx_, 20);

  size_t heap_before = alloc->GetHeapOffset();

  // Free should be a no-op
  REQUIRE_NOTHROW(ctx_alloc->Free(ctx_alloc.ctx_, ptr1));
  REQUIRE_NOTHROW(ctx_alloc->Free(ctx_alloc.ctx_, ptr2));

  // Heap offset should not change
  REQUIRE(alloc->GetHeapOffset() == heap_before);

  delete alloc;
  backend.shm_destroy();
}

TEST_CASE("ArenaAllocator - Remaining Space", "[ArenaAllocator]") {
  hipc::MallocBackend backend;
  size_t test_arena_size = 1000;
  hipc::ArenaAllocator<false> *alloc = CreateArenaAllocator<false>(backend, test_arena_size);

  hipc::CtxAllocator<hipc::ArenaAllocator<false>> ctx_alloc(HSHM_MCTX, alloc);

  REQUIRE(alloc->GetRemainingSize() == test_arena_size);

  ctx_alloc->AllocateOffset(ctx_alloc.ctx_, 300);
  REQUIRE(alloc->GetRemainingSize() == 700);

  ctx_alloc->AllocateOffset(ctx_alloc.ctx_, 200);
  REQUIRE(alloc->GetRemainingSize() == 500);

  alloc->Reset();
  REQUIRE(alloc->GetRemainingSize() == test_arena_size);

  delete alloc;
  backend.shm_destroy();
}

TEST_CASE("ArenaAllocator - Atomic Version", "[ArenaAllocator][atomic]") {
  hipc::MallocBackend backend;
  size_t arena_size = 1024 * 1024;
  hipc::ArenaAllocator<true> *alloc = CreateArenaAllocator<true>(backend, arena_size);

  SECTION("Basic atomic allocations") {
    hipc::CtxAllocator<hipc::ArenaAllocator<true>> ctx_alloc(HSHM_MCTX, alloc);

    auto ptr1 = ctx_alloc->AllocateOffset(ctx_alloc.ctx_, 100);
    auto ptr2 = ctx_alloc->AllocateOffset(ctx_alloc.ctx_, 200);

    REQUIRE(ptr1.off_.load() == 0);
    REQUIRE(ptr2.off_.load() == 100);
    REQUIRE(alloc->GetHeapOffset() == 300);
  }

  SECTION("Atomic reset") {
    hipc::CtxAllocator<hipc::ArenaAllocator<true>> ctx_alloc(HSHM_MCTX, alloc);

    ctx_alloc->AllocateOffset(ctx_alloc.ctx_, 500);
    REQUIRE(alloc->GetHeapOffset() == 500);

    alloc->Reset();
    REQUIRE(alloc->GetHeapOffset() == 0);
  }

  delete alloc;
  backend.shm_destroy();
}

// Note: Type allocation tests are skipped because ArenaAllocator with MallocBackend
// doesn't provide a real memory buffer (MallocBackend has data_=nullptr).
// ArenaAllocator is designed to work with backends that provide actual buffers
// (like PosixShmMmap or ArrayBackend from sub-allocators).

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
#include "hermes_shm/memory/allocator/malloc_allocator.h"

using hshm::testing::AllocatorTest;

/**
 * Helper function to create a MallocBackend and MallocAllocator
 * Returns the allocator pointer (caller must manage backend lifetime)
 */
hipc::MallocAllocator* CreateMallocAllocator(hipc::MallocBackend &backend) {
  // Initialize backend with 256 MB
  size_t backend_size = 256 * 1024 * 1024;
  backend.shm_init(hipc::MemoryBackendId(0, 0), backend_size);

  // Create allocator on top of backend
  hipc::MallocAllocator *alloc = new hipc::MallocAllocator();
  alloc->shm_init(hipc::AllocatorId(hipc::MemoryBackendId(0, 0), 0), 0, backend);

  return alloc;
}

TEST_CASE("SubAllocator - Basic Creation and Destruction", "[SubAllocator]") {
  hipc::MallocBackend backend;
  hipc::MallocAllocator *parent_alloc = CreateMallocAllocator(backend);

  SECTION("Create and destroy a single sub-allocator") {
    // Create a sub-allocator with 64 MB
    size_t sub_alloc_size = 64 * 1024 * 1024;
    hipc::MallocAllocator *sub_alloc = parent_alloc->CreateSubAllocator<hipc::_MallocAllocator>(
        HSHM_MCTX, 1, sub_alloc_size, 0);

    REQUIRE(sub_alloc != nullptr);
    REQUIRE(sub_alloc->GetId().backend_id_ == parent_alloc->GetId().backend_id_);
    REQUIRE(sub_alloc->GetId().sub_id_ == 1);

    // Free the sub-allocator
    parent_alloc->FreeSubAllocator(HSHM_MCTX, sub_alloc);
  }

  SECTION("Create multiple sub-allocators with different IDs") {
    size_t sub_alloc_size = 32 * 1024 * 1024;

    hipc::MallocAllocator *sub_alloc1 = parent_alloc->CreateSubAllocator<hipc::_MallocAllocator>(
        HSHM_MCTX, 1, sub_alloc_size, 0);
    hipc::MallocAllocator *sub_alloc2 = parent_alloc->CreateSubAllocator<hipc::_MallocAllocator>(
        HSHM_MCTX, 2, sub_alloc_size, 0);
    hipc::MallocAllocator *sub_alloc3 = parent_alloc->CreateSubAllocator<hipc::_MallocAllocator>(
        HSHM_MCTX, 3, sub_alloc_size, 0);

    REQUIRE(sub_alloc1 != nullptr);
    REQUIRE(sub_alloc2 != nullptr);
    REQUIRE(sub_alloc3 != nullptr);

    REQUIRE(sub_alloc1->GetId().sub_id_ == 1);
    REQUIRE(sub_alloc2->GetId().sub_id_ == 2);
    REQUIRE(sub_alloc3->GetId().sub_id_ == 3);

    // Free all sub-allocators
    parent_alloc->FreeSubAllocator(HSHM_MCTX, sub_alloc1);
    parent_alloc->FreeSubAllocator(HSHM_MCTX, sub_alloc2);
    parent_alloc->FreeSubAllocator(HSHM_MCTX, sub_alloc3);
  }

  delete parent_alloc;
  backend.shm_destroy();
}

TEST_CASE("SubAllocator - Allocations within SubAllocator", "[SubAllocator]") {
  hipc::MallocBackend backend;
  hipc::MallocAllocator *parent_alloc = CreateMallocAllocator(backend);

  // Create a sub-allocator with 64 MB
  size_t sub_alloc_size = 64 * 1024 * 1024;
  hipc::MallocAllocator *sub_alloc = parent_alloc->CreateSubAllocator<hipc::_MallocAllocator>(
      HSHM_MCTX, 1, sub_alloc_size, 0);

  REQUIRE(sub_alloc != nullptr);

  SECTION("Allocate and free immediately") {
    hipc::CtxAllocator<hipc::MallocAllocator> ctx_alloc(HSHM_MCTX, sub_alloc);

    for (size_t i = 0; i < 1000; ++i) {
      auto ptr = ctx_alloc->template AlignedAllocate<void>(ctx_alloc.ctx_, 1024, 64);
      REQUIRE_FALSE(ptr.IsNull());
      ctx_alloc->Free(ctx_alloc.ctx_, ptr);
    }
  }

  SECTION("Batch allocations") {
    hipc::CtxAllocator<hipc::MallocAllocator> ctx_alloc(HSHM_MCTX, sub_alloc);
    std::vector<hipc::FullPtr<void>> ptrs;

    // Allocate batch
    for (size_t i = 0; i < 100; ++i) {
      auto ptr = ctx_alloc->template AlignedAllocate<void>(ctx_alloc.ctx_, 4096, 64);
      REQUIRE_FALSE(ptr.IsNull());
      ptrs.push_back(ptr);
    }

    // Free batch
    for (auto &ptr : ptrs) {
      ctx_alloc->Free(ctx_alloc.ctx_, ptr);
    }
  }

  // Free the sub-allocator
  parent_alloc->FreeSubAllocator(HSHM_MCTX, sub_alloc);

  delete parent_alloc;
  backend.shm_destroy();
}

TEST_CASE("SubAllocator - Random Allocation Test", "[SubAllocator]") {
  hipc::MallocBackend backend;
  hipc::MallocAllocator *parent_alloc = CreateMallocAllocator(backend);

  // Create a sub-allocator with 64 MB
  size_t sub_alloc_size = 64 * 1024 * 1024;
  hipc::MallocAllocator *sub_alloc = parent_alloc->CreateSubAllocator<hipc::_MallocAllocator>(
      HSHM_MCTX, 1, sub_alloc_size, 0);

  REQUIRE(sub_alloc != nullptr);

  // Use the AllocatorTest framework to run random tests
  AllocatorTest<hipc::MallocAllocator> tester(sub_alloc);

  SECTION("16 iterations of random allocations") {
    REQUIRE_NOTHROW(tester.TestRandomAllocation(16));
  }

  SECTION("32 iterations of random allocations") {
    REQUIRE_NOTHROW(tester.TestRandomAllocation(32));
  }

  // Free the sub-allocator
  parent_alloc->FreeSubAllocator(HSHM_MCTX, sub_alloc);

  delete parent_alloc;
  backend.shm_destroy();
}

TEST_CASE("SubAllocator - Multiple SubAllocators with Random Tests", "[SubAllocator]") {
  hipc::MallocBackend backend;
  hipc::MallocAllocator *parent_alloc = CreateMallocAllocator(backend);

  // Create 3 sub-allocators, each with 32 MB
  size_t sub_alloc_size = 32 * 1024 * 1024;
  hipc::MallocAllocator *sub_alloc1 = parent_alloc->CreateSubAllocator<hipc::_MallocAllocator>(
      HSHM_MCTX, 1, sub_alloc_size, 0);
  hipc::MallocAllocator *sub_alloc2 = parent_alloc->CreateSubAllocator<hipc::_MallocAllocator>(
      HSHM_MCTX, 2, sub_alloc_size, 0);
  hipc::MallocAllocator *sub_alloc3 = parent_alloc->CreateSubAllocator<hipc::_MallocAllocator>(
      HSHM_MCTX, 3, sub_alloc_size, 0);

  REQUIRE(sub_alloc1 != nullptr);
  REQUIRE(sub_alloc2 != nullptr);
  REQUIRE(sub_alloc3 != nullptr);

  SECTION("Run random tests on all three sub-allocators") {
    AllocatorTest<hipc::MallocAllocator> tester1(sub_alloc1);
    AllocatorTest<hipc::MallocAllocator> tester2(sub_alloc2);
    AllocatorTest<hipc::MallocAllocator> tester3(sub_alloc3);

    REQUIRE_NOTHROW(tester1.TestRandomAllocation(8));
    REQUIRE_NOTHROW(tester2.TestRandomAllocation(8));
    REQUIRE_NOTHROW(tester3.TestRandomAllocation(8));
  }

  // Free all sub-allocators
  parent_alloc->FreeSubAllocator(HSHM_MCTX, sub_alloc1);
  parent_alloc->FreeSubAllocator(HSHM_MCTX, sub_alloc2);
  parent_alloc->FreeSubAllocator(HSHM_MCTX, sub_alloc3);

  delete parent_alloc;
  backend.shm_destroy();
}

TEST_CASE("SubAllocator - Nested SubAllocators", "[SubAllocator][nested]") {
  hipc::MallocBackend backend;
  hipc::MallocAllocator *parent_alloc = CreateMallocAllocator(backend);

  // Create a sub-allocator from parent (64 MB)
  size_t sub_alloc1_size = 64 * 1024 * 1024;
  hipc::MallocAllocator *sub_alloc1 = parent_alloc->CreateSubAllocator<hipc::_MallocAllocator>(
      HSHM_MCTX, 1, sub_alloc1_size, 0);

  REQUIRE(sub_alloc1 != nullptr);

  // Create a nested sub-allocator from the first sub-allocator (16 MB)
  size_t sub_alloc2_size = 16 * 1024 * 1024;
  hipc::MallocAllocator *sub_alloc2 = sub_alloc1->CreateSubAllocator<hipc::_MallocAllocator>(
      HSHM_MCTX, 2, sub_alloc2_size, 0);

  REQUIRE(sub_alloc2 != nullptr);
  REQUIRE(sub_alloc2->GetId().sub_id_ == 2);

  // Test allocations in the nested sub-allocator
  AllocatorTest<hipc::MallocAllocator> tester(sub_alloc2);
  REQUIRE_NOTHROW(tester.TestRandomAllocation(8));

  // Free nested sub-allocator first, then parent sub-allocator
  sub_alloc1->FreeSubAllocator(HSHM_MCTX, sub_alloc2);
  parent_alloc->FreeSubAllocator(HSHM_MCTX, sub_alloc1);

  delete parent_alloc;
  backend.shm_destroy();
}

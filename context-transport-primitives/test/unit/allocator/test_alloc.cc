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
  // Initialize backend with 128 MB
  size_t backend_size = 128 * 1024 * 1024;
  backend.shm_init(hipc::MemoryBackendId(0), backend_size);

  // Create allocator on top of backend
  hipc::MallocAllocator *alloc = new hipc::MallocAllocator();
  alloc->shm_init(hipc::AllocatorId(0, 1), 0, backend);

  return alloc;
}

TEST_CASE("MallocAllocator - Allocate and Free Immediate", "[MallocAllocator]") {
  hipc::MallocBackend backend;
  hipc::MallocAllocator *alloc = CreateMallocAllocator(backend);

  AllocatorTest<hipc::MallocAllocator> tester(alloc);

  SECTION("Small allocations (1KB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeImmediate(10000, 1024));
  }

  SECTION("Medium allocations (64KB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeImmediate(1000, 64 * 1024));
  }

  SECTION("Large allocations (1MB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeImmediate(100, 1024 * 1024));
  }

  delete alloc;
  backend.shm_destroy();
}

TEST_CASE("MallocAllocator - Batch Allocate and Free", "[MallocAllocator]") {
  hipc::MallocBackend backend;
  hipc::MallocAllocator *alloc = CreateMallocAllocator(backend);

  AllocatorTest<hipc::MallocAllocator> tester(alloc);

  SECTION("Small batches (10 allocations of 4KB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeBatch(1000, 10, 4096));
  }

  SECTION("Medium batches (100 allocations of 4KB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeBatch(100, 100, 4096));
  }

  SECTION("Large batches (1000 allocations of 1KB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeBatch(10, 1000, 1024));
  }

  delete alloc;
  backend.shm_destroy();
}

TEST_CASE("MallocAllocator - Random Allocation", "[MallocAllocator]") {
  hipc::MallocBackend backend;
  hipc::MallocAllocator *alloc = CreateMallocAllocator(backend);

  AllocatorTest<hipc::MallocAllocator> tester(alloc);

  SECTION("16 iterations of random allocations") {
    REQUIRE_NOTHROW(tester.TestRandomAllocation(16));
  }

  SECTION("32 iterations of random allocations") {
    REQUIRE_NOTHROW(tester.TestRandomAllocation(32));
  }

  delete alloc;
  backend.shm_destroy();
}

TEST_CASE("MallocAllocator - Multi-threaded Random", "[MallocAllocator][multithread]") {
  hipc::MallocBackend backend;
  hipc::MallocAllocator *alloc = CreateMallocAllocator(backend);

  AllocatorTest<hipc::MallocAllocator> tester(alloc);

  SECTION("8 threads, 2 iterations each") {
    REQUIRE_NOTHROW(tester.TestMultiThreadedRandom(8, 2));
  }

  SECTION("4 threads, 4 iterations each") {
    REQUIRE_NOTHROW(tester.TestMultiThreadedRandom(4, 4));
  }

  delete alloc;
  backend.shm_destroy();
}

TEST_CASE("MallocAllocator - Run All Tests", "[MallocAllocator][all]") {
  hipc::MallocBackend backend;
  hipc::MallocAllocator *alloc = CreateMallocAllocator(backend);

  AllocatorTest<hipc::MallocAllocator> tester(alloc);

  REQUIRE_NOTHROW(tester.RunAllTests());

  delete alloc;
  backend.shm_destroy();
}

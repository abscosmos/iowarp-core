/**
 * Single-threaded multi-process unit test for BuddyAllocator
 *
 * Usage: test_buddy_allocator_singlethread <rank> <duration_sec>
 *
 * rank 0: Initializes shared memory and optionally runs for duration_sec
 * rank 1+: Attaches to shared memory and runs for duration_sec
 *
 * This test validates BuddyAllocator in a single-threaded environment across
 * multiple processes using small allocations (1 byte to 16KB).
 */

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <chrono>

#include "hermes_shm/memory/allocator/buddy_allocator.h"
#include "hermes_shm/memory/backend/posix_shm_mmap.h"
#include "allocator_test.h"

using namespace hshm::ipc;
using namespace hshm::testing;

// Shared memory configuration
constexpr size_t kShmSize = 512UL * 1024UL * 1024UL;  // 512 MB
const std::string kShmUrl = "/buddy_allocator_singlethread_test";

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <rank> <duration_sec>" << std::endl;
    return 1;
  }

  int rank = std::atoi(argv[1]);
  int duration_sec = std::atoi(argv[2]);

  std::cout << "Rank " << rank << ": Starting single-threaded test for "
            << duration_sec << " seconds" << std::endl;

  // Create or attach to shared memory
  PosixShmMmap backend;
  bool success = false;

  if (rank == 0) {
    // Rank 0 initializes
    std::cout << "Rank 0: Initializing shared memory" << std::endl;
    success = backend.shm_init(MemoryBackendId(0, 0), kShmSize, kShmUrl);
    if (!success) {
      std::cerr << "Rank 0: Failed to initialize shared memory" << std::endl;
      return 1;
    }
    std::cout << "Rank 0: Shared memory initialized successfully" << std::endl;
    std::cout << "  Shared memory size: " << kShmSize << " bytes ("
              << (kShmSize / (1024UL * 1024UL)) << " MB)" << std::endl;
  } else {
    // Other ranks attach to existing shared memory
    std::cout << "Rank " << rank << ": Attaching to shared memory" << std::endl;

    // Give rank 0 time to fully initialize before we try to attach
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    success = backend.shm_attach(kShmUrl);
    if (!success) {
      std::cerr << "Rank " << rank << ": Failed to attach to shared memory"
                << std::endl;
      return 1;
    }
    std::cout << "Rank " << rank << ": Attached to shared memory successfully"
              << std::endl;
  }

  // Initialize or attach allocator
  BuddyAllocator *allocator = nullptr;

  if (rank == 0) {
    std::cout << "Rank 0: Initializing BuddyAllocator" << std::endl;
    std::cout << "  Backend data capacity: " << backend.data_capacity_
              << " bytes" << std::endl;

    allocator = backend.MakeAlloc<BuddyAllocator>();
    if (allocator == nullptr) {
      std::cerr << "Rank 0: Failed to initialize BuddyAllocator" << std::endl;
      backend.shm_destroy();
      return 1;
    }

    std::cout << "Rank 0: BuddyAllocator initialized successfully" << std::endl;
    std::cout << "  Allocator size: " << sizeof(BuddyAllocator) << " bytes"
              << std::endl;
  } else {
    std::cout << "Rank " << rank << ": Attaching to BuddyAllocator"
              << std::endl;

    // Attach to existing allocator without reinitializing
    allocator = backend.AttachAlloc<BuddyAllocator>();
    if (allocator == nullptr) {
      std::cerr << "Rank " << rank << ": Failed to attach to BuddyAllocator"
                << std::endl;
      return 1;
    }

    std::cout << "Rank " << rank
              << ": Attached to BuddyAllocator successfully" << std::endl;
  }

  // Run test if duration > 0
  if (duration_sec > 0) {
    std::cout << "Rank " << rank
              << ": Starting single-threaded timed workload test for "
              << duration_sec << " seconds" << std::endl;
    std::cout << "Rank " << rank
              << ": Testing SMALL allocations only (1 byte to 16KB)"
              << std::endl;

    // Create allocator tester and run timed workload with SMALL allocations
    AllocatorTest<BuddyAllocator> tester(allocator);
    constexpr size_t kSmallMin = 1UL;          // 1 byte
    constexpr size_t kSmallMax = 16UL * 1024UL;  // 16 KB

    // Run the timed random allocation test (single-threaded)
    tester.TestRandomAllocationTimed(duration_sec, kSmallMin, kSmallMax);

    std::cout << "Rank " << rank << ": TEST PASSED" << std::endl;
  } else {
    std::cout << "Rank " << rank << ": Initialization complete, exiting"
              << std::endl;
  }

  // Only rank 0 should clean up shared memory, and only if it ran the test
  // (if duration was 0, other ranks may still be using it)
  if (rank == 0 && duration_sec > 0) {
    std::cout << "Rank 0: Cleaning up shared memory" << std::endl;
    backend.shm_destroy();
  }

  return 0;
}

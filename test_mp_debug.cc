#include <iostream>
#include <cstring>
#include "hermes_shm/memory/backend/posix_mmap.h"
#include "hermes_shm/memory/allocator/mp_allocator.h"

int main() {
    using namespace hshm::ipc;

    // Create backend with 512MB heap
    PosixMmap backend;
    size_t heap_size = 512 * 1024 * 1024;  // 512 MB heap
    size_t alloc_size = sizeof(MultiProcessAllocator);
    backend.shm_init(MemoryBackendId(0, 0), alloc_size + heap_size);

    // Initialize the allocator
    auto *alloc = backend.MakeAlloc<MultiProcessAllocator>();

    std::cout << "Allocator initialized. Header size: " << sizeof(MultiProcessAllocator) << " bytes\n";
    std::cout << "Process unit: " << alloc->process_unit_ << " bytes\n";
    std::cout << "Thread unit: " << alloc->thread_unit_ << " bytes\n";
    std::cout << "\n";

    // First, allocate two 1MB chunks to trigger the second ProcessBlock creation
    std::cout << "=== Allocating first 1MB chunk ===\n";
    auto ptr1 = alloc->Allocate<char>(1024 * 1024, 64);
    if (ptr1.IsNull()) {
        std::cerr << "First 1MB allocation failed!\n";
        return 1;
    }
    std::cout << "First 1MB allocation succeeded at offset: " << ptr1.shm_.off_.load() << "\n";

    // This should fill up most of the first thread block (1MB)
    // The next allocation will need to expand from ProcessBlock

    std::cout << "\n=== Allocating second 1MB chunk ===\n";
    auto ptr2 = alloc->Allocate<char>(1024 * 1024, 64);
    if (ptr2.IsNull()) {
        std::cerr << "Second 1MB allocation failed!\n";
        return 1;
    }
    std::cout << "Second 1MB allocation succeeded at offset: " << ptr2.shm_.off_.load() << "\n";

    // Now allocate a series of smaller chunks to trigger process block expansions
    std::cout << "\n=== Allocating smaller chunks to force ProcessBlock expansion ===\n";
    for (int i = 0; i < 20; i++) {
        auto ptr = alloc->Allocate<char>(512 * 1024, 64);  // 512KB chunks
        if (ptr.IsNull()) {
            std::cerr << "Allocation " << i << " failed (512KB)!\n";
            break;
        }
        std::cout << "Allocation " << i << " succeeded at offset: " << ptr.shm_.off_.load() << "\n";

        // Try to detect if we've allocated memory that overlaps with ThreadBlock
        // ThreadBlock is typically allocated early, so check for low offsets being reused
        ProcessBlock *pblock = alloc->GetProcessBlock();
        if (pblock) {
            void *tblock_data = HSHM_THREAD_MODEL->GetTls<void>(pblock->tblock_key_);
            ThreadBlock *tblock = reinterpret_cast<ThreadBlock*>(tblock_data);
            if (tblock) {
                size_t tblock_offset = reinterpret_cast<char*>(tblock) - backend.GetBackendData();
                std::cout << "  ThreadBlock is at offset: " << tblock_offset << "\n";

                // Check if the allocation overlaps with the ThreadBlock
                size_t alloc_start = ptr.shm_.off_.load();
                size_t alloc_end = alloc_start + 512 * 1024;
                size_t tblock_end = tblock_offset + sizeof(ThreadBlock) + alloc->thread_unit_;

                if (alloc_start < tblock_end && alloc_end > tblock_offset) {
                    std::cerr << "ERROR: Allocation overlaps with ThreadBlock!\n";
                    std::cerr << "  Allocation range: [" << alloc_start << ", " << alloc_end << ")\n";
                    std::cerr << "  ThreadBlock range: [" << tblock_offset << ", " << tblock_end << ")\n";
                    return 1;
                }
            }
        }
    }

    std::cout << "\nTest completed successfully.\n";
    return 0;
}
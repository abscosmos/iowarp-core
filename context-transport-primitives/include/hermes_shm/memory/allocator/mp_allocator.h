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

#ifndef HSHM_MEMORY_ALLOCATOR_MP_ALLOCATOR_H_
#define HSHM_MEMORY_ALLOCATOR_MP_ALLOCATOR_H_

#include "hermes_shm/memory/allocator/allocator.h"
#include "hermes_shm/memory/allocator/buddy_allocator.h"
#include "hermes_shm/data_structures/ipc/slist_pre.h"
#include "hermes_shm/thread/lock/mutex.h"
#include "hermes_shm/thread/thread_model_manager.h"
#include <sys/types.h>
#include <unistd.h>

namespace hshm::ipc {

/**
 * Forward declarations
 */
class ProcessBlock;
class MultiProcessAllocator;

/**
 * Per-thread allocator block providing lock-free fast-path allocation.
 *
 * Each thread has its own ThreadBlock with a private BuddyAllocator,
 * enabling concurrent allocations without contention. When a ThreadBlock
 * runs out of memory, it requests expansion from its parent ProcessBlock.
 */
class ThreadBlock : public pre::slist_node {
 public:
  int tid_;                    /**< Thread ID */
  BuddyAllocator alloc_;       /**< Private buddy allocator for this thread */

  /**
   * Default constructor
   */
  ThreadBlock() : tid_(-1) {}

  /**
   * Initialize the thread block with a memory region
   *
   * @param backend Memory backend containing the region for this thread
   * @param size Size of the memory region in bytes
   * @param tid Thread ID for this block
   * @return true on success, false on failure
   */
  bool shm_init(const MemoryBackend &backend, size_t size, int tid) {
    tid_ = tid;

    // Create a shifted backend that starts after the ThreadBlock header
    MemoryBackend thread_backend = backend;
    size_t header_size = sizeof(ThreadBlock);
    thread_backend.Shift(header_size);

    // Initialize the buddy allocator with the shifted backend
    return alloc_.shm_init(thread_backend);
  }

  /**
   * Allocate memory from this thread block
   *
   * @param size Size in bytes to allocate
   * @return Offset pointer to allocated memory, or null on failure
   */
  OffsetPtr Allocate(size_t size) {
    return alloc_.AllocateOffset(size);
  }

  /**
   * Free memory back to this thread block
   *
   * @param ptr Offset pointer to memory to free
   */
  void Free(OffsetPtr ptr) {
    alloc_.FreeOffset(ptr);
  }

  /**
   * Reallocate memory within this thread block
   *
   * @param ptr Original offset pointer
   * @param new_size New size in bytes
   * @return New offset pointer, or null on failure
   */
  OffsetPtr Reallocate(OffsetPtr ptr, size_t new_size) {
    // Simple implementation: allocate new, copy, free old
    // TODO: Optimize with in-place reallocation when possible
    OffsetPtr new_ptr = Allocate(new_size);
    if (new_ptr.IsNull()) {
      return OffsetPtr::GetNull();
    }

    // For now, just return the new pointer
    // In a complete implementation, we would copy data and free old pointer
    Free(ptr);
    return new_ptr;
  }
};

/**
 * Per-process allocator block managing multiple ThreadBlocks.
 *
 * Each process has one ProcessBlock that manages a pool of ThreadBlocks.
 * The ProcessBlock uses a mutex to protect its thread list and can
 * expand by requesting more memory from the global allocator.
 */
class ProcessBlock : public pre::slist_node {
 public:
  int pid_;                    /**< Process ID */
  int tid_count_;              /**< Number of thread blocks allocated */
  hshm::Mutex lock_;           /**< Mutex protecting thread list */
  BuddyAllocator alloc_;       /**< Allocator for managing ThreadBlock regions */
  pre::slist<false> threads_;  /**< List of ThreadBlocks */

  /**
   * Default constructor
   */
  ProcessBlock() : pid_(-1), tid_count_(0) {}

  /**
   * Initialize the process block with a memory region
   *
   * @param backend Memory backend for this process
   * @param region Start of the memory region
   * @param size Size of the memory region in bytes
   * @param pid Process ID
   * @return true on success, false on failure
   */
  bool shm_init(const MemoryBackend &backend, char *region, size_t size, int pid) {
    pid_ = pid;
    tid_count_ = 0;
    lock_.Init();
    threads_.Init();

    // Calculate space for ProcessBlock header and thread list
    size_t header_size = sizeof(ProcessBlock);

    // Create a shifted backend for the buddy allocator
    MemoryBackend process_backend = backend;
    // Calculate offset from root to this region, then shift past header
    size_t region_offset = region - backend.data_;
    process_backend.Shift(region_offset + header_size);
    process_backend.data_size_ = size - header_size;

    // Initialize buddy allocator for managing thread blocks
    return alloc_.shm_init(process_backend);
  }

  /**
   * Allocate a new ThreadBlock from this ProcessBlock
   *
   * @param backend Root memory backend (for FullPtr construction)
   * @param region_size Size of region to allocate for the ThreadBlock
   * @return FullPtr to the newly allocated ThreadBlock, or null on failure
   */
  FullPtr<ThreadBlock> AllocateThreadBlock(const MemoryBackend &backend, size_t region_size) {
    // Lock to protect thread list
    lock_.Lock(0);

    // Allocate memory for ThreadBlock from our buddy allocator
    OffsetPtr thread_offset = alloc_.AllocateOffset(region_size);
    if (thread_offset.IsNull()) {
      lock_.Unlock();
      return FullPtr<ThreadBlock>::GetNull();
    }

    // Convert offset to pointer
    char *thread_ptr = alloc_.backend_.data_ + thread_offset.load();
    ThreadBlock *tblock = reinterpret_cast<ThreadBlock*>(thread_ptr);

    // Initialize ThreadBlock in-place
    new (tblock) ThreadBlock();

    // Create backend for this ThreadBlock (keep root data_, set offset)
    MemoryBackend thread_backend = backend;
    thread_backend.data_offset_ = thread_offset.load();
    thread_backend.data_size_ = region_size;

    if (!tblock->shm_init(thread_backend, region_size, tid_count_++)) {
      alloc_.FreeOffset(thread_offset);
      lock_.Unlock();
      return FullPtr<ThreadBlock>::GetNull();
    }

    // Add to thread list
    ShmPtr thread_shm;
    thread_shm.off_ = thread_offset;
    FullPtr<pre::slist_node> node_ptr(reinterpret_cast<pre::slist_node*>(tblock), thread_shm);
    threads_.emplace(&alloc_, node_ptr);

    lock_.Unlock();

    // Return FullPtr to the ThreadBlock
    FullPtr<ThreadBlock> result;
    result.ptr_ = tblock;
    result.shm_.off_ = thread_offset;
    result.shm_.alloc_id_ = AllocatorId(backend.GetId(), 0);
    return result;
  }

  /**
   * Expand this ProcessBlock by freeing a region back to it
   *
   * This is called when memory is returned to the ProcessBlock allocator,
   * typically when a ThreadBlock is freed or when coalescing occurs.
   *
   * @param ptr Offset pointer to memory region to return
   */
  void Expand(OffsetPtr ptr) {
    lock_.Lock(0);
    alloc_.FreeOffset(ptr);
    lock_.Unlock();
  }
};

/**
 * Global header for the MultiProcessAllocator
 *
 * This header is stored at the beginning of the shared memory region
 * and contains the process list and global state.
 */
class MultiProcessAllocatorHeader {
 public:
  int pid_count_;                /**< Number of processes */
  pre::slist<false> alloc_procs_;/**< Active ProcessBlocks */
  pre::slist<false> free_procs_; /**< Free ProcessBlocks (not currently used) */
  hshm::Mutex lock_;             /**< Mutex protecting process lists */

  /**
   * Default constructor
   */
  MultiProcessAllocatorHeader() : pid_count_(0) {}

  /**
   * Initialize the header
   */
  void Init() {
    pid_count_ = 0;
    lock_.Init();
    alloc_procs_.Init();
    free_procs_.Init();
  }
};

/**
 * Multi-process allocator with thread-local storage for lock-free fast path.
 *
 * Architecture:
 * - Global BuddyAllocator manages the entire shared memory region
 * - ProcessBlocks are allocated per process, each managing ThreadBlocks
 * - ThreadBlocks provide lock-free allocation for individual threads
 *
 * Allocation Strategy (3-tier fallback):
 * 1. Fast path: Allocate from thread-local ThreadBlock (no locks)
 * 2. Medium path: Expand ThreadBlock from ProcessBlock (one lock)
 * 3. Slow path: Expand ProcessBlock from global allocator (global lock)
 *
 * This design minimizes contention and provides malloc-like performance
 * for the common case of thread-local allocations.
 */
class MultiProcessAllocator {
 public:
  MemoryBackend backend_;                    /**< Root memory backend (public for FullPtr) */

 private:
  MultiProcessAllocatorHeader *header_;      /**< Global header */
  BuddyAllocator alloc_;                     /**< Global buddy allocator */
  size_t process_unit_;                      /**< Default ProcessBlock size (1GB) */
  size_t thread_unit_;                       /**< Default ThreadBlock size (16MB) */

  // Thread-local storage keys
  ThreadLocalKey tblock_key_;                /**< TLS key for ThreadBlock* */
  ThreadLocalKey pblock_key_;                /**< TLS key for ProcessBlock* */

 public:
  /**
   * Default constructor
   */
  MultiProcessAllocator() : header_(nullptr), process_unit_(0), thread_unit_(0) {}

  /**
   * Initialize the allocator with a new memory region
   *
   * @param backend Memory backend to use
   * @param size Total size of the memory region
   * @return true on success, false on failure
   */
  bool shm_init(const MemoryBackend &backend, size_t size) {
    backend_ = backend;  // Store original backend for FullPtr construction

    // Place header at the beginning
    header_ = reinterpret_cast<MultiProcessAllocatorHeader*>(backend_.data_);
    new (header_) MultiProcessAllocatorHeader();
    header_->Init();

    // Create a shifted backend for the global allocator (don't modify backend_!)
    MemoryBackend alloc_backend = backend_;
    alloc_backend.Shift(sizeof(MultiProcessAllocatorHeader));

    // Initialize global buddy allocator with shifted backend
    if (!alloc_.shm_init(alloc_backend)) {
      return false;
    }

    // Set default sizes based on available memory
    // For testing with smaller memory regions, use smaller defaults
    if (size < 1ULL * 1024 * 1024 * 1024) {
      // For regions < 1GB, use smaller units
      process_unit_ = size / 4;                  // Use 1/4 of available space
      thread_unit_ = 4ULL * 1024 * 1024;         // 4MB
    } else {
      process_unit_ = 1ULL * 1024 * 1024 * 1024;  // 1GB
      thread_unit_ = 16ULL * 1024 * 1024;         // 16MB
    }

    // Allocate first ProcessBlock
    OffsetPtr pblock_offset = alloc_.AllocateOffset(process_unit_);
    if (pblock_offset.IsNull()) {
      return false;
    }

    // Initialize ProcessBlock
    char *pblock_ptr = backend_.data_ + pblock_offset.load();
    ProcessBlock *pblock = reinterpret_cast<ProcessBlock*>(pblock_ptr);
    new (pblock) ProcessBlock();

    // The pblock_offset is already relative to backend_.data_, so pblock_ptr is correct
    if (!pblock->shm_init(backend_, pblock_ptr, process_unit_, getpid())) {
      alloc_.FreeOffset(pblock_offset);
      return false;
    }

    // Add to process list
    ShmPtr pblock_shm;
    pblock_shm.off_ = pblock_offset;
    FullPtr<pre::slist_node> pblock_node(&alloc_, pblock_shm);
    header_->alloc_procs_.emplace(&alloc_, pblock_node);
    header_->pid_count_++;

    // Call shm_attach to set up TLS
    return shm_attach(process_unit_, thread_unit_);
  }

  /**
   * Attach to an existing allocator and set up TLS
   *
   * @param process_unit Default size for ProcessBlocks (default: 1GB)
   * @param thread_unit Default size for ThreadBlocks (default: 16MB)
   * @return true on success, false on failure
   */
  bool shm_attach(size_t process_unit = 1ULL * 1024 * 1024 * 1024,
                  size_t thread_unit = 16ULL * 1024 * 1024) {
    process_unit_ = process_unit;
    thread_unit_ = thread_unit;

    // Create TLS keys
    void *null_ptr = nullptr;
    if (!HSHM_THREAD_MODEL->CreateTls<void>(tblock_key_, null_ptr)) {
      return false;
    }
    if (!HSHM_THREAD_MODEL->CreateTls<void>(pblock_key_, null_ptr)) {
      return false;
    }

    return true;
  }

  /**
   * Detach from the allocator (cleanup TLS)
   */
  void shm_detach() {
    // TLS cleanup is handled automatically by pthread_key_create destructor
  }

  /**
   * Ensure thread-local storage is set up for the current thread
   *
   * This is the critical function for lock-free fast path allocation.
   * It checks TLS for a ThreadBlock, and if not found, allocates one
   * from the ProcessBlock (creating a ProcessBlock if needed).
   *
   * @return Pointer to the thread's ThreadBlock, or nullptr on failure
   */
  ThreadBlock* EnsureTls() {
    // Check if we already have a ThreadBlock in TLS
    void *tblock_data = HSHM_THREAD_MODEL->GetTls<void>(tblock_key_);
    if (tblock_data != nullptr) {
      return reinterpret_cast<ThreadBlock*>(tblock_data);
    }

    // Get or create ProcessBlock
    void *pblock_data = HSHM_THREAD_MODEL->GetTls<void>(pblock_key_);
    ProcessBlock *pblock = reinterpret_cast<ProcessBlock*>(pblock_data);
    if (pblock == nullptr) {
      // Find or allocate ProcessBlock for this process
      pblock = AllocateProcessBlock();
      if (pblock == nullptr) {
        return nullptr;
      }
      HSHM_THREAD_MODEL->SetTls<void>(pblock_key_, reinterpret_cast<void*>(pblock));
    }

    // Allocate ThreadBlock from ProcessBlock
    FullPtr<ThreadBlock> tblock_ptr = pblock->AllocateThreadBlock(backend_, thread_unit_);
    if (tblock_ptr.IsNull()) {
      // Try expanding ProcessBlock
      OffsetPtr expand_offset = alloc_.AllocateOffset(thread_unit_);
      if (!expand_offset.IsNull()) {
        pblock->Expand(expand_offset);
        tblock_ptr = pblock->AllocateThreadBlock(backend_, thread_unit_);
      }
    }

    if (tblock_ptr.IsNull()) {
      return nullptr;
    }

    // Store in TLS
    ThreadBlock *tblock = tblock_ptr.ptr_;
    HSHM_THREAD_MODEL->SetTls<void>(tblock_key_, reinterpret_cast<void*>(tblock));
    return tblock;
  }

  /**
   * Allocate a ProcessBlock for the current process
   *
   * @return Pointer to ProcessBlock, or nullptr on failure
   */
  ProcessBlock* AllocateProcessBlock() {
    int pid = getpid();

    // Lock the global process list
    header_->lock_.Lock(0);

    // Search for existing ProcessBlock for this PID
    if (!header_->alloc_procs_.empty()) {
      auto node = header_->alloc_procs_.peek(&alloc_);
      if (!node.IsNull()) {
        ProcessBlock *pblock = reinterpret_cast<ProcessBlock*>(node.ptr_);
        if (pblock->pid_ == pid) {
          header_->lock_.Unlock();
          return pblock;
        }
      }
    }

    // Allocate new ProcessBlock
    OffsetPtr pblock_offset = alloc_.AllocateOffset(process_unit_);
    if (pblock_offset.IsNull()) {
      header_->lock_.Unlock();
      return nullptr;
    }

    // Initialize ProcessBlock
    char *pblock_ptr = alloc_.backend_.data_ + pblock_offset.load();
    ProcessBlock *pblock = reinterpret_cast<ProcessBlock*>(pblock_ptr);
    new (pblock) ProcessBlock();

    if (!pblock->shm_init(backend_, pblock_ptr, process_unit_, pid)) {
      alloc_.FreeOffset(pblock_offset);
      header_->lock_.Unlock();
      return nullptr;
    }

    // Add to process list
    ShmPtr pblock_shm;
    pblock_shm.off_ = pblock_offset;
    FullPtr<pre::slist_node> pblock_node(&alloc_, pblock_shm);
    header_->alloc_procs_.emplace(&alloc_, pblock_node);
    header_->pid_count_++;

    header_->lock_.Unlock();
    return pblock;
  }

  /**
   * Allocate memory using the 3-tier allocation strategy
   *
   * @param size Size in bytes to allocate
   * @return Offset pointer to allocated memory, or null on failure
   */
  OffsetPtr AllocateOffset(size_t size) {
    // Tier 1: Fast path - try thread-local allocator (lock-free)
    ThreadBlock *tblock = EnsureTls();
    if (tblock != nullptr) {
      OffsetPtr ptr = tblock->Allocate(size);
      if (!ptr.IsNull()) {
        return ptr;
      }
    }

    // Tier 2: Medium path - try expanding ThreadBlock from ProcessBlock
    void *pblock_data = HSHM_THREAD_MODEL->GetTls<void>(pblock_key_);
    ProcessBlock *pblock = reinterpret_cast<ProcessBlock*>(pblock_data);
    if (pblock != nullptr) {
      OffsetPtr expand_offset = alloc_.AllocateOffset(size);
      if (!expand_offset.IsNull()) {
        pblock->Expand(expand_offset);
        if (tblock != nullptr) {
          OffsetPtr ptr = tblock->Allocate(size);
          if (!ptr.IsNull()) {
            return ptr;
          }
        }
      }
    }

    // Tier 3: Slow path - allocate directly from global allocator
    header_->lock_.Lock(0);
    OffsetPtr ptr = alloc_.AllocateOffset(size);
    header_->lock_.Unlock();
    return ptr;
  }

  /**
   * Reallocate memory to a new size
   *
   * @param offset Original offset pointer
   * @param new_size New size in bytes
   * @return New offset pointer, or null on failure
   */
  OffsetPtr ReallocateOffset(OffsetPtr offset, size_t new_size) {
    if (offset.IsNull()) {
      return AllocateOffset(new_size);
    }

    // Allocate new memory
    OffsetPtr new_offset = AllocateOffset(new_size);
    if (new_offset.IsNull()) {
      return new_offset;
    }

    // Copy old data to new location
    // We need to determine the size of the old allocation
    // For now, copy up to new_size (assuming old is at least as large)
    // TODO: Store allocation sizes to copy exact old size
    char *old_data = backend_.data_ + offset.load();
    char *new_data = backend_.data_ + new_offset.load();
    memcpy(new_data, old_data, new_size);

    // Free old allocation
    FreeOffset(offset);

    return new_offset;
  }

  /**
   * Free allocated memory
   *
   * @param offset Offset pointer to memory to free
   */
  void FreeOffset(OffsetPtr offset) {
    if (offset.IsNull()) {
      return;
    }

    // Determine which allocator owns this memory
    // For simplicity, try thread-local first, then global
    void *tblock_data = HSHM_THREAD_MODEL->GetTls<void>(tblock_key_);
    ThreadBlock *tblock = reinterpret_cast<ThreadBlock*>(tblock_data);
    if (tblock != nullptr) {
      // Check if offset is within thread block range
      // For now, just try to free it
      tblock->Free(offset);
      return;
    }

    // Fall back to global allocator
    header_->lock_.Lock(0);
    alloc_.FreeOffset(offset);
    header_->lock_.Unlock();
  }

  /**
   * Get the allocator ID (for compatibility with allocator interface)
   *
   * @return Allocator ID
   */
  AllocatorId GetId() const {
    return AllocatorId(backend_.GetId(), 0);
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_MEMORY_ALLOCATOR_MP_ALLOCATOR_H_

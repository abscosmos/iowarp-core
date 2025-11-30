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

#ifndef HSHM_MEMORY_ALLOCATOR_HEAP_H_
#define HSHM_MEMORY_ALLOCATOR_HEAP_H_

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/types/atomic.h"
#include "hermes_shm/util/errors.h"

namespace hshm::ipc {

/**
 * Heap helper class for simple bump-pointer allocation
 *
 * This is not an allocator itself, but a utility for implementing
 * allocators that need monotonically increasing offset allocation.
 *
 * @tparam ATOMIC Whether the heap pointer should be atomic
 */
template<bool ATOMIC>
class Heap {
 private:
  hipc::opt_atomic<size_t, ATOMIC> heap_;  /// Current heap offset
  size_t max_size_;                        /// Maximum heap size

 public:
  /**
   * Default constructor
   */
  HSHM_CROSS_FUN
  Heap() : heap_(0), max_size_(0) {}

  /**
   * Constructor with initial offset and max size
   *
   * @param initial_offset Initial heap offset
   * @param max_size Maximum size the heap can grow to
   */
  HSHM_CROSS_FUN
  Heap(size_t initial_offset, size_t max_size)
      : heap_(initial_offset), max_size_(max_size) {}

  /**
   * Initialize the heap
   *
   * @param initial_offset Initial heap offset
   * @param max_size Maximum size the heap can grow to
   */
  HSHM_CROSS_FUN
  void Init(size_t initial_offset, size_t max_size) {
    heap_.store(initial_offset);
    max_size_ = max_size;
  }

  /**
   * Allocate space from the heap
   *
   * @param size Number of bytes to allocate
   * @param align Alignment requirement (must be power of 2)
   * @return Offset of the allocated region
   * @throws OUT_OF_MEMORY if allocation would exceed max_size
   */
  HSHM_CROSS_FUN
  size_t Allocate(size_t size, size_t align = 8) {
    size_t off;
    size_t aligned_off;
    size_t end_off;

    do {
      // Get current heap offset
      off = heap_.load();

      // Align the offset to the specified alignment
      aligned_off = AlignSize(off, align);

      // Calculate end offset after this allocation
      end_off = aligned_off + size;

      // Check if allocation would exceed maximum size
      if (end_off > max_size_) {
        HSHM_THROW_ERROR(OUT_OF_MEMORY,
                         "Heap allocation exceeded max size: " +
                         std::to_string(end_off) + " > " +
                         std::to_string(max_size_));
      }

      // Try to atomically update heap pointer
      // If another thread modified heap_ between load() and compare_exchange,
      // this will fail and we'll retry
    } while (!heap_.compare_exchange_weak(off, end_off));

    return aligned_off;
  }

  /**
   * Get the current heap offset
   *
   * @return Current offset at the top of the heap
   */
  HSHM_CROSS_FUN
  size_t GetOffset() const {
    return heap_.load();
  }

  /**
   * Get the maximum heap size
   *
   * @return Maximum size the heap can grow to
   */
  HSHM_CROSS_FUN
  size_t GetMaxSize() const {
    return max_size_;
  }

  /**
   * Get the remaining space in the heap
   *
   * @return Number of bytes remaining
   */
  HSHM_CROSS_FUN
  size_t GetRemainingSize() const {
    size_t current = heap_.load();
    return (current < max_size_) ? (max_size_ - current) : 0;
  }

 private:
  /**
   * Align a size to the specified alignment
   *
   * @param size Size to align
   * @param align Alignment (must be power of 2)
   * @return Aligned size
   */
  HSHM_CROSS_FUN
  static size_t AlignSize(size_t size, size_t align) {
    return ((size + align - 1) / align) * align;
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_MEMORY_ALLOCATOR_HEAP_H_

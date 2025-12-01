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

#ifndef HSHM_INCLUDE_HSHM_DATA_STRUCTURES_IPC_RING_QUEUE_PRE_H
#define HSHM_INCLUDE_HSHM_DATA_STRUCTURES_IPC_RING_QUEUE_PRE_H

#include <cstddef>
#include <stdexcept>

namespace hshm::ipc::pre {

/**
 * Preallocated fixed-size ring queue
 *
 * A circular queue with compile-time fixed capacity. Uses modulo arithmetic
 * to wrap indices around the circular buffer. Thread-safe operations require
 * external synchronization.
 *
 * @tparam T Type of elements stored in the queue
 * @tparam COUNT Maximum number of elements the queue can hold
 */
template<typename T, size_t COUNT>
class ring_queue {
 private:
  size_t head_;      /**< Index of the next element to pop (front of queue) */
  size_t tail_;      /**< Index where the next element will be pushed (back of queue) */
  T data_[COUNT];    /**< Fixed-size array storing queue elements */

 public:
  /**
   * Default constructor
   * Initializes an empty queue
   */
  ring_queue() : head_(0), tail_(0) {}

  /**
   * Push an element to the back of the queue
   *
   * @param entry Element to add to the queue
   * @throws std::overflow_error if queue is full
   */
  void push(const T &entry) {
    if (size() >= COUNT) {
      throw std::overflow_error("ring_queue is full");
    }
    size_t tail = tail_ % COUNT;
    data_[tail] = entry;
    tail_++;
  }

  /**
   * Pop an element from the front of the queue
   *
   * @return The element at the front of the queue
   * @throws std::underflow_error if queue is empty
   */
  T pop() {
    if (tail_ - head_ == 0) {
      throw std::underflow_error("ring_queue is empty");
    }
    size_t head = head_ % COUNT;
    T result = data_[head];
    head_++;
    return result;
  }

  /**
   * Get the number of elements currently in the queue
   *
   * @return Number of elements in the queue
   */
  size_t size() const {
    return tail_ - head_;
  }

  /**
   * Check if the queue is empty
   *
   * @return true if queue has no elements, false otherwise
   */
  bool empty() const {
    return size() == 0;
  }

  /**
   * Check if the queue is full
   *
   * @return true if queue is at maximum capacity, false otherwise
   */
  bool full() const {
    return size() >= COUNT;
  }

  /**
   * Get the maximum capacity of the queue
   *
   * @return Maximum number of elements the queue can hold
   */
  constexpr size_t capacity() const {
    return COUNT;
  }

  /**
   * Peek at the front element without removing it
   *
   * @return Reference to the element at the front of the queue
   * @throws std::underflow_error if queue is empty
   */
  const T& front() const {
    if (empty()) {
      throw std::underflow_error("ring_queue is empty");
    }
    size_t head = head_ % COUNT;
    return data_[head];
  }

  /**
   * Peek at the back element without removing it
   *
   * @return Reference to the element at the back of the queue
   * @throws std::underflow_error if queue is empty
   */
  const T& back() const {
    if (empty()) {
      throw std::underflow_error("ring_queue is empty");
    }
    size_t tail = (tail_ - 1) % COUNT;
    return data_[tail];
  }

  /**
   * Clear all elements from the queue
   */
  void clear() {
    head_ = 0;
    tail_ = 0;
  }
};

}  // namespace hshm::ipc::pre

#endif  // HSHM_INCLUDE_HSHM_DATA_STRUCTURES_IPC_RING_QUEUE_PRE_H

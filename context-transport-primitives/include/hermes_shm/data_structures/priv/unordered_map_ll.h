/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef HSHM_DATA_STRUCTURES_PRIV_UNORDERED_MAP_LL_H_
#define HSHM_DATA_STRUCTURES_PRIV_UNORDERED_MAP_LL_H_

#include "hermes_shm/data_structures/priv/vector.h"
#include "hermes_shm/types/hash.h"
#include "hermes_shm/memory/allocator/malloc_allocator.h"

namespace hshm::priv {

/** Result of insert / insert_or_assign operations */
template <typename T>
struct InsertResult {
  bool inserted;
  T *value;
};

/**
 * GPU-compatible unordered map using open addressing with linear probing.
 *
 * Backed by a single priv::vector of slots. Each slot is either empty,
 * occupied, or a tombstone. The map does not auto-rehash; callers must
 * ensure the capacity is sufficient for their workload.
 *
 * External locking is required for thread safety.
 *
 * @tparam Key      Key type (must support copy/move and operator==)
 * @tparam T        Mapped value type
 * @tparam AllocT   Allocator type (e.g., BuddyAllocator, ThreadAllocator)
 * @tparam Hash     Hash functor (defaults to hshm::hash<Key>)
 * @tparam KeyEqual Equality functor (defaults to hshm::equal_to<Key>)
 */
template <typename Key, typename T,
          typename AllocT = hshm::ipc::MallocAllocator,
          typename Hash = hshm::hash<Key>,
          typename KeyEqual = hshm::equal_to<Key>>
class unordered_map_ll {
 public:
  using key_type = Key;
  using mapped_type = T;
  using size_type = std::size_t;
  using hasher = Hash;
  using key_equal = KeyEqual;

 private:
  static constexpr uint32_t kEmpty = 0;
  static constexpr uint32_t kOccupied = 1;
  static constexpr uint32_t kTombstone = 2;

  struct Slot {
    uint32_t state_;
    Key key_;
    T value_;

    HSHM_CROSS_FUN Slot() : state_(kEmpty), key_(), value_() {}
    HSHM_CROSS_FUN Slot(const Slot &o)
        : state_(o.state_), key_(o.key_), value_(o.value_) {}
    HSHM_CROSS_FUN Slot &operator=(const Slot &o) {
      if (this != &o) {
        state_ = o.state_;
        key_ = o.key_;
        value_ = o.value_;
      }
      return *this;
    }
  };

  vector<Slot, AllocT> slots_;
  size_type size_;
  AllocT *alloc_;
  Hash hash_fn_;
  KeyEqual key_eq_;

  /** Find the slot index for a key (returns capacity if not found) */
  HSHM_INLINE_CROSS_FUN
  size_type find_slot(const Key &key) const {
    size_type cap = slots_.size();
    if (cap == 0) return cap;
    size_type h = hash_fn_(key) % cap;
    for (size_type i = 0; i < cap; ++i) {
      size_type idx = (h + i) % cap;
      if (slots_[idx].state_ == kEmpty) return cap;
      if (slots_[idx].state_ == kOccupied && key_eq_(slots_[idx].key_, key)) {
        return idx;
      }
    }
    return cap;
  }

  /** Find the first available slot (empty or tombstone) for insertion.
   *  Also checks for existing key. Returns {found_idx, is_existing}. */
  HSHM_INLINE_CROSS_FUN
  void find_insert_slot(const Key &key, size_type &out_idx,
                        bool &out_existing) const {
    size_type cap = slots_.size();
    size_type h = hash_fn_(key) % cap;
    size_type first_avail = cap;
    out_existing = false;
    out_idx = cap;
    for (size_type i = 0; i < cap; ++i) {
      size_type idx = (h + i) % cap;
      if (slots_[idx].state_ == kOccupied && key_eq_(slots_[idx].key_, key)) {
        out_idx = idx;
        out_existing = true;
        return;
      }
      if (slots_[idx].state_ != kOccupied && first_avail == cap) {
        first_avail = idx;
      }
      if (slots_[idx].state_ == kEmpty) {
        break;
      }
    }
    out_idx = first_avail;
  }

 public:
  /**
   * Constructor (host-only, uses global MallocAllocator)
   * @param capacity Initial number of slots (hash table size)
   */
#if HSHM_IS_HOST
  explicit unordered_map_ll(size_type capacity = 16)
      : slots_(HSHM_MALLOC), size_(0), alloc_(HSHM_MALLOC), hash_fn_(), key_eq_() {
    slots_.resize(capacity);
  }
#endif

  /**
   * Constructor with explicit allocator
   * @param alloc Allocator for the backing vector
   * @param capacity Initial number of slots (hash table size)
   */
  HSHM_CROSS_FUN
  explicit unordered_map_ll(AllocT *alloc, size_type capacity = 16)
      : slots_(alloc), size_(0), alloc_(alloc), hash_fn_(), key_eq_() {
    slots_.resize(capacity);
  }

  HSHM_CROSS_FUN ~unordered_map_ll() = default;

  /** Rehash the map to a new capacity, re-inserting all occupied entries.
   *  Returns false if allocation fails; the map is left unchanged. */
  HSHM_CROSS_FUN
  bool rehash(size_type new_cap) {
    // Try to allocate new slots first (before destroying old)
    vector<Slot, AllocT> new_slots(alloc_);
    if (!new_slots.resize(new_cap)) {
      return false;  // Allocation failed; keep existing map intact
    }

    // Move old slots out; slots_ becomes empty after move
    vector<Slot, AllocT> old_slots(static_cast<vector<Slot, AllocT>&&>(slots_));
    size_type old_size = old_slots.size();
    // Install the new (empty) slots
    slots_ = static_cast<vector<Slot, AllocT>&&>(new_slots);
    size_ = 0;
    for (size_type i = 0; i < old_size; ++i) {
      if (old_slots[i].state_ == kOccupied) {
        insert_no_rehash(old_slots[i].key_, old_slots[i].value_);
      }
    }
    return true;
  }

 private:
  /** Check load factor and rehash if needed (>75% full).
   *  Silently skips rehash if allocation fails. */
  HSHM_INLINE_CROSS_FUN
  void maybe_rehash() {
    // Rehash when load factor > 75%: size * 4 > capacity * 3
    if (size_ * 4 > slots_.size() * 3) {
      rehash(slots_.size() * 2);  // Best-effort; map degrades on failure
    }
  }

  /** Insert without rehash check (used internally by rehash) */
  HSHM_CROSS_FUN
  InsertResult<T> insert_no_rehash(const Key &key, const T &value) {
    size_type idx;
    bool existing;
    find_insert_slot(key, idx, existing);
    if (idx >= slots_.size()) {
      return {false, nullptr};
    }
    if (existing) {
      return {false, &slots_[idx].value_};
    }
    slots_[idx].state_ = kOccupied;
    slots_[idx].key_ = key;
    slots_[idx].value_ = value;
    ++size_;
    return {true, &slots_[idx].value_};
  }

 public:
  /** Insert or update a key-value pair */
  HSHM_CROSS_FUN
  InsertResult<T> insert_or_assign(const Key &key, const T &value) {
    size_type idx;
    bool existing;
    find_insert_slot(key, idx, existing);
    if (existing) {
      slots_[idx].value_ = value;
      return {false, &slots_[idx].value_};
    }
    if (idx >= slots_.size()) {
      if (!rehash(slots_.size() * 2)) return {false, nullptr};
      return insert_or_assign(key, value);
    }
    slots_[idx].state_ = kOccupied;
    slots_[idx].key_ = key;
    slots_[idx].value_ = value;
    ++size_;
    maybe_rehash();
    return {true, &slots_[idx].value_};
  }

  /** Insert a key-value pair (only if key doesn't exist) */
  HSHM_CROSS_FUN
  InsertResult<T> insert(const Key &key, const T &value) {
    size_type idx;
    bool existing;
    find_insert_slot(key, idx, existing);
    if (existing) {
      return {false, &slots_[idx].value_};
    }
    if (idx >= slots_.size()) {
      if (!rehash(slots_.size() * 2)) return {false, nullptr};
      return insert(key, value);
    }
    slots_[idx].state_ = kOccupied;
    slots_[idx].key_ = key;
    slots_[idx].value_ = value;
    ++size_;
    maybe_rehash();
    return {true, &slots_[idx].value_};
  }

  /** Access element (creates with default value if absent) */
  HSHM_CROSS_FUN
  T &operator[](const Key &key) {
    size_type idx;
    bool existing;
    find_insert_slot(key, idx, existing);
    if (existing) {
      return slots_[idx].value_;
    }
    if (idx >= slots_.size()) {
      rehash(slots_.size() * 2);
      return operator[](key);
    }
    slots_[idx].state_ = kOccupied;
    slots_[idx].key_ = key;
    slots_[idx].value_ = T();
    ++size_;
    maybe_rehash();
    return slots_[idx].value_;
  }

  /** Find an element */
  HSHM_CROSS_FUN
  T *find(const Key &key) {
    size_type idx = find_slot(key);
    if (idx < slots_.size()) {
      return &slots_[idx].value_;
    }
    return nullptr;
  }

  /** Find an element (const) */
  HSHM_CROSS_FUN
  const T *find(const Key &key) const {
    size_type idx = find_slot(key);
    if (idx < slots_.size()) {
      return &slots_[idx].value_;
    }
    return nullptr;
  }

  /** Check if key exists */
  HSHM_CROSS_FUN
  bool contains(const Key &key) const {
    return find(key) != nullptr;
  }

  /** Count occurrences (0 or 1) */
  HSHM_CROSS_FUN
  size_type count(const Key &key) const {
    return contains(key) ? 1 : 0;
  }

  /** Erase element by key */
  HSHM_CROSS_FUN
  size_type erase(const Key &key) {
    size_type idx = find_slot(key);
    if (idx < slots_.size()) {
      slots_[idx].state_ = kTombstone;
      slots_[idx].key_ = Key();
      slots_[idx].value_ = T();
      --size_;
      return 1;
    }
    return 0;
  }

  /** Clear all elements */
  HSHM_CROSS_FUN
  void clear() {
    for (size_type i = 0; i < slots_.size(); ++i) {
      if (slots_[i].state_ == kOccupied) {
        slots_[i].key_ = Key();
        slots_[i].value_ = T();
      }
      slots_[i].state_ = kEmpty;
    }
    size_ = 0;
  }

  /** Total number of elements */
  HSHM_INLINE_CROSS_FUN
  size_type size() const { return size_; }

  /** Check if empty */
  HSHM_INLINE_CROSS_FUN
  bool empty() const { return size_ == 0; }

  /** Number of slots */
  HSHM_INLINE_CROSS_FUN
  size_type bucket_count() const { return slots_.size(); }

  /** Apply function to each occupied entry */
  template <typename Func>
  HSHM_CROSS_FUN void for_each(Func fn) {
    for (size_type i = 0; i < slots_.size(); ++i) {
      if (slots_[i].state_ == kOccupied) {
        fn(slots_[i].key_, slots_[i].value_);
      }
    }
  }

  /** Apply function to each occupied entry (const) */
  template <typename Func>
  HSHM_CROSS_FUN void for_each(Func fn) const {
    for (size_type i = 0; i < slots_.size(); ++i) {
      if (slots_[i].state_ == kOccupied) {
        fn(slots_[i].key_, slots_[i].value_);
      }
    }
  }
};

}  // namespace hshm::priv

#endif  // HSHM_DATA_STRUCTURES_PRIV_UNORDERED_MAP_LL_H_

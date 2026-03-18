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

/**
 * GPU implementation of CTE Core ChiMod methods.
 *
 * Uses chi::priv data structures (string, vector) backed by the
 * ThreadAllocator which provides per-block BuddyAllocator partitions,
 * eliminating cross-block allocator contention (CUDA Error 700).
 *
 * Note: core_tasks.h is included here (not in the header) to keep GPU
 * compilation isolated from CPU-only task constructors that use HSHM_MALLOC.
 */

#include "wrp_cte/core/core_gpu_runtime.h"
#include "wrp_cte/core/core_tasks.h"
#include <hermes_shm/data_structures/priv/vector.h>
#include <hermes_shm/data_structures/priv/string.h>
#include <hermes_shm/thread/lock/mutex.h>

namespace wrp_cte::core {

/** Default number of blob map slots (open addressing) */
static constexpr chi::u32 kDefaultBlobMapCapacity = 4096;

/** FNV-1a hash for chi::priv::string */
HSHM_GPU_FUN static chi::u32 HashString(const chi::priv::string &s) {
  chi::u32 hash = 2166136261u;
  const char *data = s.data();
  size_t len = s.size();
  for (size_t i = 0; i < len; ++i) {
    hash ^= static_cast<chi::u32>(data[i]);
    hash *= 16777619u;
  }
  return hash;
}

/**
 * GPU-side blob entry using chi::priv data structures.
 */
struct GpuBlobEntry {
  chi::priv::string key_;       // compound key "major.minor.blob_name"
  chi::u64 data_ptr_;           // GPU pointer to blob data
  chi::u64 size_;               // blob size in bytes
  float score_;
  Timestamp last_modified_;
  Timestamp last_read_;

  HSHM_CROSS_FUN GpuBlobEntry()
      : key_(CHI_PRIV_ALLOC),
        data_ptr_(0), size_(0), score_(0.0f),
        last_modified_(0), last_read_(0) {}

  HSHM_CROSS_FUN GpuBlobEntry(const GpuBlobEntry &other)
      : key_(other.key_),
        data_ptr_(other.data_ptr_), size_(other.size_),
        score_(other.score_),
        last_modified_(other.last_modified_),
        last_read_(other.last_read_) {}

  HSHM_CROSS_FUN GpuBlobEntry &operator=(const GpuBlobEntry &other) {
    if (this != &other) {
      key_ = other.key_;
      data_ptr_ = other.data_ptr_;
      size_ = other.size_;
      score_ = other.score_;
      last_modified_ = other.last_modified_;
      last_read_ = other.last_read_;
    }
    return *this;
  }
};

/**
 * GPU-compatible hash map for blob entries using open addressing.
 *
 * Fixed-size slot array with linear probing. Each slot is either empty
 * (key_.size() == 0) or occupied. Tombstones use a sentinel key.
 *
 * Uses a single mutex for simplicity (blob operations are already
 * fast with O(1) lookup; contention is the rare case).
 */
static constexpr chi::u32 kBlobMapEmpty = 0;
static constexpr chi::u32 kBlobMapOccupied = 1;
static constexpr chi::u32 kBlobMapTombstone = 2;

struct GpuBlobSlot {
  chi::u32 state_;
  GpuBlobEntry entry_;

  HSHM_CROSS_FUN GpuBlobSlot()
      : state_(kBlobMapEmpty), entry_() {}

  HSHM_CROSS_FUN GpuBlobSlot(const GpuBlobSlot &other)
      : state_(other.state_), entry_(other.entry_) {}

  HSHM_CROSS_FUN GpuBlobSlot &operator=(const GpuBlobSlot &other) {
    if (this != &other) {
      state_ = other.state_;
      entry_ = other.entry_;
    }
    return *this;
  }
};

struct GpuBlobMap {
  chi::priv::vector<GpuBlobSlot> slots_;
  chi::u32 size_;
  hshm::Mutex lock_;

  HSHM_GPU_FUN GpuBlobMap() : slots_(CHI_PRIV_ALLOC), size_(0) {}

  /** Initialize the map with a given number of slots */
  HSHM_GPU_FUN void Init(chi::u32 capacity) {
    size_ = 0;
    lock_.Init();
    slots_.resize(capacity);
  }

  /** Get the lock */
  HSHM_GPU_FUN hshm::Mutex &Lock(const chi::priv::string &key) {
    (void)key;
    return lock_;
  }

  /** Find a blob entry by key. Returns pointer or nullptr. */
  HSHM_GPU_FUN GpuBlobEntry *Find(const chi::priv::string &key) {
    chi::u32 cap = slots_.size();
    if (cap == 0) return nullptr;
    chi::u32 h = HashString(key) % cap;
    for (chi::u32 i = 0; i < cap; ++i) {
      chi::u32 idx = (h + i) % cap;
      if (slots_[idx].state_ == kBlobMapEmpty) return nullptr;
      if (slots_[idx].state_ == kBlobMapOccupied &&
          slots_[idx].entry_.key_ == key) {
        return &slots_[idx].entry_;
      }
    }
    return nullptr;
  }

  /** Insert or find a blob entry. Returns pointer to the entry. */
  HSHM_GPU_FUN GpuBlobEntry *InsertOrFind(const chi::priv::string &key) {
    chi::u32 cap = slots_.size();
    if (cap == 0) return nullptr;
    chi::u32 h = HashString(key) % cap;
    chi::u32 first_avail = cap;
    for (chi::u32 i = 0; i < cap; ++i) {
      chi::u32 idx = (h + i) % cap;
      if (slots_[idx].state_ == kBlobMapOccupied &&
          slots_[idx].entry_.key_ == key) {
        return &slots_[idx].entry_;
      }
      if (slots_[idx].state_ != kBlobMapOccupied && first_avail == cap) {
        first_avail = idx;
      }
      if (slots_[idx].state_ == kBlobMapEmpty) {
        break;
      }
    }
    if (first_avail >= cap) return nullptr;
    slots_[first_avail].state_ = kBlobMapOccupied;
    slots_[first_avail].entry_.key_ = key;
    ++size_;
    return &slots_[first_avail].entry_;
  }

  /** Erase a blob entry by key. Returns true if erased. */
  HSHM_GPU_FUN bool Erase(const chi::priv::string &key) {
    chi::u32 cap = slots_.size();
    if (cap == 0) return false;
    chi::u32 h = HashString(key) % cap;
    for (chi::u32 i = 0; i < cap; ++i) {
      chi::u32 idx = (h + i) % cap;
      if (slots_[idx].state_ == kBlobMapEmpty) return false;
      if (slots_[idx].state_ == kBlobMapOccupied &&
          slots_[idx].entry_.key_ == key) {
        slots_[idx].state_ = kBlobMapTombstone;
        slots_[idx].entry_.key_ = chi::priv::string(CHI_PRIV_ALLOC);
        --size_;
        return true;
      }
    }
    return false;
  }

  /** Erase all entries whose key starts with prefix. Returns count erased. */
  HSHM_GPU_FUN int EraseByPrefix(const chi::priv::string &prefix) {
    int erased = 0;
    chi::u32 cap = slots_.size();
    for (chi::u32 i = 0; i < cap; ++i) {
      if (slots_[i].state_ != kBlobMapOccupied) continue;
      const chi::priv::string &key = slots_[i].entry_.key_;
      bool match = (key.size() >= prefix.size());
      if (match) {
        const char *kd = key.data();
        const char *pd = prefix.data();
        for (size_t c = 0; c < prefix.size() && match; ++c) {
          match = (kd[c] == pd[c]);
        }
      }
      if (match) {
        slots_[i].state_ = kBlobMapTombstone;
        slots_[i].entry_.key_ = chi::priv::string(CHI_PRIV_ALLOC);
        --size_;
        ++erased;
      }
    }
    return erased;
  }
};

/**
 * GPU-resident metadata store for CTE Core GpuRuntime.
 * Uses chi::priv data structures backed by ThreadAllocator.
 */
struct GpuMetadata {
  hshm::Mutex tag_lock_;
  GpuBlobMap blob_map_;
  chi::priv::vector<TagInfo> tags_;

  HSHM_GPU_FUN GpuMetadata() : tags_(CHI_PRIV_ALLOC) {
    tag_lock_.Init();
    blob_map_.Init(kDefaultBlobMapCapacity);
  }
};

//==============================================================================
// Helper: build compound key "major.minor.blob_name"
//==============================================================================

/**
 * Append the decimal digits of a u32 to a priv::string.
 */
HSHM_GPU_FUN static void AppendU32(chi::priv::string &s, chi::u32 val) {
  if (val == 0) {
    s.push_back('0');
    return;
  }
  // Find digit count
  int ndigits = 0;
  chi::u32 tmp = val;
  while (tmp > 0) { ++ndigits; tmp /= 10; }
  // Append digits in forward order
  size_t start = s.size();
  s.resize(start + ndigits);
  for (int i = ndigits - 1; i >= 0; --i) {
    s[start + i] = '0' + static_cast<char>(val % 10);
    val /= 10;
  }
}

HSHM_GPU_FUN static chi::priv::string MakeCompoundKey(
    const TagId &tag_id, const char *blob_name, int blob_name_len) {
  // Build directly into heap-backed priv::string (no stack buffers)
  chi::priv::string ck(CHI_PRIV_ALLOC);
  // Reserve: up to 10+1+10+1+blob_name_len
  ck.reserve(22 + blob_name_len);
  AppendU32(ck, tag_id.major_);
  ck.push_back('.');
  AppendU32(ck, tag_id.minor_);
  ck.push_back('.');
  for (int i = 0; i < blob_name_len; ++i) {
    ck.push_back(blob_name[i]);
  }
  return ck;
}

//==============================================================================
// Stub methods (no-ops on GPU)
//==============================================================================

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::RegisterTarget(
    hipc::FullPtr<RegisterTargetTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::UnregisterTarget(
    hipc::FullPtr<UnregisterTargetTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::ListTargets(
    hipc::FullPtr<ListTargetsTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::StatTargets(
    hipc::FullPtr<StatTargetsTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

//==============================================================================
// EnsureMetaInit — double-checked locking with threadfence
//==============================================================================

HSHM_GPU_FUN void GpuRuntime::EnsureMetaInit() {
  GpuMetadata *m = *reinterpret_cast<GpuMetadata *volatile *>(&meta_);
  if (m != nullptr) return;
  hshm::ScopedMutex guard(init_lock_, 0);
  m = *reinterpret_cast<GpuMetadata *volatile *>(&meta_);
  if (m != nullptr) return;
  CHI_PRIV_ALLOC_T *alloc = CHI_PRIV_ALLOC;
  hipc::FullPtr<GpuMetadata> ptr = alloc->template AllocateObjs<GpuMetadata>(1);
  new (ptr.ptr_) GpuMetadata();
  __threadfence();
  meta_ = ptr.ptr_;
  __threadfence();
}

//==============================================================================
// Tag operations
//==============================================================================

HSHM_GPU_FUN TagInfo *GpuRuntime::FindTagById(const TagId &tag_id) {
  if (!meta_) return nullptr;
  for (size_t i = 0; i < meta_->tags_.size(); ++i) {
    if (meta_->tags_[i].tag_id_ == tag_id) {
      return &meta_->tags_[i];
    }
  }
  return nullptr;
}

HSHM_GPU_FUN TagId *GpuRuntime::FindTagIdByName(const chi::priv::string &name) {
  if (!meta_) return nullptr;
  for (size_t i = 0; i < meta_->tags_.size(); ++i) {
    if (meta_->tags_[i].tag_name_ == name) {
      return &meta_->tags_[i].tag_id_;
    }
  }
  return nullptr;
}

HSHM_GPU_FUN TagInfo *GpuRuntime::UpsertTag(const chi::priv::string &tag_name,
                                             const TagId &tag_id) {
  if (!meta_) return nullptr;
  // Check if tag exists by name
  for (size_t i = 0; i < meta_->tags_.size(); ++i) {
    if (meta_->tags_[i].tag_name_ == tag_name) {
      return &meta_->tags_[i];
    }
  }
  // Insert new
  TagInfo info(tag_name, tag_id);
  meta_->tags_.push_back(info);
  return &meta_->tags_.back();
}

//==============================================================================
// Blob map operations
//==============================================================================

HSHM_GPU_FUN chi::priv::string GpuRuntime::MakeBlobKey(const TagId &tag_id,
                                                         const chi::priv::string &blob_name) {
  return MakeCompoundKey(tag_id, blob_name.data(),
                         static_cast<int>(blob_name.size()));
}

//==============================================================================
// GetOrCreateTag
//==============================================================================

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::GetOrCreateTag(
    hipc::FullPtr<GetOrCreateTagTask<CreateParams>> task,
    chi::gpu::RunContext &rctx) {
  (void)rctx;
  if (!chi::IpcManager::IsWarpScheduler()) co_return;
  EnsureMetaInit();
  hshm::ScopedMutex guard(meta_->tag_lock_, 0);

  chi::priv::string name(CHI_PRIV_ALLOC, task->tag_name_.data());
  TagId preferred_id = task->tag_id_;

  // Look up existing tag by name
  TagId *existing = FindTagIdByName(name);
  if (existing != nullptr) {
    task->tag_id_ = *existing;
    task->return_code_ = 0;
    co_return;
  }

  // Assign new ID
  TagId tag_id;
  if (preferred_id.major_ != 0 || preferred_id.minor_ != 0) {
    tag_id = preferred_id;
  } else {
    tag_id.major_ = container_id_;
    tag_id.minor_ = atomicAdd(&next_tag_minor_, 1u) + 1;
  }

  // Insert
  UpsertTag(name, tag_id);

  task->tag_id_ = tag_id;
  task->return_code_ = 0;
  co_return;
}

//==============================================================================
// GetTagSize
//==============================================================================

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::GetTagSize(
    hipc::FullPtr<GetTagSizeTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

//==============================================================================
// DelTag
//==============================================================================

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::DelTag(
    hipc::FullPtr<DelTagTask> task, chi::gpu::RunContext &rctx) {
  (void)rctx;
  if (!chi::IpcManager::IsWarpScheduler()) co_return;
  EnsureMetaInit();
  hshm::ScopedMutex tag_guard(meta_->tag_lock_, 0);

  TagId tag_id = task->tag_id_;

  // Resolve tag_id from name if needed
  if (tag_id.IsNull()) {
    chi::priv::string name(CHI_PRIV_ALLOC, task->tag_name_.data());
    if (name.size() == 0) {
      task->return_code_ = 1;
      co_return;
    }
    TagId *found = FindTagIdByName(name);
    if (found == nullptr) {
      task->return_code_ = 1;
      co_return;
    }
    tag_id = *found;
    task->tag_id_ = tag_id;
  }

  // Build prefix "major.minor." for matching and erase all matching blobs
  chi::priv::string prefix = MakeCompoundKey(tag_id, "", 0);
  meta_->blob_map_.EraseByPrefix(prefix);

  // Erase tag from tag store
  TagInfo *tag_info = FindTagById(tag_id);
  if (tag_info != nullptr) {
    size_t tag_idx = tag_info - meta_->tags_.data();
    size_t last = meta_->tags_.size() - 1;
    if (tag_idx != last) {
      meta_->tags_[tag_idx] = meta_->tags_[last];
    }
    meta_->tags_.pop_back();
  }

  task->return_code_ = 0;
  co_return;
}

//==============================================================================
// GetContainedBlobs
//==============================================================================

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::GetContainedBlobs(
    hipc::FullPtr<GetContainedBlobsTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

//==============================================================================
// PutBlob
//==============================================================================

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::PutBlob(
    hipc::FullPtr<PutBlobTask> task, chi::gpu::RunContext &rctx) {
  (void)rctx;
  if (!chi::IpcManager::IsWarpScheduler()) co_return;
  EnsureMetaInit();
  TagId tag_id = task->tag_id_;
  const char *blob_name = task->blob_name_.data();
  int blob_name_len = static_cast<int>(task->blob_name_.size());
  chi::u64 size = task->size_;
  float blob_score = task->score_;

  // Validate inputs
  if (size == 0) { task->return_code_ = 2; co_return; }
  if (task->blob_data_.IsNull()) { task->return_code_ = 3; co_return; }
  if (blob_name_len == 0) { task->return_code_ = 4; co_return; }
  if (blob_score < 0.0f) blob_score = 1.0f;
  if (blob_score > 1.0f) { task->return_code_ = 5; co_return; }

  // Resolve blob data pointer
  auto data_ptr = CHI_IPC->ToFullPtr(task->blob_data_);
  if (data_ptr.IsNull()) { task->return_code_ = 6; co_return; }

  // Build compound key and lock the bucket
  chi::priv::string ck = MakeCompoundKey(tag_id, blob_name, blob_name_len);
  hshm::ScopedMutex blob_guard(meta_->blob_map_.Lock(ck), 0);

  GpuBlobEntry *entry = meta_->blob_map_.Find(ck);
  if (entry == nullptr) {
    // Create new blob entry
    entry = meta_->blob_map_.InsertOrFind(ck);
    entry->data_ptr_ = reinterpret_cast<chi::u64>(data_ptr.ptr_);
    entry->size_ = size;
    entry->score_ = blob_score;
    entry->last_modified_ = GetCurrentTimeNs();
    entry->last_read_ = 0;

    // Update tag total_size_
    hshm::ScopedMutex tag_guard(meta_->tag_lock_, 0);
    TagInfo *tag = FindTagById(tag_id);
    if (tag != nullptr) tag->total_size_ += size;
  } else {
    // Update existing blob
    chi::u64 old_size = entry->size_;
    entry->data_ptr_ = reinterpret_cast<chi::u64>(data_ptr.ptr_);
    entry->size_ = size;
    entry->score_ = blob_score;
    entry->last_modified_ = GetCurrentTimeNs();

    // Update tag total_size_
    hshm::ScopedMutex tag_guard(meta_->tag_lock_, 0);
    TagInfo *tag = FindTagById(tag_id);
    if (tag != nullptr) {
      tag->total_size_ = tag->total_size_ - old_size + size;
    }
  }

  task->return_code_ = 0;
  co_return;
}

//==============================================================================
// GetBlob
//==============================================================================

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::GetBlob(
    hipc::FullPtr<GetBlobTask> task, chi::gpu::RunContext &rctx) {
  (void)rctx;
  if (!chi::IpcManager::IsWarpScheduler()) co_return;
  EnsureMetaInit();
  TagId tag_id = task->tag_id_;
  const char *blob_name = task->blob_name_.data();
  int blob_name_len = static_cast<int>(task->blob_name_.size());
  chi::u64 offset = task->offset_;
  chi::u64 size = task->size_;

  if (size == 0 || blob_name_len == 0) { task->return_code_ = 1; co_return; }

  chi::priv::string ck = MakeCompoundKey(tag_id, blob_name, blob_name_len);
  hshm::ScopedMutex blob_guard(meta_->blob_map_.Lock(ck), 0);

  GpuBlobEntry *entry = meta_->blob_map_.Find(ck);
  if (entry == nullptr) { task->return_code_ = 1; co_return; }

  // Resolve output buffer
  auto out_ptr = CHI_IPC->ToFullPtr(task->blob_data_);
  if (out_ptr.IsNull()) { task->return_code_ = 1; co_return; }
  // Copy data from blob to output buffer
  char *src = reinterpret_cast<char *>(entry->data_ptr_) + offset;
  char *dst = reinterpret_cast<char *>(out_ptr.ptr_);
  chi::u64 can_read = (offset < entry->size_) ? (entry->size_ - offset) : 0;
  chi::u64 to_read = (can_read < size) ? can_read : size;
  memcpy(dst, src, to_read);

  entry->last_read_ = GetCurrentTimeNs();
  task->return_code_ = (to_read == size) ? 0 : 1;
  co_return;
}

//==============================================================================
// ReorganizeBlob
//==============================================================================

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::ReorganizeBlob(
    hipc::FullPtr<ReorganizeBlobTask> task, chi::gpu::RunContext &rctx) {
  (void)rctx;
  if (!chi::IpcManager::IsWarpScheduler()) co_return;
  EnsureMetaInit();
  TagId tag_id = task->tag_id_;
  const char *blob_name = task->blob_name_.data();
  int blob_name_len = static_cast<int>(task->blob_name_.size());
  float new_score = task->new_score_;

  if (blob_name_len == 0 || new_score < 0.0f || new_score > 1.0f) {
    task->return_code_ = 1;
    co_return;
  }

  chi::priv::string ck = MakeCompoundKey(tag_id, blob_name, blob_name_len);
  hshm::ScopedMutex blob_guard(meta_->blob_map_.Lock(ck), 0);

  GpuBlobEntry *entry = meta_->blob_map_.Find(ck);
  if (entry == nullptr) { task->return_code_ = 3; co_return; }

  entry->score_ = new_score;
  task->return_code_ = 0;
  co_return;
}

//==============================================================================
// DelBlob
//==============================================================================

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::DelBlob(
    hipc::FullPtr<DelBlobTask> task, chi::gpu::RunContext &rctx) {
  (void)rctx;
  if (!chi::IpcManager::IsWarpScheduler()) co_return;
  EnsureMetaInit();
  TagId tag_id = task->tag_id_;
  const char *blob_name = task->blob_name_.data();
  int blob_name_len = static_cast<int>(task->blob_name_.size());

  if (blob_name_len == 0) { task->return_code_ = 1; co_return; }

  chi::priv::string ck = MakeCompoundKey(tag_id, blob_name, blob_name_len);
  hshm::ScopedMutex blob_guard(meta_->blob_map_.Lock(ck), 0);

  GpuBlobEntry *entry = meta_->blob_map_.Find(ck);
  if (entry == nullptr) { task->return_code_ = 1; co_return; }

  chi::u64 blob_size = entry->size_;
  meta_->blob_map_.Erase(ck);

  // Update tag total_size_
  hshm::ScopedMutex tag_guard(meta_->tag_lock_, 0);
  TagInfo *tag = FindTagById(tag_id);
  if (tag != nullptr) {
    tag->total_size_ =
        (blob_size <= tag->total_size_) ? tag->total_size_ - blob_size : 0;
  }

  task->return_code_ = 0;
  co_return;
}

//==============================================================================
// Remaining stubs
//==============================================================================

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::GetBlobScore(
    hipc::FullPtr<GetBlobScoreTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::GetBlobSize(
    hipc::FullPtr<GetBlobSizeTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::GetBlobInfo(
    hipc::FullPtr<GetBlobInfoTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::PollTelemetryLog(
    hipc::FullPtr<PollTelemetryLogTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::TagQuery(
    hipc::FullPtr<TagQueryTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::BlobQuery(
    hipc::FullPtr<BlobQueryTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::GetTargetInfo(
    hipc::FullPtr<GetTargetInfoTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::FlushMetadata(
    hipc::FullPtr<FlushMetadataTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::FlushData(
    hipc::FullPtr<FlushDataTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

//==============================================================================
// LocalAllocLoadTask / LocalSaveTask — GPU implementations
//==============================================================================

HSHM_GPU_FUN hipc::FullPtr<chi::Task> GpuRuntime::LocalAllocLoadTask(
    chi::u32 method, chi::LocalLoadTaskArchive &archive) {
  auto *ipc = CHI_IPC;
  hipc::FullPtr<chi::Task> task_ptr;
  switch (method) {
    case Method::kPutBlob: {
      auto new_task = ipc->template NewTask<PutBlobTask>();
      archive >> *new_task.ptr_;
      task_ptr = new_task.template Cast<chi::Task>();
      break;
    }
    case Method::kGetBlob: {
      auto new_task = ipc->template NewTask<GetBlobTask>();
      archive >> *new_task.ptr_;
      task_ptr = new_task.template Cast<chi::Task>();
      break;
    }
    case Method::kGetOrCreateTag: {
      auto new_task = ipc->template NewTask<GetOrCreateTagTask<CreateParams>>();
      archive >> *new_task.ptr_;
      task_ptr = new_task.template Cast<chi::Task>();
      break;
    }
    default:
      task_ptr = hipc::FullPtr<chi::Task>::GetNull();
      break;
  }
  return task_ptr;
}

HSHM_GPU_FUN void GpuRuntime::LocalSaveTask(
    chi::u32 method, chi::LocalSaveTaskArchive &archive,
    const hipc::FullPtr<chi::Task> &task) {
  switch (method) {
    case Method::kPutBlob: {
      auto typed = task.template Cast<PutBlobTask>();
      archive << *typed.ptr_;
      break;
    }
    case Method::kGetBlob: {
      auto typed = task.template Cast<GetBlobTask>();
      archive << *typed.ptr_;
      break;
    }
    case Method::kGetOrCreateTag: {
      auto typed = task.template Cast<GetOrCreateTagTask<CreateParams>>();
      archive << *typed.ptr_;
      break;
    }
    default: break;
  }
}

HSHM_GPU_FUN void GpuRuntime::LocalLoadTaskOutput(
    chi::u32 method, chi::LocalLoadTaskArchive &archive,
    const hipc::FullPtr<chi::Task> &task) {
  switch (method) {
    case Method::kPutBlob: {
      auto typed = task.template Cast<PutBlobTask>();
      typed->SerializeOut(archive);
      break;
    }
    case Method::kGetBlob: {
      auto typed = task.template Cast<GetBlobTask>();
      typed->SerializeOut(archive);
      break;
    }
    case Method::kGetOrCreateTag: {
      auto typed = task.template Cast<GetOrCreateTagTask<CreateParams>>();
      typed->SerializeOut(archive);
      break;
    }
    case Method::kRegisterTarget: {
      auto typed = task.template Cast<RegisterTargetTask>();
      typed->SerializeOut(archive);
      break;
    }
    default: break;
  }
}

}  // namespace wrp_cte::core

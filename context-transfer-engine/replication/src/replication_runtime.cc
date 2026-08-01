/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#include <algorithm>
#include <string>
#include <vector>

#include <clio_cte/replication/replication_runtime.h>

namespace clio::cte::replication {

/**
 * Chunk size for primary → replica copies. Bounds the bounce buffer for
 * arbitrarily large blobs; each chunk is one GetBlob + one replica-targeted
 * PutBlob, and the CTE serializes each op under the blob's write token.
 */
static constexpr clio::run::u64 kReplicateChunkBytes = 4ULL * 1024 * 1024;

clio::run::TaskResume Runtime::Create(clio::run::shared_ptr<CreateTask> &task) {
  CLIO_TASK_BODY_BEGIN
  config_ = task->GetParams();
  if (!config_.next_pool_id_.IsNull()) {
    core_client_ =
        std::make_unique<clio::cte::core::Client>(config_.next_pool_id_);
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Destroy(clio::run::shared_ptr<DestroyTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Monitor(clio::run::shared_ptr<MonitorTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::cte::core::Client *Runtime::GetCoreClient() {
  if (!core_client_) {
    clio::run::PoolId core_id = !config_.next_pool_id_.IsNull()
                                    ? config_.next_pool_id_
                                    : clio::cte::core::kCtePoolId;
    core_client_ = std::make_unique<clio::cte::core::Client>(core_id);
  }
  return core_client_.get();
}

clio::run::TaskResume Runtime::ReplicateOne(
    const TagId &tag_id, const std::string &blob_name,
    int replica_idx, const Context &context,
    clio::run::u64 &bytes_copied, clio::run::u32 &rc) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  rc = 0;
  auto *cte = GetCoreClient();

  clio::run::u64 total = 0;
  {
    auto size_task = cte->AsyncGetBlobSize(tag_id, blob_name);
    CLIO_CO_AWAIT(size_task);
    if (size_task->GetReturnCode() != 0) {
      rc = 10 + size_task->GetReturnCode();
      CLIO_CO_RETURN;
    }
    total = size_task->size_;
  }
  if (total == 0) {
    // Empty primary: nothing to copy. Not an error — a FlushTag sweep may
    // legitimately see just-created blobs.
    CLIO_CO_RETURN;
  }

  for (clio::run::u64 off = 0; off < total; off += kReplicateChunkBytes) {
    clio::run::u64 len = std::min(kReplicateChunkBytes, total - off);
    auto buf = CLIO_IPC->AllocateBuffer(len);
    if (buf.IsNull()) {
      rc = 2;
      CLIO_CO_RETURN;
    }
    ctp::ipc::ShmPtr<> buf_ptr = buf.shm_.template Cast<void>();

    auto get_task = cte->AsyncGetBlob(tag_id, blob_name, off, len,
                                      /*flags=*/0, buf_ptr);
    CLIO_CO_AWAIT(get_task);
    if (get_task->GetReturnCode() != 0) {
      CLIO_IPC->FreeBuffer(buf);
      rc = 20 + get_task->GetReturnCode();
      CLIO_CO_RETURN;
    }

    // The one field this module owns on the way down: aim the put at the
    // replica. Everything else in the caller's context (persistence level,
    // target, preallocation) passes through to place the replica's blocks.
    Context put_ctx = context;
    put_ctx.replica_ = replica_idx;
    auto put_task = cte->AsyncPutBlob(tag_id, blob_name, off, len, buf_ptr,
                                      /*score=*/-1.0f, put_ctx);
    CLIO_CO_AWAIT(put_task);
    CLIO_IPC->FreeBuffer(buf);
    if (put_task->GetReturnCode() != 0) {
      rc = 30 + put_task->GetReturnCode();
      CLIO_CO_RETURN;
    }
    bytes_copied += len;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ReplicateBlob(
    clio::run::shared_ptr<ReplicateBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->bytes_copied_ = 0;
  if (task->replica_ <= 0 || task->blob_name_.size() == 0) {
    task->return_code_ = 1;
    CLIO_CO_RETURN;
  }
  {
    std::string blob_name = task->blob_name_.str();
    clio::run::u32 rc = 0;
    CLIO_CO_AWAIT(ReplicateOne(task->tag_id_, blob_name, task->replica_,
                          task->context_, task->bytes_copied_, rc));
    task->return_code_ = rc;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::FlushTag(clio::run::shared_ptr<FlushTagTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->blobs_replicated_ = 0;
  task->bytes_copied_ = 0;
  if (task->replica_ <= 0) {
    task->return_code_ = 1;
    CLIO_CO_RETURN;
  }
  {
    auto *cte = GetCoreClient();

    // Blob names are sharded across containers by HashBlobToContainer, so
    // the listing must be a Broadcast (GetContainedBlobsTask AggregateOut
    // merges the per-container name lists).
    std::vector<std::string> blob_names;
    {
      auto list_task = cte->AsyncGetContainedBlobs(
          task->tag_id_, clio::run::PoolQuery::Broadcast());
      CLIO_CO_AWAIT(list_task);
      if (list_task->GetReturnCode() != 0) {
        task->return_code_ = 40 + list_task->GetReturnCode();
        CLIO_CO_RETURN;
      }
      blob_names = list_task->blob_names_;
    }

    clio::run::u32 first_rc = 0;
    for (size_t i = 0; i < blob_names.size(); ++i) {
      if (task->min_score_ > 0.0f) {
        auto score_task = cte->AsyncGetBlobScore(task->tag_id_, blob_names[i]);
        CLIO_CO_AWAIT(score_task);
        if (score_task->GetReturnCode() != 0 ||
            score_task->score_ < task->min_score_) {
          continue;
        }
      }
      clio::run::u32 rc = 0;
      CLIO_CO_AWAIT(ReplicateOne(task->tag_id_, blob_names[i], task->replica_,
                            task->context_, task->bytes_copied_, rc));
      if (rc != 0) {
        // Keep sweeping — a durability flush should save what it can — but
        // report the first failure so the caller knows the sweep is partial.
        if (first_rc == 0) {
          first_rc = rc;
        }
        HLOG(kWarning, "FlushTag: failed to replicate blob '{}' (rc={})",
             blob_names[i], rc);
        continue;
      }
      task->blobs_replicated_++;
    }
    task->return_code_ = first_rc;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::CachedPut(
    clio::run::shared_ptr<CachedPutTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->replicas_written_ = 0;
  if (task->size_ == 0 || task->blob_data_.IsNull() ||
      task->blob_name_.size() == 0) {
    task->return_code_ = 1;
    CLIO_CO_RETURN;
  }
  {
    auto *cte = GetCoreClient();
    std::string blob_name = task->blob_name_.str();

    // Primary FIRST: the DRAM cache copy must never serve stale bytes, so
    // it is updated before the durable copies. If a later replica write
    // fails the caller sees the error while reads already see new data —
    // durability lagging is reported; a stale cache never is.
    // The cache copy is explicitly volatile-eligible (min persistence 0) and
    // high-scored so the DPE pins it to the fast tier; it carries NO
    // replica flags — being droppable by the organizer is its whole deal.
    {
      Context prim_ctx = task->context_;
      prim_ctx.replica_ = 0;
      prim_ctx.replica_flags_ = 0;
      prim_ctx.min_persistence_level_ = 0;
      float cache_score =
          (task->score_ >= 0.0f) ? task->score_ : config_.cache_score_;
      auto put = cte->AsyncPutBlob(task->tag_id_, blob_name, task->offset_,
                                   task->size_, task->blob_data_, cache_score,
                                   prim_ctx);
      CLIO_CO_AWAIT(put);
      if (put->GetReturnCode() != 0) {
        task->return_code_ = 20 + put->GetReturnCode();
        CLIO_CO_RETURN;
      }
    }

    // Write through to the fixed persistent set. Each copy is pinned
    // (REPLICA_FIXED) and volatile-banned (REPLICA_PERSISTENT); scores stay
    // per-replica (-1 keeps each copy's own).
    for (int i = 1; i <= config_.num_replicas_; ++i) {
      Context rep_ctx = task->context_;
      rep_ctx.replica_ = i;
      rep_ctx.replica_flags_ =
          task->context_.replica_flags_ | clio::cte::core::REPLICA_FIXED |
          clio::cte::core::REPLICA_PERSISTENT;
      if (rep_ctx.min_persistence_level_ < 1) {
        rep_ctx.min_persistence_level_ = 1;
      }
      auto put = cte->AsyncPutBlob(task->tag_id_, blob_name, task->offset_,
                                   task->size_, task->blob_data_,
                                   config_.replica_score_, rep_ctx);
      CLIO_CO_AWAIT(put);
      if (put->GetReturnCode() != 0) {
        task->return_code_ = 30 + put->GetReturnCode();
        CLIO_CO_RETURN;
      }
      task->replicas_written_++;
    }
    task->return_code_ = 0;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::RecachePrimary(const TagId &tag_id,
                                              const std::string &blob_name,
                                              int replica_idx,
                                              clio::run::u64 rep_size,
                                              clio::run::u64 &recached) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  auto *cte = GetCoreClient();
  for (clio::run::u64 off = 0; off < rep_size; off += kReplicateChunkBytes) {
    clio::run::u64 len = std::min(kReplicateChunkBytes, rep_size - off);
    auto buf = CLIO_IPC->AllocateBuffer(len);
    if (buf.IsNull()) {
      CLIO_CO_RETURN;  // best-effort: keep the valid prefix
    }
    ctp::ipc::ShmPtr<> buf_ptr = buf.shm_.template Cast<void>();

    Context get_ctx;
    get_ctx.replica_ = replica_idx;
    auto get_task = cte->AsyncGetBlob(tag_id, blob_name, off, len,
                                      /*flags=*/0, buf_ptr,
                                      clio::run::PoolQuery::Dynamic(),
                                      get_ctx);
    CLIO_CO_AWAIT(get_task);
    if (get_task->GetReturnCode() != 0) {
      CLIO_IPC->FreeBuffer(buf);
      CLIO_CO_RETURN;
    }

    Context put_ctx;
    put_ctx.replica_ = 0;
    put_ctx.min_persistence_level_ = 0;
    auto put_task = cte->AsyncPutBlob(tag_id, blob_name, off, len, buf_ptr,
                                      config_.cache_score_, put_ctx);
    CLIO_CO_AWAIT(put_task);
    CLIO_IPC->FreeBuffer(buf);
    if (put_task->GetReturnCode() != 0) {
      // No room in the fast tiers (or any tier): stop. The sequential-from-0
      // order means everything already copied is a valid prefix.
      CLIO_CO_RETURN;
    }
    recached += len;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::CachedGet(
    clio::run::shared_ptr<CachedGetTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->from_replica_ = 0;
  task->recached_bytes_ = 0;
  if (task->size_ == 0 || task->blob_data_.IsNull() ||
      task->blob_name_.size() == 0) {
    task->return_code_ = 1;
    CLIO_CO_RETURN;
  }
  {
    auto *cte = GetCoreClient();
    std::string blob_name = task->blob_name_.str();
    const clio::run::u64 end = task->offset_ + task->size_;

    // Cache hit iff the primary COVERS the requested range. A dropped
    // primary reads as size 0; an interrupted re-cache leaves a prefix that
    // still hits for ranges inside it.
    clio::run::u64 primary_size = 0;
    {
      auto size_task = cte->AsyncGetBlobSize(task->tag_id_, blob_name);
      CLIO_CO_AWAIT(size_task);
      if (size_task->GetReturnCode() != 0) {
        task->return_code_ = 2;  // Blob not found at all
        CLIO_CO_RETURN;
      }
      primary_size = size_task->size_;
    }
    if (primary_size >= end) {
      Context get_ctx = task->context_;
      get_ctx.replica_ = 0;
      auto get_task = cte->AsyncGetBlob(task->tag_id_, blob_name,
                                        task->offset_, task->size_,
                                        /*flags=*/0, task->blob_data_,
                                        clio::run::PoolQuery::Dynamic(),
                                        get_ctx);
      CLIO_CO_AWAIT(get_task);
      task->return_code_ = get_task->GetReturnCode();
      CLIO_CO_RETURN;
    }

    // Miss: serve from the first persistent replica that covers the range,
    // then restore the DRAM fast path by copying the WHOLE replica back
    // into the primary (best-effort).
    for (int r = 1; r <= config_.num_replicas_; ++r) {
      clio::run::u64 rep_size = 0;
      {
        auto size_task = cte->AsyncGetBlobSize(
            task->tag_id_, blob_name, clio::run::PoolQuery::Dynamic(), r);
        CLIO_CO_AWAIT(size_task);
        if (size_task->GetReturnCode() != 0) {
          continue;
        }
        rep_size = size_task->size_;
      }
      if (rep_size < end) {
        continue;
      }
      Context get_ctx = task->context_;
      get_ctx.replica_ = r;
      auto get_task = cte->AsyncGetBlob(task->tag_id_, blob_name,
                                        task->offset_, task->size_,
                                        /*flags=*/0, task->blob_data_,
                                        clio::run::PoolQuery::Dynamic(),
                                        get_ctx);
      CLIO_CO_AWAIT(get_task);
      if (get_task->GetReturnCode() != 0) {
        continue;
      }
      task->from_replica_ = r;
      CLIO_CO_AWAIT(RecachePrimary(task->tag_id_, blob_name, r, rep_size,
                              task->recached_bytes_));
      task->return_code_ = 0;
      CLIO_CO_RETURN;
    }
    task->return_code_ = 3;  // No copy covers the requested range
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::replication

// Define ChiMod entry points (alloc/new/name/destroy) so the runtime's module
// manager can dlopen and instantiate this chimod.
CLIO_TASK_CC(clio::cte::replication::Runtime)

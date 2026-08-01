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
    size_task.Wait();
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
    get_task.Wait();
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
    put_task.Wait();
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
      list_task.Wait();
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
        score_task.Wait();
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

}  // namespace clio::cte::replication

// Define ChiMod entry points (alloc/new/name/destroy) so the runtime's module
// manager can dlopen and instantiate this chimod.
CLIO_TASK_CC(clio::cte::replication::Runtime)

/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <clio_cte/cache/cache_runtime.h>

namespace clio::cte::cache {

/** Chunk size for cache <-> chain copies (flush and re-populate). */
static constexpr clio::run::u64 kCacheChunkBytes = 4ULL * 1024 * 1024;

static std::string DirtyKey(const clio::cte::core::TagId &tag_id,
                            const std::string &blob_name) {
  return std::to_string(tag_id.major_) + "." + std::to_string(tag_id.minor_) +
         "." + blob_name;
}

clio::run::TaskResume Runtime::Create(clio::run::shared_ptr<CreateTask> &task) {
  CLIO_TASK_BODY_BEGIN
  config_ = task->GetParams();
  interposer_next_pool_ = config_.next_pool_id_;  // base forwarding target
  if (!config_.next_pool_id_.IsNull()) {
    next_client_ =
        std::make_unique<clio::cte::core::Client>(config_.next_pool_id_);
  }
  // Write-back sweep: fire-and-forget periodic, like the replication
  // chimod's. SetPeriod alone does NOT mark a task periodic (learned the
  // hard way there) — TASK_PERIODIC must be set too.
  if (config_.flush_period_ms_ > 0) {
    auto *ipc = CLIO_CPU_IPC;
    auto sweep = ipc->NewTask<FlushSweepTask>(
        clio::run::CreateTaskId(), pool_id_, clio::run::PoolQuery::Local());
    sweep->SetPeriod(static_cast<double>(config_.flush_period_ms_),
                     clio::run::kMilli);
    sweep->SetFlags(TASK_PERIODIC);
    ipc->Send(sweep);
    HLOG(kInfo, "cache: write-back flush sweep every {} ms",
         config_.flush_period_ms_);
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

clio::cte::core::Client *Runtime::GetNextClient() {
  if (!next_client_) {
    next_client_ = std::make_unique<clio::cte::core::Client>(CorePoolId());
  }
  return next_client_.get();
}

void Runtime::EnqueueFlush(const TagId &tag_id, const std::string &blob_name,
                           const Context &context, float score) {
  DirtyEntry e{tag_id, blob_name, context, score};
  std::lock_guard<std::mutex> lk(dirty_mtx_);
  dirty_[DirtyKey(tag_id, blob_name)] = std::move(e);
}

bool Runtime::IsDirty(const TagId &tag_id, const std::string &blob_name) {
  std::lock_guard<std::mutex> lk(dirty_mtx_);
  return dirty_.find(DirtyKey(tag_id, blob_name)) != dirty_.end();
}

clio::run::TaskResume Runtime::FlushOne(const DirtyEntry &entry,
                                        clio::run::u32 &rc) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  rc = 0;
  auto *next = GetNextClient();

  // The cache replica's raw size (== logical size).
  clio::run::u64 total = 0;
  {
    auto sz = next->AsyncGetBlobSize(entry.tag_id_, entry.blob_name_,
                                     clio::run::PoolQuery::Local(),
                                     clio::cte::core::kCacheReplica);
    CLIO_CO_AWAIT(sz);
    if (sz->GetReturnCode() != 0 || sz->size_ == 0) {
      rc = 1;  // blob or cache copy gone — nothing to flush; drop the entry
      CLIO_CO_RETURN;
    }
    total = sz->size_;
  }

  for (clio::run::u64 off = 0; off < total; off += kCacheChunkBytes) {
    clio::run::u64 len = std::min(kCacheChunkBytes, total - off);
    auto buf = CLIO_IPC->AllocateBuffer(len);
    if (buf.IsNull()) {
      rc = 2;
      CLIO_CO_RETURN;
    }
    ctp::ipc::ShmPtr<> buf_ptr = buf.shm_.template Cast<void>();

    Context get_ctx;
    get_ctx.replica_ = clio::cte::core::kCacheReplica;
    auto get = next->AsyncGetBlob(entry.tag_id_, entry.blob_name_, off, len,
                                  /*flags=*/0, buf_ptr,
                                  clio::run::PoolQuery::Local(), get_ctx);
    CLIO_CO_AWAIT(get);
    if (get->GetReturnCode() != 0) {
      CLIO_IPC->FreeBuffer(buf);
      rc = 20 + get->GetReturnCode();
      CLIO_CO_RETURN;
    }

    // Down the chain with the put's remembered context: the compressor
    // transforms, replication replicates, the core stores.
    Context put_ctx = entry.context_;
    put_ctx.replica_ = 0;
    auto put = next->AsyncPutBlob(entry.tag_id_, entry.blob_name_, off, len,
                                  buf_ptr, entry.score_, put_ctx,
                                  /*flags=*/0, clio::run::PoolQuery::Local());
    CLIO_CO_AWAIT(put);
    CLIO_IPC->FreeBuffer(buf);
    if (put->GetReturnCode() != 0) {
      rc = 30 + put->GetReturnCode();
      CLIO_CO_RETURN;
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::FlushSweep(
    clio::run::shared_ptr<FlushSweepTask> &task) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  task->blobs_flushed_ = 0;
  {
    // Swap the dirty set out under the lock; flush outside it. A put racing
    // the sweep re-inserts its key and is caught next period.
    std::unordered_map<std::string, DirtyEntry> batch;
    {
      std::lock_guard<std::mutex> lk(dirty_mtx_);
      batch.swap(dirty_);
    }
    for (auto it = batch.begin(); it != batch.end(); ++it) {
      clio::run::u32 rc = 0;
      CLIO_CO_AWAIT(FlushOne(it->second, rc));
      if (rc == 0 || rc == 1) {
        // Flushed, or the blob/cache copy is gone — either way, done.
        if (rc == 0) {
          task->blobs_flushed_++;
        }
      } else {
        // Keep dirty for the next period (best-effort, periodic).
        std::lock_guard<std::mutex> lk(dirty_mtx_);
        dirty_.emplace(it->first, it->second);
      }
    }
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::PutBlob(
    clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Explicit replica addressing passes through untouched.
  if (task->context_.replica_ != 0) {
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPutBlob,
                           task.template Cast<clio::run::Task>()));
    CLIO_CO_RETURN;
  }
  {
    // 1. Cache write FIRST: the same task, re-aimed at THE cache replica —
    //    raw bytes (transform cleared), REPLICA_CACHE flag, the configured
    //    score floor. The interposers below forward explicit replica
    //    addressing verbatim, so this lands on the core untransformed, and
    //    the mirror publishes it for the zero-IPC fast path. Vectored
    //    segments ride along unchanged.
    const Context orig_ctx = task->context_;
    task->context_.replica_ = clio::cte::core::kCacheReplica;
    task->context_.replica_flags_ =
        orig_ctx.replica_flags_ | clio::cte::core::REPLICA_CACHE;
    task->context_.replica_min_score_ = config_.min_score_;
    task->context_.transform_flags_ = 0;
    task->context_.min_persistence_level_ = 0;
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPutBlob,
                           task.template Cast<clio::run::Task>()));
    const clio::run::u32 cache_rc = task->GetReturnCode();
    task->context_ = orig_ctx;
    if (cache_rc != 0) {
      // Cache write failed (e.g. fast tier full and eviction exhausted):
      // fall back to a plain synchronous pass-down so the data is never
      // lost to a caching problem.
      HLOG(kWarning, "cache: cache-replica write failed rc={}, pass-through",
           cache_rc);
      CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPutBlob,
                             task.template Cast<clio::run::Task>()));
      CLIO_CO_RETURN;
    }
    if (config_.flush_period_ms_ > 0) {
      // 2a. WRITE-BACK: ack now; the sweep pushes the raw bytes down with
      //     this put's context. Overwrites coalesce (latest context wins).
      EnqueueFlush(task->tag_id_, task->blob_name_.str(), orig_ctx,
                   task->score_);
      task->return_code_ = 0;
      CLIO_CO_RETURN;
    }
    // 2b. WRITE-THROUGH (flush_period_ms == 0): push down before acking.
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPutBlob,
                           task.template Cast<clio::run::Task>()));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::GetBlob(
    clio::run::shared_ptr<clio::cte::core::GetBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Explicit replica addressing passes through untouched.
  if (task->context_.replica_ != 0) {
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kGetBlob,
                           task.template Cast<clio::run::Task>()));
    CLIO_CO_RETURN;
  }
  {
    auto *next = GetNextClient();
    const std::string blob_name = task->blob_name_.str();
    clio::run::u64 req_lo = 0, req_hi = 0;
    clio::cte::core::BlobRequestRange(*task, &req_lo, &req_hi);

    // 1. Cache replica first — fresh by construction (writes land there
    //    before the ack) and RAW, so a hit needs no transform undo.
    clio::run::u64 cache_size = 0;
    {
      auto sz = next->AsyncGetBlobSize(task->tag_id_, blob_name,
                                       clio::run::PoolQuery::Local(),
                                       clio::cte::core::kCacheReplica);
      CLIO_CO_AWAIT(sz);
      if (sz->GetReturnCode() == 0) {
        cache_size = sz->size_;
      }
    }
    if (cache_size >= req_hi && req_hi > 0) {
      task->context_.replica_ = clio::cte::core::kCacheReplica;
      CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kGetBlob,
                             task.template Cast<clio::run::Task>()));
      task->context_.replica_ = 0;
      if (task->GetReturnCode() == 0) {
        CLIO_CO_RETURN;
      }
    }

    // 2. Miss: the authoritative chain serves (the compressor below undoes
    //    any transform).
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kGetBlob,
                           task.template Cast<clio::run::Task>()));
    if (task->GetReturnCode() != 0) {
      CLIO_CO_RETURN;
    }

    // 3. Best-effort re-population — but never while dirty: a dirty blob's
    //    cache is NEWER than the chain, and re-populating from downstream
    //    would roll fresh bytes back.
    if (!IsDirty(task->tag_id_, blob_name)) {
      CLIO_CO_AWAIT(Repopulate(task->tag_id_, blob_name));
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Repopulate(const TagId &tag_id,
                                          const std::string &blob_name) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  auto *next = GetNextClient();
  clio::run::u64 total = 0;
  {
    // Logical size from the chain (the compressor reports original sizes).
    auto sz = next->AsyncGetBlobSize(tag_id, blob_name,
                                     clio::run::PoolQuery::Local());
    CLIO_CO_AWAIT(sz);
    if (sz->GetReturnCode() != 0 || sz->size_ == 0) {
      CLIO_CO_RETURN;
    }
    total = sz->size_;
  }
  for (clio::run::u64 off = 0; off < total; off += kCacheChunkBytes) {
    clio::run::u64 len = std::min(kCacheChunkBytes, total - off);
    auto buf = CLIO_IPC->AllocateBuffer(len);
    if (buf.IsNull()) {
      CLIO_CO_RETURN;  // best-effort: keep the valid prefix
    }
    ctp::ipc::ShmPtr<> buf_ptr = buf.shm_.template Cast<void>();
    auto get = next->AsyncGetBlob(tag_id, blob_name, off, len, /*flags=*/0,
                                  buf_ptr, clio::run::PoolQuery::Local());
    CLIO_CO_AWAIT(get);
    if (get->GetReturnCode() != 0) {
      CLIO_IPC->FreeBuffer(buf);
      CLIO_CO_RETURN;
    }
    Context put_ctx;
    put_ctx.replica_ = clio::cte::core::kCacheReplica;
    put_ctx.replica_flags_ = clio::cte::core::REPLICA_CACHE;
    put_ctx.replica_min_score_ = config_.min_score_;
    put_ctx.transform_flags_ = 0;
    put_ctx.min_persistence_level_ = 0;
    auto put = next->AsyncPutBlob(tag_id, blob_name, off, len, buf_ptr,
                                  /*score=*/-1.0f, put_ctx, /*flags=*/0,
                                  clio::run::PoolQuery::Local());
    CLIO_CO_AWAIT(put);
    CLIO_IPC->FreeBuffer(buf);
    if (put->GetReturnCode() != 0) {
      CLIO_CO_RETURN;  // abandon; prefix stays valid
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::GetBlobSize(
    clio::run::shared_ptr<clio::cte::core::GetBlobSizeTask> &task) {
  CLIO_TASK_BODY_BEGIN
  if (task->replica_ != 0) {
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kGetBlobSize,
                           task.template Cast<clio::run::Task>()));
    CLIO_CO_RETURN;
  }
  {
    // Cache replica first: its raw size IS the logical size, and for a
    // dirty blob the chain's answer is stale or absent entirely.
    auto *next = GetNextClient();
    auto sz = next->AsyncGetBlobSize(task->tag_id_, task->blob_name_.str(),
                                     clio::run::PoolQuery::Local(),
                                     clio::cte::core::kCacheReplica);
    CLIO_CO_AWAIT(sz);
    if (sz->GetReturnCode() == 0 && sz->size_ > 0) {
      task->size_ = sz->size_;
      task->return_code_ = 0;
      CLIO_CO_RETURN;
    }
  }
  CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kGetBlobSize,
                         task.template Cast<clio::run::Task>()));
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::MultiPutBlob(
    clio::run::shared_ptr<clio::cte::core::MultiPutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Batches forward down verbatim (write-through): batch records carry no
  // per-record Context to remember for a write-back flush, and the chain
  // below already executes them with the core's exact semantics.
  CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kMultiPutBlob,
                         task.template Cast<clio::run::Task>()));
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::cache

// ChiMod entry points (alloc/new/name/destroy) for the module manager.
CLIO_TASK_CC(clio::cte::cache::Runtime)

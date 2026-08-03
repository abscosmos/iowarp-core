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

/** Write-back flush re-batching (issue #886 follow-up): small dirty blobs
 *  ship down the chain as MultiPutBlob batches — one task per batch instead
 *  of a get+put pair per blob — mirroring the client defer pipeline's
 *  constants (64 records / 128KB chunk; anything bigger flushes scalar,
 *  the same large-value rule the client applies). */
static constexpr size_t kFlushBatchMax = 64;
static constexpr clio::run::u64 kFlushBatchBytes = 128ULL * 1024;

/** True when a remembered put context carries no semantics beyond what a
 *  batch-wide context can express for a MERGED batch — i.e. every field
 *  that steers the downstream chain is at its default. Entries with richer
 *  contexts flush scalar; correctness first, amortization second. */
static bool BatchableContext(const clio::cte::core::Context &ctx) {
  const clio::cte::core::Context def;
  return ctx.replica_ == 0 && ctx.replica_flags_ == 0 &&
         ctx.replica_min_score_ < 0.0f && ctx.transform_flags_ == 0 &&
         ctx.dynamic_compress_ == def.dynamic_compress_ &&
         ctx.compress_lib_ == def.compress_lib_ &&
         ctx.min_persistence_level_ == def.min_persistence_level_ &&
         ctx.persistence_target_ == def.persistence_target_ &&
         ctx.preallocate_ == def.preallocate_ && !ctx.emulate_;
}

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
    auto *next = GetNextClient();

    // Pass 1: size every entry (cache-replica raw size == bytes to flush)
    // and split scalar-vs-batchable. Blobs whose cache copy vanished are
    // done; rich contexts and large blobs flush scalar; the rest re-batch.
    std::vector<DirtyEntry> scalars;
    std::vector<std::pair<DirtyEntry, clio::run::u64>> small;
    for (auto it = batch.begin(); it != batch.end(); ++it) {
      auto sz = next->AsyncGetBlobSize(it->second.tag_id_,
                                       it->second.blob_name_,
                                       clio::run::PoolQuery::Local(),
                                       clio::cte::core::kCacheReplica);
      CLIO_CO_AWAIT(sz);
      if (sz->GetReturnCode() != 0 || sz->size_ == 0) {
        continue;  // blob or cache copy gone — nothing to flush
      }
      if (!BatchableContext(it->second.context_) ||
          sz->size_ > kFlushBatchBytes) {
        scalars.push_back(it->second);
      } else {
        small.emplace_back(it->second, sz->size_);
      }
    }

    for (size_t i = 0; i < scalars.size(); ++i) {
      clio::run::u32 rc = 0;
      CLIO_CO_AWAIT(FlushOne(scalars[i], rc));
      if (rc == 0) {
        task->blobs_flushed_++;
      } else if (rc != 1) {
        // Keep dirty for the next period (best-effort, periodic). emplace
        // never overwrites — a NEWER re-dirtied entry wins.
        std::lock_guard<std::mutex> lk(dirty_mtx_);
        dirty_.emplace(DirtyKey(scalars[i].tag_id_, scalars[i].blob_name_),
                       scalars[i]);
      }
    }

    // Pass 2: pack small entries into MultiPutBlob batches — read each raw
    // copy straight into the staging chunk, ship one task per chunk. The
    // chunk's ownership transfers to the task (TASK_DATA_OWNER inside
    // AsyncMultiPutVectored), so each shipped batch gets a fresh chunk.
    size_t gi = 0;
    while (gi < small.size()) {
      auto chunk = CLIO_IPC->AllocateBuffer(kFlushBatchBytes);
      if (chunk.IsNull()) {
        // No staging memory: fall back to scalar for the remainder.
        for (; gi < small.size(); ++gi) {
          clio::run::u32 rc = 0;
          CLIO_CO_AWAIT(FlushOne(small[gi].first, rc));
          if (rc == 0) {
            task->blobs_flushed_++;
          } else if (rc != 1) {
            std::lock_guard<std::mutex> lk(dirty_mtx_);
            dirty_.emplace(DirtyKey(small[gi].first.tag_id_,
                                    small[gi].first.blob_name_),
                           small[gi].first);
          }
        }
        break;
      }
      std::vector<clio::cte::core::MultiPutDesc> descs;
      std::vector<size_t> members;  // indices into `small` in this batch
      clio::run::u64 used = 0;
      while (gi < small.size() && descs.size() < kFlushBatchMax &&
             used + small[gi].second <= kFlushBatchBytes) {
        const DirtyEntry &e = small[gi].first;
        const clio::run::u64 len = small[gi].second;
        Context get_ctx;
        get_ctx.replica_ = clio::cte::core::kCacheReplica;
        auto get = next->AsyncGetBlob(
            e.tag_id_, e.blob_name_, 0, len, /*flags=*/0,
            ctp::ipc::ShmPtr<>::FromRaw(chunk.ptr_ + used),
            clio::run::PoolQuery::Local(), get_ctx);
        CLIO_CO_AWAIT(get);
        if (get->GetReturnCode() != 0) {
          // Raced away between sizing and reading — requeue, next period
          // re-sizes it.
          std::lock_guard<std::mutex> lk(dirty_mtx_);
          dirty_.emplace(DirtyKey(e.tag_id_, e.blob_name_), e);
          ++gi;
          continue;
        }
        clio::cte::core::MultiPutDesc d;
        d.tag_id_ = e.tag_id_;
        d.blob_name_ = e.blob_name_;
        d.offset_ = 0;
        d.size_ = len;
        d.payload_off_ = used;
        descs.push_back(std::move(d));
        members.push_back(gi);
        used += len;
        ++gi;
      }
      if (descs.empty()) {
        CLIO_IPC->FreeBuffer(chunk);
        continue;
      }
      Context put_ctx = small[members.front()].first.context_;
      put_ctx.replica_ = 0;
      auto ship = next->AsyncMultiPutVectored(
          chunk.shm_.template Cast<void>(), used, descs,
          clio::run::PoolQuery::Local(), put_ctx);
      CLIO_CO_AWAIT(ship);
      if (ship->GetReturnCode() == 0 && ship->first_rc_ == 0) {
        task->blobs_flushed_ += descs.size();
      } else {
        for (size_t m = 0; m < members.size(); ++m) {
          const DirtyEntry &e = small[members[m]].first;
          std::lock_guard<std::mutex> lk(dirty_mtx_);
          dirty_.emplace(DirtyKey(e.tag_id_, e.blob_name_), e);
        }
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
  // Batches get SCALAR-EQUIVALENT semantics (issue #886 follow-up): the
  // batch context applies to every record, so the whole batch is first
  // re-aimed at the cache replica slots — one forwarded task writes every
  // record's raw copy — then acked and marked dirty for the write-back
  // sweep (which re-batches on the way down). Batching is never "passed
  // on": a batched put and a scalar put leave identical state behind.
  if (task->context_.replica_ != 0) {
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kMultiPutBlob,
                           task.template Cast<clio::run::Task>()));
    CLIO_CO_RETURN;
  }
  {
    const Context orig_ctx = task->context_;
    task->context_.replica_ = clio::cte::core::kCacheReplica;
    task->context_.replica_flags_ =
        orig_ctx.replica_flags_ | clio::cte::core::REPLICA_CACHE;
    task->context_.replica_min_score_ = config_.min_score_;
    task->context_.transform_flags_ = 0;
    task->context_.min_persistence_level_ = 0;
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kMultiPutBlob,
                           task.template Cast<clio::run::Task>()));
    const clio::run::u32 cache_rc = task->GetReturnCode();
    task->context_ = orig_ctx;
    if (cache_rc != 0) {
      HLOG(kWarning, "cache: batch cache-replica write failed rc={}, "
           "pass-through", cache_rc);
      task->num_ok_ = 0;
      task->first_rc_ = 0;
      CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kMultiPutBlob,
                             task.template Cast<clio::run::Task>()));
      CLIO_CO_RETURN;
    }
    if (config_.flush_period_ms_ > 0) {
      // Write-back: every record dirty under the batch's context; overwrites
      // coalesce per blob (latest wins), exactly like the scalar path.
      const std::vector<clio::cte::core::MultiPutDesc> descs =
          clio::cte::core::DecodeMultiPutDescs(task->descs_);
      for (size_t i = 0; i < descs.size(); ++i) {
        EnqueueFlush(descs[i].tag_id_, descs[i].blob_name_, orig_ctx,
                     /*score=*/-1.0f);
      }
      task->return_code_ = 0;
      CLIO_CO_RETURN;
    }
    // Write-through: push the original batch down before acking.
    task->num_ok_ = 0;
    task->first_rc_ = 0;
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kMultiPutBlob,
                           task.template Cast<clio::run::Task>()));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::cache

// ChiMod entry points (alloc/new/name/destroy) for the module manager.
CLIO_TASK_CC(clio::cte::cache::Runtime)

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

/** Chunk size for chain -> cache re-population copies. */
static constexpr clio::run::u64 kCacheChunkBytes = 4ULL * 1024 * 1024;

clio::run::TaskResume Runtime::Create(clio::run::shared_ptr<CreateTask> &task) {
  CLIO_TASK_BODY_BEGIN
  config_ = task->GetParams();
  interposer_next_pool_ = config_.next_pool_id_;  // base forwarding target
  if (!config_.next_pool_id_.IsNull()) {
    next_client_ =
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

clio::cte::core::Client *Runtime::GetNextClient() {
  if (!next_client_) {
    next_client_ = std::make_unique<clio::cte::core::Client>(CorePoolId());
  }
  return next_client_.get();
}

/** Mutate a put context to aim at THE cache replica: raw bytes, the
 *  configured score floor, cache-slot addressing. */
void Runtime::AimAtCacheReplica(Context *ctx) const {
  ctx->replica_ = clio::cte::core::kCacheReplica;
  ctx->replica_flags_ |= clio::cte::core::REPLICA_CACHE;
  ctx->replica_min_score_ = config_.min_score_;
  ctx->transform_flags_ = 0;
  ctx->min_persistence_level_ = 0;
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
    // ASYNCHRONOUS WRITE-THROUGH (same discipline as the replication
    // chimod): the AUTHORITATIVE write goes down the chain FIRST and must
    // succeed before the ack — after every ack the source of truth is
    // current, and a crash loses nothing that was acked. The layers below
    // keep their own async machinery (replication's durable copies are
    // sweep-driven); this layer defers nothing.
    const Context orig_ctx = task->context_;
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPutBlob,
                           task.template Cast<clio::run::Task>()));
    if (task->GetReturnCode() != 0) {
      CLIO_CO_RETURN;  // authoritative write failed — nothing was cached
    }
    const Context out_ctx = task->context_;  // chain's OUT fields (compress
                                             // telemetry etc.) survive below

    // Then the raw node-local copy: the SAME task re-aimed at the cache
    // replica slot (the interposers below forward explicit replica
    // addressing verbatim), which also publishes the serving-replica mirror
    // record for the zero-IPC fast path. Best-effort: the authoritative
    // write already landed, so a cache failure (e.g. fast tier full)
    // degrades reads, never the put.
    task->context_ = orig_ctx;
    AimAtCacheReplica(&task->context_);
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPutBlob,
                           task.template Cast<clio::run::Task>()));
    if (task->GetReturnCode() != 0) {
      HLOG(kWarning, "cache: cache-replica write failed rc={} (put ok)",
           task->GetReturnCode());
    }
    task->context_ = out_ctx;
    task->return_code_ = 0;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::MultiPutBlob(
    clio::run::shared_ptr<clio::cte::core::MultiPutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Batches get the same asynchronous write-through as scalar puts: the
  // authoritative batch lands below first, then ONE re-aimed batch writes
  // every record's raw cache copy (scalar-equivalent semantics; batching is
  // never passed on).
  if (task->context_.replica_ != 0) {
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kMultiPutBlob,
                           task.template Cast<clio::run::Task>()));
    CLIO_CO_RETURN;
  }
  {
    const Context orig_ctx = task->context_;
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kMultiPutBlob,
                           task.template Cast<clio::run::Task>()));
    if (task->GetReturnCode() != 0) {
      CLIO_CO_RETURN;
    }
    const clio::run::u32 auth_ok = task->num_ok_;
    const int auth_rc = task->first_rc_;

    task->context_ = orig_ctx;
    AimAtCacheReplica(&task->context_);
    task->num_ok_ = 0;
    task->first_rc_ = 0;
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kMultiPutBlob,
                           task.template Cast<clio::run::Task>()));
    if (task->GetReturnCode() != 0 || task->first_rc_ != 0) {
      HLOG(kWarning, "cache: batch cache-replica write failed rc={}/{} "
           "(batch ok)", task->GetReturnCode(), task->first_rc_);
    }
    // Report the AUTHORITATIVE batch's outcome.
    task->context_ = orig_ctx;
    task->num_ok_ = auth_ok;
    task->first_rc_ = auth_rc;
    task->SetReturnCode(auth_rc == 0 ? 0 : auth_rc);
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

    // 1. Cache replica first — never stale (write-through updates it before
    //    the ack) and RAW, so a hit needs no transform undo.
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

    // 3. Best-effort re-population, so the next read of this blob (and the
    //    zero-IPC fast path) hits the raw local copy.
    CLIO_CO_AWAIT(Repopulate(task->tag_id_, blob_name));
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
    AimAtCacheReplica(&put_ctx);
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
    // Cache replica first: its raw size IS the logical size (no transform),
    // and write-through keeps it current — one local answer instead of a
    // chain traversal + header probe.
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

}  // namespace clio::cte::cache

// ChiMod entry points (alloc/new/name/destroy) for the module manager.
CLIO_TASK_CC(clio::cte::cache::Runtime)

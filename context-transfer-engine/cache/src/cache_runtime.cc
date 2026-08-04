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

clio::cte::core::Client *Runtime::GetLocalClient() {
  if (!local_client_) {
    local_client_ =
        std::make_unique<clio::cte::core::Client>(clio::cte::core::kCtePoolId);
  }
  return local_client_.get();
}

clio::run::PoolQuery Runtime::ScheduleTask(
    const clio::run::shared_ptr<clio::run::Task> &task) {
  // Reads AND writes route SUBMITTER-LOCAL (issue #886 locality): the
  // local handler serves/updates the node-local raw copy and reaches the
  // blob's owner through the chain client — so the hot path (a rank
  // touching its own data) is all PoolQuery::Local, and only the
  // authoritative hop crosses nodes. Explicit-replica ops keep owner
  // routing — they address a concrete copy that lives with the blob.
  switch (task->method_) {
    case clio::cte::core::Method::kGetBlob: {
      auto typed = task.template Cast<clio::cte::core::GetBlobTask>();
      if (typed->context_.replica_ == 0) {
        return clio::run::PoolQuery::Local();
      }
      break;
    }
    case clio::cte::core::Method::kGetBlobSize: {
      auto typed = task.template Cast<clio::cte::core::GetBlobSizeTask>();
      if (typed->replica_ == 0) {
        return clio::run::PoolQuery::Local();
      }
      break;
    }
    case clio::cte::core::Method::kPutBlob: {
      auto typed = task.template Cast<clio::cte::core::PutBlobTask>();
      if (typed->context_.replica_ == 0) {
        return clio::run::PoolQuery::Local();
      }
      break;
    }
    default:
      break;
  }
  return CoreInterposer::ScheduleTask(task);
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
  // Explicit replica addressing passes through untouched (owner-routed).
  if (task->context_.replica_ != 0) {
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPutBlob,
                           task.template Cast<clio::run::Task>()));
    CLIO_CO_RETURN;
  }
  {
    // WRITER-LOCAL asynchronous write-through (issue #886 locality). This
    // handler runs on the SUBMITTER's node. The node-local raw copy is
    // written FIRST, then the authoritative put goes to the blob's owner
    // chain carrying origin_node_ — the owner invalidates every OTHER
    // registered copy and registers ours atomically under the write token,
    // so the fresh local copy is coherent by construction. Ordering is the
    // safety: the local write strictly precedes the registration, so any
    // LATER foreign put's invalidation always catches it.
    //
    // The local copy is CREATED only when this put demonstrably covers the
    // whole blob (single region at offset 0 AND the authoritative blob does
    // not exist yet) — a partial write must never mint a prefix that
    // pretends to be complete. An EXISTING local copy mirrors every put
    // in place, staying byte- and size-identical to the authoritative blob
    // for all writes issued through this node.
    auto *local = GetLocalClient();
    auto *next = GetNextClient();
    const std::string blob_name = task->blob_name_.str();
    const Context orig_ctx = task->context_;

    bool have_local = false;
    {
      auto sz = local->AsyncGetBlobSize(task->tag_id_, blob_name,
                                        clio::run::PoolQuery::Local(),
                                        clio::cte::core::kCacheReplica);
      CLIO_CO_AWAIT(sz);
      have_local = (sz->GetReturnCode() == 0 && sz->size_ > 0);
    }
    bool create_local = false;
    if (!have_local && task->segments_.empty() && task->offset_ == 0) {
      // Whole-blob-covering iff the authoritative blob does not exist yet.
      auto sz = next->AsyncGetBlobSize(task->tag_id_, blob_name);
      CLIO_CO_AWAIT(sz);
      create_local = !(sz->GetReturnCode() == 0 && sz->size_ > 0);
    }

    bool wrote_local = false;
    if (have_local || create_local) {
      task->context_ = orig_ctx;
      AimAtCacheReplica(&task->context_);
      CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPutBlob,
                             task.template Cast<clio::run::Task>()));
      wrote_local = (task->GetReturnCode() == 0);
      if (!wrote_local) {
        HLOG(kWarning, "cache: local raw-copy write failed rc={}",
             task->GetReturnCode());
      }
    }

    // Authoritative put(s) to the owner chain (Dynamic hash-routes there).
    // Vectored puts replay their regions in list order (the core's
    // last-writer-wins rule, same as the replication sync path).
    {
      Context auth_ctx = orig_ctx;
      if (wrote_local) {
        auth_ctx.origin_node_ = CLIO_IPC->GetNodeId();
      }
      std::vector<clio::cte::core::BlobRegion> regions;
      clio::cte::core::ForEachBlobRegion(*task,
          [&regions](const clio::cte::core::BlobRegion &r) {
            regions.push_back(r);
            return true;
          });
      clio::run::u32 rc = 0;
      Context out_ctx = auth_ctx;
      for (size_t i = 0; i < regions.size() && rc == 0; ++i) {
        auto put = next->AsyncPutBlob(task->tag_id_, blob_name,
                                      regions[i].blob_off_, regions[i].size_,
                                      regions[i].data_, task->score_,
                                      auth_ctx, task->flags_);
        CLIO_CO_AWAIT(put);
        rc = put->GetReturnCode();
        out_ctx = put->context_;
      }
      if (rc != 0 && wrote_local) {
        // The unacked bytes must not survive locally.
        auto del = local->AsyncDelBlob(task->tag_id_, blob_name,
                                       clio::run::PoolQuery::Local());
        CLIO_CO_AWAIT(del);
      }
      out_ctx.replica_ = orig_ctx.replica_;
      out_ctx.origin_node_ = Context::kNoOriginNode;
      task->context_ = out_ctx;
      task->SetReturnCode(rc);
    }
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
  // Explicit replica addressing passes through untouched (these tasks were
  // routed to the blob's OWNER container by ScheduleTask).
  if (task->context_.replica_ != 0) {
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kGetBlob,
                           task.template Cast<clio::run::Task>()));
    CLIO_CO_RETURN;
  }
  {
    // This handler runs READER-LOCAL (ScheduleTask). The node-local raw
    // copy lives in the LOCAL core container's cache-replica slot — the
    // same slot the write path fills — so writer and remote reader serve
    // through one code path, and the node's SHM mirror record makes the
    // copy zero-IPC readable. INVARIANT: a present local copy is COMPLETE
    // and CURRENT (creation covers the whole blob; every put through this
    // node mirrors into it; invalidation and populate-failure delete it
    // whole), so the read is OPPORTUNISTIC — no size probe, just try it.
    const std::string blob_name = task->blob_name_.str();
    task->context_.replica_ = clio::cte::core::kCacheReplica;
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kGetBlob,
                           task.template Cast<clio::run::Task>()));
    task->context_.replica_ = 0;
    if (task->GetReturnCode() == 0) {
      CLIO_CO_RETURN;
    }

    // Miss: fetch from the blob's authoritative owner chain (Dynamic
    // hash-routes to the owner; the layers there decompress/heal). Each
    // region reads straight into the caller's buffer.
    auto *next = GetNextClient();
    std::vector<clio::cte::core::BlobRegion> regions;
    clio::cte::core::ForEachBlobRegion(*task,
        [&regions](const clio::cte::core::BlobRegion &r) {
          regions.push_back(r);
          return true;
        });
    {
      clio::run::u32 rc = 0;
      clio::run::u32 out_tflags = 0;
      for (size_t i = 0; i < regions.size() && rc == 0; ++i) {
        auto get = next->AsyncGetBlob(
            task->tag_id_, blob_name, regions[i].blob_off_, regions[i].size_,
            task->flags_, regions[i].data_, clio::run::PoolQuery::Dynamic(),
            task->context_);
        CLIO_CO_AWAIT(get);
        rc = get->GetReturnCode();
        out_tflags = get->context_.transform_flags_;
      }
      task->context_.transform_flags_ = out_tflags;
      task->SetReturnCode(rc);
      if (rc != 0) {
        CLIO_CO_RETURN;
      }
    }

    // Best-effort local population + coherence registration, so the next
    // read (and the zero-IPC fast path) is node-local. When this read
    // fetched the WHOLE blob (the file-per-process page pattern), the
    // caller's buffer seeds the copy directly — no re-fetch.
    {
      clio::run::u64 total = 0;
      {
        auto sz = next->AsyncGetBlobSize(task->tag_id_, blob_name);
        CLIO_CO_AWAIT(sz);
        if (sz->GetReturnCode() != 0 || sz->size_ == 0) {
          CLIO_CO_RETURN;  // raced away — nothing to populate
        }
        total = sz->size_;
      }
      if (regions.size() == 1 && regions[0].blob_off_ == 0 &&
          regions[0].size_ >= total &&
          task->context_.transform_flags_ == 0) {
        auto *local = GetLocalClient();
        Context put_ctx;
        AimAtCacheReplica(&put_ctx);
        auto put = local->AsyncPutBlob(task->tag_id_, blob_name, 0, total,
                                       regions[0].data_, /*score=*/-1.0f,
                                       put_ctx, /*flags=*/0,
                                       clio::run::PoolQuery::Local());
        CLIO_CO_AWAIT(put);
        if (put->GetReturnCode() == 0) {
          auto reg = next->AsyncRegisterReplicaContainer(
              task->tag_id_, blob_name, CLIO_IPC->GetNodeId());
          CLIO_CO_AWAIT(reg);
        }
      } else {
        CLIO_CO_AWAIT(PopulateLocal(task->tag_id_, blob_name));
      }
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::PopulateLocal(const TagId &tag_id,
                                             const std::string &blob_name) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  auto *next = GetNextClient();
  auto *local = GetLocalClient();
  clio::run::u64 total = 0;
  {
    // Logical size from the owner chain (the compressor reports original
    // sizes; Dynamic hash-routes to the owner).
    auto sz = next->AsyncGetBlobSize(tag_id, blob_name);
    CLIO_CO_AWAIT(sz);
    if (sz->GetReturnCode() != 0 || sz->size_ == 0) {
      CLIO_CO_RETURN;
    }
    total = sz->size_;
  }
  {
    bool failed = false;
    for (clio::run::u64 off = 0; off < total && !failed;
         off += kCacheChunkBytes) {
      clio::run::u64 len = std::min(kCacheChunkBytes, total - off);
      auto buf = CLIO_IPC->AllocateBuffer(len);
      if (buf.IsNull()) {
        failed = true;
        break;
      }
      ctp::ipc::ShmPtr<> buf_ptr = buf.shm_.template Cast<void>();
      auto get = next->AsyncGetBlob(tag_id, blob_name, off, len, /*flags=*/0,
                                    buf_ptr);
      CLIO_CO_AWAIT(get);
      if (get->GetReturnCode() != 0) {
        CLIO_IPC->FreeBuffer(buf);
        failed = true;
        break;
      }
      // Into THIS node's core container cache-replica slot (Local query
      // bypasses owner routing): raw bytes, mirror record published for the
      // node's zero-IPC fast path.
      Context put_ctx;
      AimAtCacheReplica(&put_ctx);
      auto put = local->AsyncPutBlob(tag_id, blob_name, off, len, buf_ptr,
                                     /*score=*/-1.0f, put_ctx, /*flags=*/0,
                                     clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(put);
      CLIO_IPC->FreeBuffer(buf);
      if (put->GetReturnCode() != 0) {
        failed = true;
      }
    }
    if (failed) {
      // The present ⇒ COMPLETE invariant (the opportunistic read and the
      // local size answer both rely on it) forbids keeping a prefix: a
      // partial population is deleted whole.
      auto del = local->AsyncDelBlob(tag_id, blob_name,
                                     clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(del);
      CLIO_CO_RETURN;
    }
  }
  {
    // Coherence registration at the blob's OWNER (Dynamic hash-routes
    // there): its next primary write invalidates this node's copy BEFORE
    // acking, so a covering local copy is never stale.
    auto reg = next->AsyncRegisterReplicaContainer(tag_id, blob_name,
                                                   CLIO_IPC->GetNodeId());
    CLIO_CO_AWAIT(reg);
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
    // Node-local raw copy first (this handler runs READER-LOCAL): its raw
    // size IS the logical size (no transform), and invalidation keeps a
    // present copy current. NOTE: the copy is populated whole-blob and
    // reclaimed/invalidated whole-blob, so a present size is the full
    // logical size.
    auto *local = GetLocalClient();
    auto sz = local->AsyncGetBlobSize(task->tag_id_, task->blob_name_.str(),
                                      clio::run::PoolQuery::Local(),
                                      clio::cte::core::kCacheReplica);
    CLIO_CO_AWAIT(sz);
    if (sz->GetReturnCode() == 0 && sz->size_ > 0) {
      task->size_ = sz->size_;
      task->return_code_ = 0;
      CLIO_CO_RETURN;
    }
  }
  {
    // Miss: ask the owner chain (Dynamic hash-routes there).
    auto sz = GetNextClient()->AsyncGetBlobSize(task->tag_id_,
                                                task->blob_name_.str());
    CLIO_CO_AWAIT(sz);
    task->size_ = sz->size_;
    task->SetReturnCode(sz->GetReturnCode());
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::cache

// ChiMod entry points (alloc/new/name/destroy) for the module manager.
CLIO_TASK_CC(clio::cte::cache::Runtime)

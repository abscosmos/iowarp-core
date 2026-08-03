/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#ifndef CLIO_CTE_CACHE_CACHE_RUNTIME_H_
#define CLIO_CTE_CACHE_CACHE_RUNTIME_H_

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_interposer.h>
#include <clio_cte/cache/cache_client.h>
#include <clio_cte/cache/cache_tasks.h>

namespace clio::cte::cache {

/**
 * Cache chimod runtime (issue #886 cache/replication split) — the TOP of
 * the interposition chain (cache -> compressor -> replication -> core).
 * Keeps a node-local UNTRANSFORMED copy of each blob in the core's
 * REPLICA_CACHE slot, which the SHM zero-IPC fast path and raw task reads
 * serve directly, while the authoritative bytes flow down `next` (where
 * compression, persistent replication and storage happen) via a WRITE-BACK
 * sweep:
 *
 *  - PutBlob writes the caller's raw bytes into the cache replica (through
 *    the chain with Context::replica_ = kCacheReplica — the interposers
 *    below forward explicit replica addressing verbatim), acks, and marks
 *    the blob dirty; the periodic FlushSweep pushes each dirty blob down
 *    `next` with the put's remembered Context. flush_period_ms = 0 makes
 *    puts write-through instead.
 *  - GetBlob serves from the cache replica when it covers the request
 *    (fresh by construction: the cache is written before the ack), else
 *    forwards down (decompressed below) and best-effort re-populates.
 *  - GetBlobSize answers from the cache replica (raw size == logical size)
 *    before consulting the chain — a dirty blob's downstream size is stale
 *    or absent.
 *  - MultiPutBlob batches forward down verbatim (write-through; batch
 *    records carry no per-record Context to remember).
 */
class Runtime : public clio::cte::core::CoreInterposer {
 public:
  using CreateParams = CacheConfig;  // required by CLIO_TASK_CC

  Runtime() = default;
  ~Runtime() override = default;

  // ---- Method handlers ----
  clio::run::TaskResume Create(clio::run::shared_ptr<CreateTask> &task);
  clio::run::TaskResume Destroy(clio::run::shared_ptr<DestroyTask> &task);
  clio::run::TaskResume Monitor(clio::run::shared_ptr<MonitorTask> &task);
  clio::run::TaskResume FlushSweep(clio::run::shared_ptr<FlushSweepTask> &task);
  clio::run::TaskResume PutBlob(
      clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task);
  clio::run::TaskResume GetBlob(
      clio::run::shared_ptr<clio::cte::core::GetBlobTask> &task);
  clio::run::TaskResume GetBlobSize(
      clio::run::shared_ptr<clio::cte::core::GetBlobSizeTask> &task);
  clio::run::TaskResume MultiPutBlob(
      clio::run::shared_ptr<clio::cte::core::MultiPutBlobTask> &task);

  // ---- Container virtuals (defined in autogen/cache_lib_exec.cc) ----
  void Init(const clio::run::PoolId &pool_id, const std::string &pool_name,
            clio::run::u32 container_id = 0) override;
  clio::run::TaskResume Run(clio::run::u32 method,
                      clio::run::shared_ptr<clio::run::Task> task_ptr) override;
  clio::run::u64 GetWorkRemaining() const override;
  void LocalLoadTask(clio::run::u32 method, clio::run::DefaultLoadArchive &archive,
                     clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> LocalAllocLoadTask(
      clio::run::u32 method, clio::run::DefaultLoadArchive &archive) override;
  void LocalSaveTask(clio::run::u32 method, clio::run::DefaultSaveArchive &archive,
                     clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  void AggregateOut(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task,
                    const clio::run::shared_ptr<clio::run::Task> &replica_task) override;
  void AggregateIn(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &agg_task,
                   const clio::run::shared_ptr<clio::run::Task> &member_task) override;
  void SaveTask(clio::run::u32 method, clio::run::SaveTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  void LoadTask(clio::run::u32 method, clio::run::LoadTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> AllocLoadTask(clio::run::u32 method,
                                             clio::run::LoadTaskArchive &archive) override;
  clio::run::shared_ptr<clio::run::Task> NewCopyTask(clio::run::u32 method,
                                           clio::run::shared_ptr<clio::run::Task> &orig,
                                           bool deep) override;
  clio::run::shared_ptr<clio::run::Task> NewTask(clio::run::u32 method) override;

 private:
  /** One dirty blob awaiting write-back: the put's Context (compression
   *  settings and all) and score travel down with the flush. */
  struct DirtyEntry {
    TagId tag_id_;
    std::string blob_name_;
    Context context_;
    float score_;
  };

  /** Mark a blob dirty (deduped by key; the LATEST context/score wins —
   *  overwrites coalesce into one flush of the final bytes). */
  void EnqueueFlush(const TagId &tag_id, const std::string &blob_name,
                    const Context &context, float score);

  /**
   * Push one dirty blob's raw bytes from its cache replica down `next` in
   * bounded chunks with the remembered context. rc: 0 ok, 1 = blob/cache
   * copy gone (drop the entry), else keep dirty.
   */
  clio::run::TaskResume FlushOne(const DirtyEntry &entry, clio::run::u32 &rc);

  /**
   * Best-effort re-population of the cache replica from the authoritative
   * chain (used on read misses): logical size down, chunked raw reads down
   * (decompressed by the chain), chunked cache-replica puts. Sequential from
   * 0, so an interruption leaves a valid prefix.
   */
  clio::run::TaskResume Repopulate(const TagId &tag_id,
                                   const std::string &blob_name);

  /** Lazily bind the next-pool client (compose next_pool_id). */
  clio::cte::core::Client *GetNextClient();

  /** True if the blob is dirty (cache newer than downstream). */
  bool IsDirty(const TagId &tag_id, const std::string &blob_name);

  CacheConfig config_;
  std::unique_ptr<clio::cte::core::Client> next_client_;

  std::mutex dirty_mtx_;
  std::unordered_map<std::string, DirtyEntry> dirty_;
};

}  // namespace clio::cte::cache

#endif  // CLIO_CTE_CACHE_CACHE_RUNTIME_H_

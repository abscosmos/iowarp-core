/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#ifndef CLIO_CTE_REPLICATION_REPLICATION_RUNTIME_H_
#define CLIO_CTE_REPLICATION_REPLICATION_RUNTIME_H_

#include <memory>
#include <string>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/replication/replication_client.h>
#include <clio_cte/replication/replication_tasks.h>

namespace clio::cte::replication {

/**
 * Replication chimod runtime (issue #886). The CTE core stores replicas and
 * addresses them (Context::replica_); this module owns the POLICY of keeping
 * copies of cached data on persistent tiers: ReplicateBlob copies one blob's
 * primary bytes into a chosen replica, FlushTag sweeps a whole tag. Both go
 * through the public core client API (GetBlob → PutBlob with
 * Context::replica_ set), so replica placement obeys the same DPE and
 * persistence constraints as any put.
 */
class Runtime : public clio::run::Container {
 public:
  using CreateParams = ReplicationConfig;  // required by CLIO_TASK_CC

  Runtime() = default;
  ~Runtime() override = default;

  // ---- Method handlers ----
  clio::run::TaskResume Create(clio::run::shared_ptr<CreateTask> &task);
  clio::run::TaskResume Destroy(clio::run::shared_ptr<DestroyTask> &task);
  clio::run::TaskResume Monitor(clio::run::shared_ptr<MonitorTask> &task);
  clio::run::TaskResume ReplicateBlob(
      clio::run::shared_ptr<ReplicateBlobTask> &task);
  clio::run::TaskResume FlushTag(clio::run::shared_ptr<FlushTagTask> &task);
  clio::run::TaskResume CachedPut(clio::run::shared_ptr<CachedPutTask> &task);
  clio::run::TaskResume CachedGet(clio::run::shared_ptr<CachedGetTask> &task);

  // ---- Container virtuals (defined in autogen/replication_lib_exec.cc) ----
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
  /**
   * Copy one blob's primary bytes into replica `replica_idx`, chunked so a
   * large blob never needs a blob-sized bounce buffer. rc uses ReplicateBlob's
   * return-code space: 2 alloc failure, 10+x GetBlobSize, 20+x GetBlob,
   * 30+x PutBlob.
   */
  clio::run::TaskResume ReplicateOne(const TagId &tag_id,
                                     const std::string &blob_name,
                                     int replica_idx,
                                     const Context &context,
                                     clio::run::u64 &bytes_copied,
                                     clio::run::u32 &rc);

  /**
   * Copy replica `replica_idx`'s FULL contents back into the primary,
   * sequentially from offset 0 in bounded chunks, so an interruption leaves
   * a valid prefix (future CachedGets treat uncovered ranges as misses).
   * Best-effort: a failed chunk stops the copy without failing the read that
   * triggered it. recached reports bytes restored.
   */
  clio::run::TaskResume RecachePrimary(const TagId &tag_id,
                                       const std::string &blob_name,
                                       int replica_idx,
                                       clio::run::u64 rep_size,
                                       clio::run::u64 &recached);

  /**
   * Populate THIS node's local cache copy of a remote blob (issue #886
   * distributed coherence): chunked copy of the owner's primary into the
   * LOCAL container (PoolQuery::Local puts), then register this node with
   * the owner (RegisterReplicaContainer) so the next primary write
   * invalidates the copy. Best-effort — a failed chunk abandons the local
   * copy without failing the read that triggered it, and registration only
   * happens after a COMPLETE copy (a partial local copy is never served
   * because CachedGet requires coverage, but registering it would earn a
   * pointless invalidation).
   */
  clio::run::TaskResume CacheLocalCopy(const TagId &tag_id,
                                       const std::string &blob_name,
                                       clio::run::u64 total,
                                       bool &cached);

  /** Lazily bind the core client (compose next_pool_id, default kCtePoolId) */
  clio::cte::core::Client *GetCoreClient();

  ReplicationConfig config_;
  std::unique_ptr<clio::cte::core::Client> core_client_;
};

}  // namespace clio::cte::replication

#endif  // CLIO_CTE_REPLICATION_REPLICATION_RUNTIME_H_

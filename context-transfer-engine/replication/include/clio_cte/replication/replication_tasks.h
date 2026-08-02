/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#ifndef CLIO_CTE_REPLICATION_REPLICATION_TASKS_H_
#define CLIO_CTE_REPLICATION_REPLICATION_TASKS_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/task.h>
#include <clio_runtime/admin/admin_tasks.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/replication/autogen/replication_methods.h>

#include <string>

namespace clio::cte::replication {

/** The replication chimod speaks the CTE core's vocabulary. */
using Context = clio::cte::core::Context;
using TagId = clio::cte::core::TagId;

/**
 * Well-known default pool id/name, following the filesystem chimod's
 * kCfsPoolId(560,0) convention (the CTE core is 512.0).
 */
static constexpr clio::run::PoolId kReplicationPoolId(561, 0);
static constexpr const char *kReplicationPoolName = "clio_cte_replication";

/**
 * Container creation params. next_pool_id_ is the CTE core pool whose blobs
 * this module replicates (where the tags/blobs actually live).
 */
struct ReplicationConfig {
  static constexpr const char* chimod_lib_name = "clio_cte_replication";

  clio::run::PoolId next_pool_id_;  ///< CTE core pool id (e.g. 512.0)
  /// The FIXED SET of persistent replicas the caching verbs maintain: every
  /// CachedPut writes through to replicas 1..num_replicas_, each marked
  /// REPLICA_FIXED | REPLICA_PERSISTENT so the organizer neither migrates
  /// nor volatilizes them.
  int num_replicas_ = 1;
  /// Score for the primary (the DRAM cache copy) on CachedPut / re-cache.
  /// High by default so the DPE pins the fast copy to the fast tier.
  float cache_score_ = 1.0f;
  /// Score CachedPut stamps on the persistent replicas. This is also the
  /// organizer's DROP threshold for the cache copy: when the primary's score
  /// sinks below the best persistent replica's, the primary is dropped
  /// rather than migrated. Modest by default — it should mirror the slow
  /// tier the durable copies live on, not the cache's.
  float replica_score_ = 0.2f;

  ReplicationConfig() : next_pool_id_(clio::run::PoolId::GetNull()) {}
  ReplicationConfig(const clio::run::PoolId &pool_id,
                    const ReplicationConfig &other)
      : next_pool_id_(other.next_pool_id_),
        num_replicas_(other.num_replicas_),
        cache_score_(other.cache_score_),
        replica_score_(other.replica_score_) {
    (void)pool_id;
  }

  template <class Archive>
  void serialize(Archive &ar) {
    ar(next_pool_id_, num_replicas_, cache_score_, replica_score_);
  }

  /** Load configuration from compose YAML (next_pool_id: "major.minor",
   *  num_replicas: int, cache_score: float). */
  void LoadConfig(const clio::run::PoolConfig &pool_config) {
    if (!pool_config.config_.empty()) {
      try {
        YAML::Node node = YAML::Load(pool_config.config_);
        if (node["next_pool_id"]) {
          std::string next_str = node["next_pool_id"].as<std::string>();
          auto dot = next_str.find('.');
          if (dot != std::string::npos) {
            clio::run::u32 major = std::stoul(next_str.substr(0, dot));
            clio::run::u32 minor = std::stoul(next_str.substr(dot + 1));
            next_pool_id_ = clio::run::PoolId(major, minor);
          }
        }
        if (node["num_replicas"]) {
          num_replicas_ = node["num_replicas"].as<int>();
        }
        if (node["cache_score"]) {
          cache_score_ = node["cache_score"].as<float>();
        }
        if (node["replica_score"]) {
          replica_score_ = node["replica_score"].as<float>();
        }
      } catch (...) {
        // Config parsing is best-effort
      }
    }
  }
};

/** Standard pool creation. */
using CreateTask = clio::run::admin::GetOrCreatePoolTask<ReplicationConfig>;

/** Cleanup the replication container. */
struct DestroyTask : public clio::run::Task {
  DestroyTask() : clio::run::Task() {}

  explicit DestroyTask(const clio::run::TaskId &task_id,
                       const clio::run::PoolId &pool_id,
                       const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kDestroy) {}

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<DestroyTask>());
  }

  void Copy(const ctp::ipc::FullPtr<DestroyTask>& other) {
    Task::Copy(other.template Cast<clio::run::Task>());
  }

  // SerializeIn/SerializeOut, NOT the compressor's SerializeStart/End: the
  // client-mode transport drives In/Out, and a chimod meant to be called
  // from a separate client process must speak it (the filesystem chimod
  // convention). Same for every task below.
  template <typename Ar> void SerializeIn(Ar &ar) { Task::SerializeIn(ar); }
  template <typename Ar> void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

using MonitorTask = clio::run::admin::MonitorTask;

/**
 * ReplicateBlobTask — bring replica `replica_` of one blob up to date with
 * the primary (full-blob copy through GetBlob → PutBlob with
 * Context::replica_ set). The context carries the placement constraints for
 * the replica's blocks — min_persistence_level_ is the whole point: a
 * RAM-cached primary gets a persistent copy without being demoted.
 */
struct ReplicateBlobTask : public clio::run::Task {
  IN TagId tag_id_;                        // Tag the blob lives under
  IN clio::run::priv::string blob_name_;   // Blob to replicate
  IN int replica_;              // Target replica index (>= 1)
  IN Context context_;                     // Placement for the replica blocks
  OUT clio::run::u64 bytes_copied_;        // Bytes copied primary -> replica

  ReplicateBlobTask()
      : clio::run::Task(), tag_id_(TagId::GetNull()), blob_name_(CTP_MALLOC),
        replica_(0), context_(), bytes_copied_(0) {}

  explicit ReplicateBlobTask(const clio::run::TaskId &task_id,
                             const clio::run::PoolId &pool_id,
                             const clio::run::PoolQuery &pool_query,
                             const TagId &tag_id, const std::string &blob_name,
                             int replica, const Context &context)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kReplicateBlob),
        tag_id_(tag_id), blob_name_(CTP_MALLOC, blob_name), replica_(replica),
        context_(context), bytes_copied_(0) {}

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<ReplicateBlobTask>());
  }

  void Copy(const ctp::ipc::FullPtr<ReplicateBlobTask>& other) {
    Task::Copy(other.template Cast<clio::run::Task>());
    tag_id_ = other->tag_id_;
    blob_name_ = other->blob_name_;
    replica_ = other->replica_;
    context_ = other->context_;
    bytes_copied_ = other->bytes_copied_;
  }

  template <typename Ar>
  void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar);
    ar(tag_id_, blob_name_, replica_, context_);
  }

  template <typename Ar>
  void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar);
    ar(bytes_copied_);
  }
};

/**
 * FlushTagTask — ReplicateBlob for every blob in a tag whose score is at
 * least min_score_ (0 = all of them). This is the "make this dataset's RAM
 * cache durable" verb: pair it with a context whose min_persistence_level_
 * pins the replica blocks to persistent tiers.
 */
struct FlushTagTask : public clio::run::Task {
  IN TagId tag_id_;                     // Tag to flush
  IN int replica_;           // Target replica index (>= 1)
  IN float min_score_;                  // Only blobs with score_ >= this
  IN Context context_;                  // Placement for the replica blocks
  OUT clio::run::u64 blobs_replicated_;  // Blobs actually copied
  OUT clio::run::u64 bytes_copied_;      // Total bytes copied

  FlushTagTask()
      : clio::run::Task(), tag_id_(TagId::GetNull()), replica_(0),
        min_score_(0.0f), context_(), blobs_replicated_(0), bytes_copied_(0) {}

  explicit FlushTagTask(const clio::run::TaskId &task_id,
                        const clio::run::PoolId &pool_id,
                        const clio::run::PoolQuery &pool_query,
                        const TagId &tag_id, int replica,
                        float min_score, const Context &context)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kFlushTag),
        tag_id_(tag_id), replica_(replica), min_score_(min_score),
        context_(context), blobs_replicated_(0), bytes_copied_(0) {}

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<FlushTagTask>());
  }

  void Copy(const ctp::ipc::FullPtr<FlushTagTask>& other) {
    Task::Copy(other.template Cast<clio::run::Task>());
    tag_id_ = other->tag_id_;
    replica_ = other->replica_;
    min_score_ = other->min_score_;
    context_ = other->context_;
    blobs_replicated_ = other->blobs_replicated_;
    bytes_copied_ = other->bytes_copied_;
  }

  template <typename Ar>
  void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar);
    ar(tag_id_, replica_, min_score_, context_);
  }

  template <typename Ar>
  void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar);
    ar(blobs_replicated_, bytes_copied_);
  }
};

/**
 * CachedPutTask — write-through cached put (issue #886 caching model).
 * Writes the primary (the DRAM cache copy, high score, volatile-ok) FIRST so
 * reads always see the newest bytes, then writes through to the module's
 * fixed set of persistent replicas (1..num_replicas from config), each
 * marked REPLICA_FIXED | REPLICA_PERSISTENT. Any copy failing surfaces in
 * the return code; the primary-first order means a reported failure can
 * leave durability lagging but never a stale cache.
 */
struct CachedPutTask : public clio::run::Task {
  IN TagId tag_id_;
  IN clio::run::priv::string blob_name_;
  IN clio::run::u64 offset_;
  IN clio::run::u64 size_;
  IN ctp::ipc::ShmPtr<> blob_data_;
  IN float score_;        // Cache-copy score; -1 = config cache_score_
  IN Context context_;    // Passed through (its replica fields are overridden)
  OUT clio::run::u32 replicas_written_;  // Persistent copies updated

  CachedPutTask()
      : clio::run::Task(), tag_id_(TagId::GetNull()), blob_name_(CTP_MALLOC),
        offset_(0), size_(0), blob_data_(ctp::ipc::ShmPtr<>::GetNull()),
        score_(-1.0f), context_(), replicas_written_(0) {}

  explicit CachedPutTask(const clio::run::TaskId &task_id,
                         const clio::run::PoolId &pool_id,
                         const clio::run::PoolQuery &pool_query,
                         const TagId &tag_id, const std::string &blob_name,
                         clio::run::u64 offset, clio::run::u64 size,
                         ctp::ipc::ShmPtr<> blob_data, float score,
                         const Context &context)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kCachedPut),
        tag_id_(tag_id), blob_name_(CTP_MALLOC, blob_name), offset_(offset),
        size_(size), blob_data_(blob_data), score_(score), context_(context),
        replicas_written_(0) {}

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<CachedPutTask>());
  }

  void Copy(const ctp::ipc::FullPtr<CachedPutTask>& other) {
    Task::Copy(other.template Cast<clio::run::Task>());
    tag_id_ = other->tag_id_;
    blob_name_ = other->blob_name_;
    offset_ = other->offset_;
    size_ = other->size_;
    blob_data_ = other->blob_data_;
    score_ = other->score_;
    context_ = other->context_;
    replicas_written_ = other->replicas_written_;
  }

  template <typename Ar>
  void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar);
    ar(tag_id_, blob_name_, offset_, size_, blob_data_, score_, context_);
    ar.bulk(blob_data_, size_, BULK_XFER);
  }

  template <typename Ar>
  void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar);
    ar(replicas_written_);
  }
};

/**
 * CachedGetTask — cached read (issue #886 caching model). Served from the
 * primary when it covers the range; otherwise (the organizer dropped the
 * cache copy) from the first persistent replica that covers it, after which
 * the WHOLE replica is copied back into the primary — sequentially from 0,
 * so an interrupted re-cache leaves a valid prefix, never a torn copy — to
 * restore the DRAM fast path. The re-populated primary is an ordinary
 * primary: no REPLICA_PERSISTENT anywhere near it, so the organizer remains
 * free to drop it again.
 */
struct CachedGetTask : public clio::run::Task {
  IN TagId tag_id_;
  IN clio::run::priv::string blob_name_;
  IN clio::run::u64 offset_;
  IN clio::run::u64 size_;
  IN ctp::ipc::ShmPtr<> blob_data_;  // Caller's destination buffer
  IN Context context_;               // Passed through (replica overridden)
  OUT int from_replica_;             // 0 = cache hit; N = served from replica N
  OUT clio::run::u64 recached_bytes_;  // Bytes copied back into the primary

  CachedGetTask()
      : clio::run::Task(), tag_id_(TagId::GetNull()), blob_name_(CTP_MALLOC),
        offset_(0), size_(0), blob_data_(ctp::ipc::ShmPtr<>::GetNull()),
        context_(), from_replica_(0), recached_bytes_(0) {}

  explicit CachedGetTask(const clio::run::TaskId &task_id,
                         const clio::run::PoolId &pool_id,
                         const clio::run::PoolQuery &pool_query,
                         const TagId &tag_id, const std::string &blob_name,
                         clio::run::u64 offset, clio::run::u64 size,
                         ctp::ipc::ShmPtr<> blob_data, const Context &context)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kCachedGet),
        tag_id_(tag_id), blob_name_(CTP_MALLOC, blob_name), offset_(offset),
        size_(size), blob_data_(blob_data), context_(context),
        from_replica_(0), recached_bytes_(0) {}

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<CachedGetTask>());
  }

  void Copy(const ctp::ipc::FullPtr<CachedGetTask>& other) {
    Task::Copy(other.template Cast<clio::run::Task>());
    tag_id_ = other->tag_id_;
    blob_name_ = other->blob_name_;
    offset_ = other->offset_;
    size_ = other->size_;
    blob_data_ = other->blob_data_;
    context_ = other->context_;
    from_replica_ = other->from_replica_;
    recached_bytes_ = other->recached_bytes_;
  }

  template <typename Ar>
  void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar);
    ar(tag_id_, blob_name_, offset_, size_, blob_data_, context_);
    ar.bulk(blob_data_, size_, BULK_EXPOSE);
  }

  template <typename Ar>
  void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar);
    ar(from_replica_, recached_bytes_);
    ar.bulk(blob_data_, size_, BULK_XFER);
  }
};

}  // namespace clio::cte::replication

#endif  // CLIO_CTE_REPLICATION_REPLICATION_TASKS_H_

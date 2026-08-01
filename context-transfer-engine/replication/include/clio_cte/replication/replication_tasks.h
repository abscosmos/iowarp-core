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

  ReplicationConfig() : next_pool_id_(clio::run::PoolId::GetNull()) {}
  ReplicationConfig(const clio::run::PoolId &pool_id,
                    const ReplicationConfig &other)
      : next_pool_id_(other.next_pool_id_) {
    (void)pool_id;
  }

  template <class Archive>
  void serialize(Archive &ar) {
    ar(next_pool_id_);
  }

  /** Load configuration from compose YAML (next_pool_id: "major.minor"). */
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

  void Copy(const ctp::ipc::FullPtr<DestroyTask>& other) {
    (void)other;  // no fields beyond clio::run::Task
  }

  template <typename Ar> void SerializeStart(Ar &ar) { task_serialize<Ar>(ar); }
  template <typename Ar> void SerializeEnd(Ar &ar) {}
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

  void Copy(const ctp::ipc::FullPtr<ReplicateBlobTask>& other) {
    tag_id_ = other->tag_id_;
    blob_name_ = other->blob_name_;
    replica_ = other->replica_;
    context_ = other->context_;
    bytes_copied_ = other->bytes_copied_;
  }

  template <typename Ar>
  void SerializeStart(Ar &ar) {
    task_serialize<Ar>(ar);
    ar(tag_id_, blob_name_, replica_, context_, bytes_copied_);
  }

  template <typename Ar>
  void SerializeEnd(Ar &ar) {
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

  void Copy(const ctp::ipc::FullPtr<FlushTagTask>& other) {
    tag_id_ = other->tag_id_;
    replica_ = other->replica_;
    min_score_ = other->min_score_;
    context_ = other->context_;
    blobs_replicated_ = other->blobs_replicated_;
    bytes_copied_ = other->bytes_copied_;
  }

  template <typename Ar>
  void SerializeStart(Ar &ar) {
    task_serialize<Ar>(ar);
    ar(tag_id_, replica_, min_score_, context_, blobs_replicated_,
       bytes_copied_);
  }

  template <typename Ar>
  void SerializeEnd(Ar &ar) {
    ar(blobs_replicated_, bytes_copied_);
  }
};

}  // namespace clio::cte::replication

#endif  // CLIO_CTE_REPLICATION_REPLICATION_TASKS_H_

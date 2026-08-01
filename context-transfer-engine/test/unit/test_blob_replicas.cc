/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * BLOB REPLICA TESTS (issue #886)
 *
 * The CTE mechanism:
 *  - Context::replica_ == N > 0 targets replica N on put/get; a first write
 *    creates the replica lazily; primary bytes are untouched by replica
 *    writes and vice versa (isolation).
 *  - Reads of a replica no write created fail cleanly, as do reads with
 *    replica_ == kAllReplicas (a write-through selector, not a source).
 *  - Context::replica_ == kAllReplicas writes through: primary AND every
 *    existing replica receive the bytes under one write-token hold.
 *  - DelBlob destroys replicas with the blob (their blocks are freed).
 *
 * The replication chimod's policy verbs on top:
 *  - ReplicateBlob copies one blob's primary into a chosen replica.
 *  - FlushTag sweeps every qualifying blob in a tag into a replica.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/replication/replication_client.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace fs = std::filesystem;

static std::string chi_test_data_dir() {
  const char *d = clio::run::env::GetCompat("TEST_DATA_DIR");
  return (d && *d) ? d : ".";
}

static constexpr clio::run::u64 kValSize = 4096;

class BlobReplicasFixture {
 public:
  std::string config_path_;
  std::string file_storage_path_;

  BlobReplicasFixture() {
    config_path_ = chi_test_data_dir() + "/blob_replicas_config.yaml";
    file_storage_path_ = chi_test_data_dir() + "/blob_replicas_file.dat";
    Cleanup();
    CreateConfigFile();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path_.c_str(), 1);

    bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    REQUIRE(success);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    success = clio::cte::core::CLIO_CTE_CLIENT_INIT();
    REQUIRE(success);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  ~BlobReplicasFixture() { Cleanup(); }

  void Cleanup() {
    if (fs::exists(config_path_)) fs::remove(config_path_);
    if (fs::exists(file_storage_path_)) fs::remove(file_storage_path_);
  }

  void CreateConfigFile() {
    std::ofstream config_file(config_path_);
    REQUIRE(config_file.is_open());
    // Two tiers: a fast VOLATILE RAM tier and a small non-volatile file
    // tier. The file tier is deliberately tiny (1MB) so the
    // REPLICA_PERSISTENT test can prove enforcement by exhaustion: a
    // persistent replica larger than 1MB has nowhere legal to go and must
    // fail, while the same payload without the flag lands in RAM.
    config_file << R"(
# Blob replica test configuration - RAM (volatile) + small file (temporary)
runtime:
  num_threads: 2
  queue_depth: 1024
  first_busy_wait: 10000
  max_sleep: 50000

compose:
  - mod_name: clio_cte_core
    pool_name: clio_cte
    pool_query: local
    pool_id: 512.0

    targets:
      neighborhood: 1
      default_target_timeout_ms: 30000
      poll_period_ms: 5000

    storage:
      - path: "ram::blob_replicas_dram"
        bdev_type: "ram"
        capacity_limit: "64MB"
        score: 1.0

      - path: ")" << file_storage_path_ << R"("
        bdev_type: "file"
        capacity_limit: "1MB"
        score: 0.2
        persistence_level: "temporary"

    dpe:
      dpe_type: "max_bw"
)";
    config_file.close();
  }
};

static clio::cte::core::Context ReplicaCtx(int replica) {
  clio::cte::core::Context ctx;
  ctx.replica_ = replica;
  return ctx;
}

/** Put `val` into the given replica (0 = primary) of tag/name. */
static void PutTo(clio::cte::core::Client *client,
                  const clio::cte::core::TagId &tag_id,
                  const std::string &name, const std::string &val,
                  int replica) {
  auto fut = client->AsyncPutBlob(tag_id, name, 0, val.size(), val.data(),
                                  /*score=*/-1.0f, ReplicaCtx(replica));
  fut.Wait();
  REQUIRE(fut->GetReturnCode() == 0);
}

/** Read kValSize bytes from the given replica; returns the task rc. */
static clio::run::u32 GetFrom(clio::cte::core::Client *client,
                              const clio::cte::core::TagId &tag_id,
                              const std::string &name, std::vector<char> *out,
                              int replica) {
  out->assign(kValSize, 0);
  auto fut = client->AsyncGetBlob(tag_id, name, 0, kValSize, /*flags=*/0,
                                  out->data(), clio::run::PoolQuery::Dynamic(),
                                  ReplicaCtx(replica));
  fut.Wait();
  return fut->GetReturnCode();
}

TEST_CASE("BlobReplicas - replica write/read isolation",
          "[cte][replicas][886]") {
  BlobReplicasFixture fixture;
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);

  clio::cte::core::Tag tag("replica_iso_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  const std::string primary_val(kValSize, 'P');
  const std::string replica_val(kValSize, 'R');

  // Primary put, then a DIFFERENT payload into replica 1.
  PutTo(client, tag_id, "iso_blob", primary_val, 0);
  PutTo(client, tag_id, "iso_blob", replica_val, 1);

  // Primary read is untouched by the replica write; replica read returns the
  // replica's bytes.
  std::vector<char> got;
  REQUIRE(GetFrom(client, tag_id, "iso_blob", &got, 0) == 0);
  REQUIRE(std::memcmp(got.data(), primary_val.data(), kValSize) == 0);
  REQUIRE(GetFrom(client, tag_id, "iso_blob", &got, 1) == 0);
  REQUIRE(std::memcmp(got.data(), replica_val.data(), kValSize) == 0);

  // A replica no write created fails cleanly, as does the write-through
  // selector used as a read source.
  REQUIRE(GetFrom(client, tag_id, "iso_blob", &got, 2) != 0);
  REQUIRE(GetFrom(client, tag_id, "iso_blob", &got,
                  clio::cte::core::kAllReplicas) != 0);

  // A later primary overwrite leaves the replica's bytes alone.
  const std::string primary_v2(kValSize, 'Q');
  PutTo(client, tag_id, "iso_blob", primary_v2, 0);
  REQUIRE(GetFrom(client, tag_id, "iso_blob", &got, 1) == 0);
  REQUIRE(std::memcmp(got.data(), replica_val.data(), kValSize) == 0);
}

TEST_CASE("BlobReplicas - write-through to all replicas",
          "[cte][replicas][886]") {
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);

  clio::cte::core::Tag tag("replica_wt_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  // Seed primary + replica 1 with distinct old bytes.
  PutTo(client, tag_id, "wt_blob", std::string(kValSize, 'p'), 0);
  PutTo(client, tag_id, "wt_blob", std::string(kValSize, 'r'), 1);

  // Write-through: one put updates primary AND the existing replica.
  const std::string new_val(kValSize, 'W');
  PutTo(client, tag_id, "wt_blob", new_val, clio::cte::core::kAllReplicas);

  std::vector<char> got;
  REQUIRE(GetFrom(client, tag_id, "wt_blob", &got, 0) == 0);
  REQUIRE(std::memcmp(got.data(), new_val.data(), kValSize) == 0);
  REQUIRE(GetFrom(client, tag_id, "wt_blob", &got, 1) == 0);
  REQUIRE(std::memcmp(got.data(), new_val.data(), kValSize) == 0);

  // Write-through does NOT invent replicas: replica 2 still doesn't exist.
  REQUIRE(GetFrom(client, tag_id, "wt_blob", &got, 2) != 0);
}

TEST_CASE("BlobReplicas - delete destroys replicas with the blob",
          "[cte][replicas][886]") {
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);

  clio::cte::core::Tag tag("replica_del_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  PutTo(client, tag_id, "del_blob", std::string(kValSize, 'p'), 0);
  PutTo(client, tag_id, "del_blob", std::string(kValSize, 'r'), 1);

  auto del = client->AsyncDelBlob(tag_id, "del_blob");
  del.Wait();
  REQUIRE(del->GetReturnCode() == 0);

  std::vector<char> got;
  REQUIRE(GetFrom(client, tag_id, "del_blob", &got, 0) != 0);
  REQUIRE(GetFrom(client, tag_id, "del_blob", &got, 1) != 0);
}

TEST_CASE("BlobReplicas - per-replica score drives reorganizer migration",
          "[cte][replicas][reorganize][886]") {
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);

  clio::cte::core::Tag tag("replica_reorg_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  const std::string primary_val(kValSize, 'p');
  const std::string replica_val(kValSize, 'r');
  PutTo(client, tag_id, "reorg_blob", primary_val, 0);
  PutTo(client, tag_id, "reorg_blob", replica_val, 1);

  // Migrate the REPLICA to the slow tier by its own score; the primary's
  // score and placement are untouched.
  auto move_down = client->AsyncReorganizeBlob(
      tag_id, "reorg_blob", 0.2f, clio::run::PoolQuery::Dynamic(), /*replica=*/1);
  move_down.Wait();
  REQUIRE(move_down->GetReturnCode() == 0);

  std::vector<char> got;
  REQUIRE(GetFrom(client, tag_id, "reorg_blob", &got, 1) == 0);
  REQUIRE(std::memcmp(got.data(), replica_val.data(), kValSize) == 0);
  REQUIRE(GetFrom(client, tag_id, "reorg_blob", &got, 0) == 0);
  REQUIRE(std::memcmp(got.data(), primary_val.data(), kValSize) == 0);

  // The replica remembers ITS score: the same target score again is a
  // below-threshold no-op, and moving back up works.
  auto again = client->AsyncReorganizeBlob(
      tag_id, "reorg_blob", 0.2f, clio::run::PoolQuery::Dynamic(), 1);
  again.Wait();
  REQUIRE(again->GetReturnCode() == 0);
  auto move_up = client->AsyncReorganizeBlob(
      tag_id, "reorg_blob", 1.0f, clio::run::PoolQuery::Dynamic(), 1);
  move_up.Wait();
  REQUIRE(move_up->GetReturnCode() == 0);
  REQUIRE(GetFrom(client, tag_id, "reorg_blob", &got, 1) == 0);
  REQUIRE(std::memcmp(got.data(), replica_val.data(), kValSize) == 0);

  // Reorganizing a replica no write created fails cleanly.
  auto absent = client->AsyncReorganizeBlob(
      tag_id, "reorg_blob", 0.5f, clio::run::PoolQuery::Dynamic(), 7);
  absent.Wait();
  REQUIRE(absent->GetReturnCode() != 0);
}

TEST_CASE("BlobReplicas - REPLICA_FIXED pins a replica against the organizer",
          "[cte][replicas][reorganize][886]") {
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);

  clio::cte::core::Tag tag("replica_fixed_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  const std::string replica_val(kValSize, 'F');
  PutTo(client, tag_id, "fixed_blob", std::string(kValSize, 'p'), 0);
  {
    clio::cte::core::Context ctx = ReplicaCtx(1);
    ctx.replica_flags_ = clio::cte::core::REPLICA_FIXED;
    auto fut = client->AsyncPutBlob(tag_id, "fixed_blob", 0, kValSize,
                                    replica_val.data(), /*score=*/-1.0f, ctx);
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
  }

  // The reorganizer must not touch a FIXED replica: success (the organizer
  // sweeps blindly; FIXED is the filter), bytes stay put.
  auto fut = client->AsyncReorganizeBlob(
      tag_id, "fixed_blob", 0.2f, clio::run::PoolQuery::Dynamic(), 1);
  fut.Wait();
  REQUIRE(fut->GetReturnCode() == 0);
  std::vector<char> got;
  REQUIRE(GetFrom(client, tag_id, "fixed_blob", &got, 1) == 0);
  REQUIRE(std::memcmp(got.data(), replica_val.data(), kValSize) == 0);
}

TEST_CASE("BlobReplicas - REPLICA_PERSISTENT excludes volatile tiers",
          "[cte][replicas][persistence][886]") {
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);

  clio::cte::core::Tag tag("replica_persist_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  // A payload bigger than the ONLY non-volatile tier (1MB file): as a plain
  // replica it lands in RAM; as a PERSISTENT replica it has nowhere legal
  // to go and the put must fail — proof the flag really excludes the
  // volatile tier rather than merely preferring the file one.
  const clio::run::u64 kBig = 4ULL * 1024 * 1024;
  const std::string big(kBig, 'B');
  {
    auto fut = client->AsyncPutBlob(tag_id, "persist_blob", 0, kBig,
                                    big.data(), -1.0f, ReplicaCtx(1));
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
  }
  {
    clio::cte::core::Context ctx = ReplicaCtx(2);
    ctx.replica_flags_ = clio::cte::core::REPLICA_PERSISTENT;
    auto fut = client->AsyncPutBlob(tag_id, "persist_blob", 0, kBig,
                                    big.data(), -1.0f, ctx);
    fut.Wait();
    REQUIRE(fut->GetReturnCode() != 0);
  }

  // A small PERSISTENT replica fits the file tier; seed it with a LOW score,
  // then reorganize it toward the volatile tier's score — the move must
  // succeed while staying on non-volatile storage (rc 0, bytes intact).
  const clio::run::u64 kSmall = 64 * 1024;
  const std::string small_val(kSmall, 'S');
  PutTo(client, tag_id, "persist_small", std::string(kSmall, 'p'), 0);
  {
    clio::cte::core::Context ctx = ReplicaCtx(1);
    ctx.replica_flags_ = clio::cte::core::REPLICA_PERSISTENT;
    auto fut = client->AsyncPutBlob(tag_id, "persist_small", 0, kSmall,
                                    small_val.data(), /*score=*/0.2f, ctx);
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
  }
  {
    auto fut = client->AsyncReorganizeBlob(
        tag_id, "persist_small", 1.0f, clio::run::PoolQuery::Dynamic(), 1);
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
    std::vector<char> got(kSmall, 0);
    auto get = client->AsyncGetBlob(tag_id, "persist_small", 0, kSmall,
                                    /*flags=*/0, got.data(),
                                    clio::run::PoolQuery::Dynamic(),
                                    ReplicaCtx(1));
    get.Wait();
    REQUIRE(get->GetReturnCode() == 0);
    REQUIRE(std::memcmp(got.data(), small_val.data(), kSmall) == 0);
  }
}

TEST_CASE("BlobReplicas - replication module ReplicateBlob and FlushTag",
          "[cte][replicas][replication][886]") {
  auto *core_client = CLIO_CTE_CLIENT;
  REQUIRE(core_client != nullptr);

  // Create/bind the replication pool over the default CTE core pool.
  clio::cte::replication::Client repl(
      clio::cte::replication::kReplicationPoolId, clio::cte::core::kCtePoolId);
  {
    clio::cte::replication::ReplicationConfig params;
    params.next_pool_id_ = clio::cte::core::kCtePoolId;
    auto create = repl.AsyncCreateReplication(
        clio::run::PoolQuery::Local(),
        clio::cte::replication::kReplicationPoolName,
        clio::cte::replication::kReplicationPoolId, params);
    create.Wait();
    REQUIRE(create->GetReturnCode() == 0);
  }

  clio::cte::core::Tag tag("replica_mod_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  // Seed primaries only.
  constexpr int kNumBlobs = 5;
  for (int i = 0; i < kNumBlobs; ++i) {
    PutTo(core_client, tag_id, "mod_blob_" + std::to_string(i),
          std::string(kValSize, static_cast<char>('a' + i)), 0);
  }

  // ReplicateBlob: one blob into replica 1; the replica must read back the
  // primary's bytes.
  {
    auto fut = repl.AsyncReplicateBlob(tag_id, "mod_blob_0", 1);
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
    REQUIRE(fut->bytes_copied_ == kValSize);
    std::vector<char> got;
    REQUIRE(GetFrom(core_client, tag_id, "mod_blob_0", &got, 1) == 0);
    REQUIRE(std::memcmp(got.data(), std::string(kValSize, 'a').data(),
                        kValSize) == 0);
  }

  // FlushTag: every blob in the tag gains an up-to-date replica 1.
  {
    auto fut = repl.AsyncFlushTag(tag_id, /*replica=*/1);
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
    REQUIRE(fut->blobs_replicated_ == kNumBlobs);
    for (int i = 0; i < kNumBlobs; ++i) {
      std::vector<char> got;
      REQUIRE(GetFrom(core_client, tag_id, "mod_blob_" + std::to_string(i),
                      &got, 1) == 0);
      REQUIRE(std::memcmp(got.data(),
                          std::string(kValSize, static_cast<char>('a' + i))
                              .data(),
                          kValSize) == 0);
    }
  }

  // Replica stays a snapshot until re-replicated: change a primary, replica
  // still holds old bytes; ReplicateBlob refreshes it.
  {
    const std::string v2(kValSize, 'Z');
    PutTo(core_client, tag_id, "mod_blob_1", v2, 0);
    std::vector<char> got;
    REQUIRE(GetFrom(core_client, tag_id, "mod_blob_1", &got, 1) == 0);
    REQUIRE(std::memcmp(got.data(), std::string(kValSize, 'b').data(),
                        kValSize) == 0);
    auto fut = repl.AsyncReplicateBlob(tag_id, "mod_blob_1", 1);
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
    REQUIRE(GetFrom(core_client, tag_id, "mod_blob_1", &got, 1) == 0);
    REQUIRE(std::memcmp(got.data(), v2.data(), kValSize) == 0);
  }
}

TEST_CASE("BlobReplicas - reorganizer drops the cache copy below a "
          "persistent replica",
          "[cte][replicas][cache][886]") {
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);

  clio::cte::core::Tag tag("replica_drop_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  const std::string primary_val(kValSize, 'p');
  const std::string replica_val(kValSize, 'q');
  PutTo(client, tag_id, "drop_blob", primary_val, 0);
  {
    // Persistent copy at a modest score — the organizer's drop threshold.
    clio::cte::core::Context ctx = ReplicaCtx(1);
    ctx.replica_flags_ = clio::cte::core::REPLICA_PERSISTENT;
    auto fut = client->AsyncPutBlob(tag_id, "drop_blob", 0, kValSize,
                                    replica_val.data(), /*score=*/0.5f, ctx);
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
  }

  // Rescoring the primary BELOW the persistent replica must DROP it (free
  // its blocks) rather than migrate it — a durable copy already exists.
  {
    auto fut = client->AsyncReorganizeBlob(tag_id, "drop_blob", 0.1f);
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
  }
  {
    auto sz = client->AsyncGetBlobSize(tag_id, "drop_blob");
    sz.Wait();
    REQUIRE(sz->GetReturnCode() == 0);
    REQUIRE(sz->size_ == 0);  // primary dropped
    auto rsz = client->AsyncGetBlobSize(tag_id, "drop_blob",
                                        clio::run::PoolQuery::Dynamic(), 1);
    rsz.Wait();
    REQUIRE(rsz->GetReturnCode() == 0);
    REQUIRE(rsz->size_ == kValSize);  // replica untouched
    std::vector<char> got;
    REQUIRE(GetFrom(client, tag_id, "drop_blob", &got, 1) == 0);
    REQUIRE(std::memcmp(got.data(), replica_val.data(), kValSize) == 0);
  }

  // Re-scoring the (now empty) primary is a clean no-op.
  {
    auto fut = client->AsyncReorganizeBlob(tag_id, "drop_blob", 0.9f);
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
  }
}

TEST_CASE("BlobReplicas - CachedPut/CachedGet write-through cache model",
          "[cte][replicas][replication][cache][886]") {
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);
  auto *ipc_manager = CLIO_CPU_IPC;

  // Bind the replication pool (GetOrCreatePool is idempotent; the module
  // test earlier created it with default params: num_replicas=1,
  // cache_score=1.0, replica_score=0.2).
  clio::cte::replication::Client repl(
      clio::cte::replication::kReplicationPoolId, clio::cte::core::kCtePoolId);
  {
    clio::cte::replication::ReplicationConfig params;
    params.next_pool_id_ = clio::cte::core::kCtePoolId;
    auto create = repl.AsyncCreateReplication(
        clio::run::PoolQuery::Local(),
        clio::cte::replication::kReplicationPoolName,
        clio::cte::replication::kReplicationPoolId, params);
    create.Wait();
    REQUIRE(create->GetReturnCode() == 0);
  }

  clio::cte::core::Tag tag("cache_model_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  // Stage the payload in SHM for the cached verbs.
  const std::string val(kValSize, 'C');
  ctp::ipc::FullPtr<char> put_buf = ipc_manager->AllocateBuffer(kValSize);
  REQUIRE(!put_buf.IsNull());
  std::memcpy(put_buf.ptr_, val.data(), kValSize);

  // 1. Write-through: one CachedPut updates the DRAM primary AND the fixed
  //    persistent set.
  {
    auto fut = repl.AsyncCachedPut(tag_id, "cache_blob", 0, kValSize,
                                   ctp::ipc::ShmPtr<>(put_buf.shm_));
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
    REQUIRE(fut->replicas_written_ == 1);
  }
  {
    auto psz = client->AsyncGetBlobSize(tag_id, "cache_blob");
    psz.Wait();
    REQUIRE(psz->size_ == kValSize);
    auto rsz = client->AsyncGetBlobSize(tag_id, "cache_blob",
                                        clio::run::PoolQuery::Dynamic(), 1);
    rsz.Wait();
    REQUIRE(rsz->GetReturnCode() == 0);
    REQUIRE(rsz->size_ == kValSize);
  }

  // 2. Cache hit: served from the primary.
  ctp::ipc::FullPtr<char> get_buf = ipc_manager->AllocateBuffer(kValSize);
  REQUIRE(!get_buf.IsNull());
  {
    std::memset(get_buf.ptr_, 0, kValSize);
    auto fut = repl.AsyncCachedGet(tag_id, "cache_blob", 0, kValSize,
                                   ctp::ipc::ShmPtr<>(get_buf.shm_));
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
    REQUIRE(fut->from_replica_ == 0);
    REQUIRE(std::memcmp(get_buf.ptr_, val.data(), kValSize) == 0);
  }

  // 3. The organizer drops the cache copy (primary score sinks below the
  //    persistent replica's replica_score).
  {
    auto fut = client->AsyncReorganizeBlob(tag_id, "cache_blob", 0.05f);
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
    auto psz = client->AsyncGetBlobSize(tag_id, "cache_blob");
    psz.Wait();
    REQUIRE(psz->size_ == 0);
  }

  // 4. Cache miss: served from the persistent replica, and the DRAM primary
  //    is re-populated in full.
  {
    std::memset(get_buf.ptr_, 0, kValSize);
    auto fut = repl.AsyncCachedGet(tag_id, "cache_blob", 0, kValSize,
                                   ctp::ipc::ShmPtr<>(get_buf.shm_));
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
    REQUIRE(fut->from_replica_ == 1);
    REQUIRE(fut->recached_bytes_ == kValSize);
    REQUIRE(std::memcmp(get_buf.ptr_, val.data(), kValSize) == 0);
  }

  // 5. Fast path restored: the next read is a primary hit again.
  {
    auto psz = client->AsyncGetBlobSize(tag_id, "cache_blob");
    psz.Wait();
    REQUIRE(psz->size_ == kValSize);
    std::memset(get_buf.ptr_, 0, kValSize);
    auto fut = repl.AsyncCachedGet(tag_id, "cache_blob", 0, kValSize,
                                   ctp::ipc::ShmPtr<>(get_buf.shm_));
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
    REQUIRE(fut->from_replica_ == 0);
    REQUIRE(std::memcmp(get_buf.ptr_, val.data(), kValSize) == 0);
  }

  ipc_manager->FreeBuffer(put_buf);
  ipc_manager->FreeBuffer(get_buf);
}

SIMPLE_TEST_MAIN()

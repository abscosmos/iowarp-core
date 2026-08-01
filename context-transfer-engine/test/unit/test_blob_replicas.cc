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

  BlobReplicasFixture() {
    config_path_ = chi_test_data_dir() + "/blob_replicas_config.yaml";
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
  }

  void CreateConfigFile() {
    std::ofstream config_file(config_path_);
    REQUIRE(config_file.is_open());
    config_file << R"(
# Blob replica test configuration - single 64MB DRAM tier
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

SIMPLE_TEST_MAIN()

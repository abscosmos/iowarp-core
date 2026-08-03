/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * CACHE CHIMOD TESTS (issue #886 cache/replication split)
 *
 * cache(563) -> core(512) driven by ONE plain clio::cte::core::Client at the
 * top. Verifies the cache module's contract:
 *   - WRITE-BACK: a put acks after the raw bytes land in the core's
 *     REPLICA_CACHE slot; the primary fills in later via the periodic flush
 *     sweep, and reads/sizes through the cache are correct in the window
 *     where the primary is still empty;
 *   - overwrites in one flush window COALESCE (the primary ends at the
 *     final bytes);
 *   - the zero-IPC SHM fast path serves the blob from its (untransformed,
 *     RAM-local) cache replica BEFORE the flush has materialized a primary;
 *   - a read MISS (blob written behind the cache's back) forwards down and
 *     best-effort re-populates the cache replica.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/cache/cache_client.h>

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

static constexpr clio::run::u64 kValSize = 64 * 1024;
/** Long enough to assert the pre-flush window, short enough to wait out. */
static constexpr int kFlushPeriodMs = 300;

class CacheInterposeFixture {
 public:
  std::string config_path_;
  std::string restart_log_path_;

  CacheInterposeFixture() {
    config_path_ = chi_test_data_dir() + "/cache_interpose_config.yaml";
    restart_log_path_ = chi_test_data_dir() + "/cache_interpose_restart.bin";
    Cleanup();
    CreateConfigFile();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path_.c_str(), 1);
    // Hermetic pool set: don't let ~/.clio's restart log resurrect pools
    // from earlier tests (their create-params would win over ours).
    ctp::SystemInfo::Setenv("CLIO_RESTART_LOG", restart_log_path_.c_str(), 1);

    bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    REQUIRE(success);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    success = clio::cte::core::CLIO_CTE_CLIENT_INIT();
    REQUIRE(success);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  ~CacheInterposeFixture() { Cleanup(); }

  void Cleanup() {
    if (fs::exists(config_path_)) fs::remove(config_path_);
    if (fs::exists(restart_log_path_)) fs::remove(restart_log_path_);
  }

  void CreateConfigFile() {
    std::ofstream config_file(config_path_);
    REQUIRE(config_file.is_open());
    config_file << R"(
# Cache chimod test configuration
runtime:
  num_threads: 2
  queue_depth: 1024

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
      - path: "ram::cache_interpose_dram"
        bdev_type: "ram"
        capacity_limit: "128MB"
        score: 1.0

    dpe:
      dpe_type: "max_bw"
)";
    config_file.close();
  }
};

static std::string Payload(clio::run::u64 size, char seed) {
  std::string v(size, seed);
  for (clio::run::u64 i = 0; i < size; ++i) {
    v[i] = static_cast<char>(seed + (i % 61));
  }
  return v;
}

/** Poll until the PRIMARY of tag/name reports `want` bytes via the raw core. */
static bool WaitPrimarySize(clio::cte::core::Client *core,
                            const clio::cte::core::TagId &tag_id,
                            const std::string &name, clio::run::u64 want) {
  for (int attempt = 0; attempt < 300; ++attempt) {
    auto sz = core->AsyncGetBlobSize(tag_id, name);
    sz.Wait();
    if (sz->GetReturnCode() == 0 && sz->size_ == want) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return false;
}

/** Poll until the CACHE replica of tag/name reports `want` bytes. */
static bool WaitCacheReplicaSize(clio::cte::core::Client *core,
                                 const clio::cte::core::TagId &tag_id,
                                 const std::string &name,
                                 clio::run::u64 want) {
  for (int attempt = 0; attempt < 300; ++attempt) {
    auto sz = core->AsyncGetBlobSize(tag_id, name,
                                     clio::run::PoolQuery::Dynamic(),
                                     clio::cte::core::kCacheReplica);
    sz.Wait();
    if (sz->GetReturnCode() == 0 && sz->size_ == want) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return false;
}

TEST_CASE("CacheInterpose - write-back cache over core",
          "[cte][cache][886]") {
  CacheInterposeFixture fixture;
  auto *core = CLIO_CTE_CLIENT;
  REQUIRE(core != nullptr);

  // cache(563) -> core(512), write-back with a wide-enough flush window to
  // observe the pre-flush state.
  {
    clio::cte::cache::Client cache(clio::cte::cache::kCachePoolId,
                                   clio::cte::core::kCtePoolId);
    clio::cte::cache::CacheConfig params;
    params.next_pool_id_ = clio::cte::core::kCtePoolId;
    params.flush_period_ms_ = kFlushPeriodMs;
    auto create = cache.AsyncCreateCache(
        clio::run::PoolQuery::Local(), clio::cte::cache::kCachePoolName,
        clio::cte::cache::kCachePoolId, params);
    create.Wait();
    REQUIRE(create->GetReturnCode() == 0);
  }

  // Task-path client at the top of the chain (no SHM mirror), plus a
  // mirror-attached one for the fast-path assertions.
  clio::cte::core::Client cache_io(clio::cte::cache::kCachePoolId);
  clio::cte::core::Client shm_io(clio::cte::cache::kCachePoolId);
  REQUIRE(shm_io.AttachShmCacheOf(clio::cte::core::kCtePoolId));

  clio::cte::core::Tag tag("cache_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();
  const std::string val = Payload(kValSize, 'a');

  // ======================================================================
  // 1. WRITE-BACK: the ack precedes the primary. In the pre-flush window
  //    the cache replica holds the bytes, sizes/reads through the cache are
  //    already correct, and the SHM fast path serves from the replica.
  // ======================================================================
  {
    auto put = cache_io.AsyncPutBlob(tag_id, "wb_blob", 0, kValSize,
                                     val.data());
    put.Wait();
    REQUIRE(put->GetReturnCode() == 0);

    // The cache replica has the bytes NOW (written before the ack).
    {
      auto rsz = core->AsyncGetBlobSize(tag_id, "wb_blob",
                                        clio::run::PoolQuery::Dynamic(),
                                        clio::cte::core::kCacheReplica);
      rsz.Wait();
      REQUIRE(rsz->GetReturnCode() == 0);
      REQUIRE(rsz->size_ == kValSize);
    }

    // Size through the cache answers the logical size even though the
    // primary hasn't been flushed yet.
    {
      auto sz = cache_io.AsyncGetBlobSize(tag_id, "wb_blob");
      sz.Wait();
      REQUIRE(sz->GetReturnCode() == 0);
      REQUIRE(sz->size_ == kValSize);
    }

    // Read through the cache: served from the cache replica.
    {
      std::vector<char> got(kValSize, 0);
      auto get = cache_io.AsyncGetBlob(tag_id, "wb_blob", 0, kValSize,
                                       /*flags=*/0, got.data());
      get.Wait();
      REQUIRE(get->GetReturnCode() == 0);
      REQUIRE(std::memcmp(got.data(), val.data(), kValSize) == 0);
    }

    // Zero-IPC fast path from the SERVING REPLICA: the cache copy is
    // untransformed, inline-sized, RAM and local, so the mirror serves it
    // even while the primary is still empty (issue #886 task 8).
    {
      const char *view = nullptr;
      clio::run::u64 view_size = 0, gen = 0;
      REQUIRE(shm_io.TryGetBlobViewShm(tag_id, "wb_blob", &view, &view_size,
                                       &gen));
      REQUIRE(view_size == kValSize);
      REQUIRE(std::memcmp(view, val.data(), kValSize) == 0);
      REQUIRE(shm_io.CheckBlobGenShm(tag_id, "wb_blob", gen));
    }

    // The sweep materializes the primary with the same bytes.
    REQUIRE(WaitPrimarySize(core, tag_id, "wb_blob", kValSize));
    {
      std::vector<char> got(kValSize, 0);
      auto get = core->AsyncGetBlob(tag_id, "wb_blob", 0, kValSize,
                                    /*flags=*/0, got.data());
      get.Wait();
      REQUIRE(get->GetReturnCode() == 0);
      REQUIRE(std::memcmp(got.data(), val.data(), kValSize) == 0);
    }
  }

  // ======================================================================
  // 2. COALESCING: two overwrites inside one flush window end with the
  //    LAST bytes downstream — the dirty set dedupes by blob.
  // ======================================================================
  {
    const std::string v1 = Payload(kValSize, 'b');
    const std::string v2 = Payload(kValSize, 'c');
    auto p1 = cache_io.AsyncPutBlob(tag_id, "co_blob", 0, kValSize,
                                    v1.data());
    p1.Wait();
    REQUIRE(p1->GetReturnCode() == 0);
    auto p2 = cache_io.AsyncPutBlob(tag_id, "co_blob", 0, kValSize,
                                    v2.data());
    p2.Wait();
    REQUIRE(p2->GetReturnCode() == 0);

    REQUIRE(WaitPrimarySize(core, tag_id, "co_blob", kValSize));
    // Give a possible in-flight first flush time to be superseded, then
    // confirm the primary settles at the SECOND payload.
    std::this_thread::sleep_for(std::chrono::milliseconds(2 * kFlushPeriodMs));
    std::vector<char> got(kValSize, 0);
    auto get = core->AsyncGetBlob(tag_id, "co_blob", 0, kValSize,
                                  /*flags=*/0, got.data());
    get.Wait();
    REQUIRE(get->GetReturnCode() == 0);
    REQUIRE(std::memcmp(got.data(), v2.data(), kValSize) == 0);
  }

  // ======================================================================
  // 3. MISS + REPOPULATE: a blob written straight to the core (behind the
  //    cache's back) reads correctly through the cache, and the read
  //    re-populates the cache replica.
  // ======================================================================
  {
    const std::string vm = Payload(kValSize, 'd');
    auto put = core->AsyncPutBlob(tag_id, "miss_blob", 0, kValSize,
                                  vm.data());
    put.Wait();
    REQUIRE(put->GetReturnCode() == 0);

    std::vector<char> got(kValSize, 0);
    auto get = cache_io.AsyncGetBlob(tag_id, "miss_blob", 0, kValSize,
                                     /*flags=*/0, got.data());
    get.Wait();
    REQUIRE(get->GetReturnCode() == 0);
    REQUIRE(std::memcmp(got.data(), vm.data(), kValSize) == 0);

    // The miss re-populated the REPLICA_CACHE slot.
    REQUIRE(WaitCacheReplicaSize(core, tag_id, "miss_blob", kValSize));
  }
}

SIMPLE_TEST_MAIN()

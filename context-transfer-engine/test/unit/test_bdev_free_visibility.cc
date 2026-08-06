/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

// Freed space must be visible to placement immediately (issue #919).
//
// The CTE keeps two copies of every target's remaining_space_:
//   * registered_targets_[id] -- the CANONICAL entry. PutBlob debits it and
//     Free credits it, lock-free, on the data path.
//   * target_list_[i]         -- a contiguous mirror used for O(N) iteration.
//     Its remaining_space_ is refreshed ONLY by the periodic StatTargets sweep
//     (performance.stat_targets_period_ms, 5 s by default).
//
// ExtendBlob used to hand the DPE the mirror's copy, so placement decided
// against free space that could be a whole tick out of date. The failure is not
// symmetric: a StatTargets tick that lands while a tier is FULL pins the mirror
// at ~0, and every later put is then rejected for "no target has space" even
// after the blobs holding that space have been deleted -- until the next tick.
//
// That is the race behind the cte_bdev_fragmentation flake: its fill phase
// drives the tier to capacity, and on a slow runner a tick lands inside that
// window, so the large puts that follow the deletes all EIO.
//
// This test drives the same window deliberately instead of waiting to get
// unlucky: fill the tier, force a one-shot StatTargets (mirror := 0), delete a
// slice, then immediately put a blob that only fits in the just-freed space.
// The cycle repeats -- a stray periodic tick can rescue any single iteration by
// refreshing the mirror, but rescuing every iteration is not plausible.

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace {

const char *kTargetName = "free_vis_target";
// Small enough that "full" is a few thousand puts away, matching the geometry
// of the fragmentation test this regression was extracted from.
constexpr clio::run::u64 kTargetSize = 32ULL * 1024 * 1024;

class Fixture {
 public:
  bool initialized_ = false;
  Fixture() {
    bool ok = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    REQUIRE(ok);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    ok = clio::cte::core::CLIO_CTE_CLIENT_INIT();
    REQUIRE(ok);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    auto *cte = CLIO_CTE_CLIENT;
    clio::run::PoolId bdev_pool_id(932, 0);
    clio::run::bdev::Client bdev_client(bdev_pool_id);
    auto create = bdev_client.AsyncCreate(
        clio::run::PoolQuery::Dynamic(), kTargetName, bdev_pool_id,
        clio::run::bdev::BdevType::kRam, kTargetSize);
    create.Wait();
    auto reg = cte->AsyncRegisterTarget(
        kTargetName, clio::run::bdev::BdevType::kRam, kTargetSize,
        clio::run::PoolQuery::Local(), bdev_pool_id);
    reg.Wait();
    REQUIRE(reg->GetReturnCode() == 0);
    initialized_ = true;
  }
};

Fixture *g_fixture = nullptr;

constexpr size_t kSmall = 8 * 1024;
constexpr size_t kLarge = 1024 * 1024;

// Put kSmall blobs named "<prefix>_<i>" until the tier refuses one. Returns how
// many landed, which is what drives the allocator to its capacity watermark.
int FillToCapacity(clio::cte::core::Client *cte,
                   const clio::cte::core::TagId &tag_id,
                   const std::string &prefix, int start_index) {
  auto sb = CLIO_IPC->AllocateBuffer(kSmall);
  REQUIRE(!sb.IsNull());
  std::memset(sb.ptr_, 'x', kSmall);
  int filled = 0;
  const int kMaxSmall = 200000;
  for (int i = 0; i < kMaxSmall; ++i) {
    const std::string nm = prefix + std::to_string(start_index + i);
    auto p = cte->AsyncPutBlob(tag_id, nm, 0, kSmall,
                               ctp::ipc::ShmPtr<>(sb.shm_));
    p.Wait();
    if (p->GetReturnCode() != 0) break;  // tier full
    ++filled;
  }
  CLIO_IPC->FreeBuffer(sb);
  return filled;
}

}  // namespace

TEST_CASE("Deleted space is visible to placement without a stats tick (#919)",
          "[cte][bdev][placement][919]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);
  auto *cte = CLIO_CTE_CLIENT;
  clio::cte::core::Tag tag("free_vis_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();

  std::vector<char> lbuf(kLarge);
  for (size_t j = 0; j < kLarge; ++j) {
    lbuf[j] = static_cast<char>((j * 13) % 251);
  }

  // Freeing this many kSmall blobs yields 4 MiB -- comfortably more than the
  // single 1 MiB blob we then place, so a pass cannot be luck about sizing.
  const int kFreeCount = 512;
  const int kCycles = 4;
  int next_index = 0;

  for (int cycle = 0; cycle < kCycles; ++cycle) {
    // 1. Drive the tier to its capacity watermark.
    const std::string prefix = "v" + std::to_string(cycle) + "_";
    const int filled = FillToCapacity(cte, tag_id, prefix, next_index);
    REQUIRE(filled >= kFreeCount);

    // 2. Force the mirror to snapshot the tier while it is FULL. This is the
    //    state a periodic tick lands in by chance on a slow runner; taking it
    //    deliberately is what makes the regression reproducible.
    auto stat = cte->AsyncStatTargets(clio::run::PoolQuery::Local());
    stat.Wait();

    // 3. Free 4 MiB. Only the canonical entry learns about it.
    for (int i = 0; i < kFreeCount; ++i) {
      const std::string nm = prefix + std::to_string(next_index + i);
      auto d = cte->AsyncDelBlob(tag_id, nm);
      d.Wait();
      REQUIRE(d->GetReturnCode() == 0);
    }

    // 4. A 1 MiB blob now fits only in the space freed by step 3. Placement
    //    reading the stale mirror sees a full tier and refuses it.
    auto lb = CLIO_IPC->AllocateBuffer(kLarge);
    REQUIRE(!lb.IsNull());
    std::memcpy(lb.ptr_, lbuf.data(), kLarge);
    const std::string lname = "V_" + std::to_string(cycle);
    auto p = cte->AsyncPutBlob(tag_id, lname, 0, kLarge,
                               ctp::ipc::ShmPtr<>(lb.shm_));
    p.Wait();
    const int put_rc = static_cast<int>(p->GetReturnCode());
    CLIO_IPC->FreeBuffer(lb);
    std::printf("[#919] cycle %d: filled %d, freed %d, 1MiB put rc=%d\n", cycle,
                filled, kFreeCount, put_rc);
    REQUIRE(put_rc == 0);

    // Read back: the blob was assembled from the freed small extents, so a
    // correct placement decision must also produce correct data.
    auto rd = CLIO_IPC->AllocateBuffer(kLarge);
    REQUIRE(!rd.IsNull());
    std::memset(rd.ptr_, 0, kLarge);
    auto g = cte->AsyncGetBlob(tag_id, lname, 0, kLarge, /*flags=*/0,
                               ctp::ipc::ShmPtr<>(rd.shm_));
    g.Wait();
    REQUIRE(g->GetReturnCode() == 0);
    REQUIRE(std::memcmp(rd.ptr_, lbuf.data(), kLarge) == 0);
    CLIO_IPC->FreeBuffer(rd);

    // 5. Hand the tier back so the next cycle starts from a clean slate.
    auto dl = cte->AsyncDelBlob(tag_id, lname);
    dl.Wait();
    REQUIRE(dl->GetReturnCode() == 0);
    for (int i = kFreeCount; i < filled; ++i) {
      const std::string nm = prefix + std::to_string(next_index + i);
      auto d = cte->AsyncDelBlob(tag_id, nm);
      d.Wait();
      REQUIRE(d->GetReturnCode() == 0);
    }
    next_index += filled;
  }
}

int main(int argc, char **argv) {
  g_fixture = new Fixture();
  std::string filter = (argc > 1) ? argv[1] : "";
  int rc = SimpleTest::run_all_tests(filter);
  delete g_fixture;
  g_fixture = nullptr;
  return rc;
}

/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * issue #785 — a stalled worker must not strand blocked / in-flight tasks.
 *
 * #781 made the runtime route NEW work around a worker wedged by a
 * non-yielding task. Work already committed to that worker is still stranded:
 * its lane backlog, its blocked / periodic / retry queues, its completion event
 * queue, and — worst — tasks suspended on co_await, which live in no queue at
 * all and are reachable only through the subtask future's parent handle plus the
 * raw EventQueue()/Lane() addresses cached in their RunContext.
 *
 * These tests are the deterministic repro for that. They deliberately use only
 * EXISTING module primitives so they reproduce the bug against unmodified code:
 *
 *   MOD_NAME::Custom   with spin_us_  -> a NON-YIELDING busy spin (the bad task)
 *   MOD_NAME::WaitTest with depth > 1 -> recursively self-sends a subtask and
 *                                        CLIO_CO_AWAITs it (the blocked task)
 *
 * A WaitTest parent suspended on co_await has its completion routed to whichever
 * worker it first ran on. If that worker is wedged inside a Custom spin, the
 * subtask can complete on a perfectly healthy worker and the parent still never
 * wakes.
 *
 * EXPECTED STATE: these FAIL on this branch today. They are the acceptance
 * criteria for the migration work, not a regression guard for it. Every timed
 * wait is a bounded poll on Future::IsComplete() — a stranded task makes a test
 * fail, never hang.
 */

#include "simple_test.h"

#include <atomic>
#include <cstdlib>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/pool_query.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/task.h>
#include <clio_runtime/types.h>

#include <clio_runtime/MOD_NAME/MOD_NAME_client.h>
#include <clio_runtime/MOD_NAME/MOD_NAME_tasks.h>

#include <clio_runtime/admin/admin_client.h>
#include <clio_runtime/admin/admin_tasks.h>

namespace {

/** Busy-spin length of the "bad" task, microseconds. Long enough that a test
 *  bound well under it cannot be satisfied by simply waiting the spin out. */
constexpr clio::run::u32 kSpinUs = 12'000'000;  // 12 s

/** How long a chained task is allowed to take while spinners are running.
 *
 *  DERIVED FROM THE RESCUE MECHANISM, not tuned until green. Rescue is not
 *  instant: LoadBalance declares a stall only after kStallThresholdSec (1 s)
 *  and runs on a 500 ms cadence, so each rescue costs up to ~1.5 s to trigger.
 *  A depth-D chain can hit D such stalls in sequence — each hop may be queued
 *  behind a different heavy task — so the floor is
 *
 *      kChainDepth * (kStallThresholdSec + monitor_period) = 2 * 1.5 s = 3 s
 *
 *  6 s therefore leaves 2x margin for scheduling variance while still being
 *  HALF the spin duration, so passing genuinely means "does not track the bad
 *  task's runtime".
 *
 *  Measured while calibrating: at depth 3 the floor is 4.5 s, which left almost
 *  no margin — a 3 s bound passed 2/5 runs, 4 s passed 1/5, 6 s passed 2/5 and
 *  7 s passed 3/3, every failure being rescue latency rather than stranding.
 *  The fix is fewer SEQUENTIAL rescue cycles (depth 2), not a looser bound:
 *  loosening it toward the spin duration would quietly destroy what the test
 *  asserts. */
constexpr int kChainDeadlineMs = 6000;

/** Generous bound used only to distinguish "slow" from "lost". */
constexpr int kDropDeadlineMs = 30000;

/** Number of non-yielding spinners.
 *
 * This MUST stay below the number of general-purpose (I/O) workers, or the test
 * measures saturation instead of stranding: with every compute worker spinning,
 * nothing runs and the result says nothing about whether blocked tasks can be
 * rescued. The control group below enforces that empirically.
 *
 * Note the default runtime has only ONE general-purpose worker: num_threads=4
 * divides into 1 scheduler + 1 I/O + net_send + net_recv, so a single bad task
 * wedges all compute. These tests therefore need CLIO_NUM_THREADS raised to
 * leave spare compute workers. Overridable via CLIO_STALL_TEST_SPINNERS. */
int NumSpinners() {
  if (const char *env = std::getenv("CLIO_STALL_TEST_SPINNERS")) {
    int n = std::atoi(env);
    if (n > 0) return n;
  }
  return 2;
}
constexpr int kNumChained = 8;
constexpr clio::run::u32 kChainDepth = 2;

constexpr clio::run::PoolId kStallPoolId = clio::run::PoolId(9100, 0);

bool g_initialized = false;

class StallFixture {
 public:
  StallFixture() {
    if (!g_initialized) {
      // The default num_threads=4 divides into 1 scheduler + 1 I/O + net_send +
      // net_recv, leaving exactly ONE general-purpose worker — so a single
      // non-yielding task wedges all compute and every run degenerates into
      // saturation. Raise it before CLIO_INIT reads the config so there is
      // provably spare compute capacity for the control group to land on.
      // setenv with overwrite=0 keeps an externally-supplied value.
      setenv("CLIO_NUM_THREADS", "8", 0);
      bool success =
          clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
      if (success) {
        g_initialized = true;
        SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
        std::this_thread::sleep_for(500ms);
      }
    }
  }

  /** Create the MOD_NAME container and return the pool id the runtime assigned
   *  (AsyncCreate remaps it, so callers must use the returned value). */
  bool createContainer(clio::run::PoolId pool_id, clio::run::PoolId &out) {
    clio::run::MOD_NAME::Client client(pool_id);
    auto create_task = client.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                          "stall_migration_pool", pool_id);
    create_task.Wait();
    if (create_task->return_code_ != 0) {
      return false;
    }
    out = create_task->new_pool_id_;
    std::this_thread::sleep_for(100ms);
    return true;
  }
};

/** Poll a set of futures until all complete or the deadline expires.
 *  @return number still incomplete when we stopped. */
template <typename FutureT>
size_t WaitAllBounded(std::vector<FutureT> &futures, int deadline_ms) {
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(deadline_ms);
  for (;;) {
    size_t pending = 0;
    for (auto &f : futures) {
      if (!f.IsComplete()) {
        ++pending;
      }
    }
    if (pending == 0) {
      return 0;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return pending;
    }
    std::this_thread::sleep_for(5ms);
  }
}

}  // namespace

//==============================================================================
// TEST 1 — a wedged worker must not stall tasks blocked on co_await
//==============================================================================

TEST_CASE("stall_does_not_strand_blocked_tasks", "[stall][migration]") {
  StallFixture fixture;
  clio::run::PoolId pool_id;
  REQUIRE(fixture.createContainer(kStallPoolId, pool_id));

  clio::run::MOD_NAME::Client client(pool_id);

  SECTION("chained co_await tasks complete while spinners wedge workers") {
    // Wedge the pool with non-yielding spinners. These are submitted and NOT
    // waited on — they own their workers for kSpinUs.
    std::vector<clio::run::Future<clio::run::MOD_NAME::CustomTask>> spinners;
    const int num_spinners = NumSpinners();
    spinners.reserve(num_spinners);
    for (int i = 0; i < num_spinners; ++i) {
      spinners.push_back(client.AsyncCustom(clio::run::PoolQuery::Local(), "s",
                                            0, kSpinUs));
    }

    // Let the spinners actually get picked up and enter their spin loops.
    std::this_thread::sleep_for(300ms);

    // Now submit chained tasks. Each recursively self-sends a subtask and
    // co_awaits it, so each parent parks with its completion routed to the
    // worker it first ran on.
    std::vector<clio::run::Future<clio::run::MOD_NAME::WaitTestTask>> chained;
    chained.reserve(kNumChained);
    for (int i = 0; i < kNumChained; ++i) {
      chained.push_back(client.AsyncWaitTest(clio::run::PoolQuery::Local(),
                                             kChainDepth,
                                             static_cast<clio::run::u32>(i)));
    }

    // CONTROL GROUP. Flat, dependency-free quick tasks submitted at the same
    // moment. #781 already routes these around a stalled worker, so they must
    // sail through. Without this control the test cannot tell the bug
    // ("dependency chains are stranded behind a wedged worker") apart from mere
    // saturation ("every worker is busy, nothing runs"). If the control fails
    // too, the spinner count is simply swamping the pool and the run proves
    // nothing about stranding.
    std::vector<clio::run::Future<clio::run::MOD_NAME::CustomTask>> control;
    control.reserve(kNumChained);
    for (int i = 0; i < kNumChained; ++i) {
      control.push_back(
          client.AsyncCustom(clio::run::PoolQuery::Local(), "c", 0, 10));
    }

    size_t control_pending = WaitAllBounded(control, kChainDeadlineMs);
    size_t stranded = WaitAllBounded(chained, kChainDeadlineMs);
    INFO("after " + std::to_string(kChainDeadlineMs) +
         "ms — control (flat) pending: " + std::to_string(control_pending) +
         ", chained (co_await) pending: " + std::to_string(stranded));

    // Runtime liveness: if this fails, the pool is saturated and the chained
    // result below is not evidence of anything.
    REQUIRE(control_pending == 0);

    // The acceptance criterion for #785: chained-task completion must not track
    // the spin duration. Any non-zero count here is a task stranded behind a
    // wedged worker while the runtime is demonstrably still alive.
    REQUIRE(stranded == 0);

    // Drain the spinners so the next test starts from a quiet runtime.
    WaitAllBounded(spinners, kDropDeadlineMs);
  }
}

//==============================================================================
// TEST 2 — a worker wedged AFTER a task parks must not strand that task
//
// This is the category the lane rescue does NOT cover. Test 1's chained tasks
// were stuck because their subtasks sat in a stalled worker's LANE, so moving
// the lane freed them. Here the parents are already parked on a co_await before
// any spinner is submitted, so what is stranded is the completion path: the
// parent is in no queue at all, and its wakeup is routed to the event queue of
// whichever worker it first ran on. Wedge that worker and the event has nowhere
// to land, however healthy the rest of the pool is.
//==============================================================================

TEST_CASE("stall_after_park_does_not_strand_completion", "[stall][event]") {
  StallFixture fixture;
  clio::run::PoolId pool_id;
  REQUIRE(fixture.createContainer(kStallPoolId, pool_id));

  clio::run::MOD_NAME::Client client(pool_id);

  SECTION("parents parked on a subtask still wake when their worker wedges") {
    // CONCURRENCY BUDGET. Every simultaneously-spinning task occupies one
    // general-purpose worker, and there are only (num_threads - 4) of those —
    // 4 at the fixture's CLIO_NUM_THREADS=8. Each parked parent spawns a child
    // that SPINS, so parents count against the budget too. Exceed it and the
    // run degenerates into saturation, which is exactly what the control group
    // catches. Budget here: 2 children + 1 spinner = 3 busy, 1 left for the
    // control to land on.
    constexpr int kNumParked = 2;
    constexpr int kParkSpinners = 1;
    constexpr clio::run::u32 kParkSpinUs = 2'000'000;  // 2 s

    std::vector<clio::run::Future<clio::run::MOD_NAME::CustomTask>> parked;
    parked.reserve(kNumParked);
    for (int i = 0; i < kNumParked; ++i) {
      parked.push_back(client.AsyncCustom(clio::run::PoolQuery::Local(), "p", 0,
                                          kParkSpinUs, /*chain_depth=*/1));
    }

    // Let every parent actually reach its co_await and park.
    std::this_thread::sleep_for(500ms);

    // NOW wedge workers. Any parent parked on one of these has its completion
    // routed to a worker that will not drain its event queue for kSpinUs.
    std::vector<clio::run::Future<clio::run::MOD_NAME::CustomTask>> spinners;
    spinners.reserve(kParkSpinners);
    for (int i = 0; i < kParkSpinners; ++i) {
      spinners.push_back(
          client.AsyncCustom(clio::run::PoolQuery::Local(), "s", 0, kSpinUs));
    }

    // Same liveness control as test 1.
    std::vector<clio::run::Future<clio::run::MOD_NAME::CustomTask>> control;
    control.reserve(kNumParked);
    for (int i = 0; i < kNumParked; ++i) {
      control.push_back(
          client.AsyncCustom(clio::run::PoolQuery::Local(), "c", 0, 10));
    }
    size_t control_pending = WaitAllBounded(control, kChainDeadlineMs);

    // Budget: the child's own spin, plus slack for the rescue to notice and act
    // (LoadBalance runs on a 500 ms cadence). Still far below kSpinUs, so a
    // parent that simply waits out the spinner cannot pass this.
    const int park_deadline_ms =
        static_cast<int>(kParkSpinUs / 1000) + kChainDeadlineMs;
    size_t stranded = WaitAllBounded(parked, park_deadline_ms);

    INFO("after " + std::to_string(park_deadline_ms) +
         "ms — control pending: " + std::to_string(control_pending) +
         ", parked-then-wedged pending: " + std::to_string(stranded));

    REQUIRE(control_pending == 0);
    REQUIRE(stranded == 0);

    WaitAllBounded(spinners, kDropDeadlineMs);
  }
}

//==============================================================================
// TEST 3 — no task may be DROPPED, however slow the runtime gets
//==============================================================================

TEST_CASE("no_tasks_dropped_under_stall_pressure", "[stall][drop]") {
  StallFixture fixture;
  clio::run::PoolId pool_id;
  REQUIRE(fixture.createContainer(kStallPoolId, pool_id));

  clio::run::MOD_NAME::Client client(pool_id);

  SECTION("every submitted task eventually completes") {
    // Deliberately generous deadline. This test does not care about latency —
    // it distinguishes "slow" from "lost". A task that never completes even
    // when given far longer than the spin duration was dropped, not delayed.
    std::vector<clio::run::Future<clio::run::MOD_NAME::CustomTask>> spinners;
    std::vector<clio::run::Future<clio::run::MOD_NAME::CustomTask>> quick;
    std::vector<clio::run::Future<clio::run::MOD_NAME::WaitTestTask>> chained;

    for (int i = 0; i < 4; ++i) {
      spinners.push_back(client.AsyncCustom(clio::run::PoolQuery::Local(), "s",
                                            0, kSpinUs / 4));
    }
    for (int i = 0; i < 64; ++i) {
      quick.push_back(
          client.AsyncCustom(clio::run::PoolQuery::Local(), "q", 0, 10));
    }
    for (int i = 0; i < 8; ++i) {
      chained.push_back(client.AsyncWaitTest(clio::run::PoolQuery::Local(),
                                             kChainDepth,
                                             static_cast<clio::run::u32>(i)));
    }

    size_t lost_quick = WaitAllBounded(quick, kDropDeadlineMs);
    size_t lost_chained = WaitAllBounded(chained, kDropDeadlineMs);
    size_t lost_spinners = WaitAllBounded(spinners, kDropDeadlineMs);

    INFO("dropped: quick=" + std::to_string(lost_quick) +
         " chained=" + std::to_string(lost_chained) +
         " spinners=" + std::to_string(lost_spinners));

    REQUIRE(lost_quick == 0);
    REQUIRE(lost_chained == 0);
    REQUIRE(lost_spinners == 0);
  }
}

//==============================================================================
// MAIN TEST RUNNER
//==============================================================================

SIMPLE_TEST_MAIN()

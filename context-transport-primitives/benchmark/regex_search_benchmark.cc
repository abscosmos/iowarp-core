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
 * RegexSearchEngine Search/mutate benchmark (issue #919).
 *
 * Measures the cost of making Search point-in-time consistent. The optimistic
 * design retries a whole query when a mutation lands mid-flight, and falls back
 * to one fully-locked pass after repeated losses, so there are three things to
 * watch and they pull against each other:
 *
 *   1. UNCONTENDED SEARCH -- the tax paid by every query that never races. Should
 *      be ~free: one relaxed atomic load and one compare.
 *   2. CONTENDED SEARCH   -- retry amplification. Each retry repeats the
 *      candidate scan AND the regex pass, so the worst case is
 *      kOptimisticAttempts wasted passes plus a locked pass.
 *   3. WRITER PROGRESS    -- the regression guard for #680, which moved the regex
 *      pass OUT of the lock precisely because 3 tight-loop searchers vs 8
 *      writers hung the parallel test. The locked fallback puts it back, rarely.
 *      If writer throughput collapses under search load, that is the fix
 *      reintroducing the bug it must not.
 *
 * The rename workload matters specifically: renames are what make an optimistic
 * pass fail (an insert or delete of a NON-matching key does not disturb a
 * query's result, but it does bump the version, so it costs a retry anyway --
 * that conservatism is a deliberate part of the design and this measures its
 * price).
 *
 * Usage:
 *   regex_search_benchmark [keys] [seconds] [searchers] [writers] [mode]
 *     keys       entries in the index          (default 5000)
 *     seconds    duration per phase            (default 3)
 *     searchers  concurrent Search threads     (default 3)
 *     writers    concurrent mutator threads    (default 8)
 *     mode       rename | insert_delete | mixed  (default rename)
 *
 * The searcher/writer defaults reproduce the #680 starvation shape on purpose.
 */

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "clio_ctp/search/regex_search_engine.h"

using ctp::search::RegexSearchEngine;
using Clock = std::chrono::steady_clock;

namespace {

struct Stats {
  std::atomic<uint64_t> ops{0};
  std::atomic<uint64_t> nanos{0};
  std::atomic<uint64_t> results{0};
};

std::string KeyName(const char *side, int i) {
  return std::string("/bench/dir/") + side + std::to_string(i) + ".dat";
}

// Percentile-free summary: this benchmark is about throughput and starvation,
// and a mean plus a total is enough to see both. Latency percentiles live in
// lock_latency_benchmark.
void Report(const char *label, const Stats &s, double seconds) {
  const uint64_t ops = s.ops.load();
  if (ops == 0) {
    std::printf("  %-22s no completed operations\n", label);
    return;
  }
  const double per_sec = static_cast<double>(ops) / seconds;
  const double mean_us =
      static_cast<double>(s.nanos.load()) / static_cast<double>(ops) / 1000.0;
  std::printf("  %-22s %10.0f ops/s   mean %8.2f us   (%llu ops)\n", label,
              per_sec, mean_us,
              static_cast<unsigned long long>(ops));
}

}  // namespace

int main(int argc, char **argv) {
  const int num_keys = (argc > 1) ? std::atoi(argv[1]) : 5000;
  const int seconds = (argc > 2) ? std::atoi(argv[2]) : 3;
  const int num_searchers = (argc > 3) ? std::atoi(argv[3]) : 3;
  const int num_writers = (argc > 4) ? std::atoi(argv[4]) : 8;
  const std::string mode = (argc > 5) ? argv[5] : "rename";

  std::printf(
      "RegexSearchEngine benchmark: keys=%d seconds=%d searchers=%d writers=%d "
      "mode=%s\n",
      num_keys, seconds, num_searchers, num_writers, mode.c_str());

  RegexSearchEngine<int> eng;
  for (int i = 0; i < num_keys; ++i) eng.Insert(KeyName("a", i), i);

  // Directory-listing shape: the pattern readdir actually issues, which matches
  // every key in the index and so exercises the full candidate scan.
  const std::string kPattern = "^/bench/dir/[^/]+$";

  // ---- Phase 1: uncontended search -------------------------------------
  {
    Stats s;
    const auto deadline = Clock::now() + std::chrono::seconds(seconds);
    while (Clock::now() < deadline) {
      const auto t0 = Clock::now();
      auto res = eng.Search(kPattern);
      const auto t1 = Clock::now();
      s.nanos.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
          std::memory_order_relaxed);
      s.results.fetch_add(res.keys().size(), std::memory_order_relaxed);
      s.ops.fetch_add(1, std::memory_order_relaxed);
    }
    std::printf("\nPhase 1 -- search alone (no writers)\n");
    Report("search", s, seconds);
  }

  // ---- Phase 2: searchers + writers ------------------------------------
  {
    Stats search_stats;
    Stats write_stats;
    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;

    for (int t = 0; t < num_searchers; ++t) {
      threads.emplace_back([&]() {
        while (!stop.load(std::memory_order_relaxed)) {
          const auto t0 = Clock::now();
          auto res = eng.Search(kPattern);
          const auto t1 = Clock::now();
          search_stats.nanos.fetch_add(
              std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
                  .count(),
              std::memory_order_relaxed);
          search_stats.results.fetch_add(res.keys().size(),
                                         std::memory_order_relaxed);
          search_stats.ops.fetch_add(1, std::memory_order_relaxed);
        }
      });
    }

    for (int t = 0; t < num_writers; ++t) {
      threads.emplace_back([&, t]() {
        // Each writer owns a disjoint stripe so writers never fight each other
        // over the same key -- the contention under test is search-vs-write.
        int round = 0;
        while (!stop.load(std::memory_order_relaxed)) {
          for (int i = t; i < num_keys && !stop.load(std::memory_order_relaxed);
               i += num_writers) {
            const auto t0 = Clock::now();
            if (mode == "insert_delete") {
              eng.Delete(KeyName("a", i));
              eng.Insert(KeyName("a", i), i);
            } else if (mode == "mixed" && (i & 1)) {
              eng.Delete(KeyName("a", i));
              eng.Insert(KeyName("a", i), i);
            } else {
              const char *from = (round % 2 == 0) ? "a" : "b";
              const char *to = (round % 2 == 0) ? "b" : "a";
              eng.Rename(KeyName(from, i), KeyName(to, i));
            }
            const auto t1 = Clock::now();
            write_stats.nanos.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
                    .count(),
                std::memory_order_relaxed);
            write_stats.ops.fetch_add(1, std::memory_order_relaxed);
          }
          ++round;
        }
      });
    }

    std::this_thread::sleep_for(std::chrono::seconds(seconds));
    stop.store(true, std::memory_order_relaxed);
    for (auto &th : threads) th.join();

    std::printf("\nPhase 2 -- %d searchers vs %d writers (%s)\n", num_searchers,
                num_writers, mode.c_str());
    Report("search (contended)", search_stats, seconds);
    Report("mutate", write_stats, seconds);

    // Every listing must still see the whole directory -- a throughput win that
    // loses entries is not a win. Renames keep the population constant.
    const uint64_t ops = search_stats.ops.load();
    if (ops > 0 && mode == "rename") {
      const double mean_results =
          static_cast<double>(search_stats.results.load()) /
          static_cast<double>(ops);
      std::printf("  %-22s %10.2f  (expected %d; any shortfall is a dropped\n"
                  "  %-22s              entry, i.e. the #919 bug)\n",
                  "mean entries/listing", mean_results, num_keys, "");
    }
  }

  std::printf("\n");
  return 0;
}

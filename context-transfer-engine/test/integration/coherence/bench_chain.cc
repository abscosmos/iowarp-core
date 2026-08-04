/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * IOR-like file-per-process chain benchmark (issue #886 performance).
 *
 * Every node runs this binary (rank = NODE_ID-1). Each rank writes its OWN
 * tag's blobs (file-per-process), then reads them back (IOR read-verify),
 * then reads its neighbor's (cross-node phase). Blob-based barriers between
 * phases. The pool under test comes from CLIO_CTE_POOL (the chain top or
 * the core directly), so one binary measures both configurations.
 *
 * Output (stderr, machine-parsable):
 *   BENCH <phase> rank=<r> mb=<total> secs=<s> mbps=<rate>
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kNumNodes = 4;
constexpr int kBlobsPerRank = 64;
constexpr clio::run::u64 kBlobSize = 1ULL * 1024 * 1024;  // 1 MiB pages

int Rank() {
  const char *env = std::getenv("NODE_ID");
  return env ? std::atoi(env) - 1 : 0;
}

clio::cte::core::Client *Cte() { return CLIO_CTE_CLIENT; }

int g_barrier_epoch = 0;
bool Barrier(const clio::cte::core::TagId &tag_id) {
  const int epoch = g_barrier_epoch++;
  char one = 1;
  {
    std::string mine = "bar_" + std::to_string(epoch) + "_" +
                       std::to_string(Rank());
    auto put = Cte()->AsyncPutBlob(tag_id, mine, 0, 1, &one);
    put.Wait();
    if (put->GetReturnCode() != 0) return false;
  }
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(180);
  for (int r = 0; r < kNumNodes; ++r) {
    std::string name = "bar_" + std::to_string(epoch) + "_" + std::to_string(r);
    while (true) {
      auto sz = Cte()->AsyncGetBlobSize(tag_id, name);
      sz.Wait();
      if (sz->GetReturnCode() == 0 && sz->size_ == 1) break;
      if (std::chrono::steady_clock::now() > deadline) return false;
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
  return true;
}

double Mbps(clio::run::u64 bytes, double secs) {
  return secs > 0 ? (bytes / (1024.0 * 1024.0)) / secs : 0.0;
}

}  // namespace

int main(int, char **) {
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    fprintf(stderr, "CLIO_INIT failed\n");
    return 2;
  }
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    fprintf(stderr, "CTE client init failed\n");
    return 2;
  }
  const int rank = Rank();
  fprintf(stderr, "bench: rank %d ready\n", rank);

  clio::cte::core::Tag bar_tag("bench_barrier");
  const clio::cte::core::TagId bar_id = bar_tag.GetTagId();

  // File-per-process: one tag per rank, kBlobsPerRank 1MiB page blobs.
  clio::cte::core::Tag my_tag("bench_file_" + std::to_string(rank));
  const clio::cte::core::TagId my_id = my_tag.GetTagId();
  const int peer = (rank + 1) % kNumNodes;
  clio::cte::core::Tag peer_tag("bench_file_" + std::to_string(peer));
  const clio::cte::core::TagId peer_id = peer_tag.GetTagId();

  std::vector<char> buf(kBlobSize);
  const clio::run::u64 total = kBlobsPerRank * kBlobSize;

  if (!Barrier(bar_id)) return 3;

  // ---- WRITE own file ----
  {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kBlobsPerRank; ++i) {
      std::memset(buf.data(), 'a' + ((rank + i) % 26), kBlobSize);
      auto put = Cte()->AsyncPutBlob(my_id, "page_" + std::to_string(i), 0,
                                     kBlobSize, buf.data());
      put.Wait();
      if (put->GetReturnCode() != 0) {
        fprintf(stderr, "BENCH write FAILED rank=%d blob=%d rc=%u\n", rank, i,
                put->GetReturnCode());
        return 4;
      }
    }
    double s = std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "BENCH write rank=%d mb=%llu secs=%.3f mbps=%.1f\n", rank,
            (unsigned long long)(total >> 20), s, Mbps(total, s));
  }
  if (!Barrier(bar_id)) return 3;

  // ---- READ own file (IOR read-verify) ----
  {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kBlobsPerRank; ++i) {
      auto get = Cte()->AsyncGetBlob(my_id, "page_" + std::to_string(i), 0,
                                     kBlobSize, /*flags=*/0, buf.data());
      get.Wait();
      if (get->GetReturnCode() != 0 ||
          buf[0] != static_cast<char>('a' + ((rank + i) % 26))) {
        fprintf(stderr, "BENCH read_own FAILED rank=%d blob=%d rc=%u\n", rank,
                i, get->GetReturnCode());
        return 5;
      }
    }
    double s = std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "BENCH read_own rank=%d mb=%llu secs=%.3f mbps=%.1f\n",
            rank, (unsigned long long)(total >> 20), s, Mbps(total, s));
  }
  if (!Barrier(bar_id)) return 3;

  // ---- READ own file AGAIN (steady-state cached reads) ----
  {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kBlobsPerRank; ++i) {
      auto get = Cte()->AsyncGetBlob(my_id, "page_" + std::to_string(i), 0,
                                     kBlobSize, /*flags=*/0, buf.data());
      get.Wait();
      if (get->GetReturnCode() != 0) return 5;
    }
    double s = std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "BENCH read_own2 rank=%d mb=%llu secs=%.3f mbps=%.1f\n",
            rank, (unsigned long long)(total >> 20), s, Mbps(total, s));
  }
  if (!Barrier(bar_id)) return 3;

  // ---- READ peer's file (cross-node) ----
  {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kBlobsPerRank; ++i) {
      auto get = Cte()->AsyncGetBlob(peer_id, "page_" + std::to_string(i), 0,
                                     kBlobSize, /*flags=*/0, buf.data());
      get.Wait();
      if (get->GetReturnCode() != 0) return 6;
    }
    double s = std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "BENCH read_peer rank=%d mb=%llu secs=%.3f mbps=%.1f\n",
            rank, (unsigned long long)(total >> 20), s, Mbps(total, s));
  }
  if (!Barrier(bar_id)) return 3;

  fprintf(stderr, "BENCH done rank=%d\n", rank);
  return 0;
}

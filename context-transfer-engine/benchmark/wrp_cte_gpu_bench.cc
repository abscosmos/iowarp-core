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
 * CTE GPU Benchmark Application
 *
 * Measures the performance of GPU-initiated PutBlob/GetBlob operations.
 * GPU client kernels create tasks using the Client API and submit them
 * via the GPU->GPU queue. The GPU orchestrator dispatches to GpuRuntime.
 *
 * Usage:
 *   wrp_cte_gpu_bench <test_case> <client_blocks> <client_threads>
 *                     <runtime_blocks> <runtime_threads>
 *                     <io_size> <io_count>
 *
 * Parameters:
 *   test_case:       Put, Get, or PutGet
 *   client_blocks:   Number of GPU blocks for client kernels
 *   client_threads:  Threads per block for client kernels (only thread 0 works)
 *   runtime_blocks:  Number of GPU blocks for the orchestrator
 *   runtime_threads: Threads per block for the orchestrator
 *   io_size:         Size of each I/O (supports k/K, m/M, g/G suffixes)
 *   io_count:        Number of I/O operations per client block
 */

#include <chimaera/chimaera.h>
#include <wrp_cte/core/core_client.h>
#include <hermes_shm/util/logging.h>
#include <hermes_shm/util/gpu_api.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

/**
 * Per-block result from the GPU kernel.
 */
struct GpuBenchResult {
  int status;
  long long elapsed_ns;  // GPU clock cycles (not wall-clock ns)
};

extern "C" int run_gpu_bench(
    chi::PoolId pool_id,
    wrp_cte::core::TagId tag_id,
    chi::u64 io_size,
    int io_count,
    int mode,
    int num_blocks,
    int num_threads,
    GpuBenchResult *results);

namespace {

chi::u64 ParseSize(const std::string &size_str) {
  double size = 0.0;
  chi::u64 multiplier = 1;
  std::string num_str;
  char suffix = 0;

  for (char c : size_str) {
    if (std::isdigit(c) || c == '.') {
      num_str += c;
    } else if (c == 'k' || c == 'K' || c == 'm' || c == 'M' ||
               c == 'g' || c == 'G') {
      suffix = std::tolower(c);
      break;
    }
  }
  if (num_str.empty()) return 0;
  size = std::stod(num_str);

  switch (suffix) {
    case 'k': multiplier = 1024; break;
    case 'm': multiplier = 1024 * 1024; break;
    case 'g': multiplier = 1024ULL * 1024 * 1024; break;
    default: break;
  }
  return static_cast<chi::u64>(size * multiplier);
}

std::string FormatSize(chi::u64 bytes) {
  if (bytes >= 1024ULL * 1024 * 1024)
    return std::to_string(bytes / (1024ULL * 1024 * 1024)) + " GB";
  if (bytes >= 1024 * 1024)
    return std::to_string(bytes / (1024 * 1024)) + " MB";
  if (bytes >= 1024)
    return std::to_string(bytes / 1024) + " KB";
  return std::to_string(bytes) + " B";
}

double CalcBandwidth(chi::u64 total_bytes, double seconds) {
  if (seconds <= 0.0) return 0.0;
  return static_cast<double>(total_bytes) / (1024.0 * 1024.0) / seconds;
}

int ModeFromString(const std::string &s) {
  if (s == "Put") return 0;
  if (s == "Get") return 1;
  if (s == "PutGet") return 2;
  return -1;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 8) {
    HLOG(kError,
         "Usage: {} <test_case> <client_blocks> <client_threads>"
         " <runtime_blocks> <runtime_threads> <io_size> <io_count>",
         argv[0]);
    HLOG(kError, "  test_case:       Put, Get, or PutGet");
    HLOG(kError, "  client_blocks:   GPU blocks for client kernels");
    HLOG(kError, "  client_threads:  Threads/block for client kernels");
    HLOG(kError, "  runtime_blocks:  GPU blocks for orchestrator");
    HLOG(kError, "  runtime_threads: Threads/block for orchestrator");
    HLOG(kError, "  io_size:         I/O size (e.g., 4k, 1m)");
    HLOG(kError, "  io_count:        I/Os per client block");
    return 1;
  }

  std::string test_case = argv[1];
  int client_blocks = std::atoi(argv[2]);
  int client_threads = std::atoi(argv[3]);
  int runtime_blocks = std::atoi(argv[4]);
  int runtime_threads = std::atoi(argv[5]);
  chi::u64 io_size = ParseSize(argv[6]);
  int io_count = std::atoi(argv[7]);

  int mode = ModeFromString(test_case);
  if (mode < 0 || client_blocks <= 0 || client_threads <= 0 ||
      runtime_blocks <= 0 || runtime_threads <= 0 ||
      io_size == 0 || io_count <= 0) {
    HLOG(kError, "Invalid parameters");
    return 1;
  }

  // Check GPU availability
  int num_gpus = hshm::GpuApi::GetDeviceCount();
  if (num_gpus == 0) {
    HLOG(kError, "No GPUs available");
    return 1;
  }

  // Initialize Chimaera with embedded runtime (fork mode)
  // GPU orchestrator launches automatically during runtime init
  HLOG(kInfo, "Initializing Chimaera runtime...");
  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true)) {
    HLOG(kError, "Failed to initialize Chimaera runtime");
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Create a separate CTE pool with a unique ID so that CreatePool
  // goes through the full path including GPU container allocation.
  // (The compose-config pool is reused by WRP_CTE_CLIENT_INIT and skips
  // GPU container registration.)
  chi::PoolId gpu_pool_id(wrp_cte::core::kCtePoolId.major_ + 1,
                           wrp_cte::core::kCtePoolId.minor_);
  wrp_cte::core::Client cte_client_obj(gpu_pool_id);
  wrp_cte::core::CreateParams params;
  auto create_task = cte_client_obj.AsyncCreate(
      chi::PoolQuery::Dynamic(),
      "cte_gpu_bench_pool", gpu_pool_id, params);
  create_task.Wait();
  if (create_task->GetReturnCode() != 0) {
    HLOG(kError, "Failed to create CTE GPU pool: {}", create_task->GetReturnCode());
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // Create tag via CPU-path Local() (matching test fixture pattern)
  auto *cte_client = &cte_client_obj;
  auto tag_task = cte_client->AsyncGetOrCreateTag(
      "gpu_bench_tag", wrp_cte::core::TagId::GetNull(),
      chi::PoolQuery::Local());
  tag_task.Wait();
  if (tag_task->GetReturnCode() != 0) {
    HLOG(kError, "Failed to create tag");
    return 1;
  }
  wrp_cte::core::TagId tag_id = tag_task->tag_id_;
  HLOG(kInfo, "Pool ID: {}.{}", gpu_pool_id.major_, gpu_pool_id.minor_);
  HLOG(kInfo, "Tag created: {}.{}", tag_id.major_, tag_id.minor_);

  // Print benchmark info
  int total_ios = client_blocks * io_count;
  int ops_multiplier = (mode == 2) ? 2 : 1;
  chi::u64 total_bytes = static_cast<chi::u64>(total_ios) * io_size * ops_multiplier;

  HLOG(kInfo, "=== CTE GPU Benchmark ===");
  HLOG(kInfo, "Test case: {}", test_case);
  HLOG(kInfo, "Client: {} blocks x {} threads", client_blocks, client_threads);
  HLOG(kInfo, "Runtime: {} blocks x {} threads", runtime_blocks, runtime_threads);
  HLOG(kInfo, "I/O size: {}", FormatSize(io_size));
  HLOG(kInfo, "I/O count per block: {}", io_count);
  HLOG(kInfo, "Total I/Os: {} ({} ops)", total_ios,
       total_ios * ops_multiplier);
  HLOG(kInfo, "Total data: {}", FormatSize(total_bytes));
  HLOG(kInfo, "=========================");

  // Allocate pinned results array
  GpuBenchResult *results = nullptr;
  cudaMallocHost(reinterpret_cast<void**>(&results),
                 client_blocks * sizeof(GpuBenchResult));
  if (!results) {
    HLOG(kError, "Failed to allocate results array");
    return 1;
  }

  // Run benchmark with wall-clock timing
  auto start = std::chrono::high_resolution_clock::now();

  int rc = run_gpu_bench(
      cte_client->pool_id_, tag_id,
      io_size, io_count, mode,
      client_blocks, client_threads,
      results);

  auto end = std::chrono::high_resolution_clock::now();
  double wall_seconds = std::chrono::duration<double>(end - start).count();

  if (rc != 0) {
    HLOG(kError, "GPU benchmark failed with error: {}", rc);
    for (int i = 0; i < client_blocks; ++i) {
      if (results[i].status != 1) {
        HLOG(kError, "  Block {}: status={}", i, results[i].status);
      }
    }
    return 1;
  }

  // Print results
  double bw = CalcBandwidth(total_bytes, wall_seconds);
  double iops = static_cast<double>(total_ios * ops_multiplier) / wall_seconds;

  HLOG(kInfo, "");
  HLOG(kInfo, "=== {} Results ===", test_case);
  HLOG(kInfo, "Wall time: {} s", wall_seconds);
  HLOG(kInfo, "Bandwidth: {} MB/s", bw);
  HLOG(kInfo, "IOPS: {}", iops);
  HLOG(kInfo, "Latency (avg): {} us/op",
       (wall_seconds * 1e6) / (total_ios * ops_multiplier));
  HLOG(kInfo, "==================");

  return 0;
}

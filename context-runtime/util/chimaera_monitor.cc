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
 * Chimaera worker monitoring utility
 *
 * This utility connects to a running Chimaera runtime and displays
 * real-time statistics about worker threads, including:
 * - Number of queued, blocked, and periodic tasks
 * - Worker idle status and suspend periods
 * - Overall system load and utilization
 */

#include <chrono>
#include <csignal>
#include <iostream>
#include <thread>
#include <iomanip>

#include "chimaera/chimaera.h"
#include "chimaera/singletons.h"
#include "chimaera/types.h"
#include "chimaera/admin/admin_client.h"

namespace {

volatile bool g_keep_running = true;

/**
 * Print usage information
 */
void PrintUsage(const char* program_name) {
  HIPRINT("Usage: {} [OPTIONS]", program_name);
  HIPRINT("");
  HIPRINT("Options:");
  HIPRINT("  -h, --help        Show this help message");
  HIPRINT("  -i, --interval N  Set monitoring interval in seconds (default: 1)");
  HIPRINT("  -o, --once        Run once and exit (default: continuous monitoring)");
  HIPRINT("  -j, --json        Output raw JSON format");
  HIPRINT("  -v, --verbose     Enable verbose output");
  HIPRINT("");
  HIPRINT("Examples:");
  HIPRINT("  {}              # Continuous monitoring at 1 second intervals", program_name);
  HIPRINT("  {} -i 5         # Update every 5 seconds", program_name);
  HIPRINT("  {} -o           # Run once and exit", program_name);
  HIPRINT("  {} -j           # Output raw JSON", program_name);
}

/**
 * Parse command line arguments
 */
struct MonitorOptions {
  int interval_sec = 1;
  bool once = false;
  bool json_output = false;
  bool verbose = false;
};

bool ParseArgs(int argc, char* argv[], MonitorOptions& opts) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "-h" || arg == "--help") {
      PrintUsage(argv[0]);
      return false;
    } else if (arg == "-i" || arg == "--interval") {
      if (i + 1 < argc) {
        opts.interval_sec = std::atoi(argv[++i]);
        if (opts.interval_sec < 1) {
          HLOG(kError, "Interval must be >= 1 second");
          return false;
        }
      } else {
        HLOG(kError, "-i/--interval requires an argument");
        return false;
      }
    } else if (arg == "-o" || arg == "--once") {
      opts.once = true;
    } else if (arg == "-j" || arg == "--json") {
      opts.json_output = true;
    } else if (arg == "-v" || arg == "--verbose") {
      opts.verbose = true;
    } else {
      HLOG(kError, "Unknown option: {}", arg);
      PrintUsage(argv[0]);
      return false;
    }
  }

  return true;
}

/**
 * Print worker statistics in human-readable format
 */
void PrintStats(const chimaera::admin::MonitorTask& task) {
  // Clear screen and move cursor to top
  HIPRINT("\033[2J\033[H");

  // Print header
  auto now = std::chrono::system_clock::now();
  auto now_t = std::chrono::system_clock::to_time_t(now);
  std::ostringstream time_ss;
  time_ss << std::put_time(std::localtime(&now_t), "%Y-%m-%d %H:%M:%S");
  HIPRINT("==================================================");
  HIPRINT("        Chimaera Worker Monitor");
  HIPRINT("        {}", time_ss.str());
  HIPRINT("==================================================");
  HIPRINT("");

  // Calculate summary statistics
  chi::u32 total_queued = 0;
  chi::u32 total_blocked = 0;
  chi::u32 total_periodic = 0;

  for (const auto& stats : task.info_) {
    total_queued += stats.num_queued_tasks_;
    total_blocked += stats.num_blocked_tasks_;
    total_periodic += stats.num_periodic_tasks_;
  }

  // Print summary
  HIPRINT("Summary:");
  HIPRINT("  Total Workers:        {}", task.info_.size());
  HIPRINT("  Total Queued Tasks:   {}", total_queued);
  HIPRINT("  Total Blocked Tasks:  {}", total_blocked);
  HIPRINT("  Total Periodic Tasks: {}", total_periodic);
  HIPRINT("");

  // Print table header using std::ostringstream for setw formatting
  std::ostringstream header;
  header << std::setw(6) << "ID"
         << std::setw(10) << "Running"
         << std::setw(10) << "Active"
         << std::setw(12) << "Idle Iters"
         << std::setw(10) << "Queued"
         << std::setw(10) << "Blocked"
         << std::setw(10) << "Periodic"
         << std::setw(15) << "Suspend (us)";
  HIPRINT("Worker Details:");
  HIPRINT("{}", header.str());
  HIPRINT("{}", std::string(83, '-'));

  // Print worker statistics
  for (const auto& stats : task.info_) {
    std::ostringstream row;
    row << std::setw(6) << stats.worker_id_
        << std::setw(10) << (stats.is_running_ ? "Yes" : "No")
        << std::setw(10) << (stats.is_active_ ? "Yes" : "No")
        << std::setw(12) << stats.idle_iterations_
        << std::setw(10) << stats.num_queued_tasks_
        << std::setw(10) << stats.num_blocked_tasks_
        << std::setw(10) << stats.num_periodic_tasks_
        << std::setw(15) << stats.suspend_period_us_;
    HIPRINT("{}", row.str());
  }

  HIPRINT("");
  HIPRINT("Press Ctrl+C to exit");
}

}  // namespace

int main(int argc, char* argv[]) {
  // Parse command line arguments
  MonitorOptions opts;
  if (!ParseArgs(argc, argv, opts)) {
    return (argc > 1) ? 1 : 0;  // Return 0 if help was requested, 1 for errors
  }

  if (opts.verbose) {
    HLOG(kInfo, "Initializing Chimaera client...");
  }

  // Initialize Chimaera in client mode
  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, false)) {
    HLOG(kError, "Failed to initialize Chimaera client");
    HLOG(kError, "Make sure the Chimaera runtime is running");
    return 1;
  }

  if (opts.verbose) {
    HLOG(kInfo, "Chimaera client initialized successfully");
  }

  // Get admin client
  HLOG(kInfo, "Getting admin client...");
  auto* admin_client = CHI_ADMIN;
  if (!admin_client) {
    HLOG(kError, "Failed to get admin client");
    return 1;
  }

  HLOG(kInfo, "Admin client obtained successfully");
  if (opts.verbose) {
    HLOG(kInfo, "Connected to admin module");
  }

  // Main monitoring loop
  while (g_keep_running) {
    try {
      // Request worker statistics
      HLOG(kInfo, "Sending AsyncMonitor request...");
      auto future = admin_client->AsyncMonitor(chi::PoolQuery::Local());
      HLOG(kInfo, "AsyncMonitor returned future");

      HLOG(kInfo, "About to call future.Wait()...");
      // Wait for the result
      future.Wait();
      HLOG(kInfo, "future.Wait() returned - monitor task completed");

      // Get the task result (Future has operator->)
      if (future->GetReturnCode() != 0) {
        HLOG(kError, "Monitor task failed with return code {}",
             future->GetReturnCode());
        break;
      }

      // Display the results
      if (opts.json_output) {
        // Output JSON format using ostringstream
        std::ostringstream json;
        json << "{\"workers\":[";
        bool first = true;
        for (const auto& stats : future->info_) {
          if (!first) json << ",";
          first = false;
          json << "{"
               << "\"worker_id\":" << stats.worker_id_ << ","
               << "\"is_running\":" << (stats.is_running_ ? "true" : "false") << ","
               << "\"is_active\":" << (stats.is_active_ ? "true" : "false") << ","
               << "\"idle_iterations\":" << stats.idle_iterations_ << ","
               << "\"num_queued_tasks\":" << stats.num_queued_tasks_ << ","
               << "\"num_blocked_tasks\":" << stats.num_blocked_tasks_ << ","
               << "\"num_periodic_tasks\":" << stats.num_periodic_tasks_ << ","
               << "\"suspend_period_us\":" << stats.suspend_period_us_ << "}";
        }
        json << "]}";
        HIPRINT("{}", json.str());
      } else {
        // Print formatted output
        PrintStats(*future);
      }

      // Exit if running once
      if (opts.once) {
        break;
      }

      // Wait for the specified interval
      for (int i = 0; i < opts.interval_sec && g_keep_running; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }

    } catch (const std::exception& e) {
      HLOG(kError, "Exception during monitoring: {}", e.what());
      break;
    }
  }

  if (opts.verbose) {
    HLOG(kInfo, "Shutting down Chimaera client...");
  }

  // Chimaera cleanup happens automatically
  return 0;
}

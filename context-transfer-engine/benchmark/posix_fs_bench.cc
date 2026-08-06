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
 * POSIX baseline for clio_cte_reliability_bench.
 *
 * Runs the SAME eight-operation mixed workload — write_4k, read_4k, write_1m,
 * read_1m, stat_size, readdir_small (10 entries), readdir_large (100 entries),
 * rename — against an ordinary directory on an ordinary filesystem, using
 * nothing but POSIX calls (pwrite/pread/stat/opendir/rename). Point --root at
 * an NFS mount to get "what you would have got by just using the shared
 * filesystem", or at local storage for a same-hardware reference.
 *
 * It deliberately mirrors the clio benchmark exactly: same CLI flags, same
 * time-based structure, same warmup handling, same per-operation latency
 * sampling, and the same CSV schema — so the same driver script can run both
 * and the numbers line up column for column.
 *
 * Fairness notes:
 *   - Buffered I/O, no O_DIRECT and no fsync, because clio-fs does not fsync
 *     either (its backing bdev opens O_RDWR|O_CREAT). Both sides are therefore
 *     measured with the page cache in play.
 *   - One open file descriptor per thread, held for the run, matching the clio
 *     benchmark's one open handle per thread.
 *   - Writes land at random block-aligned offsets inside the thread's slice of
 *     --max-data, so the dataset never grows past the cap.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

enum OpKind {
  kWrite4K = 0,
  kRead4K,
  kWrite1M,
  kRead1M,
  kStatSize,
  kReaddirSmall,
  kReaddirLarge,
  kRename,
  kNumOps
};

const char *OpName(int op) {
  switch (op) {
    case kWrite4K: return "write_4k";
    case kRead4K: return "read_4k";
    case kWrite1M: return "write_1m";
    case kRead1M: return "read_1m";
    case kStatSize: return "stat_size";
    case kReaddirSmall: return "readdir_small";
    case kReaddirLarge: return "readdir_large";
    case kRename: return "rename";
    default: return "unknown";
  }
}

bool IsDataOp(int op) {
  return op == kWrite4K || op == kRead4K || op == kWrite1M || op == kRead1M;
}

constexpr size_t kSmallDirEntries = 10;
constexpr size_t kLargeDirEntries = 100;
constexpr uint64_t kMaxIoSize = 1024 * 1024;

struct BenchConfig {
  double duration_s = 30.0;
  double warmup_s = 2.0;
  size_t threads = 8;
  uint64_t max_data = 2ULL << 30;
  uint64_t small_io = 4096;
  std::string root = "/tmp/posix_fs_bench";
  std::string label = "posix";
  std::string csv_path;
  double mix[kNumOps] = {12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5};
};

uint64_t ParseSize(const std::string &s) {
  if (s.empty()) return 0;
  char suffix = static_cast<char>(std::tolower(s.back()));
  uint64_t mult = 1;
  std::string num = s;
  if (suffix == 'k') { mult = 1024ULL; num.pop_back(); }
  else if (suffix == 'm') { mult = 1024ULL * 1024; num.pop_back(); }
  else if (suffix == 'g') { mult = 1024ULL * 1024 * 1024; num.pop_back(); }
  return static_cast<uint64_t>(std::stoull(num)) * mult;
}

bool ParseMix(const std::string &spec, double (&mix)[kNumOps]) {
  for (int i = 0; i < kNumOps; ++i) mix[i] = 0.0;
  std::stringstream ss(spec);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) continue;
    size_t eq = item.find('=');
    if (eq == std::string::npos) return false;
    const std::string name = item.substr(0, eq);
    double pct = std::stod(item.substr(eq + 1));
    int found = -1;
    for (int i = 0; i < kNumOps; ++i) {
      if (name == OpName(i)) { found = i; break; }
    }
    if (found < 0) {
      std::cerr << "unknown operation in --mix: " << name << "\n";
      return false;
    }
    mix[found] = pct;
  }
  double total = 0.0;
  for (int i = 0; i < kNumOps; ++i) total += mix[i];
  return total > 0.0;
}

std::string Pad(const std::string &s, size_t w) {
  return s.size() >= w ? s : std::string(w - s.size(), ' ') + s;
}
std::string PadRight(const std::string &s, size_t w) {
  return s.size() >= w ? s : s + std::string(w - s.size(), ' ');
}
std::string Num(double v, size_t w, int prec = 2) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(prec) << v;
  return Pad(os.str(), w);
}

double Pctl(const std::vector<double> &sorted, double q) {
  if (sorted.empty()) return 0.0;
  size_t idx = static_cast<size_t>(q * (sorted.size() - 1) + 0.5);
  return sorted[std::min(idx, sorted.size() - 1)];
}

struct OpStats {
  std::vector<double> lat_us;
  uint64_t bytes = 0;
  uint64_t errors = 0;
  void Merge(const OpStats &o) {
    lat_us.insert(lat_us.end(), o.lat_us.begin(), o.lat_us.end());
    bytes += o.bytes;
    errors += o.errors;
  }
};

std::string DataFilePath(const BenchConfig &cfg, size_t tid) {
  return cfg.root + "/data/f" + std::to_string(tid);
}
std::string RenamePath(const BenchConfig &cfg, size_t tid, uint64_t seq) {
  return cfg.root + "/renames/t" + std::to_string(tid) + "_" +
         std::to_string(seq);
}

bool EnsureDir(const std::string &p) {
  return ::mkdir(p.c_str(), 0755) == 0 || errno == EEXIST;
}

/** Create/extend `path` to `size` bytes so every read offset has real data. */
bool CreateAndFill(const std::string &path, uint64_t size,
                   const std::vector<char> &buf) {
  int fd = ::open(path.c_str(), O_CREAT | O_RDWR, 0644);
  if (fd < 0) {
    std::cerr << "setup: open " << path << " failed: " << strerror(errno)
              << "\n";
    return false;
  }
  uint64_t off = 0;
  while (off < size) {
    uint64_t n = std::min<uint64_t>(kMaxIoSize, size - off);
    ssize_t w = ::pwrite(fd, buf.data(), n, static_cast<off_t>(off));
    if (w < 0 || static_cast<uint64_t>(w) != n) {
      std::cerr << "setup: pwrite " << path << " failed: " << strerror(errno)
                << "\n";
      ::close(fd);
      return false;
    }
    off += n;
  }
  ::close(fd);
  return true;
}

bool PopulateDir(const std::string &dir, size_t count,
                 const std::vector<char> &buf) {
  if (!EnsureDir(dir)) return false;
  for (size_t i = 0; i < count; ++i) {
    const std::string p = dir + "/e" + std::to_string(i);
    int fd = ::open(p.c_str(), O_CREAT | O_RDWR, 0644);
    if (fd < 0) return false;
    ::pwrite(fd, buf.data(), 4096, 0);
    ::close(fd);
  }
  return true;
}

struct ThreadContext {
  size_t tid = 0;
  int fd = -1;
  uint64_t file_size = 0;
  uint64_t rename_seq = 0;
  std::vector<char> buf;
  OpStats stats[kNumOps];
};

void Worker(const BenchConfig &cfg, ThreadContext &ctx,
            const std::vector<double> &cdf,
            std::chrono::steady_clock::time_point measure_start,
            std::chrono::steady_clock::time_point deadline) {
  std::mt19937_64 rng(0x9E3779B97F4A7C15ULL ^ (ctx.tid * 1000003ULL));
  std::uniform_real_distribution<double> pick(0.0, 1.0);
  const std::string small_dir = cfg.root + "/small_dir";
  const std::string large_dir = cfg.root + "/large_dir";
  const std::string stat_path = DataFilePath(cfg, ctx.tid);

  while (std::chrono::steady_clock::now() < deadline) {
    const double r = pick(rng);
    int op = kNumOps - 1;
    for (int i = 0; i < kNumOps; ++i) {
      if (r <= cdf[i]) { op = i; break; }
    }

    uint64_t io_size = 0;
    if (op == kWrite4K || op == kRead4K) io_size = cfg.small_io;
    else if (op == kWrite1M || op == kRead1M) io_size = kMaxIoSize;

    uint64_t offset = 0;
    if (io_size > 0) {
      uint64_t blocks = ctx.file_size / io_size;
      offset = (blocks > 0 ? (rng() % blocks) : 0) * io_size;
    }

    const auto t0 = std::chrono::steady_clock::now();
    bool ok = true;
    uint64_t moved = 0;

    switch (op) {
      case kWrite4K:
      case kWrite1M: {
        ssize_t w = ::pwrite(ctx.fd, ctx.buf.data(), io_size,
                             static_cast<off_t>(offset));
        ok = (w >= 0 && static_cast<uint64_t>(w) == io_size);
        if (ok) moved = io_size;
        break;
      }
      case kRead4K:
      case kRead1M: {
        ssize_t rd = ::pread(ctx.fd, ctx.buf.data(), io_size,
                             static_cast<off_t>(offset));
        ok = (rd >= 0);
        if (ok) moved = static_cast<uint64_t>(rd);
        break;
      }
      case kStatSize: {
        struct stat st;
        ok = (::stat(stat_path.c_str(), &st) == 0);
        break;
      }
      case kReaddirSmall:
      case kReaddirLarge: {
        const std::string &d = (op == kReaddirSmall) ? small_dir : large_dir;
        DIR *dp = ::opendir(d.c_str());
        if (dp == nullptr) {
          ok = false;
          break;
        }
        // Read every entry — the clio Readdir returns the full listing, so a
        // fair comparison must walk the whole directory, not just open it.
        while (::readdir(dp) != nullptr) {
        }
        ::closedir(dp);
        break;
      }
      case kRename: {
        const std::string src = RenamePath(cfg, ctx.tid, ctx.rename_seq);
        const std::string dst = RenamePath(cfg, ctx.tid, ctx.rename_seq + 1);
        ok = (::rename(src.c_str(), dst.c_str()) == 0);
        if (ok) ++ctx.rename_seq;
        break;
      }
      default:
        ok = false;
        break;
    }

    const auto t1 = std::chrono::steady_clock::now();
    if (t1 < measure_start) continue;  // warmup
    const double us =
        std::chrono::duration<double, std::micro>(t1 - t0).count();
    if (ok) {
      ctx.stats[op].lat_us.push_back(us);
      ctx.stats[op].bytes += moved;
    } else {
      ctx.stats[op].errors++;
    }
  }
}

void PrintUsage(const char *argv0) {
  std::cout << "Usage: " << argv0 << " [options]\n"
            << "  --duration <sec>   --warmup <sec>    --threads <N>\n"
            << "  --max-data <size>  --small-io <size> --mix <op=pct,...>\n"
            << "  --root <path>      --label <name>    --csv <path>\n"
            << "  --filler <N>       (accepted, ignored: no tag index here)\n";
}

}  // namespace

int main(int argc, char **argv) {
  BenchConfig cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--duration" && i + 1 < argc)
      cfg.duration_s = std::stod(argv[++i]);
    else if (a == "--warmup" && i + 1 < argc)
      cfg.warmup_s = std::stod(argv[++i]);
    else if (a == "--threads" && i + 1 < argc)
      cfg.threads = std::stoull(argv[++i]);
    else if (a == "--max-data" && i + 1 < argc)
      cfg.max_data = ParseSize(argv[++i]);
    else if (a == "--small-io" && i + 1 < argc)
      cfg.small_io = ParseSize(argv[++i]);
    else if (a == "--root" && i + 1 < argc) cfg.root = argv[++i];
    else if (a == "--label" && i + 1 < argc) cfg.label = argv[++i];
    else if (a == "--csv" && i + 1 < argc) cfg.csv_path = argv[++i];
    else if (a == "--filler" && i + 1 < argc) ++i;  // accepted for CLI parity
    else if (a == "--mix" && i + 1 < argc) {
      if (!ParseMix(argv[++i], cfg.mix)) return 1;
    } else if (a == "--verbose" || a == "-v") {
    } else if (a == "--help" || a == "-h") {
      PrintUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << a << "\n";
      PrintUsage(argv[0]);
      return 1;
    }
  }
  if (cfg.threads == 0 || cfg.max_data / cfg.threads < 2 * kMaxIoSize) {
    std::cerr << "--max-data too small for --threads\n";
    return 1;
  }

  std::cout << "=== POSIX Filesystem Baseline ===\n"
            << "Label: " << cfg.label << "\nRoot: " << cfg.root
            << "\nThreads: " << cfg.threads << "  Duration: " << cfg.duration_s
            << " s  Warmup: " << cfg.warmup_s << " s\n"
            << "Max dataset: " << cfg.max_data << " bytes ("
            << (cfg.max_data / (1024 * 1024)) << " MB)\n";

  const uint64_t per_thread = cfg.max_data / cfg.threads;
  std::vector<char> setup_buf(kMaxIoSize, 0x5A);

  if (!EnsureDir(cfg.root) || !EnsureDir(cfg.root + "/data") ||
      !EnsureDir(cfg.root + "/renames")) {
    std::cerr << "setup: could not create directories under " << cfg.root
              << ": " << strerror(errno) << "\n";
    return 1;
  }
  std::cout << "Setup: " << cfg.threads << " data files x "
            << (per_thread / (1024 * 1024)) << " MB ...\n";
  for (size_t t = 0; t < cfg.threads; ++t) {
    if (!CreateAndFill(DataFilePath(cfg, t), per_thread, setup_buf)) return 1;
    // Clear a few rename slots so the walk starts from a known name.
    for (uint64_t s = 0; s < 4; ++s) {
      ::unlink(RenamePath(cfg, t, s).c_str());
    }
    int fd = ::open(RenamePath(cfg, t, 0).c_str(), O_CREAT | O_RDWR, 0644);
    if (fd < 0) {
      std::cerr << "setup: rename fixture failed: " << strerror(errno) << "\n";
      return 1;
    }
    ::close(fd);
  }
  if (!PopulateDir(cfg.root + "/small_dir", kSmallDirEntries, setup_buf) ||
      !PopulateDir(cfg.root + "/large_dir", kLargeDirEntries, setup_buf)) {
    std::cerr << "setup: could not populate the readdir fixtures\n";
    return 1;
  }
  std::cout << "Setup complete (" << cfg.max_data << " bytes materialized)\n";

  double total_pct = 0.0;
  for (int i = 0; i < kNumOps; ++i) total_pct += cfg.mix[i];
  std::vector<double> cdf(kNumOps, 0.0);
  double acc = 0.0;
  for (int i = 0; i < kNumOps; ++i) {
    acc += cfg.mix[i] / total_pct;
    cdf[i] = acc;
  }

  std::vector<ThreadContext> ctxs(cfg.threads);
  for (size_t t = 0; t < cfg.threads; ++t) {
    ctxs[t].tid = t;
    ctxs[t].file_size = per_thread;
    ctxs[t].buf.assign(kMaxIoSize, static_cast<char>(0xA5));
    ctxs[t].fd = ::open(DataFilePath(cfg, t).c_str(), O_RDWR);
    if (ctxs[t].fd < 0) {
      std::cerr << "could not open data file for thread " << t << "\n";
      return 1;
    }
  }

  const auto start = std::chrono::steady_clock::now();
  const auto measure_start =
      start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                  std::chrono::duration<double>(cfg.warmup_s));
  const auto deadline =
      measure_start +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(cfg.duration_s));

  std::cout << "\nRunning ...\n";
  std::vector<std::thread> workers;
  for (size_t t = 0; t < cfg.threads; ++t) {
    workers.emplace_back(Worker, std::cref(cfg), std::ref(ctxs[t]),
                         std::cref(cdf), measure_start, deadline);
  }
  for (auto &w : workers) w.join();
  const double elapsed_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    measure_start)
          .count();
  for (auto &c : ctxs) {
    if (c.fd >= 0) ::close(c.fd);
  }

  OpStats agg[kNumOps];
  for (size_t t = 0; t < cfg.threads; ++t) {
    for (int i = 0; i < kNumOps; ++i) agg[i].Merge(ctxs[t].stats[i]);
  }

  std::cout << "\n=== Results (" << cfg.label << ") — " << elapsed_s
            << " s measured ===\n";
  std::cout << PadRight("operation", 16) << Pad("count", 10)
            << Pad("ops/sec", 12) << Pad("avg_us", 11) << Pad("p99_us", 11)
            << Pad("MB/s", 11) << "\n";

  std::ofstream csv;
  if (!cfg.csv_path.empty()) {
    const bool exists = std::ifstream(cfg.csv_path).good();
    csv.open(cfg.csv_path, std::ios::app);
    if (csv.is_open() && !exists) {
      csv << "label,operation,count,errors,ops_per_sec,avg_us,p50_us,p99_us,"
             "mb_per_sec,bytes\n";
    }
  }

  for (int i = 0; i < kNumOps; ++i) {
    if (agg[i].lat_us.empty() && agg[i].errors == 0) continue;
    std::sort(agg[i].lat_us.begin(), agg[i].lat_us.end());
    const double count = static_cast<double>(agg[i].lat_us.size());
    double sum = 0.0;
    for (double v : agg[i].lat_us) sum += v;
    const double avg = count > 0 ? sum / count : 0.0;
    const double p50 = Pctl(agg[i].lat_us, 0.50);
    const double p99 = Pctl(agg[i].lat_us, 0.99);
    const double ops_s = elapsed_s > 0 ? count / elapsed_s : 0.0;
    const double mb_s =
        elapsed_s > 0
            ? (static_cast<double>(agg[i].bytes) / elapsed_s) / (1024 * 1024)
            : 0.0;
    std::cout << PadRight(OpName(i), 16)
              << Pad(std::to_string(static_cast<uint64_t>(count)), 10)
              << Num(ops_s, 12) << Num(avg, 11) << Num(p99, 11)
              << (IsDataOp(i) ? Num(mb_s, 11) : Pad("-", 11)) << "\n";
    if (agg[i].errors > 0) {
      std::cout << "  WARNING: " << OpName(i) << " had " << agg[i].errors
                << " failed operations\n";
    }
    if (csv.is_open()) {
      csv << cfg.label << "," << OpName(i) << ","
          << static_cast<uint64_t>(count) << "," << agg[i].errors << ","
          << ops_s << "," << avg << "," << p50 << "," << p99 << "," << mb_s
          << "," << agg[i].bytes << "\n";
    }
  }
  if (csv.is_open()) {
    csv.close();
    std::cout << "\nCSV appended: " << cfg.csv_path << "\n";
  }
  return 0;
}

/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * clio-fs zero-IPC read path (issue #817).
 *
 * Proves three things, in order of importance:
 *   1. CORRECTNESS -- bytes returned by the fast path are identical to the
 *      bytes the RPC path returns, including at EOF and across page bounds.
 *   2. IT ACTUALLY RAN -- a fast path that silently never engages looks
 *      exactly like a fast path that works, so the test fails loudly when the
 *      cache is unavailable rather than skipping.
 *   3. LATENCY -- the same read, timed on both paths, same query.
 */

#include <fcntl.h>

#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "../../adapter/cfs/cfs_io.h"
#include "clio_cte/core/core_client.h"
#include "clio_cte/filesystem/filesystem_client.h"
#include "clio_runtime/bdev/bdev_client.h"
#include "clio_runtime/clio_runtime.h"
#include "simple_test.h"

using namespace std::chrono_literals;

namespace {

// A RAM target is what makes page payloads SHM-resident and therefore
// direct-readable; a file target would (correctly) refuse the fast path.
constexpr clio::run::u64 kRamTargetBytes = 256ULL * 1024 * 1024;
const std::string kBackendPath = "/tmp/clio_cfs_shm_read_test.dat";
const std::string kClioPath = "clio::" + kBackendPath;

/** Microseconds, as a double. */
double NowUs() {
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

bool InitRuntime() {
  static bool ok = false;
  static bool tried = false;
  if (tried) {
    return ok;
  }
  tried = true;

  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    return false;
  }
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    return false;
  }

  // Register a RAM target on the default CTE core pool -- the filesystem
  // chimod sits over that pool, so this is the pool whose placement decides
  // whether a page is direct-readable.
  auto *cte_client = CLIO_CTE_CLIENT;
  clio::run::PoolId bdev_pool_id(8171, 0);
  clio::run::bdev::Client bdev_client(bdev_pool_id);
  auto create_task = bdev_client.AsyncCreate(
      clio::run::PoolQuery::Dynamic(), "cfs_shm_read_ram", bdev_pool_id,
      clio::run::bdev::BdevType::kRam);
  create_task.Wait();
  std::this_thread::sleep_for(100ms);

  auto reg = cte_client->AsyncRegisterTarget(
      "cfs_shm_read_ram", clio::run::bdev::BdevType::kRam, kRamTargetBytes,
      clio::run::PoolQuery::Local(), bdev_pool_id);
  reg.Wait();
  if (reg->GetReturnCode() != 0) {
    return false;
  }
  std::this_thread::sleep_for(100ms);

  ok = true;
  return ok;
}

/** Read `count` bytes at `off` through the chimod RPC, bypassing the cache.
 *  This is the exact work CfsIo::DoRead does when the fast path declines, and
 *  is the honest baseline to time the fast path against. */
ssize_t RpcRead(clio::run::u64 handle, clio::run::u64 off, void *buf,
                size_t count) {
  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> shm = ipc->AllocateBuffer(count);
  if (shm.IsNull()) {
    return -1;
  }
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncRead(handle, off, count, ctp::ipc::ShmPtr<>(shm.shm_));
  t.Wait();
  ssize_t ret = -1;
  if (t->GetReturnCode() == 0) {
    std::memcpy(buf, shm.ptr_, static_cast<size_t>(t->bytes_read_));
    ret = static_cast<ssize_t>(t->bytes_read_);
  }
  ipc->FreeBuffer(shm);
  return ret;
}

}  // namespace

TEST_CASE("clio-fs SHM read: correctness and latency", "[cfs][shm][noleak]") {
  REQUIRE(InitRuntime());

  auto *cfs_io = CLIO_CTE_CFS;
  REQUIRE(cfs_io != nullptr);

  // ---- write a file through the adapter ----------------------------------
  const size_t kFileSize = 3 * 1024 * 1024 + 4096;  // spans 4 pages, unaligned
  std::vector<char> src(kFileSize);
  for (size_t i = 0; i < kFileSize; ++i) {
    src[i] = static_cast<char>((i * 31 + 7) % 251);
  }

  cfs_io->RemovePath(kClioPath);  // start from a known state
  int fd = cfs_io->Open(kClioPath, O_CREAT | O_RDWR | O_TRUNC, 0644);
  REQUIRE(fd >= 0);
  REQUIRE(cfs_io->Write(fd, src.data(), kFileSize) ==
          static_cast<ssize_t>(kFileSize));

  // ---- the cache must actually be attached -------------------------------
  // A silently-disabled cache would make every assertion below pass on the
  // RPC path, so this is checked explicitly rather than being inferred.
  auto *cte_client = CLIO_CTE_CLIENT;
  auto *fs_client = CLIO_CFS_CLIENT;
  REQUIRE(cte_client->HasShmCache());
  REQUIRE(fs_client->HasShmCache());

  clio::cte::filesystem::ShmFileRecord rec;
  REQUIRE(fs_client->TryGetFileRecordShm(kBackendPath, &rec));
  REQUIRE(rec.IsFastPathable());
  REQUIRE(rec.size_ == kFileSize);

  // The page payloads must be direct-readable, or the fast path can only ever
  // decline and the latency numbers below would be measuring the RPC path.
  clio::cte::core::ShmBlobRecord page0;
  REQUIRE(cte_client->TryGetBlobRecordShm(rec.tag_id_, "0", &page0));
  std::printf(
      "[#817] page 0: size=%llu covered=%llu blocks=%u flags=0x%x direct=%d "
      "(record %zu B)\n",
      static_cast<unsigned long long>(page0.total_size_),
      static_cast<unsigned long long>(page0.CoveredBytes()), page0.num_blocks_,
      page0.flags_, page0.IsDirectReadable() ? 1 : 0,
      sizeof(clio::cte::core::ShmBlobRecord));
  REQUIRE(page0.IsDirectReadable());

  // ---- correctness: fast path == RPC path, byte for byte ------------------
  struct Case {
    const char *what;
    clio::run::u64 off;
    size_t len;
  };
  const Case cases[] = {
      {"4K at 0", 0, 4096},
      {"4K mid-page", 12345, 4096},
      {"1 byte", 1048575, 1},
      {"crosses a page boundary", 1048576 - 100, 200},
      {"short read at EOF", kFileSize - 10, 4096},
      {"entirely past EOF", kFileSize + 1, 4096},
  };
  for (const Case &c : cases) {
    std::vector<char> fast(c.len, 0), rpc(c.len, 0);
    ssize_t nf = cfs_io->Pread(fd, fast.data(), c.len,
                               static_cast<off_t>(c.off));
    ssize_t nr = RpcRead(cfs_io->HandleOf(fd), c.off, rpc.data(), c.len);
    REQUIRE(nf >= 0);
    REQUIRE(nr == nf);  // the two paths must agree on the length...
    if (nf > 0) {
      REQUIRE(std::memcmp(fast.data(), rpc.data(),
                          static_cast<size_t>(nf)) == 0);  // ...and the bytes
    }
    // Compare against the source buffer -- the authoritative answer.
    clio::run::u64 expect = 0;
    if (c.off < kFileSize) {
      expect = std::min<clio::run::u64>(c.len, kFileSize - c.off);
    }
    REQUIRE(static_cast<clio::run::u64>(nf) == expect);
    if (expect > 0) {
      REQUIRE(std::memcmp(fast.data(), src.data() + c.off,
                          static_cast<size_t>(expect)) == 0);
    }
  }

  // ---- a shrink must not keep being served from the old mirror -----------
  // Truncate frees page blobs back to the bdev, where another blob can take
  // them. If the mirror kept the old size the client would happily read
  // through freed storage, and the placement_gen_ guard could not catch it
  // either -- an un-republished record shows the reader the same generation
  // twice. Both halves (fs record + blob record) are invalidated by the
  // runtime before the pages go away.
  {
    const clio::run::u64 kShrunk = 8192;
    REQUIRE(cfs_io->FtruncateFd(fd, static_cast<off_t>(kShrunk)) == 0);

    clio::cte::filesystem::ShmFileRecord shrunk;
    REQUIRE(fs_client->TryGetFileRecordShm(kBackendPath, &shrunk));
    REQUIRE(shrunk.size_ == kShrunk);

    std::vector<char> after(4096, 0x5A);
    // Inside the surviving prefix: still correct, still served.
    REQUIRE(cfs_io->Pread(fd, after.data(), 4096, 4096) == 4096);
    REQUIRE(std::memcmp(after.data(), src.data() + 4096, 4096) == 0);
    // Past the new EOF: nothing, not stale bytes.
    REQUIRE(cfs_io->Pread(fd, after.data(), 4096,
                          static_cast<off_t>(kShrunk)) == 0);

    // Restore the file for the latency measurement below.
    REQUIRE(cfs_io->Pwrite(fd, src.data(), kFileSize, 0) ==
            static_cast<ssize_t>(kFileSize));
  }

  // ---- latency: same 4 KiB read, both paths ------------------------------
  const size_t kIoSize = 4096;
  const int kIters = 20000;
  std::vector<char> buf(kIoSize);

  // Warm: first touch attaches the RAM bdev segment in this process.
  REQUIRE(cfs_io->Pread(fd, buf.data(), kIoSize, 0) ==
          static_cast<ssize_t>(kIoSize));

  double t0 = NowUs();
  for (int i = 0; i < kIters; ++i) {
    ssize_t n = cfs_io->Pread(fd, buf.data(), kIoSize, 0);
    if (n != static_cast<ssize_t>(kIoSize)) {
      REQUIRE(false);
    }
  }
  double shm_us = (NowUs() - t0) / kIters;

  // Fewer RPC iterations: at ~100 us each, 20000 would take half an hour.
  const int kRpcIters = 300;

  // Time the RPC path through the filesystem client directly, using the same
  // chimod handle the adapter holds, so the only difference between the two
  // measurements is which path serves the bytes.
  clio::run::u64 rpc_handle = cfs_io->HandleOf(fd);
  REQUIRE(rpc_handle != 0);

  REQUIRE(RpcRead(rpc_handle, 0, buf.data(), kIoSize) ==
          static_cast<ssize_t>(kIoSize));
  t0 = NowUs();
  for (int i = 0; i < kRpcIters; ++i) {
    if (RpcRead(rpc_handle, 0, buf.data(), kIoSize) !=
        static_cast<ssize_t>(kIoSize)) {
      REQUIRE(false);
    }
  }
  double rpc_us = (NowUs() - t0) / kRpcIters;

  std::printf(
      "\n[#817] clio-fs 4 KiB pread: SHM %.3f us vs RPC %.3f us (%.1fx)\n",
      shm_us, rpc_us, rpc_us / shm_us);
  std::printf("[#817]   (%.6f ms vs %.6f ms)\n", shm_us / 1000.0,
              rpc_us / 1000.0);

  // The point of the exercise: a cached read must be sub-microsecond.
  REQUIRE(shm_us < 1.0);

  cfs_io->Close(fd);
  cfs_io->RemovePath(kClioPath);
}

SIMPLE_TEST_MAIN()

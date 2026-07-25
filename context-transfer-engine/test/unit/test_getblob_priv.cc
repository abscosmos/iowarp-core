/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * Private-memory AsyncGetBlob (issue #823).
 *
 * CoreClient::AsyncGetBlob(char*) reads a blob region straight into a
 * caller-owned PRIVATE buffer instead of making the caller hand-manage an SHM
 * buffer. It resolves to one of three paths, and this test exercises them in
 * BOTH runtime configurations, because the path taken depends on the config:
 *
 *   - Runtime started INLINE (co-located): the daemon shares this address
 *     space, so the private pointer is wrapped as a null-allocator ShmPtr and
 *     the read lands directly in the caller's buffer. `CLIO_GETBLOB_PRIV_MODE`
 *     unset / "inline".
 *   - Runtime started SEPARATE (its own process): a pure client cannot expose
 *     private memory, so the read is staged through an SHM buffer, the task is
 *     marked TASK_DATA_OWNER, and GetBlobTask::PostWait() copies the staged
 *     bytes into the caller's buffer. `CLIO_GETBLOB_PRIV_MODE=separate`.
 *
 * Both configs additionally exercise the zero-IPC shared-cache fast path when
 * the cache is attachable.
 *
 * The buffers handed to the API here are genuine private memory
 * (std::vector<char> / stack), never SHM — that is the whole point of the API.
 */

#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "clio_cte/core/core_client.h"
#include "clio_runtime/bdev/bdev_client.h"
#include "clio_runtime/clio_runtime.h"
#include "runtime_server.h"
#include "simple_test.h"

using namespace std::chrono_literals;

namespace {

constexpr clio::run::u64 kRamTargetBytes = 256ULL * 1024 * 1024;
constexpr unsigned kPort = 10623;
const char *kTargetName = "getblob_priv_target";

/** True when CLIO_GETBLOB_PRIV_MODE=separate: bring up a daemon in its own
 *  process and attach as a pure client (exercises the client-staging path).
 *  Otherwise the runtime is started inline (exercises the direct-write path). */
bool SeparateMode() {
  const char *m = std::getenv("CLIO_GETBLOB_PRIV_MODE");
  return m != nullptr && std::string(m) == "separate";
}

/** Run `clio_run <args>` to completion, returning its exit code. */
int RunCli(const std::vector<std::string> &args, int timeout_sec) {
  std::vector<std::string> full;
  full.push_back(CLIO_RUN_EXE);
  full.insert(full.end(), args.begin(), args.end());
  std::vector<char *> argv;
  for (auto &a : full) argv.push_back(a.data());
  argv.push_back(nullptr);
  pid_t pid = fork();
  if (pid < 0) return -1;
  if (pid == 0) {
    execv(argv[0], argv.data());
    _exit(127);
  }
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
  int status = 0;
  while (true) {
    pid_t r = waitpid(pid, &status, WNOHANG);
    if (r == pid) return WIFEXITED(status) ? WEXITSTATUS(status) : -2;
    if (std::chrono::steady_clock::now() >= deadline) {
      kill(pid, SIGKILL);
      waitpid(pid, &status, 0);
      return -3;
    }
    std::this_thread::sleep_for(100ms);
  }
}

clio::run::test::RuntimeServer *g_server = nullptr;

/** Deterministic byte at logical position i of the test pattern. */
char PatternByte(size_t i) { return static_cast<char>((i * 131 + 7) & 0xFF); }

class Fixture {
 public:
  bool initialized_ = false;
  bool separate_ = false;

  Fixture() {
    separate_ = SeparateMode();
    if (separate_) {
      initialized_ = InitSeparate();
    } else {
      initialized_ = InitInline();
    }
  }

  /** Register a RAM target into the pool CLIO_CTE_CLIENT actually talks to
   *  (the default clio_cte_core pool). Shared by the inline and separate
   *  fixtures so both place the target where the client's PutBlobs look for it.
   *  In separate mode the bdev create + RegisterTarget are serviced by the
   *  daemon just as they are in-process inline. */
  static bool RegisterRamTarget() {
    auto *cte = CLIO_CTE_CLIENT;
    clio::run::PoolId bdev_pool_id(915, 0);
    clio::run::bdev::Client bdev_client(bdev_pool_id);
    auto create = bdev_client.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                          kTargetName, bdev_pool_id,
                                          clio::run::bdev::BdevType::kRam,
                                          kRamTargetBytes);
    create.Wait();
    auto reg = cte->AsyncRegisterTarget(kTargetName,
                                        clio::run::bdev::BdevType::kRam,
                                        kRamTargetBytes,
                                        clio::run::PoolQuery::Local(),
                                        bdev_pool_id);
    reg.Wait();
    return reg->GetReturnCode() == 0;
  }

  /** Inline: this process IS the runtime. Register a RAM target by hand. */
  bool InitInline() {
    if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
      return false;
    }
    std::this_thread::sleep_for(300ms);
    if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
      return false;
    }
    std::this_thread::sleep_for(200ms);

    if (!RegisterRamTarget()) return false;
    // Best-effort: attach the SHM metadata cache so the zero-IPC path can run.
    (void)CLIO_CTE_CLIENT->AttachShmCache();
    return true;
  }

  /** Separate: compose a real daemon in its own process, attach as a client. */
  bool InitSeparate() {
    const std::string work = "/tmp/clio_getblob_priv_test";
    std::filesystem::remove_all(work);
    std::filesystem::create_directories(work);
    const std::string yaml = work + "/compose.yaml";
    {
      std::ofstream f(yaml);
      f << "compose:\n"
           "  - mod_name: clio_cte_core\n"
           "    pool_name: \"getblob_priv_cte\"\n"
           "    pool_query: local\n"
           "    pool_id: \"515.0\"\n"
           "    storage:\n"
           "      - path: "
        << work << "/ram_dev\n"
           "        bdev_type: ram\n"
           "        capacity_limit: "
        << (kRamTargetBytes / (1024 * 1024)) << "mb\n"
           "    dpe:\n"
           "      dpe_type: random\n";
    }

    setenv("CLIO_WAIT_SERVER", "15", 1);
    setenv("CLIO_BIND_ADDR", "127.0.0.1", 1);

    static clio::run::test::RuntimeServer server;
    g_server = &server;
    if (!server.Start(kPort) || !server.WaitForReady()) {
      return false;
    }
    if (RunCli({"compose", "start", yaml}, 60) != 0) {
      return false;
    }
    if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
      return false;
    }
    if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
      return false;
    }
    // Register the RAM target into the pool this client actually uses.
    //
    // The compose above brings up the separate daemon, but its `storage:`
    // section registers targets into the COMPOSED pool (getblob_priv_cte, id
    // 515.0), whereas CLIO_CTE_CLIENT talks to the default CTE pool
    // (clio_cte_core, id 512.0) that CLIO_CTE_CLIENT_INIT creates. Those are
    // different containers with independent target lists, so the client's
    // PutBlobs would find no placement target -- ExtendBlob reports "no
    // targets" (error_code 1) and the setup put fails with rc 11. On a fast
    // native host the ids happened to line up so it passed; on a slow /
    // CPU-constrained host (the deps-cpu CI container under the Boost fiber
    // backend) they did not, so the test failed deterministically there.
    //
    // Registering through the client (exactly as InitInline does) puts the
    // target where the client looks for it, on every platform. The daemon
    // services the bdev create + RegisterTarget the same way it would inline.
    if (!RegisterRamTarget()) {
      return false;
    }
    (void)CLIO_CTE_CLIENT->AttachShmCache();
    return true;
  }
};

Fixture *g_fixture = nullptr;

/** Put `n` bytes of the deterministic pattern (starting at pattern index
 *  `base`) at [off, off+n) of `blob`, through an ordinary SHM PutBlob. */
void PutPattern(clio::cte::core::TagId tag_id, const std::string &blob,
                clio::run::u64 off, size_t n, size_t base) {
  auto *ipc = CLIO_IPC;
  auto *cte = CLIO_CTE_CLIENT;
  ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(n);
  REQUIRE(!buf.IsNull());
  for (size_t i = 0; i < n; ++i) buf.ptr_[i] = PatternByte(base + i);
  auto p = cte->AsyncPutBlob(tag_id, blob, off, n, ctp::ipc::ShmPtr<>(buf.shm_));
  p.Wait();
  REQUIRE(p->GetReturnCode() == 0);
  ipc->FreeBuffer(buf);
}

}  // namespace

// The heart of the issue: read into PRIVATE memory and get the right bytes,
// under whichever runtime configuration the fixture chose.
TEST_CASE("GetBlob into private memory returns correct bytes", "[cte][823]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);
  auto *cte = CLIO_CTE_CLIENT;

  clio::cte::core::Tag tag("getblob_priv_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();

  const std::string blob = "priv_blob";
  const size_t kN = 64 * 1024;  // 64 KiB
  PutPattern(tag_id, blob, 0, kN, 0);

  // Full read into a genuinely private buffer (heap, not SHM).
  std::vector<char> priv(kN, 0);
  auto g = cte->AsyncGetBlob(tag_id, blob, 0, kN, /*flags=*/0, priv.data());
  g.Wait();  // Empty (cache-hit) future Wait()s to true; otherwise task rc==0.
  for (size_t i = 0; i < kN; ++i) {
    REQUIRE(priv[i] == PatternByte(i));
  }

  // Offset read of a sub-range lands the right slice at the buffer start.
  const size_t kOff = 4096, kLen = 8192;
  std::vector<char> sub(kLen, 0);
  auto g2 = cte->AsyncGetBlob(tag_id, blob, kOff, kLen, 0, sub.data());
  g2.Wait();
  for (size_t i = 0; i < kLen; ++i) {
    REQUIRE(sub[i] == PatternByte(kOff + i));
  }
}

// Repeated reads must not leak or hang: the client path allocates a staging
// buffer per call and relies on TASK_DATA_OWNER to reclaim it. A leak here
// exhausts the main SHM segment and later reads fail; a missed free would trip
// the allocator's leak checker at shutdown.
TEST_CASE("Private GetBlob is leak-free under repetition", "[cte][823]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);
  auto *cte = CLIO_CTE_CLIENT;

  clio::cte::core::Tag tag("getblob_priv_loop_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();

  const std::string blob = "loop_blob";
  const size_t kN = 32 * 1024;
  PutPattern(tag_id, blob, 0, kN, 11);

  for (int iter = 0; iter < 256; ++iter) {
    std::vector<char> priv(kN, 0);
    auto g = cte->AsyncGetBlob(tag_id, blob, 0, kN, 0, priv.data());
    g.Wait();
    // Spot-check a few positions each iteration (full compare would dominate
    // runtime); correctness of the whole buffer is covered by the case above.
    REQUIRE(priv[0] == PatternByte(11));
    REQUIRE(priv[kN / 2] == PatternByte(11 + kN / 2));
    REQUIRE(priv[kN - 1] == PatternByte(11 + kN - 1));
  }
}

// The zero-IPC shared-cache fast path, when the cache is attachable, must
// deliver identical bytes (it copies straight out of the RAM bdev's segment
// and returns an empty, already-satisfied future).
TEST_CASE("Private GetBlob shared-cache fast path is correct", "[cte][823]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);
  auto *cte = CLIO_CTE_CLIENT;
  if (!cte->HasShmCache() && !cte->AttachShmCache()) {
    // Cache unavailable in this configuration; the direct/staged paths are
    // covered by the cases above. Nothing to assert here.
    return;
  }

  clio::cte::core::Tag tag("getblob_priv_cache_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();
  const std::string blob = "cache_blob";
  const size_t kN = 16 * 1024;
  PutPattern(tag_id, blob, 0, kN, 99);

  std::vector<char> priv(kN, 0);
  auto g = cte->AsyncGetBlob(tag_id, blob, 0, kN, 0, priv.data());
  g.Wait();
  for (size_t i = 0; i < kN; ++i) {
    REQUIRE(priv[i] == PatternByte(99 + i));
  }
}

int main(int argc, char **argv) {
  static Fixture fixture;
  g_fixture = &fixture;
  std::string filter = (argc > 1) ? argv[1] : "";
  int rc = SimpleTest::run_all_tests(filter);
  clio::run::CLIO_RUNTIME_FINALIZE();
  return rc;
}

/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * REPLICATION PERSISTENCE STRESS (issue #886)
 *
 * Two-mode program orchestrated by test_repl_persist.sh, which reboots the
 * runtime between phases:
 *   --put-blobs    Phase 1: CachedPut 50 blobs through the replication
 *                  chimod (DRAM primary + persistent disk replica each),
 *                  verify both copies, flush metadata.
 *   --verify-blobs Phase 2 (after runtime reboot): RestartContainers, then
 *                  verify the DRAM primaries are GONE (volatile blocks are
 *                  filtered at replay) while every disk replica is intact
 *                  and byte-correct — and that CachedGet transparently
 *                  serves from the replica and re-populates the DRAM cache.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <clio_ctp/util/logging.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/admin/admin_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/replication/replication_client.h>

static constexpr int kNumBlobs = 50;
static constexpr clio::run::u64 kBlobSize = 4096;
static const char* kTagName = "repl_persist_tag";

static std::string BlobName(int i) {
  return "persist_blob_" + std::to_string(i);
}

/** Deterministic per-blob fill byte. */
static char PatternByte(int i) { return static_cast<char>('A' + (i % 26)); }

/** Phase 1: CachedPut kNumBlobs blobs, verify both copies, flush. */
static int PutBlobs() {
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    HLOG(kError, "Phase 1: Failed to init client");
    return 1;
  }

  clio::cte::replication::Client repl(
      clio::cte::replication::kReplicationPoolId, clio::cte::core::kCtePoolId);
  clio::cte::core::Client &cte = repl;  // inherited core API

  auto tag_task = cte.AsyncGetOrCreateTag(kTagName);
  tag_task.Wait();
  clio::cte::core::TagId tag_id = tag_task->tag_id_;

  for (int i = 0; i < kNumBlobs; ++i) {
    ctp::ipc::FullPtr<char> buf = CLIO_IPC->AllocateBuffer(kBlobSize);
    if (buf.IsNull()) {
      HLOG(kError, "Phase 1: SHM allocation failed for blob {}", i);
      return 1;
    }
    memset(buf.ptr_, PatternByte(i), kBlobSize);

    auto put = repl.AsyncCachedPut(tag_id, BlobName(i), 0, kBlobSize,
                                   ctp::ipc::ShmPtr<>(buf.shm_));
    put.Wait();
    clio::run::u32 rc = put->GetReturnCode();
    clio::run::u32 reps = put->replicas_written_;
    CLIO_IPC->FreeBuffer(buf);
    if (rc != 0 || reps != 1) {
      HLOG(kError, "Phase 1: CachedPut failed for blob {} rc={} replicas={}",
           i, rc, reps);
      return 1;
    }
  }

  // Both copies of every blob must exist before the reboot.
  for (int i = 0; i < kNumBlobs; ++i) {
    auto psz = cte.AsyncGetBlobSize(tag_id, BlobName(i));
    psz.Wait();
    auto rsz = cte.AsyncGetBlobSize(tag_id, BlobName(i),
                                    clio::run::PoolQuery::Dynamic(), 1);
    rsz.Wait();
    if (psz->GetReturnCode() != 0 || psz->size_ != kBlobSize ||
        rsz->GetReturnCode() != 0 || rsz->size_ != kBlobSize) {
      HLOG(kError,
           "Phase 1: size check failed for blob {} primary={} replica={}",
           i, psz->size_, rsz->size_);
      return 1;
    }
  }

  // One-shot metadata flush: snapshot (including the type-3 replica entries)
  // + WAL sync, so the reboot has durable metadata regardless of timing.
  auto flush_meta = cte.AsyncFlushMetadata(clio::run::PoolQuery::Local(), 0);
  flush_meta.Wait();

  HLOG(kSuccess, "Phase 1: SUCCESS - {} blobs cached in DRAM + disk", kNumBlobs);
  return 0;
}

/** Phase 2 (post-reboot): verify the disk replicas and the re-cache path. */
static int VerifyBlobs() {
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    HLOG(kError, "Phase 2: Failed to init client");
    return 1;
  }

  // Recreate the composed pools from their saved restart configs.
  clio::run::admin::Client admin_client(clio::run::kAdminPoolId);
  auto restart_task =
      admin_client.AsyncRestartContainers(clio::run::PoolQuery::Local());
  restart_task.Wait();
  if (restart_task->GetReturnCode() != 0) {
    HLOG(kError, "Phase 2: RestartContainers failed rc={}",
         restart_task->GetReturnCode());
    return 1;
  }
  HLOG(kInfo, "Phase 2: RestartContainers restarted {} containers",
       restart_task->containers_restarted_);

  clio::cte::replication::Client repl(
      clio::cte::replication::kReplicationPoolId, clio::cte::core::kCtePoolId);
  clio::cte::core::Client &cte = repl;

  auto tag_task = cte.AsyncGetOrCreateTag(kTagName);
  tag_task.Wait();
  clio::cte::core::TagId tag_id = tag_task->tag_id_;

  ctp::ipc::FullPtr<char> buf = CLIO_IPC->AllocateBuffer(kBlobSize);
  if (buf.IsNull()) {
    HLOG(kError, "Phase 2: SHM allocation failed");
    return 1;
  }

  int verified = 0;
  for (int i = 0; i < kNumBlobs; ++i) {
    // The DRAM primary must be GONE: volatile blocks are filtered at
    // metadata replay. This is the premise of the whole feature — the cache
    // copy dies with the machine, the replica does not.
    auto psz = cte.AsyncGetBlobSize(tag_id, BlobName(i));
    psz.Wait();
    if (psz->GetReturnCode() != 0 || psz->size_ != 0) {
      HLOG(kError, "Phase 2: blob {} primary survived reboot?! size={} rc={}",
           i, psz->size_, psz->GetReturnCode());
      return 1;
    }
    // The disk replica must be fully intact.
    auto rsz = cte.AsyncGetBlobSize(tag_id, BlobName(i),
                                    clio::run::PoolQuery::Dynamic(), 1);
    rsz.Wait();
    if (rsz->GetReturnCode() != 0 || rsz->size_ != kBlobSize) {
      HLOG(kError, "Phase 2: blob {} replica lost! size={} rc={}", i,
           rsz->size_, rsz->GetReturnCode());
      return 1;
    }
    // Byte-for-byte via a direct replica read.
    {
      memset(buf.ptr_, 0, kBlobSize);
      clio::cte::core::Context ctx;
      ctx.replica_ = 1;
      auto get = cte.AsyncGetBlob(tag_id, BlobName(i), 0, kBlobSize,
                                  /*flags=*/0, ctp::ipc::ShmPtr<>(buf.shm_),
                                  clio::run::PoolQuery::Dynamic(), ctx);
      get.Wait();
      if (get->GetReturnCode() != 0) {
        HLOG(kError, "Phase 2: blob {} replica read failed rc={}", i,
             get->GetReturnCode());
        return 1;
      }
      for (clio::run::u64 b = 0; b < kBlobSize; ++b) {
        if (buf.ptr_[b] != PatternByte(i)) {
          HLOG(kError, "Phase 2: blob {} replica byte {} corrupt", i, b);
          return 1;
        }
      }
    }
    // And the cache path heals itself: CachedGet serves from the replica
    // and re-populates the DRAM primary.
    {
      memset(buf.ptr_, 0, kBlobSize);
      auto get = repl.AsyncCachedGet(tag_id, BlobName(i), 0, kBlobSize,
                                     ctp::ipc::ShmPtr<>(buf.shm_));
      get.Wait();
      if (get->GetReturnCode() != 0 || get->from_replica_ != 1 ||
          get->recached_bytes_ != kBlobSize ||
          buf.ptr_[0] != PatternByte(i)) {
        HLOG(kError,
             "Phase 2: blob {} CachedGet failed rc={} from={} recached={}",
             i, get->GetReturnCode(), get->from_replica_,
             get->recached_bytes_);
        return 1;
      }
      auto psz2 = cte.AsyncGetBlobSize(tag_id, BlobName(i));
      psz2.Wait();
      if (psz2->size_ != kBlobSize) {
        HLOG(kError, "Phase 2: blob {} primary not re-cached (size={})", i,
             psz2->size_);
        return 1;
      }
    }
    verified++;
  }
  CLIO_IPC->FreeBuffer(buf);

  HLOG(kSuccess,
       "Phase 2: SUCCESS - {}/{} disk replicas survived the reboot intact "
       "and re-cached into DRAM",
       verified, kNumBlobs);
  return 0;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    HLOG(kError, "Usage: {} [--put-blobs|--verify-blobs]", argv[0]);
    return 1;
  }
  std::string mode(argv[1]);
  if (mode == "--put-blobs") return PutBlobs();
  if (mode == "--verify-blobs") return VerifyBlobs();
  HLOG(kError, "Unknown mode '{}'", mode);
  return 1;
}

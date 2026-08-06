/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#ifndef CLIO_CTE_FILESYSTEM_FILESYSTEM_CLIENT_H_
#define CLIO_CTE_FILESYSTEM_FILESYSTEM_CLIENT_H_

#include <clio_cte/core/core_client.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <clio_cte/filesystem/filesystem_tasks.h>
#include <clio_cte/filesystem/shm_fs_cache.h>

namespace clio::cte::filesystem {

/**
 * Filesystem client — the single API every interceptor (POSIX, STDIO,
 * libfuse, HDF5 VFD, MPI-IO) calls. Inherits the full CTE core client (so all
 * blob/tag/target operations remain available) and adds POSIX-shaped
 * filesystem operations routed through the filesystem chimod, which owns the
 * path->tag mapping, page-blob I/O, and per-file logical-size metadata.
 */
class Client : public clio::cte::core::Client {
 public:
  Client() = default;
  explicit Client(const clio::run::PoolId &fs_pool_id) { Init(fs_pool_id); }

#if CTP_IS_HOST
  // =========================================================================
  // issue #817: shared-memory attribute cache (client side).
  //
  // Same contract as the CTE core's cache: strictly an OPTIMIZATION, every
  // accessor reports failure rather than guessing, and `false` means "ask the
  // runtime", never "the file does not exist".
  // =========================================================================

  /** Attach this filesystem pool's cache root via the segment directory. */
  bool AttachShmCache() {
    shm_fs_root_ = nullptr;
    auto *ipc = CLIO_CPU_IPC;
    if (ipc == nullptr) {
      return false;
    }
    auto *alloc = ipc->GetMetadataAllocator();
    auto *dir = ipc->GetMetadataDirectory();
    if (alloc == nullptr || dir == nullptr) {
      // No metadata segment (e.g. TCP client, remote node).
      HLOG(kDebug, "[#817] AttachShmCache: no metadata segment (alloc={}, dir={})",
           static_cast<const void *>(alloc), static_cast<const void *>(dir));
      return false;
    }
    clio::run::u64 root_off = dir->FindRoot(pool_id_.ToU64());
    if (root_off == 0) {
      // Either this pool is not caching, or its chimod has not registered its
      // root YET -- a client that raced pool creation must be able to attach
      // later, which is why callers retry rather than latching a failure.
      HLOG(kDebug, "[#817] AttachShmCache: no root for fs pool {} ({} entries)",
           pool_id_.ToString(), dir->num_entries_);
      return false;
    }
    auto *root = reinterpret_cast<ShmFsCacheRoot *>(
        reinterpret_cast<char *>(alloc) + root_off);
    // Refuse anything unrecognized: the cache is derived state, so declining
    // is always safe, while guessing at a layout is not.
    if (root->ready_ != 1 ||
        root->version_ != ShmFsCacheRoot::kLayoutVersion) {
      return false;
    }
    shm_fs_root_ = root;
    return true;
  }

  bool HasShmCache() const { return shm_fs_root_ != nullptr; }

  /**
   * Zero-IPC path lookup.
   *
   * @return true if a consistent record was read. false means "not cached /
   *         could not read consistently" -- fall back to the RPC path.
   */
  bool TryGetFileRecordShm(const std::string &path, ShmFileRecord *out) const {
    if (shm_fs_root_ == nullptr || out == nullptr) {
      return false;
    }
    return shm_fs_root_->path_to_file_.TryGetBytes(path.data(), path.size(),
                                                   out);
  }
#endif  // CTP_IS_HOST

#if CTP_IS_HOST
  /** Create/initialize the filesystem container over a CTE core pool. */
  clio::run::Future<CreateTask> AsyncCreate(const clio::run::PoolQuery &pool_query,
                                      const std::string &pool_name,
                                      const clio::run::PoolId &custom_pool_id,
                                      const FilesystemConfig &params) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<CreateTask>(
        clio::run::CreateTaskId(), clio::run::kAdminPoolId, pool_query,
        FilesystemConfig::chimod_lib_name, pool_name, custom_pool_id, this,
        params);
    return ipc->Send(task);
  }

  clio::run::Future<OpenTask> AsyncOpen(const std::string &path, clio::run::u32 flags,
                                  clio::run::u32 mode = 0644) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<OpenTask>(clio::run::CreateTaskId(), pool_id_,
                                       clio::run::PoolQuery::Local(), path, flags,
                                       mode);
    return ipc->Send(task);
  }

  clio::run::Future<CloseTask> AsyncClose(clio::run::u64 handle) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<CloseTask>(clio::run::CreateTaskId(), pool_id_,
                                        clio::run::PoolQuery::Local(), handle);
    return ipc->Send(task);
  }

  clio::run::Future<ReadTask> AsyncRead(clio::run::u64 handle, clio::run::u64 offset,
                                  clio::run::u64 size, ctp::ipc::ShmPtr<> data) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<ReadTask>(clio::run::CreateTaskId(), pool_id_,
                                       clio::run::PoolQuery::Local(), handle, offset,
                                       size, data);
    return ipc->Send(task);
  }

  clio::run::Future<WriteTask> AsyncWrite(clio::run::u64 handle, clio::run::u64 offset,
                                    clio::run::u64 size, ctp::ipc::ShmPtr<> data) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<WriteTask>(clio::run::CreateTaskId(), pool_id_,
                                        clio::run::PoolQuery::Local(), handle, offset,
                                        size, data);
    return ipc->Send(task);
  }

  clio::run::Future<AppendTask> AsyncAppend(clio::run::u64 handle, clio::run::u64 size,
                                      ctp::ipc::ShmPtr<> data) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<AppendTask>(clio::run::CreateTaskId(), pool_id_,
                                         clio::run::PoolQuery::Local(), handle, size,
                                         data);
    return ipc->Send(task);
  }

  clio::run::Future<GetattrTask> AsyncGetattr(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<GetattrTask>(clio::run::CreateTaskId(), pool_id_,
                                          clio::run::PoolQuery::Local(), path);
    return ipc->Send(task);
  }

  clio::run::Future<TruncateTask> AsyncTruncate(const std::string &path,
                                          clio::run::u64 new_size) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<TruncateTask>(clio::run::CreateTaskId(), pool_id_,
                                           clio::run::PoolQuery::Local(), path,
                                           new_size);
    return ipc->Send(task);
  }

  clio::run::Future<UnlinkTask> AsyncUnlink(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<UnlinkTask>(clio::run::CreateTaskId(), pool_id_,
                                         clio::run::PoolQuery::Local(), path);
    return ipc->Send(task);
  }

  clio::run::Future<UtimensTask> AsyncUtimens(const std::string &path,
                                              clio::run::u64 atime_ns,
                                              clio::run::u64 mtime_ns,
                                              clio::run::u32 flags) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<UtimensTask>(clio::run::CreateTaskId(), pool_id_,
                                          clio::run::PoolQuery::Local(), path,
                                          atime_ns, mtime_ns, flags);
    return ipc->Send(task);
  }

  clio::run::Future<ChownTask> AsyncChown(const std::string &path,
                                          clio::run::u32 uid,
                                          clio::run::u32 gid,
                                          clio::run::u32 mode = 0xFFFFFFFFu) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<ChownTask>(clio::run::CreateTaskId(), pool_id_,
                                        clio::run::PoolQuery::Local(), path, uid,
                                        gid, mode);
    return ipc->Send(task);
  }

  // chmod reuses the ChownTask (per-file mode override) with uid/gid left
  // unchanged, avoiding a separate RPC method for a single stored field.
  clio::run::Future<ChownTask> AsyncChmod(const std::string &path,
                                          clio::run::u32 mode) {
    return AsyncChown(path, 0xFFFFFFFFu, 0xFFFFFFFFu, mode);
  }

  clio::run::Future<MkdirTask> AsyncMkdir(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<MkdirTask>(clio::run::CreateTaskId(), pool_id_,
                                        clio::run::PoolQuery::Local(), path);
    return ipc->Send(task);
  }

  clio::run::Future<RmdirTask> AsyncRmdir(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<RmdirTask>(clio::run::CreateTaskId(), pool_id_,
                                        clio::run::PoolQuery::Local(), path);
    return ipc->Send(task);
  }

  clio::run::Future<RenameTask> AsyncRename(const std::string &src,
                                      const std::string &dst) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<RenameTask>(clio::run::CreateTaskId(), pool_id_,
                                         clio::run::PoolQuery::Local(), src, dst);
    return ipc->Send(task);
  }

  clio::run::Future<LinkTask> AsyncLink(const std::string &target,
                                  const std::string &link) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<LinkTask>(clio::run::CreateTaskId(), pool_id_,
                                       clio::run::PoolQuery::Local(), target, link);
    return ipc->Send(task);
  }

  clio::run::Future<SymlinkTask> AsyncSymlink(const std::string &target,
                                              const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<SymlinkTask>(clio::run::CreateTaskId(), pool_id_,
                                          clio::run::PoolQuery::Local(), target,
                                          path);
    return ipc->Send(task);
  }

  clio::run::Future<ReadlinkTask> AsyncReadlink(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<ReadlinkTask>(clio::run::CreateTaskId(), pool_id_,
                                           clio::run::PoolQuery::Local(), path);
    return ipc->Send(task);
  }

  clio::run::Future<SetxattrTask> AsyncSetxattr(const std::string &path,
                                                const std::string &name,
                                                const std::string &value,
                                                clio::run::u32 flags) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<SetxattrTask>(clio::run::CreateTaskId(), pool_id_,
                                           clio::run::PoolQuery::Local(), path,
                                           name, value, flags);
    return ipc->Send(task);
  }

  clio::run::Future<GetxattrTask> AsyncGetxattr(const std::string &path,
                                                const std::string &name) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<GetxattrTask>(clio::run::CreateTaskId(), pool_id_,
                                           clio::run::PoolQuery::Local(), path,
                                           name);
    return ipc->Send(task);
  }

  clio::run::Future<ListxattrTask> AsyncListxattr(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<ListxattrTask>(clio::run::CreateTaskId(), pool_id_,
                                            clio::run::PoolQuery::Local(), path);
    return ipc->Send(task);
  }

  clio::run::Future<RemovexattrTask> AsyncRemovexattr(const std::string &path,
                                                      const std::string &name) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<RemovexattrTask>(clio::run::CreateTaskId(),
                                              pool_id_,
                                              clio::run::PoolQuery::Local(),
                                              path, name);
    return ipc->Send(task);
  }

  // ---- deferred-append pipeline ----
  /** Kick off (or tick) the periodic local pending-append drain. */
  clio::run::Future<AppendSequenceTask> AsyncAppendSequence(
      double period_us = 0.0,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Local()) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<AppendSequenceTask>(clio::run::CreateTaskId(), pool_id_,
                                                 pool_query);
    if (period_us > 0.0) {
      task->SetPeriod(period_us, clio::run::kMicro);
      task->SetFlags(TASK_PERIODIC);
    }
    return ipc->Send(task);
  }

  /** Collect one tag's pending appends at its sequencer (ManyToOne batch). */
  clio::run::Future<AppendCollectTask> AsyncAppendCollect(
      const clio::cte::core::TagId &tag_id,
      const std::vector<AppendEntry> &entries, const clio::run::PoolQuery &pool_query) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<AppendCollectTask>(clio::run::CreateTaskId(), pool_id_,
                                                pool_query, tag_id, entries);
    return ipc->Send(task);
  }

  /** Plan + dispatch one tag's batch (suspendable; submitted by AppendCollect). */
  clio::run::Future<AppendPlanTask> AsyncAppendPlan(
      const clio::cte::core::TagId &tag_id,
      const std::vector<AppendEntry> &entries, const clio::run::PoolQuery &pool_query) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<AppendPlanTask>(clio::run::CreateTaskId(), pool_id_,
                                             pool_query, tag_id, entries);
    return ipc->Send(task);
  }

  /** Apply a slice of the merge plan (GetBlob->PutBlob->DelBlob). */
  clio::run::Future<AppendExecutionTask> AsyncAppendExecution(
      const clio::cte::core::TagId &tag_id,
      const clio::cte::core::TagId &staging_tag_id,
      const std::vector<AppendPlanStep> &steps, const clio::run::PoolQuery &pool_query) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<AppendExecutionTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id, staging_tag_id, steps);
    return ipc->Send(task);
  }

  clio::run::Future<ReaddirTask> AsyncReaddir(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<ReaddirTask>(clio::run::CreateTaskId(), pool_id_,
                                          clio::run::PoolQuery::Local(), path);
    return ipc->Send(task);
  }

  clio::run::Future<StatSizeTask> AsyncStatSize(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<StatSizeTask>(clio::run::CreateTaskId(), pool_id_,
                                           clio::run::PoolQuery::Local(), path);
    return ipc->Send(task);
  }

  // =========================================================================
  // Byte-oriented filesystem I/O with POSIX semantics (issues #817, #862).
  //
  // THIS is the interface an interceptor calls. An adapter (POSIX, STDIO,
  // MPI-IO, libfuse) should own nothing but its handle table and its seek
  // offsets: no pending-write queue, no staging allocator, no drain
  // bookkeeping, no fast-path of its own. Every one of those used to live in
  // the POSIX adapter, which meant the next adapter either duplicated them or
  // silently did without.
  //
  // The write-behind window is the CTE core's deferred-write registry, not a
  // private one — it already owns the in-flight FIFO, the per-key pending
  // table, the byte budget, the recycled staging pool (issue #892) and the
  // per-key error latch. Two consequences worth knowing:
  //
  //  - The byte budget is now GLOBAL rather than per-file. It bounds what it
  //    is actually protecting (shared-memory staging capacity), which a
  //    per-file bound never did: 100 files under a 64 MiB per-file window
  //    could pin 6.4 GiB.
  //  - Read-your-own-writes no longer WAITS. The registry keeps each in-flight
  //    write's staging bytes addressable, so a read overlapping one is served
  //    from them (DeferTryServe) and only a partially-covered read has to
  //    drain. The old adapter drained on any overlap.
  // =========================================================================

  /** Whether write(2) may return before the runtime has the bytes (issue
   *  #817). ON by default; CLIO_CFS_ASYNC_WRITES=0 restores blocking writes.
   *
   *  Measured on 128 x 64 KiB overwrite + fsync, five runs each, medians:
   *  queued 5.77 ms vs blocking 6.86 ms — "no worse, probably a little
   *  better", not a headline.
   *
   *  A NOTE ON HOW EASY THIS IS TO MEASURE WRONG: the first pass over a fresh
   *  file allocates every block and a second pass does not, so timing one mode
   *  on a new file and the other on the warmed one compares allocation against
   *  overwrite. Doing exactly that produced a confident, entirely spurious
   *  "async is 2.3x SLOWER". Warm first; time only overwrites. */
  static bool AsyncWritesEnabled() {
    static const bool v = [] {
      if (const char *e = std::getenv("CLIO_CFS_ASYNC_WRITES")) {
        return !(std::string(e) == "0" || std::string(e) == "false");
      }
      return true;
    }();
    return v;
  }

  /** Deferred writes allowed in flight before a write blocks on the oldest.
   *
   *  This is the bound that actually binds for small I/O -- see
   *  AwaitPutsUntilSpace for why a byte-only window lets 4 KiB writes queue
   *  16k deep and lose 1.59x throughput to page-token contention. 256 was the
   *  best of {256, 512, 1024, 2048, 4096, 16384} measured at 4 KiB, and is
   *  well above what large I/O ever reaches (a 1 MiB writer hits the 64 MiB
   *  byte bound at 64 in flight), so it costs large writes nothing.
   *  CLIO_CFS_WRITE_WINDOW_COUNT overrides. */
  static clio::run::u64 WriteWindowCount() {
    static const clio::run::u64 v = [] {
      if (const char *e = std::getenv("CLIO_CFS_WRITE_WINDOW_COUNT")) {
        char *end = nullptr;
        unsigned long long n = std::strtoull(e, &end, 10);
        if (end != e && n > 0) return static_cast<clio::run::u64>(n);
      }
      return static_cast<clio::run::u64>(256);
    }();
    return v;
  }

  /** Staging bytes allowed in flight before a write blocks on the oldest.
   *  Back-pressure, never a failure. CLIO_CFS_WRITE_WINDOW_BYTES overrides. */
  static clio::run::u64 WriteWindowBytes() {
    static const clio::run::u64 v = [] {
      if (const char *e = std::getenv("CLIO_CFS_WRITE_WINDOW_BYTES")) {
        char *end = nullptr;
        unsigned long long n = std::strtoull(e, &end, 10);
        if (end != e && n > 0) return static_cast<clio::run::u64>(n);
      }
      return static_cast<clio::run::u64>(64ULL * 1024 * 1024);
    }();
    return v;
  }

  /** Registry key for a file. Writes and reads of one path must agree on it,
   *  and it is what makes read-your-own-writes a property of the FILE rather
   *  than of a handle — two descriptors on one file see each other's writes. */
  static clio::run::u64 FileKey(const std::string &path) {
    return DeferKeyHashName(path);
  }

  /**
   * write(2) at an explicit offset. Copies `buf` into pooled shared memory
   * (mandatory: the caller may reuse its buffer the instant write(2) returns),
   * submits, and — unless `sync` — registers the write as deferred and
   * returns without waiting.
   *
   * ALWAYS SUCCEEDS when deferred: a queued write that fails latches its
   * errno against the path for Flush/close to report, exactly as the kernel
   * page cache reports writeback failures to fsync rather than to an
   * unrelated later write(2).
   *
   * @param sync O_SYNC/O_DSYNC — wait for completion. The caller asked for
   *        durability over latency and honouring that is the point of the flag.
   * @return bytes accepted, or -1 with errno set.
   */
  ssize_t Write(clio::run::u64 handle, const std::string &path,
                clio::run::u64 off, const void *buf, size_t count,
                bool sync = false) {
    if (count == 0) {
      return 0;
    }
    const bool blocking = sync || !AsyncWritesEnabled();
    // Recycle completed writes' staging so this burst feeds its own pool
    // instead of allocating fault-cold buffers (issue #892 measured cold
    // allocation as THE submit bottleneck). Opportunistic: it gives up rather
    // than queue when another submitter is already reaping, which is what
    // keeps this off the critical path as thread count rises.
    const bool prof = ProfileEnabled();
    auto tick = [] { return std::chrono::steady_clock::now(); };
    auto t0 = prof ? tick() : std::chrono::steady_clock::time_point{};
    DeferReapCompletedIfUncontended();
    ctp::ipc::FullPtr<char> staging = PoolTryAlloc(count);
    if (staging.IsNull()) {
      // Pool miss. NOW it is worth reaping completed writes -- that refills
      // the pool with pre-faulted buffers, and a fresh allocation is the
      // expensive thing we are trying to avoid (issue #892 measured cold
      // allocation as THE submit bottleneck). Doing this on every write
      // instead, hit or miss, is what made the global registry mutex the
      // busiest lock in the client.
      DeferReapCompleted();
      staging = PoolAllocStaging(count);
    }
    if (staging.IsNull()) {
      // Draining hands those buffers straight back, so a write that would
      // have failed with ENOMEM waits instead — the same trade the window
      // bound already makes, triggered by the allocator rather than a count.
      AwaitPutsUntilSpace(0);
      staging = PoolAllocStaging(count);
      if (staging.IsNull()) {
        errno = ENOMEM;
        return -1;
      }
    }
    auto t1 = prof ? tick() : std::chrono::steady_clock::time_point{};
    std::memcpy(staging.ptr_, buf, count);
    auto t2 = prof ? tick() : std::chrono::steady_clock::time_point{};
    auto fut = AsyncWrite(handle, off, count,
                          ctp::ipc::ShmPtr<>(staging.shm_));
    auto t3 = prof ? tick() : std::chrono::steady_clock::time_point{};
    if (blocking) {
      fut.Wait();
      ssize_t ret;
      if (fut->GetReturnCode() == 0) {
        ret = static_cast<ssize_t>(fut->bytes_written_);
      } else {
        errno = EIO;
        ret = -1;
      }
      PoolFreeStaging(staging, count);
      return ret;
    }
    // Deferred. The staging pointer is handed to the registry as the write's
    // readable extent, which is what lets a subsequent read of these bytes be
    // served without waiting for the task.
    //
    // Note what is NOT ordered: two writes to overlapping bytes with no
    // intervening flush race the runtime's write token, and a client-side
    // wait cannot fix it (measured: 512 rewrites of one range lose the last
    // write ~30% of runs even WITH a same-range wait). An application needing
    // strict last-writer-wins across overlapping unsynced writes must flush
    // between them or open O_SYNC. Disjoint writes — every sequential or
    // random writer — are unaffected.
    DeferRegisterWrite(fut, FileKey(path), off, count, staging.ptr_, staging,
                       count);
    auto t4 = prof ? tick() : std::chrono::steady_clock::time_point{};
    // Enforce the window AFTER accounting, so this write is included in the
    // bound rather than being the one that overshoots it. This can block —
    // that is the back-pressure — but it cannot fail.
    AwaitPutsUntilSpace(WriteWindowBytes(), WriteWindowCount());
    if (prof) {
      auto t5 = tick();
      using ns = std::chrono::nanoseconds;
      SubmitProfile &p = Profile();
      p.n.fetch_add(1, std::memory_order_relaxed);
      p.stage_ns.fetch_add(std::chrono::duration_cast<ns>(t1 - t0).count(),
                           std::memory_order_relaxed);
      p.copy_ns.fetch_add(std::chrono::duration_cast<ns>(t2 - t1).count(),
                          std::memory_order_relaxed);
      p.send_ns.fetch_add(std::chrono::duration_cast<ns>(t3 - t2).count(),
                          std::memory_order_relaxed);
      p.reg_ns.fetch_add(std::chrono::duration_cast<ns>(t4 - t3).count(),
                         std::memory_order_relaxed);
      p.window_ns.fetch_add(std::chrono::duration_cast<ns>(t5 - t4).count(),
                            std::memory_order_relaxed);
    }
    return static_cast<ssize_t>(count);
  }

  /**
   * read(2) at an explicit offset, with read-your-own-writes.
   *
   * Three tiers, cheapest first: bytes still in flight are copied out of the
   * pending write's staging (no wait, no IPC); committed bytes come from the
   * shared-memory mirror (no IPC); anything else is one RPC to the chimod.
   *
   * @return bytes read (possibly 0 at EOF), or -1 with errno set.
   */
  ssize_t Read(clio::run::u64 handle, const std::string &path,
               clio::run::u64 off, void *buf, size_t count) {
    if (count == 0) {
      return 0;
    }
    const clio::run::u64 key = FileKey(path);
    int served = DeferTryServe(key, off, static_cast<char *>(buf), count);
    if (served > 0) {
      // Fully covered by in-flight writes: those bytes ARE the current value,
      // and the file is necessarily at least this long.
      return static_cast<ssize_t>(count);
    }
    if (served == -1) {
      // PARTIAL coverage: the bytes outside the in-flight writes are stale in
      // the file until those writes land, so this is the one case that has to
      // wait. -2 means writes are in flight for this file but none touches
      // these bytes -- the overwhelmingly common case for a mixed workload --
      // and waiting there would let any concurrent writer stall every reader
      // of the same file.
      DeferAwaitKey(key);
    }
    ssize_t fast = TryReadShm(path, off, buf, count);
    if (fast >= 0) {
      return fast;
    }
    auto *ipc = CLIO_IPC;
    ctp::ipc::FullPtr<char> shm = ipc->AllocateBuffer(count);
    if (shm.IsNull()) {
      errno = ENOMEM;
      return -1;
    }
    auto t = AsyncRead(handle, off, count, ctp::ipc::ShmPtr<>(shm.shm_));
    t.Wait();
    ssize_t ret;
    if (t->GetReturnCode() == 0) {
      size_t got = static_cast<size_t>(t->bytes_read_);
      if (got > 0) {
        std::memcpy(buf, shm.ptr_, got);
      }
      ret = static_cast<ssize_t>(got);
    } else {
      errno = EIO;
      ret = -1;
    }
    ipc->FreeBuffer(shm);
    return ret;
  }

  /**
   * fsync(2): wait for every deferred write to `path` and report — once —
   * any failure they latched. Before deferred writes this was a no-op,
   * because every write blocked.
   */
  int Flush(const std::string &path) {
    const clio::run::u64 key = FileKey(path);
    DeferAwaitKey(key);
    int err = DeferTakeKeyError(key);
    if (err != 0) {
      errno = err;
      return -1;
    }
    return 0;
  }

  /** Wait for every deferred write this client has outstanding, on any path.
   *  For teardown and for benchmarks that must time a complete flush. */
  static void FlushAll() { AwaitPutsUntilSpace(0); }

  /**
   * stat(2)-shaped query: existence and logical size.
   *
   * Drains `path` first — a queued write has not updated the file's logical
   * size yet, and stat(2) must report the size a completed write(2)
   * established. The drain does NOT consume a latched error: stat is not one
   * of the two calls (fsync, close) that owns that report.
   *
   * @return false when the query itself failed (ask again); *exists says
   *         whether the file is there.
   */
  bool GetAttr(const std::string &path, bool *exists, clio::run::u64 *size) {
    const clio::run::u64 key = FileKey(path);
    // Serve from the shared-memory mirror when nothing is in flight for this
    // file. stat(2) is the most frequent metadata call a filesystem sees, and
    // the record the runtime already publishes carries both answers -- paying
    // a round trip for them made stat cost ~325 us against ~4 us for a
    // fast-pathed 4 KiB read, i.e. metadata became the bottleneck of a
    // workload whose DATA path had stopped doing IPC at all.
    //
    // The pending-write check is what keeps this honest: a queued write has
    // not advanced the published size yet, so with anything in flight we take
    // the slow path (drain, then ask the runtime) exactly as before. Same
    // staleness argument as TryReadShm otherwise -- the runtime publishes the
    // new size at the end of the Write handler, so it lands before the
    // write(2) that grew the file returns to its caller.
    if (HasShmCache() || AttachShmCache()) {
      ShmFileRecord rec;
      if (TryGetFileRecordShm(path, &rec)) {
        const clio::run::u64 published = rec.Exists() ? rec.size_ : 0;
        // The published size is already correct unless something in flight
        // reaches PAST the current end -- an overwrite cannot change a file's
        // size, and overwrites are what a steady-state workload mostly does.
        // Draining on any pending write instead (the obvious version of this
        // check) left stat at ~264 us because it waited on writes that could
        // not possibly have affected the answer.
        if (DeferMaxPendingEnd(key) <= published) {
          *exists = rec.Exists();
          *size = published;
          ShmStatHits().fetch_add(1, std::memory_order_relaxed);
          return true;
        }
      }
    }
    ShmStatMisses().fetch_add(1, std::memory_order_relaxed);
    DeferAwaitKey(key);
    auto t = AsyncGetattr(path);
    t.Wait();
    if (t->GetReturnCode() != 0) {
      return false;
    }
    *exists = (t->exists_ != 0);
    *size = t->size_;
    return true;
  }

  /** Logical size of `path` (0 when absent). See GetAttr for the drain. */
  bool GetSize(const std::string &path, clio::run::u64 *size) {
    bool exists = false;
    if (!GetAttr(path, &exists, size)) {
      return false;
    }
    if (!exists) {
      *size = 0;
    }
    return true;
  }

  /**
   * Zero-IPC read of COMMITTED bytes straight out of shared memory (issue
   * #817). Resolves the path to its tag + logical size in this pool's SHM
   * mirror, then copies each page blob out of the RAM bdev segment via the
   * core's payload fast path.
   *
   * Distinct from the deferred-registry fast path in Read(), which serves
   * bytes that have NOT been committed yet: the two cover disjoint cases and
   * a read may need either.
   *
   * @return bytes read (possibly 0 at EOF), or -1 meaning "not
   *         fast-pathable" — NOT an error. -1 covers every refusal: cache
   *         absent, path not mirrored, pending appends, a hole, a page on a
   *         file/remote/GPU tier, or placement moving mid-copy. The caller
   *         must then use the RPC path, which is always correct.
   */
  ssize_t TryReadShm(const std::string &path, clio::run::u64 off, void *buf,
                     size_t count) {
    // Attach lazily, and RETRY when not yet attached, rather than latching
    // the result of one attempt at init. A client can legitimately come up
    // before its pool has been composed (or before the chimod has registered
    // its cache root), and a one-shot attach would leave that process on the
    // RPC path forever -- silently, since a disabled cache and a working one
    // look identical from the outside. Retrying costs a scan of a <=32-entry
    // directory, against an RPC that costs ~100 us.
    if (!HasShmCache() && !AttachShmCache()) {
      ShmReadMisses().fetch_add(1, std::memory_order_relaxed);
      return -1;
    }
    auto *cte = CLIO_CTE_CLIENT;
    if (cte == nullptr || (!cte->HasShmCache() && !cte->AttachShmCache())) {
      // Page payloads come from the core cache; both are required.
      ShmReadMisses().fetch_add(1, std::memory_order_relaxed);
      return -1;
    }

    ShmFileRecord rec;
    if (!TryGetFileRecordShm(path, &rec) || !rec.IsFastPathable()) {
      ShmReadMisses().fetch_add(1, std::memory_order_relaxed);
      return -1;
    }

    // Clamp to the LOGICAL size. This is the number the chimod owns and the
    // one POSIX read() semantics are defined against; the tag's physical byte
    // total would be wrong after a sparse write or an ftruncate-grow.
    //
    // Reading a stale (too small) size here cannot produce a short read for a
    // properly ordered caller: the runtime publishes the new size at the end
    // of the Write handler, so it lands before the write() that grew the file
    // returns to its caller.
    if (off >= rec.size_) {
      return 0;  // at or past EOF
    }
    clio::run::u64 want = count;
    if (off + want > rec.size_) {
      want = rec.size_ - off;
    }

    char *dst = static_cast<char *>(buf);
    clio::run::u64 done = 0;
    clio::run::u64 cur = off;
    while (done < want) {
      clio::run::u64 page_off = cur % kFsPageSize;
      clio::run::u64 to_read = kFsPageSize - page_off;
      if (to_read > want - done) {
        to_read = want - done;
      }
      if (!cte->TryReadBlobShm(rec.tag_id_, PageName(cur), dst + done,
                               static_cast<size_t>(to_read),
                               static_cast<size_t>(page_off))) {
        // Abandon the WHOLE request rather than mixing sources. A hole reads
        // as zeros and a missing page is indistinguishable from an uncached
        // one, so the only safe reading of a failure here is "let the runtime
        // answer".
        ShmReadMisses().fetch_add(1, std::memory_order_relaxed);
        return -1;
      }
      done += to_read;
      cur += to_read;
    }
    // Say so, once, the first time a read is actually served from shared
    // memory. "The fast path is on" is otherwise unobservable from outside --
    // a disabled cache and a working one differ only in latency -- and the
    // failure this whole path was chasing was precisely a fast path that was
    // silently off everywhere. Operators need something to grep for; the
    // counters below are for anyone who needs the RATE rather than the fact.
    static std::once_flag announced;
    std::call_once(announced, [] {
      HLOG(kInfo,
           "clio-fs: serving reads from shared memory (zero-IPC fast path "
           "active)");
    });
    ShmReadHits().fetch_add(1, std::memory_order_relaxed);
    return static_cast<ssize_t>(done);
  }

  // Zero-IPC read accounting. A silently-disabled fast path and a working one
  // differ only in latency, which is exactly the failure mode that is easiest
  // to misdiagnose as "the runtime is slow" -- so make the rate observable.
  static std::atomic<clio::run::u64> &ShmReadHits() {
    static std::atomic<clio::run::u64> v{0};
    return v;
  }
  static std::atomic<clio::run::u64> &ShmReadMisses() {
    static std::atomic<clio::run::u64> v{0};
    return v;
  }
  static std::atomic<clio::run::u64> &ShmStatHits() {
    static std::atomic<clio::run::u64> v{0};
    return v;
  }
  static std::atomic<clio::run::u64> &ShmStatMisses() {
    static std::atomic<clio::run::u64> v{0};
    return v;
  }

  // ---- submit-path profile (CLIO_CFS_PROFILE_SUBMIT=1) --------------------
  // A deferred write does not wait for completion, so single-thread write
  // throughput IS one over the submit cost -- measured at ~12.6 us for 4 KiB,
  // which is far more than the ~0.3 us the payload copy can explain. These
  // counters attribute it. Off by default: five clock reads is ~125 ns, small
  // against 12.6 us but not free.
  struct SubmitProfile {
    std::atomic<clio::run::u64> n{0};
    std::atomic<clio::run::u64> stage_ns{0};   // acquire staging buffer
    std::atomic<clio::run::u64> copy_ns{0};    // memcpy payload
    std::atomic<clio::run::u64> send_ns{0};    // build + send the task
    std::atomic<clio::run::u64> reg_ns{0};     // register in the registry
    std::atomic<clio::run::u64> window_ns{0};  // enforce the in-flight window
  };
  static SubmitProfile &Profile() {
    static SubmitProfile p;
    return p;
  }
  static bool ProfileEnabled() {
    static const bool v = [] {
      const char *e = std::getenv("CLIO_CFS_PROFILE_SUBMIT");
      return e != nullptr && std::string(e) != "0";
    }();
    return v;
  }
#endif  // CTP_IS_HOST

#if CTP_IS_HOST
 private:
  // Resolved address of the filesystem cache root in THIS process, or nullptr
  // when caching is unavailable. Not owned: the runtime owns the segment and
  // may drop the cache at any time, which is why every read is validated.
  ShmFsCacheRoot *shm_fs_root_ = nullptr;
#endif
};

// Process-wide filesystem client singleton (reuses the clio_cte export macro).
// Declared inside the namespace so it resolves as
// clio::cte::filesystem::g_fs_client (matching CLIO_CFS_CLIENT below and the
// definition in filesystem_client.cc), exactly like core's g_cte_client.
CLIO_CTE_DEFINE_GLOBAL_PTR_VAR_H(clio::cte::filesystem::Client, g_fs_client);

/** Initialize the filesystem client singleton (creates/binds the pool). */
bool CLIO_CFS_CLIENT_INIT(const std::string &config_path = "",
                          const clio::run::PoolQuery &pool_query =
                              clio::run::PoolQuery::Dynamic());

}  // namespace clio::cte::filesystem

#define CLIO_CFS_CLIENT                                  \
  (&(*CTP_GET_GLOBAL_PTR_VAR(clio::cte::filesystem::Client, \
                             clio::cte::filesystem::g_fs_client)))

#endif  // CLIO_CTE_FILESYSTEM_FILESYSTEM_CLIENT_H_

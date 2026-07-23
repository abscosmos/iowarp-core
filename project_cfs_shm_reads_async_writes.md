# clio-fs over the CTE shared-memory cache + async writes (issue #817)

Status: **PLANNED** (no code yet)
Branch: `817-cfs-shm-reads-async-writes`, cut from `origin/dev` @ 2dc4911d

Companion issue: #818 — compressed blobs in the SHM cache (out of scope here,
but it constrains §4.3).

## 1. Problem

Every `read()` and `write()` through clio-fs pays a blocking client→runtime
round-trip, even when the bytes live in a node-local RAM bdev that the calling
process has already mapped.

Read path today — **two dispatches per page**:

1. `CfsIo::DoRead` (`context-transfer-engine/adapter/cfs/cfs_io.cc:53`)
   allocates a staging buffer, calls `AsyncRead`, and **blocks** on `t.Wait()`.
2. `Runtime::Read`
   (`context-transfer-engine/filesystem/src/filesystem_runtime.cc:351`) loops
   over 1 MiB pages issuing `cte_.AsyncGetBlob(tag_id, PageName(cur), ...)` —
   a second dispatch per page, potentially to a different container.
3. The client `memcpy`s out of the staging buffer and frees it.

Write path today — `CfsIo::DoWrite` (`cfs_io.cc:79`) copies into a staging
buffer and blocks on `t.Wait()`. `CfsIo::Sync` is a no-op that documents
"writes are synchronous" (`cfs_io.h:136`).

Measured floors from #783: SHM transport round trip ~72 µs, a real CTE
`GetBlob` RPC ~90–120 µs, versus **0.207 µs** for a direct shared-memory
payload read of a 4 KiB blob. A cached small `pread()` through clio-fs costs
roughly two round trips where it should cost one `memcpy`.

## 2. The #783 fast path is dead outside its own tests

`Tag::GetBlob` already attempts it (`core/src/tag.cc:194`):

```cpp
auto *cte_client = CLIO_CTE_CLIENT;
if (cte_client->HasShmCache() &&
    cte_client->TryReadBlobShm(tag_id_, blob_name, data, data_size, off)) {
  return;
}
```

`HasShmCache()` is always false in production. The **only** callers of
`AttachShmCache()` in the tree are `test/unit/test_core_functionality.cc:362`,
`:2754`, `:2912`. `ContentTransferEngine::ClientInit`
(`core/src/content_transfer_engine.cc:46`) creates/binds the pool, assigns
`cte_client->pool_id_`, and never attaches.

So phase 0 of this work is one call — and it benefits every adapter (FUSE,
HDF5 VFD, MPI-IO, `gpu_vector`), not just clio-fs.

## 3. What makes the client-side read path possible

Everything the client needs to build a cache key is derivable without asking
the runtime:

- **Tag name == the stripped path.** `Runtime::Open`
  (`filesystem_runtime.cc:268`) resolves the tag with
  `cte_.AsyncGetOrCreateTag(path, ...)`, so `TryGetTagIdShm(path)` returns the
  same `TagId` the chimod uses.
- **Page blob names are arithmetic.** `PageName(off)` is
  `std::to_string(off / kFsPageSize)` with `kFsPageSize = 1 MiB`
  (`filesystem_tasks.h:40`, `filesystem_runtime.cc:57`).
- **The CTE core client is already up.** `CLIO_CFS_CLIENT_INIT`
  (`filesystem/src/filesystem_client.cc`) calls
  `clio::cte::core::CLIO_CTE_CLIENT_INIT()` first, and the fs pool is layered
  over `kCtePoolId`, so `CLIO_CTE_CLIENT` is bound to exactly the pool whose
  cache root the client must attach.

## 4. Design

### 4.1 Read

```
CfsIo::Pread(fd, buf, count, off)
  ├─ dirty-page overlap with in-flight writes?     → RPC path (ordered behind them)
  ├─ file has pending deferred appends?            → RPC path
  ├─ logical size unknown / read crosses EOF?      → RPC path
  └─ per page in [off, off+count):
       TryReadBlobShm(tag_id, PageName(cur), dst, n, page_off)
         any failure → abandon the whole request, take the RPC path
```

Fallback is **per request, not per page**: a partially-fast-pathed read has to
reason about which bytes came from where, and the RPC already handles the whole
range correctly. Simplicity wins over the marginal case.

`TryReadBlobShm` already refuses non-local, non-RAM, GPU-tier, block-list-
truncated and placement-moved blobs, and validates `placement_gen_` before and
after the copy. Nothing in this design weakens that; a `false` always means
"use RPC", never "hole" or "EOF".

### 4.2 Logical size — the one real gap

EOF clamping and hole zeroing use `FileInfo::size_`
(`filesystem_runtime.cc:359`, `:370-384`), which is owned by the fs chimod and
is **not** the tag's physical size (`ShmTagRecord::total_size_`). They diverge
after `ftruncate`-grow, sparse writes, and deferred appends. Reading a hole
must produce zeros, and reading past EOF must produce a short read — getting
this from the wrong number is a correctness bug, not a perf regression.

**Chosen: publish the fs logical size into shared memory.** The
`MetadataDirectory` registration mechanism is already generic
(`RegisterRoot`/`FindRoot`, added in #783 phase 4), so the filesystem chimod
gets its own root and mirrors `path → {tag_id, logical_size, mode, uid, gid,
atime, mtime, ctime, flags}` on every change, best-effort and always *after*
the authoritative update so the cache can only lag.

This subsumes a second win: `getattr`/`stat` become zero-IPC. Today `ls -l`
costs one RPC per file (`CfsIo::StatPath` → `QueryGetattr` → `AsyncGetattr`).

Rejected alternative: have the client track size from `Open` plus its own
writes and revalidate by RPC when a read would cross the believed EOF. Cheap,
but silently wrong the moment a second process writes the file.

### 4.3 Write

Keep the staging copy (the caller's buffer is reusable on return, so the copy
is mandatory), drop the wait:

- `DoWrite` parks the `Future<WriteTask>` + its staging buffer in a
  per-descriptor in-flight list and returns `count` immediately.
- **Bounded window.** Configurable caps on in-flight bytes and count; on
  exceeding either, reap completions until under the limit. This is the
  back-pressure that keeps staging memory bounded — an unbounded queue turns a
  `dd` into an OOM.
- `fsync`/`fdatasync`/`close` become real: drain, free buffers, surface errors.
- **Sticky error latch.** A failed async write cannot set `errno` on the
  `write()` that queued it. Latch the error on the descriptor and report it
  from the next `write()`/`fsync()`/`close()` — the same contract as kernel
  page-cache writeback. Document it in `cfs_io.h` where the "writes are
  synchronous" comment lives today.
- `O_SYNC`/`O_DSYNC` keep today's synchronous behaviour.
- **RYOW.** Track dirty page ranges per descriptor; a read overlapping one
  takes the RPC path (ordered behind the queued write) or is served from the
  staging buffer. This is the per-client pending-write set that #783 §5.4
  anticipated but never needed, because writes stayed synchronous.

### 4.4 Interactions to respect

- **Deferred appends.** `Runtime::Append` (`filesystem_runtime.cc:456`) stages
  bytes under `staging_tag_id_` and merges later via
  AppendSequence→AppendCollect→AppendExecution. Until a batch is sequenced the
  file's pages are not stably addressable, and `fi->size_` is explicitly
  best-effort. A file with pending appends must not fast-path.
- **Compressed blobs.** `ShmBlobRecord` carries no compression state and
  `kShmBlobDirectReadable` does not consult it (#818). Out of scope here; this
  design assumes raw page bytes and must be revisited if #818 lands option (B).
- **DataOrganizer.** Handled by the existing `placement_gen_` bracket — no new
  machinery.

## 5. Phasing

| Phase | Deliverable | Exit criterion |
|---|---|---|
| 0 | `ClientInit` attaches the SHM cache | ordinary client reports `HasShmCache()==true` against a composed runtime |
| 1 | fs chimod publishes per-path attrs (size/mode/times) to SHM | attach from an unrelated process, read a live file's size |
| 2 | zero-IPC `getattr`/`stat` in `CfsIo` | `stat()` latency before/after; miss falls back |
| 3 | zero-IPC page reads in `Read`/`Pread` | 4 KiB cached `pread` latency; forced-miss test covers fallback |
| 4 | async writes + bounded window + real `fsync`/`close` | write throughput before/after; error latch test |
| 5 | RYOW dirty-page tracking | write-then-read same offset, with and without `fsync` |
| 6 | regression sweep | `cfs_distributed`, `scripts/xfstests` subset, FUSE + HDF5-VFD suites |

Phases 0–2 are independently valuable and independently mergeable; 3 depends on
1; 5 must land with or before 4 is enabled by default.

## 6. Benchmarks to report (all timings in ms)

- 4 KiB cached `pread` latency through the adapter, SHM vs RPC, same query.
- Sequential 1 MiB `write` throughput, sync vs async, at several window sizes.
- `stat()` latency, and `ls -l` wall time over a 1000-file directory.
- Write-then-read cycle — #783 measured only ~2x there because the cycle is
  write-dominated. Async writes attack exactly that half, so the cycle number
  is the one that shows whether the two halves compose.

## 7. Open questions

- Window sizing default, and whether it is per-descriptor or per-process.
- Whether `fsync` should also flush the fs chimod's own append pipeline, or
  only this descriptor's queued writes.
- Whether the fs attr mirror should be a separate map or reuse the CTE tag map
  with an fs-specific record (separate map is cleaner; costs another fixed
  capacity to size).

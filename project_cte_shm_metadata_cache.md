# CTE Shared-Memory Metadata Cache (issue #783)

Status: design / not yet implemented
Branch: `783-cte-shm-metadata-cache`

## 1. Problem

Blobs are arbitrary-sized, so the fixed-size page-cache trick does not apply. Every
metadata lookup is a runtime round-trip, and the round-trip is expensive no matter
which transport we pick. Measured on a 6-core WSL host (`clio_run_thrpt_bench
--test-case latency`, 4 reps, medians):

| Transport | 1 thread | 4 threads | 8 threads |
|---|---|---|---|
| SHM | 70.6 µs | 145.3 µs | 240.5 µs |
| IPC (unix socket) | 124.5 µs | 175.5 µs | 297.1 µs |
| TCP (loopback) | 471.7 µs | 574.8 µs | 691.5 µs |

The floor is ~71 µs and it degrades with concurrency. No hardware extension helps.
The only way to make small reads cheap is to remove the round-trip entirely.

**Target: node-local RAM-resident `GetBlob` in < 5 µs with zero IPC.**

## 2. Model

- The **runtime owns** all shared metadata. It is the only writer.
- **Clients are strictly read-only**, mapping the segment `PROT_READ`.
- **Reads are synchronous** (in-process SHM lookup). **Writes are fully asynchronous**
  (write-through to the runtime; the SHM copy is updated by the runtime afterward).
- Clients are **untrusted for liveness**: any client may be `SIGKILL`ed at any
  instant, including mid-read. Nothing the runtime does may ever block on a client.

That last point is the load-bearing constraint. It rules out the obvious designs.

## 3. What already exists (and what does not)

Useful groundwork already in the tree:

- `ctp::ipc::vector<AllocT>` and `ShmContainer<AllocT>` —
  `context-transport-primitives/include/clio_ctp/data_structures/ipc/`
- A segment/allocator pattern: `backend.shm_init(alloc_id, size, name)` →
  `MakeAlloc<T>()` on the runtime, `AttachAlloc<T>()` on the client
  (`context-runtime/src/ipc_manager.cc:839-945`)
- `MemorySegment` enum with `kMainSegment` / `kClientDataSegment` / `kQueueSegment`
  (`context-runtime/include/clio_runtime/types.h:596`)
- `TagInfo`/`BlobInfo` are **already allocator-parameterized** (`priv::string`,
  `priv::vector`, `CLIO_PRIV_ALLOC`) — they are not hard-wired to `new`/`delete`.

Missing, and must be built:

- **`ipc::string`** — does not exist.
- **`ipc::unordered_map`** — does not exist. Only `priv::unordered_map_ll`
  (malloc-backed, lock-taking) and `priv::unordered_map_lhash` exist.
- Any read-only / lock-free lookup path.
- SHM-backed RAM bdev. Today `MemBdevTransport` uses `new char[1 GiB]` pages
  (`mem_bdev_transport.cc:80`).

Note `CLIO_PRIV_ALLOC` expands to `CTP_MALLOC` on host (`types.h:551`) — the `priv::*`
containers are allocator-aware but currently bound to malloc. Retargeting them is
plausible; it is *not* free, because the map values are `std::shared_ptr`.

## 4. Architecture

### 4.1 Metadata segment

New `kMetadataSegment` in the `MemorySegment` enum. 8 GB fixed size, created at
context-runtime startup, **runtime-wide** (not CTE-owned — other modules will want
it). Not pre-faulted: `shm_open` + `ftruncate(8 GB)` + `mmap` is lazily populated, so
only touched pages cost RAM. `IpcManager` creates it on the server path and attaches
it read-only on the client path, exposing it next to `main_allocator_`.

### 4.2 Containers

- `ipc::string` — offset-based, SSO optional, no heap pointers.
- `ipc::unordered_map` — open addressing with tombstones, power-of-two capacity,
  per-slot generation counter. Chosen over chaining because a lock-free reader can
  probe a flat array safely; chasing chain pointers while the writer recycles nodes
  needs epoch protection on *every* node.
- Reuse `ipc::vector` for `blocks_` and `aliases_`.

All internal references are **segment-relative offsets**, never raw pointers — the
segment maps at different base addresses in each process.

### 4.3 Data structures

`ShmTagInfo` / `ShmBlobInfo` mirror the existing structs with SHM-safe members.
Notable conversions:

| Current | Becomes | Note |
|---|---|---|
| `priv::string blob_name_` | `ipc::string` | |
| `priv::vector<BlobBlock> blocks_` | `ipc::vector<BlobBlock>` | verify `BlobBlock` is POD-safe; it embeds a `bdev::Client` |
| `ctp::Mutex prealloc_lock_` | runtime-side only | clients never take it; must not be on the RO read path |
| `std::shared_ptr<T>` in maps | offset + generation | see §5.1 — this is the hard part |

### 4.4 Client read path

```
GetBlob(tag, name)
  -> seqlock-read blob slot from SHM map
  -> all blocks node-local && kRam?
       yes -> memcpy from SHM RAM bdev, re-validate generation, return
       no  -> fall back to existing async RPC path
```

Fallback is always available, which is what makes incremental rollout safe.

## 5. The hard problems

### 5.1 Reclamation without `shared_ptr`

`core_runtime.h:366` documents the current safety property explicitly: *"Values are
`std::shared_ptr`, so a concurrent erase just drops the map's reference while any
in-flight handle keeps the object alive — no use-after-free."* There are 61
`shared_ptr<TagInfo>`/`shared_ptr<BlobInfo>` sites. Shared memory has no `shared_ptr`,
and the readers are in other processes.

Proposal: **epoch-based reclamation.**
- Global epoch counter in SHM, bumped by the runtime.
- Each attached client publishes a per-client epoch slot (entering/leaving a read).
- The runtime defers freeing a slot until every live client has advanced past the
  epoch in which it was retired.
- **Liveness:** a client that dies inside a read section would pin the epoch forever.
  Each slot carries a PID + heartbeat; the runtime reaps slots whose owner is gone.
  Reaping must be conservative (a false reap is a use-after-free).

This is the single riskiest piece of the design and deserves its own standalone
test harness before it is wired into CTE.

### 5.2 Optimistic reads only

Because the runtime may never wait on a client, **no client-held lock may exist**.
All reads are seqlock-style: read sequence, read payload, re-read sequence, retry on
mismatch or odd value.

Direct consequence: **clients cannot call `unordered_map_ll::find()`** — it acquires
locks, i.e. it writes, which both blocks the runtime and faults a `PROT_READ` mapping.
The runtime's writer path and the client's reader path are therefore *different code*
over the *same* layout. That asymmetry needs to be explicit in the API, not implied.

### 5.3 Coherence against the DataOrganizer

The frecency organizer (`data_organizer/frecency_organizer.cc`) moves and evicts
blobs. Sequence that corrupts data:

1. Client reads `BlobInfo`, gets block offsets into the RAM bdev.
2. Runtime reorganizes the blob; blocks are freed and reused by another blob.
3. Client `memcpy`s from those offsets → **silently returns another blob's bytes.**

No error surfaces.

**Scope note:** locking only the *metadata copy-out* does not fix this. The hazard is
the **payload** read. Whatever mechanism we use must span the `memcpy` from the RAM
bdev, not just the `BlobInfo` field read.

#### Chosen design: generation for correctness, lease for progress

A per-blob **timed lease** (proposed: 500 ms) that the reorganizer respects is the
right shape, but it cannot be the *correctness* mechanism, for four reasons found in
the existing code:

1. **A lock requires the client to write.** Clients map `PROT_READ` (§5.5). Acquiring
   any mutex is a store. This forces the lock words into a **separate small
   read-write mapped region**, with the payload staying read-only. Consequence: a
   buggy or hostile client can corrupt lock state, and corrupting it to *unlocked*
   breaks safety rather than merely liveness. So the lock cannot be load-bearing.
2. **`ctp::Mutex` is a ticket lock and cannot be reclaimed.**
   `thread/lock/mutex.h:47` — `Lock()` does `lock_.fetch_add(1)` and spins until
   `head_ == tkt`. If a client dies holding it, `head_` never advances and *every*
   later waiter blocks forever. Forcibly advancing `head_` releases multiple waiters
   at once, destroying mutual exclusion. A reclaimable lease needs a different
   primitive: a CAS word carrying **owner PID + acquire timestamp**.
3. **The reorganizer is a coroutine on a runtime worker.**
   `ReorganizeBlobInternal` / `DynamicReorganize` return `TaskResume`, and
   `core_tasks.h:820-831` already warns that a thread-blocking lock "CANNOT be used
   here — it would deadlock the single worker the instant the holder suspends at a
   `co_await`." A 500 ms blocking wait would stall a whole worker; with 4 workers, a
   few stuck blobs wedge the runtime. **The reorganizer must `try_lock` and skip** —
   it is background work and is free to reorganize a different blob.
4. **Timeout-and-steal is probabilistic, not correct.** If the timeout fires on a
   *live but slow* client — page fault against an undersized `/dev/shm` (§5.7),
   descheduled, swapped — the reorganizer moves data under an active reader, which is
   exactly the corruption being prevented. 500 ms makes that rare; rare **plus
   silent** is the worst failure mode to ship.

Therefore:

- **Correctness** = a per-blob **generation counter**, validated before *and* after
  the payload copy. Requires no client writes, costs nothing on the read path, and
  stays correct even if the lease is stolen, corrupted, or skipped.
- **The lease** = a *progress* optimization, so readers do not livelock retrying while
  the organizer churns a hot blob.
- **Neither side ever blocks on the other.** Client: `try_lock` with a tiny budget,
  else fall back to RPC. Runtime: `try_lock`, else skip this blob this pass.
- **500 ms becomes the stale-lease janitor threshold, not a wait.** If a lease has
  been held longer than that, check the owner PID for liveness and reclaim only if it
  is gone — never as a blind timeout.

**Size cap.** The motivation is *small* I/O; for a 1 GB blob a 71 µs round-trip is
already noise. Restricting the fast path to blobs under ~1 MB bounds a client's
legitimate lease hold to well under 100 µs, which lets the janitor threshold be far
tighter than 500 ms and shrinks the damage a dead client can do. 500 ms is ~100,000x
the 5 µs target — it is sized for large-blob copies that should not use this path at
all. **Exact cap is TBD from measurement (open question 6).**

**Lock ordering:** `TagInfo` before `BlobInfo`, enforced identically on both sides.

### 5.4 Read-your-own-writes

Fully-async write-through means a client that does `PutBlob` then `GetBlob` can miss
its own write — which breaks the filesystem adapter's expectations.

Mitigation: a client-local `std::unordered_map` of **pending writes**, consulted
before every fast-path read; a hit forces the slow path until the write is
acknowledged. This is client-local plain heap — no SHM, no cross-process concern.

Two requirements that are easy to miss:
- It must track **deletes, truncates and renames**, not just puts. Otherwise a client
  reads a blob it just deleted straight out of the stale SHM cache.
- It must be consulted by **metadata-only** fast paths (`GetBlobSize`, `GetBlobScore`)
  too, not only payload reads — a stale size is just as wrong as stale bytes.

### 5.5 `PROT_READ` discipline

A stray write on the client read path is a `SIGSEGV`, not a wrong answer. No lock
acquisition, no lazy init, no refcount updates, no `operator[]` (which inserts).
Worth enforcing in CI with a test that maps the segment read-only and exercises every
client path.

### 5.6 Trust boundary — **needs an explicit decision**

A shared metadata segment lets *any* client read *every* tag and blob name/size on
the node. Today a client sees only what the runtime returns. This is a genuine change
to the isolation model. Options: accept it (single-tenant assumption), or partition
segments per tenant/pool. This should be decided deliberately and recorded, not
inherited by accident.

### 5.7 `/dev/shm` sizing

`/dev/shm` is **64 MB** on this dev host. An 8 GB sparse segment maps fine, but
touching past the tmpfs limit raises **`SIGBUS`**, not `ENOMEM` — a crash, not a
clean failure. We need (a) a graceful "cache full → fall back to RPC" path, and
(b) documented tmpfs sizing for realistic benchmarking.

## 6. Phasing

Dual-write is what keeps a 127-call-site refactor from becoming a flag-day.

| Phase | Work | Exit criterion |
|---|---|---|
| 0 | CTE metadata benchmark harness | Baseline recorded |
| 1 | `kMetadataSegment` + `IpcManager` plumbing | Runtime creates it; client attaches RO |
| 2 | `ipc::string`, `ipc::unordered_map`, reclaimable lease primitive + epoch reclamation | Standalone torture test, incl. random reader kills mid-lease |
| 3 | `ShmTagInfo`/`ShmBlobInfo`; runtime **dual-writes** | Old path still authoritative; all CTE tests green |
| 4 | Propagate map locations via `CreateTask` | Client holds valid handles |
| 5 | Client fast path for **metadata-only** ops (`GetBlobSize`, `GetBlobScore`) | Correctness parity; latency win measured |
| 6 | SHM RAM bdev + client payload fast path | `GetBlob` < 5 µs, zero IPC |
| 7 | Retire dual-write and legacy structures | Single source of truth |

Phases 0-2 are self-contained and carry essentially no risk to existing behavior.
Phase 3 is where the blast radius starts. Phase 5 before Phase 6 deliberately: proving
the read path on metadata-only ops is far cheaper to debug than on payload reads.

## 7. Open questions

1. Trust boundary (§5.6) — accept node-wide metadata visibility, or partition?
2. Epoch reclamation vs. hazard pointers vs. never-reclaim (bounded arena + compaction
   at quiesce)? Never-reclaim is dramatically simpler and may be adequate at 8 GB.
3. Do we need `TagId -> TagInfo` *and* `string -> TagId` in SHM, or can the client
   resolve names once and cache the ID privately?
4. `BlobBlock` embeds a `bdev::Client` — is that SHM-safe as-is, or does the SHM
   variant need a slimmed block descriptor?
5. Does `kPinned` (GPU page-locked) RAM bdev stay on the old private-heap path?
   `cudaMallocHost` memory cannot trivially be SHM-backed.
6. **Fast-path blob size cap** (§5.3) — measure where the SHM path stops beating the
   RPC path and set the cap there. Expected to land near 256 KB - 1 MB.
7. The reclaimable lease needs a **new lock primitive** (CAS word + owner PID +
   timestamp); `ctp::Mutex` cannot be reclaimed. Does it belong in
   `clio_ctp/thread/lock/` next to `Mutex`, or in the SHM-cache layer as a
   special-purpose type?

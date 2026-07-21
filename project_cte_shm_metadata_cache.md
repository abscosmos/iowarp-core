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

### 5.3b `TimedMutex` and `Lease` (replaces `shared_ptr`)

Decision: a reclaimable `TimedMutex` (`clio_ctp/thread/lock/timed_mutex.h`, beside
`Mutex`) plus a SHM smart pointer `Lease` (`clio_ctp/memory/smart_ptr/lease.h`, beside
`shared_ptr`/`unique_ptr`). A `Lease` acquires the target's `TimedMutex` on
construction and releases on destruction; because the lock is reclaimable, a client
that dies holding one is eventually serviced. Clients **copy metadata out and release
immediately** rather than holding a lease across work.

This replaces epochs/hazard pointers for the **lifetime** problem. It does *not*
replace the generation counter, which remains the **coherence** mechanism (§5.3).
They are complementary: the lease stops the object being freed under a reader; the
generation catches the payload being moved under a reader.

Four constraints on the implementation:

**1. Mapping is `PROT_READ|PROT_WRITE`; read-only is enforced by discipline.**
DECIDED: clients map the segment read-write, because acquiring a lease is a store and
splitting lock words into a separate region is not worth the layout complexity. **The
only bytes a client may write are lock words. Metadata is never client-writable.**

This trades a hardware guarantee for a convention, so two invariants become
load-bearing:
- The runtime **must never read the SHM cache as authoritative**. It keeps its own
  structures; the cache is derived state, and a client that scribbles on it can only
  harm other clients, never the runtime. (This is why Phase 7 keeps it a pure cache.)
- A corrupted cache must be **droppable and rebuildable** wholesale, since we can no
  longer rely on `PROT_READ` to prevent corruption from happening at all.

Acceptable because §7.1 already accepts a single-tenant trust model. Under
multi-tenancy this decision would have to be revisited along with §5.6.

**2. Default to a lock-free read; take a lease only when you need a stable view.**
An exclusive `TimedMutex` serializes concurrent readers of the same hot blob, and the
CAS ping-pongs that cache line across every reading core — the opposite of what a
read-mostly cache wants. So the common path is a **seqlock read that performs zero
stores** (read generation, copy, re-read generation, retry), and `Lease` is reserved
for operations needing a stable view for longer than a retry loop tolerates. Keeping
leases *rare* is also what keeps reclamation simple: an exclusive lock has a single
owner, so a dead holder is identifiable, which a shared/reader-counted lease would
not be (that would need per-holder tracking, i.e. epochs again by another name).

**What if a client dies mid-seqlock-read?** Nothing happens, and this is the whole
reason to prefer a seqlock for the default path. A seqlock *reader* acquires nothing
and stores nothing — it reads the sequence, copies, and re-reads the sequence. A dead
reader therefore leaves **no state to release, reclaim, or unwind**; there is no
counter to decrement and no lock to steal. It has copied garbage into its own address
space, but it is dead, so nobody consumes it. No cross-process effect whatsoever.

The seqlock's one death hazard is on the **writer** side: a writer that dies between
its two sequence bumps leaves the sequence odd forever, and readers retry forever. But
the writer is always the **runtime** — the single trusted owner — and if the runtime
dies the whole session is being torn down anyway. The asymmetry is exactly the one we
want: the untrusted-liveness side (clients) is stateless, and the stateful side is the
one process whose death is already fatal.

This is precisely why the seqlock is the *default* and `Lease` is the *exception*: a
lease is the only client-side construct whose abandonment needs servicing, so the
design minimizes how often one is held.

**3. Reclamation needs PID + process start time, not a bare timeout.** A timeout alone
cannot distinguish a dead holder from a live-but-slow one (§5.3, point 4). The lock
word carries `{owner_pid, process_start_time, acquire_timestamp}`; a waiter past the
threshold checks liveness and steals **only if the owner is genuinely gone**. Start
time is required because PIDs are recycled — on Linux, field 22 of `/proc/<pid>/stat`.
A steal must bump a steal counter so a resurrected holder can detect it lost the lock.

**4. The runtime may wait on a lease — but the *task* waits, never the *worker
thread*.** DECIDED: waiting is unavoidable, since a lease that the runtime could
ignore would not protect anything. The constraint is not *whether* to wait but *how*.

A thread-blocking wait is not available here. `ReorganizeBlobInternal` /
`DynamicReorganize` are coroutines on workers, and `core_tasks.h:820-831` already
documents why: a thread-blocking lock "CANNOT be used here — it would deadlock the
single worker the instant the holder suspends at a `co_await`." With 4 workers, a few
contended blobs would wedge the runtime — the exact failure class issue #781 exists to
prevent.

The codebase already has the right pattern, in that same comment: the `write_owner_`
contender "busy-polls via `co_await yield()` ... which is lost-wakeup-proof because it
re-checks every worker iteration." So:

- `Lease` acquisition on the runtime side is a **`co_await` retry loop**, not a
  blocking acquire. The task suspends; the worker goes and runs other tasks.
- The reorganizer, being pure background work, still prefers `try_lock`-and-skip — it
  has no obligation to reorganize any particular blob right now.
- Metadata writes wait via `co_await`, bounded by the `TimedMutex` threshold.

Two consequences to keep in view:
- The `TimedMutex` timeout now **bounds runtime write latency**, not just reclamation
  lag. A 500 ms threshold means a metadata write can stall that long behind a wedged
  client. This is the strongest argument for the §5.3 size cap: bounding legitimate
  lease hold time lets the threshold come down proportionally.
- A client holding a lease can now delay runtime progress, which is a denial-of-service
  vector even under a single-tenant model — a merely *buggy* client that leaks a lease
  is enough. The timeout is the backstop, and clients must never hold a lease across a
  syscall, an I/O, or anything else unbounded.

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
| 2 | `ipc::string`, `ipc::unordered_map`, `TimedMutex`, `Lease` | Standalone torture test, incl. random reader kills mid-lease |
| 3 | `ShmTagInfo`/`ShmBlobInfo`; runtime **dual-writes** | Old path still authoritative; all CTE tests green |
| 4 | Propagate map locations via `CreateTask` | Client holds valid handles |
| 5 | Client fast path for **metadata-only** ops (`GetBlobSize`, `GetBlobScore`) | Correctness parity; latency win measured |
| 6 | SHM RAM bdev + client payload fast path | `GetBlob` < 5 µs, zero IPC |
| 7 | Retire the *scaffolding*, keep SHM as a pure cache | Steady state (see below) |

**Phase 7 revised.** The original plan was to retire the legacy structures and make SHM
the single source of truth. That is now rejected, primarily because of §5.3b constraint
1: clients map the segment **read-write**, so a buggy client can corrupt it. Derived
state that can be dropped and rebuilt is a recoverable annoyance; *authoritative* state
that a client can corrupt is data loss. The runtime therefore keeps its own structures
permanently, and the SHM segment stays a **pure read cache** it may defer updating or
invalidate wholesale at any time.

(A secondary argument also applies: an authoritative cache would put client-held leases
on the critical path of every metadata write, rather than merely on cache freshness.
The runtime is willing to wait on leases — §5.3b constraint 4 — but there is no reason
to make correctness, rather than staleness, depend on that wait.)

Phase 7 therefore only removes benchmarking scaffolding and redundant bookkeeping, not
the legacy path.

Phases 0-2 are self-contained and carry essentially no risk to existing behavior.
Phase 3 is where the blast radius starts. Phase 5 before Phase 6 deliberately: proving
the read path on metadata-only ops is far cheaper to debug than on payload reads.

## 7. Decisions

1. **Trust boundary — DECIDED: accept node-wide metadata visibility.** Any client may
   read every tag/blob name and size on the node. Single-tenant assumption; revisit if
   multi-tenancy becomes a requirement.
2. **Reclamation — DECIDED: `Lease` + `TimedMutex` (§5.3b).** Not epochs, not hazard
   pointers, not never-reclaim. Lifetime is handled by a reclaimable lease; coherence
   stays with the generation counter.
3. **Both maps in SHM — DECIDED.** `string -> TagId` *and* `TagId -> TagInfo` (and the
   blob equivalents). No client-private ID caching scheme.
4. **`BlobBlock` — RESOLVED BY INSPECTION: a slimmed POD descriptor is required.**
   `BlobBlock` embeds a `bdev::Client`, which derives from `ContainerClient`
   (`container.h:590`). Its *data* is only `PoolId pool_id_` + `u32 return_code_`, but
   it declares `virtual void Init()` and `virtual ~ContainerClient()`, so the object
   **carries a vtable pointer**. A vptr is a process-local address (and moves under
   ASLR / differing .so load addresses), so it is meaningless — and dangerous — in a
   segment mapped by another process. The SHM block descriptor must therefore be plain
   POD: `{PoolId target_pool, PoolQuery target_query, u64 target_offset, u64 size,
   u64 capacity}`, with the `bdev::Client` reconstructed runtime-side on demand.
   **General rule for this work: no virtual functions in any type stored in SHM.**
5. **`kPinned` — DECIDED: stays on the private heap.** `cudaMallocHost` memory is not
   SHM-backed; the GPU-pinned RAM bdev keeps the existing path and simply does not
   participate in the client fast path.
6. **Fast-path blob size cap — measure in Phase 0/6.** Set it where the SHM path stops
   beating RPC; expected near 256 KB - 1 MB.
7. **Lock primitive — DECIDED: new `TimedMutex`** in `clio_ctp/thread/lock/timed_mutex.h`,
   alongside `mutex.h` / `rwlock.h` / `spin_lock.h` / `cvrwlock.h`.
8. **Smart pointer — DECIDED: new `Lease`** in `clio_ctp/memory/smart_ptr/lease.h`,
   alongside `shared_ptr.h` / `unique_ptr.h`. See §5.3b for its four constraints.

9. **Mapping — DECIDED: `PROT_READ|PROT_WRITE`.** Clients map read-write; only lock
   words are client-writable, enforced by convention rather than by the MMU. See
   §5.3b constraint 1 for the two invariants this makes load-bearing.
10. **Runtime waiting — DECIDED: the runtime may wait on leases.** Unavoidable, since
    an ignorable lease protects nothing. But it waits by `co_await` retry, never by
    blocking a worker thread (§5.3b constraint 4).

### Still open

- The exact fast-path size cap (item 6) — needs measurement, not a decision.
- Whether the `TimedMutex` threshold stays at 500 ms or drops once the size cap bounds
  legitimate hold time. This now matters more than it did: with the runtime waiting on
  leases (item 10), the threshold bounds **runtime metadata-write latency**, not just
  reclamation lag.

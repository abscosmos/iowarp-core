# Worker-local task batching (issue #820)

Status: **DESIGN** — no code yet. Branch `820-worker-task-batching` off `origin/dev` @ 04ebc438.

Companion to #817 (async writes, which surfaced the tail) and #680 (the per-blob
write-token contention this sidesteps).

## 1. Why

The clio-fs page model stores each 1 MiB file page as one blob. A sequential
4 KiB workload therefore puts **256 tasks on each page-blob**. Every `PutBlob`
holds that blob's #680 write token across its *entire* body — allocation **and**
the bdev data copy (`core_runtime.cc` `PutBlobImpl`, the `ModifyExistingData`
co_await at ~1678) — and a contender re-polls on a periodic cadence
(`kBlobWriteLockPollUs`, 1554). So the 256 writes drain single-file.

Measured, QD1, RAM target, 4 KiB: p50 **7.3 µs**, mean **~160 µs** → ~6.3K IOPS,
tail **p99.9 ≈ 10 ms, max ~1 s**. Block-size sweep pins the cause to
writes-per-blob: max stall **1082 ms** (4k, 256/page) → 10.5 ms (64k, 16/page) →
8 ms (1M, 1/page).

The 256 writes are to **disjoint byte ranges** of the page — they never conflict
on bytes — yet the token is per-*blob*, so they serialize anyway. The cheapest
place to fix that is *before* they reach the runtime: coalesce them into one
PutBlob over the union of their ranges. One token acquire, one bdev transfer,
256× fewer tasks.

## 2. Not the existing BatchManager

`BatchManager` (`batch_manager.{h,cc}`) already batches — but it is a **cross-node
collective** for `PoolQuery::ManyToOne`: park N tasks by
`(pool, method, container, batch_key)`, flush after a time window, combine their
inputs `AggregateIn` **N→1**, run **one** aggregate, then broadcast the single OUT
`1→N` back to every member (`OnAggregateComplete` → per-member `SetReturnCode` /
`SetCompleter` / `EndTask`). Every member gets the *same* result. Used today for
the clio-fs deferred-append pipeline (`AppendCollect`).

This proposal is different in kind: **intra-node coalescing**. N independent tasks
become a **minimal M-task subset** (not 1), and each output task completes a
*different* set of parents, scattering a *slice* of its OUT to each. Reduction vs
merge. We reuse BatchManager's completion-fan-out *mechanic* (its
`OnAggregateComplete` is the proof it works) but not its routing or its
single-result semantics.

## 3. Where it slots in

Worker main loop today (`worker.cc:276`): drain SHM shards → `ProcessNewTasks`
(pop ≤16, `RouteAndExec` each) → `ContinueBlockedTasks` → ManyToOne `FlushDue`.

`ProcessNewTask` pops one `Future<Task>`, resolves the container, and
`RouteAndExec` → `RouteTask`; on `RouteResult::ExecHere` it runs inline. Batching
inserts a phase between the dequeue and the execute, and only over tasks that
route **ExecHere** (a task destined for another node/container must route first,
untouched — we never batch across containers).

### The phase

```
BatchPhase(lane):
  groups := worker.batch_groups          // unordered_map<BatchKey, BatchQueue>, reused
  passthrough := worker.passthrough      // SPSC<Future<Task>>, reused

  for i in 0..MAX_BATCH_DEQUEUE (=64):
    if not lane.Pop(future): break
    route := RouteTask(future)
    if route != ExecHere:                // routed elsewhere; RouteTask already enqueued it
      continue
    container := container_of(future)
    container.BuildBatch(future.task, groups)   // routes into a group, OR...
      // default BuildBatch: passthrough.Push(future)   (not batchable)

  // run the non-batchable tasks as-is, in arrival order, first
  while passthrough.Pop(f): RouteAndExec(f, lane)   // (already ExecHere)

  // then let each container collapse its groups into minimal output tasks
  for each container with nonempty groups:
    container.SmashBatch(groups, sink)   // sink enqueues merged tasks + wires parents
  groups.clear()
```

Key decisions:

- **Bounded, backlog-only.** The `≤64` dequeue only ever *has* 64 to batch when a
  backlog exists. A lone write finds an empty lane after it, `BuildBatch`es a
  group of one, and `SmashBatch` emits it unchanged — **no added latency for the
  unbatched case**. Batching kicks in exactly when there is contention to amortize.
- **Passthrough first, merged last** (per the plan): preserves the arrival order
  of non-batchable work, and lets a merged task see a settled world.
- **Route before batch.** We only batch `ExecHere` tasks, so groups are always
  same-node, same-container — the merge never has to reason about placement.

## 4. Container interface

Two new virtuals on `Container` (`container.h`), default no-op / passthrough so
every existing container is unaffected:

```cpp
// Route `task` into the batch groups, or decline (caller sends it to passthrough).
// Default: return false (not batchable).
virtual bool BuildBatch(const shared_ptr<Task>& task, BatchGroups& groups) {
  return false;
}

// Collapse each group into the minimal set of output tasks. For each output,
// record the parent tasks it completes and how to scatter its OUT to them, then
// hand it to `sink` to enqueue. Default: no-op.
virtual void SmashBatch(BatchGroups& groups, BatchSink& sink) {}
```

`BatchGroups` = `unordered_map<BatchKey, priv::vector<shared_ptr<Task>>>` owned by
the worker and cleared each phase. `BatchKey` is opaque to the worker
(`{PoolId, method, u64 key}`); the container chooses `key` (for CTE:
`hash(tag_id, blob_name)` — the page).

`BatchSink` wraps "enqueue this merged task, and on its completion run this parent
fan-out." Implementation reuses BatchManager's pattern: a worker-side
`unordered_map<merged_task_uid, ParentCompletion>` consulted in `Worker::EndTask`
when a task is flagged `TASK_BATCH_MERGED`, mirroring `IsAggregate` /
`OnAggregateComplete`.

### Parent completion

Each merged output task carries (via the sink's side table) a list of
`{parent_task, out_slice}`. When the merged task reaches `EndTask`:

1. For each parent, copy the relevant slice of the merged OUT into the parent
   (for GetBlob: `memcpy` the parent's sub-range out of the merged read buffer
   into the parent's `blob_data_`; for PutBlob: set `bytes_written`/rc).
2. `parent->SetReturnCode(...)`, `SetCompleter(...)`, `worker->EndTask(parent)`.

This is exactly `OnAggregateComplete` generalized from one-result-broadcast to
per-parent-slice-scatter.

## 5. CTE PutBlob / GetBlob policy

`CteCoreContainer::BuildBatch`: batchable iff method ∈ {kPutBlob, kGetBlob} and
the task is an ordinary positioned op (no compression transform, no GPU page
suffix in v1 — those decline to passthrough). Group key = `hash(tag_id,
blob_name)`.

`SmashBatch`, per group:

1. **Sort** members by `(tag_id, blob_name, offset)`, ties broken by **submission
   order** (a monotonic seq the worker stamps at `BuildBatch` time). Submission
   order is what makes overlap resolution correct.
2. **Merge** into runs of contiguous/overlapping ranges → the minimal set of
   output tasks. Two members merge when `off_i + size_i >= off_{i+1}` (adjacent or
   overlapping) on the same blob.
   - **PutBlob:** allocate one contiguous staging buffer spanning
     `[min_off, max_end)`; copy each member's data in at its offset. **Overlaps
     resolve by submission order** — the later writer's bytes land last, which is
     the correct last-writer-wins the #680 token race currently gets *wrong*.
     Holes inside a run (gap between two non-adjacent-but-mergeable members) are
     not created — a gap ends the run and starts a new output task, so we never
     invent bytes.
   - **GetBlob:** one read over `[min_off, max_end)`; on completion scatter each
     parent's `[off_i, off_i+size_i)` sub-range out of the merged buffer.
3. Each output task's parent set = the members it covers; each parent's
   `out_slice` = its offset/size within the merged buffer.
4. Emit via `sink`.

Net for the 4 KiB sequential case: 256 same-page PutBlobs → **1** PutBlob spanning
the 1 MiB page. One token acquire, one bdev write. The tail's root cause is gone,
and the merge also fixes the overlap-ordering correctness gap #680 leaves open
(see #817's write notes).

## 6. Data structures to add

- `Worker`: `BatchGroups batch_groups_;` and
  `ctp SPSC<Future<Task>> passthrough_;` (the ctp ring buffer,
  `data_structures/ipc/ring_buffer.h`), both reused across iterations.
- `Worker`: `unordered_map<u64, ParentCompletion> batch_pending_;` +
  `TASK_BATCH_MERGED` flag (mirrors BatchManager's `pending_`/`TASK_BATCH_AGGREGATE`).
- `Container`: `BuildBatch` / `SmashBatch` virtuals + `BatchGroups` / `BatchSink`
  types.
- CTE container: the PutBlob/GetBlob policy above.

## 7. Open questions / hazards

1. **Staging-buffer gather cost (PutBlob).** Merging needs the members' bytes
   contiguous. v1 gathers into one buffer (a memcpy per member — the same copy
   cfs_io already does once). A later v2 could give PutBlob a scatter-list
   `(offset, ShmPtr, size)[]` to avoid the extra copy; out of scope for v1.
2. **Partial failure.** If the merged bdev transfer half-fails, how granular is
   the error to parents? v1: all covered parents get the merged rc (conservative).
   Only matters if the bdev can report partial byte counts; note and defer.
3. **Fairness / starvation.** The bounded 64 dequeue caps how long batching
   delays the passthrough tasks (which run first anyway). Confirm the phase can't
   starve `ContinueBlockedTasks` — keep it inside the existing per-iteration work
   budget.
4. **Interaction with routing side effects.** `RouteTask` may mutate RunContext /
   re-enqueue. We batch only `ExecHere` results, after routing — but confirm no
   task both routes ExecHere *and* expects to be re-dequeued.
5. **WAL / ordering vs the append pipeline.** clio-fs already has `AppendCollect`.
   Ensure positioned-write batching and the append collective don't double-handle
   the same bytes (they target different task methods, but write the same pages).
6. **Merged task accounting.** `GetTaskStats` / the #781 scheduler model sees one
   big task instead of 256 small ones — check the load model doesn't misprice it.
7. **GetBlob RYOW.** A merged read must still respect a pending overlapping write
   (the cfs `DrainIfOverlap` is client-side; server-side merged reads are new).

## 8. Phasing

| Phase | Deliverable |
|---|---|
| 0 | Worker batch phase + `BatchGroups`/passthrough SPSC, `Container::BuildBatch`/`SmashBatch` default no-ops. **No behavior change** — every container passes through. |
| 1 | Parent-completion side table + `TASK_BATCH_MERGED` fan-out in `EndTask` (unit-tested with a trivial echo container). |
| 2 | CTE `PutBlob` policy: sort/merge/gather + scatter completion. Correctness first (overlap = last-writer-wins), then the fio tail measurement. |
| 3 | CTE `GetBlob` policy: merge + scatter-read, RYOW respected. |
| 4 | Bench + tail: the 4 KiB seq write max-stall should collapse from ~1 s toward the 1-MiB-write floor; verify p99.9 and mean, not just p50. |

## 9. Success criteria

- 4 KiB sequential write **mean** latency and **p99.9/max** drop toward the
  1-writer-per-page floor (the block-size sweep's 8 ms end), not just p50.
- No regression on the full fsx/xfstests adapter suite (this touches the write
  path indirectly via coalescing).
- Unbatched single-op latency unchanged (the empty-backlog path adds nothing).

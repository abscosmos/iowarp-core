# Project: Migrating Blocked / In-Flight Tasks Off a Stalled Worker (issue #785)

Follow-up to **#781** (merged in #782). Branch `785-blocked-task-stealing`, based on
`origin/dev` @ `6b2913a3`.

---

## 1. Where #781 left us

#781 stopped a mislabeled / non-yielding task from wedging the runtime, but only for **new**
work: `RuntimeMapTask` routes on measured `RealtimeLoad()`, and the `WorkOrchestrator` monitor
thread calls `Scheduler::LoadBalance()` every 500 ms and detects a worker stuck > 1 s inside one
`ExecTask` (`default_sched.cc:331`).

Still stubs in the tree:

* `default_sched.cc:388` — `SpawnAdditionalWorker()` + steal the stalled lane.
* `default_sched.cc:404` — `StealWork()` returns `false`.
* `work_orchestrator.cc:432/439` — `SpawnAdditionalWorker()` / `RetireWorker()`.

So today a bad task strands **one thread plus everything already committed to it**. New tasks flow
around the wedge; in-flight work behind it does not.

---

## 2. What is stranded (verified code map)

Five worker structures are reachable only from that worker's own thread:

| # | State | Where | Why it is stuck |
|---|---|---|---|
| 1 | Lane backlog | `assigned_lane_` (`worker.h:475`), SHM MPSC ring | Wedged worker is the ring's only consumer |
| 2 | Blocked queues | `blocked_queues_[4]` (`worker.h:490`), `std::queue` | No lock; owner thread only |
| 3 | Periodic queues | `periodic_queues_[4]` (`worker.h:524`), `std::queue` | Same |
| 4 | Retry queue | `retry_queue_` (`worker.h:497`), `std::queue` | Same |
| 5 | Completion events | `event_queue_` (`worker.h:508`), process-local MPSC ring | Producers push from anywhere, but only the wedged worker pops (`ProcessEventQueue`, `worker.cc:1017`) |

Plus a sixth category that is not merely stranded but **invisible**: a task suspended on `co_await`
(`IsYielded() && YieldTimeUs() == 0`) is placed in **no queue at all** — `ExecTask`
(`worker.cc:720-733`) deliberately skips `AddToBlockedQueue` for it. Its only owner is the subtask
future's `parent_task_` handle (`future.h:138`, set in `ipc_cpu2self.cc:60`). The runtime cannot
enumerate its own suspended tasks. **Migration must therefore also give these tasks a home**, or
they remain unreachable by any migrator (see D4).

### 2.1 Why they are pinned

`IpcCpu2Self::SendOut` (`ipc_cpu2self.cc:109-123`):

```cpp
if (!parent_task.IsNull() && parent_task->EventQueue()) {
  auto *parent_event_queue = reinterpret_cast<...>(parent_task->EventQueue());
  parent_event_queue->Emplace(task_ptr->RunFuture());
  if (parent_task->Lane()) CLIO_IPC->AwakenWorker(parent_task->Lane());
}
```

`EventQueue()` / `Lane()` are raw addresses stamped into the parent's `RunContext` the first time
it was popped off a lane (`worker.cc:429-432`) and never revisited. A completed subtask therefore
cannot wake its parent when that worker is wedged, even though the subtask ran to completion on a
healthy worker. Same design already produced the #680 lost-wakeup class — hence the two
`AwakenAllWorkers()` fallbacks in `IpcManager::AwakenWorker` (`ipc_manager.cc:750-793`).

---

## 3. Decisions taken

Recorded because they shape everything below:

* **D-a — `LoadBalance()` is the only migrator.** No idle-thief work stealing in this issue.
  Migration happens on the monitor thread, at 500 ms cadence, triggered by stall detection.
* **D-b — Migration is triggered by stall, and moves everything.** When a worker is detected
  stalled, all of its pending and blocked tasks move to a replacement worker.
* **D-c — Migrated tasks get their event-queue pointer updated.** Signal redirection is done by
  rewriting the address in the task, not by making the address unnecessary.
* **D-d — The event queue gets a consumer mutex**, so a foreign thread can drain it.

This is a deliberately smaller design than a lock-free late-binding scheduler, and the
single-migrator constraint is what makes it tractable: **there is exactly one writer**, so no
migrator/migrator races exist and no per-task CAS state machine is needed. What remains is a
two-party problem (migrator vs. signalling producer), solved by D3 below.

---

## 4. Design

### D1 — Consumer mutex on the event queue: use `mpmc_ring_buffer`

The runtime already has the exact primitive. `ctp::ipc::mpmc_ring_buffer`
(`ring_buffer.h:790`) is *identical* to `mpsc_ring_buffer` except `RING_BUFFER_LOCK_POP`
serialises `Pop()` behind a `ctp::Mutex`. So D-d is a one-line type change:

```cpp
// worker.h:508
ctp::ipc::mpsc_ring_buffer<Future<Task, ...>, ctp::ipc::MallocAllocator> *event_queue_;
// becomes
ctp::ipc::mpmc_ring_buffer<Future<Task, ...>, ctp::ipc::MallocAllocator> *event_queue_;
```

Cost is one uncontended lock per **drain batch**, not per element, provided `ProcessEventQueue`
keeps its `while (Pop(...))` loop. The event queue is process-local (`MallocAllocator`), so an
in-process mutex here carries none of the SHM robustness concerns that apply to the lane (D5).

The lock is safe to take from the monitor thread because a wedged worker is inside `ExecTask` and
therefore provably not inside `Pop()`. **Invariant: never hold this lock across `ExecTask`** — pop
a batch, release, then execute.

### D2 — Park mutex on the blocked / periodic / retry queues

One `std::mutex park_mtx_` per worker guarding `blocked_queues_[4]`, `periodic_queues_[4]`,
`retry_queue_`. Taken by the owning worker in `AddToBlockedQueue` / `AddToRetryQueue` /
`ContinueBlockedTasks`, and by the migrator. Contention is negligible — these are touched at
park/unpark boundaries and a migration is a 500 ms-cadence event.

Same invariant: the owner pops a task under the lock, **releases**, then calls `ExecTask`. This is
what guarantees a wedged worker never holds a lock the migrator needs.

Migration splices whole `std::queue`s rather than moving element-by-element.

### D3 — The producer-side handshake (the piece the consumer mutex alone does not cover)

**A consumer mutex makes the drain safe, but it does not make the pointer swap safe.** The
signalling producer runs on a third thread and does an unsynchronised read of
`parent_task->EventQueue()`. Interleaving:

```
producer (worker B)                    migrator (monitor thread)
-------------------------------------  ------------------------------------
q = parent->EventQueue()   // OLD
                                       lock old->event_mtx_
                                       drain OLD -> NEW
                                       parent->SetEventQueue(NEW)
                                       unlock
q->Emplace(future)         // into OLD, now orphaned
AwakenWorker(parent->Lane())
```

The event lands in a queue nobody will ever drain again → the parent sleeps forever. This is the
#680 failure mode reintroduced by a different route, so it must be closed explicitly.

**Protocol.** Give each task a `sig_mtx_` (or a spinlock — it is per-task and essentially
uncontended) plus a `sig_gen_` counter incremented on every re-point.

*Producer* (`IpcCpu2Self::SendOut`):

```cpp
u32 g0; EventQ* q; TaskLane* lane;
{ std::lock_guard lk(parent->SigMtx()); q = parent->EventQueue();
  lane = parent->Lane(); g0 = parent->SigGen(); }

q->Emplace(future);                      // OUTSIDE the lock — see below

{ std::lock_guard lk(parent->SigMtx());
  if (parent->SigGen() != g0) {          // migrated under us
    q = parent->EventQueue(); lane = parent->Lane(); repush = true; } }
if (repush) q->Emplace(future);          // duplicate is harmless — §6.3

CLIO_IPC->AwakenWorker(lane);
```

*Migrator*, per task, **before** draining the old queue:

```cpp
{ std::lock_guard lk(task->SigMtx());
  task->SetEventQueue(new_q); task->Lane() = new_lane; task->BumpSigGen(); }
```

Order matters: **re-point every task first, then drain the old queue.** Anything pushed to the old
queue before its task was re-pointed is picked up by the drain; anything pushed after is caught by
the generation re-check. Between them the window is closed.

The `Emplace` deliberately happens **outside** the lock. `mpsc_ring_buffer::Emplace` with
`RING_BUFFER_WAIT_FOR_SPACE` claims a tail slot and then **spins forever** if the ring is full
(`ring_buffer.h:509-521`) — holding `sig_mtx_` across that would let a full orphaned queue
deadlock the migrator, converting a one-thread stall into a two-thread stall. See §7.2.

### D4 — Giving `co_await`-suspended tasks a home

Category 6 (§2) is not in any queue, so "migrate all blocked tasks" cannot reach it. Two ways:

* **(i) Suspended registry.** An intrusive list on the worker, under `park_mtx_`, that
  `ExecTask`'s `YieldTimeUs() == 0` path adds to and every resume path removes from. The migrator
  walks it and re-points each task. Also gives us the observability we currently lack
  (`WorkerStats::num_blocked_tasks_`, `worker.h:77`, badly undercounts today).
* **(ii) Do nothing.** Argue that a suspended task needs no migration: it is not consuming the
  wedged worker, and its wakeup arrives via the event queue — which D3 has already re-pointed…
  except D3 re-points *tasks the migrator can enumerate*, and this one it cannot. **So (ii) does
  not actually work under D-c.** A suspended task the migrator never sees keeps its stale
  `EventQueue()` pointer forever, and its completion signal goes to the wedged worker's queue.

**Recommendation: (i) is required, not optional, given D-c.** This is the main place where
"rewrite the address" costs more than "make the address unnecessary" — it forces the runtime to
maintain an enumerable set of suspended tasks that it does not have today. Worth knowing the
price, but it is a bounded, mechanical change and the observability is independently valuable.

### D5 — The lane: transfer it, do not lock its pop path

Symmetry would suggest making the lane `mpmc` too. Recommend **not**:

* `TaskLane` is `multi_mpsc_ring_buffer<Future<Task>, CLIO_QUEUE_ALLOC_T>::ring_buffer_type`
  (`task.h:766`) — it lives in **shared memory** and external client processes push into it.
  `LOCK_POP` puts a `ctp::Mutex` in SHM on the hottest path in the runtime, where a killed holder
  wedges the lane permanently.
* Transferring the lane costs nothing on the hot path and is simpler:
  1. `Worker *fresh = work_orch_->SpawnAdditionalWorker();`
  2. `fresh->AdoptLane(stalled->assigned_lane_)` → `lane->SetTid(fresh->GetTid())` so
     `AwakenWorker`'s `tgkill` (`ipc_manager.cc:771`) reaches the new consumer.
  3. Hand the stalled worker a fresh empty lane; quarantine it so the mapper places nothing on it.
  4. Requires `assigned_lane_` → `std::atomic<TaskLane*>`, re-read at the top of each `Run()`
     iteration (`worker.cc:273`), so the wedged worker sees the new lane when it finally returns.

This also **resolves #781's open "steal-safe MPSC pop" question** by making it moot.

Bonus: because the lane *object* moves with its tid, `task->Lane()` stays valid for every migrated
task — only `EventQueue()` genuinely needs re-pointing, halving the mutable state in D3.

### D6 — Load accounting must move with the tasks

Migrated tasks carry `sched_reserved_us_` reservations counted in the stalled worker's
`queued_load_us_` (#781). If they are not released from the old worker and re-reserved on the new
one, the stalled worker looks permanently loaded (harmless) and the replacement looks idle
(harmful — the mapper will pile new work onto it). Also update `SetRunWorkerId`.

---

## 5. Phases

### P0 — Deterministic repro + telemetry *(prereq, no behaviour change)*

* Extend `clio_run_thrpt_bench --test-case sched_variety` with a **dependency-chain** class: a
  `Custom` task that self-sends a subtask and `co_await`s it, while a sibling `spin_us_ = 10s`
  task wedges a worker. Today's `sched_variety` is flat (independent tasks), which is exactly why
  the in-flight stall does not appear in the #781 numbers.
* Success criterion for the whole issue: **chained-task p99 must not track the spin duration.**
  Measure before touching anything.
* Counters: parked / suspended per worker, event-queue depth, stalls detected, rescues performed,
  tasks migrated. Wire into `WorkerStats` + the `LoadBalance` 5 s telemetry dump.

### P1 — Replacement workers + lane transfer (D5)

* Real `WorkOrchestrator::SpawnAdditionalWorker()`: construct `Worker` + lane, register in
  `all_workers_` / `worker_threads_`, `thread_model_->Spawn(Run)`. `Worker::Run` already does its
  own per-thread setup (`CLIO_IPC->GetTls()`, `AddSignalEvent`, `lane->SetTid`,
  `worker.cc:247-260`), so a late-spawned worker is self-initialising.
* `assigned_lane_` → atomic; `AdoptLane`; quarantine flag; `RetireWorker` with idle-cooldown
  hysteresis.
* Wire the rescue into `LoadBalance`'s existing stall branch (`default_sched.cc:388`).
* **Clears stranded category 1.** Unstarted lane tasks have no coroutine state, so this phase
  needs no cross-thread-resume guarantee beyond what the runtime already does (§7.1).

### P2 — Migration of parked + suspended state (D1–D4, D6)

* `event_queue_` → `mpmc_ring_buffer`; `park_mtx_`; suspended registry; `sig_mtx_`/`sig_gen_`
  and the D3 producer handshake; `LoadBalance::MigrateAllFrom(stalled, fresh)`.
* **Clears stranded categories 2–6.** This is the phase that stops one bad task from stalling an
  unbounded dependency tree.
* Requires §7.1 closed, since started fibers now move deliberately.

### P3 — Observability + quarantine policy

* Suspended-duration histogram, migration counters surfaced through `clio_run_cmd_monitor`.
* Wedged-worker readmission policy (§8.3).

---

## 6. Correctness

### 6.1 Why the monitor thread can take a wedged worker's locks

Both `event_mtx_` (inside the mpmc ring) and `park_mtx_` are, by construction, never held across
`ExecTask`. A wedged worker is by definition *inside* `ExecTask`, so it holds neither. The
migrator therefore acquires promptly. Belt and braces: use `try_lock` with a bounded attempt and
skip to the next `LoadBalance` tick on failure — a missed rescue tick costs 500 ms, a blocked
monitor thread costs the whole safety net.

### 6.2 Ordering proof for the re-point

Let *T* be a migrated task and *P* a producer signalling *T*'s completion.

* If *P* releases `sig_mtx_` (first critical section) **before** the migrator acquires it, *P*
  read the OLD queue. The migrator's re-point happens after, and the drain happens after that, so
  *P*'s `Emplace` — which completes before *P*'s second critical section, which itself must wait
  for the migrator's release — is either already in the old queue when drained, or *P* observes
  `sig_gen_` changed and re-pushes to the new queue. Either way the event is delivered.
* If *P* acquires **after** the migrator's re-point, it reads the NEW queue directly.

No third case exists, because both parties serialise on the same per-task lock.

### 6.3 Duplicate events are already tolerated

The D3 re-push can deliver the same future twice (once via the drain, once via the re-push).
`ProcessEventQueue` already skips events whose future is not the parent's `AwaitedFshm()`
(`worker.cc:1056`, the #705 guard) and events whose parent `IsCoroCompleted()`
(`worker.cc:1042`, the "orphan events from parallel subtasks" case). A duplicate hits one of those
two guards. **Verify this explicitly with a test** rather than relying on the reading — it is now
load-bearing where before it was defensive.

### 6.4 Double execution after migration

A task can be simultaneously in a blocked queue *and* the target of an event — today that is
benign because both resumes happen on one thread and the second finds `IsCoroCompleted()`. After
migration both still happen on the *new* worker's single thread, so the property is preserved.
The migrator must move (not copy) under `park_mtx_`, so the old worker finds its queues empty when
it un-wedges.

### 6.5 `EndTask`'s #680 ordering

`EndTask` (`worker.cc:864-899`) orders `break_self_cycle()` against `SendOut` differently for the
in-process-client case vs. the subtask case, specifically to avoid a `shared_ptr<Task>`
double-release found by TSan. P2 changes what the subtask branch of `SendOut` does, so that
ordering must be re-derived, not assumed. **TSan run required on the P2 diff.**

### 6.6 Lost wakeups

`AwakenWorker` stays unconditional-`tgkill` (`ipc_manager.cc:759-769` — do **not** reintroduce a
park-flag gate; that regression is documented). Its two `AwakenAllWorkers()` fallbacks should
become unreachable once lanes republish their tid on transfer; keep them but log at `kWarning`, so
they act as a bug detector for this issue.

---

## 7. Risks and spikes

### 7.1 Cross-thread resume of a started fiber — *probably already happens* (gates P2)

`BoostStackPool()` (`boost_stack_allocator.h:65`) is a `SlabAllocator` with a **per-thread reuse
cache**; a stack allocated on worker A and freed on worker B lands in B's cache. Establish:

1. Does `ctp::ipc::SlabAllocator` tolerate a foreign-thread `Free` (no owner assertion, no
   intrusive header written by the owning thread)? **Spike this first.**
2. Is cross-thread resume already happening? Strong evidence it is: `ProcessRetryQueue`
   (`worker.cc:1225`) re-routes with `force_enqueue=true`, which can land a **started** task on
   another worker's lane, where `ProcessNewTask` (`worker.cc:440`) calls
   `ExecTask(task, is_started=true)`. If confirmed, the audit validates existing behaviour rather
   than introducing new risk. Confirm with a targeted test; do not assume.

The C++20 stackless backend heap-allocates frames and is thread-agnostic; only Boost needs this.

### 7.2 Full event queue is a spin-forever, not a failure

`Emplace` under `WAIT_FOR_SPACE` claims a tail slot then busy-loops until the consumer advances
(`ring_buffer.h:509-521`). Today the queue is sized `2 × GetQueueDepth()` under a #620 assumption
("a parent fills at most ~queue_depth subtask slots in its own lane", `worker.h:499-508`). After
migration a single replacement worker may receive the events of *several* workers' worth of tasks,
so that bound no longer holds. Options: size the replacement's queue larger, use
`ErrorOnNoSpace` + a spill list, or migrate to at most one worker's worth of tasks per rescue.
**Must be settled during P2 design, not discovered at runtime.**

### 7.3 TLS captured across a suspend point

Any handler caching `CLIO_CUR_WORKER` / a `Worker*` / a `GetCurrentTask()` reference in a local
across a `co_await` is already fragile and becomes wrong under migration. Grep the ChiMods and
audit each use for suspend-point crossing — cheap and mechanical, do it in P0.

### 7.4 Per-thread SHM server affinity

Each worker `ServerInit`s a named segment `clio-<pid>-<tid>` on its own thread
(`worker.cc:242-247`). If any response path is keyed to the *worker's* tid rather than the
*client's*, migration breaks it. Read `IpcManager::GetTls()` / `SendRuntime` before P2.

### 7.5 CI noise

`force_net` / stress tests were already flaky on `dev` at the #782 merge (different test each run,
clear on retry). Do not read one red run as a regression; re-run and compare against a same-day
`dev` baseline.

---

## 8. Open decisions

1. **Suspended registry (D4).** Confirm it is in scope — under D-c it is *required*, not
   optional, because a task the migrator cannot enumerate keeps a stale event-queue pointer
   forever. This is the one real cost of "rewrite the address" over "remove the address".
2. **Event-queue overflow (§7.2).** Grow the replacement's queue, make `Emplace` fallible with a
   spill list, or cap tasks migrated per rescue?
3. **Wedged-worker fate.** When the bad task finally returns, does the worker rejoin the pool or
   retire? Rejoining risks re-wedging on the same task class; retiring leaks a thread per bad task
   until exit. Recommendation: rejoin with exponential-backoff quarantine.
4. **All-workers-stalled.** `LoadBalance` already detects it (`default_sched.cc:383`). With an
   unbounded elastic pool the answer is "spawn as many replacements as needed" — confirm there is
   no cap, and that the loud warning stays.
5. **Do the blocked/periodic buckets stay four-deep?** The `% 2/4/8/16` scheme is a hand-rolled
   timer wheel; migration would be simpler over one parked list plus a deadline-ordered structure.
   Bigger diff, entirely optional.

---

## 9. Test plan

* **Unit:** D3 handshake under a stress harness (N producers signalling, 1 migrator re-pointing in
  a loop, 1 parent) — assert every signal is delivered exactly once *or* duplicated-and-skipped,
  never lost. This is the piece worth proving outside the runtime.
* **Integration** (self-launching runtime tests):
  * chained task + wedged worker → parent completes within a bound independent of spin length;
  * rescue: lane transfer republished the tid, backlog drained, blocked queues emptied;
  * migration racing a completion → no lost wakeup, no double-execute (per-task exec counter);
  * duplicate-event tolerance (§6.3) as an explicit case.
* **TSan** on the P2 diff (§6.5).
* **Benchmark:** `sched_variety` chained-class p50/p99/max per phase.
* **Regression:** the embedded-FUSE xfstests named in `SuspendMe`'s comment
  (`generic/006/007/011/013/089/100/113/127/286/363/438/471`) are the historical canary for
  worker-starvation changes; run on a 2-core-constrained config, which is where they hang.

---

## 10. Deferred / out of scope

* **General work stealing by idle workers.** Explicitly deferred per D-a. `StealWork()` stays a
  stub. This is the efficiency half; P1–P2 are the safety half, and keeping them apart means a
  perf regression there cannot un-fix liveness.
* **Late-bound completion signalling** (resume target chosen at signal time instead of rewriting
  stored addresses). This removes D3 and D4 entirely and makes idle-thief stealing nearly free,
  but needs a per-task CAS state machine. Revisit if per-migration re-pointing cost or the
  suspended registry proves awkward.
* **Preemption.** A non-yielding task still owns its thread until it returns; we bound the blast
  radius, we do not interrupt it.
* **Cross-node stealing.** Everything here is intra-process.
* **Poison-task quarantine** (a method that stalls N times gets confined to a blocking pool).
  Separate issue — it changes placement policy, not mechanism.

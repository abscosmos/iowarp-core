# Project: Stealing and Reassigning Blocked / In-Flight Tasks (issue #785)

Follow-up to **#781** (merged in #782). Branch `785-blocked-task-stealing`, based on
`origin/dev` @ `6b2913a3`.

---

## 1. Where #781 left us

#781 made the runtime stop staking liveness on tasks being honestly labeled, but only
for **new** work:

* `RuntimeMapTask` routes on measured `RealtimeLoad()` (executing + reserved + overrun),
  so a wedged worker is avoided by tasks that have not been placed yet.
* `WorkOrchestrator`'s monitor thread calls `Scheduler::LoadBalance()` every 500 ms and
  detects a worker stuck > 1 s inside one `ExecTask` (`default_sched.cc:331`).

What it explicitly did **not** do — the two TODOs still in the tree:

* `default_sched.cc:388` — `work_orch_->SpawnAdditionalWorker()` + steal the stalled lane.
* `default_sched.cc:404` — `StealWork()` is a `return false` stub.
* `work_orchestrator.cc:432/439` — `SpawnAdditionalWorker()` / `RetireWorker()` stubs.

So today: a bad task strands **one thread plus everything already committed to it**.
New tasks flow around the wedge; in-flight work behind it does not. That is the residual
hang this issue closes.

---

## 2. What is actually stranded (verified code map)

Five pieces of worker state are reachable **only** from that worker's own thread.

| # | State | Where | Why it is stuck |
|---|---|---|---|
| 1 | Lane backlog | `Worker::assigned_lane_` (`worker.h:475`), MPSC ring | Wedged worker is the ring's only consumer |
| 2 | Blocked queues | `blocked_queues_[4]` (`worker.h:490`), plain `std::queue` | No lock; owner thread only |
| 3 | Periodic queues | `periodic_queues_[4]` (`worker.h:524`), plain `std::queue` | Same |
| 4 | Retry queue | `retry_queue_` (`worker.h:497`), plain `std::queue` | Same |
| 5 | Completion events | `event_queue_` (`worker.h:508`), MPSC ring of `Future<Task>` | Producers can push from anywhere, but **only the wedged worker pops** (`ProcessEventQueue`, `worker.cc:1017`) |

And a sixth category that is worse than stranded — it is **invisible**:

> A task suspended on `co_await` (i.e. `IsYielded() && YieldTimeUs() == 0`) is placed in
> **no queue at all**. `ExecTask` (`worker.cc:720-733`) deliberately skips
> `AddToBlockedQueue` for it. Its only owner is the subtask future's
> `parent_task_` handle (`future.h:138`, set in `ipc_cpu2self.cc:60`). The runtime cannot
> enumerate its own suspended tasks.

### 2.1 Root cause: completion signalling is *worker-addressed*

`IpcCpu2Self::SendOut` (`ipc_cpu2self.cc:109-123`) is the whole story:

```cpp
const clio::run::shared_ptr<Task> &parent_task = task_ptr->GetParentTask();
if (!parent_task.IsNull() && parent_task->EventQueue()) {
  auto *parent_event_queue = reinterpret_cast<...>(parent_task->EventQueue());
  parent_event_queue->Emplace(task_ptr->RunFuture());   // <-- worker-addressed
  if (parent_task->Lane()) CLIO_IPC->AwakenWorker(parent_task->Lane());
}
```

`EventQueue()` and `Lane()` were stamped into the parent's `RunContext` the first time it
was popped off a lane (`worker.cc:429-432`). They are **raw addresses of one specific
worker**, chosen at first execution and never revisited. Three consequences:

1. **A parent task is permanently pinned** to its first worker for the rest of its life,
   however many times it suspends and resumes.
2. **A completed subtask cannot wake its parent** if that worker is wedged — even though
   the subtask ran to completion on a perfectly healthy worker.
3. **Reassignment is hard for exactly this reason.** Moving a blocked task means rewriting
   a signal target that in-flight subtasks on other threads are concurrently reading. There
   is no synchronisation on that pointer at all.

This addressing model has already produced the lost-wakeup bug class: `IpcManager::AwakenWorker`
(`ipc_manager.cc:750-793`) carries two `AwakenAllWorkers()` fallbacks and a long comment about
#680, where "a completed `PutBlob` subtask emplaced its result on the parent `WriteTask`'s
event queue but could not signal, so the parent slept forever while all workers idled."

**Design conclusion: do not try to rewrite signal targets on migration. Remove the target
from the task in the first place.**

---

## 3. Design principles

* **P1 — Late binding.** The worker that will resume a task is chosen *at the moment the task
  becomes runnable*, not at first execution. Then "reassigning a blocked task" and
  "reassigning its completion signals" are the same operation, and it is free.
* **P2 — Exactly one waker.** Every parked→runnable transition is a CAS on a single atomic
  word owned by the task. The CAS winner owns the obligation to enqueue it exactly once.
  Signaller, thief, timer expiry and rescue all race through the same gate.
* **P3 — No lock ever spans `ExecTask`.** A wedged worker must never be holding something a
  rescuer needs. Park-state locks are held for queue mutation only.
* **P4 — Affinity is a hint, never an address.** Default target stays the last worker (cache
  locality, no behaviour change on the happy path); re-target only when that worker is
  stalled, retired, or over-loaded.
* **P5 — Transfer containers, not contents, where possible.** Handing a whole lane to a new
  worker is cheaper and safer than draining an MPSC ring from a foreign thread.

---

## 4. Mechanisms

### M1 — Per-task scheduling state word

Add to `RunContext` a single `std::atomic<u32> sched_word_` packing `{state:4, wake_pending:1,
gen:27}`. All transitions are CAS on the whole word, so state and pending-wake can never be
observed torn or reordered relative to each other.

```
kRunning        executing on some worker
kSuspendEvent   co_await on a subtask future (today: in no queue)
kParkTimer      periodic / timed yield        (today: periodic_queues_)
kParkYield      cooperative yield             (today: blocked_queues_)
kParkRetry      container migrated/plugged    (today: retry_queue_)
kReady          runnable, owned by exactly one ready queue
kMigrating      transient, held by a thief
```

Legal edges:

```
kRunning     -> kSuspendEvent | kParkTimer | kParkYield | kParkRetry | (done)
any parked   -> kReady        (CAS; winner MUST enqueue exactly once)
any parked   -> kMigrating    (CAS by thief)
kMigrating   -> parked        (thief re-parks on new owner)
kReady       -> kRunning      (CAS by the worker that pops it)
```

`gen` increments on every park so a stale waker (an orphan event from a sibling subtask)
loses its CAS instead of resurrecting a task twice.

### M2 — Late-bound completion signalling (the core change)

`Worker::event_queue_` keeps its type and its `MallocAllocator` MPSC ring, but changes
*meaning*: from **"completed subtask futures addressed to worker W"** to **"tasks that are
ready to resume on worker W"**. The diff is small; the semantics are what matter.

New `IpcCpu2Self::SendOut` parent branch:

1. `future.Complete()` — set `FUTURE_COMPLETE` **at the signaller**, before touching the parent.
   (Today this is deferred to the parent's thread; §6.1 argues why the CAS protocol replaces
   that thread-serialisation.)
2. Read `parent->AwaitedFshm()`. If it is non-null and not this future, stop — the parent is
   awaiting a *different* subtask (#705). Its own await will observe our `FUTURE_COMPLETE`
   without suspending.
3. CAS `parent.sched_word_`: `kSuspendEvent -> kReady`.
   * **Won** → ask the scheduler for a target: `sched->PickResumeWorker(parent)` — the parent's
     `preferred_worker_id_` unless that worker is stalled/retiring/over-loaded (P4). Push the
     parent onto that worker's ready queue and `AwakenWorker(target->GetLane())`.
   * **Lost because state was `kRunning`** → set `wake_pending`; the parent's own suspend CAS
     will see it and not suspend (§6.2, wait-morphing).
   * **Lost because state was `kMigrating`** → set `wake_pending`; the thief completes the
     ready transition after it re-parks (§6.3).

`parent->EventQueue()` and the `parent->Lane()` wakeup disappear from this path entirely.
`ProcessEventQueue` becomes `ProcessReadyQueue`: pop task, CAS `kReady -> kRunning`, `ExecTask`.
The `IsCoroCompleted()` and `AwaitedFshm` guards it carries today (`worker.cc:1042-1060`) move
to the signaller in step 2, where they belong.

**This is what makes reassignment free.** A migrated parent needs no signal rewriting, because
no one holds its address — the target is resolved from scheduler state at signal time.

### M3 — Lane handoff instead of steal-safe MPSC pop

#781 left "steal-safe pop on a single-consumer MPSC lane" as an open question. Sidestep it:
**do not steal from the ring — transfer the ring.**

`LoadBalance()` rescue path, on detecting `w->IsStalled()`:

1. `Worker *fresh = work_orch_->SpawnAdditionalWorker();`
2. `fresh->AdoptLane(w->assigned_lane_)`, which does `lane->SetTid(fresh->GetTid())` so
   `AwakenWorker`'s `tgkill` (`ipc_manager.cc:771`) reaches the new consumer.
3. Give the wedged worker a fresh empty lane, and mark it `kQuarantined` so the mapper places
   nothing on it.
4. When the bad task finally returns, the wedged worker re-reads `assigned_lane_` and finds the
   new empty one. **This requires `assigned_lane_` to become `std::atomic<TaskLane*>`, re-read
   at the top of each `Run()` iteration** (`worker.cc:273`) — today it is a plain pointer read
   once per loop, which is fine but not publication-safe across threads.

There is exactly one consumer of the ring at all times: the handoff happens while the wedged
worker is *inside* `ExecTask` and provably not popping. Ordering: `SetTid` before the old
worker can return; guaranteed by publishing the new lane last.

### M4 — Stealable parked state

Give each worker one `std::mutex park_mtx_` guarding `blocked_queues_`, `periodic_queues_`,
`retry_queue_`, and the new suspended registry. Held only across queue mutation, never across
`ExecTask` (P3). Contention is negligible: these are touched at park/unpark boundaries, and a
steal is a 500 ms-cadence event.

`Scheduler::StealWork(Worker *thief)` and the rescue path both use:

```cpp
size_t MigrateParked(Worker *from, Worker *to, size_t max);
// per task: CAS parked -> kMigrating; move handle; re-park on `to`;
//           CAS kMigrating -> parked; if wake_pending was set, do the
//           parked -> kReady transition + enqueue on `to` (§6.3).
```

`kMigrating` is the interlock that makes a concurrent completion signal safe. A task the CAS
cannot claim (already `kReady`, already `kRunning`) is simply skipped — someone else owns it.

### M5 — Suspended-task registry (diagnostics + timeout, not the critical path)

Note the pleasing consequence of M2: **`co_await`-suspended tasks no longer need to migrate at
all.** They live nowhere; whoever signals them enqueues them onto a *currently healthy* worker.
The wedged worker was never really holding them.

The registry is still worth building, for reasons that are not liveness:

* Observability — "which tasks are suspended, on what, for how long" is currently unanswerable.
* Cycle / timeout detection — a parent suspended forever because its subtask was itself
  stranded is exactly the failure we are chasing, and today it is silent.
* `WorkerStats::num_blocked_tasks_` (`worker.h:77`) already exists and is already surfaced via
  `clio_run_cmd_monitor` — it just undercounts badly.

Implementation: intrusive list hooks on `Task` (O(1) insert/erase), under `park_mtx_`.

---

## 5. Phased plan

Each phase is independently mergeable and independently benchmarkable. Phase order is chosen so
that the highest-liveness-value change (P2) lands *after* the audit that de-risks it (P1).

### P0 — Deterministic repro + telemetry *(prereq; no behaviour change)*

* Extend `clio_run_thrpt_bench --test-case sched_variety` with a **dependency-chain** class:
  `Custom` task that self-sends a subtask and `co_await`s it, while a sibling `spin_us_ = 10s`
  task wedges a worker. Today's `sched_variety` is flat (independent tasks), which is precisely
  why the in-flight stall does not show up in the #781 numbers.
* Success criterion for the whole issue: **chained-task p99 must not track the spin duration.**
  Measure it before touching anything.
* Add counters: parked/suspended per worker, ready-queue depth, stalls detected, rescues
  performed, tasks migrated. Wire into `WorkerStats` + the `LoadBalance` 5 s telemetry dump.
* Files: `benchmarks/`, `modules/MOD_NAME/`, `worker.h` (`WorkerStats`), `default_sched.cc`.

### P1 — Lane handoff rescue (M3) + real `SpawnAdditionalWorker`/`RetireWorker`

* `WorkOrchestrator::SpawnAdditionalWorker()`: construct `Worker` + lane, register in
  `all_workers_`/`worker_threads_`, `thread_model_->Spawn(Run)`. Note `Worker::Run` already does
  its own per-thread setup (`CLIO_IPC->GetTls()`, `AddSignalEvent`, `lane->SetTid`,
  `worker.cc:247-260`), so a late-spawned worker is self-initialising — good.
* `assigned_lane_` → `std::atomic<TaskLane*>`, re-read per loop iteration.
* `AdoptLane` + quarantine flag + `LoadBalance` rescue wiring.
* `RetireWorker`: idle-cooldown park + join, hysteresis so we do not flap.
* **Fixes stranded category 1.** Unstarted tasks have no coroutine state, so this phase needs no
  cross-thread-resume guarantees beyond what the runtime already does (see §7.1).

### P2 — Late-bound completion signalling (M1 + M2) *(the core change)*

* `sched_word_` on `RunContext`; all park/unpark sites converted to CAS.
* `SendOut` rewritten per §M2; `ProcessEventQueue` → `ProcessReadyQueue`.
* `Scheduler::PickResumeWorker()` — affinity-preserving by default (P4), so the happy path is
  bit-for-bit today's behaviour and only the stalled/retired case re-targets. This is the
  de-risking lever: **cross-thread resume only happens on the exceptional path in P2.**
* Delete `RunContext::event_queue_` as an *address* (`task.h:829`) once no reader remains.
* **Fixes stranded category 5 and the invisible category 6.** This is the phase that stops one
  bad task from stalling an unbounded dependency tree.

### P3 — Stealable parked state (M4)

* `park_mtx_`; `MigrateParked`; rescue path extended to drain blocked/periodic/retry.
* **Fixes stranded categories 2, 3, 4.**
* Requires the §7.1 audit to be closed, since started fibers now move deliberately.

### P4 — General work stealing

* `DefaultScheduler::StealWork(thief)` for real: an idle worker pulls from the ready queue and
  parked state of the most-loaded worker (by `RealtimeLoad`), ring-neighbour first.
* Called from `SuspendMe` before sleeping, and from `LoadBalance`.
* This is the *efficiency* half; P1–P3 are the *safety* half. Keep them separate so a
  performance regression here cannot un-fix liveness.

### P5 — Registry + observability + poison-task quarantine

* M5 registry, suspended-duration histogram, `clio_run_cmd_monitor` surfacing.
* Optional: a method that stalls N times gets its `TaskStat` flagged so the mapper confines it
  to a dedicated blocking pool. This is the "learn which tasks are bad" layer — deliberately
  last, because the runtime must be correct without it.

---

## 6. Correctness arguments

### 6.1 Why `future.Complete()` can move to the signaller

Today's comment (`worker.cc:1019-1023`) says deferring `FUTURE_COMPLETE` to the parent's thread
"avoids stale `RunContext*` pointers since `FUTURE_COMPLETE` is never set before the event is
consumed" — i.e. thread-serialisation is being used as the ordering primitive. Under M1 the
ordering primitive is the `sched_word_` CAS instead: the parent cannot be resumed by anyone who
did not win the CAS, and the CAS winner is the same agent that set `FUTURE_COMPLETE`, in that
order. The parent therefore never observes a resumption whose future is not already complete.
The reverse (complete future, parent not yet resumed) is benign and is the normal poll case.

**Verification obligation:** the `#680` fix in `EndTask` (`worker.cc:864-899`) orders
`break_self_cycle()` against `SendOut` differently for the in-process-client case versus the
subtask case, specifically to avoid a `shared_ptr<Task>` double-release. P2 changes what
`SendOut` does in the subtask branch, so that ordering must be re-derived, not assumed. TSan run
required (`#680` was found by TSan).

### 6.2 Signal arriving before the parent suspends (wait-morphing)

A subtask can complete before the parent reaches its `co_await`. Sequence at the parent:

```
1. if (future.IsComplete()) -> do not suspend, continue
2. publish awaited_fshm_
3. CAS sched_word_: kRunning -> kSuspendEvent, requiring wake_pending == 0
   - success -> genuinely suspended, only a CAS winner can wake us
   - failure (wake_pending set) -> clear it, goto 1
```

The loop terminates because `wake_pending` is only set by a signaller that has already set
`FUTURE_COMPLETE` on some future, so step 1 succeeds on the retry for the awaited one.

### 6.3 Signal racing a migration

Thief holds `kMigrating`. A signaller's `parked -> kReady` CAS fails and it sets `wake_pending`
(a CAS on the same word, so it cannot be lost or reordered). The thief, after re-parking on the
new owner, re-reads the word; if `wake_pending` is set it performs the `parked -> kReady`
transition and the enqueue itself, onto the **new** owner. Exactly one enqueue happens, on the
correct worker. No spinning, no unbounded wait.

### 6.4 Orphan / duplicate events

`gen` in the state word is bumped on each park. A waker captures `gen` at read time and includes
it in the CAS, so a stale sibling completion (the case `worker.cc:1039-1044` currently guards
with `IsCoroCompleted()`) fails its CAS instead of resuming an already-resumed task.

### 6.5 Lost wakeups vs the signal path

`AwakenWorker` remains unconditional-`tgkill` (`ipc_manager.cc:759-769` — do **not** reintroduce
a park-flag gate; that regression is documented). Because the target lane now belongs to a
worker chosen at signal time, and lane tid is republished on handoff, the two
`AwakenAllWorkers()` fallbacks should become genuinely unreachable rather than load-bearing.
Keep them, but log at `kWarning` when hit — they become a bug detector for this issue.

---

## 7. Risks and spikes (run these before P2/P3)

### 7.1 Cross-thread resume of a started fiber — *probably already happens*

Boost stackful fibers are the concern: `BoostStackPool()` (`boost_stack_allocator.h:65`) is a
`SlabAllocator` with a **per-thread reuse cache**. A stack allocated on worker A and freed on
worker B lands in B's cache. Two things to establish:

1. Does `ctp::ipc::SlabAllocator` tolerate a foreign-thread `Free` (no owner assertion, no
   intrusive header written by the owning thread)? If not, migration needs an explicit
   "return to origin" path. **Spike this first — it gates P3.**
2. Is cross-thread resume *already* happening today? Strong evidence that it is:
   `ProcessRetryQueue` (`worker.cc:1225`) re-routes with `force_enqueue=true`, which can land a
   **started** task on another worker's lane, and `ProcessNewTask` (`worker.cc:440`) then calls
   `ExecTask(task, is_started=true)` there. If confirmed, the audit is *validating existing
   behaviour* rather than introducing new risk — a large de-risk. Confirm with a targeted test,
   do not assume.

The C++20 stackless backend heap-allocates frames and is thread-agnostic; only the Boost path
needs this.

### 7.2 TLS captured across a suspend point

Any handler that caches `CLIO_CUR_WORKER` (or a `Worker*`/`GetCurrentTask()` reference) in a
local across a `co_await` is already fragile and becomes wrong under migration. Grep the ChiMods
for `CLIO_CUR_WORKER` and audit each use for suspend-point crossing. Cheap, mechanical, do it in
P0.

### 7.3 Per-thread SHM server affinity

Each worker `ServerInit`s a named MPSC segment `clio-<pid>-<tid>` on its own thread
(`worker.cc:242-247`). If any task's response path is keyed to *the worker's* tid rather than
the *client's*, migration breaks it. Read `IpcManager::GetTls()` / `SendRuntime` before P2.

### 7.4 Ready-queue capacity

`event_queue_` is sized `2 x GetQueueDepth()` with a `WAIT_FOR_SPACE` `Emplace`
(`worker.h:499-508`) — sized under the assumption "a parent fills at most ~queue_depth subtask
slots in its own lane" (#620). Under M2 the queue holds *ready tasks from anywhere*, so that
bound no longer holds and a blocking `Emplace` on a full ready queue would deadlock the
signaller. Either make the ready queue growable, or make the enqueue fallible with a
spill-to-global-ready-list path. **Must be settled during P2 design, not discovered at runtime.**

### 7.5 Non-determinism in CI

`force_net`/stress tests were already flaky on `dev` at the #782 merge (a different test each
run, clear on retry). Do not read a single red CI run as a regression from this work — re-run
first, and compare against a `dev` baseline run from the same day.

---

## 8. Decisions needed

1. **Scope of P2's re-targeting.** Recommendation: affinity-preserving by default, re-target
   only on stall/retire/over-load. The alternative (always re-target through
   `RuntimeMapTask`) is more work-conserving but makes cross-thread resume the *common* path and
   loses cache locality on every resume. Confirm the conservative default.
2. **Do we keep `blocked_queues_`/`periodic_queues_` as four buckets?** The `% 2/4/8/16` iteration
   scheme is a hand-rolled timer wheel. Under M1 it could collapse into one parked list plus a
   deadline-ordered structure, which would be simpler to migrate. That is a bigger diff; it can
   also be left exactly as-is. Preference?
3. **Ready-queue overflow policy** (§7.4): growable ring vs fallible enqueue + global spill list.
4. **Poison-task quarantine (P5)** — in scope for this issue, or a separate one? It is the only
   part that changes *placement policy* rather than mechanism.
5. **`kQuarantined` worker fate.** After a rescue, does the wedged worker rejoin the pool when it
   finally returns, or retire? Rejoining risks re-wedging on the same class of task; retiring
   leaks a thread per bad task until the process exits. Recommendation: rejoin, but with an
   exponential-backoff quarantine so a repeatedly-wedging worker is effectively retired.

---

## 9. Test plan

* **Unit:** `sched_word_` state machine under a stress harness (N signallers, N thieves, one
  parent) — assert exactly-one-resume and no lost wakeup. This is the piece worth proving
  outside the runtime.
* **Integration (self-launching runtime tests):**
  * chained task + wedged worker → parent completes within a bound independent of spin length;
  * rescue path: assert lane handoff republished the tid and the backlog drained;
  * migration under concurrent completion: assert no double-execute (per-task execution counter).
* **TSan** on the P2 diff (§6.1).
* **Benchmark:** `sched_variety` chained-class p50/p99/max, before/after, per phase.
* **Regression:** the embedded-FUSE xfstests called out in `SuspendMe`'s comment
  (`generic/006/007/011/013/089/100/113/127/286/363/438/471`) are the historical canary for
  worker-starvation changes. Run them on a 2-core-constrained config, which is where they hang.

---

## 10. Out of scope

* Preemption of a running task. A non-yielding task still owns its thread until it returns; we
  bound the *blast radius*, we do not interrupt it.
* Cross-node stealing. Everything here is intra-process.
* Replacing the coroutine backend.

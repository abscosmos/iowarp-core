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
| 3 | Periodic queues | `periodic_queues_[4]` (`worker.h:524`), `std::queue` | Same — **and this is the most severe row**, see §2.3 |
| 4 | Retry queue | `retry_queue_` (`worker.h:497`), `std::queue` | Same |
| 5 | Completion events | `event_queue_` (`worker.h:508`), process-local MPSC ring | Producers push from anywhere, but only the wedged worker pops (`ProcessEventQueue`, `worker.cc:1017`) |

Plus a sixth category that is not merely stranded but **invisible**: a task suspended on `co_await`
(`IsYielded() && YieldTimeUs() == 0`) is placed in **no queue at all** — `ExecTask`
(`worker.cc:720-733`) deliberately skips `AddToBlockedQueue` for it. Its only owner is the subtask
future's `parent_task_` handle (`future.h:138`, set in `ipc_cpu2self.cc:60`). The runtime cannot
enumerate its own suspended tasks. **D4 fixes this by parking them in the blocked queue**, which
folds category 6 into category 2 and makes the whole set enumerable by one mechanism.

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

### 2.2 `blocked_queues_` is currently vestigial

Worth stating plainly, because D4 depends on it. `AddToBlockedQueue`'s `YieldTimeUs() == 0` branch
(`worker.cc:1151`) is the one that feeds `blocked_queues_`, and essentially nothing reaches it:

* `ExecTask` (`worker.cc:725`) only calls it when `YieldTimeUs() > 0` — i.e. always the *periodic*
  branch.
* `ProcessBlockedQueue`'s own re-add (`worker.cc:951`) is **dead code**, unreachable behind a
  `continue` at line 947.
* `ReschedulePeriodicTask` (`worker.cc:1268`) reaches it only for a periodic task whose period is 0.

So `blocked_queues_[4]`, its four-bucket backoff, and `WorkerStats::num_blocked_tasks_` are
scaffolding for a case that never populates them. D4 puts them to their intended use rather than
adding a parallel registry.

`periodic_queues_[4]`, by contrast, is very much alive — and is where the runtime's own
infrastructure lives.

### 2.3 Periodic tasks are the highest-stakes stranded state

The periodic queue holds the runtime's **network and client-IPC pollers**. Both schedulers
hard-route admin periodic methods to specific worker roles (`default_sched.cc:167-195`,
`local_sched.cc:126-136`):

```
14 = kSend       peer DEALER pool          -> net_send_worker_
15 = kRecv       peer ROUTER (9413)        -> net_recv_worker_
20 = kClientRecv client-facing ROUTER      -> net_recv_worker_
21 = kClientSend same client-facing ROUTER -> net_recv_worker_
```

Consequences the plan has to answer for:

* **Severity.** If `net_recv_worker_` stalls, its stranded pollers mean the node stops receiving
  peer traffic *and* stops servicing client IPC. One bad task becomes a whole-node outage. This is
  worse than a stalled compute worker, and the plan previously treated periodic as a lesser sibling
  of blocked.
* **The routing table itself is being deleted.** D8 replaces these roles with dedicated-worker
  periodic tasks placed by load, so the pollers become migratable like anything else. The
  socket-ownership constraint the table encodes does *not* go away — it has to be re-expressed, and
  that is most of what D8 is about.

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
* **D-e — Net-worker roles are deleted in favour of a `TASK_DEDICATED` flag.** Exclusivity is
  declared by the task, not inferred from an inflated cost that the learner would erase. See D8/D8a.

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

### D4 — Park `co_await`-suspended tasks in the blocked queue

**Decision:** the `YieldTimeUs() == 0` yield path in `ExecTask` (`worker.cc:720-733`) now calls
`AddToBlockedQueue`, so an event-waiting task is parked in `blocked_queues_[]` instead of vanishing
into the subtask future's `parent_task_` handle. The blocked queue becomes the enumerable set of
suspended tasks — for migration (D-b), for `WorkerStats`, and for diagnostics. No parallel registry.

The one-line version of this is wrong, though. `ProcessBlockedQueue` (`worker.cc:902-953`) today is
a *resume-once-and-forget* scan, and both halves of that are hostile to event-waiting tasks:

1. **It resumes unconditionally.** It pops a task and calls `ExecTask(task, is_started)` with no
   check that what the task is waiting on has actually happened (`worker.cc:944`). An event-waiting
   task resumed mid-`co_await` returns from the await while the subtask is still running and reads
   its outputs early — precisely the #705 failure (short reads/writes, freed buffers).
2. **It never re-queues.** The re-add at `worker.cc:951` is dead code behind the `continue` at 947,
   so a popped task leaves the queue permanently. Tracking would be lost on first scan.

So `ProcessBlockedQueue` is rewritten as a **readiness scan**:

```
for each parked task (bounded batch):
  if (task->IsCoroCompleted())            -> erase        // already finished elsewhere
  else if (awaited future is complete)    -> erase, resume, LOG(kWarning)
  else                                    -> leave parked, bump miss count / backoff bucket
```

The middle branch is a genuine bonus: it is a **self-healing net for lost wakeups**. Normally
`ProcessEventQueue` resumes the parent long before the scan sees it, so reaching that branch means
an event was dropped — which is exactly the #680 class, and it should be loud rather than silent.
The existing four buckets (`% 2/4/8/16`) become the backoff for how often a long-waiter is
re-checked, which is what they were shaped for.

Supporting changes:

* **`std::queue` → `std::list` + an iterator stored on the task**, so `ProcessEventQueue` can erase
  a task in O(1) before resuming it. Without O(1) erase, a resumed task lingers as a stale entry
  and can be double-parked when it suspends again.
* **`AddToBlockedQueue`'s `wait_for_task` parameter is deleted.** It currently means "do not add"
  (`worker.cc:1145`) — the exact behaviour being reversed — and no caller passes `true`.
* **Delete the dead `continue` + unreachable re-add** at `worker.cc:946-951`.
* **Awaited-future lifetime.** The scan needs to test completeness of what the task awaits.
  `RunContext::awaited_fshm_` is a raw `const void*` (`task.h:885`). Dereferencing it is safe *while
  the parent is suspended* (the parent's coroutine frame holds the owning `Future`, which holds the
  `shared_ptr<FutureShm>`), but that is an invariant worth making explicit — consider storing an
  owning `Future` instead of the raw pointer. See open decision §8.1.
* **Ownership.** Parking adds a second owning `shared_ptr<Task>` reference. A task that completes
  through a path which never erases its entry is retained forever by the queue. The
  `IsCoroCompleted()` branch above makes this self-correcting, but given #620/#680 were both
  reference-lifetime bugs, the leak-detection tests should be run against this specifically.

### D4b — Periodic tasks: the migration silently undoes itself unless the role moves too

`ProcessPeriodicQueue` (`worker.cc:955-1015`) does not merely resume a due task — it **re-routes**
it every period: `CLIO_IPC->RouteTask(task->RunFuture())`, executing only on `ExecHere`. That has a
consequence the rest of the plan has to respect:

> Migrate a role-pinned periodic task to a replacement worker, and on its very next period
> `RouteTask` → `RuntimeMapTask` sends it **straight back to the stalled worker's lane**, because
> the routing table keys on role, not on health. The rescue reverts itself within one period, and
> the symptom is a rescue that appears to succeed and then quietly stops working.

This splits periodic tasks in two, and they need opposite treatment:

**Non-pinned periodic tasks — migrate the queue, placement self-heals.** `RuntimeMapTask` routes
these by measured `RealtimeLoad`, so they will naturally avoid the stalled worker. All they need is
*somebody to scan them*. Moving `periodic_queues_[]` to a worker that runs `ProcessPeriodicQueue` is
sufficient; the existing per-period re-route does the placement. This is the cheapest correct fix in
the whole plan.

**Role-pinned periodic tasks (admin 14/15/20/21) — do not migrate; re-point the role, or neither.**
Moving the task without moving the role reverts (above); moving the role without moving the socket
is UB (§2.3). So a stalled net worker is *not* rescuable by task migration at all — see D7.

Periodic-specific mechanics for whichever tasks do move:

* **`Lane()` must stay valid.** `ReschedulePeriodicTask` (`worker.cc:1247-1251`) **silently drops**
  a periodic task whose `Lane()` is null — `if (!lane) return;`, no log, task gone. Lane transfer
  (D5) keeps the lane object alive and valid, which is another argument for it over lane draining.
* **Reset `BlockStart()` on migration.** A periodic task stranded 10 s behind a wedged worker has
  `elapsed >> yield_time`, so it fires the instant it is scanned — and a whole migrated queue fires
  simultaneously. `AddToBlockedQueue` already has staleness handling for this
  (`worker.cc:1179-1185`, the >10 ms reset); migration should reuse it rather than invent a rule.
* **`ProcessPeriodicQueue` is the model for D4's readiness scan.** It already does exactly the right
  shape — check whether the task is due, resume if so, **re-queue if not** (`worker.cc:1010-1013`).
  D4's rewrite of `ProcessBlockedQueue` is that same loop with "awaited future is complete"
  substituted for "deadline reached". Framing it as *make blocked look like periodic* keeps the
  change small and well-precedented.

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

### D7 — Migratability is a property of the worker's role

Not every worker can be rescued by moving its tasks. This has to be explicit or the rescue path
will do something unsafe.

| Role | Migratable? | Why | Rescue strategy |
|---|---|---|---|
| `scheduler_worker_` | Yes | General purpose | Lane transfer + task migration |
| `io_workers_` | Yes | General purpose | Lane transfer + task migration |
| `net_send_worker_` | **Deleted** | Becomes a dedicated-worker task — D8 | Migratable once D8 lands |
| `net_recv_worker_` | **Deleted** | Same | Migratable once D8 lands |
| GPU worker | **No** (probably) | Bound to `gpu_lanes_`; `SuspendMe` early-returns because "GPU workers must never sleep" (`worker.cc:483-486`) | Spike §7.6 |
| Worker 0 | Special | Hardcoded `worker_id_ == 0` drives `BatchManager::FlushDue` (`worker.cc:288`) | Make it a role pointer so the duty can move |

`LoadBalance` branches on this before rescuing. After D8 the non-migratable set shrinks to the GPU
worker (pending §7.6) and worker 0's batch duty, which is the point: **role gating is a temporary
scaffold, not the destination.**

Note this also dissolves what was open decision §8.2 — `DefaultScheduler::PickAltWorker`
(`default_sched.cc:302-319`) deliberately falls back to the net workers, which made "protect them by
prevention" unworkable. With no net worker to protect, the hole closes by construction.

### D8 — Delete the net-worker roles; make them expensive dedicated-worker tasks

**Decision:** remove `net_send_worker_` / `net_recv_worker_` as scheduler roles. The net pollers
become ordinary periodic tasks, placed by load-based routing, declared costly enough that
`RuntimeMapTask` + `LoadBalance` give each its own worker and spawn one when needed.

The hook already exists and is already used for exactly this purpose. `Runtime::GetTaskStats`
(`admin_runtime.cc:1870-1886`) already reports these four methods with `io_size_ = 1 MiB` and
`compute_ = ` the net-queue backlog, commented *"io_size_ here is just a stand-in so the scheduler
doesn't route them as zero-cost metadata."* And #781 already **reserves** predicted cost on the
target worker (`ReserveLoad` / `queued_load_us_`), so a task declared expensive repels other work
from its worker automatically. Exclusivity becomes an emergent property of the load model rather
than a hardcoded table. Three things have to be true or it breaks:

**(a) The learned model will erase an inflated constant.** `Container::InferCpuTime =
coef * (compute + 1)`, with `coef` driven by SGD from *measured* CPU time in `ReinforceCpuModel`
(`EndTask`, `worker.cc:778`). A poller that returns `EAGAIN` on most ticks measures near-zero CPU,
so `coef` converges toward zero and the inflated cost stops repelling anything — quietly, over
minutes, after appearing to work. Setting `compute_` high is self-defeating against a subsystem
built to learn true costs.

**Decided:** declare exclusivity directly with a `TASK_DEDICATED` flag rather than encoding it as a
fake cost. It survives the learner, it is self-documenting, and it is what we actually mean — the
requirement is *exclusivity*, not expense. Cost inflation stays as a secondary placement hint but
stops being load-bearing. Semantics in D8a.

**(b) Socket-ownership groups must survive the roles.** The routing comment is explicit:
*"The split is by SOCKET OWNERSHIP, not by verb — ZeroMQ sockets are not safe to share across
threads, so each socket has exactly one owner thread"* (`default_sched.cc:169-171`). The roles
encode three groups:

```
{kSend}                    peer DEALER pool
{kRecv}                    peer ROUTER (9413)
{kClientRecv, kClientSend} SHARE the client-facing ROUTER  <-- must not run concurrently
```

Delete the roles without replacing this and methods 20/21 can be placed on two different workers,
concurrently driving the same ZMQ socket. Options: a co-location/affinity group in the mapper, or
**merge `kClientRecv` + `kClientSend` into a single periodic task** — one socket, one task, one
thread by construction, no new scheduler concept. Recommend the merge; add a general affinity-group
mechanism only when a second resource needs it.

**(c) "Exactly one live instance" becomes a load-bearing invariant.** ØMQ permits moving a socket
between threads given a full barrier and no concurrent use. A periodic task supplies both: it exists
in exactly one periodic queue, and `ProcessPeriodicQueue` re-routes it *between* executions, so
period N finishes on worker A before period N+1 starts on worker B, with the lane push/pop providing
the barrier. **So dynamic reassignment is safe iff a method never has two live instances.** That is
currently true but unasserted, and at least two pieces of code silently depend on it — both of which
were safe only because of the pinning being deleted:

* `Runtime::Recv`: *"No socket lock needed - single net worker processes all Recv tasks."*
* `Runtime::ClientSend` (`admin_runtime.cc:652`) keeps a function-local
  `static std::vector<shared_ptr<Task>> deferred_deletes`, mutated with no lock. Two concurrent
  instances would be a data race on a `shared_ptr` vector — i.e. a double-free, the #680 shape.

Assert the invariant (one live instance per net periodic method) rather than assuming it, and audit
these two sites as part of the change.

**What this buys.** Deletes the hardcoded method-id routing from *both* schedulers; removes the
non-migratable role class; and converts the §2.3 failure — a stalled net worker taking the node's
networking down with no rescue possible — into an ordinary migratable stall. That is the real win:
not that the poller moves, but that when its worker wedges, it *can* move.

### D8a — `TASK_DEDICATED` semantics

```c
#define TASK_DEDICATED BIT_OPT(clio::run::u32, 10)   // next free bit; 1/4/5/6 are retired
```

Set alongside `TASK_PERIODIC` where the pollers are spawned (`admin_client.h:128` and friends), and
**cleared on remote receive** next to the existing `ClearFlags(TASK_PERIODIC)`
(`ipc_run2run.cc:424`) — it is a local scheduling property, not a wire property.

**1. Sticky binding, not per-period re-selection.** The scheduler holds a `{dedicated task → worker}`
binding and `RuntimeMapTask` returns the bound worker every period. This is deliberate:
`ProcessPeriodicQueue` re-routes on *every* tick, so without stickiness a poller would hand its ZMQ
socket to a different thread several times a second — legal under §7.5, but constant churn and
constant opportunity to get the barrier wrong. With stickiness the socket moves only at a rebinding
point chosen by `LoadBalance`. This is precisely "pin to any worker, then adjust dynamically": the
*binding* is dynamic, the per-period placement is not.

**2. Repulsion by exclusion, not by load.** A worker hosting a dedicated task is skipped in
`RuntimeMapTask`'s least-loaded scan over `io_workers_`. Exclusion rather than inflated
`RealtimeLoad` because it is O(1) and cannot be learned away (§D8a). If every worker is dedicated,
spawn a new one — the elastic pool is unbounded by #781's decision — rather than overflowing onto a
dedicated worker.

**3. Provisioning is a `LoadBalance` responsibility.** Each tick: every `TASK_DEDICATED` task has a
live, unstalled worker; spawn if short; rebind if its worker stalled *and the task itself is parked*;
retire the worker when the task goes away.

**Degradation under core pressure.** One worker per dedicated task means three threads for
{peer-send, peer-recv, client-combined}, where today there are two role workers — worse on a 2-core
CI runner (§7.7). Let `LoadBalance` collapse several dedicated tasks onto one shared worker when
workers exceed cores. That is safe by inspection because it *is* today's behaviour —
`net_recv_worker_` already hosts three pollers — provided the D8b socket groups stay intact.

**Honest limitation: this contains a stall, it does not cure one.** If a dedicated task wedges
*inside its own handler* (say `Recv` blocking on a socket) it cannot be migrated: migration moves
only *parked* tasks, and starting a second instance is forbidden by the one-live-instance invariant
(D8c). What is actually gained:

* no other work ever queues behind it (repulsion);
* the *other* pollers survive — today a wedged `kRecv` also takes down `kClientRecv` and
  `kClientSend`, because all three share `net_recv_worker_`;
* a poller merely *parked* on a worker wedged by something else does migrate.

Curing a wedged poller needs handler-level timeout/abandon with provably sane socket state, which is
out of scope for this issue. Worth being explicit so nobody reads "LoadBalance spawns it a worker"
as "networking always recovers".

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

Split into two mergeable steps, because D4 is independently testable and carries the #705 risk:

**P2a — park event-waiting tasks (D4), no migration yet.** `ExecTask` parks them;
`blocked_queues_` → `std::list` + task-held iterator; `ProcessBlockedQueue` becomes the readiness
scan; `ProcessEventQueue` erases before resuming; delete `wait_for_task` and the dead re-add.
Behaviour-preserving by design — nothing should resume differently — so any change in the FUSE /
stress suites here is a real regression and is easy to attribute. Lands the lost-wakeup safety net
and fixes `WorkerStats::num_blocked_tasks_` on its own.

**P2b — dissolve the net-worker roles (D8).** `TASK_DEDICATED_WORKER` (or the reinforcement-exempt
declared cost); merge `kClientRecv`+`kClientSend`; delete the method-id routing tables from
`default_sched.cc` and `local_sched.cc`, and `net_send_worker_`/`net_recv_worker_` with them; assert
one-live-instance per net periodic; drop the net fallbacks from `PickAltWorker`. Standalone and
independently valuable — it is a simplification of #781's routing that happens to unblock the
rescue. Land it before P2c so periodic migration never has to special-case a role.

**P2c — periodic queue migration (D4b).** Migrate `periodic_queues_[]` under `park_mtx_`; reset
`BlockStart()`; D7 gating for whatever remains non-migratable (GPU, worker 0). Needs no event-queue
changes at all — the per-period `RouteTask` self-heals placement — so it is the cheapest large win
available and it clears the most severe stranded category (§2.3).

**P2d — blocked/suspended migration.** `event_queue_` → `mpmc_ring_buffer`; `sig_mtx_`/`sig_gen_`
and the D3 producer handshake; `LoadBalance::MigrateAllFrom(stalled, fresh)` + re-drain of
quarantined queues (§7.2); retry queue; load accounting moves (D6).

* **Clears stranded categories 2–6.** This is what stops one bad task from stalling an unbounded
  dependency tree.
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

### 7.2 Full event queue — accepted, keep blocking `Emplace` *(closed)*

`Emplace` under `WAIT_FOR_SPACE` claims a tail slot then busy-loops until the consumer advances
(`ring_buffer.h:509-521`). Decision: **leave it as is.** The argument that makes it safe:

* The push is **outside** `sig_mtx_` (D3), so a spinning producer can never block the migrator.
  This is the constraint that must not be relaxed; if the push is ever moved inside the lock, a full
  orphaned queue converts a one-thread stall into a two-thread stall.
* The migrator drains the old queue as part of the re-point sequence, so a producer that arrives
  with a stale pointer finds space and completes, then catches the generation bump and re-pushes.
* Residual window: enough producers to refill a just-drained queue before any of them re-checks the
  generation. Closed cheaply by having `LoadBalance` **re-drain a quarantined worker's event queue
  on every subsequent tick**, not just at rescue time. No protocol change, ~10 lines.

### 7.3 TLS captured across a suspend point

Any handler caching `CLIO_CUR_WORKER` / a `Worker*` / a `GetCurrentTask()` reference in a local
across a `co_await` is already fragile and becomes wrong under migration. Grep the ChiMods and
audit each use for suspend-point crossing — cheap and mechanical, do it in P0.

### 7.4 Per-thread SHM server affinity

Each worker `ServerInit`s a named segment `clio-<pid>-<tid>` on its own thread
(`worker.cc:242-247`). If any response path is keyed to the *worker's* tid rather than the
*client's*, migration breaks it. Read `IpcManager::GetTls()` / `SendRuntime` before P2.

### 7.5 ZMQ socket thread-ownership (gates D8)

*"ZeroMQ sockets are not safe to share across threads, so each socket has exactly one owner
thread"* (`default_sched.cc:170-171`). D8 relies on the weaker, documented ØMQ property — a socket
*may* migrate between threads given a full memory barrier and no concurrent use — rather than on
socket recreation. **Verify that reading of ØMQ against the version in use before building on it**;
if it does not hold, D8 needs socket teardown/recreate on handoff, which is a much larger change and
would push the net pollers back to non-migratable.

Second-order: `IpcCpu2CpuZmq` and the admin recv threads cache `GetMainTransport()`
(`ipc_manager.h:280`) on their own background threads. Those are unaffected by worker placement, but
confirm none of them shares a socket with the periodic pollers.

### 7.6 GPU worker rescue (spike)

`gpu_lanes_` are polled by their owning worker, and `SuspendMe` (`worker.cc:483-486`) early-returns
for GPU workers because they must never sleep. Determine whether `SetGpuLanes` can transfer lanes to
a replacement thread, or whether device-context affinity makes GPU workers non-migratable like net
workers. Only needed if we intend to rescue them at all.

### 7.7 CI noise

`force_net` / stress tests were already flaky on `dev` at the #782 merge (different test each run,
clear on retry). Do not read one red run as a regression; re-run and compare against a same-day
`dev` baseline.

---

## 8. Open decisions

*Closed:* suspended-task tracking → D4 (park in the blocked queue). Event-queue overflow → §7.2
(keep blocking `Emplace`, re-drain quarantined queues). How exclusivity is expressed → D8a
(`TASK_DEDICATED` flag, bit 10).

1. **Awaited-future handle (D4).** Keep `awaited_fshm_` as a raw `const void*` (`task.h:885`) and
   document the "safe while suspended" invariant, or promote it to an owning `Future`? The raw
   pointer is sound today only because the suspended parent's frame keeps the `FutureShm` alive —
   an invariant the readiness scan now depends on, where before it only did a pointer comparison.
   Recommendation: promote it; it is a small change and removes a lifetime argument from the
   critical path.
2. **Client socket group (D8b).** Merge `kClientRecv`+`kClientSend` into one periodic task, or add a
   co-location group to the mapper? Recommendation: merge — no new scheduler concept.
3. **Readiness-scan backoff.** Reuse the existing `% 2/4/8/16` buckets keyed on yield count, or
   re-key them on *time* parked? Yield count no longer means much for a task that suspends once and
   waits a long time.
4. **Wedged-worker fate.** When the bad task finally returns, does the worker rejoin the pool or
   retire? Rejoining risks re-wedging on the same task class; retiring leaks a thread per bad task
   until exit. Recommendation: rejoin with exponential-backoff quarantine.
5. **All-workers-stalled.** `LoadBalance` already detects it (`default_sched.cc:383`). With an
   unbounded elastic pool the answer is "spawn as many replacements as needed" — confirm there is
   no cap, and that the loud warning stays.
6. **Do the periodic buckets stay four-deep?** The `% 2/4/8/16` scheme is a hand-rolled timer wheel.
   Optional cleanup, unrelated to correctness.

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
* **P2a-specific** (the #705 risk surface):
  * **no spurious resume** — a task parked awaiting a slow subtask must not be resumed by the
    readiness scan before that subtask completes. Assert with a counter on the await return path;
    this is the regression that would silently corrupt reads/writes rather than hang.
  * parked task is resumed exactly once when its event arrives, and its blocked-queue entry is
    gone afterwards (no stale entry, no double-park on re-suspend);
  * the lost-wakeup branch fires and logs when an event is deliberately dropped;
  * `num_blocked_tasks_` now tracks the real suspended count;
  * leak-detection suite against the new owning reference (D4, ownership note).
* **P2b-specific** (D8, net roles):
  * **the learner test** — run a net poller for long enough that SGD converges, then assert it still
    holds a worker to itself. This is the failure mode that passes on day one and regresses quietly.
    With `TASK_DEDICATED` this should be structurally impossible; the test is there to prove nobody
    reintroduced cost-based exclusivity.
  * a `TASK_DEDICATED` task keeps the **same** worker across many periods (sticky binding, D8a-1) —
    its socket must not hop threads every tick;
  * no non-dedicated task is ever routed onto a dedicated worker (D8a-2);
  * with all workers dedicated, a new ordinary task causes a spawn rather than landing on one;
  * degraded mode: with workers > cores, dedicated tasks collapse onto shared workers **without**
    splitting a socket group;
  * `kClientRecv`/`kClientSend` work never runs on two workers at once (assert one live instance);
  * kill the worker hosting a poller mid-flight → the poller migrates and networking recovers, which
    is the §2.3 outage becoming survivable and is the headline result for this phase;
  * throughput/latency A/B vs. the static net workers — dedicated-by-load must not be worse than
    dedicated-by-role.
* **P2c-specific** (periodic, D4b):
  * **the self-undo test** — migrate a periodic task, let one full period elapse, assert it did
    *not* route back onto the stalled worker's lane. This is the failure that looks like a working
    rescue for 500 ms and then silently stops.
  * a role-pinned admin poller (14/15/20/21) is **not** migrated, and the rescue path says so;
  * a queue of long-stranded periodic tasks does not all fire in the same tick after migration
    (`BlockStart()` reset);
  * a migrated periodic task whose `Lane()` went null is never silently dropped
    (`worker.cc:1247-1251`) — add the missing log there regardless.
* **Node-level:** stall a compute worker under network load and assert peer traffic and client IPC
  keep flowing — i.e. the §2.3 outage does not occur. Worth having even though compute workers are
  not where that risk lives, because it is the regression test for D7's role gating.
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

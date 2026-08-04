# Leader Election & Failover: Design (issue #856)

Status: DRAFT for review · Branch: `856-leader-election-failover` · 2026-08-04

## 1. What the leader is for

Clio's "leader election" is not consensus. Every node independently computes

```
leader = lowest alive node_id            (ipc_manager.cc GetLeaderNodeId)
```

from its **local** SWIM membership view. The leader's purposes, in order of
importance:

1. **Exactly-once failure recovery.** When SWIM confirms a node dead, only the
   leader runs `TriggerRecovery` (admin_runtime.cc): it redistributes the dead
   node's containers and broadcasts the new container→node mappings. The
   single-coordinator rule is what prevents N survivors from concurrently
   remapping the same containers.
2. **Collective aggregation rendezvous.** Neighborhood leaders (lowest alive
   id within a neighborhood) are the batch points for ManyToOne collectives.

Everything else (pool creation, routing, barriers) is leader-agnostic.

## 2. Current failure evidence (dev, 2026-08-04, 3 consecutive CI timeouts)

The historical #856 crash (fiber frame corruption in `TriggerRecovery`) was
fixed in July (PR #878: exactly-once origin completion + strict event-resume
guard). The step now fails differently. From the attempt-3 log of run
30910530528:

```
Step 2: kill local runtime (the leader, node 0)
Step 3: Creating pool on new host after reconnection
  Phase 1: ReconnectToOriginalHost x5 (shm_attach ENOENT) — expected, 5s budget
  Phase 2: Trying 16 random hosts
    172.24.0.10 not responding            (the dead leader — expected)
    Connected to 172.24.0.11 (generation=1040199788761)   ← FAILOVER WORKED
  <silence for 15 minutes — step timeout>
```

**The client-side transport failover succeeds.** The hang is the FIRST
operation submitted over the new connection: a `PoolQuery::Dynamic()`
`CreatePool`, whose container fan-out includes the dead node. That request
never completes and never errors. Three protocol gaps compound here:

- **Gap A — tasks addressed to dead nodes hang forever (issue #896).** Before
  SWIM confirms the death (30s direct-probe + suspicion), `SendIn` to the dead
  node waits/retries; after `SetDead`, sends are skipped and
  `ScanSendMapTimeouts` logs "timed out waiting for dead node" — but the
  origin future is never completed. A broadcast with one replica on a dead
  node therefore never finishes. The 15-minute timeout proves this is a
  permanent hang, not slow detection.
- **Gap B — no epoch/fencing on membership-derived decisions.** "Leader" and
  "alive" are per-node opinions. During churn, two nodes can act as leader
  simultaneously (double recovery) or a stale leader can keep coordinating
  after the survivors moved on. Recovery, container remaps, and collectives
  carry no membership version, so stale actors cannot be detected or refused.
- **Gap C — failure detection is starvation-fragile.** SWIM probe responses
  ride ordinary admin tasks; on an oversubscribed host a >30s CPU stall reads
  as death (observed in #894: a demonstrably-alive node was declared dead,
  recovery reshuffled its containers, every barrier cascaded). False-positive
  death is *destructive* — recovery moves containers — so the cost asymmetry
  demands a high-confidence signal.

## 3. Design goals

1. A task whose target is (or becomes) dead completes with a network-timeout
   RC within a bounded time. No caller ever hangs on a dead node.
2. Recovery is exactly-once **per membership change**, enforceable even under
   divergent views: a stale coordinator's actions are refused, not raced.
3. Failure detection distinguishes "dead" from "starved" cheaply enough for
   CI-class CPU budgets.
4. The existing deterministic min-alive-id rule stays. It is simple, has no
   election traffic, and is fine *once views agree*; the fixes are fencing and
   completion, not Paxos/Raft.

## 4. Proposed design

### 4.1 Dead-node task completion (Gap A — fixes the live CI hang, issue #896)

Ownership of "complete the origin with an error" must live in exactly one
place: the send map.

- `SetDead(node)` walks `send_map_` and completes every in-flight task whose
  target is the dead node with the #628 network-timeout RC, via the same
  `ClaimOrigin()` linearization used by the exactly-once completion path
  (PR #878), so a late response cannot double-complete.
- `SendIn` to a node already marked dead completes the task immediately with
  the same RC instead of parking it in a retry queue ("skip" today = park
  forever).
- Broadcast/replicated tasks: a dead replica contributes its timeout RC to
  aggregation instead of blocking it; the origin sees a partial-failure RC and
  can retry against the post-recovery layout.
- `ScanSendMapTimeouts` becomes the backstop (bounded deadline), not the
  primary mechanism, and it must *complete* the origin, not only log.

Acceptance: kill a node, submit a task routed to it → `Wait()` returns nonzero
within the retry deadline. This alone should un-hang the leader_elect test.

### 4.2 Membership epochs + fencing (Gap B)

Introduce a monotonically increasing **membership epoch** (u64), bumped by the
leader on every confirmed membership change (death or join), carried in the
host table gossip and stamped on:

- `TriggerRecovery` / container-remap broadcasts: receivers reject stamps
  older than their current epoch (`kStaleEpochRc`). A stale leader's recovery
  is refused everywhere rather than half-applied.
- Recovery idempotence: `(epoch, dead_node_id)` identifies one recovery
  round; a re-elected or duplicate leader re-running the same round is a
  no-op, two leaders in different epochs cannot interleave destructively.
- `UpdateContainerNodeMapping`: mappings record the epoch that produced them;
  an older-epoch update never overwrites a newer one (today: last-writer-wins
  by arrival order).

This is fencing, not consensus: nodes may still disagree transiently, but a
loser's actions are refused deterministically instead of corrupting state.

### 4.3 Failure-detection hardening (Gap C)

- **Priority heartbeats:** SWIM probe/ack handling moves to (or is mirrored
  on) the dedicated network worker so a starved task-worker pool cannot delay
  acks. A node that can move bytes is not dead.
- **Cheap liveness cross-check before `SetDead`:** the confirming node
  verifies the TCP connection is actually refused/unreachable (transport-level
  probe) rather than relying purely on task-level ack timeouts. Process death
  (the real signal in leader_elect: `shm_attach` ENOENT, connection refused)
  is distinguishable from slowness in one syscall.
- Timeouts stay configurable per deployment (`swim:` YAML block, already
  present); CI keeps lax-or-disabled settings for suites that don't test
  failure detection (done for coherence in #895).

### 4.4 Post-failover client contract

Client failover (Phase 1 retry-original → Phase 2 random survivors) already
works. Contract additions:

- After `ReconnectToNewHost`, the first submissions may race the cluster's own
  death detection. With 4.1 they fail fast with a retryable RC instead of
  hanging; the client API surfaces this as a normal retryable error.
- Phase 2 should skip hosts currently marked dead in the client's host table
  and prefer the presumptive leader (lowest alive id) rather than random
  order — deterministic reconnect targets make post-failover behavior
  reproducible under test.

## 5. Implementation plan (phased, each independently landable)

| Phase | Scope | Risk | Validates |
|---|---|---|---|
| 1 | 4.1 dead-node completion (send-map sweep on SetDead, SendIn fast-fail, broadcast partial-RC) + regression test | Medium (touches completion paths hardened by #878 — reuse `ClaimOrigin`) | leader_elect Phase-1 hang; #896 |
| 2 | 4.4 client failover polish (skip-dead, leader-first ordering) | Low | leader_elect determinism |
| 3 | 4.2 membership epoch + fenced recovery/remap | Medium | double-recovery under partition-ish churn (new test) |
| 4 | 4.3 priority heartbeats + transport-level liveness check | Medium | SWIM false-positive rate under cpulimit CI conditions |

Phase 1 is the CI unblocker and should ship first and alone.

## 6. Test plan

- **Unit/integration (new):** `test_dead_node_completion` — mark a node dead,
  assert every submission shape (Local-remote, Dynamic, Broadcast replica)
  errors within the deadline. Runs single-cluster in the existing
  leader_elect docker harness.
- **leader_elect suite:** unchanged assertions; Phase 1 of this plan should
  turn the current 15-minute hang into a pass (or a fast, diagnosable RC
  failure). Keep the `cpus: 0.5` overlay technique from #894 to reproduce CI
  timing locally.
- **Churn test (Phase 3):** kill the leader *during* an in-flight recovery of
  another node; assert single effective recovery per epoch and no divergent
  container maps (compare `GetContainerNodeId` across survivors).
- **Starvation test (Phase 4):** cpulimit cluster under load with SWIM
  enabled and default timeouts; assert zero false `SetDead` over N minutes.

## 7. Non-goals

- Raft/Paxos-style consensus, leases, or a replicated log. The min-alive-id
  rule plus epoch fencing covers Clio's coordination needs at far lower
  complexity.
- Changing SWIM's gossip structure or membership wire format beyond the epoch
  field.
- The July #856 fiber-corruption class: fixed by PR #878; nothing here
  reopens those paths beyond reusing `ClaimOrigin`.

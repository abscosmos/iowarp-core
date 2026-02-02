# IOWarp Scheduler Development Guide

## Overview

The Chimaera runtime uses a pluggable scheduler architecture to control how tasks are mapped to workers and how workers are organized. This document explains how to build custom schedulers for the IOWarp runtime.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Scheduler Interface](#scheduler-interface)
3. [Worker Lifecycle](#worker-lifecycle)
4. [Implementing a Custom Scheduler](#implementing-a-custom-scheduler)
5. [DefaultScheduler Example](#defaultscheduler-example)
6. [Best Practices](#best-practices)
7. [Integration Points](#integration-points)

## Architecture Overview

### Component Responsibilities

The IOWarp runtime separates concerns across three main components:

- **ConfigManager**: Manages configuration (number of threads, queue depth, etc.)
- **WorkOrchestrator**: Creates workers, spawns threads, assigns lanes to workers
- **Scheduler**: Decides worker partitioning, task-to-worker mapping, and load balancing
- **IpcManager**: Manages shared memory, queues, and provides task routing infrastructure

### Data Flow

```
┌─────────────────┐
│  ConfigManager  │──→ num_threads, queue_depth
└─────────────────┘
         │
         ↓
┌─────────────────┐
│ WorkOrchestrator│──→ Creates num_threads + 1 workers
└─────────────────┘
         │
         ↓
┌─────────────────┐
│   Scheduler     │──→ Partitions workers into groups
└─────────────────┘    Updates IpcManager with scheduler worker count
         │
         ↓
┌─────────────────┐
│ WorkOrchestrator│──→ Gets task-processing workers from scheduler
└─────────────────┘    Maps lanes to those workers (1:1)
         │
         ↓
┌─────────────────┐
│   IpcManager    │──→ num_sched_queues used for client task mapping
└─────────────────┘
```

## Scheduler Interface

All schedulers must inherit from the `Scheduler` base class and implement the following methods:

### Required Methods

```cpp
class Scheduler {
public:
  // Partition workers into groups after WorkOrchestrator creates them
  virtual void DivideWorkers(WorkOrchestrator *work_orch) = 0;

  // Get the list of workers that process tasks from worker queues
  virtual std::vector<Worker*> GetTaskProcessingWorkers() = 0;

  // Map tasks from clients to worker lanes
  virtual u32 ClientMapTask(IpcManager *ipc_manager, const Future<Task> &task) = 0;

  // Map tasks from runtime workers to other workers
  virtual u32 RuntimeMapTask(Worker *worker, const Future<Task> &task) = 0;

  // Rebalance load across workers (called periodically by workers)
  virtual void RebalanceWorker(Worker *worker) = 0;

  // Adjust polling intervals for periodic tasks
  virtual void AdjustPolling(RunContext *run_ctx) = 0;
};
```

### Method Details

#### `DivideWorkers(WorkOrchestrator *work_orch)`

**Purpose**: Partition workers into functional groups after they've been created.

**Called**: Once during initialization, after WorkOrchestrator creates all workers.

**Responsibilities**:
- Access workers via `work_orch->GetWorker(worker_id)`
- Set worker thread types via `worker->SetThreadType(thread_type)`
- Organize workers into scheduler-specific groups
- **Update IpcManager** with actual scheduler worker count via `IpcManager::SetNumSchedQueues()`

**Example**:
```cpp
void MyScheduler::DivideWorkers(WorkOrchestrator *work_orch) {
  u32 total_workers = work_orch->GetTotalWorkerCount();

  // Partition workers: first N-1 are task workers, last is network
  for (u32 i = 0; i < total_workers - 1; ++i) {
    Worker *worker = work_orch->GetWorker(i);
    worker->SetThreadType(kSchedWorker);
    task_workers_.push_back(worker);
  }

  // Last worker is network worker
  Worker *net_worker = work_orch->GetWorker(total_workers - 1);
  net_worker->SetThreadType(kNetWorker);
  net_worker_ = net_worker;

  // IMPORTANT: Update IpcManager with actual scheduler worker count
  IpcManager *ipc = CHI_IPC;
  if (ipc) {
    ipc->SetNumSchedQueues(task_workers_.size());
  }
}
```

#### `GetTaskProcessingWorkers()`

**Purpose**: Return the list of workers that should process tasks from worker queues.

**Called**: By WorkOrchestrator during `SpawnWorkerThreads()` for lane assignment.

**Responsibilities**:
- Return a vector of workers that will be assigned to task queue lanes
- These workers receive tasks from clients and runtime
- Typically excludes network workers or other specialized workers

**Example**:
```cpp
std::vector<Worker*> MyScheduler::GetTaskProcessingWorkers() {
  return task_workers_;  // Return your scheduler-specific worker group
}
```

#### `ClientMapTask(IpcManager *ipc_manager, const Future<Task> &task)`

**Purpose**: Determine which worker lane a task from a client should be assigned to.

**Called**: When clients submit tasks to the runtime.

**Responsibilities**:
- Return a lane ID in range `[0, num_sched_queues)`
- Use `ipc_manager->GetNumSchedQueues()` to get valid lane count
- Common strategies: PID+TID hash, round-robin, locality-aware

**Example**:
```cpp
u32 MyScheduler::ClientMapTask(IpcManager *ipc_manager, const Future<Task> &task) {
  u32 num_lanes = ipc_manager->GetNumSchedQueues();

  // PID+TID hash-based mapping
  auto *sys_info = HSHM_SYSTEM_INFO;
  pid_t pid = sys_info->pid_;
  auto tid = HSHM_THREAD_MODEL->GetTid();

  size_t hash = std::hash<pid_t>{}(pid) ^ (std::hash<void*>{}(&tid) << 1);
  return static_cast<u32>(hash % num_lanes);
}
```

#### `RuntimeMapTask(Worker *worker, const Future<Task> &task)`

**Purpose**: Determine which worker should execute a task when routing from within the runtime.

**Called**: When workers route tasks to other workers (work stealing, load balancing).

**Responsibilities**:
- Return a worker ID for task execution
- Can implement work migration, affinity, or return current worker

**Example**:
```cpp
u32 MyScheduler::RuntimeMapTask(Worker *worker, const Future<Task> &task) {
  // No migration - task stays on current worker
  return worker ? worker->GetId() : 0;
}
```

#### `RebalanceWorker(Worker *worker)`

**Purpose**: Balance load across workers by stealing or delegating tasks.

**Called**: Periodically by workers after processing tasks.

**Responsibilities**:
- Implement work stealing algorithms
- Migrate tasks between workers
- Optional - can be a no-op for simple schedulers

**Example**:
```cpp
void MyScheduler::RebalanceWorker(Worker *worker) {
  // Simple schedulers can leave this empty
  // Advanced schedulers can implement work stealing here
}
```

#### `AdjustPolling(RunContext *run_ctx)`

**Purpose**: Adjust polling intervals for periodic tasks based on work done.

**Called**: After each execution of a periodic task.

**Responsibilities**:
- Modify `run_ctx->yield_time_us_` based on `run_ctx->did_work_`
- Implement adaptive polling (exponential backoff when idle)
- Reduce CPU usage for idle periodic tasks

**Example**:
```cpp
void MyScheduler::AdjustPolling(RunContext *run_ctx) {
  const double kMaxPollingIntervalUs = 100000.0; // 100ms

  if (run_ctx->did_work_) {
    // Reset to original period when work is done
    run_ctx->yield_time_us_ = run_ctx->true_period_ns_ / 1000.0;
  } else {
    // Exponential backoff when idle
    run_ctx->yield_time_us_ = std::min(
      run_ctx->yield_time_us_ * 2.0,
      kMaxPollingIntervalUs
    );
  }
}
```

## Worker Lifecycle

Understanding the worker lifecycle is crucial for scheduler implementation:

```
1. ConfigManager loads configuration (num_threads, queue_depth)
   ↓
2. WorkOrchestrator::Init()
   - Creates num_threads + 1 workers (all initially kSchedWorker type)
   - Calls Scheduler::DivideWorkers()
   ↓
3. Scheduler::DivideWorkers()
   - Partitions workers into functional groups
   - Sets thread types (kSchedWorker, kNetWorker, etc.)
   - Updates IpcManager::SetNumSchedQueues()
   ↓
4. WorkOrchestrator::StartWorkers()
   - Calls SpawnWorkerThreads()
   - Gets task-processing workers from Scheduler::GetTaskProcessingWorkers()
   - Maps lanes to those workers (1:1 mapping)
   - Spawns actual OS threads
   ↓
5. Workers run task processing loops
   - Process tasks from assigned lanes
   - Call Scheduler::RuntimeMapTask() for task routing
   - Call Scheduler::RebalanceWorker() periodically
```

## Implementing a Custom Scheduler

### Step 1: Create Header File

Create `context-runtime/include/chimaera/scheduler/my_scheduler.h`:

```cpp
#ifndef CHIMAERA_INCLUDE_CHIMAERA_SCHEDULER_MY_SCHEDULER_H_
#define CHIMAERA_INCLUDE_CHIMAERA_SCHEDULER_MY_SCHEDULER_H_

#include <vector>
#include "chimaera/scheduler/scheduler.h"

namespace chi {

class MyScheduler : public Scheduler {
public:
  MyScheduler() = default;
  ~MyScheduler() override = default;

  // Implement required interface methods
  void DivideWorkers(WorkOrchestrator *work_orch) override;
  std::vector<Worker*> GetTaskProcessingWorkers() override;
  u32 ClientMapTask(IpcManager *ipc_manager, const Future<Task> &task) override;
  u32 RuntimeMapTask(Worker *worker, const Future<Task> &task) override;
  void RebalanceWorker(Worker *worker) override;
  void AdjustPolling(RunContext *run_ctx) override;

private:
  // Your scheduler-specific state
  std::vector<Worker*> task_workers_;
  Worker *net_worker_;
};

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_SCHEDULER_MY_SCHEDULER_H_
```

### Step 2: Implement Methods

Create `context-runtime/src/scheduler/my_scheduler.cc`:

```cpp
#include "chimaera/scheduler/my_scheduler.h"
#include "chimaera/config_manager.h"
#include "chimaera/ipc_manager.h"
#include "chimaera/work_orchestrator.h"
#include "chimaera/worker.h"

namespace chi {

void MyScheduler::DivideWorkers(WorkOrchestrator *work_orch) {
  u32 total_workers = work_orch->GetTotalWorkerCount();

  // Partition: all but last worker are task workers
  for (u32 i = 0; i < total_workers - 1; ++i) {
    Worker *worker = work_orch->GetWorker(i);
    worker->SetThreadType(kSchedWorker);
    task_workers_.push_back(worker);
  }

  // Last worker is network worker
  Worker *net_worker = work_orch->GetWorker(total_workers - 1);
  net_worker->SetThreadType(kNetWorker);
  net_worker_ = net_worker;

  // CRITICAL: Update IpcManager with actual scheduler worker count
  IpcManager *ipc = CHI_IPC;
  if (ipc) {
    ipc->SetNumSchedQueues(task_workers_.size());
  }
}

std::vector<Worker*> MyScheduler::GetTaskProcessingWorkers() {
  return task_workers_;
}

u32 MyScheduler::ClientMapTask(IpcManager *ipc_manager, const Future<Task> &task) {
  u32 num_lanes = ipc_manager->GetNumSchedQueues();
  // Implement your mapping strategy here
  return 0; // Simple: always map to lane 0
}

u32 MyScheduler::RuntimeMapTask(Worker *worker, const Future<Task> &task) {
  return worker ? worker->GetId() : 0;
}

void MyScheduler::RebalanceWorker(Worker *worker) {
  // Implement work stealing or leave empty
}

void MyScheduler::AdjustPolling(RunContext *run_ctx) {
  // Implement adaptive polling or leave with default behavior
}

}  // namespace chi
```

### Step 3: Register Scheduler

Update `context-runtime/src/ipc_manager.cc` to create your scheduler:

```cpp
bool IpcManager::ServerInit() {
  // ... existing initialization code ...

  // Create scheduler based on configuration
  ConfigManager *config = CHI_CONFIG_MANAGER;
  std::string sched_name = config->GetLocalSched();

  if (sched_name == "my_scheduler") {
    scheduler_ = new MyScheduler();
  } else if (sched_name == "default") {
    scheduler_ = new DefaultScheduler();
  } else {
    HLOG(kError, "Unknown scheduler: {}", sched_name);
    return false;
  }

  return true;
}
```

### Step 4: Configure

Update your configuration file to use the new scheduler:

```yaml
runtime:
  local_sched: "my_scheduler"
  num_threads: 4
  queue_depth: 1024
```

## DefaultScheduler Example

The `DefaultScheduler` provides a reference implementation with these characteristics:

### Worker Partitioning
- All `num_threads` workers are scheduler workers (task-processing)
- Last worker (beyond num_threads) is network worker
- No separate slow worker pool

### Task Mapping Strategy
- **Client Tasks**: PID+TID hash-based mapping
  - Ensures different processes/threads use different lanes
  - Provides load distribution
- **Runtime Tasks**: No migration (tasks stay on current worker)

### Load Balancing
- No active rebalancing (simple design)
- Tasks processed by worker that picks them up

### Polling Adjustment
- Exponential backoff for idle periodic tasks
- Doubles polling interval when no work done
- Caps at 100ms maximum interval
- Resets to original period when work resumes

### Code Reference

See implementation in:
- Header: `context-runtime/include/chimaera/scheduler/default_sched.h`
- Implementation: `context-runtime/src/scheduler/default_sched.cc`

## Best Practices

### 1. Always Update IpcManager in DivideWorkers

```cpp
void MyScheduler::DivideWorkers(WorkOrchestrator *work_orch) {
  // ... partition workers ...

  // CRITICAL: Update IpcManager with actual scheduler worker count
  IpcManager *ipc = CHI_IPC;
  if (ipc) {
    ipc->SetNumSchedQueues(actual_scheduler_worker_count);
  }
}
```

**Why**: Clients use `GetNumSchedQueues()` to map tasks to lanes. If this doesn't match the actual number of task-processing workers, tasks will be mapped to non-existent or wrong workers.

### 2. Return Consistent Worker Lists

```cpp
std::vector<Worker*> MyScheduler::GetTaskProcessingWorkers() {
  return task_workers_;  // Must be stable after DivideWorkers
}
```

**Why**: WorkOrchestrator uses this list for lane assignment. Changing the list after initialization will break lane mappings.

### 3. Validate Lane IDs

```cpp
u32 MyScheduler::ClientMapTask(IpcManager *ipc_manager, const Future<Task> &task) {
  u32 num_lanes = ipc_manager->GetNumSchedQueues();
  u32 lane = ComputeLane(...);

  // Ensure lane is in valid range
  return lane % num_lanes;
}
```

### 4. Handle Null Pointers

```cpp
void MyScheduler::DivideWorkers(WorkOrchestrator *work_orch) {
  if (!work_orch) {
    HLOG(kError, "work_orch is null");
    return;
  }
  // ... proceed ...
}
```

### 5. Consider Thread Safety

If your scheduler maintains shared state accessed by multiple workers:
- Use atomic operations for counters
- Use mutexes for complex data structures
- Prefer lock-free designs when possible

### 6. Minimize DivideWorkers Overhead

`DivideWorkers` is called during initialization:
- Avoid expensive operations
- Don't allocate large data structures
- Keep partitioning logic simple

### 7. Test with Different Configurations

Test your scheduler with various `num_threads` values:
- Single thread (num_threads = 1)
- Small (num_threads = 2-4)
- Large (num_threads = 16+)

## Integration Points

### Singletons and Macros

Access runtime components via global macros:

```cpp
// Configuration
ConfigManager *config = CHI_CONFIG_MANAGER;
u32 num_threads = config->GetNumThreads();

// IPC Manager
IpcManager *ipc = CHI_IPC;
u32 num_lanes = ipc->GetNumSchedQueues();

// System Info
auto *sys_info = HSHM_SYSTEM_INFO;
pid_t pid = sys_info->pid_;

// Thread Model
auto tid = HSHM_THREAD_MODEL->GetTid();
```

### Worker Access

Access workers through WorkOrchestrator:

```cpp
u32 total_workers = work_orch->GetTotalWorkerCount();
Worker *worker = work_orch->GetWorker(worker_id);

// Set thread type
worker->SetThreadType(kSchedWorker);

// Get worker properties
u32 id = worker->GetId();
ThreadType type = worker->GetThreadType();
```

### Logging

Use Hermes logging macros:

```cpp
HLOG(kInfo, "Scheduler initialized with {} workers", num_workers);
HLOG(kDebug, "Mapping task to lane {}", lane_id);
HLOG(kWarning, "Worker {} has empty queue", worker_id);
HLOG(kError, "Invalid configuration: {}", error_msg);
```

### Configuration Access

Read configuration values:

```cpp
ConfigManager *config = CHI_CONFIG_MANAGER;
u32 num_threads = config->GetNumThreads();
u32 queue_depth = config->GetQueueDepth();
std::string sched_name = config->GetLocalSched();
```

## Advanced Topics

### Work Stealing

Implement work stealing in `RebalanceWorker`:

```cpp
void MyScheduler::RebalanceWorker(Worker *worker) {
  // Check if worker has no work
  TaskLane *my_lane = worker->GetLane();
  if (my_lane->Empty()) {
    // Try to steal from other workers
    for (Worker *victim : task_workers_) {
      if (victim == worker) continue;

      TaskLane *victim_lane = victim->GetLane();
      if (!victim_lane->Empty()) {
        // Steal task from victim
        Future<Task> stolen_task;
        if (victim_lane->Pop(stolen_task)) {
          my_lane->Push(stolen_task);
          break;
        }
      }
    }
  }
}
```

### Locality-Aware Mapping

Map tasks based on data locality:

```cpp
u32 MyScheduler::ClientMapTask(IpcManager *ipc_manager, const Future<Task> &task) {
  // Extract data location from task
  PoolId pool_id = task->pool_id_;

  // Map to worker closest to data
  return ComputeLocalityMap(pool_id, ipc_manager->GetNumSchedQueues());
}
```

### Priority-Based Scheduling

Use task priorities for scheduling:

```cpp
void MyScheduler::DivideWorkers(WorkOrchestrator *work_orch) {
  // Partition workers into high-priority and low-priority groups
  u32 total = work_orch->GetTotalWorkerCount();
  u32 high_prio_count = total / 2;

  for (u32 i = 0; i < high_prio_count; ++i) {
    high_priority_workers_.push_back(work_orch->GetWorker(i));
  }

  for (u32 i = high_prio_count; i < total - 1; ++i) {
    low_priority_workers_.push_back(work_orch->GetWorker(i));
  }

  // Network worker
  net_worker_ = work_orch->GetWorker(total - 1);
}
```

## Troubleshooting

### Tasks Not Being Processed

**Symptom**: Tasks submitted but never execute

**Check**:
1. Did you call `IpcManager::SetNumSchedQueues()` in `DivideWorkers`?
2. Does `GetTaskProcessingWorkers()` return the correct workers?
3. Are lanes properly mapped in WorkOrchestrator?

### Client Mapping Errors

**Symptom**: Assertion failures or crashes in `ClientMapTask`

**Check**:
1. Is returned lane ID in range `[0, num_sched_queues)`?
2. Did you check for `num_lanes == 0`?
3. Are you using modulo to wrap lane IDs?

### Worker Crashes

**Symptom**: Workers crash during initialization

**Check**:
1. Are you checking for null pointers?
2. Does `DivideWorkers` handle `total_workers < expected`?
3. Are worker thread types set correctly?

## References

- **Scheduler Interface**: `context-runtime/include/chimaera/scheduler/scheduler.h`
- **DefaultScheduler**: `context-runtime/src/scheduler/default_sched.cc`
- **WorkOrchestrator**: `context-runtime/src/work_orchestrator.cc`
- **IpcManager**: `context-runtime/src/ipc_manager.cc`
- **Configuration**: `context-runtime/docs/deployment.md`

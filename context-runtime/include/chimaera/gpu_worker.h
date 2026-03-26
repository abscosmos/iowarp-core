/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CHIMAERA_INCLUDE_CHIMAERA_GPU_WORKER_H_
#define CHIMAERA_INCLUDE_CHIMAERA_GPU_WORKER_H_

// Set to 1 to enable verbose GPU worker debug printf (very slow)
#ifndef CHI_GPU_WORKER_DEBUG
#define CHI_GPU_WORKER_DEBUG 0
#endif

#if CHI_GPU_WORKER_DEBUG
#define GPU_WORKER_DPRINTF(...) printf(__VA_ARGS__)
#else
#define GPU_WORKER_DPRINTF(...) ((void)0)
#endif

#include "chimaera/gpu_container.h"
#include "chimaera/gpu_pool_manager.h"
#include "chimaera/gpu_work_orchestrator.h"
#include "chimaera/task.h"
#include "chimaera/local_task_archives.h"
#include "hermes_shm/lightbeam/shm_transport.h"

namespace chi {
namespace gpu {

// kWarpSize defined in gpu_coroutine.h

/**
 * GPU-side warp-level worker.
 *
 * Each warp in the GPU work orchestrator gets one Worker. Lane 0 of the
 * warp acts as the scheduler (queue polling, deserialization, serialization,
 * suspended task management). ALL 32 threads call container->Run() so that
 * tasks can distribute work across the warp via rctx.lane_id_.
 *
 * No STL, no TLS — all HSHM_GPU_FUN.  Container::Run() returns
 * chi::gpu::TaskResume (a C++20 coroutine) enabling cooperative yielding.
 */
class Worker {
 public:
  u32 worker_id_;                    /**< Worker identity (= warp ID) */
  u32 lane_id_;                      /**< Queue lane this warp polls */
  volatile bool is_running_;         /**< Running flag for the poll loop */
  TaskQueue *cpu2gpu_queue_;         /**< CPU → GPU queue (GPU work orchestrator polls) */
  TaskQueue *gpu2gpu_queue_;         /**< GPU → GPU queue (GPU work orchestrator polls) */
  TaskQueue *internal_queue_;        /**< Internal subtask queue (GPU orchestrator polls) */
  TaskQueue *warp_group_queue_;      /**< Cross-warp sub-task queue */
  char *warp_group_queue_base_;      /**< Base of warp group queue backend */
  PoolManager *pool_mgr_;            /**< GPU-side container lookup table */
  char *queue_backend_base_;         /**< Base of queue backend for ShmPtr resolution */
  RunContext rctx_;                   /**< Template RunContext (copied per task) */
  WorkOrchestratorControl *dbg_ctrl_; /**< Debug control struct (pinned, CPU-readable) */

  /**
   * Active RunContext pointer. Lane 0 sets this, then broadcasts
   * to all lanes via __shfl_sync after __syncwarp().
   */
  RunContext *active_ctx_;

  /** Suspended task queue (ext_ring_buffer of RunContext*), lane 0 only */
  using SuspendedQueue = hshm::ipc::ext_ring_buffer<RunContext*, CHI_PRIV_ALLOC_T>;
  SuspendedQueue *suspended_;
  u32 num_suspended_;

  /**
   * Initialize the worker with queue and pool manager pointers.
   */
  HSHM_GPU_FUN void Init(u32 worker_id,
                          u32 lane_id,
                          TaskQueue *cpu2gpu_queue,
                          TaskQueue *gpu2gpu_queue,
                          TaskQueue *internal_queue,
                          PoolManager *pool_mgr,
                          char *queue_backend_base,
                          WorkOrchestratorControl *dbg_ctrl,
                          TaskQueue *warp_group_queue = nullptr,
                          char *warp_group_queue_base = nullptr) {
    worker_id_ = worker_id;
    lane_id_ = lane_id;
    cpu2gpu_queue_ = cpu2gpu_queue;
    gpu2gpu_queue_ = gpu2gpu_queue;
    internal_queue_ = internal_queue;
    warp_group_queue_ = warp_group_queue;
    warp_group_queue_base_ = warp_group_queue_base;
    pool_mgr_ = pool_mgr;
    queue_backend_base_ = queue_backend_base;
    dbg_ctrl_ = dbg_ctrl;
    active_ctx_ = nullptr;
    is_running_ = true;
    num_suspended_ = 0;
    suspended_ = nullptr;
#if HSHM_IS_GPU_COMPILER
    u32 warp_id = IpcManager::GetWarpId();
    u32 warp_lane = IpcManager::GetLaneId();
    if (warp_lane == 0) {
      auto fp = CHI_IPC->NewObj<SuspendedQueue>(CHI_PRIV_ALLOC, 128);
      suspended_ = fp.ptr_;
    }
    rctx_ = RunContext(blockIdx.x, threadIdx.x, warp_id, warp_lane);
#else
    rctx_ = RunContext(0, 0, 0, 0);
#endif
  }

  /** Write debug state to pinned memory (CPU-readable) */
  HSHM_GPU_FUN void DbgState([[maybe_unused]] u32 state) {
#ifndef NDEBUG
    if (dbg_ctrl_ && lane_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_last_state[lane_id_] = state;
      dbg_ctrl_->dbg_num_suspended[lane_id_] = num_suspended_;
    }
#endif
  }

  HSHM_GPU_FUN void DbgPoll() {
#ifndef NDEBUG
    if (dbg_ctrl_ && lane_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_poll_count[lane_id_] = dbg_ctrl_->dbg_poll_count[lane_id_] + 1;
    }
#endif
  }

  HSHM_GPU_FUN void DbgTaskPopped() {
    if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_tasks_popped[worker_id_] = dbg_ctrl_->dbg_tasks_popped[worker_id_] + 1;
    }
  }
  HSHM_GPU_FUN void DbgTaskCompleted() {
    if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_tasks_completed[worker_id_] = dbg_ctrl_->dbg_tasks_completed[worker_id_] + 1;
#ifdef HSHM_BUDDY_ALLOC_DEBUG
      u32 completed = dbg_ctrl_->dbg_tasks_completed[worker_id_];
      if (completed % 50 == 0) {
        auto *priv = CHI_PRIV_ALLOC;
        if (priv) {
          printf("[W%u] task#%u priv: allocs=%llu frees=%llu net=%llu "
                 "bigheap=%llu/%llu\n",
                 worker_id_, completed,
                 (unsigned long long)priv->DbgAllocCount(),
                 (unsigned long long)priv->DbgFreeCount(),
                 (unsigned long long)priv->DbgNetBytes(),
                 (unsigned long long)priv->DbgBigHeapOffset(),
                 (unsigned long long)priv->DbgBigHeapMax());
        }
      }
#endif
    }
  }
  HSHM_GPU_FUN void DbgTaskResumed() {
    if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_tasks_resumed[worker_id_] = dbg_ctrl_->dbg_tasks_resumed[worker_id_] + 1;
    }
  }
  HSHM_GPU_FUN void DbgAllocFailure() {
    if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_alloc_failures[worker_id_] = dbg_ctrl_->dbg_alloc_failures[worker_id_] + 1;
    }
  }
  HSHM_GPU_FUN void DbgQueuePop() {
    if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_queue_pops[worker_id_] = dbg_ctrl_->dbg_queue_pops[worker_id_] + 1;
    }
  }
  HSHM_GPU_FUN void DbgNoContainer(unsigned int pool_major, unsigned int pool_minor) {
    if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_no_container[worker_id_] = dbg_ctrl_->dbg_no_container[worker_id_] + 1;
      // Stash the last pool_id that failed
      dbg_ctrl_->dbg_last_method[worker_id_] =
          (pool_major << 16) | (pool_minor & 0xFFFF);
    }
  }

  /**
   * Process one iteration of the warp-level poll loop.
   *
   * Phase 1 (lane 0): Check suspended tasks, pop new tasks from queues,
   *   deserialize, allocate single GpuRunContext, set __shared__ pointer.
   * Phase 2 (all lanes or lane 0): Execute active task — lane 0 creates
   *   the coroutine, all participating lanes resume it via SIMT.
   * Phase 3 (lane 0): Check completion, serialize output or suspend.
   *
   * @param lane_id Warp lane of the calling thread (0-31)
   */
  HSHM_GPU_FUN bool PollOnce(u32 lane_id) {
    bool did_work = false;

    // ================================================================
    // Phase 1: Lane 0 schedules work
    // ================================================================
    if (lane_id == 0) {
      DbgPoll();
      active_ctx_ = nullptr;

      // Check suspended tasks for completion
      if (num_suspended_ > 0) {
        did_work |= CheckAndResumeSuspended();
      }

      // Pop new task if no resumed task and we have capacity
      if (active_ctx_ == nullptr) {
        did_work |= TryPopNewTask();
      }
    }

    // Broadcast active_ctx_ from lane 0 to all lanes
#if HSHM_IS_GPU_COMPILER
    {
      u64 ctx_bits = reinterpret_cast<u64>(active_ctx_);
      u32 lo = static_cast<u32>(ctx_bits);
      u32 hi = static_cast<u32>(ctx_bits >> 32);
      lo = __shfl_sync(0xFFFFFFFF, lo, 0);
      hi = __shfl_sync(0xFFFFFFFF, hi, 0);
      active_ctx_ = reinterpret_cast<RunContext *>(
          (static_cast<u64>(hi) << 32) | lo);
    }
#endif

    // ================================================================
    // Phase 2: Execute the active task (all participating lanes)
    // ================================================================
    RunContext *ctx = active_ctx_;
    if (ctx != nullptr) {
      if (lane_id == 0) {
        GPU_WORKER_DPRINTF("[W%u] Phase2: method %u, parallelism %u\n",
                           worker_id_, ctx->method_id_, ctx->parallelism_);
      }
      bool participate = (ctx->parallelism_ > 1) || (lane_id == 0);

      if (participate) {
        // Determine if coroutines need creation (broadcast from lane 0)
        bool needs_create = (ctx->task_coros_[0] == nullptr);
#if HSHM_IS_GPU_COMPILER
        if (ctx->parallelism_ > 1) {
          int nc = __shfl_sync(0xFFFFFFFF, (int)needs_create, 0);
          needs_create = (nc != 0);
        }
#endif
        // Each participating lane creates its own coroutine.
        // operator new bulk-allocates 32 frames internally via __shfl_sync.
        if (needs_create) {
          auto *container = static_cast<Container *>(ctx->container_);
          TaskResume tmp = container->Run(ctx->method_id_, ctx->task_ptr_, *ctx);
          if (!tmp.get_handle()) {
            if (lane_id == 0) {
              printf("[W%u] CORO ALLOC FAILED: method=%u\n",
                     worker_id_, ctx->method_id_);
            }
          } else {
            tmp.get_handle().promise().set_run_context(ctx);
            ctx->task_coros_[lane_id] = tmp.release();
          }
        }
#if HSHM_IS_GPU_COMPILER
        if (ctx->parallelism_ > 1) __syncwarp();
#endif
        // Each lane resumes its own coroutine
        if (ctx->task_coros_[lane_id] && !ctx->task_coros_[lane_id].done()) {
          if (lane_id == 0) {
            ctx->is_yielded_ = false;
          }
#if HSHM_IS_GPU_COMPILER
          if (ctx->parallelism_ > 1) __syncwarp();
#endif
          // Set per-lane identity — each lane writes to its own slot,
          // no race. Coroutine body reads rctx.lane_ids_[threadIdx.x % 32].
          ctx->lane_ids_[lane_id] = lane_id;
          auto &coro_h = ctx->coro_handles_[lane_id];
          if (coro_h && !coro_h.done()) {
            coro_h.resume();
          } else {
            ctx->task_coros_[lane_id].resume();
          }
        }
        did_work = true;
      }

      // ================================================================
      // Phase 2.5: Destroy completed coroutine frames (all lanes)
      // ================================================================
      // All participating lanes must call destroy() together so that
      // operator delete's __syncwarp + lane-0-free works correctly.
      if (participate && ctx->task_coros_[lane_id] &&
          ctx->task_coros_[lane_id].done()) {
        ctx->task_coros_[lane_id].destroy();
        ctx->task_coros_[lane_id] = nullptr;
        ctx->coro_handles_[lane_id] = nullptr;
      }
    }

#if HSHM_IS_GPU_COMPILER
    __syncwarp();
#endif

    // ================================================================
    // Phase 3: Lane 0 handles completion / suspension
    // ================================================================
    if (lane_id == 0 && ctx != nullptr) {
      // After Phase 2.5, task_coros_[0] == nullptr means the task completed
      bool task_done = (ctx->task_coros_[0] == nullptr);

      if (task_done) {
        DbgState(5);
        DbgTaskCompleted();
        GPU_WORKER_DPRINTF("[W%u] Task DONE: method %u\n",
                           worker_id_, ctx->method_id_);
        auto *fshm = static_cast<FutureShm *>(ctx->task_fshm_);
        if (fshm->total_warps_ > 1) {
          // Cross-warp: increment completion counter
#if !HSHM_IS_HOST
          u32 prev = fshm->completion_counter_.fetch_add(1);
          __threadfence();
#else
          u32 prev = 0;
#endif
          if (prev + 1 < fshm->total_warps_) {
            // Not the last warp — free context and continue
            FreeContext(ctx);
            active_ctx_ = nullptr;
          } else {
            // Last warp: serialize and signal completion
            if (ctx->is_copy_path_) {
              auto *container = static_cast<Container *>(ctx->container_);
              SerializeAndComplete(fshm, container,
                                   ctx->method_id_, ctx->task_ptr_,
                                   ctx->is_gpu2gpu_);
            } else {
              CompleteAndResumeParent(fshm, ctx->is_gpu2gpu_);
            }
            FreeContext(ctx);
            active_ctx_ = nullptr;
          }
        } else {
          // Single-warp path
          if (ctx->is_copy_path_) {
            auto *container = static_cast<Container *>(ctx->container_);
            SerializeAndComplete(fshm, container,
                                 ctx->method_id_, ctx->task_ptr_,
                                 ctx->is_gpu2gpu_);
          } else {
            CompleteAndResumeParent(fshm, ctx->is_gpu2gpu_);
          }
          FreeContext(ctx);
          active_ctx_ = nullptr;
        }
      } else {
        DbgState(3);
        GPU_WORKER_DPRINTF("[W%u] Task SUSPEND: method %u (nsus=%u)\n",
                           worker_id_, ctx->method_id_, num_suspended_ + 1);
        suspended_->Push(ctx);
        ++num_suspended_;
        active_ctx_ = nullptr;
      }
    }

#if HSHM_IS_GPU_COMPILER
    __syncwarp();
#endif

    return did_work;
  }

  /**
   * Signal the worker to stop polling.
   */
  HSHM_GPU_FUN void Stop() {
    is_running_ = false;
  }

  /**
   * Finalize and cleanup the worker.
   */
  HSHM_GPU_FUN void Finalize() {
    is_running_ = false;
  }

 private:
  // ================================================================
  // Context allocation
  // ================================================================

  /**
   * Allocate and initialize a RunContext for a new task.
   * Stores FullPtr<Task> components and dispatch info.
   */
  HSHM_GPU_FUN RunContext *AllocContext(
      Container *container, u32 method_id,
      hipc::FullPtr<Task> task_ptr, FutureShm *fshm,
      bool is_gpu2gpu, bool is_copy_path, u32 parallelism,
      u32 range_off = 0, u32 range_width = 0) {
    auto *ipc = CHI_IPC;
    auto alloc_result = ipc->gpu_alloc_->AllocateObjs<RunContext>(1);
    RunContext *ctx = alloc_result.ptr_;
    if (!ctx) return nullptr;

    // Copy template fields (identity)
    *ctx = rctx_;
    // Set task dispatch fields
    ctx->container_ = container;
    ctx->method_id_ = method_id;
    ctx->parallelism_ = parallelism;
    ctx->task_ptr_ = task_ptr;
    ctx->task_fshm_ = fshm;
    ctx->is_gpu2gpu_ = is_gpu2gpu;
    ctx->is_copy_path_ = is_copy_path;
    ctx->range_off_ = range_off;
    ctx->range_width_ = (range_width > 0) ? range_width : parallelism;
    // Coroutine state starts clean
    ctx->is_yielded_ = false;
    ctx->awaited_fshm_ = nullptr;
    ctx->awaited_task_ = nullptr;
    for (u32 i = 0; i < kWarpSize; ++i) {
      ctx->task_coros_[i] = nullptr;
      ctx->coro_handles_[i] = nullptr;
    }
    return ctx;
  }

  /**
   * Free coroutine frames (if any) and the RunContext.
   *
   * For the active-task path, Phase 2.5 already destroyed frames via
   * operator delete (__syncwarp + lane-0 bulk free).  task_coros_[0]
   * will be nullptr, so FreeFramesDirect() is a no-op.
   *
   * For the suspended-task path (lane 0 only), frames may still exist.
   * FreeFramesDirect() frees the bulk block directly without needing
   * all 32 lanes for __syncwarp.
   */
  HSHM_GPU_FUN void FreeContext(RunContext *ctx) {
    ctx->FreeFramesDirect();
    auto *ipc = CHI_IPC;
    hipc::FullPtr<RunContext> ptr(
        reinterpret_cast<hipc::Allocator *>(ipc->gpu_alloc_), ctx);
    ipc->gpu_alloc_->Free(ptr);
  }


  // ================================================================
  // Suspended task management
  // ================================================================

  /**
   * Check suspended tasks. Pop each, check readiness, re-push if not ready.
   * If one is ready to resume, set active_ctx_.
   * Called only by lane 0.
   */
  HSHM_GPU_FUN bool CheckAndResumeSuspended() {
    bool did_work = false;
    u32 count = num_suspended_;
    for (u32 i = 0; i < count; ++i) {
      RunContext *ctx;
      if (!suspended_->Pop(ctx)) break;

      // Check if the awaited sub-task is complete
      if (ctx->awaited_fshm_) {
        auto *awaited = reinterpret_cast<FutureShm *>(ctx->awaited_fshm_);
        if (!awaited->flags_.AnyDevice(FutureShm::FUTURE_COMPLETE)) {
          // Not ready — re-enqueue
          if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
            u32 raw_flags = awaited->flags_.bits_.load_device();
            dbg_ctrl_->dbg_resume_checks[worker_id_] = dbg_ctrl_->dbg_resume_checks[worker_id_] + 1;
            if ((dbg_ctrl_->dbg_resume_checks[worker_id_] % 100000) == 1) {
              printf("[W%u] awaiting fshm=%p flags=0x%x method=%u\n",
                     worker_id_, (void*)awaited, raw_flags, ctx->method_id_);
            }
          }
          suspended_->Push(ctx);
          continue;
        }
        GPU_WORKER_DPRINTF("[W%u] Suspended: sub-task complete, resuming\n",
                           worker_id_);
#if !HSHM_IS_HOST
        DeserializeAwaitedOutput(*ctx, awaited);
#endif
        ctx->awaited_fshm_ = nullptr;
        ctx->awaited_task_ = nullptr;
      }

      // Check if the top-level coroutine completed (from previous resume)
      if (ctx->task_coros_[0] && ctx->task_coros_[0].done()) {
        DbgTaskCompleted();
        --num_suspended_;
        if (ctx->is_copy_path_) {
          auto *container = static_cast<Container *>(ctx->container_);
          auto *fshm = static_cast<FutureShm *>(ctx->task_fshm_);
          SerializeAndComplete(fshm, container,
                               ctx->method_id_, ctx->task_ptr_,
                               ctx->is_gpu2gpu_);
        } else {
          auto *fshm = static_cast<FutureShm *>(ctx->task_fshm_);
          CompleteAndResumeParent(fshm, ctx->is_gpu2gpu_);
        }
        FreeContext(ctx);
        did_work = true;
        continue;
      }

      // Ready to resume: set as active
      DbgTaskResumed();
      active_ctx_ = ctx;
      --num_suspended_;
      did_work = true;
      break;  // Resume one at a time
    }
    return did_work;
  }

  // ================================================================
  // New task processing
  // ================================================================

  /**
   * Try to pop a new task from gpu2gpu or cpu2gpu queue.
   * On success, prepares a RunContext and sets active_ctx_.
   * Called only by lane 0.
   */
  HSHM_GPU_FUN bool TryPopNewTask() {
    // Poll internal queue first (highest priority) to prevent deadlock:
    // orchestrator subtasks must drain before client tasks can unblock.
    if (TryPopFromQueue(internal_queue_, lane_id_, true)) return true;
    // Poll cross-warp sub-task queue (higher priority than client queues)
    if (TryPopCrossWarpTask()) return true;
    if (TryPopFromQueue(gpu2gpu_queue_, lane_id_, true)) return true;
    if (lane_id_ == 0) {
      if (TryPopFromQueue(cpu2gpu_queue_, 0, false)) return true;
    }
    return false;
  }

  HSHM_GPU_FUN bool TryPopFromQueue(TaskQueue *queue, u32 qlane,
                                      bool is_gpu2gpu) {
    if (!queue) return false;

    auto &lane = queue->GetLane(qlane, 0);

    // Debug: snapshot ring buffer head/tail via device-scope reads (bypass L1)
    if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      u64 h = lane.GetHeadDevice();
      u64 t = lane.GetTailDevice();
      if (queue == internal_queue_) {
        dbg_ctrl_->dbg_iq_head[worker_id_] = h;
        dbg_ctrl_->dbg_iq_tail[worker_id_] = t;
      } else if (is_gpu2gpu) {
        dbg_ctrl_->dbg_input_tw[worker_id_] = t;
        dbg_ctrl_->dbg_input_cs[worker_id_] = h;
        // Store queue pointer for verification (first time only)
        if (dbg_ctrl_->dbg_ser_total_written[worker_id_] == 0) {
          dbg_ctrl_->dbg_ser_total_written[worker_id_] =
              reinterpret_cast<unsigned long long>(queue);
        }
      }
    }

    Future<Task> future;
    if (is_gpu2gpu) {
      if (!lane.PopDevice(future)) return false;
    } else {
      if (!lane.Pop(future)) return false;
    }

    // Count every successful pop (before sptr check)
    DbgQueuePop();
    if (queue == internal_queue_ && dbg_ctrl_ &&
        worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_iq_pops[worker_id_] = dbg_ctrl_->dbg_iq_pops[worker_id_] + 1;
    }

    // Resolve FutureShm
    hipc::ShmPtr<FutureShm> sptr = future.GetFutureShmPtr();
    if (sptr.IsNull()) return true;

    size_t off = sptr.off_.load();
    FutureShm *fshm;
    if (!is_gpu2gpu) {
      fshm = reinterpret_cast<FutureShm *>(queue_backend_base_ + off);
    } else {
      fshm = reinterpret_cast<FutureShm *>(off);
    }
    if (!fshm) return true;

    // Fence: the client's SendGpu fenced before Push, and our PopDevice
    // used device-scope atomics. This fence ensures we see all of the
    // client's writes to the FutureShm fields (pool_id, method_id, flags,
    // copy_space data) that were flushed to L2 by the client's fence.
    hipc::threadfence();

    // Look up target container
    PoolId pool_id = fshm->pool_id_;
    u32 method_id = fshm->method_id_;
    Container *container = pool_mgr_->GetContainer(pool_id);
    if (!container) {
      DbgNoContainer(pool_id.major_, pool_id.minor_);
      CompleteAndResumeParent(fshm, is_gpu2gpu);
      return true;
    }

    DbgTaskPopped();
    bool is_copy = fshm->flags_.AnyDevice(FutureShm::FUTURE_COPY_FROM_CLIENT);
    if (is_copy) {
      return PrepareTaskCopy(fshm, container, method_id, is_gpu2gpu);
    } else {
      return PrepareTaskDirect(fshm, container, method_id, is_gpu2gpu);
    }
  }

  /**
   * Prepare a copy-path task: deserialize input, allocate contexts.
   * Sets active_ctx_ on success.
   */
  /** Compact descriptor for a cross-warp sub-task pushed to warp_group_queue */
  struct CrossWarpDescriptor {
    FutureShm *fshm;
    Container *container;
    hipc::FullPtr<Task> task_ptr;
    u32 method_id;
    u32 range_off;
    u32 range_width;
    u32 parallelism;  /**< Per-warp parallelism (capped at 32) */
    bool is_gpu2gpu;
    bool is_copy_path;
  };

  /**
   * Push a cross-warp sub-task descriptor to the warp group queue.
   * Called by lane 0 of the originating warp for each additional warp needed.
   */
  HSHM_GPU_FUN void PushCrossWarpSubTask(
      FutureShm *fshm, Container *container, u32 method_id,
      hipc::FullPtr<Task> task_ptr, bool is_gpu2gpu, bool is_copy_path,
      u32 range_off, u32 range_width) {
    if (!warp_group_queue_) return;
    auto *ipc = CHI_IPC;
    auto *alloc = ipc->gpu_alloc_;
    auto desc_fp = alloc->template AllocateObjs<CrossWarpDescriptor>(1);
    if (desc_fp.IsNull()) return;
    auto *desc = desc_fp.ptr_;
    desc->fshm = fshm;
    desc->container = container;
    desc->task_ptr = task_ptr;
    desc->method_id = method_id;
    desc->range_off = range_off;
    desc->range_width = range_width;
    desc->parallelism = range_width;
    desc->is_gpu2gpu = is_gpu2gpu;
    desc->is_copy_path = is_copy_path;

    // Pack descriptor address into a Future<Task> for the queue
    Future<Task> future;
    hipc::ShmPtr<FutureShm> sptr;
    sptr.off_ = reinterpret_cast<size_t>(desc);
    sptr.alloc_id_ = hipc::AllocatorId::GetNull();
    future = Future<Task>(sptr);

    // Distribute sub-tasks across all lanes based on range_off
    u32 num_lanes = warp_group_queue_->GetNumLanes();
    u32 lane = (range_off / 32) % num_lanes;
    auto &qlane = warp_group_queue_->GetLane(lane, 0);
    bool pushed = qlane.Push(future);
    if (!pushed) {
      // Push failed - free descriptor
      ipc->gpu_alloc_->Free(desc_fp);
      GPU_WORKER_DPRINTF("[W%u] CrossWarp PUSH FAILED: lane=%u off=%u\n",
                         worker_id_, lane, range_off);
    }
  }

  /**
   * Try to pop a cross-warp sub-task from the warp group queue.
   * Returns true if a task was popped and active_ctx_ptr_ was set.
   */
  HSHM_GPU_FUN bool TryPopCrossWarpTask() {
    if (!warp_group_queue_) return false;

    u32 lane = worker_id_ % warp_group_queue_->GetNumLanes();
    auto &qlane = warp_group_queue_->GetLane(lane, 0);

    Future<Task> future;
    if (!qlane.PopDevice(future)) return false;

    hipc::ShmPtr<FutureShm> sptr = future.GetFutureShmPtr();
    if (sptr.IsNull()) return true;

    auto *desc = reinterpret_cast<CrossWarpDescriptor *>(sptr.off_.load());
    if (!desc) return true;

    RunContext *rctx = AllocContext(
        desc->container, desc->method_id, desc->task_ptr, desc->fshm,
        desc->is_gpu2gpu, desc->is_copy_path, desc->parallelism,
        desc->range_off, desc->range_width);
    if (!rctx) {
      // Cannot allocate context — push back? For now just drop and let
      // completion counter handle the partial completion.
      auto *ipc = CHI_IPC;
      hipc::FullPtr<CrossWarpDescriptor> fp(
          reinterpret_cast<hipc::Allocator *>(ipc->gpu_alloc_), desc);
      ipc->gpu_alloc_->Free(fp);
      return true;
    }

    // Free the descriptor
    auto *ipc = CHI_IPC;
    hipc::FullPtr<CrossWarpDescriptor> fp(
        reinterpret_cast<hipc::Allocator *>(ipc->gpu_alloc_), desc);
    ipc->gpu_alloc_->Free(fp);

    active_ctx_ = rctx;
    return true;
  }

  HSHM_GPU_FUN void DbgStep(unsigned int step) {
    if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_dispatch_step[worker_id_] = step;
    }
  }

  HSHM_GPU_FUN bool PrepareTaskCopy(FutureShm *fshm, Container *container,
                                      u32 method_id, bool is_gpu2gpu) {
#if !HSHM_IS_HOST
    auto *ipc = CHI_IPC;
    auto *alloc = ipc->gpu_alloc_;

    GPU_WORKER_DPRINTF("[W%u] PrepCopy: start method %u\n", worker_id_, method_id);

    // Set allocator on container
    container->gpu_alloc_ = reinterpret_cast<HSHM_DEFAULT_ALLOC_GPU_T *>(
        static_cast<void *>(alloc));

    // Deserialize input
    hshm::lbm::LbmContext in_ctx;
    in_ctx.copy_space = fshm->copy_space;
    in_ctx.shm_info_ = &fshm->input_;

    GPU_WORKER_DPRINTF("[W%u] PrepCopy: deserializing\n", worker_id_);
    auto *priv_alloc = CHI_PRIV_ALLOC;
    if (!priv_alloc) {
      if (is_gpu2gpu) {
        fshm->flags_.SetBits(FutureShm::FUTURE_COMPLETE);
      } else {
        fshm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);
      }
      return true;
    }
    // Validate copy_space data: check total_written_ is sane
    size_t tw = fshm->input_.total_written_.load_device();
    size_t cs = fshm->input_.copy_space_size_.load_device();
    if (tw == 0 || tw > cs) {
      if (is_gpu2gpu) {
        fshm->flags_.SetBits(FutureShm::FUTURE_COMPLETE);
      } else {
        fshm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);
      }
      return true;
    }
    LocalLoadTaskArchive load_ar(CHI_PRIV_ALLOC);
    if (is_gpu2gpu) {
      hshm::lbm::ShmTransport::RecvDevice(load_ar, in_ctx);
    } else {
      hshm::lbm::ShmTransport::Recv(load_ar, in_ctx);
    }
    load_ar.SetMsgType(LocalMsgType::kSerializeIn);

    hipc::FullPtr<Task> task_ptr =
        container->LocalAllocLoadTask(method_id, load_ar);
    if (task_ptr.IsNull()) {
      GPU_WORKER_DPRINTF("[W%u] PrepCopy: task alloc FAILED\n", worker_id_);
      if (is_gpu2gpu) {
        fshm->flags_.SetBits(FutureShm::FUTURE_COMPLETE);
      } else {
        fshm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);
      }
      return true;
    }

    // Allocate RunContext
    GPU_WORKER_DPRINTF("[W%u] PrepCopy: allocating context\n", worker_id_);
    u32 parallelism = task_ptr.ptr_->pool_query_.GetParallelism();

    if (parallelism <= 32) {
      // Single-warp path
      RunContext *rctx = AllocContext(
          container, method_id, task_ptr, fshm, is_gpu2gpu, true,
          parallelism, 0, parallelism);
      if (!rctx) {
        DbgAllocFailure();
        GPU_WORKER_DPRINTF("[W%u] PrepCopy: context alloc FAILED\n", worker_id_);
        if (is_gpu2gpu) {
          fshm->flags_.SetBits(FutureShm::FUTURE_COMPLETE);
        } else {
          fshm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);
        }
        return true;
      }
      GPU_WORKER_DPRINTF("[W%u] PrepCopy: done\n", worker_id_);
      active_ctx_ = rctx;
    } else {
      // Cross-warp path
      u32 num_warps_needed = (parallelism + 31) / 32;
      fshm->total_warps_ = num_warps_needed;
      fshm->completion_counter_.store(0);

      // This warp takes range [0, 32)
      RunContext *rctx = AllocContext(
          container, method_id, task_ptr, fshm, is_gpu2gpu, true,
          32, 0, 32);
      if (!rctx) {
        DbgAllocFailure();
        if (is_gpu2gpu) {
          fshm->flags_.SetBits(FutureShm::FUTURE_COMPLETE);
        } else {
          fshm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);
        }
        return true;
      }

      // Push remaining sub-ranges to warp_group_queue
      for (u32 w = 1; w < num_warps_needed; ++w) {
        u32 off = w * 32;
        u32 width = 32;
        if (off + width > parallelism) width = parallelism - off;
        PushCrossWarpSubTask(fshm, container, method_id, task_ptr,
                             is_gpu2gpu, true, off, width);
      }
      active_ctx_ = rctx;
    }
    return true;
#else
    return false;
#endif
  }

  /**
   * Prepare a direct-path task: no deserialization needed.
   * Sets active_ctx_ on success.
   */
  HSHM_GPU_FUN bool PrepareTaskDirect(FutureShm *fshm, Container *container,
                                        u32 method_id, bool is_gpu2gpu) {
    auto *ipc = CHI_IPC;
    auto *alloc = ipc->gpu_alloc_;

    container->gpu_alloc_ = reinterpret_cast<HSHM_DEFAULT_ALLOC_GPU_T *>(
        static_cast<void *>(alloc));

    hipc::FullPtr<Task> task_ptr;
    task_ptr.ptr_ = reinterpret_cast<Task *>(fshm->client_task_vaddr_);
    task_ptr.shm_.off_ = fshm->client_task_vaddr_;
    task_ptr.shm_.alloc_id_ = hipc::AllocatorId::GetNull();

    u32 parallelism = task_ptr.ptr_->pool_query_.GetParallelism();

    if (parallelism <= 32) {
      // Single-warp path
      RunContext *rctx = AllocContext(
          container, method_id, task_ptr, fshm, is_gpu2gpu, false,
          parallelism, 0, parallelism);
      if (!rctx) {
        DbgAllocFailure();
        if (is_gpu2gpu) {
          fshm->flags_.SetBits(FutureShm::FUTURE_COMPLETE);
        } else {
          fshm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);
        }
        return true;
      }
      active_ctx_ = rctx;
    } else {
      // Cross-warp path
      u32 num_warps_needed = (parallelism + 31) / 32;
      fshm->total_warps_ = num_warps_needed;
      fshm->completion_counter_.store(0);

      RunContext *rctx = AllocContext(
          container, method_id, task_ptr, fshm, is_gpu2gpu, false,
          32, 0, 32);
      if (!rctx) {
        DbgAllocFailure();
        if (is_gpu2gpu) {
          fshm->flags_.SetBits(FutureShm::FUTURE_COMPLETE);
        } else {
          fshm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);
        }
        return true;
      }

      for (u32 w = 1; w < num_warps_needed; ++w) {
        u32 off = w * 32;
        u32 width = 32;
        if (off + width > parallelism) width = parallelism - off;
        PushCrossWarpSubTask(fshm, container, method_id, task_ptr,
                             is_gpu2gpu, false, off, width);
      }
      active_ctx_ = rctx;
    }
    return true;
  }

  // ================================================================
  // Output serialization and completion
  // ================================================================

  HSHM_GPU_FUN void SerializeAndComplete(FutureShm *fshm, Container *container,
                                           u32 method_id,
                                           hipc::FullPtr<Task> &task_ptr,
                                           bool is_gpu2gpu) {
#if !HSHM_IS_HOST
    hshm::lbm::LbmContext out_ctx;
    out_ctx.copy_space = fshm->copy_space;
    out_ctx.shm_info_ = &fshm->output_;

    auto *priv_alloc = CHI_PRIV_ALLOC;
    LocalSaveTaskArchive save_ar(LocalMsgType::kSerializeOut, priv_alloc);
    container->LocalSaveTask(method_id, save_ar, task_ptr);
    if (is_gpu2gpu) {
      hshm::lbm::ShmTransport::SendDevice(save_ar, out_ctx);
    } else {
      hshm::lbm::ShmTransport::Send(save_ar, out_ctx);
    }

    container->LocalDestroyTask(method_id, task_ptr);
    CHI_IPC->gpu_alloc_->Free(task_ptr.template Cast<char>());
    hipc::threadfence();

    if (is_gpu2gpu) {
      fshm->flags_.SetBits(FutureShm::FUTURE_COMPLETE);
    } else {
      fshm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);
    }

    ResumeParentIfPresent(fshm);
#endif
  }

  HSHM_GPU_FUN void CompleteAndResumeParent(FutureShm *fshm,
                                              bool is_gpu2gpu) {
    // Fence before setting FUTURE_COMPLETE so that all prior writes
    // (task output, FutureShm fields) are visible to the waiter.
    if (is_gpu2gpu) {
      hipc::threadfence();
      fshm->flags_.SetBits(FutureShm::FUTURE_COMPLETE);
    } else {
      fshm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);
    }
    ResumeParentIfPresent(fshm);
  }

  /**
   * Signal that a sub-task is complete.
   *
   * Do NOT resume the parent coroutine here — the sub-task may have been
   * processed by a different warp than the one that owns the parent.
   * Cross-warp coroutine resumption is unsafe.  Instead, just mark the
   * FutureShm complete and let the parent warp's CheckAndResumeSuspended
   * pick it up on its next poll iteration.
   */
  HSHM_GPU_FUN void ResumeParentIfPresent(FutureShm *fshm) {
    (void)fshm;
    // The FutureShm FUTURE_COMPLETE flag was already set by the caller
    // (SerializeAndComplete / CompleteAndResumeParent).  The parent warp
    // will detect this in CheckAndResumeSuspended → DeserializeAwaitedOutput.
  }

#if !HSHM_IS_HOST
  HSHM_GPU_FUN void DeserializeAwaitedOutput(RunContext &rctx,
                                              FutureShm *awaited) {
    if (!rctx.awaited_task_ || awaited->output_.total_written_.load_device() == 0) {
      return;
    }
    hipc::threadfence();
    auto *sub_task = reinterpret_cast<Task *>(rctx.awaited_task_);
    Container *sub_container = pool_mgr_->GetContainer(sub_task->pool_id_);
    if (!sub_container) return;

    hshm::lbm::LbmContext ctx;
    ctx.copy_space = awaited->copy_space;
    ctx.shm_info_ = &awaited->output_;
    LocalLoadTaskArchive load_ar(CHI_PRIV_ALLOC);
    hshm::lbm::ShmTransport::RecvDevice(load_ar, ctx);
    hipc::FullPtr<Task> sub_task_ptr;
    sub_task_ptr.ptr_ = sub_task;
    sub_container->LocalLoadTaskOutput(
        sub_task->method_, load_ar, sub_task_ptr);
  }
#endif
};

}  // namespace gpu
}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_GPU_WORKER_H_

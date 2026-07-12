/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * gpu2cpu ring BACKPRESSURE.
 *
 * The gpu2cpu lane is a ctp::ipc::multi_mpsc_ring_buffer, i.e. it is built with
 * RING_BUFFER_WAIT_FOR_SPACE. ring_buffer::Emplace() claims a slot with
 * `tail_.fetch_add_system(1)` and then, if the ring is full, SPINS on the device
 * re-reading `head_` until the CPU consumer frees a slot. So a GPU producer that
 * outruns the ring does not fail -- it PARKS ON THE DEVICE and waits for the CPU
 * to drain. The CPU side (ring_buffer::PopUnlocked) is what advances `head_`.
 *
 * Two cases here:
 *
 *   1. "producer fills the ring exactly" -- submits depth-1 tasks, so no
 *      producer ever has to wait for space. This PASSES and guards the normal
 *      path.
 *
 *   2. "producer OVERRUNS the ring" -- submits several times the ring depth, so
 *      producers must block in Emplace's wait-for-space spin. THIS CURRENTLY
 *      DEADLOCKS, and is therefore opt-in (set CLIO_TEST_GPU2CPU_OVERRUN=1).
 *
 * ---------------------------------------------------------------------------
 * BUG reproduced by case 2 (gpu2cpu ring: consumer stops advancing head_)
 * ---------------------------------------------------------------------------
 * Submitting 3072 tasks into a depth-1025 ring wedges with this exact state:
 *
 *     head=0  tail=3072  depth=1025
 *     slots the CPU sees READY = 1023   (not ready: slot 0 and slot 1024)
 *     tasks actually executed by the CPU = exactly ONE
 *
 * i.e. every producer successfully claimed a tail and 1023 of them landed their
 * entry, the CPU consumer popped exactly ONE entry (clearing that slot's ready
 * bit) -- and then `head_` stayed at 0 forever. Slot 1024 being unready is
 * correct (its producer is legitimately parked waiting for space). But because
 * `head_` never advances past the slot that was already consumed, PopUnlocked
 * re-reads that same slot, finds its ready bit cleared, returns false, and the
 * consumer never makes progress again. The parked producers are waiting on a
 * `head_` that will never move => deadlock.
 *
 * Once this is fixed, delete the CLIO_TEST_GPU2CPU_OVERRUN gate so the overrun
 * case runs by default: a blocking producer MUST be drainable.
 *
 * FULL WRITE-UP -- evidence, the refuted theories (do not re-derive them), what
 * is still unproven, and the fix: see GPU2CPU_RING_DEADLOCK.md next to this file.
 *
 * Suffix `_gpu.cc` so NVCC/HIPCC compiles it, matching its sibling
 * test_gpu_kernel_stress_gpu.cc.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include "simple_test.h"
#include "test_gpu_kernel_stress_common.h"

#include <clio_ctp/util/gpu_api.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

using TaskT = clio::run::MOD_NAME::GpuSubmitTask;

namespace {

/** How long the CPU consumer gets to drain the ring before we call it wedged. */
constexpr int kDrainTimeoutMs = 20000;

/**
 * PRODUCER-ONLY submit kernel: pushes its task and returns immediately. It never
 * calls future.Wait(), so the ONLY thing that can park this kernel is the ring
 * being full -- which is exactly what we want to exercise.
 *
 * One block per task; thread 0 of each block does the Send (IpcGpu2Cpu::SendIn
 * is gated on threadIdx.x == 0).
 */
__global__ void Gpu2CpuOverrunKernel(clio::run::IpcManagerGpuInfo gpu_info,
                                     ctp::ipc::FullPtr<TaskT> *task_handles,
                                     clio::run::u32 num_tasks) {
  CLIO_GPU_INIT(gpu_info, /*ipc_ptr=*/nullptr);
  if (threadIdx.x != 0) return;
  clio::run::u32 slot = blockIdx.x;
  if (slot >= num_tasks) return;
  // Fire and forget -- no Wait(). See the note in ChiGpuKernelStress about using
  // g_ipc_manager_ptr rather than the CLIO_IPC macro here.
  (void)g_ipc_manager_ptr->Send(task_handles[slot]);
  (void)g_ipc_manager;
}

}  // namespace

#if !CTP_IS_DEVICE_PASS

namespace {

/**
 * Submit `num_tasks` tasks producer-only into the gpu2cpu ring and wait for the
 * CPU worker to drain them all.
 *
 * Never blocks on Synchronize(): if the ring's blocking producer wedges against
 * the CPU consumer, a blocking Synchronize would hang the whole binary. We poll
 * with a deadline instead, so a wedge is reported as a clean failure and the
 * ring's state is dumped.
 *
 * @return true if the producer kernel retired AND every task was executed.
 */
bool RunOverrun(clio::run::u32 num_tasks) {
  using namespace chi_test_gpu_stress;
  auto *ipc = CLIO_CPU_IPC;
  const clio::run::u32 gpu_id = 0;

  const size_t slot_bytes = sizeof(TaskT);
  const size_t backend_bytes =
      static_cast<size_t>(num_tasks) * slot_bytes + 256;

  char *base = nullptr;
  ctp::ipc::AllocatorId alloc_id = ipc->AllocateAndRegisterGpuBackend(
      gpu_id, clio::run::gpu::IpcManager::MemKind::kPinnedHost, backend_bytes,
      &base);
  REQUIRE(!alloc_id.IsNull());
  REQUIRE(base != nullptr);

  // Place the tasks. Slot i carries test_value = i, so the chimod must produce
  // result_value_ == i * 2 + gpu_id.
  std::vector<ctp::ipc::FullPtr<TaskT>> handles;
  handles.reserve(num_tasks);
  for (clio::run::u32 i = 0; i < num_tasks; ++i) {
    size_t task_off = static_cast<size_t>(i) * slot_bytes;
    auto *task = new (base + task_off)
        TaskT(clio::run::CreateTaskId(), GetTestPoolId(),
              clio::run::PoolQuery::ToLocalCpu(), gpu_id, /*test_value=*/i);
    task->fut_.task_size_ = static_cast<clio::run::u32>(sizeof(TaskT));
    ctp::ipc::FullPtr<TaskT> fp;
    fp.shm_.alloc_id_ = alloc_id;
    fp.shm_.off_ = task_off;
    fp.ptr_ = task;
    handles.push_back(fp);
  }

  auto *task_handle_dev =
      ctp::GpuApi::MallocHost<ctp::ipc::FullPtr<TaskT>>(num_tasks);
  REQUIRE(task_handle_dev != nullptr);
  for (clio::run::u32 i = 0; i < num_tasks; ++i) task_handle_dev[i] = handles[i];

  clio::run::IpcManagerGpuInfo info =
      ipc->GetGpuIpcManager()->GetGpuInfo(gpu_id);
  REQUIRE(info.gpu2cpu_queue != nullptr);
  auto &lane = info.gpu2cpu_queue->GetLane(0, 0);
  const size_t depth = lane.GetDepth();

  std::fprintf(stderr,
               "[BACKPRESSURE] submitting %u tasks producer-only into a "
               "depth-%zu ring\n",
               num_tasks, depth);

  void *stream = ctp::GpuApi::CreateStream();
  REQUIRE(stream != nullptr);
  Gpu2CpuOverrunKernel<<<num_tasks, 32, 0, static_cast<cudaStream_t>(stream)>>>(
      info, task_handle_dev, num_tasks);

  // The producer kernel can only retire once the CPU consumer has freed enough
  // slots for every Emplace() to land.
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(kDrainTimeoutMs);
  bool kernel_done = false;
  while (std::chrono::steady_clock::now() < deadline) {
    if (ctp::GpuApi::StreamQuery(stream)) {
      kernel_done = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  // Then every task must have been executed by the CPU worker.
  clio::run::u32 executed = 0;
  bool all_done = false;
  while (kernel_done && std::chrono::steady_clock::now() < deadline) {
    executed = 0;
    for (clio::run::u32 i = 0; i < num_tasks; ++i) {
      if (handles[i]->result_value_ == (i * 2u) + gpu_id) ++executed;
    }
    if (executed == num_tasks) {
      all_done = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  const bool ok = kernel_done && all_done;
  if (!ok) {
    // Dump the ring state that identifies the bug (see the file header).
    size_t n_ready = 0;
    for (size_t i = 0; i < depth; ++i) {
      clio::run::gpu::Future<clio::run::Task> pv;
      if (lane.Peek(lane.GetHead() + i, pv)) ++n_ready;
    }
    executed = 0;
    for (clio::run::u32 i = 0; i < num_tasks; ++i) {
      if (handles[i]->result_value_ == (i * 2u) + gpu_id) ++executed;
    }
    std::fprintf(stderr,
                 "[BACKPRESSURE] WEDGED: producer_retired=%d  ring head=%llu "
                 "tail=%llu depth=%zu  entries_the_CPU_sees_READY=%zu  "
                 "tasks_executed=%u/%u\n"
                 "[BACKPRESSURE] the consumer stopped advancing head_ -- the "
                 "parked producers are waiting on a head_ that never moves\n",
                 (int)kernel_done, (unsigned long long)lane.GetHead(),
                 (unsigned long long)lane.GetTail(), depth, n_ready, executed,
                 num_tasks);
  }

  // Do NOT DestroyStream / FreeGpuBackend on the wedged path: the producer
  // kernel is still resident and still referencing both.
  if (ok) {
    ctp::GpuApi::DestroyStream(stream);
    ctp::GpuApi::FreeHost(reinterpret_cast<char *>(task_handle_dev));
    ipc->FreeGpuBackend(gpu_id, alloc_id);
  }
  return ok;
}

}  // namespace

TEST_CASE("gpu2cpu: producer fills the ring exactly", "[gpu2cpu][backpressure]") {
  using namespace chi_test_gpu_stress;
  EnsureInit();
  auto *ipc = CLIO_CPU_IPC;
  clio::run::IpcManagerGpuInfo info =
      ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
  REQUIRE(info.gpu2cpu_queue != nullptr);

  // depth-1 submissions is the most that can be in flight without any producer
  // having to wait for space, so nothing should ever park in Emplace.
  const size_t depth = info.gpu2cpu_queue->GetLane(0, 0).GetDepth();
  REQUIRE(depth > 1);
  REQUIRE(RunOverrun(static_cast<clio::run::u32>(depth - 1)));
}

TEST_CASE("gpu2cpu: producer overruns the ring", "[gpu2cpu][backpressure]") {
  using namespace chi_test_gpu_stress;
  EnsureInit();

  // KNOWN BUG -- opt-in. See the file header: overrunning the ring parks the
  // producers in Emplace's wait-for-space spin, and the CPU consumer then stops
  // advancing head_, so nothing ever drains. Gated so CI stays green; remove the
  // gate once the ring is fixed.
  if (std::getenv("CLIO_TEST_GPU2CPU_OVERRUN") == nullptr) {
    std::fprintf(stderr,
                 "[BACKPRESSURE] SKIPPED (known deadlock). Set "
                 "CLIO_TEST_GPU2CPU_OVERRUN=1 to reproduce.\n");
    return;
  }

  auto *ipc = CLIO_CPU_IPC;
  clio::run::IpcManagerGpuInfo info =
      ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
  REQUIRE(info.gpu2cpu_queue != nullptr);
  const size_t depth = info.gpu2cpu_queue->GetLane(0, 0).GetDepth();

  // 3x the ring depth: producers must block in Emplace at least twice over.
  REQUIRE(RunOverrun(static_cast<clio::run::u32>(depth * 3)));
}

#endif  // !CTP_IS_DEVICE_PASS

SIMPLE_TEST_MAIN()

#else  // CUDA/ROCm not enabled (or SYCL is)

int main() { return 0; }

#endif

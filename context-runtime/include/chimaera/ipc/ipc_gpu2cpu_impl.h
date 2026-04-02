/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CHIMAERA_INCLUDE_CHIMAERA_IPC_GPU2CPU_IMPL_H_
#define CHIMAERA_INCLUDE_CHIMAERA_IPC_GPU2CPU_IMPL_H_

#include "chimaera/ipc/ipc_gpu2cpu.h"

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

namespace chi {

#if HSHM_IS_GPU_COMPILER
/**
 * GPU-side ClientSend: enqueue task to gpu2cpu_queue (pinned host).
 * The CPU GPU worker polls this queue and dispatches on the CPU side.
 */
template <typename TaskT>
HSHM_GPU_FUN gpu::Future<TaskT> IpcGpu2Cpu::ClientSend(
    gpu::IpcManager *ipc, const hipc::FullPtr<TaskT> &task_ptr) {
  u32 lane = gpu::IpcManager::GetLaneId();
  gpu::Future<TaskT> future;

  if (lane == 0) {
    if (!task_ptr.IsNull() && ipc->gpu_info_.gpu2cpu_queue) {
      gpu::FutureShm *fshm = reinterpret_cast<gpu::FutureShm *>(
          reinterpret_cast<char *>(task_ptr.ptr_) + sizeof(TaskT));
      fshm->Reset(task_ptr->pool_id_, task_ptr->method_);
      fshm->client_task_vaddr_ =
          reinterpret_cast<size_t>(static_cast<Task *>(task_ptr.ptr_));
      fshm->flags_.SetBits(gpu::FutureShm::FUTURE_DEVICE_SCOPE);

      hipc::ShmPtr<gpu::FutureShm> fshmptr;
      fshmptr.alloc_id_ = hipc::AllocatorId::GetNull();
      fshmptr.off_ = reinterpret_cast<size_t>(fshm);
      future = gpu::Future<TaskT>(fshmptr, task_ptr);

      auto &qlane = ipc->gpu_info_.gpu2cpu_queue->GetLane(0, 0);
      gpu::Future<Task> task_future(future.GetFutureShmPtr());
      hipc::threadfence_system();
      qlane.Push(task_future);
    }
  }
  __syncwarp();
  return future;
}
#endif  // HSHM_IS_GPU_COMPILER

#if HSHM_IS_HOST
template <typename TaskT, typename AllocT>
bool IpcGpu2Cpu::ClientRecv(Future<TaskT, AllocT> &future, float max_sec) {
  hipc::FullPtr<FutureShm> future_full =
      CHI_CPU_IPC->ToFullPtr(future.future_shm_);
  if (future_full.IsNull()) {
    HLOG(kError, "IpcGpu2Cpu::ClientRecv: ToFullPtr returned null");
    return false;
  }

  // Poll FUTURE_COMPLETE (set by GPU with system-scope atomics)
  hshm::abitfield32_t &flags = future_full->flags_;
  auto start = std::chrono::steady_clock::now();
  while (!flags.AnySystem(FutureShm::FUTURE_COMPLETE)) {
    HSHM_THREAD_MODEL->Yield();
    if (max_sec > 0) {
      float elapsed = std::chrono::duration<float>(
                          std::chrono::steady_clock::now() - start)
                          .count();
      if (elapsed >= max_sec) {
        future.task_ptr_->SetReturnCode(static_cast<u32>(-3));
        return false;
      }
    }
  }

  // Deserialize output from GPU FutureShm ring buffer if present
  if (future_full->output_.total_written_.load() > 0) {
    hshm::lbm::LbmContext ctx;
    ctx.copy_space = future_full->copy_space;
    ctx.shm_info_ = &future_full->output_;
    chi::priv::vector<char> load_buf;
    load_buf.reserve(256);
    DefaultLoadArchive load_ar(load_buf);
    load_ar.SetMsgType(LocalMsgType::kSerializeOut);
    hshm::lbm::ShmTransport::Recv(load_ar, ctx);
    future.task_ptr_->SerializeOut(load_ar);
  }
  return true;
}
#endif  // HSHM_IS_HOST

}  // namespace chi

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
#endif  // CHIMAERA_INCLUDE_CHIMAERA_IPC_GPU2CPU_IMPL_H_

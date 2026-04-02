/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CHIMAERA_INCLUDE_CHIMAERA_IPC_CPU2GPU_IMPL_H_
#define CHIMAERA_INCLUDE_CHIMAERA_IPC_CPU2GPU_IMPL_H_

#include "chimaera/ipc/ipc_cpu2gpu.h"

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include "hermes_shm/util/gpu_api.h"

namespace chi {

#if HSHM_IS_HOST
template <typename TaskT>
chi::Future<TaskT> IpcCpu2Gpu::ClientSend(
    gpu::IpcManager *ipc, const hipc::FullPtr<TaskT> &task_ptr, u32 gpu_id) {
  if (task_ptr.IsNull() || gpu_id >= ipc->gpu_devices_.size()) {
    return chi::Future<TaskT>();
  }
  size_t task_size = sizeof(TaskT);
  size_t total_size = task_size + sizeof(gpu::FutureShm);
  auto *device_buf = hshm::GpuApi::MallocAndCopy(
      reinterpret_cast<const char *>(task_ptr.ptr_), task_size, total_size);
  if (!device_buf) return chi::Future<TaskT>();
  auto *host_fshm =
      hshm::GpuApi::MallocHost<gpu::FutureShm>(sizeof(gpu::FutureShm));
  if (!host_fshm) {
    hshm::GpuApi::Free(device_buf);
    return chi::Future<TaskT>();
  }
  memset(host_fshm, 0, sizeof(gpu::FutureShm));
  host_fshm->pool_id_ = task_ptr->pool_id_;
  host_fshm->method_id_ = task_ptr->method_;
  host_fshm->origin_ = gpu::FutureShm::FUTURE_CLIENT_CPU2GPU;
  host_fshm->client_task_vaddr_ = reinterpret_cast<uintptr_t>(device_buf);
  host_fshm->task_device_ptr_ = reinterpret_cast<uintptr_t>(device_buf);
  host_fshm->task_size_ = static_cast<u32>(task_size);
  host_fshm->flags_.SetBits(gpu::FutureShm::FUTURE_POD_COPY);
  hshm::GpuApi::Memcpy(device_buf + task_size,
      reinterpret_cast<const char *>(host_fshm), sizeof(gpu::FutureShm));
  hipc::ShmPtr<gpu::FutureShm> gpu_fshmptr;
  gpu_fshmptr.alloc_id_ = gpu::FutureShm::GetCpu2GpuAllocId();
  gpu_fshmptr.off_ = reinterpret_cast<size_t>(host_fshm);
  auto &lane = ipc->gpu_devices_[gpu_id].cpu2gpu_queue.ptr_->GetLane(0, 0);
  gpu::Future<Task> task_future(gpu_fshmptr);
  lane.Push(task_future);
  hipc::ShmPtr<chi::FutureShm> chi_fshmptr;
  chi_fshmptr.alloc_id_ = gpu::FutureShm::GetCpu2GpuAllocId();
  chi_fshmptr.off_ = reinterpret_cast<size_t>(host_fshm);
  return chi::Future<TaskT>(chi_fshmptr, task_ptr);
}

template <typename TaskT, typename AllocT>
bool IpcCpu2Gpu::ClientRecv(Future<TaskT, AllocT> &future, float max_sec) {
  // ShmPtr offset points to pinned-host gpu::FutureShm
  void *host_fshm = reinterpret_cast<void *>(future.future_shm_.off_.load());
  auto start = std::chrono::steady_clock::now();

  // Poll flags on pinned-host gpu::FutureShm
  u32 flags_val = 0;
  size_t flags_offset = offsetof(gpu::FutureShm, flags_);
  while (!(flags_val & gpu::FutureShm::FUTURE_COMPLETE)) {
    hshm::GpuApi::Memcpy(
        &flags_val,
        reinterpret_cast<u32 *>(
            static_cast<char *>(host_fshm) + flags_offset),
        sizeof(u32));
    if (flags_val & gpu::FutureShm::FUTURE_COMPLETE) break;
    HSHM_THREAD_MODEL->Yield();
    if (max_sec > 0) {
      float elapsed = std::chrono::duration<float>(
                          std::chrono::steady_clock::now() - start)
                          .count();
      if (elapsed >= max_sec) return false;
    }
  }

  // Read device task address and copy result D2H
  auto *gpu_fshm = reinterpret_cast<gpu::FutureShm *>(host_fshm);
  void *device_task = reinterpret_cast<void *>(gpu_fshm->task_device_ptr_);
  if (future.task_ptr_.ptr_ && device_task) {
    hshm::GpuApi::Memcpy(
        future.task_ptr_.ptr_,
        static_cast<TaskT *>(device_task),
        sizeof(TaskT));
    future.task_ptr_->FixupAfterCopy();
  }

  // Free device buffer and pinned-host gpu::FutureShm
  if (device_task) hshm::GpuApi::Free(device_task);
  hshm::GpuApi::FreeHost(gpu_fshm);
  return true;
}

#endif  // HSHM_IS_HOST

}  // namespace chi

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
#endif  // CHIMAERA_INCLUDE_CHIMAERA_IPC_CPU2GPU_IMPL_H_

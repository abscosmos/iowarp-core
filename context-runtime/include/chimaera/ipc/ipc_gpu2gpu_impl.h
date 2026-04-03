/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CHIMAERA_INCLUDE_CHIMAERA_IPC_GPU2GPU_IMPL_H_
#define CHIMAERA_INCLUDE_CHIMAERA_IPC_GPU2GPU_IMPL_H_

#include "chimaera/ipc/ipc_gpu2gpu.h"

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

namespace chi {
namespace gpu {

#if HSHM_IS_GPU_COMPILER

template <typename TaskT>
HSHM_GPU_FUN Future<TaskT> IpcGpu2Gpu::ClientSend(
    IpcManager *ipc, const hipc::FullPtr<TaskT> &task_ptr) {
  u32 lane = IpcManager::GetLaneId();
  Future<TaskT> future;
  if (lane == 0) {
    if (!task_ptr.IsNull() && ipc->gpu_info_.gpu2gpu_queue) {
      FutureShm *fshm = reinterpret_cast<FutureShm *>(
          reinterpret_cast<char *>(task_ptr.ptr_) + sizeof(TaskT));
      fshm->Reset(task_ptr->pool_id_, task_ptr->method_);
      fshm->client_task_vaddr_ =
          reinterpret_cast<size_t>(static_cast<Task *>(task_ptr.ptr_));
      fshm->flags_.SetBits(FutureShm::FUTURE_DEVICE_SCOPE);
      hipc::ShmPtr<FutureShm> fshmptr;
      fshmptr.alloc_id_ = hipc::AllocatorId::GetNull();
      fshmptr.off_ = reinterpret_cast<size_t>(fshm);
      future = Future<TaskT>(fshmptr, task_ptr);
      u32 queue_lane_id = 0;
      if (ipc->gpu_info_.gpu2gpu_num_lanes > 1) {
        queue_lane_id =
            IpcManager::GetWarpId() % ipc->gpu_info_.gpu2gpu_num_lanes;
      }
      auto &qlane = ipc->gpu_info_.gpu2gpu_queue->GetLane(queue_lane_id, 0);
      Future<Task> task_future(future.GetFutureShmPtr());
      hipc::threadfence_system();
      qlane.PushSystem(task_future);
    }
  }
  __syncwarp();
  return future;
}

template <typename TaskT>
HSHM_GPU_FUN void IpcGpu2Gpu::ClientRecv(
    IpcManager *ipc, Future<TaskT> &future, TaskT *task_ptr) {
  (void)ipc;
  u32 lane = IpcManager::GetLaneId();
  unsigned long long fshm_ull = 0;
  if (lane == 0) {
    hipc::FullPtr<FutureShm> fshm_full = future.GetFutureShm();
    if (!fshm_full.IsNull())
      fshm_ull = reinterpret_cast<unsigned long long>(fshm_full.ptr_);
  }
  fshm_ull = hipc::shfl_sync_u64(0xFFFFFFFF, fshm_ull, 0);
  if (fshm_ull == 0) return;
  FutureShm *fshm = reinterpret_cast<FutureShm *>(fshm_ull);
  if (lane == 0) {
    int spin_count = 0;
    while (!fshm->flags_.AnyDevice(FutureShm::FUTURE_COMPLETE)) {
      HSHM_THREAD_MODEL->Yield();
      if (++spin_count == 1000000) {
        printf("[RECV-STUCK] blk=%u waiting FUTURE_COMPLETE spin=%d\n",
               blockIdx.x, spin_count);
        spin_count = 0;
      }
    }
    hipc::threadfence();
  }
  __syncwarp();
  if (lane == 0) {
    future.Destroy(true);
  }
}

template <typename TaskT>
HSHM_GPU_FUN Future<TaskT> IpcGpu2Self::ClientSend(
    IpcManager *ipc, const hipc::FullPtr<TaskT> &task_ptr) {
  if (task_ptr.IsNull()) return Future<TaskT>();
  GpuTaskQueue *queue = ipc->gpu_info_.internal_queue
                            ? ipc->gpu_info_.internal_queue
                            : ipc->gpu_info_.gpu2gpu_queue;
  if (!queue) return Future<TaskT>();
  FutureShm *fshm = reinterpret_cast<FutureShm *>(
      reinterpret_cast<char *>(task_ptr.ptr_) + sizeof(TaskT));
  fshm->Reset(task_ptr->pool_id_, task_ptr->method_);
  fshm->origin_ = FutureShm::FUTURE_CLIENT_SHM;
  fshm->client_task_vaddr_ =
      reinterpret_cast<size_t>(static_cast<Task *>(task_ptr.ptr_));
  fshm->flags_.SetBits(FutureShm::FUTURE_DEVICE_SCOPE);
  hipc::ShmPtr<FutureShm> fshmptr;
  fshmptr.alloc_id_ = hipc::AllocatorId::GetNull();
  fshmptr.off_ = reinterpret_cast<size_t>(fshm);
  Future<TaskT> future(fshmptr, task_ptr);
  u32 lane_id = 0;
  if (queue == ipc->gpu_info_.internal_queue) {
    if (ipc->gpu_info_.internal_num_lanes > 1)
      lane_id =
          IpcManager::GetWarpId() % ipc->gpu_info_.internal_num_lanes;
  } else {
    if (ipc->gpu_info_.gpu2gpu_num_lanes > 1)
      lane_id =
          IpcManager::GetWarpId() % ipc->gpu_info_.gpu2gpu_num_lanes;
  }
  auto &qlane = queue->GetLane(lane_id, 0);
  Future<Task> task_future(future.GetFutureShmPtr());
  hipc::threadfence_system();
  qlane.PushSystem(task_future);
  return future;
}

#endif  // HSHM_IS_GPU_COMPILER

}  // namespace gpu
}  // namespace chi

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
#endif  // CHIMAERA_INCLUDE_CHIMAERA_IPC_GPU2GPU_IMPL_H_

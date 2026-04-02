/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CHIMAERA_INCLUDE_CHIMAERA_IPC_CPU2GPU_H_
#define CHIMAERA_INCLUDE_CHIMAERA_IPC_CPU2GPU_H_

#include "chimaera/types.h"
#include "chimaera/task.h"

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

namespace chi {

class IpcManager;

/**
 * IPC transport for CPU client → GPU runtime.
 *
 * CPU allocates device buffer, copies task H2D, pushes to cpu2gpu_queue.
 * GPU orchestrator dequeues and dispatches via CDP.
 * GPU sets FUTURE_COMPLETE on device FutureShm, worker relays to host.
 * CPU polls pinned-host FutureShm, then copies result D2H.
 */
struct IpcCpu2Gpu {
  /** ClientSend: allocate device buffer, copy H2D, push to cpu2gpu_queue. */
  // Implemented as gpu::IpcManager::SendCpuToGpu (template, stays in gpu_ipc_manager.h)

  /** RuntimeSend: set FUTURE_COMPLETE on CPU side. */
  static void RuntimeSend(
      IpcManager *ipc, const FullPtr<Task> &task_ptr,
      RunContext *run_ctx, Container *container);

  /** ClientRecv: poll pinned-host gpu::FutureShm, copy result D2H. */
  // Implemented as Future::WaitCpu2Gpu (template, stays in ipc_manager.h)
};

}  // namespace chi

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
#endif  // CHIMAERA_INCLUDE_CHIMAERA_IPC_CPU2GPU_H_

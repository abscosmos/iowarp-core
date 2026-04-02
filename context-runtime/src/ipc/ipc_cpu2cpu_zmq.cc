/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#include "chimaera/ipc_manager.h"

namespace chi {

hipc::FullPtr<Task> IpcCpu2CpuZmq::RuntimeRecv(
    IpcManager *ipc, Future<Task> &future, Container *container,
    u32 method_id, hshm::lbm::Transport *recv_transport) {
  // ZMQ tasks are deserialized by the net worker receive thread before
  // reaching this point. By the time the worker calls RecvRuntime,
  // the task is already a pointer. Delegate to IpcCpu2Cpu::RuntimeRecv
  // which handles the SHM deserialization path.
  return IpcCpu2Cpu::RuntimeRecv(ipc, future, container,
                                  method_id, recv_transport);
}

void IpcCpu2CpuZmq::RuntimeSend(
    IpcManager *ipc, RunContext *run_ctx, u32 origin) {
  if (origin == FutureShm::FUTURE_CLIENT_TCP) {
    ipc->EnqueueNetTask(run_ctx->future_, NetQueuePriority::kClientSendTcp);
  } else {
    ipc->EnqueueNetTask(run_ctx->future_, NetQueuePriority::kClientSendIpc);
  }
}

}  // namespace chi

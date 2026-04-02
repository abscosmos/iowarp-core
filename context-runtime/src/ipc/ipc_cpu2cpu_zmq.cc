/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#include "chimaera/ipc_manager.h"

namespace chi {

void IpcCpu2CpuZmq::RuntimeSend(
    IpcManager *ipc, RunContext *run_ctx, u32 origin) {
  if (origin == FutureShm::FUTURE_CLIENT_TCP) {
    ipc->EnqueueNetTask(run_ctx->future_, NetQueuePriority::kClientSendTcp);
  } else {
    ipc->EnqueueNetTask(run_ctx->future_, NetQueuePriority::kClientSendIpc);
  }
}

}  // namespace chi

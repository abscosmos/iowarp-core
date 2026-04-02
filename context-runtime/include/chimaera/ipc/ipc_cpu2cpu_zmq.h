/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CHIMAERA_INCLUDE_CHIMAERA_IPC_CPU2CPU_ZMQ_H_
#define CHIMAERA_INCLUDE_CHIMAERA_IPC_CPU2CPU_ZMQ_H_

#include "chimaera/types.h"
#include "chimaera/task.h"

namespace chi {

class IpcManager;

/**
 * IPC transport for CPU client → CPU runtime via ZeroMQ (TCP or IPC).
 */
struct IpcCpu2CpuZmq {
  /** Serialize and send via ZMQ. */
  template <typename TaskT>
  static Future<TaskT> ClientSend(IpcManager *ipc,
                                   const hipc::FullPtr<TaskT> &task_ptr,
                                   IpcMode mode);

  /** Enqueue to net queue for TCP/IPC response. */
  static void RuntimeSend(IpcManager *ipc, RunContext *run_ctx, u32 origin);

  /** Wait for COMPLETE, deserialize from pending archives. */
  template <typename TaskT>
  static bool ClientRecv(IpcManager *ipc,
                          Future<TaskT> &future, float max_sec);

  /** Re-send a task via ZMQ after server restart. */
  template <typename TaskT>
  static void ResendTask(IpcManager *ipc, Future<TaskT> &future);
};

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_IPC_CPU2CPU_ZMQ_H_

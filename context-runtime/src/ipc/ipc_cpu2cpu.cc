/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#include "clio_runtime/ipc/ipc_cpu2cpu.h"
#include "clio_runtime/ipc_manager.h"
#include "clio_runtime/singletons.h"

namespace clio::run {

bool IpcCpu2Cpu::RecvIn(IpcManager *ipc, u32 shard) {
  // Drain ONE inbound shard ring (shm_in_servers_[shard], DONTWAIT) and enqueue
  // any received external-client task onto the normal dispatch path. issue #807:
  // there are now S shard rings, each with its own dedicated drain thread, so
  // each ring still has exactly one consumer — the MPSC single-consumer contract
  // holds per shard — and S clients-worth of deserialize+route runs on S cores.
  if (!ipc->shm_in_server_ok_ || shard >= ipc->shm_in_servers_.size() ||
      ipc->shm_in_servers_[shard] == nullptr) {
    return false;
  }
  LoadTaskArchive archive;
  ctp::lbm::ClientInfo info =
      ipc->shm_in_servers_[shard]->Recv(archive, ctp::lbm::SHM_MPSC_DONTWAIT);
  if (info.rc != 0) {
    return false;
  }
  const auto &tis = archive.GetTaskInfos();
  if (tis.empty()) {
    return false;
  }
  const auto &ti = tis[0];
  auto container = CLIO_POOL_MANAGER->GetStaticContainer(ti.pool_id_).get();
  if (container == nullptr) {
    return false;
  }
  clio::run::shared_ptr<clio::run::Task> tp =
      container->AllocLoadTask(ti.method_id_, archive);
  if (tp.IsNull()) {
    return false;
  }
  tp->SetFlags(TASK_EXTERNAL_CLIENT);
  // The Future owns the FutureShm via shared_ptr; pushing it onto the lane
  // copies the Future, so the FutureShm stays alive until the worker (and its
  // RunContext copy) is done with it.
  // The Future ctor ensures tp's RunContext exists (its embedded route_ holds
  // the routing state); BeginRunContext below reuses it (idempotent).
  Future<Task> f(ti.pool_id_, ti.method_id_, tp);
  auto fs = f.GetFutureShm();
  fs->origin_ = ClientOrigin::kClientShm;
  fs->client_pid_ = ti.task_id_.pid_;
  // Preserve the CLIENT's response-matching key (issue #774 / #768). The
  // runtime repurposes task_id_.net_key_ for its own bookkeeping when the task
  // is forwarded cross-node (IpcManagerRun2Run::SendIn overwrites it with the
  // daemon-side send_map key), so a forwarded task's response would otherwise
  // reach the client bearing the DAEMON's key — the client's net_key demux
  // then stashes it forever and Future::Wait() never returns. Mirror of the
  // ZMQ path's save (ipc_cpu2cpu_zmq.cc RecvIn) / restore (SendOut).
  fs->client_net_key_ = ti.task_id_.net_key_;
  // The SHM client blocks on its own MPSC server clio-<pid>-<tid>; SendOut routes
  // the result back there using the waiter (the OS pid/tid stamped by SendIn,
  // carried in task_id_). net_key (task_id_.net_key_) is already on the task.
  tp->SetWaiter(ti.task_id_.pid_, ti.task_id_.tid_);
  // Allocate the task's RunContext (and resolve its container) now that it is
  // deserialized, so RouteTask / the worker have an active RunContext.
  f.GetTaskPtr()->BeginRunContext();
  // issue #781: ClientMapTask removed. Like the ZMQ client recv thread
  // (IpcCpu2CpuZmq::RecvIn), this SHM recv path deposits on the shared ingress
  // lane (0); the runtime maps the task to a worker via RuntimeMapTask. Always
  // signal: see ipc_cpu2cpu_impl.h for the enqueue/sleep race this closes.
  LaneId lane_id = 0;
  auto &lane_ref = ipc->GetTaskQueue()->GetLane(lane_id, 0);
  lane_ref.Push(f);
  ipc->AwakenWorker(&lane_ref);
  return true;
}

clio::run::shared_ptr<clio::run::Task> IpcCpu2Cpu::RecvIn(
    IpcManager *ipc, Future<Task> &future,
    u32 method_id, ctp::lbm::Transport *recv_transport) {
  // The inbound MPSC SHM drain (the RecvIn(ipc, lane) overload above) already
  // deserialized the task from the client's serialized bytes and stamped the
  // Future's task pointer before enqueuing it. There is no inline copy_space to
  // deserialize from here, so just return the already-resolved task pointer.
  (void)ipc;
  (void)method_id;
  (void)recv_transport;
  return future.GetTaskPtr();
}

void IpcCpu2Cpu::SendOut(
    IpcManager *ipc, const clio::run::shared_ptr<Task> &task_ptr,
    ctp::lbm::Transport *send_transport) {
  // issue #807: do NOT serialize or transfer on the worker. Enqueue an OWNING
  // future onto the destination client's per-client send queue and return; the
  // background sender thread does the work via SendOutTransfer.
  //
  // The owning copy is mandatory, exactly as in IpcCpu2CpuZmq::EnqueueSendOut:
  // task_ptr->RunFuture()'s task_ptr_ may be a NON-OWNING self-handle (RouteGlobal
  // / RouteManyToOne rebind it non-owning to break the leak cycle), so enqueuing
  // that handle could let the task free before the sender drains it — its
  // GetFutureShm() would then resolve through freed memory and the client's
  // Wait() would hang. An owning copy keeps the task alive until the response is
  // on the wire; run_ctx_->future_ stays non-owning, so no leak is added.
  (void)send_transport;  // the sender thread owns the transport it serializes on
  clio::run::Future<Task> owning = task_ptr->RunFuture();
  owning.GetTaskPtr() = task_ptr;
  std::string dest =
      "clio-" + std::to_string(task_ptr->WaiterPid()) + "-shm-out";
  ipc->EnqueueShmSend(dest, std::move(owning));
}

void IpcCpu2Cpu::SendOutTransfer(
    IpcManager *ipc, const clio::run::shared_ptr<Task> &task_ptr,
    ctp::lbm::Transport *send_transport) {
  clio::run::ContainerHold container =
      CLIO_POOL_MANAGER->GetStaticContainer(task_ptr->pool_id_).get();
  auto future_shm = task_ptr->RunFuture().GetFutureShm();

  // Serialize the result and high-level Send it to the originating client's
  // SINGLE per-process response ring ("clio-<client_pid>-shm-out"), drained by
  // that client's RecvShmClientThread. The waiter tid is no longer part of the
  // ring name (one ring per process); it is still carried on the task so the
  // client recv thread can Signal the exact waiting thread. send_transport is
  // used only to Expose bulk descriptors while building the archive; conn->Send
  // performs the actual MPSC transfer (metadata + data).
  std::string name =
      "clio-" + std::to_string(task_ptr->WaiterPid()) + "-shm-out";
  ctp::lbm::ShmMpscTransport *conn = ipc->GetOrCreateShmConn(name);
  if (conn != nullptr) {
    // Restore the CLIENT's response-matching key before serializing: a
    // cross-node round trip leaves task_id_.net_key_ holding run2run's
    // send_map key (see RecvIn above); the client demuxes responses by ITS
    // key, so echo back what it sent. Mirror of ipc_cpu2cpu_zmq.cc SendOut.
    if (!future_shm.IsNull() && future_shm->client_net_key_ != 0) {
      task_ptr->task_id_.net_key_ = future_shm->client_net_key_;
    }
    SaveTaskArchive archive(MsgType::kSerializeOut, send_transport);
    // SaveTask takes a non-const shared_ptr&; copy the (const) handle.
    clio::run::shared_ptr<clio::run::Task> save_handle = task_ptr;
    container->SaveTask(save_handle->method_, archive, save_handle);
    conn->Send(archive);
  } else {
    HLOG(kError, "IpcCpu2Cpu::SendOut: no client server '{}'", name);
  }

  // Signal completion (per-process: the client's own task is woken via the MPSC
  // response above). The task frees via RAII when the owning shared_ptr (held by
  // the worker's RunContext/Future) drops — no explicit DelTask.
  task_ptr->SetComplete();
  task_ptr->ClearFlags(TASK_DATA_OWNER);
}

}  // namespace clio::run

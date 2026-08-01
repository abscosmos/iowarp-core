/**
 * Container virtual API dispatch for the replication ChiMod (Run / Save /
 * Load / NewTask / Aggregate), switch-case over Method ids. Hand-maintained
 * to match clio_mod.yaml + replication_methods.h (mirrors the filesystem
 * chimod's autogen/filesystem_lib_exec.cc).
 */
#include "clio_cte/replication/replication_runtime.h"
#include "clio_cte/replication/autogen/replication_methods.h"
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/task.h>

namespace clio::cte::replication {

// One case body per (function, method); declared once to keep the dispatch
// functions in sync.
#define CLIO_REPL_FOR_EACH_METHOD(X)               \
  X(kCreate, CreateTask, Create)                   \
  X(kDestroy, DestroyTask, Destroy)                \
  X(kMonitor, MonitorTask, Monitor)                \
  X(kReplicateBlob, ReplicateBlobTask, ReplicateBlob) \
  X(kFlushTag, FlushTagTask, FlushTag)

void Runtime::Init(const clio::run::PoolId &pool_id, const std::string &pool_name,
                   clio::run::u32 container_id) {
  clio::run::Container::Init(pool_id, pool_name, container_id);
  DefineModel(Method::kMaxMethodId);
  SetMethodNames(Method::GetMethodNames());
}

clio::run::u64 Runtime::GetWorkRemaining() const { return 0; }

clio::run::TaskResume Runtime::Run(clio::run::u32 method,
                             clio::run::shared_ptr<clio::run::Task> task_ptr) {
  CLIO_TASK_BODY_BEGIN
  switch (method) {
#define X(MID, TASK, HANDLER)                                       \
    case Method::MID: {                                             \
      auto &typed = task_ptr.template Cast<TASK>();                 \
      CLIO_CO_AWAIT(HANDLER(typed));                                \
      break;                                                        \
    }
    CLIO_REPL_FOR_EACH_METHOD(X)
#undef X
    default:
      break;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::SaveTask(clio::run::u32 method, clio::run::SaveTaskArchive &archive,
                       clio::run::shared_ptr<clio::run::Task> &task_ptr) {
  switch (method) {
#define X(MID, TASK, HANDLER)                                  \
    case Method::MID:                                          \
      archive << *task_ptr.template Cast<TASK>();              \
      break;
    CLIO_REPL_FOR_EACH_METHOD(X)
#undef X
    default:
      break;
  }
}

void Runtime::LoadTask(clio::run::u32 method, clio::run::LoadTaskArchive &archive,
                       clio::run::shared_ptr<clio::run::Task> &task_ptr) {
  switch (method) {
#define X(MID, TASK, HANDLER)                                  \
    case Method::MID:                                          \
      archive >> *task_ptr.template Cast<TASK>();              \
      break;
    CLIO_REPL_FOR_EACH_METHOD(X)
#undef X
    default:
      break;
  }
}

clio::run::shared_ptr<clio::run::Task> Runtime::AllocLoadTask(
    clio::run::u32 method, clio::run::LoadTaskArchive &archive) {
  clio::run::shared_ptr<clio::run::Task> task_ptr = NewTask(method);
  if (!task_ptr.IsNull()) {
    LoadTask(method, archive, task_ptr);
  }
  return task_ptr;
}

void Runtime::LocalLoadTask(clio::run::u32 method, clio::run::DefaultLoadArchive &archive,
                            clio::run::shared_ptr<clio::run::Task> &task_ptr) {
  switch (method) {
#define X(MID, TASK, HANDLER)                                  \
    case Method::MID:                                          \
      archive >> *task_ptr.template Cast<TASK>();              \
      break;
    CLIO_REPL_FOR_EACH_METHOD(X)
#undef X
    default:
      break;
  }
}

clio::run::shared_ptr<clio::run::Task> Runtime::LocalAllocLoadTask(
    clio::run::u32 method, clio::run::DefaultLoadArchive &archive) {
  clio::run::shared_ptr<clio::run::Task> task_ptr = NewTask(method);
  if (!task_ptr.IsNull()) {
    LocalLoadTask(method, archive, task_ptr);
  }
  return task_ptr;
}

void Runtime::LocalSaveTask(clio::run::u32 method, clio::run::DefaultSaveArchive &archive,
                            clio::run::shared_ptr<clio::run::Task> &task_ptr) {
  switch (method) {
#define X(MID, TASK, HANDLER)                                  \
    case Method::MID:                                          \
      archive << *task_ptr.template Cast<TASK>();              \
      break;
    CLIO_REPL_FOR_EACH_METHOD(X)
#undef X
    default:
      break;
  }
}

clio::run::shared_ptr<clio::run::Task> Runtime::NewCopyTask(
    clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task_ptr, bool deep) {
  auto *ipc_manager = CLIO_IPC;
  if (!ipc_manager) {
    return clio::run::shared_ptr<clio::run::Task>();
  }
  (void)deep;
  switch (method) {
#define X(MID, TASK, HANDLER)                                            \
    case Method::MID: {                                                  \
      auto new_task = ipc_manager->NewTask<TASK>();                      \
      if (!new_task.IsNull()) {                                          \
        new_task->Copy(ctp::ipc::FullPtr<TASK>(                          \
            orig_task_ptr.template Cast<TASK>().get()));                 \
        return new_task.template Cast<clio::run::Task>();                \
      }                                                                  \
      break;                                                             \
    }
    CLIO_REPL_FOR_EACH_METHOD(X)
#undef X
    default: {
      auto new_task = ipc_manager->NewTask<clio::run::Task>();
      if (!new_task.IsNull()) {
        new_task->Copy(ctp::ipc::FullPtr<clio::run::Task>(orig_task_ptr.get()));
        return new_task;
      }
      break;
    }
  }
  return clio::run::shared_ptr<clio::run::Task>();
}

clio::run::shared_ptr<clio::run::Task> Runtime::NewTask(clio::run::u32 method) {
  auto *ipc_manager = CLIO_IPC;
  if (!ipc_manager) {
    return clio::run::shared_ptr<clio::run::Task>();
  }
  switch (method) {
#define X(MID, TASK, HANDLER)                                  \
    case Method::MID:                                          \
      return ipc_manager->NewTask<TASK>().template Cast<clio::run::Task>();
    CLIO_REPL_FOR_EACH_METHOD(X)
#undef X
    default:
      return clio::run::shared_ptr<clio::run::Task>();
  }
}

void Runtime::AggregateOut(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task,
                           const clio::run::shared_ptr<clio::run::Task> &replica_task) {
  switch (method) {
#define X(MID, TASK, HANDLER)                                            \
    case Method::MID:                                                    \
      orig_task.template Cast<TASK>()->AggregateOut(                     \
          ctp::ipc::FullPtr<clio::run::Task>(replica_task.get()));       \
      break;
    CLIO_REPL_FOR_EACH_METHOD(X)
#undef X
    default:
      orig_task->AggregateOut(
          ctp::ipc::FullPtr<clio::run::Task>(replica_task.get()));
      break;
  }
}

void Runtime::AggregateIn(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &agg_task,
                          const clio::run::shared_ptr<clio::run::Task> &member_task) {
  // No ManyToOne methods; the aggregate stays a copy of the first member.
  (void)method;
  (void)agg_task;
  (void)member_task;
}

#undef CLIO_REPL_FOR_EACH_METHOD

}  // namespace clio::cte::replication

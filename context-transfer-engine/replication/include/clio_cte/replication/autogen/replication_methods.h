/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#ifndef CLIO_CTE_REPLICATION_AUTOGEN_METHODS_H_
#define CLIO_CTE_REPLICATION_AUTOGEN_METHODS_H_

#include <clio_runtime/clio_runtime.h>
#include <string>
#include <vector>

/**
 * Method ids for the replication chimod. Hand-maintained (same as the CTE
 * core, compressor and filesystem chimods). Keep in sync with clio_mod.yaml
 * and the switch cases in autogen/replication_lib_exec.cc.
 */
namespace clio::cte::replication {

namespace Method {
GLOBAL_CROSS_CONST clio::run::u32 kCreate = 0;
GLOBAL_CROSS_CONST clio::run::u32 kDestroy = 1;
GLOBAL_CROSS_CONST clio::run::u32 kMonitor = 9;

// replication-specific methods
GLOBAL_CROSS_CONST clio::run::u32 kReplicateBlob = 10;
GLOBAL_CROSS_CONST clio::run::u32 kFlushTag = 11;
GLOBAL_CROSS_CONST clio::run::u32 kCachedPut = 12;
GLOBAL_CROSS_CONST clio::run::u32 kCachedGet = 13;

GLOBAL_CROSS_CONST clio::run::u32 kMaxMethodId = 14;

inline const std::vector<std::string>& GetMethodNames() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v(kMaxMethodId);
    v[0] = "Create";
    v[1] = "Destroy";
    v[9] = "Monitor";
    v[10] = "ReplicateBlob";
    v[11] = "FlushTag";
    v[12] = "CachedPut";
    v[13] = "CachedGet";
    return v;
  }();
  return names;
}
}  // namespace Method

}  // namespace clio::cte::replication

#endif  // CLIO_CTE_REPLICATION_AUTOGEN_METHODS_H_

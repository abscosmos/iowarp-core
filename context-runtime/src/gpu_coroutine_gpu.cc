/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * See COPYING file in the top-level directory.
 */

/**
 * GPU-side coroutine frame allocator wrappers.
 *
 * These functions wrap CHI_PRIV_ALLOC (PrivateBuddyAllocator) for use by
 * gpu_coroutine.h's promise_type::operator new/delete and RunContext.
 * Defined in a _gpu.cc file so that ipc_manager.h (heavy) is not pulled
 * into the low-level coroutine header.
 */

#include "chimaera/singletons.h"

namespace chi {
namespace gpu {

HSHM_GPU_FUN hipc::FullPtr<char> GpuCoroAlloc(size_t size) {
  auto *alloc = CHI_PRIV_ALLOC;
  if (!alloc) return hipc::FullPtr<char>::GetNull();
  return alloc->AllocateObjs<char>(size);
}

HSHM_GPU_FUN void GpuCoroFree(hipc::FullPtr<char> fp) {
  auto *alloc = CHI_PRIV_ALLOC;
  if (!alloc || fp.IsNull()) return;
  alloc->Free(fp);
}

HSHM_GPU_FUN void GpuCoroFreeRaw(void *ptr) {
  if (!ptr) return;
  auto *alloc = CHI_PRIV_ALLOC;
  if (!alloc) return;
  hipc::FullPtr<char> fp(static_cast<char *>(ptr));
  fp.shm_.off_ = static_cast<size_t>(
      static_cast<char *>(ptr) - alloc->GetBackendData());
  fp.shm_.alloc_id_ = alloc->GetId();
  alloc->Free(fp);
}

}  // namespace gpu
}  // namespace chi

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of Hermes. The full Hermes copyright notice, including  *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the top directory. If you do not  *
 * have access to the file, you may request a copy from help@hdfgroup.org.   *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef HSHM_INCLUDE_MEMORY_BACKEND_POSIX_SHM_MMAP_H
#define HSHM_INCLUDE_MEMORY_BACKEND_POSIX_SHM_MMAP_H

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/util/errors.h"
#include "hermes_shm/util/logging.h"
#include "memory_backend.h"

namespace hshm::ipc {

class PosixShmMmap : public MemoryBackend, public UrlMemoryBackend {
 protected:
  File fd_;
  std::string url_;
  size_t total_size_;

 public:
  /** Constructor */
  HSHM_CROSS_FUN
  PosixShmMmap() {}

  /** Destructor */
  HSHM_CROSS_FUN
  ~PosixShmMmap() {
#if HSHM_IS_HOST
    if (IsOwned()) {
      _Destroy();
    } else {
      _Detach();
    }
#endif
  }

  /** Initialize backend */
  bool shm_init(const MemoryBackendId &backend_id, size_t size,
                const std::string &url) {
    // Enforce minimum backend size of 1MB
    constexpr size_t kMinBackendSize = 1024 * 1024;  // 1MB
    if (size < kMinBackendSize) {
      size = kMinBackendSize;
    }

    // Initialize flags before calling methods that use it
    flags_.Clear();
    SetInitialized();
    Own();

    // Calculate sizes: header + md section + alignment + data section
    constexpr size_t kAlignment = 4096;  // 4KB alignment
    size_t header_size = sizeof(MemoryBackendHeader);
    size_t md_size = header_size;  // md section stores the header
    size_t aligned_md_size = ((md_size + kAlignment - 1) / kAlignment) * kAlignment;
    total_size_ = aligned_md_size + size;

    // Create shared memory
    SystemInfo::DestroySharedMemory(url);
    if (!SystemInfo::CreateNewSharedMemory(fd_, url, total_size_)) {
      char *err_buf = strerror(errno);
      HILOG(kError, "shm_open failed: {}", err_buf);
      return false;
    }
    url_ = url;

    // Map the entire shared memory region as one contiguous block
    char *ptr = _ShmMap(total_size_, 0);
    if (!ptr) {
      return false;
    }

    // Layout: [MemoryBackendHeader | padding to 4KB] [data]
    header_ = reinterpret_cast<MemoryBackendHeader *>(ptr);
    new (header_) MemoryBackendHeader();
    header_->id_ = backend_id;
    header_->md_size_ = md_size;
    header_->data_size_ = size;
    header_->data_id_ = -1;
    header_->flags_.Clear();

    // md_ points to the header itself (metadata for process connection)
    md_ = ptr;
    md_size_ = md_size;

    // data_ starts at 4KB aligned boundary after md section
    data_ = ptr + aligned_md_size;
    data_size_ = size;
    data_id_ = -1;

    return true;
  }

  /** Deserialize the backend */
  bool shm_attach(const std::string &url) {
    flags_.Clear();
    SetInitialized();
    Disown();

    if (!SystemInfo::OpenSharedMemory(fd_, url)) {
      const char *err_buf = strerror(errno);
      HILOG(kError, "shm_open failed: {}", err_buf);
      return false;
    }
    url_ = url;

    // First, map just the header to get the size information
    constexpr size_t kAlignment = 4096;
    header_ = (MemoryBackendHeader *)_ShmMap(kAlignment, 0);

    // Calculate total size based on header information
    size_t md_size = header_->md_size_;
    size_t aligned_md_size = ((md_size + kAlignment - 1) / kAlignment) * kAlignment;
    total_size_ = aligned_md_size + header_->data_size_;

    // Unmap the header
    SystemInfo::UnmapMemory(header_, kAlignment);

    // Map the entire region
    char *ptr = _ShmMap(total_size_, 0);
    if (!ptr) {
      return false;
    }

    // Set up pointers
    header_ = reinterpret_cast<MemoryBackendHeader *>(ptr);
    md_ = ptr;
    md_size_ = header_->md_size_;
    data_ = ptr + aligned_md_size;
    data_size_ = header_->data_size_;
    data_id_ = header_->data_id_;

    return true;
  }

  /** Detach the mapped memory */
  void shm_detach() { _Detach(); }

  /** Destroy the mapped memory */
  void shm_destroy() { _Destroy(); }

 protected:
  /** Map shared memory */
  char *_ShmMap(size_t size, i64 off) {
    char *ptr =
        reinterpret_cast<char *>(SystemInfo::MapSharedMemory(fd_, size, off));
    if (!ptr) {
      HSHM_THROW_ERROR(SHMEM_CREATE_FAILED);
    }
    return ptr;
  }

  /** Unmap shared memory */
  void _Detach() {
    if (!IsInitialized()) {
      return;
    }
    // Unmap the entire contiguous region
    SystemInfo::UnmapMemory(reinterpret_cast<void *>(header_), total_size_);
    SystemInfo::CloseSharedMemory(fd_);
    UnsetInitialized();
  }

  /** Destroy shared memory */
  void _Destroy() {
    if (!IsInitialized()) {
      return;
    }
    _Detach();
    SystemInfo::DestroySharedMemory(url_);
    UnsetInitialized();
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_INCLUDE_MEMORY_BACKEND_POSIX_SHM_MMAP_H

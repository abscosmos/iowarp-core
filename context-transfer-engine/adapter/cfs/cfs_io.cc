/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 *
 * Implementation of the shared CTE adapter I/O core (see cfs_io.h).
 */
#include "cfs_io.h"

#include <algorithm>

namespace clio::cae {

CTP_DEFINE_GLOBAL_PTR_VAR_CC(CfsIo, g_cfs_io);

bool CfsIo::EnsureClient() {
  if (client_ready_) {
    return true;
  }
  client_ready_ = clio::cte::filesystem::CLIO_CFS_CLIENT_INIT();
  if (!client_ready_) {
    HLOG(kError, "CfsIo: failed to initialize the filesystem client");
  }
  return client_ready_;
}

bool CfsIo::QueryGetattr(const std::string &path, bool *exists,
                         clio::run::u64 *size) {
  if (!EnsureClient()) {
    return false;
  }
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncGetattr(path);
  t.Wait();
  if (t->GetReturnCode() != 0) {
    return false;
  }
  *exists = (t->exists_ != 0);
  *size = t->size_;
  return true;
}

bool CfsIo::QuerySize(const std::string &path, clio::run::u64 *size) {
  bool exists = false;
  if (!QueryGetattr(path, &exists, size)) {
    return false;
  }
  if (!exists) {
    *size = 0;
  }
  return true;
}

ssize_t CfsIo::TryReadShm(const std::string &path, clio::run::u64 off,
                          void *buf, size_t count) {
  // Attach lazily, and RETRY when not yet attached, rather than latching the
  // result of one attempt at init. A client can legitimately come up before
  // its pool has been composed (or before the chimod has registered its cache
  // root), and a one-shot attach at init would leave that process on the RPC
  // path forever -- silently, since a disabled cache and a working one look
  // identical from the outside. Retrying costs a scan of a <=32-entry
  // directory, against an RPC that costs ~100 us.
  auto *cfs = CLIO_CFS_CLIENT;
  if (cfs == nullptr || (!cfs->HasShmCache() && !cfs->AttachShmCache())) {
    return -1;
  }
  auto *cte = CLIO_CTE_CLIENT;
  if (cte == nullptr || (!cte->HasShmCache() && !cte->AttachShmCache())) {
    return -1;  // page payloads come from the core cache; both are required
  }

  clio::cte::filesystem::ShmFileRecord rec;
  if (!cfs->TryGetFileRecordShm(path, &rec) || !rec.IsFastPathable()) {
    return -1;
  }

  // Clamp to the LOGICAL size. This is the number the chimod owns and the one
  // POSIX read() semantics are defined against; the tag's physical byte total
  // would be wrong after a sparse write or an ftruncate-grow.
  //
  // Reading a stale (too small) size here cannot produce a short read for a
  // properly ordered caller: the runtime publishes the new size at the end of
  // the Write handler, so it lands before the write() that grew the file
  // returns to its caller.
  if (off >= rec.size_) {
    return 0;  // at or past EOF
  }
  clio::run::u64 want = count;
  if (off + want > rec.size_) {
    want = rec.size_ - off;
  }

  char *dst = static_cast<char *>(buf);
  clio::run::u64 done = 0;
  clio::run::u64 cur = off;
  while (done < want) {
    clio::run::u64 page_off = cur % clio::cte::filesystem::kFsPageSize;
    clio::run::u64 to_read =
        std::min(clio::cte::filesystem::kFsPageSize - page_off, want - done);
    if (!cte->TryReadBlobShm(rec.tag_id_, clio::cte::filesystem::PageName(cur),
                             dst + done, static_cast<size_t>(to_read),
                             static_cast<size_t>(page_off))) {
      // Abandon the WHOLE request rather than mixing sources. A hole reads as
      // zeros and a missing page is indistinguishable from an uncached one, so
      // the only safe reading of a failure here is "let the runtime answer".
      return -1;
    }
    done += to_read;
    cur += to_read;
  }
  return static_cast<ssize_t>(done);
}

ssize_t CfsIo::DoRead(clio::run::u64 handle, const std::string &path,
                      clio::run::u64 off, void *buf, size_t count) {
  if (count == 0) {
    return 0;
  }
  // issue #817: try the zero-IPC path first. It either answers completely or
  // declines, and declining costs one hash lookup.
  ssize_t fast = TryReadShm(path, off, buf, count);
  if (fast >= 0) {
    return fast;
  }
  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> shm = ipc->AllocateBuffer(count);
  if (shm.IsNull()) {
    errno = ENOMEM;
    return -1;
  }
  ctp::ipc::ShmPtr<> shm_ptr(shm.shm_);
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncRead(handle, off, count, shm_ptr);
  t.Wait();
  ssize_t ret;
  if (t->GetReturnCode() == 0) {
    size_t got = static_cast<size_t>(t->bytes_read_);
    if (got > 0) {
      std::memcpy(buf, shm.ptr_, got);
    }
    ret = static_cast<ssize_t>(got);
  } else {
    errno = EIO;
    ret = -1;
  }
  ipc->FreeBuffer(shm);
  return ret;
}

ssize_t CfsIo::DoWrite(clio::run::u64 handle, clio::run::u64 off, const void *buf,
                       size_t count) {
  if (count == 0) {
    return 0;
  }
  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> shm = ipc->AllocateBuffer(count);
  if (shm.IsNull()) {
    errno = ENOMEM;
    return -1;
  }
  std::memcpy(shm.ptr_, buf, count);
  ctp::ipc::ShmPtr<> shm_ptr(shm.shm_);
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncWrite(handle, off, count, shm_ptr);
  t.Wait();
  ssize_t ret;
  if (t->GetReturnCode() == 0) {
    ret = static_cast<ssize_t>(t->bytes_written_);
  } else {
    errno = EIO;
    ret = -1;
  }
  ipc->FreeBuffer(shm);
  return ret;
}

int CfsIo::Open(const std::string &raw_path, int flags, int mode) {
  if (!EnsureClient()) {
    errno = EIO;
    return -1;
  }
  std::string path = StripClioPrefix(raw_path);
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncOpen(path, static_cast<clio::run::u32>(flags),
                          static_cast<clio::run::u32>(mode));
  t.Wait();
  if (t->GetReturnCode() != 0) {
    errno = EIO;
    return -1;
  }
  if (t->handle_ == 0) {
    // Plain open of a missing file (chimod honors O_CREAT).
    errno = ENOENT;
    return -1;
  }
  clio::run::u64 size = t->size_;
  // O_TRUNC: drop the logical size to zero.
  if (flags & O_TRUNC) {
    auto tr = cfs->AsyncTruncate(path, 0);
    tr.Wait();
    size = 0;
  }
  std::lock_guard<std::mutex> g(mu_);
  int fd = next_fd_++;
  OpenFile of;
  of.handle = t->handle_;
  of.path = path;
  of.flags = flags;
  of.off = (flags & O_APPEND) ? size : 0;
  fds_[fd] = of;
  return fd;
}

ssize_t CfsIo::Read(int fd, void *buf, size_t count) {
  clio::run::u64 handle, off;
  std::string path;
  {
    std::lock_guard<std::mutex> g(mu_);
    auto it = fds_.find(fd);
    if (it == fds_.end()) {
      errno = EBADF;
      return -1;
    }
    handle = it->second.handle;
    off = it->second.off;
    path = it->second.path;
  }
  ssize_t n = DoRead(handle, path, off, buf, count);
  if (n > 0) {
    std::lock_guard<std::mutex> g(mu_);
    auto it = fds_.find(fd);
    if (it != fds_.end()) {
      it->second.off += static_cast<clio::run::u64>(n);
    }
  }
  return n;
}

ssize_t CfsIo::Write(int fd, const void *buf, size_t count) {
  clio::run::u64 handle, off;
  {
    std::lock_guard<std::mutex> g(mu_);
    auto it = fds_.find(fd);
    if (it == fds_.end()) {
      errno = EBADF;
      return -1;
    }
    handle = it->second.handle;
    off = it->second.off;
  }
  ssize_t n = DoWrite(handle, off, buf, count);
  if (n > 0) {
    std::lock_guard<std::mutex> g(mu_);
    auto it = fds_.find(fd);
    if (it != fds_.end()) {
      it->second.off += static_cast<clio::run::u64>(n);
    }
  }
  return n;
}

ssize_t CfsIo::Pread(int fd, void *buf, size_t count, off_t offset) {
  clio::run::u64 handle;
  std::string path;
  {
    std::lock_guard<std::mutex> g(mu_);
    auto it = fds_.find(fd);
    if (it == fds_.end()) {
      errno = EBADF;
      return -1;
    }
    handle = it->second.handle;
    path = it->second.path;
  }
  return DoRead(handle, path, static_cast<clio::run::u64>(offset), buf, count);
}

ssize_t CfsIo::Pwrite(int fd, const void *buf, size_t count, off_t offset) {
  clio::run::u64 handle;
  {
    std::lock_guard<std::mutex> g(mu_);
    auto it = fds_.find(fd);
    if (it == fds_.end()) {
      errno = EBADF;
      return -1;
    }
    handle = it->second.handle;
  }
  return DoWrite(handle, static_cast<clio::run::u64>(offset), buf, count);
}

off_t CfsIo::Seek(int fd, off_t offset, int whence) {
  std::string path;
  clio::run::u64 cur;
  {
    std::lock_guard<std::mutex> g(mu_);
    auto it = fds_.find(fd);
    if (it == fds_.end()) {
      errno = EBADF;
      return -1;
    }
    path = it->second.path;
    cur = it->second.off;
  }
  clio::run::u64 base = 0;
  switch (whence) {
    case SEEK_SET:
      base = 0;
      break;
    case SEEK_CUR:
      base = cur;
      break;
    case SEEK_END: {
      clio::run::u64 size = 0;
      if (!QuerySize(path, &size)) {
        errno = EIO;
        return -1;
      }
      base = size;
      break;
    }
    default:
      errno = EINVAL;
      return -1;
  }
  off_t newoff = static_cast<off_t>(base) + offset;
  if (newoff < 0) {
    errno = EINVAL;
    return -1;
  }
  std::lock_guard<std::mutex> g(mu_);
  auto it = fds_.find(fd);
  if (it == fds_.end()) {
    errno = EBADF;
    return -1;
  }
  it->second.off = static_cast<clio::run::u64>(newoff);
  return newoff;
}

off_t CfsIo::Tell(int fd) {
  std::lock_guard<std::mutex> g(mu_);
  auto it = fds_.find(fd);
  if (it == fds_.end()) {
    errno = EBADF;
    return -1;
  }
  return static_cast<off_t>(it->second.off);
}

off_t CfsIo::SizeFd(int fd) {
  std::string path;
  {
    std::lock_guard<std::mutex> g(mu_);
    auto it = fds_.find(fd);
    if (it == fds_.end()) {
      errno = EBADF;
      return -1;
    }
    path = it->second.path;
  }
  clio::run::u64 size = 0;
  if (!QuerySize(path, &size)) {
    return -1;
  }
  return static_cast<off_t>(size);
}

int CfsIo::Sync(int fd) {
  // Writes are synchronous (each AsyncWrite is awaited), nothing to flush.
  return IsFdTracked(fd) ? 0 : -1;
}

int CfsIo::FtruncateFd(int fd, off_t length) {
  std::string path;
  {
    std::lock_guard<std::mutex> g(mu_);
    auto it = fds_.find(fd);
    if (it == fds_.end()) {
      errno = EBADF;
      return -1;
    }
    path = it->second.path;
  }
  return TruncatePath(std::string(kClioPrefix) + path, length);
}

int CfsIo::TruncatePath(const std::string &raw_path, off_t length) {
  if (!EnsureClient()) {
    errno = EIO;
    return -1;
  }
  std::string path = StripClioPrefix(raw_path);
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncTruncate(path, static_cast<clio::run::u64>(length));
  t.Wait();
  if (t->GetReturnCode() != 0) {
    errno = EIO;
    return -1;
  }
  return 0;
}

int CfsIo::RemovePath(const std::string &raw_path) {
  if (!EnsureClient()) {
    errno = EIO;
    return -1;
  }
  std::string path = StripClioPrefix(raw_path);
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncUnlink(path);
  t.Wait();
  if (t->GetReturnCode() != 0) {
    errno = EIO;
    return -1;
  }
  return 0;
}

int CfsIo::Close(int fd) {
  clio::run::u64 handle;
  {
    std::lock_guard<std::mutex> g(mu_);
    auto it = fds_.find(fd);
    if (it == fds_.end()) {
      errno = EBADF;
      return -1;
    }
    handle = it->second.handle;
    fds_.erase(it);
  }
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncClose(handle);
  t.Wait();
  return (t->GetReturnCode() == 0) ? 0 : -1;
}

}  // namespace clio::cae

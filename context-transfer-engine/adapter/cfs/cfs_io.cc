/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 *
 * Implementation of the shared CTE adapter I/O core (see cfs_io.h).
 *
 * Every operation here is descriptor bookkeeping plus one call into
 * clio::cte::filesystem::Client. Deferred writes, read-your-own-writes, the
 * staging pool, the in-flight window, the zero-IPC read path and the
 * write-behind error latch all live in that client (and, beneath it, in the
 * CTE core's deferred-write registry) — NOT here. An adapter that reimplements
 * any of them is a bug: the next adapter would have to reimplement it too.
 */
#include "cfs_io.h"

#include <mutex>
#include <string>
#include <vector>

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

bool CfsIo::Lookup(int fd, OpenFile *out) {
  std::lock_guard<std::mutex> g(mu_);
  auto it = fds_.find(fd);
  if (it == fds_.end()) {
    errno = EBADF;
    return false;
  }
  *out = it->second;
  return true;
}

void CfsIo::Advance(int fd, clio::run::u64 n) {
  std::lock_guard<std::mutex> g(mu_);
  auto it = fds_.find(fd);
  if (it != fds_.end()) {
    it->second.off += n;
  }
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
  OpenFile of;
  if (!Lookup(fd, &of)) {
    return -1;
  }
  ssize_t n = CLIO_CFS_CLIENT->Read(of.handle, of.path, of.off, buf, count);
  if (n > 0) {
    Advance(fd, static_cast<clio::run::u64>(n));
  }
  return n;
}

ssize_t CfsIo::Write(int fd, const void *buf, size_t count) {
  OpenFile of;
  if (!Lookup(fd, &of)) {
    return -1;
  }
  ssize_t n = CLIO_CFS_CLIENT->Write(of.handle, of.path, of.off, buf, count,
                                     IsSyncFd(of.flags));
  if (n > 0) {
    Advance(fd, static_cast<clio::run::u64>(n));
  }
  return n;
}

ssize_t CfsIo::Pread(int fd, void *buf, size_t count, off_t offset) {
  OpenFile of;
  if (!Lookup(fd, &of)) {
    return -1;
  }
  return CLIO_CFS_CLIENT->Read(of.handle, of.path,
                               static_cast<clio::run::u64>(offset), buf, count);
}

ssize_t CfsIo::Pwrite(int fd, const void *buf, size_t count, off_t offset) {
  OpenFile of;
  if (!Lookup(fd, &of)) {
    return -1;
  }
  return CLIO_CFS_CLIENT->Write(of.handle, of.path,
                                static_cast<clio::run::u64>(offset), buf, count,
                                IsSyncFd(of.flags));
}

off_t CfsIo::Seek(int fd, off_t offset, int whence) {
  OpenFile of;
  if (!Lookup(fd, &of)) {
    return -1;
  }
  clio::run::u64 base = 0;
  switch (whence) {
    case SEEK_SET:
      base = 0;
      break;
    case SEEK_CUR:
      base = of.off;
      break;
    case SEEK_END: {
      // GetSize drains this file's deferred writes first, so EOF includes
      // them (issue #817).
      clio::run::u64 size = 0;
      if (!CLIO_CFS_CLIENT->GetSize(of.path, &size)) {
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
  OpenFile of;
  if (!Lookup(fd, &of)) {
    return -1;
  }
  return static_cast<off_t>(of.off);
}

off_t CfsIo::SizeFd(int fd) {
  OpenFile of;
  if (!Lookup(fd, &of)) {
    return -1;
  }
  clio::run::u64 size = 0;
  if (!CLIO_CFS_CLIENT->GetSize(of.path, &size)) {
    return -1;
  }
  return static_cast<off_t>(size);
}

int CfsIo::Sync(int fd) {
  OpenFile of;
  if (!Lookup(fd, &of)) {
    return -1;
  }
  // Wait for every deferred write on this file and report a latched failure
  // exactly once — fsync and close are the only two places a deferred write's
  // failure can reach the application.
  return CLIO_CFS_CLIENT->Flush(of.path);
}

int CfsIo::FtruncateFd(int fd, off_t length) {
  OpenFile of;
  if (!Lookup(fd, &of)) {
    return -1;
  }
  return TruncatePath(std::string(kClioPrefix) + of.path, length);
}

int CfsIo::TruncatePath(const std::string &raw_path, off_t length) {
  if (!EnsureClient()) {
    errno = EIO;
    return -1;
  }
  std::string path = StripClioPrefix(raw_path);
  auto *cfs = CLIO_CFS_CLIENT;
  // Order matters: a deferred write that landed AFTER the truncate would undo
  // it, so drain before resizing (issue #817).
  cfs->Flush(path);
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
  // Drain first: a write still deferred against a deleted path would
  // resurrect the tag.
  cfs->Flush(path);
  auto t = cfs->AsyncUnlink(path);
  t.Wait();
  if (t->GetReturnCode() != 0) {
    errno = EIO;
    return -1;
  }
  return 0;
}

int CfsIo::RenamePath(const std::string &raw_src, const std::string &raw_dst) {
  if (!EnsureClient()) {
    errno = EIO;
    return -1;
  }
  std::string src = StripClioPrefix(raw_src);
  std::string dst = StripClioPrefix(raw_dst);
  auto *cfs = CLIO_CFS_CLIENT;
  // Both sides drain: a deferred write to either path must land before the
  // namespace moves, or it would be applied to a name that no longer means
  // the same file.
  cfs->Flush(src);
  cfs->Flush(dst);
  auto t = cfs->AsyncRename(src, dst);
  t.Wait();
  if (t->GetReturnCode() != 0) {
    errno = EIO;
    return -1;
  }
  return 0;
}

int CfsIo::Readdir(const std::string &raw_path, std::vector<std::string> *out) {
  if (!EnsureClient()) {
    errno = EIO;
    return -1;
  }
  std::string path = StripClioPrefix(raw_path);
  auto t = CLIO_CFS_CLIENT->AsyncReaddir(path);
  t.Wait();
  if (t->GetReturnCode() != 0) {
    errno = ENOENT;
    return -1;
  }
  out->clear();
  out->reserve(t->entries_.size());
  for (const auto &e : t->entries_) {
    out->emplace_back(e.str());
  }
  return 0;
}

int CfsIo::Close(int fd) {
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
    fds_.erase(it);
  }
  auto *cfs = CLIO_CFS_CLIENT;
  // Drain BEFORE releasing the chimod handle: a deferred write names that
  // handle, and the runtime answers EBADF once it is gone (issue #817).
  // close(2) is also the last chance to report a latched write failure, which
  // is why its return code is not simply the Close task's.
  int werr = cfs->Flush(path);
  auto t = cfs->AsyncClose(handle);
  t.Wait();
  if (werr != 0) {
    errno = EIO;
    return -1;
  }
  return (t->GetReturnCode() == 0) ? 0 : -1;
}

}  // namespace clio::cae

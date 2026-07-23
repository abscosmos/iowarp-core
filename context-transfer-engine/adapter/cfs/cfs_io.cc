/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 *
 * Implementation of the shared CTE adapter I/O core (see cfs_io.h).
 */
#include "cfs_io.h"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <utility>

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
  // Read-your-own-writes. A queued write is not visible to either path until
  // its task runs -- the SHM mirror still carries the old size and the old
  // page bytes -- so a read that overlaps one must wait for it. This is the
  // per-client pending-write set the #783 design anticipated; it only became
  // necessary once writes stopped blocking.
  DrainIfOverlap(path, off, count);
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

// ---- issue #817: the async write window -----------------------------------

bool CfsIo::AsyncWritesEnabled() {
  // ON by default (issue #817). write(2) hands the bytes to the chimod and
  // returns; fsync/close drain and report.
  //
  // Measured on 128 x 64 KiB overwrite + fsync, five runs each, medians:
  // queued 5.77 ms vs blocking 6.86 ms. A modest ~13% with overlapping
  // ranges -- the honest summary is "no worse, probably a little better",
  // not a headline.
  //
  // A NOTE ON HOW EASY THIS IS TO MEASURE WRONG: the first pass over a fresh
  // file allocates every block and a second pass does not, so timing one mode
  // on a new file and the other on the warmed one compares allocation against
  // overwrite. Doing exactly that produced a confident, entirely spurious
  // "async is 2.3x SLOWER" -- with the two arms differing by 40% even when
  // both ran the SAME code path. The benchmark now warms first and times only
  // overwrites; keep it that way.
  //
  // Set CLIO_CFS_ASYNC_WRITES=0 to fall back to blocking writes.
  static const bool v = [] {
    if (const char *e = std::getenv("CLIO_CFS_ASYNC_WRITES")) {
      return !(std::string(e) == "0" || std::string(e) == "false");
    }
    return true;
  }();
  return v;
}

clio::run::u64 CfsIo::MaxInflightBytes() {
  static const clio::run::u64 v = [] {
    if (const char *e = std::getenv("CLIO_CFS_WRITE_WINDOW_BYTES")) {
      char *end = nullptr;
      unsigned long long n = std::strtoull(e, &end, 10);
      if (end != e && n > 0) return static_cast<clio::run::u64>(n);
    }
    return static_cast<clio::run::u64>(8ULL * 1024 * 1024);
  }();
  return v;
}

size_t CfsIo::MaxInflightWrites() {
  static const size_t v = [] {
    if (const char *e = std::getenv("CLIO_CFS_WRITE_WINDOW_COUNT")) {
      char *end = nullptr;
      unsigned long long n = std::strtoull(e, &end, 10);
      if (end != e && n > 0) return static_cast<size_t>(n);
    }
    return static_cast<size_t>(64);
  }();
  return v;
}

std::shared_ptr<PathWrites> CfsIo::PendingFor(const std::string &path,
                                              bool create) {
  std::lock_guard<std::mutex> g(pw_mu_);
  auto it = pending_writes_.find(path);
  if (it != pending_writes_.end()) {
    return it->second;
  }
  if (!create) {
    return nullptr;
  }
  auto pw = std::make_shared<PathWrites>();
  pending_writes_[path] = pw;
  return pw;
}

int CfsIo::ReapAndBound(PathWrites *pw) {
  auto *ipc = CLIO_IPC;
  std::lock_guard<std::mutex> g(pw->mu);

  // Retire anything already finished. Front to back: the queue is in
  // submission order, and a completed write behind an outstanding one still
  // has to keep its slot so `bytes` and the drain order stay coherent.
  size_t done = 0;
  while (done < pw->q.size() && pw->q[done].fut.IsComplete()) {
    PendingWrite &p = pw->q[done];
    if (p.fut->GetReturnCode() != 0 && pw->sticky == 0) {
      pw->sticky = EIO;
    }
    ipc->FreeBuffer(p.buf);
    pw->bytes -= p.size;
    ++done;
  }
  if (done > 0) {
    pw->q.erase(pw->q.begin(), pw->q.begin() + static_cast<long>(done));
  }

  // Still over the window: block on the oldest until we are under it. This is
  // the back-pressure -- without it a writer outruns the runtime and the
  // staging buffers grow without bound.
  while (pw->bytes > MaxInflightBytes() || pw->q.size() > MaxInflightWrites()) {
    PendingWrite &p = pw->q.front();
    p.fut.Wait();
    if (p.fut->GetReturnCode() != 0 && pw->sticky == 0) {
      pw->sticky = EIO;
    }
    ipc->FreeBuffer(p.buf);
    pw->bytes -= p.size;
    pw->q.erase(pw->q.begin());
  }
  return pw->sticky;
}

int CfsIo::DrainPath(const std::string &path) {
  auto pw = PendingFor(path, /*create=*/false);
  if (!pw) {
    return 0;
  }
  auto *ipc = CLIO_IPC;
  std::lock_guard<std::mutex> g(pw->mu);
  for (auto &p : pw->q) {
    p.fut.Wait();
    if (p.fut->GetReturnCode() != 0 && pw->sticky == 0) {
      pw->sticky = EIO;
    }
    ipc->FreeBuffer(p.buf);
  }
  pw->q.clear();
  pw->bytes = 0;
  // Report a latched error exactly once, to whoever asks first.
  int err = pw->sticky;
  pw->sticky = 0;
  return err;
}

void CfsIo::ForgetIfIdle(const std::string &path) {
  std::lock_guard<std::mutex> g(pw_mu_);
  auto it = pending_writes_.find(path);
  if (it == pending_writes_.end()) {
    return;
  }
  // Only drop a window that is genuinely finished: an empty queue with no
  // error still owed to someone. Otherwise a long-lived process (an
  // interceptor lives as long as the application) accumulates one entry per
  // file it ever wrote.
  std::lock_guard<std::mutex> g2(it->second->mu);
  if (it->second->q.empty() && it->second->sticky == 0) {
    pending_writes_.erase(it);
  }
}

void CfsIo::WaitForPageOverlap(PathWrites *pw, clio::run::u64 off,
                               clio::run::u64 len) {
  const clio::run::u64 page = clio::cte::filesystem::kFsPageSize;
  const clio::run::u64 first = off / page;
  const clio::run::u64 last = (off + len - 1) / page;
  auto *ipc = CLIO_IPC;
  std::lock_guard<std::mutex> g(pw->mu);
  size_t keep = 0;
  for (size_t i = 0; i < pw->q.size(); ++i) {
    PendingWrite &p = pw->q[i];
    const clio::run::u64 pf = p.off / page;
    const clio::run::u64 pl = (p.off + p.size - 1) / page;
    if (pl < first || pf > last) {
      // Different pages: leave it in flight, it cannot contend.
      if (keep != i) pw->q[keep] = std::move(pw->q[i]);
      ++keep;
      continue;
    }
    p.fut.Wait();
    if (p.fut->GetReturnCode() != 0 && pw->sticky == 0) {
      pw->sticky = EIO;
    }
    ipc->FreeBuffer(p.buf);
    pw->bytes -= p.size;
  }
  pw->q.resize(keep);
}

void CfsIo::DrainIfOverlap(const std::string &path, clio::run::u64 off,
                           clio::run::u64 len) {
  auto pw = PendingFor(path, /*create=*/false);
  if (!pw) {
    return;
  }
  bool overlap = false;
  {
    std::lock_guard<std::mutex> g(pw->mu);
    for (const auto &p : pw->q) {
      if (off < p.off + p.size && p.off < off + len) {
        overlap = true;
        break;
      }
    }
  }
  if (overlap) {
    // Drain everything, not just the overlapping entries: the queue is
    // ordered, and waiting on a later write while an earlier one is still
    // outstanding would leave the file in a state no single read reflects.
    // The sticky error is deliberately NOT consumed here -- a read must not
    // swallow the report owed to the next write/fsync/close.
    auto *ipc = CLIO_IPC;
    std::lock_guard<std::mutex> g(pw->mu);
    for (auto &p : pw->q) {
      p.fut.Wait();
      if (p.fut->GetReturnCode() != 0 && pw->sticky == 0) {
        pw->sticky = EIO;
      }
      ipc->FreeBuffer(p.buf);
    }
    pw->q.clear();
    pw->bytes = 0;
  }
}

ssize_t CfsIo::DoWrite(clio::run::u64 handle, const std::string &path,
                       clio::run::u64 off, const void *buf, size_t count,
                       bool sync) {
  if (count == 0) {
    return 0;
  }
  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> shm = ipc->AllocateBuffer(count);
  if (shm.IsNull()) {
    errno = ENOMEM;
    return -1;
  }
  // The copy is mandatory either way: the caller's buffer is theirs to reuse
  // the moment write(2) returns.
  std::memcpy(shm.ptr_, buf, count);
  ctp::ipc::ShmPtr<> shm_ptr(shm.shm_);
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncWrite(handle, off, count, shm_ptr);

  if (sync || !AsyncWritesEnabled()) {
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

  // Asynchronous: hand the task off and return. The bytes are already in
  // shared memory and the chimod task is queued, so ordering against later
  // operations on this file is the runtime's to keep.
  auto pw = PendingFor(path, /*create=*/true);

  // ONE OUTSTANDING WRITE PER PAGE. Two PutBlobs to the same page blob
  // serialize on the runtime's per-blob write token (issue #680), and the
  // contender does not block -- it busy-polls via `co_await yield()` on a
  // ~2 ms cadence. So overlapping two writes to one page does not pipeline
  // them, it adds milliseconds. Measured on 64 KiB sequential writes (16 to a
  // page): unrestricted queuing was 1.7-3.3x SLOWER than blocking writes, and
  // it got worse the deeper the window.
  //
  // Waiting here keeps the concurrency exactly where the runtime can actually
  // exploit it -- across DIFFERENT pages -- and makes same-page writes no
  // worse than the synchronous path they replace.
  WaitForPageOverlap(pw.get(), off, count);

  {
    std::lock_guard<std::mutex> g(pw->mu);
    PendingWrite p;
    p.fut = t;
    p.buf = shm;
    p.off = off;
    p.size = count;
    pw->q.push_back(std::move(p));
    pw->bytes += count;
  }
  // Reap + enforce the window AFTER accounting, so this write is included in
  // the bound rather than being the one that overshoots it.
  int err = ReapAndBound(pw.get());
  if (err != 0) {
    // A previously queued write failed. Report it here -- POSIX writeback
    // semantics: the error surfaces on a later write/fsync/close, not on the
    // call that queued the doomed one (which had already returned success).
    {
      std::lock_guard<std::mutex> g(pw->mu);
      pw->sticky = 0;
    }
    errno = err;
    return -1;
  }
  return static_cast<ssize_t>(count);
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
  std::string path;
  int flags;
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
    flags = it->second.flags;
  }
  ssize_t n = DoWrite(handle, path, off, buf, count, IsSyncFd(flags));
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
  std::string path;
  int flags;
  {
    std::lock_guard<std::mutex> g(mu_);
    auto it = fds_.find(fd);
    if (it == fds_.end()) {
      errno = EBADF;
      return -1;
    }
    handle = it->second.handle;
    path = it->second.path;
    flags = it->second.flags;
  }
  return DoWrite(handle, path, static_cast<clio::run::u64>(offset), buf, count,
                 IsSyncFd(flags));
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
      DrainPath(path);  // EOF must include queued writes (issue #817)
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
  // Queued writes have not advanced the logical size yet (issue #817).
  DrainPath(path);
  clio::run::u64 size = 0;
  if (!QuerySize(path, &size)) {
    return -1;
  }
  return static_cast<off_t>(size);
}

int CfsIo::Sync(int fd) {
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
  // Wait for every queued write on this file and report a latched failure
  // exactly once. Before #817 every write blocked, so this was a no-op.
  int err = DrainPath(path);
  if (err != 0) {
    errno = err;
    return -1;
  }
  return 0;
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
  // Order matters: a queued write that landed AFTER the truncate would undo
  // it, so drain before resizing (issue #817).
  DrainPath(path);
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
  // Drain first: a write still queued against a deleted path would resurrect
  // the tag. Then drop the window so the path's entry does not leak.
  DrainPath(path);
  {
    std::lock_guard<std::mutex> g(pw_mu_);
    pending_writes_.erase(path);
  }
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
  // Drain BEFORE releasing the chimod handle: a queued write names that
  // handle, and the runtime answers EBADF once it is gone (issue #817).
  // close(2) is also the last chance to report a latched write failure, which
  // is why its return code is not simply the Close task's.
  int werr = DrainPath(path);
  ForgetIfIdle(path);
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncClose(handle);
  t.Wait();
  if (werr != 0) {
    errno = werr;
    return -1;
  }
  return (t->GetReturnCode() == 0) ? 0 : -1;
}

}  // namespace clio::cae

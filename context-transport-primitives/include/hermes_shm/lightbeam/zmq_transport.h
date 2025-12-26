#pragma once
#if HSHM_ENABLE_ZMQ
#include <unistd.h>
#include <zmq.h>

#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

#include "hermes_shm/util/logging.h"
#include "lightbeam.h"

// Cereal serialization for Bulk
// Note: data is transferred separately via bulk transfer mechanism, not
// serialized here
namespace cereal {
template <class Archive>
void serialize(Archive& ar, hshm::lbm::Bulk& bulk) {
  ar(bulk.size, bulk.flags);
}

template <class Archive>
void serialize(Archive& ar, hshm::lbm::LbmMeta& meta) {
  ar(meta.send, meta.recv);
}
}  // namespace cereal

namespace hshm::lbm {

// Lightbeam context flags for Send operations
constexpr uint32_t LBM_SYNC =
    0x1; /**< Synchronous send (wait for completion) */

/**
 * Context for lightbeam Send operations
 * Controls send behavior (sync vs async)
 */
struct LbmContext {
  uint32_t flags; /**< Combination of LBM_* flags */

  LbmContext() : flags(0) {}

  explicit LbmContext(uint32_t f) : flags(f) {}

  bool IsSync() const { return flags & LBM_SYNC; }
};

class ZeroMqClient : public Client {
 private:
  /**
   * Get or create the shared ZeroMQ context for all clients
   * Uses a static local variable for thread-safe singleton initialization
   */
  static void* GetSharedContext() {
    static void* shared_ctx = nullptr;
    static std::mutex ctx_mutex;

    std::lock_guard<std::mutex> lock(ctx_mutex);
    if (!shared_ctx) {
      shared_ctx = zmq_ctx_new();
      // Set I/O threads to 2 for better throughput
      zmq_ctx_set(shared_ctx, ZMQ_IO_THREADS, 2);
      HLOG(kInfo, "[ZeroMqClient] Created shared context with 2 I/O threads");
    }
    return shared_ctx;
  }

 public:
  explicit ZeroMqClient(const std::string& addr,
                        const std::string& protocol = "tcp", int port = 8192)
      : addr_(addr),
        protocol_(protocol),
        port_(port),
        ctx_(GetSharedContext()),
        owns_ctx_(false),
        socket_(zmq_socket(ctx_, ZMQ_PUSH)) {
    std::string full_url =
        protocol_ + "://" + addr_ + ":" + std::to_string(port_);
    HLOG(kDebug, "ZeroMqClient connecting to URL: {}", full_url);

    // Disable ZMQ_IMMEDIATE - let messages queue until connection is
    // established With ZMQ_IMMEDIATE=1, messages may be dropped if no peer is
    // immediately available
    int immediate = 0;
    zmq_setsockopt(socket_, ZMQ_IMMEDIATE, &immediate, sizeof(immediate));

    // Set a reasonable send timeout (5 seconds)
    int timeout = 5000;
    zmq_setsockopt(socket_, ZMQ_SNDTIMEO, &timeout, sizeof(timeout));

    int rc = zmq_connect(socket_, full_url.c_str());
    if (rc == -1) {
      std::string err = "ZeroMqClient failed to connect to URL '" + full_url +
                        "': " + zmq_strerror(zmq_errno());
      zmq_close(socket_);
      throw std::runtime_error(err);
    }

    // Wait for socket to become writable (connection established)
    // zmq_connect is asynchronous, so we use poll to verify readiness
    zmq_pollitem_t poll_item = {socket_, 0, ZMQ_POLLOUT, 0};
    int poll_timeout_ms = 5000;  // 5 second timeout for connection
    int poll_rc = zmq_poll(&poll_item, 1, poll_timeout_ms);

    if (poll_rc < 0) {
      HLOG(kError, "[ZeroMqClient] Poll failed for {}: {}", full_url,
           zmq_strerror(zmq_errno()));
    } else if (poll_rc == 0) {
      HLOG(kWarning,
           "[ZeroMqClient] Poll timeout - connection to {} may not be ready",
           full_url);
    } else if (poll_item.revents & ZMQ_POLLOUT) {
      HLOG(kInfo, "[ZeroMqClient] Socket ready for writing to {}", full_url);
    }

    HLOG(kDebug, "ZeroMqClient connected to {} (poll_rc={})", full_url,
         poll_rc);
  }

  ~ZeroMqClient() override {
    HLOG(kDebug, "ZeroMqClient destructor - closing socket to {}:{}", addr_,
         port_);

    // Set linger to ensure any remaining messages are sent
    int linger = 5000;
    zmq_setsockopt(socket_, ZMQ_LINGER, &linger, sizeof(linger));

    zmq_close(socket_);
    // Don't destroy the shared context - it's shared across all clients
    HLOG(kDebug, "ZeroMqClient destructor - socket closed");
  }

  // Base Expose implementation - accepts hipc::FullPtr
  Bulk Expose(const hipc::FullPtr<char>& ptr, size_t data_size,
              u32 flags) override {
    Bulk bulk;
    bulk.data = ptr;
    bulk.size = data_size;
    bulk.flags = hshm::bitfield32_t(flags);
    return bulk;
  }

  template <typename MetaT>
  int Send(MetaT& meta, const LbmContext& ctx = LbmContext()) {
    HLOG(kDebug, "ZeroMqClient::Send - START to {}:{}", addr_, port_);

    // Serialize metadata (includes both send and recv vectors)
    std::ostringstream oss(std::ios::binary);
    {
      cereal::BinaryOutputArchive ar(oss);
      ar(meta);
    }
    std::string meta_str = oss.str();
    HLOG(kDebug, "ZeroMqClient::Send - serialized metadata size={}",
         meta_str.size());

    // Count bulks marked for WRITE
    size_t write_bulk_count = 0;
    for (const auto& bulk : meta.send) {
      if (bulk.flags.Any(BULK_XFER)) {
        write_bulk_count++;
      }
    }

    // IMPORTANT: Always use blocking send for distributed messaging
    // ZMQ_DONTWAIT with newly-created connections causes messages to be lost
    // because the connection may not be established when send is called
    int base_flags = 0;  // Use blocking sends

    HLOG(kInfo, "ZeroMqClient::Send - write_bulk_count={}, base_flags={}",
         write_bulk_count, base_flags);

    // Send metadata - use ZMQ_SNDMORE only if there are WRITE bulks to follow
    int flags = base_flags;
    if (write_bulk_count > 0) {
      flags |= ZMQ_SNDMORE;
    }

    int rc = zmq_send(socket_, meta_str.data(), meta_str.size(), flags);
    HLOG(kDebug, "ZeroMqClient::Send - zmq_send metadata rc={}, errno={}", rc,
         rc == -1 ? zmq_errno() : 0);
    if (rc == -1) {
      HLOG(kDebug, "ZeroMqClient::Send - FAILED: {}",
           zmq_strerror(zmq_errno()));
      return zmq_errno();
    }

    // Send only bulks marked with BULK_XFER
    size_t sent_count = 0;
    for (size_t i = 0; i < meta.send.size(); ++i) {
      if (!meta.send[i].flags.Any(BULK_XFER)) {
        continue;  // Skip bulks not marked for WRITE
      }

      flags = base_flags;
      sent_count++;
      if (sent_count < write_bulk_count) {
        flags |= ZMQ_SNDMORE;
      }

      rc = zmq_send(socket_, meta.send[i].data.ptr_, meta.send[i].size, flags);
      if (rc == -1) {
        HLOG(kDebug, "ZeroMqClient::Send - bulk {} FAILED: {}", i,
             zmq_strerror(zmq_errno()));
        return zmq_errno();
      }
    }

    HLOG(kDebug, "ZeroMqClient::Send - SUCCESS to {}:{}", addr_, port_);

    // Give TCP stack time to transmit the message before socket is destroyed.
    // This is a diagnostic workaround - proper fix would be to reuse
    // connections.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    HLOG(kDebug, "ZeroMqClient::Send - post-send delay completed");

    return 0;  // Success
  }

 private:
  std::string addr_;
  std::string protocol_;
  int port_;
  void* ctx_;
  bool owns_ctx_;  // Whether this client owns the context (should destroy on
                   // cleanup)
  void* socket_;
};

class ZeroMqServer : public Server {
 public:
  explicit ZeroMqServer(const std::string& addr,
                        const std::string& protocol = "tcp", int port = 8192)
      : addr_(addr),
        protocol_(protocol),
        port_(port),
        ctx_(zmq_ctx_new()),
        socket_(zmq_socket(ctx_, ZMQ_PULL)) {
    std::string full_url =
        protocol_ + "://" + addr_ + ":" + std::to_string(port_);
    HLOG(kDebug, "ZeroMqServer binding to URL: {}", full_url);
    int rc = zmq_bind(socket_, full_url.c_str());
    if (rc == -1) {
      std::string err = "ZeroMqServer failed to bind to URL '" + full_url +
                        "': " + zmq_strerror(zmq_errno());
      zmq_close(socket_);
      zmq_ctx_destroy(ctx_);
      throw std::runtime_error(err);
    }
    HLOG(kDebug, "ZeroMqServer bound successfully to {} (socket={})", full_url,
         reinterpret_cast<uintptr_t>(socket_));
  }

  ~ZeroMqServer() override {
    zmq_close(socket_);
    zmq_ctx_destroy(ctx_);
  }

  // Base Expose implementation - accepts hipc::FullPtr
  Bulk Expose(const hipc::FullPtr<char>& ptr, size_t data_size,
              u32 flags) override {
    Bulk bulk;
    bulk.data = ptr;
    bulk.size = data_size;
    bulk.flags = hshm::bitfield32_t(flags);
    return bulk;
  }

  template <typename MetaT>
  int RecvMetadata(MetaT& meta, bool* has_more_parts = nullptr) {
    static thread_local int recv_attempt_count = 0;
    recv_attempt_count++;

    // Check socket events periodically for diagnostics
    if (recv_attempt_count % 1000 == 1) {
      int events = 0;
      size_t events_size = sizeof(events);
      zmq_getsockopt(socket_, ZMQ_EVENTS, &events, &events_size);
      HLOG(kDebug,
           "ZeroMqServer::RecvMetadata - ZMQ_EVENTS={} "
           "(POLLIN={}, POLLOUT={}), attempt={}, socket={}",
           events, (events & ZMQ_POLLIN) ? 1 : 0,
           (events & ZMQ_POLLOUT) ? 1 : 0, recv_attempt_count,
           reinterpret_cast<uintptr_t>(socket_));
    }

    // Receive metadata message (non-blocking)
    zmq_msg_t msg;
    zmq_msg_init(&msg);
    int rc = zmq_msg_recv(&msg, socket_, ZMQ_DONTWAIT);

    if (rc == -1) {
      int err = zmq_errno();
      zmq_msg_close(&msg);
      // Only log every 1000th EAGAIN to avoid spam
      if (err != EAGAIN || recv_attempt_count % 1000 == 0) {
        HLOG(kInfo,
             "ZeroMqServer::RecvMetadata - err={} ({}), attempt={}, socket={}",
             err, zmq_strerror(err), recv_attempt_count,
             reinterpret_cast<uintptr_t>(socket_));
      }
      return err;
    }

    HLOG(kInfo,
         "ZeroMqServer::RecvMetadata - RECEIVED message! size={}, "
         "attempt={}",
         zmq_msg_size(&msg), recv_attempt_count);

    // Check if there are more message parts (bulk data following metadata)
    int more = 0;
    size_t more_size = sizeof(more);
    zmq_getsockopt(socket_, ZMQ_RCVMORE, &more, &more_size);
    if (has_more_parts) {
      *has_more_parts = (more != 0);
    }

    // Deserialize metadata
    size_t msg_size = zmq_msg_size(&msg);
    try {
      std::string meta_str(static_cast<char*>(zmq_msg_data(&msg)), msg_size);
      std::istringstream iss(meta_str, std::ios::binary);
      cereal::BinaryInputArchive ar(iss);
      ar(meta);
    } catch (const std::exception& e) {
      // If deserialization fails, this might be bulk data from a previous
      // incomplete receive or a spurious epoll trigger. Discard and return.
      if (more) {
        // Multi-part message with stale data - discard remaining parts
        HLOG(kDebug,
             "ZeroMQ RecvMetadata: Discarding stale multi-part message "
             "(msg_size={}, has_more=true)",
             msg_size);
        DiscardRemainingParts();
      } else if (msg_size > 10000) {
        // Large standalone message (likely bulk data) - log as debug
        HLOG(kDebug,
             "ZeroMQ RecvMetadata: Received large non-metadata message "
             "(msg_size={}), skipping",
             msg_size);
      } else {
        // Small message that failed to parse - this is a real error
        HLOG(kError,
             "ZeroMQ RecvMetadata: Deserialization failed - {} (msg_size={})",
             e.what(), msg_size);
      }
      zmq_msg_close(&msg);
      return -1;  // Deserialization error
    }

    zmq_msg_close(&msg);
    return 0;  // Success
  }

  /**
   * Discard remaining parts of a multi-part message
   * Used to re-synchronize after a failed receive
   */
  void DiscardRemainingParts() {
    int more = 1;
    while (more) {
      zmq_msg_t msg;
      zmq_msg_init(&msg);
      int rc = zmq_msg_recv(&msg, socket_, 0);
      if (rc == -1) {
        zmq_msg_close(&msg);
        break;
      }
      size_t more_size = sizeof(more);
      zmq_getsockopt(socket_, ZMQ_RCVMORE, &more, &more_size);
      zmq_msg_close(&msg);
    }
  }

  template <typename MetaT>
  int RecvBulks(MetaT& meta) {
    // Count bulks marked with BULK_XFER (only these will be received)
    size_t write_bulk_count = 0;
    for (const auto& bulk : meta.recv) {
      if (bulk.flags.Any(BULK_XFER)) {
        write_bulk_count++;
      }
    }

    // If no WRITE bulks, return immediately
    if (write_bulk_count == 0) {
      return 0;
    }

    // Receive only bulks marked with BULK_XFER
    size_t recv_count = 0;
    for (size_t i = 0; i < meta.recv.size(); ++i) {
      if (!meta.recv[i].flags.Any(BULK_XFER)) {
        continue;  // Skip bulks not marked for WRITE
      }

      int rc = zmq_recv(socket_, meta.recv[i].data.ptr_, meta.recv[i].size, 0);
      if (rc == -1) {
        return zmq_errno();
      }
      recv_count++;

      // Check if there are more message parts
      int more = 0;
      size_t more_size = sizeof(more);
      zmq_getsockopt(socket_, ZMQ_RCVMORE, &more, &more_size);

      // If this is the last expected WRITE bulk but more parts exist, it's an
      // error
      if (recv_count == write_bulk_count && more) {
        return -1;  // More parts than expected
      }

      // If we expect more WRITE bulks but no more parts, it's incomplete
      if (recv_count < write_bulk_count && !more) {
        return -1;  // Fewer parts than expected
      }
    }

    return 0;  // Success
  }

  std::string GetAddress() const override { return addr_; }

  /**
   * Get the file descriptor for the ZeroMQ socket
   * Can be used with epoll for efficient event-driven I/O
   * @return File descriptor for the socket
   */
  int GetFd() const {
    int fd;
    size_t fd_size = sizeof(fd);
    zmq_getsockopt(socket_, ZMQ_FD, &fd, &fd_size);
    return fd;
  }

  /**
   * Lock the socket for exclusive access during multi-part receive
   * Use this when you need to perform a multi-part receive (RecvMetadata +
   * RecvBulks)
   * @return A lock guard that will unlock automatically when destroyed
   */
  std::unique_lock<std::mutex> LockSocket() {
    return std::unique_lock<std::mutex>(socket_mutex_);
  }

 private:
  std::string addr_;
  std::string protocol_;
  int port_;
  void* ctx_;
  void* socket_;
  mutable std::mutex socket_mutex_; /**< Mutex to serialize socket access */
};

// --- Base Class Template Implementations ---
// These delegate to the derived class implementations
template <typename MetaT>
int Client::Send(MetaT& meta, const LbmContext& ctx) {
  // Forward to ZeroMqClient implementation with provided context
  return static_cast<ZeroMqClient*>(this)->Send(meta, ctx);
}

template <typename MetaT>
int Server::RecvMetadata(MetaT& meta, bool* has_more_parts) {
  return static_cast<ZeroMqServer*>(this)->RecvMetadata(meta, has_more_parts);
}

template <typename MetaT>
int Server::RecvBulks(MetaT& meta) {
  return static_cast<ZeroMqServer*>(this)->RecvBulks(meta);
}

// --- TransportFactory Implementations ---
inline std::unique_ptr<Client> TransportFactory::GetClient(
    const std::string& addr, Transport t, const std::string& protocol,
    int port) {
  if (t == Transport::kZeroMq) {
    return std::make_unique<ZeroMqClient>(addr, protocol, port);
  }
  throw std::runtime_error("Unsupported transport type");
}

inline std::unique_ptr<Client> TransportFactory::GetClient(
    const std::string& addr, Transport t, const std::string& protocol, int port,
    const std::string& domain) {
  if (t == Transport::kZeroMq) {
    return std::make_unique<ZeroMqClient>(addr, protocol, port);
  }
  throw std::runtime_error("Unsupported transport type");
}

inline std::unique_ptr<Server> TransportFactory::GetServer(
    const std::string& addr, Transport t, const std::string& protocol,
    int port) {
  if (t == Transport::kZeroMq) {
    return std::make_unique<ZeroMqServer>(addr, protocol, port);
  }
  throw std::runtime_error("Unsupported transport type");
}

inline std::unique_ptr<Server> TransportFactory::GetServer(
    const std::string& addr, Transport t, const std::string& protocol, int port,
    const std::string& domain) {
  if (t == Transport::kZeroMq) {
    return std::make_unique<ZeroMqServer>(addr, protocol, port);
  }
  throw std::runtime_error("Unsupported transport type");
}

}  // namespace hshm::lbm

#endif  // HSHM_ENABLE_ZMQ
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

#ifndef HSHM_MEMORY_ALLOCATOR_ALLOCATOR_H_
#define HSHM_MEMORY_ALLOCATOR_ALLOCATOR_H_

#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/memory/backend/memory_backend.h"
#include "hermes_shm/thread/thread_model/thread_model.h"
#include "hermes_shm/types/atomic.h"
#include "hermes_shm/types/bitfield.h"
#include "hermes_shm/types/numbers.h"
#include "hermes_shm/types/real_number.h"
#include "hermes_shm/util/errors.h"

namespace hshm::ipc {

/**
 * The identifier for an allocator
 * */
union AllocatorId {
  struct {
    i32 major_;  // Typically some sort of process id
    i32 minor_;  // Typically a process-local id
  } bits_;
  u64 int_;

  HSHM_INLINE_CROSS_FUN AllocatorId() = default;

  /**
   * Constructor which sets major & minor
   * */
  HSHM_INLINE_CROSS_FUN explicit AllocatorId(i32 major, i32 minor) {
    bits_.major_ = major;
    bits_.minor_ = minor;
  }

  /**
   * Set this allocator to null
   * */
  HSHM_INLINE_CROSS_FUN void SetNull() { (*this) = GetNull(); }

  /**
   * Check if this is the null allocator
   * */
  HSHM_INLINE_CROSS_FUN bool IsNull() const { return (*this) == GetNull(); }

  /** Equality check */
  HSHM_INLINE_CROSS_FUN bool operator==(const AllocatorId &other) const {
    return other.int_ == int_;
  }

  /** Inequality check */
  HSHM_INLINE_CROSS_FUN bool operator!=(const AllocatorId &other) const {
    return other.int_ != int_;
  }

  /** Get the null allocator */
  HSHM_INLINE_CROSS_FUN static AllocatorId GetNull() {
    return AllocatorId(-1, -1);
  }

  /** To index */
  HSHM_INLINE_CROSS_FUN uint32_t ToIndex() const {
    return bits_.major_ * 2 + bits_.minor_;
  }

  /** Serialize an hipc::allocator_id */
  template <typename Ar>
  HSHM_INLINE_CROSS_FUN void serialize(Ar &ar) {
    ar & int_;
  }

  /** Print */
  HSHM_CROSS_FUN
  void Print() const {
    printf("(%s) Allocator ID: %u.%u\n", kCurrentDevice, bits_.major_,
           bits_.minor_);
  }

  friend std::ostream &operator<<(std::ostream &os, const AllocatorId &id) {
    os << id.bits_.major_ << "." << id.bits_.minor_;
    return os;
  }
};

class Allocator;

/** Pointer type base */
class ShmPointer {};

/**
 * The basic shared-memory allocator header.
 * Allocators inherit from this.
 * */
struct AllocatorHeader {
  AllocatorId alloc_id_;
  size_t custom_header_size_;
  hipc::atomic<hshm::size_t> total_alloc_;

  HSHM_CROSS_FUN
  AllocatorHeader() = default;

  HSHM_CROSS_FUN
  void Configure(AllocatorId allocator_id, size_t custom_header_size) {
    alloc_id_ = allocator_id;
    custom_header_size_ = custom_header_size;
    total_alloc_ = 0;
  }

  HSHM_INLINE_CROSS_FUN
  void AddSize(hshm::size_t size) {
#ifdef HSHM_ALLOC_TRACK_SIZE
    total_alloc_ += size;
#endif
  }

  HSHM_INLINE_CROSS_FUN
  void SubSize(hshm::size_t size) {
#ifdef HSHM_ALLOC_TRACK_SIZE
    total_alloc_ -= size;
#endif
  }

  HSHM_INLINE_CROSS_FUN
  hshm::size_t GetCurrentlyAllocatedSize() { return total_alloc_.load(); }
};

/** Memory context */
class MemContext {
 public:
  ThreadId tid_ = ThreadId::GetNull();

 public:
  /** Default constructor */
  HSHM_INLINE_CROSS_FUN
  MemContext() = default;

  /** Constructor */
  HSHM_INLINE_CROSS_FUN
  MemContext(const ThreadId &tid) : tid_(tid) {}
};

/** The allocator information struct */
class Allocator {
 public:
  AllocatorId id_;
  MemoryBackend backend_;
  char *buffer_;
  size_t buffer_size_;
  char *custom_header_;

 public:
  /** Default constructor */
  HSHM_INLINE_CROSS_FUN
  Allocator() : custom_header_(nullptr) {}

  /** Get the allocator identifier */
  HSHM_INLINE_CROSS_FUN
  AllocatorId &GetId() { return id_; }

  /** Get the allocator identifier (const) */
  HSHM_INLINE_CROSS_FUN
  const AllocatorId &GetId() const { return id_; }

  /**
   * Construct custom header
   */
  template <typename HEADER_T>
  HSHM_INLINE_CROSS_FUN HEADER_T *ConstructHeader(void *buffer) {
    new ((HEADER_T *)buffer) HEADER_T();
    return reinterpret_cast<HEADER_T *>(buffer);
  }

  /**
   * Get the custom header of the shared-memory allocator
   *
   * @return Custom header pointer
   * */
  template <typename HEADER_T>
  HSHM_INLINE_CROSS_FUN HEADER_T *GetCustomHeader() {
    return reinterpret_cast<HEADER_T *>(custom_header_);
  }

  /**
   * Determine whether or not this allocator contains a process-specific
   * pointer
   *
   * @param ptr process-specific pointer
   * @return True or false
   * */
  template <typename T = void>
  HSHM_INLINE_CROSS_FUN bool ContainsPtr(const T *ptr) {
    return reinterpret_cast<size_t>(buffer_) <= reinterpret_cast<size_t>(ptr) &&
           reinterpret_cast<size_t>(ptr) <
               reinterpret_cast<size_t>(buffer_) + buffer_size_;
  }

  template<bool ATOMIC>
  HSHM_INLINE_CROSS_FUN bool ContainsPtr(const OffsetPointerBase<ATOMIC> &ptr) const {
    return ptr.off_.load() < buffer_size_;
  }

  template<bool ATOMIC>
  HSHM_INLINE_CROSS_FUN bool ContainsPtr(const PointerBase<ATOMIC> &ptr) const {
    return ptr.off_.load() < buffer_size_;
  }

  /** Print */
  HSHM_CROSS_FUN
  void Print() {
    printf("(%s) Allocator: id: %d.%d, custom_header: %p\n",
           kCurrentDevice, GetId().bits_.major_,
           GetId().bits_.minor_, custom_header_);
  }

  /**====================================
   * Object Constructors
   * ===================================*/

  /**
   * Construct each object in an array of objects.
   *
   * @param ptr the array of objects (potentially archived)
   * @param old_count the original size of the ptr
   * @param new_count the new size of the ptr
   * @param args parameters to construct object of type T
   * @return None
   * */
  template <typename T, typename... Args>
  HSHM_INLINE_CROSS_FUN static void ConstructObjs(T *ptr, size_t old_count,
                                                  size_t new_count,
                                                  Args &&...args) {
    if (ptr == nullptr) {
      return;
    }
    for (size_t i = old_count; i < new_count; ++i) {
      ConstructObj<T>(*(ptr + i), std::forward<Args>(args)...);
    }
  }

  /**
   * Construct an object.
   *
   * @param ptr the object to construct (potentially archived)
   * @param args parameters to construct object of type T
   * @return None
   * */
  template <typename T, typename... Args>
  HSHM_INLINE_CROSS_FUN static void ConstructObj(T &obj, Args &&...args) {
    new (&obj) T(std::forward<Args>(args)...);
  }

  /**
   * Destruct an array of objects
   *
   * @param ptr the object to destruct (potentially archived)
   * @param count the length of the object array
   * @return None
   * */
  template <typename T>
  HSHM_INLINE_CROSS_FUN static void DestructObjs(T *ptr, size_t count) {
    if (ptr == nullptr) {
      return;
    }
    for (size_t i = 0; i < count; ++i) {
      DestructObj<T>(*(ptr + i));
    }
  }

  /**
   * Destruct an object
   *
   * @param ptr the object to destruct (potentially archived)
   * @param count the length of the object array
   * @return None
   * */
  template <typename T>
  HSHM_INLINE_CROSS_FUN static void DestructObj(T &obj) {
    obj.~T();
  }
};

/**
 * The allocator base class.
 * */
template <typename CoreAllocT>
class BaseAllocator : public CoreAllocT {
 public:
  /**====================================
   * Constructors
   * ===================================*/
  /**
   * Create the shared-memory allocator with \a id unique allocator id over
   * the particular slot of a memory backend.
   *
   * The shm_init function is required, but cannot be marked virtual as
   * each allocator has its own arguments to this method. Though each
   * allocator must have "id" as its first argument.
   * */
  template <typename... Args>
  HSHM_CROSS_FUN void shm_init(AllocatorId id, Args... args) {
    CoreAllocT::shm_init(id, std::forward<Args>(args)...);
  }

  /**
   * Deserialize allocator from a buffer.
   * */
  HSHM_CROSS_FUN
  void shm_deserialize(const MemoryBackend &backend) {
    CoreAllocT::shm_deserialize(backend);
  }

  /**====================================
   * Core Allocator API
   * ===================================*/
 public:
  /**
   * Allocate a region of memory of \a size size
   * */
  HSHM_CROSS_FUN
  OffsetPointer AllocateOffset(const MemContext &ctx, size_t size) {
    return CoreAllocT::AllocateOffset(ctx, size);
  }

  /**
   * Allocate a region of memory of \a size size
   * and \a alignment alignment. Assumes that
   * alignment is not 0.
   * */
  HSHM_CROSS_FUN
  OffsetPointer AlignedAllocateOffset(const MemContext &ctx, size_t size,
                                      size_t alignment) {
    return CoreAllocT::AlignedAllocateOffset(ctx, size, alignment);
  }

  /**
   * Reallocate \a pointer to \a new_size new size.
   * Assumes that p is not kNulFullPtr.
   *
   * @return true if p was modified.
   * */
  HSHM_CROSS_FUN
  OffsetPointer ReallocateOffsetNoNullCheck(const MemContext &ctx,
                                            OffsetPointer p, size_t new_size) {
    return CoreAllocT::ReallocateOffsetNoNullCheck(ctx, p, new_size);
  }

  /**
   * Free the memory pointed to by \a ptr Pointer
   * */
  HSHM_CROSS_FUN
  void FreeOffsetNoNullCheck(const MemContext &ctx, OffsetPointer p) {
    CoreAllocT::FreeOffsetNoNullCheck(ctx, p);
  }

  /**
   * Create a thread-local storage segment. This storage
   * is unique even across processes.
   * */
  HSHM_CROSS_FUN
  void CreateTls(MemContext &ctx) { CoreAllocT::CreateTls(ctx); }

  /**
   * Free a thread-local storage segment.
   * */
  HSHM_CROSS_FUN
  void FreeTls(const MemContext &ctx) { CoreAllocT::FreeTls(ctx); }

  /** Get the allocator identifier */
  HSHM_INLINE_CROSS_FUN
  AllocatorId &GetId() { return CoreAllocT::GetId(); }

  /** Get the allocator identifier (const) */
  HSHM_INLINE_CROSS_FUN
  const AllocatorId &GetId() const { return CoreAllocT::GetId(); }

  /**
   * Get the amount of memory that was allocated, but not yet freed.
   * Useful for memory leak checks.
   * */
  HSHM_CROSS_FUN
  size_t GetCurrentlyAllocatedSize() {
    return CoreAllocT::GetCurrentlyAllocatedSize();
  }

  /**====================================
   * SHM Pointer Allocator
   * ===================================*/
 public:
  /**
   * Allocate a region of memory to a specific pointer type
   * */
  template <typename T = void, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN FullPtr<T, PointerT> Allocate(const MemContext &ctx, size_t size) {
    FullPtr<T, PointerT> result;
    result.shm_ = PointerT(GetId(), AllocateOffset(ctx, size).load());
    result.ptr_ = reinterpret_cast<T*>(CoreAllocT::buffer_ + result.shm_.off_.load());
    return result;
  }

  /**
   * Allocate a region of memory to a specific pointer type
   * */
  template <typename T = void, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN FullPtr<T, PointerT> AlignedAllocate(const MemContext &ctx,
                                                 size_t size,
                                                 size_t alignment) {
    FullPtr<T, PointerT> result;
    result.shm_ = PointerT(GetId(),
                    AlignedAllocateOffset(ctx, size, alignment).load());
    result.ptr_ = reinterpret_cast<T*>(CoreAllocT::buffer_ + result.shm_.off_.load());
    return result;
  }

  /**
   * Allocate a region of \a size size and \a alignment
   * alignment. Will fall back to regular Allocate if
   * alignmnet is 0.
   * */
  template <typename T = void, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN FullPtr<T, PointerT> Allocate(const MemContext &ctx, size_t size,
                                          size_t alignment) {
    if (alignment == 0) {
      return Allocate<T, PointerT>(ctx, size);
    } else {
      return AlignedAllocate<T, PointerT>(ctx, size, alignment);
    }
  }

  /**
   * Reallocate \a pointer to \a new_size new size
   * If p is kNulFullPtr, will internally call Allocate.
   *
   * @return the reallocated FullPtr.
   * */
  template <typename T = void, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN FullPtr<T, PointerT> Reallocate(const MemContext &ctx, const FullPtr<T, PointerT> &p,
                                        size_t new_size) {
    if (p.IsNull()) {
      return Allocate<T, PointerT>(ctx, new_size);
    }
    auto new_off =
        ReallocateOffsetNoNullCheck(ctx, p.shm_.ToOffsetPointer(), new_size);
    FullPtr<T, PointerT> result;
    result.shm_ = PointerT(GetId(), new_off.load());
    result.ptr_ = reinterpret_cast<T*>(CoreAllocT::buffer_ + result.shm_.off_.load());
    return result;
  }

  /**
   * Free the memory pointed to by \a p Pointer
   * */
  template <typename T = void, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN void Free(const MemContext &ctx, const FullPtr<T, PointerT> &p) {
    if (p.IsNull()) {
      HSHM_THROW_ERROR(INVALID_FREE);
    }
    FreeOffsetNoNullCheck(ctx, OffsetPointer(p.shm_.off_.load()));
  }





  /**====================================
   * Private Object Allocators
   * ===================================*/

  /**
   * Allocate an array of objects (but don't construct).
   *
   * @return A FullPtr to the allocated memory
   * */
  template <typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN FullPtr<T, PointerT> AllocateObjs(const MemContext &ctx, size_t count) {
    return Allocate<T, PointerT>(ctx, count * sizeof(T));
  }

  /** Allocate + construct an array of objects */
  template <typename T, typename... Args>
  HSHM_INLINE_CROSS_FUN FullPtr<T> NewObjs(const MemContext &ctx, size_t count,
                                   Args &&...args) {
    auto alloc_result = AllocateObjs<T, Pointer>(ctx, count);
    ConstructObjs<T>(alloc_result.ptr_, 0, count, std::forward<Args>(args)...);
    return alloc_result;
  }

  /** Allocate + construct a single object */
  template <typename T, typename... Args>
  HSHM_INLINE_CROSS_FUN FullPtr<T> NewObj(const MemContext &ctx, Args &&...args) {
    return NewObjs<T>(ctx, 1, std::forward<Args>(args)...);
  }

  /**
   * Reallocate a pointer of objects to a new size.
   *
   * @param p FullPtr to reallocate (input & output)
   * @param new_count the new number of objects
   *
   * @return A FullPtr to the reallocated objects
   * */
  template <typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN FullPtr<T, PointerT> ReallocateObjs(const MemContext &ctx, 
                                                             FullPtr<T, PointerT> &p,
                                                             size_t new_count) {
    FullPtr<void, PointerT> old_full_ptr(reinterpret_cast<void*>(p.ptr_), p.shm_);
    auto new_full_ptr = Reallocate<void, PointerT>(ctx, old_full_ptr, new_count * sizeof(T));
    p.shm_ = new_full_ptr.shm_;
    p.ptr_ = reinterpret_cast<T*>(new_full_ptr.ptr_);
    return p;
  }

  /**
   * Free + destruct objects
   * */
  template <typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN void DelObjs(const MemContext &ctx, 
                                     FullPtr<T, PointerT> &p,
                                     size_t count) {
    DestructObjs<T>(p.ptr_, count);
    FullPtr<void, PointerT> void_ptr(reinterpret_cast<void*>(p.ptr_), p.shm_);
    Free<void, PointerT>(ctx, void_ptr);
  }

  /**
   * Free + destruct an object
   * */
  template <typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN void DelObj(const MemContext &ctx, 
                                    FullPtr<T, PointerT> &p) {
    DelObjs<T, PointerT>(ctx, p, 1);
  }


  /**====================================
   * Object Constructors
   * ===================================*/

  /**
   * Construct each object in an array of objects.
   *
   * @param ptr the array of objects (potentially archived)
   * @param old_count the original size of the ptr
   * @param new_count the new size of the ptr
   * @param args parameters to construct object of type T
   * @return None
   * */
  template <typename T, typename... Args>
  HSHM_INLINE_CROSS_FUN static void ConstructObjs(T *ptr, size_t old_count,
                                                  size_t new_count,
                                                  Args &&...args) {
    CoreAllocT::template ConstructObjs<T>(ptr, old_count, new_count,
                                          std::forward<Args>(args)...);
  }

  /**
   * Construct an object.
   *
   * @param ptr the object to construct (potentially archived)
   * @param args parameters to construct object of type T
   * @return None
   * */
  template <typename T, typename... Args>
  HSHM_INLINE_CROSS_FUN static void ConstructObj(T &obj, Args &&...args) {
    CoreAllocT::template ConstructObj<T>(obj, std::forward<Args>(args)...);
  }

  /**
   * Destruct an array of objects
   *
   * @param ptr the object to destruct (potentially archived)
   * @param count the length of the object array
   * @return None
   * */
  template <typename T>
  HSHM_INLINE_CROSS_FUN static void DestructObjs(T *ptr, size_t count) {
    CoreAllocT::template DestructObjs<T>(ptr, count);
  }

  /**
   * Destruct an object
   *
   * @param ptr the object to destruct (potentially archived)
   * @param count the length of the object array
   * @return None
   * */
  template <typename T>
  HSHM_INLINE_CROSS_FUN static void DestructObj(T &obj) {
    CoreAllocT::template DestructObj<T>(obj);
  }

  /**====================================
   * Helpers
   * ===================================*/

  /**
   * Get the custom header of the shared-memory allocator
   *
   * @return Custom header pointer
   * */
  template <typename HEADER_T>
  HSHM_INLINE_CROSS_FUN HEADER_T *GetCustomHeader() {
    return CoreAllocT::template GetCustomHeader<HEADER_T>();
  }

  /**
   * Determine whether or not this allocator contains a process-specific
   * pointer
   *
   * @param ptr process-specific pointer
   * @return True or false
   * */
  template <typename T = void>
  HSHM_INLINE_CROSS_FUN bool ContainsPtr(const T *ptr) {
    return CoreAllocT::template ContainsPtr<T>(ptr);
  }

  /** Print */
  HSHM_CROSS_FUN
  void Print() { CoreAllocT::Print(); }
};

/** Get the full allocator within core allocator */
#define HSHM_ALLOCATOR(ALLOC_NAME)                    \
 public:                                              \
  typedef hipc::BaseAllocator<ALLOC_NAME> BaseAllocT; \
  HSHM_INLINE_CROSS_FUN                               \
  BaseAllocT *GetAllocator() { return (BaseAllocT *)(this); }

/** Demonstration allocator */
class _NullAllocator : public Allocator {
 public:
  /**====================================
   * Constructors
   * ===================================*/
  /**
   * Create the shared-memory allocator with \a id unique allocator id over
   * the particular slot of a memory backend.
   *
   * The shm_init function is required, but cannot be marked virtual as
   * each allocator has its own arguments to this method. Though each
   * allocator must have "id" as its first argument.
   * */
  HSHM_CROSS_FUN
  void shm_init(AllocatorId id, size_t custom_header_size,
                MemoryBackend backend) {
    id_ = id;
    if (backend.IsCopyGpu()) {
      buffer_ = backend.accel_data_;
      buffer_size_ = backend.accel_data_size_;
    } else {
      buffer_ = backend.data_;
      buffer_size_ = backend.data_size_;
    }
  }

  /**
   * Deserialize allocator from a buffer.
   * */
  HSHM_CROSS_FUN
  void shm_deserialize(char *buffer, size_t buffer_size) {}

  /**====================================
   * Core Allocator API
   * ===================================*/
 public:
  /**
   * Allocate a region of memory of \a size size
   * */
  HSHM_CROSS_FUN
  OffsetPointer AllocateOffset(const MemContext &ctx, size_t size) {
    return OffsetPointer::GetNull();
  }

  /**
   * Allocate a region of memory of \a size size
   * and \a alignment alignment. Assumes that
   * alignment is not 0.
   * */
  HSHM_CROSS_FUN
  OffsetPointer AlignedAllocateOffset(const MemContext &ctx, size_t size,
                                      size_t alignment) {
    return OffsetPointer::GetNull();
  }

  /**
   * Reallocate \a pointer to \a new_size new size.
   * Assumes that p is not kNulFullPtr.
   *
   * @return true if p was modified.
   * */
  HSHM_CROSS_FUN
  OffsetPointer ReallocateOffsetNoNullCheck(const MemContext &ctx,
                                            OffsetPointer p, size_t new_size) {
    return p;
  }

  /**
   * Free the memory pointed to by \a ptr Pointer
   * */
  HSHM_CROSS_FUN
  void FreeOffsetNoNullCheck(const MemContext &ctx, OffsetPointer p) {}

  /**
   * Create a globally-unique thread ID
   * */
  HSHM_CROSS_FUN
  void CreateTls(MemContext &ctx) {}

  /**
   * Free the memory pointed to by \a ptr Pointer
   * */
  HSHM_CROSS_FUN
  void FreeTls(const MemContext &ctx) {}

  /**
   * Get the amount of memory that was allocated, but not yet freed.
   * Useful for memory leak checks.
   * */
  HSHM_CROSS_FUN
  size_t GetCurrentlyAllocatedSize() { return 0; }
};
typedef BaseAllocator<_NullAllocator> NullAllocator;

/**
 * Allocator with thread-local storage identifier
 * */
template <typename AllocT>
struct CtxAllocator {
  MemContext ctx_;
  AllocT *alloc_;

  /** Default constructor */
  HSHM_INLINE_CROSS_FUN
  CtxAllocator() = default;

  /** Allocator-only constructor */
  HSHM_INLINE_CROSS_FUN
  CtxAllocator(AllocT *alloc) : alloc_(alloc), ctx_() {}

  /** Allocator and thread identifier constructor */
  HSHM_INLINE_CROSS_FUN
  CtxAllocator(AllocT *alloc, const ThreadId &tid) : alloc_(alloc), ctx_(tid) {}

  /** Allocator and thread identifier constructor */
  HSHM_INLINE_CROSS_FUN
  CtxAllocator(const ThreadId &tid, AllocT *alloc) : alloc_(alloc), ctx_(tid) {}

  /** Allocator and ctx constructor */
  HSHM_INLINE_CROSS_FUN
  CtxAllocator(const MemContext &ctx, AllocT *alloc)
      : alloc_(alloc), ctx_(ctx) {}

  /** ctx and Allocator constructor */
  HSHM_INLINE_CROSS_FUN
  CtxAllocator(AllocT *alloc, const MemContext &ctx)
      : alloc_(alloc), ctx_(ctx) {}

  /** Arrow operator */
  HSHM_INLINE_CROSS_FUN
  AllocT *operator->() { return alloc_; }

  /** Arrow operator (const) */
  HSHM_INLINE_CROSS_FUN
  AllocT *operator->() const { return alloc_; }

  /** Star operator */
  HSHM_INLINE_CROSS_FUN
  AllocT *operator*() { return alloc_; }

  /** Star operator (const) */
  HSHM_INLINE_CROSS_FUN
  AllocT *operator*() const { return alloc_; }

  /** Equality operator */
  HSHM_INLINE_CROSS_FUN
  bool operator==(const CtxAllocator &rhs) const {
    return alloc_ == rhs.alloc_;
  }

  /** Inequality operator */
  HSHM_INLINE_CROSS_FUN
  bool operator!=(const CtxAllocator &rhs) const {
    return alloc_ != rhs.alloc_;
  }
};

/**
 * Scoped Allocator (thread-local)
 * */
template <typename AllocT>
class ScopedTlsAllocator {
 public:
  CtxAllocator<AllocT> alloc_;

 public:
  HSHM_INLINE_CROSS_FUN
  ScopedTlsAllocator(const MemContext &ctx, AllocT *alloc)
      : alloc_(ctx, alloc) {
    alloc_->CreateTls(alloc_.ctx_);
  }

  HSHM_INLINE_CROSS_FUN
  ScopedTlsAllocator(const CtxAllocator<AllocT> &alloc) : alloc_(alloc) {
    alloc_->CreateTls(alloc_.ctx_);
  }

  HSHM_INLINE_CROSS_FUN
  ~ScopedTlsAllocator() { alloc_->FreeTls(alloc_.ctx_); }

  /** Arrow operator */
  HSHM_INLINE_CROSS_FUN
  CtxAllocator<AllocT> &operator->() { return alloc_; }

  /** Arrow operator (const) */
  HSHM_INLINE_CROSS_FUN
  const CtxAllocator<AllocT> &operator->() const { return alloc_; }

  /** Star operator */
  HSHM_INLINE_CROSS_FUN
  CtxAllocator<AllocT> &operator*() { return alloc_; }

  /** Star operator (const) */
  HSHM_INLINE_CROSS_FUN
  const CtxAllocator<AllocT> &operator*() const { return alloc_; }
};

/** Thread-local storage manager */
template <typename AllocT>
class TlsAllocatorInfo : public thread::ThreadLocalData {
 public:
  AllocT *alloc_;
  ThreadId tid_;

 public:
  HSHM_CROSS_FUN
  TlsAllocatorInfo() : alloc_(nullptr), tid_(ThreadId::GetNull()) {}

  HSHM_CROSS_FUN
  void destroy() { alloc_->FreeTls(tid_); }
};

/**
 * Stores an offset into a memory region. Assumes the developer knows
 * which allocator the pointer comes from.
 * */
template <bool ATOMIC = false>
struct OffsetPointerBase : public ShmPointer {
  hipc::opt_atomic<hshm::size_t, ATOMIC>
      off_; /**< Offset within the allocator's slot */

  /** Serialize an hipc::OffsetPointerBase */
  template <typename Ar>
  HSHM_INLINE_CROSS_FUN void serialize(Ar &ar) {
    ar & off_;
  }

  /** ostream operator */
  friend std::ostream &operator<<(std::ostream &os,
                                  const OffsetPointerBase &ptr) {
    os << ptr.off_.load();
    return os;
  }

  /** Default constructor */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase() = default;

  /** Full constructor */
  HSHM_INLINE_CROSS_FUN explicit OffsetPointerBase(size_t off) : off_(off) {}

  /** Full constructor */
  HSHM_INLINE_CROSS_FUN explicit OffsetPointerBase(
      hipc::opt_atomic<hshm::size_t, ATOMIC> off)
      : off_(off.load()) {}

  /** Pointer constructor */
  HSHM_INLINE_CROSS_FUN explicit OffsetPointerBase(AllocatorId alloc_id,
                                                   size_t off)
      : off_(off) {
    (void)alloc_id;
  }

  /** Pointer constructor (alloc + atomic offset)*/
  HSHM_INLINE_CROSS_FUN explicit OffsetPointerBase(AllocatorId id,
                                                   OffsetPointerBase<true> off)
      : off_(off.load()) {
    (void)id;
  }

  /** Pointer constructor (alloc + non-offeset) */
  HSHM_INLINE_CROSS_FUN explicit OffsetPointerBase(AllocatorId id,
                                                   OffsetPointerBase<false> off)
      : off_(off.load()) {
    (void)id;
  }

  /** Copy constructor */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase(const OffsetPointerBase &other)
      : off_(other.off_.load()) {}

  /** Other copy constructor */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase(
      const OffsetPointerBase<!ATOMIC> &other)
      : off_(other.off_.load()) {}

  /** Move constructor */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase(OffsetPointerBase &&other) noexcept
      : off_(other.off_.load()) {
    other.SetNull();
  }

  /** Get the offset pointer */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase<false> ToOffsetPointer() const {
    return OffsetPointerBase<false>(off_.load());
  }

  /** Set to null (offsets can be 0, so not 0) */
  HSHM_INLINE_CROSS_FUN void SetNull() { off_ = (size_t)-1; }

  /** Check if null */
  HSHM_INLINE_CROSS_FUN bool IsNull() const {
    return off_.load() == (size_t)-1;
  }

  /** Get the null pointer */
  HSHM_INLINE_CROSS_FUN static OffsetPointerBase GetNull() {
    return OffsetPointerBase((size_t)-1);
  }

  /** Atomic load wrapper */
  HSHM_INLINE_CROSS_FUN size_t
  load(std::memory_order order = std::memory_order_seq_cst) const {
    return off_.load(order);
  }

  /** Atomic exchange wrapper */
  HSHM_INLINE_CROSS_FUN void exchange(
      size_t count, std::memory_order order = std::memory_order_seq_cst) {
    off_.exchange(count, order);
  }

  /** Atomic compare exchange weak wrapper */
  HSHM_INLINE_CROSS_FUN bool compare_exchange_weak(
      size_t &expected, size_t desired,
      std::memory_order order = std::memory_order_seq_cst) {
    return off_.compare_exchange_weak(expected, desired, order);
  }

  /** Atomic compare exchange strong wrapper */
  HSHM_INLINE_CROSS_FUN bool compare_exchange_strong(
      size_t &expected, size_t desired,
      std::memory_order order = std::memory_order_seq_cst) {
    return off_.compare_exchange_weak(expected, desired, order);
  }

  /** Atomic add operator */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase operator+(size_t count) const {
    return OffsetPointerBase(off_ + count);
  }

  /** Atomic subtract operator */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase operator-(size_t count) const {
    return OffsetPointerBase(off_ - count);
  }

  /** Atomic add assign operator */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase &operator+=(size_t count) {
    off_ += count;
    return *this;
  }

  /** Atomic subtract assign operator */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase &operator-=(size_t count) {
    off_ -= count;
    return *this;
  }

  /** Atomic increment (post) */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase operator++(int) {
    return OffsetPointerBase(off_++);
  }

  /** Atomic increment (pre) */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase &operator++() {
    ++off_;
    return *this;
  }

  /** Atomic decrement (post) */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase operator--(int) {
    return OffsetPointerBase(off_--);
  }

  /** Atomic decrement (pre) */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase &operator--() {
    --off_;
    return *this;
  }

  /** Atomic assign operator */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase &operator=(size_t count) {
    off_ = count;
    return *this;
  }

  /** Atomic copy assign operator */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase &operator=(
      const OffsetPointerBase &count) {
    off_ = count.load();
    return *this;
  }

  /** Equality check */
  HSHM_INLINE_CROSS_FUN bool operator==(const OffsetPointerBase &other) const {
    return off_ == other.off_;
  }

  /** Inequality check */
  HSHM_INLINE_CROSS_FUN bool operator!=(const OffsetPointerBase &other) const {
    return off_ != other.off_;
  }

  /** Mark first bit */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase Mark() const {
    return OffsetPointerBase(MARK_FIRST_BIT(size_t, off_.load()));
  }

  /** Check if first bit is marked */
  HSHM_INLINE_CROSS_FUN bool IsMarked() const {
    return IS_FIRST_BIT_MARKED(size_t, off_.load());
  }

  /** Unmark first bit */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase Unmark() const {
    return OffsetPointerBase(UNMARK_FIRST_BIT(size_t, off_.load()));
  }

  /** Set to 0 */
  HSHM_INLINE_CROSS_FUN void SetZero() { off_ = 0; }
};

/** Non-atomic offset */
typedef OffsetPointerBase<false> OffsetPointer;

/** Atomic offset */
typedef OffsetPointerBase<true> AtomicOffsetPointer;

/** Typed offset pointer */
template <typename T>
using TypedOffsetPointer = OffsetPointer;

/** Typed atomic pointer */
template <typename T>
using TypedAtomicOffsetPointer = AtomicOffsetPointer;

/**
 * A process-independent pointer, which stores both the allocator's
 * information and the offset within the allocator's region
 * */
template <bool ATOMIC = false>
struct PointerBase : public ShmPointer {
  AllocatorId alloc_id_;           /// Allocator the pointer comes from
  OffsetPointerBase<ATOMIC> off_;  /// Offset within the allocator's slot

  /** Serialize a pointer */
  template <typename Ar>
  HSHM_INLINE_CROSS_FUN void serialize(Ar &ar) {
    ar & alloc_id_;
    ar & off_;
  }

  /** Ostream operator */
  friend std::ostream &operator<<(std::ostream &os, const PointerBase &ptr) {
    os << ptr.alloc_id_ << "::" << ptr.off_;
    return os;
  }

  /** Default constructor */
  HSHM_INLINE_CROSS_FUN PointerBase() = default;

  /** Full constructor */
  HSHM_INLINE_CROSS_FUN explicit PointerBase(AllocatorId id, size_t off)
      : alloc_id_(id), off_(off) {}

  /** Full constructor using offset pointer */
  HSHM_INLINE_CROSS_FUN explicit PointerBase(AllocatorId id, OffsetPointer off)
      : alloc_id_(id), off_(off) {}

  /** Copy constructor */
  HSHM_INLINE_CROSS_FUN PointerBase(const PointerBase &other)
      : alloc_id_(other.alloc_id_), off_(other.off_) {}

  /** Other copy constructor */
  HSHM_INLINE_CROSS_FUN PointerBase(const PointerBase<!ATOMIC> &other)
      : alloc_id_(other.alloc_id_), off_(other.off_.load()) {}

  /** Move constructor */
  HSHM_INLINE_CROSS_FUN PointerBase(PointerBase &&other) noexcept
      : alloc_id_(other.alloc_id_), off_(other.off_) {
    other.SetNull();
  }

  /** Get the offset pointer */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase<false> ToOffsetPointer() const {
    return OffsetPointerBase<false>(off_.load());
  }

  /** Set to null */
  HSHM_INLINE_CROSS_FUN void SetNull() { alloc_id_.SetNull(); }

  /** Check if null */
  HSHM_INLINE_CROSS_FUN bool IsNull() const { return alloc_id_.IsNull(); }

  /** Get the null pointer */
  HSHM_INLINE_CROSS_FUN static PointerBase GetNull() {
    return PointerBase{AllocatorId::GetNull(), OffsetPointer::GetNull()};
  }

  /** Copy assignment operator */
  HSHM_INLINE_CROSS_FUN PointerBase &operator=(const PointerBase &other) {
    if (this != &other) {
      alloc_id_ = other.alloc_id_;
      off_ = other.off_;
    }
    return *this;
  }

  /** Move assignment operator */
  HSHM_INLINE_CROSS_FUN PointerBase &operator=(PointerBase &&other) {
    if (this != &other) {
      alloc_id_ = other.alloc_id_;
      off_.exchange(other.off_.load());
      other.SetNull();
    }
    return *this;
  }

  /** Addition operator */
  HSHM_INLINE_CROSS_FUN PointerBase operator+(size_t size) const {
    PointerBase p;
    p.alloc_id_ = alloc_id_;
    p.off_ = off_ + size;
    return p;
  }

  /** Subtraction operator */
  HSHM_INLINE_CROSS_FUN PointerBase operator-(size_t size) const {
    PointerBase p;
    p.alloc_id_ = alloc_id_;
    p.off_ = off_ - size;
    return p;
  }

  /** Addition assignment operator */
  HSHM_INLINE_CROSS_FUN PointerBase &operator+=(size_t size) {
    off_ += size;
    return *this;
  }

  /** Subtraction assignment operator */
  HSHM_INLINE_CROSS_FUN PointerBase &operator-=(size_t size) {
    off_ -= size;
    return *this;
  }

  /** Increment operator (pre) */
  HSHM_INLINE_CROSS_FUN PointerBase &operator++() {
    off_++;
    return *this;
  }

  /** Decrement operator (pre) */
  HSHM_INLINE_CROSS_FUN PointerBase &operator--() {
    off_--;
    return *this;
  }

  /** Increment operator (post) */
  HSHM_INLINE_CROSS_FUN PointerBase operator++(int) {
    PointerBase tmp(*this);
    operator++();
    return tmp;
  }

  /** Decrement operator (post) */
  HSHM_INLINE_CROSS_FUN PointerBase operator--(int) {
    PointerBase tmp(*this);
    operator--();
    return tmp;
  }

  /** Equality check */
  HSHM_INLINE_CROSS_FUN bool operator==(const PointerBase &other) const {
    return (other.alloc_id_ == alloc_id_ && other.off_ == off_);
  }

  /** Inequality check */
  HSHM_INLINE_CROSS_FUN bool operator!=(const PointerBase &other) const {
    return (other.alloc_id_ != alloc_id_ || other.off_ != off_);
  }

  /** Mark first bit */
  HSHM_INLINE_CROSS_FUN PointerBase Mark() const {
    return PointerBase(alloc_id_, off_.Mark());
  }

  /** Check if first bit is marked */
  HSHM_INLINE_CROSS_FUN bool IsMarked() const { return off_.IsMarked(); }

  /** Unmark first bit */
  HSHM_INLINE_CROSS_FUN PointerBase Unmark() const {
    return PointerBase(alloc_id_, off_.Unmark());
  }

  /** Set to 0 */
  HSHM_INLINE_CROSS_FUN void SetZero() { off_.SetZero(); }
};

/** Non-atomic pointer */
typedef PointerBase<false> Pointer;

/** Atomic pointer */
typedef PointerBase<true> AtomicPointer;

/** Typed pointer */
template <typename T>
using TypedPointer = Pointer;

/** Typed atomic pointer */
template <typename T>
using TypedAtomicPointer = AtomicPointer;

/** Struct containing both private and shared pointer */
template <typename T = char, typename PointerT = Pointer>
struct FullPtr : public ShmPointer {
  T *ptr_;
  PointerT shm_;

  /** Serialize hipc::FullPtr */
  template <typename Ar>
  HSHM_INLINE_CROSS_FUN void serialize(Ar &ar) {
    ar & shm_;
  }

  // /** Serialize an hipc::FullPtr */
  // template <typename Ar>
  // HSHM_INLINE_CROSS_FUN void save(Ar &ar) const {
  //   ar & shm_;
  // }

  // /** Deserialize an hipc::FullPtr */
  // template <typename Ar>
  // HSHM_INLINE_CROSS_FUN void load(Ar &ar) {
  //   ar & shm_;
  //   ptr_ = FullPtr<T>(shm_).ptr_;
  // }

  /** Ostream operator */
  friend std::ostream &operator<<(std::ostream &os, const FullPtr &ptr) {
    os << (void *)ptr.ptr_ << " " << ptr.shm_;
    return os;
  }

  /** Default constructor */
  HSHM_INLINE_CROSS_FUN FullPtr() = default;

  /** Full constructor */
  HSHM_INLINE_CROSS_FUN FullPtr(const T *ptr, const PointerT &shm)
      : ptr_(const_cast<T *>(ptr)), shm_(shm) {}

  /** Private half + alloc constructor */
  template<typename AllocT>
  HSHM_INLINE_CROSS_FUN explicit FullPtr(const hipc::CtxAllocator<AllocT> &ctx_alloc, const T *ptr) {
    if (ctx_alloc->ContainsPtr(ptr)) {
      shm_.off_ = (size_t)(reinterpret_cast<const char*>(ptr) - ctx_alloc->buffer_);
      shm_.alloc_id_ = ctx_alloc->id_;
      ptr_ = const_cast<T*>(ptr);
    } else {
        HSHM_THROW_ERROR(PTR_NOT_IN_ALLOCATOR);
    }
  }

  /** Shared half + alloc constructor for OffsetPointer */
  template<typename AllocT, bool ATOMIC>
  HSHM_INLINE_CROSS_FUN explicit FullPtr(const hipc::CtxAllocator<AllocT> &ctx_alloc,
                                         const OffsetPointerBase<ATOMIC> &shm) {
    if (ctx_alloc->ContainsPtr(shm)) {
      shm_.off_ = shm.load();
      shm_.alloc_id_ = ctx_alloc->id_;
      ptr_ = reinterpret_cast<T*>(ctx_alloc->buffer_ + shm.load());
    } else {
        HSHM_THROW_ERROR(PTR_NOT_IN_ALLOCATOR);
    }
 }

 /** Shared half + alloc constructor for Pointer */
  template<typename AllocT, bool ATOMIC>
  HSHM_INLINE_CROSS_FUN explicit FullPtr(const hipc::CtxAllocator<AllocT> &ctx_alloc,
                                         const PointerBase<ATOMIC> &shm) {
    if (ctx_alloc->ContainsPtr(shm)) {
      shm_.off_ = shm.off_.load();
      shm_.alloc_id_ = shm.alloc_id_;
      ptr_ = reinterpret_cast<T*>(ctx_alloc->buffer_ + shm.off_.load());
    } else {
        HSHM_THROW_ERROR(PTR_NOT_IN_ALLOCATOR);
    }
 }

  /** Copy constructor */
  HSHM_INLINE_CROSS_FUN FullPtr(const FullPtr &other)
      : ptr_(other.ptr_), shm_(other.shm_) {}

  /** Move constructor */
  HSHM_INLINE_CROSS_FUN FullPtr(FullPtr &&other) noexcept
      : ptr_(other.ptr_), shm_(other.shm_) {
    other.SetNull();
  }

  /** Copy assignment operator */
  HSHM_INLINE_CROSS_FUN FullPtr &operator=(const FullPtr &other) {
    if (this != &other) {
      ptr_ = other.ptr_;
      shm_ = other.shm_;
    }
    return *this;
  }

  /** Move assignment operator */
  HSHM_INLINE_CROSS_FUN FullPtr &operator=(FullPtr &&other) {
    if (this != &other) {
      ptr_ = other.ptr_;
      shm_ = other.shm_;
      other.SetNull();
    }
    return *this;
  }

  /** Overload arrow */
  template<typename U = T>
  HSHM_INLINE_CROSS_FUN typename std::enable_if<!std::is_void<U>::value, U*>::type
  operator->() const { return ptr_; }

  /** Overload dereference */
  template<typename U = T>
  HSHM_INLINE_CROSS_FUN typename std::enable_if<!std::is_void<U>::value, U&>::type
  operator*() const { return *ptr_; }

  /** Equality operator */
  HSHM_INLINE_CROSS_FUN bool operator==(const FullPtr &other) const {
    return ptr_ == other.ptr_ && shm_ == other.shm_;
  }

  /** Inequality operator */
  HSHM_INLINE_CROSS_FUN bool operator!=(const FullPtr &other) const {
    return ptr_ != other.ptr_ || shm_ != other.shm_;
  }

  /** Addition operator */
  HSHM_INLINE_CROSS_FUN FullPtr operator+(size_t size) const {
    return FullPtr(ptr_ + size, shm_ + size);
  }

  /** Subtraction operator */
  HSHM_INLINE_CROSS_FUN FullPtr operator-(size_t size) const {
    return FullPtr(ptr_ - size, shm_ - size);
  }

  /** Addition assignment operator */
  HSHM_INLINE_CROSS_FUN FullPtr &operator+=(size_t size) {
    ptr_ += size;
    shm_ += size;
    return *this;
  }

  /** Subtraction assignment operator */
  HSHM_INLINE_CROSS_FUN FullPtr &operator-=(size_t size) {
    ptr_ -= size;
    shm_ -= size;
    return *this;
  }

  /** Increment operator (pre) */
  HSHM_INLINE_CROSS_FUN FullPtr &operator++() {
    ptr_++;
    shm_++;
    return *this;
  }

  /** Decrement operator (pre) */
  HSHM_INLINE_CROSS_FUN FullPtr &operator--() {
    ptr_--;
    shm_--;
    return *this;
  }

  /** Increment operator (post) */
  HSHM_INLINE_CROSS_FUN FullPtr operator++(int) {
    FullPtr tmp(*this);
    operator++();
    return tmp;
  }

  /** Decrement operator (post) */
  HSHM_INLINE_CROSS_FUN FullPtr operator--(int) {
    FullPtr tmp(*this);
    operator--();
    return tmp;
  }

  /** Check if null */
  HSHM_INLINE_CROSS_FUN bool IsNull() const { return ptr_ == nullptr; }

  /** Get null */
  HSHM_INLINE_CROSS_FUN static FullPtr GetNull() {
    return FullPtr(nullptr, Pointer::GetNull());
  }

  /** Set to null */
  HSHM_INLINE_CROSS_FUN void SetNull() { ptr_ = nullptr; }

  /** Reintrepret cast to other internal type */
  template <typename U>
  HSHM_INLINE_CROSS_FUN FullPtr<U, PointerT> &Cast() {
    return DeepCast<FullPtr<U, PointerT>>();
  }

  /** Reintrepret cast to other internal type (const) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN const FullPtr<U, PointerT> &Cast() const {
    return DeepCast<FullPtr<U, PointerT>>();
  }

  /** Reintrepret cast to another FullPtr */
  template <typename FullPtrT>
  HSHM_INLINE_CROSS_FUN FullPtrT &DeepCast() {
    return *((FullPtrT *)this);
  }

  /** Reintrepret cast to another FullPtr (const) */
  template <typename FullPtrT>
  HSHM_INLINE_CROSS_FUN const FullPtrT &DeepCast() const {
    return *((FullPtrT *)this);
  }

  /** Mark first bit */
  HSHM_INLINE_CROSS_FUN FullPtr Mark() const {
    return FullPtr(ptr_, shm_.Mark());
  }

  /** Check if first bit is marked */
  HSHM_INLINE_CROSS_FUN bool IsMarked() const { return shm_.IsMarked(); }

  /** Unmark first bit */
  HSHM_INLINE_CROSS_FUN FullPtr Unmark() const {
    return FullPtr(ptr_, shm_.Unmark());
  }

  /** Set to 0 */
  HSHM_INLINE_CROSS_FUN void SetZero() { shm_.SetZero(); }
};

/** Alias to full pointer (deprecated) */
template <typename T = char, typename PointerT = Pointer>
using LPointer = FullPtr<T, PointerT>;


class MemoryAlignment {
 public:
  /**
   * Round up to the nearest multiple of the alignment
   * @param alignment the alignment value (e.g., 4096)
   * @param size the size to make a multiple of alignment (e.g., 4097)
   * @return the new size  (e.g., 8192)
   * */
  static size_t AlignTo(size_t alignment, size_t size) {
    auto page_size = HSHM_SYSTEM_INFO->page_size_;
    size_t new_size = size;
    size_t page_off = size % alignment;
    if (page_off) {
      new_size = size + page_size - page_off;
    }
    return new_size;
  }

  /**
   * Round up to the nearest multiple of page size
   * @param size the size to align to the PAGE_SIZE
   * */
  static size_t AlignToPageSize(size_t size) {
    auto page_size = HSHM_SYSTEM_INFO->page_size_;
    size_t new_size = AlignTo(page_size, size);
    return new_size;
  }
};

}  // namespace hshm::ipc

namespace std {

/** Allocator ID hash */
template <>
struct hash<hshm::ipc::AllocatorId> {
  std::size_t operator()(const hshm::ipc::AllocatorId &key) const {
    return std::hash<uint64_t>{}(key.int_);
  }
};

}  // namespace std

namespace hshm {

/** Allocator ID hash */
template <>
struct hash<hshm::ipc::AllocatorId> {
  HSHM_INLINE_CROSS_FUN std::size_t operator()(
      const hshm::ipc::AllocatorId &key) const {
    return hshm::hash<uint64_t>{}(key.int_);
  }
};

}  // namespace hshm

#define IS_SHM_POINTER(T) std::is_base_of_v<hipc::ShmPointer, T>

#endif  // HSHM_MEMORY_ALLOCATOR_ALLOCATOR_H_

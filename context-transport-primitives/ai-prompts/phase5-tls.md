@CLAUDE.md

Remove AlignedAllocateOffset and similar functions as a method from allocators.

# FullPtr

Add the following constructors:

```
FullPtr(AllocT *alloc, const OffsetPointerBase<ATOMIC> &shm);
FullPtr(AllocT *alloc, const PointerBase<ATOMIC> &shm);
```

They should be effectively the same implementation as the CtxAllocator ones.




# BuddyAllocator

## shm_init

### Parameters
1. An OffsetPointer to the beginning of the heap in shared memory
2. The atomic Heap object
3. The size of page metadata.

### Implementation

Store the Heap inside the shm header.
Create a fixed table for storing free lists by allocating from the heap.
round_up_list: Free list for every power of two between 32 bytes and 16KB should have a free list. 
round_down_list: Free list for every power of two between 16KB and 1MB.

## AllocateOffset
Takes as input size. HSHM_MCTX is ignored.

Case 1: Size < 16KB
1. Get the free list for this size. Identify the power-of-two using a logarithm of request size - size of page metadata. Round up.
2. Check if there is a page existing in the free lists. If so, return it.
3. Check if a large page exists in the free lists, divide into smaller pages of this size. Return it.
4. Run Coalesce for all smaller page sizes to produce pages of this size. Recheck our free list. If page existing, return it.
5. Try allocating from heap. If successful, return.
6. Return OffsetPointer::GetNull().

Case 2: Size > 16KB
1. Identify power-of-two using logarithm. Round down. Cap at 20 (2^20 = 1MB).
2. Check each entry if there is a fit (i.e., the page size > requested size).
3. If not, check if a larger page exists in any of the larger free lists. If yes, remove the first match and then subset the requested size. Move the remainder to the most appropriate free list. return.
4. Run Coalesce for all smaller than or equal to page sizes. If coalesce yields a page fitting the size, then return it.
5. Try allocating from heap. If successful, return
6. Return OffsetPointer::GetNull()

## FreeOffset

Add page to the free list matching its size.

## Coalesce

Build a balanced binary search tree.





# ThreadLocalAllocator

I want to create an allocator that levearages thread-local storage for most operations.
Each process will have its own large block of shared memory allocated to it. 
Each process then will need 

## shm_init

### Additional Parameters

The maximum number of processes that can connect to the shared memory. Default is 256.

### Implementation
Store an atomic heap in the shared memory header.

Use heap to allocate an array of Process blocks.

Store a counter 

Allocate one block to this process.

## shm_attach
### Additional Parameters
1. The max amount of shared memory for this process.
2. Number of thread blocks to create for this process. Default 64.

### Implementation



## shm_detach

Delete all memory associated with this process and free back 

## AllocateOffset

If the MemContext has a ``tid < 0``, then we must attempt to derive it automatically.



## AlignedAllocateOffset

## FreeOffsetNoNullCheck

## Coalescing

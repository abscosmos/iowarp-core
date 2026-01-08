@CLAUDE.md We are doing a hard refactoring of hshm. Delete the following. Remove any tests that need to be
removed to get this code compiling again. Use the debug preset. Do not remove any code outside of context-transport-primitives.
When you compile, enusre that context-assimilation-engine, context-exploration-engine, context-runtime, and context-transfer-engine
are disabled.

context-transport-primitives/include/hermes_shm/memory/memory_manager_.h
context-transport-primitives/include/hermes_shm/memory/memory_manager.h
context-transport-primitives/test/unit/allocators
context-transport-primitives/test/unit/allocators_mpi
context-transport-primitives/test/unit/cuda
context-transport-primitives/test/unit/data_structures
context-transport-primitives/test/unit/rocm
context-transport-primitives/include/hermes_shm/data_structures/ipc/charwrap.h
context-transport-primitives/include/hermes_shm/data_structures/ipc/dynamic_queue.h
context-transport-primitives/include/hermes_shm/data_structures/ipc/functional.h
context-transport-primitives/include/hermes_shm/data_structures/ipc/hash.h
context-transport-primitives/include/hermes_shm/data_structures/ipc/key_set.h
context-transport-primitives/include/hermes_shm/data_structures/ipc/lifo_list_queue.h
context-transport-primitives/include/hermes_shm/data_structures/ipc/list.h
context-transport-primitives/include/hermes_shm/data_structures/ipc/mpsc_lifo_list_queue.h
context-transport-primitives/include/hermes_shm/data_structures/ipc/multi_ring_buffer.h
context-transport-primitives/include/hermes_shm/data_structures/ipc/pair.h
context-transport-primitives/include/hermes_shm/data_structures/ipc/ring_ptr_queue.h
context-transport-primitives/include/hermes_shm/data_structures/ipc/ring_queue_flags.h
context-transport-primitives/include/hermes_shm/data_structures/ipc/ring_queue.h
context-transport-primitives/include/hermes_shm/data_structures/ipc/slist.h
context-transport-primitives/include/hermes_shm/data_structures/ipc/split_ticket_queue.h
context-transport-primitives/include/hermes_shm/data_structures/ipc/spsc_fifo_list_queue.h
context-transport-primitives/include/hermes_shm/data_structures/ipc/string_common.h
context-transport-primitives/include/hermes_shm/data_structures/ipc/string.h
context-transport-primitives/include/hermes_shm/data_structures/ipc/stringstream.h
context-transport-primitives/include/hermes_shm/data_structures/ipc/ticket_queue.h
context-transport-primitives/include/hermes_shm/data_structures/ipc/tuple_base.h
context-transport-primitives/include/hermes_shm/data_structures/ipc/unordered_map.h
context-transport-primitives/include/hermes_shm/data_structures/ipc/vector.h
context-transport-primitives/benchmark


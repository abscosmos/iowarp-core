@CLAUDE.md

# Ring Buffer

In the main branch, I have a ring_buffer implementation that provides various compile-time options, such as support for lock-free multiple-producer, single-consumer access. 

There are technically two, but I want you to ignore the ring_buffer_ptr_queue. Focus only on the ring_buffer.

I want you to adapt that to this current branch.

You should have hipc typedefs, but not hshm typedefs. Read the file to see what that means.

Instead of using hipc::pair for the queue, just make your own custom data structure for holding two entries.


@CLAUDE.md 

Please also add the relevant typedefs from the main branch. Every typedef from the ring_queue.h that is in hshm::ipc namespace please. Add them to the ring_buffer.h in this branch. These are the ones I remember:
1. ext_ring_buffer: An extensible ring buffer, single-thread only. It should extend buffer if we reach capacity limit.
2. spsc_ring_buffer: A fixed-size ring buffer, also single-thread only. It should error if we reach the capacity limit.
3. mpsc_ring_buffer: A fixed-size ring buffer, multiple can emplace, but only one can consume. It should NOT error if we reach capacity limit and assume the consumer will free up space eventually.

We should have a test verifying each typedef data structure.
We should have a single workload generator class testing all angles of the queues.
We may not use each workload for each typedef, but they should all be in a single class.

For mpsc_ring_buffer, we need the following test: 
1. We will spawn 4 producer threads. Each producer thread will emplace 256 elements. The queue should have size 8.
1. One thread consumes constantly. It will consume until X entries are received
So one thread that consumes, and then N threads that are emplacing. The queue for emplacing should be 2*number of threads. Each thread should emplace elements for at least 2 seconds constantly.
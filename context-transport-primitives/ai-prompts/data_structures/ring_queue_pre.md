@CLAUDE.md 

# Preallocated Ring Queue

Implement this data structure and add some unit tests.

In context-transport-primitives/include/hermes_shm/data_structures/ipc/ring_queue_pre.md
```
namespace hshm::ipc::pre {

template<typename T, size_t COUNT>
class ring_queue {
};

}
```

Template parameters:
1. T (the type stored in the vector)
2. COUNT (the max number of T stored in the vector)

Class variables:
1. size_t head_;
2. size_t tail_; 
2. T data_[COUNT];

## push

verify size() < COUNT
size_t tail = tail_ % COUNT;
data_[tail] = entry;
tail_++;

## pop

verify tail - head > 0
size_t head = head_ % COUNT;
data_[head]
head_++;

## size

tail_ - head_;
#include <cstdio>
#include <atomic>

struct TestAtomic {
  std::atomic<size_t> head_;
  std::atomic<size_t> tail_;

  TestAtomic() : head_(0), tail_(0) {
    printf("TestAtomic constructor completed\n");
  }
};

int main() {
  printf("Creating TestAtomic...\n");
  TestAtomic t;
  printf("Created TestAtomic successfully\n");
  printf("head=%zu, tail=%zu\n", t.head_.load(), t.tail_.load());
  return 0;
}

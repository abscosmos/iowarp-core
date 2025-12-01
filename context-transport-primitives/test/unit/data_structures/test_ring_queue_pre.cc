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

#include <catch2/catch_test_macros.hpp>
#include "hermes_shm/data_structures/ipc/ring_queue_pre.h"
#include <string>

using namespace hshm::ipc::pre;

TEST_CASE("RingQueue - Initialization", "[ring_queue]") {
  ring_queue<int, 10> queue;

  REQUIRE(queue.size() == 0);
  REQUIRE(queue.empty());
  REQUIRE_FALSE(queue.full());
  REQUIRE(queue.capacity() == 10);
}

TEST_CASE("RingQueue - Basic Push and Pop", "[ring_queue]") {
  ring_queue<int, 5> queue;

  SECTION("Single push and pop") {
    queue.push(42);
    REQUIRE(queue.size() == 1);
    REQUIRE_FALSE(queue.empty());

    int val = queue.pop();
    REQUIRE(val == 42);
    REQUIRE(queue.size() == 0);
    REQUIRE(queue.empty());
  }

  SECTION("Multiple push and pop") {
    queue.push(1);
    queue.push(2);
    queue.push(3);

    REQUIRE(queue.size() == 3);
    REQUIRE(queue.pop() == 1);
    REQUIRE(queue.pop() == 2);
    REQUIRE(queue.pop() == 3);
    REQUIRE(queue.empty());
  }
}

TEST_CASE("RingQueue - FIFO Order", "[ring_queue]") {
  ring_queue<int, 10> queue;

  // Push sequence
  for (int i = 0; i < 5; ++i) {
    queue.push(i * 10);
  }

  // Verify FIFO order
  REQUIRE(queue.pop() == 0);
  REQUIRE(queue.pop() == 10);
  REQUIRE(queue.pop() == 20);
  REQUIRE(queue.pop() == 30);
  REQUIRE(queue.pop() == 40);
}

TEST_CASE("RingQueue - Full Capacity", "[ring_queue]") {
  ring_queue<int, 3> queue;

  SECTION("Fill to capacity") {
    queue.push(1);
    queue.push(2);
    queue.push(3);

    REQUIRE(queue.size() == 3);
    REQUIRE(queue.full());
    REQUIRE_FALSE(queue.empty());
  }

  SECTION("Overflow throws exception") {
    queue.push(1);
    queue.push(2);
    queue.push(3);

    REQUIRE_THROWS_AS(queue.push(4), std::overflow_error);
  }
}

TEST_CASE("RingQueue - Empty Queue", "[ring_queue]") {
  ring_queue<int, 5> queue;

  SECTION("Pop from empty throws exception") {
    REQUIRE_THROWS_AS(queue.pop(), std::underflow_error);
  }

  SECTION("Front on empty throws exception") {
    REQUIRE_THROWS_AS(queue.front(), std::underflow_error);
  }

  SECTION("Back on empty throws exception") {
    REQUIRE_THROWS_AS(queue.back(), std::underflow_error);
  }
}

TEST_CASE("RingQueue - Wraparound Behavior", "[ring_queue]") {
  ring_queue<int, 4> queue;

  SECTION("Simple wraparound") {
    // Fill queue
    queue.push(1);
    queue.push(2);
    queue.push(3);
    queue.push(4);

    // Pop two elements
    REQUIRE(queue.pop() == 1);
    REQUIRE(queue.pop() == 2);

    // Push two more (should wrap around)
    queue.push(5);
    queue.push(6);

    // Verify order is maintained
    REQUIRE(queue.size() == 4);
    REQUIRE(queue.pop() == 3);
    REQUIRE(queue.pop() == 4);
    REQUIRE(queue.pop() == 5);
    REQUIRE(queue.pop() == 6);
    REQUIRE(queue.empty());
  }

  SECTION("Multiple wraparound cycles") {
    // Perform multiple wrap cycles
    for (int cycle = 0; cycle < 3; ++cycle) {
      // Fill and empty queue
      for (int i = 0; i < 4; ++i) {
        queue.push(cycle * 10 + i);
      }

      for (int i = 0; i < 4; ++i) {
        REQUIRE(queue.pop() == cycle * 10 + i);
      }

      REQUIRE(queue.empty());
    }
  }
}

TEST_CASE("RingQueue - Front and Back", "[ring_queue]") {
  ring_queue<int, 5> queue;

  queue.push(10);
  REQUIRE(queue.front() == 10);
  REQUIRE(queue.back() == 10);

  queue.push(20);
  REQUIRE(queue.front() == 10);
  REQUIRE(queue.back() == 20);

  queue.push(30);
  REQUIRE(queue.front() == 10);
  REQUIRE(queue.back() == 30);

  queue.pop();
  REQUIRE(queue.front() == 20);
  REQUIRE(queue.back() == 30);
}

TEST_CASE("RingQueue - Clear", "[ring_queue]") {
  ring_queue<int, 5> queue;

  queue.push(1);
  queue.push(2);
  queue.push(3);

  REQUIRE(queue.size() == 3);

  queue.clear();

  REQUIRE(queue.size() == 0);
  REQUIRE(queue.empty());
  REQUIRE_FALSE(queue.full());
}

TEST_CASE("RingQueue - Different Types", "[ring_queue]") {
  SECTION("String type") {
    ring_queue<std::string, 3> queue;

    queue.push("hello");
    queue.push("world");

    REQUIRE(queue.pop() == "hello");
    REQUIRE(queue.pop() == "world");
  }

  SECTION("Struct type") {
    struct Point {
      int x, y;
      bool operator==(const Point &other) const {
        return x == other.x && y == other.y;
      }
    };

    ring_queue<Point, 4> queue;

    queue.push({1, 2});
    queue.push({3, 4});

    Point p1 = queue.pop();
    REQUIRE(p1.x == 1);
    REQUIRE(p1.y == 2);

    Point p2 = queue.pop();
    REQUIRE(p2.x == 3);
    REQUIRE(p2.y == 4);
  }
}

TEST_CASE("RingQueue - Stress Test", "[ring_queue]") {
  ring_queue<int, 100> queue;

  SECTION("Fill and empty multiple times") {
    for (int iteration = 0; iteration < 10; ++iteration) {
      // Fill queue
      for (size_t i = 0; i < queue.capacity(); ++i) {
        queue.push(static_cast<int>(iteration * 1000 + i));
      }

      REQUIRE(queue.full());
      REQUIRE(queue.size() == 100);

      // Empty queue
      for (size_t i = 0; i < queue.capacity(); ++i) {
        int expected = static_cast<int>(iteration * 1000 + i);
        REQUIRE(queue.pop() == expected);
      }

      REQUIRE(queue.empty());
    }
  }

  SECTION("Alternating push and pop") {
    int pushed = 0;
    for (int i = 0; i < 500; ++i) {
      if (!queue.full()) {
        queue.push(i);
        pushed++;
      }

      if (i % 3 == 0 && !queue.empty()) {
        queue.pop();
      }
    }

    REQUIRE(pushed > 0);  // Verify we pushed some elements

    // Drain remaining elements
    int last = -1;
    while (!queue.empty()) {
      int val = queue.pop();
      REQUIRE(val > last);  // Verify FIFO order
      last = val;
    }
  }
}

TEST_CASE("RingQueue - Edge Cases", "[ring_queue]") {
  SECTION("Size 1 queue") {
    ring_queue<int, 1> queue;

    queue.push(42);
    REQUIRE(queue.full());
    REQUIRE(queue.size() == 1);

    REQUIRE(queue.pop() == 42);
    REQUIRE(queue.empty());
  }

  SECTION("Large index wraparound") {
    ring_queue<int, 3> queue;

    // Simulate many operations to force large head/tail values
    for (int cycle = 0; cycle < 1000; ++cycle) {
      if (!queue.full()) {
        queue.push(cycle);
      }
      if (!queue.empty()) {
        queue.pop();
      }
    }

    // Queue should still work correctly
    queue.clear();
    queue.push(1);
    queue.push(2);
    REQUIRE(queue.pop() == 1);
    REQUIRE(queue.pop() == 2);
  }
}

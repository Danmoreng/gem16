#include <atomic>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "server/request_queue.h"
#include "test.h"

namespace {

void TestFifoAndCapacity() {
  gem16::server::RequestQueue queue(1U, 4U);
  auto first_result = queue.Acquire();
  GEM16_CHECK(first_result.ok());
  if (!first_result.ok()) return;
  gem16::server::RequestAdmission first = std::move(first_result).value();

  std::mutex order_mutex;
  std::vector<int> order;
  std::atomic<bool> first_waiter_started{false};
  std::atomic<bool> second_waiter_started{false};
  std::thread waiter_one([&] {
    first_waiter_started.store(true);
    auto admission = queue.Acquire();
    GEM16_CHECK(admission.ok());
    if (admission.ok()) {
      std::lock_guard lock(order_mutex);
      order.push_back(1);
    }
  });
  while (!first_waiter_started.load()) std::this_thread::yield();
  while (queue.Snapshot().queued != 1U) std::this_thread::yield();
  std::thread waiter_two([&] {
    second_waiter_started.store(true);
    auto admission = queue.Acquire();
    GEM16_CHECK(admission.ok());
    if (admission.ok()) {
      std::lock_guard lock(order_mutex);
      order.push_back(2);
    }
  });
  while (!second_waiter_started.load()) std::this_thread::yield();
  while (queue.Snapshot().queued != 2U) std::this_thread::yield();

  first = gem16::server::RequestAdmission();
  waiter_one.join();
  waiter_two.join();
  GEM16_CHECK(order.size() == 2U);
  if (order.size() == 2U) {
    GEM16_CHECK(order[0] == 1);
    GEM16_CHECK(order[1] == 2);
  }
  GEM16_CHECK(queue.Snapshot().active == 0U);
  GEM16_CHECK(queue.Snapshot().high_watermark == 2U);
}

void TestBoundedAndDraining() {
  gem16::server::RequestQueue bounded(1U, 1U);
  auto active = bounded.Acquire();
  GEM16_CHECK(active.ok());
  std::atomic<bool> waiter_started{false};
  std::atomic<bool> waiter_cancelled{false};
  std::thread waiter([&] {
    waiter_started.store(true);
    auto result = bounded.Acquire();
    waiter_cancelled.store(
        !result.ok() && result.status().code() ==
                            gem16::StatusCode::kResourceExhausted);
  });
  while (!waiter_started.load()) std::this_thread::yield();
  while (bounded.Snapshot().queued != 1U) std::this_thread::yield();

  auto overflow = bounded.Acquire();
  GEM16_CHECK(!overflow.ok());
  if (!overflow.ok()) {
    GEM16_CHECK(overflow.status().code() ==
                gem16::StatusCode::kResourceExhausted);
  }
  bounded.StartDraining();
  waiter.join();
  GEM16_CHECK(waiter_cancelled.load());
  GEM16_CHECK(bounded.Snapshot().draining);
  GEM16_CHECK(bounded.Snapshot().queued == 0U);
}

void TestParallelCapacity() {
  gem16::server::RequestQueue queue(2U, 1U);
  auto first = queue.Acquire();
  auto second = queue.Acquire();
  GEM16_CHECK(first.ok());
  GEM16_CHECK(second.ok());
  GEM16_CHECK(queue.Snapshot().active == 2U);
}

}  // namespace

void RunRequestQueueTests() {
  TestFifoAndCapacity();
  TestBoundedAndDraining();
  TestParallelCapacity();
}

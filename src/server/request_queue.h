#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>

#include "gem16/status.h"

namespace gem16::server {

class RequestQueue;

class RequestAdmission {
 public:
  RequestAdmission() = default;
  RequestAdmission(const RequestAdmission&) = delete;
  RequestAdmission& operator=(const RequestAdmission&) = delete;
  RequestAdmission(RequestAdmission&& other) noexcept;
  RequestAdmission& operator=(RequestAdmission&& other) noexcept;
  ~RequestAdmission();

  [[nodiscard]] std::uint64_t wait_microseconds() const {
    return wait_microseconds_;
  }

 private:
  friend class RequestQueue;
  RequestAdmission(RequestQueue* queue, std::uint64_t wait_microseconds)
      : queue_(queue), wait_microseconds_(wait_microseconds) {}
  void Release();

  RequestQueue* queue_ = nullptr;
  std::uint64_t wait_microseconds_ = 0U;
};

struct RequestQueueSnapshot {
  std::size_t active = 0U;
  std::size_t queued = 0U;
  std::size_t capacity = 0U;
  std::size_t max_queued = 0U;
  std::size_t high_watermark = 0U;
  bool draining = false;
};

// Bounded FIFO admission for the finite set of resident execution slots.
// Cancellation and observability endpoints intentionally bypass this queue.
class RequestQueue {
 public:
  RequestQueue(std::size_t capacity, std::size_t max_queued);
  RequestQueue(const RequestQueue&) = delete;
  RequestQueue& operator=(const RequestQueue&) = delete;

  [[nodiscard]] Result<RequestAdmission> Acquire(
      std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max(),
      const std::function<bool()>& cancelled = {});
  void StartDraining();
  [[nodiscard]] RequestQueueSnapshot Snapshot() const;

 private:
  friend class RequestAdmission;
  void Release();

  const std::size_t capacity_;
  const std::size_t max_queued_;
  mutable std::mutex mutex_;
  std::condition_variable available_;
  std::deque<std::uint64_t> waiters_;
  std::uint64_t next_ticket_ = 1U;
  std::size_t active_ = 0U;
  std::size_t high_watermark_ = 0U;
  bool draining_ = false;
};

}  // namespace gem16::server

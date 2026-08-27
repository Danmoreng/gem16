#include "server/request_queue.h"

#include <algorithm>
#include <utility>

namespace gem16::server {

RequestAdmission::RequestAdmission(RequestAdmission&& other) noexcept
    : queue_(std::exchange(other.queue_, nullptr)),
      wait_microseconds_(other.wait_microseconds_) {}

RequestAdmission& RequestAdmission::operator=(
    RequestAdmission&& other) noexcept {
  if (this == &other) return *this;
  Release();
  queue_ = std::exchange(other.queue_, nullptr);
  wait_microseconds_ = other.wait_microseconds_;
  return *this;
}

RequestAdmission::~RequestAdmission() { Release(); }

void RequestAdmission::Release() {
  if (queue_ == nullptr) return;
  RequestQueue* queue = std::exchange(queue_, nullptr);
  queue->Release();
}

RequestQueue::RequestQueue(std::size_t capacity, std::size_t max_queued)
    : capacity_(std::max<std::size_t>(capacity, 1U)),
      max_queued_(max_queued) {}

Result<RequestAdmission> RequestQueue::Acquire() {
  const auto started = std::chrono::steady_clock::now();
  std::unique_lock lock(mutex_);
  if (draining_) {
    return Status(StatusCode::kResourceExhausted, "server is draining");
  }
  if (waiters_.empty() && active_ < capacity_) {
    ++active_;
    return RequestAdmission(this, 0U);
  }
  if (waiters_.size() >= max_queued_) {
    return Status(StatusCode::kResourceExhausted,
                  "request queue is full");
  }

  const std::uint64_t ticket = next_ticket_++;
  waiters_.push_back(ticket);
  high_watermark_ = std::max(high_watermark_, waiters_.size());
  available_.wait(lock, [&] {
    return draining_ ||
           (!waiters_.empty() && waiters_.front() == ticket &&
            active_ < capacity_);
  });
  if (draining_) {
    const auto found = std::find(waiters_.begin(), waiters_.end(), ticket);
    if (found != waiters_.end()) waiters_.erase(found);
    available_.notify_all();
    return Status(StatusCode::kResourceExhausted, "server is draining");
  }
  waiters_.pop_front();
  ++active_;
  available_.notify_all();
  const auto waited = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - started);
  return RequestAdmission(this, static_cast<std::uint64_t>(waited.count()));
}

void RequestQueue::Release() {
  {
    std::lock_guard lock(mutex_);
    if (active_ != 0U) --active_;
  }
  available_.notify_all();
}

void RequestQueue::StartDraining() {
  {
    std::lock_guard lock(mutex_);
    draining_ = true;
  }
  available_.notify_all();
}

RequestQueueSnapshot RequestQueue::Snapshot() const {
  std::lock_guard lock(mutex_);
  return {active_,       waiters_.size(), capacity_, max_queued_,
          high_watermark_, draining_};
}

}  // namespace gem16::server

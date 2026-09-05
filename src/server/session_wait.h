#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>

#include "server/request_queue.h"

namespace gem16::server {

// One HTTP admission budget also covers waits for an existing resident session.
struct SessionWaitOptions {
  std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::time_point::max();
  std::function<bool()> cancelled;

  [[nodiscard]] Status Check(const RequestQueue& queue) const {
    if (cancelled && cancelled()) {
      return Status(StatusCode::kCancelled,
                    "request cancelled while waiting for a session");
    }
    if (queue.Snapshot().draining) {
      return Status(StatusCode::kResourceExhausted, "server is draining");
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return Status(StatusCode::kResourceExhausted,
                    "request admission deadline exceeded");
    }
    return Status::Ok();
  }

  template <typename Ready>
  [[nodiscard]] Status Wait(std::condition_variable& changed,
                            std::unique_lock<std::mutex>& lock,
                            const RequestQueue& queue, Ready ready) const {
    for (;;) {
      const Status status = Check(queue);
      if (!status.ok()) return status;
      if (ready()) return Status::Ok();
      changed.wait_until(lock, std::min(deadline,
          std::chrono::steady_clock::now() + std::chrono::milliseconds(25)));
    }
  }
};

}  // namespace gem16::server

#pragma once

#include "platform_process.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

namespace gem16::studio {

struct ModelInstallState {
  bool target_26b_ready = false;
  bool assistant_26b_ready = false;
  bool downloading = false;
  bool cancel_requested = false;
  std::string current_file;
  std::string error;
  std::uint64_t completed_bytes = 0;
  std::uint64_t total_bytes = 0;

  [[nodiscard]] bool All26BReady() const {
    return target_26b_ready && assistant_26b_ready;
  }
};

class ModelManager final {
 public:
  ModelManager();
  ~ModelManager();
  ModelManager(const ModelManager&) = delete;
  ModelManager& operator=(const ModelManager&) = delete;

  [[nodiscard]] ModelInstallState State() const;
  void Refresh();
  void DownloadQualified26B();
  void Cancel();

 private:
  void DownloadWorker();

  mutable std::mutex mutex_;
  ModelInstallState state_;
  std::jthread worker_;
  std::atomic<bool> cancel_{false};
  PlatformProcess* active_process_ = nullptr;
};

}  // namespace gem16::studio

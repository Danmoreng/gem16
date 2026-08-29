#pragma once

#include "platform_process.h"
#include "types.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace gem16::studio {

struct ProfileInstallState {
  bool target_ready = false;
  bool assistant_ready = false;
  bool storage_available = false;
  bool sufficient_space = false;
  std::uint64_t completed_bytes = 0;
  std::uint64_t total_bytes = 0;
  std::uint64_t required_download_bytes = 0;
  std::uint64_t available_disk_bytes = 0;

  [[nodiscard]] bool Ready() const { return target_ready && assistant_ready; }
};

struct ModelInstallState {
  std::array<ProfileInstallState, 2> profiles{};
  bool downloading = false;
  bool cancel_requested = false;
  ModelProfile downloading_profile = ModelProfile::kGemma4Unified12B;
  std::string current_file;
  std::string error;

  [[nodiscard]] const ProfileInstallState& For(ModelProfile profile) const {
    return profiles[profile == ModelProfile::kGemma4Moe26BA4B ? 1U : 0U];
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
  void DownloadProfile(ModelProfile profile);
  void Cancel();

 private:
  void DownloadWorker(ModelProfile profile);

  mutable std::mutex mutex_;
  ModelInstallState state_;
  std::jthread worker_;
  std::atomic<bool> cancel_{false};
  std::shared_ptr<PlatformProcess> active_process_;
};

}  // namespace gem16::studio

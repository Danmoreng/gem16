#pragma once

#include "model_catalog.h"
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
  std::array<bool, kModelComponentKindCount> component_ready{};
  bool required_ready = false;
  bool storage_available = false;
  bool sufficient_space = false;
  std::uint64_t completed_bytes = 0;
  std::uint64_t total_bytes = 0;
  std::uint64_t required_download_bytes = 0;
  std::uint64_t available_disk_bytes = 0;

  [[nodiscard]] bool ComponentReady(ModelComponentKind kind) const {
    return component_ready[ModelComponentKindIndex(kind)];
  }
  [[nodiscard]] bool Ready() const { return required_ready; }
};

struct ModelInstallState {
  std::array<ProfileInstallState, kModelProfileCount> profiles{};
  bool downloading = false;
  bool cancel_requested = false;
  ModelProfile downloading_profile = ModelProfile::kGemma4Unified12B;
  std::string current_file;
  std::string error;

  [[nodiscard]] const ProfileInstallState& For(ModelProfile profile) const {
    return profiles[ModelProfileIndex(profile)];
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
  void RemoveComponent(ModelProfile profile, ModelComponentKind kind);
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

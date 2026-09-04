#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "model_cache.h"
#include "model_catalog.h"
#include "platform_process.h"
#include "types.h"

namespace gem16::studio {

enum class ComponentInstallStatus {
  kMissing,
  kPartial,
  kUnverified,
  kDamaged,
  kVerified
};
const char* ComponentStatusLabel(ComponentInstallStatus status);
struct ProfileInstallState {
  std::array<ComponentInstallStatus, kModelComponentKindCount>
      component_status{};
  bool all_ready = false;
  std::uint64_t unique_bytes = 0, shared_bytes = 0;
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
  bool maintenance = false;
  bool cache_review_ready = false;
  CacheCleanupPlan cache_plan;
  std::string cache_status;
  bool Busy() const { return downloading || verifying || maintenance; }
  bool downloading = false;
  bool verifying = false;
  std::uint64_t verification_bytes = 0;
  std::uint64_t verification_total_bytes = 0;
  std::string verification_status;
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
  // Catalog storage must outlive this manager and its worker. Production uses
  // the static lock-derived catalog.
  explicit ModelManager(
      std::span<const ModelProfileCatalog> catalog = ModelCatalog());
  ~ModelManager();
  ModelManager(const ModelManager&) = delete;
  ModelManager& operator=(const ModelManager&) = delete;

  [[nodiscard]] ModelInstallState State() const;
  void Refresh();
  void VerifyInstalled();
  void DownloadProfile(ModelProfile profile);
  void ReviewCache();
  void CleanCache();
  void RemoveProfile(ModelProfile profile);
  void RemoveComponent(ModelProfile profile, ModelComponentKind kind);
  void Cancel();

 private:
  void DownloadWorker(ModelProfile profile);

  std::span<const ModelProfileCatalog> catalog_;
  mutable std::mutex mutex_;
  ModelInstallState state_;
  std::jthread worker_;
  std::atomic<bool> cancel_{false};
  std::shared_ptr<PlatformProcess> active_process_;
};

}  // namespace gem16::studio

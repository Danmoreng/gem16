#pragma once
#include <cstdint>
#include <filesystem>
#include <vector>

#include "model_catalog.h"

namespace gem16::studio {
// Same per-blob lock location as the Hub client. Fail closed on unsupported
// locks.
class HubBlobLock {
 public:
  explicit HubBlobLock(const std::filesystem::path& path);
  ~HubBlobLock();
  HubBlobLock(const HubBlobLock&) = delete;
  HubBlobLock& operator=(const HubBlobLock&) = delete;
  bool Locked() const { return locked_; }

 private:
  std::intptr_t handle_ = -1;
  bool locked_ = false;
};
std::filesystem::path HubBlobLockPath(const ModelCatalogFile& file);
struct CacheCleanupFile {
  std::filesystem::path path, lock;
  std::uint64_t bytes = 0;
  std::filesystem::file_time_type modified;
};
struct CacheCleanupPlan {
  std::vector<CacheCleanupFile> files;
  std::uint64_t cached_bytes = 0, reclaimable_bytes = 0, partial_bytes = 0;
};
CacheCleanupPlan InspectModelCache(
    std::span<const ModelProfileCatalog> catalog);
std::uint64_t CleanModelCache(const CacheCleanupPlan& approved,
                              std::span<const ModelProfileCatalog> catalog);
}  // namespace gem16::studio

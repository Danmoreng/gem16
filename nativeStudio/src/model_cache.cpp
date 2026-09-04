#include "model_cache.h"

#include <set>
#include <stdexcept>

#include "settings.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace gem16::studio {
namespace {
void RejectParentLinks(const std::filesystem::path& path) {
  for (auto p = path; !p.empty() && p != p.root_path(); p = p.parent_path())
    if (std::filesystem::is_symlink(std::filesystem::symlink_status(p)))
      throw std::runtime_error(
          "Model cache maintenance refuses symlinked directories.");
}
std::set<std::filesystem::path> LinkTargets(const std::filesystem::path& hub) {
  std::set<std::filesystem::path> targets;
  if (!std::filesystem::exists(hub)) return targets;
  RejectParentLinks(hub);
  std::size_t entries = 0;
  for (const auto& entry : std::filesystem::recursive_directory_iterator(hub)) {
    if (++entries > 1000000)
      throw std::runtime_error(
          "Cache scan exceeded one million entries; nothing was removed.");
    if (!entry.is_symlink()) continue;
    if (entry.is_directory())
      throw std::runtime_error(
          "Cache has directory symlinks; cleanup requires an unambiguous "
          "reference scan.");
    targets.insert(std::filesystem::weakly_canonical(entry.path()));
  }
  return targets;
}
bool Unreferenced(const std::filesystem::path& file,
                  const std::set<std::filesystem::path>& targets) {
  return !std::filesystem::is_symlink(std::filesystem::symlink_status(file)) &&
         std::filesystem::is_regular_file(file) &&
         std::filesystem::hard_link_count(file) == 1 &&
         !targets.contains(std::filesystem::weakly_canonical(file));
}
}  // namespace
HubBlobLock::HubBlobLock(const std::filesystem::path& path) {
  RejectParentLinks(path);
  std::filesystem::create_directories(path.parent_path());
#ifdef _WIN32
  handle_ = reinterpret_cast<std::intptr_t>(
      CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE,
                  FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS,
                  FILE_ATTRIBUTE_NORMAL, nullptr));
  if (handle_ != -1) {
    OVERLAPPED offset{};
    locked_ = LockFileEx(reinterpret_cast<HANDLE>(handle_),
                         LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0,
                         1, 0, &offset) != 0;
  }
#else
  handle_ = open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (handle_ != -1)
    locked_ = flock(static_cast<int>(handle_), LOCK_EX | LOCK_NB) == 0;
#endif
}
HubBlobLock::~HubBlobLock() {
#ifdef _WIN32
  if (handle_ != -1) CloseHandle(reinterpret_cast<HANDLE>(handle_));
#else
  if (handle_ != -1) close(static_cast<int>(handle_));
#endif
}
std::filesystem::path HubBlobLockPath(const ModelCatalogFile& file) {
  const auto hub = HuggingFaceHubRoot();
  return hub / ".locks" /
         RepositoryDirectory(file.source_repository, hub).filename() /
         (std::string(file.blob_id) + ".lock");
}
CacheCleanupPlan InspectModelCache(
    std::span<const ModelProfileCatalog> catalog) {
  CacheCleanupPlan plan;
  const auto hub = HuggingFaceHubRoot();
  const auto targets = LinkTargets(hub);
  std::set<std::filesystem::path> seen;
  for (const auto& profile : catalog)
    for (const auto& component : profile.components)
      for (const auto& file : component.catalog->files) {
        const auto blob = RepositoryDirectory(file.source_repository, hub) /
                          "blobs" / file.blob_id;
        for (const auto& path :
             {blob, std::filesystem::path(blob.string() + ".incomplete")}) {
          if (!seen.insert(path).second) continue;
          RejectParentLinks(path.parent_path());
          if (std::filesystem::is_symlink(
                  std::filesystem::symlink_status(path)) ||
              !std::filesystem::is_regular_file(path))
            continue;
          auto bytes = std::filesystem::file_size(path);
          plan.cached_bytes += bytes;
          if (path != blob) plan.partial_bytes += bytes;
          if (Unreferenced(path, targets)) {
            plan.files.push_back({path, HubBlobLockPath(file), bytes,
                                  std::filesystem::last_write_time(path)});
            plan.reclaimable_bytes += bytes;
          }
        }
      }
  return plan;
}
std::uint64_t CleanModelCache(const CacheCleanupPlan& approved,
                              std::span<const ModelProfileCatalog> catalog) {
  const auto current = InspectModelCache(catalog);
  std::uint64_t freed = 0;
  for (const auto& candidate : current.files) {
    bool accepted = false;
    for (const auto& old : approved.files)
      if (old.path == candidate.path && old.bytes == candidate.bytes &&
          old.modified == candidate.modified) {
        accepted = true;
        break;
      }
    if (!accepted) continue;
    HubBlobLock lock(candidate.lock);
    if (!lock.Locked()) continue;
    // Recheck references under the download lock, including other Hub
    // repositories.
    const auto targets = LinkTargets(HuggingFaceHubRoot());
    if (!Unreferenced(candidate.path, targets) ||
        std::filesystem::file_size(candidate.path) != candidate.bytes ||
        std::filesystem::last_write_time(candidate.path) != candidate.modified)
      continue;
    if (std::filesystem::remove(candidate.path)) freed += candidate.bytes;
  }
  return freed;
}
}  // namespace gem16::studio

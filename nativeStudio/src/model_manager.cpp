#include "model_manager.h"

#include "compiler/sha256.h"
#include "model_catalog.h"
#include "settings.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <string_view>
#include <system_error>
#include <vector>

namespace gem16::studio {
namespace {

constexpr std::uint64_t kDownloadReserveBytes = 256ULL * 1024ULL * 1024ULL;

bool SafeRelativePath(std::string_view value) {
  if (value.empty()) return false;
  const std::filesystem::path path(value);
  if (path.is_absolute() || path.has_root_name() || path.has_root_directory()) return false;
  for (const auto& part : path) {
    if (part == "." || part == "..") return false;
  }
  return true;
}

std::filesystem::path SourceRepositoryDirectory(const ModelCatalogFile& file) {
  return RepositoryDirectory(file.source_repository, HuggingFaceHubRoot());
}

std::filesystem::path SourceSnapshotPath(const ModelCatalogFile& file) {
  return SourceRepositoryDirectory(file) / "snapshots" / file.source_revision /
         file.source_path;
}

std::filesystem::path BlobPath(const ModelCatalogFile& file) {
  return SourceRepositoryDirectory(file) / "blobs" / file.blob_id;
}

std::filesystem::path VerificationMarker(const ModelCatalogFile& file) {
  return VerificationMarkerPath(file, HuggingFaceHubRoot());
}

bool HasSize(const std::filesystem::path& path, std::uint64_t size) {
  std::error_code error;
  const auto observed = std::filesystem::file_size(path, error);
  return !error && observed == size;
}

bool HasVerificationMarker(const ModelCatalogFile& file) {
  std::ifstream input(VerificationMarker(file));
  std::string hash;
  std::getline(input, hash);
  return hash == file.sha256;
}

bool HasDamageMarker(const ModelCatalogFile& file) {
  std::error_code error;
  const bool present=std::filesystem::exists(std::filesystem::path(VerificationMarker(file).string()+".damaged"),error);
  return present || static_cast<bool>(error);
}
bool FileReady(const ModelComponentCatalog& component,
               const ModelCatalogFile& file) {
  return HasSize(
             ComponentDirectory(component, HuggingFaceHubRoot()) / file.path,
             file.size) &&
         HasSize(BlobPath(file), file.size) && HasVerificationMarker(file) &&
         !HasDamageMarker(file);
}

ComponentInstallStatus ComponentStatus(const ModelComponentCatalog& component) {
  bool all_ready = true, all_present = true, any_present = false,
       damaged = false;
  for (const auto& file : component.files) {
    std::error_code ec;
    const auto blob = BlobPath(file),
               view = ComponentDirectory(component, HuggingFaceHubRoot()) /
                      file.path;
    const bool present = HasSize(blob, file.size) && HasSize(view, file.size);
    all_present &= present;
    all_ready &= FileReady(component, file);
    any_present |=
        std::filesystem::exists(blob, ec) ||
        std::filesystem::exists(view, ec) ||
        std::filesystem::exists(
            std::filesystem::path(blob.string() + ".incomplete"), ec);
    damaged |= std::filesystem::exists(
                   std::filesystem::path(VerificationMarker(file).string() +
                                         ".damaged"),
                   ec) ||
               (std::filesystem::is_regular_file(blob, ec) &&
                !HasSize(blob, file.size)) ||
               (std::filesystem::is_regular_file(view, ec) &&
                !HasSize(view, file.size));
  }
  if (damaged) return ComponentInstallStatus::kDamaged;
  if (all_ready) return ComponentInstallStatus::kVerified;
  if (all_present) return ComponentInstallStatus::kUnverified;
  return any_present ? ComponentInstallStatus::kPartial
                     : ComponentInstallStatus::kMissing;
}
const ModelProfileCatalog& FindProfile(
    std::span<const ModelProfileCatalog> catalog, ModelProfile profile) {
  for (const auto& entry : catalog)
    if (entry.profile == profile) return entry;
  throw std::runtime_error("Unknown model profile.");
}

std::uint64_t ComponentBytes(const ModelComponentCatalog& component) {
  std::uint64_t result = 0;
  for (const auto& file : component.files) result += file.size;
  return result;
}

std::uint64_t ReadyBytes(const ModelProfileCatalog& profile) {
  std::uint64_t result = 0;
  for (const auto& profile_component : profile.components) {
    const auto& component = *profile_component.catalog;
    for (const auto& file : component.files) {
      if (FileReady(component, file)) result += file.size;
    }
  }
  return result;
}

std::uint64_t RequiredDownloadBytes(const ModelProfileCatalog& profile) {
  std::uint64_t result = 0;
  std::set<std::filesystem::path> missing_blobs;
  for (const auto& profile_component : profile.components) {
    for (const auto& file : profile_component.catalog->files) {
      const auto blob = BlobPath(file);
      if ((!HasSize(blob, file.size) ||
           HasDamageMarker(file)) &&
          missing_blobs.insert(blob).second) {
        std::error_code ec;
        const auto partial =
            std::filesystem::path(blob.string() + ".incomplete");
        const auto size = std::filesystem::file_size(partial, ec);
        result +=
            file.size - (ec ? 0 : std::min<std::uint64_t>(size, file.size));
      }
    }
  }
  return result;
}

std::optional<std::uint64_t> AvailableDiskBytes() {
  std::filesystem::path probe = HuggingFaceHubRoot();
  std::error_code error;
  while (!probe.empty() && !std::filesystem::exists(probe, error)) {
    error.clear();
    const auto parent = probe.parent_path();
    if (parent == probe) break;
    probe = parent;
  }
  if (probe.empty()) return std::nullopt;
  const auto space = std::filesystem::space(probe, error);
  if (error) return std::nullopt;
  return space.available;
}

ProfileInstallState InspectProfile(const ModelProfileCatalog& profile,
                                   std::span<const ModelProfileCatalog> all) {
  ProfileInstallState state;
  state.required_ready = true;
  state.all_ready = true;
  std::set<std::filesystem::path> unique;
  for (const auto& profile_component : profile.components) {
    const auto status = ComponentStatus(*profile_component.catalog);
    const bool ready = status == ComponentInstallStatus::kVerified;
    state.component_status[ModelComponentKindIndex(profile_component.kind)] =
        status;
    state.all_ready &= ready;
    for (const auto& file : profile_component.catalog->files)
      if (unique.insert(BlobPath(file)).second) {
        state.unique_bytes += file.size;
        bool shared = false;
        for (const auto& other : all)
          if (other.profile != profile.profile)
            for (const auto& component : other.components)
              for (const auto& other_file : component.catalog->files)
                if (BlobPath(file) == BlobPath(other_file)) shared = true;
        if (shared) state.shared_bytes += file.size;
      }
    state.component_ready[ModelComponentKindIndex(profile_component.kind)] =
        ready;
    if (profile_component.required && !ready) state.required_ready = false;
    state.total_bytes += ComponentBytes(*profile_component.catalog);
  }
  state.completed_bytes = ReadyBytes(profile);
  state.required_download_bytes = RequiredDownloadBytes(profile);
  if (const auto available = AvailableDiskBytes()) {
    state.storage_available = true;
    state.available_disk_bytes = *available;
    state.sufficient_space =
        state.required_download_bytes <= *available &&
        kDownloadReserveBytes <= *available - state.required_download_bytes;
  }
  return state;
}

std::optional<std::filesystem::path> CurlExecutable() {
  const char* path_value = std::getenv("PATH");
  if (!path_value) return std::nullopt;
#ifdef _WIN32
  constexpr char separator = ';';
  constexpr const char* executable = "curl.exe";
#else
  constexpr char separator = ':';
  constexpr const char* executable = "curl";
#endif
  std::string_view paths(path_value);
  while (!paths.empty()) {
    const auto end = paths.find(separator);
    const auto directory = paths.substr(0, end);
    if (!directory.empty()) {
      const auto candidate = std::filesystem::path(directory) / executable;
      if (std::filesystem::is_regular_file(candidate)) return candidate;
    }
    if (end == std::string_view::npos) break;
    paths.remove_prefix(end + 1);
  }
  return std::nullopt;
}

std::string FileSha256(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return {};
  gem16::compiler::Sha256 digest;
  std::vector<char> buffer(8 * 1024 * 1024);
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (count > 0) digest.Update(buffer.data(), static_cast<std::size_t>(count));
  }
  return input.eof() ? digest.HexDigest() : std::string{};
}

void PruneLegacyVisionViewAuxiliaryLinks() {
  const auto& profile = CatalogForProfile(
      ModelProfile::kGemma4Moe26BTrellis35VisionFp8);
  const auto* vision =
      ComponentForProfile(profile, ModelComponentKind::kVision);
  if (vision == nullptr) return;
  const auto root = ComponentDirectory(*vision->catalog,
                                       HuggingFaceHubRoot());
  constexpr std::array<std::string_view, 3> legacy_files{
      "LICENSE", "NOTICE", "README.md"};
  for (const std::string_view name : legacy_files) {
    const auto path = root / name;
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (!error && (std::filesystem::is_regular_file(status) ||
                   std::filesystem::is_symlink(status))) {
      std::filesystem::remove(path, error);
    }
  }
}

bool LinkVerifiedFile(const std::filesystem::path& blob,
                      const std::filesystem::path& destination,
                      std::string& error_message) {
  std::error_code error;
  std::filesystem::create_directories(destination.parent_path(), error);
  if (error) {
    error_message = "Could not create snapshot directory: " + error.message();
    return false;
  }
  std::filesystem::remove(destination, error);
  if (error) {
    error_message = "Could not replace snapshot entry: " + error.message();
    return false;
  }
  std::filesystem::create_hard_link(blob, destination, error);
  if (error) {
    error_message = "Could not hardlink verified Hub blob: " + error.message();
    return false;
  }
  return true;
}

}  // namespace

ModelManager::ModelManager(std::span<const ModelProfileCatalog> catalog) : catalog_(catalog) {
  PruneLegacyVisionViewAuxiliaryLinks();
  Refresh();
}

ModelManager::~ModelManager() {
  Cancel();
  if (worker_.joinable()) worker_.join();
}

ModelInstallState ModelManager::State() const {
  std::scoped_lock lock(mutex_);
  return state_;
}

void ModelManager::Refresh() {
  std::scoped_lock lock(mutex_);
  if (state_.Busy()) return;
  for (const auto& profile : catalog_) {
    state_.profiles[ModelProfileIndex(profile.profile)] =
        InspectProfile(profile, catalog_);
  }
}

void ModelManager::VerifyInstalled() {
  {
    std::scoped_lock lock(mutex_);
    if (state_.Busy()) return;
    state_.verifying = true;
    state_.verification_bytes = 0;
    state_.verification_total_bytes = 0;
    state_.verification_status.clear();
    state_.error.clear();
    state_.cancel_requested = false;
  }
  cancel_.store(false);
  if (worker_.joinable()) worker_.join();
  worker_ = std::jthread([this] {
    try {
      struct File {
        std::filesystem::path path, marker;
        std::uint64_t size;
        std::string hash;
      };
      std::vector<File> files;
      std::set<std::filesystem::path> seen;
      const auto hub = HuggingFaceHubRoot();
      std::uint64_t total = 0;
      for (const auto& profile : catalog_) {
        for (const auto& component : profile.components) {
          for (const auto& file : component.catalog->files) {
            const auto blob = BlobPath(file);
            const auto view =
                ComponentDirectory(*component.catalog, hub) / file.path;
            for (const auto& path : {blob, view}) {
              std::error_code ec;
              if (!std::filesystem::exists(path, ec) ||
                  !seen.insert(path).second)
                continue;
              // Hash a hardlinked/symlinked runtime view only once with its
              // blob.
              if (path != blob && std::filesystem::equivalent(path, blob, ec))
                continue;
              files.push_back(
                  {path, VerificationMarker(file), file.size, file.sha256});
              total += file.size;
            }
          }
        }
      }
      {
        std::scoped_lock lock(mutex_);
        state_.verification_total_bytes = total;
      }
      std::vector<char> buffer(8U * 1024U * 1024U);
      std::set<std::filesystem::path> invalid_markers;
      std::string failure;
      std::uint64_t completed = 0;
      for (const auto& file : files) {
        if (cancel_.load()) break;
        {
          std::scoped_lock lock(mutex_);
          state_.current_file = file.path.filename().string();
        }
        std::error_code ec;
        std::filesystem::remove(file.marker, ec);
        if (ec) {
          failure = "Could not invalidate verification marker: " + ec.message();
          break;
        }
        gem16::compiler::Sha256 digest;
        std::ifstream input(file.path, std::ios::binary);
        std::uint64_t read = 0;
        while (input && read < file.size && !cancel_.load()) {
          input.read(buffer.data(),
                     static_cast<std::streamsize>(std::min<std::uint64_t>(
                         buffer.size(), file.size - read)));
          const auto count = input.gcount();
          if (count > 0) {
            digest.Update(buffer.data(), static_cast<std::size_t>(count));
            read += static_cast<std::uint64_t>(count);
          }
          std::scoped_lock lock(mutex_);
          state_.verification_bytes = completed + read;
        }
        if (cancel_.load()) break;
        const bool valid = read == file.size && HasSize(file.path, file.size) &&
                           digest.HexDigest() == file.hash;
        if (!valid) {
          invalid_markers.insert(file.marker);
          std::filesystem::create_directories(file.marker.parent_path(), ec);
          std::ofstream(
              std::filesystem::path(file.marker.string() + ".damaged"))
              << file.hash << '\n';
          if (failure.empty())
            failure =
                "Size or SHA-256 verification failed: " + file.path.string();
        } else if (!invalid_markers.contains(file.marker)) {
          std::filesystem::remove(
              std::filesystem::path(file.marker.string() + ".damaged"), ec);
          std::filesystem::create_directories(file.marker.parent_path(), ec);
          std::ofstream marker(file.marker, std::ios::trunc);
          marker << file.hash << '\n';
          if (ec || !marker)
            failure =
                "Could not write verification marker: " + file.marker.string();
        }
        completed += file.size;
      }
      // A bad detached view invalidates the shared marker even if another view
      // passed.
      for (const auto& marker : invalid_markers) {
        std::error_code ec;
        std::filesystem::remove(marker, ec);
      }
      std::scoped_lock lock(mutex_);
      for (const auto& profile : catalog_)
        state_.profiles[ModelProfileIndex(profile.profile)] =
            InspectProfile(profile, catalog_);
      state_.error = std::move(failure);
      state_.verification_status =
          cancel_.load() ? "Verification cancelled"
          : !state_.error.empty()
              ? "Verification failed; reinstall the affected component"
          : files.empty() ? "No installed files to verify"
                          : "SHA-256 verification complete";
      state_.current_file.clear();
      state_.verifying = false;
    } catch (const std::exception& e) {
      std::scoped_lock lock(mutex_);
      state_.verifying = false;
      state_.error = e.what();
    }
  });
}

void ModelManager::DownloadProfile(ModelProfile profile) {
  {
    std::scoped_lock lock(mutex_);
    if (state_.Busy()) return;
    const auto inspected =
        InspectProfile(FindProfile(catalog_, profile), catalog_);
    state_.profiles[ModelProfileIndex(profile)] = inspected;
    state_.error.clear();
    if (!inspected.storage_available) {
      state_.error = "Could not determine free space for the Hugging Face cache";
      return;
    }
    if (!inspected.sufficient_space) {
      state_.error = "Not enough free space in the Hugging Face cache filesystem";
      return;
    }
    state_.downloading = true;
    state_.downloading_profile = profile;
    state_.cancel_requested = false;
    state_.current_file.clear();
  }
  cancel_.store(false);
  if (worker_.joinable()) worker_.join();
  worker_ = std::jthread([this, profile] {
    try {
      DownloadWorker(profile);
    } catch (const std::exception& e) {
      std::scoped_lock lock(mutex_);
      state_.downloading = false;
      state_.error = e.what();
      active_process_.reset();
    }
  });
}

void ModelManager::RemoveComponent(ModelProfile profile,
                                   ModelComponentKind kind) {
  std::scoped_lock lock(mutex_);
  if (state_.Busy()) return;
  const auto* profile_component =
      ComponentForProfile(FindProfile(catalog_, profile), kind);
  if (!profile_component) return;
  for(const auto& other:catalog_)if(other.profile!=profile&&state_.For(other.profile).Ready())
    for(const auto& shared:other.components)if(shared.catalog==profile_component->catalog){state_.error="This component is shared by another installed profile and was kept.";return;}
  const auto root = ComponentDirectory(*profile_component->catalog,
                                       HuggingFaceHubRoot());
  std::error_code error;
  for(auto parent=root;!parent.empty()&&parent!=parent.root_path();parent=parent.parent_path()) {
    const bool linked=std::filesystem::is_symlink(std::filesystem::symlink_status(parent,error));
    if(linked || (error && error!=std::errc::no_such_file_or_directory)){state_.error="Refusing to remove a model view through an inaccessible or symlinked directory.";return;}
    error.clear();
  }
  for (const auto& file : profile_component->catalog->files) {
    std::filesystem::remove(root / file.path, error);
    if (error) {
      state_.error = "Could not remove component view: " + error.message();
      return;
    }
  }
  for (const auto& catalog : catalog_) {
    state_.profiles[ModelProfileIndex(catalog.profile)] =
        InspectProfile(catalog, catalog_);
  }
  state_.error.clear();
}

void ModelManager::Cancel() {
  cancel_.store(true);
  std::shared_ptr<PlatformProcess> process;
  {
    std::scoped_lock lock(mutex_);
    state_.cancel_requested = state_.Busy();
    process = active_process_;
  }
  if (process) process->Stop();
}

void ModelManager::DownloadWorker(ModelProfile selected_profile) {
  const ModelProfileCatalog& profile = FindProfile(catalog_, selected_profile);
  const auto profile_index = ModelProfileIndex(selected_profile);
  const auto finish = [this](std::string error) {
    std::array<ProfileInstallState, kModelProfileCount> inspected{};
    for (const auto& catalog : catalog_) {
      inspected[ModelProfileIndex(catalog.profile)] =
          InspectProfile(catalog, catalog_);
    }
    std::scoped_lock lock(mutex_);
    state_.downloading = false;
    state_.cancel_requested = cancel_.load();
    state_.error = std::move(error);
    state_.current_file.clear();
    state_.profiles = inspected;
  };

  const auto curl = CurlExecutable();
  if (!curl) {
    finish("curl was not found on PATH; install curl to download Hugging Face files");
    return;
  }

  std::uint64_t completed = ReadyBytes(profile);
  for (const auto& profile_component : profile.components) {
    const ModelComponentCatalog* component = profile_component.catalog;
    const auto component_directory =
        ComponentDirectory(*component, HuggingFaceHubRoot());
    for (const auto& file : component->files) {
      if (!SafeRelativePath(file.path) || !SafeRelativePath(file.source_path)) {
        finish("The embedded model catalog contains an unsafe relative path");
        return;
      }
      if (cancel_.load()) {
        finish("Download paused; partial files are retained for resume");
        return;
      }
      if (FileReady(*component, file)) continue;

      const auto blob = BlobPath(file);
      const auto partial = std::filesystem::path(blob.string() + ".incomplete");
      const auto marker = VerificationMarker(file);
      std::error_code error;
      std::filesystem::create_directories(blob.parent_path(), error);
      std::filesystem::create_directories(marker.parent_path(), error);
      if (error) {
        finish("Could not create the Hugging Face cache directories: " +
               error.message());
        return;
      }

      HubBlobLock blob_lock(HubBlobLockPath(file));
      if (!blob_lock.Locked()) {
        finish(
            "This model file is in use by another cache operation; retry "
            "shortly.");
        return;
      }
      bool verified = false;
      if (HasSize(blob, file.size)) {
        verified = (HasVerificationMarker(file) &&
                    !std::filesystem::exists(
                        std::filesystem::path(marker.string() + ".damaged"))) ||
                   FileSha256(blob) == file.sha256;
      }
      if (!verified) {
        if (std::filesystem::exists(blob, error)) {
          std::filesystem::remove(blob, error);
          if (error) {
            finish("Could not replace an invalid Hub blob for " +
                   std::string(file.path) + ": " + error.message());
            return;
          }
        }
        error.clear();
        if (std::filesystem::is_regular_file(partial, error) &&
            std::filesystem::file_size(partial, error) > file.size) {
          std::filesystem::remove(partial, error);
        }
        if (HasSize(partial, file.size)) {
          verified = FileSha256(partial) == file.sha256;
          if (!verified) std::filesystem::remove(partial, error);
        }
        if (!verified) {
          const std::string url =
              "https://huggingface.co/" + std::string(file.source_repository) +
              "/resolve/" + file.source_revision + "/" + file.source_path;
          std::vector<std::string> arguments{
              curl->string(), "--fail", "--location", "--silent", "--show-error",
              "--retry", "5", "--retry-all-errors", "--continue-at", "-",
              "--output", partial.string(), url};
          std::mutex wait_mutex;
          std::condition_variable wait_condition;
          bool exited = false;
          int exit_code = -1;
          auto process = std::make_shared<PlatformProcess>();
          {
            std::scoped_lock lock(mutex_);
            state_.current_file = std::string(component->label) + " · " + file.path;
            state_.profiles[profile_index].completed_bytes = completed;
            active_process_ = process;
          }
          std::string start_error;
          if (!process->Start(arguments, RepositoryRoot().string(),
                              [](std::string) {},
                              [&](int code) {
                                std::scoped_lock wait_lock(wait_mutex);
                                exit_code = code;
                                exited = true;
                                wait_condition.notify_all();
                              },
                              start_error)) {
            {
              std::scoped_lock lock(mutex_);
              active_process_.reset();
            }
            finish("Could not start curl: " + start_error);
            return;
          }
          std::unique_lock wait_lock(wait_mutex);
          while (!exited) {
            wait_condition.wait_for(wait_lock, std::chrono::milliseconds(250));
            std::error_code size_error;
            const auto partial_size =
                std::filesystem::is_regular_file(partial, size_error)
                    ? std::filesystem::file_size(partial, size_error)
                    : 0;
            std::scoped_lock state_lock(mutex_);
            state_.profiles[profile_index].completed_bytes =
                completed + std::min<std::uint64_t>(partial_size, file.size);
          }
          wait_lock.unlock();
          process->Stop();
          {
            std::scoped_lock lock(mutex_);
            active_process_.reset();
          }
          if (cancel_.load()) {
            finish("Download paused; partial files are retained for resume");
            return;
          }
          if (exit_code != 0) {
            finish("Hugging Face download failed for " + std::string(file.path) +
                   " (curl exit " + std::to_string(exit_code) + ")");
            return;
          }
          verified = HasSize(partial, file.size) &&
                     FileSha256(partial) == file.sha256;
        }
        if (!verified) {
          finish("Size or SHA-256 verification failed for " +
                 std::string(file.path));
          return;
        }
        std::filesystem::rename(partial, blob, error);
        if (error) {
          finish("Could not finalize " + std::string(file.path) + ": " +
                 error.message());
          return;
        }
      }

      {
        std::ofstream output(marker, std::ios::trunc);
        output << file.sha256 << '\n';
        if (!output) {
          finish("Could not record verified Hub blob identity for " +
                 std::string(file.path));
          return;
        }
      }
      std::string link_error;
      std::filesystem::remove(
          std::filesystem::path(marker.string() + ".damaged"), error);
      const auto source_snapshot = SourceSnapshotPath(file);
      if (!LinkVerifiedFile(blob, source_snapshot, link_error)) {
        finish(link_error + " (" + std::string(file.path) + ")");
        return;
      }
      const auto component_path = component_directory / file.path;
      if (component_path != source_snapshot &&
          !LinkVerifiedFile(blob, component_path, link_error)) {
        finish(link_error + " (" + std::string(file.path) + ")");
        return;
      }
      completed += file.size;
      {
        std::scoped_lock lock(mutex_);
        state_.profiles[profile_index].completed_bytes = completed;
      }
    }
  }
  finish({});
}

const char* ComponentStatusLabel(ComponentInstallStatus status) {
  switch (status) {
    case ComponentInstallStatus::kMissing:
      return "Missing";
    case ComponentInstallStatus::kPartial:
      return "Partial / resumable";
    case ComponentInstallStatus::kUnverified:
      return "Needs verification";
    case ComponentInstallStatus::kDamaged:
      return "Damaged / repair needed";
    case ComponentInstallStatus::kVerified:
      return "Verified";
  }
  return "Unknown";
}
void ModelManager::ReviewCache() {
  {
    std::scoped_lock lock(mutex_);
    if (state_.Busy()) return;
    state_.maintenance = true;
    state_.cache_review_ready = false;
    state_.cache_status = "Scanning shared-cache references...";
  }
  if (worker_.joinable()) worker_.join();
  worker_ = std::jthread([this] {
    try {
      auto plan = InspectModelCache(catalog_);
      std::scoped_lock lock(mutex_);
      state_.cache_plan = std::move(plan);
      state_.cache_review_ready = true;
      state_.cache_status =
          "Review complete. Referenced blobs and unknown cache files are "
          "retained.";
      state_.maintenance = false;
    } catch (const std::exception& e) {
      std::scoped_lock lock(mutex_);
      state_.cache_status = e.what();
      state_.maintenance = false;
    }
  });
}
void ModelManager::CleanCache() {
  CacheCleanupPlan plan;
  {
    std::scoped_lock lock(mutex_);
    if (state_.Busy() || !state_.cache_review_ready) return;
    plan = state_.cache_plan;
    state_.maintenance = true;
    state_.cache_review_ready = false;
    state_.cache_status = "Rechecking references and cleaning unused files...";
  }
  if (worker_.joinable()) worker_.join();
  worker_ = std::jthread([this, plan = std::move(plan)] {
    try {
      const auto freed = CleanModelCache(plan, catalog_);
      auto current = InspectModelCache(catalog_);
      std::scoped_lock lock(mutex_);
      state_.cache_plan = std::move(current);
      state_.cache_status =
          "Removed " + std::to_string(freed) +
          " bytes. Changed, referenced or locked files were kept.";
      state_.maintenance = false;
      for (const auto& profile : catalog_)
        state_.profiles[ModelProfileIndex(profile.profile)] =
            InspectProfile(profile, catalog_);
    } catch (const std::exception& e) {
      std::scoped_lock lock(mutex_);
      state_.cache_status = e.what();
      state_.maintenance = false;
    }
  });
}
void ModelManager::RemoveProfile(ModelProfile profile) {
  if (State().Busy()) return;
  for (const auto& component : FindProfile(catalog_, profile).components) {
    bool shared = false;
    for (const auto& other : catalog_)
      if (other.profile != profile && State().For(other.profile).Ready())
        for (const auto& used : other.components)
          if (used.catalog == component.catalog) shared = true;
    if (!shared) {
      RemoveComponent(profile, component.kind);
      if (!State().error.empty()) return;
    }
  }
}

}  // namespace gem16::studio

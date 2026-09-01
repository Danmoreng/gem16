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

bool FileReady(const ModelComponentCatalog& component,
               const ModelCatalogFile& file) {
  return HasSize(ComponentDirectory(component, HuggingFaceHubRoot()) / file.path,
                 file.size) &&
         HasSize(BlobPath(file), file.size) && HasVerificationMarker(file);
}

bool ComponentReady(const ModelComponentCatalog& component) {
  return std::ranges::all_of(component.files, [&](const ModelCatalogFile& file) {
    return FileReady(component, file);
  });
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
      if (!HasSize(blob, file.size) && missing_blobs.insert(blob).second)
        result += file.size;
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

ProfileInstallState InspectProfile(const ModelProfileCatalog& profile) {
  ProfileInstallState state;
  state.required_ready = true;
  for (const auto& profile_component : profile.components) {
    const bool ready = ComponentReady(*profile_component.catalog);
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

ModelManager::ModelManager() {
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
  if (state_.downloading) return;
  for (const auto& profile : ModelCatalog()) {
    state_.profiles[ModelProfileIndex(profile.profile)] = InspectProfile(profile);
  }
}

void ModelManager::DownloadProfile(ModelProfile profile) {
  {
    std::scoped_lock lock(mutex_);
    if (state_.downloading) return;
    const auto inspected = InspectProfile(CatalogForProfile(profile));
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
  worker_ = std::jthread([this, profile] { DownloadWorker(profile); });
}

void ModelManager::RemoveComponent(ModelProfile profile,
                                   ModelComponentKind kind) {
  std::scoped_lock lock(mutex_);
  if (state_.downloading) return;
  const auto* profile_component =
      ComponentForProfile(CatalogForProfile(profile), kind);
  if (!profile_component) return;
  const auto root = ComponentDirectory(*profile_component->catalog,
                                       HuggingFaceHubRoot());
  std::error_code error;
  for (const auto& file : profile_component->catalog->files) {
    std::filesystem::remove(root / file.path, error);
    if (error) {
      state_.error = "Could not remove component view: " + error.message();
      return;
    }
  }
  for (const auto& catalog : ModelCatalog()) {
    state_.profiles[ModelProfileIndex(catalog.profile)] =
        InspectProfile(catalog);
  }
  state_.error.clear();
}

void ModelManager::Cancel() {
  cancel_.store(true);
  std::shared_ptr<PlatformProcess> process;
  {
    std::scoped_lock lock(mutex_);
    state_.cancel_requested = state_.downloading;
    process = active_process_;
  }
  if (process) process->Stop();
}

void ModelManager::DownloadWorker(ModelProfile selected_profile) {
  const ModelProfileCatalog& profile = CatalogForProfile(selected_profile);
  const auto profile_index = ModelProfileIndex(selected_profile);
  const auto finish = [this](std::string error) {
    std::array<ProfileInstallState, kModelProfileCount> inspected{};
    for (const auto& catalog : ModelCatalog()) {
      inspected[ModelProfileIndex(catalog.profile)] = InspectProfile(catalog);
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

      bool verified = false;
      if (HasSize(blob, file.size)) {
        verified = HasVerificationMarker(file) || FileSha256(blob) == file.sha256;
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

}  // namespace gem16::studio

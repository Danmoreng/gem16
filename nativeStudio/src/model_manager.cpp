#include "model_manager.h"

#include "compiler/sha256.h"
#include "settings.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <string_view>
#include <system_error>
#include <vector>

namespace gem16::studio {
namespace {

struct LockedFile {
  const char* path;
  std::uint64_t size;
  const char* sha256;
  const char* blob_id;
};

struct LockedRepository {
  const char* repository;
  const char* revision;
  std::span<const LockedFile> files;
};

constexpr std::array kTargetFiles{
    LockedFile{".gitattributes", 95, "f15b349b90d515d0af07dc0d7da57e1a3ef9f00a723c70e8bdb8ff7fcb943b83", "c1ad41cbcce50cb9c28dbc5c41f4b17d8ce29f00"},
    LockedFile{"LICENSE", 634, "e32c5cdb30142b7c670d61be0f2de18ab0007f72ef131e2869ef51687539c86d", "506ff98349a36c9b78904645c9ae492a9b62e826"},
    LockedFile{"NOTICE", 778, "e26e0f5e20bc05d54543aae77d45686b614f948e1af7346597e1b4ef533d990b", "a365f24b0c9e70ebfff6128d45fb9f0c62bba3d5"},
    LockedFile{"README.md", 31418, "7349b325178b75830da550d010c313b2756beb1ebabec5923ba71f108e359a28", "1535491512e42a3ceab0eba250b0810368901ba4"},
    LockedFile{"chat_template.jinja", 18683, "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4", "4741bf6e4132ba23a5537f9d6e74e9a6d613d7cd"},
    LockedFile{"config.json", 4152, "8a647d5444c9e77b03bd80ac802683ffdbf64b3d9296ca5257ced8e50349ea16", "ef0ecec886c4ee7fb9151bbd2a4395973e96c061"},
    LockedFile{"gem16.lock.json", 4231, "d7d2d30743e7c42aa55537a83f563047c9cbd2d83e0d583a2d2c8bcacbfa51a4", "d2f5aa90480db072e7b3258e16abca22e6bd93cf"},
    LockedFile{"gem16_compilation.json", 3061406, "7e5e78b9c6f61fbe8829866395634085261e1261a8c783f69affc5a16bd1847a", "fe671accce31db603d8d139bcef8523ee3baac34"},
    LockedFile{"gem16_model.json", 671, "c1015228120105ff5e54efe300c77c7e6a831ed598425f4f4c1b27c673732049", "02cf4afbfe3051e630eecd85c313fbb2a11ef015"},
    LockedFile{"generation_config.json", 203, "b69207f9be617e982d13cc273cce6fd88c98dda99a4bdc5e2d52ffe0a0d9f0a9", "5a376e9fca28bfac73fe81b623bd6e4d00bb0817"},
    LockedFile{"model.gem16", 14696668160ULL, "1ed73cf105b68db937ac0992283d31fdb2225474204341440721f41fe871bb72", "1ed73cf105b68db937ac0992283d31fdb2225474204341440721f41fe871bb72"},
    LockedFile{"tokenizer.json", 32169626, "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f", "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"},
    LockedFile{"tokenizer_config.json", 3729, "3ab5c7b94dc97d65ca7064496fa69b88ff875378e1cb7ee3e43070c3a8170999", "bd297dce3d66737f7d6f9136691db3ad05991a2f"},
};

constexpr std::array kAssistantFiles{
    LockedFile{".gitattributes", 101, "a0e948ca693c63e0b5139f66bd709003cbcf8bc764acb1ee687a552885c8fe2c", "0fd17d2f7c330afce8bc2b472393143fa66f68a1"},
    LockedFile{"LICENSE", 634, "e32c5cdb30142b7c670d61be0f2de18ab0007f72ef131e2869ef51687539c86d", "506ff98349a36c9b78904645c9ae492a9b62e826"},
    LockedFile{"NOTICE", 791, "555f14e6346372367e5facd6979a9594fd1502f319548933f7c867a1b00ec9d3", "80da8080b8090add2bedb8bec8dc62682b16c88d"},
    LockedFile{"README.md", 31457, "d93671a2e9b9064f670525925ba085c43b4023e70ec76f29b0c4ea0c9a1f5dcc", "1e883f1f67f7231ac09567790a012a84997293da"},
    LockedFile{"chat_template.jinja", 18683, "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4", "4741bf6e4132ba23a5537f9d6e74e9a6d613d7cd"},
    LockedFile{"config.json", 2720, "d3c79600ce09c86c993c89bed7ce05baa376770967ffaa1ac9a0c29347a633e4", "049ec62ef9a48d2f0732b580d90b48cbb0f3616b"},
    LockedFile{"gem16.lock.json", 1664, "f2d2278a53ecfb3bf7a093a3207dce70148a88708a51b5050dd8d436c3d69455", "bd555c8d0653bd74ec609bb51720e462174c3545"},
    LockedFile{"gem16_compilation.json", 217371, "d2a722ccad675807ff73a1a645a0f1fbcc0fc200c402a14e823a5927618a116d", "6e938657610d03a7f30aeec5172849238d8708a0"},
    LockedFile{"gem16_model.json", 613, "ef69d5384d6cca061e9941375bb2b18ebc9fee6587e492ef16701dce29b0131e", "f34e42339b16ee761c8724ee21aef629dfd84b65"},
    LockedFile{"generation_config.json", 209, "fb53f4c64e58896a63472e8eb304397db4a39453e1da0f5d57625ec5a8c1050e", "76d3a00538aca6a1b85db3328e1b0f4e93316420"},
    LockedFile{"model-00001-of-00001.safetensors", 258317280, "4d3ce2102ad0631d9e7e0586be0b108d5789cbc5b90d21b4c50613979228d927", "4d3ce2102ad0631d9e7e0586be0b108d5789cbc5b90d21b4c50613979228d927"},
    LockedFile{"model.safetensors.index.json", 8327, "8fcc4b09fb62f710558176406385e9a04423119cd6e81301171d9a64bbfd9165", "c094c640185f76249eb96a10d0d07dd7afd9ec89"},
    LockedFile{"tokenizer.json", 32169440, "75a6583c1a418e2bbd79c60d95d28e0f5bf549ad3f2990b5bdb5238c6c2bf70c", "75a6583c1a418e2bbd79c60d95d28e0f5bf549ad3f2990b5bdb5238c6c2bf70c"},
    LockedFile{"tokenizer_config.json", 3023, "01f2ff1c21ef2e722891380323edcaecd9c86a776aeb9b40148e2f35e3cee4d3", "0672fbe45a4922b10e6ccd13947ecdb166bead28"},
};

constexpr LockedRepository kTarget{
    "danmoreng/gemma-4-26B-A4B-it-GEM16",
    "63508b5826527484e707b4b46e2eacf077cf2b35", kTargetFiles};
constexpr LockedRepository kAssistant{
    "danmoreng/gemma-4-26B-A4B-it-assistant-GEM16",
    "466cc26d157fad0cc946f094ae904445147c38b4", kAssistantFiles};
constexpr std::array kRepositories{kTarget, kAssistant};
constexpr std::uint64_t kTotalBytes = 15022736099ULL;

bool HasSize(const std::filesystem::path& path, std::uint64_t size) {
  std::error_code error;
  return std::filesystem::is_regular_file(path, error) && !error &&
         std::filesystem::file_size(path, error) == size && !error;
}

std::filesystem::path RepositoryDirectory(const LockedRepository& repository) {
  std::string name = repository.repository;
  std::replace(name.begin(), name.end(), '/', '-');
  const auto delimiter = name.find('-');
  if (delimiter != std::string::npos) name.replace(delimiter, 1, "--");
  return HuggingFaceHubRoot() / ("models--" + name);
}

std::filesystem::path SnapshotDirectory(const LockedRepository& repository) {
  return RepositoryDirectory(repository) / "snapshots" / repository.revision;
}

std::filesystem::path BlobPath(const LockedRepository& repository,
                               const LockedFile& file) {
  return RepositoryDirectory(repository) / "blobs" / file.blob_id;
}

std::filesystem::path VerificationMarker(const LockedFile& file) {
  return HuggingFaceHubRoot() / ".gem16-verified" /
         (std::string(file.blob_id) + ".sha256");
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

bool SnapshotReady(const LockedRepository& repository) {
  const auto snapshot = SnapshotDirectory(repository);
  for (const auto& file : repository.files) {
    if (!HasSize(snapshot / file.path, file.size)) return false;
  }
  return true;
}

std::uint64_t ReadyBytes() {
  std::uint64_t bytes = 0;
  for (const auto& repository : kRepositories) {
    const auto snapshot = SnapshotDirectory(repository);
    for (const auto& file : repository.files) {
      if (HasSize(snapshot / file.path, file.size)) bytes += file.size;
    }
  }
  return bytes;
}

}  // namespace

ModelManager::ModelManager() {
  state_.total_bytes = kTotalBytes;
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
  state_.target_26b_ready = SnapshotReady(kTarget);
  state_.assistant_26b_ready = SnapshotReady(kAssistant);
  state_.completed_bytes = ReadyBytes();
  state_.total_bytes = kTotalBytes;
}

void ModelManager::DownloadQualified26B() {
  {
    std::scoped_lock lock(mutex_);
    if (state_.downloading) return;
    state_.downloading = true;
    state_.cancel_requested = false;
    state_.error.clear();
    state_.current_file.clear();
    state_.completed_bytes = ReadyBytes();
    state_.total_bytes = kTotalBytes;
  }
  cancel_.store(false);
  if (worker_.joinable()) worker_.join();
  worker_ = std::jthread([this] { DownloadWorker(); });
}

void ModelManager::Cancel() {
  cancel_.store(true);
  PlatformProcess* process = nullptr;
  {
    std::scoped_lock lock(mutex_);
    state_.cancel_requested = state_.downloading;
    process = active_process_;
  }
  if (process != nullptr) process->Stop();
}

void ModelManager::DownloadWorker() {
  const auto finish = [this](std::string error) {
    std::scoped_lock lock(mutex_);
    state_.downloading = false;
    state_.cancel_requested = cancel_.load();
    state_.error = std::move(error);
    state_.current_file.clear();
    state_.target_26b_ready = SnapshotReady(kTarget);
    state_.assistant_26b_ready = SnapshotReady(kAssistant);
    state_.completed_bytes = ReadyBytes();
  };

  const auto curl = CurlExecutable();
  if (!curl) {
    finish("curl was not found on PATH; install curl to download the pinned Hugging Face files");
    return;
  }
  std::uint64_t completed = ReadyBytes();
  for (const auto& repository : kRepositories) {
    const auto snapshot = SnapshotDirectory(repository);
    for (const auto& file : repository.files) {
      if (cancel_.load()) {
        finish("Download paused; partial files are retained for resume");
        return;
      }
      const auto target = snapshot / file.path;
      if (HasSize(target, file.size)) continue;

      const auto blob = BlobPath(repository, file);
      const auto partial = std::filesystem::path(blob.string() + ".incomplete");
      const auto marker = VerificationMarker(file);
      std::error_code error;
      std::filesystem::create_directories(blob.parent_path(), error);
      std::filesystem::create_directories(target.parent_path(), error);
      std::filesystem::create_directories(marker.parent_path(), error);
      if (error) {
        finish("Could not create the Hugging Face cache directories: " + error.message());
        return;
      }

      bool verified = false;
      if (HasSize(blob, file.size)) {
        std::ifstream marker_input(marker);
        std::string marker_hash;
        std::getline(marker_input, marker_hash);
        verified = marker_hash == file.sha256 || FileSha256(blob) == file.sha256;
      }
      if (!verified) {
        if (std::filesystem::exists(blob, error)) {
          std::filesystem::remove(blob, error);
          if (error) {
            finish("Could not replace an invalid cache blob for " +
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
          if (!verified) {
            std::filesystem::remove(partial, error);
            if (error) {
              finish("Could not replace an invalid partial file for " +
                     std::string(file.path) + ": " + error.message());
              return;
            }
          }
        }
        if (!verified) {
          const std::string url = "https://huggingface.co/" +
                                  std::string(repository.repository) + "/resolve/" +
                                  repository.revision + "/" + file.path;
          std::vector<std::string> arguments{
              curl->string(), "--fail", "--location", "--silent", "--show-error",
              "--retry", "5", "--retry-all-errors", "--continue-at", "-",
              "--output", partial.string(), url};
          std::mutex wait_mutex;
          std::condition_variable wait_condition;
          bool exited = false;
          int exit_code = -1;
          PlatformProcess process;
          {
            std::scoped_lock lock(mutex_);
            state_.current_file = std::string(repository.repository) + "/" + file.path;
            state_.completed_bytes = completed;
            active_process_ = &process;
          }
          std::string start_error;
          if (!process.Start(arguments, RepositoryRoot().string(), [](std::string) {},
                             [&](int code) {
                               std::scoped_lock wait_lock(wait_mutex);
                               exit_code = code;
                               exited = true;
                               wait_condition.notify_all();
                             }, start_error)) {
            {
              std::scoped_lock lock(mutex_);
              active_process_ = nullptr;
            }
            finish("Could not start curl: " + start_error);
            return;
          }
          std::unique_lock wait_lock(wait_mutex);
          while (!exited) {
            wait_condition.wait_for(wait_lock, std::chrono::milliseconds(250));
            std::error_code size_error;
            const auto partial_size = std::filesystem::is_regular_file(partial, size_error)
                                          ? std::filesystem::file_size(partial, size_error)
                                          : 0;
            std::scoped_lock state_lock(mutex_);
            state_.completed_bytes = completed +
                std::min<std::uint64_t>(partial_size, file.size);
          }
          wait_lock.unlock();
          process.Stop();
          {
            std::scoped_lock lock(mutex_);
            active_process_ = nullptr;
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
          verified = HasSize(partial, file.size) && FileSha256(partial) == file.sha256;
        }
        if (!verified) {
          finish("Size or SHA-256 verification failed for " + std::string(file.path));
          return;
        }
        std::filesystem::rename(partial, blob, error);
        if (error) {
          finish("Could not finalize " + std::string(file.path) + ": " + error.message());
          return;
        }
      }

      {
        std::ofstream output(marker, std::ios::trunc);
        output << file.sha256 << '\n';
      }
      std::filesystem::remove(target, error);
      error.clear();
      std::filesystem::create_hard_link(blob, target, error);
      if (error) {
        finish("Could not hardlink " + std::string(file.path) +
               " into its immutable snapshot: " + error.message());
        return;
      }
      completed += file.size;
      {
        std::scoped_lock lock(mutex_);
        state_.completed_bytes = completed;
      }
    }
  }
  finish({});
}

}  // namespace gem16::studio

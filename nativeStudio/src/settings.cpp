#include "settings.h"

#include <cstdlib>
#include <fstream>
#include <string_view>

#ifdef _WIN32
#include <windows.h>
#endif

namespace gem16::studio {
namespace {

std::string EscapeLine(std::string_view value) {
  std::string result;
  result.reserve(value.size());
  for (const char character : value) {
    if (character == '\\') result += "\\\\";
    else if (character == '\n') result += "\\n";
    else if (character != '\r') result += character;
  }
  return result;
}

std::string UnescapeLine(std::string_view value) {
  std::string result;
  result.reserve(value.size());
  for (std::size_t index = 0; index < value.size(); ++index) {
    if (value[index] == '\\' && index + 1 < value.size()) {
      const char next = value[++index];
      result += next == 'n' ? '\n' : next;
    } else {
      result += value[index];
    }
  }
  return result;
}

bool ParseBool(std::string_view value, bool fallback) {
  if (value == "1" || value == "true") return true;
  if (value == "0" || value == "false") return false;
  return fallback;
}

std::filesystem::path ResolveHubRoot() {
  if (const char* value = std::getenv("HF_HUB_CACHE"); value && *value) return value;
  if (const char* value = std::getenv("HF_HOME"); value && *value) return std::filesystem::path(value) / "hub";
  if (const char* value = std::getenv("XDG_CACHE_HOME"); value && *value) {
    return std::filesystem::path(value) / "huggingface/hub";
  }
  if (const char* value = std::getenv("HOME"); value && *value) {
    return std::filesystem::path(value) / ".cache/huggingface/hub";
  }
  return RepositoryRoot() / ".cache/huggingface/hub";
}

std::filesystem::path ExecutableDirectory() {
#ifdef _WIN32
  std::wstring buffer(32768, L'\0');
  const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
  if (length > 0 && length < buffer.size()) {
    buffer.resize(length);
    return std::filesystem::path(buffer).parent_path();
  }
#else
  std::error_code error;
  const auto executable = std::filesystem::read_symlink("/proc/self/exe", error);
  if (!error && !executable.empty()) return executable.parent_path();
#endif
  return std::filesystem::current_path();
}

}  // namespace

std::filesystem::path HuggingFaceHubRoot() { return ResolveHubRoot(); }

std::filesystem::path Qualified26BTargetDirectory() {
  return ResolveHubRoot() /
         "models--danmoreng--gemma-4-26B-A4B-it-GEM16/snapshots/"
         "b5feb4d146c5ce943160514df0c70a31059885bd";
}

std::filesystem::path Qualified26BAssistantDirectory() {
  return ResolveHubRoot() /
         "models--danmoreng--gemma-4-26B-A4B-it-assistant-GEM16/snapshots/"
         "a741c642353ccdaefc6f987a3120f434dc9487c7";
}

const char* ProfileLabel(ModelProfile profile) {
  return profile == ModelProfile::kGemma4Moe26BA4B ? "Gemma 4 26B A4B" : "Gemma 4 12B Unified";
}

const char* ProfileWireName(ModelProfile profile) {
  return profile == ModelProfile::kGemma4Moe26BA4B ? "gemma4_26b_a4b" : "gemma4_12b";
}

std::filesystem::path RepositoryRoot() {
  if (const char* override_root = std::getenv("GEM16_REPO_ROOT"); override_root && *override_root) {
    return std::filesystem::absolute(override_root).lexically_normal();
  }
  const std::filesystem::path compiled_root =
      std::filesystem::path(GEM16_REPOSITORY_ROOT).lexically_normal();
  if (std::filesystem::is_regular_file(compiled_root / "CMakeLists.txt")) return compiled_root;
  const std::filesystem::path portable_root = ExecutableDirectory().parent_path();
  return portable_root.empty() ? std::filesystem::current_path() : portable_root;
}

std::filesystem::path SettingsPath() {
#ifdef _WIN32
  const char* base = std::getenv("APPDATA");
#else
  const char* base = std::getenv("XDG_CONFIG_HOME");
  if (!base || !*base) base = std::getenv("HOME");
#endif
  std::filesystem::path directory = base && *base ? std::filesystem::path(base) : RepositoryRoot();
#ifndef _WIN32
  if (!std::getenv("XDG_CONFIG_HOME") && base && *base) directory /= ".config";
#endif
  return directory / "gem16" / "studio.conf";
}

void ApplyProfileDefaults(ServerConfig& config, ModelProfile profile) {
  config.profile = profile;
  config.max_sessions = 1;
  config.max_context_tokens = 32768;
  config.mtp_draft_tokens = 2;
  config.mtp_adaptive = false;
  if (profile == ModelProfile::kGemma4Moe26BA4B) {
    config.model_name = "gemma4-26b-a4b";
    config.model_directory = Qualified26BTargetDirectory().string();
    config.assistant_directory = Qualified26BAssistantDirectory().string();
    config.max_context_tokens = 73728;
  } else {
    config.model_name = "gem16-12b";
    config.model_directory =
        (ResolveHubRoot() / ".gem16/snapshots/unsloth--gemma-4-12b-it-NVFP4--b1f649734b34aa5575b03d186abd1b9be3d0d5c4").string();
    config.assistant_directory =
        (ResolveHubRoot() / "models--google--gemma-4-12B-it-assistant/snapshots/364bd03c9952e5b7da73665ee30c9eccfc408345").string();
  }
}

StudioSettings DefaultSettings() {
  StudioSettings settings;
  const auto root = RepositoryRoot();
#ifdef _WIN32
  const auto adjacent_server = ExecutableDirectory() / "gem16-server.exe";
  settings.server.executable = std::filesystem::is_regular_file(adjacent_server)
                                   ? adjacent_server.string()
                                   : (root / "build/Windows/blackwell-release/bin/gem16-server.exe").string();
#else
  const auto adjacent_server = ExecutableDirectory() / "gem16-server";
  settings.server.executable = std::filesystem::is_regular_file(adjacent_server)
                                   ? adjacent_server.string()
                                   : (root / "build/Linux/blackwell-release/bin/gem16-server").string();
#endif
  if (const char* override_server = std::getenv("GEM16_STUDIO_SERVER_EXECUTABLE");
      override_server && *override_server) {
    settings.server.executable = override_server;
  }
  ApplyProfileDefaults(settings.server, ModelProfile::kGemma4Unified12B);
  return settings;
}

StudioSettings LoadSettings() {
  StudioSettings result = DefaultSettings();
  std::ifstream input(SettingsPath());
  std::string line;
  while (std::getline(input, line)) {
    const std::size_t delimiter = line.find('=');
    if (delimiter == std::string::npos) continue;
    const std::string key = line.substr(0, delimiter);
    const std::string value = UnescapeLine(std::string_view(line).substr(delimiter + 1));
    try {
      if (key == "profile") result.server.profile = value == "gemma4_26b_a4b" ? ModelProfile::kGemma4Moe26BA4B : ModelProfile::kGemma4Unified12B;
      else if (key == "executable") result.server.executable = value;
      else if (key == "model_directory") result.server.model_directory = value;
      else if (key == "assistant_directory") result.server.assistant_directory = value;
      else if (key == "model_name") result.server.model_name = value;
      else if (key == "host") result.server.host = value;
      else if (key == "port") result.server.port = std::stoi(value);
      else if (key == "max_context") result.server.max_context_tokens = std::stoll(value);
      else if (key == "mtp_draft_tokens") result.server.mtp_draft_tokens = std::stoi(value);
      else if (key == "greedy") result.server.greedy = ParseBool(value, result.server.greedy);
      else if (key == "reasoning_effort") result.generation.reasoning_effort = value;
      else if (key == "max_output_tokens") result.generation.max_output_tokens = std::stoll(value);
      else if (key == "system_prompt") result.generation.system_prompt = value;
      else if (key == "dark_theme") result.dark_theme = ParseBool(value, result.dark_theme);
    } catch (...) {
      // A malformed setting is ignored and retains the safe default.
    }
  }
  if (const char* override_server = std::getenv("GEM16_STUDIO_SERVER_EXECUTABLE");
      override_server && *override_server) {
    result.server.executable = override_server;
  }
  return result;
}

bool SaveSettings(const StudioSettings& settings) {
  std::error_code error;
  std::filesystem::create_directories(SettingsPath().parent_path(), error);
  std::ofstream output(SettingsPath(), std::ios::trunc);
  if (!output) return false;
  const ServerConfig& server = settings.server;
  output << "profile=" << ProfileWireName(server.profile) << '\n'
         << "executable=" << EscapeLine(server.executable) << '\n'
         << "model_directory=" << EscapeLine(server.model_directory) << '\n'
         << "assistant_directory=" << EscapeLine(server.assistant_directory) << '\n'
         << "model_name=" << EscapeLine(server.model_name) << '\n'
         << "host=" << EscapeLine(server.host) << '\n'
         << "port=" << server.port << '\n'
         << "max_context=" << server.max_context_tokens << '\n'
         << "mtp_draft_tokens=" << server.mtp_draft_tokens << '\n'
         << "greedy=" << (server.greedy ? 1 : 0) << '\n'
         << "reasoning_effort=" << EscapeLine(settings.generation.reasoning_effort) << '\n'
         << "max_output_tokens=" << settings.generation.max_output_tokens << '\n'
         << "system_prompt=" << EscapeLine(settings.generation.system_prompt) << '\n'
         << "dark_theme=" << (settings.dark_theme ? 1 : 0) << '\n';
  return static_cast<bool>(output);
}

}  // namespace gem16::studio

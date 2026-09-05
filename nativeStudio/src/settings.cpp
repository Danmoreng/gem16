#include "settings.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <string_view>

#include "model_catalog.h"

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

bool SupportedUiScale(float value) {
  return value == 0.0f || value == 1.0f || value == 1.25f || value == 1.5f;
}

std::filesystem::path ResolveHubRoot() {
  if (const char* value = std::getenv("HF_HUB_CACHE"); value && *value) return value;
  if (const char* value = std::getenv("HF_HOME"); value && *value) return std::filesystem::path(value) / "hub";
#ifdef _WIN32
  if (const char* value = std::getenv("USERPROFILE"); value && *value) {
    return std::filesystem::path(value) / ".cache/huggingface/hub";
  }
  if (const char* value = std::getenv("LOCALAPPDATA"); value && *value) {
    return std::filesystem::path(value) / "huggingface/hub";
  }
#else
  if (const char* value = std::getenv("XDG_CACHE_HOME"); value && *value) {
    return std::filesystem::path(value) / "huggingface/hub";
  }
  if (const char* value = std::getenv("HOME"); value && *value) {
    return std::filesystem::path(value) / ".cache/huggingface/hub";
  }
#endif
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

float ResolveUiScale(float configured_scale, float platform_scale,
                     bool linux_platform) {
  if (SupportedUiScale(configured_scale) && configured_scale != 0.0f) {
    return configured_scale;
  }
  const float platform = std::clamp(platform_scale, 1.0f, 2.0f);
  return linux_platform ? std::max(platform, 1.25f) : platform;
}

std::filesystem::path ProfileTargetDirectory(ModelProfile profile) {
  const auto* component = ComponentForProfile(
      CatalogForProfile(profile), ModelComponentKind::kTarget);
  return component ? ComponentDirectory(*component->catalog, ResolveHubRoot())
                   : std::filesystem::path{};
}

std::filesystem::path ProfileAssistantDirectory(ModelProfile profile) {
  const auto* component = ComponentForProfile(
      CatalogForProfile(profile), ModelComponentKind::kAssistant);
  return component ? ComponentDirectory(*component->catalog, ResolveHubRoot())
                   : std::filesystem::path{};
}

std::filesystem::path ProfileVisionDirectory(ModelProfile profile) {
  const auto* component = ComponentForProfile(
      CatalogForProfile(profile), ModelComponentKind::kVision);
  return component ? ComponentDirectory(*component->catalog, ResolveHubRoot())
                   : std::filesystem::path{};
}

const char* ProfileLabel(ModelProfile profile) {
  switch (profile) {
    case ModelProfile::kGemma4Unified12B:
      return "Gemma 4 12B Unified";
    case ModelProfile::kGemma4Moe26BA4B:
      return "Gemma 4 26B A4B";
    case ModelProfile::kGemma4Moe26BTrellis35VisionFp8:
      return "Gemma 4 26B A4B Compact Vision";
  }
  return "Unknown profile";
}

const char* ProfileWireName(ModelProfile profile) {
  switch (profile) {
    case ModelProfile::kGemma4Unified12B:
      return "gemma4_12b";
    case ModelProfile::kGemma4Moe26BA4B:
      return "gemma4_26b_a4b";
    case ModelProfile::kGemma4Moe26BTrellis35VisionFp8:
      return "gemma4_26b_trellis35_vision_fp8";
  }
  return "unknown";
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
  config.vision_soft_token_budget = 280;
  config.vision_directory.clear();
  if (profile == ModelProfile::kGemma4Moe26BA4B) {
    config.model_name = "gemma4-26b-a4b";
    config.model_directory = ProfileTargetDirectory(profile).string();
    config.assistant_directory = ProfileAssistantDirectory(profile).string();
    config.max_context_tokens = 86016;
  } else if (profile == ModelProfile::kGemma4Moe26BTrellis35VisionFp8) {
    config.model_name = "gemma4-26b-a4b-trellis35-vision-fp8";
    config.model_directory = ProfileTargetDirectory(profile).string();
    config.assistant_directory = ProfileAssistantDirectory(profile).string();
    config.vision_directory = ProfileVisionDirectory(profile).string();
    // Leave room for WDDM and desktop applications on Windows. The qualified
    // fixed-D2 maximum remains available as an explicit context setting.
#ifdef _WIN32
    config.max_context_tokens = 170000;
#else
    config.max_context_tokens = 229120;
#endif
  } else {
    config.model_name = "gem16-12b";
    config.model_directory = ProfileTargetDirectory(profile).string();
    config.assistant_directory = ProfileAssistantDirectory(profile).string();
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
  const auto settings_path = SettingsPath();
  std::error_code file_error;
  const bool existing_settings = std::filesystem::is_regular_file(settings_path, file_error);
  std::ifstream input(settings_path);
  bool saw_onboarding = false;
  std::string line;
  while (std::getline(input, line)) {
    const std::size_t delimiter = line.find('=');
    if (delimiter == std::string::npos) continue;
    const std::string key = line.substr(0, delimiter);
    const std::string value = UnescapeLine(std::string_view(line).substr(delimiter + 1));
    try {
      if (key == "profile") {
        if (value == "gemma4_26b_a4b")
          result.server.profile = ModelProfile::kGemma4Moe26BA4B;
        else if (value == "gemma4_26b_trellis35_vision_fp8")
          result.server.profile =
              ModelProfile::kGemma4Moe26BTrellis35VisionFp8;
        else
          result.server.profile = ModelProfile::kGemma4Unified12B;
      }
      else if (key == "onboarding_complete") {
        result.onboarding_complete = ParseBool(value, result.onboarding_complete);
        saw_onboarding = true;
      } else if (key == "auto_start_server")
        result.auto_start_server = ParseBool(value, result.auto_start_server);
      else if (key == "executable") result.server.executable = value;
      else if (key == "previous_model_selection" &&
               value.size() <= 128U * 1024U)
        result.previous_model_selection = value;
      else if (key == "model_directory") result.server.model_directory = value;
      else if (key == "assistant_directory") result.server.assistant_directory = value;
      else if (key == "vision_directory") result.server.vision_directory = value;
      else if (key == "model_name") result.server.model_name = value;
      else if (key == "host") result.server.host = value;
      else if (key == "port") result.server.port = std::stoi(value);
      else if (key == "max_context") result.server.max_context_tokens = std::stoll(value);
      else if (key == "max_sessions")
        result.server.max_sessions = std::clamp(std::stoi(value), 1, 2);
      else if (key == "mtp_adaptive")
        result.server.mtp_adaptive =
            ParseBool(value, result.server.mtp_adaptive);
      else if (key == "mtp_draft_tokens") result.server.mtp_draft_tokens = std::stoi(value);
      else if (key == "vision_soft_token_budget") {
        const int budget = std::stoi(value);
        if (budget == 70 || budget == 140 || budget == 280)
          result.server.vision_soft_token_budget = budget;
      }
      else if (key == "greedy") result.server.greedy = ParseBool(value, result.server.greedy);
      else if (key == "reasoning_effort") result.generation.reasoning_effort = value;
      else if (key == "max_output_tokens") result.generation.max_output_tokens = std::stoll(value);
      else if (key == "system_prompt") result.generation.system_prompt = value;
      else if (key == "dark_theme") result.dark_theme = ParseBool(value, result.dark_theme);
      else if (key == "ui_scale") {
        const float scale = std::stof(value);
        if (SupportedUiScale(scale)) result.ui_scale = scale;
      }
    } catch (...) {
      // A malformed setting is ignored and retains the safe default.
    }
  }
  if (const char* override_server = std::getenv("GEM16_STUDIO_SERVER_EXECUTABLE");
      override_server && *override_server) {
    result.server.executable = override_server;
  }
  if (existing_settings && !saw_onboarding) result.onboarding_complete = true;
  return result;
}

bool SaveSettings(const StudioSettings& settings) {
  std::error_code error;
  std::filesystem::create_directories(SettingsPath().parent_path(), error);
  if (error) return false;
  const auto stamp = std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  const auto temporary =
      std::filesystem::path(SettingsPath().string() + "." + stamp + ".tmp");
  std::ofstream output(temporary, std::ios::trunc);
  if (!output) return false;
  const ServerConfig& server = settings.server;
  output << "onboarding_complete=" << (settings.onboarding_complete ? 1 : 0)
         << '\n'
         << "auto_start_server=" << (settings.auto_start_server ? 1 : 0) << '\n'
         << "previous_model_selection="
         << EscapeLine(settings.previous_model_selection) << '\n'
         << "profile=" << ProfileWireName(server.profile) << '\n'
         << "executable=" << EscapeLine(server.executable) << '\n'
         << "model_directory=" << EscapeLine(server.model_directory) << '\n'
         << "assistant_directory=" << EscapeLine(server.assistant_directory)
         << '\n'
         << "vision_directory=" << EscapeLine(server.vision_directory) << '\n'
         << "model_name=" << EscapeLine(server.model_name) << '\n'
         << "host=" << EscapeLine(server.host) << '\n'
         << "port=" << server.port << '\n'
         << "max_context=" << server.max_context_tokens << '\n'
         << "max_sessions=" << server.max_sessions << '\n'
         << "mtp_adaptive=" << (server.mtp_adaptive ? 1 : 0) << '\n'
         << "mtp_draft_tokens=" << server.mtp_draft_tokens << '\n'
         << "vision_soft_token_budget=" << server.vision_soft_token_budget
         << '\n'
         << "greedy=" << (server.greedy ? 1 : 0) << '\n'
         << "reasoning_effort="
         << EscapeLine(settings.generation.reasoning_effort) << '\n'
         << "max_output_tokens=" << settings.generation.max_output_tokens
         << '\n'
         << "system_prompt=" << EscapeLine(settings.generation.system_prompt)
         << '\n'
         << "dark_theme=" << (settings.dark_theme ? 1 : 0) << '\n';
  output << "ui_scale=" << settings.ui_scale << '\n';
  output.close();
  if (!output) {
    std::filesystem::remove(temporary, error);
    return false;
  }
#ifdef _WIN32
  if (!MoveFileExW(temporary.c_str(), SettingsPath().c_str(),
                   MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    std::filesystem::remove(temporary, error);
    return false;
  }
#else
  std::filesystem::rename(temporary, SettingsPath(), error);
  if (error) {
    std::filesystem::remove(temporary, error);
    return false;
  }
#endif
  return true;
}

}  // namespace gem16::studio

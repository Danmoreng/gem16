#include "api_client.h"
#include "chat_history.h"
#include "fonts.h"
#include "gem16_logo.generated.h"
#include "image_texture.h"
#include "markdown.h"
#include "media_loader.h"
#include "model_catalog.h"
#include "model_manager.h"
#include "model_widgets.h"
#include "server_manager.h"
#include "settings.h"
#include "selectable_text.h"
#include "util/json.h"

#include "httplib.h"
#include "imgui.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <fstream>
#include <set>
#include <string_view>
#include <thread>
#ifndef _WIN32
#include <sys/resource.h>
#endif

bool TestExtendedMarkdown();
bool TestChatStore();
bool TestCanvas();
bool TestStudioLifecycle(const std::filesystem::path& executable);
int RunStudioTestServer(int argc, char** argv);
bool TestModelLifecycle();
namespace {

void CaptureClipboard(void* user_data, const char* text) {
  *static_cast<std::string*>(user_data) = text == nullptr ? "" : text;
}

bool Contains(const std::vector<std::string>& values, const std::string& expected) {
  return std::find(values.begin(), values.end(), expected) != values.end();
}

void SetEnvironment(const char* name, const std::string& value) {
#ifdef _WIN32
  _putenv_s(name, value.c_str());
#else
  setenv(name, value.c_str(), 1);
#endif
}

void ClearEnvironment(const char* name) {
#ifdef _WIN32
  _putenv_s(name, "");
#else
  unsetenv(name);
#endif
}

bool TestServerCommand() {
  gem16::studio::ServerConfig config;
  config.profile = gem16::studio::ModelProfile::kGemma4Moe26BA4B;
  config.executable = "/tmp/gem16-server";
  config.model_directory = "/tmp/target";
  config.assistant_directory = "/tmp/assistant";
  config.model_name = "gemma4-26b-a4b";
  config.mtp_draft_tokens = 2;
  const auto command = gem16::studio::BuildServerCommand(config);
  return command.front() == config.executable && Contains(command, "--assistant-model") &&
         Contains(command, config.assistant_directory) && Contains(command, "--mtp-draft-tokens") &&
         Contains(command, "2") && !Contains(command, "--mtp-adaptive") &&
         !Contains(command, "--vision-model");
}

bool TestQualified26BDefaults() {
  gem16::studio::ServerConfig config;
  gem16::studio::ApplyProfileDefaults(
      config, gem16::studio::ModelProfile::kGemma4Moe26BA4B);
  return config.max_context_tokens == 86016 && config.max_sessions == 1 &&
         config.mtp_draft_tokens == 2 && !config.mtp_adaptive &&
         config.model_directory ==
             gem16::studio::ProfileTargetDirectory(
                 gem16::studio::ModelProfile::kGemma4Moe26BA4B).string() &&
         config.assistant_directory ==
             gem16::studio::ProfileAssistantDirectory(
                 gem16::studio::ModelProfile::kGemma4Moe26BA4B).string();
}

bool TestVision26BDefaultsAndCommand() {
  gem16::studio::ServerConfig config;
  config.executable = "/tmp/gem16-server";
  gem16::studio::ApplyProfileDefaults(
      config,
      gem16::studio::ModelProfile::kGemma4Moe26BTrellis35VisionFp8);
  const auto command = gem16::studio::BuildServerCommand(config);
  return config.max_context_tokens == 229120 && config.max_sessions == 1 &&
         config.mtp_draft_tokens == 2 && !config.mtp_adaptive &&
         config.vision_soft_token_budget == 280 &&
         config.model_directory ==
             gem16::studio::ProfileTargetDirectory(config.profile).string() &&
         config.assistant_directory ==
             gem16::studio::ProfileAssistantDirectory(config.profile).string() &&
         config.vision_directory ==
             gem16::studio::ProfileVisionDirectory(config.profile).string() &&
         Contains(command, "--vision-model") &&
         Contains(command, config.vision_directory) &&
         Contains(command, "--vision-max-soft-token-budget") &&
         Contains(command, "280") &&
         Contains(command, "--assistant-model");
}

bool TestVisionAttachmentPolicyAndEstimate() {
  using gem16::studio::AttachmentPolicyError;
  using gem16::studio::MediaKind;
  using gem16::studio::ModelProfile;
  if (!AttachmentPolicyError(ModelProfile::kGemma4Unified12B,
                             MediaKind::kAudio, 0U)
           .empty() ||
      !AttachmentPolicyError(ModelProfile::kGemma4Moe26BTrellis35VisionFp8,
                             MediaKind::kDocument, 1U)
           .empty() ||
      !AttachmentPolicyError(ModelProfile::kGemma4Moe26BTrellis35VisionFp8,
                             MediaKind::kImage, 0U)
           .empty() ||
      AttachmentPolicyError(ModelProfile::kGemma4Moe26BTrellis35VisionFp8,
                            MediaKind::kImage, 1U)
          .empty() ||
      AttachmentPolicyError(ModelProfile::kGemma4Moe26BTrellis35VisionFp8,
                            MediaKind::kAudio, 0U)
          .empty() ||
      AttachmentPolicyError(ModelProfile::kGemma4Moe26BA4B,
                            MediaKind::kImage, 0U)
          .empty()) {
    return false;
  }
  gem16::studio::MediaAttachment image;
  image.kind = MediaKind::kImage;
  image.image_width = 256;
  image.image_height = 256;
  return gem16::studio::EstimateVisionSoftTokens(image, 70U) == 64U &&
         gem16::studio::EstimateVisionSoftTokens(image, 140U) == 121U &&
         gem16::studio::EstimateVisionSoftTokens(image, 280U) == 256U &&
         gem16::studio::EstimateVisionSoftTokens(image, 71U) == 0U;
}

bool TestLiveServerCompatibility() {
  using gem16::studio::HealthCompatibilityError;
  using gem16::studio::HealthSnapshot;
  using gem16::studio::ModelProfile;
  using gem16::studio::ServerConfig;
  ServerConfig config;
  config.profile = ModelProfile::kGemma4Moe26BTrellis35VisionFp8;
  config.mtp_draft_tokens = 2;
  HealthSnapshot health;
  health.available = true;
  health.profile_id = "gemma4-26b-a4b-trellis35-vision-fp8";
  health.decode_mode = "fixed-d2";
  health.qualification_state = "production_candidate";
  health.supports_vision = true;
  health.vision_module_loaded = true;
  health.supports_mtp = true;
  health.vision_mtp_supported = true;
  health.mtp_draft_tokens = 2;
  health.vision_max_soft_token_budget = config.vision_soft_token_budget;
  if (!HealthCompatibilityError(config, health).empty()) return false;

  health.vision_max_soft_token_budget = 140;
  if (HealthCompatibilityError(config, health).empty()) return false;
  health.vision_max_soft_token_budget = config.vision_soft_token_budget;

  health.qualification_state = "development";
  if (HealthCompatibilityError(config, health).empty()) return false;
  health.qualification_state = "production_candidate";

  health.vision_mtp_supported = false;
  if (HealthCompatibilityError(config, health).empty()) return false;
  health.vision_mtp_supported = true;
  health.vision_module_loaded = false;
  if (HealthCompatibilityError(config, health).empty()) return false;
  health.vision_module_loaded = true;
  health.profile_id = "gemma4-26b-a4b-nvfp4";
  if (HealthCompatibilityError(config, health).empty()) return false;

  config.mtp_draft_tokens = 0;
  health.profile_id = "gemma4-26b-a4b-trellis35-vision-fp8";
  health.decode_mode = "ordinary";
  health.mtp_draft_tokens = 0;
  health.supports_mtp = false;
  health.vision_mtp_supported = false;
  health.qualification_state = "production_candidate";
  return HealthCompatibilityError(config, health).empty();
}

bool TestModelCatalog() {
  const auto catalog = gem16::studio::ModelCatalog();
  if (catalog.size() != gem16::studio::kModelProfileCount) return false;
  const auto public_profiles = gem16::studio::PublicModelProfiles();
  if (public_profiles.size() != 2U ||
      public_profiles[0] !=
          gem16::studio::ModelProfile::kGemma4Unified12B ||
      public_profiles[1] !=
          gem16::studio::ModelProfile::kGemma4Moe26BTrellis35VisionFp8 ||
      std::ranges::find(public_profiles,
                        gem16::studio::ModelProfile::kGemma4Moe26BA4B) !=
          public_profiles.end()) {
    return false;
  }
  const auto& twelve = gem16::studio::CatalogForProfile(
      gem16::studio::ModelProfile::kGemma4Unified12B);
  const auto& twenty_six = gem16::studio::CatalogForProfile(
      gem16::studio::ModelProfile::kGemma4Moe26BA4B);
  const auto& vision = gem16::studio::CatalogForProfile(
      gem16::studio::ModelProfile::kGemma4Moe26BTrellis35VisionFp8);
  const auto* twelve_target = gem16::studio::ComponentForProfile(
      twelve, gem16::studio::ModelComponentKind::kTarget);
  const auto* twelve_assistant = gem16::studio::ComponentForProfile(
      twelve, gem16::studio::ModelComponentKind::kAssistant);
  const auto* twenty_six_target = gem16::studio::ComponentForProfile(
      twenty_six, gem16::studio::ModelComponentKind::kTarget);
  const auto* twenty_six_assistant = gem16::studio::ComponentForProfile(
      twenty_six, gem16::studio::ModelComponentKind::kAssistant);
  const auto* vision_target = gem16::studio::ComponentForProfile(
      vision, gem16::studio::ModelComponentKind::kTarget);
  const auto* vision_module = gem16::studio::ComponentForProfile(
      vision, gem16::studio::ModelComponentKind::kVision);
  const auto* vision_assistant = gem16::studio::ComponentForProfile(
      vision, gem16::studio::ModelComponentKind::kAssistant);
  if (!twelve_target || !twelve_assistant || !twenty_six_target ||
      !twenty_six_assistant || !vision_target || !vision_module ||
      !vision_assistant || vision.components.size() != 3U ||
      std::ranges::any_of(vision.components,
                          [](const auto& component) {
                            return !component.required;
                          })) {
    return false;
  }
  if (std::string_view(twelve_target->catalog->repository) !=
          "unsloth/gemma-4-12b-it-NVFP4" ||
      std::string_view(twelve_assistant->catalog->repository) !=
          "google/gemma-4-12B-it-assistant" ||
      !twelve_target->catalog->composed_view ||
      twelve_assistant->catalog->composed_view ||
      twenty_six_target->catalog->composed_view ||
      !twenty_six_assistant->catalog->composed_view ||
      std::string_view(twenty_six_assistant->catalog->repository) !=
          "danmoreng/gemma-4-26B-A4B-it-GEM16" ||
      !vision_target->catalog->composed_view ||
      !vision_module->catalog->composed_view ||
      vision_module->catalog->files.size() != 4U ||
      std::string_view(vision_target->catalog->composed_view_suffix) !=
          "trellis35" ||
      std::string_view(vision_module->catalog->composed_view_suffix) !=
          "vision") {
    return false;
  }
  if (std::ranges::any_of(
          vision_module->catalog->files, [](const auto& file) {
            const std::string_view path = file.path;
            return path == "LICENSE" || path == "NOTICE" ||
                   path == "README.md";
          })) {
    return false;
  }
  bool external_tokenizer = false;
  for (const auto& file : twelve_target->catalog->files) {
    external_tokenizer |=
        std::string_view(file.path) == "tokenizer_config.json" &&
        std::string_view(file.source_repository) == "google/gemma-4-12B-it";
  }
  const auto root = std::filesystem::path("/hub");
  const auto& first_target_file = twelve_target->catalog->files.front();
  return external_tokenizer &&
         gem16::studio::ComponentDirectory(*twelve_target->catalog, root) ==
             root / ".gem16/snapshots/"
                    "unsloth--gemma-4-12b-it-NVFP4--"
                    "b1f649734b34aa5575b03d186abd1b9be3d0d5c4" &&
         gem16::studio::ComponentDirectory(*twenty_six_target->catalog, root) ==
             root / "models--danmoreng--gemma-4-26B-A4B-it-GEM16/snapshots/"
                    "6de2a057f11332420819f8e6efd08e42d7a03bc7" &&
         gem16::studio::ComponentDirectory(*twenty_six_assistant->catalog, root) ==
             root / ".gem16/snapshots/"
                    "danmoreng--gemma-4-26B-A4B-it-GEM16--"
                    "6de2a057f11332420819f8e6efd08e42d7a03bc7--assistant" &&
         gem16::studio::VerificationMarkerPath(first_target_file, root) ==
             root / "models--unsloth--gemma-4-12b-it-NVFP4/.gem16-verified" /
                    (std::string(first_target_file.blob_id) + ".sha256");
}

bool TestNeutralFirstRunDefaults() {
  const auto settings = gem16::studio::DefaultSettings();
  return !settings.onboarding_complete &&
         settings.server.model_directory ==
             gem16::studio::ProfileTargetDirectory(
                 gem16::studio::ModelProfile::kGemma4Unified12B).string() &&
         settings.server.assistant_directory ==
             gem16::studio::ProfileAssistantDirectory(
                 gem16::studio::ModelProfile::kGemma4Unified12B).string();
}

bool TestUiScaleResolution() {
  return gem16::studio::ResolveUiScale(0.0f, 1.0f, true) == 1.25f &&
         gem16::studio::ResolveUiScale(0.0f, 1.5f, true) == 1.5f &&
         gem16::studio::ResolveUiScale(0.0f, 1.0f, false) == 1.0f &&
         gem16::studio::ResolveUiScale(1.0f, 1.5f, true) == 1.0f &&
         gem16::studio::ResolveUiScale(1.25f, 1.0f, false) == 1.25f;
}

bool TestOnboardingPersistence() {
#ifdef _WIN32
  constexpr const char* environment_name = "APPDATA";
#else
  constexpr const char* environment_name = "XDG_CONFIG_HOME";
#endif
  const char* previous = std::getenv(environment_name);
  const std::string previous_value = previous == nullptr ? "" : previous;
  const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto config = std::filesystem::temp_directory_path() /
                      ("gem16-studio-settings-test-" + std::to_string(suffix));
  std::filesystem::create_directories(config);
  SetEnvironment(environment_name, config.string());

  auto settings = gem16::studio::LoadSettings();
  bool valid = !settings.onboarding_complete;
  settings.onboarding_complete = true;
  settings.ui_scale = 1.25f;
  gem16::studio::ApplyProfileDefaults(
      settings.server,
      gem16::studio::ModelProfile::kGemma4Moe26BTrellis35VisionFp8);
  settings.server.vision_soft_token_budget = 140;
  valid = valid && gem16::studio::SaveSettings(settings);
  const auto loaded = gem16::studio::LoadSettings();
  valid = valid && loaded.onboarding_complete && loaded.ui_scale == 1.25f &&
          loaded.server.profile ==
              gem16::studio::ModelProfile::kGemma4Moe26BTrellis35VisionFp8 &&
          loaded.server.vision_soft_token_budget == 140 &&
          loaded.server.vision_directory == settings.server.vision_directory;

  if (previous == nullptr)
    ClearEnvironment(environment_name);
  else
    SetEnvironment(environment_name, previous_value);
  std::error_code error;
  std::filesystem::remove_all(config, error);
  return valid && !error;
}

bool TestMediaPayload() {
  gem16::studio::MediaAttachment image;
  image.kind = gem16::studio::MediaKind::kImage;
  image.file_name = "tiny.png";
  image.mime_type = "image/png";
  image.format = "png";
  image.bytes = {0x01, 0x02, 0x03};
  gem16::studio::MediaAttachment audio;
  audio.kind = gem16::studio::MediaKind::kAudio;
  audio.file_name = "tiny.wav";
  audio.mime_type = "audio/wav";
  audio.format = "wav";
  audio.bytes = {0x04, 0x05};
  gem16::studio::MediaAttachment document;
  document.kind = gem16::studio::MediaKind::kDocument;
  document.file_name = "notes.txt";
  document.document_text = "native studio";
  gem16::studio::ChatMessage message{"user", "Inspect", {}, false, false};
  message.attachments = {image, audio, document};
  gem16::studio::ServerConfig server;
  gem16::studio::GenerationConfig generation;
  const std::string payload = gem16::studio::BuildChatPayload(
      server, generation, {message});
  const auto parsed = gem16::json::Parse(payload);
  server.profile =
      gem16::studio::ModelProfile::kGemma4Moe26BTrellis35VisionFp8;
  server.vision_soft_token_budget = 140;
  const std::string vision_payload = gem16::studio::BuildChatPayload(
      server, generation, {message});
  return parsed.ok() && payload.find("image_url") != std::string::npos &&
         payload.find("input_audio") != std::string::npos &&
         payload.find("native studio") != std::string::npos &&
         payload.find("AQID") != std::string::npos &&
         payload.find("vision_soft_token_budget") == std::string::npos &&
         vision_payload.find("\"vision_soft_token_budget\":140") !=
             std::string::npos;
}

bool TestPerformanceMetrics() {
  const std::string before_text =
      "# TYPE gem16_input_tokens_total counter\n"
      "gem16_input_tokens_total 100\n"
      "gem16_cache_write_tokens_total 80\n"
      "gem16_prompt_microseconds_total 40000\n"
      "gem16_decode_microseconds_total 90000\n"
      "gem16_decode_measured_tokens_total 9\n";
  const std::string after_text =
      "gem16_input_tokens_total 112\n"
      "gem16_cache_write_tokens_total 92\n"
      "gem16_prompt_microseconds_total 46000\n"
      "gem16_decode_microseconds_total 230000\n"
      "gem16_decode_measured_tokens_total 16\n";
  const auto before = gem16::studio::ParseServerMetrics(before_text);
  const auto after = gem16::studio::ParseServerMetrics(after_text);
  if (!before || !after) return false;
  const auto performance = gem16::studio::PerformanceDifference(*before, *after);
  return performance &&
         std::abs(performance->decode_tokens_per_second - 50.0) < 0.001 &&
         std::abs(performance->prefill_tokens_per_second - 2000.0) < 0.001 &&
         std::abs(performance->prefill_milliseconds - 6.0) < 0.001 &&
         std::abs(performance->decode_milliseconds - 140.0) < 0.001;
}

bool TestPreviewImageDecode() {
  const auto logo = gem16::studio::DecodePreviewImage(
      gem16::studio::kGem16LogoPng, gem16::studio::kGem16LogoPngSize);
  const std::array<std::uint8_t, 8> invalid{0, 1, 2, 3, 4, 5, 6, 7};
  const auto rejected = gem16::studio::DecodePreviewImage(
      invalid.data(), invalid.size());
  return logo.width == 256 && logo.height == 256 &&
         logo.rgba.size() == 256U * 256U * 4U && rejected.rgba.empty();
}

bool TestMediaLoader() {
  const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto directory = std::filesystem::temp_directory_path() /
                         ("gem16-media-test-" + std::to_string(suffix));
  std::filesystem::create_directories(directory);
  const auto document = directory / "notes.md";
  {
    std::ofstream output(document, std::ios::binary);
    output << "# Native Studio\n\nUTF-8: gruen";
  }
  gem16::studio::MediaAttachment attachment;
  std::string error;
  const bool loaded = gem16::studio::LoadMediaAttachment(
      document, attachment, error);
  std::error_code remove_error;
  std::filesystem::remove_all(directory, remove_error);
  return loaded && error.empty() && !remove_error &&
         attachment.id != 0 &&
         attachment.kind == gem16::studio::MediaKind::kDocument &&
         attachment.file_name == "notes.md" &&
         attachment.document_text.find("Native Studio") != std::string::npos;
}

bool TestEmptyCacheInstallState() {
  const char* previous = std::getenv("HF_HUB_CACHE");
  const std::string previous_value = previous == nullptr ? "" : previous;
  const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto cache = std::filesystem::temp_directory_path() /
                     ("gem16-studio-catalog-test-" + std::to_string(suffix));
  std::filesystem::create_directories(cache);
  SetEnvironment("HF_HUB_CACHE", cache.string());

  std::array<std::uint64_t, gem16::studio::kModelProfileCount> expected{};
  for (const auto& profile : gem16::studio::ModelCatalog()) {
    const auto index = gem16::studio::ModelProfileIndex(profile.profile);
    std::set<std::string> blobs;
    for (const auto& component : profile.components) {
      for (const auto& file : component.catalog->files) {
        const std::string identity = std::string(file.source_repository) + "/" +
                                     file.blob_id;
        if (blobs.insert(identity).second) expected[index] += file.size;
      }
    }
  }
  gem16::studio::ModelManager manager;
  const auto state = manager.State();
  const bool valid =
      !state.downloading && state.For(gem16::studio::ModelProfile::kGemma4Unified12B)
                                .required_download_bytes == expected[0] &&
      state.For(gem16::studio::ModelProfile::kGemma4Moe26BA4B)
              .required_download_bytes == expected[1] &&
      state.For(gem16::studio::ModelProfile::kGemma4Moe26BTrellis35VisionFp8)
              .required_download_bytes == expected[2] &&
      state.For(gem16::studio::ModelProfile::kGemma4Unified12B).storage_available &&
      state.For(gem16::studio::ModelProfile::kGemma4Moe26BA4B).storage_available &&
      state.For(gem16::studio::ModelProfile::kGemma4Moe26BTrellis35VisionFp8)
          .storage_available;

  if (previous == nullptr)
    ClearEnvironment("HF_HUB_CACHE");
  else
    SetEnvironment("HF_HUB_CACHE", previous_value);
  std::error_code error;
  std::filesystem::remove_all(cache, error);
  return valid && !error;
}

bool TestComponentRemovalKeepsSharedBlob() {
  const char* previous = std::getenv("HF_HUB_CACHE");
  const std::string previous_value = previous == nullptr ? "" : previous;
  const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto cache = std::filesystem::temp_directory_path() /
                     ("gem16-studio-remove-test-" + std::to_string(suffix));
  std::filesystem::create_directories(cache);
  SetEnvironment("HF_HUB_CACHE", cache.string());

  const auto profile =
      gem16::studio::ModelProfile::kGemma4Moe26BTrellis35VisionFp8;
  const auto* component = gem16::studio::ComponentForProfile(
      gem16::studio::CatalogForProfile(profile),
      gem16::studio::ModelComponentKind::kVision);
  bool valid = component != nullptr;
  if (component) {
    const auto& file = component->catalog->files.front();
    const auto view = gem16::studio::ComponentDirectory(*component->catalog,
                                                        cache) /
                      file.path;
    const auto blob = gem16::studio::RepositoryDirectory(
                          file.source_repository, cache) /
                      "blobs" / file.blob_id;
    std::filesystem::create_directories(view.parent_path());
    std::filesystem::create_directories(blob.parent_path());
    {
      std::ofstream(view) << "view";
      std::ofstream(blob) << "shared";
    }
    gem16::studio::ModelManager manager;
    manager.RemoveComponent(profile, gem16::studio::ModelComponentKind::kVision);
    valid = !std::filesystem::exists(view) &&
            std::filesystem::is_regular_file(blob);
  }

  if (previous == nullptr)
    ClearEnvironment("HF_HUB_CACHE");
  else
    SetEnvironment("HF_HUB_CACHE", previous_value);
  std::error_code error;
  std::filesystem::remove_all(cache, error);
  return valid && !error;
}

bool TestLegacyVisionViewMigration() {
  const char* previous = std::getenv("HF_HUB_CACHE");
  const std::string previous_value = previous == nullptr ? "" : previous;
  const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto cache = std::filesystem::temp_directory_path() /
                     ("gem16-studio-vision-migration-" +
                      std::to_string(suffix));
  std::filesystem::create_directories(cache);
  SetEnvironment("HF_HUB_CACHE", cache.string());
  const auto& profile = gem16::studio::CatalogForProfile(
      gem16::studio::ModelProfile::kGemma4Moe26BTrellis35VisionFp8);
  const auto* vision = gem16::studio::ComponentForProfile(
      profile, gem16::studio::ModelComponentKind::kVision);
  bool valid = vision != nullptr;
  if (vision) {
    const auto root = gem16::studio::ComponentDirectory(*vision->catalog, cache);
    std::filesystem::create_directories(root);
    std::ofstream(root / "LICENSE") << "legacy hardlink view";
    std::ofstream(root / "NOTICE") << "legacy hardlink view";
    std::ofstream(root / "README.md") << "legacy hardlink view";
    gem16::studio::ModelManager manager;
    valid = !std::filesystem::exists(root / "LICENSE") &&
            !std::filesystem::exists(root / "NOTICE") &&
            !std::filesystem::exists(root / "README.md");
  }
  if (previous == nullptr)
    ClearEnvironment("HF_HUB_CACHE");
  else
    SetEnvironment("HF_HUB_CACHE", previous_value);
  std::error_code error;
  std::filesystem::remove_all(cache, error);
  return valid && !error;
}

bool TestQualifiedVisionCacheIfProvided() {
  const char* cache = std::getenv("GEM16_STUDIO_TEST_VISION_CACHE");
  if (cache == nullptr || *cache == '\0') return true;
  const char* previous = std::getenv("HF_HUB_CACHE");
  const std::string previous_value = previous == nullptr ? "" : previous;
  SetEnvironment("HF_HUB_CACHE", cache);
  gem16::studio::ModelManager manager;
  const auto& state = manager.State().For(
      gem16::studio::ModelProfile::kGemma4Moe26BTrellis35VisionFp8);
  const bool valid = state.Ready() &&
                     state.ComponentReady(
                         gem16::studio::ModelComponentKind::kTarget) &&
                     state.ComponentReady(
                         gem16::studio::ModelComponentKind::kVision) &&
                     state.ComponentReady(
                         gem16::studio::ModelComponentKind::kAssistant) &&
                     state.required_download_bytes == 0U;
  if (previous == nullptr)
    ClearEnvironment("HF_HUB_CACHE");
  else
    SetEnvironment("HF_HUB_CACHE", previous_value);
  return valid;
}

bool TestTextSelection() {
  using gem16::studio::selectable_text::NormalizedRange;
  using gem16::studio::selectable_text::SelectedText;
  using gem16::studio::selectable_text::WordRange;
  if (NormalizedRange(8, 2, 6) != std::pair<std::size_t, std::size_t>{2, 6})
    return false;
  const std::string text = "Hello grünes Gemma!";
  if (SelectedText(text, 5, 0) != "Hello") return false;
  const auto [begin, end] = WordRange(text, 8);
  if (text.substr(begin, end - begin) != "grünes") return false;
  return true;
}

bool TestModelCardLayoutAndProgress() {
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.IniFilename = nullptr;
  io.DisplaySize = {1600.0f, 1200.0f};
  io.DeltaTime = 1.0f / 60.0f;
  unsigned char* pixels = nullptr;
  int width = 0;
  int height = 0;
  io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
  bool passed = true;
  for (float scale : {1.0f, 1.25f, 1.5f}) {
    for (float card_width : {340.0f, 900.0f}) {
      for (bool ready : {false, true}) {
        std::vector<ImU32> frame_colors[2];
        for (int frame = 0; frame < 5; ++frame) {
          ImGui::NewFrame();
          ImGui::SetNextWindowPos({0, 0});
          ImGui::SetNextWindowSize({card_width * scale, 900.0f});
          ImGui::Begin("##model-layout", nullptr, ImGuiWindowFlags_NoDecoration |
                                                    ImGuiWindowFlags_NoSavedSettings);
          ImGui::SetWindowFontScale(scale);
          gem16::studio::BeginModelCard("##card", scale);
          ImGui::TextWrapped("Gemma 4 26B A4B Compact Vision");
          ImGui::TextWrapped(
              "Text (Trellis35 W4A8) / Vision (FP8 E4M3FN) / Fixed MTP D2 / "
              "229,120 tokens");
          bool first = true;
          for (const char* component : {"Target", "Vision", "Assistant"}) {
            const std::string status = std::string(component) +
                                      (ready ? ": Verified" : ": Missing");
            const float extent = ImGui::CalcTextSize(status.c_str()).x +
                (ready ? ImGui::CalcTextSize("Remove").x +
                             ImGui::GetStyle().ItemSpacing.x +
                             ImGui::GetStyle().FramePadding.x * 2.0f : 0.0f);
            if (!first) gem16::studio::ModelComponentSameLine(extent, 22.0f * scale);
            first = false;
            ImGui::BeginGroup();
            ImGui::TextUnformatted(status.c_str());
            if (ready) {
              ImGui::SameLine();
              ImGui::PushID(component);
              ImGui::SmallButton("Remove");
              ImGui::PopID();
            }
            ImGui::EndGroup();
          }
          const int vertex_begin = ImGui::GetWindowDrawList()->VtxBuffer.Size;
          gem16::studio::ModelDownloadProgress(0.65f,
              {ImGui::GetContentRegionAvail().x, ImGui::GetFrameHeight()},
              "7.9 / 12.2 GiB", frame == 4 ? 1.0 : 0.0);
          if (frame >= 3) {
            auto& colors = frame_colors[frame - 3];
            const auto& vertices = ImGui::GetWindowDrawList()->VtxBuffer;
            for (int index = vertex_begin; index < vertices.Size; ++index)
              colors.push_back(vertices[index].col);
          }
          ImGui::TextWrapped("26B Trellis35 Target / model.gem16");
          if (frame >= 3) {
            if (ImGui::GetScrollMaxY() != 0.0f || ImGui::GetScrollMaxX() != 0.0f ||
                ImGui::GetWindowHeight() >= 260.0f * scale) {
              std::fprintf(stderr, "card scale=%.2f width=%.0f ready=%d frame=%d height=%.1f scroll=(%.1f,%.1f)\n",
                  scale, card_width, ready, frame, ImGui::GetWindowHeight(),
                  ImGui::GetScrollMaxX(), ImGui::GetScrollMaxY());
            }
            passed &= ImGui::GetScrollMaxY() == 0.0f && ImGui::GetScrollY() == 0.0f;
            passed &= ImGui::GetScrollMaxX() == 0.0f;
            passed &= ImGui::GetWindowHeight() < 260.0f * scale;
          }
          gem16::studio::EndModelCard();
          ImGui::End();
          ImGui::Render();
        }
        if (frame_colors[0] == frame_colors[1])
          std::fprintf(stderr, "shimmer unchanged scale=%.2f width=%.0f ready=%d\n",
                       scale, card_width, ready);
        passed &= frame_colors[0] != frame_colors[1];
      }
    }
  }
  ImGui::DestroyContext();
  return passed;
}

bool TestSelectableTextWidgetClipboard() {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.IniFilename = nullptr;
  io.DisplaySize = {480.0f, 260.0f};
  io.DeltaTime = 1.0f / 60.0f;
  io.ConfigInputTrickleEventQueue = false;
  std::string clipboard;
  io.SetClipboardTextFn = CaptureClipboard;
  io.ClipboardUserData = &clipboard;
  unsigned char* pixels = nullptr;
  int atlas_width = 0;
  int atlas_height = 0;
  io.Fonts->GetTexDataAsRGBA32(&pixels, &atlas_width, &atlas_height);

  const std::string response = "A selectable wrapped Gem 16 response.";
  const auto draw_frame = [&response] {
    ImGui::NewFrame();
    ImGui::SetNextWindowPos({0.0f, 0.0f});
    ImGui::SetNextWindowSize({480.0f, 260.0f});
    ImGui::Begin("##selection-test", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings);
    ImGui::SetCursorScreenPos({24.0f, 24.0f});
    gem16::studio::selectable_text::Wrapped(
        "##selectable-response", response, {.width = 340.0f});
    ImGui::End();
    ImGui::Render();
  };

  draw_frame();
  io.AddMousePosEvent(48.0f, 29.0f);
  io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
  draw_frame();
  io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
  draw_frame();
  io.AddKeyEvent(ImGuiMod_Ctrl, true);
  io.AddKeyEvent(ImGuiKey_A, true);
  io.AddKeyEvent(ImGuiKey_C, true);
  draw_frame();

  io.AddKeyEvent(ImGuiKey_C, false);
  io.AddKeyEvent(ImGuiKey_A, false);
  io.AddKeyEvent(ImGuiMod_Ctrl, false);
  ImGui::DestroyContext();
  return clipboard == response;
}

bool TestGroupedSelectableTextClipboard() {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.IniFilename = nullptr;
  io.DisplaySize = {480.0f, 260.0f};
  io.DeltaTime = 1.0f / 60.0f;
  io.ConfigInputTrickleEventQueue = false;
  std::string clipboard;
  io.SetClipboardTextFn = CaptureClipboard;
  io.ClipboardUserData = &clipboard;
  unsigned char* pixels = nullptr;
  int atlas_width = 0;
  int atlas_height = 0;
  io.Fonts->GetTexDataAsRGBA32(&pixels, &atlas_width, &atlas_height);

  const std::string first = "First line";
  const std::string second = "Second line";
  const std::string combined = first + "\n" + second;
  ImVec2 first_min{};
  ImVec2 first_max{};
  ImVec2 second_min{};
  ImVec2 second_max{};
  const auto draw_frame = [&] {
    ImGui::NewFrame();
    ImGui::SetNextWindowPos({0.0f, 0.0f});
    ImGui::SetNextWindowSize({480.0f, 260.0f});
    ImGui::Begin("##grouped-selection-test", nullptr,
                 ImGuiWindowFlags_NoDecoration |
                     ImGuiWindowFlags_NoSavedSettings);
    ImGui::SetCursorScreenPos({24.0f, 24.0f});
    const ImGuiID group = ImGui::GetID("##selection-group");
    gem16::studio::selectable_text::Wrapped(
        "##first", first,
        {.width = 340.0f,
         .selection_group = group,
         .selection_text = combined,
         .selection_offset = 0});
    first_min = ImGui::GetItemRectMin();
    first_max = ImGui::GetItemRectMax();
    gem16::studio::selectable_text::Wrapped(
        "##second", second,
        {.width = 340.0f,
         .selection_group = group,
         .selection_text = combined,
         .selection_offset = first.size() + 1U});
    second_min = ImGui::GetItemRectMin();
    second_max = ImGui::GetItemRectMax();
    ImGui::End();
    ImGui::Render();
  };

  draw_frame();
  io.AddMousePosEvent(first_min.x + 2.0f,
                      (first_min.y + first_max.y) * 0.5f);
  io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
  draw_frame();
  io.AddMousePosEvent(second_max.x - 2.0f,
                      (second_min.y + second_max.y) * 0.5f);
  draw_frame();
  io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
  draw_frame();
  io.AddKeyEvent(ImGuiMod_Ctrl, true);
  io.AddKeyEvent(ImGuiKey_C, true);
  draw_frame();

  io.AddKeyEvent(ImGuiKey_C, false);
  io.AddKeyEvent(ImGuiMod_Ctrl, false);
  ImGui::DestroyContext();
  if (clipboard != combined) {
    std::fprintf(stderr, "grouped selection copied '%s', expected '%s'\n",
                 clipboard.c_str(), combined.c_str());
  }
  return clipboard == combined;
}

bool TestMarkdownParser() {
  using gem16::studio::markdown::BlockKind;
  const std::string source =
      "# Gem 16\n\n"
      "A **strong**, *calm*, `local` [assistant](https://example.com).\n\n"
      "- first item\n"
      "2. second item\n"
      "> quoted answer\n\n"
      "```cpp\nint value = 16;\n```\n"
      "---\n";
  const auto blocks = gem16::studio::markdown::Parse(source);
  if (blocks.size() != 7 || blocks[0].kind != BlockKind::kHeading ||
      blocks[0].level != 1 || blocks[0].text != "Gem 16" ||
      blocks[1].kind != BlockKind::kParagraph ||
      blocks[1].text != "A strong, calm, local assistant." ||
      blocks[2].kind != BlockKind::kBulletItem ||
      blocks[3].kind != BlockKind::kOrderedItem || blocks[3].ordinal != 2 ||
      blocks[4].kind != BlockKind::kQuote ||
      blocks[5].kind != BlockKind::kCode || blocks[5].info != "cpp" ||
      blocks[5].text != "int value = 16;" ||
      blocks[6].kind != BlockKind::kRule) {
    return false;
  }
  bool strong = false;
  bool emphasis = false;
  bool code = false;
  bool link = false;
  for (const auto& span : blocks[1].spans) {
    strong |= span.strong;
    emphasis |= span.emphasis;
    code |= span.code;
    link |= span.link && span.destination == "https://example.com";
    if (span.begin >= span.end || span.end > blocks[1].text.size()) return false;
  }
  const auto streaming = gem16::studio::markdown::Parse("```html\n<div>streaming");
  return strong && emphasis && code && link && streaming.size() == 1 &&
         streaming[0].kind == BlockKind::kCode &&
         streaming[0].text == "<div>streaming";
}

bool TestChatHistoryUndo() {
  std::vector<gem16::studio::ChatMessage> messages{
      {"user", "first", {}, false, false},
      {"assistant", "first answer", {}, false, false},
      {"user", "second", {}, false, false},
      {"assistant", "second answer", {}, false, false}};
  if (!gem16::studio::RemoveLastExchange(messages) || messages.size() != 2 ||
      messages.back().content != "first answer") {
    return false;
  }
  if (!gem16::studio::RemoveLastExchange(messages) || !messages.empty())
    return false;
  messages.push_back({"assistant", "orphan", {}, false, false});
  return !gem16::studio::RemoveLastExchange(messages) && messages.size() == 1;
}

bool TestChatFontsAndSpacing() {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.IniFilename = nullptr;
  io.DisplaySize = {900, 700};
  io.DeltaTime = 1.0f / 60.0f;
  ImFont* font = gem16::studio::InitializeStudioFonts();
  unsigned char* pixels = nullptr;
  int atlas_width = 0, atlas_height = 0;
  io.Fonts->GetTexDataAsRGBA32(&pixels, &atlas_width, &atlas_height);
  bool passed = sizeof(ImWchar) == 4 && pixels && atlas_width > 0;
#ifdef _WIN32
  // Actual Windows fallback glyphs, not the replacement character. Verify
  // supplementary UTF-8 measures as ONE glyph as well as baking visibly.
  const std::pair<ImWchar, const char*> emojis[] = {
      {0x1f60a, "\xf0\x9f\x98\x8a"}, {0x1f44d, "\xf0\x9f\x91\x8d"},
      {0x1f680, "\xf0\x9f\x9a\x80"}};
  for (const auto& [codepoint, utf8] : emojis) {
    const ImFontGlyph* glyph = font->GetFontBaked(17.0f)->FindGlyphNoFallback(codepoint);
    passed &= glyph && glyph->Visible && glyph->Codepoint == codepoint;
    if (glyph) {
      const float width = font->CalcTextSizeA(17.0f, 1000, 0, utf8).x;
      passed &= std::abs(width - glyph->AdvanceX) < 0.01f;
    }
  }
#endif
  for (const float scale : {1.0f, 1.25f, 1.5f}) {
    ImGui::NewFrame();
    ImGui::SetNextWindowSize({900, 700});
    ImGui::Begin("##chat-format-test");
    ImGui::SetWindowFontScale(scale);
    ImGui::GetStyle().ItemSpacing = {11 * scale, 11 * scale};
    const float unit = ImGui::GetFontSize() / 17.0f;
    const float line = ImGui::GetFontSize() + 3 * unit;
    const auto height = [](const char* id, const std::string& source, float width) {
      const float before = ImGui::GetCursorPosY();
      gem16::studio::markdown::Render(id, source, width);
      return ImGui::GetCursorPosY() - before;
    };
    const float bullets = height("bullets", "- First\n\n- Second", 600);
    const float ordered = height("ordered", "1. First\n2. Second", 600);
    const float paragraphs = height("paragraphs", "First\n\nSecond", 600);
    passed &= std::abs(bullets - (2 * line + 2 * unit)) < 2.0f;
    passed &= std::abs(ordered - bullets) < 1.0f;
    passed &= paragraphs > bullets + 6 * unit;
    const float wrapped = height("wrapped", "- A longer item that wraps to more than one line\n- Second", 140);
    passed &= wrapped > bullets + line;
    passed &= ImGui::GetStyle().ItemSpacing.y == 11 * scale;
    ImGui::End();
    ImGui::Render();
  }
  ImGui::DestroyContext();
  return passed;
}

bool TestComposerEnterBehavior() {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.IniFilename = nullptr;
  io.DisplaySize = {480.0f, 260.0f};
  io.DeltaTime = 1.0f / 60.0f;
  io.ConfigInputTrickleEventQueue = false;
  unsigned char* pixels = nullptr;
  int atlas_width = 0;
  int atlas_height = 0;
  io.Fonts->GetTexDataAsRGBA32(&pixels, &atlas_width, &atlas_height);
  std::array<char, 128> buffer{};
  std::snprintf(buffer.data(), buffer.size(), "hello");

  const auto draw_frame = [&buffer] {
    ImGui::NewFrame();
    ImGui::SetNextWindowPos({0.0f, 0.0f});
    ImGui::SetNextWindowSize({480.0f, 260.0f});
    ImGui::Begin("##composer-test", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings);
    ImGui::SetCursorScreenPos({24.0f, 24.0f});
    const bool submitted = ImGui::InputTextMultiline(
        "##message", buffer.data(), buffer.size(), {360.0f, 100.0f},
        ImGuiInputTextFlags_EnterReturnsTrue |
            ImGuiInputTextFlags_CtrlEnterForNewLine);
    ImGui::End();
    ImGui::Render();
    return submitted;
  };

  draw_frame();
  io.AddMousePosEvent(70.0f, 45.0f);
  io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
  draw_frame();
  io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
  draw_frame();
  io.AddKeyEvent(ImGuiKey_Enter, true);
  const bool enter_submitted = draw_frame();
  io.AddKeyEvent(ImGuiKey_Enter, false);
  draw_frame();

  io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
  draw_frame();
  io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
  draw_frame();
  io.AddKeyEvent(ImGuiMod_Shift, true);
  io.AddKeyEvent(ImGuiKey_Enter, true);
  const bool shift_submitted = draw_frame();
  io.AddKeyEvent(ImGuiKey_Enter, false);
  io.AddKeyEvent(ImGuiMod_Shift, false);
  draw_frame();

  const std::string result(buffer.data());
  ImGui::DestroyContext();
  return enter_submitted && !shift_submitted && result.find('\n') != std::string::npos;
}

bool TestFailedChatHistory() {
  using namespace gem16::studio;
  ChatMessage partial{"assistant", "Useful partial answer", "thought", true};
  ApplyChatEvent(partial, {ChatEvent::Kind::kError, "Transport failed"});
  if (partial.content != "Useful partial answer" || partial.reasoning != "thought" ||
      partial.error_message != "Transport failed" || !partial.error || partial.streaming) return false;
  std::vector<ChatMessage> messages{{"user", "Old question"}, partial, {"user", "New question"}};
  const auto payload = BuildChatPayload({}, {}, messages);
  if (payload.find("Transport failed") != std::string::npos ||
      payload.find("Useful partial") != std::string::npos ||
      payload.find("Old question") != std::string::npos ||
      payload.find("New question") == std::string::npos) return false;
  ChatMessage cancelled{"assistant", "Still useful", {}, true};
  ApplyChatEvent(cancelled, {ChatEvent::Kind::kFinished, "cancelled"});
  return cancelled.content == "Still useful" && cancelled.error &&
         !cancelled.streaming && cancelled.error_message == "Generation stopped.";
}

bool TestCancelBeforeChatDispatch() {
  using namespace std::chrono_literals;
  httplib::Server server;
  std::promise<void> entered, release;
  auto released = release.get_future().share();
  std::atomic<int> posts{0};
  server.Get("/metrics", [&](const auto&, auto& response) {
    entered.set_value();
    released.wait_for(3s);
    response.set_content("", "text/plain");
  });
  server.Post("/v1/chat/completions", [&](const auto&, auto& response) {
    ++posts;
    response.set_content("data: [DONE]\n\n", "text/event-stream");
  });
  gem16::studio::ServerConfig config;
  config.port = server.bind_to_any_port("127.0.0.1");
  if (config.port <= 0) return false;
  std::jthread listener([&] { server.listen_after_bind(); });
  gem16::studio::ApiClient client;
  client.StreamChat(config, {}, {{"user", "Hi"}}, {});
  const bool started = entered.get_future().wait_for(2s) == std::future_status::ready;
  auto cancel = std::async(std::launch::async, [&] { client.Cancel(); });
  const bool prompt = cancel.wait_for(250ms) == std::future_status::ready;
  release.set_value();
  cancel.get();
  for (int i = 0; i < 300 && client.Busy(); ++i) std::this_thread::sleep_for(10ms);
  const bool finished = !client.Busy();
  const auto events = client.DrainEvents();
  server.stop();
  listener.join();
  return started && prompt && finished && posts == 0 &&
         std::count_if(events.begin(), events.end(), [](const auto& event) {
           return event.kind == gem16::studio::ChatEvent::Kind::kFinished && event.value == "cancelled";
         }) == 1;
}

bool TestStreamingErrorPreservesPartial() {
  using namespace gem16::studio;
  httplib::Server server;
  server.Post("/v1/chat/completions", [](const auto&, auto& response) {
    response.set_content(
        "data: {\"choices\":[{\"delta\":{\"content\":\"Partial\"}}]}\n\n"
        "data: {\"error\":{\"message\":\"Original server error\"}}\n\n",
        "text/event-stream");
  });
  ServerConfig config;
  config.port = server.bind_to_any_port("127.0.0.1");
  if (config.port <= 0) return false;
  std::jthread listener([&] { server.listen_after_bind(); });
  ApiClient client;
  client.StreamChat(config, {}, {{"user", "Hi"}}, {});
  for (int i = 0; i < 300 && client.Busy(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  const auto events = client.DrainEvents();
  const bool finished = !client.Busy();
  client.Cancel();
  server.stop();
  listener.join();
  ChatMessage message{"assistant", {}, {}, true};
  int errors = 0;
  for (const auto& event : events) {
    errors += event.kind == ChatEvent::Kind::kError ? 1 : 0;
    ApplyChatEvent(message, event);
  }
  return finished && errors == 1 && message.content == "Partial" &&
         message.error_message == "Original server error" && message.error;
}

bool TestReverifySameSizeCorruption() {
  using namespace gem16::studio;
  const char* previous = std::getenv("HF_HUB_CACHE");
  const std::optional<std::string> saved = previous ? std::optional<std::string>(previous) : std::nullopt;
  const auto cache = std::filesystem::temp_directory_path() /
      ("gem16-reverify-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  SetEnvironment("HF_HUB_CACHE", cache.string());
  const ModelCatalogFile file{"weights.bin", 3,
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "blob", "test/model", "revision", "weights.bin"};
  const ModelComponentCatalog component{"test", "Test", "test/model", "revision", {&file, 1}, false, ""};
  const ModelProfileComponent target{ModelComponentKind::kTarget, &component, true};
  const ModelProfileCatalog profile{ModelProfile::kGemma4Unified12B, "test", "test", {&target, 1}};
  const auto blob = RepositoryDirectory(file.source_repository, cache) / "blobs" / file.blob_id;
  const auto view = ComponentDirectory(component, cache) / file.path;
  const auto marker = VerificationMarkerPath(file, cache);
  std::filesystem::create_directories(blob.parent_path());
  std::filesystem::create_directories(view.parent_path());
  std::ofstream(blob, std::ios::binary) << "abc";
  std::filesystem::create_hard_link(blob, view);
  bool valid = true;
  {
    ModelManager manager({&profile, 1});
    const auto verify = [&] {
      manager.VerifyInstalled();
      for (int i = 0; i < 300 && manager.State().verifying; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      return manager.State();
    };
    auto state = verify();
    valid = !state.verifying && state.error.empty() && state.For(profile.profile).Ready() &&
            state.verification_bytes == 3 && state.verification_total_bytes == 3;
    std::ofstream(blob, std::ios::binary | std::ios::trunc) << "xbc";
    manager.Refresh();
    valid = valid && manager.State().For(profile.profile).Ready(); // Cached status alone cannot see this.
    state = verify();
    valid = valid && !state.verifying && !state.error.empty() &&
            !state.For(profile.profile).Ready() && !std::filesystem::exists(marker) &&
            std::filesystem::exists(blob); // Never delete shared payloads during verification.
    std::ofstream(blob, std::ios::binary | std::ios::trunc) << "abc";
    state = verify();
    valid = valid && state.error.empty() && state.For(profile.profile).Ready();
    std::filesystem::remove(view);
    std::ofstream(view, std::ios::binary) << "bad"; // Detached, same-size view.
    state = verify();
    valid = valid && !state.error.empty() && !state.For(profile.profile).Ready() &&
            !std::filesystem::exists(marker);
  }
  if (saved) SetEnvironment("HF_HUB_CACHE", *saved); else ClearEnvironment("HF_HUB_CACHE");
  std::filesystem::remove_all(cache);
  return valid;
}

bool TestStreamingClient() {
  httplib::Server server;
  server.Post("/v1/chat/completions", [](const httplib::Request& request,
                                         httplib::Response& response) {
    if (request.get_header_value("X-Gem16-Session-Id") != "session_existing") {
      response.status = 400;
      return;
    }
    response.set_header("X-Gem16-Session-Id", "session_returned");
    const std::string stream =
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"Think\"}}]}\n\n"
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hello \\ud83d\\ude0a\"}}]}\n\n"
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":2,\"total_tokens\":9}}\n\n"
        "data: [DONE]\n\n";
    response.set_content(stream, "text/event-stream");
  });
  const int port = server.bind_to_any_port("127.0.0.1");
  if (port <= 0) return false;
  std::jthread server_thread([&server] { server.listen_after_bind(); });

  gem16::studio::ApiClient client;
  gem16::studio::ServerConfig config;
  config.port = port;
  config.model_name = "gem16-test";
  gem16::studio::GenerationConfig generation;
  std::vector<gem16::studio::ChatMessage> messages{
      {"user", "Say hello", {}, false, false}};
  client.StreamChat(config, generation, messages, "session_existing");
  for (int attempt = 0; attempt < 200 && client.Busy(); ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  const auto events = client.DrainEvents();
  client.Cancel();
  server.stop();
  server_thread.join();

  bool text = false;
  bool reasoning = false;
  bool finished = false;
  bool session = false;
  bool usage = false;
  bool error = false;
  for (const auto& event : events) {
    text |= event.kind == gem16::studio::ChatEvent::Kind::kText &&
            event.value == "Hello \xf0\x9f\x98\x8a";
    reasoning |= event.kind == gem16::studio::ChatEvent::Kind::kReasoning && event.value == "Think";
    finished |= event.kind == gem16::studio::ChatEvent::Kind::kFinished;
    session |= event.kind == gem16::studio::ChatEvent::Kind::kSession && event.value == "session_returned";
    usage |= event.kind == gem16::studio::ChatEvent::Kind::kUsage &&
             event.prompt_tokens == 7 && event.completion_tokens == 2;
    error |= event.kind == gem16::studio::ChatEvent::Kind::kError;
  }
  return text && reasoning && usage && finished && session && !error;
}

}  // namespace

int main(int argc, char** argv) {
#ifndef _WIN32
  // Host UI tests must never exhaust the desktop's RAM, even on malformed
  // input.
  rlimit memory{};
  if (getrlimit(RLIMIT_AS, &memory) != 0) return 1;
  constexpr rlim_t cap = 2ULL * 1024ULL * 1024ULL * 1024ULL;
  memory.rlim_cur = std::min(memory.rlim_cur, cap);
  memory.rlim_max = std::min(memory.rlim_max, cap);
  if (setrlimit(RLIMIT_AS, &memory) != 0) return 1;
#endif
  if (argc > 2 && std::string_view(argv[1]) == "--model")
    return RunStudioTestServer(argc, argv);
  if (argc == 2 && std::string_view(argv[1]) == "--studio-lifecycle")
    return TestStudioLifecycle(std::filesystem::absolute(argv[0])) ? 0 : 1;
  if (argc == 2 && std::string_view(argv[1]) == "--markdown")
    return TestExtendedMarkdown() ? 0 : 1;
  if (!TestCanvas() || !TestChatStore()) return 1;
  if (!TestModelLifecycle()) return 1;
  if (!TestFailedChatHistory() || !TestStreamingErrorPreservesPartial() || !TestCancelBeforeChatDispatch() || !TestReverifySameSizeCorruption()) {
    std::fprintf(stderr, "chat cancellation/history or model re-verification regression failed\n");
    return 1;
  }
  if (!TestModelCardLayoutAndProgress()) {
    std::fprintf(stderr, "compact model-card layout/progress animation test failed\n");
    return 1;
  }
  if (!TestTextSelection()) {
    std::fprintf(stderr, "selectable text test failed\n");
    return 1;
  }
  if (!TestSelectableTextWidgetClipboard()) {
    std::fprintf(stderr, "selectable text clipboard integration test failed\n");
    return 1;
  }
  if (!TestGroupedSelectableTextClipboard()) {
    std::fprintf(stderr, "grouped selectable text clipboard test failed\n");
    return 1;
  }
  if (!TestMarkdownParser()) {
    std::fprintf(stderr, "markdown parser test failed\n");
    return 1;
  }

  if (!TestChatFontsAndSpacing()) {
    std::fprintf(stderr, "chat font/emoji and Markdown spacing test failed\n");
    return 1;
  }
  if (!TestChatHistoryUndo()) {
    std::fprintf(stderr, "chat history undo test failed\n");
    return 1;
  }
  if (!TestComposerEnterBehavior()) {
    std::fprintf(stderr, "composer Enter/Shift+Enter test failed\n");
    return 1;
  }
  if (!TestServerCommand()) {
    std::fprintf(stderr, "server command test failed\n");
    return 1;
  }
  if (!TestQualified26BDefaults()) {
    std::fprintf(stderr, "qualified 26B defaults test failed\n");
    return 1;
  }
  if (!TestVision26BDefaultsAndCommand()) {
    std::fprintf(stderr, "Vision 26B defaults/command test failed\n");
    return 1;
  }
  if (!TestVisionAttachmentPolicyAndEstimate()) {
    std::fprintf(stderr, "Vision attachment policy/estimate test failed\n");
    return 1;
  }
  if (!TestLiveServerCompatibility()) {
    std::fprintf(stderr, "live server compatibility test failed\n");
    return 1;
  }
  if (!TestModelCatalog()) {
    std::fprintf(stderr, "model catalog test failed\n");
    return 1;
  }
  if (!TestNeutralFirstRunDefaults()) {
    std::fprintf(stderr, "neutral first-run defaults test failed\n");
    return 1;
  }
  if (!TestUiScaleResolution()) {
    std::fprintf(stderr, "UI scale resolution test failed\n");
    return 1;
  }
  if (!TestOnboardingPersistence()) {
    std::fprintf(stderr, "onboarding persistence test failed\n");
    return 1;
  }
  if (!TestMediaPayload()) {
    std::fprintf(stderr, "multimodal payload test failed\n");
    return 1;
  }
  if (!TestPerformanceMetrics()) {
    std::fprintf(stderr, "performance metrics test failed\n");
    return 1;
  }
  if (!TestPreviewImageDecode()) {
    std::fprintf(stderr, "preview image decode test failed\n");
    return 1;
  }
  if (!TestMediaLoader()) {
    std::fprintf(stderr, "media loader test failed\n");
    return 1;
  }
  if (!TestEmptyCacheInstallState()) {
    std::fprintf(stderr, "empty-cache install state test failed\n");
    return 1;
  }
  if (!TestComponentRemovalKeepsSharedBlob()) {
    std::fprintf(stderr, "component removal/shared blob test failed\n");
    return 1;
  }
  if (!TestLegacyVisionViewMigration()) {
    std::fprintf(stderr, "legacy Vision view migration test failed\n");
    return 1;
  }
  if (!TestQualifiedVisionCacheIfProvided()) {
    std::fprintf(stderr, "qualified Vision cache test failed\n");
    return 1;
  }
  if (!TestStreamingClient()) {
    std::fprintf(stderr, "streaming client test failed\n");
    return 1;
  }
  std::puts("gem16 native Studio host tests passed");
  return 0;
}

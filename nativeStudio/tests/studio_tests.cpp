#include "api_client.h"
#include "chat_history.h"
#include "gem16_logo.generated.h"
#include "image_texture.h"
#include "markdown.h"
#include "media_loader.h"
#include "model_catalog.h"
#include "model_manager.h"
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
#include <fstream>
#include <string_view>
#include <thread>

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
         Contains(command, "2") && !Contains(command, "--mtp-adaptive");
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

bool TestModelCatalog() {
  const auto catalog = gem16::studio::ModelCatalog();
  if (catalog.size() != 2) return false;
  const auto& twelve = gem16::studio::CatalogForProfile(
      gem16::studio::ModelProfile::kGemma4Unified12B);
  const auto& twenty_six = gem16::studio::CatalogForProfile(
      gem16::studio::ModelProfile::kGemma4Moe26BA4B);
  if (std::string_view(twelve.target->repository) !=
          "unsloth/gemma-4-12b-it-NVFP4" ||
      std::string_view(twelve.assistant->repository) !=
          "google/gemma-4-12B-it-assistant" ||
      !twelve.target->composed_view || twelve.assistant->composed_view ||
      twenty_six.target->composed_view || !twenty_six.assistant->composed_view ||
      std::string_view(twenty_six.assistant->repository) !=
          "danmoreng/gemma-4-26B-A4B-it-GEM16") {
    return false;
  }
  bool external_tokenizer = false;
  for (const auto& file : twelve.target->files) {
    external_tokenizer |=
        std::string_view(file.path) == "tokenizer_config.json" &&
        std::string_view(file.source_repository) == "google/gemma-4-12B-it";
  }
  const auto root = std::filesystem::path("/hub");
  const auto& first_target_file = twelve.target->files.front();
  return external_tokenizer &&
         gem16::studio::ComponentDirectory(*twelve.target, root) ==
             root / ".gem16/snapshots/"
                    "unsloth--gemma-4-12b-it-NVFP4--"
                    "b1f649734b34aa5575b03d186abd1b9be3d0d5c4" &&
         gem16::studio::ComponentDirectory(*twenty_six.target, root) ==
             root / "models--danmoreng--gemma-4-26B-A4B-it-GEM16/snapshots/"
                    "31842e12882d09bab7109c0ad52a4ee2e945069c" &&
         gem16::studio::ComponentDirectory(*twenty_six.assistant, root) ==
             root / ".gem16/snapshots/"
                    "danmoreng--gemma-4-26B-A4B-it-GEM16--"
                    "31842e12882d09bab7109c0ad52a4ee2e945069c--assistant" &&
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
      settings.server, gem16::studio::ModelProfile::kGemma4Moe26BA4B);
  valid = valid && gem16::studio::SaveSettings(settings);
  const auto loaded = gem16::studio::LoadSettings();
  valid = valid && loaded.onboarding_complete && loaded.ui_scale == 1.25f &&
          loaded.server.profile ==
              gem16::studio::ModelProfile::kGemma4Moe26BA4B;

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
  return parsed.ok() && payload.find("image_url") != std::string::npos &&
         payload.find("input_audio") != std::string::npos &&
         payload.find("native studio") != std::string::npos &&
         payload.find("AQID") != std::string::npos;
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

  std::array<std::uint64_t, 2> expected{};
  for (const auto& profile : gem16::studio::ModelCatalog()) {
    const auto index = profile.profile ==
                               gem16::studio::ModelProfile::kGemma4Moe26BA4B
                           ? 1U
                           : 0U;
    for (const auto& file : profile.target->files) expected[index] += file.size;
    for (const auto& file : profile.assistant->files) expected[index] += file.size;
  }
  gem16::studio::ModelManager manager;
  const auto state = manager.State();
  const bool valid =
      !state.downloading && state.For(gem16::studio::ModelProfile::kGemma4Unified12B)
                                .required_download_bytes == expected[0] &&
      state.For(gem16::studio::ModelProfile::kGemma4Moe26BA4B)
              .required_download_bytes == expected[1] &&
      state.For(gem16::studio::ModelProfile::kGemma4Unified12B).storage_available &&
      state.For(gem16::studio::ModelProfile::kGemma4Moe26BA4B).storage_available;

  if (previous == nullptr)
    ClearEnvironment("HF_HUB_CACHE");
  else
    SetEnvironment("HF_HUB_CACHE", previous_value);
  std::error_code error;
  std::filesystem::remove_all(cache, error);
  return valid && !error;
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
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n"
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
    text |= event.kind == gem16::studio::ChatEvent::Kind::kText && event.value == "Hello";
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

int main() {
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
  if (!TestStreamingClient()) {
    std::fprintf(stderr, "streaming client test failed\n");
    return 1;
  }
  std::puts("gem16 native Studio host tests passed");
  return 0;
}

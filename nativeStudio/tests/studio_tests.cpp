#include "api_client.h"
#include "chat_history.h"
#include "markdown.h"
#include "server_manager.h"
#include "selectable_text.h"

#include "httplib.h"
#include "imgui.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <thread>

namespace {

void CaptureClipboard(void* user_data, const char* text) {
  *static_cast<std::string*>(user_data) = text == nullptr ? "" : text;
}

bool Contains(const std::vector<std::string>& values, const std::string& expected) {
  return std::find(values.begin(), values.end(), expected) != values.end();
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
  bool error = false;
  for (const auto& event : events) {
    text |= event.kind == gem16::studio::ChatEvent::Kind::kText && event.value == "Hello";
    reasoning |= event.kind == gem16::studio::ChatEvent::Kind::kReasoning && event.value == "Think";
    finished |= event.kind == gem16::studio::ChatEvent::Kind::kFinished;
    session |= event.kind == gem16::studio::ChatEvent::Kind::kSession && event.value == "session_returned";
    error |= event.kind == gem16::studio::ChatEvent::Kind::kError;
  }
  return text && reasoning && finished && session && !error;
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
  if (!TestStreamingClient()) {
    std::fprintf(stderr, "streaming client test failed\n");
    return 1;
  }
  std::puts("gem16 native Studio host tests passed");
  return 0;
}

#include "app.h"

#include "chat_history.h"
#include "markdown.h"
#include "media_loader.h"
#include "model_catalog.h"
#include "selectable_text.h"
#include "settings.h"
#include "platform_ui.h"

#include "imgui.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numbers>

namespace gem16::studio {
namespace {

constexpr ImVec4 kAccent{0.31f, 0.91f, 0.65f, 1.0f};
constexpr ImVec4 kAccentDim{0.11f, 0.34f, 0.24f, 1.0f};
float g_ui_scale = 1.0f;

float Ui(float value) { return value * g_ui_scale; }

template <std::size_t Size>
void CopyTo(std::array<char, Size>& destination, const std::string& source) {
  const std::size_t count = std::min(source.size(), Size - 1);
  std::memcpy(destination.data(), source.data(), count);
  destination[count] = '\0';
}

const char* ScreenTitle(Screen screen) {
  switch (screen) {
    case Screen::kChat: return "Chat";
    case Screen::kModels: return "Models";
    case Screen::kServer: return "Server";
    case Screen::kSettings: return "Settings";
  }
  return "gem16";
}

const char* PhaseLabel(ServerPhase phase) {
  switch (phase) {
    case ServerPhase::kStopped: return "Stopped";
    case ServerPhase::kStarting: return "Starting";
    case ServerPhase::kRunning: return "Running";
    case ServerPhase::kExternal: return "Attached";
    case ServerPhase::kStopping: return "Stopping";
    case ServerPhase::kError: return "Error";
  }
  return "Unknown";
}

ImVec4 PhaseColor(ServerPhase phase) {
  if (phase == ServerPhase::kRunning || phase == ServerPhase::kExternal) return kAccent;
  if (phase == ServerPhase::kError) return {1.0f, 0.39f, 0.39f, 1.0f};
  if (phase == ServerPhase::kStarting || phase == ServerPhase::kStopping) return {1.0f, 0.76f, 0.30f, 1.0f};
  return {0.57f, 0.61f, 0.60f, 1.0f};
}

void DrawGemstone(ImDrawList* draw, ImVec2 center, float radius) {
  const ImU32 edge = IM_COL32(38, 245, 170, 255);
  const ImU32 deep = IM_COL32(5, 88, 65, 255);
  const ImU32 middle = IM_COL32(8, 161, 109, 255);
  const ImU32 shine = IM_COL32(112, 255, 205, 255);
  ImVec2 outer[8];
  ImVec2 inner[8];
  for (int index = 0; index < 8; ++index) {
    const float angle = -std::numbers::pi_v<float> * 0.5f +
                        static_cast<float>(index) * std::numbers::pi_v<float> * 0.25f;
    outer[index] = {center.x + std::cos(angle) * radius,
                    center.y + std::sin(angle) * radius};
    inner[index] = {center.x + std::cos(angle) * radius * 0.56f,
                    center.y + std::sin(angle) * radius * 0.56f};
  }
  draw->AddConvexPolyFilled(outer, 8, deep);
  for (int index = 0; index < 8; ++index) {
    const ImVec2 triangle[3] = {outer[index], outer[(index + 1) % 8],
                                inner[(index + 1) % 8]};
    draw->AddConvexPolyFilled(triangle, 3,
                              index % 3 == 0 ? shine : middle);
    const ImVec2 bridge[3] = {outer[index], inner[(index + 1) % 8],
                              inner[index]};
    draw->AddConvexPolyFilled(bridge, 3,
                              index % 2 == 0 ? middle : deep);
  }
  draw->AddConvexPolyFilled(inner, 8, IM_COL32(3, 69, 52, 255));
  draw->AddPolyline(outer, 8, edge, ImDrawFlags_Closed, Ui(1.2f));
  draw->AddLine(inner[5], inner[1], IM_COL32(100, 255, 203, 125), Ui(1.0f));
}

void DrawNavIcon(ImDrawList* draw, Screen screen, ImVec2 center, ImU32 color) {
  if (screen == Screen::kChat) {
    draw->AddCircle(center, Ui(9.0f), color, 18, Ui(1.7f));
    draw->AddLine({center.x - Ui(6.0f), center.y + Ui(6.0f)},
                  {center.x - Ui(9.0f), center.y + Ui(11.0f)}, color,
                  Ui(1.7f));
  } else if (screen == Screen::kModels) {
    draw->AddRect({center.x - Ui(9.0f), center.y - Ui(8.0f)},
                  {center.x + Ui(9.0f), center.y + Ui(8.0f)}, color, Ui(2.0f),
                  0, Ui(1.5f));
    draw->AddLine({center.x, center.y - Ui(8.0f)},
                  {center.x, center.y + Ui(8.0f)}, color, Ui(1.2f));
    draw->AddLine({center.x - Ui(9.0f), center.y - Ui(2.0f)},
                  {center.x + Ui(9.0f), center.y - Ui(2.0f)}, color,
                  Ui(1.2f));
  } else if (screen == Screen::kServer) {
    for (int row = -1; row <= 1; ++row) {
      draw->AddRect({center.x - Ui(10.0f), center.y + row * Ui(7.0f) - Ui(2.0f)},
                    {center.x + Ui(10.0f), center.y + row * Ui(7.0f) + Ui(2.0f)},
                    color, Ui(1.5f), 0, Ui(1.4f));
      draw->AddCircleFilled(
          {center.x + Ui(6.5f), center.y + row * Ui(7.0f)}, Ui(1.2f), color);
    }
  } else {
    draw->AddCircle(center, Ui(8.0f), color, 16, Ui(1.7f));
    draw->AddCircle(center, Ui(2.5f), color, 12, Ui(1.5f));
    for (int index = 0; index < 8; ++index) {
      const float angle = static_cast<float>(index) * std::numbers::pi_v<float> * 0.25f;
      draw->AddLine({center.x + std::cos(angle) * Ui(8.0f),
                     center.y + std::sin(angle) * Ui(8.0f)},
                    {center.x + std::cos(angle) * Ui(11.0f),
                     center.y + std::sin(angle) * Ui(11.0f)}, color, Ui(1.5f));
    }
  }
}

bool NavButton(const char* label, Screen screen, bool selected, float width) {
  static std::array<float, 4> glow_strength{};
  ImGui::InvisibleButton(label, {width, Ui(48.0f)});
  const bool clicked = ImGui::IsItemClicked();
  const bool hovered = ImGui::IsItemHovered();
  const std::size_t glow_index = static_cast<std::size_t>(screen);
  const float target = selected ? 1.0f : (hovered ? 0.72f : 0.0f);
  const float response =
      1.0f - std::exp(-ImGui::GetIO().DeltaTime * (target > glow_strength[glow_index]
                                                       ? 18.0f
                                                       : 11.0f));
  glow_strength[glow_index] +=
      (target - glow_strength[glow_index]) * response;

  const float glow = glow_strength[glow_index];
  const ImVec2 minimum = ImGui::GetItemRectMin();
  const ImVec2 maximum = ImGui::GetItemRectMax();
  const ImVec2 center{minimum.x + Ui(28.0f), (minimum.y + maximum.y) * 0.5f};
  ImDrawList* draw = ImGui::GetWindowDrawList();
  draw->PushClipRect(minimum, maximum, true);
  if (selected || glow > 0.01f) {
    draw->AddRectFilled(
        minimum, maximum,
        IM_COL32(8, 61, 43,
                 static_cast<int>((selected ? 162.0f : 105.0f) * glow)),
        Ui(11.0f), ImDrawFlags_RoundCornersRight);
    draw->AddCircleFilled(
        {minimum.x - Ui(2.0f), center.y}, Ui(34.0f),
        IM_COL32(38, 244, 164, static_cast<int>(42.0f * glow)), 36);
    draw->AddRectFilledMultiColor(
        {minimum.x + Ui(1.0f), minimum.y + Ui(2.0f)},
        {minimum.x + width * 0.72f, maximum.y - Ui(2.0f)},
        IM_COL32(37, 239, 160, static_cast<int>(84.0f * glow)),
        IM_COL32(37, 239, 160, 0), IM_COL32(37, 239, 160, 0),
        IM_COL32(37, 239, 160, static_cast<int>(84.0f * glow)));
    draw->AddRectFilled(
        {minimum.x, minimum.y + Ui(5.0f)},
        {minimum.x + Ui(2.0f), maximum.y - Ui(5.0f)},
        IM_COL32(76, 255, 190, static_cast<int>(245.0f * glow)), Ui(1.0f));
    draw->AddRectFilled(
        {minimum.x + Ui(2.0f), minimum.y + Ui(8.0f)},
        {minimum.x + Ui(5.0f), maximum.y - Ui(8.0f)},
        IM_COL32(40, 244, 164, static_cast<int>(92.0f * glow)), Ui(2.0f));
  }
  draw->PopClipRect();

  const ImVec4 idle{0.52f, 0.58f, 0.57f, 1.0f};
  const ImVec4 lit = selected || hovered
                         ? ImVec4(kAccent.x, kAccent.y, kAccent.z, 1.0f)
                         : idle;
  DrawNavIcon(draw, screen, center, ImGui::ColorConvertFloat4ToU32(lit));
  const ImVec2 text_size = ImGui::CalcTextSize(label);
  draw->AddText({minimum.x + Ui(53.0f),
                 minimum.y + (Ui(48.0f) - text_size.y) * 0.5f},
                ImGui::ColorConvertFloat4ToU32(
                    selected ? kAccent
                             : (hovered ? ImVec4(0.78f, 0.97f, 0.88f, 1.0f)
                                        : ImGui::GetStyleColorVec4(ImGuiCol_Text))),
                label);
  return clicked;
}

void StatusPill(const char* text, ImVec4 color) {
  const ImVec2 padding(Ui(10), Ui(5));
  const ImVec2 text_size = ImGui::CalcTextSize(text);
  const ImVec2 min = ImGui::GetCursorScreenPos();
  const ImVec2 max(min.x + text_size.x + padding.x * 2, min.y + text_size.y + padding.y * 2);
  ImGui::GetWindowDrawList()->AddRectFilled(min, max, ImGui::ColorConvertFloat4ToU32({color.x * 0.20f, color.y * 0.20f, color.z * 0.20f, 0.95f}), Ui(12));
  ImGui::GetWindowDrawList()->AddCircleFilled({min.x + Ui(9), (min.y + max.y) * 0.5f}, Ui(3), ImGui::ColorConvertFloat4ToU32(color));
  ImGui::SetCursorScreenPos({min.x + padding.x + Ui(6), min.y + padding.y});
  ImGui::TextColored(color, "%s", text);
  ImGui::SetCursorScreenPos({max.x, min.y});
  ImGui::Dummy({0, max.y - min.y});
}

void PanelHeading(const char* title, const char* description) {
  ImGui::SetWindowFontScale(1.16f);
  ImGui::TextColored(kAccent, "%s", title);
  ImGui::SetWindowFontScale(1.0f);
  if (description != nullptr && description[0] != '\0') {
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    ImGui::TextWrapped("%s", description);
    ImGui::PopStyleColor();
  }
  ImGui::Dummy({0, Ui(8)});
}

void FieldLabel(const char* label) {
  ImGui::TextColored({0.72f, 0.77f, 0.75f, 1.0f}, "%s", label);
  ImGui::SetNextItemWidth(-1.0f);
}

template <std::size_t Size>
bool TextField(const char* label, const char* id,
               std::array<char, Size>& buffer) {
  FieldLabel(label);
  return ImGui::InputText(id, buffer.data(), buffer.size());
}

template <std::size_t Size>
bool PathField(const char* label, const char* id, const char* browse_id,
               std::array<char, Size>& buffer, bool directory) {
  FieldLabel(label);
  ImGui::SetNextItemWidth(std::max(Ui(140.0f), ImGui::GetContentRegionAvail().x - Ui(92.0f)));
  bool changed = ImGui::InputText(id, buffer.data(), buffer.size());
  ImGui::SameLine();
  if (ImGui::Button(browse_id, {Ui(80.0f), 0})) {
    const auto path = directory ? OpenDirectoryDialog() : OpenExecutableDialog();
    if (!path.empty()) {
      CopyTo(buffer, path.string());
      changed = true;
    }
  }
  return changed;
}

void CapabilityChip(const char* text, bool active = true) {
  ImGui::PushStyleColor(
      ImGuiCol_Button,
      active ? ImVec4(0.08f, 0.31f, 0.22f, 0.96f)
             : ImVec4(0.11f, 0.13f, 0.13f, 0.92f));
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                        ImGui::GetStyleColorVec4(ImGuiCol_Button));
  ImGui::PushStyleColor(ImGuiCol_Text,
                        active ? kAccent
                               : ImVec4(0.48f, 0.53f, 0.52f, 1.0f));
  ImGui::SmallButton(text);
  ImGui::PopStyleColor(3);
}

std::string FormatBytes(std::uint64_t bytes) {
  constexpr double gib = 1024.0 * 1024.0 * 1024.0;
  constexpr double mib = 1024.0 * 1024.0;
  char result[64]{};
  if (bytes >= static_cast<std::uint64_t>(gib)) {
    std::snprintf(result, sizeof(result), "%.2f GiB", static_cast<double>(bytes) / gib);
  } else {
    std::snprintf(result, sizeof(result), "%.1f MiB", static_cast<double>(bytes) / mib);
  }
  return result;
}

enum class ComposerIcon { kUndo, kDelete, kSend, kStop };

bool ComposerButton(const char* id, ComposerIcon icon, float size,
                    const char* tooltip, bool disabled = false) {
  const bool clicked = ImGui::Button(id, {size, size});
  const ImVec2 minimum = ImGui::GetItemRectMin();
  const ImVec2 maximum = ImGui::GetItemRectMax();
  const ImVec2 center{(minimum.x + maximum.x) * 0.5f,
                      (minimum.y + maximum.y) * 0.5f};
  ImDrawList* draw = ImGui::GetWindowDrawList();
  const ImU32 color = ImGui::GetColorU32(
      disabled ? ImGuiCol_TextDisabled : ImGuiCol_Text);
  if (icon == ComposerIcon::kUndo) {
    draw->PathArcTo(center, Ui(8.0f), 0.15f * std::numbers::pi_v<float>,
                    1.55f * std::numbers::pi_v<float>, 20);
    draw->PathStroke(color, 0, Ui(1.8f));
    const ImVec2 arrow[3] = {{center.x - Ui(9.0f), center.y - Ui(5.0f)},
                             {center.x - Ui(10.0f), center.y + Ui(3.0f)},
                             {center.x - Ui(3.0f), center.y - Ui(1.0f)}};
    draw->AddConvexPolyFilled(arrow, 3, color);
  } else if (icon == ComposerIcon::kDelete) {
    draw->AddRect({center.x - Ui(6.0f), center.y - Ui(5.0f)},
                  {center.x + Ui(6.0f), center.y + Ui(8.0f)}, color, Ui(2.0f),
                  0, Ui(1.7f));
    draw->AddLine({center.x - Ui(8.0f), center.y - Ui(8.0f)},
                  {center.x + Ui(8.0f), center.y - Ui(8.0f)}, color, Ui(1.7f));
    draw->AddLine({center.x - Ui(3.0f), center.y - Ui(11.0f)},
                  {center.x + Ui(3.0f), center.y - Ui(11.0f)}, color, Ui(1.7f));
  } else if (icon == ComposerIcon::kStop) {
    draw->AddRectFilled({center.x - Ui(6.0f), center.y - Ui(6.0f)},
                        {center.x + Ui(6.0f), center.y + Ui(6.0f)}, color, Ui(2.0f));
  } else {
    const ImVec2 send[3] = {{center.x - Ui(7.0f), center.y - Ui(8.0f)},
                            {center.x + Ui(9.0f), center.y},
                            {center.x - Ui(7.0f), center.y + Ui(8.0f)}};
    draw->AddTriangleFilled(send[0], send[1], send[2], color);
  }
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    ImGui::SetTooltip("%s", tooltip);
  return clicked;
}

int ComposerLineCount(const char* text, float width) {
  if (text == nullptr || text[0] == '\0') return 1;
  width = std::max(width, 80.0f);
  int lines = 0;
  const char* begin = text;
  while (true) {
    const char* end = std::strchr(begin, '\n');
    const char* line_end = end == nullptr ? begin + std::strlen(begin) : end;
    const float measured = ImGui::CalcTextSize(begin, line_end).x;
    lines += std::max(1, static_cast<int>(std::ceil(measured / width)));
    if (end == nullptr) break;
    begin = end + 1;
  }
  return std::clamp(lines, 1, 8);
}

}  // namespace

StudioApp::StudioApp(StudioSettings settings, float ui_scale)
    : settings_(std::move(settings)), ui_scale_(ui_scale) {
  g_ui_scale = ui_scale_;
  sidebar_width_ = Ui(176.0f);
  if (!settings_.onboarding_complete) screen_ = Screen::kModels;
  SyncBuffersFromSettings();
  server_.Configure(settings_.server);
  ApplyTheme();
}

StudioApp::~StudioApp() {
  SyncSettingsFromBuffers();
  (void)SaveSettings(settings_);
  api_.Cancel();
}

void StudioApp::ApplyTheme() const {
  ImGuiStyle& style = ImGui::GetStyle();
  style.WindowRounding = Ui(16);
  style.ChildRounding = Ui(14);
  style.FrameRounding = Ui(10);
  style.PopupRounding = Ui(12);
  style.ScrollbarRounding = Ui(10);
  style.GrabRounding = Ui(9);
  style.WindowBorderSize = 0;
  style.ChildBorderSize = 1;
  style.FrameBorderSize = 0;
  style.WindowPadding = {Ui(18), Ui(18)};
  style.FramePadding = {Ui(13), Ui(10)};
  style.ItemSpacing = {Ui(11), Ui(11)};
  style.ButtonTextAlign = {0.5f, 0.5f};
  style.ScrollbarSize = Ui(14);
  style.GrabMinSize = Ui(12);
  ImVec4* colors = style.Colors;
  if (settings_.dark_theme) {
    colors[ImGuiCol_Text] = {0.91f, 0.94f, 0.93f, 1};
    colors[ImGuiCol_TextDisabled] = {0.47f, 0.52f, 0.51f, 1};
    colors[ImGuiCol_WindowBg] = {0.025f, 0.033f, 0.033f, 0.48f};
    colors[ImGuiCol_ChildBg] = {0.055f, 0.070f, 0.068f, 0.84f};
    colors[ImGuiCol_Border] = {0.16f, 0.21f, 0.20f, 0.72f};
    colors[ImGuiCol_FrameBg] = {0.080f, 0.100f, 0.097f, 0.96f};
    colors[ImGuiCol_FrameBgHovered] = {0.11f, 0.15f, 0.14f, 1};
    colors[ImGuiCol_FrameBgActive] = {0.12f, 0.19f, 0.17f, 1};
    colors[ImGuiCol_Button] = {0.14f, 0.50f, 0.35f, 1};
    colors[ImGuiCol_ButtonHovered] = {0.18f, 0.63f, 0.44f, 1};
    colors[ImGuiCol_ButtonActive] = {0.12f, 0.42f, 0.30f, 1};
    colors[ImGuiCol_Header] = {0.11f, 0.34f, 0.24f, 1};
    colors[ImGuiCol_ScrollbarBg] = {0, 0, 0, 0};
    colors[ImGuiCol_CheckMark] = kAccent;
  } else {
    ImGui::StyleColorsLight();
    style.WindowRounding = Ui(16);
    style.ChildRounding = Ui(14);
    style.FrameRounding = Ui(10);
    colors[ImGuiCol_CheckMark] = {0.05f, 0.42f, 0.25f, 1};
    colors[ImGuiCol_Button] = {0.12f, 0.58f, 0.39f, 1};
    colors[ImGuiCol_ButtonHovered] = {0.09f, 0.48f, 0.32f, 1};
  }
}

void StudioApp::Render() {
  DrainChatEvents();
  const auto dropped = DrainDroppedFiles();
  if (!dropped.empty()) {
    if (screen_ == Screen::kChat)
      AddAttachments(dropped);
    else
      attachment_error_ = "Open Chat before dropping attachments.";
  }
  if (!settings_.onboarding_complete &&
      screen_ != Screen::kModels && screen_ != Screen::kSettings) {
    screen_ = Screen::kModels;
  }
  const ImGuiViewport* viewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(viewport->WorkPos);
  ImGui::SetNextWindowSize(viewport->WorkSize);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
  ImGui::PushStyleColor(ImGuiCol_WindowBg, {0.02f, 0.03f, 0.03f, 0.24f});
  ImGui::Begin("##gem16-root", nullptr,
               ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);
  ImGui::PopStyleColor();
  ImGui::PopStyleVar();

  ImGui::BeginChild("##sidebar", {sidebar_width_, 0}, ImGuiChildFlags_Borders);
  DrawSidebar();
  ImGui::EndChild();
  ImGui::SameLine(0, Ui(12));
  ImGui::BeginChild("##content", {0, 0}, ImGuiChildFlags_Borders);
  DrawHeader();
  ImGui::Separator();
  ImGui::Spacing();
  switch (screen_) {
    case Screen::kChat: DrawChat(); break;
    case Screen::kModels: DrawModels(); break;
    case Screen::kServer: DrawServer(); break;
    case Screen::kSettings: DrawSettings(); break;
  }
  ImGui::EndChild();
  ImGui::End();
}

void StudioApp::DrainChatEvents() {
  for (ChatEvent& event : api_.DrainEvents()) {
    if (event.kind == ChatEvent::Kind::kSession) {
      session_id_ = std::move(event.value);
      continue;
    }
    if (event.kind == ChatEvent::Kind::kUsage) {
      prompt_tokens_ = event.prompt_tokens;
      completion_tokens_ = event.completion_tokens;
      continue;
    }
    if (messages_.empty() || messages_.back().role != "assistant") continue;
    ChatMessage& message = messages_.back();
    if (event.kind == ChatEvent::Kind::kText ||
        event.kind == ChatEvent::Kind::kReasoning) {
      if (streamed_chunks_ == 0) first_token_at_ = std::chrono::steady_clock::now();
      ++streamed_chunks_;
      if (event.kind == ChatEvent::Kind::kText)
        message.content += event.value;
      else
        message.reasoning += event.value;
    }
    else if (event.kind == ChatEvent::Kind::kError) {
      message.content = std::move(event.value);
      message.error = true;
      message.streaming = false;
    } else if (event.kind == ChatEvent::Kind::kFinished) {
      message.streaming = false;
      if (event.value == "cancelled" && message.content.empty())
        message.content = "Generation stopped.";
      generation_finished_ = std::chrono::steady_clock::now();
    }
    if (auto_follow_) scroll_to_bottom_ = true;
  }
}

void StudioApp::DrawSidebar() {
  const ImVec2 logo_position = ImGui::GetCursorScreenPos();
  DrawGemstone(ImGui::GetWindowDrawList(),
               {logo_position.x + Ui(21.0f), logo_position.y + Ui(22.0f)}, Ui(19.0f));
  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + Ui(52.0f));
  ImGui::SetWindowFontScale(1.34f);
  ImGui::TextUnformatted("Gem 16");
  ImGui::SetWindowFontScale(1.0f);
  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + Ui(52.0f));
  ImGui::TextDisabled("Local AI Studio");
  ImGui::Dummy({0, Ui(32)});
  const float width = ImGui::GetWindowWidth() - 2.0f;
  ImGui::SetCursorPosX(1.0f);
  ImGui::BeginDisabled(!settings_.onboarding_complete);
  if (NavButton("Chat", Screen::kChat, screen_ == Screen::kChat, width))
    screen_ = Screen::kChat;
  ImGui::EndDisabled();
  ImGui::SetCursorPosX(1.0f);
  if (NavButton("Models", Screen::kModels, screen_ == Screen::kModels, width))
    screen_ = Screen::kModels;
  ImGui::SetCursorPosX(1.0f);
  ImGui::BeginDisabled(!settings_.onboarding_complete);
  if (NavButton("Server", Screen::kServer, screen_ == Screen::kServer, width))
    screen_ = Screen::kServer;
  ImGui::EndDisabled();
  ImGui::SetCursorPosX(1.0f);
  if (NavButton("Settings", Screen::kSettings, screen_ == Screen::kSettings,
                width))
    screen_ = Screen::kSettings;
  ImGui::SetCursorPosX(ImGui::GetStyle().WindowPadding.x);
  ImGui::SetCursorPosY(ImGui::GetWindowHeight() - Ui(104));
  ImGui::Separator();
  ImGui::Spacing();
  const ServerPhase phase = server_.Phase();
  StatusPill(PhaseLabel(phase), PhaseColor(phase));
  ImGui::TextDisabled("%s", settings_.onboarding_complete
                                ? ProfileLabel(settings_.server.profile)
                                : "No model selected");
}

void StudioApp::DrawHeader() {
  ImGui::SetWindowFontScale(1.42f);
  ImGui::TextUnformatted(screen_ == Screen::kChat ? "Gem 16" : ScreenTitle(screen_));
  ImGui::SetWindowFontScale(1.0f);
  ImGui::SameLine(0, Ui(14));
  StatusPill(PhaseLabel(server_.Phase()), PhaseColor(server_.Phase()));
  if (screen_ != Screen::kChat) {
    ImGui::SameLine(0, Ui(10));
    ImGui::TextDisabled("%s", settings_.onboarding_complete
                                  ? ProfileLabel(settings_.server.profile)
                                  : "Choose a model to continue");
  }
}

void StudioApp::DrawChat() {
  const float available_width = ImGui::GetContentRegionAvail().x;
  const float composer_content_width =
      std::max(Ui(300.0f), available_width - ImGui::GetStyle().WindowPadding.x * 2.0f);
  const float action_width = Ui(40.0f) + Ui(6.0f) + Ui(40.0f) + Ui(8.0f) + Ui(44.0f);
  const float input_width =
      std::max(Ui(180.0f), composer_content_width - action_width - Ui(6.0f));
  const int composer_lines = ComposerLineCount(composer_.data(), input_width - Ui(24.0f));
  const float input_height = std::clamp(
      Ui(18.0f) + static_cast<float>(composer_lines) * (ImGui::GetFontSize() + Ui(3.0f)),
      Ui(44.0f), Ui(126.0f));
  const float toolbar_height = Ui(32.0f);
  const float context_height = ImGui::GetFontSize();
  const float attachment_height = pending_attachments_.empty() ? 0.0f : Ui(28.0f);
  const float error_height = attachment_error_.empty() ? 0.0f : ImGui::GetFontSize();
  const int composer_gaps = 2 + (!pending_attachments_.empty() ? 1 : 0) +
                            (!attachment_error_.empty() ? 1 : 0);
  const float composer_height = ImGui::GetStyle().WindowPadding.y * 2.0f +
                                toolbar_height + context_height + attachment_height +
                                error_height + input_height +
                                static_cast<float>(composer_gaps) * Ui(6.0f);
  ImGui::BeginChild("##conversation", {0, -composer_height}, ImGuiChildFlags_None,
                    ImGuiWindowFlags_None);
  if (messages_.empty()) {
    const float y = std::max(Ui(30.0f), ImGui::GetContentRegionAvail().y * 0.23f);
    ImGui::Dummy({0, y});
    const ImVec2 center = {ImGui::GetCursorScreenPos().x + Ui(35.0f),
                           ImGui::GetCursorScreenPos().y + Ui(34.0f)};
    DrawGemstone(ImGui::GetWindowDrawList(), center, Ui(29.0f));
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + Ui(82.0f));
    ImGui::SetWindowFontScale(1.48f);
    ImGui::TextColored(kAccent, "What should we build today?");
    ImGui::SetWindowFontScale(1.0f);
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + Ui(82.0f));
    ImGui::TextWrapped(
        "Chat locally with %s. Responses stream directly from the resident GPU session.",
        ProfileLabel(settings_.server.profile));
  }
  for (std::size_t index = 0; index < messages_.size(); ++index) DrawMessage(messages_[index], index);
  const bool at_bottom = ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - Ui(18.0f);
  if (ImGui::IsWindowHovered() && ImGui::GetIO().MouseWheel != 0.0f && !at_bottom)
    auto_follow_ = false;
  if (at_bottom) auto_follow_ = true;
  if (!auto_follow_ && !messages_.empty()) {
    ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(),
                                  ImGui::GetWindowWidth() - Ui(150.0f)));
    if (ImGui::Button("Jump to latest", {Ui(130.0f), Ui(34.0f)})) {
      auto_follow_ = true;
      scroll_to_bottom_ = true;
    }
  }
  if (scroll_to_bottom_) {
    ImGui::SetScrollHereY(1.0f);
    scroll_to_bottom_ = false;
  }
  ImGui::EndChild();

  ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, Ui(18.0f));
  ImGui::BeginChild("##composer", {0, 0}, ImGuiChildFlags_Borders,
                    ImGuiWindowFlags_NoScrollbar |
                        ImGuiWindowFlags_NoScrollWithMouse);
  ImGui::PopStyleVar();
  const HealthSnapshot health = server_.Health();
  const bool busy = api_.Busy();
  const bool has_draft = composer_[0] != '\0' || !pending_attachments_.empty();
  const bool can_send = !busy && has_draft && health.available;

  const ImVec2 toolbar_origin = ImGui::GetCursorScreenPos();
  float toolbar_x = toolbar_origin.x;
  ImGui::SetCursorScreenPos({toolbar_x, toolbar_origin.y});
  if (ImGui::Button("+ Attach", {Ui(88.0f), toolbar_height}))
    AddAttachments(OpenAttachmentDialog());
  toolbar_x += Ui(96.0f);
  const bool media_profile = settings_.server.profile == ModelProfile::kGemma4Unified12B;
  ImGui::SetCursorScreenPos({toolbar_x, toolbar_origin.y});
  ImGui::BeginDisabled(!media_profile || (busy && !recorder_.Recording()));
  if (ImGui::Button(recorder_.Recording() ? "Stop mic" : "Record mic",
                    {Ui(96.0f), Ui(32.0f)})) {
    if (recorder_.Recording()) {
      MediaAttachment recording;
      if (recorder_.Stop(recording, attachment_error_))
        pending_attachments_.push_back(std::move(recording));
    } else {
      (void)recorder_.Start(attachment_error_);
    }
  }
  ImGui::EndDisabled();
  toolbar_x += Ui(104.0f);
  ImGui::SetCursorScreenPos({toolbar_x, toolbar_origin.y});
  ImGui::SetNextItemWidth(Ui(132.0f));
  const char* efforts[] = {"none", "low", "medium", "high"};
  int effort = 2;
  for (int index = 0; index < 4; ++index)
    if (settings_.generation.reasoning_effort == efforts[index]) effort = index;
  if (ImGui::Combo("##chat-effort", &effort, efforts, 4))
    settings_.generation.reasoning_effort = efforts[effort];
  toolbar_x += Ui(140.0f);
  char maximum_label[48]{};
  std::snprintf(maximum_label, sizeof(maximum_label), "%lld max",
                static_cast<long long>(settings_.generation.max_output_tokens));
  const float toolbar_text_y = toolbar_origin.y +
                               (toolbar_height - ImGui::GetFontSize()) * 0.5f;
  ImDrawList* composer_draw = ImGui::GetWindowDrawList();
  const ImU32 disabled_text = ImGui::GetColorU32(ImGuiCol_TextDisabled);
  composer_draw->AddText({toolbar_x, toolbar_text_y}, disabled_text, maximum_label);
  toolbar_x += ImGui::CalcTextSize(maximum_label).x + Ui(18.0f);
  char status_label[160]{};
  if (generation_started_.time_since_epoch().count() != 0) {
    const auto end = busy ? std::chrono::steady_clock::now() : generation_finished_;
    const double seconds = std::chrono::duration<double>(end - generation_started_).count();
    const std::int64_t observed_tokens = busy ? streamed_chunks_ : completion_tokens_;
    const double decode_seconds = first_token_at_.time_since_epoch().count() == 0
                                      ? 0.0
                                      : std::chrono::duration<double>(end - first_token_at_).count();
    const double rate = decode_seconds > 0.0
                            ? static_cast<double>(observed_tokens) / decode_seconds
                            : 0.0;
    if (busy && streamed_chunks_ == 0)
      std::snprintf(status_label, sizeof(status_label), "Prefilling...");
    else
      std::snprintf(status_label, sizeof(status_label),
                    "%lld in · %lld out · %.1f tok/s · %.1fs",
                    static_cast<long long>(prompt_tokens_),
                    static_cast<long long>(observed_tokens), rate, seconds);
  } else {
    std::snprintf(status_label, sizeof(status_label),
                  "Drop files anywhere · media require 12B Unified");
  }
  const float toolbar_right = ImGui::GetWindowPos().x +
                              ImGui::GetWindowContentRegionMax().x;
  if (toolbar_x + ImGui::CalcTextSize(status_label).x <= toolbar_right)
    composer_draw->AddText({toolbar_x, toolbar_text_y},
                           busy ? ImGui::GetColorU32(kAccent) : disabled_text,
                           status_label);
  ImGui::SetCursorScreenPos({toolbar_origin.x,
                             toolbar_origin.y + toolbar_height + Ui(6.0f)});

  const std::int64_t context_limit = health.max_context_tokens > 0
                                         ? health.max_context_tokens
                                         : settings_.server.max_context_tokens;
  const float context_fraction = context_limit > 0
      ? std::clamp(static_cast<float>(prompt_tokens_) /
                       static_cast<float>(context_limit), 0.0f, 1.0f)
      : 0.0f;
  const ImVec2 context_origin = ImGui::GetCursorScreenPos();
  char context_label[64]{};
  std::snprintf(context_label, sizeof(context_label), "%lld / %lld",
                static_cast<long long>(prompt_tokens_),
                static_cast<long long>(context_limit));
  const float context_right = context_origin.x + ImGui::GetContentRegionAvail().x;
  const float count_width = ImGui::CalcTextSize(context_label).x;
  const float context_text_y = context_origin.y +
                               (context_height - ImGui::GetFontSize()) * 0.5f;
  composer_draw->AddText({context_origin.x, context_text_y}, disabled_text, "Context");
  composer_draw->AddText({context_right - count_width, context_text_y}, disabled_text,
                         context_label);
  const float bar_left = context_origin.x + ImGui::CalcTextSize("Context").x + Ui(14.0f);
  const float bar_right = context_right - count_width - Ui(14.0f);
  const float bar_y = context_origin.y + context_height * 0.5f;
  if (bar_right > bar_left) {
    composer_draw->AddRectFilled({bar_left, bar_y - Ui(3.0f)},
                                 {bar_right, bar_y + Ui(3.0f)},
                                 ImGui::GetColorU32(ImGuiCol_FrameBg), Ui(3.0f));
    composer_draw->AddRectFilled({bar_left, bar_y - Ui(3.0f)},
                                 {bar_left + (bar_right - bar_left) * context_fraction,
                                  bar_y + Ui(3.0f)},
                                 ImGui::GetColorU32(kAccent), Ui(3.0f));
  }
  ImGui::SetCursorScreenPos({context_origin.x,
                             context_origin.y + context_height + Ui(6.0f)});

  if (!pending_attachments_.empty()) {
    const ImVec2 attachment_origin = ImGui::GetCursorScreenPos();
    float attachment_x = attachment_origin.x;
    const float attachment_right = attachment_origin.x + ImGui::GetContentRegionAvail().x;
    for (std::size_t index = 0; index < pending_attachments_.size();) {
      const auto& attachment = pending_attachments_[index];
      const char* kind = attachment.kind == MediaKind::kImage ? "Image" :
                         attachment.kind == MediaKind::kAudio ? "Audio" : "Text";
      const std::string label = std::string(kind) + " · " + attachment.file_name + "  ×";
      const float chip_width = std::clamp(ImGui::CalcTextSize(label.c_str()).x + Ui(24.0f),
                                          Ui(120.0f), Ui(245.0f));
      if (attachment_x + chip_width > attachment_right) {
        const std::string remaining = "+" +
            std::to_string(pending_attachments_.size() - index) + " more";
        composer_draw->AddText(
            {attachment_x, attachment_origin.y +
                               (attachment_height - ImGui::GetFontSize()) * 0.5f},
            disabled_text, remaining.c_str());
        break;
      }
      ImGui::PushID(static_cast<int>(index));
      ImGui::SetCursorScreenPos({attachment_x, attachment_origin.y});
      ImGui::PushStyleColor(ImGuiCol_Button, kAccentDim);
      ImGui::PushStyleColor(ImGuiCol_Text, kAccent);
      const bool remove = ImGui::Button(label.c_str(), {chip_width, attachment_height});
      ImGui::PopStyleColor(2);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Remove %s", attachment.file_name.c_str());
      if (remove) {
        pending_attachments_.erase(pending_attachments_.begin() +
                                   static_cast<std::ptrdiff_t>(index));
        ImGui::PopID();
        continue;
      }
      attachment_x += chip_width + Ui(8.0f);
      ++index;
      ImGui::PopID();
    }
    ImGui::SetCursorScreenPos({attachment_origin.x,
                               attachment_origin.y + attachment_height + Ui(6.0f)});
  }
  if (!attachment_error_.empty()) {
    ImGui::TextColored({1.0f, 0.47f, 0.42f, 1.0f}, "%s", attachment_error_.c_str());
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + Ui(6.0f));
  }

  const ImVec2 input_origin = ImGui::GetCursorScreenPos();
  ImGui::BeginDisabled(busy);
  const bool enter = ImGui::InputTextMultiline(
      "##message", composer_.data(), composer_.size(), {input_width, input_height},
      ImGuiInputTextFlags_EnterReturnsTrue |
          ImGuiInputTextFlags_CtrlEnterForNewLine);
  ImGui::EndDisabled();
  if (composer_[0] == '\0' && !ImGui::IsItemActive()) {
    ImGui::GetWindowDrawList()->AddText(
      {input_origin.x + Ui(13.0f), input_origin.y + Ui(12.0f)},
        IM_COL32(122, 137, 132, 190),
        "Message Gem 16...  Enter to send · Shift+Enter for a new line");
  }
  float action_x = input_origin.x + input_width + Ui(6.0f);
  const float small_action_y = input_origin.y + (input_height - Ui(40.0f)) * 0.5f;
  const float send_action_y = input_origin.y + (input_height - Ui(44.0f)) * 0.5f;
  ImGui::SetCursorScreenPos({action_x, small_action_y});
  ImGui::BeginDisabled(busy || messages_.empty());
  if (ComposerButton("##undo-turn", ComposerIcon::kUndo, Ui(40.0f),
                     "Remove the last exchange", busy || messages_.empty()))
    RemoveLastExchange();
  ImGui::EndDisabled();
  action_x += Ui(46.0f);
  ImGui::SetCursorScreenPos({action_x, small_action_y});
  ImGui::BeginDisabled(busy || messages_.empty());
  if (ComposerButton("##clear-chat", ComposerIcon::kDelete, Ui(40.0f),
                     "Delete the complete chat", busy || messages_.empty()))
    ClearChat();
  ImGui::EndDisabled();
  action_x += Ui(48.0f);
  ImGui::SetCursorScreenPos({action_x, send_action_y});
  ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, Ui(30.0f));
  if (busy) {
    if (ComposerButton("##stop", ComposerIcon::kStop, Ui(44.0f),
                       "Stop generation"))
      api_.Cancel();
  } else {
    ImGui::BeginDisabled(!can_send);
    const bool send_clicked = ComposerButton(
        "##send", ComposerIcon::kSend, Ui(44.0f),
        health.available ? "Send message" : "Server is offline", !can_send);
    ImGui::EndDisabled();
    if (send_clicked || (enter && can_send)) SendMessage();
  }
  ImGui::PopStyleVar();
  ImGui::SetCursorScreenPos({input_origin.x, input_origin.y + input_height});
  ImGui::Dummy({0.0f, 0.0f});
  ImGui::EndChild();
}

void StudioApp::DrawMessage(const ChatMessage& message, std::size_t index) {
  const bool user = message.role == "user";
  const float available = ImGui::GetContentRegionAvail().x;
  const float width = available * (user ? 0.68f : 0.80f);
  const float original_x = ImGui::GetCursorPosX();
  if (user) {
    ImGui::SetCursorPosX(original_x + available - width - Ui(10.0f));
  } else {
    const ImVec2 avatar = ImGui::GetCursorScreenPos();
    DrawGemstone(ImGui::GetWindowDrawList(), {avatar.x + Ui(21.0f), avatar.y + Ui(28.0f)}, Ui(18.0f));
    ImGui::SetCursorPosX(original_x + Ui(52.0f));
  }
  ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, Ui(17.0f));
  ImGui::PushStyleColor(ImGuiCol_ChildBg,
                        user ? ImVec4(0.035f, 0.25f, 0.17f, 0.93f)
                             : ImVec4(0.060f, 0.078f, 0.074f, 0.92f));
  ImGui::PushStyleColor(ImGuiCol_Border,
                        user ? ImVec4(0.10f, 0.55f, 0.38f, 0.75f)
                             : ImVec4(0.19f, 0.26f, 0.24f, 0.86f));
  if (message.error) ImGui::PushStyleColor(ImGuiCol_Border, {0.8f, 0.18f, 0.18f, 1});
  const std::string id = "##message-" + std::to_string(index);
  ImGui::BeginChild(id.c_str(), {width, 0}, ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Borders);
  ImGui::TextColored(user ? kAccent : ImVec4(0.62f, 0.68f, 0.66f, 1), "%s", user ? "You" : ProfileLabel(settings_.server.profile));
  if (!message.content.empty()) {
    const bool recently_copied =
        copied_message_index_ == index &&
        ImGui::GetTime() - copied_message_at_ < 1.6;
    ImGui::SameLine();
    ImGui::SetCursorPosX(ImGui::GetWindowWidth() - Ui(64.0f));
    ImGui::PushStyleColor(ImGuiCol_Button, {0, 0, 0, 0});
    if (ImGui::SmallButton(
            (std::string(recently_copied ? "Copied##" : "Copy##") +
             std::to_string(index))
                .c_str())) {
      ImGui::SetClipboardText(message.content.c_str());
      copied_message_index_ = index;
      copied_message_at_ = ImGui::GetTime();
    }
    ImGui::PopStyleColor();
  }
  if (!message.attachments.empty()) {
    ImGui::Spacing();
    for (const auto& attachment : message.attachments) {
      const char* kind = attachment.kind == MediaKind::kImage ? "Image" :
                         attachment.kind == MediaKind::kAudio ? "Audio" : "Document";
      ImGui::PushStyleColor(ImGuiCol_Button, {0.07f, 0.18f, 0.14f, 0.96f});
      ImGui::SmallButton((std::string(kind) + " · " + attachment.file_name + "##attachment").c_str());
      ImGui::PopStyleColor();
    }
  }
  if (show_reasoning_ && (!message.reasoning.empty() || message.streaming)) {
    ImGui::Spacing();
    const bool reasoning_expanded = expanded_reasoning_.contains(index);
    ImGui::PushStyleColor(ImGuiCol_Button, {0.04f, 0.11f, 0.09f, 0.88f});
    const std::string reasoning_label =
        std::string("Reasoning##") + std::to_string(index);
    if (ImGui::Button(reasoning_label.c_str(), {-1.0f, Ui(32.0f)})) {
      if (reasoning_expanded)
        expanded_reasoning_.erase(index);
      else
        expanded_reasoning_.insert(index);
    }
    const ImVec2 reasoning_min = ImGui::GetItemRectMin();
    const float reasoning_center_y =
        (reasoning_min.y + ImGui::GetItemRectMax().y) * 0.5f;
    if (reasoning_expanded) {
      ImGui::GetWindowDrawList()->AddTriangleFilled(
          {reasoning_min.x + Ui(15.0f), reasoning_center_y - Ui(3.0f)},
          {reasoning_min.x + Ui(23.0f), reasoning_center_y - Ui(3.0f)},
          {reasoning_min.x + Ui(19.0f), reasoning_center_y + Ui(4.0f)},
          ImGui::GetColorU32(ImGuiCol_TextDisabled));
    } else {
      ImGui::GetWindowDrawList()->AddTriangleFilled(
          {reasoning_min.x + Ui(16.0f), reasoning_center_y - Ui(5.0f)},
          {reasoning_min.x + Ui(16.0f), reasoning_center_y + Ui(5.0f)},
          {reasoning_min.x + Ui(23.0f), reasoning_center_y},
          ImGui::GetColorU32(ImGuiCol_TextDisabled));
    }
    ImGui::PopStyleColor();
    if (reasoning_expanded && !message.reasoning.empty()) {
      ImGui::PushStyleColor(ImGuiCol_Text, {0.55f, 0.62f, 0.60f, 1});
      selectable_text::Wrapped(
          (std::string("reasoning##") + std::to_string(index)).c_str(),
          message.reasoning,
          {.width = ImGui::GetContentRegionAvail().x,
           .text_color = ImGui::ColorConvertFloat4ToU32(
               {0.55f, 0.62f, 0.60f, 1}),
           .selection_color = IM_COL32(42, 123, 94, 190)});
      ImGui::PopStyleColor();
    }
  }
  if (message.content.empty() && message.streaming) {
    const int dots = 1 + static_cast<int>(std::fmod(ImGui::GetTime() * 2.5, 3.0));
    ImGui::TextColored(kAccent, "%.*s", dots, "...");
  } else {
    ImGui::Spacing();
    markdown::Render((std::string("content##") + std::to_string(index)).c_str(),
                     message.content, ImGui::GetContentRegionAvail().x);
  }
  ImGui::EndChild();
  const ImVec2 bubble_min = ImGui::GetItemRectMin();
  const ImVec2 bubble_max = ImGui::GetItemRectMax();
  if (user) {
    const ImVec2 tail[3] = {{bubble_max.x - Ui(18.0f), bubble_max.y - Ui(1.0f)},
                            {bubble_max.x + Ui(7.0f), bubble_max.y - Ui(1.0f)},
                            {bubble_max.x - Ui(5.0f), bubble_max.y - Ui(15.0f)}};
    ImGui::GetWindowDrawList()->AddConvexPolyFilled(
        tail, 3, ImGui::ColorConvertFloat4ToU32({0.035f, 0.25f, 0.17f, 0.93f}));
  } else {
    const ImVec2 tail[3] = {{bubble_min.x + Ui(16.0f), bubble_min.y + Ui(17.0f)},
                            {bubble_min.x - Ui(8.0f), bubble_min.y + Ui(28.0f)},
                            {bubble_min.x + Ui(2.0f), bubble_min.y + Ui(8.0f)}};
    ImGui::GetWindowDrawList()->AddConvexPolyFilled(
        tail, 3, ImGui::ColorConvertFloat4ToU32({0.060f, 0.078f, 0.074f, 0.92f}));
  }
  if (message.error) ImGui::PopStyleColor();
  ImGui::PopStyleColor(2);
  ImGui::PopStyleVar();
  ImGui::SetCursorPosX(original_x);
  ImGui::Dummy({0, Ui(12)});
}

void StudioApp::DrawModels() {
  const ModelInstallState install = models_.State();
  PanelHeading(settings_.onboarding_complete ? "Local model profiles"
                                             : "Welcome to Gem 16",
               settings_.onboarding_complete
                   ? "Install either profile or keep both side by side in the shared Hugging Face cache."
                   : "Choose and install a model profile. Nothing is selected by default on a new system.");
  ImGui::TextDisabled("Hub cache: %s", HuggingFaceHubRoot().string().c_str());
  ImGui::SameLine();
  if (ImGui::SmallButton("Open cache")) OpenInFileManager(HuggingFaceHubRoot());
  ImGui::SameLine();
  ImGui::BeginDisabled(install.downloading);
  if (ImGui::SmallButton("Verify again")) models_.Refresh();
  ImGui::EndDisabled();
  ImGui::Dummy({0, Ui(8)});
  const auto draw_profile = [this, &install](const ModelProfileCatalog& catalog) {
    const ModelProfile profile = catalog.profile;
    const ProfileInstallState& profile_state = install.For(profile);
    const bool selected = settings_.onboarding_complete &&
                          settings_.server.profile == profile;
    const bool downloading = install.downloading &&
                             install.downloading_profile == profile;
    ImGui::PushStyleColor(ImGuiCol_ChildBg, selected ? ImVec4(0.07f, 0.22f, 0.16f, 0.96f)
                                                     : ImGui::GetStyleColorVec4(ImGuiCol_ChildBg));
    ImGui::PushStyleColor(ImGuiCol_Border, selected ? kAccent : ImGui::GetStyleColorVec4(ImGuiCol_Border));
    const std::string id = std::string("##profile-") + ProfileWireName(profile);
    ImGui::BeginChild(id.c_str(), {0, Ui(190)}, ImGuiChildFlags_Borders,
                      ImGuiWindowFlags_NoScrollbar);
    const ImVec2 gem_origin = ImGui::GetCursorScreenPos();
    DrawGemstone(ImGui::GetWindowDrawList(),
                 {gem_origin.x + Ui(27.0f), gem_origin.y + Ui(29.0f)}, Ui(22.0f));
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + Ui(64.0f));
    ImGui::SetWindowFontScale(1.18f);
    ImGui::TextUnformatted(ProfileLabel(profile));
    ImGui::SetWindowFontScale(1.0f);
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + Ui(64.0f));
    ImGui::TextWrapped("%s", catalog.description);
    ImGui::Dummy({0, Ui(4)});
    ImGui::TextColored(kAccent, "%s", catalog.capabilities);
    ImGui::Text("Target: %s", profile_state.target_ready ? "Verified" : "Missing");
    ImGui::SameLine(Ui(180.0f));
    ImGui::Text("Assistant: %s", profile_state.assistant_ready ? "Verified" : "Missing");
    if (downloading) {
      const float progress = profile_state.total_bytes == 0 ? 0.0f :
          static_cast<float>(static_cast<double>(profile_state.completed_bytes) /
                             static_cast<double>(profile_state.total_bytes));
      ImGui::ProgressBar(std::clamp(progress, 0.0f, 1.0f), {-Ui(110.0f), 0.0f},
                         FormatBytes(profile_state.completed_bytes).c_str());
      ImGui::SameLine();
      if (ImGui::Button((std::string("Pause##") + ProfileWireName(profile)).c_str())) {
        models_.Cancel();
      }
      if (!install.current_file.empty()) ImGui::TextDisabled("%s", install.current_file.c_str());
    } else if (!profile_state.Ready()) {
      const bool blocked = install.downloading || !profile_state.storage_available ||
                           !profile_state.sufficient_space;
      ImGui::BeginDisabled(blocked);
      const std::string button = "Install " + FormatBytes(profile_state.required_download_bytes) +
                                 "##" + ProfileWireName(profile);
      if (ImGui::Button(button.c_str())) models_.DownloadProfile(profile);
      ImGui::EndDisabled();
      ImGui::SameLine();
      if (!profile_state.storage_available) {
        ImGui::TextDisabled("Free space unavailable");
      } else if (!profile_state.sufficient_space) {
        ImGui::TextColored({1.0f, 0.48f, 0.36f, 1.0f}, "Need %s + 256 MiB reserve · %s free",
                           FormatBytes(profile_state.required_download_bytes).c_str(),
                           FormatBytes(profile_state.available_disk_bytes).c_str());
      } else {
        ImGui::TextDisabled("%s free · resumable · SHA-256 verified",
                            FormatBytes(profile_state.available_disk_bytes).c_str());
      }
    } else if (selected) {
      ImGui::TextColored(kAccent, "Installed and selected");
    } else {
      const std::string button = "Use this profile##" +
                                 std::string(ProfileWireName(profile));
      if (ImGui::Button(button.c_str())) SelectProfile(profile);
      ImGui::SameLine();
      ImGui::TextDisabled("Installed in the shared Hub cache");
    }
    ImGui::EndChild();
    ImGui::PopStyleColor(2);
    ImGui::Dummy({0, Ui(10)});
  };
  for (const auto& catalog : ModelCatalog()) draw_profile(catalog);
  if (!install.error.empty()) {
    ImGui::TextColored({1.0f, 0.45f, 0.45f, 1.0f}, "%s", install.error.c_str());
  }
  ImGui::TextDisabled("Profiles may coexist. Changing the active profile updates launch paths; restart a running server to apply it.");
}

void StudioApp::DrawServer() {
  const HealthSnapshot health = server_.Health();
  if (!server_.Error().empty()) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, {0.27f, 0.07f, 0.07f, 0.92f});
    const float wrap_width = std::max(Ui(240.0f), ImGui::GetContentRegionAvail().x - Ui(36.0f));
    const float error_height = ImGui::CalcTextSize(server_.Error().c_str(), nullptr,
                                                   false, wrap_width).y + Ui(52.0f);
    ImGui::BeginChild("##server-error", {0, error_height}, ImGuiChildFlags_Borders,
                      ImGuiWindowFlags_NoScrollbar);
    ImGui::TextColored({1, 0.55f, 0.55f, 1}, "Server error");
    ImGui::TextWrapped("%s", server_.Error().c_str());
    ImGui::EndChild();
    ImGui::PopStyleColor();
    ImGui::Dummy({0, Ui(10)});
  }
  const float available = ImGui::GetContentRegionAvail().x;
  const bool columns = available >= Ui(920.0f);
  const float config_width = columns ? available * 0.60f : available;
  ImGuiChildFlags config_flags = ImGuiChildFlags_Borders;
  if (!columns) config_flags |= ImGuiChildFlags_AutoResizeY;
  ImGui::BeginChild("##server-config", {config_width, 0.0f}, config_flags,
                    columns ? ImGuiWindowFlags_None : ImGuiWindowFlags_NoScrollbar);
  PanelHeading("Server configuration",
               "Configure the local executable, resident model and generation path.");
  ImGui::TextDisabled("ACTIVE PROFILE");
  ImGui::SameLine();
  ImGui::TextColored(kAccent, "%s", ProfileLabel(settings_.server.profile));
  ImGui::Dummy({0, Ui(5)});
  PathField("Server executable", "##server-executable", "Browse##server", executable_, false);
  PathField("Compiled target model", "##target-model", "Browse##target", model_directory_, true);
  PathField("Compiled MTP assistant", "##mtp-assistant", "Browse##assistant", assistant_directory_, true);
  TextField("Served model name", "##served-name", model_name_);

  if (ImGui::BeginTable("##network-fields", 2,
                        ImGuiTableFlags_SizingStretchSame)) {
    ImGui::TableNextColumn();
    TextField("Server host", "##server-host", host_);
    ImGui::TableNextColumn();
    FieldLabel("Port");
    ImGui::InputInt("##server-port", &settings_.server.port);
    ImGui::EndTable();
  }
  int context = static_cast<int>(settings_.server.max_context_tokens);
  FieldLabel("Context tokens");
  if (ImGui::InputInt("##context-tokens", &context))
    settings_.server.max_context_tokens = std::max(context, 1);
  const int mtp_values[] = {0, 1, 2, 4};
  const char* mtp_labels[] = {"Off", "D1", "D2", "D4"};
  int mtp_index = 2;
  for (int index = 0; index < 4; ++index) {
    if (settings_.server.mtp_draft_tokens == mtp_values[index]) mtp_index = index;
  }
  FieldLabel("MTP draft profile");
  if (ImGui::Combo("##mtp-profile", &mtp_index, mtp_labels, 4)) {
    settings_.server.mtp_draft_tokens = mtp_values[mtp_index];
  }
  ImGui::Checkbox("Greedy sampling", &settings_.server.greedy);
  ImGui::SameLine(0, Ui(18));
  CapabilityChip(settings_.server.mtp_draft_tokens == 0 ? "MTP disabled" : "GPU MTP enabled",
                 settings_.server.mtp_draft_tokens != 0);
  SyncSettingsFromBuffers();
  server_.Configure(settings_.server);
  ImGui::Dummy({0, Ui(6)});
  const bool executable_ready = std::filesystem::is_regular_file(settings_.server.executable);
  const bool target_ready = std::filesystem::is_directory(settings_.server.model_directory);
  const bool assistant_ready = settings_.server.mtp_draft_tokens == 0 ||
      std::filesystem::is_directory(settings_.server.assistant_directory);
  const bool preflight_ready = executable_ready && target_ready && assistant_ready &&
      settings_.server.port > 0 && settings_.server.port <= 65535;
  ImGui::TextColored(preflight_ready ? kAccent : ImVec4(1.0f, 0.48f, 0.36f, 1.0f),
                     "%s  Executable · %s  Target · %s  Assistant",
                     executable_ready ? "Ready" : "Missing",
                     target_ready ? "Ready" : "Missing",
                     assistant_ready ? "Ready" : "Missing");
  const ServerPhase phase = server_.Phase();
  if (phase == ServerPhase::kRunning || phase == ServerPhase::kStarting || phase == ServerPhase::kStopping) {
    ImGui::BeginDisabled(phase == ServerPhase::kStopping);
    if (ImGui::Button("Stop server", {Ui(140), Ui(42)})) server_.Stop();
    ImGui::EndDisabled();
  } else {
    ImGui::BeginDisabled(!preflight_ready);
    if (ImGui::Button(phase == ServerPhase::kExternal ? "External server" : "Start server", {Ui(140), Ui(42)}) && phase != ServerPhase::kExternal) {
      (void)SaveSettings(settings_);
      server_.Start(settings_.server);
    }
    ImGui::EndDisabled();
  }
  ImGui::EndChild();
  if (columns)
    ImGui::SameLine(0, Ui(12));
  else
    ImGui::Dummy({0, Ui(12)});
  ImGui::BeginChild("##server-log", {0, columns ? 0.0f : Ui(390.0f)}, ImGuiChildFlags_Borders);
  PanelHeading("Runtime status", "Resident session health and server output.");
  StatusPill(PhaseLabel(server_.Phase()), PhaseColor(server_.Phase()));
  if (health.available) {
    ImGui::Dummy({0, Ui(6)});
    ImGui::TextColored(kAccent, "%s", health.status.c_str());
    ImGui::Text("Variant: %s", health.model_variant.c_str());
    ImGui::Text("Sessions: %d / %d", health.resident_sessions, health.session_limit);
    ImGui::Text("Context: %lld", static_cast<long long>(health.max_context_tokens));
    ImGui::Text("MTP: %s · D%d", health.supports_mtp ? "available" : "off", health.mtp_draft_tokens);
    ImGui::Separator();
  }
  const auto logs = server_.Logs();
  std::string log_text;
  for (const std::string& line : logs) {
    if (!log_text.empty()) log_text.push_back('\n');
    log_text += line;
  }
  if (log_text.empty()) {
    ImGui::TextDisabled("Server output will appear here.");
  } else {
    selectable_text::Wrapped(
        "##server-log-text", log_text,
        {.width = ImGui::GetContentRegionAvail().x,
         .text_color = ImGui::ColorConvertFloat4ToU32(
             {0.72f, 0.77f, 0.75f, 1.0f}),
         .selection_color = IM_COL32(38, 144, 102, 205)});
  }
  ImGui::Dummy({0, Ui(6)});
  if (ImGui::Button("Clear log")) server_.ClearLogs();
  ImGui::SameLine();
  ImGui::BeginDisabled(log_text.empty());
  if (ImGui::Button("Copy log")) ImGui::SetClipboardText(log_text.c_str());
  ImGui::EndDisabled();
  ImGui::EndChild();
}

void StudioApp::DrawSettings() {
  PanelHeading("Studio settings",
               "Tune the interface and local generation defaults.");
  const float available = ImGui::GetContentRegionAvail().x;
  const bool columns = available >= Ui(760.0f);
  const float left_width = columns ? available * 0.34f : available;
  ImGui::BeginChild("##appearance-card", {left_width, columns ? 0.0f : Ui(285.0f)},
                    ImGuiChildFlags_Borders, ImGuiWindowFlags_NoScrollbar);
  PanelHeading("Appearance", "The interface remains local and GPU-rendered.");
  if (ImGui::Checkbox("Dark glass theme", &settings_.dark_theme)) ApplyTheme();
  ImGui::Dummy({0, Ui(8)});
  FieldLabel("Interface scale");
  const char* scale_labels[] = {"Auto", "100%", "125%", "150%"};
  const float scale_values[] = {0.0f, 1.0f, 1.25f, 1.5f};
  int scale_index = 0;
  for (int index = 0; index < 4; ++index)
    if (settings_.ui_scale == scale_values[index]) scale_index = index;
  if (ImGui::Combo("##ui-scale", &scale_index, scale_labels, 4))
    settings_.ui_scale = scale_values[scale_index];
  ImGui::TextDisabled("Current %.0f%% · changes apply after restart", ui_scale_ * 100.0f);
  ImGui::Dummy({0, Ui(8)});
  ImGui::TextWrapped(
      "The animated science-fiction wave is rendered by the native OpenGL and Direct3D 11 backends.");
  ImGui::Dummy({0, Ui(12)});
  CapabilityChip("Animated shader");
  ImGui::SameLine();
  CapabilityChip("OS cursor");
  ImGui::Dummy({0, Ui(8)});
  ImGui::TextDisabled("Native Studio %s", GEM16_VERSION_STRING);
  ImGui::EndChild();
  if (columns)
    ImGui::SameLine(0, Ui(12));
  else
    ImGui::Dummy({0, Ui(12)});
  ImGui::BeginChild("##generation-card", {0, columns ? 0.0f : Ui(620.0f)},
                    ImGuiChildFlags_Borders, ImGuiWindowFlags_NoScrollbar);
  PanelHeading("Generation", "Defaults applied to new local chat requests.");
  const char* efforts[] = {"none", "low", "medium", "high"};
  int current = 2;
  for (int index = 0; index < 4; ++index) if (settings_.generation.reasoning_effort == efforts[index]) current = index;
  FieldLabel("Thinking effort");
  if (ImGui::Combo("##thinking-effort", &current, efforts, 4)) settings_.generation.reasoning_effort = efforts[current];
  int output_tokens = static_cast<int>(settings_.generation.max_output_tokens);
  FieldLabel("Maximum output tokens");
  if (ImGui::InputInt("##maximum-output-tokens", &output_tokens)) settings_.generation.max_output_tokens = std::max(output_tokens, 1);
  ImGui::Checkbox("Show reasoning", &show_reasoning_);
  FieldLabel("System prompt");
  ImGui::InputTextMultiline("##system-prompt", system_prompt_.data(), system_prompt_.size(), {0, Ui(70)});
  SyncSettingsFromBuffers();
  if (ImGui::Button("Save settings", {Ui(140), Ui(42)})) (void)SaveSettings(settings_);
  ImGui::EndChild();
}

void StudioApp::SendMessage() {
  std::string text = composer_.data();
  if ((text.empty() && pending_attachments_.empty()) || api_.Busy()) return;
  ChatMessage user{"user", std::move(text), {}, false, false, {}};
  user.attachments = std::move(pending_attachments_);
  messages_.push_back(std::move(user));
  composer_[0] = '\0';
  pending_attachments_.clear();
  attachment_error_.clear();
  const auto request_messages = messages_;
  messages_.push_back({"assistant", {}, {}, true, false, {}});
  generation_started_ = std::chrono::steady_clock::now();
  first_token_at_ = {};
  generation_finished_ = {};
  prompt_tokens_ = 0;
  completion_tokens_ = 0;
  streamed_chunks_ = 0;
  api_.StreamChat(settings_.server, settings_.generation, request_messages, session_id_);
  auto_follow_ = true;
  scroll_to_bottom_ = true;
}

void StudioApp::AddAttachments(
    const std::vector<std::filesystem::path>& paths) {
  attachment_error_.clear();
  for (const auto& path : paths) {
    MediaAttachment attachment;
    std::string error;
    if (!LoadMediaAttachment(path, attachment, error)) {
      attachment_error_ = std::move(error);
      continue;
    }
    if (settings_.server.profile == ModelProfile::kGemma4Moe26BA4B &&
        attachment.kind != MediaKind::kDocument) {
      attachment_error_ = "Gemma 4 26B A4B is text-only. Select 12B Unified for images and audio.";
      continue;
    }
    pending_attachments_.push_back(std::move(attachment));
  }
}

void StudioApp::ClearChat() {
  if (api_.Busy()) return;
  if (recorder_.Recording()) {
    MediaAttachment discarded;
    std::string ignored;
    (void)recorder_.Stop(discarded, ignored);
  }
  messages_.clear();
  session_id_.clear();
  composer_[0] = '\0';
  pending_attachments_.clear();
  attachment_error_.clear();
  expanded_reasoning_.clear();
}

void StudioApp::RemoveLastExchange() {
  if (api_.Busy() || messages_.empty()) return;
  if (!gem16::studio::RemoveLastExchange(messages_)) return;
  session_id_.clear();
  expanded_reasoning_.clear();
}

void StudioApp::SelectProfile(ModelProfile profile) {
  if (!models_.State().For(profile).Ready()) return;
  const bool restart_managed_server = server_.Phase() == ServerPhase::kRunning;
  if (restart_managed_server) server_.Stop();
  SyncSettingsFromBuffers();
  ApplyProfileDefaults(settings_.server, profile);
  settings_.onboarding_complete = true;
  SyncBuffersFromSettings();
  server_.Configure(settings_.server);
  if (api_.Busy()) api_.Cancel();
  ClearChat();
  (void)SaveSettings(settings_);
  if (restart_managed_server) server_.Start(settings_.server);
}

void StudioApp::SyncBuffersFromSettings() {
  CopyTo(executable_, settings_.server.executable);
  CopyTo(model_directory_, settings_.server.model_directory);
  CopyTo(assistant_directory_, settings_.server.assistant_directory);
  CopyTo(model_name_, settings_.server.model_name);
  CopyTo(host_, settings_.server.host);
  CopyTo(system_prompt_, settings_.generation.system_prompt);
}

void StudioApp::SyncSettingsFromBuffers() {
  settings_.server.executable = executable_.data();
  settings_.server.model_directory = model_directory_.data();
  settings_.server.assistant_directory = assistant_directory_.data();
  settings_.server.model_name = model_name_.data();
  settings_.server.host = host_.data();
  settings_.generation.system_prompt = system_prompt_.data();
}

}  // namespace gem16::studio

#include "app.h"

#include "chat_history.h"
#include "markdown.h"
#include "model_catalog.h"
#include "selectable_text.h"
#include "settings.h"

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
  draw->AddPolyline(outer, 8, edge, ImDrawFlags_Closed, 1.2f);
  draw->AddLine(inner[5], inner[1], IM_COL32(100, 255, 203, 125), 1.0f);
}

void DrawNavIcon(ImDrawList* draw, Screen screen, ImVec2 center, ImU32 color) {
  if (screen == Screen::kChat) {
    draw->AddCircle(center, 9.0f, color, 18, 1.7f);
    draw->AddLine({center.x - 6.0f, center.y + 6.0f},
                  {center.x - 9.0f, center.y + 11.0f}, color, 1.7f);
  } else if (screen == Screen::kModels) {
    draw->AddRect({center.x - 9.0f, center.y - 8.0f},
                  {center.x + 9.0f, center.y + 8.0f}, color, 2.0f, 0, 1.5f);
    draw->AddLine({center.x, center.y - 8.0f}, {center.x, center.y + 8.0f}, color, 1.2f);
    draw->AddLine({center.x - 9.0f, center.y - 2.0f},
                  {center.x + 9.0f, center.y - 2.0f}, color, 1.2f);
  } else if (screen == Screen::kServer) {
    for (int row = -1; row <= 1; ++row) {
      draw->AddRect({center.x - 10.0f, center.y + row * 7.0f - 2.0f},
                    {center.x + 10.0f, center.y + row * 7.0f + 2.0f}, color,
                    1.5f, 0, 1.4f);
      draw->AddCircleFilled({center.x + 6.5f, center.y + row * 7.0f}, 1.2f, color);
    }
  } else {
    draw->AddCircle(center, 8.0f, color, 16, 1.7f);
    draw->AddCircle(center, 2.5f, color, 12, 1.5f);
    for (int index = 0; index < 8; ++index) {
      const float angle = static_cast<float>(index) * std::numbers::pi_v<float> * 0.25f;
      draw->AddLine({center.x + std::cos(angle) * 8.0f,
                     center.y + std::sin(angle) * 8.0f},
                    {center.x + std::cos(angle) * 11.0f,
                     center.y + std::sin(angle) * 11.0f}, color, 1.5f);
    }
  }
}

bool NavButton(const char* label, Screen screen, bool selected, float width) {
  static std::array<float, 4> glow_strength{};
  ImGui::InvisibleButton(label, {width, 48.0f});
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
  const ImVec2 center{minimum.x + 28.0f, (minimum.y + maximum.y) * 0.5f};
  ImDrawList* draw = ImGui::GetWindowDrawList();
  draw->PushClipRect(minimum, maximum, true);
  if (selected || glow > 0.01f) {
    draw->AddRectFilled(
        minimum, maximum,
        IM_COL32(8, 61, 43,
                 static_cast<int>((selected ? 162.0f : 105.0f) * glow)),
        11.0f, ImDrawFlags_RoundCornersRight);
    draw->AddCircleFilled(
        {minimum.x - 2.0f, center.y}, 34.0f,
        IM_COL32(38, 244, 164, static_cast<int>(42.0f * glow)), 36);
    draw->AddRectFilledMultiColor(
        {minimum.x + 1.0f, minimum.y + 2.0f},
        {minimum.x + width * 0.72f, maximum.y - 2.0f},
        IM_COL32(37, 239, 160, static_cast<int>(84.0f * glow)),
        IM_COL32(37, 239, 160, 0), IM_COL32(37, 239, 160, 0),
        IM_COL32(37, 239, 160, static_cast<int>(84.0f * glow)));
    draw->AddRectFilled(
        {minimum.x, minimum.y + 5.0f},
        {minimum.x + 2.0f, maximum.y - 5.0f},
        IM_COL32(76, 255, 190, static_cast<int>(245.0f * glow)), 1.0f);
    draw->AddRectFilled(
        {minimum.x + 2.0f, minimum.y + 8.0f},
        {minimum.x + 5.0f, maximum.y - 8.0f},
        IM_COL32(40, 244, 164, static_cast<int>(92.0f * glow)), 2.0f);
  }
  draw->PopClipRect();

  const ImVec4 idle{0.52f, 0.58f, 0.57f, 1.0f};
  const ImVec4 lit = selected || hovered
                         ? ImVec4(kAccent.x, kAccent.y, kAccent.z, 1.0f)
                         : idle;
  DrawNavIcon(draw, screen, center, ImGui::ColorConvertFloat4ToU32(lit));
  const ImVec2 text_size = ImGui::CalcTextSize(label);
  draw->AddText({minimum.x + 53.0f,
                 minimum.y + (48.0f - text_size.y) * 0.5f},
                ImGui::ColorConvertFloat4ToU32(
                    selected ? kAccent
                             : (hovered ? ImVec4(0.78f, 0.97f, 0.88f, 1.0f)
                                        : ImGui::GetStyleColorVec4(ImGuiCol_Text))),
                label);
  return clicked;
}

void StatusPill(const char* text, ImVec4 color) {
  const ImVec2 padding(10, 5);
  const ImVec2 text_size = ImGui::CalcTextSize(text);
  const ImVec2 min = ImGui::GetCursorScreenPos();
  const ImVec2 max(min.x + text_size.x + padding.x * 2, min.y + text_size.y + padding.y * 2);
  ImGui::GetWindowDrawList()->AddRectFilled(min, max, ImGui::ColorConvertFloat4ToU32({color.x * 0.20f, color.y * 0.20f, color.z * 0.20f, 0.95f}), 12);
  ImGui::GetWindowDrawList()->AddCircleFilled({min.x + 9, (min.y + max.y) * 0.5f}, 3, ImGui::ColorConvertFloat4ToU32(color));
  ImGui::SetCursorScreenPos({min.x + padding.x + 6, min.y + padding.y});
  ImGui::TextColored(color, "%s", text);
  ImGui::SetCursorScreenPos({max.x, min.y});
  ImGui::Dummy({0, max.y - min.y});
}

void SectionLabel(const char* text) {
  ImGui::TextColored({0.55f, 0.60f, 0.59f, 1.0f}, "%s", text);
  ImGui::Spacing();
}

void PanelHeading(const char* title, const char* description) {
  ImGui::SetWindowFontScale(1.16f);
  ImGui::TextColored(kAccent, "%s", title);
  ImGui::SetWindowFontScale(1.0f);
  if (description != nullptr && description[0] != '\0') {
    ImGui::TextDisabled("%s", description);
  }
  ImGui::Dummy({0, 8});
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
    draw->PathArcTo(center, 8.0f, 0.15f * std::numbers::pi_v<float>,
                    1.55f * std::numbers::pi_v<float>, 20);
    draw->PathStroke(color, 0, 1.8f);
    const ImVec2 arrow[3] = {{center.x - 9.0f, center.y - 5.0f},
                             {center.x - 10.0f, center.y + 3.0f},
                             {center.x - 3.0f, center.y - 1.0f}};
    draw->AddConvexPolyFilled(arrow, 3, color);
  } else if (icon == ComposerIcon::kDelete) {
    draw->AddRect({center.x - 6.0f, center.y - 5.0f},
                  {center.x + 6.0f, center.y + 8.0f}, color, 2.0f, 0, 1.7f);
    draw->AddLine({center.x - 8.0f, center.y - 8.0f},
                  {center.x + 8.0f, center.y - 8.0f}, color, 1.7f);
    draw->AddLine({center.x - 3.0f, center.y - 11.0f},
                  {center.x + 3.0f, center.y - 11.0f}, color, 1.7f);
  } else if (icon == ComposerIcon::kStop) {
    draw->AddRectFilled({center.x - 6.0f, center.y - 6.0f},
                        {center.x + 6.0f, center.y + 6.0f}, color, 2.0f);
  } else {
    const ImVec2 send[3] = {{center.x - 7.0f, center.y - 8.0f},
                            {center.x + 9.0f, center.y},
                            {center.x - 7.0f, center.y + 8.0f}};
    draw->AddTriangle(send[0], send[1], send[2], color, 2.0f);
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

StudioApp::StudioApp() : settings_(LoadSettings()) {
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
  style.WindowRounding = 16;
  style.ChildRounding = 14;
  style.FrameRounding = 10;
  style.PopupRounding = 12;
  style.ScrollbarRounding = 10;
  style.GrabRounding = 9;
  style.WindowBorderSize = 0;
  style.ChildBorderSize = 1;
  style.FrameBorderSize = 0;
  style.WindowPadding = {18, 18};
  style.FramePadding = {13, 10};
  style.ItemSpacing = {11, 11};
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
  } else {
    ImGui::StyleColorsLight();
    style.WindowRounding = 16;
    style.ChildRounding = 14;
    style.FrameRounding = 10;
    colors[ImGuiCol_Button] = {0.12f, 0.58f, 0.39f, 1};
    colors[ImGuiCol_ButtonHovered] = {0.09f, 0.48f, 0.32f, 1};
  }
}

void StudioApp::Render() {
  DrainChatEvents();
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
  ImGui::SameLine(0, 14);
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
    if (messages_.empty() || messages_.back().role != "assistant") continue;
    ChatMessage& message = messages_.back();
    if (event.kind == ChatEvent::Kind::kText) message.content += event.value;
    else if (event.kind == ChatEvent::Kind::kReasoning) message.reasoning += event.value;
    else if (event.kind == ChatEvent::Kind::kError) {
      message.content = std::move(event.value);
      message.error = true;
      message.streaming = false;
    } else if (event.kind == ChatEvent::Kind::kFinished) {
      message.streaming = false;
    }
    scroll_to_bottom_ = true;
  }
}

void StudioApp::DrawSidebar() {
  const ImVec2 logo_position = ImGui::GetCursorScreenPos();
  DrawGemstone(ImGui::GetWindowDrawList(),
               {logo_position.x + 21.0f, logo_position.y + 22.0f}, 19.0f);
  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 52.0f);
  ImGui::SetWindowFontScale(1.34f);
  ImGui::TextUnformatted("Gem 16");
  ImGui::SetWindowFontScale(1.0f);
  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 52.0f);
  ImGui::TextDisabled("Local AI Studio");
  ImGui::Dummy({0, 32});
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
  ImGui::SetCursorPosY(ImGui::GetWindowHeight() - 104);
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
  ImGui::SameLine(0, 14);
  StatusPill(PhaseLabel(server_.Phase()), PhaseColor(server_.Phase()));
  if (screen_ != Screen::kChat) {
    ImGui::SameLine(0, 10);
    ImGui::TextDisabled("%s", settings_.onboarding_complete
                                  ? ProfileLabel(settings_.server.profile)
                                  : "Choose a model to continue");
  }
}

void StudioApp::DrawChat() {
  const float available_width = ImGui::GetContentRegionAvail().x;
  const float action_width = 44.0f * 3.0f + 18.0f;
  const float input_width = std::max(180.0f, available_width - action_width - 38.0f);
  const int composer_lines = ComposerLineCount(composer_.data(), input_width - 24.0f);
  const float input_height = std::clamp(
      22.0f + static_cast<float>(composer_lines) * (ImGui::GetFontSize() + 3.0f),
      48.0f, 154.0f);
  const float composer_height = input_height + 36.0f;
  ImGui::BeginChild("##conversation", {0, -composer_height}, ImGuiChildFlags_None,
                    ImGuiWindowFlags_None);
  if (messages_.empty()) {
    const float y = std::max(30.0f, ImGui::GetContentRegionAvail().y * 0.23f);
    ImGui::Dummy({0, y});
    const ImVec2 center = {ImGui::GetCursorScreenPos().x + 35.0f,
                           ImGui::GetCursorScreenPos().y + 34.0f};
    DrawGemstone(ImGui::GetWindowDrawList(), center, 29.0f);
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 82.0f);
    ImGui::SetWindowFontScale(1.48f);
    ImGui::TextColored(kAccent, "What should we build today?");
    ImGui::SetWindowFontScale(1.0f);
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 82.0f);
    ImGui::TextWrapped(
        "Chat locally with %s. Responses stream directly from the resident GPU session.",
        ProfileLabel(settings_.server.profile));
  }
  for (std::size_t index = 0; index < messages_.size(); ++index) DrawMessage(messages_[index], index);
  if (scroll_to_bottom_) {
    ImGui::SetScrollHereY(1.0f);
    scroll_to_bottom_ = false;
  }
  ImGui::EndChild();

  ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 18.0f);
  ImGui::BeginChild("##composer", {0, 0}, ImGuiChildFlags_Borders,
                    ImGuiWindowFlags_NoScrollbar |
                        ImGuiWindowFlags_NoScrollWithMouse);
  ImGui::PopStyleVar();
  const HealthSnapshot health = server_.Health();
  const bool busy = api_.Busy();
  const bool has_draft = composer_[0] != '\0';
  const bool can_send = !busy && has_draft && health.available;
  const ImVec2 input_origin = ImGui::GetCursorScreenPos();
  ImGui::BeginDisabled(busy);
  const bool enter = ImGui::InputTextMultiline(
      "##message", composer_.data(), composer_.size(), {input_width, input_height},
      ImGuiInputTextFlags_EnterReturnsTrue |
          ImGuiInputTextFlags_CtrlEnterForNewLine);
  ImGui::EndDisabled();
  if (composer_[0] == '\0' && !ImGui::IsItemActive()) {
    ImGui::GetWindowDrawList()->AddText(
      {input_origin.x + 13.0f, input_origin.y + 12.0f},
        IM_COL32(122, 137, 132, 190),
        "Message Gem 16...  Enter to send · Shift+Enter for a new line");
  }
  const float action_y = input_origin.y + input_height - 44.0f;
  ImGui::SameLine(0, 6);
  ImGui::SetCursorScreenPos({ImGui::GetCursorScreenPos().x, action_y});
  ImGui::BeginDisabled(busy || messages_.empty());
  if (ComposerButton("##undo-turn", ComposerIcon::kUndo, 40.0f,
                     "Remove the last exchange", busy || messages_.empty()))
    RemoveLastExchange();
  ImGui::EndDisabled();
  ImGui::SameLine(0, 6);
  ImGui::BeginDisabled(busy || messages_.empty());
  if (ComposerButton("##clear-chat", ComposerIcon::kDelete, 40.0f,
                     "Delete the complete chat", busy || messages_.empty()))
    ClearChat();
  ImGui::EndDisabled();
  ImGui::SameLine(0, 8);
  ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 30.0f);
  if (busy) {
    if (ComposerButton("##stop", ComposerIcon::kStop, 44.0f,
                       "Stop generation"))
      api_.Cancel();
  } else {
    ImGui::BeginDisabled(!can_send);
    const bool send_clicked = ComposerButton(
        "##send", ComposerIcon::kSend, 44.0f,
        health.available ? "Send message" : "Server is offline", !can_send);
    ImGui::EndDisabled();
    if (send_clicked || (enter && can_send)) SendMessage();
  }
  ImGui::PopStyleVar();
  ImGui::EndChild();
}

void StudioApp::DrawMessage(const ChatMessage& message, std::size_t index) {
  const bool user = message.role == "user";
  const float available = ImGui::GetContentRegionAvail().x;
  const float width = available * (user ? 0.68f : 0.80f);
  const float original_x = ImGui::GetCursorPosX();
  if (user) {
    ImGui::SetCursorPosX(original_x + available - width - 10.0f);
  } else {
    const ImVec2 avatar = ImGui::GetCursorScreenPos();
    DrawGemstone(ImGui::GetWindowDrawList(), {avatar.x + 21.0f, avatar.y + 28.0f}, 18.0f);
    ImGui::SetCursorPosX(original_x + 52.0f);
  }
  ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 17.0f);
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
    ImGui::SetCursorPosX(ImGui::GetWindowWidth() - 64.0f);
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
  if (show_reasoning_ && (!message.reasoning.empty() || message.streaming)) {
    ImGui::Spacing();
    const bool reasoning_expanded = expanded_reasoning_.contains(index);
    ImGui::PushStyleColor(ImGuiCol_Button, {0.04f, 0.11f, 0.09f, 0.88f});
    const std::string reasoning_label =
        std::string("Reasoning##") + std::to_string(index);
    if (ImGui::Button(reasoning_label.c_str(), {-1.0f, 32.0f})) {
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
          {reasoning_min.x + 15.0f, reasoning_center_y - 3.0f},
          {reasoning_min.x + 23.0f, reasoning_center_y - 3.0f},
          {reasoning_min.x + 19.0f, reasoning_center_y + 4.0f},
          ImGui::GetColorU32(ImGuiCol_TextDisabled));
    } else {
      ImGui::GetWindowDrawList()->AddTriangleFilled(
          {reasoning_min.x + 16.0f, reasoning_center_y - 5.0f},
          {reasoning_min.x + 16.0f, reasoning_center_y + 5.0f},
          {reasoning_min.x + 23.0f, reasoning_center_y},
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
    const ImVec2 tail[3] = {{bubble_max.x - 18.0f, bubble_max.y - 1.0f},
                            {bubble_max.x + 7.0f, bubble_max.y - 1.0f},
                            {bubble_max.x - 5.0f, bubble_max.y - 15.0f}};
    ImGui::GetWindowDrawList()->AddConvexPolyFilled(
        tail, 3, ImGui::ColorConvertFloat4ToU32({0.035f, 0.25f, 0.17f, 0.93f}));
  } else {
    const ImVec2 tail[3] = {{bubble_min.x + 16.0f, bubble_min.y + 17.0f},
                            {bubble_min.x - 8.0f, bubble_min.y + 28.0f},
                            {bubble_min.x + 2.0f, bubble_min.y + 8.0f}};
    ImGui::GetWindowDrawList()->AddConvexPolyFilled(
        tail, 3, ImGui::ColorConvertFloat4ToU32({0.060f, 0.078f, 0.074f, 0.92f}));
  }
  if (message.error) ImGui::PopStyleColor();
  ImGui::PopStyleColor(2);
  ImGui::PopStyleVar();
  ImGui::SetCursorPosX(original_x);
  ImGui::Dummy({0, 12});
}

void StudioApp::DrawModels() {
  const ModelInstallState install = models_.State();
  PanelHeading(settings_.onboarding_complete ? "Local model profiles"
                                             : "Welcome to Gem 16",
               settings_.onboarding_complete
                   ? "Install either profile or keep both side by side in the shared Hugging Face cache."
                   : "Choose and install a model profile. Nothing is selected by default on a new system.");
  ImGui::TextDisabled("Hub cache: %s", HuggingFaceHubRoot().string().c_str());
  ImGui::Dummy({0, 8});
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
    ImGui::BeginChild(id.c_str(), {0, 218}, ImGuiChildFlags_Borders);
    const ImVec2 gem_origin = ImGui::GetCursorScreenPos();
    DrawGemstone(ImGui::GetWindowDrawList(),
                 {gem_origin.x + 27.0f, gem_origin.y + 29.0f}, 22.0f);
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 64.0f);
    ImGui::SetWindowFontScale(1.18f);
    ImGui::TextUnformatted(ProfileLabel(profile));
    ImGui::SetWindowFontScale(1.0f);
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 64.0f);
    ImGui::TextWrapped("%s", catalog.description);
    ImGui::Dummy({0, 4});
    ImGui::TextColored(kAccent, "%s", catalog.capabilities);
    ImGui::Text("Target: %s", profile_state.target_ready ? "Verified" : "Missing");
    ImGui::SameLine(180.0f);
    ImGui::Text("Assistant: %s", profile_state.assistant_ready ? "Verified" : "Missing");
    if (downloading) {
      const float progress = profile_state.total_bytes == 0 ? 0.0f :
          static_cast<float>(static_cast<double>(profile_state.completed_bytes) /
                             static_cast<double>(profile_state.total_bytes));
      ImGui::ProgressBar(std::clamp(progress, 0.0f, 1.0f), {-110.0f, 0.0f},
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
      ImGui::TextColored(kAccent, "● Installed and selected");
    } else {
      const std::string button = "Use this profile##" +
                                 std::string(ProfileWireName(profile));
      if (ImGui::Button(button.c_str())) SelectProfile(profile);
      ImGui::SameLine();
      ImGui::TextDisabled("Installed in the shared Hub cache");
    }
    ImGui::EndChild();
    ImGui::PopStyleColor(2);
    ImGui::Dummy({0, 10});
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
    ImGui::BeginChild("##server-error", {0, 64}, ImGuiChildFlags_Borders);
    ImGui::TextColored({1, 0.55f, 0.55f, 1}, "Server error");
    ImGui::TextWrapped("%s", server_.Error().c_str());
    ImGui::EndChild();
    ImGui::PopStyleColor();
    ImGui::Dummy({0, 10});
  }
  const float config_width = std::max(520.0f, ImGui::GetContentRegionAvail().x * 0.62f);
  ImGui::BeginChild("##server-config", {config_width, 0}, ImGuiChildFlags_Borders);
  PanelHeading("Server configuration",
               "Configure the local executable, resident model and generation path.");
  ImGui::TextDisabled("ACTIVE PROFILE");
  ImGui::SameLine();
  ImGui::TextColored(kAccent, "%s", ProfileLabel(settings_.server.profile));
  ImGui::Dummy({0, 5});
  TextField("Server executable", "##server-executable", executable_);
  TextField("Compiled target model", "##target-model", model_directory_);
  TextField("Compiled MTP assistant", "##mtp-assistant", assistant_directory_);
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
  ImGui::SameLine(0, 18);
  CapabilityChip(settings_.server.mtp_draft_tokens == 0 ? "MTP disabled" : "GPU MTP enabled",
                 settings_.server.mtp_draft_tokens != 0);
  SyncSettingsFromBuffers();
  server_.Configure(settings_.server);
  ImGui::Dummy({0, 6});
  const ServerPhase phase = server_.Phase();
  if (phase == ServerPhase::kRunning || phase == ServerPhase::kStarting || phase == ServerPhase::kStopping) {
    ImGui::BeginDisabled(phase == ServerPhase::kStopping);
    if (ImGui::Button("Stop server", {140, 42})) server_.Stop();
    ImGui::EndDisabled();
  } else {
    if (ImGui::Button(phase == ServerPhase::kExternal ? "External server" : "Start server", {140, 42}) && phase != ServerPhase::kExternal) {
      (void)SaveSettings(settings_);
      server_.Start(settings_.server);
    }
  }
  ImGui::EndChild();
  ImGui::SameLine(0, 12);
  ImGui::BeginChild("##server-log", {0, 0}, ImGuiChildFlags_Borders);
  PanelHeading("Runtime status", "Resident session health and server output.");
  StatusPill(PhaseLabel(server_.Phase()), PhaseColor(server_.Phase()));
  if (health.available) {
    ImGui::Dummy({0, 6});
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
  ImGui::Dummy({0, 6});
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
  const float left_width = std::max(280.0f, ImGui::GetContentRegionAvail().x * 0.34f);
  ImGui::BeginChild("##appearance-card", {left_width, 0},
                    ImGuiChildFlags_Borders);
  PanelHeading("Appearance", "The interface remains local and GPU-rendered.");
  if (ImGui::Checkbox("Dark glass theme", &settings_.dark_theme)) ApplyTheme();
  ImGui::Dummy({0, 8});
  ImGui::TextWrapped(
      "The animated science-fiction wave is rendered by the native OpenGL and Direct3D 11 backends.");
  ImGui::Dummy({0, 12});
  CapabilityChip("Animated shader");
  ImGui::SameLine();
  CapabilityChip("OS cursor");
  ImGui::EndChild();
  ImGui::SameLine(0, 12);
  ImGui::BeginChild("##generation-card", {0, 0}, ImGuiChildFlags_Borders);
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
  ImGui::InputTextMultiline("##system-prompt", system_prompt_.data(), system_prompt_.size(), {0, 120});
  SyncSettingsFromBuffers();
  if (ImGui::Button("Save settings", {140, 42})) (void)SaveSettings(settings_);
  ImGui::Dummy({0, 18});
  SectionLabel("ABOUT");
  ImGui::TextWrapped("gem16 Native Studio %s · C++20 · Dear ImGui · Linux OpenGL / Windows Direct3D 11",
                     GEM16_VERSION_STRING);
  ImGui::TextDisabled("Visual foundation adapted from Free Solace at bb35bb3 (MIT). Native operating-system cursor enabled.");
  ImGui::EndChild();
}

void StudioApp::SendMessage() {
  std::string text = composer_.data();
  if (text.empty() || api_.Busy()) return;
  messages_.push_back({"user", std::move(text), {}, false, false});
  composer_[0] = '\0';
  const auto request_messages = messages_;
  messages_.push_back({"assistant", {}, {}, true, false});
  api_.StreamChat(settings_.server, settings_.generation, request_messages, session_id_);
  scroll_to_bottom_ = true;
}

void StudioApp::ClearChat() {
  if (api_.Busy()) return;
  messages_.clear();
  session_id_.clear();
  composer_[0] = '\0';
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
  SyncSettingsFromBuffers();
  ApplyProfileDefaults(settings_.server, profile);
  settings_.onboarding_complete = true;
  SyncBuffersFromSettings();
  server_.Configure(settings_.server);
  if (api_.Busy()) api_.Cancel();
  ClearChat();
  (void)SaveSettings(settings_);
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

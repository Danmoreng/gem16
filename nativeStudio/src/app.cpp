#include "app.h"

#include "chat_history.h"
#include "gem16_logo.generated.h"
#include "markdown.h"
#include "media_loader.h"
#include "model_catalog.h"
#include "model_widgets.h"
#include "selectable_text.h"
#include "settings.h"
#include "platform_ui.h"

#include "imgui.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <numbers>

namespace gem16::studio {
namespace {

constexpr ImVec4 kAccent{0.31f, 0.91f, 0.65f, 1.0f};
constexpr ImVec4 kAccentDim{0.11f, 0.34f, 0.24f, 1.0f};
float g_ui_scale = 1.0f;

float Ui(float value) { return value * g_ui_scale; }

std::string UserFacingPath(const std::filesystem::path& path) {
  const std::filesystem::path normalized = path.lexically_normal();
  for (const char* variable : {"USERPROFILE", "HOME"}) {
    const char* value = std::getenv(variable);
    if (!value || !*value) continue;
    const std::filesystem::path user_root =
        std::filesystem::path(value).lexically_normal();
    const std::filesystem::path relative =
        normalized.lexically_relative(user_root);
    if (relative.empty() ||
        (relative.begin() != relative.end() && *relative.begin() == "..")) {
      continue;
    }
    if (relative == ".") return "~";
    return (std::filesystem::path("~") / relative).generic_string();
  }
  return normalized.string();
}

template <std::size_t Size>
void CopyTo(std::array<char, Size>& destination, const std::string& source) {
  const std::size_t count = std::min(source.size(), Size - 1);
  std::memcpy(destination.data(), source.data(), count);
  destination[count] = '\0';
}

std::size_t CountImages(const std::vector<ChatMessage>& messages,
                        const std::vector<MediaAttachment>& pending = {}) {
  std::size_t count = std::count_if(
      pending.begin(), pending.end(), [](const MediaAttachment& attachment) {
        return attachment.kind == MediaKind::kImage;
      });
  for (const ChatMessage& message : messages) {
    count += std::count_if(
        message.attachments.begin(), message.attachments.end(),
        [](const MediaAttachment& attachment) {
          return attachment.kind == MediaKind::kImage;
        });
  }
  return count;
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
    const ImVec2 bubble_min{center.x - Ui(10.0f), center.y - Ui(7.5f)};
    const ImVec2 bubble_max{center.x + Ui(10.0f), center.y + Ui(7.0f)};
    draw->AddRect(bubble_min, bubble_max, color, Ui(6.0f), 0, Ui(1.7f));
    draw->AddLine({center.x - Ui(5.5f), bubble_max.y - Ui(0.5f)},
                  {center.x - Ui(9.0f), center.y + Ui(11.0f)}, color,
                  Ui(1.7f));
    draw->AddLine({center.x - Ui(9.0f), center.y + Ui(11.0f)},
                  {center.x - Ui(1.0f), bubble_max.y}, color, Ui(1.7f));
    for (int dot = -1; dot <= 1; ++dot)
      draw->AddCircleFilled({center.x + dot * Ui(4.0f), center.y},
                            Ui(1.05f), color, 8);
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

bool NavButton(const char* label, Screen screen, bool selected, float width,
               ImTextureID flame_texture) {
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
    if (selected && flame_texture != ImTextureID_Invalid) {
      draw->AddImageRounded(
          ImTextureRef(flame_texture), minimum, maximum, {0.0f, 0.0f},
          {1.0f, 1.0f},
          IM_COL32(255, 255, 255, static_cast<int>(235.0f * glow)),
          Ui(11.0f), ImDrawFlags_RoundCornersRight);
    } else {
      draw->AddCircleFilled(
          {minimum.x - Ui(2.0f), center.y}, Ui(34.0f),
          IM_COL32(38, 244, 164, static_cast<int>(42.0f * glow)), 36);
      draw->AddRectFilledMultiColor(
          {minimum.x + Ui(1.0f), minimum.y + Ui(2.0f)},
          {minimum.x + width * 0.72f, maximum.y - Ui(2.0f)},
          IM_COL32(37, 239, 160, static_cast<int>(84.0f * glow)),
          IM_COL32(37, 239, 160, 0), IM_COL32(37, 239, 160, 0),
          IM_COL32(37, 239, 160, static_cast<int>(84.0f * glow)));
    }
    draw->AddRectFilled(
        minimum, {minimum.x + Ui(2.5f), maximum.y},
        IM_COL32(76, 255, 190, static_cast<int>(245.0f * glow)));
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
  } else if (bytes >= static_cast<std::uint64_t>(mib)) {
    std::snprintf(result, sizeof(result), "%.1f MiB", static_cast<double>(bytes) / mib);
  } else if (bytes >= 1024U) {
    std::snprintf(result, sizeof(result), "%.1f KiB",
                  static_cast<double>(bytes) / 1024.0);
  } else {
    std::snprintf(result, sizeof(result), "%llu B",
                  static_cast<unsigned long long>(bytes));
  }
  return result;
}

enum class ComposerIcon { kAttach, kMic, kMicStop, kUndo, kDelete, kSend, kStop };

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
  if (icon == ComposerIcon::kAttach) {
    draw->PathLineTo({center.x + Ui(2.0f), center.y - Ui(10.0f)});
    draw->PathBezierCubicCurveTo(
        {center.x + Ui(8.0f), center.y - Ui(10.0f)},
        {center.x + Ui(10.0f), center.y - Ui(4.0f)},
        {center.x + Ui(6.0f), center.y}, 12);
    draw->PathLineTo({center.x - Ui(2.0f), center.y + Ui(8.0f)});
    draw->PathBezierCubicCurveTo(
        {center.x - Ui(6.0f), center.y + Ui(12.0f)},
        {center.x - Ui(12.0f), center.y + Ui(5.0f)},
        {center.x - Ui(7.0f), center.y}, 12);
    draw->PathLineTo({center.x + Ui(1.0f), center.y - Ui(8.0f)});
    draw->PathBezierCubicCurveTo(
        {center.x + Ui(4.0f), center.y - Ui(11.0f)},
        {center.x + Ui(8.0f), center.y - Ui(6.0f)},
        {center.x + Ui(5.0f), center.y - Ui(3.0f)}, 10);
    draw->PathLineTo({center.x - Ui(2.0f), center.y + Ui(4.0f)});
    draw->PathStroke(color, 0, Ui(1.8f));
  } else if (icon == ComposerIcon::kMic ||
             icon == ComposerIcon::kMicStop) {
    if (icon == ComposerIcon::kMicStop) {
      draw->AddRectFilled({center.x - Ui(5.0f), center.y - Ui(5.0f)},
                          {center.x + Ui(5.0f), center.y + Ui(5.0f)}, color,
                          Ui(1.5f));
    } else {
      draw->AddRect({center.x - Ui(4.0f), center.y - Ui(9.0f)},
                    {center.x + Ui(4.0f), center.y + Ui(3.0f)}, color,
                    Ui(4.0f), 0, Ui(1.8f));
      draw->PathArcTo({center.x, center.y + Ui(1.0f)}, Ui(8.0f), 0.0f,
                      std::numbers::pi_v<float>, 16);
      draw->PathStroke(color, 0, Ui(1.8f));
      draw->AddLine({center.x, center.y + Ui(9.0f)},
                    {center.x, center.y + Ui(12.0f)}, color, Ui(1.8f));
      draw->AddLine({center.x - Ui(5.0f), center.y + Ui(12.0f)},
                    {center.x + Ui(5.0f), center.y + Ui(12.0f)}, color,
                    Ui(1.8f));
    }
  } else if (icon == ComposerIcon::kUndo) {
    const ImVec2 start{center.x - Ui(8.0f), center.y - Ui(3.0f)};
    const ImVec2 arrow[3] = {
        {start.x, start.y},
        {start.x + Ui(6.0f), start.y - Ui(5.0f)},
        {start.x + Ui(6.0f), start.y + Ui(5.0f)},
    };
    draw->AddConvexPolyFilled(arrow, 3, color);
    draw->PathLineTo({start.x + Ui(5.0f), start.y});
    draw->PathBezierCubicCurveTo(
        {center.x + Ui(9.0f), center.y - Ui(4.0f)},
        {center.x + Ui(10.0f), center.y + Ui(8.0f)},
        {center.x + Ui(2.0f), center.y + Ui(9.0f)}, 18);
    draw->PathStroke(color, 0, Ui(1.9f));
  } else if (icon == ComposerIcon::kDelete) {
    draw->AddRect({center.x - Ui(7.0f), center.y - Ui(5.0f)},
                  {center.x + Ui(7.0f), center.y + Ui(9.0f)}, color, Ui(2.0f),
                  0, Ui(1.7f));
    draw->AddLine({center.x - Ui(9.0f), center.y - Ui(8.0f)},
                  {center.x + Ui(9.0f), center.y - Ui(8.0f)}, color, Ui(1.7f));
    draw->AddLine({center.x - Ui(3.0f), center.y - Ui(11.0f)},
                  {center.x + Ui(3.0f), center.y - Ui(11.0f)}, color, Ui(1.7f));
    draw->AddLine({center.x - Ui(3.0f), center.y - Ui(2.0f)},
                  {center.x - Ui(3.0f), center.y + Ui(6.0f)}, color, Ui(1.3f));
    draw->AddLine({center.x + Ui(3.0f), center.y - Ui(2.0f)},
                  {center.x + Ui(3.0f), center.y + Ui(6.0f)}, color, Ui(1.3f));
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

float AttachmentGalleryHeight(std::size_t count, float width) {
  if (count == 0) return 0.0f;
  const float card_width = Ui(152.0f);
  const float gap = Ui(8.0f);
  const std::size_t columns = std::max<std::size_t>(
      1, static_cast<std::size_t>((std::max(width, card_width) + gap) /
                                  (card_width + gap)));
  const std::size_t rows = (count + columns - 1U) / columns;
  return static_cast<float>(rows) * Ui(112.0f) +
         static_cast<float>(rows - 1U) * gap;
}

std::string FormatRecordingTime(std::uint64_t milliseconds) {
  const std::uint64_t total_seconds = milliseconds / 1000U;
  const unsigned minutes = static_cast<unsigned>(
      std::min<std::uint64_t>(total_seconds / 60U, 999U));
  const unsigned seconds = static_cast<unsigned>(total_seconds % 60U);
  char value[16]{};
  std::snprintf(value, sizeof(value), "%02u:%02u", minutes, seconds);
  return value;
}

}  // namespace

StudioApp::StudioApp(StudioSettings settings, float automatic_ui_scale)
    : settings_(std::move(settings)), automatic_ui_scale_(automatic_ui_scale) {
  ApplyUiScale(settings_.ui_scale);
  if (!settings_.onboarding_complete) screen_ = Screen::kModels;
  SyncBuffersFromSettings();
  server_.Configure(settings_.server);
  (void)logo_texture_.Load(kGem16LogoPng, kGem16LogoPngSize);
}

StudioApp::~StudioApp() {
  SyncSettingsFromBuffers();
  (void)SaveSettings(settings_);
  api_.Cancel();
}

void StudioApp::DrawAppLogo(ImVec2 position, float size) {
  if (logo_texture_.Valid()) {
    ImGui::GetWindowDrawList()->AddImageRounded(
        ImTextureRef(logo_texture_.Id()), position,
        {position.x + size, position.y + size}, {0.0f, 0.0f}, {1.0f, 1.0f},
        IM_COL32_WHITE, Ui(7.0f));
    return;
  }
  DrawGemstone(ImGui::GetWindowDrawList(),
               {position.x + size * 0.5f, position.y + size * 0.5f},
               size * 0.46f);
}

void StudioApp::ApplyUiScale(float configured_scale) {
  ui_scale_ = configured_scale > 0.0f ? configured_scale : automatic_ui_scale_;
  g_ui_scale = ui_scale_;
  sidebar_width_ = Ui(176.0f);
  ImGui::GetStyle().FontScaleMain = ui_scale_;
  ApplyTheme();
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
  if (pending_profile_ && !api_.Busy()) {
    const auto profile = *pending_profile_;
    pending_profile_.reset();
    SelectProfile(profile);
  }
  if (recorder_.Active() && !recorder_.Recording()) FinishRecording();
  PruneAttachmentTextures();
  const auto dropped = DrainDroppedFiles();
  if (!dropped.empty()) {
    if (screen_ == Screen::kChat)
      AddAttachments(dropped);
    else
      attachment_error_ = "Open Chat before dropping attachments.";
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
    if (event.kind == ChatEvent::Kind::kPerformance) {
      performance_ = event.performance;
      continue;
    }
    if (messages_.empty() || messages_.back().role != "assistant") continue;
    ChatMessage& message = messages_.back();
    if (event.kind == ChatEvent::Kind::kText ||
        event.kind == ChatEvent::Kind::kReasoning) {
      if (streamed_chunks_ == 0) first_token_at_ = std::chrono::steady_clock::now();
      ++streamed_chunks_;
    }
    ApplyChatEvent(message, event);
    if (message.error) session_id_.clear();
    if (event.kind == ChatEvent::Kind::kFinished || event.kind == ChatEvent::Kind::kError)
      generation_finished_ = std::chrono::steady_clock::now();
    if (auto_follow_) scroll_to_bottom_ = true;
  }
}

void StudioApp::DrawSidebar() {
  const ImVec2 logo_position = ImGui::GetCursorScreenPos();
  DrawAppLogo(logo_position, Ui(44.0f));
  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + Ui(52.0f));
  ImGui::SetWindowFontScale(1.34f);
  ImGui::TextUnformatted("Gem 16");
  ImGui::SetWindowFontScale(1.0f);
  ImGui::Dummy({0, Ui(32)});
  const float width = ImGui::GetWindowWidth() - Ui(11.0f);
  ImGui::SetCursorPosX(1.0f);
  if (NavButton("Chat", Screen::kChat, screen_ == Screen::kChat, width,
                navigation_flame_texture_))
    screen_ = Screen::kChat;
  ImGui::SetCursorPosX(1.0f);
  if (NavButton("Models", Screen::kModels, screen_ == Screen::kModels, width,
                navigation_flame_texture_))
    screen_ = Screen::kModels;
  ImGui::SetCursorPosX(1.0f);
  if (NavButton("Server", Screen::kServer, screen_ == Screen::kServer, width,
                navigation_flame_texture_))
    screen_ = Screen::kServer;
  ImGui::SetCursorPosX(1.0f);
  if (NavButton("Settings", Screen::kSettings, screen_ == Screen::kSettings,
                width, navigation_flame_texture_))
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
  const float recording_height = recorder_.Active() ? Ui(28.0f) : 0.0f;
  const float context_height = ImGui::GetFontSize();
  const float attachment_height = AttachmentGalleryHeight(
      pending_attachments_.size(), composer_content_width);
  const float error_height = attachment_error_.empty() ? 0.0f : ImGui::GetFontSize();
  const int composer_gaps = 2 + (recorder_.Active() ? 1 : 0) +
                            (!pending_attachments_.empty() ? 1 : 0) +
                            (!attachment_error_.empty() ? 1 : 0);
  const float composer_height = ImGui::GetStyle().WindowPadding.y * 2.0f +
                                recording_height + toolbar_height + context_height +
                                attachment_height +
                                error_height + input_height +
                                static_cast<float>(composer_gaps) * Ui(6.0f);
  ImGui::BeginChild("##conversation", {0, -composer_height}, ImGuiChildFlags_None,
                    ImGuiWindowFlags_None);
  if (messages_.empty()) {
    const float y = std::max(Ui(30.0f), ImGui::GetContentRegionAvail().y * 0.23f);
    ImGui::Dummy({0, y});
    const ImVec2 logo = {ImGui::GetCursorScreenPos().x,
                         ImGui::GetCursorScreenPos().y};
    DrawAppLogo(logo, Ui(68.0f));
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + Ui(82.0f));
    ImGui::SetWindowFontScale(1.48f);
    ImGui::TextColored(kAccent, "%s",
                       settings_.onboarding_complete
                           ? "What should we build today?"
                           : "Select a model to begin");
    ImGui::SetWindowFontScale(1.0f);
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + Ui(82.0f));
    if (settings_.onboarding_complete) {
      ImGui::TextWrapped(
          "Chat locally with %s. Responses stream directly from the resident GPU session.",
          ProfileLabel(settings_.server.profile));
    } else {
      ImGui::TextWrapped(
          "Open Models and select either qualified profile. You can return here at any time.");
    }
  }
  for (std::size_t index = 0; index < messages_.size(); ++index)
    DrawMessage(messages_[index], index);
  if (retry_requested_) {
    retry_requested_ = false;
    RetryLastRequest();
  }
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
  const std::string live_mismatch =
      health.available ? HealthCompatibilityError(settings_.server, health)
                       : std::string{};
  const ServerPhase server_phase = server_.Phase();
  const bool live_compatible =
      live_mismatch.empty() &&
      (server_phase == ServerPhase::kRunning ||
       server_phase == ServerPhase::kExternal);
  const bool can_send = settings_.onboarding_complete && !busy &&
                        !recorder_.Active() && has_draft && health.available &&
                        live_compatible;

  if (recorder_.Active()) {
    const ImVec2 recording_origin = ImGui::GetCursorScreenPos();
    ImDrawList* draw = ImGui::GetWindowDrawList();
    const std::uint64_t elapsed = recorder_.ElapsedMilliseconds();
    const std::uint64_t maximum = kMaximumRecordingSeconds * 1000U;
    const std::uint64_t remaining = elapsed >= maximum ? 0U : maximum - elapsed;
    const std::string label = "Recording " + FormatRecordingTime(elapsed) +
                              " / 00:30 · " +
                              FormatRecordingTime(remaining) + " left";
    const float center_y = recording_origin.y + recording_height * 0.5f;
    draw->AddCircleFilled({recording_origin.x + Ui(5.0f), center_y},
                          Ui(4.0f), IM_COL32(255, 91, 83, 255));
    draw->AddText({recording_origin.x + Ui(16.0f),
                   center_y - ImGui::GetFontSize() * 0.5f},
                  ImGui::GetColorU32(ImGuiCol_Text), label.c_str());
    const float label_width = ImGui::CalcTextSize(label.c_str()).x;
    const float bar_left = recording_origin.x + Ui(28.0f) + label_width;
    const float bar_right = recording_origin.x + ImGui::GetContentRegionAvail().x;
    if (bar_right > bar_left + Ui(30.0f)) {
      draw->AddRectFilled({bar_left, center_y - Ui(3.0f)},
                          {bar_right, center_y + Ui(3.0f)},
                          ImGui::GetColorU32(ImGuiCol_FrameBg), Ui(3.0f));
      const float progress = std::clamp(
          static_cast<float>(elapsed) / static_cast<float>(maximum), 0.0f,
          1.0f);
      draw->AddRectFilled({bar_left, center_y - Ui(3.0f)},
                          {bar_left + (bar_right - bar_left) * progress,
                           center_y + Ui(3.0f)},
                          IM_COL32(255, 91, 83, 255), Ui(3.0f));
    }
    ImGui::SetCursorScreenPos(
        {recording_origin.x, recording_origin.y + recording_height + Ui(6.0f)});
  }

  const ImVec2 toolbar_origin = ImGui::GetCursorScreenPos();
  ImDrawList* composer_draw = ImGui::GetWindowDrawList();
  const ImU32 disabled_text = ImGui::GetColorU32(ImGuiCol_TextDisabled);
  const float toolbar_text_y = toolbar_origin.y +
                               (toolbar_height - ImGui::GetFontSize()) * 0.5f;
  float toolbar_x = toolbar_origin.x;
  ImGui::SetCursorScreenPos({toolbar_x, toolbar_origin.y});
  const bool attach_disabled = busy || recorder_.Active();
  ImGui::BeginDisabled(attach_disabled);
  if (ComposerButton("##attach-files", ComposerIcon::kAttach, toolbar_height,
                     "Attach files", attach_disabled))
    AddAttachments(OpenAttachmentDialog());
  ImGui::EndDisabled();
  toolbar_x += toolbar_height + Ui(8.0f);
  const bool media_profile = settings_.onboarding_complete &&
                             settings_.server.profile == ModelProfile::kGemma4Unified12B;
  ImGui::SetCursorScreenPos({toolbar_x, toolbar_origin.y});
  const bool mic_disabled =
      !media_profile || (busy && !recorder_.Active());
  ImGui::BeginDisabled(mic_disabled);
  if (ComposerButton("##record-mic",
                     recorder_.Active() ? ComposerIcon::kMicStop
                                        : ComposerIcon::kMic,
                     toolbar_height,
                     recorder_.Active() ? "Stop and attach recording"
                                        : "Record audio",
                     mic_disabled)) {
    if (recorder_.Active()) {
      FinishRecording();
    } else {
      (void)recorder_.Start(attachment_error_);
    }
  }
  ImGui::EndDisabled();
  toolbar_x += toolbar_height + Ui(8.0f);
  ImGui::SetCursorScreenPos({toolbar_x, toolbar_origin.y});
  ImGui::SetNextItemWidth(Ui(150.0f));
  const char* effort_labels[] = {"Thinking: Off", "Thinking: Low (1K)",
                                "Thinking: Med (4K)", "Thinking: High (8K)"};
  const char* effort_keys[] = {"none", "low", "medium", "high"};
  int effort = 2;
  for (int index = 0; index < 4; ++index)
    if (settings_.generation.reasoning_effort == effort_keys[index]) effort = index;
  if (ImGui::Combo("##chat-effort", &effort, effort_labels, 4)) {
    settings_.generation.reasoning_effort = effort_keys[effort];
    (void)SaveSettings(settings_);
  }
  toolbar_x += Ui(158.0f);
  ImGui::SetCursorScreenPos({toolbar_x, toolbar_origin.y});
  ImGui::SetNextItemWidth(Ui(120.0f));
  constexpr std::array<std::pair<std::int64_t, const char*>, 6U> kOutputPresets{{
      {4096, "4K max"},
      {8192, "8K max"},
      {16384, "16K max"},
      {32768, "32K max"},
      {65536, "64K max"},
      {131072, "128K max"},
  }};
  int output_idx = 3;
  bool custom_output = true;
  for (std::size_t i = 0; i < kOutputPresets.size(); ++i) {
    if (settings_.generation.max_output_tokens == kOutputPresets[i].first) {
      output_idx = static_cast<int>(i);
      custom_output = false;
      break;
    }
  }
  char output_preview[32]{};
  if (custom_output) {
    std::snprintf(output_preview, sizeof(output_preview), "%lld max",
                  static_cast<long long>(settings_.generation.max_output_tokens));
  } else {
    std::snprintf(output_preview, sizeof(output_preview), "%s",
                  kOutputPresets[output_idx].second);
  }
  if (ImGui::BeginCombo("##chat-max-tokens", output_preview)) {
    for (std::size_t i = 0; i < kOutputPresets.size(); ++i) {
      const bool is_selected = (!custom_output && output_idx == static_cast<int>(i));
      if (ImGui::Selectable(kOutputPresets[i].second, is_selected)) {
        settings_.generation.max_output_tokens = kOutputPresets[i].first;
        (void)SaveSettings(settings_);
      }
      if (is_selected) ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }
  toolbar_x += Ui(128.0f);
  char status_label[192]{};
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
    else if (!busy && performance_)
      std::snprintf(
          status_label, sizeof(status_label),
          "%lld in · %lld out · Decode %.1f tok/s · Prefill %.0f tok/s · %.1fs",
          static_cast<long long>(prompt_tokens_),
          static_cast<long long>(completion_tokens_),
          performance_->decode_tokens_per_second,
          performance_->prefill_tokens_per_second, seconds);
    else
      std::snprintf(status_label, sizeof(status_label),
                    "%lld in · %lld out · Stream %.1f tok/s · %.1fs",
                    static_cast<long long>(prompt_tokens_),
                    static_cast<long long>(observed_tokens), rate, seconds);
  } else if (!settings_.onboarding_complete) {
    std::snprintf(status_label, sizeof(status_label),
                  "Select a model profile in Models to begin");
  } else {
    if (settings_.server.profile ==
        ModelProfile::kGemma4Moe26BTrellis35VisionFp8) {
      std::snprintf(status_label, sizeof(status_label),
                    "Vision · one image · %d-token budget",
                    settings_.server.vision_soft_token_budget);
    } else {
      std::snprintf(status_label, sizeof(status_label),
                    "Drop files anywhere · media follow the selected profile");
    }
  }
  const float toolbar_right = ImGui::GetWindowPos().x +
                              ImGui::GetWindowContentRegionMax().x;
  if (toolbar_x + ImGui::CalcTextSize(status_label).x > toolbar_right &&
      !busy && performance_) {
    std::snprintf(status_label, sizeof(status_label),
                  "%lld/%lld · D %.1f · P %.0f tok/s",
                  static_cast<long long>(prompt_tokens_),
                  static_cast<long long>(completion_tokens_),
                  performance_->decode_tokens_per_second,
                  performance_->prefill_tokens_per_second);
  }
  if (toolbar_x < toolbar_right) {
    composer_draw->PushClipRect({toolbar_x, toolbar_origin.y},
                                {toolbar_right, toolbar_origin.y + toolbar_height},
                                true);
    composer_draw->AddText({toolbar_x, toolbar_text_y},
                           busy ? ImGui::GetColorU32(kAccent) : disabled_text,
                           status_label);
    composer_draw->PopClipRect();
  }
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

  if (!pending_attachments_.empty())
    DrawAttachmentGallery(pending_attachments_, &pending_attachments_);
  if (!live_mismatch.empty() && health.available) {
    ImGui::TextColored({1.0f, 0.47f, 0.42f, 1.0f}, "%s",
                       live_mismatch.c_str());
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + Ui(6.0f));
  }
  if (!attachment_error_.empty()) {
    ImGui::TextColored({1.0f, 0.47f, 0.42f, 1.0f}, "%s", attachment_error_.c_str());
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + Ui(6.0f));
  }

  const ImVec2 input_origin = ImGui::GetCursorScreenPos();
  ImGui::BeginDisabled(busy || recorder_.Active());
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
    const char* send_hint = !settings_.onboarding_complete
                                ? "Select a model profile first"
                            : !health.available
                                ? "Server is offline"
                            : !live_mismatch.empty()
                                ? "Live server profile or capabilities do not match"
                                : "Send message";
    const bool send_clicked = ComposerButton(
        "##send", ComposerIcon::kSend, Ui(44.0f), send_hint, !can_send);
    ImGui::EndDisabled();
    if (send_clicked || (enter && can_send)) SendMessage();
  }
  ImGui::PopStyleVar();
  ImGui::SetCursorScreenPos({input_origin.x, input_origin.y + input_height});
  ImGui::Dummy({0.0f, 0.0f});
  ImGui::EndChild();
}

ImageTexture* StudioApp::AttachmentTexture(
    const MediaAttachment& attachment) {
  if (attachment.kind != MediaKind::kImage || attachment.id == 0 ||
      attachment.bytes.empty())
    return nullptr;
  if (const auto found = attachment_textures_.find(attachment.id);
      found != attachment_textures_.end())
    return found->second->Valid() ? found->second.get() : nullptr;
  auto texture = std::make_unique<ImageTexture>();
  const bool loaded = texture->Load(attachment.bytes.data(), attachment.bytes.size());
  ImageTexture* result = loaded ? texture.get() : nullptr;
  attachment_textures_.emplace(attachment.id, std::move(texture));
  return result;
}

void StudioApp::PruneAttachmentTextures() {
  std::unordered_set<std::uint64_t> active;
  for (const MediaAttachment& attachment : pending_attachments_)
    if (attachment.kind == MediaKind::kImage) active.insert(attachment.id);
  for (const ChatMessage& message : messages_)
    for (const MediaAttachment& attachment : message.attachments)
      if (attachment.kind == MediaKind::kImage) active.insert(attachment.id);
  std::erase_if(attachment_textures_, [&active](const auto& entry) {
    return !active.contains(entry.first);
  });
}

void StudioApp::DrawAttachmentGallery(
    const std::vector<MediaAttachment>& attachments,
    std::vector<MediaAttachment>* removable) {
  if (attachments.empty()) return;
  constexpr ImU32 card_fill = IM_COL32(17, 42, 34, 246);
  constexpr ImU32 card_border = IM_COL32(53, 83, 72, 235);
  const float card_width = Ui(152.0f);
  const float card_height = Ui(112.0f);
  const float gap = Ui(8.0f);
  const ImVec2 origin = ImGui::GetCursorScreenPos();
  const float right = origin.x + ImGui::GetContentRegionAvail().x;
  float x = origin.x;
  float y = origin.y;
  ImDrawList* draw = ImGui::GetWindowDrawList();
  for (std::size_t index = 0; index < attachments.size(); ++index) {
    if (index > 0 && x + card_width > right) {
      x = origin.x;
      y += card_height + gap;
    }
    const MediaAttachment& attachment = attachments[index];
    ImGui::PushID(attachment.id == 0 ? static_cast<int>(index)
                                     : static_cast<int>(attachment.id));
    ImGui::SetCursorScreenPos({x, y});
    ImGui::InvisibleButton("##attachment-card", {card_width, card_height});
    const ImVec2 minimum{x, y};
    const ImVec2 maximum{x + card_width, y + card_height};
    draw->AddRectFilled(minimum, maximum, card_fill, Ui(10.0f));
    draw->AddRect(minimum, maximum,
                  ImGui::IsItemHovered() ? ImGui::GetColorU32(kAccent)
                                         : card_border,
                  Ui(10.0f), 0, Ui(1.0f));

    const float inset = Ui(7.0f);
    const ImVec2 preview_min{x + inset, y + inset};
    const ImVec2 preview_max{x + card_width - inset, y + Ui(67.0f)};
    if (ImageTexture* texture = AttachmentTexture(attachment)) {
      const float source_aspect = static_cast<float>(texture->Width()) /
                                  static_cast<float>(texture->Height());
      const float target_aspect = (preview_max.x - preview_min.x) /
                                  (preview_max.y - preview_min.y);
      ImVec2 uv_min{0.0f, 0.0f};
      ImVec2 uv_max{1.0f, 1.0f};
      if (source_aspect > target_aspect) {
        const float visible = target_aspect / source_aspect;
        uv_min.x = (1.0f - visible) * 0.5f;
        uv_max.x = 1.0f - uv_min.x;
      } else {
        const float visible = source_aspect / target_aspect;
        uv_min.y = (1.0f - visible) * 0.5f;
        uv_max.y = 1.0f - uv_min.y;
      }
      draw->AddImageRounded(ImTextureRef(texture->Id()), preview_min,
                            preview_max, uv_min, uv_max, IM_COL32_WHITE,
                            Ui(7.0f));
    } else {
      draw->AddRectFilled(preview_min, preview_max, IM_COL32(12, 32, 27, 255),
                          Ui(7.0f));
      const ImVec2 center{(preview_min.x + preview_max.x) * 0.5f,
                          (preview_min.y + preview_max.y) * 0.5f};
      if (attachment.kind == MediaKind::kAudio) {
        const ImU32 color = ImGui::GetColorU32(kAccent);
        draw->AddCircleFilled({center.x - Ui(4.0f), center.y + Ui(7.0f)},
                              Ui(4.0f), color);
        draw->AddRectFilled({center.x, center.y - Ui(11.0f)},
                            {center.x + Ui(3.0f), center.y + Ui(7.0f)}, color,
                            Ui(1.0f));
        draw->AddLine({center.x + Ui(2.0f), center.y - Ui(10.0f)},
                      {center.x + Ui(10.0f), center.y - Ui(13.0f)}, color,
                      Ui(3.0f));
      } else {
        const ImU32 color = ImGui::GetColorU32(kAccent);
        draw->AddRect({center.x - Ui(9.0f), center.y - Ui(13.0f)},
                      {center.x + Ui(9.0f), center.y + Ui(13.0f)}, color,
                      Ui(2.0f), 0, Ui(1.7f));
        for (int row = -1; row <= 1; ++row)
          draw->AddLine({center.x - Ui(5.0f), center.y + row * Ui(5.0f)},
                        {center.x + Ui(5.0f), center.y + row * Ui(5.0f)},
                        color, Ui(1.3f));
      }
    }

    const char* kind = attachment.kind == MediaKind::kImage
                           ? "Image"
                           : (attachment.kind == MediaKind::kAudio
                                  ? "Audio"
                                  : (attachment.format == "pdf" ? "PDF" : "Text"));
    const ImVec2 text_min{x + inset, y + Ui(73.0f)};
    const ImVec2 text_max{x + card_width - inset, maximum.y - inset};
    draw->PushClipRect(text_min, text_max, true);
    draw->AddText(text_min, ImGui::GetColorU32(ImGuiCol_Text),
                  attachment.file_name.c_str());
    std::string detail = std::string(kind) + " · " +
                         FormatBytes(attachment.byte_size);
    if (attachment.kind == MediaKind::kImage &&
        settings_.server.profile ==
            ModelProfile::kGemma4Moe26BTrellis35VisionFp8) {
      const std::uint32_t estimate = EstimateVisionSoftTokens(
          attachment,
          static_cast<std::uint32_t>(settings_.server.vision_soft_token_budget));
      detail = std::to_string(settings_.server.vision_soft_token_budget) +
               " budget · ~" + std::to_string(estimate) + " tokens";
    }
    draw->AddText({text_min.x, y + Ui(92.0f)},
                  ImGui::GetColorU32(ImGuiCol_TextDisabled), detail.c_str());
    draw->PopClipRect();

    if (removable != nullptr) {
      const ImVec2 remove_min{x + card_width - Ui(25.0f), y + Ui(3.0f)};
      ImGui::SetCursorScreenPos(remove_min);
      ImGui::InvisibleButton("##remove-attachment", {Ui(22.0f), Ui(22.0f)});
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Remove %s", attachment.file_name.c_str());
      const ImU32 remove_color = ImGui::GetColorU32(
          ImGui::IsItemHovered() ? ImGuiCol_Text : ImGuiCol_TextDisabled);
      draw->AddCircleFilled({remove_min.x + Ui(11.0f), remove_min.y + Ui(11.0f)},
                            Ui(9.0f), IM_COL32(7, 21, 17, 220));
      draw->AddLine({remove_min.x + Ui(7.0f), remove_min.y + Ui(7.0f)},
                    {remove_min.x + Ui(15.0f), remove_min.y + Ui(15.0f)},
                    remove_color, Ui(1.5f));
      draw->AddLine({remove_min.x + Ui(15.0f), remove_min.y + Ui(7.0f)},
                    {remove_min.x + Ui(7.0f), remove_min.y + Ui(15.0f)},
                    remove_color, Ui(1.5f));
      if (ImGui::IsItemClicked()) {
        removable->erase(removable->begin() +
                         static_cast<std::ptrdiff_t>(index));
        ImGui::PopID();
        ImGui::SetCursorScreenPos(
            {origin.x, origin.y + AttachmentGalleryHeight(
                                      removable->size(), right - origin.x) + gap});
        return;
      }
    }
    ImGui::PopID();
    x += card_width + gap;
  }
  ImGui::SetCursorScreenPos(
      {origin.x, y + card_height + Ui(6.0f)});
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
    DrawAppLogo({avatar.x + Ui(3.0f), avatar.y + Ui(10.0f)}, Ui(38.0f));
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
    ImGui::SetCursorPosX(
        ImGui::GetWindowWidth() - Ui(message.error ? 120.0f : 64.0f));
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
  if (!user && message.error && index + 1U == messages_.size()) {
    ImGui::SameLine();
    if (ImGui::SmallButton(
            (std::string("Retry##") + std::to_string(index)).c_str())) {
      retry_requested_ = true;
    }
  }
  if (!message.error_message.empty()) {
    ImGui::PushStyleColor(ImGuiCol_Text, {1.0f, 0.55f, 0.55f, 1.0f});
    ImGui::TextWrapped("%s", message.error_message.c_str());
    ImGui::TextWrapped("This exchange is excluded from future context. Retry to include it.");
    ImGui::PopStyleColor();
  }
  if (!message.attachments.empty()) {
    ImGui::Spacing();
    DrawAttachmentGallery(message.attachments);
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
                     message.content, ImGui::GetContentRegionAvail().x, &svg_previews_);
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
  const std::filesystem::path hub_root = HuggingFaceHubRoot();
  const std::string hub_root_label = UserFacingPath(hub_root);
  PanelHeading(settings_.onboarding_complete ? "Local model profiles"
                                             : "Welcome to Gem 16",
               settings_.onboarding_complete
                   ? "Install either profile or keep both side by side in the shared Hugging Face cache."
                   : "Choose and install a model profile. Nothing is selected by default on a new system.");
  ImGui::TextDisabled("Hub cache: %s", hub_root_label.c_str());
  ImGui::SameLine();
  if (ImGui::SmallButton("Open cache")) OpenInFileManager(hub_root);
  ImGui::SameLine();
  ImGui::BeginDisabled(install.downloading || install.verifying);
  if (ImGui::SmallButton("Verify again")) models_.VerifyInstalled();
  ImGui::EndDisabled();
  ImGui::Dummy({0, Ui(8)});
  if (install.verifying) {
    ImGui::TextWrapped("Verifying SHA-256: %s / %s · %s",
        FormatBytes(install.verification_bytes).c_str(),
        FormatBytes(install.verification_total_bytes).c_str(), install.current_file.c_str());
    if (ImGui::SmallButton("Cancel verification")) models_.Cancel();
  } else if (!install.verification_status.empty()) {
    ImGui::TextWrapped("%s", install.verification_status.c_str());
  }
  const auto draw_profile = [this, &install](const ModelProfileCatalog& catalog) {
    ImGui::BeginDisabled(install.verifying);
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
    BeginModelCard(id.c_str(), g_ui_scale);
    const ImVec2 gem_origin = ImGui::GetCursorScreenPos();
    DrawGemstone(ImGui::GetWindowDrawList(),
                 {gem_origin.x + Ui(19.0f), gem_origin.y + Ui(21.0f)}, Ui(17.0f));
    ImGui::Dummy({Ui(42.0f), Ui(42.0f)});
    ImGui::SameLine();
    ImGui::BeginGroup();
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x);
    ImGui::SetWindowFontScale(1.08f);
    ImGui::TextWrapped("%s", ProfileLabel(profile));
    ImGui::SetWindowFontScale(1.0f);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", catalog.description);
    ImGui::PushStyleColor(ImGuiCol_Text, kAccent);
    ImGui::TextWrapped("%s", catalog.capabilities);
    ImGui::PopStyleColor();
    ImGui::PopTextWrapPos();
    ImGui::EndGroup();
    bool first_component = true;
    for (const auto& component : catalog.components) {
      const bool ready = profile_state.ComponentReady(component.kind);
      const std::string status = std::string(ComponentKindLabel(component.kind)) +
          ": " + (ready ? "Verified" : "Missing") +
          (component.required ? "" : " (optional)");
      const float component_width = ImGui::CalcTextSize(status.c_str()).x +
          (ready ? ImGui::GetStyle().ItemSpacing.x +
                       ImGui::CalcTextSize("Remove").x +
                       ImGui::GetStyle().FramePadding.x * 2.0f : 0.0f);
      if (!first_component) ModelComponentSameLine(component_width, Ui(22.0f));
      first_component = false;
      ImGui::BeginGroup();
      ImGui::TextUnformatted(status.c_str());
      if (ready) {
        bool component_in_use = false;
        if (settings_.onboarding_complete) {
          for (const auto& active_component :
               CatalogForProfile(settings_.server.profile).components) {
            component_in_use |=
                active_component.catalog == component.catalog;
          }
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(component_in_use || install.downloading);
        const std::string remove_id =
            "Remove##" + std::string(ProfileWireName(profile)) + "-" +
            std::to_string(ModelComponentKindIndex(component.kind));
        if (ImGui::SmallButton(remove_id.c_str()))
          models_.RemoveComponent(profile, component.kind);
        ImGui::EndDisabled();
      }
      ImGui::EndGroup();
    }
    if (downloading) {
      const float progress = profile_state.total_bytes == 0 ? 0.0f :
          static_cast<float>(static_cast<double>(profile_state.completed_bytes) /
                             static_cast<double>(profile_state.total_bytes));
      const float pause_width = ImGui::CalcTextSize("Pause").x +
                                ImGui::GetStyle().FramePadding.x * 2.0f;
      const std::string progress_label = FormatBytes(profile_state.completed_bytes) +
                                         " / " + FormatBytes(profile_state.total_bytes);
      ModelDownloadProgress(progress,
          {std::max(Ui(40.0f), ImGui::GetContentRegionAvail().x - pause_width -
                                   ImGui::GetStyle().ItemSpacing.x), ImGui::GetFrameHeight()},
          progress_label.c_str(), ImGui::GetTime());
      ImGui::SameLine();
      if (ImGui::Button((std::string("Pause##") + ProfileWireName(profile)).c_str())) {
        models_.Cancel();
      }
      if (!install.current_file.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        ImGui::TextWrapped("%s", install.current_file.c_str());
        ImGui::PopStyleColor();
      }
    } else if (!profile_state.Ready()) {
      const bool blocked = install.downloading || !profile_state.storage_available ||
                           !profile_state.sufficient_space;
      ImGui::BeginDisabled(blocked);
      const std::string button = "Install " + FormatBytes(profile_state.required_download_bytes) +
                                 "##" + ProfileWireName(profile);
      if (ImGui::Button(button.c_str())) models_.DownloadProfile(profile);
      ImGui::EndDisabled();
      ModelComponentSameLine(Ui(310.0f), ImGui::GetStyle().ItemSpacing.x);
      ImGui::PushTextWrapPos(0.0f);
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
      ImGui::PopTextWrapPos();
    } else if (selected) {
      ImGui::TextColored(kAccent, "Installed and selected");
    } else {
      const std::string button = "Use this profile##" +
                                 std::string(ProfileWireName(profile));
      if (ImGui::Button(button.c_str())) SelectProfile(profile);
      ModelComponentSameLine(ImGui::CalcTextSize("Installed in the shared Hub cache").x,
                             ImGui::GetStyle().ItemSpacing.x);
      ImGui::TextDisabled("Installed in the shared Hub cache");
    }
    EndModelCard();
    ImGui::PopStyleColor(2);
    ImGui::Dummy({0, Ui(3)});
    ImGui::EndDisabled();
  };
  for (const ModelProfile profile : PublicModelProfiles()) {
    draw_profile(CatalogForProfile(profile));
  }
  if (!install.error.empty()) {
    ImGui::TextColored({1.0f, 0.45f, 0.45f, 1.0f}, "%s", install.error.c_str());
  }
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
  ImGui::TextColored(settings_.onboarding_complete
                         ? kAccent
                         : ImVec4(1.0f, 0.65f, 0.32f, 1.0f),
                     "%s", settings_.onboarding_complete
                               ? ProfileLabel(settings_.server.profile)
                               : "None selected");
  if (!settings_.onboarding_complete) {
    ImGui::TextWrapped(
        "Select a qualified profile in Models before starting the server.");
  }
  ImGui::Dummy({0, Ui(5)});
  PathField("Server executable", "##server-executable", "Browse##server", executable_, false);
  PathField("Compiled target model", "##target-model", "Browse##target", model_directory_, true);
  PathField("Compiled MTP assistant", "##mtp-assistant", "Browse##assistant", assistant_directory_, true);
  const bool vision_profile = settings_.server.profile ==
      ModelProfile::kGemma4Moe26BTrellis35VisionFp8;
  if (vision_profile) {
    PathField("Compiled Vision module", "##vision-model", "Browse##vision",
              vision_directory_, true);
  }
  TextField("Served model name", "##served-name", model_name_);

  if (ImGui::BeginTable("##network-fields", 2,
                        ImGuiTableFlags_SizingStretchProp)) {
    ImGui::TableSetupColumn("Host", ImGuiTableColumnFlags_WidthStretch, 1.35f);
    ImGui::TableSetupColumn("Port", ImGuiTableColumnFlags_WidthStretch, 1.0f);
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::TextColored({0.72f, 0.77f, 0.75f, 1.0f}, "Server host");
    ImGui::TableSetColumnIndex(1);
    ImGui::TextColored({0.72f, 0.77f, 0.75f, 1.0f}, "Port");
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::SetNextItemWidth(-1.0f);
    ImGui::InputText("##server-host", host_.data(), host_.size());
    ImGui::TableSetColumnIndex(1);
    ImGui::SetNextItemWidth(-1.0f);
    ImGui::InputInt("##server-port", &settings_.server.port);
    ImGui::EndTable();
  }
  int context = static_cast<int>(settings_.server.max_context_tokens);
  FieldLabel("Context tokens");
  if (ImGui::InputInt("##context-tokens", &context))
    settings_.server.max_context_tokens = std::max(context, 1);
  const int generic_mtp_values[] = {0, 1, 2, 4};
  const char* generic_mtp_labels[] = {"Off", "D1", "D2", "D4"};
  const int moe_mtp_values[] = {0, 2};
  const char* moe_mtp_labels[] = {"Off", "D2 (fixed)"};
  const bool moe_profile = settings_.server.profile !=
                           ModelProfile::kGemma4Unified12B;
  const int* mtp_values = moe_profile ? moe_mtp_values : generic_mtp_values;
  const char* const* mtp_labels =
      moe_profile ? moe_mtp_labels : generic_mtp_labels;
  const int mtp_count = moe_profile ? 2 : 4;
  int mtp_index = settings_.server.mtp_draft_tokens == 0 ? 0
                                                        : (moe_profile ? 1 : 2);
  for (int index = 0; index < mtp_count; ++index) {
    if (settings_.server.mtp_draft_tokens == mtp_values[index])
      mtp_index = index;
  }
  FieldLabel("MTP draft profile");
  if (ImGui::Combo("##mtp-profile", &mtp_index, mtp_labels, mtp_count)) {
    settings_.server.mtp_draft_tokens = mtp_values[mtp_index];
  }
  if (vision_profile) {
    const int budget_values[] = {70, 140, 280};
    const char* budget_labels[] = {"Fast · 70 soft tokens",
                                   "Balanced · 140 soft tokens",
                                   "Maximum detail · 280 soft tokens"};
    int budget_index = settings_.server.vision_soft_token_budget == 70
                           ? 0
                           : settings_.server.vision_soft_token_budget == 140
                                 ? 1
                                 : 2;
    FieldLabel("Vision startup capacity and processing budget");
    if (ImGui::Combo("##vision-budget", &budget_index, budget_labels, 3))
      settings_.server.vision_soft_token_budget = budget_values[budget_index];
  }
  ImGui::Checkbox("Greedy sampling", &settings_.server.greedy);
  ImGui::SameLine(0, Ui(18));
  CapabilityChip(settings_.server.mtp_draft_tokens == 0 ? "MTP disabled" : "GPU MTP enabled",
                 settings_.server.mtp_draft_tokens != 0);
  if (vision_profile) {
    const HealthSnapshot health = server_.Health();
    ImGui::SameLine(0, Ui(10));
    const bool ordinary = settings_.server.mtp_draft_tokens == 0;
    const bool d2_live = health.available && health.vision_mtp_supported;
    CapabilityChip(ordinary ? "Vision Ordinary selected"
                            : d2_live ? "Vision+D2 live-qualified"
                                      : "Vision+D2 awaiting live capability",
                   ordinary || d2_live);
  }
  SyncSettingsFromBuffers();
  server_.Configure(settings_.server);
  ImGui::Dummy({0, Ui(6)});
  const bool executable_ready = std::filesystem::is_regular_file(settings_.server.executable);
  const bool target_ready = std::filesystem::is_directory(settings_.server.model_directory);
  const bool assistant_ready = settings_.server.mtp_draft_tokens == 0 ||
      std::filesystem::is_directory(settings_.server.assistant_directory);
  const bool vision_ready = !vision_profile ||
      std::filesystem::is_directory(settings_.server.vision_directory);
  const bool preflight_ready = executable_ready && target_ready && assistant_ready &&
      vision_ready &&
      settings_.server.port > 0 && settings_.server.port <= 65535;
  const bool can_start = settings_.onboarding_complete && preflight_ready;
  ImGui::TextColored(preflight_ready ? kAccent : ImVec4(1.0f, 0.48f, 0.36f, 1.0f),
                     "%s  Executable · %s  Target · %s  Assistant",
                     executable_ready ? "Ready" : "Missing",
                     target_ready ? "Ready" : "Missing",
                     assistant_ready ? "Ready" : "Missing");
  if (vision_profile) {
    ImGui::SameLine();
    ImGui::Text(" · %s Vision", vision_ready ? "Ready" : "Missing");
  }
  const ServerPhase phase = server_.Phase();
  if (phase == ServerPhase::kRunning || phase == ServerPhase::kStarting || phase == ServerPhase::kStopping) {
    ImGui::BeginDisabled(phase == ServerPhase::kStopping);
    if (ImGui::Button("Stop server", {Ui(140), Ui(42)})) server_.Stop();
    ImGui::EndDisabled();
  } else {
    ImGui::BeginDisabled(!can_start);
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
    if (health.supports_vision || health.vision_module_loaded) {
      ImGui::Text("Vision: %s · Vision+D2: %s",
                  health.vision_module_loaded ? "loaded" : "unavailable",
                  health.vision_mtp_supported ? "qualified" : "disabled");
      ImGui::TextDisabled("Profile: %s · %s", health.profile_id.c_str(),
                          health.qualification_state.c_str());
      ImGui::TextDisabled("Decode: %s · Max image budget: %d soft tokens",
                          health.decode_mode.c_str(),
                          health.vision_max_soft_token_budget);
      if (health.last_vision_soft_token_budget > 0) {
        ImGui::Text("Last image processing budget: %d soft tokens",
                    health.last_vision_soft_token_budget);
      }
    }
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
  if (ImGui::Combo("##ui-scale", &scale_index, scale_labels, 4)) {
    settings_.ui_scale = scale_values[scale_index];
    ApplyUiScale(settings_.ui_scale);
  }
  ImGui::TextDisabled("Current %.0f%% · applies immediately", ui_scale_ * 100.0f);
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
  const char* effort_labels[] = {"none (0 tokens)", "low (1,024 tokens)",
                                 "medium (4,096 tokens)", "high (8,192 tokens)"};
  const char* effort_keys[] = {"none", "low", "medium", "high"};
  int current = 2;
  for (int index = 0; index < 4; ++index)
    if (settings_.generation.reasoning_effort == effort_keys[index]) current = index;
  FieldLabel("Thinking effort (reasoning token budget)");
  if (ImGui::Combo("##thinking-effort", &current, effort_labels, 4))
    settings_.generation.reasoning_effort = effort_keys[current];
  int output_tokens = static_cast<int>(settings_.generation.max_output_tokens);
  FieldLabel("Maximum output tokens (e.g. 32768, bounded by context window)");
  if (ImGui::InputInt("##maximum-output-tokens", &output_tokens, 1024, 4096))
    settings_.generation.max_output_tokens =
        std::clamp<std::int64_t>(output_tokens, 1, 262144);
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
  const HealthSnapshot health = server_.Health();
  const std::string mismatch =
      HealthCompatibilityError(settings_.server, health);
  if (!mismatch.empty()) {
    attachment_error_ = mismatch;
    return;
  }
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
  performance_.reset();
  api_.StreamChat(settings_.server, settings_.generation, request_messages, session_id_);
  auto_follow_ = true;
  scroll_to_bottom_ = true;
}

void StudioApp::RetryLastRequest() {
  if (api_.Busy() || messages_.size() < 2U ||
      messages_.back().role != "assistant" || !messages_.back().error ||
      messages_[messages_.size() - 2U].role != "user") {
    return;
  }
  const HealthSnapshot health = server_.Health();
  const std::string mismatch =
      HealthCompatibilityError(settings_.server, health);
  if (!mismatch.empty()) {
    attachment_error_ = mismatch;
    return;
  }
  messages_.pop_back();
  const auto request_messages = messages_;
  messages_.push_back({"assistant", {}, {}, true, false, {}});
  session_id_.clear();
  generation_started_ = std::chrono::steady_clock::now();
  first_token_at_ = {};
  generation_finished_ = {};
  prompt_tokens_ = 0;
  completion_tokens_ = 0;
  streamed_chunks_ = 0;
  performance_.reset();
  api_.StreamChat(settings_.server, settings_.generation, request_messages,
                  session_id_);
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
    const std::string policy_error = AttachmentPolicyError(
        settings_.server.profile, attachment.kind,
        CountImages(messages_, pending_attachments_));
    if (!policy_error.empty()) {
      attachment_error_ = policy_error;
      continue;
    }
    pending_attachments_.push_back(std::move(attachment));
  }
}

void StudioApp::FinishRecording() {
  if (!recorder_.Active()) return;
  MediaAttachment recording;
  if (recorder_.Stop(recording, attachment_error_))
    pending_attachments_.push_back(std::move(recording));
}

void StudioApp::ClearChat() {
  if (api_.Busy()) return;
  if (recorder_.Active()) {
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
  performance_.reset();
}

void StudioApp::RemoveLastExchange() {
  if (api_.Busy() || messages_.empty()) return;
  if (!gem16::studio::RemoveLastExchange(messages_)) return;
  session_id_.clear();
  expanded_reasoning_.clear();
}

void StudioApp::SelectProfile(ModelProfile profile) {
  if (api_.Busy()) {
    pending_profile_ = profile;
    api_.Cancel();
    return;
  }
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
  CopyTo(vision_directory_, settings_.server.vision_directory);
  CopyTo(model_name_, settings_.server.model_name);
  CopyTo(host_, settings_.server.host);
  CopyTo(system_prompt_, settings_.generation.system_prompt);
}

void StudioApp::SyncSettingsFromBuffers() {
  settings_.server.executable = executable_.data();
  settings_.server.model_directory = model_directory_.data();
  settings_.server.assistant_directory = assistant_directory_.data();
  settings_.server.vision_directory = vision_directory_.data();
  settings_.server.model_name = model_name_.data();
  settings_.server.host = host_.data();
  settings_.generation.system_prompt = system_prompt_.data();
}

}  // namespace gem16::studio

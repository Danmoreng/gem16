#pragma once

#include "imgui.h"

#include <algorithm>
#include <cmath>

namespace gem16::studio {

inline void BeginModelCard(const char* id, float scale) {
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {14.0f * scale, 12.0f * scale});
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {12.0f * scale, 7.0f * scale});
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {12.0f * scale, 6.0f * scale});
  ImGui::BeginChild(id, {0, 0},
                    ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY |
                        ImGuiChildFlags_AlwaysAutoResize,
                    ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
}

inline void EndModelCard() {
  ImGui::EndChild();
  ImGui::PopStyleVar(3);
}

// Move the next component onto the same row only if its complete status/action
// group fits. Never squeeze a Remove button outside the card's content area.
inline void ModelComponentSameLine(float width, float gap) {
  const float next_x = ImGui::GetItemRectMax().x + gap;
  const float right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
  if (next_x + width <= right) ImGui::SameLine(0, gap);
}

inline void ModelDownloadProgress(float fraction, ImVec2 size,
                                  const char* label, double time) {
  fraction = std::isfinite(fraction) ? std::clamp(fraction, 0.0f, 1.0f) : 0.0f;
  ImGui::PushStyleColor(ImGuiCol_FrameBg, {0.045f, 0.095f, 0.075f, 1.0f});
  ImGui::PushStyleColor(ImGuiCol_PlotHistogram, {0.08f, 0.48f, 0.31f, 1.0f});
  ImGui::ProgressBar(fraction, size, "");
  ImGui::PopStyleColor(2);
  const ImVec2 min = ImGui::GetItemRectMin();
  const ImVec2 max = ImGui::GetItemRectMax();
  ImDrawList* draw = ImGui::GetWindowDrawList();
  const float inset = std::max(1.0f, (max.y - min.y) * 0.06f);
  const float left = min.x + inset;
  const float right = max.x - inset;
  const float top = min.y + inset;
  const float bottom = max.y - inset;
  const float fill_right = left + (right - left) * fraction;
  const float radius = std::min(ImGui::GetStyle().FrameRounding,
                                (bottom - top) * 0.5f);
  // Soft travelling emerald waves stay inside the actual completed extent.
  // Drawing through ImGui keeps the animation identical on DX11 and OpenGL.
  const float phase = static_cast<float>(std::fmod(time, 60.0)) * 1.7f;
  draw->PushClipRect(min, max, true);
  for (float x = left + radius; x < fill_right - radius; x += 3.0f) {
    const float wave = 0.5f + 0.5f * std::sin((x - left) * 0.025f - phase);
    const int alpha = static_cast<int>(28.0f + 65.0f * wave * wave);
    draw->AddRectFilledMultiColor(
        {x, top}, {std::min(x + 3.0f, fill_right - radius), bottom},
        IM_COL32(100, 255, 198, alpha), IM_COL32(100, 255, 198, alpha),
        IM_COL32(20, 174, 108, alpha / 3), IM_COL32(20, 174, 108, alpha / 3));
  }
  const ImVec2 text = ImGui::CalcTextSize(label);
  const ImVec2 position{min.x + (max.x - min.x - text.x) * 0.5f,
                        min.y + (max.y - min.y - text.y) * 0.5f};
  draw->AddText({position.x + 1.0f, position.y + 1.0f}, IM_COL32(0, 24, 15, 230), label);
  draw->AddText(position, IM_COL32(226, 255, 242, 255), label);
  draw->PopClipRect();
}

}  // namespace gem16::studio

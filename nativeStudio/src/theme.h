#pragma once

#include "imgui.h"

namespace gem16::studio {

constexpr ImVec4 ThemeRgb(unsigned rgb, float alpha = 1.0f) {
  return {static_cast<float>((rgb >> 16) & 255) / 255.0f,
          static_cast<float>((rgb >> 8) & 255) / 255.0f,
          static_cast<float>(rgb & 255) / 255.0f, alpha};
}

// Native Studio owns one ImGui style. Custom drawing must use the same theme
// as the widgets, rather than keeping dark surfaces after StyleColorsLight.
inline bool g_studio_dark_theme = true;
inline ImVec4 ThemeColor(ImVec4 dark, ImVec4 light) {
  return g_studio_dark_theme ? dark : light;
}

struct StudioPalette {
  ImVec4 accent, accent_dim, field_text, muted, root_background;
  ImVec4 user_background, user_border, assistant_background, assistant_border;
  ImVec4 subtle_background, code_background, code_border, code_text;
  ImVec4 reasoning_text, inline_code_background, selection;
  ImVec4 warning, error, error_background;
};

inline const StudioPalette& StudioColors() {
  static const StudioPalette dark{
      {0.31f, 0.91f, 0.65f, 1}, {0.11f, 0.34f, 0.24f, 1},
      {0.72f, 0.77f, 0.75f, 1}, {0.62f, 0.68f, 0.66f, 1},
      {0.02f, 0.03f, 0.03f, 0.24f},
      {0.035f, 0.25f, 0.17f, 0.93f}, {0.10f, 0.55f, 0.38f, 0.75f},
      {0.060f, 0.078f, 0.074f, 0.92f}, {0.19f, 0.26f, 0.24f, 0.86f},
      {0.04f, 0.11f, 0.09f, 0.88f}, {0.025f, 0.045f, 0.041f, 0.96f},
      {0.12f, 0.31f, 0.25f, 0.92f}, {0.82f, 0.89f, 0.86f, 1},
      {0.55f, 0.62f, 0.60f, 1}, ThemeRgb(0x1b302a, 235.0f / 255.0f),
      ThemeRgb(0x269066, 205.0f / 255.0f),
      {1.0f, 0.76f, 0.30f, 1}, {1.0f, 0.48f, 0.36f, 1},
      {0.27f, 0.07f, 0.07f, 0.92f}};
  // The former Compose GemLight palette: neutral white/gray surfaces, dark
  // text, forest-green accents and pale mint containers.
  static const StudioPalette light{
      ThemeRgb(0x126c3d), ThemeRgb(0xb9f6ca), ThemeRgb(0x404040),
      ThemeRgb(0x575757), ThemeRgb(0xf3f3f3, 0.08f),
      ThemeRgb(0xb9f6ca), ThemeRgb(0x85be97),
      ThemeRgb(0xeaeaea), ThemeRgb(0xd0d0d0),
      ThemeRgb(0xe2e2e2), ThemeRgb(0xf7f7f7), ThemeRgb(0xd0d0d0),
      ThemeRgb(0x202020), ThemeRgb(0x575757), ThemeRgb(0xddebe2),
      ThemeRgb(0x94d7ad, 0.60f), ThemeRgb(0x805400), ThemeRgb(0xa12622),
      ThemeRgb(0xffe9e6)};
  return g_studio_dark_theme ? dark : light;
}

inline void ApplyStudioThemeColors(bool dark) {
  g_studio_dark_theme = dark;
  if (dark) ImGui::StyleColorsDark();
  else ImGui::StyleColorsLight();
  if (dark) return;
  auto* colors = ImGui::GetStyle().Colors;
  colors[ImGuiCol_Text] = ThemeRgb(0x202020);
  colors[ImGuiCol_TextDisabled] = StudioColors().muted;
  colors[ImGuiCol_WindowBg] = StudioColors().root_background;
  colors[ImGuiCol_ChildBg] = ThemeRgb(0xffffff);
  colors[ImGuiCol_PopupBg] = ThemeRgb(0xffffff);
  colors[ImGuiCol_Border] = ThemeRgb(0xd0d0d0);
  colors[ImGuiCol_BorderShadow] = {0, 0, 0, 0};
  colors[ImGuiCol_FrameBg] = ThemeRgb(0xf7f7f7);
  colors[ImGuiCol_FrameBgHovered] = ThemeRgb(0xe5eee8);
  colors[ImGuiCol_FrameBgActive] = ThemeRgb(0xddebe2);
  colors[ImGuiCol_Button] = ThemeRgb(0xb9f6ca);
  colors[ImGuiCol_ButtonHovered] = ThemeRgb(0xa6e6b8);
  colors[ImGuiCol_ButtonActive] = ThemeRgb(0x94d7ad);
  colors[ImGuiCol_Header] = ThemeRgb(0xd5efdf);
  colors[ImGuiCol_HeaderHovered] = ThemeRgb(0xc5e8d2);
  colors[ImGuiCol_HeaderActive] = ThemeRgb(0xb9f6ca);
  colors[ImGuiCol_CheckMark] = StudioColors().accent;
  colors[ImGuiCol_SliderGrab] = StudioColors().accent;
  colors[ImGuiCol_SliderGrabActive] = ThemeRgb(0x0b542e);
  colors[ImGuiCol_ScrollbarBg] = ThemeRgb(0xf1f1f1);
  colors[ImGuiCol_ScrollbarGrab] = ThemeRgb(0xb5bdb8);
  colors[ImGuiCol_ScrollbarGrabHovered] = ThemeRgb(0x929e97);
  colors[ImGuiCol_ScrollbarGrabActive] = ThemeRgb(0x788a7f);
  colors[ImGuiCol_Separator] = ThemeRgb(0xd0d0d0);
  colors[ImGuiCol_SeparatorHovered] = StudioColors().accent;
  colors[ImGuiCol_SeparatorActive] = StudioColors().accent;
  colors[ImGuiCol_TextSelectedBg] = StudioColors().selection;
  colors[ImGuiCol_TextLink] = StudioColors().accent;
  colors[ImGuiCol_NavCursor] = StudioColors().accent;
  colors[ImGuiCol_PlotHistogram] = ThemeRgb(0x85be97);
  colors[ImGuiCol_TableHeaderBg] = ThemeRgb(0xe2e2e2);
  colors[ImGuiCol_TableBorderStrong] = ThemeRgb(0xb5bdb8);
  colors[ImGuiCol_TableBorderLight] = ThemeRgb(0xd0d0d0);
  colors[ImGuiCol_TableRowBgAlt] = ThemeRgb(0xf1f1f1, 0.65f);
}

}  // namespace gem16::studio

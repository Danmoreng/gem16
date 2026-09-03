#include "fonts.h"
#include "math_renderer.h"

#include <filesystem>
#include <cstring>
#include "imgui.h"

namespace gem16::studio {

ImFont* InitializeStudioFonts() {
  static_assert(sizeof(ImWchar) == 4, "Studio requires supplementary Unicode glyphs");
  ImFontAtlas* atlas = ImGui::GetIO().Fonts;
#ifdef _WIN32
  const char* text_paths[] = {"C:/Windows/Fonts/segoeui.ttf"};
  const char* emoji_paths[] = {"C:/Windows/Fonts/seguiemj.ttf"};
  const char* code_paths[] = {"C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/cour.ttf"};
#else
  const char* text_paths[] = {"/usr/share/fonts/noto/NotoSans-Regular.ttf",
      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"};
  const char* code_paths[] = {"/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
      "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
      "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf"};
  // stb rasterizes outline fonts, not the bitmap-only NotoColorEmoji font.
  const char* emoji_paths[] = {"/usr/share/fonts/truetype/noto/NotoEmoji-Regular.ttf",
      "/usr/share/fonts/noto/NotoEmoji-Regular.ttf",
      "/usr/share/fonts/truetype/ancient-scripts/Symbola_hint.ttf"};
#endif
  ImFont* font = nullptr;
  static constexpr ImWchar text_ranges[] = {0x20, 0x024f, 0x2000, 0x206f, 0};
  for (const char* path : text_paths) {
    if (std::filesystem::is_regular_file(path)) {
      font = atlas->AddFontFromFileTTF(path, 17.0f, nullptr, text_ranges);
      if (font) break;
    }
  }
  if (!font) font = atlas->AddFontDefault();
  // Ranges are also used by the legacy atlas build path in headless tests.
  static constexpr ImWchar emoji_ranges[] = {
      0x200d, 0x200d, 0x2300, 0x27ff, 0xfe0e, 0xfe0f,
      0x1f000, 0x1faff, 0};
  for (const char* path : emoji_paths) {
    if (!std::filesystem::is_regular_file(path)) continue;
    ImFontConfig config;
    config.MergeMode = true;
    if (atlas->AddFontFromFileTTF(path, 17.0f, &config, emoji_ranges)) break;
  }
  ImFontConfig code_config;
  // Consolas has visibly taller glyphs than the UI face at the same pixel
  // height. Scale at rasterization so drawing, advances, wrapping and hit
  // testing all agree, without applying the application's DPI factor twice.
  code_config.ExtraSizeScale = 0.80f;
  constexpr char code_name[] = "Studio monospace";
  static_assert(sizeof(code_name) <= sizeof(code_config.Name));
  std::memcpy(code_config.Name, code_name, sizeof(code_name));
  ImFont* code_font = nullptr;
  for (const char* path : code_paths) {
    if (std::filesystem::is_regular_file(path))
      code_font = atlas->AddFontFromFileTTF(path, 17.0f, &code_config, text_ranges);
    if (code_font) break;
  }
  if (!code_font) atlas->AddFontDefaultBitmap(&code_config); // Embedded monospaced ProggyClean.
  InitializeMathFonts();
  return font;
}

ImFont* StudioCodeFont() {
  for (ImFont* font : ImGui::GetIO().Fonts->Fonts)
    if (std::strcmp(font->GetDebugName(), "Studio monospace") == 0) return font;
  return ImGui::GetFont();
}

}  // namespace gem16::studio

#include "fonts.h"

#include <filesystem>
#include "imgui.h"

namespace gem16::studio {

ImFont* InitializeStudioFonts() {
  static_assert(sizeof(ImWchar) == 4, "Studio requires supplementary Unicode glyphs");
  ImFontAtlas* atlas = ImGui::GetIO().Fonts;
#ifdef _WIN32
  const char* text_paths[] = {"C:/Windows/Fonts/segoeui.ttf"};
  const char* emoji_paths[] = {"C:/Windows/Fonts/seguiemj.ttf"};
#else
  const char* text_paths[] = {"/usr/share/fonts/noto/NotoSans-Regular.ttf",
      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"};
  // stb rasterizes outline fonts, not the bitmap-only NotoColorEmoji font.
  const char* emoji_paths[] = {"/usr/share/fonts/truetype/noto/NotoEmoji-Regular.ttf",
      "/usr/share/fonts/noto/NotoEmoji-Regular.ttf",
      "/usr/share/fonts/truetype/ancient-scripts/Symbola_hint.ttf"};
#endif
  ImFont* font = nullptr;
  for (const char* path : text_paths) {
    if (std::filesystem::is_regular_file(path)) {
      font = atlas->AddFontFromFileTTF(path, 17.0f);
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
  return font;
}

}  // namespace gem16::studio

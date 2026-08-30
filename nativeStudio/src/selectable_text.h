#pragma once

#include "imgui.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gem16::studio::selectable_text {

struct StyleSpan {
  std::size_t begin = 0;
  std::size_t end = 0;
  ImU32 text_color = 0;
  ImU32 background_color = 0;
  bool strong = false;
  bool emphasis = false;
  bool underline = false;
};

struct Options {
  float width = 0.0f;
  ImU32 text_color = IM_COL32_WHITE;
  ImU32 selection_color = IM_COL32(37, 132, 96, 190);
  float line_spacing = 3.0f;
  const std::vector<StyleSpan>* spans = nullptr;
  // Widgets that share a group use one selection over selection_text. The
  // offset maps this widget's local text into that shared string.
  ImGuiID selection_group = 0;
  std::string_view selection_text{};
  std::size_t selection_offset = 0;
};

// Draws wrapped text with native desktop-style character selection. Selection
// is local to this widget and supports drag, double-click, Shift+click,
// keyboard extension, Ctrl+A/C, and a right-click copy menu.
void Wrapped(const char* id, const std::string& text, const Options& options = {});

[[nodiscard]] std::pair<std::size_t, std::size_t> NormalizedRange(
    std::size_t anchor, std::size_t caret, std::size_t text_size);
[[nodiscard]] std::pair<std::size_t, std::size_t> WordRange(
    std::string_view text, std::size_t position);
[[nodiscard]] std::string SelectedText(std::string_view text,
                                       std::size_t anchor,
                                       std::size_t caret);

}  // namespace gem16::studio::selectable_text

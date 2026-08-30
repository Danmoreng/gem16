#include "selectable_text.h"

#include "imgui_internal.h"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <vector>

namespace gem16::studio::selectable_text {
namespace {

struct WrappedLine {
  std::size_t begin = 0;
  std::size_t end = 0;
};

struct SelectionState {
  ImGuiID id = 0;
  std::size_t anchor = 0;
  std::size_t caret = 0;
  bool dragging = false;
  int keyboard_frame = -1;
};

struct ResolvedStyle {
  ImU32 text_color = 0;
  ImU32 background_color = 0;
  bool strong = false;
  bool emphasis = false;
  bool underline = false;

  bool operator==(const ResolvedStyle&) const = default;
};

SelectionState g_selection;

std::size_t NextCodepoint(std::string_view text, std::size_t position) {
  if (position >= text.size()) return text.size();
  const unsigned char first = static_cast<unsigned char>(text[position]);
  std::size_t count = 1;
  if ((first & 0xE0U) == 0xC0U) count = 2;
  else if ((first & 0xF0U) == 0xE0U) count = 3;
  else if ((first & 0xF8U) == 0xF0U) count = 4;
  if (position + count > text.size()) return position + 1;
  for (std::size_t index = position + 1; index < position + count; ++index) {
    if ((static_cast<unsigned char>(text[index]) & 0xC0U) != 0x80U) {
      return position + 1;
    }
  }
  return position + count;
}

std::size_t PreviousCodepoint(std::string_view text, std::size_t position) {
  position = std::min(position, text.size());
  if (position == 0) return 0;
  --position;
  while (position > 0 &&
         (static_cast<unsigned char>(text[position]) & 0xC0U) == 0x80U) {
    --position;
  }
  return position;
}

bool IsBreakByte(char value) {
  return value == ' ' || value == '\t';
}

bool IsWordByte(unsigned char value) {
  return value >= 0x80U || std::isalnum(value) != 0 || value == '_';
}

float Measure(ImFont* font, float size, std::string_view text,
              std::size_t begin, std::size_t end) {
  if (end <= begin) return 0.0f;
  return font->CalcTextSizeA(size, FLT_MAX, 0.0f, text.data() + begin,
                             text.data() + end).x;
}

std::vector<WrappedLine> BuildLines(std::string_view text, ImFont* font,
                                    float font_size, float width) {
  std::vector<WrappedLine> lines;
  if (text.empty()) {
    lines.push_back({0, 0});
    return lines;
  }
  width = std::max(width, font_size * 2.0f);
  std::size_t line_begin = 0;
  std::size_t position = 0;
  std::size_t last_break = std::string_view::npos;
  float line_width = 0.0f;
  while (position < text.size()) {
    if (text[position] == '\n') {
      std::size_t end = position;
      if (end > line_begin && text[end - 1] == '\r') --end;
      lines.push_back({line_begin, end});
      position += 1;
      line_begin = position;
      last_break = std::string_view::npos;
      line_width = 0.0f;
      continue;
    }
    const std::size_t next = NextCodepoint(text, position);
    const float glyph_width = Measure(font, font_size, text, position, next);
    if (IsBreakByte(text[position])) last_break = next;
    if (line_width + glyph_width > width && position > line_begin) {
      std::size_t line_end = position;
      std::size_t next_begin = position;
      if (last_break != std::string_view::npos && last_break > line_begin) {
        line_end = last_break;
        while (line_end > line_begin && IsBreakByte(text[line_end - 1])) --line_end;
        next_begin = last_break;
        while (next_begin < text.size() && IsBreakByte(text[next_begin])) ++next_begin;
      }
      lines.push_back({line_begin, line_end});
      line_begin = next_begin;
      position = next_begin;
      last_break = std::string_view::npos;
      line_width = 0.0f;
      continue;
    }
    line_width += glyph_width;
    position = next;
  }
  lines.push_back({line_begin, text.size()});
  return lines;
}

std::size_t ByteAtX(std::string_view text, const WrappedLine& line,
                    ImFont* font, float font_size, float local_x) {
  if (local_x <= 0.0f || line.begin == line.end) return line.begin;
  float x = 0.0f;
  for (std::size_t position = line.begin; position < line.end;) {
    const std::size_t next = NextCodepoint(text, position);
    const float width = Measure(font, font_size, text, position, next);
    if (local_x < x + width * 0.5f) return position;
    x += width;
    if (local_x < x) return next;
    position = next;
  }
  return line.end;
}

std::size_t ByteAtPoint(std::string_view text,
                        const std::vector<WrappedLine>& lines, ImFont* font,
                        float font_size, float line_height,
                        const ImVec2& origin, const ImVec2& point) {
  if (lines.empty()) return 0;
  const float relative_y = point.y - origin.y;
  const std::size_t line_index = relative_y <= 0.0f
                                     ? 0
                                     : std::min(
                                           static_cast<std::size_t>(relative_y / line_height),
                                           lines.size() - 1);
  return ByteAtX(text, lines[line_index], font, font_size,
                 point.x - origin.x);
}

float XAtByte(std::string_view text, const WrappedLine& line, ImFont* font,
              float font_size, std::size_t byte) {
  return Measure(font, font_size, text, line.begin,
                 std::clamp(byte, line.begin, line.end));
}

void CopySelection(std::string_view text) {
  const std::string selected =
      SelectedText(text, g_selection.anchor, g_selection.caret);
  if (!selected.empty()) ImGui::SetClipboardText(selected.c_str());
}

void MoveCaret(std::string_view text, bool right, bool extend) {
  const std::size_t target = right ? NextCodepoint(text, g_selection.caret)
                                   : PreviousCodepoint(text, g_selection.caret);
  g_selection.caret = target;
  if (!extend) g_selection.anchor = target;
}

ResolvedStyle ResolveStyle(const std::vector<StyleSpan>* spans,
                           std::size_t position, ImU32 default_color) {
  ResolvedStyle result;
  result.text_color = default_color;
  if (spans == nullptr) return result;
  for (const StyleSpan& span : *spans) {
    if (position < span.begin || position >= span.end) continue;
    if (span.text_color != 0) result.text_color = span.text_color;
    if (span.background_color != 0)
      result.background_color = span.background_color;
    result.strong = result.strong || span.strong;
    result.emphasis = result.emphasis || span.emphasis;
    result.underline = result.underline || span.underline;
  }
  return result;
}

struct StyledRun {
  std::size_t begin = 0;
  std::size_t end = 0;
  ResolvedStyle style;
};

std::vector<StyledRun> BuildStyledRuns(std::string_view text,
                                       const WrappedLine& line,
                                       const std::vector<StyleSpan>* spans,
                                       ImU32 default_color) {
  std::vector<StyledRun> runs;
  if (line.begin == line.end) return runs;
  std::size_t begin = line.begin;
  ResolvedStyle style = ResolveStyle(spans, begin, default_color);
  for (std::size_t position = NextCodepoint(text, begin);
       position < line.end; position = NextCodepoint(text, position)) {
    const ResolvedStyle next = ResolveStyle(spans, position, default_color);
    if (next == style) continue;
    runs.push_back({begin, position, style});
    begin = position;
    style = next;
  }
  runs.push_back({begin, line.end, style});
  return runs;
}

}  // namespace

std::pair<std::size_t, std::size_t> NormalizedRange(
    std::size_t anchor, std::size_t caret, std::size_t text_size) {
  anchor = std::min(anchor, text_size);
  caret = std::min(caret, text_size);
  return {std::min(anchor, caret), std::max(anchor, caret)};
}

std::pair<std::size_t, std::size_t> WordRange(std::string_view text,
                                              std::size_t position) {
  if (text.empty()) return {0, 0};
  position = std::min(position, text.size() - 1);
  while (position > 0 &&
         (static_cast<unsigned char>(text[position]) & 0xC0U) == 0x80U) {
    --position;
  }
  const bool word = IsWordByte(static_cast<unsigned char>(text[position]));
  std::size_t begin = position;
  while (begin > 0) {
    const std::size_t previous = PreviousCodepoint(text, begin);
    if (IsWordByte(static_cast<unsigned char>(text[previous])) != word) break;
    begin = previous;
  }
  std::size_t end = NextCodepoint(text, position);
  while (end < text.size() &&
         IsWordByte(static_cast<unsigned char>(text[end])) == word) {
    end = NextCodepoint(text, end);
  }
  return {begin, end};
}

std::string SelectedText(std::string_view text, std::size_t anchor,
                         std::size_t caret) {
  const auto [begin, end] = NormalizedRange(anchor, caret, text.size());
  return std::string(text.substr(begin, end - begin));
}

void Wrapped(const char* id, const std::string& text, const Options& options) {
  ImGuiWindow* window = ImGui::GetCurrentWindow();
  if (window->SkipItems) return;
  ImFont* font = ImGui::GetFont();
  const float font_size = ImGui::GetFontSize();
  const float width = options.width > 0.0f ? options.width
                                           : ImGui::GetContentRegionAvail().x;
  const float line_height = font_size + options.line_spacing;
  const std::vector<WrappedLine> lines =
      BuildLines(text, font, font_size, width);
  const float height = std::max(line_height, line_height * lines.size());
  const ImVec2 origin = ImGui::GetCursorScreenPos();
  ImGui::InvisibleButton(id, {width, height},
                         ImGuiButtonFlags_MouseButtonLeft |
                             ImGuiButtonFlags_MouseButtonRight);
  const ImGuiID item_id = ImGui::GetItemID();
  const bool grouped = options.selection_group != 0 &&
                       options.selection_offset <= options.selection_text.size() &&
                       text.size() <= options.selection_text.size() -
                                          options.selection_offset;
  const ImGuiID selection_id = grouped ? options.selection_group : item_id;
  const std::string_view selection_text =
      grouped ? options.selection_text : std::string_view(text);
  const std::size_t selection_offset = grouped ? options.selection_offset : 0U;
  const bool hovered = ImGui::IsItemHovered(
      ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);

  if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
    const std::size_t byte = selection_offset +
        ByteAtPoint(text, lines, font, font_size, line_height, origin,
                    ImGui::GetMousePos());
    g_selection.id = selection_id;
    if (ImGui::GetIO().KeyShift) {
      g_selection.caret = byte;
    } else {
      g_selection.anchor = byte;
      g_selection.caret = byte;
    }
    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
      const auto [begin, end] = WordRange(selection_text, byte);
      g_selection.anchor = begin;
      g_selection.caret = end;
    }
    g_selection.dragging = true;
  }
  if (g_selection.id == selection_id && g_selection.dragging && hovered &&
      ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
    g_selection.caret = selection_offset +
        ByteAtPoint(text, lines, font, font_size, line_height, origin,
                    ImGui::GetMousePos());
    const float mouse_y = ImGui::GetMousePos().y;
    if (mouse_y < window->InnerRect.Min.y + 18.0f)
      ImGui::SetScrollY(window, std::max(0.0f, window->Scroll.y - line_height));
    else if (mouse_y > window->InnerRect.Max.y - 18.0f)
      ImGui::SetScrollY(window, window->Scroll.y + line_height);
  }
  if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) g_selection.dragging = false;

  if (g_selection.id == selection_id) {
    g_selection.anchor = std::min(g_selection.anchor, selection_text.size());
    g_selection.caret = std::min(g_selection.caret, selection_text.size());
    const ImGuiIO& io = ImGui::GetIO();
    if (g_selection.keyboard_frame != ImGui::GetFrameCount()) {
      g_selection.keyboard_frame = ImGui::GetFrameCount();
      if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_A)) {
        g_selection.anchor = 0;
        g_selection.caret = selection_text.size();
      }
      if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_C))
        CopySelection(selection_text);
      if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow))
        MoveCaret(selection_text, false, io.KeyShift);
      if (ImGui::IsKeyPressed(ImGuiKey_RightArrow))
        MoveCaret(selection_text, true, io.KeyShift);
      if (ImGui::IsKeyPressed(ImGuiKey_Home)) {
        g_selection.caret = 0;
        if (!io.KeyShift) g_selection.anchor = 0;
      }
      if (ImGui::IsKeyPressed(ImGuiKey_End)) {
        g_selection.caret = selection_text.size();
        if (!io.KeyShift) g_selection.anchor = selection_text.size();
      }
    }
  }

  const auto [selection_begin, selection_end] =
      NormalizedRange(g_selection.anchor, g_selection.caret,
                      selection_text.size());
  ImDrawList* draw = ImGui::GetWindowDrawList();
  for (std::size_t index = 0; index < lines.size(); ++index) {
    const WrappedLine& line = lines[index];
    const ImVec2 position(origin.x, origin.y + line_height * index);
    const std::vector<StyledRun> runs =
        BuildStyledRuns(text, line, options.spans, options.text_color);
    for (const StyledRun& run : runs) {
      if (run.style.background_color == 0) continue;
      const float x1 = position.x + XAtByte(text, line, font, font_size,
                                            run.begin);
      const float x2 = position.x + XAtByte(text, line, font, font_size,
                                            run.end);
      draw->AddRectFilled({x1 - 2.0f, position.y - 1.0f},
                          {x2 + 2.0f, position.y + font_size + 2.0f},
                          run.style.background_color, 3.0f);
    }
    const std::size_t global_line_begin = selection_offset + line.begin;
    const std::size_t global_line_end = selection_offset + line.end;
    if (g_selection.id == selection_id &&
        selection_end > global_line_begin &&
        selection_begin < global_line_end) {
      const std::size_t begin =
          std::max(selection_begin, global_line_begin) - selection_offset;
      const std::size_t end =
          std::min(selection_end, global_line_end) - selection_offset;
      const float x1 = position.x + XAtByte(text, line, font, font_size, begin);
      const float x2 = position.x + XAtByte(text, line, font, font_size, end);
      draw->AddRectFilled({x1, position.y - 1.0f},
                          {std::max(x1 + 1.0f, x2), position.y + font_size + 2.0f},
                          options.selection_color, 2.0f);
    }
    for (const StyledRun& run : runs) {
      const float x = position.x +
                      XAtByte(text, line, font, font_size, run.begin);
      const ImVec2 text_position(
          x, position.y + (run.style.emphasis ? -0.35f : 0.0f));
      if (run.style.strong) {
        draw->AddText(font, font_size,
                      {text_position.x + 0.55f, text_position.y},
                      run.style.text_color, text.data() + run.begin,
                      text.data() + run.end);
      }
      draw->AddText(font, font_size, text_position, run.style.text_color,
                    text.data() + run.begin, text.data() + run.end);
      if (run.style.underline) {
        const float x2 = position.x +
                         XAtByte(text, line, font, font_size, run.end);
        draw->AddLine({x, position.y + font_size + 1.0f},
                      {x2, position.y + font_size + 1.0f},
                      run.style.text_color, 1.0f);
      }
    }
  }

  if (ImGui::BeginPopupContextItem()) {
    if (ImGui::MenuItem("Copy selection", "Ctrl+C",
                        false, selection_begin != selection_end)) {
      CopySelection(selection_text);
    }
    if (ImGui::MenuItem("Select all", "Ctrl+A")) {
      g_selection.id = selection_id;
      g_selection.anchor = 0;
      g_selection.caret = selection_text.size();
    }
    if (ImGui::MenuItem("Copy full text")) {
      const std::string copy(selection_text);
      ImGui::SetClipboardText(copy.c_str());
    }
    ImGui::EndPopup();
  }
  if (hovered) ImGui::SetMouseCursor(ImGuiMouseCursor_TextInput);
}

}  // namespace gem16::studio::selectable_text

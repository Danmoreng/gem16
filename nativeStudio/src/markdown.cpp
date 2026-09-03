#include "markdown.h"

#include "selectable_text.h"

#include "imgui.h"

#include <algorithm>
#include <cctype>
#include <string_view>
#include <utility>

namespace gem16::studio::markdown {
namespace {

constexpr ImVec4 kAccent{0.31f, 0.91f, 0.65f, 1.0f};

std::string_view TrimLeft(std::string_view value) {
  while (!value.empty() &&
         (value.front() == ' ' || value.front() == '\t' ||
          value.front() == '\r')) {
    value.remove_prefix(1);
  }
  return value;
}

std::string_view Trim(std::string_view value) {
  value = TrimLeft(value);
  while (!value.empty() &&
         (value.back() == ' ' || value.back() == '\t' ||
          value.back() == '\r')) {
    value.remove_suffix(1);
  }
  return value;
}

std::vector<std::string_view> Lines(std::string_view source) {
  std::vector<std::string_view> result;
  std::size_t begin = 0;
  while (begin <= source.size()) {
    const std::size_t end = source.find('\n', begin);
    if (end == std::string_view::npos) {
      result.push_back(source.substr(begin));
      break;
    }
    result.push_back(source.substr(begin, end - begin));
    begin = end + 1;
  }
  return result;
}

void AppendSpan(std::vector<InlineSpan>& spans, std::size_t begin,
                std::size_t end, const InlineSpan& style) {
  if (begin == end || (!style.strong && !style.emphasis && !style.code &&
                       !style.link)) {
    return;
  }
  InlineSpan span = style;
  span.begin = begin;
  span.end = end;
  if (!spans.empty()) {
    InlineSpan& previous = spans.back();
    if (previous.end == span.begin && previous.strong == span.strong &&
        previous.emphasis == span.emphasis && previous.code == span.code &&
        previous.link == span.link &&
        previous.destination == span.destination) {
      previous.end = span.end;
      return;
    }
  }
  spans.push_back(std::move(span));
}

void ParseInlineFragment(std::string_view source, std::string& output,
                         std::vector<InlineSpan>& spans,
                         InlineSpan inherited) {
  std::size_t position = 0;
  while (position < source.size()) {
    if (source[position] == '\\' && position + 1 < source.size()) {
      const std::size_t begin = output.size();
      output.push_back(source[position + 1]);
      AppendSpan(spans, begin, output.size(), inherited);
      position += 2;
      continue;
    }
    if (source[position] == '`') {
      const std::size_t close = source.find('`', position + 1);
      if (close != std::string_view::npos) {
        InlineSpan style = inherited;
        style.code = true;
        const std::size_t begin = output.size();
        output.append(source.substr(position + 1, close - position - 1));
        AppendSpan(spans, begin, output.size(), style);
        position = close + 1;
        continue;
      }
    }
    const bool image = source[position] == '!' && position + 1 < source.size() &&
                       source[position + 1] == '[';
    const bool link = source[position] == '[' || image;
    if (link) {
      const std::size_t label_begin = position + (image ? 2 : 1);
      const std::size_t label_end = source.find("](", label_begin);
      const std::size_t destination_end =
          label_end == std::string_view::npos
              ? std::string_view::npos
              : source.find(')', label_end + 2);
      if (label_end != std::string_view::npos &&
          destination_end != std::string_view::npos) {
        const std::string destination(
            source.substr(label_end + 2, destination_end - label_end - 2));
        if (image) {
          const std::size_t begin = output.size();
          output += "[Image: ";
          output.append(source.substr(label_begin, label_end - label_begin));
          output.push_back(']');
          InlineSpan style = inherited;
          style.emphasis = true;
          AppendSpan(spans, begin, output.size(), style);
        } else {
          InlineSpan style = inherited;
          style.link = true;
          style.destination = destination;
          ParseInlineFragment(
              source.substr(label_begin, label_end - label_begin), output,
              spans, std::move(style));
        }
        position = destination_end + 1;
        continue;
      }
    }
    const bool strong = position + 1 < source.size() &&
                        ((source[position] == '*' && source[position + 1] == '*') ||
                         (source[position] == '_' && source[position + 1] == '_'));
    if (strong) {
      const std::string_view marker = source.substr(position, 2);
      const std::size_t close = source.find(marker, position + 2);
      if (close != std::string_view::npos) {
        InlineSpan style = inherited;
        style.strong = true;
        ParseInlineFragment(source.substr(position + 2, close - position - 2),
                            output, spans, std::move(style));
        position = close + 2;
        continue;
      }
    }
    if (source[position] == '*' || source[position] == '_') {
      const char marker = source[position];
      const std::size_t close = source.find(marker, position + 1);
      if (close != std::string_view::npos) {
        InlineSpan style = inherited;
        style.emphasis = true;
        ParseInlineFragment(source.substr(position + 1, close - position - 1),
                            output, spans, std::move(style));
        position = close + 1;
        continue;
      }
    }
    const std::size_t begin = output.size();
    output.push_back(source[position]);
    AppendSpan(spans, begin, output.size(), inherited);
    ++position;
  }
}

void ParseInline(std::string_view source, Block& block) {
  block.text.clear();
  block.spans.clear();
  ParseInlineFragment(source, block.text, block.spans, {});
}

bool Fence(std::string_view line, char& marker, std::size_t& count,
           std::string_view& info) {
  line = TrimLeft(line);
  if (line.size() < 3 || (line.front() != '`' && line.front() != '~'))
    return false;
  marker = line.front();
  count = 0;
  while (count < line.size() && line[count] == marker) ++count;
  if (count < 3) return false;
  info = Trim(line.substr(count));
  return true;
}

int HeadingLevel(std::string_view line) {
  line = TrimLeft(line);
  int level = 0;
  while (level < 6 && static_cast<std::size_t>(level) < line.size() &&
         line[level] == '#') {
    ++level;
  }
  return level > 0 && static_cast<std::size_t>(level) < line.size() &&
                 line[level] == ' '
             ? level
             : 0;
}

bool ThematicRule(std::string_view line) {
  line = Trim(line);
  char marker = 0;
  int count = 0;
  for (char value : line) {
    if (value == ' ' || value == '\t') continue;
    if (marker == 0) marker = value;
    if (value != marker || (value != '-' && value != '*' && value != '_'))
      return false;
    ++count;
  }
  return count >= 3;
}

bool Bullet(std::string_view line, std::string_view& content) {
  line = TrimLeft(line);
  if (line.size() < 2 || (line[0] != '-' && line[0] != '*' && line[0] != '+') ||
      line[1] != ' ') {
    return false;
  }
  content = TrimLeft(line.substr(2));
  return true;
}

bool Ordered(std::string_view line, int& ordinal, std::string_view& content) {
  line = TrimLeft(line);
  std::size_t digits = 0;
  while (digits < line.size() && std::isdigit(static_cast<unsigned char>(line[digits])) != 0)
    ++digits;
  if (digits == 0 || digits + 1 >= line.size() ||
      (line[digits] != '.' && line[digits] != ')') || line[digits + 1] != ' ')
    return false;
  ordinal = 0;
  for (std::size_t index = 0; index < digits; ++index)
    ordinal = std::min(100000, ordinal * 10 + (line[index] - '0'));
  content = TrimLeft(line.substr(digits + 2));
  return true;
}

bool StartsBlock(std::string_view line) {
  char marker = 0;
  std::size_t count = 0;
  std::string_view info;
  std::string_view content;
  int ordinal = 0;
  const std::string_view trimmed = TrimLeft(line);
  return trimmed.empty() || Fence(line, marker, count, info) ||
         HeadingLevel(line) != 0 || ThematicRule(line) ||
         Bullet(line, content) || Ordered(line, ordinal, content) ||
         trimmed.starts_with('>') ||
         (line.size() >= 4 && line.substr(0, 4) == "    ");
}

std::vector<selectable_text::StyleSpan> DrawSpans(const Block& block) {
  std::vector<selectable_text::StyleSpan> result;
  result.reserve(block.spans.size());
  for (const InlineSpan& span : block.spans) {
    selectable_text::StyleSpan draw;
    draw.begin = span.begin;
    draw.end = span.end;
    draw.strong = span.strong;
    draw.emphasis = span.emphasis;
    draw.underline = span.link;
    if (span.link) draw.text_color = ImGui::ColorConvertFloat4ToU32(kAccent);
    else if (span.emphasis)
      draw.text_color = ImGui::ColorConvertFloat4ToU32(
          {0.73f, 0.80f, 0.77f, 1.0f});
    if (span.code)
      draw.background_color = IM_COL32(27, 48, 42, 235);
    result.push_back(draw);
  }
  return result;
}

void DrawInline(const char* id, const Block& block, float width,
                ImU32 text_color, ImGuiID selection_group,
                std::string_view selection_text,
                std::size_t selection_offset) {
  const std::vector<selectable_text::StyleSpan> spans = DrawSpans(block);
  selectable_text::Wrapped(
      id, block.text,
      {.width = width,
       .text_color = text_color,
       .selection_color = IM_COL32(38, 144, 102, 205),
       .line_spacing = 3.0f * ImGui::GetFontSize() / 17.0f,
       .spans = &spans,
       .selection_group = selection_group,
       .selection_text = selection_text,
       .selection_offset = selection_offset});
}

}  // namespace

std::vector<Block> Parse(std::string_view source) {
  const std::vector<std::string_view> lines = Lines(source);
  std::vector<Block> blocks;
  for (std::size_t index = 0; index < lines.size();) {
    const std::string_view trimmed = Trim(lines[index]);
    if (trimmed.empty()) {
      ++index;
      continue;
    }

    char fence_marker = 0;
    std::size_t fence_count = 0;
    std::string_view info;
    if (Fence(lines[index], fence_marker, fence_count, info)) {
      Block block;
      block.kind = BlockKind::kCode;
      block.info = std::string(info);
      ++index;
      while (index < lines.size()) {
        char close_marker = 0;
        std::size_t close_count = 0;
        std::string_view close_info;
        if (Fence(lines[index], close_marker, close_count, close_info) &&
            close_marker == fence_marker && close_count >= fence_count) {
          ++index;
          break;
        }
        if (!block.text.empty()) block.text.push_back('\n');
        block.text.append(lines[index]);
        ++index;
      }
      blocks.push_back(std::move(block));
      continue;
    }

    if (lines[index].size() >= 4 && lines[index].substr(0, 4) == "    ") {
      Block block;
      block.kind = BlockKind::kCode;
      while (index < lines.size() && lines[index].size() >= 4 &&
             lines[index].substr(0, 4) == "    ") {
        if (!block.text.empty()) block.text.push_back('\n');
        block.text.append(lines[index].substr(4));
        ++index;
      }
      blocks.push_back(std::move(block));
      continue;
    }

    const int heading = HeadingLevel(lines[index]);
    if (heading != 0) {
      Block block;
      block.kind = BlockKind::kHeading;
      block.level = heading;
      std::string_view content =
          Trim(lines[index].substr(static_cast<std::size_t>(heading) + 1));
      while (!content.empty() && content.back() == '#')
        content = Trim(content.substr(0, content.size() - 1));
      ParseInline(content, block);
      blocks.push_back(std::move(block));
      ++index;
      continue;
    }

    if (ThematicRule(lines[index])) {
      Block block;
      block.kind = BlockKind::kRule;
      blocks.push_back(std::move(block));
      ++index;
      continue;
    }

    std::string_view content;
    if (Bullet(lines[index], content)) {
      Block block;
      block.kind = BlockKind::kBulletItem;
      ParseInline(content, block);
      blocks.push_back(std::move(block));
      ++index;
      continue;
    }

    int ordinal = 0;
    if (Ordered(lines[index], ordinal, content)) {
      Block block;
      block.kind = BlockKind::kOrderedItem;
      block.ordinal = ordinal;
      ParseInline(content, block);
      blocks.push_back(std::move(block));
      ++index;
      continue;
    }

    if (TrimLeft(lines[index]).starts_with('>')) {
      std::string quote;
      while (index < lines.size() && TrimLeft(lines[index]).starts_with('>')) {
        std::string_view line = TrimLeft(lines[index]);
        line.remove_prefix(1);
        line = TrimLeft(line);
        if (!quote.empty()) quote.push_back('\n');
        quote.append(line);
        ++index;
      }
      Block block;
      block.kind = BlockKind::kQuote;
      ParseInline(quote, block);
      blocks.push_back(std::move(block));
      continue;
    }

    std::string paragraph;
    bool previous_hard_break = false;
    while (index < lines.size() && !Trim(lines[index]).empty() &&
           (paragraph.empty() || !StartsBlock(lines[index]))) {
      std::string_view line = Trim(lines[index]);
      if (!paragraph.empty())
        paragraph.push_back(previous_hard_break ? '\n' : ' ');
      paragraph.append(line);
      previous_hard_break = lines[index].size() >= 2 &&
                            lines[index].substr(lines[index].size() - 2) ==
                                "  ";
      ++index;
    }
    if (paragraph.empty()) {
      paragraph = std::string(trimmed);
      ++index;
    }
    Block block;
    block.kind = BlockKind::kParagraph;
    ParseInline(paragraph, block);
    blocks.push_back(std::move(block));
  }
  return blocks;
}

void Render(const char* id, const std::string& source, float width) {
  const std::vector<Block> blocks = Parse(source);
  std::string selection_text;
  std::vector<std::size_t> selection_offsets(blocks.size(), 0U);
  bool has_selection_text = false;
  for (std::size_t index = 0; index < blocks.size(); ++index) {
    if (blocks[index].text.empty()) {
      selection_offsets[index] = selection_text.size();
      continue;
    }
    if (has_selection_text) selection_text.push_back('\n');
    selection_offsets[index] = selection_text.size();
    selection_text += blocks[index].text;
    has_selection_text = true;
  }
  ImGui::PushID(id);
  const ImGuiID selection_group = ImGui::GetID("##markdown-selection");
  for (std::size_t index = 0; index < blocks.size(); ++index) {
    const Block& block = blocks[index];
    ImGui::PushID(static_cast<int>(index));
    const float scale = ImGui::GetFontSize() / 17.0f;
    const bool next_same_list = index + 1 < blocks.size() &&
        (block.kind == BlockKind::kBulletItem ||
         block.kind == BlockKind::kOrderedItem) &&
        blocks[index + 1].kind == block.kind;
    // Exactly one block gap, independent of the large application-control
    // spacing. Blank lines between list items must not create paragraph gaps.
    const float gap = index + 1 == blocks.size() ? 0.0f :
                      (next_same_list ? 2.0f : 10.0f) * scale;
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,
                        {ImGui::GetStyle().ItemSpacing.x, gap});
    const float available = std::max(60.0f, width);
    switch (block.kind) {
      case BlockKind::kHeading: {
        const float scales[] = {1.0f, 1.42f, 1.30f, 1.20f, 1.12f, 1.06f, 1.03f};
        ImGui::SetWindowFontScale(scales[std::clamp(block.level, 1, 6)]);
        DrawInline("##heading", block, available,
                   ImGui::ColorConvertFloat4ToU32(kAccent), selection_group,
                   selection_text, selection_offsets[index]);
        ImGui::SetWindowFontScale(1.0f);
        break;
      }
      case BlockKind::kCode: {
        ImGui::PushStyleColor(ImGuiCol_ChildBg,
                              {0.025f, 0.045f, 0.041f, 0.96f});
        ImGui::PushStyleColor(ImGuiCol_Border,
                              {0.12f, 0.31f, 0.25f, 0.92f});
        ImGui::BeginChild("##code-block", {available, 0},
                          ImGuiChildFlags_AutoResizeY |
                              ImGuiChildFlags_Borders);
        ImGui::TextColored(kAccent, "%s",
                           block.info.empty() ? "Code" : block.info.c_str());
        ImGui::SameLine();
        ImGui::SetCursorPosX(ImGui::GetWindowWidth() - 66.0f);
        if (ImGui::SmallButton("Copy##code"))
          ImGui::SetClipboardText(block.text.c_str());
        ImGui::Separator();
        selectable_text::Wrapped(
            "##code-text", block.text,
            {.width = ImGui::GetContentRegionAvail().x,
             .text_color = ImGui::ColorConvertFloat4ToU32(
                 {0.82f, 0.89f, 0.86f, 1.0f}),
             .selection_color = IM_COL32(38, 144, 102, 205),
             .line_spacing = 4.0f,
             .selection_group = selection_group,
             .selection_text = selection_text,
             .selection_offset = selection_offsets[index]});
        ImGui::EndChild();
        ImGui::PopStyleColor(2);
        break;
      }
      case BlockKind::kBulletItem:
      case BlockKind::kOrderedItem: {
        const std::string marker = block.kind == BlockKind::kBulletItem
                                       ? "•"
                                       : std::to_string(block.ordinal) + ".";
        ImGui::TextColored(kAccent, "%s", marker.c_str());
        ImGui::SameLine(0, 7);
        DrawInline("##list-item", block,
                   std::max(40.0f, available - 30.0f),
                   ImGui::GetColorU32(ImGuiCol_Text), selection_group,
                   selection_text, selection_offsets[index]);
        break;
      }
      case BlockKind::kQuote: {
        const ImVec2 rail = ImGui::GetCursorScreenPos();
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 14.0f);
        DrawInline("##quote", block, std::max(40.0f, available - 14.0f),
                   ImGui::ColorConvertFloat4ToU32(
                       {0.72f, 0.80f, 0.77f, 1.0f}), selection_group,
                   selection_text, selection_offsets[index]);
        ImGui::GetWindowDrawList()->AddRectFilled(
            rail, {rail.x + 3.0f, ImGui::GetItemRectMax().y},
            ImGui::ColorConvertFloat4ToU32(kAccent), 2.0f);
        break;
      }
      case BlockKind::kRule:
        ImGui::Separator();
        break;
      case BlockKind::kParagraph:
        DrawInline("##paragraph", block, available,
                   ImGui::GetColorU32(ImGuiCol_Text), selection_group,
                   selection_text, selection_offsets[index]);
        break;
    }
    ImGui::PopStyleVar();
    ImGui::PopID();
  }
  ImGui::PopID();
}

}  // namespace gem16::studio::markdown

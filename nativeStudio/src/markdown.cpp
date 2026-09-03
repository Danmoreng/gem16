#include "markdown.h"
#include "selectable_text.h"
#include "math_renderer.h"
#include "platform_ui.h"
#include <algorithm>

namespace gem16::studio::markdown {
namespace {
constexpr ImVec4 kAccent{0.20f, 0.83f, 0.60f, 1.0f};

std::vector<selectable_text::StyleSpan> DrawSpans(const Block& block, float width) {
  std::vector<selectable_text::StyleSpan> result;
  result.reserve(block.spans.size());
  for (const InlineSpan& span : block.spans) {
    selectable_text::StyleSpan draw;
    draw.begin = span.begin;
    draw.end = span.end;
    draw.strong = span.strong;
    draw.emphasis = span.emphasis;
    draw.strike = span.strike;
    if (span.link && IsSafeWebLink(span.destination)) draw.link = span.destination;
    if (span.math) {
      auto math = LayoutMath(std::string_view(block.text).substr(span.begin, span.end - span.begin),
          span.display_math, ImGui::GetFontSize(), width);
      if (math.data) draw.math = std::make_shared<MathLayout>(std::move(math));
      else draw.background_color = IM_COL32(70, 45, 20, 170);
    }
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
  const std::vector<selectable_text::StyleSpan> spans = DrawSpans(block, width);
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
    const float origin_x = ImGui::GetCursorPosX();
    const float indent = (block.indent * 22.0f + block.quote_depth * 10.0f) * scale;
    ImGui::SetCursorPosX(origin_x + indent);
    const float available = std::max(60.0f, width - indent);
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
        const std::string marker = block.task ? (block.checked ? "☑" : "☐") : block.kind == BlockKind::kBulletItem
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
      case BlockKind::kTable: {
        if (block.alignments.empty()) break;
        ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, {8.0f * scale, 5.0f * scale});
        if (ImGui::BeginTable("##table", static_cast<int>(block.alignments.size()),
              ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchSame,
              {available, 0})) {
          std::size_t offset = selection_offsets[index];
          for (std::size_t row = 0; row < block.rows.size(); ++row) {
            ImGui::TableNextRow();
            if (row) ++offset;
            for (std::size_t col = 0; col < block.rows[row].size(); ++col) {
              ImGui::TableNextColumn();
              if (col) ++offset;
              Block cell = block.rows[row][col];
              if (row == 0) { InlineSpan header; header.end = cell.text.size(); header.strong = true; cell.spans.push_back(header); }
              const float cell_width = ImGui::GetContentRegionAvail().x;
              const float text_width = ImGui::CalcTextSize(cell.text.c_str()).x;
              float shift = 0;
              if (text_width < cell_width && col < block.alignments.size()) {
                if (block.alignments[col] == 2) shift = (cell_width - text_width) * 0.5f;
                if (block.alignments[col] == 3) shift = cell_width - text_width;
              }
              ImGui::SetCursorPosX(ImGui::GetCursorPosX() + shift);
              ImGui::PushID(static_cast<int>(row * 32 + col));
              DrawInline("##cell", cell, cell_width - shift, ImGui::GetColorU32(ImGuiCol_Text),
                  selection_group, selection_text, offset);
              ImGui::PopID();
              offset += cell.text.size();
            }
          }
          ImGui::EndTable();
        }
        ImGui::PopStyleVar();
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
    ImGui::SetCursorPosX(origin_x);
    ImGui::PopID();
  }
  ImGui::PopID();
}

}  // namespace gem16::studio::markdown

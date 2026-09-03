#include "markdown.h"
#include "selectable_text.h"
#include "math_renderer.h"
#include "platform_ui.h"
#include "fonts.h"
#include "svg_preview.h"
#include <algorithm>

namespace gem16::studio::markdown {
namespace {
constexpr ImVec4 kAccent{0.20f, 0.83f, 0.60f, 1.0f};

void SvgCanvas(const ImageTexture& texture, ImVec2 size, float scale) {
  const ImVec2 origin = ImGui::GetCursorScreenPos();
  auto* draw = ImGui::GetWindowDrawList();
  const float inset = std::min(20 * scale, size.x * 0.05f);
  const float fit = std::min((size.x - 2*inset) / texture.Width(),
                              (size.y - 2*inset) / texture.Height());
  const ImVec2 image_size(texture.Width()*fit, texture.Height()*fit);
  const ImVec2 image_pos(origin.x + (size.x-image_size.x)*0.5f,
                         origin.y + (size.y-image_size.y)*0.5f);
  draw->AddRectFilled(origin, {origin.x+size.x, origin.y+size.y}, IM_COL32(19,25,24,255), 8*scale);
  draw->AddRectFilled({image_pos.x+2*scale,image_pos.y+3*scale},
      {image_pos.x+image_size.x+2*scale,image_pos.y+image_size.y+3*scale}, IM_COL32(0,0,0,60), 3*scale);
  draw->AddRectFilled(image_pos, {image_pos.x+image_size.x,image_pos.y+image_size.y}, IM_COL32_WHITE);
  draw->AddImage(ImTextureRef(texture.Id()), image_pos, {image_pos.x+image_size.x,image_pos.y+image_size.y});
  ImGui::Dummy(size);
}

// Returns true when the caller should draw the selectable source below us.
bool SvgArtifact(const Block& block, SvgPreviewCache& cache, float scale) {
  auto* storage = ImGui::GetStateStorage();
  const ImGuiID mode_id = ImGui::GetID("##svg-code-view");
  const ImGuiID copied_id = ImGui::GetID("##svg-copied-until");
  bool code = storage->GetBool(mode_id, false);
  const float x = ImGui::GetCursorPosX(), y = ImGui::GetCursorPosY();
  const float width = ImGui::GetContentRegionAvail().x;
  const float button_h = 32*scale, tab_w = 72*scale, gap = 6*scale;
  const float copy_w = 116*scale, expand_w = 80*scale;
  const bool stacked = width < 48*scale + 2*tab_w + copy_w + expand_w + 3*gap;
  ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 8*scale);
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {8*scale,6*scale});
  ImGui::PushStyleColor(ImGuiCol_Button, {0,0,0,0});
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.13f,0.25f,0.21f,1});
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, {0.17f,0.34f,0.27f,1});
  const auto badge = ImGui::GetCursorScreenPos();
  ImGui::GetWindowDrawList()->AddRectFilled({badge.x,badge.y+6*scale},
      {badge.x+38*scale,badge.y+27*scale}, IM_COL32(39,48,45,255), 5*scale);
  ImGui::GetWindowDrawList()->AddText({badge.x+6*scale,badge.y+8*scale}, IM_COL32(177,190,184,255), "SVG");
  ImGui::SetCursorPos({x+48*scale,y});
  ImGui::PushStyleColor(ImGuiCol_Button, code ? ImVec4(0.12f,0.24f,0.19f,1) : ImVec4(0,0,0,0));
  if (ImGui::Button("Code##svg-tab", {tab_w,button_h})) code = true;
  ImGui::PopStyleColor();
  ImGui::SameLine(0,0);
  ImGui::PushStyleColor(ImGuiCol_Button, !code ? ImVec4(0.12f,0.24f,0.19f,1) : ImVec4(0,0,0,0));
  if (ImGui::Button("Preview##svg-tab", {tab_w,button_h})) code = false;
  ImGui::PopStyleColor();
  storage->SetBool(mode_id, code);
  ImGui::SetCursorPos({x+std::max(0.0f,width-copy_w-expand_w-gap), y+(stacked ? 38*scale : 0)});
  const bool copied = storage->GetFloat(copied_id, -1) > ImGui::GetTime();
  if (ImGui::Button(copied ? "Copied##svg-copy" : "Copy code##svg-copy", {copy_w,button_h})) {
    ImGui::SetClipboardText(block.text.c_str());
    storage->SetFloat(copied_id, static_cast<float>(ImGui::GetTime()+1.8));
  }
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Copy the original SVG source");
  ImGui::SameLine(0,gap);
  if (ImGui::Button("Expand##svg-expand", {expand_w,button_h})) ImGui::OpenPopup("SVG preview##expanded");
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Open a larger preview");
  ImGui::PopStyleColor(3);
  ImGui::PopStyleVar(2);
  ImGui::SetCursorPos({x,y+(stacked ? 82 : 44)*scale});
  ImGui::Separator();
  bool draw_code = code;
  if (!code) {
    auto* preview = cache.Get(block.text);
    if (preview && preview->texture.Valid()) {
      SvgCanvas(preview->texture, {width,std::clamp(width*0.60f,280*scale,620*scale)}, scale);
    } else {
      ImGui::PushTextWrapPos();
      ImGui::TextColored({0.95f,0.73f,0.35f,1}, "%s", preview ? preview->error.c_str() :
          "SVG preview limit reached (256 KiB per SVG, eight cached previews).");
      ImGui::PopTextWrapPos();
      draw_code = true;
    }
  }
  const auto* viewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowSize({viewport->WorkSize.x*0.9f,viewport->WorkSize.y*0.9f}, ImGuiCond_Appearing);
  ImGui::SetNextWindowPos(viewport->GetWorkCenter(), ImGuiCond_Appearing, {0.5f,0.5f});
  bool open = true;
  if (ImGui::BeginPopupModal("SVG preview##expanded", &open, ImGuiWindowFlags_NoSavedSettings)) {
    if (ImGui::Button("Close")) ImGui::CloseCurrentPopup();
    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
    auto* preview = cache.Get(block.text);
    if (preview && preview->texture.Valid()) {
      const auto available = ImGui::GetContentRegionAvail();
      if (available.x > 40*scale && available.y > 40*scale) SvgCanvas(preview->texture, available, scale);
    } else ImGui::TextWrapped("%s", preview ? preview->error.c_str() : "SVG preview limit reached.");
    ImGui::EndPopup();
  }
  return draw_code;
}

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
    if (span.code) {
      draw.background_color = IM_COL32(27, 48, 42, 235);
      draw.font = StudioCodeFont();
    }
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

void Render(const char* id, const std::string& source, float width, SvgPreviewCache* svg_cache) {
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
        const bool svg = svg_cache && IsSvgCode(block.info, block.text);
        if (svg) ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 13*scale);
        ImGui::PushStyleColor(ImGuiCol_ChildBg,
                              {0.025f, 0.045f, 0.041f, 0.96f});
        ImGui::PushStyleColor(ImGuiCol_Border,
                              {0.12f, 0.31f, 0.25f, 0.92f});
        ImGui::BeginChild("##code-block", {available, 0},
                          ImGuiChildFlags_AutoResizeY |
                              ImGuiChildFlags_Borders);
        bool draw_code = true;
        if (svg) draw_code = SvgArtifact(block, *svg_cache, scale);
        else {
        const float header_x = ImGui::GetCursorPosX();
        const float header_width = ImGui::GetContentRegionAvail().x;
        const float button_width = ImGui::CalcTextSize("Copy").x + 2 * ImGui::GetStyle().FramePadding.x;
        const float label_width = std::max(0.0f, header_width - button_width - 8 * scale);
        const ImVec2 label_pos = ImGui::GetCursorScreenPos();
        ImGui::PushClipRect(label_pos, {label_pos.x + label_width, label_pos.y + ImGui::GetTextLineHeight()}, true);
        ImGui::GetWindowDrawList()->AddText(label_pos, ImGui::ColorConvertFloat4ToU32(kAccent),
            block.info.empty() ? "Code" : block.info.c_str());
        ImGui::PopClipRect();
        ImGui::SetCursorPosX(header_x + std::max(0.0f, header_width - button_width));
        if (ImGui::SmallButton("Copy##code"))
          ImGui::SetClipboardText(block.text.c_str());
        ImGui::Separator();
        }
        if (draw_code) {
          ImGui::PushFont(StudioCodeFont(), 0.0f);
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
          ImGui::PopFont();
        }
        ImGui::EndChild();
        ImGui::PopStyleColor(2);
        if (svg) ImGui::PopStyleVar();
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

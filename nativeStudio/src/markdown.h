#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace gem16::studio { class SvgPreviewCache; }

namespace gem16::studio::markdown {

enum class BlockKind {
  kParagraph,
  kHeading,
  kCode,
  kBulletItem,
  kOrderedItem,
  kQuote,
  kRule,
  kTable,
};

struct InlineSpan {
  std::size_t begin = 0;
  std::size_t end = 0;
  bool strong = false;
  bool emphasis = false;
  bool code = false;
  bool link = false;
  bool strike = false;
  bool math = false;
  bool display_math = false;
  std::string destination;
};

struct Block {
  BlockKind kind = BlockKind::kParagraph;
  std::string text;
  std::string info;
  int level = 0;
  int ordinal = 0;
  std::vector<InlineSpan> spans;
  int indent = 0;
  int quote_depth = 0;
  bool task = false;
  bool checked = false;
  std::vector<std::vector<Block>> rows;
  std::vector<int> alignments;
};

[[nodiscard]] std::vector<Block> Parse(std::string_view source);

// md4c CommonMark/GFM parsing with native selectable blocks, nested lists,
// tables, tasks, combined inline styles, safe web links, and bounded MicroTeX
// math. HTML and remote image loading are deliberately disabled.
void Render(const char* id, const std::string& source, float width, SvgPreviewCache* svg_cache = nullptr);

}  // namespace gem16::studio::markdown

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace gem16::studio::markdown {

enum class BlockKind {
  kParagraph,
  kHeading,
  kCode,
  kBulletItem,
  kOrderedItem,
  kQuote,
  kRule,
};

struct InlineSpan {
  std::size_t begin = 0;
  std::size_t end = 0;
  bool strong = false;
  bool emphasis = false;
  bool code = false;
  bool link = false;
  std::string destination;
};

struct Block {
  BlockKind kind = BlockKind::kParagraph;
  std::string text;
  std::string info;
  int level = 0;
  int ordinal = 0;
  std::vector<InlineSpan> spans;
};

[[nodiscard]] std::vector<Block> Parse(std::string_view source);

// Renders a compact CommonMark-compatible subset while retaining the native
// selectable-text interactions. Supported blocks are headings, paragraphs,
// fenced/indented code, ordered and unordered lists, quotes, and rules;
// supported inline styles are emphasis, strong emphasis, code, and links.
void Render(const char* id, const std::string& source, float width);

}  // namespace gem16::studio::markdown

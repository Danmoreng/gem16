#include "markdown.h"
#include "md4c.h"
extern "C" {
#include "entity.h"
}
#include <charconv>
#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>

namespace gem16::studio::markdown {
namespace {
void Utf8(std::string& out, unsigned cp) {
  if (cp == 0 || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) cp = 0xfffd;
  if (cp < 0x80) out += static_cast<char>(cp);
  else if (cp < 0x800) { out += static_cast<char>(0xc0 | (cp >> 6)); out += static_cast<char>(0x80 | (cp & 63)); }
  else if (cp < 0x10000) { out += static_cast<char>(0xe0 | (cp >> 12)); out += static_cast<char>(0x80 | ((cp >> 6) & 63)); out += static_cast<char>(0x80 | (cp & 63)); }
  else { out += static_cast<char>(0xf0 | (cp >> 18)); out += static_cast<char>(0x80 | ((cp >> 12) & 63)); out += static_cast<char>(0x80 | ((cp >> 6) & 63)); out += static_cast<char>(0x80 | (cp & 63)); }
}
std::string Entity(std::string_view value) {
  std::string out;
  if (value.starts_with("&#") && value.ends_with(';')) {
    value.remove_prefix(2); value.remove_suffix(1);
    const bool hex = value.starts_with('x') || value.starts_with('X');
    if (hex) value.remove_prefix(1);
    unsigned cp = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), cp, hex ? 16 : 10);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) cp = 0xfffd;
    Utf8(out, cp);
  } else if (const ENTITY* entity = entity_lookup(value.data(), value.size())) {
    Utf8(out, entity->codepoints[0]);
    if (entity->codepoints[1]) Utf8(out, entity->codepoints[1]);
  } else out = value;
  return out;
}
std::string Attribute(const MD_ATTRIBUTE& attr) {
  std::string out;
  for (unsigned i = 0; attr.size && attr.substr_offsets[i] < attr.size; ++i) {
    const std::string_view part(attr.text + attr.substr_offsets[i], attr.substr_offsets[i + 1] - attr.substr_offsets[i]);
    out += attr.substr_types[i] == MD_TEXT_ENTITY ? Entity(part) : std::string(part);
  }
  return out;
}

// md4c's math extension is inline-only: block parsing can interpret an '='
// inside $$ ... $$ as a Setext heading before math spans are recognized.
// Shield complete, line-leading display formulas as opaque fences, retaining
// their original TeX out of band. Never alter actual fenced/indented code.
std::string ProtectDisplayMath(std::string_view source, std::map<std::string, std::string>& formulas) {
  std::string prefix = "gem16-display-math-";
  while (source.find(prefix) != source.npos) prefix += 'x';
  std::string out;
  char fence = 0;
  std::size_t fence_length = 0;
  for (std::size_t pos = 0; pos < source.size();) {
    const auto newline = source.find('\n', pos);
    const auto end = newline == source.npos ? source.size() : newline + 1;
    auto line = source.substr(pos, end - pos);
    const auto indent = line.find_first_not_of(' ');
    auto trimmed = indent != line.npos && indent <= 3 ? line.substr(indent) : std::string_view{};
    if (!trimmed.empty() && (trimmed[0] == '`' || trimmed[0] == '~')) {
      const char ch = trimmed[0];
      const auto count = trimmed.find_first_not_of(ch);
      const auto run = count == trimmed.npos ? trimmed.size() : count;
      const auto tail = trimmed.substr(run);
      if (!fence && run >= 3 && (ch != '`' || tail.find('`') == tail.npos)) { fence = ch; fence_length = run; }
      else if (fence == ch && run >= fence_length && tail.find_first_not_of(" \t\r\n") == tail.npos) fence = 0;
    } else if (!fence && trimmed.starts_with("$$") && !trimmed.starts_with("$$$")) {
      const auto start = pos + indent + 2;
      const auto candidate = source.substr(start, std::min<std::size_t>(4098, source.size() - start));
      std::size_t close = 0;
      while ((close = candidate.find("$$", close)) != candidate.npos) {
        std::size_t escapes = 0;
        for (std::size_t i = close; i > 0 && candidate[i - 1] == '\\'; --i) ++escapes;
        if (escapes % 2 == 0) break;
        close += 2;
      }
      if (close != candidate.npos) {
        const auto after = start + close + 2;
        const auto last_newline = source.find('\n', after);
        const auto last_end = last_newline == source.npos ? source.size() : last_newline + 1;
        if (source.substr(after, last_end - after).find_first_not_of(" \t\r\n") == source.npos) {
          auto formula = source.substr(start, close);
          const auto first = formula.find_first_not_of(" \t\r\n");
          if (first != formula.npos) {
            formula = formula.substr(first, formula.find_last_not_of(" \t\r\n") - first + 1);
            const auto key = prefix + std::to_string(formulas.size());
            formulas.emplace(key, formula);
            out += "\n```" + key + "\n.\n```\n\n";
            pos = last_end;
            continue;
          }
        }
      }
    }
    out += line;
    pos = end;
  }
  return out;
}
struct Parser {
  struct List { bool ordered; unsigned ordinal; };
  struct Item { bool pending = true; bool task = false; bool checked = false; };
  std::vector<Block> blocks;
  std::vector<List> lists;
  std::vector<Item> items;
  std::vector<InlineSpan> styles{InlineSpan{}};
  Block current;
  bool active = false, cell = false;
  int quotes = 0;
  std::size_t table = 0;
  unsigned depth = 0;
  std::map<std::string, std::string> display_formulas;
  void Start(BlockKind kind = BlockKind::kParagraph) {
    current = {}; current.kind = kind; active = true;
    current.indent = static_cast<int>(lists.size()); current.quote_depth = quotes;
    if (kind == BlockKind::kParagraph && !items.empty() && items.back().pending) {
      items.back().pending = false;
      current.kind = lists.back().ordered ? BlockKind::kOrderedItem : BlockKind::kBulletItem;
      current.ordinal = static_cast<int>(lists.back().ordinal++);
      current.task = items.back().task; current.checked = items.back().checked;
      current.indent = static_cast<int>(lists.size()) - 1;
    }
    else if (kind == BlockKind::kParagraph && quotes > 0 && items.empty()) current.kind = BlockKind::kQuote;
  }
  void Flush() {
    if (!active) return;
    if (current.kind == BlockKind::kCode) {
      if (auto it = display_formulas.find(current.info); it != display_formulas.end()) {
        current.kind = BlockKind::kParagraph;
        current.info.clear();
        current.text = it->second;
        InlineSpan math; math.end = current.text.size(); math.math = math.display_math = true;
        current.spans = {math};
      }
    }
    if (current.kind == BlockKind::kCode && current.text.ends_with('\n')) current.text.pop_back();
    if (cell) blocks[table].rows.back().push_back(std::move(current));
    else if (!current.text.empty() || current.kind == BlockKind::kRule) blocks.push_back(std::move(current));
    active = false;
    if (blocks.size() > 10000) throw std::runtime_error("Markdown block limit");
  }
  void Text(MD_TEXTTYPE type, std::string_view text) {
    if (!active) Start();
    InlineSpan style = styles.back();
    style.begin = current.text.size();
    if (type == MD_TEXT_ENTITY) current.text += Entity(text);
    else if (type == MD_TEXT_NULLCHAR) Utf8(current.text, 0xfffd);
    else if (type == MD_TEXT_BR) current.text += '\n';
    else if (type == MD_TEXT_SOFTBR) current.text += ' ';
    else current.text += text;
    style.end = current.text.size();
    if (style.end > style.begin) {
      if (style.math && !current.spans.empty() && current.spans.back().math &&
          current.spans.back().begin >= styles.back().begin && current.spans.back().end == style.begin &&
          current.spans.back().display_math == style.display_math)
        current.spans.back().end = style.end;
      else current.spans.push_back(std::move(style));
    }
  }
  int Enter(MD_BLOCKTYPE type, void* detail) {
    if (++depth > 64) return 1;
    switch (type) {
      case MD_BLOCK_QUOTE: Flush(); ++quotes; break;
      case MD_BLOCK_UL: Flush(); lists.push_back({false, 1}); break;
      case MD_BLOCK_OL: Flush(); lists.push_back({true, static_cast<MD_BLOCK_OL_DETAIL*>(detail)->start}); break;
      case MD_BLOCK_LI: {
        Flush(); auto* d = static_cast<MD_BLOCK_LI_DETAIL*>(detail);
        items.push_back({true, d->is_task != 0, d->task_mark == 'x' || d->task_mark == 'X'}); break;
      }
      case MD_BLOCK_P: Flush(); Start(); break;
      case MD_BLOCK_H: Flush(); Start(BlockKind::kHeading); current.level = static_cast<int>(static_cast<MD_BLOCK_H_DETAIL*>(detail)->level); break;
      case MD_BLOCK_CODE: Flush(); Start(BlockKind::kCode); current.info = Attribute(static_cast<MD_BLOCK_CODE_DETAIL*>(detail)->lang); break;
      case MD_BLOCK_HR: Flush(); Start(BlockKind::kRule); Flush(); break;
      case MD_BLOCK_TABLE: {
        Flush(); Block b; b.kind = BlockKind::kTable; b.indent = static_cast<int>(lists.size()); b.quote_depth = quotes;
        const auto columns = static_cast<MD_BLOCK_TABLE_DETAIL*>(detail)->col_count;
        if (columns > 32) return 1;
        b.alignments.resize(columns); blocks.push_back(std::move(b)); table = blocks.size() - 1; break;
      }
      case MD_BLOCK_TR: if (blocks[table].rows.size() >= 1000) return 1; blocks[table].rows.emplace_back(); break;
      case MD_BLOCK_TH: case MD_BLOCK_TD: {
        cell = true; Start();
        const auto col = blocks[table].rows.back().size();
        if (col < blocks[table].alignments.size()) blocks[table].alignments[col] = static_cast<int>(static_cast<MD_BLOCK_TD_DETAIL*>(detail)->align);
        break;
      }
      default: break;
    }
    return 0;
  }
  int Leave(MD_BLOCKTYPE type) {
    --depth;
    switch (type) {
      case MD_BLOCK_QUOTE: Flush(); --quotes; break;
      case MD_BLOCK_UL: case MD_BLOCK_OL: Flush(); lists.pop_back(); break;
      case MD_BLOCK_LI: Flush(); items.pop_back(); break;
      case MD_BLOCK_TH: case MD_BLOCK_TD: Flush(); cell = false; break;
      case MD_BLOCK_TABLE:
        for (const auto& row : blocks[table].rows) {
          if (!blocks[table].text.empty()) blocks[table].text += '\n';
          for (std::size_t col = 0; col < row.size(); ++col) {
            if (col) blocks[table].text += '\t';
            blocks[table].text += row[col].text;
          }
        }
        break;
      case MD_BLOCK_P: case MD_BLOCK_H: case MD_BLOCK_CODE: Flush(); break;
      default: break;
    }
    return 0;
  }
  int Span(MD_SPANTYPE type, void* detail) {
    if (styles.size() >= 64) return 1;
    auto s = styles.back();
    switch (type) {
      case MD_SPAN_EM: s.emphasis = true; break;
      case MD_SPAN_STRONG: s.strong = true; break;
      case MD_SPAN_CODE: s.code = true; break;
      case MD_SPAN_DEL: s.strike = true; break;
      case MD_SPAN_A: s.link = true; s.destination = Attribute(static_cast<MD_SPAN_A_DETAIL*>(detail)->href); break;
      case MD_SPAN_LATEXMATH: case MD_SPAN_LATEXMATH_DISPLAY:
        s.math = true; s.display_math = type == MD_SPAN_LATEXMATH_DISPLAY;
        s.begin = current.text.size(); break;
      case MD_SPAN_IMG: Text(MD_TEXT_NORMAL, "Image: "); break;
      default: break;
    }
    styles.push_back(std::move(s)); return 0;
  }
};
template<class F> int Guard(F&& fn) noexcept { try { return fn(); } catch (...) { return 1; } }
}
std::vector<Block> Parse(std::string_view source) {
  if (source.size() > 2U * 1024U * 1024U) return {{BlockKind::kParagraph, "Markdown exceeds the 2 MiB rendering limit."}};
  Parser context;
  const std::string protected_source = ProtectDisplayMath(source, context.display_formulas);
  MD_PARSER parser{};
  parser.flags = MD_DIALECT_GITHUB | MD_FLAG_LATEXMATHSPANS | MD_FLAG_NOHTML;
  parser.enter_block = [](MD_BLOCKTYPE t, void* d, void* p) { return Guard([&] { return static_cast<Parser*>(p)->Enter(t,d); }); };
  parser.leave_block = [](MD_BLOCKTYPE t, void*, void* p) { return Guard([&] { return static_cast<Parser*>(p)->Leave(t); }); };
  parser.enter_span = [](MD_SPANTYPE t, void* d, void* p) { return Guard([&] { return static_cast<Parser*>(p)->Span(t,d); }); };
  parser.leave_span = [](MD_SPANTYPE, void*, void* p) { static_cast<Parser*>(p)->styles.pop_back(); return 0; };
  parser.text = [](MD_TEXTTYPE t, const MD_CHAR* s, MD_SIZE n, void* p) { return Guard([&] { static_cast<Parser*>(p)->Text(t, {s,n}); return 0; }); };
  if (md_parse(protected_source.data(), static_cast<MD_SIZE>(protected_source.size()), &parser, &context) != 0)
    return {{BlockKind::kCode, std::string(source), "Markdown rendering limit reached"}};
  context.Flush();
  return std::move(context.blocks);
}
}  // namespace gem16::studio::markdown

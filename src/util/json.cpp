#include "util/json.h"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>

namespace gem16gb::json {
namespace {

class Parser {
 public:
  Parser(std::string_view input, ParseLimits limits) : input_(input), limits_(limits) {}

  Result<Value> Run() {
    SkipWhitespace();
    auto value = ParseValue(0);
    if (!value.ok()) {
      return value.status();
    }
    SkipWhitespace();
    if (position_ != input_.size()) {
      return Error("trailing data after the root value");
    }
    return std::move(value).value();
  }

 private:
  Result<Value> ParseValue(std::size_t depth) {
    if (depth > limits_.max_depth) {
      return Error("nesting limit exceeded");
    }
    if (++values_seen_ > limits_.max_values) {
      return Error("value count limit exceeded");
    }
    if (position_ >= input_.size()) {
      return Error("unexpected end of input");
    }

    switch (input_[position_]) {
      case 'n': return ParseLiteral("null", Value(nullptr));
      case 't': return ParseLiteral("true", Value(true));
      case 'f': return ParseLiteral("false", Value(false));
      case '"': {
        auto string = ParseString();
        if (!string.ok()) {
          return string.status();
        }
        return Value(std::move(string).value());
      }
      case '[': return ParseArray(depth + 1);
      case '{': return ParseObject(depth + 1);
      default:
        if (input_[position_] == '-' || IsDigit(input_[position_])) {
          return ParseNumber();
        }
        return Error("unexpected character");
    }
  }

  Result<Value> ParseLiteral(std::string_view literal, Value value) {
    if (input_.substr(position_, literal.size()) != literal) {
      return Error("invalid literal");
    }
    position_ += literal.size();
    return value;
  }

  Result<Value> ParseArray(std::size_t depth) {
    ++position_;
    SkipWhitespace();
    Value::Array array;
    if (Consume(']')) {
      return Value(std::move(array));
    }
    while (true) {
      auto item = ParseValue(depth);
      if (!item.ok()) {
        return item.status();
      }
      array.push_back(std::move(item).value());
      SkipWhitespace();
      if (Consume(']')) {
        return Value(std::move(array));
      }
      if (!Consume(',')) {
        return Error("expected ',' or ']' in array");
      }
      SkipWhitespace();
    }
  }

  Result<Value> ParseObject(std::size_t depth) {
    ++position_;
    SkipWhitespace();
    Value::Object object;
    if (Consume('}')) {
      return Value(std::move(object));
    }
    while (true) {
      if (position_ >= input_.size() || input_[position_] != '"') {
        return Error("expected a string object key");
      }
      auto key = ParseString();
      if (!key.ok()) {
        return key.status();
      }
      SkipWhitespace();
      if (!Consume(':')) {
        return Error("expected ':' after object key");
      }
      SkipWhitespace();
      auto value = ParseValue(depth);
      if (!value.ok()) {
        return value.status();
      }
      auto [unused, inserted] = object.emplace(std::move(key).value(), std::move(value).value());
      (void)unused;
      if (!inserted) {
        return Error("duplicate object key");
      }
      SkipWhitespace();
      if (Consume('}')) {
        return Value(std::move(object));
      }
      if (!Consume(',')) {
        return Error("expected ',' or '}' in object");
      }
      SkipWhitespace();
    }
  }

  Result<Value> ParseNumber() {
    const std::size_t start = position_;
    if (Consume('-') && position_ >= input_.size()) {
      return Error("incomplete number");
    }
    if (Consume('0')) {
      if (position_ < input_.size() && IsDigit(input_[position_])) {
        return Error("leading zero in number");
      }
    } else {
      if (position_ >= input_.size() || !IsDigit19(input_[position_])) {
        return Error("invalid integer part");
      }
      while (position_ < input_.size() && IsDigit(input_[position_])) {
        ++position_;
      }
    }

    bool integer = true;
    if (Consume('.')) {
      integer = false;
      if (position_ >= input_.size() || !IsDigit(input_[position_])) {
        return Error("fraction requires a digit");
      }
      while (position_ < input_.size() && IsDigit(input_[position_])) {
        ++position_;
      }
    }
    if (position_ < input_.size() && (input_[position_] == 'e' || input_[position_] == 'E')) {
      integer = false;
      ++position_;
      if (position_ < input_.size() && (input_[position_] == '+' || input_[position_] == '-')) {
        ++position_;
      }
      if (position_ >= input_.size() || !IsDigit(input_[position_])) {
        return Error("exponent requires a digit");
      }
      while (position_ < input_.size() && IsDigit(input_[position_])) {
        ++position_;
      }
    }

    const std::string_view token = input_.substr(start, position_ - start);
    if (integer) {
      std::int64_t parsed = 0;
      const auto result = std::from_chars(token.data(), token.data() + token.size(), parsed);
      if (result.ec == std::errc{} && result.ptr == token.data() + token.size()) {
        return Value(parsed);
      }
      if (result.ec != std::errc::result_out_of_range) {
        return Error("invalid integer");
      }
    }

    // JSON itself does not limit integer magnitude. Preserve exactly representable
    // int64 values above, and represent larger finite JSON numbers as binary64.
    // Typed consumers that require an integer continue to reject this value via
    // is_integer(), while metadata sentinels such as tokenizer model_max_length
    // remain parseable.
    std::string terminated(token);
    char* end = nullptr;
    const double parsed = std::strtod(terminated.c_str(), &end);
    if (end != terminated.c_str() + terminated.size() || !std::isfinite(parsed)) {
      return Error("invalid or non-finite number");
    }
    return Value(parsed);
  }

  Result<std::string> ParseString() {
    ++position_;
    std::string output;
    while (position_ < input_.size()) {
      const unsigned char byte = static_cast<unsigned char>(input_[position_++]);
      if (byte == '"') {
        if (!IsValidUtf8(output)) {
          return Error("string is not valid UTF-8");
        }
        return output;
      }
      if (byte < 0x20U) {
        return Error("unescaped control character in string");
      }
      if (byte != '\\') {
        output.push_back(static_cast<char>(byte));
      } else {
        if (position_ >= input_.size()) {
          return Error("incomplete string escape");
        }
        const char escaped = input_[position_++];
        switch (escaped) {
          case '"': output.push_back('"'); break;
          case '\\': output.push_back('\\'); break;
          case '/': output.push_back('/'); break;
          case 'b': output.push_back('\b'); break;
          case 'f': output.push_back('\f'); break;
          case 'n': output.push_back('\n'); break;
          case 'r': output.push_back('\r'); break;
          case 't': output.push_back('\t'); break;
          case 'u': {
            auto codepoint = ParseUnicodeEscape();
            if (!codepoint.ok()) {
              return codepoint.status();
            }
            AppendUtf8(codepoint.value(), output);
            break;
          }
          default: return Error("invalid string escape");
        }
      }
      if (output.size() > limits_.max_string_bytes) {
        return Error("string byte limit exceeded");
      }
    }
    return Error("unterminated string");
  }

  Result<std::uint32_t> ParseUnicodeEscape() {
    auto first = ParseHex4();
    if (!first.ok()) {
      return first.status();
    }
    std::uint32_t codepoint = first.value();
    if (codepoint >= 0xD800U && codepoint <= 0xDBFFU) {
      if (position_ + 2U > input_.size() || input_[position_] != '\\' || input_[position_ + 1U] != 'u') {
        return Error("high surrogate is not followed by a low surrogate");
      }
      position_ += 2U;
      auto second = ParseHex4();
      if (!second.ok()) {
        return second.status();
      }
      if (second.value() < 0xDC00U || second.value() > 0xDFFFU) {
        return Error("invalid low surrogate");
      }
      codepoint = 0x10000U + ((codepoint - 0xD800U) << 10U) + (second.value() - 0xDC00U);
    } else if (codepoint >= 0xDC00U && codepoint <= 0xDFFFU) {
      return Error("unexpected low surrogate");
    }
    return codepoint;
  }

  Result<std::uint32_t> ParseHex4() {
    if (position_ + 4U > input_.size()) {
      return Error("incomplete unicode escape");
    }
    std::uint32_t value = 0;
    for (int index = 0; index < 4; ++index) {
      const char character = input_[position_++];
      value <<= 4U;
      if (character >= '0' && character <= '9') {
        value += static_cast<std::uint32_t>(character - '0');
      } else if (character >= 'a' && character <= 'f') {
        value += static_cast<std::uint32_t>(character - 'a' + 10);
      } else if (character >= 'A' && character <= 'F') {
        value += static_cast<std::uint32_t>(character - 'A' + 10);
      } else {
        return Error("non-hexadecimal unicode escape");
      }
    }
    return value;
  }

  static void AppendUtf8(std::uint32_t codepoint, std::string& output) {
    if (codepoint <= 0x7FU) {
      output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FFU) {
      output.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
      output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else if (codepoint <= 0xFFFFU) {
      output.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
      output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
      output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else {
      output.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
      output.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
      output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
      output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    }
  }

  static bool IsValidUtf8(std::string_view text) {
    std::size_t index = 0;
    while (index < text.size()) {
      const auto first = static_cast<unsigned char>(text[index]);
      if (first <= 0x7FU) {
        ++index;
        continue;
      }
      std::size_t continuation_count = 0;
      std::uint32_t codepoint = 0;
      std::uint32_t minimum = 0;
      if ((first & 0xE0U) == 0xC0U) {
        continuation_count = 1;
        codepoint = first & 0x1FU;
        minimum = 0x80U;
      } else if ((first & 0xF0U) == 0xE0U) {
        continuation_count = 2;
        codepoint = first & 0x0FU;
        minimum = 0x800U;
      } else if ((first & 0xF8U) == 0xF0U) {
        continuation_count = 3;
        codepoint = first & 0x07U;
        minimum = 0x10000U;
      } else {
        return false;
      }
      if (index + continuation_count >= text.size()) return false;
      for (std::size_t offset = 1; offset <= continuation_count; ++offset) {
        const auto continuation = static_cast<unsigned char>(text[index + offset]);
        if ((continuation & 0xC0U) != 0x80U) return false;
        codepoint = (codepoint << 6U) | (continuation & 0x3FU);
      }
      if (codepoint < minimum || codepoint > 0x10FFFFU || (codepoint >= 0xD800U && codepoint <= 0xDFFFU)) return false;
      index += continuation_count + 1U;
    }
    return true;
  }

  Status Error(std::string message) const {
    std::ostringstream stream;
    stream << "JSON parse error at byte " << position_ << ": " << message;
    return Status(StatusCode::kDataLoss, stream.str());
  }

  void SkipWhitespace() {
    while (position_ < input_.size()) {
      const char character = input_[position_];
      if (character != ' ' && character != '\t' && character != '\n' && character != '\r') {
        break;
      }
      ++position_;
    }
  }

  bool Consume(char expected) {
    if (position_ < input_.size() && input_[position_] == expected) {
      ++position_;
      return true;
    }
    return false;
  }

  static bool IsDigit(char value) { return value >= '0' && value <= '9'; }
  static bool IsDigit19(char value) { return value >= '1' && value <= '9'; }

  std::string_view input_;
  ParseLimits limits_;
  std::size_t position_ = 0;
  std::size_t values_seen_ = 0;
};

}  // namespace

double Value::as_number() const {
  if (is_integer()) {
    return static_cast<double>(as_integer());
  }
  return std::get<double>(storage_);
}

const Value* Value::find(std::string_view key) const {
  if (!is_object()) {
    return nullptr;
  }
  const auto iterator = as_object().find(key);
  return iterator == as_object().end() ? nullptr : &iterator->second;
}

Result<Value> Parse(std::string_view input, ParseLimits limits) {
  if (input.size() > limits.max_string_bytes * 2U) {
    return Status(StatusCode::kDataLoss, "JSON document exceeds configured byte limit");
  }
  return Parser(input, limits).Run();
}

std::string Escape(std::string_view input) {
  std::ostringstream output;
  for (const unsigned char byte : input) {
    switch (byte) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\b': output << "\\b"; break;
      case '\f': output << "\\f"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (byte < 0x20U) {
          constexpr char digits[] = "0123456789abcdef";
          output << "\\u00" << digits[byte >> 4U] << digits[byte & 0x0FU];
        } else {
          output << static_cast<char>(byte);
        }
    }
  }
  return output.str();
}

}  // namespace gem16gb::json

#include "runtime/tool_call_parser.h"

#include <cctype>
#include <charconv>
#include <string>
#include <utility>

#include "util/json.h"

namespace gem16::internal {
namespace {

constexpr std::string_view kToolOpen = "<|tool_call>";
constexpr std::string_view kCallPrefix = "call:";
constexpr std::string_view kToolClose = "<tool_call|>";
constexpr std::string_view kQuote = "<|\"|>";

Status ParseError(std::string message) { return Status(StatusCode::kDataLoss, std::move(message)); }

std::size_t RetainedPrefixSuffix(std::string_view value, std::string_view marker) {
  const std::size_t maximum = std::min(value.size(), marker.size() - 1U);
  for (std::size_t length = maximum; length != 0U; --length) {
    if (value.substr(value.size() - length) == marker.substr(0U, length)) {
      return length;
    }
  }
  return 0U;
}

class ArgumentDslParser {
 public:
  explicit ArgumentDslParser(std::string_view input) : input_(input) {}

  Result<std::string> Parse() {
    auto value = ParseValue();
    if (!value.ok()) return value.status();
    if (position_ != input_.size()) {
      return ParseError("Gemma tool arguments contain trailing data");
    }
    return value;
  }

 private:
  Result<std::string> ParseValue() {
    if (position_ >= input_.size()) {
      return ParseError("Gemma tool arguments contain an incomplete value");
    }
    if (input_.substr(position_).starts_with(kQuote)) return ParseString();
    if (input_[position_] == '{') return ParseObject();
    if (input_[position_] == '[') return ParseArray();
    const std::size_t begin = position_;
    while (position_ < input_.size() && input_[position_] != ',' && input_[position_] != '}' &&
           input_[position_] != ']') {
      ++position_;
    }
    if (begin == position_) {
      return ParseError("Gemma tool arguments contain an empty value");
    }
    const std::string_view literal = input_.substr(begin, position_ - begin);
    if (literal == "true" || literal == "false" || literal == "null") {
      return std::string(literal);
    }
    double number = 0.0;
    const auto parsed = std::from_chars(literal.data(), literal.data() + literal.size(), number);
    if (parsed.ec != std::errc{} || parsed.ptr != literal.data() + literal.size()) {
      return ParseError("Gemma tool arguments contain an unsupported literal");
    }
    return std::string(literal);
  }

  Result<std::string> ParseString() {
    position_ += kQuote.size();
    const std::size_t close = input_.find(kQuote, position_);
    if (close == std::string_view::npos) {
      return ParseError("Gemma tool arguments contain an unterminated string");
    }
    std::string result = "\"";
    result.append(json::Escape(input_.substr(position_, close - position_)));
    result.push_back('\"');
    position_ = close + kQuote.size();
    return result;
  }

  Result<std::string> ParseObject() {
    ++position_;
    std::string result = "{";
    bool first = true;
    while (position_ < input_.size() && input_[position_] != '}') {
      const std::size_t key_begin = position_;
      while (position_ < input_.size() &&
             (std::isalnum(static_cast<unsigned char>(input_[position_])) || input_[position_] == '_')) {
        ++position_;
      }
      if (key_begin == position_ || position_ >= input_.size() || input_[position_] != ':') {
        return ParseError("Gemma tool arguments contain an invalid object key");
      }
      const std::string_view key = input_.substr(key_begin, position_ - key_begin);
      ++position_;
      auto value = ParseValue();
      if (!value.ok()) return value.status();
      if (!first) result.push_back(',');
      first = false;
      result.push_back('\"');
      result.append(json::Escape(key));
      result.append("\":");
      result.append(value.value());
      if (position_ < input_.size() && input_[position_] == ',') {
        ++position_;
      } else if (position_ >= input_.size() || input_[position_] != '}') {
        return ParseError("Gemma tool arguments contain an unterminated object");
      }
    }
    if (position_ >= input_.size()) {
      return ParseError("Gemma tool arguments contain an unterminated object");
    }
    ++position_;
    result.push_back('}');
    return result;
  }

  Result<std::string> ParseArray() {
    ++position_;
    std::string result = "[";
    bool first = true;
    while (position_ < input_.size() && input_[position_] != ']') {
      auto value = ParseValue();
      if (!value.ok()) return value.status();
      if (!first) result.push_back(',');
      first = false;
      result.append(value.value());
      if (position_ < input_.size() && input_[position_] == ',') {
        ++position_;
      } else if (position_ >= input_.size() || input_[position_] != ']') {
        return ParseError("Gemma tool arguments contain an unterminated array");
      }
    }
    if (position_ >= input_.size()) {
      return ParseError("Gemma tool arguments contain an unterminated array");
    }
    ++position_;
    result.push_back(']');
    return result;
  }

  std::string_view input_;
  std::size_t position_ = 0U;
};

}  // namespace

Result<std::vector<GenerationEvent>> GemmaToolCallParser::Push(std::string_view text, bool final) {
  pending_.append(text);
  return Consume(final);
}

Result<std::vector<GenerationEvent>> GemmaToolCallParser::Consume(bool final) {
  std::vector<GenerationEvent> events;
  while (true) {
    if (state_ == State::kText) {
      const std::size_t open = pending_.find(kToolOpen);
      if (open != std::string::npos) {
        if (open != 0U) {
          GenerationEvent event;
          event.kind = GenerationEventKind::kTextDelta;
          event.text_delta = pending_.substr(0U, open);
          visible_text_.append(event.text_delta);
          events.push_back(std::move(event));
        }
        pending_.erase(0U, open + kToolOpen.size());
        state_ = State::kName;
        continue;
      }
      const std::size_t retained = final ? 0U : RetainedPrefixSuffix(pending_, kToolOpen);
      const std::size_t emitted = pending_.size() - retained;
      if (emitted != 0U) {
        GenerationEvent event;
        event.kind = GenerationEventKind::kTextDelta;
        event.text_delta = pending_.substr(0U, emitted);
        visible_text_.append(event.text_delta);
        events.push_back(std::move(event));
        pending_.erase(0U, emitted);
      }
      break;
    }

    if (state_ == State::kName) {
      if (pending_.size() < kCallPrefix.size()) {
        if (!kCallPrefix.starts_with(pending_)) {
          return ParseError("Gemma tool call is missing the call: prefix");
        }
        break;
      }
      if (!pending_.starts_with(kCallPrefix)) {
        return ParseError("Gemma tool call is missing the call: prefix");
      }
      const std::size_t brace = pending_.find('{', kCallPrefix.size());
      if (brace == std::string::npos) break;
      current_name_ = pending_.substr(kCallPrefix.size(), brace - kCallPrefix.size());
      if (current_name_.empty()) {
        return ParseError("Gemma tool call has an empty function name");
      }
      for (const char character : current_name_) {
        if (!std::isalnum(static_cast<unsigned char>(character)) && character != '_') {
          return ParseError("Gemma tool call has an invalid function name");
        }
      }
      current_id_ = "call_" + std::to_string(tool_calls_.size() + 1U);
      current_arguments_ = "{";
      pending_.erase(0U, brace + 1U);
      state_ = State::kArguments;
      GenerationEvent event;
      event.kind = GenerationEventKind::kToolCallStart;
      event.tool_call_id = current_id_;
      event.tool_name = current_name_;
      events.push_back(std::move(event));
      continue;
    }

    const std::size_t close = pending_.find(kToolClose);
    if (close == std::string::npos) {
      const std::size_t retained = RetainedPrefixSuffix(pending_, kToolClose);
      current_arguments_.append(pending_.substr(0U, pending_.size() - retained));
      pending_.erase(0U, pending_.size() - retained);
      break;
    }
    current_arguments_.append(pending_.substr(0U, close));
    pending_.erase(0U, close + kToolClose.size());
    auto arguments = ArgumentDslParser(current_arguments_).Parse();
    if (!arguments.ok()) return arguments.status();
    tool_calls_.push_back({current_id_, current_name_, arguments.value()});
    GenerationEvent arguments_event;
    arguments_event.kind = GenerationEventKind::kToolCallArgumentsDelta;
    arguments_event.text_delta = arguments.value();
    arguments_event.tool_call_id = current_id_;
    arguments_event.tool_name = current_name_;
    events.push_back(std::move(arguments_event));
    GenerationEvent end_event;
    end_event.kind = GenerationEventKind::kToolCallEnd;
    end_event.tool_call_id = current_id_;
    end_event.tool_name = current_name_;
    events.push_back(std::move(end_event));
    current_name_.clear();
    current_arguments_.clear();
    current_id_.clear();
    state_ = State::kText;
  }
  if (final && (state_ != State::kText || !pending_.empty())) {
    return ParseError("model response ends inside a Gemma tool call marker");
  }
  return events;
}

}  // namespace gem16::internal

#pragma once

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "gem16/chat.h"

namespace gem16::internal {

[[nodiscard]] Status ValidateToolDefinitions(
    std::span<const GenerationToolDefinition> tools);
[[nodiscard]] Status ValidateGeneratedToolCalls(
    std::span<const GenerationToolDefinition> tools,
    const GenerationToolChoice& tool_choice,
    std::span<const GenerationToolCall> calls);

// Incremental parser for the checkpoint-native repeatable form
// <|tool_call>call:name{...}<tool_call|>. Input chunks may split any marker,
// function name, or argument payload. Events own their text so callbacks may
// consume them after the next parser call.
class GemmaToolCallParser {
 public:
  [[nodiscard]] Result<std::vector<GenerationEvent>> Push(std::string_view text, bool final = false);
  [[nodiscard]] const std::string& visible_text() const { return visible_text_; }
  [[nodiscard]] const std::vector<GenerationToolCall>& tool_calls() const { return tool_calls_; }

 private:
  enum class State { kText, kName, kArguments };

  [[nodiscard]] Result<std::vector<GenerationEvent>> Consume(bool final);

  State state_ = State::kText;
  std::string pending_;
  std::string visible_text_;
  std::string current_name_;
  std::string current_arguments_;
  std::string current_id_;
  std::vector<GenerationToolCall> tool_calls_;
};

}  // namespace gem16::internal

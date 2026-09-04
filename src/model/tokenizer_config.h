#pragma once

#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "gem16/status.h"
#include "gem16/tokenizer.h"

namespace gem16::internal {

// Pure host renderer used by the pinned tokenizer processor and template contract tests.
[[nodiscard]] Result<std::string> RenderGemmaChat(
    std::span<const ChatMessage> messages, bool enable_thinking, bool add_generation_prompt,
    std::span<const ChatToolDefinition> tools, std::string_view thinking_open,
    std::string_view thinking_close, std::span<const std::string> content_close_tokens,
    std::string_view tool_call_start_token);

struct TokenizerConfig {
  std::string tokenizer_class;
  std::string bos_token;
  std::string eos_token;
  std::string eot_token;
  std::string tool_response_end_token;
  std::string tool_call_start_token;
  double model_max_length = 0.0;
  std::string response_role;
  std::vector<std::string> response_start_anchors;
  std::vector<std::string> content_close_tokens;
  std::string thinking_open;
  std::string thinking_close;
  std::string tool_call_open_pattern;
  std::string tool_call_close;
  bool tool_calls_repeat = false;
};

[[nodiscard]] Result<TokenizerConfig> LoadTokenizerConfig(const std::filesystem::path& path);
[[nodiscard]] Status ValidatePrimaryTokenizerConfig(const TokenizerConfig& config);
[[nodiscard]] Result<std::string> ExtractResponseContent(std::string_view text, std::string_view thinking_open,
                                                         std::string_view thinking_close,
                                                         std::span<const std::string> content_close_tokens,
                                                         std::string_view tool_call_start_token);
[[nodiscard]] Result<std::string> RenderGemmaToolDefinition(std::string_view name, std::string_view description,
                                                            std::string_view parameters_json);
[[nodiscard]] Result<std::string> RenderGemmaToolCall(std::string_view name, std::string_view arguments_json);

}  // namespace gem16::internal

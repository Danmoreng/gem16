#pragma once

#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "gem16gb/status.h"

namespace gem16gb::internal {

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

[[nodiscard]] Result<TokenizerConfig> LoadTokenizerConfig(
    const std::filesystem::path& path);
[[nodiscard]] Status ValidatePrimaryTokenizerConfig(
    const TokenizerConfig& config);
[[nodiscard]] Result<std::string> ExtractResponseContent(
    std::string_view text, std::string_view thinking_open,
    std::string_view thinking_close,
    std::span<const std::string> content_close_tokens,
    std::string_view tool_call_start_token);

}  // namespace gem16gb::internal

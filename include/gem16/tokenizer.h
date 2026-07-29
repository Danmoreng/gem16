#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "gem16/sampling.h"
#include "gem16/status.h"

namespace gem16 {

struct ChatMessage {
  std::string role;
  std::string content;
};

struct GenerationTokenControls {
  std::vector<std::uint32_t> stop_token_ids;
  std::vector<std::uint32_t> suppressed_token_ids;
  std::vector<std::uint32_t> thinking_open_token_ids;
  std::uint32_t thinking_close_token_id = 0U;
  SamplingOptions recommended_sampling;
};

enum class ResponseTokenChannel {
  kControl,
  kReasoning,
  kText,
};

// Allocation-free per-token recognizer for the checkpoint-qualified thinking
// channel. Construct it before generation from GenerationTokenControls.
class ResponseChannelTracker {
 public:
  explicit ResponseChannelTracker(const GenerationTokenControls& controls)
      : thinking_open_token_ids_(controls.thinking_open_token_ids),
        thinking_close_token_id_(controls.thinking_close_token_id) {}

  [[nodiscard]] ResponseTokenChannel Observe(std::uint32_t token_id);
  [[nodiscard]] bool in_reasoning() const { return in_reasoning_; }
  [[nodiscard]] std::uint64_t reasoning_token_count() const {
    return reasoning_token_count_;
  }

 private:
  std::vector<std::uint32_t> thinking_open_token_ids_;
  std::uint32_t thinking_close_token_id_ = 0U;
  std::size_t open_match_length_ = 0U;
  std::uint64_t reasoning_token_count_ = 0U;
  bool in_reasoning_ = false;
};

class Tokenizer {
 public:
  struct Impl;

  [[nodiscard]] static Result<Tokenizer> Load(const std::filesystem::path& tokenizer_json);
  [[nodiscard]] Result<std::vector<std::uint32_t>> Encode(std::string_view text) const;
  [[nodiscard]] Result<std::string> Decode(std::span<const std::uint32_t> token_ids,
                                           bool skip_special_tokens) const;
  [[nodiscard]] Status WriteDecodedToken(std::uint32_t token_id,
                                         bool skip_special_tokens,
                                         std::ostream& output) const;

 private:
  explicit Tokenizer(std::shared_ptr<const Impl> implementation)
      : implementation_(std::move(implementation)) {}

  std::shared_ptr<const Impl> implementation_;
};

class GemmaChatProcessor {
 public:
  [[nodiscard]] static Result<GemmaChatProcessor> Load(
      const std::filesystem::path& model_directory);

  [[nodiscard]] Result<std::string> Render(
      std::span<const ChatMessage> messages, bool enable_thinking,
      bool add_generation_prompt = true) const;
  [[nodiscard]] Result<std::vector<std::uint32_t>> Encode(
      std::span<const ChatMessage> messages, bool enable_thinking,
      bool add_generation_prompt = true) const;
  [[nodiscard]] Result<std::vector<std::uint32_t>> EncodeContinuation(
      std::string_view user_content, bool enable_thinking) const;
  [[nodiscard]] Result<std::string> Decode(std::span<const std::uint32_t> token_ids,
                                           bool skip_special_tokens) const;
  [[nodiscard]] Result<std::string> DecodeResponseText(
      std::span<const std::uint32_t> token_ids) const;
  [[nodiscard]] Status WriteDecodedToken(std::uint32_t token_id,
                                         bool skip_special_tokens,
                                         std::ostream& output) const;

  [[nodiscard]] const GenerationTokenControls& generation_controls() const {
    return generation_controls_;
  }

 private:
  GemmaChatProcessor(Tokenizer tokenizer, GenerationTokenControls controls,
                     std::string thinking_open, std::string thinking_close,
                     std::vector<std::string> content_close_tokens,
                     std::string tool_call_start_token)
      : tokenizer_(std::move(tokenizer)),
        generation_controls_(std::move(controls)),
        thinking_open_(std::move(thinking_open)),
        thinking_close_(std::move(thinking_close)),
        content_close_tokens_(std::move(content_close_tokens)),
        tool_call_start_token_(std::move(tool_call_start_token)) {}

  Tokenizer tokenizer_;
  GenerationTokenControls generation_controls_;
  std::string thinking_open_;
  std::string thinking_close_;
  std::vector<std::string> content_close_tokens_;
  std::string tool_call_start_token_;
};

}  // namespace gem16

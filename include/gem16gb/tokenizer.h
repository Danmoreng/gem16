#pragma once

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "gem16gb/status.h"

namespace gem16gb {

struct ChatMessage {
  std::string role;
  std::string content;
};

struct GenerationTokenControls {
  std::vector<std::uint32_t> stop_token_ids;
  std::vector<std::uint32_t> suppressed_token_ids;
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
  [[nodiscard]] Status WriteDecodedToken(std::uint32_t token_id,
                                         bool skip_special_tokens,
                                         std::ostream& output) const;

  [[nodiscard]] const GenerationTokenControls& generation_controls() const {
    return generation_controls_;
  }

 private:
  GemmaChatProcessor(Tokenizer tokenizer, GenerationTokenControls controls)
      : tokenizer_(std::move(tokenizer)), generation_controls_(std::move(controls)) {}

  Tokenizer tokenizer_;
  GenerationTokenControls generation_controls_;
};

}  // namespace gem16gb

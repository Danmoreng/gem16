#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "gem16/chat.h"

namespace gem16::server {

struct OpenAiChatRequest {
  std::string model;
  ChatGenerationRequest generation;
  bool stream = false;
  bool include_usage = false;
};

struct OpenAiResponseIdentity {
  std::string id;
  std::string model;
  std::int64_t created = 0;
};

struct OpenAiChatAdapterOptions {
  std::uint64_t context_tokens = 8192U;
};

[[nodiscard]] Result<OpenAiChatRequest> ParseChatCompletionsRequest(
    std::string_view body,
    const OpenAiChatAdapterOptions& options = {});
[[nodiscard]] std::string ChatCompletionJson(
    const OpenAiResponseIdentity& identity,
    const ChatGenerationResponse& response);
[[nodiscard]] std::string ChatCompletionChunkJson(
    const OpenAiResponseIdentity& identity, std::string_view delta_json,
    std::optional<GenerationFinishReason> finish_reason = std::nullopt,
    const ChatGenerationResponse* usage = nullptr);
[[nodiscard]] std::string OpenAiErrorJson(std::string_view message,
                                          std::string_view type);

}  // namespace gem16::server

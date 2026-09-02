#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

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
  std::optional<std::int64_t> completed;
};

struct OpenAiChatAdapterOptions {
  std::uint64_t context_tokens = 8192U;
  bool gemma4_moe26b_vision = false;
  std::uint32_t vision_max_soft_token_budget = 280U;
};

struct OpenAiResponsesRequest {
  std::string model;
  ChatGenerationRequest generation;
  std::optional<std::string> previous_response_id;
  std::optional<std::string> instructions;
  bool stream = false;
  bool store = true;
  bool tools_present = false;
  bool tool_choice_present = false;
};

[[nodiscard]] Result<OpenAiChatRequest> ParseChatCompletionsRequest(
    std::string_view body,
    const OpenAiChatAdapterOptions& options = {});
[[nodiscard]] Result<OpenAiResponsesRequest> ParseResponsesRequest(
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
                                          std::string_view type,
                                          std::string_view code = {});
[[nodiscard]] std::string ResponseJson(
    const OpenAiResponseIdentity& identity,
    const OpenAiResponsesRequest& request,
    const ChatGenerationResponse& response);
[[nodiscard]] std::string ResponseShellJson(
    const OpenAiResponseIdentity& identity,
    const OpenAiResponsesRequest& request, std::string_view status,
    const ChatGenerationResponse* response = nullptr);

}  // namespace gem16::server

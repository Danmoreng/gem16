#include "gem16/chat.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <span>
#include <utility>

namespace gem16 {

std::uint64_t ThinkingBudgetTokens(ThinkingEffort effort) {
  switch (effort) {
    case ThinkingEffort::kOff: return 0U;
    case ThinkingEffort::kSmall: return 1024U;
    case ThinkingEffort::kMedium: return 4096U;
    case ThinkingEffort::kHigh: return 8192U;
  }
  return 0U;
}

const char* ThinkingEffortName(ThinkingEffort effort) {
  switch (effort) {
    case ThinkingEffort::kOff: return "off";
    case ThinkingEffort::kSmall: return "small";
    case ThinkingEffort::kMedium: return "medium";
    case ThinkingEffort::kHigh: return "high";
  }
  return "unknown";
}

namespace {

Result<std::string> TextContent(const GenerationMessage& message) {
  if (message.content.empty()) {
    return Status(StatusCode::kInvalidArgument,
                  "generation message content must not be empty");
  }
  std::string text;
  for (const GenerationContentPart& part : message.content) {
    if (part.kind != GenerationContentKind::kText) {
      return Status(StatusCode::kUnsupported,
                    "only text generation content is currently supported");
    }
    text.append(part.text);
  }
  return text;
}

Result<std::vector<ChatMessage>> MaterializeTextMessages(
    std::span<const GenerationMessage> messages) {
  if (messages.empty()) {
    return Status(StatusCode::kInvalidArgument,
                  "generation request requires at least one message");
  }
  std::vector<ChatMessage> materialized;
  materialized.reserve(messages.size());
  for (const GenerationMessage& message : messages) {
    auto content = TextContent(message);
    if (!content.ok()) return content.status();
    materialized.push_back({message.role, std::move(content).value()});
  }
  return materialized;
}

struct EventBridge {
  GenerationEventCallback callback = nullptr;
  void* context = nullptr;
};

Status ForwardTokenEvent(void* opaque_context, std::uint32_t token_id) {
  auto* bridge = static_cast<EventBridge*>(opaque_context);
  if (bridge == nullptr || bridge->callback == nullptr) {
    return Status(StatusCode::kInternal,
                  "generation event bridge is not initialized");
  }
  const GenerationEvent event{GenerationEventKind::kToken, token_id};
  return bridge->callback(bridge->context, event);
}

}  // namespace

struct ChatSession::Impl {
  Impl(GemmaChatProcessor chat_processor,
       ConversationSession conversation_session)
      : processor(std::move(chat_processor)),
        session(std::move(conversation_session)) {}

  GemmaChatProcessor processor;
  ConversationSession session;
  std::vector<GenerationMessage> committed_messages;
  std::vector<std::uint32_t> cached_prefix_token_ids;
  std::optional<std::uint32_t> pending_assistant_token_id;
  std::uint64_t max_context_tokens = 0U;
};

ChatSession::ChatSession(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ChatSession::ChatSession(ChatSession&&) noexcept = default;
ChatSession& ChatSession::operator=(ChatSession&&) noexcept = default;
ChatSession::~ChatSession() = default;

Result<ChatSession> ChatSession::Create(const ChatSessionOptions& options) {
  auto processor = GemmaChatProcessor::Load(options.model_directory);
  if (!processor.ok()) return processor.status();
  return Create(options, std::move(processor).value());
}

Result<ChatSession> ChatSession::Create(const ChatSessionOptions& options,
                                        GemmaChatProcessor processor) {
  ConversationSessionOptions session_options;
  session_options.model_directory = options.model_directory;
  session_options.assistant_model_directory =
      options.assistant_model_directory;
  session_options.stop_token_ids =
      processor.generation_controls().stop_token_ids;
  session_options.suppressed_token_ids =
      processor.generation_controls().suppressed_token_ids;
  session_options.max_context_tokens = options.max_context_tokens;
  session_options.kv_cache_mode = options.kv_cache_mode;
  session_options.sampling = options.sampling;
  session_options.mtp_draft_tokens = options.mtp_draft_tokens;
  session_options.mtp_adaptive = options.mtp_adaptive;
  auto session = ConversationSession::Create(session_options);
  if (!session.ok()) return session.status();

  auto impl = std::make_unique<Impl>(std::move(processor),
                                     std::move(session).value());
  impl->cached_prefix_token_ids.reserve(
      static_cast<std::size_t>(options.max_context_tokens));
  impl->max_context_tokens = options.max_context_tokens;
  return ChatSession(std::move(impl));
}

Result<ChatGenerationResponse> ChatSession::Generate(
    const ChatGenerationRequest& request, GenerationEventCallback callback,
    void* callback_context) {
  if (impl_ == nullptr) {
    return Status(StatusCode::kInternal, "chat session was moved from");
  }
  if (request.max_generated_tokens.has_value() &&
      *request.max_generated_tokens == 0U) {
    return Status(StatusCode::kInvalidArgument,
                  "generation token limit must be positive when specified");
  }
  auto messages = MaterializeTextMessages(request.messages);
  if (!messages.ok()) return messages.status();
  if (messages.value().back().role != "user") {
    return Status(StatusCode::kInvalidArgument,
                  "generation request must end with a user message");
  }
  if (!impl_->committed_messages.empty()) {
    if (request.messages.size() != impl_->committed_messages.size() + 1U ||
        !std::equal(impl_->committed_messages.begin(),
                    impl_->committed_messages.end(),
                    request.messages.begin())) {
      return Status(
          StatusCode::kInvalidArgument,
          "generation request does not exactly extend the resident conversation");
    }
  }

  Result<std::vector<std::uint32_t>> prompt_ids = [&]() {
    if (impl_->cached_prefix_token_ids.empty()) {
      return impl_->processor.Encode(
          messages.value(), request.thinking.effort != ThinkingEffort::kOff);
    }
    auto continuation = impl_->processor.EncodeContinuation(
        messages.value().back().content,
        request.thinking.effort != ThinkingEffort::kOff);
    if (!continuation.ok()) {
      return Result<std::vector<std::uint32_t>>(continuation.status());
    }
    std::vector<std::uint32_t> token_ids =
        impl_->cached_prefix_token_ids;
    if (impl_->pending_assistant_token_id.has_value()) {
      token_ids.push_back(*impl_->pending_assistant_token_id);
    }
    token_ids.insert(token_ids.end(), continuation.value().begin(),
                     continuation.value().end());
    return Result<std::vector<std::uint32_t>>(std::move(token_ids));
  }();
  if (!prompt_ids.ok()) return prompt_ids.status();
  if (prompt_ids.value().size() > impl_->max_context_tokens) {
    return Status(StatusCode::kInvalidArgument,
                  "conversation prompt exceeds the session context capacity");
  }
  const std::uint64_t remaining_output_capacity =
      impl_->max_context_tokens - prompt_ids.value().size() + 1U;
  const std::uint64_t max_generated_tokens =
      request.max_generated_tokens.value_or(remaining_output_capacity);
  if (max_generated_tokens > remaining_output_capacity) {
    return Status(StatusCode::kInvalidArgument,
                  "requested output exceeds the remaining context capacity");
  }

  ReasoningTokenOptions reasoning;
  reasoning.enabled = request.thinking.effort != ThinkingEffort::kOff;
  if (reasoning.enabled) {
    reasoning.channel_open_token_ids =
        impl_->processor.generation_controls().thinking_open_token_ids;
    reasoning.channel_close_token_id =
        impl_->processor.generation_controls().thinking_close_token_id;
    const std::uint64_t answer_reserve =
        max_generated_tokens > 128U ? 128U : max_generated_tokens / 2U;
    reasoning.max_reasoning_tokens = std::min(
        ThinkingBudgetTokens(request.thinking.effort),
        std::max<std::uint64_t>(1U, max_generated_tokens - answer_reserve));
  }

  EventBridge bridge{callback, callback_context};
  auto inference = impl_->session.Generate(
      prompt_ids.value(), max_generated_tokens, reasoning,
      callback == nullptr ? nullptr : ForwardTokenEvent,
      callback == nullptr ? nullptr : &bridge);
  if (!inference.ok()) return inference.status();

  impl_->cached_prefix_token_ids = prompt_ids.value();
  if (inference.value().output_token_ids.size() > 1U) {
    impl_->cached_prefix_token_ids.insert(
        impl_->cached_prefix_token_ids.end(),
        inference.value().output_token_ids.begin(),
        inference.value().output_token_ids.end() - 1);
  }
  impl_->pending_assistant_token_id.reset();
  if (!inference.value().stopped &&
      !inference.value().output_token_ids.empty()) {
    impl_->pending_assistant_token_id =
        inference.value().output_token_ids.back();
  }

  std::vector<std::uint32_t> content_ids =
      inference.value().output_token_ids;
  const auto& stop_tokens =
      impl_->processor.generation_controls().stop_token_ids;
  if (!content_ids.empty() &&
      std::find(stop_tokens.begin(), stop_tokens.end(), content_ids.back()) !=
          stop_tokens.end()) {
    content_ids.pop_back();
  }
  auto assistant_content = impl_->processor.Decode(content_ids, false);
  if (!assistant_content.ok()) return assistant_content.status();
  auto assistant_text = impl_->processor.DecodeResponseText(content_ids);
  if (!assistant_text.ok()) return assistant_text.status();

  ChatGenerationResponse response;
  response.assistant_content = std::move(assistant_content).value();
  response.assistant_text = std::move(assistant_text).value();
  response.prompt_token_ids = std::move(prompt_ids).value();
  response.finish_reason = inference.value().stopped
                               ? GenerationFinishReason::kStop
                               : GenerationFinishReason::kLength;
  response.inference = std::move(inference).value();

  impl_->committed_messages = request.messages;
  impl_->committed_messages.push_back(GenerationMessage::Text(
      "assistant", response.assistant_content));
  return response;
}

std::uint64_t ChatSession::cached_token_count() const {
  return impl_ == nullptr ? 0U : impl_->session.cached_token_count();
}

const char* GenerationFinishReasonName(GenerationFinishReason reason) {
  switch (reason) {
    case GenerationFinishReason::kStop: return "stop";
    case GenerationFinishReason::kLength: return "length";
  }
  return "unknown";
}

}  // namespace gem16

#include "gem16/chat.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
#include <utility>

#include "runtime/tool_call_parser.h"
#include "runtime/chat_internal.h"

namespace gem16 {

std::uint64_t ThinkingBudgetTokens(ThinkingEffort effort) {
  switch (effort) {
    case ThinkingEffort::kOff:
      return 0U;
    case ThinkingEffort::kSmall:
      return 1024U;
    case ThinkingEffort::kMedium:
      return 4096U;
    case ThinkingEffort::kHigh:
      return 8192U;
  }
  return 0U;
}

const char* ThinkingEffortName(ThinkingEffort effort) {
  switch (effort) {
    case ThinkingEffort::kOff:
      return "off";
    case ThinkingEffort::kSmall:
      return "small";
    case ThinkingEffort::kMedium:
      return "medium";
    case ThinkingEffort::kHigh:
      return "high";
  }
  return "unknown";
}

namespace {

constexpr std::size_t kAudioSamplesPerToken = 640U;
constexpr std::size_t kMaximumAudioTokens = 750U;

std::string_view TrimAsciiWhitespace(std::string_view value) {
  std::size_t begin = 0U;
  std::size_t end = value.size();
  while (begin < end &&
         std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
    ++begin;
  }
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(value[end - 1U])) != 0) {
    --end;
  }
  return value.substr(begin, end - begin);
}

struct MaterializedMessages {
  std::vector<ChatMessage> messages;
  std::vector<std::vector<float>> audio_frames;
  std::vector<VisionImage> images;
  std::vector<Gemma4Moe26BVisionImage> moe26b_images;
};

Result<std::string> MaterializeContent(const GenerationMessage& message, ChatMessage& chat_message,
                                       std::vector<std::vector<float>>& audio_frames,
                                       std::vector<VisionImage>& images,
                                       std::vector<Gemma4Moe26BVisionImage>&
                                           moe26b_images) {
  if (message.content.empty()) {
    return Status(StatusCode::kInvalidArgument, "generation message content must not be empty");
  }
  std::string text;
  for (const GenerationContentPart& part : message.content) {
    if (part.kind == GenerationContentKind::kText) {
      text.append(part.text);
      continue;
    }
    if (part.kind == GenerationContentKind::kToolCall) {
      if (message.role != "assistant") {
        return Status(StatusCode::kInvalidArgument, "tool calls are supported only in assistant messages");
      }
      chat_message.tool_calls.push_back({part.tool_call.id, part.tool_call.name, part.tool_call.arguments_json});
      continue;
    }
    if (part.kind == GenerationContentKind::kToolResult) {
      if (message.role != "tool" || message.content.size() != 1U) {
        return Status(StatusCode::kInvalidArgument, "a tool message must contain exactly one tool result");
      }
      chat_message.tool_call_id = part.tool_result.call_id;
      text.append(part.tool_result.output);
      continue;
    }
    if (part.kind == GenerationContentKind::kImage) {
      if (message.role != "user") {
        return Status(StatusCode::kInvalidArgument, "image content is supported only in user messages");
      }
      if (part.image.patch_count == 0U || part.image.patch_count > 280U ||
          part.image.patches.size() != static_cast<std::size_t>(part.image.patch_count) * 6912U ||
          part.image.positions.size() != static_cast<std::size_t>(part.image.patch_count) * 2U) {
        return Status(StatusCode::kInvalidArgument, "image content has invalid processed patch geometry");
      }
      images.push_back(part.image);
      text.append("<|image>");
      for (std::uint32_t patch = 0U; patch < part.image.patch_count; ++patch) {
        text.append("<|image|>");
      }
      text.append("<image|>");
      continue;
    }
    if (part.kind == GenerationContentKind::kGemma4Moe26BImage) {
      if (message.role != "user") {
        return Status(StatusCode::kInvalidArgument,
                      "image content is supported only in user messages");
      }
      const auto& image = part.moe26b_image;
      if (image.soft_token_count == 0U || image.soft_token_count > 280U ||
          image.raw_patch_count != image.soft_token_count * 9U ||
          image.patches.size() !=
              static_cast<std::size_t>(image.raw_patch_count) * 768U ||
          image.positions.size() !=
              static_cast<std::size_t>(image.raw_patch_count) * 2U) {
        return Status(StatusCode::kInvalidArgument,
                      "Gemma 4 26B image has invalid processed patch geometry");
      }
      moe26b_images.push_back(image);
      text.append("<|image>");
      for (std::uint32_t token = 0U; token < image.soft_token_count; ++token) {
        text.append("<|image|>");
      }
      text.append("<image|>");
      continue;
    }
    if (part.kind != GenerationContentKind::kAudio) {
      return Status(StatusCode::kUnsupported, "generation content kind is unsupported");
    }
    if (message.role != "user") {
      return Status(StatusCode::kInvalidArgument, "audio content is supported only in user messages");
    }
    if (part.audio.sample_rate != 16000U || part.audio.samples.empty()) {
      return Status(StatusCode::kInvalidArgument, "audio content must contain mono 16 kHz samples");
    }
    if (!std::all_of(part.audio.samples.begin(), part.audio.samples.end(),
                     [](float sample) { return std::isfinite(sample); })) {
      return Status(StatusCode::kInvalidArgument, "audio content contains a non-finite sample");
    }
    const std::size_t frame_count = (part.audio.samples.size() + kAudioSamplesPerToken - 1U) / kAudioSamplesPerToken;
    if (frame_count == 0U || frame_count > kMaximumAudioTokens) {
      return Status(StatusCode::kUnsupported, "audio content must be at most 30 seconds");
    }
    std::vector<float> padded(frame_count * kAudioSamplesPerToken, 0.0F);
    std::copy(part.audio.samples.begin(), part.audio.samples.end(), padded.begin());
    audio_frames.push_back(std::move(padded));
    text.append("<|audio>");
    for (std::size_t frame = 0U; frame < frame_count; ++frame) {
      text.append("<|audio|>");
    }
    text.append("<audio|>");
  }
  return text;
}

Result<MaterializedMessages> MaterializeMessages(
    std::span<const GenerationMessage> messages,
    std::span<const GenerationMessage> canonical_prefix = {}) {
  if (messages.empty()) {
    return Status(StatusCode::kInvalidArgument, "generation request requires at least one message");
  }
  MaterializedMessages materialized;
  materialized.messages.reserve(messages.size());
  for (std::size_t index = 0U; index < messages.size(); ++index) {
    const GenerationMessage& message =
        index < canonical_prefix.size() ? canonical_prefix[index]
                                        : messages[index];
    ChatMessage chat_message;
    chat_message.role = message.role;
    auto content = MaterializeContent(
        message, chat_message, materialized.audio_frames,
        materialized.images, materialized.moe26b_images);
    if (!content.ok()) return content.status();
    chat_message.content = std::move(content).value();
    materialized.messages.push_back(std::move(chat_message));
  }
  return materialized;
}

Result<std::vector<AudioEmbeddingSegment>> LocateAudioSegments(std::span<const std::uint32_t> prompt_ids,
                                                               const std::vector<std::vector<float>>& audio_frames) {
  std::vector<AudioEmbeddingSegment> segments;
  segments.reserve(audio_frames.size());
  std::size_t search = 0U;
  for (const auto& frames : audio_frames) {
    const std::size_t frame_count = frames.size() / kAudioSamplesPerToken;
    const auto boa = std::find(prompt_ids.begin() + search, prompt_ids.end(), 256000U);
    if (boa == prompt_ids.end()) {
      return Status(StatusCode::kDataLoss, "rendered prompt is missing the audio boundary token");
    }
    const std::size_t first = static_cast<std::size_t>(boa - prompt_ids.begin()) + 1U;
    if (frame_count > prompt_ids.size() - first) {
      return Status(StatusCode::kDataLoss, "rendered audio placeholder extent is truncated");
    }
    for (std::size_t frame = 0U; frame < frame_count; ++frame) {
      if (prompt_ids[first + frame] != 258881U) {
        return Status(StatusCode::kDataLoss, "rendered audio placeholder count is inconsistent");
      }
    }
    if (first + frame_count >= prompt_ids.size() || prompt_ids[first + frame_count] != 258883U) {
      return Status(StatusCode::kDataLoss, "rendered prompt is missing the audio end token");
    }
    segments.push_back(AudioEmbeddingSegment{first, frames});
    search = first + frame_count + 1U;
  }
  return segments;
}

Result<std::vector<VisionEmbeddingSegment>> LocateVisionSegments(std::span<const std::uint32_t> prompt_ids,
                                                                 const std::vector<VisionImage>& images) {
  std::vector<VisionEmbeddingSegment> segments;
  segments.reserve(images.size());
  std::size_t search = 0U;
  for (const VisionImage& image : images) {
    const auto begin = std::find(prompt_ids.begin() + search, prompt_ids.end(), 255999U);
    if (begin == prompt_ids.end()) {
      return Status(StatusCode::kDataLoss, "rendered prompt is missing the image boundary token");
    }
    const std::size_t first = static_cast<std::size_t>(begin - prompt_ids.begin()) + 1U;
    if (image.patch_count > prompt_ids.size() - first) {
      return Status(StatusCode::kDataLoss, "rendered image placeholder extent is truncated");
    }
    for (std::size_t patch = 0U; patch < image.patch_count; ++patch) {
      if (prompt_ids[first + patch] != 258880U) {
        return Status(StatusCode::kDataLoss, "rendered image placeholder count is inconsistent");
      }
    }
    if (first + image.patch_count >= prompt_ids.size() || prompt_ids[first + image.patch_count] != 258882U) {
      return Status(StatusCode::kDataLoss, "rendered prompt is missing the image end token");
    }
    segments.push_back(VisionEmbeddingSegment{first, image.patches, image.positions});
    search = first + image.patch_count + 1U;
  }
  return segments;
}

Result<std::vector<Gemma4Moe26BVisionInputSegment>>
LocateGemma4Moe26BVisionSegments(
    std::span<const std::uint32_t> prompt_ids,
    const std::vector<Gemma4Moe26BVisionImage>& images) {
  std::vector<Gemma4Moe26BVisionInputSegment> segments;
  segments.reserve(images.size());
  std::size_t search = 0U;
  for (const auto& image : images) {
    const auto begin =
        std::find(prompt_ids.begin() + search, prompt_ids.end(), 255999U);
    if (begin == prompt_ids.end()) {
      return Status(StatusCode::kDataLoss,
                    "rendered prompt is missing the image boundary token");
    }
    const std::size_t first =
        static_cast<std::size_t>(begin - prompt_ids.begin()) + 1U;
    if (image.soft_token_count > prompt_ids.size() - first) {
      return Status(StatusCode::kDataLoss,
                    "rendered image placeholder extent is truncated");
    }
    for (std::uint32_t token = 0U; token < image.soft_token_count; ++token) {
      if (prompt_ids[first + token] != 258880U) {
        return Status(StatusCode::kDataLoss,
                      "rendered image placeholder count is inconsistent");
      }
    }
    if (first + image.soft_token_count >= prompt_ids.size() ||
        prompt_ids[first + image.soft_token_count] != 258882U) {
      return Status(StatusCode::kDataLoss,
                    "rendered prompt is missing the image end token");
    }
    segments.push_back(Gemma4Moe26BVisionInputSegment{
        first, image.soft_token_count, image.soft_token_budget,
        image.raw_patch_count, image.patches, image.positions});
    search = first + image.soft_token_count + 1U;
  }
  return segments;
}

struct EventBridge {
  GenerationEventCallback callback = nullptr;
  void* context = nullptr;
};

Status ForwardTokenEvent(void* opaque_context, std::uint32_t token_id) {
  auto* bridge = static_cast<EventBridge*>(opaque_context);
  if (bridge == nullptr || bridge->callback == nullptr) {
    return Status(StatusCode::kInternal, "generation event bridge is not initialized");
  }
  GenerationEvent event;
  event.kind = GenerationEventKind::kToken;
  event.token_id = token_id;
  return bridge->callback(bridge->context, event);
}

}  // namespace

namespace internal {

namespace {

bool StructurallyEquivalent(const VisionImage& left,
                            const VisionImage& right) {
  return left.patches == right.patches &&
         left.positions == right.positions &&
         left.patch_count == right.patch_count &&
         left.source_width == right.source_width &&
         left.source_height == right.source_height &&
         left.processed_width == right.processed_width &&
         left.processed_height == right.processed_height &&
         left.soft_token_budget == right.soft_token_budget;
}

bool StructurallyEquivalent(const Gemma4Moe26BVisionImage& left,
                            const Gemma4Moe26BVisionImage& right) {
  return left.patches == right.patches &&
         left.positions == right.positions &&
         left.raw_patch_count == right.raw_patch_count &&
         left.soft_token_count == right.soft_token_count &&
         left.source_width == right.source_width &&
         left.source_height == right.source_height &&
         left.processed_width == right.processed_width &&
         left.processed_height == right.processed_height &&
         left.soft_token_budget == right.soft_token_budget;
}

template <typename Image>
bool ResidentImageEquivalent(const Image& left, const Image& right) {
  if (left.source_identity.available() &&
      right.source_identity.available()) {
    return left.source_identity == right.source_identity;
  }
  return StructurallyEquivalent(left, right);
}

}  // namespace

bool ResidentMessageEquivalent(const GenerationMessage& cached,
                               const GenerationMessage& supplied) {
  if (cached.role != supplied.role ||
      cached.content.size() != supplied.content.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < cached.content.size(); ++index) {
    const GenerationContentPart& left = cached.content[index];
    const GenerationContentPart& right = supplied.content[index];
    if (left.kind != right.kind) return false;
    if (left.kind == GenerationContentKind::kImage) {
      if (!ResidentImageEquivalent(left.image, right.image)) return false;
      continue;
    }
    if (left.kind == GenerationContentKind::kGemma4Moe26BImage) {
      if (!ResidentImageEquivalent(left.moe26b_image,
                                   right.moe26b_image)) {
        return false;
      }
      continue;
    }
    // DecodeResponseText stores the authoritative generated assistant text
    // without surrounding whitespace, while the streaming API necessarily
    // emits token bytes before it knows which trailing bytes will be trimmed.
    // Treat the API-visible round trip as equivalent to the resident value.
    if (cached.role == "assistant" &&
        left.kind == GenerationContentKind::kText) {
      if (TrimAsciiWhitespace(left.text) !=
          TrimAsciiWhitespace(right.text)) {
        return false;
      }
      continue;
    }
    if (!(left == right)) return false;
  }
  return true;
}

ResponseTokenChannel ProjectResponseChannel(
    ResponseTokenChannel channel, bool flatten_reasoning_to_text) {
  return flatten_reasoning_to_text &&
                 channel == ResponseTokenChannel::kReasoning
             ? ResponseTokenChannel::kText
             : channel;
}

std::vector<std::uint32_t> ExtractReasoningTokenIds(
    std::span<const std::uint32_t> token_ids,
    const GenerationTokenControls& controls,
    bool starts_in_reasoning) {
  std::vector<std::uint32_t> reasoning_ids;
  ResponseChannelTracker response_channels(controls, starts_in_reasoning);
  for (const std::uint32_t token_id : token_ids) {
    if (response_channels.Observe(token_id) ==
        ResponseTokenChannel::kReasoning) {
      reasoning_ids.push_back(token_id);
    }
  }
  return reasoning_ids;
}

std::vector<std::uint32_t> ExtractVisibleTokenIds(
    std::span<const std::uint32_t> token_ids,
    const GenerationTokenControls& controls,
    bool starts_in_reasoning,
  bool flatten_reasoning_to_text) {
  std::vector<std::uint32_t> visible_ids;
  ResponseChannelTracker response_channels(controls, starts_in_reasoning);
  response_channels.SetSuppressAdditionalReasoning(
      !flatten_reasoning_to_text);
  for (const std::uint32_t token_id : token_ids) {
    const ResponseTokenChannel channel = ProjectResponseChannel(
        response_channels.Observe(token_id), flatten_reasoning_to_text);
    if (channel == ResponseTokenChannel::kText) {
      visible_ids.push_back(token_id);
    }
  }
  return visible_ids;
}

}  // namespace internal

struct ChatSession::Impl {
  Impl(GemmaChatProcessor chat_processor, ConversationSession conversation_session)
      : processor(std::move(chat_processor)), session(std::move(conversation_session)) {}

  GemmaChatProcessor processor;
  ConversationSession session;
  std::vector<GenerationMessage> committed_messages;
  std::vector<GenerationToolDefinition> committed_tools;
  GenerationToolChoice committed_tool_choice;
  std::vector<std::uint32_t> cached_prefix_token_ids;
  std::optional<std::uint32_t> pending_assistant_token_id;
  std::uint64_t max_context_tokens = 0U;
  bool poisoned = false;
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

Result<ChatSession> ChatSession::Create(const ChatSessionOptions& options, GemmaChatProcessor processor) {
  auto runtime = ModelRuntime::Load(
      {options.model_directory, options.assistant_model_directory,
       options.max_context_tokens, 0, true,
       options.vision_model_directory,
       options.vision_max_soft_token_budget});
  if (!runtime.ok()) return runtime.status();
  return Create(std::move(runtime).value(), options, std::move(processor));
}

Result<ChatSession> ChatSession::Create(
    std::shared_ptr<ModelRuntime> runtime, const ChatSessionOptions& options,
    GemmaChatProcessor processor) {
  ConversationSessionOptions session_options;
  session_options.model_directory = options.model_directory;
  session_options.assistant_model_directory = options.assistant_model_directory;
  session_options.vision_model_directory = options.vision_model_directory;
  session_options.vision_max_soft_token_budget =
      options.vision_max_soft_token_budget;
  session_options.stop_token_ids = processor.generation_controls().stop_token_ids;
  session_options.suppressed_token_ids = processor.generation_controls().suppressed_token_ids;
  session_options.max_context_tokens = options.max_context_tokens;
  session_options.kv_cache_mode = options.kv_cache_mode;
  session_options.sampling = options.sampling;
  session_options.mtp_draft_tokens = options.mtp_draft_tokens;
  session_options.mtp_adaptive = options.mtp_adaptive;
  auto session = ConversationSession::Create(std::move(runtime), session_options);
  if (!session.ok()) return session.status();

  auto impl = std::make_unique<Impl>(std::move(processor), std::move(session).value());
  impl->cached_prefix_token_ids.reserve(static_cast<std::size_t>(options.max_context_tokens));
  impl->max_context_tokens = options.max_context_tokens;
  return ChatSession(std::move(impl));
}

Result<ChatGenerationResponse> ChatSession::Generate(const ChatGenerationRequest& request,
                                                     GenerationEventCallback callback, void* callback_context) {
  if (impl_ == nullptr) {
    return Status(StatusCode::kInternal, "chat session was moved from");
  }
  if (impl_->poisoned || impl_->session.is_poisoned()) {
    return Status(StatusCode::kInternal,
                  "chat session cannot continue after a state-mutating failure");
  }
  if (request.max_generated_tokens.has_value() && *request.max_generated_tokens == 0U) {
    return Status(StatusCode::kInvalidArgument, "generation token limit must be positive when specified");
  }
  if (request.tool_choice.mode != GenerationToolChoiceMode::kAuto &&
      request.tool_choice.mode != GenerationToolChoiceMode::kNone) {
    return Status(StatusCode::kUnsupported,
                  "native Gemma generation currently supports auto or none tool choice");
  }
  if (!request.parallel_tool_calls) {
    return Status(StatusCode::kUnsupported,
                  "native Gemma generation cannot yet constrain output to one tool call");
  }
  const Status tool_definition_status =
      internal::ValidateToolDefinitions(request.tools);
  if (!tool_definition_status.ok()) return tool_definition_status;
  if (request.messages.empty()) {
    return Status(StatusCode::kInvalidArgument,
                  "generation request requires at least one message");
  }
  if (request.messages.back().role != "user" &&
      request.messages.back().role != "tool") {
    return Status(StatusCode::kInvalidArgument, "generation request must end with a user or tool message");
  }
  std::vector<ChatToolDefinition> tools;
  if (request.tool_choice.mode != GenerationToolChoiceMode::kNone) {
    tools.reserve(request.tools.size());
    for (const GenerationToolDefinition& tool : request.tools) {
      tools.push_back({tool.name, tool.description, tool.parameters_json});
    }
  }
  if (!impl_->committed_messages.empty()) {
    const bool extends_prefix =
        request.messages.size() > impl_->committed_messages.size() &&
        std::equal(
            impl_->committed_messages.begin(), impl_->committed_messages.end(),
            request.messages.begin(),
            [](const GenerationMessage& cached,
               const GenerationMessage& supplied) {
              return internal::ResidentMessageEquivalent(cached, supplied);
            });
    const auto added_begin = extends_prefix
                                 ? request.messages.begin() +
                                       static_cast<std::ptrdiff_t>(
                                           impl_->committed_messages.size())
                                 : request.messages.end();
    const bool one_user =
        extends_prefix && request.messages.size() ==
                              impl_->committed_messages.size() + 1U &&
        added_begin->role == "user";
    const bool tool_results =
        extends_prefix &&
        std::all_of(added_begin, request.messages.end(),
                    [](const GenerationMessage& message) {
                      return message.role == "tool";
                    });
    if (!one_user && !tool_results) {
      return Status(StatusCode::kInvalidArgument,
                    "generation request must append one user turn or consecutive tool results");
    }
    if (request.tools != impl_->committed_tools) {
      return Status(StatusCode::kInvalidArgument, "resident conversation tool definitions must not change");
    }
    if (request.tool_choice != impl_->committed_tool_choice) {
      return Status(StatusCode::kInvalidArgument,
                    "resident conversation tool choice must not change");
    }
  }
  auto messages = MaterializeMessages(request.messages,
                                      impl_->committed_messages);
  if (!messages.ok()) return messages.status();
  const bool tool_result_continuation =
      messages.value().messages.back().role == "tool";

  Result<std::vector<std::uint32_t>> prompt_ids = [&]() {
    if (impl_->cached_prefix_token_ids.empty()) {
      return impl_->processor.Encode(messages.value().messages, request.thinking.effort != ThinkingEffort::kOff, true,
                                     tools);
    }
    if (messages.value().messages.back().role == "tool") {
      std::vector<ChatToolResult> results;
      for (std::size_t index = impl_->committed_messages.size();
           index < request.messages.size(); ++index) {
        const GenerationContentPart& result_part =
            request.messages[index].content.front();
        std::string tool_name;
        for (const GenerationContentPart& assistant_part :
             impl_->committed_messages.back().content) {
          if (assistant_part.kind == GenerationContentKind::kToolCall &&
              assistant_part.tool_call.id == result_part.tool_result.call_id) {
            tool_name = assistant_part.tool_call.name;
            break;
          }
        }
        if (tool_name.empty()) {
          return Result<std::vector<std::uint32_t>>(Status(
              StatusCode::kInvalidArgument,
              "tool result does not match the resident assistant calls"));
        }
        results.push_back({tool_name, result_part.tool_result.output});
      }
      auto continuation = impl_->processor.EncodeToolResultsContinuation(
          results, request.thinking.effort != ThinkingEffort::kOff);
      if (!continuation.ok()) {
        return Result<std::vector<std::uint32_t>>(continuation.status());
      }
      std::vector<std::uint32_t> token_ids = impl_->cached_prefix_token_ids;
      if (impl_->pending_assistant_token_id.has_value()) {
        token_ids.push_back(*impl_->pending_assistant_token_id);
      }
      token_ids.insert(token_ids.end(), continuation.value().begin(),
                       continuation.value().end());
      return Result<std::vector<std::uint32_t>>(std::move(token_ids));
    }
    auto continuation = impl_->processor.EncodeContinuation(messages.value().messages.back().content,
                                                            request.thinking.effort != ThinkingEffort::kOff);
    if (!continuation.ok()) {
      return Result<std::vector<std::uint32_t>>(continuation.status());
    }
    std::vector<std::uint32_t> token_ids = impl_->cached_prefix_token_ids;
    if (impl_->pending_assistant_token_id.has_value()) {
      token_ids.push_back(*impl_->pending_assistant_token_id);
    }
    token_ids.insert(token_ids.end(), continuation.value().begin(), continuation.value().end());
    return Result<std::vector<std::uint32_t>>(std::move(token_ids));
  }();
  if (!prompt_ids.ok()) return prompt_ids.status();
  auto audio_segments = LocateAudioSegments(prompt_ids.value(), messages.value().audio_frames);
  if (!audio_segments.ok()) return audio_segments.status();
  auto vision_segments = LocateVisionSegments(prompt_ids.value(), messages.value().images);
  if (!vision_segments.ok()) return vision_segments.status();
  auto moe26b_vision_segments = LocateGemma4Moe26BVisionSegments(
      prompt_ids.value(), messages.value().moe26b_images);
  if (!moe26b_vision_segments.ok()) return moe26b_vision_segments.status();
  if (prompt_ids.value().size() > impl_->max_context_tokens) {
    return Status(StatusCode::kInvalidArgument, "conversation prompt exceeds the session context capacity");
  }
  const std::uint64_t remaining_output_capacity = impl_->max_context_tokens - prompt_ids.value().size() + 1U;
  const std::uint64_t max_generated_tokens = request.max_generated_tokens.value_or(remaining_output_capacity);
  if (max_generated_tokens > remaining_output_capacity) {
    return Status(StatusCode::kInvalidArgument, "requested output exceeds the remaining context capacity");
  }

  ReasoningTokenOptions reasoning;
  reasoning.enabled = request.thinking.effort != ThinkingEffort::kOff ||
                      tool_result_continuation;
  reasoning.starts_in_reasoning = tool_result_continuation;
  if (reasoning.enabled) {
    reasoning.channel_open_token_ids = impl_->processor.generation_controls().thinking_open_token_ids;
    reasoning.channel_close_token_id = impl_->processor.generation_controls().thinking_close_token_id;
    const std::uint64_t answer_reserve = max_generated_tokens > 128U ? 128U : max_generated_tokens / 2U;
    const std::uint64_t requested_budget =
        request.thinking.effort == ThinkingEffort::kOff
            ? 1U
            : ThinkingBudgetTokens(request.thinking.effort);
    reasoning.max_reasoning_tokens = std::min(
        requested_budget,
        std::max<std::uint64_t>(1U, max_generated_tokens - answer_reserve));
  }

  EventBridge bridge{callback, callback_context};
  auto inference = impl_->session.Generate(
      prompt_ids.value(), max_generated_tokens, reasoning, callback == nullptr ? nullptr : ForwardTokenEvent,
      callback == nullptr ? nullptr : &bridge, audio_segments.value(),
      vision_segments.value(), moe26b_vision_segments.value());
  if (!inference.ok()) {
    impl_->poisoned = impl_->session.is_poisoned();
    return inference.status();
  }
  if (!messages.value().moe26b_images.empty()) {
    const auto& image = messages.value().moe26b_images.front();
    inference.value().image_decode_milliseconds = image.decode_milliseconds;
    inference.value().image_resize_patchify_milliseconds =
        image.resize_patchify_milliseconds;
  }
  const auto poison = [this](Status status) {
    impl_->poisoned = true;
    return status;
  };

  impl_->cached_prefix_token_ids = prompt_ids.value();
  if (inference.value().output_token_ids.size() > 1U) {
    impl_->cached_prefix_token_ids.insert(impl_->cached_prefix_token_ids.end(),
                                          inference.value().output_token_ids.begin(),
                                          inference.value().output_token_ids.end() - 1);
  }
  impl_->pending_assistant_token_id.reset();
  if (!inference.value().stopped && !inference.value().output_token_ids.empty()) {
    impl_->pending_assistant_token_id = inference.value().output_token_ids.back();
  }

  std::vector<std::uint32_t> content_ids = inference.value().output_token_ids;
  const auto& stop_tokens = impl_->processor.generation_controls().stop_token_ids;
  if (!content_ids.empty() &&
      std::find(stop_tokens.begin(), stop_tokens.end(), content_ids.back()) != stop_tokens.end()) {
    content_ids.pop_back();
  }
  auto assistant_content = impl_->processor.Decode(content_ids, false);
  if (!assistant_content.ok()) return poison(assistant_content.status());
  const bool flatten_reasoning_to_text =
      request.thinking.effort == ThinkingEffort::kOff &&
      !tool_result_continuation;
  std::vector<std::uint32_t> visible_ids =
      internal::ExtractVisibleTokenIds(
          content_ids, impl_->processor.generation_controls(),
          tool_result_continuation, flatten_reasoning_to_text);
  auto assistant_text = impl_->processor.DecodeResponseText(visible_ids);
  if (!assistant_text.ok()) return poison(assistant_text.status());
  std::vector<std::uint32_t> reasoning_ids;
  if (!flatten_reasoning_to_text) {
    reasoning_ids = internal::ExtractReasoningTokenIds(
        content_ids, impl_->processor.generation_controls(),
        tool_result_continuation);
  }
  if (reasoning_ids.size() != inference.value().reasoning_tokens) {
    return poison(Status(
        StatusCode::kInternal,
        "decoded reasoning channel has " +
            std::to_string(reasoning_ids.size()) +
            " tokens but inference accounted " +
            std::to_string(inference.value().reasoning_tokens)));
  }
  auto reasoning_text = impl_->processor.Decode(reasoning_ids, true);
  if (!reasoning_text.ok()) return poison(reasoning_text.status());
  internal::GemmaToolCallParser tool_parser;
  auto parsed_tool_events = tool_parser.Push(assistant_content.value(), true);
  if (!parsed_tool_events.ok()) return poison(parsed_tool_events.status());
  const Status tool_call_status = internal::ValidateGeneratedToolCalls(
      request.tools, request.tool_choice, tool_parser.tool_calls());
  if (!tool_call_status.ok()) return poison(tool_call_status);

  ChatGenerationResponse response;
  response.assistant_content = std::move(assistant_content).value();
  response.assistant_text = std::move(assistant_text).value();
  response.reasoning_text = std::move(reasoning_text).value();
  response.tool_calls = tool_parser.tool_calls();
  response.prompt_token_ids = std::move(prompt_ids).value();
  response.finish_reason = !response.tool_calls.empty() ? GenerationFinishReason::kToolCalls
                           : inference.value().stopped  ? GenerationFinishReason::kStop
                                                        : GenerationFinishReason::kLength;
  response.inference = std::move(inference).value();

  if (impl_->committed_messages.empty()) {
    impl_->committed_messages = request.messages;
  } else {
    impl_->committed_messages.insert(
        impl_->committed_messages.end(),
        request.messages.begin() + static_cast<std::ptrdiff_t>(
                                       impl_->committed_messages.size()),
        request.messages.end());
  }
  GenerationMessage assistant;
  assistant.role = "assistant";
  if (!response.assistant_text.empty() || response.tool_calls.empty()) {
    assistant.content.push_back(
        GenerationContentPart::Text(response.assistant_text));
  }
  for (const GenerationToolCall& call : response.tool_calls) {
    assistant.content.push_back(GenerationContentPart::ToolCall(call));
  }
  impl_->committed_messages.push_back(std::move(assistant));
  impl_->committed_tools = request.tools;
  impl_->committed_tool_choice = request.tool_choice;
  return response;
}

std::uint64_t ChatSession::cached_token_count() const {
  return impl_ == nullptr ? 0U : impl_->session.cached_token_count();
}

std::uint64_t ChatSession::reserved_device_bytes() const {
  return impl_ == nullptr ? 0U : impl_->session.reserved_device_bytes();
}

bool ChatSession::is_poisoned() const {
  return impl_ == nullptr || impl_->poisoned || impl_->session.is_poisoned();
}

const char* GenerationFinishReasonName(GenerationFinishReason reason) {
  switch (reason) {
    case GenerationFinishReason::kStop:
      return "stop";
    case GenerationFinishReason::kLength:
      return "length";
    case GenerationFinishReason::kToolCalls:
      return "tool_calls";
  }
  return "unknown";
}

}  // namespace gem16

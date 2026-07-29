#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "gem16/engine.h"
#include "gem16/audio.h"
#include "gem16/status.h"
#include "gem16/tokenizer.h"

namespace gem16 {

// Server-neutral content boundary. Network locations, MIME decoding, base64,
// and OpenAI JSON belong to adapters above this interface. Text and raw audio
// are qualified here; codecs and transport remain adapter concerns.
enum class GenerationContentKind {
  kText,
  kAudio,
};

struct GenerationContentPart {
  GenerationContentKind kind = GenerationContentKind::kText;
  std::string text;
  AudioWaveform audio;

  bool operator==(const GenerationContentPart&) const = default;

  [[nodiscard]] static GenerationContentPart Text(std::string value) {
    GenerationContentPart part;
    part.kind = GenerationContentKind::kText;
    part.text = std::move(value);
    return part;
  }

  [[nodiscard]] static GenerationContentPart Audio(AudioWaveform value) {
    GenerationContentPart part;
    part.kind = GenerationContentKind::kAudio;
    part.audio = std::move(value);
    return part;
  }
};

struct GenerationMessage {
  std::string role;
  std::vector<GenerationContentPart> content;

  bool operator==(const GenerationMessage&) const = default;

  [[nodiscard]] static GenerationMessage Text(std::string role,
                                              std::string content) {
    GenerationMessage message;
    message.role = std::move(role);
    message.content.push_back(GenerationContentPart::Text(std::move(content)));
    return message;
  }
};

enum class ThinkingEffort {
  kOff,
  kSmall,
  kMedium,
  kHigh,
};

struct ThinkingOptions {
  ThinkingEffort effort = ThinkingEffort::kMedium;
};

[[nodiscard]] std::uint64_t ThinkingBudgetTokens(ThinkingEffort effort);
[[nodiscard]] const char* ThinkingEffortName(ThinkingEffort effort);

struct ChatGenerationRequest {
  // Complete conversation through the new user turn. A resident session
  // requires every later request to extend the previously committed messages.
  std::vector<GenerationMessage> messages;
  // Unset generates until a checkpoint stop token or the session's remaining
  // context capacity. A value sets a stricter per-turn output limit.
  std::optional<std::uint64_t> max_generated_tokens;
  ThinkingOptions thinking;
};

enum class GenerationEventKind {
  kToken,
};

struct GenerationEvent {
  GenerationEventKind kind = GenerationEventKind::kToken;
  std::uint32_t token_id = 0U;
};

using GenerationEventCallback = Status (*)(void* context,
                                            const GenerationEvent& event);

enum class GenerationFinishReason {
  kStop,
  kLength,
};

struct ChatGenerationResponse {
  std::string assistant_content;
  std::string assistant_text;
  std::vector<std::uint32_t> prompt_token_ids;
  GenerationFinishReason finish_reason = GenerationFinishReason::kLength;
  GreedyInferenceResult inference;
};

struct ChatSessionOptions {
  std::filesystem::path model_directory;
  std::filesystem::path assistant_model_directory;
  std::uint64_t max_context_tokens = 1024U;
  KvCacheMode kv_cache_mode = KvCacheMode::kCheckpointFp8;
  SamplingOptions sampling;
  std::uint32_t mtp_draft_tokens = 0U;
  bool mtp_adaptive = false;
};

// High-level batch-one generation boundary shared by CLI and future protocol
// adapters. It owns prompt materialization and exact resident-prefix tracking;
// callers provide messages rather than CUDA-facing token/cache state.
class ChatSession {
 public:
  ChatSession(const ChatSession&) = delete;
  ChatSession& operator=(const ChatSession&) = delete;
  ChatSession(ChatSession&&) noexcept;
  ChatSession& operator=(ChatSession&&) noexcept;
  ~ChatSession();

  [[nodiscard]] static Result<ChatSession> Create(
      const ChatSessionOptions& options);
  [[nodiscard]] static Result<ChatSession> Create(
      const ChatSessionOptions& options, GemmaChatProcessor processor);

  [[nodiscard]] Result<ChatGenerationResponse> Generate(
      const ChatGenerationRequest& request,
      GenerationEventCallback callback = nullptr,
      void* callback_context = nullptr);
  [[nodiscard]] std::uint64_t cached_token_count() const;

 private:
  struct Impl;
  explicit ChatSession(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] const char* GenerationFinishReasonName(
    GenerationFinishReason reason);

}  // namespace gem16

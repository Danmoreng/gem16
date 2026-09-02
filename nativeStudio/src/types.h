#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace gem16::studio {

enum class ModelProfile {
  kGemma4Unified12B,
  kGemma4Moe26BA4B,
  kGemma4Moe26BTrellis35VisionFp8,
};
inline constexpr std::size_t kModelProfileCount = 3U;

[[nodiscard]] constexpr std::size_t ModelProfileIndex(ModelProfile profile) {
  switch (profile) {
    case ModelProfile::kGemma4Unified12B:
      return 0U;
    case ModelProfile::kGemma4Moe26BA4B:
      return 1U;
    case ModelProfile::kGemma4Moe26BTrellis35VisionFp8:
      return 2U;
  }
  return 0U;
}
enum class ServerPhase { kStopped, kStarting, kRunning, kExternal, kStopping, kError };
enum class Screen { kChat, kModels, kServer, kSettings };
enum class MediaKind { kImage, kAudio, kDocument };

struct ServerConfig {
  ModelProfile profile = ModelProfile::kGemma4Unified12B;
  std::string executable;
  std::string model_directory;
  std::string assistant_directory;
  std::string vision_directory;
  std::string model_name = "gem16-12b";
  std::string host = "127.0.0.1";
  int port = 8080;
  std::int64_t max_context_tokens = 32768;
  int max_sessions = 1;
  int mtp_draft_tokens = 2;
  int vision_soft_token_budget = 280;
  bool mtp_adaptive = false;
  bool greedy = false;
};

struct GenerationConfig {
  std::string reasoning_effort = "medium";
  std::int64_t max_output_tokens = 32768;
  std::string system_prompt = "You are a helpful assistant.";
};

struct StudioSettings {
  ServerConfig server;
  GenerationConfig generation;
  bool dark_theme = true;
  bool onboarding_complete = false;
  // Zero selects the platform-aware automatic scale.
  float ui_scale = 0.0f;
};

struct HealthSnapshot {
  bool available = false;
  std::string status;
  std::string model_variant;
  std::string profile_id;
  std::string decode_mode;
  std::string qualification_state;
  bool text_only = false;
  bool supports_mtp = false;
  bool supports_vision = false;
  bool vision_module_loaded = false;
  bool vision_mtp_supported = false;
  int resident_sessions = 0;
  int session_limit = 0;
  std::int64_t max_context_tokens = 0;
  int mtp_draft_tokens = 0;
  int vision_max_soft_token_budget = 0;
  int last_vision_soft_token_budget = 0;
  std::int64_t vision_max_context_tokens = 0;
  bool sampling_enabled = false;
};

struct MediaAttachment {
  std::uint64_t id = 0;
  MediaKind kind = MediaKind::kDocument;
  std::string file_name;
  std::string mime_type;
  std::string format;
  std::vector<std::uint8_t> bytes;
  std::string document_text;
  std::uint64_t byte_size = 0;
  int image_width = 0;
  int image_height = 0;
};

struct PerformanceStats {
  double decode_tokens_per_second = 0.0;
  double prefill_tokens_per_second = 0.0;
  double prefill_milliseconds = 0.0;
  double decode_milliseconds = 0.0;
};

struct ChatMessage {
  std::string role;
  std::string content;
  std::string reasoning;
  bool streaming = false;
  bool error = false;
  std::vector<MediaAttachment> attachments;
};

struct ChatEvent {
  enum class Kind {
    kText,
    kReasoning,
    kUsage,
    kPerformance,
    kFinished,
    kError,
    kSession,
  };
  Kind kind = Kind::kText;
  std::string value;
  std::int64_t prompt_tokens = 0;
  std::int64_t completion_tokens = 0;
  PerformanceStats performance;

  ChatEvent() = default;
  ChatEvent(Kind event_kind, std::string event_value)
      : kind(event_kind), value(std::move(event_value)) {}
};

[[nodiscard]] const char* ProfileLabel(ModelProfile profile);
[[nodiscard]] const char* ProfileWireName(ModelProfile profile);

}  // namespace gem16::studio

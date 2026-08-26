#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gem16::studio {

enum class ModelProfile { kGemma4Unified12B, kGemma4Moe26BA4B };
enum class ServerPhase { kStopped, kStarting, kRunning, kExternal, kStopping, kError };
enum class Screen { kChat, kModels, kServer, kSettings };

struct ServerConfig {
  ModelProfile profile = ModelProfile::kGemma4Unified12B;
  std::string executable;
  std::string model_directory;
  std::string assistant_directory;
  std::string model_name = "gem16-12b";
  std::string host = "127.0.0.1";
  int port = 8080;
  std::int64_t max_context_tokens = 32768;
  int max_sessions = 1;
  int mtp_draft_tokens = 2;
  bool mtp_adaptive = false;
  bool greedy = false;
};

struct GenerationConfig {
  std::string reasoning_effort = "medium";
  std::int64_t max_output_tokens = 4096;
  std::string system_prompt = "You are a helpful assistant.";
};

struct StudioSettings {
  ServerConfig server;
  GenerationConfig generation;
  bool dark_theme = true;
};

struct HealthSnapshot {
  bool available = false;
  std::string status;
  std::string model_variant;
  bool text_only = false;
  bool supports_mtp = false;
  int resident_sessions = 0;
  int session_limit = 0;
  std::int64_t max_context_tokens = 0;
  int mtp_draft_tokens = 0;
  bool sampling_enabled = false;
};

struct ChatMessage {
  std::string role;
  std::string content;
  std::string reasoning;
  bool streaming = false;
  bool error = false;
};

struct ChatEvent {
  enum class Kind { kText, kReasoning, kFinished, kError, kSession };
  Kind kind = Kind::kText;
  std::string value;
};

[[nodiscard]] const char* ProfileLabel(ModelProfile profile);
[[nodiscard]] const char* ProfileWireName(ModelProfile profile);

}  // namespace gem16::studio


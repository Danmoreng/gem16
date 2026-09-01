#include "server_manager.h"

#include "settings.h"
#include "util/json.h"

#include "httplib.h"

#include <filesystem>
#include <iomanip>
#include <sstream>

namespace gem16::studio {
namespace {

const json::Value* Member(const json::Value* value, std::string_view key) {
  return value && value->is_object() ? value->find(key) : nullptr;
}

std::string StringValue(const json::Value* value) {
  return value && value->is_string() ? value->as_string() : std::string{};
}

bool BoolValue(const json::Value* value) {
  return value && value->is_bool() && value->as_bool();
}

std::int64_t IntegerValue(const json::Value* value) {
  return value && value->is_integer() ? value->as_integer() : 0;
}

bool IsMoe26B(ModelProfile profile) {
  switch (profile) {
    case ModelProfile::kGemma4Unified12B:
      return false;
    case ModelProfile::kGemma4Moe26BA4B:
    case ModelProfile::kGemma4Moe26BTrellis35VisionFp8:
      return true;
  }
  return false;
}

bool IsVision26B(ModelProfile profile) {
  return profile == ModelProfile::kGemma4Moe26BTrellis35VisionFp8;
}

std::string Timestamp() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t time = std::chrono::system_clock::to_time_t(now);
  std::tm local{};
#ifdef _WIN32
  localtime_s(&local, &time);
#else
  localtime_r(&time, &local);
#endif
  std::ostringstream stream;
  stream << std::put_time(&local, "%H:%M:%S");
  return stream.str();
}

}  // namespace

std::string HealthCompatibilityError(const ServerConfig& config,
                                     const HealthSnapshot& health) {
  if (!health.available) return "Server health is unavailable";
  std::string_view expected_profile;
  switch (config.profile) {
    case ModelProfile::kGemma4Unified12B:
      expected_profile = "gemma4-12b-unified";
      if (!health.supports_vision || health.text_only) {
        return "The live server does not expose the selected 12B multimodal capabilities";
      }
      break;
    case ModelProfile::kGemma4Moe26BA4B:
      expected_profile = "gemma4-26b-a4b-nvfp4";
      if (!health.text_only || health.supports_vision ||
          health.vision_module_loaded) {
        return "The live server does not expose the selected qualified 26B text-only capabilities";
      }
      break;
    case ModelProfile::kGemma4Moe26BTrellis35VisionFp8:
      expected_profile = config.mtp_draft_tokens == 2
                             ? "gemma4-26b-a4b-trellis35-vision-fp8-d2"
                             : "gemma4-26b-a4b-trellis35-vision-fp8";
      if (!health.supports_vision || !health.vision_module_loaded ||
          health.text_only) {
        return "The live server has not loaded the selected 26B Vision module";
      }
      if (config.mtp_draft_tokens == 2 &&
          (!health.supports_mtp || !health.vision_mtp_supported ||
           health.qualification_state != "experimental_v20_frozen")) {
        return "The live server does not qualify Vision with fixed-D2 for this component set";
      }
      if (config.mtp_draft_tokens == 0 && health.vision_mtp_supported) {
        return "The live Vision server has a different MTP configuration";
      }
      break;
  }
  if (health.profile_id != expected_profile) {
    return "Live server profile mismatch: expected " +
           std::string(expected_profile) + ", received " +
           (health.profile_id.empty() ? std::string("<missing>")
                                      : health.profile_id);
  }
  if (config.mtp_draft_tokens != 0 && !health.supports_mtp) {
    return "The live server does not expose the selected MTP capability";
  }
  if (health.mtp_draft_tokens != config.mtp_draft_tokens) {
    return "Live server MTP mismatch: expected D" +
           std::to_string(config.mtp_draft_tokens) + ", received D" +
           std::to_string(health.mtp_draft_tokens);
  }
  return {};
}

ServerManager::ServerManager() : poller_([this](std::stop_token token) { PollLoop(token); }) {}

ServerManager::~ServerManager() {
  poller_.request_stop();
  Stop();
}

void ServerManager::Configure(const ServerConfig& config) {
  std::lock_guard lock(mutex_);
  config_ = config;
}

void ServerManager::Start(const ServerConfig& config) {
  if (process_.IsRunning()) return;
  Configure(config);
  {
    std::lock_guard lock(mutex_);
    phase_ = ServerPhase::kStarting;
    error_.clear();
  }
  AppendLog("Checking " + config.host + ":" + std::to_string(config.port) + " for an existing server");
  const HealthSnapshot existing = FetchHealth(config);
  if (existing.available) {
    std::lock_guard lock(mutex_);
    health_ = existing;
    const std::string mismatch = HealthCompatibilityError(config, existing);
    if (mismatch.empty()) {
      phase_ = ServerPhase::kExternal;
      logs_.push_back(Timestamp() + "  Attached to compatible existing server");
    } else {
      phase_ = ServerPhase::kError;
      error_ = mismatch;
      logs_.push_back(Timestamp() + "  Existing server rejected: " + mismatch);
    }
    return;
  }
  if (const std::string validation = Validate(config); !validation.empty()) {
    std::lock_guard lock(mutex_);
    phase_ = ServerPhase::kError;
    error_ = validation;
    logs_.push_back(Timestamp() + "  Start failed: " + validation);
    return;
  }
  AppendLog("Starting " + config.executable);
  std::string start_error;
  const bool started = process_.Start(
      BuildServerCommand(config), RepositoryRoot().string(),
      [this](std::string line) { AppendLog(std::move(line)); },
      [this](int exit_code) {
        AppendLog("Server exited with code " + std::to_string(exit_code));
        std::lock_guard lock(mutex_);
        health_ = {};
        if (phase_ == ServerPhase::kStopping) {
          phase_ = ServerPhase::kStopped;
        } else {
          phase_ = exit_code == 0 ? ServerPhase::kStopped : ServerPhase::kError;
          if (exit_code != 0) error_ = "gem16-server exited with code " + std::to_string(exit_code);
        }
      },
      start_error);
  if (!started) {
    std::lock_guard lock(mutex_);
    phase_ = ServerPhase::kError;
    error_ = start_error;
    logs_.push_back(Timestamp() + "  Start failed: " + start_error);
  }
}

void ServerManager::Stop() {
  if (!process_.IsRunning()) {
    std::lock_guard lock(mutex_);
    if (phase_ != ServerPhase::kExternal) phase_ = ServerPhase::kStopped;
    health_ = {};
    return;
  }
  {
    std::lock_guard lock(mutex_);
    phase_ = ServerPhase::kStopping;
  }
  AppendLog("Stopping managed server");
  process_.Stop();
}

void ServerManager::ClearLogs() {
  std::lock_guard lock(mutex_);
  logs_.clear();
}

ServerPhase ServerManager::Phase() const {
  std::lock_guard lock(mutex_);
  return phase_;
}

HealthSnapshot ServerManager::Health() const {
  std::lock_guard lock(mutex_);
  return health_;
}

std::string ServerManager::Error() const {
  std::lock_guard lock(mutex_);
  return error_;
}

std::vector<std::string> ServerManager::Logs() const {
  std::lock_guard lock(mutex_);
  return {logs_.begin(), logs_.end()};
}

void ServerManager::PollLoop(std::stop_token stop_token) {
  while (!stop_token.stop_requested()) {
    ServerConfig config;
    {
      std::lock_guard lock(mutex_);
      config = config_;
    }
    const HealthSnapshot snapshot = FetchHealth(config);
    {
      std::lock_guard lock(mutex_);
      health_ = snapshot;
      if (snapshot.available) {
        const std::string mismatch = HealthCompatibilityError(config, snapshot);
        if (mismatch.empty()) {
          error_.clear();
          phase_ = process_.IsRunning() ? ServerPhase::kRunning
                                        : ServerPhase::kExternal;
        } else {
          error_ = mismatch;
          phase_ = ServerPhase::kError;
        }
      } else if (!process_.IsRunning() && phase_ == ServerPhase::kExternal) {
        phase_ = ServerPhase::kStopped;
      }
    }
    for (int step = 0; step < 15 && !stop_token.stop_requested(); ++step) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
}

HealthSnapshot ServerManager::FetchHealth(const ServerConfig& config) const {
  HealthSnapshot snapshot;
  if (config.port <= 0 || config.port > 65535) return snapshot;
  const std::string host = config.host == "0.0.0.0" ? "127.0.0.1" : config.host;
  httplib::Client client(host, config.port);
  client.set_connection_timeout(std::chrono::seconds(1));
  client.set_read_timeout(std::chrono::seconds(2));
  const auto response = client.Get("/health");
  if (!response || response->status != 200) return snapshot;
  const auto parsed = json::Parse(response->body);
  if (!parsed.ok() || !parsed.value().is_object()) return snapshot;
  const json::Value& root = parsed.value();
  const json::Value* capabilities = Member(&root, "capabilities");
  const json::Value* sampling = Member(&root, "sampling");
  snapshot.available = true;
  snapshot.status = StringValue(Member(&root, "status"));
  snapshot.model_variant = StringValue(Member(&root, "model_variant"));
  snapshot.profile_id = StringValue(Member(&root, "profile_id"));
  snapshot.qualification_state =
      StringValue(Member(&root, "qualification_state"));
  snapshot.text_only = BoolValue(Member(&root, "text_only"));
  snapshot.supports_mtp = BoolValue(Member(capabilities, "mtp"));
  snapshot.supports_vision = BoolValue(Member(capabilities, "vision"));
  snapshot.vision_mtp_supported =
      BoolValue(Member(capabilities, "vision_mtp"));
  snapshot.vision_module_loaded =
      BoolValue(Member(&root, "vision_module_loaded"));
  snapshot.resident_sessions = static_cast<int>(IntegerValue(Member(&root, "resident_sessions")));
  snapshot.session_limit = static_cast<int>(IntegerValue(Member(&root, "session_limit")));
  snapshot.max_context_tokens = IntegerValue(Member(&root, "max_context_tokens"));
  snapshot.mtp_draft_tokens = static_cast<int>(IntegerValue(Member(&root, "mtp_draft_tokens")));
  snapshot.selected_vision_soft_token_budget = static_cast<int>(
      IntegerValue(Member(&root, "selected_vision_soft_token_budget")));
  snapshot.vision_max_context_tokens =
      IntegerValue(Member(&root, "vision_max_context_tokens"));
  snapshot.sampling_enabled = BoolValue(Member(sampling, "enabled"));
  return snapshot;
}

std::string ServerManager::Validate(const ServerConfig& config) const {
  if (!std::filesystem::is_regular_file(config.executable)) return "Server executable does not exist";
  if (!std::filesystem::is_directory(config.model_directory)) return "Model directory does not exist";
  if (config.mtp_draft_tokens != 0 && !std::filesystem::is_directory(config.assistant_directory)) {
    return "MTP is enabled, but the assistant directory does not exist";
  }
  if (IsVision26B(config.profile) &&
      !std::filesystem::is_directory(config.vision_directory)) {
    return "Vision profile is selected, but the Vision directory does not exist";
  }
  if (IsMoe26B(config.profile) && config.max_sessions != 1) {
    return "Gemma 4 26B supports exactly one resident session";
  }
  if (IsMoe26B(config.profile) && config.mtp_adaptive) {
    return "Gemma 4 26B supports fixed-depth MTP only";
  }
  if (IsMoe26B(config.profile) && config.mtp_draft_tokens != 0 &&
      config.mtp_draft_tokens != 2) {
    return "Gemma 4 26B supports only Ordinary decode or fixed-D2 MTP";
  }
  if (config.profile == ModelProfile::kGemma4Moe26BA4B &&
      config.mtp_draft_tokens != 0 && config.max_context_tokens > 86016) {
    return "Gemma 4 26B fixed-D2 MTP supports at most 86,016 context tokens";
  }
  if (config.profile == ModelProfile::kGemma4Moe26BA4B &&
      config.mtp_draft_tokens == 0 && config.max_context_tokens > 98304) {
    return "Gemma 4 26B Target-only supports at most 98,304 context tokens";
  }
  if (IsVision26B(config.profile) && config.mtp_draft_tokens == 2 &&
      config.max_context_tokens > 229120) {
    return "Gemma 4 26B Vision fixed-D2 supports at most 229,120 context tokens";
  }
  if (IsVision26B(config.profile) && config.mtp_draft_tokens == 0 &&
      config.max_context_tokens > 229376) {
    return "Gemma 4 26B Vision supports at most 229,376 context tokens";
  }
  if (IsVision26B(config.profile) && config.vision_soft_token_budget != 70 &&
      config.vision_soft_token_budget != 140 &&
      config.vision_soft_token_budget != 280) {
    return "Vision soft-token budget must be 70, 140, or 280";
  }
  if (config.port < 1 || config.port > 65535) return "Port must be in [1, 65535]";
  if (config.max_context_tokens < 1 || config.max_context_tokens > 262144) return "Context is outside the supported range";
  return {};
}

void ServerManager::AppendLog(std::string line) {
  std::lock_guard lock(mutex_);
  logs_.push_back(Timestamp() + "  " + std::move(line));
  while (logs_.size() > 1000) logs_.pop_front();
}

std::vector<std::string> BuildServerCommand(const ServerConfig& config) {
  std::vector<std::string> result{
      config.executable, "--model", config.model_directory,
      "--model-name", config.model_name, "--host", config.host,
      "--port", std::to_string(config.port), "--max-context",
      std::to_string(config.max_context_tokens), "--max-sessions",
      std::to_string(config.max_sessions)};
  if (config.greedy) result.push_back("--greedy");
  if (IsVision26B(config.profile)) {
    result.insert(result.end(), {"--vision-model", config.vision_directory});
  }
  if (config.mtp_draft_tokens != 0) {
    result.insert(result.end(), {"--assistant-model", config.assistant_directory,
                                 "--mtp-draft-tokens", std::to_string(config.mtp_draft_tokens)});
    if (config.mtp_adaptive) result.push_back("--mtp-adaptive");
  }
  return result;
}

}  // namespace gem16::studio

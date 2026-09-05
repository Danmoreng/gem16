#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include "windows_utf8.h"
#endif

#include "httplib.h"

#include "gem16/chat.h"
#include "gem16/tokenizer.h"
#include "model/config.h"
#include "model/model_variant.h"
#include "server/http_streaming.h"
#include "server/observability.h"
#include "server/openai_chat.h"
#include "server/request_queue.h"
#include "server/secure_id.h"
#include "server/session_pool.h"
#include "server/http_policy.h"
#include "util/json.h"

namespace {

using gem16::server::AcquireNamedSession;
using gem16::server::AcquireResponseSession;
using gem16::server::ClearActiveResponse;
using gem16::server::CommitResponsesRequest;
using gem16::server::CreateSession;
using gem16::server::CreateSessionQueued;
using gem16::server::DiscardSession;
using gem16::server::FinishSse;
using gem16::server::IndexResponse;
using gem16::server::MakeChatIdentity;
using gem16::server::MakeResponsesIdentity;
using gem16::server::MetricsText;
using gem16::server::LogField;
using gem16::server::LogFormat;
using gem16::server::LogLevel;
using gem16::server::ParseLogFormat;
using gem16::server::ParseLogLevel;
using gem16::server::PrepareResponsesRequest;
using gem16::server::RecordGeneration;
using gem16::server::RecordQueueAdmission;
using gem16::server::RecordRequestLatency;
using gem16::server::RequestAdmission;
using gem16::server::ReleaseSession;
using gem16::server::ServerState;
using gem16::server::SessionEntry;
using gem16::server::SessionLease;
using gem16::server::StructuredLogger;
using gem16::server::SetActiveResponse;
using gem16::server::UnindexResponse;
using gem16::server::WriteSse;

constexpr std::uint64_t kPrimaryServerVramSafetyBytes =
    700U * 1024U * 1024U;
constexpr std::uint64_t kLong26BServerVramSafetyBytes =
    200U * 1024U * 1024U;
constexpr std::uint64_t kLong26BMtpServerVramSafetyBytes =
    200U * 1024U * 1024U;
constexpr std::uint64_t kRequestPayloadLimit = 16U * 1024U * 1024U;

volatile std::sig_atomic_t g_shutdown_signal = 0;

void HandleShutdownSignal(int signal) { g_shutdown_signal = signal; }

struct RequestLogContext {
  std::chrono::steady_clock::time_point started{};
  std::string request_id;
  std::uint64_t queue_wait_microseconds = 0U;
};

thread_local RequestLogContext g_request_log_context;

struct SlotMemoryPlan {
  std::uint64_t slot_bytes = 0U;
  std::uint64_t configured_slot_bytes = 0U;
  std::uint64_t safety_margin_bytes = 0U;
  gem16::DeviceMemoryInfo device;
};

struct Options {
  std::filesystem::path model_directory;
  std::filesystem::path assistant_model_directory;
  std::filesystem::path vision_model_directory;
  std::uint32_t vision_max_soft_token_budget = 280U;
  std::string model_name = "gem16";
  std::string host = "127.0.0.1";
  int port = 8080;
  std::uint64_t max_context = 8192U;
  gem16::KvCacheMode kv_cache_mode = gem16::KvCacheMode::kCheckpointFp8;
  std::uint32_t mtp_draft_tokens = 0U;
  std::uint32_t max_sessions = 2U;
  std::size_t max_queued_requests = 64U;
  LogLevel log_level = LogLevel::kInfo;
  LogFormat log_format = LogFormat::kText;
  bool max_sessions_explicit = false;
  bool max_context_explicit = false;
  bool mtp_adaptive = false;
  bool greedy = false;
  bool verify_device_image_sha256 = false;
  std::uint64_t sampling_seed = 0U;
};

void PrintUsage() {
  std::cout
      << "Usage: gem16-server --model <checkpoint> [options]\n"
      << "  --model-name <id>       Served OpenAI model id (default: gem16)\n"
      << "  --host <address>        Listen address (default: 127.0.0.1)\n"
      << "  --port <port>           Listen port (default: 8080)\n"
      << "  --max-context <tokens>  Session context capacity (default: 8192)\n"
      << "  --max-sessions <count>   Resident slots (default: 2; 26B profile: 1)\n"
      << "  --max-queued-requests <count>  FIFO waiters (default: 64)\n"
      << "  --log-level debug|info|warning|error|off (default: info)\n"
      << "  --log-format text|json (default: text)\n"
      << "  --kv-cache fp8|bf16\n"
      << "  --model-integrity structural|sha256 (default: structural)\n"
      << "  --greedy                Disable checkpoint-recommended sampling\n"
      << "  --seed <integer>        Sampling seed (default: 0)\n"
      << "  --assistant-model <checkpoint> --mtp-draft-tokens 1|2|4 [--mtp-adaptive]\n"
      << "  --vision-model <compiled-vision-module>\n"
      << "  --vision-max-soft-token-budget 70|140|280 (default: 280)\n";
}

bool ParseUnsigned(std::string_view text, std::uint64_t& value) {
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

gem16::Result<Options> ParseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--model" && index + 1 < argc) {
      options.model_directory = argv[++index];
    } else if (argument == "--assistant-model" && index + 1 < argc) {
      options.assistant_model_directory = argv[++index];
    } else if (argument == "--vision-model" && index + 1 < argc) {
      options.vision_model_directory = argv[++index];
    } else if (argument == "--vision-max-soft-token-budget" &&
               index + 1 < argc) {
      std::uint64_t value = 0U;
      if (!ParseUnsigned(argv[++index], value) ||
          (value != 70U && value != 140U && value != 280U)) {
        return gem16::Status(
            gem16::StatusCode::kInvalidArgument,
            "--vision-max-soft-token-budget must be 70, 140, or 280");
      }
      options.vision_max_soft_token_budget =
          static_cast<std::uint32_t>(value);
    } else if (argument == "--model-name" && index + 1 < argc) {
      options.model_name = argv[++index];
    } else if (argument == "--host" && index + 1 < argc) {
      options.host = argv[++index];
    } else if (argument == "--port" && index + 1 < argc) {
      std::uint64_t value = 0U;
      if (!ParseUnsigned(argv[++index], value) || value == 0U || value > 65535U) {
        return gem16::Status(gem16::StatusCode::kInvalidArgument,
                             "--port must be in [1, 65535]");
      }
      options.port = static_cast<int>(value);
    } else if (argument == "--max-context" && index + 1 < argc) {
      if (!ParseUnsigned(argv[++index], options.max_context) ||
          options.max_context == 0U) {
        return gem16::Status(gem16::StatusCode::kInvalidArgument,
                             "--max-context must be positive");
      }
      options.max_context_explicit = true;
    } else if (argument == "--max-sessions" && index + 1 < argc) {
      std::uint64_t value = 0U;
      if (!ParseUnsigned(argv[++index], value) || value == 0U ||
          value > 64U) {
        return gem16::Status(gem16::StatusCode::kInvalidArgument,
                             "--max-sessions must be in [1, 64]");
      }
      options.max_sessions = static_cast<std::uint32_t>(value);
      options.max_sessions_explicit = true;
    } else if (argument == "--max-queued-requests" && index + 1 < argc) {
      std::uint64_t value = 0U;
      if (!ParseUnsigned(argv[++index], value) || value == 0U ||
          value > 64U) {
        return gem16::Status(gem16::StatusCode::kInvalidArgument,
                             "--max-queued-requests must be in [1, 64]");
      }
      options.max_queued_requests = static_cast<std::size_t>(value);
    } else if (argument == "--log-level" && index + 1 < argc) {
      auto level = ParseLogLevel(argv[++index]);
      if (!level.ok()) return level.status();
      options.log_level = level.value();
    } else if (argument == "--log-format" && index + 1 < argc) {
      auto format = ParseLogFormat(argv[++index]);
      if (!format.ok()) return format.status();
      options.log_format = format.value();
    } else if (argument == "--kv-cache" && index + 1 < argc) {
      const std::string_view mode(argv[++index]);
      if (mode == "fp8") {
        options.kv_cache_mode = gem16::KvCacheMode::kCheckpointFp8;
      } else if (mode == "bf16") {
        options.kv_cache_mode = gem16::KvCacheMode::kBf16Correctness;
      } else {
        return gem16::Status(gem16::StatusCode::kInvalidArgument,
                             "--kv-cache must be fp8 or bf16");
      }
    } else if (argument == "--mtp-draft-tokens" && index + 1 < argc) {
      std::uint64_t value = 0U;
      if (!ParseUnsigned(argv[++index], value) ||
          (value != 1U && value != 2U && value != 4U)) {
        return gem16::Status(gem16::StatusCode::kInvalidArgument,
                             "--mtp-draft-tokens must be 1, 2, or 4");
      }
      options.mtp_draft_tokens = static_cast<std::uint32_t>(value);
    } else if (argument == "--model-integrity" && index + 1 < argc) {
      const std::string_view mode(argv[++index]);
      if (mode == "structural") {
        options.verify_device_image_sha256 = false;
      } else if (mode == "sha256") {
        options.verify_device_image_sha256 = true;
      } else {
        return gem16::Status(
            gem16::StatusCode::kInvalidArgument,
            "--model-integrity must be structural or sha256");
      }
    } else if (argument == "--mtp-adaptive") {
      options.mtp_adaptive = true;
    } else if (argument == "--greedy") {
      options.greedy = true;
    } else if (argument == "--seed" && index + 1 < argc) {
      if (!ParseUnsigned(argv[++index], options.sampling_seed)) {
        return gem16::Status(gem16::StatusCode::kInvalidArgument,
                             "--seed must be an unsigned integer");
      }
    } else {
      return gem16::Status(gem16::StatusCode::kInvalidArgument,
                           "unknown or incomplete option: " +
                               std::string(argument));
    }
  }
  if (options.model_directory.empty()) {
    return gem16::Status(gem16::StatusCode::kInvalidArgument,
                         "--model is required");
  }
  if (!gem16::server::IsLoopbackHost(options.host)) {
    return gem16::Status(gem16::StatusCode::kUnsupported,
        "only loopback binding is supported; remote serving requires a separately secured gateway");
  }
  if (options.model_name.empty() || options.host.empty()) {
    return gem16::Status(gem16::StatusCode::kInvalidArgument,
                         "--model-name and --host must not be empty");
  }
  if (options.mtp_draft_tokens != 0U &&
      options.assistant_model_directory.empty()) {
    return gem16::Status(gem16::StatusCode::kInvalidArgument,
                         "active MTP requires --assistant-model");
  }
  if (options.mtp_adaptive && options.mtp_draft_tokens == 0U) {
    return gem16::Status(gem16::StatusCode::kInvalidArgument,
                         "--mtp-adaptive requires active MTP");
  }
  return options;
}

int HttpStatus(const gem16::Status& status) {
  switch (status.code()) {
    case gem16::StatusCode::kInvalidArgument: return 400;
    case gem16::StatusCode::kNotFound: return 404;
    case gem16::StatusCode::kDataLoss: return 400;
    case gem16::StatusCode::kUnsupported: return 400;
    case gem16::StatusCode::kResourceExhausted: return 503;
    case gem16::StatusCode::kCancelled: return 409;
    case gem16::StatusCode::kOk: return 200;
    default: return 500;
  }
}

bool HasVisionInput(const gem16::ChatGenerationRequest& request) {
  for (const auto& message : request.messages) {
    for (const auto& part : message.content) {
      if (part.kind == gem16::GenerationContentKind::kImage ||
          part.kind == gem16::GenerationContentKind::kGemma4Moe26BImage) {
        return true;
      }
    }
  }
  return false;
}

void RecordStatusMetric(ServerState& state, const gem16::Status& status,
                        bool vision_request = false) {
  const bool vision_failure =
      vision_request || status.message().find("Vision") != std::string::npos ||
      status.message().find("vision") != std::string::npos ||
      status.message().find("image") != std::string::npos;
  if (vision_failure) state.metrics.vision_failures.fetch_add(1U);
  if (status.code() == gem16::StatusCode::kDataLoss &&
      (status.message().find("Vision artifact") != std::string::npos ||
       status.message().find("Vision module") != std::string::npos)) {
    state.metrics.vision_artifact_validation_failures.fetch_add(1U);
  }
  switch (status.code()) {
    case gem16::StatusCode::kResourceExhausted:
      state.metrics.resource_exhaustion_count.fetch_add(1U);
      break;
    case gem16::StatusCode::kUnsupported:
      state.metrics.unsupported_feature_count.fetch_add(1U);
      break;
    case gem16::StatusCode::kDataLoss:
      state.metrics.model_validation_failure_count.fetch_add(1U);
      break;
    default:
      break;
  }
}

void SetError(ServerState& state, const gem16::Status& status,
              httplib::Response& response, bool vision_request = false) {
  RecordStatusMetric(state, status, vision_request);
  response.status = HttpStatus(status);
  std::string_view type = response.status >= 500 ? "server_error"
                                                  : "invalid_request_error";
  std::string_view code = gem16::server::VisionErrorCode(status);
  if (status.code() == gem16::StatusCode::kUnsupported) {
    type = "unsupported_feature";
    if (code.empty()) {
      code = status.message().find("Gemma 4 26B") != std::string::npos
                 ? "gemma4_26b_text_only"
                 : "unsupported_feature";
    }
  } else if (status.code() == gem16::StatusCode::kResourceExhausted) {
    type = "resource_exhausted";
    if (status.message() == "request queue is full") {
      code = "request_queue_full";
    } else if (status.message() == "server is draining") {
      code = "server_draining";
    } else {
      code = "resident_capacity_exhausted";
    }
    response.set_header("Retry-After", "1");
  }
  response.set_content(
      gem16::server::OpenAiErrorJson(status.message(), type, code),
      "application/json; charset=utf-8");
}

gem16::Result<RequestAdmission> AcquireRequestAdmission(
    ServerState& state, const gem16::server::SessionWaitOptions& wait) {
  auto admission = state.request_queue.Acquire(
      wait.deadline, wait.cancelled);
  if (!admission.ok()) {
    state.metrics.queue_rejections.fetch_add(1U);
    return admission.status();
  }
  RecordQueueAdmission(state, admission.value().wait_microseconds());
  g_request_log_context.queue_wait_microseconds =
      admission.value().wait_microseconds();
  return std::move(admission).value();
}

gem16::Status ValidateRequestCapabilities(
    ServerState& state, const gem16::ChatGenerationRequest& request) {
  if (request.tool_choice.mode != gem16::GenerationToolChoiceMode::kAuto &&
      request.tool_choice.mode != gem16::GenerationToolChoiceMode::kNone)
    return gem16::Status(gem16::StatusCode::kUnsupported,
        "required/named tool choice needs constrained generation, which is not yet implemented");
  if (!request.parallel_tool_calls)
    return gem16::Status(gem16::StatusCode::kUnsupported,
        "parallel_tool_calls=false needs constrained generation, which is not yet implemented");
  std::uint32_t image_count = 0U;
  bool has_compact_vision = false;
  for (const auto& message : request.messages) {
    for (const auto& part : message.content) {
      if (part.kind == gem16::GenerationContentKind::kAudio &&
          !state.runtime->supports_audio()) {
        return gem16::Status(
            gem16::StatusCode::kUnsupported,
            "Gemma 4 26B is a text-only profile; audio input is unsupported");
      }
      if (part.kind == gem16::GenerationContentKind::kImage &&
          !state.runtime->supports_vision()) {
        return gem16::Status(gem16::StatusCode::kUnsupported,
                             state.runtime->experimental()
                                 ? "Vision module is not loaded"
                                 : "a Vision profile is required for image input");
      }
      if (part.kind == gem16::GenerationContentKind::kImage ||
          part.kind == gem16::GenerationContentKind::kGemma4Moe26BImage) {
        ++image_count;
      }
      if (part.kind == gem16::GenerationContentKind::kGemma4Moe26BImage) {
        has_compact_vision = true;
        if (!state.runtime->supports_vision()) {
          return gem16::Status(gem16::StatusCode::kUnsupported,
                               "a Vision profile is required for 26B image input");
        }
        const std::uint32_t budget = part.moe26b_image.soft_token_budget;
        if (budget != 70U && budget != 140U && budget != 280U) {
          return gem16::Status(gem16::StatusCode::kUnsupported,
                               "Vision soft-token budget is unsupported");
        }
        state.metrics.last_vision_soft_token_budget.store(budget);
        if (budget == 70U) {
          state.metrics.vision_budget_70.fetch_add(1U);
        } else if (budget == 140U) {
          state.metrics.vision_budget_140.fetch_add(1U);
        } else {
          state.metrics.vision_budget_280.fetch_add(1U);
        }
      }
    }
  }
  if (image_count == 0U) return gem16::Status::Ok();
  state.metrics.vision_requests.fetch_add(1U);
  // The module/context/fixed-D2 qualification below belongs to 26B. 12B's
  // integrated Vision path has its own admitted context and MTP support.
  if (!has_compact_vision) return gem16::Status::Ok();
  if (image_count > state.runtime->maximum_images()) {
    return gem16::Status(gem16::StatusCode::kUnsupported,
                         "image count exceeds the active Vision profile capacity");
  }
  if (state.max_context > state.runtime->vision_max_context_tokens()) {
    return gem16::Status(gem16::StatusCode::kUnsupported,
                         "Vision context is outside the measured profile limit");
  }
  if (state.session_options.mtp_draft_tokens != 0U &&
      !state.runtime->vision_mtp_supported()) {
    state.metrics.vision_d2_rejections.fetch_add(1U);
    return gem16::Status(
        gem16::StatusCode::kUnsupported,
        "Vision with fixed-D2 is not qualified for this component set");
  }
  return gem16::Status::Ok();
}

struct CancellationContext {
  std::atomic<bool>* cancel_requested = nullptr;
  std::atomic<std::uint64_t>* cancellations_observed = nullptr;
  std::atomic<std::uint64_t>* client_disconnects = nullptr;
  httplib::DataSink* sink = nullptr;
  const httplib::Request* request = nullptr;
};

gem16::Status CheckCancellation(void* opaque_context,
                                const gem16::GenerationEvent& event) {
  auto* context = static_cast<CancellationContext*>(opaque_context);
  if (context == nullptr ||
      event.kind != gem16::GenerationEventKind::kToken) {
    return gem16::Status(gem16::StatusCode::kInternal,
                         "invalid cancellation callback");
  }
  if (g_shutdown_signal != 0 || (context->cancel_requested != nullptr &&
      context->cancel_requested->load())) {
    if (context->cancellations_observed != nullptr) {
      context->cancellations_observed->fetch_add(1U);
    }
    return gem16::Status(gem16::StatusCode::kCancelled,
                         "generation was cancelled");
  }
  if ((context->sink != nullptr && context->sink->is_writable &&
       !context->sink->is_writable()) ||
      (context->request != nullptr && context->request->is_connection_closed())) {
    if (context->client_disconnects != nullptr) {
      context->client_disconnects->fetch_add(1U);
    }
    return gem16::Status(gem16::StatusCode::kCancelled,
                         "client disconnected during generation");
  }
  return gem16::Status::Ok();
}

void HandleCompletion(ServerState& state, const httplib::Request& request,
                      httplib::Response& response) {
  state.metrics.requests_total.fetch_add(1U);
  const gem16::server::SessionWaitOptions wait{
      std::chrono::steady_clock::now() + std::chrono::seconds(30),
      request.is_connection_closed};
  auto admission_result = AcquireRequestAdmission(state, wait);
  if (!admission_result.ok()) {
    state.metrics.requests_failed.fetch_add(1U);
    SetError(state, admission_result.status(), response);
    return;
  }
  RequestAdmission admission = std::move(admission_result).value();
  auto parsed = gem16::server::ParseChatCompletionsRequest(
      request.body,
      {state.max_context, state.runtime->vision_module_loaded(),
       state.runtime->vision_max_soft_token_budget(), state.model_name});
  if (!parsed.ok()) {
    state.metrics.requests_failed.fetch_add(1U);
    SetError(state, parsed.status(), response);
    return;
  }
  const gem16::Status capability =
      ValidateRequestCapabilities(state, parsed.value().generation);
  if (!capability.ok()) {
    state.metrics.requests_failed.fetch_add(1U);
    SetError(state, capability, response);
    return;
  }
  auto identity_result = MakeChatIdentity(state);
  if (!identity_result.ok()) {
    state.metrics.requests_failed.fetch_add(1U);
    SetError(state, identity_result.status(), response);
    return;
  }
  gem16::server::OpenAiResponseIdentity identity =
      std::move(identity_result).value();
  std::string session_id;
  for (const auto header : {"X-Gem16-Session-Id", "session_id", "x-session-affinity"}) {
    const auto count = request.get_header_value_count(header);
    const auto value = request.get_header_value(header);
    if (count > 1U || (count && value.empty()) ||
        (!session_id.empty() && count && value != session_id)) {
      SetError(state, gem16::Status(gem16::StatusCode::kInvalidArgument,
          "session affinity headers are empty, duplicated or conflicting"), response);
      return;
    }
    if (count) session_id = value;
  }
  if (session_id.empty()) {
    auto generated_id = gem16::server::MakeSecureId("session_");
    if (!generated_id.ok()) {
      state.metrics.requests_failed.fetch_add(1U);
      SetError(state, generated_id.status(), response);
      return;
    }
    session_id = std::move(generated_id).value();
  }
  if (session_id.size() > 128U ||
      !std::all_of(session_id.begin(), session_id.end(), [](char value) {
        return (value >= 'a' && value <= 'z') ||
               (value >= 'A' && value <= 'Z') ||
               (value >= '0' && value <= '9') || value == '-' ||
               value == '_' || value == '.';
      })) {
    state.metrics.requests_failed.fetch_add(1U);
    SetError(state, gem16::Status(gem16::StatusCode::kInvalidArgument,
                           "X-Gem16-Session-Id is invalid"),
             response);
    return;
  }
  auto acquired = AcquireNamedSession(state, "chat:" + session_id, wait);
  if (!acquired.ok()) {
    state.metrics.requests_failed.fetch_add(1U);
    SetError(state, acquired.status(), response);
    return;
  }
  std::shared_ptr<SessionEntry> entry = std::move(acquired).value();
  SessionLease lease(state, entry);
  {
    std::lock_guard inference_lock(entry->inference_mutex);
    const auto continuation = entry->session.ValidateContinuation(parsed.value().generation);
    if (!continuation.ok()) {
      if (continuation.code() != gem16::StatusCode::kInvalidArgument) {
        lease.Discard();
        SetError(state, continuation, response);
        return;
      }
      lease.Discard();
      const auto restarted = entry->session.Restart(state.runtime, state.session_options, state.processor);
      if (!restarted.ok()) { SetError(state, restarted, response); return; }
      lease.Keep();
      state.metrics.sessions_rebuilt.fetch_add(1U);
      response.set_header("X-Gem16-Cache-Reset", "history_or_tools_changed");
    }
  }
  response.set_header("X-Gem16-Session-Id", session_id);
  if (!parsed.value().stream) {
    std::lock_guard inference_lock(entry->inference_mutex);
    entry->cancel_requested.store(false);
    CancellationContext cancellation{
        &entry->cancel_requested, &state.metrics.cancellations_observed,
        &state.metrics.client_disconnects, nullptr, &request};
    const auto generation_start = std::chrono::steady_clock::now();
    auto generated = entry->session.Generate(
        parsed.value().generation, CheckCancellation, &cancellation);
    if (!generated.ok()) {
      state.metrics.requests_failed.fetch_add(1U);
      SetError(state, generated.status(), response,
               HasVisionInput(parsed.value().generation));
      if (entry->session.is_poisoned()) lease.Discard();
      return;
    }
    RecordGeneration(state, generated.value(),
                     std::chrono::steady_clock::now() - generation_start);
    response.set_content(
        gem16::server::ChatCompletionJson(identity, generated.value()),
        "application/json; charset=utf-8");
    return;
  }

  struct ProviderState {
    ServerState* server = nullptr;
    gem16::ChatGenerationRequest generation;
    gem16::server::OpenAiResponseIdentity identity;
    std::shared_ptr<SessionEntry> entry;
    RequestAdmission admission;
    std::unique_ptr<SessionLease> lease;
    bool include_usage = false;
    bool ran = false;
  };
  auto provider = std::make_shared<ProviderState>();
  provider->server = &state;
  provider->generation = std::move(parsed.value().generation);
  provider->identity = identity;
  provider->entry = entry;
  provider->admission = std::move(admission);
  provider->lease = std::make_unique<SessionLease>(std::move(lease));
  provider->include_usage = parsed.value().include_usage;
  response.set_header("Cache-Control", "no-cache");
  response.set_header("X-Accel-Buffering", "no");
  response.set_header("Transfer-Encoding", "chunked");
  response.set_content_provider(
      "text/event-stream; charset=utf-8",
      [provider](std::size_t, httplib::DataSink& sink) {
        if (provider->ran) return false;
        provider->ran = true;
        std::lock_guard inference_lock(provider->entry->inference_mutex);
        provider->entry->cancel_requested.store(false);
        if (!WriteSse(sink, gem16::server::ChatCompletionChunkJson(
                                provider->identity,
                                "{\"role\":\"assistant\"}"))) {
          return false;
        }
        gem16::server::ChatCompletionStream stream(
            provider->server->processor, provider->identity, sink,
            {&provider->entry->cancel_requested,
             &provider->server->metrics.cancellations_observed,
             &provider->server->metrics.client_disconnects});
        const auto generation_start = std::chrono::steady_clock::now();
        provider->lease->Discard();
        auto generated = stream.Generate(provider->entry->session,
                                         provider->generation);
        const bool generation_completed = generated.ok();
        if (!generated.ok()) {
          provider->server->metrics.requests_failed.fetch_add(1U);
          RecordStatusMetric(*provider->server, generated.status(),
                             HasVisionInput(provider->generation));
          (void)WriteSse(
              sink, gem16::server::OpenAiErrorJson(
                        generated.status().message(), "server_error"));
          (void)FinishSse(sink);
          if (!generation_completed &&
              !provider->entry->session.is_poisoned()) {
            provider->lease->Keep();
          }
          return true;
        }
        provider->lease->Keep();
        RecordGeneration(*provider->server, generated.value(),
                         std::chrono::steady_clock::now() - generation_start);
        const gem16::Status tool_status =
            stream.WriteToolCalls(generated.value());
        if (!tool_status.ok()) return false;
        if (!WriteSse(sink, gem16::server::ChatCompletionChunkJson(
                                provider->identity, {},
                                generated.value().finish_reason))) {
          return false;
        }
        if (provider->include_usage &&
            !WriteSse(sink, gem16::server::ChatCompletionChunkJson(
                                    provider->identity, {}, std::nullopt,
                                    &generated.value()))) {
          return false;
        }
        if (!WriteSse(sink, "[DONE]") || !FinishSse(sink)) return false;
        provider->lease->Keep();
        return true;
      });
}

void HandleResponses(ServerState& state, const httplib::Request& request,
                     httplib::Response& response) {
  state.metrics.requests_total.fetch_add(1U);
  const gem16::server::SessionWaitOptions wait{
      std::chrono::steady_clock::now() + std::chrono::seconds(30),
      request.is_connection_closed};
  auto admission_result = AcquireRequestAdmission(state, wait);
  if (!admission_result.ok()) {
    state.metrics.requests_failed.fetch_add(1U);
    SetError(state, admission_result.status(), response);
    return;
  }
  RequestAdmission admission = std::move(admission_result).value();
  auto parsed = gem16::server::ParseResponsesRequest(
      request.body,
      {state.max_context, state.runtime->vision_module_loaded(),
       state.runtime->vision_max_soft_token_budget(), state.model_name});
  if (!parsed.ok()) {
    state.metrics.requests_failed.fetch_add(1U);
    SetError(state, parsed.status(), response);
    return;
  }
  const gem16::Status capability =
      ValidateRequestCapabilities(state, parsed.value().generation);
  if (!capability.ok()) {
    state.metrics.requests_failed.fetch_add(1U);
    SetError(state, capability, response);
    return;
  }
  auto identity_result = MakeResponsesIdentity(state);
  if (!identity_result.ok()) {
    state.metrics.requests_failed.fetch_add(1U);
    SetError(state, identity_result.status(), response);
    return;
  }
  gem16::server::OpenAiResponseIdentity identity =
      std::move(identity_result).value();
  gem16::Result<std::shared_ptr<SessionEntry>> acquired =
      parsed.value().previous_response_id.has_value()
          ? AcquireResponseSession(state,
                                   *parsed.value().previous_response_id, wait)
          : CreateSessionQueued(state, "responses:" + identity.id, wait);
  if (!acquired.ok()) {
    state.metrics.requests_failed.fetch_add(1U);
    SetError(state, acquired.status(), response);
    return;
  }
  std::shared_ptr<SessionEntry> entry = std::move(acquired).value();
  SessionLease lease(state, entry);
  struct ResponseIndexGuard {
    ServerState& state;
    const std::string& id;
    bool keep = false;
    ~ResponseIndexGuard() { if (!keep) UnindexResponse(state, id); }
  } index_guard{state, identity.id};
  IndexResponse(state, entry, identity.id);
  if (!parsed.value().stream) {
    std::lock_guard inference_lock(entry->inference_mutex);
    gem16::Status status = PrepareResponsesRequest(*entry, parsed.value());
    if (!status.ok()) {
      state.metrics.requests_failed.fetch_add(1U);
      SetError(state, status, response);
      UnindexResponse(state, identity.id);
      return;
    }
    entry->cancel_requested.store(false);
    CancellationContext cancellation{
        &entry->cancel_requested, &state.metrics.cancellations_observed,
        &state.metrics.client_disconnects, nullptr, &request};
    const auto generation_start = std::chrono::steady_clock::now();
    auto generated = entry->session.Generate(
        parsed.value().generation, CheckCancellation, &cancellation);
    if (!generated.ok()) {
      state.metrics.requests_failed.fetch_add(1U);
      SetError(state, generated.status(), response,
               HasVisionInput(parsed.value().generation));
      if (entry->session.is_poisoned()) lease.Discard();
      return;
    }
    RecordGeneration(state, generated.value(),
                     std::chrono::steady_clock::now() - generation_start);
    CommitResponsesRequest(*entry, parsed.value(), generated.value(),
                           identity.id);
    identity.completed = gem16::server::UnixSecondsNow();
    response.set_content(
        gem16::server::ResponseJson(identity, parsed.value(),
                                    generated.value()),
        "application/json; charset=utf-8");
    index_guard.keep = true;
    return;
  }

  struct ProviderState {
    ServerState* server = nullptr;
    gem16::server::OpenAiResponsesRequest request;
    gem16::server::OpenAiResponseIdentity identity;
    std::shared_ptr<SessionEntry> entry;
    RequestAdmission admission;
    std::unique_ptr<SessionLease> lease;
    std::atomic<bool> keep_response_index{false};
    bool ran = false;

    ~ProviderState() {
      if (server != nullptr &&
          !keep_response_index.load(std::memory_order_acquire)) {
        UnindexResponse(*server, identity.id);
      }
    }
  };
  auto provider = std::make_shared<ProviderState>();
  provider->server = &state;
  provider->request = std::move(parsed).value();
  provider->identity = identity;
  index_guard.keep = true;
  provider->entry = entry;
  provider->admission = std::move(admission);
  provider->lease = std::make_unique<SessionLease>(std::move(lease));
  response.set_header("Cache-Control", "no-cache");
  response.set_header("X-Accel-Buffering", "no");
  response.set_header("Transfer-Encoding", "chunked");
  response.set_content_provider(
      "text/event-stream; charset=utf-8",
      [provider](std::size_t, httplib::DataSink& sink) {
        if (provider->ran) return false;
        provider->ran = true;
        std::lock_guard inference_lock(provider->entry->inference_mutex);
        gem16::Status status = PrepareResponsesRequest(
            *provider->entry, provider->request);
        if (!status.ok()) {
          provider->server->metrics.requests_failed.fetch_add(1U);
          RecordStatusMetric(*provider->server, status,
                             HasVisionInput(provider->request.generation));
          (void)WriteSse(
              sink, "{\"type\":\"error\",\"code\":null,\"message\":" +
                        gem16::json::Quote(status.message()) +
                        ",\"param\":null,\"sequence_number\":0}");
          (void)FinishSse(sink);
          return true;
        }
        provider->entry->cancel_requested.store(false);
        SetActiveResponse(*provider->server, provider->entry,
                          provider->identity.id);
        if (!WriteSse(
                sink,
                "{\"type\":\"response.created\",\"response\":" +
                    gem16::server::ResponseShellJson(
                        provider->identity, provider->request, "in_progress") +
                    ",\"sequence_number\":0}")) {
          ClearActiveResponse(*provider->server, provider->entry,
                              provider->identity.id);
          return false;
        }
        std::uint64_t reasoning_capacity = gem16::ThinkingBudgetTokens(
            provider->request.generation.thinking.effort);
        // The checkpoint template leaves a tool-result continuation at the
        // model boundary. Even with thinking disabled, Gemma emits an empty
        // thought channel containing one newline before the visible answer.
        if (reasoning_capacity == 0U &&
            !provider->request.generation.messages.empty() &&
            provider->request.generation.messages.back().role == "tool") {
          reasoning_capacity = 1U;
        }
        if (provider->request.generation.max_generated_tokens.has_value()) {
          reasoning_capacity = std::min(
              reasoning_capacity,
              *provider->request.generation.max_generated_tokens);
        }
        gem16::server::ResponsesStream stream(
            provider->server->processor, provider->identity, sink,
            reasoning_capacity,
            {&provider->entry->cancel_requested,
             &provider->server->metrics.cancellations_observed,
             &provider->server->metrics.client_disconnects});
        const auto generation_start = std::chrono::steady_clock::now();
        provider->lease->Discard();
        auto generated = stream.Generate(provider->entry->session,
                                         provider->request.generation);
        const bool generation_completed = generated.ok();
        ClearActiveResponse(*provider->server, provider->entry,
                            provider->identity.id);
        if (!generated.ok()) {
          provider->server->metrics.requests_failed.fetch_add(1U);
          RecordStatusMetric(*provider->server, generated.status(),
                             HasVisionInput(provider->request.generation));
          (void)WriteSse(
              sink, "{\"type\":\"error\",\"code\":null,\"message\":" +
                        gem16::json::Quote(generated.status().message()) +
                        ",\"param\":null,\"sequence_number\":" +
                        std::to_string(stream.sequence()) + "}");
          (void)FinishSse(sink);
          if (!generation_completed &&
              !provider->entry->session.is_poisoned()) {
            provider->lease->Keep();
          }
          return true;
        }
        RecordGeneration(*provider->server, generated.value(),
                         std::chrono::steady_clock::now() - generation_start);
        CommitResponsesRequest(*provider->entry, provider->request,
                               generated.value(), provider->identity.id);
        // Generation and KV commit are complete at this point. The OpenAI SDK
        // may stop reading immediately after response.completed, so a later
        // final-chunk write failure must not discard an otherwise safe chain.
        provider->keep_response_index.store(true, std::memory_order_release);
        provider->lease->Keep();
        const bool written = stream.WriteFinalEvents(
            provider->request, generated.value(),
            gem16::server::UnixSecondsNow());
        return written && FinishSse(sink);
      });
}

gem16::Result<SlotMemoryPlan> PlanServerSlots(
    const std::shared_ptr<gem16::ModelRuntime>& runtime,
    const gem16::ChatSessionOptions& options,
    const gem16::GemmaChatProcessor& processor,
    std::uint32_t max_sessions) {
  auto before = gem16::QueryDeviceMemoryInfo();
  if (!before.ok()) return before.status();

  std::uint64_t reported_slot_bytes = 0U;
  gem16::DeviceMemoryInfo after;
  {
    auto probe = gem16::ChatSession::Create(runtime, options, processor);
    if (!probe.ok()) return probe.status();
    reported_slot_bytes = probe.value().reserved_device_bytes();
    auto measured = gem16::QueryDeviceMemoryInfo();
    if (!measured.ok()) return measured.status();
    after = measured.value();
  }
  const std::uint64_t measured_delta =
      before.value().free_bytes > after.free_bytes
          ? before.value().free_bytes - after.free_bytes
          : 0U;
  const std::uint64_t slot_bytes =
      std::max(reported_slot_bytes, measured_delta);
  if (slot_bytes == 0U) {
    return gem16::Status(gem16::StatusCode::kInternal,
                         "execution-slot memory probe reported zero bytes");
  }
  if (slot_bytes >
      std::numeric_limits<std::uint64_t>::max() / max_sessions) {
    return gem16::Status(gem16::StatusCode::kResourceExhausted,
                         "configured execution-slot memory overflows accounting");
  }
  const std::uint64_t additional_slots = max_sessions - 1U;
  const bool moe_26b =
      std::string_view(runtime->model_variant_name()) ==
          "gemma4_moe_26b_a4b";
  const bool long_26b = moe_26b && options.max_context_tokens >= 65536U;
  const bool long_26b_mtp =
      moe_26b && options.mtp_draft_tokens != 0U &&
      options.max_context_tokens >= 64000U;
  const std::uint64_t safety_margin =
      long_26b_mtp
          ? kLong26BMtpServerVramSafetyBytes
          : long_26b ? kLong26BServerVramSafetyBytes
                     : kPrimaryServerVramSafetyBytes;
  const std::uint64_t additional_slot_bytes = slot_bytes * additional_slots;
  const bool required_overflows =
      additional_slot_bytes >
      std::numeric_limits<std::uint64_t>::max() - safety_margin;
  const std::uint64_t required_free =
      required_overflows
          ? std::numeric_limits<std::uint64_t>::max()
          : additional_slot_bytes + safety_margin;
  if (required_overflows || after.free_bytes < required_free) {
    const std::uint64_t shortfall =
        required_free > after.free_bytes ? required_free - after.free_bytes
                                         : 0U;
    return gem16::Status(
        gem16::StatusCode::kResourceExhausted,
        "cannot admit configured execution slots: free=" +
            std::to_string(after.free_bytes) +
            " required_slot=" + std::to_string(slot_bytes) +
            " additional_required=" +
            std::to_string(additional_slot_bytes) +
            " probe_resident=" + std::to_string(slot_bytes) +
            " required_margin=" + std::to_string(safety_margin) +
            " shortfall=" + std::to_string(shortfall));
  }
  return SlotMemoryPlan{slot_bytes, slot_bytes * max_sessions, safety_margin,
                        after};
}

void HandleCancelResponse(ServerState& state, std::string_view response_id,
                          httplib::Response& response) {
  bool cancellation_requested = false;
  {
    std::lock_guard pool_lock(state.pool_mutex);
    const auto found = state.response_index.find(std::string(response_id));
    if (found != state.response_index.end()) {
      const std::shared_ptr<SessionEntry> entry = found->second.lock();
      if (entry != nullptr && entry->active_response_id == response_id &&
          entry->active_requests.load() != 0U) {
        entry->cancel_requested.store(true);
        state.metrics.cancellations_requested.fetch_add(1U);
        cancellation_requested = true;
      }
    }
  }
  if (!cancellation_requested) {
    SetError(state, gem16::Status(gem16::StatusCode::kNotFound,
                           "response is not actively generating"),
             response);
    return;
  }
  response.set_content(
      "{\"id\":" + gem16::json::Quote(response_id) +
          ",\"object\":\"response\",\"status\":\"cancelling\"}",
      "application/json; charset=utf-8");
}

int ServerMain(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--version") {
    std::cout << "gem16-server " << GEM16_VERSION_STRING << '\n';
    return 0;
  }
  if (argc == 2 && (std::string_view(argv[1]) == "--help" ||
                    std::string_view(argv[1]) == "-h")) {
    PrintUsage();
    return 0;
  }
  auto options = ParseOptions(argc, argv);
  if (!options.ok()) {
    std::cerr << "error: " << options.status().message() << '\n';
    PrintUsage();
    return 64;
  }
  StructuredLogger logger(options.value().log_level,
                          options.value().log_format, std::cerr);
  const auto startup_started = std::chrono::steady_clock::now();
  logger.Log(LogLevel::kInfo, "server_starting",
             {{"version", GEM16_VERSION_STRING},
              {"model", options.value().model_name},
              {"host", options.value().host},
              {"port", std::to_string(options.value().port)}});
  auto config = gem16::internal::LoadModelConfig(
      options.value().model_directory / "config.json");
  if (!config.ok()) {
    logger.Log(LogLevel::kError, "model_config_failed",
               {{"error", config.status().message()}});
    return 2;
  }
  const bool moe26b = gem16::internal::ClassifyModelVariant(config.value()) ==
                      gem16::internal::ModelVariant::kGemma4Moe26BA4B;
  if (!options.value().max_context_explicit && moe26b) {
    options.value().max_context = 32768U;
  }
  auto processor =
      gem16::GemmaChatProcessor::Load(options.value().model_directory);
  if (!processor.ok()) {
    logger.Log(LogLevel::kError, "chat_processor_load_failed",
               {{"error", processor.status().message()}});
    return 2;
  }
  gem16::ChatSessionOptions session_options;
  session_options.model_directory = options.value().model_directory;
  session_options.assistant_model_directory =
      options.value().assistant_model_directory;
  session_options.vision_model_directory =
      options.value().vision_model_directory;
  session_options.max_context_tokens = options.value().max_context;
  session_options.kv_cache_mode = options.value().kv_cache_mode;
  session_options.sampling =
      processor.value().generation_controls().recommended_sampling;
  if (options.value().greedy) session_options.sampling.enabled = false;
  session_options.sampling.seed = options.value().sampling_seed;
  session_options.mtp_draft_tokens = options.value().mtp_draft_tokens;
  session_options.mtp_adaptive = options.value().mtp_adaptive;
  auto runtime = gem16::ModelRuntime::Load(
      {options.value().model_directory,
       options.value().assistant_model_directory,
       options.value().max_context, 0,
       options.value().verify_device_image_sha256,
       options.value().vision_model_directory,
       options.value().vision_max_soft_token_budget});
  if (!runtime.ok()) {
    logger.Log(LogLevel::kError, "model_load_failed",
               {{"error", runtime.status().message()}});
    return 2;
  }
  if (!options.value().max_sessions_explicit &&
      runtime.value()->maximum_execution_slots() == 1U) {
    options.value().max_sessions = 1U;
  } else if (options.value().max_sessions >
             runtime.value()->maximum_execution_slots()) {
    logger.Log(
        LogLevel::kError, "execution_slot_limit_exceeded",
        {{"variant", runtime.value()->model_variant_name()},
         {"requested_sessions", std::to_string(options.value().max_sessions)},
         {"maximum_execution_slots",
          std::to_string(runtime.value()->maximum_execution_slots())}});
    return 2;
  }
  logger.Log(LogLevel::kInfo, "model_loaded",
             {{"variant", runtime.value()->model_variant_name()},
              {"native_path", runtime.value()->selected_native_path()},
              {"weight_load_path", runtime.value()->weight_load_path()},
              {"weight_bytes", std::to_string(runtime.value()->weight_bytes())},
              {"assistant_weight_bytes",
               std::to_string(runtime.value()->assistant_weight_bytes())},
              {"load_ms", std::to_string(runtime.value()->load_milliseconds())}});
  auto slot_plan = PlanServerSlots(
      runtime.value(), session_options, processor.value(),
      options.value().max_sessions);
  if (!slot_plan.ok()) {
    logger.Log(LogLevel::kError, "slot_admission_failed",
               {{"error", slot_plan.status().message()}});
    return 2;
  }
  logger.Log(LogLevel::kInfo, "execution_slots_admitted",
             {{"slot_bytes", std::to_string(slot_plan.value().slot_bytes)},
              {"configured_slot_bytes",
               std::to_string(slot_plan.value().configured_slot_bytes)},
              {"session_limit", std::to_string(options.value().max_sessions)},
              {"device_free_bytes",
               std::to_string(slot_plan.value().device.free_bytes)},
              {"safety_margin_bytes",
               std::to_string(slot_plan.value().safety_margin_bytes)}});
  ServerState state(options.value().model_name, options.value().max_context,
                    std::move(processor).value(), runtime.value(),
                    session_options, options.value().max_sessions,
                    options.value().max_queued_requests);
  state.planned_slot_device_bytes = slot_plan.value().slot_bytes;
  state.configured_slot_device_bytes =
      slot_plan.value().configured_slot_bytes;
  state.device_total_bytes = slot_plan.value().device.total_bytes;
  state.device_free_after_probe_bytes = slot_plan.value().device.free_bytes;
  state.device_safety_margin_bytes = slot_plan.value().safety_margin_bytes;
  state.model_load_microseconds = static_cast<std::uint64_t>(
      std::max(0.0, runtime.value()->load_milliseconds()) * 1000.0);
  httplib::Server server;
  const std::size_t http_task_queue_limit =
      options.value().max_queued_requests + options.value().max_sessions + 16U;
  const auto http_workers = options.value().max_queued_requests + options.value().max_sessions + 4U;
  server.new_task_queue = [http_task_queue_limit, http_workers] {
    return new httplib::ThreadPool(http_workers, http_workers,
                                   http_task_queue_limit);
  };
  server.Get("/health",
             [&state](const httplib::Request&, httplib::Response& response) {
               std::size_t resident = 0U;
               std::size_t pending = 0U;
               {
                 std::lock_guard pool_lock(state.pool_mutex);
                 resident = state.sessions.size();
                 pending = state.pending_sessions.size();
               }
               const bool is_moe26b =
                   std::string_view(state.runtime->model_variant_name()) ==
                   "gemma4_moe_26b_a4b";
               const bool vision_profile = state.runtime->vision_module_loaded();
               const auto queue = state.request_queue.Snapshot();
               if (queue.draining) response.status = 503;
               response.set_content(
                   "{\"status\":" +
                       gem16::json::Quote(queue.draining ? "draining" : "ok") +
                       ",\"version\":" +
                       gem16::json::Quote(GEM16_VERSION_STRING) +
                       ",\"request_queue_depth\":" +
                       std::to_string(queue.queued) +
                       ",\"request_queue_active\":" +
                       std::to_string(queue.active) +
                       ",\"resident_sessions\":" +
                       std::to_string(resident) +
                       ",\"pending_session_creations\":" +
                       std::to_string(pending) +
                       ",\"session_limit\":" +
                       std::to_string(state.max_sessions) +
                       ",\"max_context_tokens\":" +
                       std::to_string(state.max_context) +
                       ",\"default_context\":" +
                       std::to_string(
                           state.runtime->default_context_tokens()) +
                       ",\"default_context_tokens\":" +
                       std::to_string(
                           state.runtime->default_context_tokens()) +
                       ",\"qualified_64k\":" +
                       (is_moe26b && !vision_profile
                            ? (state.runtime->qualified_64k() ? "true"
                                                             : "false")
                            : "null") +
                       ",\"base_max_context\":" +
                       (is_moe26b && !vision_profile
                            ? std::to_string(
                                  state.runtime->base_max_context_tokens())
                            : "null") +
                       ",\"mtp_max_context\":" +
                       (state.runtime->supports_mtp()
                            ? std::to_string(state.runtime->max_context_tokens())
                            : "null") +
                       ",\"model_variant\":" +
                       gem16::json::Quote(
                           state.runtime->model_variant_name()) +
                       ",\"profile_id\":" +
                       gem16::json::Quote(state.runtime->profile_id()) +
                       ",\"decode_mode\":" +
                       gem16::json::Quote(
                           state.session_options.mtp_draft_tokens == 2U
                               ? "fixed-d2"
                               : "ordinary") +
                       ",\"text_artifact_profile\":" +
                       gem16::json::Quote(
                           state.runtime->text_artifact_profile()) +
                       ",\"vision_artifact_profile\":" +
                       gem16::json::Quote(
                           state.runtime->vision_artifact_profile()) +
                       ",\"experimental\":" +
                       (state.runtime->experimental() ? "true" : "false") +
                       ",\"qualification_state\":" +
                       gem16::json::Quote(
                           state.runtime->qualification_state()) +
                       ",\"native_path\":" +
                       gem16::json::Quote(
                           state.runtime->selected_native_path()) +
                       ",\"weight_profile\":" +
                       gem16::json::Quote(state.runtime->artifact_profile()) +
                       ",\"head_format\":" +
                       gem16::json::Quote(state.runtime->head_format()) +
                       ",\"artifact_content_sha256\":" +
                       gem16::json::Quote(
                           state.runtime->artifact_content_sha256()) +
                       ",\"source_lock_sha256\":" +
                       gem16::json::Quote(
                           state.runtime->source_lock_sha256()) +
                       ",\"compiler_commit\":" +
                       gem16::json::Quote(state.runtime->compiler_commit()) +
                       ",\"resident_weight_bytes\":" +
                       std::to_string(state.runtime->weight_bytes()) +
                       ",\"assistant_weight_bytes\":" +
                       std::to_string(
                           state.runtime->assistant_weight_bytes()) +
                       ",\"assistant_workspace_bytes\":" +
                       std::to_string(
                           state.runtime->assistant_workspace_bytes()) +
                       ",\"vision_module_loaded\":" +
                       (state.runtime->vision_module_loaded() ? "true"
                                                              : "false") +
                       ",\"vision_weight_bytes\":" +
                       std::to_string(state.runtime->vision_weight_bytes()) +
                       ",\"vision_workspace_bytes\":" +
                       std::to_string(
                           state.runtime->vision_workspace_bytes()) +
                       ",\"maximum_images\":" +
                       std::to_string(state.runtime->maximum_images()) +
                       ",\"vision_soft_token_budgets\":" +
                       (vision_profile ? "[70,140,280]" : "[]") +
                       ",\"vision_max_soft_token_budget\":" +
                       (vision_profile
                            ? std::to_string(state.runtime
                                                 ->vision_max_soft_token_budget())
                            : "null") +
                       ",\"last_vision_soft_token_budget\":" +
                       (state.metrics.last_vision_soft_token_budget.load() ==
                                0U
                            ? "null"
                            : std::to_string(state.metrics
                                                 .last_vision_soft_token_budget
                                                 .load())) +
                       ",\"vision_max_context_tokens\":" +
                       (vision_profile
                            ? std::to_string(
                                  state.runtime->vision_max_context_tokens())
                            : "null") +
                       ",\"kv_cache_bytes\":" +
                       (is_moe26b
                            ? std::to_string(state.runtime->kv_cache_bytes())
                            : "null") +
                       ",\"workspace_bytes\":" +
                       (is_moe26b
                            ? std::to_string(state.runtime->workspace_bytes())
                            : "null") +
                       ",\"execution_slot_bytes\":" +
                       std::to_string(state.planned_slot_device_bytes) +
                       ",\"admission_free_bytes\":" +
                       std::to_string(state.device_free_after_probe_bytes) +
                       ",\"required_admission_margin_bytes\":" +
                       std::to_string(state.device_safety_margin_bytes) +
                       ",\"admission_headroom_bytes\":" +
                       std::to_string(
                           state.device_free_after_probe_bytes >
                                   state.device_safety_margin_bytes
                               ? state.device_free_after_probe_bytes -
                                     state.device_safety_margin_bytes
                               : 0U) +
                       ",\"text_only\":" +
                       ((!state.runtime->supports_audio() &&
                         !state.runtime->supports_vision())
                            ? "true"
                            : "false") +
                       ",\"capabilities\":{\"text\":true,\"audio\":" +
                       (state.runtime->supports_audio() ? "true" : "false") +
                       ",\"vision\":" +
                       (state.runtime->supports_vision() ? "true" : "false") +
                       ",\"mtp\":" +
                       (state.runtime->supports_mtp() ? "true" : "false") +
                       ",\"vision_mtp\":" +
                       (state.runtime->vision_mtp_supported() ? "true"
                                                              : "false") +
                       "}" +
                       ",\"mtp_draft_tokens\":" +
                       std::to_string(state.session_options.mtp_draft_tokens) +
                       ",\"mtp_adaptive\":" +
                       (state.session_options.mtp_adaptive ? "true" : "false") +
                       ",\"sampling\":{\"enabled\":" +
                       (state.session_options.sampling.enabled ? "true" : "false") +
                       ",\"temperature\":" +
                       std::to_string(state.session_options.sampling.temperature) +
                       ",\"top_k\":" +
                       std::to_string(state.session_options.sampling.top_k) +
                       ",\"top_p\":" +
                       std::to_string(state.session_options.sampling.top_p) +
                       ",\"min_p\":" +
                       std::to_string(state.session_options.sampling.min_p) +
                       ",\"repetition_penalty\":" +
                       std::to_string(
                           state.session_options.sampling.repetition_penalty) +
                       ",\"seed\":" +
                       std::to_string(state.session_options.sampling.seed) +
                       "}}",
                   "application/json; charset=utf-8");
             });
  server.Get("/live",
             [](const httplib::Request&, httplib::Response& response) {
               response.set_content("{\"status\":\"ok\"}",
                                    "application/json; charset=utf-8");
             });
  server.Get("/ready",
             [&state](const httplib::Request&, httplib::Response& response) {
               const bool ready = !state.request_queue.Snapshot().draining;
               if (!ready) response.status = 503;
               response.set_content(
                   std::string("{\"status\":\"") +
                       (ready ? "ready" : "draining") + "\"}",
                   "application/json; charset=utf-8");
             });
  server.Get("/metrics",
             [&state](const httplib::Request&, httplib::Response& response) {
               response.set_content(MetricsText(state),
                                    "text/plain; version=0.0.4; charset=utf-8");
             });
  server.Get("/v1/models",
             [&state](const httplib::Request&, httplib::Response& response) {
               response.set_content(
                   "{\"object\":\"list\",\"data\":[{\"id\":" +
                       gem16::json::Quote(state.model_name) +
                       ",\"object\":\"model\",\"owned_by\":\"gem16\"}]}",
                   "application/json; charset=utf-8");
             });
  server.Post("/v1/chat/completions",
              [&state](const httplib::Request& request,
                       httplib::Response& response) {
                HandleCompletion(state, request, response);
              });
  server.Post("/v1/responses",
              [&state](const httplib::Request& request,
                       httplib::Response& response) {
                HandleResponses(state, request, response);
              });
  server.Post(R"(/v1/responses/([^/]+)/cancel)",
              [&state](const httplib::Request& request,
                       httplib::Response& response) {
                HandleCancelResponse(state, request.matches[1].str(), response);
              });
  server.set_pre_request_handler(
      [&state, &options](const httplib::Request& request, httplib::Response& response) {
        g_request_log_context = {};
        auto id = gem16::server::MakeSecureId("req_gem16_");
        if (!id.ok()) {
          state.metrics.requests_failed.fetch_add(1U);
          response.status = 500;
          response.set_content(
              gem16::server::OpenAiErrorJson(
                  "failed to create request identity", "server_error"),
              "application/json; charset=utf-8");
          return httplib::Server::HandlerResponse::Handled;
        }
        g_request_log_context =
            {std::chrono::steady_clock::now(), std::move(id).value()};
        response.set_header("X-Request-Id", g_request_log_context.request_id);
        const auto policy_error = gem16::server::ValidateLocalHttpRequest(request, options.value().port);
        if (!policy_error.empty()) {
          SetError(state, gem16::Status(gem16::StatusCode::kInvalidArgument, policy_error), response);
          return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
      });
  server.set_logger([&state, &logger](const httplib::Request& request,
                                      const httplib::Response& response) {
    std::uint64_t elapsed_us = 0U;
    if (!g_request_log_context.request_id.empty()) {
      const auto elapsed =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() -
              g_request_log_context.started);
      elapsed_us = elapsed.count() < 0
                       ? 0U
                       : static_cast<std::uint64_t>(elapsed.count());
    }
    RecordRequestLatency(state, elapsed_us);
    const int status = response.status < 0 ? 200 : response.status;
    LogLevel level = status >= 500 ? LogLevel::kError
                                   : (status >= 400 ? LogLevel::kWarning
                                                    : LogLevel::kInfo);
    if (request.path == "/health" || request.path == "/live" ||
        request.path == "/ready" || request.path == "/metrics") {
      level = status >= 400 ? LogLevel::kWarning : LogLevel::kDebug;
    }
    logger.Log(level, "request_completed",
               {{"request_id", g_request_log_context.request_id},
                {"method", request.method},
                {"route", request.matched_route.empty()
                              ? request.path
                              : request.matched_route},
                {"status", std::to_string(status)},
                {"duration_ms",
                 std::to_string(static_cast<double>(elapsed_us) / 1000.0)},
                {"queue_wait_ms",
                 std::to_string(
                     static_cast<double>(
                         g_request_log_context.queue_wait_microseconds) /
                     1000.0)},
                {"request_bytes", std::to_string(request.body.size())}});
    g_request_log_context = {};
  });
  server.set_exception_handler(
      [&state, &logger](const httplib::Request& request,
                        httplib::Response& response, std::exception_ptr error) {
        std::string message = "unknown exception";
        try {
          if (error != nullptr) std::rethrow_exception(error);
        } catch (const std::exception& exception) {
          message = exception.what();
        } catch (...) {
        }
        state.metrics.requests_failed.fetch_add(1U);
        logger.Log(LogLevel::kError, "request_exception",
                   {{"request_id", g_request_log_context.request_id},
                    {"route", request.matched_route.empty()
                                  ? request.path
                                  : request.matched_route},
                    {"error", message}});
        response.status = 500;
        response.set_content(
            gem16::server::OpenAiErrorJson("internal server error",
                                           "server_error"),
            "application/json; charset=utf-8");
      });
  server.set_error_handler([](const httplib::Request&,
                              httplib::Response& response) {
    if (!response.body.empty()) {
      return httplib::Server::HandlerResponse::Unhandled;
    }
    const int status = response.status < 0 ? 500 : response.status;
    response.status = status;
    response.set_content(
        gem16::server::OpenAiErrorJson(
            status == 404 ? "endpoint not found" : "HTTP request failed",
            status >= 500 ? "server_error" : "invalid_request_error"),
        "application/json; charset=utf-8");
    return httplib::Server::HandlerResponse::Handled;
  });
  server.set_payload_max_length(kRequestPayloadLimit);
  server.set_read_timeout(std::chrono::seconds(30));
  server.set_write_timeout(std::chrono::seconds(60));
  server.set_keep_alive_timeout(5);
  server.set_keep_alive_max_count(100U);

  if (!server.bind_to_port(options.value().host, options.value().port)) {
    logger.Log(LogLevel::kError, "listen_failed",
               {{"host", options.value().host},
                {"port", std::to_string(options.value().port)}});
    return 2;
  }
  const auto startup_elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - startup_started);
  state.server_startup_microseconds = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - startup_started)
          .count());
  logger.Log(LogLevel::kInfo, "server_ready",
             {{"host", options.value().host},
              {"port", std::to_string(options.value().port)},
              {"model", state.model_name},
              {"session_limit", std::to_string(state.max_sessions)},
              {"max_queued_requests",
               std::to_string(options.value().max_queued_requests)},
              {"http_task_queue_limit",
               std::to_string(http_task_queue_limit)},
              {"startup_ms", std::to_string(startup_elapsed.count())}});

  g_shutdown_signal = 0;
  std::signal(SIGINT, HandleShutdownSignal);
  std::signal(SIGTERM, HandleShutdownSignal);
  std::atomic<bool> monitor_stop{false};
  std::thread signal_monitor([&] {
    while (!monitor_stop.load()) {
      if (g_shutdown_signal != 0) {
        const int signal = g_shutdown_signal;
        state.request_queue.StartDraining();
        {
          std::lock_guard lock(state.pool_mutex);
          for (auto& [id, entry] : state.sessions) entry->cancel_requested.store(true);
        }
        state.pool_changed.notify_all();
        logger.Log(LogLevel::kInfo, "shutdown_requested",
                   {{"signal", std::to_string(signal)}});
        server.stop();
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });
  const bool listened = server.listen_after_bind();
  state.request_queue.StartDraining();
  monitor_stop.store(true);
  signal_monitor.join();
  if (!listened && g_shutdown_signal == 0) {
    logger.Log(LogLevel::kError, "server_stopped_unexpectedly");
    return 2;
  }
  logger.Log(LogLevel::kInfo, "shutdown_completed",
             {{"signal", std::to_string(g_shutdown_signal)}});
  return 0;
}

}  // namespace

#if defined(_WIN32)

int wmain(int argc, wchar_t** wide_argv) {
  std::vector<std::string> utf8_arguments;
  utf8_arguments.reserve(static_cast<std::size_t>(argc));
  for (int index = 0; index < argc; ++index) {
    auto converted = gem16::cli::WideToUtf8(wide_argv[index]);
    if (!converted.has_value()) {
      std::cerr << "error: a command-line argument is not valid Unicode\n";
      return 64;
    }
    utf8_arguments.push_back(std::move(*converted));
  }
  std::vector<char*> arguments;
  arguments.reserve(utf8_arguments.size());
  for (std::string& argument : utf8_arguments) {
    arguments.push_back(argument.data());
  }
  return ServerMain(argc, arguments.data());
}

#else

int main(int argc, char** argv) { return ServerMain(argc, argv); }

#endif

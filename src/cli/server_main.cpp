#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#include "windows_utf8.h"
#endif

#include "httplib.h"

#include "gem16/chat.h"
#include "gem16/tokenizer.h"
#include "server/http_streaming.h"
#include "server/openai_chat.h"
#include "server/secure_id.h"
#include "server/session_pool.h"
#include "util/json.h"

namespace {

using gem16::server::AcquireNamedSession;
using gem16::server::AcquireResponseSession;
using gem16::server::ClearActiveResponse;
using gem16::server::CommitResponsesRequest;
using gem16::server::CreateSession;
using gem16::server::DiscardSession;
using gem16::server::FinishSse;
using gem16::server::IndexResponse;
using gem16::server::MakeChatIdentity;
using gem16::server::MakeResponsesIdentity;
using gem16::server::MetricsText;
using gem16::server::PrepareResponsesRequest;
using gem16::server::RecordGeneration;
using gem16::server::ReleaseSession;
using gem16::server::ServerState;
using gem16::server::SessionEntry;
using gem16::server::SessionLease;
using gem16::server::SetActiveResponse;
using gem16::server::UnindexResponse;
using gem16::server::WriteSse;

constexpr std::uint64_t kServerVramSafetyBytes =
    700U * 1024U * 1024U;

struct SlotMemoryPlan {
  std::uint64_t slot_bytes = 0U;
  std::uint64_t configured_slot_bytes = 0U;
  gem16::DeviceMemoryInfo device;
};

struct Options {
  std::filesystem::path model_directory;
  std::filesystem::path assistant_model_directory;
  std::string model_name = "gem16";
  std::string host = "127.0.0.1";
  int port = 8080;
  std::uint64_t max_context = 8192U;
  gem16::KvCacheMode kv_cache_mode = gem16::KvCacheMode::kCheckpointFp8;
  std::uint32_t mtp_draft_tokens = 0U;
  std::uint32_t max_sessions = 2U;
  bool max_sessions_explicit = false;
  bool mtp_adaptive = false;
  bool greedy = false;
};

void PrintUsage() {
  std::cout
      << "Usage: gem16-server --model <checkpoint> [options]\n"
      << "  --model-name <id>       Served OpenAI model id (default: gem16)\n"
      << "  --host <address>        Listen address (default: 127.0.0.1)\n"
      << "  --port <port>           Listen port (default: 8080)\n"
      << "  --max-context <tokens>  Session context capacity (default: 8192)\n"
      << "  --max-sessions <count>   Resident slots (default: 2; 26B profile: 1)\n"
      << "  --kv-cache fp8|bf16\n"
      << "  --greedy                Disable checkpoint-recommended sampling\n"
      << "  --assistant-model <checkpoint> --mtp-draft-tokens 1|2|4 [--mtp-adaptive]\n";
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
    } else if (argument == "--max-sessions" && index + 1 < argc) {
      std::uint64_t value = 0U;
      if (!ParseUnsigned(argv[++index], value) || value == 0U ||
          value > 64U) {
        return gem16::Status(gem16::StatusCode::kInvalidArgument,
                             "--max-sessions must be in [1, 64]");
      }
      options.max_sessions = static_cast<std::uint32_t>(value);
      options.max_sessions_explicit = true;
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
    } else if (argument == "--mtp-adaptive") {
      options.mtp_adaptive = true;
    } else if (argument == "--greedy") {
      options.greedy = true;
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

void SetError(const gem16::Status& status, httplib::Response& response) {
  response.status = HttpStatus(status);
  response.set_content(
      gem16::server::OpenAiErrorJson(
          status.message(), response.status >= 500 ? "server_error"
                                                   : "invalid_request_error"),
      "application/json; charset=utf-8");
}

struct CancellationContext {
  std::atomic<bool>* cancel_requested = nullptr;
  std::atomic<std::uint64_t>* cancellations_observed = nullptr;
  std::atomic<std::uint64_t>* client_disconnects = nullptr;
  httplib::DataSink* sink = nullptr;
};

gem16::Status CheckCancellation(void* opaque_context,
                                const gem16::GenerationEvent& event) {
  auto* context = static_cast<CancellationContext*>(opaque_context);
  if (context == nullptr ||
      event.kind != gem16::GenerationEventKind::kToken) {
    return gem16::Status(gem16::StatusCode::kInternal,
                         "invalid cancellation callback");
  }
  if (context->cancel_requested != nullptr &&
      context->cancel_requested->load()) {
    if (context->cancellations_observed != nullptr) {
      context->cancellations_observed->fetch_add(1U);
    }
    return gem16::Status(gem16::StatusCode::kCancelled,
                         "generation was cancelled");
  }
  if (context->sink != nullptr && context->sink->is_writable &&
      !context->sink->is_writable()) {
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
  auto parsed = gem16::server::ParseChatCompletionsRequest(
      request.body, {state.max_context});
  if (!parsed.ok()) {
    state.metrics.requests_failed.fetch_add(1U);
    SetError(parsed.status(), response);
    return;
  }
  if (parsed.value().model != state.model_name) {
    state.metrics.requests_failed.fetch_add(1U);
    SetError(gem16::Status(gem16::StatusCode::kNotFound,
                           "requested model is not served"),
             response);
    return;
  }
  auto identity_result = MakeChatIdentity(state);
  if (!identity_result.ok()) {
    state.metrics.requests_failed.fetch_add(1U);
    SetError(identity_result.status(), response);
    return;
  }
  gem16::server::OpenAiResponseIdentity identity =
      std::move(identity_result).value();
  std::string session_id = request.get_header_value("X-Gem16-Session-Id");
  if (session_id.empty()) {
    auto generated_id = gem16::server::MakeSecureId("session_");
    if (!generated_id.ok()) {
      state.metrics.requests_failed.fetch_add(1U);
      SetError(generated_id.status(), response);
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
    SetError(gem16::Status(gem16::StatusCode::kInvalidArgument,
                           "X-Gem16-Session-Id is invalid"),
             response);
    return;
  }
  auto acquired = AcquireNamedSession(state, "chat:" + session_id);
  if (!acquired.ok()) {
    state.metrics.requests_failed.fetch_add(1U);
    SetError(acquired.status(), response);
    return;
  }
  std::shared_ptr<SessionEntry> entry = std::move(acquired).value();
  response.set_header("X-Gem16-Session-Id", session_id);
  if (!parsed.value().stream) {
    std::lock_guard inference_lock(entry->inference_mutex);
    entry->cancel_requested.store(false);
    CancellationContext cancellation{
        &entry->cancel_requested, &state.metrics.cancellations_observed,
        &state.metrics.client_disconnects, nullptr};
    const auto generation_start = std::chrono::steady_clock::now();
    auto generated = entry->session.Generate(
        parsed.value().generation, CheckCancellation, &cancellation);
    if (!generated.ok()) {
      state.metrics.requests_failed.fetch_add(1U);
      SetError(generated.status(), response);
      if (entry->session.is_poisoned()) DiscardSession(state, entry);
      ReleaseSession(state, entry);
      return;
    }
    RecordGeneration(state, generated.value(),
                     std::chrono::steady_clock::now() - generation_start);
    response.set_content(
        gem16::server::ChatCompletionJson(identity, generated.value()),
        "application/json; charset=utf-8");
    ReleaseSession(state, entry);
    return;
  }

  struct ProviderState {
    ServerState* server = nullptr;
    gem16::ChatGenerationRequest generation;
    gem16::server::OpenAiResponseIdentity identity;
    std::shared_ptr<SessionEntry> entry;
    std::unique_ptr<SessionLease> lease;
    bool include_usage = false;
    bool ran = false;
  };
  auto provider = std::make_shared<ProviderState>();
  provider->server = &state;
  provider->generation = std::move(parsed.value().generation);
  provider->identity = identity;
  provider->entry = entry;
  provider->lease = std::make_unique<SessionLease>(state, entry);
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
  auto parsed = gem16::server::ParseResponsesRequest(
      request.body, {state.max_context});
  if (!parsed.ok()) {
    state.metrics.requests_failed.fetch_add(1U);
    SetError(parsed.status(), response);
    return;
  }
  if (parsed.value().model != state.model_name) {
    state.metrics.requests_failed.fetch_add(1U);
    SetError(gem16::Status(gem16::StatusCode::kNotFound,
                           "requested model is not served"),
             response);
    return;
  }
  auto identity_result = MakeResponsesIdentity(state);
  if (!identity_result.ok()) {
    state.metrics.requests_failed.fetch_add(1U);
    SetError(identity_result.status(), response);
    return;
  }
  gem16::server::OpenAiResponseIdentity identity =
      std::move(identity_result).value();
  gem16::Result<std::shared_ptr<SessionEntry>> acquired =
      parsed.value().previous_response_id.has_value()
          ? AcquireResponseSession(state,
                                   *parsed.value().previous_response_id)
          : CreateSession(state, "responses:" + identity.id);
  if (!acquired.ok()) {
    state.metrics.requests_failed.fetch_add(1U);
    SetError(acquired.status(), response);
    return;
  }
  std::shared_ptr<SessionEntry> entry = std::move(acquired).value();
  IndexResponse(state, entry, identity.id);
  if (!parsed.value().stream) {
    std::lock_guard inference_lock(entry->inference_mutex);
    gem16::Status status = PrepareResponsesRequest(*entry, parsed.value());
    if (!status.ok()) {
      state.metrics.requests_failed.fetch_add(1U);
      SetError(status, response);
      UnindexResponse(state, identity.id);
      ReleaseSession(state, entry);
      return;
    }
    entry->cancel_requested.store(false);
    CancellationContext cancellation{
        &entry->cancel_requested, &state.metrics.cancellations_observed,
        &state.metrics.client_disconnects, nullptr};
    const auto generation_start = std::chrono::steady_clock::now();
    auto generated = entry->session.Generate(
        parsed.value().generation, CheckCancellation, &cancellation);
    if (!generated.ok()) {
      state.metrics.requests_failed.fetch_add(1U);
      SetError(generated.status(), response);
      if (entry->session.is_poisoned()) DiscardSession(state, entry);
      ReleaseSession(state, entry);
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
    ReleaseSession(state, entry);
    return;
  }

  struct ProviderState {
    ServerState* server = nullptr;
    gem16::server::OpenAiResponsesRequest request;
    gem16::server::OpenAiResponseIdentity identity;
    std::shared_ptr<SessionEntry> entry;
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
  provider->entry = entry;
  provider->lease = std::make_unique<SessionLease>(state, entry);
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
  if (after.free_bytes < kServerVramSafetyBytes ||
      additional_slots >
          (after.free_bytes - kServerVramSafetyBytes) / slot_bytes) {
    return gem16::Status(
        gem16::StatusCode::kResourceExhausted,
        "--max-sessions and --max-context exceed VRAM after the required "
        "700 MiB safety margin");
  }
  return SlotMemoryPlan{slot_bytes, slot_bytes * max_sessions, after};
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
    SetError(gem16::Status(gem16::StatusCode::kNotFound,
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
  auto processor =
      gem16::GemmaChatProcessor::Load(options.value().model_directory);
  if (!processor.ok()) {
    std::cerr << "error: " << processor.status().message() << '\n';
    return 2;
  }
  gem16::ChatSessionOptions session_options;
  session_options.model_directory = options.value().model_directory;
  session_options.assistant_model_directory =
      options.value().assistant_model_directory;
  session_options.max_context_tokens = options.value().max_context;
  session_options.kv_cache_mode = options.value().kv_cache_mode;
  session_options.sampling =
      processor.value().generation_controls().recommended_sampling;
  if (options.value().greedy) session_options.sampling.enabled = false;
  session_options.mtp_draft_tokens = options.value().mtp_draft_tokens;
  session_options.mtp_adaptive = options.value().mtp_adaptive;
  auto runtime = gem16::ModelRuntime::Load(
      {options.value().model_directory,
       options.value().assistant_model_directory,
       options.value().max_context, 0});
  if (!runtime.ok()) {
    std::cerr << "error: " << runtime.status().message() << '\n';
    return 2;
  }
  if (!options.value().max_sessions_explicit &&
      runtime.value()->maximum_execution_slots() == 1U) {
    options.value().max_sessions = 1U;
  } else if (options.value().max_sessions >
             runtime.value()->maximum_execution_slots()) {
    std::cerr << "error: model profile "
              << runtime.value()->model_variant_name() << " supports at most "
              << runtime.value()->maximum_execution_slots()
              << " resident execution slot; use --max-sessions 1\n";
    return 2;
  }
  std::cout << "model_runtime weights=" << runtime.value()->weight_bytes()
            << " assistant_weights="
            << runtime.value()->assistant_weight_bytes()
            << " load_ms=" << runtime.value()->load_milliseconds() << '\n';
  auto slot_plan = PlanServerSlots(
      runtime.value(), session_options, processor.value(),
      options.value().max_sessions);
  if (!slot_plan.ok()) {
    std::cerr << "error: server memory admission failed: "
              << slot_plan.status().message() << '\n';
    return 2;
  }
  std::cout << "execution_slot planned_bytes=" << slot_plan.value().slot_bytes
            << " configured_bytes="
            << slot_plan.value().configured_slot_bytes
            << " device_total_bytes=" << slot_plan.value().device.total_bytes
            << " free_with_probe_bytes=" << slot_plan.value().device.free_bytes
            << " safety_margin_bytes=" << kServerVramSafetyBytes << '\n';
  ServerState state(options.value().model_name, options.value().max_context,
                    std::move(processor).value(), runtime.value(),
                    session_options, options.value().max_sessions);
  state.planned_slot_device_bytes = slot_plan.value().slot_bytes;
  state.configured_slot_device_bytes =
      slot_plan.value().configured_slot_bytes;
  state.device_total_bytes = slot_plan.value().device.total_bytes;
  state.device_safety_margin_bytes = kServerVramSafetyBytes;
  httplib::Server server;
  server.Get("/health",
             [&state](const httplib::Request&, httplib::Response& response) {
               std::size_t resident = 0U;
               std::size_t pending = 0U;
               {
                 std::lock_guard pool_lock(state.pool_mutex);
                 resident = state.sessions.size();
                 pending = state.pending_sessions.size();
               }
               response.set_content(
                   "{\"status\":\"ok\",\"resident_sessions\":" +
                       std::to_string(resident) +
                       ",\"pending_session_creations\":" +
                       std::to_string(pending) +
                       ",\"session_limit\":" +
                       std::to_string(state.max_sessions) +
                       ",\"max_context_tokens\":" +
                       std::to_string(state.max_context) +
                       ",\"model_variant\":" +
                       gem16::json::Quote(
                           state.runtime->model_variant_name()) +
                       ",\"native_path\":" +
                       gem16::json::Quote(
                           state.runtime->selected_native_path()) +
                       ",\"capabilities\":{\"text\":true,\"audio\":" +
                       (state.runtime->supports_audio() ? "true" : "false") +
                       ",\"vision\":" +
                       (state.runtime->supports_vision() ? "true" : "false") +
                       ",\"mtp\":" +
                       (state.runtime->supports_mtp() ? "true" : "false") +
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
  server.set_payload_max_length(16U * 1024U * 1024U);
  std::cout << "gem16 OpenAI-compatible server listening on http://"
            << options.value().host << ':' << options.value().port
            << " (model " << state.model_name << ", max sessions "
            << state.max_sessions << ", variant "
            << state.runtime->model_variant_name() << ", native path "
            << state.runtime->selected_native_path() << ")\n";
  if (!server.listen(options.value().host, options.value().port)) {
    std::cerr << "error: failed to listen on requested address\n";
    return 2;
  }
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

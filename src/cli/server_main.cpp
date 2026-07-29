#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#include "windows_utf8.h"
#endif

#include "httplib.h"

#include "gem16/chat.h"
#include "gem16/tokenizer.h"
#include "server/openai_chat.h"
#include "util/json.h"

namespace {

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
      << "  --max-sessions <count>   Resident execution slots (default: 2)\n"
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

std::int64_t UnixSeconds() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

bool WriteSse(httplib::DataSink& sink, std::string_view payload) {
  const std::string record = "data: " + std::string(payload) + "\n\n";
  return sink.write(record.data(), record.size());
}

std::size_t CompleteUtf8Prefix(std::string_view text) {
  std::size_t position = 0U;
  while (position < text.size()) {
    const unsigned char first = static_cast<unsigned char>(text[position]);
    std::size_t width = 1U;
    if (first <= 0x7FU) {
      width = 1U;
    } else if (first >= 0xC2U && first <= 0xDFU) {
      width = 2U;
    } else if (first >= 0xE0U && first <= 0xEFU) {
      width = 3U;
    } else if (first >= 0xF0U && first <= 0xF4U) {
      width = 4U;
    } else {
      return position;
    }
    if (position + width > text.size()) return position;
    bool valid = true;
    for (std::size_t offset = 1U; offset < width; ++offset) {
      const unsigned char continuation =
          static_cast<unsigned char>(text[position + offset]);
      valid = valid && continuation >= 0x80U && continuation <= 0xBFU;
    }
    if (!valid) return position;
    position += width;
  }
  return position;
}

struct StreamingContext {
  StreamingContext(const gem16::GemmaChatProcessor& chat_processor,
                   gem16::server::OpenAiResponseIdentity response_identity,
                   httplib::DataSink& data_sink)
      : processor(&chat_processor),
        identity(std::move(response_identity)),
        sink(&data_sink),
        channels(chat_processor.generation_controls()) {}

  const gem16::GemmaChatProcessor* processor = nullptr;
  gem16::server::OpenAiResponseIdentity identity;
  httplib::DataSink* sink = nullptr;
  gem16::ResponseChannelTracker channels;
  std::map<std::string, std::size_t, std::less<>> tool_indices;
  std::string utf8_pending;
  std::string reasoning_utf8_pending;
  bool inside_tool_call = false;
  std::atomic<bool>* cancel_requested = nullptr;
  std::atomic<std::uint64_t>* cancellations_observed = nullptr;
  std::atomic<std::uint64_t>* client_disconnects = nullptr;
};

gem16::Status StreamDelta(StreamingContext& context,
                          const gem16::GenerationEvent& event) {
  std::string delta;
  if (event.kind == gem16::GenerationEventKind::kTextDelta) {
    delta = "{\"content\":" + gem16::json::Quote(event.text_delta) + "}";
  } else if (event.kind == gem16::GenerationEventKind::kToolCallStart) {
    const std::size_t index = context.tool_indices.size();
    context.tool_indices.emplace(event.tool_call_id, index);
    delta = "{\"tool_calls\":[{\"index\":" + std::to_string(index) +
            ",\"id\":" + gem16::json::Quote(event.tool_call_id) +
            ",\"type\":\"function\",\"function\":{\"name\":" +
            gem16::json::Quote(event.tool_name) +
            ",\"arguments\":\"\"}}]}";
  } else if (event.kind ==
             gem16::GenerationEventKind::kToolCallArgumentsDelta) {
    const auto index = context.tool_indices.find(event.tool_call_id);
    if (index == context.tool_indices.end()) {
      return gem16::Status(gem16::StatusCode::kInternal,
                           "tool arguments arrived before tool start");
    }
    delta = "{\"tool_calls\":[{\"index\":" +
            std::to_string(index->second) +
            ",\"function\":{\"arguments\":" +
            gem16::json::Quote(event.text_delta) + "}}]}";
  } else {
    return gem16::Status::Ok();
  }
  if (!WriteSse(*context.sink, gem16::server::ChatCompletionChunkJson(
                                   context.identity, delta))) {
    return gem16::Status(gem16::StatusCode::kIoError,
                         "client disconnected during SSE generation");
  }
  return gem16::Status::Ok();
}

gem16::Status FeedVisibleText(StreamingContext& context,
                              std::string_view bytes, bool final) {
  context.utf8_pending.append(bytes);
  const std::size_t complete = CompleteUtf8Prefix(context.utf8_pending);
  if (complete != 0U) {
    gem16::GenerationEvent event;
    event.kind = gem16::GenerationEventKind::kTextDelta;
    event.text_delta = context.utf8_pending.substr(0U, complete);
    context.utf8_pending.erase(0U, complete);
    const gem16::Status status = StreamDelta(context, event);
    if (!status.ok()) return status;
  }
  if (!final) return gem16::Status::Ok();
  if (!context.utf8_pending.empty()) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         "model response ends with incomplete UTF-8");
  }
  return gem16::Status::Ok();
}

gem16::Status FeedReasoningText(StreamingContext& context,
                                std::string_view bytes, bool final) {
  context.reasoning_utf8_pending.append(bytes);
  const std::size_t complete =
      CompleteUtf8Prefix(context.reasoning_utf8_pending);
  if (complete != 0U) {
    const std::string delta =
        "{\"reasoning_content\":" +
        gem16::json::Quote(
            std::string_view(context.reasoning_utf8_pending)
                .substr(0U, complete)) +
        "}";
    context.reasoning_utf8_pending.erase(0U, complete);
    if (!WriteSse(*context.sink,
                  gem16::server::ChatCompletionChunkJson(context.identity,
                                                          delta))) {
      return gem16::Status(gem16::StatusCode::kIoError,
                           "client disconnected during SSE generation");
    }
  }
  if (final && !context.reasoning_utf8_pending.empty()) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         "reasoning response ends with incomplete UTF-8");
  }
  return gem16::Status::Ok();
}

gem16::Status StreamToken(void* opaque_context,
                          const gem16::GenerationEvent& event) {
  auto* context = static_cast<StreamingContext*>(opaque_context);
  if (context == nullptr || context->processor == nullptr ||
      context->sink == nullptr ||
      event.kind != gem16::GenerationEventKind::kToken) {
    return gem16::Status(gem16::StatusCode::kInternal,
                         "invalid OpenAI stream callback");
  }
  if (context->cancel_requested != nullptr &&
      context->cancel_requested->load()) {
    if (context->cancellations_observed != nullptr) {
      context->cancellations_observed->fetch_add(1U);
    }
    return gem16::Status(gem16::StatusCode::kCancelled,
                         "generation was cancelled");
  }
  if (context->sink->is_writable && !context->sink->is_writable()) {
    if (context->client_disconnects != nullptr) {
      context->client_disconnects->fetch_add(1U);
    }
    return gem16::Status(gem16::StatusCode::kCancelled,
                         "client disconnected during generation");
  }
  const gem16::ResponseTokenChannel channel =
      context->channels.Observe(event.token_id);
  if (channel == gem16::ResponseTokenChannel::kControl) {
    return gem16::Status::Ok();
  }
  const std::uint32_t token_id = event.token_id;
  auto visible = context->processor->Decode(
      std::span<const std::uint32_t>(&token_id, 1U), true);
  if (!visible.ok()) return visible.status();
  if (channel == gem16::ResponseTokenChannel::kReasoning) {
    return FeedReasoningText(*context, visible.value(), false);
  }
  auto raw = context->processor->Decode(
      std::span<const std::uint32_t>(&token_id, 1U), false);
  if (!raw.ok()) return raw.status();
  if (raw.value() == "<|tool_call>") {
    context->inside_tool_call = true;
    return gem16::Status::Ok();
  }
  if (raw.value() == "<tool_call|>") {
    context->inside_tool_call = false;
    return gem16::Status::Ok();
  }
  if (context->inside_tool_call) return gem16::Status::Ok();
  return FeedVisibleText(*context, visible.value(), false);
}

struct ResponsesChain {
    std::string latest_response_id;
    std::vector<gem16::GenerationMessage> messages;
    std::vector<gem16::GenerationToolDefinition> tools;
    gem16::GenerationToolChoice tool_choice;
    bool initialized = false;
};

struct SessionEntry {
  SessionEntry(std::string session_id, gem16::ChatSession chat_session)
      : id(std::move(session_id)), session(std::move(chat_session)) {}

  std::string id;
  gem16::ChatSession session;
  std::mutex inference_mutex;
  ResponsesChain responses_chain;
  std::atomic<std::uint32_t> active_requests{0U};
  std::atomic<bool> cancel_requested{false};
  std::atomic<std::uint64_t> last_used{0U};
  std::string active_response_id;
};

struct ServerMetrics {
  std::atomic<std::uint64_t> requests_total{0U};
  std::atomic<std::uint64_t> requests_failed{0U};
  std::atomic<std::uint64_t> active_requests{0U};
  std::atomic<std::uint64_t> sessions_created{0U};
  std::atomic<std::uint64_t> sessions_evicted{0U};
  std::atomic<std::uint64_t> cancellations_requested{0U};
  std::atomic<std::uint64_t> cancellations_observed{0U};
  std::atomic<std::uint64_t> client_disconnects{0U};
  std::atomic<std::uint64_t> input_tokens{0U};
  std::atomic<std::uint64_t> output_tokens{0U};
  std::atomic<std::uint64_t> generation_microseconds{0U};
  std::atomic<std::uint64_t> last_slot_bytes{0U};
};

struct ServerState {
  ServerState(std::string served_model_name, std::uint64_t context_limit,
              gem16::GemmaChatProcessor chat_processor,
              std::shared_ptr<gem16::ModelRuntime> model_runtime,
              gem16::ChatSessionOptions chat_session_options,
              std::uint32_t session_limit)
      : model_name(std::move(served_model_name)),
        max_context(context_limit),
        processor(std::move(chat_processor)),
        runtime(std::move(model_runtime)),
        session_options(std::move(chat_session_options)),
        max_sessions(session_limit) {}

  std::string model_name;
  std::uint64_t max_context = 0U;
  gem16::GemmaChatProcessor processor;
  std::shared_ptr<gem16::ModelRuntime> runtime;
  gem16::ChatSessionOptions session_options;
  std::uint32_t max_sessions = 2U;
  std::mutex pool_mutex;
  std::unordered_map<std::string, std::shared_ptr<SessionEntry>> sessions;
  std::unordered_map<std::string, std::weak_ptr<SessionEntry>> response_index;
  std::atomic<std::uint64_t> response_counter{1U};
  std::atomic<std::uint64_t> session_counter{1U};
  std::atomic<std::uint64_t> lru_clock{1U};
  ServerMetrics metrics;
};

gem16::server::OpenAiResponseIdentity MakeIdentity(ServerState& state) {
  return {"chatcmpl-gem16-" +
              std::to_string(state.response_counter.fetch_add(1U)),
          state.model_name, UnixSeconds()};
}

gem16::server::OpenAiResponseIdentity MakeResponsesIdentity(
    ServerState& state) {
  return {"resp_gem16_" +
              std::to_string(state.response_counter.fetch_add(1U)),
          state.model_name, UnixSeconds()};
}

void EraseSessionLocked(ServerState& state, const std::string& id) {
  const auto found = state.sessions.find(id);
  if (found == state.sessions.end()) return;
  const std::shared_ptr<SessionEntry> entry = found->second;
  for (auto iterator = state.response_index.begin();
       iterator != state.response_index.end();) {
    const std::shared_ptr<SessionEntry> indexed = iterator->second.lock();
    if (indexed == nullptr || indexed == entry) {
      iterator = state.response_index.erase(iterator);
    } else {
      ++iterator;
    }
  }
  state.sessions.erase(found);
}

gem16::Result<std::shared_ptr<SessionEntry>> CreateSession(
    ServerState& state, std::string id) {
  std::lock_guard pool_lock(state.pool_mutex);
  if (state.sessions.contains(id)) {
    return gem16::Status(gem16::StatusCode::kInvalidArgument,
                         "session ID is already resident");
  }
  if (state.sessions.size() >= state.max_sessions) {
    auto victim = state.sessions.end();
    for (auto iterator = state.sessions.begin();
         iterator != state.sessions.end(); ++iterator) {
      if (iterator->second->active_requests.load() != 0U) continue;
      if (victim == state.sessions.end() ||
          iterator->second->last_used.load() <
              victim->second->last_used.load()) {
        victim = iterator;
      }
    }
    if (victim == state.sessions.end()) {
      return gem16::Status(
          gem16::StatusCode::kResourceExhausted,
          "all resident execution slots are active");
    }
    const std::string victim_id = victim->first;
    EraseSessionLocked(state, victim_id);
    state.metrics.sessions_evicted.fetch_add(1U);
  }
  auto session = gem16::ChatSession::Create(
      state.runtime, state.session_options, state.processor);
  if (!session.ok()) return session.status();
  auto entry = std::make_shared<SessionEntry>(
      std::move(id), std::move(session).value());
  entry->active_requests.store(1U);
  entry->last_used.store(state.lru_clock.fetch_add(1U));
  state.sessions.emplace(entry->id, entry);
  state.metrics.sessions_created.fetch_add(1U);
  state.metrics.active_requests.fetch_add(1U);
  return entry;
}

gem16::Result<std::shared_ptr<SessionEntry>> AcquireNamedSession(
    ServerState& state, const std::string& id) {
  {
    std::lock_guard pool_lock(state.pool_mutex);
    const auto found = state.sessions.find(id);
    if (found != state.sessions.end()) {
      if (found->second->active_requests.load() != 0U) {
        return gem16::Status(
            gem16::StatusCode::kResourceExhausted,
            "session already has an active request");
      }
      found->second->active_requests.fetch_add(1U);
      found->second->last_used.store(state.lru_clock.fetch_add(1U));
      state.metrics.active_requests.fetch_add(1U);
      return found->second;
    }
  }
  return CreateSession(state, id);
}

gem16::Result<std::shared_ptr<SessionEntry>> AcquireResponseSession(
    ServerState& state, const std::string& response_id) {
  std::lock_guard pool_lock(state.pool_mutex);
  const auto found = state.response_index.find(response_id);
  if (found == state.response_index.end()) {
    return gem16::Status(gem16::StatusCode::kNotFound,
                         "previous_response_id is not resident");
  }
  std::shared_ptr<SessionEntry> entry = found->second.lock();
  if (entry == nullptr) {
    state.response_index.erase(found);
    return gem16::Status(gem16::StatusCode::kNotFound,
                         "previous_response_id was evicted");
  }
  if (entry->active_requests.load() != 0U) {
    return gem16::Status(gem16::StatusCode::kResourceExhausted,
                         "response session already has an active request");
  }
  entry->active_requests.fetch_add(1U);
  entry->last_used.store(state.lru_clock.fetch_add(1U));
  state.metrics.active_requests.fetch_add(1U);
  return entry;
}

void ReleaseSession(ServerState& state,
                    const std::shared_ptr<SessionEntry>& entry) {
  entry->last_used.store(state.lru_clock.fetch_add(1U));
  entry->active_requests.fetch_sub(1U);
  state.metrics.active_requests.fetch_sub(1U);
}

void DiscardSession(ServerState& state,
                    const std::shared_ptr<SessionEntry>& entry) {
  std::lock_guard pool_lock(state.pool_mutex);
  EraseSessionLocked(state, entry->id);
}

class SessionLease {
 public:
  SessionLease(ServerState& state, std::shared_ptr<SessionEntry> entry)
      : state_(&state), entry_(std::move(entry)) {}
  SessionLease(const SessionLease&) = delete;
  SessionLease& operator=(const SessionLease&) = delete;
  ~SessionLease() {
    if (state_ == nullptr) return;
    if (discard_) DiscardSession(*state_, entry_);
    ReleaseSession(*state_, entry_);
  }
  void Discard() { discard_ = true; }
  void Keep() { discard_ = false; }

 private:
  ServerState* state_ = nullptr;
  std::shared_ptr<SessionEntry> entry_;
  bool discard_ = false;
};

gem16::Status PrepareResponsesRequest(
    SessionEntry& entry, gem16::server::OpenAiResponsesRequest& request) {
  ResponsesChain& chain = entry.responses_chain;
  if (!request.previous_response_id.has_value()) {
    if (chain.initialized) {
      return gem16::Status(gem16::StatusCode::kInvalidArgument,
                           "response session already has a root");
    }
    return gem16::Status::Ok();
  }
  if (!chain.initialized ||
      *request.previous_response_id != chain.latest_response_id) {
    return gem16::Status(
        gem16::StatusCode::kNotFound,
        "previous_response_id is not the latest resident response");
  }
  if (request.tools_present && request.generation.tools != chain.tools) {
    return gem16::Status(
        gem16::StatusCode::kInvalidArgument,
        "tools must remain identical on a resident response chain");
  }
  if (!request.tools_present) request.generation.tools = chain.tools;
  if (request.tool_choice_present &&
      request.generation.tool_choice != chain.tool_choice) {
    return gem16::Status(
        gem16::StatusCode::kInvalidArgument,
        "tool_choice must remain identical on a resident response chain");
  }
  if (!request.tool_choice_present) {
    request.generation.tool_choice = chain.tool_choice;
  }
  std::vector<gem16::GenerationMessage> complete = chain.messages;
  complete.insert(complete.end(), request.generation.messages.begin(),
                  request.generation.messages.end());
  request.generation.messages = std::move(complete);
  return gem16::Status::Ok();
}

void CommitResponsesRequest(
    SessionEntry& entry, const gem16::server::OpenAiResponsesRequest& request,
    const gem16::ChatGenerationResponse& response,
    std::string response_id) {
  ResponsesChain& chain = entry.responses_chain;
  chain.messages = request.generation.messages;
  gem16::GenerationMessage assistant;
  assistant.role = "assistant";
  if (!response.assistant_text.empty() || response.tool_calls.empty()) {
    assistant.content.push_back(
        gem16::GenerationContentPart::Text(response.assistant_text));
  }
  for (const gem16::GenerationToolCall& call : response.tool_calls) {
    assistant.content.push_back(
        gem16::GenerationContentPart::ToolCall(call));
  }
  chain.messages.push_back(std::move(assistant));
  chain.tools = request.generation.tools;
  chain.tool_choice = request.generation.tool_choice;
  chain.latest_response_id = std::move(response_id);
  chain.initialized = true;
}

void IndexResponse(ServerState& state,
                   const std::shared_ptr<SessionEntry>& entry,
                   const std::string& response_id) {
  std::lock_guard pool_lock(state.pool_mutex);
  state.response_index[response_id] = entry;
}

void UnindexResponse(ServerState& state, std::string_view response_id) {
  std::lock_guard pool_lock(state.pool_mutex);
  state.response_index.erase(std::string(response_id));
}

void SetActiveResponse(ServerState& state,
                       const std::shared_ptr<SessionEntry>& entry,
                       std::string_view response_id) {
  std::lock_guard pool_lock(state.pool_mutex);
  entry->active_response_id = response_id;
}

void ClearActiveResponse(ServerState& state,
                         const std::shared_ptr<SessionEntry>& entry,
                         std::string_view response_id) {
  std::lock_guard pool_lock(state.pool_mutex);
  if (entry->active_response_id == response_id) {
    entry->active_response_id.clear();
  }
}

void RecordGeneration(ServerState& state,
                      const gem16::ChatGenerationResponse& response,
                      std::chrono::steady_clock::duration elapsed) {
  state.metrics.input_tokens.fetch_add(response.prompt_token_ids.size());
  state.metrics.output_tokens.fetch_add(
      response.inference.output_token_ids.size());
  state.metrics.generation_microseconds.fetch_add(
      static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(elapsed)
              .count()));
  state.metrics.last_slot_bytes.store(
      response.inference.kv_cache_bytes + response.inference.workspace_bytes +
      response.inference.assistant_workspace_bytes +
      response.inference.decode_graph_device_bytes);
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
  const gem16::server::OpenAiResponseIdentity identity = MakeIdentity(state);
  std::string session_id = request.get_header_value("X-Gem16-Session-Id");
  if (session_id.empty()) {
    session_id = "session_" +
                 std::to_string(state.session_counter.fetch_add(1U));
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
      DiscardSession(state, entry);
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
    bool include_usage = false;
    bool ran = false;
  };
  auto provider = std::make_shared<ProviderState>();
  provider->server = &state;
  provider->generation = std::move(parsed.value().generation);
  provider->identity = identity;
  provider->entry = entry;
  provider->include_usage = parsed.value().include_usage;
  response.set_header("Cache-Control", "no-cache");
  response.set_header("X-Accel-Buffering", "no");
  response.set_chunked_content_provider(
      "text/event-stream; charset=utf-8",
      [provider](std::size_t, httplib::DataSink& sink) {
        if (provider->ran) return false;
        provider->ran = true;
        SessionLease lease(*provider->server, provider->entry);
        std::lock_guard inference_lock(provider->entry->inference_mutex);
        provider->entry->cancel_requested.store(false);
        if (!WriteSse(sink, gem16::server::ChatCompletionChunkJson(
                                provider->identity,
                                "{\"role\":\"assistant\"}"))) {
          lease.Discard();
          return false;
        }
        StreamingContext stream(provider->server->processor,
                                provider->identity, sink);
        stream.cancel_requested = &provider->entry->cancel_requested;
        stream.cancellations_observed =
            &provider->server->metrics.cancellations_observed;
        stream.client_disconnects =
            &provider->server->metrics.client_disconnects;
        const auto generation_start = std::chrono::steady_clock::now();
        lease.Discard();
        auto generated = provider->entry->session.Generate(
            provider->generation, StreamToken, &stream);
        if (generated.ok()) {
          const gem16::Status final_status = FeedVisibleText(stream, {}, true);
          if (!final_status.ok()) generated = final_status;
        }
        if (generated.ok()) {
          const gem16::Status final_status =
              FeedReasoningText(stream, {}, true);
          if (!final_status.ok()) generated = final_status;
        }
        if (generated.ok() && stream.inside_tool_call) {
          generated = gem16::Status(
              gem16::StatusCode::kDataLoss,
              "model response ends inside a streamed tool call");
        }
        if (!generated.ok()) {
          provider->server->metrics.requests_failed.fetch_add(1U);
          WriteSse(sink, gem16::server::OpenAiErrorJson(
                             generated.status().message(), "server_error"));
          sink.done();
          lease.Discard();
          return true;
        }
        RecordGeneration(*provider->server, generated.value(),
                         std::chrono::steady_clock::now() - generation_start);
        for (const gem16::GenerationToolCall& call :
             generated.value().tool_calls) {
          gem16::GenerationEvent start;
          start.kind = gem16::GenerationEventKind::kToolCallStart;
          start.tool_call_id = call.id;
          start.tool_name = call.name;
          gem16::Status stream_status = StreamDelta(stream, start);
          if (!stream_status.ok()) return false;
          gem16::GenerationEvent arguments;
          arguments.kind =
              gem16::GenerationEventKind::kToolCallArgumentsDelta;
          arguments.tool_call_id = call.id;
          arguments.tool_name = call.name;
          arguments.text_delta = call.arguments_json;
          stream_status = StreamDelta(stream, arguments);
          if (!stream_status.ok()) return false;
        }
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
        WriteSse(sink, "[DONE]");
        sink.done();
        lease.Keep();
        return true;
      });
}

bool WriteResponsesFinalEvents(
    httplib::DataSink& sink,
    const gem16::server::OpenAiResponseIdentity& identity,
    const gem16::server::OpenAiResponsesRequest& request,
    const gem16::ChatGenerationResponse& generated,
    std::uint64_t& sequence) {
  std::size_t output_index = 0U;
  if (!generated.reasoning_text.empty()) {
    const std::string item_id = "rs_" + identity.id;
    const std::string item =
        "{\"id\":" + gem16::json::Quote(item_id) +
        ",\"type\":\"reasoning\",\"summary\":[],\"content\":[],"
        "\"status\":\"in_progress\"}";
    if (!WriteSse(sink, "{\"type\":\"response.output_item.added\","
                        "\"output_index\":" +
                            std::to_string(output_index) + ",\"item\":" +
                            item + ",\"sequence_number\":" +
                            std::to_string(sequence++) + "}")) {
      return false;
    }
    if (!WriteSse(sink, "{\"type\":\"response.reasoning_text.delta\","
                        "\"item_id\":" +
                            gem16::json::Quote(item_id) +
                            ",\"output_index\":" +
                            std::to_string(output_index) +
                            ",\"content_index\":0,\"delta\":" +
                            gem16::json::Quote(generated.reasoning_text) +
                            ",\"sequence_number\":" +
                            std::to_string(sequence++) + "}")) {
      return false;
    }
    if (!WriteSse(sink, "{\"type\":\"response.reasoning_text.done\","
                        "\"item_id\":" +
                            gem16::json::Quote(item_id) +
                            ",\"output_index\":" +
                            std::to_string(output_index) +
                            ",\"content_index\":0,\"text\":" +
                            gem16::json::Quote(generated.reasoning_text) +
                            ",\"sequence_number\":" +
                            std::to_string(sequence++) + "}")) {
      return false;
    }
    ++output_index;
  }
  if (!generated.assistant_text.empty() || generated.tool_calls.empty()) {
    const std::string item_id = "msg_" + identity.id;
    const std::string added_item =
        "{\"id\":" + gem16::json::Quote(item_id) +
        ",\"type\":\"message\",\"status\":\"in_progress\","
        "\"role\":\"assistant\",\"content\":[]}";
    if (!WriteSse(sink, "{\"type\":\"response.output_item.added\","
                        "\"output_index\":" +
                            std::to_string(output_index) + ",\"item\":" +
                            added_item + ",\"sequence_number\":" +
                            std::to_string(sequence++) + "}")) {
      return false;
    }
    const std::string empty_part =
        "{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[]}";
    if (!WriteSse(sink, "{\"type\":\"response.content_part.added\","
                        "\"item_id\":" +
                            gem16::json::Quote(item_id) +
                            ",\"output_index\":" +
                            std::to_string(output_index) +
                            ",\"content_index\":0,\"part\":" + empty_part +
                            ",\"sequence_number\":" +
                            std::to_string(sequence++) + "}")) {
      return false;
    }
    if (!generated.assistant_text.empty() &&
        !WriteSse(sink, "{\"type\":\"response.output_text.delta\","
                        "\"item_id\":" +
                            gem16::json::Quote(item_id) +
                            ",\"output_index\":" +
                            std::to_string(output_index) +
                            ",\"content_index\":0,\"delta\":" +
                            gem16::json::Quote(generated.assistant_text) +
                            ",\"logprobs\":[],\"sequence_number\":" +
                            std::to_string(sequence++) + "}")) {
      return false;
    }
    const std::string completed_part =
        "{\"type\":\"output_text\",\"text\":" +
        gem16::json::Quote(generated.assistant_text) +
        ",\"annotations\":[]}";
    if (!WriteSse(sink, "{\"type\":\"response.output_text.done\","
                        "\"item_id\":" +
                            gem16::json::Quote(item_id) +
                            ",\"output_index\":" +
                            std::to_string(output_index) +
                            ",\"content_index\":0,\"text\":" +
                            gem16::json::Quote(generated.assistant_text) +
                            ",\"logprobs\":[],\"sequence_number\":" +
                            std::to_string(sequence++) + "}")) {
      return false;
    }
    if (!WriteSse(sink, "{\"type\":\"response.content_part.done\","
                        "\"item_id\":" +
                            gem16::json::Quote(item_id) +
                            ",\"output_index\":" +
                            std::to_string(output_index) +
                            ",\"content_index\":0,\"part\":" +
                            completed_part + ",\"sequence_number\":" +
                            std::to_string(sequence++) + "}")) {
      return false;
    }
    const std::string done_item =
        "{\"id\":" + gem16::json::Quote(item_id) +
        ",\"type\":\"message\",\"status\":\"completed\","
        "\"role\":\"assistant\",\"content\":[" + completed_part + "]}";
    if (!WriteSse(sink, "{\"type\":\"response.output_item.done\","
                        "\"output_index\":" +
                            std::to_string(output_index) + ",\"item\":" +
                            done_item + ",\"sequence_number\":" +
                            std::to_string(sequence++) + "}")) {
      return false;
    }
    ++output_index;
  }
  for (std::size_t call_index = 0U;
       call_index < generated.tool_calls.size(); ++call_index) {
    const gem16::GenerationToolCall& call = generated.tool_calls[call_index];
    const std::string item_id =
        "fc_" + identity.id + "_" + std::to_string(call_index);
    const auto item = [&](std::string_view status) {
      return "{\"id\":" + gem16::json::Quote(item_id) +
             ",\"type\":\"function_call\",\"status\":" +
             gem16::json::Quote(status) + ",\"arguments\":" +
             gem16::json::Quote(call.arguments_json) + ",\"call_id\":" +
             gem16::json::Quote(call.id) + ",\"name\":" +
             gem16::json::Quote(call.name) + "}";
    };
    if (!WriteSse(sink, "{\"type\":\"response.output_item.added\","
                        "\"output_index\":" +
                            std::to_string(output_index) + ",\"item\":" +
                            item("in_progress") +
                            ",\"sequence_number\":" +
                            std::to_string(sequence++) + "}")) {
      return false;
    }
    if (!WriteSse(sink, "{\"type\":\"response.function_call_arguments.done\","
                        "\"item_id\":" +
                            gem16::json::Quote(item_id) +
                            ",\"output_index\":" +
                            std::to_string(output_index) + ",\"arguments\":" +
                            gem16::json::Quote(call.arguments_json) +
                            ",\"name\":" + gem16::json::Quote(call.name) +
                            ",\"sequence_number\":" +
                            std::to_string(sequence++) + "}")) {
      return false;
    }
    if (!WriteSse(sink, "{\"type\":\"response.output_item.done\","
                        "\"output_index\":" +
                            std::to_string(output_index) + ",\"item\":" +
                            item("completed") + ",\"sequence_number\":" +
                            std::to_string(sequence++) + "}")) {
      return false;
    }
    ++output_index;
  }
  const std::string completed = gem16::server::ResponseJson(
      identity, request, generated);
  return WriteSse(sink, "{\"type\":\"response.completed\",\"response\":" +
                            completed + ",\"sequence_number\":" +
                            std::to_string(sequence++) + "}");
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
  const gem16::server::OpenAiResponseIdentity identity =
      MakeResponsesIdentity(state);
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
      DiscardSession(state, entry);
      ReleaseSession(state, entry);
      return;
    }
    RecordGeneration(state, generated.value(),
                     std::chrono::steady_clock::now() - generation_start);
    CommitResponsesRequest(*entry, parsed.value(), generated.value(),
                           identity.id);
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
    bool ran = false;
  };
  auto provider = std::make_shared<ProviderState>();
  provider->server = &state;
  provider->request = std::move(parsed).value();
  provider->identity = identity;
  provider->entry = entry;
  response.set_header("Cache-Control", "no-cache");
  response.set_header("X-Accel-Buffering", "no");
  response.set_chunked_content_provider(
      "text/event-stream; charset=utf-8",
      [provider](std::size_t, httplib::DataSink& sink) {
        if (provider->ran) return false;
        provider->ran = true;
        std::uint64_t sequence = 0U;
        SessionLease lease(*provider->server, provider->entry);
        std::lock_guard inference_lock(provider->entry->inference_mutex);
        gem16::Status status = PrepareResponsesRequest(
            *provider->entry, provider->request);
        if (!status.ok()) {
          provider->server->metrics.requests_failed.fetch_add(1U);
          WriteSse(sink, "{\"type\":\"error\",\"code\":null,\"message\":" +
                             gem16::json::Quote(status.message()) +
                             ",\"param\":null,\"sequence_number\":" +
                             std::to_string(sequence++) + "}");
          sink.done();
          UnindexResponse(*provider->server, provider->identity.id);
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
                    ",\"sequence_number\":" +
                    std::to_string(sequence++) + "}")) {
          ClearActiveResponse(*provider->server, provider->entry,
                              provider->identity.id);
          UnindexResponse(*provider->server, provider->identity.id);
          return false;
        }
        CancellationContext cancellation{
            &provider->entry->cancel_requested,
            &provider->server->metrics.cancellations_observed,
            &provider->server->metrics.client_disconnects, &sink};
        const auto generation_start = std::chrono::steady_clock::now();
        lease.Discard();
        auto generated = provider->entry->session.Generate(
            provider->request.generation, CheckCancellation, &cancellation);
        ClearActiveResponse(*provider->server, provider->entry,
                            provider->identity.id);
        if (!generated.ok()) {
          provider->server->metrics.requests_failed.fetch_add(1U);
          WriteSse(sink, "{\"type\":\"error\",\"code\":null,\"message\":" +
                             gem16::json::Quote(generated.status().message()) +
                             ",\"param\":null,\"sequence_number\":" +
                             std::to_string(sequence++) + "}");
          sink.done();
          return true;
        }
        RecordGeneration(*provider->server, generated.value(),
                         std::chrono::steady_clock::now() - generation_start);
        CommitResponsesRequest(*provider->entry, provider->request,
                               generated.value(), provider->identity.id);
        const bool written = WriteResponsesFinalEvents(
            sink, provider->identity, provider->request, generated.value(),
            sequence);
        sink.done();
        if (written) lease.Keep();
        return written;
      });
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

std::string MetricsText(ServerState& state) {
  std::size_t resident_sessions = 0U;
  {
    std::lock_guard pool_lock(state.pool_mutex);
    resident_sessions = state.sessions.size();
  }
  const auto metric = [](std::string_view name, std::uint64_t value) {
    return std::string(name) + " " + std::to_string(value) + "\n";
  };
  std::string output;
  output.append("# TYPE gem16_requests_total counter\n");
  output.append(metric("gem16_requests_total",
                       state.metrics.requests_total.load()));
  output.append("# TYPE gem16_requests_failed_total counter\n");
  output.append(metric("gem16_requests_failed_total",
                       state.metrics.requests_failed.load()));
  output.append("# TYPE gem16_active_requests gauge\n");
  output.append(metric("gem16_active_requests",
                       state.metrics.active_requests.load()));
  output.append("# TYPE gem16_resident_sessions gauge\n");
  output.append(metric("gem16_resident_sessions", resident_sessions));
  output.append(metric("gem16_session_limit", state.max_sessions));
  output.append(metric("gem16_sessions_created_total",
                       state.metrics.sessions_created.load()));
  output.append(metric("gem16_sessions_evicted_total",
                       state.metrics.sessions_evicted.load()));
  output.append(metric("gem16_cancellations_requested_total",
                       state.metrics.cancellations_requested.load()));
  output.append(metric("gem16_cancellations_observed_total",
                       state.metrics.cancellations_observed.load()));
  output.append(metric("gem16_client_disconnects_total",
                       state.metrics.client_disconnects.load()));
  output.append(metric("gem16_input_tokens_total",
                       state.metrics.input_tokens.load()));
  output.append(metric("gem16_output_tokens_total",
                       state.metrics.output_tokens.load()));
  output.append(metric("gem16_generation_microseconds_total",
                       state.metrics.generation_microseconds.load()));
  output.append(metric("gem16_model_weight_bytes",
                       state.runtime->weight_bytes()));
  output.append(metric("gem16_assistant_weight_bytes",
                       state.runtime->assistant_weight_bytes()));
  output.append(metric("gem16_last_execution_slot_bytes",
                       state.metrics.last_slot_bytes.load()));
  return output;
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
       options.value().assistant_model_directory});
  if (!runtime.ok()) {
    std::cerr << "error: " << runtime.status().message() << '\n';
    return 2;
  }
  std::cout << "model_runtime weights=" << runtime.value()->weight_bytes()
            << " assistant_weights="
            << runtime.value()->assistant_weight_bytes()
            << " load_ms=" << runtime.value()->load_milliseconds() << '\n';
  ServerState state(options.value().model_name, options.value().max_context,
                    std::move(processor).value(), runtime.value(),
                    session_options, options.value().max_sessions);
  httplib::Server server;
  server.Get("/health",
             [&state](const httplib::Request&, httplib::Response& response) {
               std::size_t resident = 0U;
               {
                 std::lock_guard pool_lock(state.pool_mutex);
                 resident = state.sessions.size();
               }
               response.set_content(
                   "{\"status\":\"ok\",\"resident_sessions\":" +
                       std::to_string(resident) +
                       ",\"session_limit\":" +
                       std::to_string(state.max_sessions) + "}",
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
            << " (model " << state.model_name
            << ", one resident conversation / one execution slot)\n";
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

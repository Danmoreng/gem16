#include "server/session_pool.h"

#include <algorithm>
#include <chrono>
#include <utility>

namespace gem16::server {
namespace {

std::int64_t UnixSeconds() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

}  // namespace

ServerState::ServerState(std::string served_model_name,
                         std::uint64_t context_limit,
                         GemmaChatProcessor chat_processor,
                         std::shared_ptr<ModelRuntime> model_runtime,
                         ChatSessionOptions chat_session_options,
                         std::uint32_t session_limit)
    : model_name(std::move(served_model_name)),
      max_context(context_limit),
      processor(std::move(chat_processor)),
      runtime(std::move(model_runtime)),
      session_options(std::move(chat_session_options)),
      max_sessions(session_limit) {}

gem16::server::OpenAiResponseIdentity MakeChatIdentity(ServerState& state) {
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

SessionLease::SessionLease(ServerState& state,
                           std::shared_ptr<SessionEntry> entry)
    : state_(&state), entry_(std::move(entry)) {}

SessionLease::~SessionLease() {
  if (state_ == nullptr) return;
  if (discard_) DiscardSession(*state_, entry_);
  ReleaseSession(*state_, entry_);
}

void SessionLease::Discard() { discard_ = true; }
void SessionLease::Keep() { discard_ = false; }

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
  state.metrics.cached_input_tokens.fetch_add(
      response.inference.prompt_cached_tokens);
  state.metrics.cache_write_tokens.fetch_add(
      response.inference.prompt_cache_write_tokens);
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
  output.append(metric("gem16_cached_input_tokens_total",
                       state.metrics.cached_input_tokens.load()));
  output.append(metric("gem16_cache_write_tokens_total",
                       state.metrics.cache_write_tokens.load()));
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


}  // namespace gem16::server

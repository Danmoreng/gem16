#include "server/session_pool.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <utility>

#include "server/secure_id.h"

namespace gem16::server {

std::int64_t UnixSecondsNow() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

ServerState::ServerState(std::string served_model_name,
                         std::uint64_t context_limit,
                         GemmaChatProcessor chat_processor,
                         std::shared_ptr<ModelRuntime> model_runtime,
                         ChatSessionOptions chat_session_options,
                         std::uint32_t session_limit,
                         std::size_t max_queued_requests)
    : model_name(std::move(served_model_name)),
      max_context(context_limit),
      processor(std::move(chat_processor)),
      runtime(std::move(model_runtime)),
      session_options(std::move(chat_session_options)),
      max_sessions(session_limit),
      request_queue(session_limit, max_queued_requests) {}

namespace {

void ObserveLatency(ServerMetrics::LatencyHistogram& histogram,
                    std::uint64_t microseconds) {
  histogram.count.fetch_add(1U);
  histogram.sum_microseconds.fetch_add(microseconds);
  for (std::size_t index = 0U;
       index < ServerMetrics::kLatencyBucketsUs.size(); ++index) {
    if (microseconds <= ServerMetrics::kLatencyBucketsUs[index]) {
      histogram.buckets[index].fetch_add(1U);
    }
  }
}

void AppendLatencyHistogram(
    std::string& output, std::string_view name,
    const ServerMetrics::LatencyHistogram& histogram) {
  output.append("# TYPE ").append(name).append(" histogram\n");
  for (std::size_t index = 0U;
       index < ServerMetrics::kLatencyBucketsUs.size(); ++index) {
    const double seconds =
        static_cast<double>(ServerMetrics::kLatencyBucketsUs[index]) /
        1'000'000.0;
    output.append(name)
        .append("_bucket{le=\"")
        .append(std::to_string(seconds))
        .append("\"} ")
        .append(std::to_string(histogram.buckets[index].load()))
        .append("\n");
  }
  const std::uint64_t count = histogram.count.load();
  output.append(name)
      .append("_bucket{le=\"+Inf\"} ")
      .append(std::to_string(count))
      .append("\n");
  output.append(name)
      .append("_sum ")
      .append(std::to_string(
          static_cast<double>(histogram.sum_microseconds.load()) /
          1'000'000.0))
      .append("\n");
  output.append(name)
      .append("_count ")
      .append(std::to_string(count))
      .append("\n");
}

}  // namespace

Result<OpenAiResponseIdentity> MakeChatIdentity(const ServerState& state) {
  auto id = MakeSecureId("chatcmpl-gem16-");
  if (!id.ok()) return id.status();
  return OpenAiResponseIdentity{std::move(id).value(), state.model_name,
                                UnixSecondsNow(), std::nullopt};
}

Result<OpenAiResponseIdentity> MakeResponsesIdentity(
    const ServerState& state) {
  auto id = MakeSecureId("resp_gem16_");
  if (!id.ok()) return id.status();
  return OpenAiResponseIdentity{std::move(id).value(), state.model_name,
                                UnixSecondsNow(), std::nullopt};
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
  {
    std::lock_guard pool_lock(state.pool_mutex);
    if (state.sessions.contains(id) || state.pending_sessions.contains(id)) {
      return gem16::Status(gem16::StatusCode::kInvalidArgument,
                           "session ID is already resident or being created");
    }
    if (state.sessions.size() + state.pending_sessions.size() >=
        state.max_sessions) {
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
            "all resident execution slots are active or being created");
      }
      const std::string victim_id = victim->first;
      EraseSessionLocked(state, victim_id);
      state.metrics.sessions_evicted.fetch_add(1U);
    }
    state.pending_sessions.insert(id);
  }

  struct Reservation {
    ServerState& state;
    const std::string& id;
    ~Reservation() {
      { std::lock_guard lock(state.pool_mutex); state.pending_sessions.erase(id); }
      state.pool_changed.notify_all();
    }
  } reservation{state, id};

  // CUDA arenas and graphs may take a material amount of time to construct.
  // The reservation above keeps the pool bounded while allowing unrelated
  // acquire, cancellation, health, and metrics operations to proceed.
  auto session = gem16::ChatSession::Create(
      state.runtime, state.session_options, state.processor);
  if (!session.ok()) {
    {
      std::lock_guard pool_lock(state.pool_mutex);
      state.pending_sessions.erase(id);
    }
    state.pool_changed.notify_all();
    return session.status();
  }
  auto entry = std::make_shared<SessionEntry>(
      id, std::move(session).value());
  entry->active_requests.store(1U);
  entry->last_used.store(state.lru_clock.fetch_add(1U));
  {
    std::lock_guard pool_lock(state.pool_mutex);
    if (state.pending_sessions.erase(entry->id) != 1U ||
        state.sessions.contains(entry->id)) {
      return gem16::Status(gem16::StatusCode::kInternal,
                           "session reservation was lost before publication");
    }
    state.sessions.emplace(entry->id, entry);
    state.metrics.sessions_created.fetch_add(1U);
    state.metrics.active_requests.fetch_add(1U);
  }
  state.pool_changed.notify_all();
  return entry;
}

gem16::Result<std::shared_ptr<SessionEntry>> CreateSessionQueued(
    ServerState& state, std::string id) {
  for (;;) {
    auto created = CreateSession(state, id);
    if (created.ok()) return created;
    if (created.status().code() != gem16::StatusCode::kResourceExhausted) {
      return created.status();
    }
    std::unique_lock pool_lock(state.pool_mutex);
    state.pool_changed.wait(pool_lock, [&] {
      if (state.sessions.contains(id) || state.pending_sessions.contains(id)) {
        return false;
      }
      if (state.sessions.size() + state.pending_sessions.size() <
          state.max_sessions) {
        return true;
      }
      return std::any_of(state.sessions.begin(), state.sessions.end(),
                         [](const auto& item) {
                           return item.second->active_requests.load() == 0U;
                         });
    });
  }
}

gem16::Result<std::shared_ptr<SessionEntry>> AcquireNamedSession(
    ServerState& state, const std::string& id) {
  for (;;) {
    std::unique_lock pool_lock(state.pool_mutex);
    const auto found = state.sessions.find(id);
    if (found != state.sessions.end()) {
      if (found->second->active_requests.load() != 0U) {
        const std::shared_ptr<SessionEntry> busy_entry = found->second;
        state.pool_changed.wait(pool_lock, [busy_entry] {
          return busy_entry->active_requests.load() == 0U;
        });
        continue;
      }
      found->second->active_requests.fetch_add(1U);
      found->second->last_used.store(state.lru_clock.fetch_add(1U));
      state.metrics.active_requests.fetch_add(1U);
      return found->second;
    }
    if (state.pending_sessions.contains(id)) {
      state.pool_changed.wait(pool_lock, [&] {
        return !state.pending_sessions.contains(id);
      });
      continue;
    }
    pool_lock.unlock();
    auto created = CreateSession(state, id);
    if (created.ok()) return created;
    if (created.status().code() != gem16::StatusCode::kResourceExhausted &&
        created.status().code() != gem16::StatusCode::kInvalidArgument) {
      return created.status();
    }
    std::unique_lock retry_lock(state.pool_mutex);
    state.pool_changed.wait(retry_lock, [&] {
      const auto current = state.sessions.find(id);
      if (current != state.sessions.end()) {
        return current->second->active_requests.load() == 0U;
      }
      if (state.pending_sessions.contains(id)) return false;
      if (state.sessions.size() + state.pending_sessions.size() <
          state.max_sessions) {
        return true;
      }
      return std::any_of(state.sessions.begin(), state.sessions.end(),
                         [](const auto& item) {
                           return item.second->active_requests.load() == 0U;
                         });
    });
  }
}

gem16::Result<std::shared_ptr<SessionEntry>> AcquireResponseSession(
    ServerState& state, const std::string& response_id) {
  for (;;) {
    std::unique_lock pool_lock(state.pool_mutex);
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
      state.pool_changed.wait(pool_lock, [entry] {
        return entry->active_requests.load() == 0U;
      });
      continue;
    }
    entry->active_requests.fetch_add(1U);
    entry->last_used.store(state.lru_clock.fetch_add(1U));
    state.metrics.active_requests.fetch_add(1U);
    return entry;
  }
}

void ReleaseSession(ServerState& state,
                    const std::shared_ptr<SessionEntry>& entry) {
  entry->last_used.store(state.lru_clock.fetch_add(1U));
  entry->active_requests.fetch_sub(1U);
  state.metrics.active_requests.fetch_sub(1U);
  state.pool_changed.notify_all();
}

void DiscardSession(ServerState& state,
                    const std::shared_ptr<SessionEntry>& entry) {
  {
    std::lock_guard pool_lock(state.pool_mutex);
    EraseSessionLocked(state, entry->id);
  }
}

SessionLease::SessionLease(ServerState& state,
                           std::shared_ptr<SessionEntry> entry)
    : state_(&state), entry_(std::move(entry)),
      uncaught_on_entry_(std::uncaught_exceptions()) {}

SessionLease::SessionLease(SessionLease&& other) noexcept
    : state_(std::exchange(other.state_, nullptr)), entry_(std::move(other.entry_)),
      discard_(other.discard_), uncaught_on_entry_(other.uncaught_on_entry_) {}

SessionLease::~SessionLease() {
  if (state_ == nullptr) return;
  if (discard_ || std::uncaught_exceptions() > uncaught_on_entry_) DiscardSession(*state_, entry_);
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
  state.metrics.prompt_microseconds.fetch_add(static_cast<std::uint64_t>(
      std::max(0.0, response.inference.prompt_milliseconds) * 1000.0));
  state.metrics.decode_microseconds.fetch_add(static_cast<std::uint64_t>(
      std::max(0.0, response.inference.decode_milliseconds) * 1000.0));
  const auto add_milliseconds = [](std::atomic<std::uint64_t>& metric,
                                   double milliseconds) {
    metric.fetch_add(static_cast<std::uint64_t>(
        std::max(0.0, milliseconds) * 1000.0));
  };
  add_milliseconds(state.metrics.image_decode_microseconds,
                   response.inference.image_decode_milliseconds);
  add_milliseconds(state.metrics.image_resize_patchify_microseconds,
                   response.inference.image_resize_patchify_milliseconds);
  add_milliseconds(state.metrics.vision_upload_microseconds,
                   response.inference.vision_upload_milliseconds);
  add_milliseconds(state.metrics.vision_tower_microseconds,
                   response.inference.vision_tower_milliseconds);
  add_milliseconds(state.metrics.vision_pool_project_microseconds,
                   response.inference.vision_pool_project_milliseconds);
  add_milliseconds(state.metrics.text_prefill_microseconds,
                   response.inference.text_prefill_milliseconds);
  state.metrics.decode_measured_tokens.fetch_add(
      response.inference.output_token_ids.empty()
          ? 0U
          : static_cast<std::uint64_t>(
                response.inference.output_token_ids.size() - 1U));
  state.metrics.mtp_proposed_tokens.fetch_add(
      response.inference.mtp_proposed_tokens);
  state.metrics.mtp_accepted_tokens.fetch_add(
      response.inference.mtp_accepted_tokens);
  state.metrics.mtp_rejected_tokens.fetch_add(
      response.inference.mtp_rejected_tokens);
  state.metrics.mtp_verification_groups.fetch_add(
      response.inference.mtp_verification_groups);
  state.metrics.mtp_d1_groups.fetch_add(response.inference.mtp_d1_groups);
  state.metrics.mtp_d2_groups.fetch_add(response.inference.mtp_d2_groups);
  state.metrics.mtp_d4_groups.fetch_add(response.inference.mtp_d4_groups);
  state.metrics.mtp_ordinary_fallback_tokens.fetch_add(
      response.inference.mtp_ordinary_fallback_tokens);
  state.metrics.fallback_count.fetch_add(response.inference.fallback_count);
  state.metrics.token_loop_allocation_count.fetch_add(
      response.inference.token_loop_allocations ? 1U : 0U);
  state.metrics.last_slot_bytes.store(
      response.inference.kv_cache_bytes + response.inference.workspace_bytes +
      response.inference.assistant_workspace_bytes +
      response.inference.decode_graph_device_bytes);
  const auto generation_us = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
  ObserveLatency(state.metrics.generation_latency, generation_us);
  ObserveLatency(state.metrics.prompt_latency, static_cast<std::uint64_t>(
      std::max(0.0, response.inference.prompt_milliseconds) * 1000.0));
  ObserveLatency(state.metrics.decode_latency, static_cast<std::uint64_t>(
      std::max(0.0, response.inference.decode_milliseconds) * 1000.0));
}

void RecordRequestLatency(ServerState& state, std::uint64_t microseconds) {
  ObserveLatency(state.metrics.request_latency, microseconds);
}

void RecordQueueAdmission(ServerState& state, std::uint64_t microseconds) {
  state.metrics.queue_admissions.fetch_add(1U);
  state.metrics.queue_wait_microseconds.fetch_add(microseconds);
  if (microseconds != 0U) state.metrics.queue_waits.fetch_add(1U);
  ObserveLatency(state.metrics.queue_latency, microseconds);
}

std::string VisionMetricsText(const ServerMetrics& metrics) {
  const auto metric = [](std::string_view name, std::uint64_t value) {
    return std::string(name) + " " + std::to_string(value) + "\n";
  };
  std::string output;
  output.reserve(1600U);
  output.append("# TYPE gem16_image_decode_microseconds_total counter\n");
  output.append(metric("gem16_image_decode_microseconds_total",
                       metrics.image_decode_microseconds.load()));
  output.append("# TYPE gem16_image_resize_patchify_microseconds_total counter\n");
  output.append(metric("gem16_image_resize_patchify_microseconds_total",
                       metrics.image_resize_patchify_microseconds.load()));
  output.append("# TYPE gem16_vision_upload_microseconds_total counter\n");
  output.append(metric("gem16_vision_upload_microseconds_total",
                       metrics.vision_upload_microseconds.load()));
  output.append("# TYPE gem16_vision_tower_microseconds_total counter\n");
  output.append(metric("gem16_vision_tower_microseconds_total",
                       metrics.vision_tower_microseconds.load()));
  output.append("# TYPE gem16_vision_pool_project_microseconds_total counter\n");
  output.append(metric("gem16_vision_pool_project_microseconds_total",
                       metrics.vision_pool_project_microseconds.load()));
  output.append("# TYPE gem16_text_prefill_microseconds_total counter\n");
  output.append(metric("gem16_text_prefill_microseconds_total",
                       metrics.text_prefill_microseconds.load()));
  output.append("# TYPE gem16_vision_requests_total counter\n");
  output.append(metric("gem16_vision_requests_total",
                       metrics.vision_requests.load()));
  output.append("# TYPE gem16_vision_failures_total counter\n");
  output.append(metric("gem16_vision_failures_total",
                       metrics.vision_failures.load()));
  output.append("# TYPE gem16_vision_d2_rejections_total counter\n");
  output.append(metric("gem16_vision_d2_rejections_total",
                       metrics.vision_d2_rejections.load()));
  output.append("# TYPE gem16_vision_budget_70_total counter\n");
  output.append(metric("gem16_vision_budget_70_total",
                       metrics.vision_budget_70.load()));
  output.append("# TYPE gem16_vision_budget_140_total counter\n");
  output.append(metric("gem16_vision_budget_140_total",
                       metrics.vision_budget_140.load()));
  output.append("# TYPE gem16_vision_budget_280_total counter\n");
  output.append(metric("gem16_vision_budget_280_total",
                       metrics.vision_budget_280.load()));
  output.append("# TYPE gem16_vision_artifact_validation_failures_total counter\n");
  output.append(metric("gem16_vision_artifact_validation_failures_total",
                       metrics.vision_artifact_validation_failures.load()));
  output.append("# TYPE gem16_last_vision_soft_token_budget gauge\n");
  output.append(metric("gem16_last_vision_soft_token_budget",
                       metrics.last_vision_soft_token_budget.load()));
  return output;
}

std::string MetricsText(ServerState& state) {
  std::size_t resident_sessions = 0U;
  std::size_t pending_sessions = 0U;
  {
    std::lock_guard pool_lock(state.pool_mutex);
    resident_sessions = state.sessions.size();
    pending_sessions = state.pending_sessions.size();
  }
  const auto metric = [](std::string_view name, std::uint64_t value) {
    return std::string(name) + " " + std::to_string(value) + "\n";
  };
  std::string output;
  const RequestQueueSnapshot queue = state.request_queue.Snapshot();
  output.append("# TYPE gem16_requests_total counter\n");
  output.append(metric("gem16_requests_total",
                       state.metrics.requests_total.load()));
  output.append("# TYPE gem16_requests_failed_total counter\n");
  output.append(metric("gem16_requests_failed_total",
                       state.metrics.requests_failed.load()));
  output.append("# TYPE gem16_active_requests gauge\n");
  output.append(metric("gem16_active_requests",
                       state.metrics.active_requests.load()));
  output.append("# TYPE gem16_request_queue_active gauge\n");
  output.append(metric("gem16_request_queue_active", queue.active));
  output.append("# TYPE gem16_request_queue_depth gauge\n");
  output.append(metric("gem16_request_queue_depth", queue.queued));
  output.append(metric("gem16_request_queue_capacity", queue.capacity));
  output.append(metric("gem16_request_queue_max_depth", queue.max_queued));
  output.append(metric("gem16_request_queue_high_watermark",
                       queue.high_watermark));
  output.append(metric("gem16_request_queue_admissions_total",
                       state.metrics.queue_admissions.load()));
  output.append(metric("gem16_request_queue_waits_total",
                       state.metrics.queue_waits.load()));
  output.append(metric("gem16_request_queue_rejections_total",
                       state.metrics.queue_rejections.load()));
  output.append(metric("gem16_request_queue_wait_microseconds_total",
                       state.metrics.queue_wait_microseconds.load()));
  output.append("# TYPE gem16_resident_sessions gauge\n");
  output.append(metric("gem16_resident_sessions", resident_sessions));
  output.append("# TYPE gem16_pending_session_creations gauge\n");
  output.append(metric("gem16_pending_session_creations", pending_sessions));
  output.append(metric("gem16_session_limit", state.max_sessions));
  output.append(metric("gem16_sessions_rebuilt_total", state.metrics.sessions_rebuilt.load()));
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
  output.append(metric("gem16_prompt_microseconds_total",
                       state.metrics.prompt_microseconds.load()));
  output.append(metric("gem16_ttft_microseconds_total",
                       state.metrics.prompt_microseconds.load()));
  output.append(metric("gem16_decode_microseconds_total",
                       state.metrics.decode_microseconds.load()));
  output.append(VisionMetricsText(state.metrics));
  output.append(metric("gem16_decode_measured_tokens_total",
                       state.metrics.decode_measured_tokens.load()));
  output.append(metric("gem16_mtp_proposed_tokens_total",
                       state.metrics.mtp_proposed_tokens.load()));
  output.append(metric("gem16_mtp_accepted_tokens_total",
                       state.metrics.mtp_accepted_tokens.load()));
  output.append(metric("gem16_mtp_rejected_tokens_total",
                       state.metrics.mtp_rejected_tokens.load()));
  output.append(metric("gem16_mtp_verification_groups_total",
                       state.metrics.mtp_verification_groups.load()));
  output.append(metric("gem16_mtp_d1_groups_total",
                       state.metrics.mtp_d1_groups.load()));
  output.append(metric("gem16_mtp_d2_groups_total",
                       state.metrics.mtp_d2_groups.load()));
  output.append(metric("gem16_mtp_d4_groups_total",
                       state.metrics.mtp_d4_groups.load()));
  output.append(metric("gem16_mtp_ordinary_fallback_tokens_total",
                       state.metrics.mtp_ordinary_fallback_tokens.load()));
  output.append("# TYPE gem16_fallback_total counter\n");
  output.append(metric("gem16_fallback_total",
                       state.metrics.fallback_count.load()));
  output.append("# TYPE gem16_resource_exhaustion_total counter\n");
  output.append(metric("gem16_resource_exhaustion_total",
                       state.metrics.resource_exhaustion_count.load()));
  output.append("# TYPE gem16_unsupported_feature_total counter\n");
  output.append(metric("gem16_unsupported_feature_total",
                       state.metrics.unsupported_feature_count.load()));
  output.append("# TYPE gem16_model_validation_failure_total counter\n");
  output.append(metric("gem16_model_validation_failure_total",
                       state.metrics.model_validation_failure_count.load()));
  output.append("# TYPE gem16_token_loop_allocation_total counter\n");
  output.append(metric("gem16_token_loop_allocation_total",
                       state.metrics.token_loop_allocation_count.load()));
  output.append(metric("gem16_model_weight_bytes",
                       state.runtime->weight_bytes()));
  output.append(metric("gem16_assistant_weight_bytes",
                       state.runtime->assistant_weight_bytes()));
  output.append(metric("gem16_vision_weight_bytes",
                       state.runtime->vision_weight_bytes()));
  output.append(metric("gem16_vision_workspace_bytes",
                       state.runtime->vision_workspace_bytes()));
  output.append(metric("gem16_execution_slot_planned_bytes",
                       state.planned_slot_device_bytes));
  output.append(metric("gem16_configured_execution_slot_bytes",
                       state.configured_slot_device_bytes));
  output.append(metric("gem16_resident_execution_slot_bytes",
                       static_cast<std::uint64_t>(resident_sessions) *
                           state.planned_slot_device_bytes));
  output.append(metric("gem16_device_total_bytes", state.device_total_bytes));
  output.append(metric("gem16_device_free_after_probe_bytes",
                       state.device_free_after_probe_bytes));
  output.append(metric("gem16_device_safety_margin_bytes",
                       state.device_safety_margin_bytes));
  output.append(metric("gem16_last_execution_slot_bytes",
                       state.metrics.last_slot_bytes.load()));
  output.append(metric("gem16_model_load_microseconds",
                       state.model_load_microseconds));
  output.append(metric("gem16_server_startup_microseconds",
                       state.server_startup_microseconds));
  AppendLatencyHistogram(output, "gem16_request_duration_seconds",
                         state.metrics.request_latency);
  AppendLatencyHistogram(output, "gem16_request_queue_wait_seconds",
                         state.metrics.queue_latency);
  AppendLatencyHistogram(output, "gem16_generation_duration_seconds",
                         state.metrics.generation_latency);
  AppendLatencyHistogram(output, "gem16_prompt_duration_seconds",
                         state.metrics.prompt_latency);
  AppendLatencyHistogram(output, "gem16_decode_duration_seconds",
                         state.metrics.decode_latency);
  return output;
}

}  // namespace gem16::server

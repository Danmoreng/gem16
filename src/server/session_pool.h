#pragma once

#include <atomic>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gem16/chat.h"
#include "server/openai_chat.h"
#include "server/request_queue.h"

namespace gem16::server {

struct ResponsesChain {
  std::string latest_response_id;
  std::vector<GenerationMessage> messages;
  std::vector<GenerationToolDefinition> tools;
  GenerationToolChoice tool_choice;
  bool initialized = false;
};

struct SessionEntry {
  SessionEntry(std::string session_id, ChatSession chat_session)
      : id(std::move(session_id)), session(std::move(chat_session)) {}

  std::string id;
  ChatSession session;
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
  std::atomic<std::uint64_t> cached_input_tokens{0U};
  std::atomic<std::uint64_t> cache_write_tokens{0U};
  std::atomic<std::uint64_t> output_tokens{0U};
  std::atomic<std::uint64_t> generation_microseconds{0U};
  std::atomic<std::uint64_t> prompt_microseconds{0U};
  std::atomic<std::uint64_t> decode_microseconds{0U};
  std::atomic<std::uint64_t> image_decode_microseconds{0U};
  std::atomic<std::uint64_t> image_resize_patchify_microseconds{0U};
  std::atomic<std::uint64_t> vision_upload_microseconds{0U};
  std::atomic<std::uint64_t> vision_tower_microseconds{0U};
  std::atomic<std::uint64_t> vision_pool_project_microseconds{0U};
  std::atomic<std::uint64_t> text_prefill_microseconds{0U};
  std::atomic<std::uint64_t> decode_measured_tokens{0U};
  std::atomic<std::uint64_t> mtp_proposed_tokens{0U};
  std::atomic<std::uint64_t> mtp_accepted_tokens{0U};
  std::atomic<std::uint64_t> mtp_rejected_tokens{0U};
  std::atomic<std::uint64_t> mtp_verification_groups{0U};
  std::atomic<std::uint64_t> mtp_d1_groups{0U};
  std::atomic<std::uint64_t> mtp_d2_groups{0U};
  std::atomic<std::uint64_t> mtp_d4_groups{0U};
  std::atomic<std::uint64_t> mtp_ordinary_fallback_tokens{0U};
  std::atomic<std::uint64_t> fallback_count{0U};
  std::atomic<std::uint64_t> vision_requests{0U};
  std::atomic<std::uint64_t> vision_failures{0U};
  std::atomic<std::uint64_t> vision_d2_rejections{0U};
  std::atomic<std::uint64_t> vision_budget_70{0U};
  std::atomic<std::uint64_t> vision_budget_140{0U};
  std::atomic<std::uint64_t> vision_budget_280{0U};
  std::atomic<std::uint64_t> vision_artifact_validation_failures{0U};
  std::atomic<std::uint32_t> last_vision_soft_token_budget{0U};
  std::atomic<std::uint64_t> resource_exhaustion_count{0U};
  std::atomic<std::uint64_t> unsupported_feature_count{0U};
  std::atomic<std::uint64_t> model_validation_failure_count{0U};
  std::atomic<std::uint64_t> token_loop_allocation_count{0U};
  std::atomic<std::uint64_t> last_slot_bytes{0U};
  std::atomic<std::uint64_t> queue_admissions{0U};
  std::atomic<std::uint64_t> queue_waits{0U};
  std::atomic<std::uint64_t> queue_rejections{0U};
  std::atomic<std::uint64_t> queue_wait_microseconds{0U};
  static constexpr std::array<std::uint64_t, 11U> kLatencyBucketsUs = {
      10'000U,     50'000U,     100'000U,    250'000U,
      500'000U,    1'000'000U,  2'500'000U,  5'000'000U,
      10'000'000U, 30'000'000U, 60'000'000U};
  struct LatencyHistogram {
    std::array<std::atomic<std::uint64_t>, kLatencyBucketsUs.size()> buckets{};
    std::atomic<std::uint64_t> count{0U};
    std::atomic<std::uint64_t> sum_microseconds{0U};
  };
  LatencyHistogram request_latency;
  LatencyHistogram queue_latency;
  LatencyHistogram generation_latency;
  LatencyHistogram prompt_latency;
  LatencyHistogram decode_latency;
};

struct ServerState {
  ServerState(std::string served_model_name, std::uint64_t context_limit,
              GemmaChatProcessor chat_processor,
              std::shared_ptr<ModelRuntime> model_runtime,
              ChatSessionOptions chat_session_options,
              std::uint32_t session_limit,
              std::size_t max_queued_requests = 64U);

  std::string model_name;
  std::uint64_t max_context = 0U;
  GemmaChatProcessor processor;
  std::shared_ptr<ModelRuntime> runtime;
  ChatSessionOptions session_options;
  std::uint32_t max_sessions = 2U;
  RequestQueue request_queue;
  std::uint64_t planned_slot_device_bytes = 0U;
  std::uint64_t configured_slot_device_bytes = 0U;
  std::uint64_t device_total_bytes = 0U;
  std::uint64_t device_free_after_probe_bytes = 0U;
  std::uint64_t device_safety_margin_bytes = 0U;
  std::uint64_t model_load_microseconds = 0U;
  std::uint64_t server_startup_microseconds = 0U;
  std::mutex pool_mutex;
  std::condition_variable pool_changed;
  std::unordered_map<std::string, std::shared_ptr<SessionEntry>> sessions;
  std::unordered_map<std::string, std::weak_ptr<SessionEntry>> response_index;
  // Session construction allocates CUDA slot state and must run without
  // pool_mutex. Pending IDs reserve capacity and prevent duplicate creation
  // while that work is in progress.
  std::unordered_set<std::string> pending_sessions;
  std::atomic<std::uint64_t> lru_clock{1U};
  ServerMetrics metrics;
};

class SessionLease {
 public:
  SessionLease(ServerState& state, std::shared_ptr<SessionEntry> entry);
  SessionLease(const SessionLease&) = delete;
  SessionLease& operator=(const SessionLease&) = delete;
  ~SessionLease();

  void Discard();
  void Keep();

 private:
  ServerState* state_ = nullptr;
  std::shared_ptr<SessionEntry> entry_;
  bool discard_ = false;
};

[[nodiscard]] std::int64_t UnixSecondsNow();
[[nodiscard]] Result<OpenAiResponseIdentity> MakeChatIdentity(
    const ServerState& state);
[[nodiscard]] Result<OpenAiResponseIdentity> MakeResponsesIdentity(
    const ServerState& state);
[[nodiscard]] Result<std::shared_ptr<SessionEntry>> CreateSession(
    ServerState& state, std::string id);
[[nodiscard]] Result<std::shared_ptr<SessionEntry>> CreateSessionQueued(
    ServerState& state, std::string id);
[[nodiscard]] Result<std::shared_ptr<SessionEntry>> AcquireNamedSession(
    ServerState& state, const std::string& id);
[[nodiscard]] Result<std::shared_ptr<SessionEntry>> AcquireResponseSession(
    ServerState& state, const std::string& response_id);
void ReleaseSession(ServerState& state,
                    const std::shared_ptr<SessionEntry>& entry);
void DiscardSession(ServerState& state,
                    const std::shared_ptr<SessionEntry>& entry);
[[nodiscard]] Status PrepareResponsesRequest(
    SessionEntry& entry, OpenAiResponsesRequest& request);
void CommitResponsesRequest(SessionEntry& entry,
                            const OpenAiResponsesRequest& request,
                            const ChatGenerationResponse& response,
                            std::string response_id);
void IndexResponse(ServerState& state,
                   const std::shared_ptr<SessionEntry>& entry,
                   const std::string& response_id);
void UnindexResponse(ServerState& state, std::string_view response_id);
void SetActiveResponse(ServerState& state,
                       const std::shared_ptr<SessionEntry>& entry,
                       std::string_view response_id);
void ClearActiveResponse(ServerState& state,
                         const std::shared_ptr<SessionEntry>& entry,
                         std::string_view response_id);
void RecordGeneration(ServerState& state,
                      const ChatGenerationResponse& response,
                      std::chrono::steady_clock::duration elapsed);
void RecordRequestLatency(ServerState& state, std::uint64_t microseconds);
void RecordQueueAdmission(ServerState& state, std::uint64_t microseconds);
[[nodiscard]] std::string VisionMetricsText(const ServerMetrics& metrics);
[[nodiscard]] std::string MetricsText(ServerState& state);

}  // namespace gem16::server

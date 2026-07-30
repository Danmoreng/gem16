#pragma once

#include <atomic>
#include <chrono>
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
  std::atomic<std::uint64_t> decode_measured_tokens{0U};
  std::atomic<std::uint64_t> mtp_proposed_tokens{0U};
  std::atomic<std::uint64_t> mtp_accepted_tokens{0U};
  std::atomic<std::uint64_t> mtp_rejected_tokens{0U};
  std::atomic<std::uint64_t> mtp_verification_groups{0U};
  std::atomic<std::uint64_t> mtp_d1_groups{0U};
  std::atomic<std::uint64_t> mtp_d2_groups{0U};
  std::atomic<std::uint64_t> mtp_d4_groups{0U};
  std::atomic<std::uint64_t> mtp_ordinary_fallback_tokens{0U};
  std::atomic<std::uint64_t> last_slot_bytes{0U};
};

struct ServerState {
  ServerState(std::string served_model_name, std::uint64_t context_limit,
              GemmaChatProcessor chat_processor,
              std::shared_ptr<ModelRuntime> model_runtime,
              ChatSessionOptions chat_session_options,
              std::uint32_t session_limit);

  std::string model_name;
  std::uint64_t max_context = 0U;
  GemmaChatProcessor processor;
  std::shared_ptr<ModelRuntime> runtime;
  ChatSessionOptions session_options;
  std::uint32_t max_sessions = 2U;
  std::uint64_t planned_slot_device_bytes = 0U;
  std::uint64_t configured_slot_device_bytes = 0U;
  std::uint64_t device_total_bytes = 0U;
  std::uint64_t device_safety_margin_bytes = 0U;
  std::mutex pool_mutex;
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
[[nodiscard]] std::string MetricsText(ServerState& state);

}  // namespace gem16::server

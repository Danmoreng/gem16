#pragma once

#include "types.h"

#include "httplib.h"

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>

namespace gem16::studio {

struct ServerMetrics {
  double input_tokens = 0.0;
  double cache_write_tokens = 0.0;
  double prompt_microseconds = 0.0;
  double decode_microseconds = 0.0;
  double decode_measured_tokens = 0.0;
};

[[nodiscard]] std::string BuildChatPayload(
    const ServerConfig& server, const GenerationConfig& generation,
    const std::vector<ChatMessage>& messages, const std::string& tools = {});
[[nodiscard]] std::optional<ServerMetrics> ParseServerMetrics(
    std::string_view body);
[[nodiscard]] std::optional<PerformanceStats> PerformanceDifference(
    const ServerMetrics& before, const ServerMetrics& after);

class ApiClient final {
 public:
  ApiClient() = default;
  ~ApiClient();
  ApiClient(const ApiClient&) = delete;
  ApiClient& operator=(const ApiClient&) = delete;

  void StreamChat(const ServerConfig& server,
                  const GenerationConfig& generation,
                  const std::vector<ChatMessage>& messages,
                  const std::string& session_id, const std::string& tools = {});
  void Cancel();
  [[nodiscard]] bool Busy() const;
  [[nodiscard]] std::vector<ChatEvent> DrainEvents();

 private:
  void Emit(ChatEvent event);

  mutable std::mutex mutex_;
  std::deque<ChatEvent> events_;
  std::shared_ptr<httplib::Client> active_client_;
  std::jthread worker_;
  std::atomic<bool> cancel_requested_{false};
  bool busy_ = false;
};

}  // namespace gem16::studio

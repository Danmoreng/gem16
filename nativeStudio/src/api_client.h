#pragma once

#include "types.h"

#include "httplib.h"

#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace gem16::studio {

class ApiClient final {
 public:
  ApiClient() = default;
  ~ApiClient();
  ApiClient(const ApiClient&) = delete;
  ApiClient& operator=(const ApiClient&) = delete;

  void StreamChat(const ServerConfig& server, const GenerationConfig& generation,
                  const std::vector<ChatMessage>& messages,
                  const std::string& session_id);
  void Cancel();
  [[nodiscard]] bool Busy() const;
  [[nodiscard]] std::vector<ChatEvent> DrainEvents();

 private:
  void Emit(ChatEvent event);

  mutable std::mutex mutex_;
  std::deque<ChatEvent> events_;
  std::shared_ptr<httplib::Client> active_client_;
  std::jthread worker_;
  bool busy_ = false;
};

}  // namespace gem16::studio


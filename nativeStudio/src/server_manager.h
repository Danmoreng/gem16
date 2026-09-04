#pragma once

#include "platform_process.h"
#include "types.h"

#include <chrono>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

namespace gem16::studio {

class ServerManager final {
 public:
  ServerManager();
  ~ServerManager();
  ServerManager(const ServerManager&) = delete;
  ServerManager& operator=(const ServerManager&) = delete;

  void Configure(const ServerConfig& config);
  void Start(const ServerConfig& config);
  void Stop();
  void ClearLogs();
  [[nodiscard]] ServerPhase Phase() const;
  [[nodiscard]] bool OwnsProcess() const { return process_.IsRunning(); }
  [[nodiscard]] HealthSnapshot Health() const;
  [[nodiscard]] std::string Error() const;
  [[nodiscard]] std::vector<std::string> Logs() const;

 private:
  void PollLoop(std::stop_token stop_token);
  [[nodiscard]] HealthSnapshot FetchHealth(const ServerConfig& config) const;
  [[nodiscard]] std::string Validate(const ServerConfig& config) const;
  void AppendLog(std::string line);

  mutable std::mutex mutex_;
  ServerConfig config_;
  ServerPhase phase_ = ServerPhase::kStopped;
  HealthSnapshot health_;
  std::string error_;
  std::deque<std::string> logs_;
  PlatformProcess process_;
  std::jthread poller_;
};

[[nodiscard]] std::vector<std::string> BuildServerCommand(const ServerConfig& config);
[[nodiscard]] std::string HealthCompatibilityError(
    const ServerConfig& config, const HealthSnapshot& health);

}  // namespace gem16::studio

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace gem16::studio {

class PlatformProcess final {
 public:
  using LogCallback = std::function<void(std::string)>;
  using ExitCallback = std::function<void(int)>;

  PlatformProcess();
  ~PlatformProcess();
  PlatformProcess(const PlatformProcess&) = delete;
  PlatformProcess& operator=(const PlatformProcess&) = delete;

  [[nodiscard]] bool Start(const std::vector<std::string>& arguments,
                           const std::string& working_directory,
                           LogCallback on_log, ExitCallback on_exit,
                           std::string& error);
  void Stop();
  [[nodiscard]] bool IsRunning() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gem16::studio


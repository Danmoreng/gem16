#pragma once

#include <initializer_list>
#include <mutex>
#include <ostream>
#include <string>
#include <string_view>

#include "gem16/status.h"

namespace gem16::server {

enum class LogLevel {
  kDebug = 0,
  kInfo = 1,
  kWarning = 2,
  kError = 3,
  kOff = 4
};
enum class LogFormat { kText, kJson };

struct LogField {
  std::string_view name;
  std::string value;
};

[[nodiscard]] Result<LogLevel> ParseLogLevel(std::string_view value);
[[nodiscard]] Result<LogFormat> ParseLogFormat(std::string_view value);
// Empty means the status is not one of the bounded Vision API failures.
[[nodiscard]] std::string_view VisionErrorCode(const Status& status);

class StructuredLogger {
 public:
  StructuredLogger(LogLevel level, LogFormat format, std::ostream& output);

  void Log(LogLevel level, std::string_view event,
           std::initializer_list<LogField> fields = {});
  [[nodiscard]] bool Enabled(LogLevel level) const;

 private:
  LogLevel level_;
  LogFormat format_;
  std::ostream* output_;
  std::mutex mutex_;
};

}  // namespace gem16::server

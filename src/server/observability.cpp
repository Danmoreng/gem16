#include "server/observability.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

#include "util/json.h"

namespace gem16::server {
namespace {

std::string_view LevelName(LogLevel level) {
  switch (level) {
    case LogLevel::kDebug: return "debug";
    case LogLevel::kInfo: return "info";
    case LogLevel::kWarning: return "warning";
    case LogLevel::kError: return "error";
    case LogLevel::kOff: return "off";
  }
  return "unknown";
}

std::string Timestamp() {
  const auto now = std::chrono::system_clock::now();
  const auto seconds = std::chrono::system_clock::to_time_t(now);
  const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                                now.time_since_epoch()) %
                            1000;
  std::tm utc{};
#if defined(_WIN32)
  gmtime_s(&utc, &seconds);
#else
  gmtime_r(&seconds, &utc);
#endif
  std::ostringstream output;
  output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S") << '.'
         << std::setfill('0') << std::setw(3) << milliseconds.count() << 'Z';
  return output.str();
}

}  // namespace

Result<LogLevel> ParseLogLevel(std::string_view value) {
  if (value == "debug") return LogLevel::kDebug;
  if (value == "info") return LogLevel::kInfo;
  if (value == "warning" || value == "warn") return LogLevel::kWarning;
  if (value == "error") return LogLevel::kError;
  if (value == "off") return LogLevel::kOff;
  return Status(StatusCode::kInvalidArgument,
                "--log-level must be debug, info, warning, error, or off");
}

Result<LogFormat> ParseLogFormat(std::string_view value) {
  if (value == "text") return LogFormat::kText;
  if (value == "json") return LogFormat::kJson;
  return Status(StatusCode::kInvalidArgument,
                "--log-format must be text or json");
}

std::string_view VisionErrorCode(const Status& status) {
  const std::string& message = status.message();
  if (message.find("supports exactly one image") != std::string::npos) {
    return "vision_multiple_images_unsupported";
  }
  if (message.find("Vision module is not loaded") != std::string::npos) {
    return "vision_module_not_loaded";
  }
  if (message.find("Vision profile is required") != std::string::npos) {
    return "vision_profile_required";
  }
  if (message.find("Vision with fixed-D2 is not qualified") !=
      std::string::npos) {
    return "vision_mtp_unqualified";
  }
  if (message.find("Vision soft-token budget") != std::string::npos) {
    return "vision_budget_unsupported";
  }
  if (message.find("Vision context is outside") != std::string::npos) {
    return "vision_context_unqualified";
  }
  if (message.find("Vision artifact") != std::string::npos) {
    return "vision_artifact_mismatch";
  }
  return {};
}

StructuredLogger::StructuredLogger(LogLevel level, LogFormat format,
                                   std::ostream& output)
    : level_(level), format_(format), output_(&output) {}

bool StructuredLogger::Enabled(LogLevel level) const {
  return level_ != LogLevel::kOff &&
         static_cast<int>(level) >= static_cast<int>(level_);
}

void StructuredLogger::Log(LogLevel level, std::string_view event,
                           std::initializer_list<LogField> fields) {
  if (!Enabled(level)) return;
  const std::string timestamp = Timestamp();
  std::lock_guard lock(mutex_);
  if (format_ == LogFormat::kJson) {
    *output_ << "{\"timestamp\":" << json::Quote(timestamp)
             << ",\"level\":" << json::Quote(LevelName(level))
             << ",\"event\":" << json::Quote(event);
    for (const LogField& field : fields) {
      *output_ << ',' << json::Quote(field.name) << ':'
               << json::Quote(field.value);
    }
    *output_ << "}\n";
  } else {
    *output_ << timestamp << ' ' << LevelName(level) << ' ' << event;
    for (const LogField& field : fields) {
      *output_ << ' ' << field.name << '=' << json::Quote(field.value);
    }
    *output_ << '\n';
  }
  output_->flush();
}

}  // namespace gem16::server

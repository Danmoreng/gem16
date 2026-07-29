#pragma once

#include <optional>
#include <string>
#include <utility>

namespace gem16 {

enum class StatusCode {
  kOk = 0,
  kInvalidArgument,
  kNotFound,
  kIoError,
  kDataLoss,
  kUnsupported,
  kResourceExhausted,
  kCancelled,
  kInternal,
};

class [[nodiscard]] Status {
 public:
  Status() = default;
  Status(StatusCode code, std::string message) : code_(code), message_(std::move(message)) {}

  [[nodiscard]] static Status Ok() { return {}; }
  [[nodiscard]] bool ok() const { return code_ == StatusCode::kOk; }
  [[nodiscard]] StatusCode code() const { return code_; }
  [[nodiscard]] const std::string& message() const { return message_; }

 private:
  StatusCode code_ = StatusCode::kOk;
  std::string message_;
};

template <typename T>
class [[nodiscard]] Result {
 public:
  Result(T value) : value_(std::move(value)) {}
  Result(Status status) : status_(std::move(status)) {}

  [[nodiscard]] bool ok() const { return value_.has_value(); }
  [[nodiscard]] const Status& status() const { return status_; }
  [[nodiscard]] T& value() & { return *value_; }
  [[nodiscard]] const T& value() const& { return *value_; }
  [[nodiscard]] T&& value() && { return std::move(*value_); }

 private:
  std::optional<T> value_;
  Status status_ = Status(StatusCode::kInternal, "Result has no value");
};

}  // namespace gem16

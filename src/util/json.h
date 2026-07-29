#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "gem16/status.h"

namespace gem16::json {

class Value {
 public:
  using Array = std::vector<Value>;
  using Object = std::map<std::string, Value, std::less<>>;
  using Storage = std::variant<std::nullptr_t, bool, std::int64_t, double, std::string, Array, Object>;

  explicit Value(Storage storage) : storage_(std::move(storage)) {}

  [[nodiscard]] bool is_null() const { return std::holds_alternative<std::nullptr_t>(storage_); }
  [[nodiscard]] bool is_bool() const { return std::holds_alternative<bool>(storage_); }
  [[nodiscard]] bool is_integer() const { return std::holds_alternative<std::int64_t>(storage_); }
  [[nodiscard]] bool is_number() const { return is_integer() || std::holds_alternative<double>(storage_); }
  [[nodiscard]] bool is_string() const { return std::holds_alternative<std::string>(storage_); }
  [[nodiscard]] bool is_array() const { return std::holds_alternative<Array>(storage_); }
  [[nodiscard]] bool is_object() const { return std::holds_alternative<Object>(storage_); }

  [[nodiscard]] bool as_bool() const { return std::get<bool>(storage_); }
  [[nodiscard]] std::int64_t as_integer() const { return std::get<std::int64_t>(storage_); }
  [[nodiscard]] double as_number() const;
  [[nodiscard]] const std::string& as_string() const { return std::get<std::string>(storage_); }
  [[nodiscard]] const Array& as_array() const { return std::get<Array>(storage_); }
  [[nodiscard]] const Object& as_object() const { return std::get<Object>(storage_); }
  [[nodiscard]] const Value* find(std::string_view key) const;

 private:
  Storage storage_;
};

struct ParseLimits {
  std::size_t max_depth = 128;
  std::size_t max_values = 2'000'000;
  std::size_t max_string_bytes = 256U * 1024U * 1024U;
};

[[nodiscard]] Result<Value> Parse(std::string_view input, ParseLimits limits = {});
[[nodiscard]] std::string Escape(std::string_view input);
[[nodiscard]] std::string Quote(std::string_view input);
[[nodiscard]] std::string Stringify(const Value& value);

}  // namespace gem16::json


#pragma once

#if defined(_WIN32)

#include <limits>
#include <optional>
#include <string>
#include <string_view>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace gem16::cli {

inline std::optional<std::string> WideToUtf8(std::wstring_view text) {
  if (text.empty()) return std::string{};
  if (text.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return std::nullopt;
  }
  const int source_length = static_cast<int>(text.size());
  const int required = WideCharToMultiByte(
      CP_UTF8, WC_ERR_INVALID_CHARS, text.data(), source_length, nullptr, 0,
      nullptr, nullptr);
  if (required <= 0) return std::nullopt;
  std::string result(static_cast<std::size_t>(required), '\0');
  const int written = WideCharToMultiByte(
      CP_UTF8, WC_ERR_INVALID_CHARS, text.data(), source_length, result.data(),
      required, nullptr, nullptr);
  if (written != required) return std::nullopt;
  return result;
}

}  // namespace gem16::cli

#endif

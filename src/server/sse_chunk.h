#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace gem16::server {

// Fixed-capacity SSE/HTTP-chunk encoder. Construction reserves the only host
// allocation; Reset/Append/Finish perform bounded writes into that storage.
class SseChunkBuilder {
 public:
  explicit SseChunkBuilder(std::size_t payload_capacity)
      : storage_(kPrefixBytes + payload_capacity + kSuffixBytes) {}

  void Reset() {
    size_ = kPrefixBytes;
    failed_ = false;
    (void)Append("data: ");
  }

  [[nodiscard]] bool Append(std::string_view value) {
    if (failed_ || value.size() > storage_.size() - size_) {
      failed_ = true;
      return false;
    }
    for (const char byte : value) storage_[size_++] = byte;
    return true;
  }

  [[nodiscard]] bool Append(char value) {
    if (failed_ || size_ == storage_.size()) {
      failed_ = true;
      return false;
    }
    storage_[size_++] = value;
    return true;
  }

  [[nodiscard]] bool AppendUnsigned(std::uint64_t value) {
    char digits[32]{};
    const auto converted =
        std::to_chars(digits, digits + sizeof(digits), value);
    return converted.ec == std::errc{} &&
           Append(std::string_view(
               digits, static_cast<std::size_t>(converted.ptr - digits)));
  }

  [[nodiscard]] bool AppendSigned(std::int64_t value) {
    char digits[32]{};
    const auto converted =
        std::to_chars(digits, digits + sizeof(digits), value);
    return converted.ec == std::errc{} &&
           Append(std::string_view(
               digits, static_cast<std::size_t>(converted.ptr - digits)));
  }

  [[nodiscard]] bool AppendJsonEscaped(std::string_view value) {
    static constexpr char kHex[] = "0123456789abcdef";
    for (const unsigned char byte : value) {
      switch (byte) {
        case '"':
          if (!Append("\\\"")) return false;
          break;
        case '\\':
          if (!Append("\\\\")) return false;
          break;
        case '\b':
          if (!Append("\\b")) return false;
          break;
        case '\f':
          if (!Append("\\f")) return false;
          break;
        case '\n':
          if (!Append("\\n")) return false;
          break;
        case '\r':
          if (!Append("\\r")) return false;
          break;
        case '\t':
          if (!Append("\\t")) return false;
          break;
        default:
          if (byte < 0x20U) {
            char escaped[] = {'\\', 'u', '0', '0', kHex[byte >> 4U],
                              kHex[byte & 0x0FU]};
            if (!Append(std::string_view(escaped, sizeof(escaped)))) {
              return false;
            }
          } else if (!Append(static_cast<char>(byte))) {
            return false;
          }
      }
    }
    return true;
  }

  [[nodiscard]] bool AppendJsonString(std::string_view value) {
    return Append('"') && AppendJsonEscaped(value) && Append('"');
  }

  [[nodiscard]] bool AppendJsonString(std::string_view prefix,
                                      std::string_view value) {
    return Append('"') && AppendJsonEscaped(prefix) &&
           AppendJsonEscaped(value) && Append('"');
  }

  // Returns a complete HTTP chunk containing one SSE record. Empty means the
  // fixed capacity was exceeded.
  [[nodiscard]] std::span<const char> Finish() {
    if (!Append("\n\n")) return {};
    const std::size_t payload_size = size_ - kPrefixBytes;
    if (!Append("\r\n")) return {};

    char hexadecimal[2U * sizeof(std::size_t)]{};
    const auto converted = std::to_chars(
        hexadecimal, hexadecimal + sizeof(hexadecimal), payload_size, 16);
    if (converted.ec != std::errc{}) return {};
    const std::size_t digits =
        static_cast<std::size_t>(converted.ptr - hexadecimal);
    const std::size_t header_size = digits + 2U;
    const std::size_t begin = kPrefixBytes - header_size;
    for (std::size_t index = 0U; index < digits; ++index) {
      storage_[begin + index] = hexadecimal[index];
    }
    storage_[begin + digits] = '\r';
    storage_[begin + digits + 1U] = '\n';
    return std::span<const char>(storage_.data() + begin, size_ - begin);
  }

  [[nodiscard]] bool failed() const { return failed_; }
  [[nodiscard]] std::span<const char> Payload() const {
    return failed_ ? std::span<const char>{}
                   : std::span<const char>(storage_.data() + kPrefixBytes,
                                           size_ - kPrefixBytes);
  }
  [[nodiscard]] std::size_t capacity() const {
    return storage_.size() - kPrefixBytes - kSuffixBytes;
  }

 private:
  static constexpr std::size_t kPrefixBytes =
      2U * sizeof(std::size_t) + 2U;
  static constexpr std::size_t kSuffixBytes = 4U;

  std::vector<char> storage_;
  std::size_t size_ = kPrefixBytes;
  bool failed_ = false;
};

inline constexpr std::string_view kFinalHttpChunk = "0\r\n\r\n";

}  // namespace gem16::server

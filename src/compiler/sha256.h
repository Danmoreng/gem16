#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace gem16::compiler {

class Sha256 {
 public:
  Sha256() noexcept;

  void Update(const void* data, std::size_t size) noexcept;
  void Update(std::string_view data) noexcept { Update(data.data(), data.size()); }
  [[nodiscard]] std::array<std::uint8_t, 32> Final() const noexcept;
  [[nodiscard]] std::string HexDigest() const;

 private:
  void Transform(const std::uint8_t* block) noexcept;

  std::array<std::uint32_t, 8> state_{};
  std::array<std::uint8_t, 64> block_{};
  std::size_t block_size_ = 0;
  std::uint64_t total_bytes_ = 0;
};

[[nodiscard]] std::string Sha256Hex(const void* data, std::size_t size);

}  // namespace gem16::compiler

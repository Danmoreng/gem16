#include "compiler/sha256.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <iomanip>
#include <sstream>

namespace gem16::compiler {
namespace {

constexpr std::array<std::uint32_t, 64> kRoundConstants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

constexpr std::uint32_t RotateRight(std::uint32_t value, unsigned count) noexcept {
  return (value >> count) | (value << (32U - count));
}
constexpr std::uint32_t Ch(std::uint32_t x, std::uint32_t y, std::uint32_t z) noexcept {
  return (x & y) ^ (~x & z);
}
constexpr std::uint32_t Maj(std::uint32_t x, std::uint32_t y, std::uint32_t z) noexcept {
  return (x & y) ^ (x & z) ^ (y & z);
}
constexpr std::uint32_t Sigma0(std::uint32_t value) noexcept {
  return RotateRight(value, 2U) ^ RotateRight(value, 13U) ^ RotateRight(value, 22U);
}
constexpr std::uint32_t Sigma1(std::uint32_t value) noexcept {
  return RotateRight(value, 6U) ^ RotateRight(value, 11U) ^ RotateRight(value, 25U);
}
constexpr std::uint32_t Gamma0(std::uint32_t value) noexcept {
  return RotateRight(value, 7U) ^ RotateRight(value, 18U) ^ (value >> 3U);
}
constexpr std::uint32_t Gamma1(std::uint32_t value) noexcept {
  return RotateRight(value, 17U) ^ RotateRight(value, 19U) ^ (value >> 10U);
}

std::uint32_t LoadBigEndian(const std::uint8_t* data) noexcept {
  return (static_cast<std::uint32_t>(data[0]) << 24U) |
         (static_cast<std::uint32_t>(data[1]) << 16U) |
         (static_cast<std::uint32_t>(data[2]) << 8U) |
         static_cast<std::uint32_t>(data[3]);
}

void StoreBigEndian(std::uint8_t* data, std::uint32_t value) noexcept {
  data[0] = static_cast<std::uint8_t>(value >> 24U);
  data[1] = static_cast<std::uint8_t>(value >> 16U);
  data[2] = static_cast<std::uint8_t>(value >> 8U);
  data[3] = static_cast<std::uint8_t>(value);
}

}  // namespace

Sha256::Sha256() noexcept
    : state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
             0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U} {}

void Sha256::Transform(const std::uint8_t* block) noexcept {
  std::array<std::uint32_t, 64> schedule{};
  for (std::size_t index = 0; index < 16; ++index) schedule[index] = LoadBigEndian(block + index * 4U);
  for (std::size_t index = 16; index < schedule.size(); ++index) {
    schedule[index] = Gamma1(schedule[index - 2U]) + schedule[index - 7U] +
                      Gamma0(schedule[index - 15U]) + schedule[index - 16U];
  }
  std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
  std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
  for (std::size_t index = 0; index < schedule.size(); ++index) {
    const std::uint32_t t1 = h + Sigma1(e) + Ch(e, f, g) + kRoundConstants[index] + schedule[index];
    const std::uint32_t t2 = Sigma0(a) + Maj(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  state_[0] += a;
  state_[1] += b;
  state_[2] += c;
  state_[3] += d;
  state_[4] += e;
  state_[5] += f;
  state_[6] += g;
  state_[7] += h;
}

void Sha256::Update(const void* input, std::size_t size) noexcept {
  const auto* data = static_cast<const std::uint8_t*>(input);
  total_bytes_ += static_cast<std::uint64_t>(size);
  while (size != 0U) {
    const std::size_t available = block_.size() - block_size_;
    const std::size_t count = size < available ? size : available;
    std::copy_n(data, count, block_.data() + block_size_);
    block_size_ += count;
    data += count;
    size -= count;
    if (block_size_ == block_.size()) {
      Transform(block_.data());
      block_size_ = 0;
    }
  }
}

std::array<std::uint8_t, 32> Sha256::Final() const noexcept {
  Sha256 copy = *this;
  const std::uint64_t bit_length = copy.total_bytes_ * 8U;
  copy.block_[copy.block_size_++] = 0x80U;
  if (copy.block_size_ > 56U) {
    std::fill(copy.block_.begin() + static_cast<std::ptrdiff_t>(copy.block_size_),
              copy.block_.end(), std::uint8_t{0});
    copy.Transform(copy.block_.data());
    copy.block_size_ = 0;
  }
  std::fill(copy.block_.begin() + static_cast<std::ptrdiff_t>(copy.block_size_),
            copy.block_.begin() + 56, std::uint8_t{0});
  for (unsigned index = 0; index < 8U; ++index) {
    copy.block_[56U + index] = static_cast<std::uint8_t>(bit_length >> (56U - index * 8U));
  }
  copy.Transform(copy.block_.data());
  std::array<std::uint8_t, 32> digest{};
  for (std::size_t index = 0; index < copy.state_.size(); ++index) {
    StoreBigEndian(digest.data() + index * 4U, copy.state_[index]);
  }
  return digest;
}

std::string Sha256::HexDigest() const {
  const auto digest = Final();
  std::ostringstream result;
  result << std::hex << std::setfill('0');
  for (const auto byte : digest) result << std::setw(2) << static_cast<unsigned>(byte);
  return result.str();
}

std::string Sha256Hex(const void* data, std::size_t size) {
  Sha256 hash;
  hash.Update(data, size);
  return hash.HexDigest();
}

}  // namespace gem16::compiler

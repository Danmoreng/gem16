#include "server/secure_id.h"

#include <array>
#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <bcrypt.h>
#else
#include <cerrno>
#include <sys/random.h>
#endif

namespace gem16::server {
namespace {

constexpr std::size_t kRandomBytes = 16U;

Status FillRandom(std::array<std::uint8_t, kRandomBytes>& bytes) {
#if defined(_WIN32)
  const NTSTATUS status = BCryptGenRandom(
      nullptr, bytes.data(), static_cast<ULONG>(bytes.size()),
      BCRYPT_USE_SYSTEM_PREFERRED_RNG);
  if (status < 0) {
    return Status(StatusCode::kInternal,
                  "Windows system random generator failed");
  }
  return Status::Ok();
#else
  std::size_t offset = 0U;
  while (offset < bytes.size()) {
    const ssize_t received =
        getrandom(bytes.data() + offset, bytes.size() - offset, 0U);
    if (received < 0) {
      if (errno == EINTR) continue;
      return Status(StatusCode::kInternal,
                    "Linux system random generator failed");
    }
    if (received == 0) {
      return Status(StatusCode::kInternal,
                    "Linux system random generator returned no data");
    }
    offset += static_cast<std::size_t>(received);
  }
  return Status::Ok();
#endif
}

}  // namespace

Result<std::string> MakeSecureId(std::string_view prefix) {
  std::array<std::uint8_t, kRandomBytes> bytes{};
  const Status status = FillRandom(bytes);
  if (!status.ok()) return status;

  constexpr char kHex[] = "0123456789abcdef";
  std::string id;
  id.reserve(prefix.size() + 2U * bytes.size());
  id.append(prefix);
  for (const std::uint8_t byte : bytes) {
    id.push_back(kHex[byte >> 4U]);
    id.push_back(kHex[byte & 0x0FU]);
  }
  return id;
}

}  // namespace gem16::server

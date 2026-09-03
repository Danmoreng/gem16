#pragma once

#include <cstdlib>

namespace gem16::internal {

// Environment switches are process-local diagnostics and rollout controls.
// MSVC deprecates getenv in favor of its allocating _dupenv_s extension, but
// a read-only lookup has the portable lifetime and concurrency contract that
// these callers need. Keep the warning suppression scoped to this wrapper.
inline const char* GetEnvironmentVariable(const char* name) noexcept {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
  const char* value = std::getenv(name);
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
  return value;
}

}  // namespace gem16::internal

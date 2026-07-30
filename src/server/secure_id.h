#pragma once

#include <string>
#include <string_view>

#include "gem16/status.h"

namespace gem16::server {

// Returns prefix followed by 128 bits of OS-generated randomness encoded as
// lowercase hexadecimal. IDs are opaque protocol handles, not counters.
[[nodiscard]] Result<std::string> MakeSecureId(std::string_view prefix);

}  // namespace gem16::server

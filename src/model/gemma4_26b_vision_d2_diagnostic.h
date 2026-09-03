#pragma once

#include <string_view>

#include "util/environment.h"

namespace gem16::internal {

// V11 is an exactness laboratory, not a product capability. Keep the switch
// process-local so model metadata and inference requests cannot enable an
// unqualified Vision + fixed-D2 path.
inline bool Gemma4Moe26BVisionD2DiagnosticEnabled() noexcept {
  const char* value = GetEnvironmentVariable("GEM16_VISION_D2_DIAGNOSTIC");
  return value != nullptr && std::string_view(value) == "1";
}

inline constexpr std::string_view kGemma4Moe26BVisionMtpUnqualified =
    "vision_mtp_unqualified";

}  // namespace gem16::internal

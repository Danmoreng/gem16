#pragma once

#include <cstdint>
#include <string_view>

#include "gem16/engine.h"

namespace gem16::internal {

struct TwelveBD3MathSelection {
  bool optimized = false;
  const char* policy = "auto";
  const char* reason = "native_reference_geometry";

  [[nodiscard]] const char* mode() const {
    return optimized ? "optimized" : "reference";
  }
};

// Startup-only shape policy, shared by the12B engine and server diagnostics.
// Auto prefers the measured optimized geometry; all other native geometries
// retain their existing implementation and disclose the selection reason.
// Explicit optimized/fp8 requests never silently select reference arithmetic.
[[nodiscard]] inline Result<TwelveBD3MathSelection> SelectTwelveBD3Math(
    const char* setting, std::uint32_t drafts, KvCacheMode kv,
    std::uint64_t context, bool sampling, bool adaptive = false) {
  const std::string_view requested = setting == nullptr ? "auto" : setting;
  if (requested != "auto" && requested != "optimized" && requested != "fp8" &&
      requested != "exact") {
    return Status(StatusCode::kInvalidArgument,
        "GEM16_12B_D3_ATTENTION requires auto, optimized (fp8), or exact");
  }
  const bool supported = drafts == 3U && kv == KvCacheMode::kCheckpointFp8 &&
                         context == 17519U && !sampling && !adaptive;
  if (requested == "exact") {
    return TwelveBD3MathSelection{false, "exact", "explicit_reference"};
  }
  if (requested == "optimized" || requested == "fp8") {
    if (!supported) {
      return Status(StatusCode::kUnsupported,
          "optimized FP8 attention requires fixed12B D3, FP8 KV, context17519 and greedy generation");
    }
    return TwelveBD3MathSelection{true, "optimized", "explicit_optimized"};
  }
  return TwelveBD3MathSelection{supported, "auto",
      supported ? "default_optimized_geometry" : "native_reference_geometry"};
}

}  // namespace gem16::internal

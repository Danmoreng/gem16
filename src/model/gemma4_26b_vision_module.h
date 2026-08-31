#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "gem16/status.h"

namespace gem16::internal {

inline constexpr std::string_view kGemma4Moe26BVisionProfile =
    "gemma4_26b_trellis35_vision_fp8";
inline constexpr std::string_view kGemma4Moe26BVisionRequiredTextProfile =
    "gem16-trellis35-w4a8-v1";
inline constexpr std::string_view kGemma4Moe26BVisionSourceLock =
    "3d9441fdebef33785e33181c335338208b8bf868cb8c7da692fd9c765cca8230";
inline constexpr std::uint64_t kGemma4Moe26BVisionAlignment = 256U;
inline constexpr std::uint64_t kGemma4Moe26BVisionTensorBytes =
    597'301'792ULL;
inline constexpr std::uint64_t kGemma4Moe26BVisionPaddingBytes = 11'232ULL;
inline constexpr std::uint64_t kGemma4Moe26BVisionPayloadBytes =
    597'313'024ULL;
inline constexpr std::uint64_t kGemma4Moe26BVisionTensorCount = 547U;

struct Gemma4Moe26BVisionTensorPlan {
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
  std::string dtype;
  std::vector<std::uint64_t> shape;
};

struct Gemma4Moe26BVisionModulePlan {
  std::filesystem::path root;
  std::filesystem::path artifact;
  std::string artifact_sha256;
  std::uint64_t artifact_bytes = 0;
  std::uint64_t payload_file_offset = 0;
  std::map<std::string, Gemma4Moe26BVisionTensorPlan, std::less<>> tensors;
};

// Initialization-only validation. It binds no capability and performs no GPU
// allocation; callers must still explicitly select the Trellis35 Vision
// profile. Every file, hash, tensor name, shape, range and zero padding gap is
// checked before the plan can be consumed by the CUDA residency loader.
[[nodiscard]] Result<Gemma4Moe26BVisionModulePlan>
LoadGemma4Moe26BVisionModulePlan(const std::filesystem::path& module_root);

}  // namespace gem16::internal

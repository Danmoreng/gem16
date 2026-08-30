#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>

#include "gem16/status.h"

namespace gem16::internal {

inline constexpr std::string_view kGemma4Moe26BTrellis35Profile =
    "gem16-trellis35-w4a8-v1";
inline constexpr std::string_view kGemma4Moe26BTrellis35SourceLock =
    "3d9441fdebef33785e33181c335338208b8bf868cb8c7da692fd9c765cca8230";
inline constexpr std::uint32_t kTrellis35LayerCount = 30U;
inline constexpr std::uint32_t kTrellis35ExpertCount = 128U;
inline constexpr std::uint32_t kTrellis35CodebookId = 2U;
inline constexpr std::uint64_t kTrellis35Alignment = 256U;
inline constexpr std::uint64_t kTrellis35LayerArtifactBytes =
    345'147'392ULL;
inline constexpr std::uint64_t kTrellis35NonRoutedBytes =
    1'850'270'720ULL;
inline constexpr std::uint64_t kTrellis35RoutedExpertBytes =
    10'354'421'760ULL;
inline constexpr std::uint64_t kTrellis35CheckpointBytes =
    12'204'692'480ULL;

struct Trellis35Region {
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
};

struct Trellis35ExpertDescriptor {
  std::uint32_t pool_offset = 0;
  std::uint16_t rate_bits = 0;
  std::uint16_t codebook_id = 0;
};
static_assert(sizeof(Trellis35ExpertDescriptor) == 8U);

struct Trellis35FamilyPlan {
  std::array<std::uint16_t, kTrellis35ExpertCount> rate_map{};
  std::array<Trellis35ExpertDescriptor, kTrellis35ExpertCount> descriptors{};
  Trellis35Region k3_payload_pool;
  Trellis35Region k4_payload_pool;
  Trellis35Region descriptor;
  Trellis35Region suh;
  Trellis35Region svh;
};

struct Trellis35FileIdentity {
  std::filesystem::path path;
  std::uint64_t bytes = 0;
  std::string sha256;
};

struct Trellis35LayerPlan {
  std::uint32_t layer = 0;
  std::uint64_t arena_offset = 0;
  Trellis35FileIdentity artifact;
  Trellis35FamilyPlan gate_up;
  Trellis35FamilyPlan down;
};

struct Trellis35NonRoutedTensor {
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
};

struct Gemma4Moe26BTrellis35CheckpointPlan {
  std::filesystem::path checkpoint_root;
  std::string checkpoint_content_sha256;
  Trellis35FileIdentity non_routed;
  std::map<std::string, Trellis35NonRoutedTensor, std::less<>>
      non_routed_tensors;
  std::array<Trellis35LayerPlan, kTrellis35LayerCount> layers{};
  std::uint64_t arena_bytes = 0;
  std::uint64_t nvfp4_routed_expert_bytes = 0;
};

// This is initialization-only host work. It validates all metadata, safe paths,
// artifact extents, K3/K4 maps and the on-disk little-endian descriptors. Large
// payload SHA-256 validation is performed while the CUDA loader uploads files.
[[nodiscard]] Result<Gemma4Moe26BTrellis35CheckpointPlan>
LoadGemma4Moe26BTrellis35CheckpointPlan(
    const std::filesystem::path& checkpoint_root);

// Public for bounded host corruption tests and future loader diagnostics. The
// plan must already contain the exact v1 regions and rate maps.
[[nodiscard]] Status ValidateGemma4Moe26BTrellis35LayerPayload(
    Trellis35LayerPlan* layer);

// WP4 has an isolated ordinary-decode routed-expert kernel, while the complete
// text-only engine remains unsupported. No qualified precision path may be
// selected as a fallback.
[[nodiscard]] Status Gemma4Moe26BTrellis35EngineDispatchStatus();

}  // namespace gem16::internal

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>

#include "gem16/status.h"
#include "model/gemma4_26b_trellis35.h"

namespace gem16::internal {

struct Trellis35DeviceFamilyBinding {
  const std::byte* k3_payload_pool = nullptr;
  const std::byte* k4_payload_pool = nullptr;
  const Trellis35ExpertDescriptor* descriptors = nullptr;
  const std::uint16_t* suh_f16 = nullptr;
  const std::uint16_t* svh_f16 = nullptr;
  std::array<std::uint16_t, kTrellis35ExpertCount> rate_map{};
};

struct Trellis35DeviceLayerBinding {
  Trellis35DeviceFamilyBinding gate_up;
  Trellis35DeviceFamilyBinding down;
};

struct Gemma4Moe26BTrellis35DeviceArtifactStats {
  std::uint64_t arena_bytes = 0;
  std::uint64_t uploaded_bytes = 0;
  std::uint64_t files = 0;
  std::uint64_t non_routed_tensors = 0;
  std::uint64_t device_allocations = 0;
  std::uint64_t host_staging_peak_bytes = 0;
  std::uint32_t storage_format_version = 0U;
  double upload_milliseconds = 0.0;
  double load_milliseconds = 0.0;
  bool runtime_payload_sha256 = false;
  std::string checkpoint_content_sha256;
  std::string load_path;
};

// One allocation containing the compact non-routed image followed by all 30
// Trellis35 layer images. All execution-facing addresses are resolved once in
// Load(); the object performs no lazy binding, repacking or filesystem access.
class Gemma4Moe26BTrellis35DeviceArtifact {
 public:
  Gemma4Moe26BTrellis35DeviceArtifact() = default;
  ~Gemma4Moe26BTrellis35DeviceArtifact();
  Gemma4Moe26BTrellis35DeviceArtifact(
      const Gemma4Moe26BTrellis35DeviceArtifact&) = delete;
  Gemma4Moe26BTrellis35DeviceArtifact& operator=(
      const Gemma4Moe26BTrellis35DeviceArtifact&) = delete;
  Gemma4Moe26BTrellis35DeviceArtifact(
      Gemma4Moe26BTrellis35DeviceArtifact&& other) noexcept;
  Gemma4Moe26BTrellis35DeviceArtifact& operator=(
      Gemma4Moe26BTrellis35DeviceArtifact&& other) noexcept;

  [[nodiscard]] static Result<Gemma4Moe26BTrellis35DeviceArtifact> Load(
      const std::filesystem::path& checkpoint_root);

  [[nodiscard]] Result<const std::byte*> NonRoutedPointer(
      std::string_view name) const;
  // Initialization-only scalar lookup. Values are copied from the immutable
  // device arena after the complete artifact upload; recurring execution does
  // not consult this object or perform a device transfer.
  [[nodiscard]] Result<float> HostFloat32(std::string_view name) const;
  [[nodiscard]] const std::array<Trellis35DeviceLayerBinding,
                                 kTrellis35LayerCount>&
  layers() const {
    return layers_;
  }
  [[nodiscard]] const Gemma4Moe26BTrellis35DeviceArtifactStats& stats() const {
    return stats_;
  }
  [[nodiscard]] const std::byte* arena() const { return arena_; }
  [[nodiscard]] std::uint64_t arena_bytes() const { return arena_bytes_; }

 private:
  std::byte* arena_ = nullptr;
  std::uint64_t arena_bytes_ = 0;
  std::map<std::string, const std::byte*, std::less<>> non_routed_;
  std::map<std::string, float, std::less<>> host_f32_;
  std::array<Trellis35DeviceLayerBinding, kTrellis35LayerCount> layers_{};
  Gemma4Moe26BTrellis35DeviceArtifactStats stats_;
};

}  // namespace gem16::internal

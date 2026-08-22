#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>

#include "gem16/status.h"
#include "gem16/types.h"
#include "model/gemma4_26b_residency.h"

namespace gem16::internal {

struct Gemma4Moe26BDeviceArtifactStats {
  std::uint64_t tensors = 0;
  std::uint64_t payload_bytes = 0;
  std::uint64_t shards = 0;
  std::uint64_t direct_tensors = 0;
  std::uint64_t tiled_weight_tensors = 0;
  std::uint64_t tiled_scale_tensors = 0;
  std::uint64_t host_staging_peak_bytes = 0;
};

// One immutable, final-layout device arena for the exact validated M08 artifact.
// Loading is an initialization operation. Pointer lookup is host metadata only;
// execution code receives resolved pointers and never consults this object.
class Gemma4Moe26BDeviceArtifact {
 public:
  Gemma4Moe26BDeviceArtifact() = default;
  ~Gemma4Moe26BDeviceArtifact();
  Gemma4Moe26BDeviceArtifact(const Gemma4Moe26BDeviceArtifact&) = delete;
  Gemma4Moe26BDeviceArtifact& operator=(const Gemma4Moe26BDeviceArtifact&) = delete;
  Gemma4Moe26BDeviceArtifact(Gemma4Moe26BDeviceArtifact&& other) noexcept;
  Gemma4Moe26BDeviceArtifact& operator=(Gemma4Moe26BDeviceArtifact&& other) noexcept;

  [[nodiscard]] static Result<Gemma4Moe26BDeviceArtifact> Load(
      const std::filesystem::path& model_directory,
      const ModelManifest& manifest,
      const Gemma4Moe26BResidencyPlan& plan);

  [[nodiscard]] Result<const std::byte*> Pointer(std::string_view name) const;
  [[nodiscard]] Result<float> HostFloat32(std::string_view name) const;
  [[nodiscard]] std::byte* arena() const { return arena_; }
  [[nodiscard]] std::uint64_t arena_bytes() const { return arena_bytes_; }
  [[nodiscard]] const Gemma4Moe26BDeviceArtifactStats& stats() const {
    return stats_;
  }

 private:
  std::byte* arena_ = nullptr;
  std::uint64_t arena_bytes_ = 0;
  std::map<std::string, std::uint64_t, std::less<>> offsets_;
  std::map<std::string, float, std::less<>> host_f32_;
  Gemma4Moe26BDeviceArtifactStats stats_;
};

}  // namespace gem16::internal

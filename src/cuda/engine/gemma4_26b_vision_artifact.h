#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>

#include "gem16/status.h"
#include "model/gemma4_26b_vision_module.h"

namespace gem16::internal {

struct Gemma4Moe26BVisionDeviceArtifactStats {
  std::uint64_t arena_bytes = 0;
  std::uint64_t uploaded_bytes = 0;
  std::uint64_t tensor_count = 0;
  std::uint64_t device_allocations = 0;
  std::uint64_t host_staging_peak_bytes = 0;
  std::string artifact_sha256;
  std::string load_path;
};

// One persistent allocation containing the already-final FP8/BF16 payload.
// Header bytes are validated but never uploaded. Tensor bindings are resolved
// once, and recurring image/text execution performs no filesystem work,
// allocation or weight repack.
class Gemma4Moe26BVisionDeviceArtifact {
 public:
  Gemma4Moe26BVisionDeviceArtifact() = default;
  ~Gemma4Moe26BVisionDeviceArtifact();
  Gemma4Moe26BVisionDeviceArtifact(
      const Gemma4Moe26BVisionDeviceArtifact&) = delete;
  Gemma4Moe26BVisionDeviceArtifact& operator=(
      const Gemma4Moe26BVisionDeviceArtifact&) = delete;
  Gemma4Moe26BVisionDeviceArtifact(
      Gemma4Moe26BVisionDeviceArtifact&& other) noexcept;
  Gemma4Moe26BVisionDeviceArtifact& operator=(
      Gemma4Moe26BVisionDeviceArtifact&& other) noexcept;

  [[nodiscard]] static Result<Gemma4Moe26BVisionDeviceArtifact> Load(
      const std::filesystem::path& module_root);
  [[nodiscard]] Result<const std::byte*> Pointer(
      std::string_view name) const;
  [[nodiscard]] const Gemma4Moe26BVisionDeviceArtifactStats& stats() const {
    return stats_;
  }
  [[nodiscard]] const std::byte* arena() const { return arena_; }
  [[nodiscard]] std::uint64_t arena_bytes() const { return arena_bytes_; }

 private:
  std::byte* arena_ = nullptr;
  std::uint64_t arena_bytes_ = 0;
  std::map<std::string, const std::byte*, std::less<>> tensors_;
  Gemma4Moe26BVisionDeviceArtifactStats stats_;
};

}  // namespace gem16::internal

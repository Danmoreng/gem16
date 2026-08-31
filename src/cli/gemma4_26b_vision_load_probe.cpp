#include <cuda_runtime_api.h>

#include <filesystem>
#include <iostream>

#include "cuda/engine/gemma4_26b_trellis35_artifact.h"
#include "cuda/engine/gemma4_26b_vision_artifact.h"

namespace {

int LoadVision(const std::filesystem::path& sidecar,
               std::uint64_t text_arena_bytes) {
  std::size_t free_before = 0U;
  std::size_t total = 0U;
  if (cudaMemGetInfo(&free_before, &total) != cudaSuccess) return 3;
  auto artifact = gem16::internal::Gemma4Moe26BVisionDeviceArtifact::Load(
      sidecar);
  if (!artifact.ok()) {
    std::cerr << "vision_load_probe_error: " << artifact.status().message()
              << '\n';
    return 4;
  }
  std::size_t free_after = 0U;
  if (cudaMemGetInfo(&free_after, &total) != cudaSuccess) return 5;
  const auto& stats = artifact.value().stats();
  std::cout << "vision_load_probe_ok arena_bytes=" << stats.arena_bytes
            << " uploaded_bytes=" << stats.uploaded_bytes
            << " tensors=" << stats.tensor_count
            << " allocations=" << stats.device_allocations
            << " text_arena_bytes=" << text_arena_bytes
            << " cuda_delta=" << (free_before - free_after)
            << " free_after=" << free_after
            << " sha256=" << stats.artifact_sha256 << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 && argc != 3) {
    std::cerr << "usage: gem16-26b-vision-load-probe SIDECAR [TRELLIS_CHECKPOINT]\n";
    return 2;
  }
  if (argc == 2) return LoadVision(std::filesystem::path(argv[1]), 0U);
  auto text = gem16::internal::Gemma4Moe26BTrellis35DeviceArtifact::Load(
      std::filesystem::path(argv[2]));
  if (!text.ok()) {
    std::cerr << "vision_load_probe_text_error: " << text.status().message()
              << '\n';
    return 6;
  }
  return LoadVision(std::filesystem::path(argv[1]),
                    text.value().arena_bytes());
}

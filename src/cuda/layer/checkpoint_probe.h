#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "gem16/status.h"

namespace gem16::internal {

struct AttentionProbeSample {
  std::uint64_t element = 0;
  float cuda_reference = 0.0F;
  float sm120_direct = 0.0F;
};

struct LayerCheckpointProbeResult {
  std::uint64_t layer = 0;
  std::uint64_t context_tokens = 0;
  std::uint64_t device_bytes = 0;
  bool global_attention = false;
  bool reused_k_projection_for_v = false;
  bool includes_mlp = false;
  bool layer_scalar_applied = false;
  std::uint64_t mlp_input_mismatched_bytes = 0;
  std::uint64_t down_input_mismatched_bytes = 0;
  double reference_native_max_abs = 0.0;
  double reference_native_rms = 0.0;
  double reference_native_cosine = 0.0;
  std::vector<AttentionProbeSample> samples;
};

[[nodiscard]] Result<LayerCheckpointProbeResult>
RunLayer0LocalAttentionCheckpointProbe(const std::filesystem::path& model_directory);

[[nodiscard]] Result<LayerCheckpointProbeResult>
RunLayer5GlobalAttentionCheckpointProbe(const std::filesystem::path& model_directory);

[[nodiscard]] Result<LayerCheckpointProbeResult>
RunLayer0DecoderCheckpointProbe(const std::filesystem::path& model_directory);

}  // namespace gem16::internal

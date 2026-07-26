#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "gem16/status.h"

namespace gem16::internal {

struct Nvfp4ProbeSample {
  std::uint64_t row = 0;
  double oracle = 0.0;
  float cuda_reference = 0.0F;
  float sm120_direct = 0.0F;
  float simt_gemv = 0.0F;
};

struct Nvfp4CheckpointProbeResult {
  std::string tensor_name;
  std::string instruction;
  std::uint64_t rows = 0;
  std::uint64_t contracting_elements = 0;
  std::uint64_t packed_weight_bytes = 0;
  std::uint64_t weight_scale_bytes = 0;
  std::uint64_t device_bytes = 0;
  float input_global_divisor = 0.0F;
  float weight_global_divisor = 0.0F;
  bool activation_bytes_match = false;
  double activation_quantize_ms = 0.0;
  double cuda_reference_ms = 0.0;
  double sm120_direct_ms = 0.0;
  double simt_gemv_ms = 0.0;
  double reference_native_max_abs = 0.0;
  double reference_native_rms = 0.0;
  double reference_native_cosine = 0.0;
  double reference_simt_max_abs = 0.0;
  double reference_simt_rms = 0.0;
  double reference_simt_cosine = 0.0;
  double oracle_reference_max_abs = 0.0;
  double oracle_native_max_abs = 0.0;
  double oracle_simt_max_abs = 0.0;
  std::vector<Nvfp4ProbeSample> samples;
};

struct Nvfp4MlpCheckpointProbeResult {
  std::string instruction;
  std::uint64_t device_bytes = 0;
  bool input_activation_bytes_match = false;
  bool reference_down_activation_bytes_match = false;
  std::uint64_t native_down_activation_mismatched_bytes = 0;
  double cuda_reference_ms = 0.0;
  double sm120_direct_ms = 0.0;
  double sm120_unfused_gate_up_ms = 0.0;
  double sm120_fused_gate_up_ms = 0.0;
  double reference_native_max_abs = 0.0;
  double reference_native_rms = 0.0;
  double reference_native_cosine = 0.0;
  double oracle_reference_max_abs = 0.0;
  double oracle_native_max_abs = 0.0;
  std::vector<Nvfp4ProbeSample> samples;
};

[[nodiscard]] Result<Nvfp4CheckpointProbeResult> RunLayer0Nvfp4CheckpointProbe(
    const std::filesystem::path& model_directory,
    std::string_view projection,
    std::uint32_t warmups,
    std::uint32_t iterations);

[[nodiscard]] Result<Nvfp4MlpCheckpointProbeResult> RunLayer0Nvfp4MlpCheckpointProbe(
    const std::filesystem::path& model_directory,
    std::uint32_t warmups,
    std::uint32_t iterations);

}  // namespace gem16::internal

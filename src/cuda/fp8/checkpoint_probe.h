#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "gem16/status.h"

namespace gem16::internal {

struct Fp8ProbeSample {
  std::uint64_t row = 0;
  double oracle = 0.0;
  float cuda_reference = 0.0F;
  float sm120_direct = 0.0F;
};

struct Fp8CheckpointProbeResult {
  std::string tensor_name;
  std::string instruction;
  std::uint64_t rows = 0;
  std::uint64_t contracting_elements = 0;
  std::uint64_t weight_bytes = 0;
  std::uint64_t weight_scale_bytes = 0;
  std::uint64_t device_bytes = 0;
  float activation_scale = 0.0F;
  bool activation_bytes_match = false;
  bool activation_scale_match = false;
  double activation_quantize_ms = 0.0;
  double cuda_reference_ms = 0.0;
  double sm120_direct_ms = 0.0;
  double reference_native_max_abs = 0.0;
  double reference_native_rms = 0.0;
  double reference_native_cosine = 0.0;
  double oracle_reference_max_abs = 0.0;
  double oracle_native_max_abs = 0.0;
  std::vector<Fp8ProbeSample> samples;
};

[[nodiscard]] Result<Fp8CheckpointProbeResult> RunLayer0Fp8CheckpointProbe(
    const std::filesystem::path& model_directory,
    std::string_view projection,
    std::uint32_t warmups,
    std::uint32_t iterations);

}  // namespace gem16::internal

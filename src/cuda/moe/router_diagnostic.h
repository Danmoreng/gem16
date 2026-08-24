#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>

#include "gem16/status.h"

namespace gem16::internal {

// Diagnostic comparison of the locked serial BF16 router projection and the
// qualified SM120 tensor-core projection. Both consume the exact same
// already-normalized router input and transform it once. Raw FP32 accumulator
// outputs are retained separately from the production BF16-rounded logits so
// an offline oracle can distinguish accumulation error from the model's cast
// boundary.
struct Gemma4RouterDiagnosticWorkspace {
  float* transformed = nullptr;          // tokens * width
  float* serial_raw_logits = nullptr;    // tokens * experts
  float* serial_bf16_logits = nullptr;   // tokens * experts
  float* tensor_raw_logits = nullptr;    // tokens * experts
  float* tensor_bf16_logits = nullptr;   // tokens * experts
};

[[nodiscard]] Status LaunchGemma4RouterProjectionDiagnostic(
    const float* normalized_input, const std::uint16_t* router_scale_bf16,
    const std::uint16_t* router_projection_bf16,
    const Gemma4RouterDiagnosticWorkspace& workspace, std::uint64_t tokens,
    std::uint64_t width, std::uint32_t experts, cudaStream_t stream);

struct Gemma4RouterComparisonSummary {
  std::uint64_t cases = 0U;
  std::uint64_t top8_set_matches = 0U;
  std::uint64_t top8_order_matches = 0U;
  std::uint64_t changed_top8_slots = 0U;
  std::uint64_t flip_cases = 0U;
  double flip_margin_sum = 0.0;
  double gating_l1_sum = 0.0;
  float maximum_logit_absolute_delta = 0.0F;
  float maximum_flip_margin_8_9 = 0.0F;
  float maximum_tensor_flip_margin_8_9 = 0.0F;
  float maximum_gating_l1 = 0.0F;
  // Exact-router #8-#9 margin for flip cases: <=1/512, <=1/256,
  // <=1/128, <=1/64, <=1/32, <=1/16, <=1/8, and >1/8.
  std::array<std::uint64_t, 8> flip_margin_histogram{};
  std::array<std::uint64_t, 8> tensor_flip_margin_histogram{};
  std::array<std::uint64_t, 8> serial_margin_histogram{};
  std::array<std::uint64_t, 8> tensor_margin_histogram{};
};

// Process-local, explicit diagnostic switch used only by the benchmark probe.
// Normal inference leaves it disabled and launches no comparison kernels.
void SetGemma4RouterComparisonEnabled(bool enabled);
[[nodiscard]] bool Gemma4RouterComparisonEnabled();
[[nodiscard]] Status ResetGemma4RouterComparison(cudaStream_t stream);
[[nodiscard]] Result<Gemma4RouterComparisonSummary>
CopyGemma4RouterComparison(cudaStream_t stream);

// Production SM120 projection. Selection remains explicit in the owning
// engine configuration; diagnostics call the same implementation.
[[nodiscard]] Status LaunchGemma4Sm120TensorRouterProjection(
    const float* transformed, const std::uint16_t* router_projection_bf16,
    float* tensor_bf16_logits, std::uint64_t tokens, std::uint64_t width,
    std::uint32_t experts, cudaStream_t stream);

[[nodiscard]] Status LaunchGemma4RouterComparisonDiagnostic(
    const float* serial_bf16_logits, const float* tensor_bf16_logits,
    const std::uint16_t* per_expert_scale_bf16, std::uint64_t tokens,
    std::uint32_t experts, std::uint32_t top_k, cudaStream_t stream);

}  // namespace gem16::internal

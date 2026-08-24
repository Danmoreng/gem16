#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "gem16/status.h"

namespace gem16::internal {

class Gemma4Moe26BDeviceArtifact;

// Correctness-only view of one NVFP4 matrix in the immutable M08/M09 arena.
// The packed values and scales use the row8/K64 layouts documented by
// sm120_layout.h. Divisors are immutable initialization-time metadata.
struct Gemma4MoeNvfp4Matrix {
  const std::uint8_t* packed_e2m1 = nullptr;
  const std::uint8_t* scales_e4m3fn = nullptr;
  std::uint64_t rows = 0;
  std::uint64_t columns = 0;
  float activation_global_divisor = 0.0F;
  float weight_global_divisor = 0.0F;
};

struct Gemma4MoeReferenceWeights {
  const std::uint16_t* pre_shared_norm_bf16 = nullptr;
  const std::uint16_t* post_shared_norm_bf16 = nullptr;
  const std::uint16_t* pre_expert_norm_bf16 = nullptr;
  const std::uint16_t* post_expert_norm_bf16 = nullptr;
  const std::uint16_t* post_combined_norm_bf16 = nullptr;
  const std::uint16_t* router_scale_bf16 = nullptr;
  const std::uint16_t* router_projection_bf16 = nullptr;
  const std::uint16_t* per_expert_scale_bf16 = nullptr;
  const std::uint16_t* layer_scalar_bf16 = nullptr;
  Gemma4MoeNvfp4Matrix shared_gate;
  Gemma4MoeNvfp4Matrix shared_up;
  Gemma4MoeNvfp4Matrix shared_down;
  // Expert matrices are flattened expert-major. gate_up has
  // experts * (2 * expert_intermediate) rows; down has experts * width rows.
  Gemma4MoeNvfp4Matrix expert_gate_up;
  Gemma4MoeNvfp4Matrix expert_down;
};

// Every pointer is caller-owned fixed device storage. The launcher performs no
// allocation, host routing, synchronization, filesystem access, or repacking.
struct Gemma4MoeReferenceWorkspace {
  float* shared_input = nullptr;          // width
  std::uint8_t* shared_input_packed = nullptr;  // width / 2
  std::uint8_t* shared_input_scales = nullptr;  // width / 16
  float* shared_gate = nullptr;           // shared_intermediate
  float* shared_up = nullptr;             // shared_intermediate
  float* shared_product = nullptr;        // shared_intermediate
  std::uint8_t* shared_product_packed = nullptr;
  std::uint8_t* shared_product_scales = nullptr;
  float* shared_output = nullptr;         // width
  float* shared_post = nullptr;           // width

  float* router_normalized = nullptr;     // width
  float* router_transformed = nullptr;    // width
  float* router_logits = nullptr;         // experts
  float* router_probabilities = nullptr;  // experts
  std::uint32_t* top_ids = nullptr;       // top_k
  float* top_weights = nullptr;           // top_k

  float* expert_input = nullptr;          // width
  std::uint8_t* expert_input_packed = nullptr;
  std::uint8_t* expert_input_scales = nullptr;
  float* expert_gate_up = nullptr;         // top_k * 2 * expert_intermediate
  float* expert_product = nullptr;         // top_k * expert_intermediate
  std::uint8_t* expert_product_packed = nullptr;
  std::uint8_t* expert_product_scales = nullptr;
  float* expert_down = nullptr;            // top_k * width
  float* expert_contributions = nullptr;   // top_k * width
  float* routed_sum = nullptr;             // width
  float* routed_post = nullptr;            // width
  float* combined = nullptr;               // width
  float* feed_forward = nullptr;           // width, optional capture
  // Optional caller-owned sticky flag. Initialize to one before an engine
  // transaction. Router kernels atomically clear it on any non-finite input
  // or invalid scale while still writing safe, non-stale routing outputs.
  int* routing_finite = nullptr;
};

enum class Gemma4MoePrefillRouter {
  kSerialExact,
  kSm120TensorCore,
};

struct Gemma4MoeReferenceConfig {
  std::uint64_t width = 0;
  std::uint64_t shared_intermediate = 0;
  std::uint64_t expert_intermediate = 0;
  std::uint32_t experts = 0;
  std::uint32_t top_k = 0;
  float epsilon = 0.0F;
  Gemma4MoePrefillRouter prefill_router =
      Gemma4MoePrefillRouter::kSerialExact;
};

// Initialization-only pointer resolution for one exact 26B layer. The
// returned view aliases the immutable M08 arena and owns no storage.
[[nodiscard]] Result<Gemma4MoeReferenceWeights>
BindGemma4Moe26BReferenceWeights(
    const Gemma4Moe26BDeviceArtifact& artifact, std::uint32_t layer);

// Correctness-only projection over the accepted row8/K64 artifact layout.
// This is shared by M11 and the provisional M13 tied head; it performs no
// allocation, repacking, synchronization, or precision selection.
[[nodiscard]] Status LaunchGemma4MoeTiledNvfp4ReferenceProjection(
    const Gemma4MoeNvfp4Matrix& matrix,
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn, float* output,
    cudaStream_t stream);

// Executes one token through the accepted M10 feed-forward sequence. Hidden
// input/output are FP32 containers holding BF16-rounded recurrent state.
[[nodiscard]] Status LaunchGemma4MoeReferenceLayer(
    const float* hidden, float* output,
    const Gemma4MoeReferenceConfig& config,
    const Gemma4MoeReferenceWeights& weights,
    const Gemma4MoeReferenceWorkspace& workspace, cudaStream_t stream);

// M14 batch-one native SM120 path. Router, normalization and deterministic
// reduction semantics are shared with M11; all NVFP4 projections dispatch the
// project-built mma.sync SM120 kernels and selected experts remain on device.
[[nodiscard]] Status LaunchGemma4MoeSm120Layer(
    const float* hidden, float* output,
    const Gemma4MoeReferenceConfig& config,
    const Gemma4MoeReferenceWeights& weights,
    const Gemma4MoeReferenceWorkspace& workspace, cudaStream_t stream,
    cudaStream_t shared_branch_stream = nullptr,
    cudaEvent_t fork_event = nullptr, cudaEvent_t join_event = nullptr);

}  // namespace gem16::internal

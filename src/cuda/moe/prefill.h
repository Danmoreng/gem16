#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "cuda/moe/prefill_plan.h"
#include "cuda/moe/reference.h"
#include "gem16/status.h"

namespace gem16::internal {

// All pointers alias caller-owned fixed M15 workspace. Region sizes are
// determined by BuildGemma4MoePrefillPlan(chunk_tokens).
struct Gemma4MoePrefillWorkspace {
  // T * experts; after routing, this first aliases ceil(T / 32) * experts
  // stable-group chunk offsets and then at most 8T expert-tile descriptors.
  // The top_k=8 model contract guarantees both intermediate capacities.
  float* router_logits = nullptr;
  float* router_probabilities = nullptr;   // T * experts
  float* token_hidden = nullptr;           // T * width, reusable
  std::uint8_t* token_packed = nullptr;     // T * width / 2
  std::uint8_t* token_scales = nullptr;     // T * width / 16
  float* expert_product = nullptr;          // 8T * expert_intermediate
  std::uint8_t* expert_product_packed = nullptr;
  std::uint8_t* expert_product_scales = nullptr;
  float* expert_down = nullptr;             // 8T * width, assignment order
  float* shared_product = nullptr;           // T * shared_intermediate
  std::uint8_t* shared_product_packed = nullptr;
  std::uint8_t* shared_product_scales = nullptr;
  float* shared_output = nullptr;            // T * width, reusable
  float* reduced_output = nullptr;           // T * width, shared post-norm

  Gemma4MoePrefillAssignment* assignments = nullptr;  // 8T token-major
  std::uint32_t* histogram = nullptr;                  // experts
  std::uint32_t* prefix = nullptr;                     // experts + 1
  std::uint32_t* permutation = nullptr;                // grouped -> original
  std::uint32_t* inverse_permutation = nullptr;        // original -> grouped
  // Optional sticky transaction flag shared with decode. See the decode
  // workspace contract for initialization and failure semantics.
  int* routing_finite = nullptr;

  // Optional production storage for the already-rounded routed-expert
  // boundaries. When both pointers are present, expert_product/expert_down
  // must be null and the layer uses the bit-identical physical-BF16 path.
  // Keeping the float-container fields preserves the independent M15
  // reference/differential harness without retaining both representations in
  // the integrated engine.
  std::uint16_t* expert_product_bf16 = nullptr;
  std::uint16_t* expert_down_bf16 = nullptr;
  // Trellis35 reuses this fixed activation scratch for Gate+Up and Down E4M3
  // rows. It is separate from physical BF16 output because assignment-major
  // strides differ and therefore cannot safely alias during tiled projection.
  std::uint8_t* trellis_activation = nullptr;
};

struct Trellis35DeviceLayerBinding;

// Deterministic token-major/top-k-major routed-expert reduction. The float
// form consumes BF16-rounded values held in FP32 containers.
[[nodiscard]] Status LaunchGemma4MoeReduceAssignments(
    const float* expert_down,
    const Gemma4MoePrefillAssignment* assignments, float* routed_sum,
    std::uint64_t width, std::uint32_t top_k, std::uint64_t tokens,
    cudaStream_t stream);

// Same exact weighted-BF16 and slot-order reduction for physical BF16 W2
// output. No change is made to assignment weights or accumulation order.
[[nodiscard]] Status LaunchGemma4MoeReduceAssignmentsBf16(
    const std::uint16_t* expert_down_bf16,
    const Gemma4MoePrefillAssignment* assignments, float* routed_sum,
    std::uint64_t width, std::uint32_t top_k, std::uint64_t tokens,
    cudaStream_t stream);

// Complete M15 bounded-workspace grouped prefill layer. Hidden/output contain
// T contiguous FP32 containers at the recurrent BF16 boundary. Routing,
// grouping, W13, W2 and inverse mapping remain entirely device-side.
[[nodiscard]] Status LaunchGemma4MoeSm120PrefillLayer(
    const float* hidden, float* output, std::uint64_t tokens,
    const Gemma4MoeReferenceConfig& config,
    const Gemma4MoeReferenceWeights& weights,
    const Gemma4MoePrefillWorkspace& workspace, cudaStream_t stream);

[[nodiscard]] Status LaunchGemma4MoeSm120Trellis35PrefillLayer(
    const float* hidden, float* output, std::uint64_t tokens,
    const Gemma4MoeReferenceConfig& config,
    const Gemma4MoeReferenceWeights& weights,
    const Trellis35DeviceLayerBinding& trellis_layer,
    const Gemma4MoePrefillWorkspace& workspace, cudaStream_t stream);

}  // namespace gem16::internal

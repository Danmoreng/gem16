#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "cuda/engine/gemma4_26b_trellis35_artifact.h"
#include "gem16/status.h"

namespace gem16::internal {

inline constexpr std::uint32_t kTrellis35M1TopK = 8U;
inline constexpr std::uint64_t kTrellis35GateUpInput = 2816U;
inline constexpr std::uint64_t kTrellis35GateUpOutput = 1408U;
inline constexpr std::uint64_t kTrellis35ExpertIntermediate = 704U;
inline constexpr std::uint64_t kTrellis35DownInput = 768U;
inline constexpr std::uint64_t kTrellis35DownOutput = 2816U;

// Fixed caller-owned storage for one ordinary-decode routed-expert operation.
// All fields are allocated and bound before graph capture; the launcher itself
// performs no allocation, synchronization, filesystem access, or repacking.
struct Trellis35M1Workspace {
  float* gate_up_input_transformed = nullptr;       // 8 * 2816
  std::uint8_t* gate_up_input_e4m3 = nullptr;       // 8 * 2816
  float* gate_up_input_scales = nullptr;            // 8
  float* gate_up_transformed_output = nullptr;      // 8 * 1408
  float* gate_up_output = nullptr;                  // 8 * 1408
  float* product = nullptr;                         // 8 * 704
  float* down_input_transformed = nullptr;          // 8 * 768
  std::uint8_t* down_input_e4m3 = nullptr;          // 8 * 768
  float* down_input_scales = nullptr;               // 8
  float* down_transformed_output = nullptr;         // 8 * 2816
  float* down_output = nullptr;                     // 8 * 2816
};

// Applies diag(SUH) followed by independent normalized 128-point Hadamard
// blocks. Values beyond logical_elements are explicit zero padding.
[[nodiscard]] Status LaunchTrellis35InputTransformM1(
    const float* input, const std::uint16_t* suh_f16,
    float* transformed_output, std::uint64_t logical_elements,
    std::uint64_t physical_elements, cudaStream_t stream);

// Correctness-first mixed-K3/K4 W4A8 projection. Packed Trellis weights are
// decoded only into registers, rounded immediately to E4M3, and multiplied by
// E4M3 activations. No complete expert matrix is written to global memory.
[[nodiscard]] Status LaunchTrellis35ReferenceW4A8ProjectionM1(
    const std::uint8_t* activation_e4m3, const float* activation_scale,
    const Trellis35DeviceFamilyBinding& family, std::uint32_t expert,
    float* transformed_output, std::uint64_t input_elements,
    std::uint64_t output_elements, cudaStream_t stream);

// Direct Trellis -> E4M3-register -> SM120 FP8-MMA selected-expert
// projection. Expert IDs remain device runtime data and each CTA dispatches
// K3/K4 from the immutable descriptor; no decoded weight matrix is stored.
[[nodiscard]] Status LaunchTrellis35MmaW4A8ProjectionM1(
    const std::uint8_t* activation_e4m3, const float* activation_scales,
    const Trellis35DeviceFamilyBinding& family,
    const std::uint32_t* selected_experts, float* transformed_output,
    std::uint64_t input_elements, std::uint64_t output_elements,
    cudaStream_t stream);

// Applies the complete normalized 128-point output Hadamard before SVH. Gate
// and Up callers must invoke this for all 1408 outputs before splitting 704/704.
[[nodiscard]] Status LaunchTrellis35OutputTransformM1(
    const float* transformed_input, const std::uint16_t* svh_f16,
    float* output, std::uint64_t elements, cudaStream_t stream);

// Complete ordinary-decode routed-expert path for the eight selected slots.
// selected_experts and route_weights are read-only device arrays in router
// slot order. The final reduction deliberately accumulates slot 0 through 7.
[[nodiscard]] Status LaunchTrellis35SelectedExpertsM1(
    const float* input, const std::uint32_t* selected_experts,
    const float* route_weights, const Trellis35DeviceLayerBinding& layer,
    const Trellis35M1Workspace& workspace, float* output,
    cudaStream_t stream);

}  // namespace gem16::internal

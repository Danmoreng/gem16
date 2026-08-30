#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "cuda/engine/gemma4_26b_trellis35_artifact.h"
#include "gem16/status.h"

namespace gem16::internal {

struct Gemma4MoePrefillWorkspace;
struct Gemma4MoePrefillAssignment;

inline constexpr std::uint32_t kTrellis35M1TopK = 8U;
inline constexpr std::uint32_t kTrellis35T3Rows = 3U;
inline constexpr std::uint32_t kTrellis35T3Assignments =
    kTrellis35T3Rows * kTrellis35M1TopK;
inline constexpr std::uint64_t kTrellis35GateUpInput = 2816U;
inline constexpr std::uint64_t kTrellis35GateUpOutput = 1408U;
inline constexpr std::uint64_t kTrellis35ExpertIntermediate = 704U;
inline constexpr std::uint64_t kTrellis35DownInput = 768U;
inline constexpr std::uint64_t kTrellis35DownOutput = 2816U;

enum class Trellis35SmallTransformMode {
  // WP11/WP14 numerical and performance rollback.
  kDirectH128,
  // WP15 bounded isolation candidates.
  kWarpInputH128,
  kWarpOutputH128,
  // WP15 warp FWHT for Ordinary M1 and Fixed-D2 T3.
  kWarpH128,
};

enum class Trellis35T3ProjectionMode {
  // WP11 rollback: one broadcast-row MMA accumulator per matching row.
  kIndependentRows,
  // WP15: one physical M16 activation tile, with at most three live rows.
  kM16,
};

enum class Trellis35SmallGeluDownMode {
  // WP20 numerical and performance rollback.
  kSeparate,
  // GELU, Down H128 transform, amax and E4M3 in one assignment CTA.
  kFusedTransformQuantize,
};

enum class Trellis35M1ProjectionOutputMode {
  // Production selection, including the packet-local environment rollback.
  kEnvironment,
  // WP25 numerical and performance rollback.
  kSeparateN32,
  // Fuse the N128 inverse transform for both routed projection families.
  kFusedN128,
  // WP25 family-isolation modes used by qualification and A/B measurement.
  kGateUpFusedN128,
  kDownFusedN128,
};

// Diagnostic-only bounded materialization used to decide whether a transient
// decoded-weight cache can beat inline Trellis decode. The slab is row-major
// E4M3 with unit BF16 row scales and is never an engine/runtime fallback.
[[nodiscard]] Status LaunchTrellis35DecodeE4M3SlabDiagnostic(
    const Trellis35DeviceFamilyBinding& family, std::uint32_t expert,
    std::uint64_t input_elements, std::uint64_t output_elements,
    std::uint64_t output_offset, std::uint64_t slab_rows,
    std::uint8_t* weight_e4m3, std::uint16_t* weight_scales_bf16,
    cudaStream_t stream);

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

// Fixed storage for the dedicated Fixed-D2 verifier operator. Assignment-major
// regions contain 3 rows * 8 router slots; outputs remain row-major.
struct Trellis35T3Workspace {
  float* gate_up_input_transformed = nullptr;       // 24 * 2816
  std::uint8_t* gate_up_input_e4m3 = nullptr;       // 24 * 2816
  float* gate_up_input_scales = nullptr;            // 24
  float* gate_up_transformed_output = nullptr;      // 24 * 1408
  float* gate_up_output = nullptr;                  // 24 * 1408
  float* product = nullptr;                         // 24 * 704
  float* down_input_transformed = nullptr;          // 24 * 768
  std::uint8_t* down_input_e4m3 = nullptr;          // 24 * 768
  float* down_input_scales = nullptr;               // 24
  float* down_transformed_output = nullptr;         // 24 * 2816
  float* down_output = nullptr;                     // 24 * 2816
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
    cudaStream_t stream,
    Trellis35SmallTransformMode transform_mode =
        Trellis35SmallTransformMode::kWarpH128,
    Trellis35SmallGeluDownMode gelu_down_mode =
        Trellis35SmallGeluDownMode::kSeparate,
    Trellis35M1ProjectionOutputMode projection_output_mode =
        Trellis35M1ProjectionOutputMode::kEnvironment);

// Dedicated Fixed-D2 T=3 path. This is one 24-assignment pipeline, not three
// M1 calls. Each first-occurrence expert group decodes a weight fragment once
// and feeds every row that selected that expert before discarding it.
[[nodiscard]] Status LaunchTrellis35SelectedExpertsT3(
    const float* input_rows, const std::uint32_t* selected_experts,
    const float* route_weights, const Trellis35DeviceLayerBinding& layer,
    const Trellis35T3Workspace& workspace, float* output_rows,
    cudaStream_t stream,
    Trellis35SmallTransformMode transform_mode =
        Trellis35SmallTransformMode::kWarpH128,
    Trellis35T3ProjectionMode projection_mode =
        Trellis35T3ProjectionMode::kM16,
    Trellis35SmallGeluDownMode gelu_down_mode =
        Trellis35SmallGeluDownMode::kSeparate);

// Routed-expert-only prefill replacement. normalized_hidden contains T
// recurrent BF16 values in FP32 containers. The launcher consumes the existing
// token-major assignments plus stable expert permutation/prefix and aliases
// only dead regions of Gemma4MoePrefillWorkspace; no second prefill arena is
// required. The reduced routed result is written to workspace.token_hidden.
enum class Trellis35PrefillScheduleMode {
  // Standalone operator tests do not run the enclosing MoE scheduler.
  kBuildStandalone,
  // Production consumes BuildExpertTileScheduleKernel's existing M32 result.
  kConsumeM32,
  // Trellis35-only hybrid schedule: full/33--63-row M64 tiles followed by
  // <=32-row M32 tails. The NVFP4 scheduler and its descriptors are unchanged.
  kBuildM64Hybrid,
};

enum class Trellis35PrefillKernelMode {
  // WP11 rollback and numerical A/B reference.
  kLegacyM4,
  // WP12 true two-M16 grouped candidate.
  kGroupedM32,
  // WP17 four-M16 candidate with M32 tail fallback.
  kGroupedM64Hybrid,
};

enum class Trellis35PrefillTransformMode {
  // WP12 numerical and performance rollback.
  kDirectH128,
  // WP13 warp FWHT with fused amax/scale/E4M3 input quantization.
  kWarpH128,
};

enum class Trellis35PrefillOutputMode {
  // WP12/WP13 rollback: one projection and inverse-transform launch per N128.
  kLoopN128,
  // WP14 candidate C: one N128 CTA with a shared projection/inverse epilogue.
  kFusedN128,
};

enum class Trellis35PrefillGeluDownMode {
  // WP19 numerical and performance rollback.
  kTwoKernel,
  // Preserves the physical BF16 product roundpoint without materializing it.
  kFusedTransformQuantize,
};

// Bounded WP19 operator entry used by production and the byte-exact oracle.
// product_bf16 remains mandatory as rollback storage but is not accessed by
// kFusedTransformQuantize.
[[nodiscard]] Status LaunchTrellis35GatedGeluDownTransformQuantizeBf16(
    const std::uint16_t* gate_up_bf16, std::uint16_t* product_bf16,
    const Trellis35DeviceFamilyBinding& down,
    const Gemma4MoePrefillAssignment* assignments,
    std::uint8_t* down_activation_e4m3, float* down_activation_scales,
    std::uint64_t assignment_count, Trellis35PrefillGeluDownMode mode,
    cudaStream_t stream);

// Bounded diagnostic entry used by the exhaustive transform oracle. It does
// not participate in model execution or allocate storage.
[[nodiscard]] Status LaunchTrellis35H128WarpDiagnostic(
    const float* input, float* output, std::uint64_t vectors,
    cudaStream_t stream);

[[nodiscard]] Status LaunchTrellis35PrefillExpertsW4A8(
    const float* normalized_hidden, std::uint64_t tokens,
    const Trellis35DeviceLayerBinding& layer,
    const Gemma4MoePrefillWorkspace& workspace,
    Trellis35PrefillScheduleMode schedule_mode,
    Trellis35PrefillKernelMode kernel_mode,
    Trellis35PrefillTransformMode transform_mode,
    Trellis35PrefillOutputMode output_mode, cudaStream_t stream);

}  // namespace gem16::internal

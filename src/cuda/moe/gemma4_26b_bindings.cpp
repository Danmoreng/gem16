#include "cuda/moe/reference.h"

#include <cstdint>
#include <string>
#include <type_traits>

#include "cuda/engine/gemma4_26b_artifact.h"
#include "cuda/engine/gemma4_26b_trellis35_artifact.h"

namespace gem16::internal {
namespace {

template <typename T, typename Artifact>
Result<const T*> Pointer(const Artifact& artifact, const std::string& name) {
  auto pointer = [&]() {
    if constexpr (std::is_same_v<Artifact, Gemma4Moe26BDeviceArtifact>) {
      return artifact.Pointer(name);
    } else {
      return artifact.NonRoutedPointer(name);
    }
  }();
  if (!pointer.ok()) return pointer.status();
  return reinterpret_cast<const T*>(pointer.value());
}

template <typename Artifact>
Result<Gemma4MoeNvfp4Matrix> Matrix(
    const Artifact& artifact, const std::string& stem,
    std::uint64_t rows, std::uint64_t columns) {
  auto packed = Pointer<std::uint8_t>(artifact, stem + ".weight_packed");
  auto scales = Pointer<std::uint8_t>(artifact, stem + ".weight_scale");
  auto activation = artifact.HostFloat32(stem + ".input_global_scale");
  auto weight = artifact.HostFloat32(stem + ".weight_global_scale");
  if (!packed.ok()) return packed.status();
  if (!scales.ok()) return scales.status();
  if (!activation.ok()) return activation.status();
  if (!weight.ok()) return weight.status();
  return Gemma4MoeNvfp4Matrix{packed.value(), scales.value(), rows, columns,
                              activation.value(), weight.value()};
}

}  // namespace

template <typename Artifact>
Result<Gemma4MoeReferenceWeights> BindWeights(
    const Artifact& artifact, std::uint32_t layer) {
  constexpr bool kTrellis35 =
      std::is_same_v<Artifact, Gemma4Moe26BTrellis35DeviceArtifact>;
  constexpr std::uint64_t kLayers = 30U;
  constexpr std::uint64_t kWidth = 2816U;
  constexpr std::uint64_t kShared = 2112U;
  constexpr std::uint64_t kExpert = 704U;
  constexpr std::uint64_t kExperts = 128U;
  if (layer >= kLayers) {
    return Status(StatusCode::kInvalidArgument,
                  "M11/M13 MoE binding layer is out of range");
  }
  const std::string prefix =
      "model.language_model.layers." + std::to_string(layer);
  Gemma4MoeReferenceWeights result;
  auto bind = [&](const char* suffix,
                  const std::uint16_t** destination) -> Status {
    auto pointer = Pointer<std::uint16_t>(artifact, prefix + suffix);
    if (!pointer.ok()) return pointer.status();
    *destination = pointer.value();
    return Status::Ok();
  };
  Status status = bind(".pre_feedforward_layernorm.weight",
                       &result.pre_shared_norm_bf16);
  if (status.ok()) status = bind(".post_feedforward_layernorm_1.weight",
                                 &result.post_shared_norm_bf16);
  if (status.ok()) status = bind(".pre_feedforward_layernorm_2.weight",
                                 &result.pre_expert_norm_bf16);
  if (status.ok()) status = bind(".post_feedforward_layernorm_2.weight",
                                 &result.post_expert_norm_bf16);
  if (status.ok()) status = bind(".post_feedforward_layernorm.weight",
                                 &result.post_combined_norm_bf16);
  if (status.ok()) status = bind(".router.scale", &result.router_scale_bf16);
  if (status.ok()) status = bind(".router.proj.weight",
                                 &result.router_projection_bf16);
  if (status.ok()) status = bind(".router.per_expert_scale",
                                 &result.per_expert_scale_bf16);
  if (status.ok()) status = bind(".layer_scalar", &result.layer_scalar_bf16);
  if (!status.ok()) return status;

  auto shared_gate = Matrix(artifact, prefix + ".mlp.gate_proj", kShared,
                            kWidth);
  auto shared_up = Matrix(artifact, prefix + ".mlp.up_proj", kShared, kWidth);
  auto shared_down = Matrix(artifact, prefix + ".mlp.down_proj", kWidth,
                            kShared);
  if (!shared_gate.ok()) return shared_gate.status();
  if (!shared_up.ok()) return shared_up.status();
  if (!shared_down.ok()) return shared_down.status();
  result.shared_gate = shared_gate.value();
  result.shared_up = shared_up.value();
  result.shared_down = shared_down.value();
  if constexpr (!kTrellis35) {
    auto expert_gate_up =
        Matrix(artifact, prefix + ".experts.gate_up_proj",
               kExperts * 2U * kExpert, kWidth);
    auto expert_down = Matrix(artifact, prefix + ".experts.down_proj",
                              kExperts * kWidth, kExpert);
    if (!expert_gate_up.ok()) return expert_gate_up.status();
    if (!expert_down.ok()) return expert_down.status();
    result.expert_gate_up = expert_gate_up.value();
    result.expert_down = expert_down.value();
  } else {
    // The fused input-boundary kernel still writes its now-dead NVFP4 expert
    // activation scratch. A positive divisor preserves that kernel contract;
    // no routed NVFP4 weight pointer or persistent fallback is bound.
    result.expert_gate_up.activation_global_divisor = 1.0F;
  }
  return result;
}

Result<Gemma4MoeReferenceWeights> BindGemma4Moe26BReferenceWeights(
    const Gemma4Moe26BDeviceArtifact& artifact, std::uint32_t layer) {
  return BindWeights(artifact, layer);
}

Result<Gemma4MoeReferenceWeights> BindGemma4Moe26BReferenceWeights(
    const Gemma4Moe26BTrellis35DeviceArtifact& artifact,
    std::uint32_t layer) {
  return BindWeights(artifact, layer);
}

}  // namespace gem16::internal

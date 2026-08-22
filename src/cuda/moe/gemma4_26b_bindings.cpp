#include "cuda/moe/reference.h"

#include <cstdint>
#include <string>

#include "cuda/engine/gemma4_26b_artifact.h"

namespace gem16::internal {
namespace {

template <typename T>
Result<const T*> Pointer(const Gemma4Moe26BDeviceArtifact& artifact,
                         const std::string& name) {
  auto pointer = artifact.Pointer(name);
  if (!pointer.ok()) return pointer.status();
  return reinterpret_cast<const T*>(pointer.value());
}

Result<Gemma4MoeNvfp4Matrix> Matrix(
    const Gemma4Moe26BDeviceArtifact& artifact, const std::string& stem,
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

Result<Gemma4MoeReferenceWeights> BindGemma4Moe26BReferenceWeights(
    const Gemma4Moe26BDeviceArtifact& artifact, std::uint32_t layer) {
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
  auto expert_gate_up =
      Matrix(artifact, prefix + ".experts.gate_up_proj",
             kExperts * 2U * kExpert, kWidth);
  auto expert_down = Matrix(artifact, prefix + ".experts.down_proj",
                            kExperts * kWidth, kExpert);
  if (!shared_gate.ok()) return shared_gate.status();
  if (!shared_up.ok()) return shared_up.status();
  if (!shared_down.ok()) return shared_down.status();
  if (!expert_gate_up.ok()) return expert_gate_up.status();
  if (!expert_down.ok()) return expert_down.status();
  result.shared_gate = shared_gate.value();
  result.shared_up = shared_up.value();
  result.shared_down = shared_down.value();
  result.expert_gate_up = expert_gate_up.value();
  result.expert_down = expert_down.value();
  return result;
}

}  // namespace gem16::internal

#include "model/gemma4_26b_manifest.h"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <string>
#include <utility>

namespace gem16::internal {
namespace {

constexpr std::uint64_t kLayerCount = 30;
constexpr std::uint64_t kHidden = 2816;
constexpr std::uint64_t kSharedIntermediate = 2112;
constexpr std::uint64_t kExpertIntermediate = 704;
constexpr std::uint64_t kExpertCount = 128;
constexpr std::uint64_t kVocabulary = 262144;
constexpr std::uint64_t kVisionHidden = 1152;
constexpr std::uint64_t kVisionIntermediate = 4304;
constexpr std::uint64_t kVisionLayerCount = 27;
constexpr std::string_view kSourceFamily = "google_gemma4_26b_bf16_source";
constexpr std::string_view kExternalFamily =
    "unsloth_gemma4_26b_nvfp4_reference";
constexpr std::string_view kCompiledFamily = "gem16_compiled_hybrid";
constexpr std::string_view kExternalProducer =
    "llm-compressor/compressed-tensors@0.17.2.a20260707";
constexpr std::string_view kCompiledProducer = "gem16";

struct ContractMetadata {
  TensorRole role;
  ResidencyClass residency;
  std::string_view quantization_component = "weight";
  std::string_view quantization_class = "BF16";
  std::string_view logical_dtype = "BF16";
  std::string_view quantization_producer = "none";
  std::string_view local_scale_dtype = "none";
  std::uint64_t local_scale_vector_size = 0;
  std::string_view global_scale_role = "none";
  std::string_view activation_scale_role = "none";
  std::string_view final_gpu_layout = "none";
  std::string_view logical_axis_order;
  std::int64_t layer_index = -1;
  std::int64_t expert_index = -1;
  std::int64_t expert_axis = -1;
  bool aliased = false;
};

Result<std::uint64_t> CheckedMultiply(std::uint64_t left,
                                      std::uint64_t right,
                                      std::string_view description) {
  if (right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right) {
    return Status(StatusCode::kDataLoss,
                  "Gemma 4 26B tensor contract overflow: " +
                      std::string(description));
  }
  return left * right;
}

Result<std::uint64_t> CheckedAdd(std::uint64_t left, std::uint64_t right,
                                 std::string_view description) {
  if (left > std::numeric_limits<std::uint64_t>::max() - right) {
    return Status(StatusCode::kDataLoss,
                  "Gemma 4 26B tensor contract overflow: " +
                      std::string(description));
  }
  return left + right;
}

Result<std::uint64_t> PhysicalBytes(std::string_view dtype,
                                    const std::vector<std::uint64_t>& shape) {
  std::uint64_t element_bytes = 0;
  if (dtype == "U8" || dtype == "F8_E4M3") {
    element_bytes = 1;
  } else if (dtype == "BF16") {
    element_bytes = 2;
  } else if (dtype == "F32") {
    element_bytes = 4;
  } else {
    return Status(StatusCode::kUnsupported,
                  "unsupported dtype in Gemma 4 26B tensor contract: " +
                      std::string(dtype));
  }
  std::uint64_t elements = 1;
  for (const auto dimension : shape) {
    auto product = CheckedMultiply(elements, dimension, "shape product");
    if (!product.ok()) return product.status();
    elements = product.value();
  }
  return CheckedMultiply(elements, element_bytes, "physical byte count");
}

Result<TensorInfo> MakeTensor(std::string name, std::string dtype,
                              std::vector<std::uint64_t> physical_shape,
                              std::vector<std::uint64_t> logical_shape,
                              std::string_view source_family,
                              const ContractMetadata& metadata) {
  auto bytes = PhysicalBytes(dtype, physical_shape);
  if (!bytes.ok()) return bytes.status();
  TensorInfo tensor;
  tensor.name = std::move(name);
  tensor.shape = std::move(physical_shape);
  tensor.logical_shape = std::move(logical_shape);
  tensor.storage_dtype = std::move(dtype);
  tensor.logical_dtype = std::string(metadata.logical_dtype);
  tensor.quantization_class = std::string(metadata.quantization_class);
  tensor.byte_length = bytes.value();
  tensor.expected_role = std::string(TensorRoleName(metadata.role));
  tensor.tensor_role = tensor.expected_role;
  tensor.residency_class =
      std::string(ResidencyClassName(metadata.residency));
  tensor.source_family = std::string(source_family);
  tensor.quantization_component =
      std::string(metadata.quantization_component);
  tensor.quantization_producer =
      std::string(metadata.quantization_producer);
  tensor.local_scale_dtype = std::string(metadata.local_scale_dtype);
  tensor.local_scale_vector_size = metadata.local_scale_vector_size;
  tensor.global_scale_role = std::string(metadata.global_scale_role);
  tensor.activation_scale_role = std::string(metadata.activation_scale_role);
  tensor.final_gpu_layout = std::string(metadata.final_gpu_layout);
  tensor.logical_axis_order = std::string(metadata.logical_axis_order);
  tensor.layer_index = metadata.layer_index;
  tensor.expert_index = metadata.expert_index;
  tensor.expert_axis = metadata.expert_axis;
  tensor.loaded_in_text_only_mode =
      metadata.residency != ResidencyClass::kCompileExcludedVision;
  tensor.aliased = metadata.aliased;
  tensor.layout = source_family == kCompiledFamily
                      ? tensor.final_gpu_layout
                      : "source Safetensors order";
  return tensor;
}

Status AddTensor(std::vector<TensorInfo>* tensors, std::string name,
                 std::string dtype,
                 std::vector<std::uint64_t> physical_shape,
                 std::vector<std::uint64_t> logical_shape,
                 std::string_view source_family,
                 const ContractMetadata& metadata) {
  auto tensor = MakeTensor(std::move(name), std::move(dtype),
                           std::move(physical_shape),
                           std::move(logical_shape), source_family, metadata);
  if (!tensor.ok()) return tensor.status();
  tensors->push_back(std::move(tensor).value());
  return Status::Ok();
}

Status AddBf16(std::vector<TensorInfo>* tensors, std::string name,
               std::vector<std::uint64_t> shape, TensorRole role,
               ResidencyClass residency, std::string_view source_family,
               std::int64_t layer_index = -1,
               std::int64_t expert_index = -1,
               std::string_view logical_axis_order = {},
               std::int64_t expert_axis = -1, bool aliased = false,
               std::string_view final_layout = "source_bf16") {
  ContractMetadata metadata{.role = role,
                            .residency = residency,
                            .final_gpu_layout = final_layout,
                            .logical_axis_order = logical_axis_order,
                            .layer_index = layer_index,
                            .expert_index = expert_index,
                            .expert_axis = expert_axis,
                            .aliased = aliased};
  auto logical_shape = shape;
  return AddTensor(tensors, std::move(name), "BF16", std::move(shape),
                   std::move(logical_shape), source_family, metadata);
}

Status AddVisionContract(std::vector<TensorInfo>* tensors,
                         std::string_view source_family) {
  constexpr auto residency = ResidencyClass::kCompileExcludedVision;
  auto status = AddBf16(
      tensors, "model.embed_vision.embedding_projection.weight",
      {kHidden, kVisionHidden}, TensorRole::kVisionProjection, residency,
      source_family, -1, -1, "output,input", -1, false, "compile_excluded");
  if (!status.ok()) return status;
  status = AddBf16(tensors,
                   "model.vision_tower.patch_embedder.input_proj.weight",
                   {kVisionHidden, 768}, TensorRole::kVisionProjection,
                   residency, source_family, -1, -1, "output,input", -1,
                   false, "compile_excluded");
  if (!status.ok()) return status;
  status = AddBf16(
      tensors,
      "model.vision_tower.patch_embedder.position_embedding_table",
      {2, 10240, kVisionHidden}, TensorRole::kVisionEmbedding, residency,
      source_family, -1, -1, "axis,position,hidden", -1, false,
      "compile_excluded");
  if (!status.ok()) return status;
  for (const auto* suffix : {"std_bias", "std_scale"}) {
    status = AddBf16(tensors, "model.vision_tower." + std::string(suffix),
                     {kVisionHidden}, TensorRole::kVisionNorm, residency,
                     source_family, -1, -1, "hidden", -1, false,
                     "compile_excluded");
    if (!status.ok()) return status;
  }

  for (std::uint64_t layer = 0; layer < kVisionLayerCount; ++layer) {
    const auto prefix = "model.vision_tower.encoder.layers." +
                        std::to_string(layer) + ".";
    constexpr std::array<std::pair<std::string_view, TensorRole>, 6> norms = {{
        {"input_layernorm.weight", TensorRole::kVisionNorm},
        {"post_attention_layernorm.weight", TensorRole::kVisionNorm},
        {"post_feedforward_layernorm.weight", TensorRole::kVisionNorm},
        {"pre_feedforward_layernorm.weight", TensorRole::kVisionNorm},
        {"self_attn.k_norm.weight", TensorRole::kVisionNorm},
        {"self_attn.q_norm.weight", TensorRole::kVisionNorm},
    }};
    for (const auto& [suffix, role] : norms) {
      const std::uint64_t extent = suffix.find("self_attn") == 0 ? 72 : 1152;
      status = AddBf16(tensors, prefix + std::string(suffix), {extent}, role,
                       residency, source_family,
                       static_cast<std::int64_t>(layer), -1, "hidden", -1,
                       false, "compile_excluded");
      if (!status.ok()) return status;
    }
    for (const auto* projection : {"k", "o", "q", "v"}) {
      status = AddBf16(
          tensors, prefix + "self_attn." + projection + "_proj.linear.weight",
          {kVisionHidden, kVisionHidden}, TensorRole::kVisionAttention,
          residency, source_family, static_cast<std::int64_t>(layer), -1,
          "output,input", -1, false, "compile_excluded");
      if (!status.ok()) return status;
    }
    status = AddBf16(tensors, prefix + "mlp.down_proj.linear.weight",
                     {kVisionHidden, kVisionIntermediate},
                     TensorRole::kVisionMlp, residency, source_family,
                     static_cast<std::int64_t>(layer), -1, "output,input", -1,
                     false, "compile_excluded");
    if (!status.ok()) return status;
    for (const auto* projection : {"gate", "up"}) {
      status = AddBf16(
          tensors, prefix + "mlp." + projection + "_proj.linear.weight",
          {kVisionIntermediate, kVisionHidden}, TensorRole::kVisionMlp,
          residency, source_family, static_cast<std::int64_t>(layer), -1,
          "output,input", -1, false, "compile_excluded");
      if (!status.ok()) return status;
    }
  }
  return Status::Ok();
}

Status AddLayerControls(std::vector<TensorInfo>* tensors,
                        std::string_view prefix, ResidencyClass residency,
                        std::string_view source_family,
                        std::int64_t layer_index,
                        std::string_view final_layout) {
  constexpr std::array<std::pair<std::string_view, TensorRole>, 7> norms = {{
      {"input_layernorm.weight", TensorRole::kInputLayerNorm},
      {"post_attention_layernorm.weight",
       TensorRole::kPostAttentionLayerNorm},
      {"pre_feedforward_layernorm.weight",
       TensorRole::kPreFeedForwardLayerNorm},
      {"pre_feedforward_layernorm_2.weight",
       TensorRole::kPreFeedForwardLayerNorm2},
      {"post_feedforward_layernorm.weight",
       TensorRole::kPostFeedForwardLayerNorm},
      {"post_feedforward_layernorm_1.weight",
       TensorRole::kPostFeedForwardLayerNorm1},
      {"post_feedforward_layernorm_2.weight",
       TensorRole::kPostFeedForwardLayerNorm2},
  }};
  for (const auto& [suffix, role] : norms) {
    auto status = AddBf16(tensors, std::string(prefix) + std::string(suffix),
                          {kHidden}, role, residency, source_family,
                          layer_index, -1, "hidden", -1, false,
                          final_layout);
    if (!status.ok()) return status;
  }
  return AddBf16(tensors, std::string(prefix) + "layer_scalar", {1},
                  TensorRole::kLayerScalar, residency, source_family,
                  layer_index, -1, "scalar", -1, false, final_layout);
}

Status AddRouter(std::vector<TensorInfo>* tensors, std::string_view prefix,
                 ResidencyClass residency, std::string_view source_family,
                 std::int64_t layer_index,
                 std::string_view final_layout) {
  auto status = AddBf16(tensors, std::string(prefix) + "router.scale",
                        {kHidden}, TensorRole::kRouterScale, residency,
                        source_family, layer_index, -1, "hidden", -1, false,
                        final_layout);
  if (!status.ok()) return status;
  status = AddBf16(tensors, std::string(prefix) + "router.proj.weight",
                   {kExpertCount, kHidden}, TensorRole::kRouterProjection,
                   residency, source_family, layer_index, -1, "expert,input",
                   0, false, final_layout);
  if (!status.ok()) return status;
  return AddBf16(tensors,
                  std::string(prefix) + "router.per_expert_scale",
                  {kExpertCount}, TensorRole::kRouterPerExpertScale, residency,
                  source_family, layer_index, -1, "expert", 0, false,
                  final_layout);
}

TensorRole AttentionRole(std::string_view projection) {
  if (projection == "q") return TensorRole::kAttentionQProjection;
  if (projection == "k") return TensorRole::kAttentionKProjection;
  if (projection == "v") return TensorRole::kAttentionVProjection;
  return TensorRole::kAttentionOProjection;
}

std::vector<std::uint64_t> AttentionShape(std::string_view projection,
                                          bool global) {
  if (projection == "q") return {global ? 8192U : 4096U, kHidden};
  if (projection == "k" || projection == "v") {
    return {global ? 1024U : 2048U, kHidden};
  }
  return {kHidden, global ? 8192U : 4096U};
}

Status AddSourceAttention(std::vector<TensorInfo>* tensors,
                          std::string_view prefix, bool global,
                          ResidencyClass residency,
                          std::string_view source_family,
                          std::int64_t layer_index) {
  const auto norm_extent = global ? 512U : 256U;
  auto status = AddBf16(
      tensors, std::string(prefix) + "self_attn.q_norm.weight", {norm_extent},
      TensorRole::kAttentionQNorm, residency, source_family, layer_index, -1,
      "head_dimension", -1, false, "compiler_source_order_only");
  if (!status.ok()) return status;
  status = AddBf16(
      tensors, std::string(prefix) + "self_attn.k_norm.weight", {norm_extent},
      TensorRole::kAttentionKNorm, residency, source_family, layer_index, -1,
      "head_dimension", -1, false, "compiler_source_order_only");
  if (!status.ok()) return status;
  for (const auto projection : {"q", "k", "o", "v"}) {
    if (global && std::string_view(projection) == "v") continue;
    auto shape = AttentionShape(projection, global);
    status = AddBf16(
        tensors,
        std::string(prefix) + "self_attn." + projection + "_proj.weight",
        shape, AttentionRole(projection), residency, source_family, layer_index,
        -1, "output,input", -1, false, "compiler_source_order_only");
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

Status AddSourceSharedMlp(std::vector<TensorInfo>* tensors,
                          std::string_view prefix, ResidencyClass residency,
                          std::string_view source_family,
                          std::int64_t layer_index) {
  auto status = AddBf16(
      tensors, std::string(prefix) + "mlp.gate_proj.weight",
      {kSharedIntermediate, kHidden}, TensorRole::kSharedMlpGate, residency,
      source_family, layer_index, -1, "output,input", -1, false,
      "compiler_source_order_only");
  if (!status.ok()) return status;
  status = AddBf16(
      tensors, std::string(prefix) + "mlp.up_proj.weight",
      {kSharedIntermediate, kHidden}, TensorRole::kSharedMlpUp, residency,
      source_family, layer_index, -1, "output,input", -1, false,
      "compiler_source_order_only");
  if (!status.ok()) return status;
  return AddBf16(
      tensors, std::string(prefix) + "mlp.down_proj.weight",
      {kHidden, kSharedIntermediate}, TensorRole::kSharedMlpDown, residency,
      source_family, layer_index, -1, "output,input", -1, false,
      "compiler_source_order_only");
}

Status AddSourceExperts(std::vector<TensorInfo>* tensors,
                        std::string_view prefix, ResidencyClass residency,
                        std::string_view source_family,
                        std::int64_t layer_index) {
  auto status = AddBf16(
      tensors, std::string(prefix) + "experts.gate_up_proj",
      {kExpertCount, 2U * kExpertIntermediate, kHidden},
      TensorRole::kRoutedExpertGateUp, residency, source_family, layer_index,
      -1, "expert,gate_then_up,input", 0, false,
      "compiler_source_order_only");
  if (!status.ok()) return status;
  return AddBf16(
      tensors, std::string(prefix) + "experts.down_proj",
      {kExpertCount, kHidden, kExpertIntermediate},
      TensorRole::kRoutedExpertDown, residency, source_family, layer_index, -1,
      "expert,output,input", 0, false, "compiler_source_order_only");
}

Status AddExternalFp8Projection(std::vector<TensorInfo>* tensors,
                                std::string_view module,
                                const std::vector<std::uint64_t>& shape,
                                TensorRole role, std::int64_t layer_index) {
  ContractMetadata weight{.role = role,
                          .residency = ResidencyClass::kExternalReferenceText,
                          .quantization_class = "FP8_WEIGHT_E4M3",
                          .quantization_producer = kExternalProducer,
                          .local_scale_dtype = "BF16",
                          .local_scale_vector_size = shape[1],
                          .activation_scale_role =
                              "dynamic_per_token_dequant_multiplier",
                          .final_gpu_layout = "external_source_order",
                          .logical_axis_order = "output,input",
                          .layer_index = layer_index};
  auto status = AddTensor(tensors, std::string(module) + ".weight", "F8_E4M3",
                          shape, shape, kExternalFamily, weight);
  if (!status.ok()) return status;
  ContractMetadata scale = weight;
  scale.quantization_component = "weight_channel_scale";
  scale.quantization_class = "FP8_WEIGHT_SCALE";
  scale.logical_dtype = "BF16";
  const auto scale_name = std::string(module) + ".weight_scale";
  status = AddTensor(tensors, scale_name, "BF16", {shape[0], 1}, {shape[0], 1},
                     kExternalFamily, scale);
  if (!status.ok()) return status;
  tensors->at(tensors->size() - 2).local_scale_tensor = scale_name;
  return Status::Ok();
}

Status AddExternalNvfp4Projection(
    std::vector<TensorInfo>* tensors, std::string_view module,
    const std::vector<std::uint64_t>& logical_shape, TensorRole role,
    std::int64_t layer_index, std::int64_t expert_index) {
  if (logical_shape.size() != 2 || logical_shape[1] % 16U != 0) {
    return Status(StatusCode::kDataLoss,
                  "invalid external NVFP4 logical shape: " +
                      std::string(module));
  }
  const std::vector<std::uint64_t> packed_shape = {logical_shape[0],
                                                   logical_shape[1] / 2U};
  const std::vector<std::uint64_t> scale_shape = {logical_shape[0],
                                                  logical_shape[1] / 16U};
  ContractMetadata base{
      .role = role,
      .residency = ResidencyClass::kExternalReferenceText,
      .quantization_class = "NVFP4_PACKED",
      .quantization_producer = kExternalProducer,
      .local_scale_dtype = "F8_E4M3",
      .local_scale_vector_size = 16,
      .global_scale_role = "divisor",
      .activation_scale_role = "divisor",
      .final_gpu_layout = "external_source_order",
      .logical_axis_order = "output,input",
      .layer_index = layer_index,
      .expert_index = expert_index};
  const auto packed_name = std::string(module) + ".weight_packed";
  const auto local_name = std::string(module) + ".weight_scale";
  const auto global_name = std::string(module) + ".weight_global_scale";
  const auto input_name = std::string(module) + ".input_global_scale";
  auto status = AddTensor(tensors, packed_name, "U8", packed_shape,
                          logical_shape, kExternalFamily, base);
  if (!status.ok()) return status;
  auto& packed = tensors->back();
  packed.local_scale_tensor = local_name;
  packed.global_scale_tensor = global_name;
  packed.input_scale_tensor = input_name;
  ContractMetadata local = base;
  local.quantization_component = "weight_local_scale";
  local.quantization_class = "NVFP4_LOCAL_SCALE_E4M3";
  status = AddTensor(tensors, local_name, "F8_E4M3", scale_shape, scale_shape,
                     kExternalFamily, local);
  if (!status.ok()) return status;
  ContractMetadata global = base;
  global.quantization_component = "weight_global_scale";
  global.quantization_class = "NVFP4_GLOBAL_SCALE";
  status = AddTensor(tensors, global_name, "F32", {1}, {1}, kExternalFamily,
                     global);
  if (!status.ok()) return status;
  ContractMetadata input = base;
  input.quantization_component = "activation_global_scale";
  input.quantization_class = "NVFP4_INPUT_SCALE";
  return AddTensor(tensors, input_name, "F32", {1}, {1}, kExternalFamily,
                   input);
}

Status AddExternalAttention(std::vector<TensorInfo>* tensors,
                            std::string_view prefix, bool global,
                            std::int64_t layer_index) {
  const auto norm_extent = global ? 512U : 256U;
  auto status = AddBf16(
      tensors, std::string(prefix) + "self_attn.q_norm.weight", {norm_extent},
      TensorRole::kAttentionQNorm, ResidencyClass::kExternalReferenceText,
      kExternalFamily, layer_index, -1, "head_dimension", -1, false,
      "external_source_order");
  if (!status.ok()) return status;
  status = AddBf16(
      tensors, std::string(prefix) + "self_attn.k_norm.weight", {norm_extent},
      TensorRole::kAttentionKNorm, ResidencyClass::kExternalReferenceText,
      kExternalFamily, layer_index, -1, "head_dimension", -1, false,
      "external_source_order");
  if (!status.ok()) return status;
  status = AddBf16(
      tensors, std::string(prefix) + "self_attn.k_scale", {1},
      TensorRole::kAttentionKScale, ResidencyClass::kExternalReferenceText,
      kExternalFamily, layer_index, -1, "scalar", -1, false,
      "external_source_order");
  if (!status.ok()) return status;
  status = AddBf16(
      tensors, std::string(prefix) + "self_attn.v_scale", {1},
      TensorRole::kAttentionVScale, ResidencyClass::kExternalReferenceText,
      kExternalFamily, layer_index, -1, "scalar", -1, false,
      "external_source_order");
  if (!status.ok()) return status;
  for (const auto projection : {"q", "k", "o", "v"}) {
    if (global && std::string_view(projection) == "v") continue;
    const auto module = std::string(prefix) + "self_attn." + projection +
                        "_proj";
    status = AddExternalFp8Projection(tensors, module,
                                      AttentionShape(projection, global),
                                      AttentionRole(projection), layer_index);
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

Status AddExternalSharedMlp(std::vector<TensorInfo>* tensors,
                            std::string_view prefix,
                            std::int64_t layer_index) {
  auto status = AddExternalNvfp4Projection(
      tensors, std::string(prefix) + "mlp.gate_proj",
      {kSharedIntermediate, kHidden}, TensorRole::kSharedMlpGate, layer_index,
      -1);
  if (!status.ok()) return status;
  status = AddExternalNvfp4Projection(
      tensors, std::string(prefix) + "mlp.up_proj",
      {kSharedIntermediate, kHidden}, TensorRole::kSharedMlpUp, layer_index,
      -1);
  if (!status.ok()) return status;
  return AddExternalNvfp4Projection(
      tensors, std::string(prefix) + "mlp.down_proj",
      {kHidden, kSharedIntermediate}, TensorRole::kSharedMlpDown, layer_index,
      -1);
}

Status AddExternalExperts(std::vector<TensorInfo>* tensors,
                          std::string_view prefix,
                          std::int64_t layer_index) {
  for (std::uint64_t expert = 0; expert < kExpertCount; ++expert) {
    const auto expert_prefix = std::string(prefix) + "experts." +
                               std::to_string(expert) + ".";
    auto status = AddExternalNvfp4Projection(
        tensors, expert_prefix + "gate_proj", {kExpertIntermediate, kHidden},
        TensorRole::kRoutedExpertGate, layer_index,
        static_cast<std::int64_t>(expert));
    if (!status.ok()) return status;
    status = AddExternalNvfp4Projection(
        tensors, expert_prefix + "up_proj", {kExpertIntermediate, kHidden},
        TensorRole::kRoutedExpertUp, layer_index,
        static_cast<std::int64_t>(expert));
    if (!status.ok()) return status;
    status = AddExternalNvfp4Projection(
        tensors, expert_prefix + "down_proj", {kHidden, kExpertIntermediate},
        TensorRole::kRoutedExpertDown, layer_index,
        static_cast<std::int64_t>(expert));
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

Status AddCompiledFp8Projection(std::vector<TensorInfo>* tensors,
                                std::string_view module,
                                const std::vector<std::uint64_t>& shape,
                                TensorRole role, std::int64_t layer_index) {
  ContractMetadata weight{.role = role,
                          .residency = ResidencyClass::kImmutableDeviceText,
                          .quantization_class = "FP8_WEIGHT_E4M3",
                          .quantization_producer = kCompiledProducer,
                          .local_scale_dtype = "BF16",
                          .local_scale_vector_size = shape[1],
                          .activation_scale_role =
                              "dynamic_per_token_dequant_multiplier",
                          .final_gpu_layout = "source_nk_fp8",
                          .logical_axis_order = "output,input",
                          .layer_index = layer_index};
  auto status = AddTensor(tensors, std::string(module) + ".weight", "F8_E4M3",
                          shape, shape, kCompiledFamily, weight);
  if (!status.ok()) return status;
  ContractMetadata scale = weight;
  scale.quantization_component = "weight_channel_scale";
  scale.quantization_class = "FP8_WEIGHT_SCALE";
  const auto scale_name = std::string(module) + ".weight_scale";
  status = AddTensor(tensors, scale_name, "BF16", {shape[0], 1}, {shape[0], 1},
                     kCompiledFamily, scale);
  if (!status.ok()) return status;
  tensors->at(tensors->size() - 2).local_scale_tensor = scale_name;
  return Status::Ok();
}

Status AddCompiledNvfp4Projection(
    std::vector<TensorInfo>* tensors, std::string_view module,
    const std::vector<std::uint64_t>& logical_shape, TensorRole role,
    std::int64_t layer_index, std::string_view axis_order,
    std::int64_t expert_axis = -1) {
  if (logical_shape.empty() || logical_shape.back() % 16U != 0) {
    return Status(StatusCode::kDataLoss,
                  "invalid compiled NVFP4 logical shape: " +
                      std::string(module));
  }
  auto packed_shape = logical_shape;
  packed_shape.back() /= 2U;
  auto scale_shape = logical_shape;
  scale_shape.back() /= 16U;
  ContractMetadata base{
      .role = role,
      .residency = ResidencyClass::kImmutableDeviceText,
      .quantization_class = "NVFP4_PACKED",
      .quantization_producer = kCompiledProducer,
      .local_scale_dtype = "F8_E4M3",
      .local_scale_vector_size = 16,
      .global_scale_role = "divisor",
      .activation_scale_role = "divisor",
      .final_gpu_layout = expert_axis == 0 ? "expert_major_sm120_row8_k64"
                                           : "sm120_row8_k64",
      .logical_axis_order = axis_order,
      .layer_index = layer_index,
      .expert_axis = expert_axis};
  const auto packed_name = std::string(module) + ".weight_packed";
  const auto local_name = std::string(module) + ".weight_scale";
  const auto global_name = std::string(module) + ".weight_global_scale";
  const auto input_name = std::string(module) + ".input_global_scale";
  auto status = AddTensor(tensors, packed_name, "U8", packed_shape,
                          logical_shape, kCompiledFamily, base);
  if (!status.ok()) return status;
  auto& packed = tensors->back();
  packed.local_scale_tensor = local_name;
  packed.global_scale_tensor = global_name;
  packed.input_scale_tensor = input_name;
  ContractMetadata local = base;
  local.quantization_component = "weight_local_scale";
  local.quantization_class = "NVFP4_LOCAL_SCALE_E4M3";
  status = AddTensor(tensors, local_name, "F8_E4M3", scale_shape, scale_shape,
                     kCompiledFamily, local);
  if (!status.ok()) return status;
  ContractMetadata global = base;
  global.quantization_component = "weight_global_scale";
  global.quantization_class = "NVFP4_GLOBAL_SCALE";
  status = AddTensor(tensors, global_name, "F32", {1}, {1}, kCompiledFamily,
                     global);
  if (!status.ok()) return status;
  ContractMetadata input = base;
  input.quantization_component = "activation_global_scale";
  input.quantization_class = "NVFP4_INPUT_SCALE";
  return AddTensor(tensors, input_name, "F32", {1}, {1}, kCompiledFamily,
                   input);
}

Status AddCompiledHead(std::vector<TensorInfo>* tensors,
                       Gemma4Moe26BHeadFormat format) {
  constexpr auto role = TensorRole::kTiedEmbeddingAndOutput;
  constexpr auto residency = ResidencyClass::kImmutableDeviceText;
  if (format == Gemma4Moe26BHeadFormat::kQ4_0) {
    ContractMetadata metadata{
        .role = role,
        .residency = residency,
        .quantization_class = "Q4_0_BLOCK32",
        .quantization_producer = kCompiledProducer,
        .local_scale_dtype = "F16",
        .local_scale_vector_size = 32,
        .final_gpu_layout = "q4_0_row_blocks32",
        .logical_axis_order = "vocabulary,hidden",
        .aliased = true};
    return AddTensor(
        tensors, "model.language_model.embed_tokens.weight_q4_0", "U8",
        {kVocabulary, 1584}, {kVocabulary, kHidden}, kCompiledFamily,
        metadata);
  }
  auto status = AddCompiledNvfp4Projection(
      tensors, "model.language_model.embed_tokens",
      {kVocabulary, kHidden}, role, -1, "vocabulary,hidden");
  if (!status.ok()) return status;
  for (std::size_t index = tensors->size() - 4; index < tensors->size();
       ++index) {
    tensors->at(index).aliased = true;
  }
  return Status::Ok();
}

Status AddCompiledAttention(std::vector<TensorInfo>* tensors,
                            std::string_view prefix, bool global,
                            std::int64_t layer_index) {
  const auto norm_extent = global ? 512U : 256U;
  auto status = AddBf16(
      tensors, std::string(prefix) + "self_attn.q_norm.weight", {norm_extent},
      TensorRole::kAttentionQNorm, ResidencyClass::kImmutableDeviceText,
      kCompiledFamily, layer_index, -1, "head_dimension", -1, false,
      "source_bf16");
  if (!status.ok()) return status;
  status = AddBf16(
      tensors, std::string(prefix) + "self_attn.k_norm.weight", {norm_extent},
      TensorRole::kAttentionKNorm, ResidencyClass::kImmutableDeviceText,
      kCompiledFamily, layer_index, -1, "head_dimension", -1, false,
      "source_bf16");
  if (!status.ok()) return status;
  status = AddBf16(
      tensors, std::string(prefix) + "self_attn.k_scale", {1},
      TensorRole::kAttentionKScale, ResidencyClass::kImmutableDeviceText,
      kCompiledFamily, layer_index, -1, "scalar", -1, false,
      "source_bf16");
  if (!status.ok()) return status;
  status = AddBf16(
      tensors, std::string(prefix) + "self_attn.v_scale", {1},
      TensorRole::kAttentionVScale, ResidencyClass::kImmutableDeviceText,
      kCompiledFamily, layer_index, -1, "scalar", -1, false,
      "source_bf16");
  if (!status.ok()) return status;
  for (const auto projection : {"q", "k", "o", "v"}) {
    if (global && std::string_view(projection) == "v") continue;
    const auto module = std::string(prefix) + "self_attn." + projection +
                        "_proj";
    status = AddCompiledFp8Projection(tensors, module,
                                      AttentionShape(projection, global),
                                      AttentionRole(projection), layer_index);
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

Status AddCompiledSharedMlp(std::vector<TensorInfo>* tensors,
                            std::string_view prefix,
                            std::int64_t layer_index) {
  auto status = AddCompiledNvfp4Projection(
      tensors, std::string(prefix) + "mlp.gate_proj",
      {kSharedIntermediate, kHidden}, TensorRole::kSharedMlpGate, layer_index,
      "output,input");
  if (!status.ok()) return status;
  status = AddCompiledNvfp4Projection(
      tensors, std::string(prefix) + "mlp.up_proj",
      {kSharedIntermediate, kHidden}, TensorRole::kSharedMlpUp, layer_index,
      "output,input");
  if (!status.ok()) return status;
  return AddCompiledNvfp4Projection(
      tensors, std::string(prefix) + "mlp.down_proj",
      {kHidden, kSharedIntermediate}, TensorRole::kSharedMlpDown, layer_index,
      "output,input");
}

Status AddCompiledExperts(std::vector<TensorInfo>* tensors,
                          std::string_view prefix,
                          std::int64_t layer_index) {
  auto status = AddCompiledNvfp4Projection(
      tensors, std::string(prefix) + "experts.gate_up_proj",
      {kExpertCount, 2U * kExpertIntermediate, kHidden},
      TensorRole::kRoutedExpertGateUp, layer_index,
      "expert,gate_then_up,input", 0);
  if (!status.ok()) return status;
  return AddCompiledNvfp4Projection(
      tensors, std::string(prefix) + "experts.down_proj",
      {kExpertCount, kHidden, kExpertIntermediate},
      TensorRole::kRoutedExpertDown, layer_index, "expert,output,input", 0);
}

void SortContract(std::vector<TensorInfo>* tensors) {
  std::sort(tensors->begin(), tensors->end(),
            [](const TensorInfo& left, const TensorInfo& right) {
              return left.name < right.name;
            });
}

bool SameQuantizationRule(const QuantizationRule& rule,
                          std::string_view format,
                          std::uint32_t weight_bits,
                          std::uint32_t activation_bits,
                          std::uint32_t group_size,
                          std::string_view scale_dtype,
                          std::string_view weight_strategy,
                          std::string_view activation_strategy,
                          bool dynamic, bool dynamic_local,
                          const std::vector<std::string>& targets) {
  return rule.format == format && rule.weight_bits == weight_bits &&
         rule.activation_bits == activation_bits &&
         rule.group_size == group_size && rule.scale_dtype == scale_dtype &&
         rule.weight_strategy == weight_strategy &&
         rule.activation_strategy == activation_strategy &&
         rule.activation_dynamic == dynamic &&
         rule.activation_dynamic_local == dynamic_local &&
         rule.regex_targets == targets;
}

bool IsExactExternalUnslothConfig(const ModelConfig& config) {
  if (config.quant_method != "compressed-tensors" ||
      config.quant_format != "mixed-precision" ||
      config.quantization_status != "compressed" ||
      config.quantization_version != "0.17.2.a20260707" ||
      config.quantization_rules.size() != 2) {
    return false;
  }
  bool fp8 = false;
  bool nvfp4 = false;
  for (const auto& rule : config.quantization_rules) {
    fp8 = fp8 || SameQuantizationRule(
                     rule, "float-quantized", 8, 8, 0, "", "channel",
                     "token", true, false,
                     {".*self_attn\\.(q|k|v|o)_proj$"});
    nvfp4 = nvfp4 || SameQuantizationRule(
                         rule, "nvfp4-pack-quantized", 4, 4, 16,
                         "torch.float8_e4m3fn", "tensor_group",
                         "tensor_group", false, true,
                         {".*\\.experts\\.\\d+\\.(gate|up|down)_proj$",
                          ".*language_model.*\\.mlp\\.(gate|up|down)_proj$"});
  }
  return fp8 && nvfp4;
}

void CopyContractMetadata(const TensorInfo& expected, TensorInfo* actual) {
  actual->logical_shape = expected.logical_shape;
  actual->logical_dtype = expected.logical_dtype;
  actual->quantization_class = expected.quantization_class;
  actual->expected_role = expected.expected_role;
  actual->tensor_role = expected.tensor_role;
  actual->residency_class = expected.residency_class;
  actual->source_family = expected.source_family;
  actual->quantization_component = expected.quantization_component;
  actual->quantization_producer = expected.quantization_producer;
  actual->local_scale_dtype = expected.local_scale_dtype;
  actual->local_scale_vector_size = expected.local_scale_vector_size;
  actual->global_scale_role = expected.global_scale_role;
  actual->activation_scale_role = expected.activation_scale_role;
  actual->final_gpu_layout = expected.final_gpu_layout;
  actual->logical_axis_order = expected.logical_axis_order;
  actual->layer_index = expected.layer_index;
  actual->expert_index = expected.expert_index;
  actual->expert_axis = expected.expert_axis;
  actual->local_scale_tensor = expected.local_scale_tensor;
  actual->global_scale_tensor = expected.global_scale_tensor;
  actual->input_scale_tensor = expected.input_scale_tensor;
  actual->loaded_in_text_only_mode = expected.loaded_in_text_only_mode;
  actual->aliased = expected.aliased;
  if (!expected.layout.empty()) actual->layout = expected.layout;
}

Status ValidatePhysicalContract(std::vector<TensorInfo>* actual,
                                const std::vector<TensorInfo>& expected,
                                bool compare_semantic_metadata) {
  if (actual == nullptr) {
    return Status(StatusCode::kInvalidArgument,
                  "Gemma 4 26B tensor inventory is null");
  }
  std::map<std::string, TensorInfo*, std::less<>> actual_by_name;
  for (auto& tensor : *actual) {
    if (!actual_by_name.emplace(tensor.name, &tensor).second) {
      return Status(StatusCode::kDataLoss,
                    "duplicate Gemma 4 26B tensor: " + tensor.name);
    }
  }
  std::map<std::string, const TensorInfo*, std::less<>> expected_by_name;
  for (const auto& tensor : expected) {
    if (!expected_by_name.emplace(tensor.name, &tensor).second) {
      return Status(StatusCode::kInternal,
                    "duplicate generated Gemma 4 26B contract tensor: " +
                        tensor.name);
    }
  }
  for (const auto& [name, expected_tensor] : expected_by_name) {
    const auto found = actual_by_name.find(name);
    if (found == actual_by_name.end()) {
      return Status(StatusCode::kDataLoss,
                    "required Gemma 4 26B tensor is missing: " + name);
    }
    const auto& tensor = *found->second;
    if (tensor.storage_dtype != expected_tensor->storage_dtype ||
        tensor.shape != expected_tensor->shape ||
        tensor.byte_length != expected_tensor->byte_length) {
      return Status(StatusCode::kDataLoss,
                    "Gemma 4 26B tensor schema mismatch: " + name);
    }
    if (compare_semantic_metadata &&
        (tensor.logical_shape != expected_tensor->logical_shape ||
         tensor.logical_dtype != expected_tensor->logical_dtype ||
         tensor.quantization_class != expected_tensor->quantization_class ||
         tensor.expected_role != expected_tensor->expected_role ||
         tensor.tensor_role != expected_tensor->tensor_role ||
         tensor.residency_class != expected_tensor->residency_class ||
         tensor.source_family != expected_tensor->source_family ||
         tensor.quantization_component !=
             expected_tensor->quantization_component ||
         tensor.quantization_producer !=
             expected_tensor->quantization_producer ||
         tensor.local_scale_dtype != expected_tensor->local_scale_dtype ||
         tensor.local_scale_vector_size !=
             expected_tensor->local_scale_vector_size ||
         tensor.global_scale_role != expected_tensor->global_scale_role ||
         tensor.activation_scale_role !=
             expected_tensor->activation_scale_role ||
         tensor.final_gpu_layout != expected_tensor->final_gpu_layout ||
         tensor.logical_axis_order != expected_tensor->logical_axis_order ||
         tensor.layer_index != expected_tensor->layer_index ||
         tensor.expert_index != expected_tensor->expert_index ||
         tensor.expert_axis != expected_tensor->expert_axis ||
         tensor.local_scale_tensor != expected_tensor->local_scale_tensor ||
         tensor.global_scale_tensor != expected_tensor->global_scale_tensor ||
         tensor.input_scale_tensor != expected_tensor->input_scale_tensor ||
         tensor.layout != expected_tensor->layout ||
         tensor.loaded_in_text_only_mode !=
             expected_tensor->loaded_in_text_only_mode ||
         tensor.aliased != expected_tensor->aliased)) {
      return Status(StatusCode::kDataLoss,
                    "Gemma 4 26B tensor semantic contract mismatch: " + name);
    }
  }
  for (const auto& [name, unused] : actual_by_name) {
    (void)unused;
    if (!expected_by_name.contains(name)) {
      return Status(StatusCode::kUnsupported,
                    "unexpected Gemma 4 26B tensor: " + name);
    }
  }
  if (actual_by_name.size() != expected_by_name.size()) {
    return Status(StatusCode::kDataLoss,
                  "Gemma 4 26B tensor count does not match its exact contract");
  }
  if (!compare_semantic_metadata) {
    for (const auto& [name, expected_tensor] : expected_by_name) {
      CopyContractMetadata(*expected_tensor, actual_by_name.at(name));
    }
  }
  return Status::Ok();
}

}  // namespace

std::string_view TensorRoleName(TensorRole role) {
  switch (role) {
    case TensorRole::kTiedEmbeddingAndOutput:
      return "tied_embedding_and_output";
    case TensorRole::kFinalNorm:
      return "final_norm";
    case TensorRole::kAttentionQProjection:
      return "attention_q_projection";
    case TensorRole::kAttentionKProjection:
      return "attention_k_projection";
    case TensorRole::kAttentionVProjection:
      return "attention_v_projection";
    case TensorRole::kAttentionOProjection:
      return "attention_o_projection";
    case TensorRole::kAttentionQNorm:
      return "attention_q_norm";
    case TensorRole::kAttentionKNorm:
      return "attention_k_norm";
    case TensorRole::kAttentionKScale:
      return "attention_k_cache_scale";
    case TensorRole::kAttentionVScale:
      return "attention_v_cache_scale";
    case TensorRole::kSharedMlpGate:
      return "shared_mlp_gate";
    case TensorRole::kSharedMlpUp:
      return "shared_mlp_up";
    case TensorRole::kSharedMlpDown:
      return "shared_mlp_down";
    case TensorRole::kRouterScale:
      return "router_norm_scale";
    case TensorRole::kRouterProjection:
      return "router_projection";
    case TensorRole::kRouterPerExpertScale:
      return "router_per_expert_scale";
    case TensorRole::kRoutedExpertGateUp:
      return "routed_expert_gate_up";
    case TensorRole::kRoutedExpertGate:
      return "routed_expert_gate";
    case TensorRole::kRoutedExpertUp:
      return "routed_expert_up";
    case TensorRole::kRoutedExpertDown:
      return "routed_expert_down";
    case TensorRole::kInputLayerNorm:
      return "input_layer_norm";
    case TensorRole::kPostAttentionLayerNorm:
      return "post_attention_layer_norm";
    case TensorRole::kPreFeedForwardLayerNorm:
      return "pre_feed_forward_layer_norm";
    case TensorRole::kPreFeedForwardLayerNorm2:
      return "pre_feed_forward_layer_norm_2";
    case TensorRole::kPostFeedForwardLayerNorm:
      return "post_feed_forward_layer_norm";
    case TensorRole::kPostFeedForwardLayerNorm1:
      return "post_feed_forward_layer_norm_1";
    case TensorRole::kPostFeedForwardLayerNorm2:
      return "post_feed_forward_layer_norm_2";
    case TensorRole::kLayerScalar:
      return "layer_scalar";
    case TensorRole::kVisionProjection:
      return "vision_projection";
    case TensorRole::kVisionEmbedding:
      return "vision_embedding";
    case TensorRole::kVisionAttention:
      return "vision_attention";
    case TensorRole::kVisionMlp:
      return "vision_mlp";
    case TensorRole::kVisionNorm:
      return "vision_norm";
  }
  return "unknown";
}

std::string_view ResidencyClassName(ResidencyClass residency) {
  switch (residency) {
    case ResidencyClass::kCompilerSourceText:
      return "compiler_source_text";
    case ResidencyClass::kExternalReferenceText:
      return "external_reference_text";
    case ResidencyClass::kImmutableDeviceText:
      return "immutable_device_text";
    case ResidencyClass::kCompileExcludedVision:
      return "compile_excluded_vision";
  }
  return "unknown";
}

std::string_view Gemma4Moe26BInventoryProfileName(
    Gemma4Moe26BInventoryProfile profile) {
  switch (profile) {
    case Gemma4Moe26BInventoryProfile::kSourceBf16:
      return "source_bf16";
    case Gemma4Moe26BInventoryProfile::kExternalUnslothNvfp4:
      return "external_unsloth_nvfp4";
  }
  return "unknown";
}

std::string_view Gemma4Moe26BHeadFormatName(
    Gemma4Moe26BHeadFormat format) {
  switch (format) {
    case Gemma4Moe26BHeadFormat::kQ4_0:
      return "q4_0";
    case Gemma4Moe26BHeadFormat::kNvfp4:
      return "nvfp4";
  }
  return "unknown";
}

Result<std::vector<TensorInfo>> BuildGemma4Moe26BSourceBf16Contract() {
  std::vector<TensorInfo> tensors;
  tensors.reserve(1013);
  auto status = AddBf16(
      &tensors, "model.language_model.embed_tokens.weight",
      {kVocabulary, kHidden}, TensorRole::kTiedEmbeddingAndOutput,
      ResidencyClass::kCompilerSourceText, kSourceFamily, -1, -1,
      "vocabulary,hidden", -1, true, "compiler_source_order_only");
  if (!status.ok()) return status;
  status = AddBf16(&tensors, "model.language_model.norm.weight", {kHidden},
                   TensorRole::kFinalNorm,
                   ResidencyClass::kCompilerSourceText, kSourceFamily, -1, -1,
                   "hidden", -1, false, "compiler_source_order_only");
  if (!status.ok()) return status;
  for (std::uint64_t layer = 0; layer < kLayerCount; ++layer) {
    const auto prefix = "model.language_model.layers." +
                        std::to_string(layer) + ".";
    const auto layer_index = static_cast<std::int64_t>(layer);
    const bool global = layer % 6U == 5U;
    status = AddLayerControls(&tensors, prefix,
                              ResidencyClass::kCompilerSourceText,
                              kSourceFamily, layer_index,
                              "compiler_source_order_only");
    if (!status.ok()) return status;
    status = AddRouter(&tensors, prefix, ResidencyClass::kCompilerSourceText,
                       kSourceFamily, layer_index,
                       "compiler_source_order_only");
    if (!status.ok()) return status;
    status = AddSourceAttention(&tensors, prefix, global,
                                ResidencyClass::kCompilerSourceText,
                                kSourceFamily, layer_index);
    if (!status.ok()) return status;
    status = AddSourceSharedMlp(&tensors, prefix,
                                ResidencyClass::kCompilerSourceText,
                                kSourceFamily, layer_index);
    if (!status.ok()) return status;
    status = AddSourceExperts(&tensors, prefix,
                              ResidencyClass::kCompilerSourceText,
                              kSourceFamily, layer_index);
    if (!status.ok()) return status;
  }
  status = AddVisionContract(&tensors, kSourceFamily);
  if (!status.ok()) return status;
  SortContract(&tensors);
  if (tensors.size() != 1013) {
    return Status(StatusCode::kInternal,
                  "generated source BF16 tensor count is not 1013");
  }
  return tensors;
}

Result<std::vector<TensorInfo>>
BuildGemma4Moe26BExternalUnslothNvfp4Contract() {
  std::vector<TensorInfo> tensors;
  tensors.reserve(47478);
  auto status = AddBf16(
      &tensors, "model.language_model.embed_tokens.weight",
      {kVocabulary, kHidden}, TensorRole::kTiedEmbeddingAndOutput,
      ResidencyClass::kExternalReferenceText, kExternalFamily, -1, -1,
      "vocabulary,hidden", -1, true, "external_source_order");
  if (!status.ok()) return status;
  status = AddBf16(&tensors, "model.language_model.norm.weight", {kHidden},
                   TensorRole::kFinalNorm,
                   ResidencyClass::kExternalReferenceText, kExternalFamily,
                   -1, -1, "hidden", -1, false, "external_source_order");
  if (!status.ok()) return status;
  for (std::uint64_t layer = 0; layer < kLayerCount; ++layer) {
    const auto prefix = "model.language_model.layers." +
                        std::to_string(layer) + ".";
    const auto layer_index = static_cast<std::int64_t>(layer);
    const bool global = layer % 6U == 5U;
    status = AddLayerControls(&tensors, prefix,
                              ResidencyClass::kExternalReferenceText,
                              kExternalFamily, layer_index,
                              "external_source_order");
    if (!status.ok()) return status;
    status = AddRouter(&tensors, prefix,
                       ResidencyClass::kExternalReferenceText, kExternalFamily,
                       layer_index, "external_source_order");
    if (!status.ok()) return status;
    status = AddExternalAttention(&tensors, prefix, global, layer_index);
    if (!status.ok()) return status;
    status = AddExternalSharedMlp(&tensors, prefix, layer_index);
    if (!status.ok()) return status;
    status = AddExternalExperts(&tensors, prefix, layer_index);
    if (!status.ok()) return status;
  }
  status = AddVisionContract(&tensors, kExternalFamily);
  if (!status.ok()) return status;
  SortContract(&tensors);
  if (tensors.size() != 47478) {
    return Status(StatusCode::kInternal,
                  "generated external NVFP4 tensor count is not 47478");
  }
  return tensors;
}

Result<std::vector<TensorInfo>> BuildGemma4Moe26BCompiledHybridContract(
    Gemma4Moe26BHeadFormat head_format) {
  std::vector<TensorInfo> tensors;
  tensors.reserve(1285);
  auto status = AddCompiledHead(&tensors, head_format);
  if (!status.ok()) return status;
  status = AddBf16(&tensors, "model.language_model.norm.weight", {kHidden},
                   TensorRole::kFinalNorm,
                   ResidencyClass::kImmutableDeviceText, kCompiledFamily, -1,
                   -1, "hidden", -1, false, "source_bf16");
  if (!status.ok()) return status;
  for (std::uint64_t layer = 0; layer < kLayerCount; ++layer) {
    const auto prefix = "model.language_model.layers." +
                        std::to_string(layer) + ".";
    const auto layer_index = static_cast<std::int64_t>(layer);
    const bool global = layer % 6U == 5U;
    status = AddLayerControls(&tensors, prefix,
                              ResidencyClass::kImmutableDeviceText,
                              kCompiledFamily, layer_index, "source_bf16");
    if (!status.ok()) return status;
    status = AddRouter(&tensors, prefix,
                       ResidencyClass::kImmutableDeviceText, kCompiledFamily,
                       layer_index, "source_bf16");
    if (!status.ok()) return status;
    status = AddCompiledAttention(&tensors, prefix, global, layer_index);
    if (!status.ok()) return status;
    status = AddCompiledSharedMlp(&tensors, prefix, layer_index);
    if (!status.ok()) return status;
    status = AddCompiledExperts(&tensors, prefix, layer_index);
    if (!status.ok()) return status;
  }
  SortContract(&tensors);
  const std::size_t expected_count =
      head_format == Gemma4Moe26BHeadFormat::kQ4_0 ? 1282 : 1285;
  if (tensors.size() != expected_count) {
    return Status(StatusCode::kInternal,
                  "generated compiled hybrid tensor count is incorrect");
  }
  return tensors;
}

Result<Gemma4Moe26BInventoryProfile>
ValidateAndAnnotateGemma4Moe26BInventory(const ModelConfig& config,
                                         std::vector<TensorInfo>* tensors) {
  if (!IsGemma4Moe26BModel(config)) {
    return Status(StatusCode::kInvalidArgument,
                  "Gemma 4 26B inventory validator received another model variant");
  }
  if (config.quant_method.empty() && config.quant_format.empty() &&
      config.quantization_status.empty() &&
      config.quantization_version.empty() &&
      config.quantization_rules.empty()) {
    auto expected = BuildGemma4Moe26BSourceBf16Contract();
    if (!expected.ok()) return expected.status();
    auto status = ValidatePhysicalContract(tensors, expected.value(), false);
    if (!status.ok()) return status;
    return Gemma4Moe26BInventoryProfile::kSourceBf16;
  }
  if (!IsExactExternalUnslothConfig(config)) {
    return Status(
        StatusCode::kUnsupported,
        "unsupported Gemma 4 26B quantization producer/scale contract");
  }
  auto expected = BuildGemma4Moe26BExternalUnslothNvfp4Contract();
  if (!expected.ok()) return expected.status();
  auto status = ValidatePhysicalContract(tensors, expected.value(), false);
  if (!status.ok()) return status;
  return Gemma4Moe26BInventoryProfile::kExternalUnslothNvfp4;
}

Status ValidateGemma4Moe26BCompiledHybridInventory(
    std::span<const TensorInfo> tensors, Gemma4Moe26BHeadFormat head_format) {
  auto expected = BuildGemma4Moe26BCompiledHybridContract(head_format);
  if (!expected.ok()) return expected.status();
  std::vector<TensorInfo> actual(tensors.begin(), tensors.end());
  return ValidatePhysicalContract(&actual, expected.value(), true);
}

Result<std::uint64_t> Gemma4Moe26BAlignedArenaBytes(
    std::span<const TensorInfo> tensors, std::uint64_t alignment) {
  if (alignment == 0 || (alignment & (alignment - 1U)) != 0) {
    return Status(StatusCode::kInvalidArgument,
                  "Gemma 4 26B arena alignment must be a nonzero power of two");
  }
  std::uint64_t total = 0;
  for (const auto& tensor : tensors) {
    auto padded = CheckedAdd(tensor.byte_length, alignment - 1U,
                             "aligned tensor byte count");
    if (!padded.ok()) return padded.status();
    const auto aligned = padded.value() & ~(alignment - 1U);
    auto next = CheckedAdd(total, aligned, "aligned immutable arena bytes");
    if (!next.ok()) return next.status();
    total = next.value();
  }
  return total;
}

Result<std::uint64_t> Gemma4Moe26B32KFp8KvBytes() {
  constexpr std::uint64_t context = 32768;
  auto local = CheckedMultiply(25, 1024, "local layer/window count");
  if (!local.ok()) return local.status();
  local = CheckedMultiply(local.value(), 8, "local KV heads");
  if (!local.ok()) return local.status();
  local = CheckedMultiply(local.value(), 256, "local head dimension");
  if (!local.ok()) return local.status();
  local = CheckedMultiply(local.value(), 2, "separate local K and V");
  if (!local.ok()) return local.status();
  auto global = CheckedMultiply(5, context, "global layer/context count");
  if (!global.ok()) return global.status();
  global = CheckedMultiply(global.value(), 2, "global KV heads");
  if (!global.ok()) return global.status();
  global = CheckedMultiply(global.value(), 512, "global head dimension");
  if (!global.ok()) return global.status();
  global = CheckedMultiply(global.value(), 2, "separate global K and V");
  if (!global.ok()) return global.status();
  return CheckedAdd(local.value(), global.value(), "32K FP8 K/V bytes");
}

}  // namespace gem16::internal

#include "model/gemma4_26b_attention.h"

#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gem16::internal {
namespace {

Status DataLoss(std::string message) {
  return Status(StatusCode::kDataLoss, std::move(message));
}

Result<std::uint64_t> CheckedMultiply(std::uint64_t left,
                                      std::uint64_t right,
                                      std::string_view label) {
  if (right != 0U && left > std::numeric_limits<std::uint64_t>::max() / right) {
    return Status(StatusCode::kInvalidArgument,
                  std::string(label) + " byte count overflows uint64");
  }
  return left * right;
}

Result<std::uint64_t> CheckedAdd(std::uint64_t left, std::uint64_t right,
                                 std::string_view label) {
  if (left > std::numeric_limits<std::uint64_t>::max() - right) {
    return Status(StatusCode::kInvalidArgument,
                  std::string(label) + " byte count overflows uint64");
  }
  return left + right;
}

bool ShapeEquals(const TensorInfo& tensor,
                 std::initializer_list<std::uint64_t> expected) {
  return tensor.logical_shape == std::vector<std::uint64_t>(expected);
}

Status RequireTensor(
    const std::map<std::string_view, const TensorInfo*, std::less<>>& by_name,
    const std::string& name, std::string_view dtype,
    std::initializer_list<std::uint64_t> shape) {
  const auto found = by_name.find(name);
  if (found == by_name.end()) {
    return DataLoss("M12 attention tensor is missing: " + name);
  }
  const TensorInfo& tensor = *found->second;
  if (tensor.storage_dtype != dtype || !ShapeEquals(tensor, shape) ||
      tensor.residency_class != "immutable_device_text") {
    return DataLoss("M12 attention tensor contract mismatch: " + name);
  }
  return Status::Ok();
}

}  // namespace

Result<Gemma4Moe26BAttentionTraits> BuildGemma4Moe26BAttentionTraits(
    const ModelConfig& config) {
  Status status = ValidateGemma4Moe26BContract(config);
  if (!status.ok()) return status;

  Gemma4Moe26BAttentionTraits result{};
  for (std::size_t layer = 0; layer < result.size(); ++layer) {
    const bool full = config.layer_types[layer] == "full_attention";
    auto& traits = result[layer];
    traits.layer = static_cast<std::uint32_t>(layer);
    traits.attention = full ? Gemma4Moe26BAttentionType::kFull
                            : Gemma4Moe26BAttentionType::kSliding;
    traits.query_heads = static_cast<std::uint32_t>(config.query_heads);
    traits.kv_heads = static_cast<std::uint32_t>(
        full ? config.global_kv_heads : config.local_kv_heads);
    traits.head_dimension = static_cast<std::uint32_t>(
        full ? config.global_head_dimension : config.local_head_dimension);
    traits.cache_capacity = static_cast<std::uint32_t>(
        full ? config.max_positions : config.sliding_window);
    traits.stores_v_projection = !full;
    traits.reuses_raw_k_for_v = full;
    traits.kv_source = Gemma4Moe26BKvSource::kOwnedProjection;
    traits.kv_producer_layer = static_cast<std::int32_t>(layer);
    traits.rope_theta = full ? config.global_rope_theta
                             : config.local_rope_theta;
    traits.rotary_factor = full ? config.global_partial_rotary_factor : 1.0;
    traits.rope_scaling_factor = 1.0;
  }
  return result;
}

Result<std::uint64_t> Gemma4Moe26BFp8KvBytes(
    const Gemma4Moe26BAttentionTraits& traits,
    std::uint64_t context_tokens) {
  if (context_tokens == 0U || context_tokens > 262144U) {
    return Status(StatusCode::kInvalidArgument,
                  "M12 context must be within 1..262144 tokens");
  }
  std::uint64_t bytes = 0U;
  for (std::size_t index = 0; index < traits.size(); ++index) {
    const auto& layer = traits[index];
    if (layer.layer != index || layer.query_heads != 16U ||
        layer.kv_source != Gemma4Moe26BKvSource::kOwnedProjection ||
        layer.kv_producer_layer != static_cast<std::int32_t>(index)) {
      return DataLoss("M12 layer trait table is not the validated owned-KV table");
    }
    const bool sliding =
        layer.attention == Gemma4Moe26BAttentionType::kSliding;
    const std::uint64_t tokens = sliding ? layer.cache_capacity : context_tokens;
    auto layer_bytes = CheckedMultiply(tokens, layer.kv_heads, "M12 KV tokens/heads");
    if (!layer_bytes.ok()) return layer_bytes.status();
    layer_bytes = CheckedMultiply(layer_bytes.value(), layer.head_dimension,
                                  "M12 KV head dimension");
    if (!layer_bytes.ok()) return layer_bytes.status();
    layer_bytes = CheckedMultiply(layer_bytes.value(), 2U,
                                  "M12 separate K and V");
    if (!layer_bytes.ok()) return layer_bytes.status();
    auto total = CheckedAdd(bytes, layer_bytes.value(), "M12 total KV");
    if (!total.ok()) return total.status();
    bytes = total.value();
  }
  return bytes;
}

Status ValidateGemma4Moe26BAttentionBindings(
    std::span<const TensorInfo> tensors,
    const Gemma4Moe26BAttentionTraits& traits) {
  std::map<std::string_view, const TensorInfo*, std::less<>> by_name;
  for (const auto& tensor : tensors) {
    if (!by_name.emplace(tensor.name, &tensor).second) {
      return DataLoss("M12 compiled inventory contains duplicate tensor: " +
                      tensor.name);
    }
  }
  for (std::size_t layer = 0; layer < traits.size(); ++layer) {
    const auto& t = traits[layer];
    if (t.layer != layer || t.query_heads != 16U ||
        (t.stores_v_projection == t.reuses_raw_k_for_v)) {
      return DataLoss("M12 attention traits are internally inconsistent");
    }
    const std::string prefix = "model.language_model.layers." +
                               std::to_string(layer) + ".";
    const std::uint64_t q_rows = t.query_heads * t.head_dimension;
    const std::uint64_t kv_rows = t.kv_heads * t.head_dimension;
    const std::array<std::pair<std::string_view, std::uint64_t>, 3>
        projections = {{{"self_attn.q_proj.weight", q_rows},
                        {"self_attn.k_proj.weight", kv_rows},
                        {"self_attn.o_proj.weight", 2816U}}};
    for (const auto& [suffix, rows] : projections) {
      const std::uint64_t columns =
          suffix.find("o_proj") == std::string_view::npos ? 2816U : q_rows;
      Status status = RequireTensor(by_name, prefix + std::string(suffix), "F8_E4M3",
                                    {rows, columns});
      if (!status.ok()) return status;
      status = RequireTensor(by_name, prefix + std::string(suffix) + "_scale", "BF16",
                             {rows, 1U});
      if (!status.ok()) return status;
    }
    const std::string v_name = prefix + "self_attn.v_proj.weight";
    const std::string v_scale_name = v_name + "_scale";
    if (t.stores_v_projection) {
      Status status = RequireTensor(by_name, v_name, "F8_E4M3",
                                    {kv_rows, 2816U});
      if (!status.ok()) return status;
      status = RequireTensor(by_name, v_scale_name, "BF16", {kv_rows, 1U});
      if (!status.ok()) return status;
    } else if (by_name.contains(v_name) || by_name.contains(v_scale_name)) {
      return DataLoss("M12 global layer unexpectedly owns V projection: " +
                      std::to_string(layer));
    }
    for (const auto* suffix : {"self_attn.q_norm.weight",
                               "self_attn.k_norm.weight"}) {
      Status status = RequireTensor(by_name, prefix + suffix, "BF16",
                                    {t.head_dimension});
      if (!status.ok()) return status;
    }
    for (const auto* suffix : {"self_attn.k_scale", "self_attn.v_scale"}) {
      Status status = RequireTensor(by_name, prefix + suffix, "BF16", {1U});
      if (!status.ok()) return status;
    }
    for (const auto* suffix : {"input_layernorm.weight",
                               "post_attention_layernorm.weight"}) {
      Status status = RequireTensor(by_name, prefix + suffix, "BF16", {2816U});
      if (!status.ok()) return status;
    }
  }
  return Status::Ok();
}

}  // namespace gem16::internal

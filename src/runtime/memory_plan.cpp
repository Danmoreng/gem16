#include "runtime/memory_plan.h"

#include <algorithm>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <string_view>
#include <utility>

namespace gem16::internal {
namespace {

Result<std::uint64_t> CheckedAdd(std::uint64_t left, std::uint64_t right, std::string_view label) {
  if (left > std::numeric_limits<std::uint64_t>::max() - right) {
    return Status(StatusCode::kInvalidArgument, std::string(label) + " byte count overflows uint64");
  }
  return left + right;
}

Result<std::uint64_t> CheckedMultiply(std::initializer_list<std::uint64_t> factors, std::string_view label) {
  std::uint64_t result = 1;
  for (const std::uint64_t factor : factors) {
    if (factor != 0 && result > std::numeric_limits<std::uint64_t>::max() / factor) {
      return Status(StatusCode::kInvalidArgument, std::string(label) + " byte count overflows uint64");
    }
    result *= factor;
  }
  return result;
}

bool IsPowerOfTwo(std::uint64_t value) {
  return value != 0 && (value & (value - 1U)) == 0;
}

Result<std::uint64_t> AlignUp(std::uint64_t value, std::uint64_t alignment) {
  if (!IsPowerOfTwo(alignment)) {
    return Status(StatusCode::kInvalidArgument, "arena alignment must be a non-zero power of two");
  }
  const std::uint64_t mask = alignment - 1U;
  if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
    return Status(StatusCode::kInvalidArgument, "aligned arena offset overflows uint64");
  }
  return (value + mask) & ~mask;
}

bool IsScaleTensor(const TensorInfo& tensor) {
  constexpr std::string_view classes[] = {
      "FP8_WEIGHT_SCALE",
      "NVFP4_GLOBAL_SCALE",
      "NVFP4_INPUT_SCALE",
      "NVFP4_LOCAL_SCALE_E4M3",
  };
  const bool scale_class =
      std::any_of(std::begin(classes), std::end(classes), [&tensor](std::string_view value) {
        return tensor.quantization_class == value;
      });
  return scale_class || tensor.expected_role == "weight_global_scale" ||
         tensor.expected_role == "activation_global_scale" ||
         tensor.expected_role == "weight_local_or_channel_scale";
}

std::string_view ProfileName(ContextProfile profile) {
  switch (profile) {
    case ContextProfile::kInteractive: return "interactive";
    case ContextProfile::kStandard: return "standard";
    case ContextProfile::kLong: return "long";
    case ContextProfile::kXlong: return "xlong";
    case ContextProfile::kMax: return "max";
  }
  return "unknown";
}

Status AddRegion(MemoryPlan& plan, std::uint64_t& cursor, std::string name, std::uint64_t bytes,
                 std::uint64_t alignment) {
  auto aligned = AlignUp(cursor, alignment);
  if (!aligned.ok()) return aligned.status();
  auto padding = CheckedAdd(plan.padding_bytes, aligned.value() - cursor, "arena padding");
  if (!padding.ok()) return padding.status();
  auto end = CheckedAdd(aligned.value(), bytes, "arena region");
  if (!end.ok()) return end.status();
  plan.padding_bytes = padding.value();
  plan.regions.push_back({std::move(name), aligned.value(), bytes, alignment});
  cursor = end.value();
  return Status::Ok();
}

}  // namespace

Result<std::uint64_t> ContextTokens(ContextProfile profile) {
  switch (profile) {
    case ContextProfile::kInteractive: return 8192;
    case ContextProfile::kStandard: return 32768;
    case ContextProfile::kLong: return 65536;
    case ContextProfile::kXlong: return 131072;
    case ContextProfile::kMax: return 262144;
  }
  return Status(StatusCode::kInvalidArgument, "unknown context profile");
}

Result<MemoryPlan> BuildMemoryPlan(const ModelConfig& config, const ModelManifest& manifest,
                                   const MemoryPlanOptions& options) {
  if (options.kv_storage == KvStorage::kUnspecified) {
    return Status(StatusCode::kInvalidArgument,
                  "KV storage must be explicit: choose shared or separate K/V storage");
  }
  if (options.kv_element_bytes == 0) {
    return Status(StatusCode::kInvalidArgument, "KV element byte width must be positive");
  }
  if (!IsPowerOfTwo(options.arena_alignment)) {
    return Status(StatusCode::kInvalidArgument, "arena alignment must be a non-zero power of two");
  }

  auto context_tokens = ContextTokens(options.context_profile);
  if (!context_tokens.ok()) return context_tokens.status();
  if (context_tokens.value() > config.max_positions) {
    return Status(StatusCode::kUnsupported, "context profile exceeds the checkpoint maximum position count");
  }
  if (config.layer_types.size() != config.layer_count || config.sliding_window == 0 ||
      config.local_kv_heads == 0 || config.global_kv_heads == 0 || config.local_head_dimension == 0 ||
      config.global_head_dimension == 0) {
    return Status(StatusCode::kInvalidArgument, "model config is incomplete for memory planning");
  }
  if (options.kv_storage == KvStorage::kShared) {
    return Status(
        StatusCode::kUnsupported,
        "shared physical K/V storage is unsupported: attention_k_eq_v reuses only the "
        "projection output; K normalization/RoPE and V normalization produce distinct cache states");
  }

  MemoryPlan plan;
  plan.model_directory = manifest.model_directory;
  plan.context_profile = ProfileName(options.context_profile);
  plan.kv_storage = "separate";
  plan.context_tokens = context_tokens.value();
  plan.kv_element_bytes = options.kv_element_bytes;
  plan.arena_alignment = options.arena_alignment;
  for (const auto& layer_type : config.layer_types) {
    if (layer_type == "sliding_attention") {
      ++plan.local_layer_count;
    } else if (layer_type == "full_attention") {
      ++plan.global_layer_count;
    } else {
      return Status(StatusCode::kUnsupported, "unknown attention layer type in memory plan: " + layer_type);
    }
  }

  for (const auto& tensor : manifest.tensors) {
    if (!tensor.loaded_in_text_only_mode) continue;
    auto& destination = IsScaleTensor(tensor) ? plan.scale_bytes : plan.model_weight_bytes;
    auto sum = CheckedAdd(destination, tensor.byte_length, "text-only tensor");
    if (!sum.ok()) return sum.status();
    destination = sum.value();
  }
  auto text_only_bytes = CheckedAdd(plan.model_weight_bytes, plan.scale_bytes, "text-only tensor");
  if (!text_only_bytes.ok()) return text_only_bytes.status();
  plan.text_only_source_bytes = text_only_bytes.value();
  if (plan.text_only_source_bytes != manifest.text_only_tensor_bytes) {
    return Status(StatusCode::kDataLoss,
                  "manifest text-only byte total disagrees with the tensor inventory");
  }

  const std::uint64_t local_tokens = std::min(plan.context_tokens, config.sliding_window);
  auto local_bytes = CheckedMultiply(
      {plan.local_layer_count, local_tokens, config.local_kv_heads, config.local_head_dimension,
       options.kv_element_bytes},
      "local KV cache");
  if (!local_bytes.ok()) return local_bytes.status();
  plan.local_shared_kv_bytes = local_bytes.value();

  auto global_bytes = CheckedMultiply(
      {plan.global_layer_count, plan.context_tokens, config.global_kv_heads, config.global_head_dimension,
       options.kv_element_bytes},
      "global KV cache");
  if (!global_bytes.ok()) return global_bytes.status();
  plan.global_shared_kv_bytes = global_bytes.value();

  auto shared_bytes = CheckedAdd(plan.local_shared_kv_bytes, plan.global_shared_kv_bytes, "shared KV cache");
  if (!shared_bytes.ok()) return shared_bytes.status();
  plan.shared_kv_bytes = shared_bytes.value();
  auto separate_bytes = CheckedMultiply({plan.shared_kv_bytes, 2}, "separate KV cache");
  if (!separate_bytes.ok()) return separate_bytes.status();
  plan.separate_kv_bytes = separate_bytes.value();
  plan.selected_kv_bytes = plan.separate_kv_bytes;

  std::uint64_t cursor = 0;
  auto status = AddRegion(plan, cursor, "immutable_model_weights", plan.model_weight_bytes,
                          options.arena_alignment);
  if (!status.ok()) return status;
  status = AddRegion(plan, cursor, "scales", plan.scale_bytes, options.arena_alignment);
  if (!status.ok()) return status;
  status = AddRegion(plan, cursor, "kv_cache", plan.selected_kv_bytes, options.arena_alignment);
  if (!status.ok()) return status;

  auto total = AlignUp(cursor, options.arena_alignment);
  if (!total.ok()) return total.status();
  auto final_padding = CheckedAdd(plan.padding_bytes, total.value() - cursor, "arena padding");
  if (!final_padding.ok()) return final_padding.status();
  plan.padding_bytes = final_padding.value();
  plan.total_arena_bytes = total.value();
  return plan;
}

}  // namespace gem16::internal

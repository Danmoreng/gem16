#include "model/gemma4_26b_residency.h"

#include <array>
#include <filesystem>
#include <limits>
#include <set>
#include <string_view>

#include "model/gemma4_26b_manifest.h"

namespace gem16::internal {
namespace {

constexpr std::uint64_t kMiB = 1024U * 1024U;
constexpr std::uint64_t kWeightAlignment = 256U;
constexpr std::uint64_t kPrimaryMargin = 700U * kMiB;
constexpr std::uint64_t kLongContextMargin = 400U * kMiB;
constexpr std::uint64_t kM08PayloadBytes = 14'696'569'196ULL;

Result<std::uint64_t> CheckedAdd(std::uint64_t left, std::uint64_t right,
                                 std::string_view label) {
  if (left > std::numeric_limits<std::uint64_t>::max() - right) {
    return Status(StatusCode::kInvalidArgument,
                  std::string(label) + " byte count overflows uint64");
  }
  return left + right;
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

Result<std::uint64_t> AlignUp(std::uint64_t value, std::uint64_t alignment) {
  const std::uint64_t mask = alignment - 1U;
  auto padded = CheckedAdd(value, mask, "aligned arena offset");
  if (!padded.ok()) return padded.status();
  return padded.value() & ~mask;
}

Result<std::uint64_t> Fp8KvBytes(std::uint64_t context_tokens) {
  constexpr std::uint64_t kLocalLayers = 25U;
  constexpr std::uint64_t kLocalWindow = 1024U;
  constexpr std::uint64_t kLocalHeads = 8U;
  constexpr std::uint64_t kLocalHeadDimension = 256U;
  constexpr std::uint64_t kGlobalLayers = 5U;
  constexpr std::uint64_t kGlobalHeads = 2U;
  constexpr std::uint64_t kGlobalHeadDimension = 512U;
  constexpr std::uint64_t kSeparateKeyValue = 2U;

  auto local = CheckedMultiply(kLocalLayers, kLocalWindow, "local layer/window");
  if (!local.ok()) return local.status();
  local = CheckedMultiply(local.value(), kLocalHeads, "local KV heads");
  if (!local.ok()) return local.status();
  local = CheckedMultiply(local.value(), kLocalHeadDimension,
                          "local head dimension");
  if (!local.ok()) return local.status();
  local = CheckedMultiply(local.value(), kSeparateKeyValue, "local K/V");
  if (!local.ok()) return local.status();

  auto global = CheckedMultiply(kGlobalLayers, context_tokens,
                                "global layer/context");
  if (!global.ok()) return global.status();
  global = CheckedMultiply(global.value(), kGlobalHeads, "global KV heads");
  if (!global.ok()) return global.status();
  global = CheckedMultiply(global.value(), kGlobalHeadDimension,
                           "global head dimension");
  if (!global.ok()) return global.status();
  global = CheckedMultiply(global.value(), kSeparateKeyValue, "global K/V");
  if (!global.ok()) return global.status();
  return CheckedAdd(local.value(), global.value(), "FP8 K/V");
}

bool IsSafeShardName(const std::string& name) {
  const std::filesystem::path path(name);
  return !path.empty() && !path.is_absolute() && !path.has_parent_path() &&
         path.extension() == ".safetensors";
}

}  // namespace

Result<Gemma4Moe26BResidencyPlan> BuildGemma4Moe26BResidencyPlan(
    const ModelManifest& manifest) {
  if (manifest.model_variant != "gemma4_moe_26b_a4b" ||
      manifest.checkpoint_profile != "sm120-text-hybrid-v1" ||
      manifest.validation_contract != "gemma4_26b_m08_compiled_hybrid_v1" ||
      !manifest.tensor_contract_validated || !manifest.supports_text ||
      manifest.supports_vision || manifest.supports_audio ||
      manifest.supports_video || manifest.supports_mtp) {
    return Status(StatusCode::kUnsupported,
                  "M09 requires the exact validated text-only M08 artifact");
  }
  if (manifest.tensors.size() != 1285U ||
      manifest.total_tensor_bytes != kM08PayloadBytes) {
    return Status(StatusCode::kDataLoss,
                  "M09 artifact tensor inventory is incomplete");
  }

  Gemma4Moe26BResidencyPlan plan;
  plan.artifact_payload_bytes = manifest.total_tensor_bytes;
  plan.fixed_regions = {
      {"moe_prefill_workspace", 192U * kMiB},
      {"router_permutation_workspace", 64U * kMiB},
      {"activation_ping", 64U * kMiB},
      {"activation_pong", 64U * kMiB},
      {"output_sampling_workspace", 16U * kMiB},
      {"graph_private_reserve", 32U * kMiB},
      {"allocator_metadata_guard", 16U * kMiB},
  };
  for (const auto& region : plan.fixed_regions) {
    auto total = CheckedAdd(plan.fixed_region_bytes, region.bytes,
                            "fixed residency regions");
    if (!total.ok()) return total.status();
    plan.fixed_region_bytes = total.value();
  }

  std::set<std::string> names;
  std::uint64_t payload_bytes = 0U;
  std::uint64_t cursor = 0U;
  for (const auto& tensor : manifest.tensors) {
    if (!names.insert(tensor.name).second || tensor.byte_length == 0U ||
        !IsSafeShardName(tensor.source_shard) ||
        tensor.residency_class != "immutable_device_text" ||
        tensor.final_gpu_layout.empty() || tensor.final_gpu_layout == "none") {
      return Status(StatusCode::kDataLoss,
                    "invalid M09 upload tensor: " + tensor.name);
    }
    auto destination = AlignUp(cursor, plan.arena_alignment);
    if (!destination.ok()) return destination.status();
    auto end = CheckedAdd(destination.value(), tensor.byte_length,
                          "immutable weight arena");
    if (!end.ok()) return end.status();
    auto payload = CheckedAdd(payload_bytes, tensor.byte_length,
                              "artifact payload");
    if (!payload.ok()) return payload.status();
    plan.upload_ranges.push_back({
        tensor.name,
        tensor.source_shard,
        tensor.quantization_class,
        tensor.final_gpu_layout,
        tensor.byte_offset,
        destination.value(),
        tensor.byte_length,
    });
    cursor = end.value();
    payload_bytes = payload.value();
  }
  if (payload_bytes != manifest.total_tensor_bytes) {
    return Status(StatusCode::kDataLoss,
                  "M09 payload bytes disagree with the validated manifest");
  }
  auto arena_bytes = AlignUp(cursor, plan.arena_alignment);
  if (!arena_bytes.ok()) return arena_bytes.status();
  auto contract_bytes = Gemma4Moe26BAlignedArenaBytes(manifest.tensors,
                                                       plan.arena_alignment);
  if (!contract_bytes.ok()) return contract_bytes.status();
  if (arena_bytes.value() != contract_bytes.value()) {
    return Status(StatusCode::kInternal,
                  "M09 upload offsets disagree with the compiled arena contract");
  }
  plan.immutable_weight_arena_bytes = arena_bytes.value();

  constexpr std::array<std::uint64_t, 4> kContexts = {
      8192U, 16384U, 32768U, 65536U};
  for (const std::uint64_t context : kContexts) {
    auto kv = Fp8KvBytes(context);
    if (!kv.ok()) return kv.status();
    auto total = CheckedAdd(plan.immutable_weight_arena_bytes, kv.value(),
                            "profile weights and K/V");
    if (!total.ok()) return total.status();
    total = CheckedAdd(total.value(), plan.fixed_region_bytes,
                       "profile fixed regions");
    if (!total.ok()) return total.status();
    const std::uint64_t margin =
        context >= 65536U ? kLongContextMargin : kPrimaryMargin;
    auto admission = CheckedAdd(total.value(), margin,
                                "profile safety margin");
    if (!admission.ok()) return admission.status();
    plan.context_profiles.push_back(
        {context, kv.value(), margin, total.value(), admission.value()});
  }
  return plan;
}

Status CheckGemma4Moe26BAdmission(const Gemma4Moe26BResidencyPlan& plan,
                                  std::uint64_t context_tokens,
                                  std::uint64_t free_device_bytes,
                                  bool include_weight_arena) {
  const Gemma4Moe26BContextResidency* profile = nullptr;
  for (const auto& candidate : plan.context_profiles) {
    if (candidate.context_tokens == context_tokens) {
      profile = &candidate;
      break;
    }
  }
  if (profile == nullptr) {
    return Status(StatusCode::kInvalidArgument,
                  "unknown Gemma 4 26B residency context profile");
  }
  auto required = CheckedAdd(profile->fp8_kv_bytes, plan.fixed_region_bytes,
                             "slot variable regions");
  if (!required.ok()) return required.status();
  required = CheckedAdd(required.value(), profile->required_free_margin_bytes,
                        "slot safety margin");
  if (!required.ok()) return required.status();
  if (include_weight_arena) {
    required = CheckedAdd(required.value(), plan.immutable_weight_arena_bytes,
                          "slot immutable weights");
    if (!required.ok()) return required.status();
  }
  if (free_device_bytes < required.value()) {
    return Status(StatusCode::kResourceExhausted,
                  "insufficient CUDA-visible memory for one complete 26B slot");
  }
  return Status::Ok();
}

}  // namespace gem16::internal

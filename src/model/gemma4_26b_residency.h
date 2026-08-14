#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gem16/status.h"
#include "gem16/types.h"

namespace gem16::internal {

struct Gemma4Moe26BUploadRange {
  std::string tensor_name;
  std::string source_shard;
  std::string quantization_class;
  std::string runtime_layout;
  std::uint64_t source_offset = 0;
  std::uint64_t destination_offset = 0;
  std::uint64_t bytes = 0;
};

struct Gemma4Moe26BFixedRegion {
  std::string name;
  std::uint64_t bytes = 0;
};

struct Gemma4Moe26BContextResidency {
  std::uint64_t context_tokens = 0;
  std::uint64_t fp8_kv_bytes = 0;
  std::uint64_t required_free_margin_bytes = 0;
  std::uint64_t total_device_bytes_without_margin = 0;
  std::uint64_t admission_bytes = 0;
};

struct Gemma4Moe26BResidencyPlan {
  std::uint64_t arena_alignment = 256;
  std::uint64_t artifact_payload_bytes = 0;
  std::uint64_t immutable_weight_arena_bytes = 0;
  std::uint64_t fixed_region_bytes = 0;
  std::vector<Gemma4Moe26BUploadRange> upload_ranges;
  std::vector<Gemma4Moe26BFixedRegion> fixed_regions;
  std::vector<Gemma4Moe26BContextResidency> context_profiles;
};

[[nodiscard]] Result<Gemma4Moe26BResidencyPlan>
BuildGemma4Moe26BResidencyPlan(const ModelManifest& manifest);

[[nodiscard]] Status CheckGemma4Moe26BAdmission(
    const Gemma4Moe26BResidencyPlan& plan, std::uint64_t context_tokens,
    std::uint64_t free_device_bytes, bool include_weight_arena);

}  // namespace gem16::internal

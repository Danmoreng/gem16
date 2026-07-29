#pragma once

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

#include "gem16/status.h"

namespace gem16 {

enum class ContextProfile { kInteractive, kStandard, kLong, kXlong, kMax };
enum class KvStorage { kUnspecified, kShared, kSeparate };

struct MemoryPlanOptions {
  ContextProfile context_profile = ContextProfile::kStandard;
  KvStorage kv_storage = KvStorage::kUnspecified;
  std::uint64_t kv_element_bytes = 1;
  std::uint64_t arena_alignment = 256;
};

struct MemoryRegion {
  std::string name;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
  std::uint64_t alignment = 1;
};

struct MemoryPlan {
  std::string model_directory;
  std::string context_profile;
  std::string kv_storage;
  std::uint64_t context_tokens = 0;
  std::uint64_t kv_element_bytes = 0;
  std::uint64_t arena_alignment = 0;
  std::uint64_t local_layer_count = 0;
  std::uint64_t global_layer_count = 0;
  std::uint64_t resident_source_bytes = 0;
  std::uint64_t model_weight_bytes = 0;
  std::uint64_t scale_bytes = 0;
  // One final attention state per token. Retained as a diagnostic lower bound;
  // Gemma 4 requires separate final K and V cache storage.
  std::uint64_t local_shared_kv_bytes = 0;
  std::uint64_t global_shared_kv_bytes = 0;
  std::uint64_t shared_kv_bytes = 0;
  std::uint64_t separate_kv_bytes = 0;
  std::uint64_t selected_kv_bytes = 0;
  std::uint64_t padding_bytes = 0;
  std::uint64_t total_arena_bytes = 0;
  bool shared_kv_storage_supported = false;
  bool execution_workspaces_planned = false;
  std::vector<MemoryRegion> regions;
};

[[nodiscard]] Result<MemoryPlan> PlanCheckpointMemory(
    const std::filesystem::path& model_directory,
    const MemoryPlanOptions& options);
[[nodiscard]] Status WriteMemoryPlanJson(const MemoryPlan& plan, std::ostream& output);
void PrintMemoryPlanSummary(const MemoryPlan& plan, std::ostream& output);

}  // namespace gem16

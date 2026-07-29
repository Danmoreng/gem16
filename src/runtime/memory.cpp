#include "gem16/memory.h"

#include <iomanip>
#include <ostream>

#include "gem16/model.h"
#include "model/config.h"
#include "runtime/memory_plan.h"
#include "util/json.h"

namespace gem16 {

Result<MemoryPlan> PlanCheckpointMemory(const std::filesystem::path& model_directory,
                                        const MemoryPlanOptions& options) {
  InspectOptions inspect_options;
  inspect_options.model_directory = model_directory;
  inspect_options.validate = true;
  auto manifest = InspectCheckpoint(inspect_options);
  if (!manifest.ok()) return manifest.status();

  auto config = internal::LoadModelConfig(model_directory / "config.json");
  if (!config.ok()) return config.status();
  auto validation = internal::ValidatePrimaryModelContract(config.value());
  if (!validation.ok()) return validation;
  return internal::BuildMemoryPlan(config.value(), manifest.value(), options);
}

Status WriteMemoryPlanJson(const MemoryPlan& plan, std::ostream& output) {
  output << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"status\": \"ok\",\n"
         << "  \"mode\": \"memory\",\n"
         << "  \"fallbacks\": 0,\n"
         << "  \"model_directory\": \"" << json::Escape(plan.model_directory) << "\",\n"
         << "  \"context_profile\": \"" << json::Escape(plan.context_profile) << "\",\n"
         << "  \"context_tokens\": " << plan.context_tokens << ",\n"
         << "  \"kv_storage\": \"" << json::Escape(plan.kv_storage) << "\",\n"
         << "  \"kv_element_bytes\": " << plan.kv_element_bytes << ",\n"
         << "  \"arena_alignment\": " << plan.arena_alignment << ",\n"
         << "  \"local_layer_count\": " << plan.local_layer_count << ",\n"
         << "  \"global_layer_count\": " << plan.global_layer_count << ",\n"
         << "  \"resident_source_bytes\": " << plan.resident_source_bytes << ",\n"
         << "  \"model_weight_bytes\": " << plan.model_weight_bytes << ",\n"
         << "  \"scale_bytes\": " << plan.scale_bytes << ",\n"
         << "  \"local_shared_kv_bytes\": " << plan.local_shared_kv_bytes << ",\n"
         << "  \"global_shared_kv_bytes\": " << plan.global_shared_kv_bytes << ",\n"
         << "  \"shared_kv_bytes\": " << plan.shared_kv_bytes << ",\n"
         << "  \"separate_kv_bytes\": " << plan.separate_kv_bytes << ",\n"
         << "  \"selected_kv_bytes\": " << plan.selected_kv_bytes << ",\n"
         << "  \"shared_kv_storage_supported\": "
         << (plan.shared_kv_storage_supported ? "true" : "false") << ",\n"
         << "  \"padding_bytes\": " << plan.padding_bytes << ",\n"
         << "  \"total_arena_bytes\": " << plan.total_arena_bytes << ",\n"
         << "  \"execution_workspaces_planned\": "
         << (plan.execution_workspaces_planned ? "true" : "false") << ",\n"
         << "  \"unplanned_regions\": [\"activations_a\",\"activations_b\",\"logits\","
            "\"sampling_workspace\",\"graph_workspace\",\"kernel_workspace\","
            "\"prefill_workspace\"],\n"
         << "  \"regions\": [\n";
  for (std::size_t index = 0; index < plan.regions.size(); ++index) {
    const auto& region = plan.regions[index];
    output << "    {\"name\":\"" << json::Escape(region.name) << "\",\"offset\":" << region.offset
           << ",\"bytes\":" << region.bytes << ",\"alignment\":" << region.alignment << '}'
           << (index + 1U == plan.regions.size() ? "\n" : ",\n");
  }
  output << "  ]\n}\n";
  if (!output) return Status(StatusCode::kIoError, "failed while writing memory-plan JSON");
  return Status::Ok();
}

void PrintMemoryPlanSummary(const MemoryPlan& plan, std::ostream& output) {
  constexpr double kMiB = 1024.0 * 1024.0;
  const auto previous_flags = output.flags();
  const auto previous_precision = output.precision();
  output << "Memory plan: profile=" << plan.context_profile << " context=" << plan.context_tokens
         << " kv=" << plan.kv_storage << " element_bytes=" << plan.kv_element_bytes << '\n'
         << "  model weights: " << std::fixed << std::setprecision(2)
         << static_cast<double>(plan.model_weight_bytes) / kMiB << " MiB\n"
         << "  scales: " << static_cast<double>(plan.scale_bytes) / kMiB << " MiB\n"
         << "  KV selected: " << static_cast<double>(plan.selected_kv_bytes) / kMiB << " MiB"
         << " (one-state lower bound=" << static_cast<double>(plan.shared_kv_bytes) / kMiB
         << ", separate=" << static_cast<double>(plan.separate_kv_bytes) / kMiB << ")\n"
         << "  known arena total: " << static_cast<double>(plan.total_arena_bytes) / kMiB << " MiB\n"
         << "  execution workspaces: not planned yet\n";
  output.flags(previous_flags);
  output.precision(previous_precision);
}

}  // namespace gem16

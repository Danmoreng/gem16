#include "test.h"

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>

#include "gem16/memory.h"
#include "runtime/memory_plan.h"
#include "util/json.h"

namespace {

gem16::internal::ModelConfig PrimaryMemoryConfig() {
  gem16::internal::ModelConfig config;
  config.layer_count = 48;
  config.max_positions = 262144;
  config.sliding_window = 1024;
  config.local_kv_heads = 8;
  config.global_kv_heads = 1;
  config.local_head_dimension = 256;
  config.global_head_dimension = 512;
  config.attention_k_eq_v = true;
  for (std::uint64_t layer = 0; layer < config.layer_count; ++layer) {
    config.layer_types.push_back(layer % 6U == 5U ? "full_attention" : "sliding_attention");
  }
  return config;
}

gem16::TensorInfo MemoryTensor(const char* name, const char* quantization_class,
                                 std::uint64_t byte_length,
                                 bool loaded_in_text_only_mode = true) {
  gem16::TensorInfo tensor;
  tensor.name = name;
  tensor.quantization_class = quantization_class;
  tensor.byte_length = byte_length;
  tensor.loaded_in_text_only_mode = loaded_in_text_only_mode;
  return tensor;
}

gem16::ModelManifest PrimaryMemoryManifest() {
  gem16::ModelManifest manifest;
  manifest.tensors = {
      MemoryTensor("weights", "BF16", 8'668'020'512),
      MemoryTensor("fp8_scales", "FP8_WEIGHT_SCALE", 1'163'264),
      MemoryTensor("nvfp4_local_scales", "NVFP4_LOCAL_SCALE_E4M3", 530'841'600),
      MemoryTensor("nvfp4_global_scales", "NVFP4_GLOBAL_SCALE", 576),
      MemoryTensor("nvfp4_input_scales", "NVFP4_INPUT_SCALE", 576),
      MemoryTensor("multimodal", "BF16", 104'759'808, false),
  };
  manifest.text_only_tensor_bytes = 9'200'026'528;
  manifest.skipped_tensor_bytes = 104'759'808;
  manifest.total_tensor_bytes = 9'304'786'336;
  return manifest;
}

}  // namespace

void RunMemoryPlanTests() {
  using gem16::internal::BuildMemoryPlan;
  using gem16::internal::ContextTokens;
  using gem16::ContextProfile;
  using gem16::KvStorage;
  using gem16::MemoryPlanOptions;

  GEM16_CHECK(ContextTokens(ContextProfile::kInteractive).value() == 8192);
  GEM16_CHECK(ContextTokens(ContextProfile::kStandard).value() == 32768);
  GEM16_CHECK(ContextTokens(ContextProfile::kLong).value() == 65536);
  GEM16_CHECK(ContextTokens(ContextProfile::kXlong).value() == 131072);
  GEM16_CHECK(ContextTokens(ContextProfile::kMax).value() == 262144);

  const auto config = PrimaryMemoryConfig();
  const auto manifest = PrimaryMemoryManifest();
  const MemoryPlanOptions shared_options{
      .context_profile = ContextProfile::kLong,
      .kv_storage = KvStorage::kShared,
  };
  auto shared = BuildMemoryPlan(config, manifest, shared_options);
  GEM16_CHECK(!shared.ok());
  if (!shared.ok()) {
    GEM16_CHECK(shared.status().code() == gem16::StatusCode::kUnsupported);
    GEM16_CHECK(shared.status().message().find("distinct cache states") != std::string::npos);
  }

  auto separate_options = shared_options;
  separate_options.kv_storage = KvStorage::kSeparate;
  auto separate = BuildMemoryPlan(config, manifest, separate_options);
  GEM16_CHECK(separate.ok());
  if (separate.ok()) {
    GEM16_CHECK(separate.value().context_profile == "long");
    GEM16_CHECK(separate.value().kv_storage == "separate");
    GEM16_CHECK(separate.value().kv_element_bytes == 1);
    GEM16_CHECK(separate.value().arena_alignment == 256);
    GEM16_CHECK(separate.value().local_layer_count == 40);
    GEM16_CHECK(separate.value().global_layer_count == 8);
    GEM16_CHECK(separate.value().model_weight_bytes == 8'772'780'320);
    GEM16_CHECK(separate.value().scale_bytes == 532'006'016);
    GEM16_CHECK(separate.value().resident_source_bytes == 9'304'786'336);
    GEM16_CHECK(separate.value().local_shared_kv_bytes == 83'886'080);
    GEM16_CHECK(separate.value().global_shared_kv_bytes == 268'435'456);
    GEM16_CHECK(separate.value().shared_kv_bytes == 352'321'536);
    GEM16_CHECK(separate.value().separate_kv_bytes == 704'643'072);
    GEM16_CHECK(separate.value().selected_kv_bytes == separate.value().separate_kv_bytes);
    GEM16_CHECK(!separate.value().shared_kv_storage_supported);
    GEM16_CHECK(separate.value().regions.size() == 3);
    for (const auto& region : separate.value().regions) {
      GEM16_CHECK(region.offset % 256U == 0);
      GEM16_CHECK(region.alignment == 256);
    }
    GEM16_CHECK(separate.value().total_arena_bytes % 256U == 0);
    GEM16_CHECK(separate.value().total_arena_bytes ==
                  separate.value().resident_source_bytes + separate.value().selected_kv_bytes +
                      separate.value().padding_bytes);
    GEM16_CHECK(!separate.value().execution_workspaces_planned);

    std::ostringstream json_output;
    GEM16_CHECK(gem16::WriteMemoryPlanJson(separate.value(), json_output).ok());
    auto parsed_json = gem16::json::Parse(json_output.str());
    GEM16_CHECK(parsed_json.ok());
    if (parsed_json.ok()) {
      const auto* total_arena = parsed_json.value().find("total_arena_bytes");
      const auto* workspaces_planned = parsed_json.value().find("execution_workspaces_planned");
      const auto* fallbacks = parsed_json.value().find("fallbacks");
      GEM16_CHECK(total_arena != nullptr);
      GEM16_CHECK(workspaces_planned != nullptr);
      GEM16_CHECK(fallbacks != nullptr);
      if (total_arena != nullptr) GEM16_CHECK(total_arena->as_integer() == 10'009'429'760);
      if (workspaces_planned != nullptr) GEM16_CHECK(!workspaces_planned->as_bool());
      if (fallbacks != nullptr) GEM16_CHECK(fallbacks->as_integer() == 0);
    }
  }

  auto bf16_options = separate_options;
  bf16_options.kv_element_bytes = 2;
  auto bf16 = BuildMemoryPlan(config, manifest, bf16_options);
  GEM16_CHECK(bf16.ok());
  if (bf16.ok()) {
    GEM16_CHECK(bf16.value().shared_kv_bytes == 704'643'072);
    GEM16_CHECK(bf16.value().selected_kv_bytes == 1'409'286'144);
  }

  GEM16_CHECK(!BuildMemoryPlan(config, manifest, MemoryPlanOptions{}).ok());

  auto invalid_alignment = separate_options;
  invalid_alignment.arena_alignment = 192;
  GEM16_CHECK(!BuildMemoryPlan(config, manifest, invalid_alignment).ok());

  auto non_shared_config = config;
  non_shared_config.attention_k_eq_v = false;
  GEM16_CHECK(!BuildMemoryPlan(non_shared_config, manifest, shared_options).ok());
  GEM16_CHECK(BuildMemoryPlan(non_shared_config, manifest, separate_options).ok());

  auto short_config = config;
  short_config.max_positions = 32768;
  GEM16_CHECK(!BuildMemoryPlan(short_config, manifest, separate_options).ok());

  auto inconsistent_manifest = manifest;
  ++inconsistent_manifest.total_tensor_bytes;
  GEM16_CHECK(!BuildMemoryPlan(config, inconsistent_manifest, separate_options).ok());

  auto overflow_options = separate_options;
  overflow_options.kv_element_bytes = std::numeric_limits<std::uint64_t>::max();
  GEM16_CHECK(!BuildMemoryPlan(config, manifest, overflow_options).ok());
}

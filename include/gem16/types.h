#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gem16 {

struct TensorInfo {
  std::string name;
  std::vector<std::uint64_t> shape;
  std::vector<std::uint64_t> logical_shape;
  std::string storage_dtype;
  std::string logical_dtype;
  std::string quantization_class;
  std::uint64_t byte_offset = 0;
  std::uint64_t byte_length = 0;
  std::uint64_t alignment = 1;
  std::string source_shard;
  std::string expected_role;
  std::string tensor_role;
  std::string residency_class;
  std::string source_family;
  std::string quantization_component;
  std::string quantization_producer;
  std::string local_scale_dtype;
  std::uint64_t local_scale_vector_size = 0;
  std::string global_scale_role;
  std::string activation_scale_role;
  std::string final_gpu_layout;
  std::string logical_axis_order;
  std::int64_t layer_index = -1;
  std::int64_t expert_index = -1;
  std::int64_t expert_axis = -1;
  std::string local_scale_tensor;
  std::string global_scale_tensor;
  std::string input_scale_tensor;
  std::string layout;
  bool loaded_in_text_only_mode = true;
  bool aliased = false;
};

struct TensorClassTotal {
  std::string quantization_class;
  std::uint64_t tensor_count = 0;
  std::uint64_t bytes = 0;
};

struct TensorGroupTotal {
  std::string name;
  std::uint64_t tensor_count = 0;
  std::uint64_t bytes = 0;
};

struct ModelManifest {
  std::string model_directory;
  std::string architecture;
  std::string model_type;
  std::string model_variant;
  std::string checkpoint_profile;
  std::string validation_contract;
  std::uint64_t layer_count = 0;
  std::uint64_t hidden_size = 0;
  std::uint64_t intermediate_size = 0;
  std::uint64_t moe_intermediate_size = 0;
  std::uint64_t expert_count = 0;
  std::uint64_t top_k_experts = 0;
  bool runtime_supported = false;
  bool supports_text = false;
  bool supports_vision = false;
  bool supports_audio = false;
  bool supports_video = false;
  bool supports_mtp = false;
  bool tensor_contract_validated = false;
  std::vector<TensorInfo> tensors;
  std::vector<TensorClassTotal> totals;
  std::vector<TensorGroupTotal> totals_by_role;
  std::vector<TensorGroupTotal> totals_by_residency;
  std::uint64_t total_tensor_bytes = 0;
  std::uint64_t text_only_tensor_bytes = 0;
  std::uint64_t skipped_tensor_bytes = 0;
};

}  // namespace gem16

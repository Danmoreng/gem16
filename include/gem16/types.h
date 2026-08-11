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
  std::string quantization_class;
  std::uint64_t byte_offset = 0;
  std::uint64_t byte_length = 0;
  std::uint64_t alignment = 1;
  std::string source_shard;
  std::string expected_role;
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

struct ModelManifest {
  std::string model_directory;
  std::string architecture;
  std::string model_type;
  std::string model_variant;
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
  std::uint64_t total_tensor_bytes = 0;
  std::uint64_t text_only_tensor_bytes = 0;
  std::uint64_t skipped_tensor_bytes = 0;
};

}  // namespace gem16

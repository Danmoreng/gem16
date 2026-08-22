#include <cuda_runtime_api.h>

#include <cstddef>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cuda/engine/gemma4_26b_artifact.h"
#include "cuda/moe/reference.h"
#include "gem16/model.h"
#include "model/config.h"
#include "model/gemma4_26b_residency.h"
#include "util/json.h"

namespace {

struct Options {
  std::filesystem::path model;
  std::filesystem::path fixture;
  std::filesystem::path output;
  int device = 0;
};

bool Parse(int argc, char** argv, Options* options) {
  for (int index = 1; index < argc; ++index) {
    if (index + 1 >= argc) return false;
    const std::string_view argument(argv[index]);
    const std::string value(argv[++index]);
    if (argument == "--model") options->model = value;
    else if (argument == "--fixture") options->fixture = value;
    else if (argument == "--output") options->output = value;
    else if (argument == "--device") {
      try {
        std::size_t consumed = 0;
        options->device = std::stoi(value, &consumed);
        if (consumed != value.size() || options->device < 0) return false;
      } catch (...) {
        return false;
      }
    } else return false;
  }
  return !options->model.empty() && !options->fixture.empty() &&
         !options->output.empty();
}

gem16::Result<std::string> Read(const std::filesystem::path& path,
                                std::uint64_t maximum) {
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  const auto size = std::filesystem::file_size(path, error);
  if (error || !std::filesystem::is_regular_file(status) ||
      std::filesystem::is_symlink(status) || size == 0U || size > maximum ||
      size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         "M11 fixture is missing, unsafe, or too large");
  }
  std::string payload(static_cast<std::size_t>(size), '\0');
  std::ifstream input(path, std::ios::binary);
  if (!input.read(payload.data(), static_cast<std::streamsize>(payload.size()))) {
    return gem16::Status(gem16::StatusCode::kIoError,
                         "cannot read M11 fixture");
  }
  return payload;
}

gem16::Result<std::vector<float>> FloatArray(const gem16::json::Value* value,
                                             std::size_t expected) {
  if (value == nullptr || !value->is_array() ||
      value->as_array().size() != expected) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         "M11 fixture float array shape mismatch");
  }
  std::vector<float> result;
  result.reserve(expected);
  for (const auto& element : value->as_array()) {
    if (!element.is_number()) {
      return gem16::Status(gem16::StatusCode::kDataLoss,
                           "M11 fixture contains a non-number");
    }
    const double number = element.as_number();
    if (!std::isfinite(number) || number < -std::numeric_limits<float>::max() ||
        number > std::numeric_limits<float>::max()) {
      return gem16::Status(gem16::StatusCode::kDataLoss,
                           "M11 fixture contains an invalid float");
    }
    result.push_back(static_cast<float>(number));
  }
  return result;
}

class Buffer {
 public:
  Buffer() = default;
  ~Buffer() { if (pointer_ != nullptr) (void)cudaFree(pointer_); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  gem16::Status Allocate(std::uint64_t bytes) {
    if (bytes == 0U || bytes > std::numeric_limits<std::size_t>::max()) {
      return {gem16::StatusCode::kInvalidArgument, "invalid M11 buffer size"};
    }
    const cudaError_t error = cudaMalloc(&pointer_, static_cast<std::size_t>(bytes));
    if (error != cudaSuccess) {
      return {gem16::StatusCode::kResourceExhausted,
              std::string("allocate M11 workspace: ") + cudaGetErrorString(error)};
    }
    bytes_ = bytes;
    return gem16::Status::Ok();
  }
  template <typename T> T* As() const { return static_cast<T*>(pointer_); }
  std::uint64_t bytes() const { return bytes_; }
 private:
  void* pointer_ = nullptr;
  std::uint64_t bytes_ = 0U;
};

template <typename T>
gem16::Status Allocate(Buffer* buffer, std::uint64_t elements) {
  if (elements > std::numeric_limits<std::uint64_t>::max() / sizeof(T)) {
    return {gem16::StatusCode::kInvalidArgument, "M11 workspace overflow"};
  }
  return buffer->Allocate(elements * sizeof(T));
}

template <typename T>
gem16::Result<const T*> Pointer(
    const gem16::internal::Gemma4Moe26BDeviceArtifact& artifact,
    const std::string& name) {
  auto pointer = artifact.Pointer(name);
  if (!pointer.ok()) return pointer.status();
  return reinterpret_cast<const T*>(pointer.value());
}

gem16::Result<gem16::internal::Gemma4MoeNvfp4Matrix> Matrix(
    const gem16::internal::Gemma4Moe26BDeviceArtifact& artifact,
    const std::string& stem, std::uint64_t rows, std::uint64_t columns) {
  auto packed = Pointer<std::uint8_t>(artifact, stem + ".weight_packed");
  auto scales = Pointer<std::uint8_t>(artifact, stem + ".weight_scale");
  auto activation = artifact.HostFloat32(stem + ".input_global_scale");
  auto weight = artifact.HostFloat32(stem + ".weight_global_scale");
  if (!packed.ok()) return packed.status();
  if (!scales.ok()) return scales.status();
  if (!activation.ok()) return activation.status();
  if (!weight.ok()) return weight.status();
  return gem16::internal::Gemma4MoeNvfp4Matrix{
      packed.value(), scales.value(), rows, columns, activation.value(),
      weight.value()};
}

void WriteFloatArray(std::ostream& output, const std::vector<float>& values) {
  output << '[' << std::setprecision(9);
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0U) output << ',';
    output << values[index];
  }
  output << ']';
}

template <typename T>
gem16::Status Copy(std::vector<T>* host, const Buffer& buffer,
                   std::uint64_t elements) {
  host->resize(static_cast<std::size_t>(elements));
  const cudaError_t error = cudaMemcpy(host->data(), buffer.As<T>(),
                                       host->size() * sizeof(T),
                                       cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    return {gem16::StatusCode::kInternal,
            std::string("copy M11 boundary: ") + cudaGetErrorString(error)};
  }
  return gem16::Status::Ok();
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!Parse(argc, argv, &options)) {
    std::cerr << "usage: gem16-26b-moe-probe --model DIR --fixture JSON --output JSON [--device N]\n";
    return 2;
  }
  if (cudaSetDevice(options.device) != cudaSuccess) {
    std::cerr << "error: cannot select CUDA device\n";
    return 3;
  }
  auto fixture_text = Read(options.fixture, 16U * 1024U * 1024U);
  if (!fixture_text.ok()) {
    std::cerr << "error: " << fixture_text.status().message() << '\n';
    return 3;
  }
  auto fixture = gem16::json::Parse(fixture_text.value());
  const auto* source = fixture.ok() ? fixture.value().find("source") : nullptr;
  const auto* layer_value = source != nullptr ? source->find("layer") : nullptr;
  if (!fixture.ok() || fixture.value().find("schema_version") == nullptr ||
      !fixture.value().find("schema_version")->is_integer() ||
      fixture.value().find("schema_version")->as_integer() != 1 ||
      source == nullptr || !source->is_object() || layer_value == nullptr ||
      !layer_value->is_integer() || layer_value->as_integer() < 0 ||
      layer_value->as_integer() >= 30) {
    std::cerr << "error: invalid M11 fixture schema\n";
    return 3;
  }
  constexpr std::uint64_t kWidth = 2816U;
  constexpr std::uint64_t kShared = 2112U;
  constexpr std::uint64_t kExpert = 704U;
  constexpr std::uint32_t kExperts = 128U;
  constexpr std::uint32_t kTopK = 8U;
  const std::uint32_t layer = static_cast<std::uint32_t>(layer_value->as_integer());
  auto hidden = FloatArray(fixture.value().find("hidden_f32"), kWidth);
  if (!hidden.ok()) {
    std::cerr << "error: " << hidden.status().message() << '\n';
    return 3;
  }
  auto manifest = gem16::InspectCheckpoint({options.model, true});
  if (!manifest.ok()) {
    std::cerr << "error: " << manifest.status().message() << '\n';
    return 4;
  }
  auto model_config =
      gem16::internal::LoadModelConfig(options.model / "config.json");
  if (!model_config.ok()) {
    std::cerr << model_config.status().message() << '\n';
    return 3;
  }
  auto plan = gem16::internal::BuildGemma4Moe26BResidencyPlan(
      manifest.value(), model_config.value());
  if (!plan.ok()) {
    std::cerr << "error: " << plan.status().message() << '\n';
    return 4;
  }
  auto artifact = gem16::internal::Gemma4Moe26BDeviceArtifact::Load(
      options.model, manifest.value(), plan.value());
  if (!artifact.ok()) {
    std::cerr << "error: " << artifact.status().message() << '\n';
    return 5;
  }
  const std::string prefix = "model.language_model.layers." +
                             std::to_string(layer);
  gem16::internal::Gemma4MoeReferenceWeights weights;
  auto bind_bf16 = [&](const std::string& suffix,
                       const std::uint16_t** destination) -> gem16::Status {
    auto pointer = Pointer<std::uint16_t>(artifact.value(), prefix + suffix);
    if (!pointer.ok()) return pointer.status();
    *destination = pointer.value();
    return gem16::Status::Ok();
  };
  gem16::Status status = bind_bf16(".pre_feedforward_layernorm.weight",
                                   &weights.pre_shared_norm_bf16);
  if (status.ok()) status = bind_bf16(".post_feedforward_layernorm_1.weight",
                                      &weights.post_shared_norm_bf16);
  if (status.ok()) status = bind_bf16(".pre_feedforward_layernorm_2.weight",
                                      &weights.pre_expert_norm_bf16);
  if (status.ok()) status = bind_bf16(".post_feedforward_layernorm_2.weight",
                                      &weights.post_expert_norm_bf16);
  if (status.ok()) status = bind_bf16(".post_feedforward_layernorm.weight",
                                      &weights.post_combined_norm_bf16);
  if (status.ok()) status = bind_bf16(".router.scale", &weights.router_scale_bf16);
  if (status.ok()) status = bind_bf16(".router.proj.weight",
                                      &weights.router_projection_bf16);
  if (status.ok()) status = bind_bf16(".router.per_expert_scale",
                                      &weights.per_expert_scale_bf16);
  if (status.ok()) status = bind_bf16(".layer_scalar",
                                      &weights.layer_scalar_bf16);
  auto shared_gate = Matrix(artifact.value(), prefix + ".mlp.gate_proj",
                            kShared, kWidth);
  auto shared_up = Matrix(artifact.value(), prefix + ".mlp.up_proj",
                          kShared, kWidth);
  auto shared_down = Matrix(artifact.value(), prefix + ".mlp.down_proj",
                            kWidth, kShared);
  auto expert_gate_up = Matrix(artifact.value(), prefix + ".experts.gate_up_proj",
                               kExperts * 2U * kExpert, kWidth);
  auto expert_down = Matrix(artifact.value(), prefix + ".experts.down_proj",
                            kExperts * kWidth, kExpert);
  if (!status.ok() || !shared_gate.ok() || !shared_up.ok() ||
      !shared_down.ok() || !expert_gate_up.ok() || !expert_down.ok()) {
    std::cerr << "error: cannot bind M11 layer tensors\n";
    return 5;
  }
  weights.shared_gate = shared_gate.value();
  weights.shared_up = shared_up.value();
  weights.shared_down = shared_down.value();
  weights.expert_gate_up = expert_gate_up.value();
  weights.expert_down = expert_down.value();

  Buffer d_hidden, d_output, shared_input, shared_input_packed,
      shared_input_scales, shared_gate_out, shared_up_out, shared_product,
      shared_product_packed, shared_product_scales, shared_output, shared_post,
      router_normalized, router_transformed, router_logits,
      router_probabilities, top_ids, top_weights, expert_input,
      expert_input_packed, expert_input_scales, expert_gate_up_out,
      expert_product, expert_product_packed, expert_product_scales,
      expert_down_out, expert_contributions, routed_sum, routed_post, combined,
      feed_forward;
  auto allocate = [&](Buffer* buffer, std::uint64_t bytes) {
    if (status.ok()) status = buffer->Allocate(bytes);
  };
  allocate(&d_hidden, kWidth * sizeof(float));
  allocate(&d_output, kWidth * sizeof(float));
  allocate(&shared_input, kWidth * sizeof(float));
  allocate(&shared_input_packed, kWidth / 2U);
  allocate(&shared_input_scales, kWidth / 16U);
  allocate(&shared_gate_out, kShared * sizeof(float));
  allocate(&shared_up_out, kShared * sizeof(float));
  allocate(&shared_product, kShared * sizeof(float));
  allocate(&shared_product_packed, kShared / 2U);
  allocate(&shared_product_scales, kShared / 16U);
  allocate(&shared_output, kWidth * sizeof(float));
  allocate(&shared_post, kWidth * sizeof(float));
  allocate(&router_normalized, kWidth * sizeof(float));
  allocate(&router_transformed, kWidth * sizeof(float));
  allocate(&router_logits, kExperts * sizeof(float));
  allocate(&router_probabilities, kExperts * sizeof(float));
  allocate(&top_ids, kTopK * sizeof(std::uint32_t));
  allocate(&top_weights, kTopK * sizeof(float));
  allocate(&expert_input, kWidth * sizeof(float));
  allocate(&expert_input_packed, kWidth / 2U);
  allocate(&expert_input_scales, kWidth / 16U);
  allocate(&expert_gate_up_out, kTopK * 2U * kExpert * sizeof(float));
  allocate(&expert_product, kTopK * kExpert * sizeof(float));
  allocate(&expert_product_packed, kTopK * kExpert / 2U);
  allocate(&expert_product_scales, kTopK * kExpert / 16U);
  allocate(&expert_down_out, kTopK * kWidth * sizeof(float));
  allocate(&expert_contributions, kTopK * kWidth * sizeof(float));
  allocate(&routed_sum, kWidth * sizeof(float));
  allocate(&routed_post, kWidth * sizeof(float));
  allocate(&combined, kWidth * sizeof(float));
  allocate(&feed_forward, kWidth * sizeof(float));
  if (!status.ok()) {
    std::cerr << "error: " << status.message() << '\n';
    return 6;
  }
  if (cudaMemcpy(d_hidden.As<float>(), hidden.value().data(),
                 hidden.value().size() * sizeof(float),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    std::cerr << "error: copy M11 hidden state\n";
    return 6;
  }
  gem16::internal::Gemma4MoeReferenceWorkspace workspace{
      shared_input.As<float>(), shared_input_packed.As<std::uint8_t>(),
      shared_input_scales.As<std::uint8_t>(), shared_gate_out.As<float>(),
      shared_up_out.As<float>(), shared_product.As<float>(),
      shared_product_packed.As<std::uint8_t>(),
      shared_product_scales.As<std::uint8_t>(), shared_output.As<float>(),
      shared_post.As<float>(), router_normalized.As<float>(),
      router_transformed.As<float>(), router_logits.As<float>(),
      router_probabilities.As<float>(), top_ids.As<std::uint32_t>(),
      top_weights.As<float>(), expert_input.As<float>(),
      expert_input_packed.As<std::uint8_t>(),
      expert_input_scales.As<std::uint8_t>(), expert_gate_up_out.As<float>(),
      expert_product.As<float>(), expert_product_packed.As<std::uint8_t>(),
      expert_product_scales.As<std::uint8_t>(), expert_down_out.As<float>(),
      expert_contributions.As<float>(), routed_sum.As<float>(),
      routed_post.As<float>(), combined.As<float>(), feed_forward.As<float>()};
  const gem16::internal::Gemma4MoeReferenceConfig config{
      kWidth, kShared, kExpert, kExperts, kTopK, 1.0e-6F};
  status = gem16::internal::LaunchGemma4MoeReferenceLayer(
      d_hidden.As<float>(), d_output.As<float>(), config, weights, workspace,
      nullptr);
  if (!status.ok() || cudaDeviceSynchronize() != cudaSuccess) {
    std::cerr << "error: M11 layer execution failed: " << status.message() << '\n';
    return 7;
  }
  std::vector<float> output_values, probability_values, weight_values,
      shared_gate_values, shared_up_values, shared_product_values,
      shared_values, contribution_values, routed_values;
  std::vector<std::uint32_t> id_values;
  status = Copy(&output_values, d_output, kWidth);
  if (status.ok()) status = Copy(&probability_values, router_probabilities, kExperts);
  if (status.ok()) status = Copy(&weight_values, top_weights, kTopK);
  if (status.ok()) status = Copy(&shared_gate_values, shared_gate_out, kShared);
  if (status.ok()) status = Copy(&shared_up_values, shared_up_out, kShared);
  if (status.ok()) status = Copy(&shared_product_values, shared_product, kShared);
  if (status.ok()) status = Copy(&shared_values, shared_output, kWidth);
  if (status.ok()) status = Copy(&contribution_values, expert_contributions,
                                 kTopK * kWidth);
  if (status.ok()) status = Copy(&routed_values, routed_sum, kWidth);
  if (status.ok()) status = Copy(&id_values, top_ids, kTopK);
  if (!status.ok()) {
    std::cerr << "error: " << status.message() << '\n';
    return 7;
  }
  std::size_t free_before_repeats = 0U;
  std::size_t visible_total = 0U;
  if (cudaMemGetInfo(&free_before_repeats, &visible_total) != cudaSuccess) {
    std::cerr << "error: cannot measure M11 memory before repeats\n";
    return 7;
  }
  for (int repeat = 0; repeat < 3; ++repeat) {
    status = gem16::internal::LaunchGemma4MoeReferenceLayer(
        d_hidden.As<float>(), d_output.As<float>(), config, weights, workspace,
        nullptr);
    if (!status.ok()) break;
  }
  if (!status.ok() || cudaDeviceSynchronize() != cudaSuccess) {
    std::cerr << "error: repeated M11 execution failed\n";
    return 7;
  }
  std::vector<float> repeated_output, repeated_contributions;
  std::vector<std::uint32_t> repeated_ids;
  status = Copy(&repeated_output, d_output, kWidth);
  if (status.ok()) status = Copy(&repeated_contributions, expert_contributions,
                                 kTopK * kWidth);
  if (status.ok()) status = Copy(&repeated_ids, top_ids, kTopK);
  std::size_t free_after_repeats = 0U;
  if (!status.ok() ||
      cudaMemGetInfo(&free_after_repeats, &visible_total) != cudaSuccess) {
    std::cerr << "error: cannot capture repeated M11 execution\n";
    return 7;
  }
  const bool deterministic = repeated_output == output_values &&
                             repeated_contributions == contribution_values &&
                             repeated_ids == id_values;
  const bool allocation_free = free_before_repeats == free_after_repeats;
  std::ofstream report(options.output, std::ios::binary | std::ios::trunc);
  if (!report) return 8;
  report << "{\"schema_version\":1,\"milestone\":\"M11\","
         << "\"path\":\"cuda_correctness_only\",\"layer\":" << layer
         << ",\"artifact_arena_bytes\":" << artifact.value().arena_bytes()
         << ",\"repeated_bitwise_identical\":"
         << (deterministic ? "true" : "false")
         << ",\"forward_allocation_free\":"
         << (allocation_free ? "true" : "false")
         << ",\"free_before_repeats_bytes\":" << free_before_repeats
         << ",\"free_after_repeats_bytes\":" << free_after_repeats
         << ",\"top_ids\":[";
  for (std::size_t index = 0; index < id_values.size(); ++index) {
    if (index != 0U) report << ',';
    report << id_values[index];
  }
  report << "],\"top_weights\":";
  WriteFloatArray(report, weight_values);
  report << ",\"router_probabilities\":";
  WriteFloatArray(report, probability_values);
  report << ",\"shared_gate\":";
  WriteFloatArray(report, shared_gate_values);
  report << ",\"shared_up\":";
  WriteFloatArray(report, shared_up_values);
  report << ",\"shared_product\":";
  WriteFloatArray(report, shared_product_values);
  report << ",\"shared_output\":";
  WriteFloatArray(report, shared_values);
  report << ",\"expert_contributions\":[";
  for (std::uint32_t slot = 0; slot < kTopK; ++slot) {
    if (slot != 0U) report << ',';
    const auto begin = contribution_values.begin() + slot * kWidth;
    WriteFloatArray(report, std::vector<float>(begin, begin + kWidth));
  }
  report << "],\"routed_sum\":";
  WriteFloatArray(report, routed_values);
  report << ",\"output\":";
  WriteFloatArray(report, output_values);
  report << "}\n";
  return report ? 0 : 8;
}

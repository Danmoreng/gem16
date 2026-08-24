#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cuda/engine/gemma4_26b_artifact.h"
#include "cuda/moe/router_diagnostic.h"
#include "gem16/model.h"
#include "model/config.h"
#include "model/gemma4_26b_residency.h"
#include "util/json.h"

namespace {

constexpr std::uint64_t kWidth = 2816U;
constexpr std::uint32_t kExperts = 128U;
constexpr std::uint32_t kTopK = 8U;

struct Options {
  std::filesystem::path model;
  std::filesystem::path output;
  std::vector<std::filesystem::path> fixtures;
  int device = 0;
};

bool Parse(int argc, char** argv, Options* options) {
  for (int index = 1; index < argc; ++index) {
    if (index + 1 >= argc) return false;
    const std::string_view argument(argv[index]);
    const std::string value(argv[++index]);
    if (argument == "--model") options->model = value;
    else if (argument == "--output") options->output = value;
    else if (argument == "--fixture") options->fixtures.emplace_back(value);
    else if (argument == "--device") {
      try {
        std::size_t consumed = 0U;
        options->device = std::stoi(value, &consumed);
        if (consumed != value.size() || options->device < 0) return false;
      } catch (...) {
        return false;
      }
    } else {
      return false;
    }
  }
  return !options->model.empty() && !options->output.empty() &&
         !options->fixtures.empty();
}

gem16::Result<std::string> Read(const std::filesystem::path& path,
                                std::uint64_t maximum) {
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  const auto size = std::filesystem::file_size(path, error);
  if (error || !std::filesystem::is_regular_file(status) ||
      std::filesystem::is_symlink(status) || size == 0U || size > maximum ||
      size > static_cast<std::uint64_t>(
                 std::numeric_limits<std::size_t>::max())) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         "router fixture is missing, unsafe, or too large");
  }
  std::string payload(static_cast<std::size_t>(size), '\0');
  std::ifstream input(path, std::ios::binary);
  if (!input.read(payload.data(),
                  static_cast<std::streamsize>(payload.size()))) {
    return gem16::Status(gem16::StatusCode::kIoError,
                         "cannot read router fixture");
  }
  return payload;
}

gem16::Result<std::vector<float>> FloatArray(const gem16::json::Value* value,
                                             std::size_t expected,
                                             std::string_view name) {
  if (value == nullptr || !value->is_array() ||
      value->as_array().size() != expected) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         std::string(name) + " has invalid shape");
  }
  std::vector<float> result;
  result.reserve(expected);
  for (const auto& element : value->as_array()) {
    if (!element.is_number() || !std::isfinite(element.as_number()) ||
        element.as_number() < -std::numeric_limits<float>::max() ||
        element.as_number() > std::numeric_limits<float>::max()) {
      return gem16::Status(gem16::StatusCode::kDataLoss,
                           std::string(name) + " contains invalid data");
    }
    result.push_back(static_cast<float>(element.as_number()));
  }
  return result;
}

gem16::Result<std::vector<std::uint32_t>> IdArray(
    const gem16::json::Value* value, std::size_t expected) {
  if (value == nullptr || !value->is_array() ||
      value->as_array().size() != expected) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         "trusted router IDs have invalid shape");
  }
  std::vector<std::uint32_t> result;
  result.reserve(expected);
  for (const auto& element : value->as_array()) {
    if (!element.is_integer() || element.as_integer() < 0 ||
        element.as_integer() >= kExperts) {
      return gem16::Status(gem16::StatusCode::kDataLoss,
                           "trusted router IDs contain invalid data");
    }
    result.push_back(static_cast<std::uint32_t>(element.as_integer()));
  }
  return result;
}

class Buffer {
 public:
  Buffer() = default;
  ~Buffer() {
    if (pointer_ != nullptr) (void)cudaFree(pointer_);
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  gem16::Status Allocate(std::uint64_t bytes) {
    if (bytes == 0U || bytes > std::numeric_limits<std::size_t>::max()) {
      return {gem16::StatusCode::kInvalidArgument,
              "invalid router diagnostic allocation"};
    }
    const cudaError_t error =
        cudaMalloc(&pointer_, static_cast<std::size_t>(bytes));
    if (error != cudaSuccess) {
      return {gem16::StatusCode::kResourceExhausted,
              std::string("allocate router diagnostic: ") +
                  cudaGetErrorString(error)};
    }
    return gem16::Status::Ok();
  }
  template <typename T>
  T* As() const {
    return static_cast<T*>(pointer_);
  }

 private:
  void* pointer_ = nullptr;
};

float DecodeBf16(std::uint16_t value) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

std::vector<std::uint32_t> TopIds(std::span<const float> logits) {
  std::vector<std::uint32_t> ids(kExperts);
  std::iota(ids.begin(), ids.end(), 0U);
  std::stable_sort(ids.begin(), ids.end(), [&](std::uint32_t left,
                                               std::uint32_t right) {
    return logits[left] > logits[right] ||
           (logits[left] == logits[right] && left < right);
  });
  ids.resize(kTopK);
  return ids;
}

std::vector<double> Softmax(std::span<const float> logits) {
  const float maximum = *std::max_element(logits.begin(), logits.end());
  std::vector<double> result(logits.size());
  double total = 0.0;
  for (std::size_t index = 0U; index < logits.size(); ++index) {
    result[index] = std::exp(static_cast<double>(logits[index] - maximum));
    total += result[index];
  }
  for (double& value : result) value /= total;
  return result;
}

std::vector<double> GatingVector(
    std::span<const float> logits, std::span<const std::uint16_t> scales) {
  const auto probabilities = Softmax(logits);
  const auto ids = TopIds(logits);
  double selected_sum = 0.0;
  for (const std::uint32_t id : ids) selected_sum += probabilities[id];
  std::vector<double> result(kExperts, 0.0);
  for (const std::uint32_t id : ids) {
    result[id] = probabilities[id] / selected_sum * DecodeBf16(scales[id]);
  }
  return result;
}

double MaximumAbsolute(std::span<const float> left,
                       std::span<const float> right) {
  double maximum = 0.0;
  for (std::size_t index = 0U; index < left.size(); ++index) {
    maximum = std::max(maximum, std::abs(static_cast<double>(left[index]) -
                                        static_cast<double>(right[index])));
  }
  return maximum;
}

double L1(std::span<const double> left, std::span<const double> right) {
  double total = 0.0;
  for (std::size_t index = 0U; index < left.size(); ++index) {
    total += std::abs(left[index] - right[index]);
  }
  return total;
}

struct ErrorSummary {
  double maximum = 0.0;
  double mean = 0.0;
  double rms = 0.0;
};

ErrorSummary OracleError(std::span<const float> values,
                         std::span<const double> oracle) {
  ErrorSummary result;
  double absolute_sum = 0.0;
  double squared_sum = 0.0;
  for (std::size_t index = 0U; index < values.size(); ++index) {
    const double error = std::abs(static_cast<double>(values[index]) -
                                  oracle[index]);
    result.maximum = std::max(result.maximum, error);
    absolute_sum += error;
    squared_sum += error * error;
  }
  result.mean = absolute_sum / static_cast<double>(values.size());
  result.rms = std::sqrt(squared_sum / static_cast<double>(values.size()));
  return result;
}

std::size_t SetOverlap(std::span<const std::uint32_t> left,
                       std::span<const std::uint32_t> right) {
  std::set<std::uint32_t> values(left.begin(), left.end());
  return static_cast<std::size_t>(std::count_if(
      right.begin(), right.end(),
      [&](std::uint32_t value) { return values.contains(value); }));
}

template <typename T>
gem16::Status CopyFromDevice(std::vector<T>* destination, const T* source,
                             std::size_t elements) {
  destination->resize(elements);
  const cudaError_t error = cudaMemcpy(destination->data(), source,
                                       elements * sizeof(T),
                                       cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    return {gem16::StatusCode::kInternal,
            std::string("copy router diagnostic: ") +
                cudaGetErrorString(error)};
  }
  return gem16::Status::Ok();
}

template <typename T>
gem16::Result<const T*> Pointer(
    const gem16::internal::Gemma4Moe26BDeviceArtifact& artifact,
    const std::string& name) {
  auto pointer = artifact.Pointer(name);
  if (!pointer.ok()) return pointer.status();
  return reinterpret_cast<const T*>(pointer.value());
}

struct CaseResult {
  std::uint32_t layer = 0U;
  std::uint64_t position = 0U;
  std::vector<std::uint32_t> trusted_ids;
  std::vector<std::uint32_t> serial_ids;
  std::vector<std::uint32_t> tensor_ids;
  ErrorSummary serial_oracle;
  ErrorSummary tensor_oracle;
  double bf16_logit_max_abs = 0.0;
  double gating_l1 = 0.0;
  double serial_margin_8_9 = 0.0;
  double tensor_margin_8_9 = 0.0;
  double serial_trusted_probability_max_abs = 0.0;
  double tensor_trusted_probability_max_abs = 0.0;
};

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!Parse(argc, argv, &options)) {
    std::cerr << "usage: gem16-26b-router-diagnostic --model DIR "
                 "--fixture JSON [--fixture JSON ...] --output JSON "
                 "[--device N]\n";
    return 2;
  }
  if (cudaSetDevice(options.device) != cudaSuccess) {
    std::cerr << "error: cannot select CUDA device\n";
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
    std::cerr << "error: " << model_config.status().message() << '\n';
    return 4;
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

  Buffer normalized_device, transformed_device, serial_raw_device,
      serial_bf16_device, tensor_raw_device, tensor_bf16_device;
  gem16::Status status = normalized_device.Allocate(kWidth * sizeof(float));
  if (status.ok()) {
    status = transformed_device.Allocate(kWidth * sizeof(float));
  }
  constexpr std::uint64_t kLogitBytes = kExperts * sizeof(float);
  for (Buffer* buffer : {&serial_raw_device, &serial_bf16_device,
                         &tensor_raw_device, &tensor_bf16_device}) {
    if (status.ok()) status = buffer->Allocate(kLogitBytes);
  }
  if (!status.ok()) {
    std::cerr << "error: " << status.message() << '\n';
    return 6;
  }
  const gem16::internal::Gemma4RouterDiagnosticWorkspace workspace{
      transformed_device.As<float>(), serial_raw_device.As<float>(),
      serial_bf16_device.As<float>(), tensor_raw_device.As<float>(),
      tensor_bf16_device.As<float>()};

  std::vector<CaseResult> results;
  results.reserve(options.fixtures.size());
  for (const auto& fixture_path : options.fixtures) {
    auto text = Read(fixture_path, 4U * 1024U * 1024U);
    auto document = text.ok() ? gem16::json::Parse(text.value())
                              : gem16::Result<gem16::json::Value>(text.status());
    const auto* source = document.ok() ? document.value().find("source")
                                       : nullptr;
    const auto* expected = document.ok() ? document.value().find("expected")
                                         : nullptr;
    const auto* layer_value = source != nullptr ? source->find("layer") : nullptr;
    const auto* position_value =
        source != nullptr ? source->find("position") : nullptr;
    if (!document.ok() || source == nullptr || !source->is_object() ||
        expected == nullptr || !expected->is_object() ||
        layer_value == nullptr || !layer_value->is_integer() ||
        layer_value->as_integer() < 0 || layer_value->as_integer() >= 30 ||
        position_value == nullptr || !position_value->is_integer() ||
        position_value->as_integer() < 0) {
      std::cerr << "error: invalid router fixture: " << fixture_path << '\n';
      return 7;
    }
    auto normalized = FloatArray(document.value().find("hidden_f32"), kWidth,
                                 "router normalized input");
    auto trusted_probabilities =
        FloatArray(expected->find("router_probabilities"), kExperts,
                   "trusted router probabilities");
    auto trusted_ids = IdArray(expected->find("top_ids"), kTopK);
    if (!normalized.ok() || !trusted_probabilities.ok() ||
        !trusted_ids.ok()) {
      const gem16::Status failure = !normalized.ok()
                                        ? normalized.status()
                                        : (!trusted_probabilities.ok()
                                               ? trusted_probabilities.status()
                                               : trusted_ids.status());
      std::cerr << "error: " << failure.message() << '\n';
      return 7;
    }
    const std::uint32_t layer =
        static_cast<std::uint32_t>(layer_value->as_integer());
    const std::string prefix = "model.language_model.layers." +
                               std::to_string(layer) + ".router.";
    auto scale = Pointer<std::uint16_t>(artifact.value(), prefix + "scale");
    auto projection =
        Pointer<std::uint16_t>(artifact.value(), prefix + "proj.weight");
    auto expert_scale = Pointer<std::uint16_t>(
        artifact.value(), prefix + "per_expert_scale");
    if (!scale.ok() || !projection.ok() || !expert_scale.ok()) {
      std::cerr << "error: cannot bind BF16 router tensors for layer "
                << layer << '\n';
      return 7;
    }
    cudaError_t error = cudaMemcpy(normalized_device.As<float>(),
                                   normalized.value().data(),
                                   kWidth * sizeof(float),
                                   cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
      std::cerr << "error: cannot copy router fixture\n";
      return 7;
    }
    status = gem16::internal::LaunchGemma4RouterProjectionDiagnostic(
        normalized_device.As<float>(), scale.value(), projection.value(),
        workspace, 1U, kWidth, kExperts, nullptr);
    if (!status.ok() || cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "error: router diagnostic launch failed: "
                << status.message() << '\n';
      return 8;
    }

    std::vector<float> transformed, serial_raw, serial_bf16, tensor_raw,
        tensor_bf16;
    std::vector<std::uint16_t> projection_host, expert_scale_host;
    status = CopyFromDevice(&transformed, transformed_device.As<float>(),
                            kWidth);
    if (status.ok()) {
      status = CopyFromDevice(&serial_raw, serial_raw_device.As<float>(),
                              kExperts);
    }
    if (status.ok()) {
      status = CopyFromDevice(&serial_bf16, serial_bf16_device.As<float>(),
                              kExperts);
    }
    if (status.ok()) {
      status = CopyFromDevice(&tensor_raw, tensor_raw_device.As<float>(),
                              kExperts);
    }
    if (status.ok()) {
      status = CopyFromDevice(&tensor_bf16, tensor_bf16_device.As<float>(),
                              kExperts);
    }
    if (status.ok()) {
      status = CopyFromDevice(&projection_host, projection.value(),
                              static_cast<std::size_t>(kExperts * kWidth));
    }
    if (status.ok()) {
      status = CopyFromDevice(&expert_scale_host, expert_scale.value(),
                              kExperts);
    }
    if (!status.ok()) {
      std::cerr << "error: " << status.message() << '\n';
      return 8;
    }

    std::vector<double> oracle(kExperts, 0.0);
    for (std::uint32_t expert = 0U; expert < kExperts; ++expert) {
      double accumulator = 0.0;
      const std::uint64_t base =
          static_cast<std::uint64_t>(expert) * kWidth;
      for (std::uint64_t column = 0U; column < kWidth; ++column) {
        accumulator +=
            static_cast<double>(DecodeBf16(projection_host[base + column])) *
            static_cast<double>(transformed[column]);
      }
      oracle[expert] = accumulator;
    }
    const auto serial_ids = TopIds(serial_bf16);
    const auto tensor_ids = TopIds(tensor_bf16);
    const auto serial_probabilities = Softmax(serial_bf16);
    const auto tensor_probabilities = Softmax(tensor_bf16);
    const auto serial_gating = GatingVector(serial_bf16, expert_scale_host);
    const auto tensor_gating = GatingVector(tensor_bf16, expert_scale_host);
    std::vector<std::uint32_t> serial_order(kExperts), tensor_order(kExperts);
    std::iota(serial_order.begin(), serial_order.end(), 0U);
    std::iota(tensor_order.begin(), tensor_order.end(), 0U);
    const auto order = [](std::span<const float> logits) {
      return [logits](std::uint32_t left, std::uint32_t right) {
        return logits[left] > logits[right] ||
               (logits[left] == logits[right] && left < right);
      };
    };
    std::stable_sort(serial_order.begin(), serial_order.end(),
                     order(serial_bf16));
    std::stable_sort(tensor_order.begin(), tensor_order.end(),
                     order(tensor_bf16));
    double serial_trusted_probability_max_abs = 0.0;
    double tensor_trusted_probability_max_abs = 0.0;
    for (std::uint32_t expert = 0U; expert < kExperts; ++expert) {
      serial_trusted_probability_max_abs = std::max(
          serial_trusted_probability_max_abs,
          std::abs(serial_probabilities[expert] -
                   trusted_probabilities.value()[expert]));
      tensor_trusted_probability_max_abs = std::max(
          tensor_trusted_probability_max_abs,
          std::abs(tensor_probabilities[expert] -
                   trusted_probabilities.value()[expert]));
    }
    results.push_back(CaseResult{
        layer,
        static_cast<std::uint64_t>(position_value->as_integer()),
        std::move(trusted_ids.value()),
        serial_ids,
        tensor_ids,
        OracleError(serial_raw, oracle),
        OracleError(tensor_raw, oracle),
        MaximumAbsolute(serial_bf16, tensor_bf16),
        L1(serial_gating, tensor_gating),
        static_cast<double>(serial_bf16[serial_order[7U]]) -
            static_cast<double>(serial_bf16[serial_order[8U]]),
        static_cast<double>(tensor_bf16[tensor_order[7U]]) -
            static_cast<double>(tensor_bf16[tensor_order[8U]]),
        serial_trusted_probability_max_abs,
        tensor_trusted_probability_max_abs});
  }

  std::size_t set_matches = 0U;
  std::size_t order_matches = 0U;
  std::size_t serial_trusted_exact = 0U;
  std::size_t tensor_trusted_exact = 0U;
  double maximum_bf16_logit_delta = 0.0;
  double maximum_gating_l1 = 0.0;
  double serial_oracle_rms_sum = 0.0;
  double tensor_oracle_rms_sum = 0.0;
  for (const auto& result : results) {
    set_matches += SetOverlap(result.serial_ids, result.tensor_ids) == kTopK;
    order_matches += result.serial_ids == result.tensor_ids;
    serial_trusted_exact +=
        SetOverlap(result.serial_ids, result.trusted_ids) == kTopK;
    tensor_trusted_exact +=
        SetOverlap(result.tensor_ids, result.trusted_ids) == kTopK;
    maximum_bf16_logit_delta =
        std::max(maximum_bf16_logit_delta, result.bf16_logit_max_abs);
    maximum_gating_l1 = std::max(maximum_gating_l1, result.gating_l1);
    serial_oracle_rms_sum += result.serial_oracle.rms;
    tensor_oracle_rms_sum += result.tensor_oracle.rms;
  }

  std::ofstream output(options.output, std::ios::binary | std::ios::trunc);
  if (!output) return 9;
  output << std::setprecision(12)
         << "{\"schema_version\":1,\"status\":\"diagnostic_only\","
         << "\"source\":\"locked_qat_bf16_selected_captures\","
         << "\"cases\":" << results.size()
         << ",\"summary\":{\"top8_set_matches\":" << set_matches
         << ",\"top8_order_matches\":" << order_matches
         << ",\"serial_trusted_top8_set_matches\":"
         << serial_trusted_exact
         << ",\"tensor_trusted_top8_set_matches\":"
         << tensor_trusted_exact
         << ",\"maximum_bf16_logit_absolute_delta\":"
         << maximum_bf16_logit_delta
         << ",\"maximum_gating_vector_l1\":" << maximum_gating_l1
         << ",\"serial_mean_fp64_rms_error\":"
         << serial_oracle_rms_sum / static_cast<double>(results.size())
         << ",\"tensor_mean_fp64_rms_error\":"
         << tensor_oracle_rms_sum / static_cast<double>(results.size())
         << "},\"records\":[";
  for (std::size_t index = 0U; index < results.size(); ++index) {
    if (index != 0U) output << ',';
    const auto& result = results[index];
    output << "{\"layer\":" << result.layer
           << ",\"position\":" << result.position
           << ",\"serial_tensor_top8_overlap\":"
           << SetOverlap(result.serial_ids, result.tensor_ids)
           << ",\"serial_trusted_top8_overlap\":"
           << SetOverlap(result.serial_ids, result.trusted_ids)
           << ",\"tensor_trusted_top8_overlap\":"
           << SetOverlap(result.tensor_ids, result.trusted_ids)
           << ",\"serial_margin_8_9\":" << result.serial_margin_8_9
           << ",\"tensor_margin_8_9\":" << result.tensor_margin_8_9
           << ",\"bf16_logit_max_abs\":" << result.bf16_logit_max_abs
           << ",\"gating_vector_l1\":" << result.gating_l1
           << ",\"serial_fp64_error\":{\"max\":"
           << result.serial_oracle.maximum << ",\"mean\":"
           << result.serial_oracle.mean << ",\"rms\":"
           << result.serial_oracle.rms
           << "},\"tensor_fp64_error\":{\"max\":"
           << result.tensor_oracle.maximum << ",\"mean\":"
           << result.tensor_oracle.mean << ",\"rms\":"
           << result.tensor_oracle.rms
           << "},\"serial_trusted_probability_max_abs\":"
           << result.serial_trusted_probability_max_abs
           << ",\"tensor_trusted_probability_max_abs\":"
           << result.tensor_trusted_probability_max_abs << '}';
  }
  output << "]}\n";
  return output ? 0 : 9;
}

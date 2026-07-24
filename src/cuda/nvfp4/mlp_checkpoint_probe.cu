#include "cuda/nvfp4/checkpoint_probe.h"

#include "cuda/nvfp4/mlp.h"
#include "cuda/nvfp4/reference.h"
#include "cuda/nvfp4/sm120.h"
#include "gem16gb/model.h"
#include "gem16gb/nvfp4.h"
#include "platform/mapped_file.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gem16gb::internal {
namespace {

constexpr char kLayer0MlpBase[] = "model.language_model.layers.0.mlp.";
constexpr char kInstruction[] = "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X";

Status ProbeError(StatusCode code, std::string message) {
  return Status(code, std::move(message));
}

Status CudaFailure(const char* operation, cudaError_t error) {
  return ProbeError(StatusCode::kInternal,
                    std::string(operation) + ": " + cudaGetErrorName(error) + ": " +
                        cudaGetErrorString(error));
}

const TensorInfo* FindTensor(const ModelManifest& manifest, const std::string& name) {
  const auto found = std::find_if(manifest.tensors.begin(), manifest.tensors.end(),
                                  [&](const TensorInfo& tensor) { return tensor.name == name; });
  return found == manifest.tensors.end() ? nullptr : &*found;
}

Result<std::span<const std::uint8_t>> TensorBytes(const MappedFile& file,
                                                  const TensorInfo& tensor) {
  if (tensor.byte_offset > file.size() || tensor.byte_length > file.size() - tensor.byte_offset ||
      tensor.byte_length > std::numeric_limits<std::size_t>::max()) {
    return ProbeError(StatusCode::kDataLoss,
                      "tensor range is outside its mapped shard: " + tensor.name);
  }
  const auto* begin = reinterpret_cast<const std::uint8_t*>(file.data() + tensor.byte_offset);
  return std::span<const std::uint8_t>(begin, static_cast<std::size_t>(tensor.byte_length));
}

Result<float> ScalarF32(const MappedFile& file, const TensorInfo& tensor) {
  const auto bytes = TensorBytes(file, tensor);
  if (!bytes.ok()) return bytes.status();
  if (bytes.value().size() != sizeof(float)) {
    return ProbeError(StatusCode::kDataLoss, "expected scalar F32 tensor: " + tensor.name);
  }
  const std::uint32_t word = static_cast<std::uint32_t>(bytes.value()[0]) |
                             (static_cast<std::uint32_t>(bytes.value()[1]) << 8U) |
                             (static_cast<std::uint32_t>(bytes.value()[2]) << 16U) |
                             (static_cast<std::uint32_t>(bytes.value()[3]) << 24U);
  const float value = std::bit_cast<float>(word);
  if (!std::isfinite(value) || value <= 0.0F) {
    return ProbeError(StatusCode::kDataLoss,
                      "global divisor must be positive and finite: " + tensor.name);
  }
  return value;
}

struct ProjectionBinding {
  std::span<const std::uint8_t> packed_weight;
  std::span<const std::uint8_t> weight_scales;
  float input_divisor = 0.0F;
  float weight_divisor = 0.0F;
  std::uint64_t rows = 0;
  std::uint64_t contracting_elements = 0;
};

Result<ProjectionBinding> BindProjection(const ModelManifest& manifest, const MappedFile& mapped,
                                         std::string_view projection,
                                         std::vector<std::uint64_t> expected_shape) {
  const std::string base = std::string(kLayer0MlpBase) + std::string(projection) + "_proj";
  const TensorInfo* packed = FindTensor(manifest, base + ".weight_packed");
  const TensorInfo* scales = FindTensor(manifest, base + ".weight_scale");
  const TensorInfo* input_divisor = FindTensor(manifest, base + ".input_global_scale");
  const TensorInfo* weight_divisor = FindTensor(manifest, base + ".weight_global_scale");
  if (packed == nullptr || scales == nullptr || input_divisor == nullptr ||
      weight_divisor == nullptr) {
    return ProbeError(StatusCode::kNotFound,
                      "layer-0 " + std::string(projection) + " tensor family is incomplete");
  }
  if (packed->logical_shape != expected_shape ||
      scales->shape !=
          std::vector<std::uint64_t>{expected_shape[0], expected_shape[1] / 16U} ||
      packed->source_shard != "model.safetensors" ||
      scales->source_shard != packed->source_shard ||
      input_divisor->source_shard != packed->source_shard ||
      weight_divisor->source_shard != packed->source_shard) {
    return ProbeError(StatusCode::kDataLoss,
                      "layer-0 " + std::string(projection) +
                          " geometry or shard differs from the pinned contract");
  }
  auto packed_bytes = TensorBytes(mapped, *packed);
  auto scale_bytes = TensorBytes(mapped, *scales);
  auto input_value = ScalarF32(mapped, *input_divisor);
  auto weight_value = ScalarF32(mapped, *weight_divisor);
  if (!packed_bytes.ok()) return packed_bytes.status();
  if (!scale_bytes.ok()) return scale_bytes.status();
  if (!input_value.ok()) return input_value.status();
  if (!weight_value.ok()) return weight_value.status();
  return ProjectionBinding{packed_bytes.value(), scale_bytes.value(), input_value.value(),
                           weight_value.value(), expected_shape[0], expected_shape[1]};
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() {
    if (data_ != nullptr) (void)cudaFree(data_);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] Status Allocate(std::uint64_t elements, const char* label) {
    if (elements == 0U || elements > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      return ProbeError(StatusCode::kInvalidArgument, std::string(label) + " size is invalid");
    }
    elements_ = static_cast<std::size_t>(elements);
    const cudaError_t error = cudaMalloc(&data_, bytes());
    return error == cudaSuccess ? Status::Ok() : CudaFailure(label, error);
  }

  [[nodiscard]] T* get() const { return static_cast<T*>(data_); }
  [[nodiscard]] std::size_t bytes() const { return elements_ * sizeof(T); }

 private:
  void* data_ = nullptr;
  std::size_t elements_ = 0;
};

class CudaEvents {
 public:
  ~CudaEvents() {
    if (start_ != nullptr) (void)cudaEventDestroy(start_);
    if (stop_ != nullptr) (void)cudaEventDestroy(stop_);
  }
  [[nodiscard]] Status Create() {
    cudaError_t error = cudaEventCreate(&start_);
    if (error != cudaSuccess) return CudaFailure("cudaEventCreate(start)", error);
    error = cudaEventCreate(&stop_);
    return error == cudaSuccess ? Status::Ok() : CudaFailure("cudaEventCreate(stop)", error);
  }
  [[nodiscard]] cudaEvent_t start() const { return start_; }
  [[nodiscard]] cudaEvent_t stop() const { return stop_; }

 private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

template <typename Launch>
Result<double> Measure(std::uint32_t warmups, std::uint32_t iterations, Launch&& launch) {
  for (std::uint32_t iteration = 0; iteration < warmups; ++iteration) {
    const Status status = launch();
    if (!status.ok()) return status;
  }
  cudaError_t error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return CudaFailure("MLP warmup synchronize", error);
  CudaEvents events;
  const Status create_status = events.Create();
  if (!create_status.ok()) return create_status;
  error = cudaEventRecord(events.start());
  if (error != cudaSuccess) return CudaFailure("cudaEventRecord(start)", error);
  for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
    const Status status = launch();
    if (!status.ok()) return status;
  }
  error = cudaEventRecord(events.stop());
  if (error != cudaSuccess) return CudaFailure("cudaEventRecord(stop)", error);
  error = cudaEventSynchronize(events.stop());
  if (error != cudaSuccess) return CudaFailure("cudaEventSynchronize(stop)", error);
  float milliseconds = 0.0F;
  error = cudaEventElapsedTime(&milliseconds, events.start(), events.stop());
  if (error != cudaSuccess) return CudaFailure("cudaEventElapsedTime", error);
  return static_cast<double>(milliseconds) / static_cast<double>(iterations);
}

std::vector<float> DeterministicActivation(std::size_t elements) {
  std::vector<float> activation(elements);
  for (std::size_t index = 0; index < elements; ++index) {
    const double position = static_cast<double>(index);
    const double ripple = static_cast<double>(static_cast<int>(index % 17U) - 8) * 0.005;
    activation[index] = static_cast<float>(0.25 * std::sin(position * 0.013) +
                                           0.125 * std::cos(position * 0.007) + ripple);
  }
  return activation;
}

Status CopyToDevice(void* destination, std::span<const std::uint8_t> source,
                    const char* label) {
  const cudaError_t error =
      cudaMemcpy(destination, source.data(), source.size(), cudaMemcpyHostToDevice);
  return error == cudaSuccess ? Status::Ok() : CudaFailure(label, error);
}

}  // namespace

Result<Nvfp4MlpCheckpointProbeResult> RunLayer0Nvfp4MlpCheckpointProbe(
    const std::filesystem::path& model_directory, std::uint32_t warmups,
    std::uint32_t iterations) {
  if (iterations == 0U) {
    return ProbeError(StatusCode::kInvalidArgument, "MLP probe iterations must be nonzero");
  }
  auto manifest = InspectCheckpoint({model_directory, true});
  if (!manifest.ok()) return manifest.status();
  auto mapped = MappedFile::Open(model_directory / "model.safetensors");
  if (!mapped.ok()) return mapped.status();
  auto gate = BindProjection(manifest.value(), mapped.value(), "gate", {15360U, 3840U});
  auto up = BindProjection(manifest.value(), mapped.value(), "up", {15360U, 3840U});
  auto down = BindProjection(manifest.value(), mapped.value(), "down", {3840U, 15360U});
  if (!gate.ok()) return gate.status();
  if (!up.ok()) return up.status();
  if (!down.ok()) return down.status();
  if (std::bit_cast<std::uint32_t>(gate.value().input_divisor) !=
      std::bit_cast<std::uint32_t>(up.value().input_divisor)) {
    return ProbeError(StatusCode::kDataLoss,
                      "Layer-0 Gate and Up input global divisors are not identical");
  }

  cudaDeviceProp properties{};
  cudaError_t cuda_error = cudaGetDeviceProperties(&properties, 0);
  if (cuda_error != cudaSuccess) return CudaFailure("cudaGetDeviceProperties", cuda_error);
  if (properties.major != 12 || properties.minor != 0) {
    return ProbeError(StatusCode::kUnsupported, "MLP checkpoint probe requires an SM120 device");
  }

  constexpr std::uint64_t hidden = 3840;
  constexpr std::uint64_t intermediate = 15360;
  std::vector<float> host_input = DeterministicActivation(hidden);
  auto host_input_quantized = nvfp4::QuantizeActivation(host_input, gate.value().input_divisor);
  if (!host_input_quantized.ok()) return host_input_quantized.status();

  DeviceBuffer<float> input;
  DeviceBuffer<std::uint8_t> input_packed;
  DeviceBuffer<std::uint8_t> input_scales;
  DeviceBuffer<std::uint8_t> gate_weight;
  DeviceBuffer<std::uint8_t> gate_scales;
  DeviceBuffer<std::uint8_t> up_weight;
  DeviceBuffer<std::uint8_t> up_scales;
  DeviceBuffer<std::uint8_t> down_weight;
  DeviceBuffer<std::uint8_t> down_scales;
  DeviceBuffer<float> gate_output;
  DeviceBuffer<float> up_output;
  DeviceBuffer<float> product;
  DeviceBuffer<std::uint8_t> down_input_packed;
  DeviceBuffer<std::uint8_t> down_input_scales;
  DeviceBuffer<float> down_output;
  DeviceBuffer<float> final_output;
  for (const Status status : {
           input.Allocate(hidden, "allocate MLP input"),
           input_packed.Allocate(hidden / 2U, "allocate MLP input packed"),
           input_scales.Allocate(hidden / 16U, "allocate MLP input scales"),
           gate_weight.Allocate(gate.value().packed_weight.size(), "allocate Gate weight"),
           gate_scales.Allocate(gate.value().weight_scales.size(), "allocate Gate scales"),
           up_weight.Allocate(up.value().packed_weight.size(), "allocate Up weight"),
           up_scales.Allocate(up.value().weight_scales.size(), "allocate Up scales"),
           down_weight.Allocate(down.value().packed_weight.size(), "allocate Down weight"),
           down_scales.Allocate(down.value().weight_scales.size(), "allocate Down scales"),
           gate_output.Allocate(intermediate, "allocate Gate output"),
           up_output.Allocate(intermediate, "allocate Up output"),
           product.Allocate(intermediate, "allocate MLP product"),
           down_input_packed.Allocate(intermediate / 2U, "allocate Down input packed"),
           down_input_scales.Allocate(intermediate / 16U, "allocate Down input scales"),
           down_output.Allocate(hidden, "allocate Down output"),
           final_output.Allocate(hidden, "allocate MLP final output"),
       }) {
    if (!status.ok()) return status;
  }
  cuda_error = cudaMemcpy(input.get(), host_input.data(), input.bytes(), cudaMemcpyHostToDevice);
  if (cuda_error != cudaSuccess) return CudaFailure("copy MLP input", cuda_error);
  const std::array<Status, 6> copy_statuses = {
      CopyToDevice(gate_weight.get(), gate.value().packed_weight, "copy Gate weight"),
      CopyToDevice(gate_scales.get(), gate.value().weight_scales, "copy Gate scales"),
      CopyToDevice(up_weight.get(), up.value().packed_weight, "copy Up weight"),
      CopyToDevice(up_scales.get(), up.value().weight_scales, "copy Up scales"),
      CopyToDevice(down_weight.get(), down.value().packed_weight, "copy Down weight"),
      CopyToDevice(down_scales.get(), down.value().weight_scales, "copy Down scales"),
  };
  for (const Status& status : copy_statuses) {
    if (!status.ok()) return status;
  }

  const auto quantize_input = [&] {
    return LaunchNvfp4ReferenceActivationQuantization(
        input.get(), input_packed.get(), input_scales.get(), hidden,
        gate.value().input_divisor, nullptr);
  };
  const Status initial_quantize_status = quantize_input();
  if (!initial_quantize_status.ok()) return initial_quantize_status;
  cuda_error = cudaDeviceSynchronize();
  if (cuda_error != cudaSuccess) return CudaFailure("initial input quantization", cuda_error);
  std::vector<std::uint8_t> gpu_input_packed(input_packed.bytes());
  std::vector<std::uint8_t> gpu_input_scales(input_scales.bytes());
  cuda_error = cudaMemcpy(gpu_input_packed.data(), input_packed.get(), input_packed.bytes(),
                          cudaMemcpyDeviceToHost);
  if (cuda_error != cudaSuccess) return CudaFailure("copy MLP input packed", cuda_error);
  cuda_error = cudaMemcpy(gpu_input_scales.data(), input_scales.get(), input_scales.bytes(),
                          cudaMemcpyDeviceToHost);
  if (cuda_error != cudaSuccess) return CudaFailure("copy MLP input scales", cuda_error);
  const bool input_match =
      gpu_input_packed == host_input_quantized.value().packed_e2m1 &&
      gpu_input_scales == host_input_quantized.value().block_scales_e4m3fn;
  if (!input_match) {
    return ProbeError(StatusCode::kDataLoss, "CPU and CUDA MLP input quantization differ");
  }

  const auto reference_chain = [&] {
    Status status = quantize_input();
    if (!status.ok()) return status;
    status = LaunchNvfp4ReferenceProjection(
        input_packed.get(), input_scales.get(), gate_weight.get(), gate_scales.get(),
        gate_output.get(), intermediate, hidden, gate.value().input_divisor,
        gate.value().weight_divisor, nullptr);
    if (!status.ok()) return status;
    status = LaunchNvfp4ReferenceProjection(
        input_packed.get(), input_scales.get(), up_weight.get(), up_scales.get(), up_output.get(),
        intermediate, hidden, up.value().input_divisor, up.value().weight_divisor, nullptr);
    if (!status.ok()) return status;
    status = LaunchGeluTanhProduct(gate_output.get(), up_output.get(), product.get(), intermediate,
                                   nullptr);
    if (!status.ok()) return status;
    status = LaunchNvfp4ReferenceActivationQuantization(
        product.get(), down_input_packed.get(), down_input_scales.get(), intermediate,
        down.value().input_divisor, nullptr);
    if (!status.ok()) return status;
    status = LaunchNvfp4ReferenceProjection(
        down_input_packed.get(), down_input_scales.get(), down_weight.get(), down_scales.get(),
        down_output.get(), hidden, intermediate, down.value().input_divisor,
        down.value().weight_divisor, nullptr);
    if (!status.ok()) return status;
    return LaunchAddResidual(down_output.get(), input.get(), final_output.get(), hidden, nullptr);
  };
  auto reference_ms = Measure(1, 1, reference_chain);
  if (!reference_ms.ok()) return reference_ms.status();
  std::vector<float> reference_output(hidden);
  std::vector<float> reference_product(intermediate);
  std::vector<std::uint8_t> reference_down_packed(down_input_packed.bytes());
  std::vector<std::uint8_t> reference_down_scales(down_input_scales.bytes());
  cuda_error = cudaMemcpy(reference_output.data(), final_output.get(), final_output.bytes(),
                          cudaMemcpyDeviceToHost);
  if (cuda_error != cudaSuccess) return CudaFailure("copy reference MLP output", cuda_error);
  cuda_error = cudaMemcpy(reference_product.data(), product.get(), product.bytes(),
                          cudaMemcpyDeviceToHost);
  if (cuda_error != cudaSuccess) return CudaFailure("copy reference MLP product", cuda_error);
  cuda_error = cudaMemcpy(reference_down_packed.data(), down_input_packed.get(),
                          down_input_packed.bytes(), cudaMemcpyDeviceToHost);
  if (cuda_error != cudaSuccess) return CudaFailure("copy reference Down input", cuda_error);
  cuda_error = cudaMemcpy(reference_down_scales.data(), down_input_scales.get(),
                          down_input_scales.bytes(), cudaMemcpyDeviceToHost);
  if (cuda_error != cudaSuccess) return CudaFailure("copy reference Down scales", cuda_error);
  auto host_down_quantized =
      nvfp4::QuantizeActivation(reference_product, down.value().input_divisor);
  if (!host_down_quantized.ok()) return host_down_quantized.status();
  const bool reference_down_match =
      reference_down_packed == host_down_quantized.value().packed_e2m1 &&
      reference_down_scales == host_down_quantized.value().block_scales_e4m3fn;
  // Continue timing when a boundary value rounds differently in the CPU oracle. All measured
  // device paths consume the same CUDA-produced bytes and the mismatch remains explicit in the
  // result instead of invalidating the kernel comparison.

  const auto unfused_gate_up = [&] {
    Status status = LaunchNvfp4Sm120DirectProjection(
        input_packed.get(), input_scales.get(), gate_weight.get(), gate_scales.get(),
        gate_output.get(), intermediate, hidden, gate.value().input_divisor,
        gate.value().weight_divisor, nullptr);
    if (!status.ok()) return status;
    status = LaunchNvfp4Sm120DirectProjection(
        input_packed.get(), input_scales.get(), up_weight.get(), up_scales.get(),
        up_output.get(), intermediate, hidden, up.value().input_divisor,
        up.value().weight_divisor, nullptr);
    if (!status.ok()) return status;
    return LaunchGeluTanhProduct(gate_output.get(), up_output.get(), product.get(),
                                 intermediate, nullptr);
  };
  auto unfused_gate_up_ms = Measure(warmups, iterations, unfused_gate_up);
  if (!unfused_gate_up_ms.ok()) return unfused_gate_up_ms.status();

  const auto fused_gate_up = [&] {
    return LaunchNvfp4Sm120FusedGateUp(
        input_packed.get(), input_scales.get(), gate_weight.get(), gate_scales.get(),
        up_weight.get(), up_scales.get(), nullptr, nullptr, product.get(),
        intermediate, hidden, gate.value().input_divisor, gate.value().weight_divisor,
        up.value().input_divisor, up.value().weight_divisor, nullptr);
  };
  auto fused_gate_up_ms = Measure(warmups, iterations, fused_gate_up);
  if (!fused_gate_up_ms.ok()) return fused_gate_up_ms.status();

  const auto native_chain = [&] {
    Status status = quantize_input();
    if (!status.ok()) return status;
    status = LaunchNvfp4Sm120DirectProjection(
        input_packed.get(), input_scales.get(), gate_weight.get(), gate_scales.get(),
        gate_output.get(), intermediate, hidden, gate.value().input_divisor,
        gate.value().weight_divisor, nullptr);
    if (!status.ok()) return status;
    status = LaunchNvfp4Sm120DirectProjection(
        input_packed.get(), input_scales.get(), up_weight.get(), up_scales.get(), up_output.get(),
        intermediate, hidden, up.value().input_divisor, up.value().weight_divisor, nullptr);
    if (!status.ok()) return status;
    status = LaunchGeluTanhProduct(gate_output.get(), up_output.get(), product.get(), intermediate,
                                   nullptr);
    if (!status.ok()) return status;
    status = LaunchNvfp4ReferenceActivationQuantization(
        product.get(), down_input_packed.get(), down_input_scales.get(), intermediate,
        down.value().input_divisor, nullptr);
    if (!status.ok()) return status;
    status = LaunchNvfp4Sm120DirectProjection(
        down_input_packed.get(), down_input_scales.get(), down_weight.get(), down_scales.get(),
        down_output.get(), hidden, intermediate, down.value().input_divisor,
        down.value().weight_divisor, nullptr);
    if (!status.ok()) return status;
    return LaunchAddResidual(down_output.get(), input.get(), final_output.get(), hidden, nullptr);
  };
  auto native_ms = Measure(warmups, iterations, native_chain);
  if (!native_ms.ok()) return native_ms.status();
  std::vector<float> native_output(hidden);
  std::vector<std::uint8_t> native_down_packed(down_input_packed.bytes());
  std::vector<std::uint8_t> native_down_scales(down_input_scales.bytes());
  cuda_error = cudaMemcpy(native_output.data(), final_output.get(), final_output.bytes(),
                          cudaMemcpyDeviceToHost);
  if (cuda_error != cudaSuccess) return CudaFailure("copy native MLP output", cuda_error);
  cuda_error = cudaMemcpy(native_down_packed.data(), down_input_packed.get(),
                          down_input_packed.bytes(), cudaMemcpyDeviceToHost);
  if (cuda_error != cudaSuccess) return CudaFailure("copy native Down input", cuda_error);
  cuda_error = cudaMemcpy(native_down_scales.data(), down_input_scales.get(),
                          down_input_scales.bytes(), cudaMemcpyDeviceToHost);
  if (cuda_error != cudaSuccess) return CudaFailure("copy native Down scales", cuda_error);

  Nvfp4MlpCheckpointProbeResult result;
  result.instruction = kInstruction;
  result.input_activation_bytes_match = input_match;
  result.reference_down_activation_bytes_match = reference_down_match;
  result.cuda_reference_ms = reference_ms.value();
  result.sm120_direct_ms = native_ms.value();
  result.sm120_unfused_gate_up_ms = unfused_gate_up_ms.value();
  result.sm120_fused_gate_up_ms = fused_gate_up_ms.value();
  result.device_bytes = input.bytes() + input_packed.bytes() + input_scales.bytes() +
                        gate_weight.bytes() + gate_scales.bytes() + up_weight.bytes() +
                        up_scales.bytes() + down_weight.bytes() + down_scales.bytes() +
                        gate_output.bytes() + up_output.bytes() + product.bytes() +
                        down_input_packed.bytes() + down_input_scales.bytes() +
                        down_output.bytes() + final_output.bytes();
  for (std::size_t index = 0; index < native_down_packed.size(); ++index) {
    result.native_down_activation_mismatched_bytes +=
        native_down_packed[index] == reference_down_packed[index] ? 0U : 1U;
  }
  for (std::size_t index = 0; index < native_down_scales.size(); ++index) {
    result.native_down_activation_mismatched_bytes +=
        native_down_scales[index] == reference_down_scales[index] ? 0U : 1U;
  }

  double square_sum = 0.0;
  double dot = 0.0;
  double reference_square_sum = 0.0;
  double native_square_sum = 0.0;
  for (std::size_t row = 0; row < hidden; ++row) {
    if (!std::isfinite(reference_output[row]) || !std::isfinite(native_output[row])) {
      return ProbeError(StatusCode::kDataLoss, "MLP probe produced a non-finite output");
    }
    const double reference = reference_output[row];
    const double native = native_output[row];
    const double difference = native - reference;
    result.reference_native_max_abs =
        std::max(result.reference_native_max_abs, std::fabs(difference));
    square_sum += difference * difference;
    dot += reference * native;
    reference_square_sum += reference * reference;
    native_square_sum += native * native;
  }
  result.reference_native_rms = std::sqrt(square_sum / static_cast<double>(hidden));
  result.reference_native_cosine = dot / std::sqrt(reference_square_sum * native_square_sum);

  constexpr std::array<std::uint64_t, 8> sample_rows = {0, 1, 7, 8, 127, 960, 1920, 3839};
  const std::size_t packed_row_bytes = intermediate / 2U;
  const std::size_t scale_row_bytes = intermediate / 16U;
  for (const std::uint64_t row : sample_rows) {
    const auto row_weights = down.value().packed_weight.subspan(
        static_cast<std::size_t>(row) * packed_row_bytes, packed_row_bytes);
    const auto row_scales = down.value().weight_scales.subspan(
        static_cast<std::size_t>(row) * scale_row_bytes, scale_row_bytes);
    auto down_oracle = nvfp4::ReferenceDotProduct(host_down_quantized.value(), row_weights,
                                                  row_scales, down.value().weight_divisor);
    if (!down_oracle.ok()) return down_oracle.status();
    const double oracle = down_oracle.value() + static_cast<double>(host_input[row]);
    result.oracle_reference_max_abs = std::max(
        result.oracle_reference_max_abs,
        std::fabs(oracle - static_cast<double>(reference_output[row])));
    result.oracle_native_max_abs = std::max(
        result.oracle_native_max_abs,
        std::fabs(oracle - static_cast<double>(native_output[row])));
    result.samples.push_back({row, oracle, reference_output[row], native_output[row]});
  }
  return result;
}

}  // namespace gem16gb::internal

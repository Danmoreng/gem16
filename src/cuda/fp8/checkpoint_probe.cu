#include "cuda/fp8/checkpoint_probe.h"

#include "cuda/fp8/reference.h"
#include "cuda/fp8/sm120.h"
#include "gem16/fp8.h"
#include "gem16/model.h"
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

namespace gem16::internal {
namespace {

constexpr char kLayer0AttentionBase[] = "model.language_model.layers.0.self_attn.";
constexpr char kInstruction[] = "QMMA.16832.F32.E4M3.E4M3";

Status Error(StatusCode code, std::string message) {
  return Status(code, std::move(message));
}

Status CudaFailure(const char* operation, cudaError_t error) {
  return Error(StatusCode::kInternal,
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
    return Error(StatusCode::kDataLoss, "tensor range is outside its mapped shard: " + tensor.name);
  }
  const auto* begin = reinterpret_cast<const std::uint8_t*>(file.data() + tensor.byte_offset);
  return std::span<const std::uint8_t>(begin, static_cast<std::size_t>(tensor.byte_length));
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
      return Error(StatusCode::kInvalidArgument, std::string(label) + " size is invalid");
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
  if (iterations == 0U) return Error(StatusCode::kInvalidArgument, "FP8 probe iterations must be nonzero");
  for (std::uint32_t iteration = 0; iteration < warmups; ++iteration) {
    const Status status = launch();
    if (!status.ok()) return status;
  }
  cudaError_t error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return CudaFailure("FP8 warmup synchronize", error);
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
    const double ripple = static_cast<double>(static_cast<int>(index % 19U) - 9) * 0.004;
    activation[index] = static_cast<float>(0.2 * std::sin(position * 0.011) +
                                           0.15 * std::cos(position * 0.017) + ripple);
  }
  return activation;
}

std::uint16_t LoadLittleU16(std::span<const std::uint8_t> bytes, std::size_t index) {
  const std::size_t offset = index * 2U;
  return static_cast<std::uint16_t>(bytes[offset]) |
         static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[offset + 1U]) << 8U);
}

}  // namespace

Result<Fp8CheckpointProbeResult> RunLayer0Fp8CheckpointProbe(
    const std::filesystem::path& model_directory, std::string_view projection,
    std::uint32_t warmups, std::uint32_t iterations) {
  std::vector<std::uint64_t> expected_shape;
  if (projection == "q") expected_shape = {4096U, 3840U};
  else if (projection == "k" || projection == "v") expected_shape = {2048U, 3840U};
  else if (projection == "o") expected_shape = {3840U, 4096U};
  else return Error(StatusCode::kInvalidArgument, "FP8 projection must be q, k, v, or o");

  auto manifest = InspectCheckpoint({model_directory, true});
  if (!manifest.ok()) return manifest.status();
  const std::string base = std::string(kLayer0AttentionBase) + std::string(projection) + "_proj";
  const std::string weight_name = base + ".weight";
  const std::string scale_name = base + ".weight_scale";
  const TensorInfo* weight_info = FindTensor(manifest.value(), weight_name);
  const TensorInfo* scale_info = FindTensor(manifest.value(), scale_name);
  if (weight_info == nullptr || scale_info == nullptr) {
    return Error(StatusCode::kNotFound,
                 "layer-0 " + std::string(projection) + " FP8 tensor family is incomplete");
  }
  if (weight_info->shape != expected_shape ||
      scale_info->shape != std::vector<std::uint64_t>{expected_shape[0], 1U} ||
      weight_info->storage_dtype != "F8_E4M3" || scale_info->storage_dtype != "BF16" ||
      weight_info->source_shard != scale_info->source_shard) {
    return Error(StatusCode::kDataLoss,
                 "layer-0 " + std::string(projection) +
                     " FP8 geometry or storage differs from the pinned contract");
  }

  auto mapped = MappedFile::Open(model_directory / weight_info->source_shard);
  if (!mapped.ok()) return mapped.status();
  auto weight_bytes = TensorBytes(mapped.value(), *weight_info);
  auto scale_bytes = TensorBytes(mapped.value(), *scale_info);
  if (!weight_bytes.ok()) return weight_bytes.status();
  if (!scale_bytes.ok()) return scale_bytes.status();

  cudaDeviceProp properties{};
  cudaError_t cuda_error = cudaGetDeviceProperties(&properties, 0);
  if (cuda_error != cudaSuccess) return CudaFailure("cudaGetDeviceProperties", cuda_error);
  if (properties.major != 12 || properties.minor != 0) {
    return Error(StatusCode::kUnsupported, "FP8 checkpoint probe requires an SM120 device");
  }

  const std::uint64_t rows = expected_shape[0];
  const std::uint64_t k_size = expected_shape[1];
  std::vector<float> host_activation = DeterministicActivation(static_cast<std::size_t>(k_size));
  auto host_quantized = fp8::QuantizeToken(host_activation);
  if (!host_quantized.ok()) return host_quantized.status();

  DeviceBuffer<float> device_input;
  DeviceBuffer<std::uint8_t> device_activation;
  DeviceBuffer<float> device_activation_scale;
  DeviceBuffer<std::uint8_t> device_weight;
  DeviceBuffer<std::uint16_t> device_weight_scales;
  DeviceBuffer<float> device_reference;
  DeviceBuffer<float> device_native;
  for (const Status status : {
           device_input.Allocate(k_size, "allocate FP8 input"),
           device_activation.Allocate(k_size, "allocate FP8 activation"),
           device_activation_scale.Allocate(1, "allocate FP8 activation scale"),
           device_weight.Allocate(weight_info->byte_length, "allocate FP8 weight"),
           device_weight_scales.Allocate(scale_info->byte_length / 2U, "allocate FP8 weight scales"),
           device_reference.Allocate(rows, "allocate FP8 reference output"),
           device_native.Allocate(rows, "allocate FP8 native output"),
       }) {
    if (!status.ok()) return status;
  }
  cuda_error = cudaMemcpy(device_input.get(), host_activation.data(), device_input.bytes(),
                          cudaMemcpyHostToDevice);
  if (cuda_error != cudaSuccess) return CudaFailure("copy FP8 input", cuda_error);
  cuda_error = cudaMemcpy(device_weight.get(), weight_bytes.value().data(), weight_bytes.value().size(),
                          cudaMemcpyHostToDevice);
  if (cuda_error != cudaSuccess) return CudaFailure("copy FP8 weight", cuda_error);
  cuda_error = cudaMemcpy(device_weight_scales.get(), scale_bytes.value().data(), scale_bytes.value().size(),
                          cudaMemcpyHostToDevice);
  if (cuda_error != cudaSuccess) return CudaFailure("copy FP8 weight scales", cuda_error);

  const auto quantize_launch = [&] {
    return LaunchFp8ReferenceTokenQuantization(device_input.get(), device_activation.get(),
                                                device_activation_scale.get(), k_size, nullptr);
  };
  auto quantize_ms = Measure(warmups, iterations, quantize_launch);
  if (!quantize_ms.ok()) return quantize_ms.status();
  std::vector<std::uint8_t> gpu_activation(static_cast<std::size_t>(k_size));
  float gpu_activation_scale = 0.0F;
  cuda_error = cudaMemcpy(gpu_activation.data(), device_activation.get(), gpu_activation.size(),
                          cudaMemcpyDeviceToHost);
  if (cuda_error != cudaSuccess) return CudaFailure("copy FP8 activation", cuda_error);
  cuda_error = cudaMemcpy(&gpu_activation_scale, device_activation_scale.get(), sizeof(float),
                          cudaMemcpyDeviceToHost);
  if (cuda_error != cudaSuccess) return CudaFailure("copy FP8 activation scale", cuda_error);
  const bool activation_bytes_match = gpu_activation == host_quantized.value().values_e4m3fn;
  const bool activation_scale_match =
      std::bit_cast<std::uint32_t>(gpu_activation_scale) ==
      std::bit_cast<std::uint32_t>(host_quantized.value().scale);
  if (!activation_bytes_match || !activation_scale_match) {
    return Error(StatusCode::kDataLoss, "CPU and CUDA FP8 token quantization differ");
  }

  const auto reference_launch = [&] {
    return LaunchFp8ReferenceProjection(
        device_activation.get(), device_activation_scale.get(), device_weight.get(),
        device_weight_scales.get(), device_reference.get(), rows, k_size, nullptr);
  };
  auto reference_ms = Measure(1, 1, reference_launch);
  if (!reference_ms.ok()) return reference_ms.status();
  const auto native_launch = [&] {
    return LaunchFp8Sm120DirectProjection(
        device_activation.get(), device_activation_scale.get(), device_weight.get(),
        device_weight_scales.get(), device_native.get(), rows, k_size, nullptr);
  };
  auto native_ms = Measure(warmups, iterations, native_launch);
  if (!native_ms.ok()) return native_ms.status();

  std::vector<float> reference_output(static_cast<std::size_t>(rows));
  std::vector<float> native_output(static_cast<std::size_t>(rows));
  cuda_error = cudaMemcpy(reference_output.data(), device_reference.get(), device_reference.bytes(),
                          cudaMemcpyDeviceToHost);
  if (cuda_error != cudaSuccess) return CudaFailure("copy FP8 reference output", cuda_error);
  cuda_error = cudaMemcpy(native_output.data(), device_native.get(), device_native.bytes(),
                          cudaMemcpyDeviceToHost);
  if (cuda_error != cudaSuccess) return CudaFailure("copy FP8 native output", cuda_error);

  Fp8CheckpointProbeResult result;
  result.tensor_name = weight_name;
  result.instruction = kInstruction;
  result.rows = rows;
  result.contracting_elements = k_size;
  result.weight_bytes = weight_info->byte_length;
  result.weight_scale_bytes = scale_info->byte_length;
  result.device_bytes = device_input.bytes() + device_activation.bytes() +
                        device_activation_scale.bytes() + device_weight.bytes() +
                        device_weight_scales.bytes() + device_reference.bytes() +
                        device_native.bytes();
  result.activation_scale = gpu_activation_scale;
  result.activation_bytes_match = activation_bytes_match;
  result.activation_scale_match = activation_scale_match;
  result.activation_quantize_ms = quantize_ms.value();
  result.cuda_reference_ms = reference_ms.value();
  result.sm120_direct_ms = native_ms.value();

  double square_sum = 0.0;
  double dot = 0.0;
  double reference_square_sum = 0.0;
  double native_square_sum = 0.0;
  for (std::size_t row = 0; row < reference_output.size(); ++row) {
    if (!std::isfinite(reference_output[row]) || !std::isfinite(native_output[row])) {
      return Error(StatusCode::kDataLoss, "FP8 checkpoint probe produced a non-finite output");
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
  result.reference_native_rms = std::sqrt(square_sum / static_cast<double>(rows));
  result.reference_native_cosine = dot / std::sqrt(reference_square_sum * native_square_sum);

  const std::array<std::uint64_t, 8> sample_rows = {
      0, 1, 7, 8, 127, rows / 4U, rows / 2U, rows - 1U};
  for (const std::uint64_t row : sample_rows) {
    const auto row_weight = weight_bytes.value().subspan(static_cast<std::size_t>(row * k_size),
                                                         static_cast<std::size_t>(k_size));
    const std::uint16_t row_scale = LoadLittleU16(scale_bytes.value(), static_cast<std::size_t>(row));
    auto oracle = fp8::ReferenceDotProduct(host_quantized.value(), row_weight, row_scale);
    if (!oracle.ok()) return oracle.status();
    result.oracle_reference_max_abs = std::max(
        result.oracle_reference_max_abs,
        std::fabs(oracle.value() - static_cast<double>(reference_output[row])));
    result.oracle_native_max_abs = std::max(
        result.oracle_native_max_abs,
        std::fabs(oracle.value() - static_cast<double>(native_output[row])));
    result.samples.push_back({row, oracle.value(), reference_output[row], native_output[row]});
  }
  return result;
}

}  // namespace gem16::internal

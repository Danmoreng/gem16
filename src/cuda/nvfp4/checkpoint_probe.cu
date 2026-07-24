#include "cuda/nvfp4/checkpoint_probe.h"

#include "cuda/nvfp4/gemv.h"
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

Result<std::span<const std::uint8_t>> TensorBytes(const MappedFile& file,
                                                  const TensorInfo& tensor) {
  if (tensor.byte_offset > file.size() || tensor.byte_length > file.size() - tensor.byte_offset ||
      tensor.byte_length > std::numeric_limits<std::size_t>::max()) {
    return Error(StatusCode::kDataLoss, "tensor range is outside its mapped shard: " + tensor.name);
  }
  const auto* begin = reinterpret_cast<const std::uint8_t*>(file.data() + tensor.byte_offset);
  return std::span<const std::uint8_t>(begin, static_cast<std::size_t>(tensor.byte_length));
}

Result<float> ScalarF32(const MappedFile& file, const TensorInfo& tensor) {
  const auto bytes = TensorBytes(file, tensor);
  if (!bytes.ok()) return bytes.status();
  if (bytes.value().size() != sizeof(float)) {
    return Error(StatusCode::kDataLoss, "expected scalar F32 tensor: " + tensor.name);
  }
  const std::uint32_t word = static_cast<std::uint32_t>(bytes.value()[0]) |
                             (static_cast<std::uint32_t>(bytes.value()[1]) << 8U) |
                             (static_cast<std::uint32_t>(bytes.value()[2]) << 16U) |
                             (static_cast<std::uint32_t>(bytes.value()[3]) << 24U);
  const float value = std::bit_cast<float>(word);
  if (!std::isfinite(value) || value <= 0.0F) {
    return Error(StatusCode::kDataLoss, "global divisor must be positive and finite: " + tensor.name);
  }
  return value;
}

Status CopyToDevice(void* destination, std::span<const std::uint8_t> source,
                    const char* label) {
  const cudaError_t error =
      cudaMemcpy(destination, source.data(), source.size(), cudaMemcpyHostToDevice);
  return error == cudaSuccess ? Status::Ok() : CudaFailure(label, error);
}

template <typename Launch>
Result<double> Measure(std::uint32_t warmups, std::uint32_t iterations, Launch&& launch) {
  if (iterations == 0U) {
    return Error(StatusCode::kInvalidArgument, "kernel probe iterations must be nonzero");
  }
  for (std::uint32_t iteration = 0; iteration < warmups; ++iteration) {
    const Status status = launch();
    if (!status.ok()) return status;
  }
  cudaError_t error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return CudaFailure("kernel warmup synchronize", error);

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

}  // namespace

Result<Nvfp4CheckpointProbeResult> RunLayer0Nvfp4CheckpointProbe(
    const std::filesystem::path& model_directory, std::string_view projection,
    std::uint32_t warmups, std::uint32_t iterations) {
  if (projection != "gate" && projection != "up" && projection != "down") {
    return Error(StatusCode::kInvalidArgument,
                 "layer-0 NVFP4 projection must be gate, up, or down");
  }
  auto manifest = InspectCheckpoint({model_directory, true});
  if (!manifest.ok()) return manifest.status();

  const std::string projection_base =
      std::string(kLayer0MlpBase) + std::string(projection) + "_proj";
  const std::string packed_name = projection_base + ".weight_packed";
  const std::string scale_name = projection_base + ".weight_scale";
  const std::string input_divisor_name = projection_base + ".input_global_scale";
  const std::string weight_divisor_name = projection_base + ".weight_global_scale";
  const TensorInfo* packed_info = FindTensor(manifest.value(), packed_name);
  const TensorInfo* scale_info = FindTensor(manifest.value(), scale_name);
  const TensorInfo* input_divisor_info = FindTensor(manifest.value(), input_divisor_name);
  const TensorInfo* weight_divisor_info = FindTensor(manifest.value(), weight_divisor_name);
  if (packed_info == nullptr || scale_info == nullptr || input_divisor_info == nullptr ||
      weight_divisor_info == nullptr) {
    return Error(StatusCode::kNotFound,
                 "layer-0 " + std::string(projection) + " NVFP4 tensor family is incomplete");
  }
  const std::vector<std::uint64_t> expected_shape =
      projection == "down" ? std::vector<std::uint64_t>{3840U, 15360U}
                           : std::vector<std::uint64_t>{15360U, 3840U};
  if (packed_info->logical_shape != expected_shape ||
      scale_info->shape !=
          std::vector<std::uint64_t>({expected_shape[0], expected_shape[1] / 16U}) ||
      packed_info->source_shard != scale_info->source_shard ||
      packed_info->source_shard != input_divisor_info->source_shard ||
      packed_info->source_shard != weight_divisor_info->source_shard) {
    return Error(StatusCode::kDataLoss,
                 "layer-0 " + std::string(projection) +
                     " NVFP4 tensor geometry or shard differs from the pinned contract");
  }

  auto mapped = MappedFile::Open(model_directory / packed_info->source_shard);
  if (!mapped.ok()) return mapped.status();
  auto packed_weight = TensorBytes(mapped.value(), *packed_info);
  auto weight_scales = TensorBytes(mapped.value(), *scale_info);
  auto input_divisor = ScalarF32(mapped.value(), *input_divisor_info);
  auto weight_divisor = ScalarF32(mapped.value(), *weight_divisor_info);
  if (!packed_weight.ok()) return packed_weight.status();
  if (!weight_scales.ok()) return weight_scales.status();
  if (!input_divisor.ok()) return input_divisor.status();
  if (!weight_divisor.ok()) return weight_divisor.status();

  cudaDeviceProp properties{};
  cudaError_t cuda_error = cudaGetDeviceProperties(&properties, 0);
  if (cuda_error != cudaSuccess) return CudaFailure("cudaGetDeviceProperties", cuda_error);
  if (properties.major != 12 || properties.minor != 0) {
    return Error(StatusCode::kUnsupported, "checkpoint probe requires an SM120 device");
  }

  const std::uint64_t rows = expected_shape[0];
  const std::uint64_t k_size = expected_shape[1];
  std::vector<float> host_activation = DeterministicActivation(k_size);
  auto host_quantized = nvfp4::QuantizeActivation(host_activation, input_divisor.value());
  if (!host_quantized.ok()) return host_quantized.status();

  DeviceBuffer<float> device_activation;
  DeviceBuffer<std::uint8_t> device_packed_activation;
  DeviceBuffer<std::uint8_t> device_activation_scales;
  DeviceBuffer<std::uint8_t> device_weight;
  DeviceBuffer<std::uint8_t> device_weight_scales;
  DeviceBuffer<float> device_reference_output;
  DeviceBuffer<float> device_native_output;
  DeviceBuffer<float> device_simt_output;
  for (const Status status : {
           device_activation.Allocate(k_size, "allocate activation"),
           device_packed_activation.Allocate(k_size / 2U, "allocate packed activation"),
           device_activation_scales.Allocate(k_size / 16U, "allocate activation scales"),
           device_weight.Allocate(packed_info->byte_length, "allocate packed weight"),
           device_weight_scales.Allocate(scale_info->byte_length, "allocate weight scales"),
           device_reference_output.Allocate(rows, "allocate reference output"),
           device_native_output.Allocate(rows, "allocate native output"),
           device_simt_output.Allocate(rows, "allocate SIMT GEMV output"),
       }) {
    if (!status.ok()) return status;
  }

  cuda_error = cudaMemcpy(device_activation.get(), host_activation.data(), device_activation.bytes(),
                          cudaMemcpyHostToDevice);
  if (cuda_error != cudaSuccess) return CudaFailure("copy activation", cuda_error);
  Status copy_status = CopyToDevice(device_weight.get(), packed_weight.value(), "copy packed weight");
  if (!copy_status.ok()) return copy_status;
  copy_status = CopyToDevice(device_weight_scales.get(), weight_scales.value(), "copy weight scales");
  if (!copy_status.ok()) return copy_status;

  const auto quantize_launch = [&] {
    return LaunchNvfp4ReferenceActivationQuantization(
        device_activation.get(), device_packed_activation.get(), device_activation_scales.get(),
        k_size, input_divisor.value(), nullptr);
  };
  auto quantize_ms = Measure(warmups, iterations, quantize_launch);
  if (!quantize_ms.ok()) return quantize_ms.status();

  std::vector<std::uint8_t> gpu_packed_activation(k_size / 2U);
  std::vector<std::uint8_t> gpu_activation_scales(k_size / 16U);
  cuda_error = cudaMemcpy(gpu_packed_activation.data(), device_packed_activation.get(),
                          gpu_packed_activation.size(), cudaMemcpyDeviceToHost);
  if (cuda_error != cudaSuccess) return CudaFailure("copy packed activation to host", cuda_error);
  cuda_error = cudaMemcpy(gpu_activation_scales.data(), device_activation_scales.get(),
                          gpu_activation_scales.size(), cudaMemcpyDeviceToHost);
  if (cuda_error != cudaSuccess) return CudaFailure("copy activation scales to host", cuda_error);
  const bool activation_match =
      gpu_packed_activation == host_quantized.value().packed_e2m1 &&
      gpu_activation_scales == host_quantized.value().block_scales_e4m3fn;
  // The projection comparison remains valid when a quantization-boundary value differs: every
  // device projection consumes the same CUDA-produced bytes. Build the sample oracle from those
  // exact bytes instead of silently comparing it with the CPU quantizer's neighboring value.
  nvfp4::QuantizedActivation projection_activation = host_quantized.value();
  projection_activation.packed_e2m1 = gpu_packed_activation;
  projection_activation.block_scales_e4m3fn = gpu_activation_scales;

  const auto reference_launch = [&] {
    return LaunchNvfp4ReferenceProjection(
        device_packed_activation.get(), device_activation_scales.get(), device_weight.get(),
        device_weight_scales.get(), device_reference_output.get(), rows, k_size,
        input_divisor.value(), weight_divisor.value(), nullptr);
  };
  auto reference_ms = Measure(1, 1, reference_launch);
  if (!reference_ms.ok()) return reference_ms.status();

  const auto native_launch = [&] {
    return LaunchNvfp4Sm120DirectProjection(
        device_packed_activation.get(), device_activation_scales.get(), device_weight.get(),
        device_weight_scales.get(), device_native_output.get(), rows, k_size,
        input_divisor.value(), weight_divisor.value(), nullptr);
  };
  auto native_ms = Measure(warmups, iterations, native_launch);
  if (!native_ms.ok()) return native_ms.status();

  const auto simt_launch = [&] {
    return LaunchNvfp4SimtGemvProjection(
        device_packed_activation.get(), device_activation_scales.get(), device_weight.get(),
        device_weight_scales.get(), device_simt_output.get(), rows, k_size,
        input_divisor.value(), weight_divisor.value(), nullptr);
  };
  auto simt_ms = Measure(warmups, iterations, simt_launch);
  if (!simt_ms.ok()) return simt_ms.status();

  std::vector<float> reference_output(rows);
  std::vector<float> native_output(rows);
  std::vector<float> simt_output(rows);
  cuda_error = cudaMemcpy(reference_output.data(), device_reference_output.get(),
                          device_reference_output.bytes(), cudaMemcpyDeviceToHost);
  if (cuda_error != cudaSuccess) return CudaFailure("copy reference output", cuda_error);
  cuda_error = cudaMemcpy(native_output.data(), device_native_output.get(),
                          device_native_output.bytes(), cudaMemcpyDeviceToHost);
  if (cuda_error != cudaSuccess) return CudaFailure("copy native output", cuda_error);
  cuda_error = cudaMemcpy(simt_output.data(), device_simt_output.get(),
                          device_simt_output.bytes(), cudaMemcpyDeviceToHost);
  if (cuda_error != cudaSuccess) return CudaFailure("copy SIMT GEMV output", cuda_error);

  double max_abs = 0.0;
  double square_sum = 0.0;
  double dot = 0.0;
  double reference_square_sum = 0.0;
  double native_square_sum = 0.0;
  double simt_max_abs = 0.0;
  double simt_square_sum = 0.0;
  double simt_dot = 0.0;
  double simt_square_sum_output = 0.0;
  for (std::size_t row = 0; row < reference_output.size(); ++row) {
    if (!std::isfinite(reference_output[row]) || !std::isfinite(native_output[row]) ||
        !std::isfinite(simt_output[row])) {
      return Error(StatusCode::kDataLoss, "checkpoint probe produced a non-finite output");
    }
    const double reference = reference_output[row];
    const double native = native_output[row];
    const double difference = native - reference;
    max_abs = std::max(max_abs, std::fabs(difference));
    square_sum += difference * difference;
    dot += reference * native;
    reference_square_sum += reference * reference;
    native_square_sum += native * native;
    const double simt = simt_output[row];
    const double simt_difference = simt - reference;
    simt_max_abs = std::max(simt_max_abs, std::fabs(simt_difference));
    simt_square_sum += simt_difference * simt_difference;
    simt_dot += reference * simt;
    simt_square_sum_output += simt * simt;
  }

  Nvfp4CheckpointProbeResult result;
  result.tensor_name = packed_name;
  result.instruction = kInstruction;
  result.rows = rows;
  result.contracting_elements = k_size;
  result.packed_weight_bytes = packed_info->byte_length;
  result.weight_scale_bytes = scale_info->byte_length;
  result.device_bytes = device_activation.bytes() + device_packed_activation.bytes() +
                        device_activation_scales.bytes() + device_weight.bytes() +
                        device_weight_scales.bytes() + device_reference_output.bytes() +
                        device_native_output.bytes() + device_simt_output.bytes();
  result.input_global_divisor = input_divisor.value();
  result.weight_global_divisor = weight_divisor.value();
  result.activation_bytes_match = activation_match;
  result.activation_quantize_ms = quantize_ms.value();
  result.cuda_reference_ms = reference_ms.value();
  result.sm120_direct_ms = native_ms.value();
  result.simt_gemv_ms = simt_ms.value();
  result.reference_native_max_abs = max_abs;
  result.reference_native_rms = std::sqrt(square_sum / static_cast<double>(rows));
  result.reference_native_cosine =
      dot / std::sqrt(reference_square_sum * native_square_sum);
  result.reference_simt_max_abs = simt_max_abs;
  result.reference_simt_rms = std::sqrt(simt_square_sum / static_cast<double>(rows));
  result.reference_simt_cosine =
      simt_dot / std::sqrt(reference_square_sum * simt_square_sum_output);

  const std::array<std::uint64_t, 8> sample_rows = {
      0, 1, 7, 8, 127, rows / 4U, rows / 2U, rows - 1U};
  const std::size_t packed_row_bytes = static_cast<std::size_t>(k_size / 2U);
  const std::size_t scale_row_bytes = static_cast<std::size_t>(k_size / 16U);
  for (const std::uint64_t row : sample_rows) {
    const auto row_weights = packed_weight.value().subspan(
        static_cast<std::size_t>(row) * packed_row_bytes, packed_row_bytes);
    const auto row_scales = weight_scales.value().subspan(
        static_cast<std::size_t>(row) * scale_row_bytes, scale_row_bytes);
    auto oracle = nvfp4::ReferenceDotProduct(projection_activation, row_weights, row_scales,
                                             weight_divisor.value());
    if (!oracle.ok()) return oracle.status();
    result.oracle_reference_max_abs = std::max(
        result.oracle_reference_max_abs,
        std::fabs(oracle.value() - static_cast<double>(reference_output[row])));
    result.oracle_native_max_abs = std::max(
        result.oracle_native_max_abs,
        std::fabs(oracle.value() - static_cast<double>(native_output[row])));
    result.oracle_simt_max_abs = std::max(
        result.oracle_simt_max_abs,
        std::fabs(oracle.value() - static_cast<double>(simt_output[row])));
    result.samples.push_back(
        {row, oracle.value(), reference_output[row], native_output[row], simt_output[row]});
  }
  return result;
}

}  // namespace gem16gb::internal

#include "cuda/layer/checkpoint_probe.h"

#include "cuda/fp8/reference.h"
#include "cuda/fp8/sm120.h"
#include "cuda/layer/reference.h"
#include "cuda/nvfp4/mlp.h"
#include "cuda/nvfp4/reference.h"
#include "cuda/nvfp4/sm120.h"
#include "cuda/nvfp4/sm120_layout.h"
#include "gem16gb/model.h"
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
#include <utility>
#include <vector>

namespace gem16gb::internal {
namespace {

constexpr std::uint64_t kHidden = 3840;
constexpr std::uint64_t kIntermediate = 15360;
constexpr std::uint64_t kQueryHeads = 16;
constexpr std::uint64_t kContextTokens = 32;
constexpr float kEpsilon = 1.0e-6F;

struct AttentionGeometry {
  std::uint64_t layer = 0;
  std::uint64_t kv_heads = 0;
  std::uint64_t head_dimension = 0;
  double theta = 0.0;
  bool proportional_rope = false;
  bool reuse_k_projection_for_v = false;

  [[nodiscard]] std::uint64_t query_elements() const {
    return kQueryHeads * head_dimension;
  }
  [[nodiscard]] std::uint64_t kv_elements() const {
    return kv_heads * head_dimension;
  }
  [[nodiscard]] std::string base() const {
    return "model.language_model.layers." + std::to_string(layer) + ".";
  }
};

constexpr AttentionGeometry kLocalGeometry{0, 8, 256, 10000.0, false, false};
constexpr AttentionGeometry kGlobalGeometry{5, 1, 512, 1000000.0, true, true};

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
    return Error(StatusCode::kDataLoss, "tensor range is outside its shard: " + tensor.name);
  }
  const auto* begin = reinterpret_cast<const std::uint8_t*>(file.data() + tensor.byte_offset);
  return std::span<const std::uint8_t>(begin, static_cast<std::size_t>(tensor.byte_length));
}

struct HostFp8Binding {
  std::span<const std::uint8_t> weight;
  std::span<const std::uint8_t> scales;
  std::uint64_t rows = 0;
  std::uint64_t contracting = 0;
};

Result<HostFp8Binding> BindFp8(const ModelManifest& manifest, const MappedFile& mapped,
                              const std::string& base, const std::string& projection,
                              std::uint64_t rows, std::uint64_t contracting) {
  const std::string name = base + "self_attn." + projection + "_proj.weight";
  const TensorInfo* weight = FindTensor(manifest, name);
  const TensorInfo* scales = FindTensor(manifest, name + "_scale");
  if (weight == nullptr || scales == nullptr) {
    return Error(StatusCode::kNotFound, "missing FP8 tensor family: " + name);
  }
  if (weight->shape != std::vector<std::uint64_t>{rows, contracting} ||
      scales->shape != std::vector<std::uint64_t>{rows, 1U} ||
      weight->storage_dtype != "F8_E4M3" || scales->storage_dtype != "BF16") {
    return Error(StatusCode::kDataLoss, "unexpected FP8 tensor geometry: " + name);
  }
  auto weight_bytes = TensorBytes(mapped, *weight);
  auto scale_bytes = TensorBytes(mapped, *scales);
  if (!weight_bytes.ok()) return weight_bytes.status();
  if (!scale_bytes.ok()) return scale_bytes.status();
  return HostFp8Binding{weight_bytes.value(), scale_bytes.value(), rows, contracting};
}

Result<std::span<const std::uint8_t>> BindBf16(const ModelManifest& manifest,
                                               const MappedFile& mapped,
                                               const std::string& name,
                                               std::uint64_t elements) {
  const TensorInfo* tensor = FindTensor(manifest, name);
  if (tensor == nullptr) return Error(StatusCode::kNotFound, "missing BF16 tensor: " + name);
  if (tensor->storage_dtype != "BF16" || tensor->byte_length != elements * 2U) {
    return Error(StatusCode::kDataLoss, "unexpected BF16 tensor geometry: " + name);
  }
  return TensorBytes(mapped, *tensor);
}

Result<float> ScalarF32(const MappedFile& mapped, const TensorInfo& tensor) {
  auto bytes = TensorBytes(mapped, tensor);
  if (!bytes.ok()) return bytes.status();
  if (tensor.storage_dtype != "F32" || bytes.value().size() != sizeof(float)) {
    return Error(StatusCode::kDataLoss, "expected scalar F32 tensor: " + tensor.name);
  }
  const auto data = bytes.value();
  const std::uint32_t word = static_cast<std::uint32_t>(data[0]) |
                             (static_cast<std::uint32_t>(data[1]) << 8U) |
                             (static_cast<std::uint32_t>(data[2]) << 16U) |
                             (static_cast<std::uint32_t>(data[3]) << 24U);
  const float value = std::bit_cast<float>(word);
  if (!std::isfinite(value) || value <= 0.0F) {
    return Error(StatusCode::kDataLoss,
                 "NVFP4 global divisor must be positive and finite: " + tensor.name);
  }
  return value;
}

struct HostNvfp4Binding {
  std::span<const std::uint8_t> packed_weight;
  std::span<const std::uint8_t> weight_scales;
  float input_divisor = 0.0F;
  float weight_divisor = 0.0F;
  std::uint64_t rows = 0;
  std::uint64_t contracting = 0;
};

Result<HostNvfp4Binding> BindNvfp4(const ModelManifest& manifest, const MappedFile& mapped,
                                  const std::string& base, const std::string& projection,
                                  std::uint64_t rows, std::uint64_t contracting) {
  const std::string name = base + "mlp." + projection + "_proj";
  const TensorInfo* packed = FindTensor(manifest, name + ".weight_packed");
  const TensorInfo* scales = FindTensor(manifest, name + ".weight_scale");
  const TensorInfo* input_divisor = FindTensor(manifest, name + ".input_global_scale");
  const TensorInfo* weight_divisor = FindTensor(manifest, name + ".weight_global_scale");
  if (packed == nullptr || scales == nullptr || input_divisor == nullptr ||
      weight_divisor == nullptr) {
    return Error(StatusCode::kNotFound, "missing NVFP4 tensor family: " + name);
  }
  if (packed->logical_shape != std::vector<std::uint64_t>{rows, contracting} ||
      packed->storage_dtype != "U8" ||
      scales->shape != std::vector<std::uint64_t>{rows, contracting / 16U} ||
      scales->storage_dtype != "F8_E4M3" || packed->source_shard != "model.safetensors" ||
      scales->source_shard != packed->source_shard ||
      input_divisor->source_shard != packed->source_shard ||
      weight_divisor->source_shard != packed->source_shard) {
    return Error(StatusCode::kDataLoss, "unexpected NVFP4 tensor geometry: " + name);
  }
  auto packed_bytes = TensorBytes(mapped, *packed);
  auto scale_bytes = TensorBytes(mapped, *scales);
  auto input_value = ScalarF32(mapped, *input_divisor);
  auto weight_value = ScalarF32(mapped, *weight_divisor);
  if (!packed_bytes.ok()) return packed_bytes.status();
  if (!scale_bytes.ok()) return scale_bytes.status();
  if (!input_value.ok()) return input_value.status();
  if (!weight_value.ok()) return weight_value.status();
  return HostNvfp4Binding{packed_bytes.value(), scale_bytes.value(), input_value.value(),
                          weight_value.value(), rows, contracting};
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { if (data_ != nullptr) (void)cudaFree(data_); }
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

struct DeviceFp8Binding {
  DeviceBuffer<std::uint8_t> weight;
  DeviceBuffer<std::uint16_t> scales;
  std::uint64_t rows = 0;
  std::uint64_t contracting = 0;
};

struct DeviceNvfp4Binding {
  DeviceBuffer<std::uint8_t> packed_weight;
  DeviceBuffer<std::uint8_t> weight_scales;
  float input_divisor = 0.0F;
  float weight_divisor = 0.0F;
  std::uint64_t rows = 0;
  std::uint64_t contracting = 0;
};

Status UploadBinding(const HostFp8Binding& host, DeviceFp8Binding& device) {
  device.rows = host.rows;
  device.contracting = host.contracting;
  Status status = device.weight.Allocate(host.weight.size(), "allocate FP8 weight");
  if (!status.ok()) return status;
  status = device.scales.Allocate(host.scales.size() / 2U, "allocate FP8 scales");
  if (!status.ok()) return status;
  cudaError_t error = cudaMemcpy(device.weight.get(), host.weight.data(), host.weight.size(),
                                 cudaMemcpyHostToDevice);
  if (error != cudaSuccess) return CudaFailure("copy FP8 weight", error);
  error = cudaMemcpy(device.scales.get(), host.scales.data(), host.scales.size(),
                     cudaMemcpyHostToDevice);
  return error == cudaSuccess ? Status::Ok() : CudaFailure("copy FP8 scales", error);
}

Status UploadBinding(const HostNvfp4Binding& host, DeviceNvfp4Binding& device) {
  device.input_divisor = host.input_divisor;
  device.weight_divisor = host.weight_divisor;
  device.rows = host.rows;
  device.contracting = host.contracting;
  Status status =
      device.packed_weight.Allocate(host.packed_weight.size(), "allocate NVFP4 weight");
  if (!status.ok()) return status;
  status = device.weight_scales.Allocate(host.weight_scales.size(), "allocate NVFP4 scales");
  if (!status.ok()) return status;
  cudaError_t error = cudaMemcpy(device.packed_weight.get(), host.packed_weight.data(),
                                 host.packed_weight.size(), cudaMemcpyHostToDevice);
  if (error != cudaSuccess) return CudaFailure("copy NVFP4 weight", error);
  error = cudaMemcpy(device.weight_scales.get(), host.weight_scales.data(),
                     host.weight_scales.size(), cudaMemcpyHostToDevice);
  return error == cudaSuccess ? Status::Ok() : CudaFailure("copy NVFP4 scales", error);
}

Status UploadSm120Scales(const HostNvfp4Binding& host,
                         DeviceNvfp4Binding& device) {
  const auto layout = PlanSm120Nvfp4SourceLayout(host.rows, host.contracting);
  if (!layout.ok()) return layout.status();
  const auto tiled = TileSm120Nvfp4WeightScales(layout.value(), host.weight_scales);
  if (!tiled.ok()) return tiled.status();
  const cudaError_t error = cudaMemcpy(device.weight_scales.get(), tiled.value().data(),
                                       tiled.value().size(), cudaMemcpyHostToDevice);
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("copy tiled NVFP4 scales", error);
}

std::vector<float> DeterministicHidden() {
  std::vector<float> values(kHidden);
  for (std::size_t index = 0; index < values.size(); ++index) {
    const double position = static_cast<double>(index);
    values[index] = static_cast<float>(0.21 * std::sin(position * 0.011) +
                                       0.13 * std::cos(position * 0.017) +
                                       static_cast<double>(static_cast<int>(index % 17U) - 8) * 0.003);
  }
  return values;
}

std::vector<float> DeterministicCache(const AttentionGeometry& geometry, bool value_cache) {
  std::vector<float> values(kContextTokens * geometry.kv_elements());
  for (std::uint64_t token = 0; token + 1U < kContextTokens; ++token) {
    for (std::uint64_t element = 0; element < geometry.kv_elements(); ++element) {
      const double phase = static_cast<double>(token * 37U + element);
      values[static_cast<std::size_t>(token * geometry.kv_elements() + element)] =
          static_cast<float>((value_cache ? 0.07 : 0.05) * std::sin(phase * 0.019) +
                             (value_cache ? 0.03 : 0.04) * std::cos(phase * 0.013));
    }
  }
  return values;
}

enum class ProjectionPath { kReference, kSm120 };

Status LaunchProjection(ProjectionPath path, const std::uint8_t* activation,
                        const float* activation_scale, const DeviceFp8Binding& binding,
                        float* output) {
  if (path == ProjectionPath::kReference) {
    return LaunchFp8ReferenceProjection(activation, activation_scale, binding.weight.get(),
                                        binding.scales.get(), output, binding.rows,
                                        binding.contracting, nullptr);
  }
  return LaunchFp8Sm120DirectProjection(activation, activation_scale, binding.weight.get(),
                                       binding.scales.get(), output, binding.rows,
                                       binding.contracting, nullptr);
}

Status LaunchProjection(ProjectionPath path, const std::uint8_t* packed_activation,
                        const std::uint8_t* activation_scales,
                        const DeviceNvfp4Binding& binding, float* output) {
  if (path == ProjectionPath::kReference) {
    return LaunchNvfp4ReferenceProjection(
        packed_activation, activation_scales, binding.packed_weight.get(),
        binding.weight_scales.get(), output, binding.rows, binding.contracting,
        binding.input_divisor, binding.weight_divisor, nullptr);
  }
  return LaunchNvfp4Sm120DirectProjection(
      packed_activation, activation_scales, binding.packed_weight.get(),
      binding.weight_scales.get(), output, binding.rows, binding.contracting,
      binding.input_divisor, binding.weight_divisor, nullptr);
}

struct PathBuffers {
  DeviceBuffer<float> q;
  DeviceBuffer<float> k;
  DeviceBuffer<float> v;
  DeviceBuffer<float> q_norm;
  DeviceBuffer<float> k_norm;
  DeviceBuffer<float> v_norm;
  DeviceBuffer<float> key_cache;
  DeviceBuffer<float> value_cache;
  DeviceBuffer<float> scores;
  DeviceBuffer<float> attention;
  DeviceBuffer<std::uint8_t> o_activation;
  DeviceBuffer<float> o_scale;
  DeviceBuffer<float> o_output;
  DeviceBuffer<float> post_norm;
  DeviceBuffer<float> final;
};

Status AllocatePath(const AttentionGeometry& geometry, PathBuffers& path) {
  for (const Status status : {
      path.q.Allocate(geometry.query_elements(), "allocate Q"),
      path.k.Allocate(geometry.kv_elements(), "allocate K"),
      path.v.Allocate(geometry.kv_elements(), "allocate V"),
      path.q_norm.Allocate(geometry.query_elements(), "allocate normalized Q"),
      path.k_norm.Allocate(geometry.kv_elements(), "allocate normalized K"),
      path.v_norm.Allocate(geometry.kv_elements(), "allocate normalized V"),
      path.key_cache.Allocate(kContextTokens * geometry.kv_elements(), "allocate K cache"),
      path.value_cache.Allocate(kContextTokens * geometry.kv_elements(), "allocate V cache"),
      path.scores.Allocate(kQueryHeads * kContextTokens, "allocate attention scores"),
      path.attention.Allocate(geometry.query_elements(), "allocate attention output"),
      path.o_activation.Allocate(geometry.query_elements(), "allocate O activation"),
      path.o_scale.Allocate(1, "allocate O activation scale"),
      path.o_output.Allocate(kHidden, "allocate O projection output"),
      path.post_norm.Allocate(kHidden, "allocate post-attention norm"),
      path.final.Allocate(kHidden, "allocate attention residual"),
  }) if (!status.ok()) return status;
  return Status::Ok();
}

std::uint64_t PathBytes(const PathBuffers& path) {
  return static_cast<std::uint64_t>(
      path.q.bytes() + path.k.bytes() + path.v.bytes() + path.q_norm.bytes() +
      path.k_norm.bytes() + path.v_norm.bytes() + path.key_cache.bytes() +
      path.value_cache.bytes() + path.scores.bytes() + path.attention.bytes() +
      path.o_activation.bytes() + path.o_scale.bytes() + path.o_output.bytes() +
      path.post_norm.bytes() + path.final.bytes());
}

struct DecoderTailBuffers {
  DeviceBuffer<float> pre_mlp_norm;
  DeviceBuffer<std::uint8_t> mlp_input_packed;
  DeviceBuffer<std::uint8_t> mlp_input_scales;
  DeviceBuffer<float> gate_output;
  DeviceBuffer<float> up_output;
  DeviceBuffer<float> product;
  DeviceBuffer<std::uint8_t> down_input_packed;
  DeviceBuffer<std::uint8_t> down_input_scales;
  DeviceBuffer<float> down_output;
  DeviceBuffer<float> post_mlp_norm;
  DeviceBuffer<float> final;
};

Status AllocateDecoderTail(DecoderTailBuffers& tail) {
  for (const Status status : {
           tail.pre_mlp_norm.Allocate(kHidden, "allocate pre-MLP norm"),
           tail.mlp_input_packed.Allocate(kHidden / 2U, "allocate MLP input packed"),
           tail.mlp_input_scales.Allocate(kHidden / 16U, "allocate MLP input scales"),
           tail.gate_output.Allocate(kIntermediate, "allocate Gate output"),
           tail.up_output.Allocate(kIntermediate, "allocate Up output"),
           tail.product.Allocate(kIntermediate, "allocate MLP product"),
           tail.down_input_packed.Allocate(kIntermediate / 2U, "allocate Down input packed"),
           tail.down_input_scales.Allocate(kIntermediate / 16U,
                                           "allocate Down input scales"),
           tail.down_output.Allocate(kHidden, "allocate Down output"),
           tail.post_mlp_norm.Allocate(kHidden, "allocate post-MLP norm"),
           tail.final.Allocate(kHidden, "allocate decoder-layer output"),
       }) {
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

std::uint64_t DecoderTailBytes(const DecoderTailBuffers& tail) {
  return static_cast<std::uint64_t>(
      tail.pre_mlp_norm.bytes() + tail.mlp_input_packed.bytes() +
      tail.mlp_input_scales.bytes() + tail.gate_output.bytes() + tail.up_output.bytes() +
      tail.product.bytes() + tail.down_input_packed.bytes() +
      tail.down_input_scales.bytes() + tail.down_output.bytes() +
      tail.post_mlp_norm.bytes() + tail.final.bytes());
}

Status RunDecoderTail(ProjectionPath projection_path, DecoderTailBuffers& tail,
                      const float* attention_residual,
                      const std::uint16_t* pre_mlp_norm_weight,
                      const std::uint16_t* post_mlp_norm_weight,
                      const std::uint16_t* layer_scalar,
                      const DeviceNvfp4Binding& gate_binding,
                      const DeviceNvfp4Binding& up_binding,
                      const DeviceNvfp4Binding& down_binding) {
  Status status = LaunchRmsNorm(attention_residual, pre_mlp_norm_weight,
                                tail.pre_mlp_norm.get(), 1, kHidden, kEpsilon, nullptr);
  if (!status.ok()) return status;
  status = LaunchNvfp4ReferenceActivationQuantization(
      tail.pre_mlp_norm.get(), tail.mlp_input_packed.get(), tail.mlp_input_scales.get(),
      kHidden, gate_binding.input_divisor, nullptr);
  if (!status.ok()) return status;
  status = LaunchProjection(projection_path, tail.mlp_input_packed.get(),
                            tail.mlp_input_scales.get(), gate_binding,
                            tail.gate_output.get());
  if (!status.ok()) return status;
  status = LaunchProjection(projection_path, tail.mlp_input_packed.get(),
                            tail.mlp_input_scales.get(), up_binding, tail.up_output.get());
  if (!status.ok()) return status;
  status = LaunchGeluTanhProduct(tail.gate_output.get(), tail.up_output.get(),
                                 tail.product.get(), kIntermediate, nullptr);
  if (!status.ok()) return status;
  status = LaunchNvfp4ReferenceActivationQuantization(
      tail.product.get(), tail.down_input_packed.get(), tail.down_input_scales.get(),
      kIntermediate, down_binding.input_divisor, nullptr);
  if (!status.ok()) return status;
  status = LaunchProjection(projection_path, tail.down_input_packed.get(),
                            tail.down_input_scales.get(), down_binding,
                            tail.down_output.get());
  if (!status.ok()) return status;
  status = LaunchRmsNorm(tail.down_output.get(), post_mlp_norm_weight,
                         tail.post_mlp_norm.get(), 1, kHidden, kEpsilon, nullptr);
  if (!status.ok()) return status;
  status = LaunchAddResidual(tail.post_mlp_norm.get(), attention_residual,
                             tail.final.get(), kHidden, nullptr);
  if (!status.ok()) return status;
  return LaunchScale(tail.final.get(), layer_scalar, kHidden, nullptr);
}

Result<std::uint64_t> CountMismatchedBytes(const DeviceBuffer<std::uint8_t>& left,
                                           const DeviceBuffer<std::uint8_t>& right) {
  if (left.bytes() != right.bytes()) {
    return Error(StatusCode::kInternal, "mismatched comparison-buffer extents");
  }
  std::vector<std::uint8_t> host_left(left.bytes());
  std::vector<std::uint8_t> host_right(right.bytes());
  cudaError_t error =
      cudaMemcpy(host_left.data(), left.get(), left.bytes(), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return CudaFailure("copy left comparison buffer", error);
  error = cudaMemcpy(host_right.data(), right.get(), right.bytes(), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return CudaFailure("copy right comparison buffer", error);
  std::uint64_t mismatches = 0;
  for (std::size_t index = 0; index < host_left.size(); ++index) {
    mismatches += host_left[index] == host_right[index] ? 0U : 1U;
  }
  return mismatches;
}

Status RunPath(const AttentionGeometry& geometry, ProjectionPath projection_path, PathBuffers& path,
               const DeviceFp8Binding& q_binding, const DeviceFp8Binding& k_binding,
               const DeviceFp8Binding* v_binding, const DeviceFp8Binding& o_binding,
               const std::uint8_t* input_activation, const float* input_scale,
               const float* residual, const std::uint16_t* q_norm_weight,
               const std::uint16_t* k_norm_weight, const std::uint16_t* post_norm_weight,
               const std::vector<float>& host_keys, const std::vector<float>& host_values) {
  cudaError_t error = cudaMemcpy(path.key_cache.get(), host_keys.data(), path.key_cache.bytes(),
                                 cudaMemcpyHostToDevice);
  if (error != cudaSuccess) return CudaFailure("initialize K cache", error);
  error = cudaMemcpy(path.value_cache.get(), host_values.data(), path.value_cache.bytes(),
                     cudaMemcpyHostToDevice);
  if (error != cudaSuccess) return CudaFailure("initialize V cache", error);
  Status status = LaunchProjection(projection_path, input_activation, input_scale, q_binding,
                                   path.q.get());
  if (!status.ok()) return status;
  status = LaunchProjection(projection_path, input_activation, input_scale, k_binding,
                            path.k.get());
  if (!status.ok()) return status;
  if (geometry.reuse_k_projection_for_v) {
    error = cudaMemcpyAsync(path.v.get(), path.k.get(), path.v.bytes(),
                            cudaMemcpyDeviceToDevice, nullptr);
    if (error != cudaSuccess) return CudaFailure("reuse K projection for V", error);
  } else {
    if (v_binding == nullptr) {
      return Error(StatusCode::kInternal, "local attention V binding is absent");
    }
    status = LaunchProjection(projection_path, input_activation, input_scale, *v_binding,
                              path.v.get());
    if (!status.ok()) return status;
  }
  for (const Status next : {
      LaunchRmsNorm(path.q.get(), q_norm_weight, path.q_norm.get(), kQueryHeads,
                    geometry.head_dimension, kEpsilon, nullptr),
      LaunchRmsNorm(path.k.get(), k_norm_weight, path.k_norm.get(), geometry.kv_heads,
                    geometry.head_dimension, kEpsilon, nullptr),
      LaunchRmsNorm(path.v.get(), nullptr, path.v_norm.get(), geometry.kv_heads,
                    geometry.head_dimension, kEpsilon, nullptr),
  }) if (!next.ok()) return next;
  if (geometry.proportional_rope) {
    status = LaunchProportionalRotaryEmbedding(
        path.q_norm.get(), kQueryHeads, geometry.head_dimension, 0.25,
        kContextTokens - 1U, geometry.theta, 1.0, nullptr);
    if (!status.ok()) return status;
    status = LaunchProportionalRotaryEmbedding(
        path.k_norm.get(), geometry.kv_heads, geometry.head_dimension, 0.25,
        kContextTokens - 1U, geometry.theta, 1.0, nullptr);
  } else {
    status = LaunchRotaryEmbedding(path.q_norm.get(), kQueryHeads, geometry.head_dimension,
                                   geometry.head_dimension, kContextTokens - 1U,
                                   geometry.theta, nullptr);
    if (!status.ok()) return status;
    status = LaunchRotaryEmbedding(path.k_norm.get(), geometry.kv_heads,
                                   geometry.head_dimension, geometry.head_dimension,
                                   kContextTokens - 1U, geometry.theta, nullptr);
  }
  if (!status.ok()) return status;
  for (const Status next : {
      LaunchAppendKv(path.k_norm.get(), path.v_norm.get(), path.key_cache.get(),
                     path.value_cache.get(), kContextTokens - 1U, geometry.kv_heads,
                     geometry.head_dimension, nullptr),
      LaunchLocalAttentionDecode(path.q_norm.get(), path.key_cache.get(), path.value_cache.get(),
                                 path.scores.get(), path.attention.get(), kQueryHeads,
                                 geometry.kv_heads, geometry.head_dimension, kContextTokens, nullptr),
      LaunchFp8ReferenceTokenQuantization(path.attention.get(), path.o_activation.get(),
                                          path.o_scale.get(), geometry.query_elements(), nullptr),
      LaunchProjection(projection_path, path.o_activation.get(), path.o_scale.get(), o_binding,
                       path.o_output.get()),
      LaunchRmsNorm(path.o_output.get(), post_norm_weight, path.post_norm.get(), 1, kHidden,
                    kEpsilon, nullptr),
      LaunchAddResidual(path.post_norm.get(), residual, path.final.get(), kHidden, nullptr),
  }) if (!next.ok()) return next;
  return Status::Ok();
}

}  // namespace

namespace {

Result<LayerCheckpointProbeResult> RunLayerCheckpointProbe(
    const std::filesystem::path& model_directory, const AttentionGeometry& geometry,
    bool include_mlp) {
  auto manifest = InspectCheckpoint({model_directory, true});
  if (!manifest.ok()) return manifest.status();
  auto mapped = MappedFile::Open(model_directory / "model.safetensors");
  if (!mapped.ok()) return mapped.status();
  const std::string base = geometry.base();
  auto q = BindFp8(manifest.value(), mapped.value(), base, "q", geometry.query_elements(), kHidden);
  auto k = BindFp8(manifest.value(), mapped.value(), base, "k", geometry.kv_elements(), kHidden);
  auto o = BindFp8(manifest.value(), mapped.value(), base, "o", kHidden, geometry.query_elements());
  HostFp8Binding v;
  if (!geometry.reuse_k_projection_for_v) {
    auto bound_v =
        BindFp8(manifest.value(), mapped.value(), base, "v", geometry.kv_elements(), kHidden);
    if (!bound_v.ok()) return bound_v.status();
    v = bound_v.value();
  } else if (FindTensor(manifest.value(), base + "self_attn.v_proj.weight") != nullptr) {
    return Error(StatusCode::kDataLoss,
                 "global attention unexpectedly contains a separate V projection");
  }
  auto input_norm = BindBf16(manifest.value(), mapped.value(), base +
                             "input_layernorm.weight", kHidden);
  auto q_norm = BindBf16(manifest.value(), mapped.value(), base +
                         "self_attn.q_norm.weight", geometry.head_dimension);
  auto k_norm = BindBf16(manifest.value(), mapped.value(), base +
                         "self_attn.k_norm.weight", geometry.head_dimension);
  auto post_norm = BindBf16(manifest.value(), mapped.value(), base +
                            "post_attention_layernorm.weight", kHidden);
  if (!q.ok()) return q.status();
  if (!k.ok()) return k.status();
  if (!o.ok()) return o.status();
  if (!input_norm.ok()) return input_norm.status();
  if (!q_norm.ok()) return q_norm.status();
  if (!k_norm.ok()) return k_norm.status();
  if (!post_norm.ok()) return post_norm.status();

  HostNvfp4Binding gate, up, down;
  std::span<const std::uint8_t> pre_mlp_norm;
  std::span<const std::uint8_t> post_mlp_norm;
  std::span<const std::uint8_t> layer_scalar;
  if (include_mlp) {
    auto bound_gate = BindNvfp4(manifest.value(), mapped.value(), base, "gate", kIntermediate,
                                kHidden);
    auto bound_up = BindNvfp4(manifest.value(), mapped.value(), base, "up", kIntermediate,
                              kHidden);
    auto bound_down = BindNvfp4(manifest.value(), mapped.value(), base, "down", kHidden,
                                kIntermediate);
    auto bound_pre_mlp = BindBf16(manifest.value(), mapped.value(),
                                  base + "pre_feedforward_layernorm.weight", kHidden);
    auto bound_post_mlp = BindBf16(manifest.value(), mapped.value(),
                                   base + "post_feedforward_layernorm.weight", kHidden);
    auto bound_layer_scalar =
        BindBf16(manifest.value(), mapped.value(), base + "layer_scalar", 1);
    if (!bound_gate.ok()) return bound_gate.status();
    if (!bound_up.ok()) return bound_up.status();
    if (!bound_down.ok()) return bound_down.status();
    if (!bound_pre_mlp.ok()) return bound_pre_mlp.status();
    if (!bound_post_mlp.ok()) return bound_post_mlp.status();
    if (!bound_layer_scalar.ok()) return bound_layer_scalar.status();
    gate = bound_gate.value();
    up = bound_up.value();
    down = bound_down.value();
    pre_mlp_norm = bound_pre_mlp.value();
    post_mlp_norm = bound_post_mlp.value();
    layer_scalar = bound_layer_scalar.value();
    if (std::bit_cast<std::uint32_t>(gate.input_divisor) !=
        std::bit_cast<std::uint32_t>(up.input_divisor)) {
      return Error(StatusCode::kDataLoss,
                   "Layer-0 Gate and Up input global divisors are not identical");
    }
  }

  cudaDeviceProp properties{};
  cudaError_t error = cudaGetDeviceProperties(&properties, 0);
  if (error != cudaSuccess) return CudaFailure("cudaGetDeviceProperties", error);
  if (properties.major != 12 || properties.minor != 0) {
    return Error(StatusCode::kUnsupported, "attention checkpoint probe requires SM120");
  }

  DeviceFp8Binding device_q, device_k, device_v, device_o;
  for (const Status status : {UploadBinding(q.value(), device_q), UploadBinding(k.value(), device_k),
                              UploadBinding(o.value(), device_o)})
    if (!status.ok()) return status;
  const DeviceFp8Binding* device_v_pointer = nullptr;
  if (!geometry.reuse_k_projection_for_v) {
    const Status upload_v = UploadBinding(v, device_v);
    if (!upload_v.ok()) return upload_v;
    device_v_pointer = &device_v;
  }
  DeviceNvfp4Binding device_gate, device_up, device_down;
  if (include_mlp) {
    for (const Status status : {UploadBinding(gate, device_gate), UploadBinding(up, device_up),
                                UploadBinding(down, device_down)}) {
      if (!status.ok()) return status;
    }
  }
  DeviceBuffer<float> hidden, normalized;
  DeviceBuffer<std::uint8_t> activation;
  DeviceBuffer<float> activation_scale;
  DeviceBuffer<std::uint16_t> input_norm_weight, q_norm_weight, k_norm_weight, post_norm_weight;
  for (const Status status : {
      hidden.Allocate(kHidden, "allocate hidden"),
      normalized.Allocate(kHidden, "allocate normalized hidden"),
      activation.Allocate(kHidden, "allocate FP8 hidden"),
      activation_scale.Allocate(1, "allocate hidden scale"),
      input_norm_weight.Allocate(kHidden, "allocate input norm weight"),
      q_norm_weight.Allocate(geometry.head_dimension, "allocate Q norm weight"),
      k_norm_weight.Allocate(geometry.head_dimension, "allocate K norm weight"),
      post_norm_weight.Allocate(kHidden, "allocate post norm weight")})
    if (!status.ok()) return status;
  DeviceBuffer<std::uint16_t> pre_mlp_norm_weight, post_mlp_norm_weight, layer_scalar_weight;
  if (include_mlp) {
    for (const Status status : {
             pre_mlp_norm_weight.Allocate(kHidden, "allocate pre-MLP norm weight"),
             post_mlp_norm_weight.Allocate(kHidden, "allocate post-MLP norm weight"),
             layer_scalar_weight.Allocate(1, "allocate layer scalar"),
         }) {
      if (!status.ok()) return status;
    }
  }
  const auto host_hidden = DeterministicHidden();
  const auto host_keys = DeterministicCache(geometry, false);
  const auto host_values = DeterministicCache(geometry, true);
  const std::array<std::pair<void*, std::span<const std::uint8_t>>, 4> norm_copies = {{
      {input_norm_weight.get(), input_norm.value()},
      {q_norm_weight.get(), q_norm.value()},
      {k_norm_weight.get(), k_norm.value()},
      {post_norm_weight.get(), post_norm.value()},
  }};
  for (const auto& copy : norm_copies) {
    error = cudaMemcpy(copy.first, copy.second.data(), copy.second.size(), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return CudaFailure("copy norm weight", error);
  }
  if (include_mlp) {
    const std::array<std::pair<void*, std::span<const std::uint8_t>>, 3> tail_copies = {{
        {pre_mlp_norm_weight.get(), pre_mlp_norm},
        {post_mlp_norm_weight.get(), post_mlp_norm},
        {layer_scalar_weight.get(), layer_scalar},
    }};
    for (const auto& copy : tail_copies) {
      error = cudaMemcpy(copy.first, copy.second.data(), copy.second.size(),
                         cudaMemcpyHostToDevice);
      if (error != cudaSuccess) return CudaFailure("copy decoder-tail weight", error);
    }
  }
  error = cudaMemcpy(hidden.get(), host_hidden.data(), hidden.bytes(), cudaMemcpyHostToDevice);
  if (error != cudaSuccess) return CudaFailure("copy hidden", error);
  Status status = LaunchRmsNorm(hidden.get(), input_norm_weight.get(), normalized.get(), 1,
                                kHidden, kEpsilon, nullptr);
  if (!status.ok()) return status;
  status = LaunchFp8ReferenceTokenQuantization(normalized.get(), activation.get(),
                                               activation_scale.get(), kHidden, nullptr);
  if (!status.ok()) return status;

  PathBuffers reference, native;
  status = AllocatePath(geometry, reference);
  if (!status.ok()) return status;
  status = AllocatePath(geometry, native);
  if (!status.ok()) return status;
  status = RunPath(geometry, ProjectionPath::kReference, reference, device_q, device_k,
                   device_v_pointer, device_o,
                   activation.get(), activation_scale.get(), hidden.get(), q_norm_weight.get(),
                   k_norm_weight.get(), post_norm_weight.get(), host_keys, host_values);
  if (!status.ok()) return status;
  status = RunPath(geometry, ProjectionPath::kSm120, native, device_q, device_k,
                   device_v_pointer, device_o,
                   activation.get(), activation_scale.get(), hidden.get(), q_norm_weight.get(),
                   k_norm_weight.get(), post_norm_weight.get(), host_keys, host_values);
  if (!status.ok()) return status;
  DecoderTailBuffers reference_tail, native_tail;
  const float* reference_output = reference.final.get();
  const float* native_output = native.final.get();
  if (include_mlp) {
    status = AllocateDecoderTail(reference_tail);
    if (!status.ok()) return status;
    status = AllocateDecoderTail(native_tail);
    if (!status.ok()) return status;
    status = RunDecoderTail(ProjectionPath::kReference, reference_tail, reference.final.get(),
                            pre_mlp_norm_weight.get(), post_mlp_norm_weight.get(),
                            layer_scalar_weight.get(), device_gate, device_up, device_down);
    if (!status.ok()) return status;
    for (const Status scale_status : {
             UploadSm120Scales(gate, device_gate),
             UploadSm120Scales(up, device_up),
             UploadSm120Scales(down, device_down),
         }) {
      if (!scale_status.ok()) return scale_status;
    }
    status = RunDecoderTail(ProjectionPath::kSm120, native_tail, native.final.get(),
                            pre_mlp_norm_weight.get(), post_mlp_norm_weight.get(),
                            layer_scalar_weight.get(), device_gate, device_up, device_down);
    if (!status.ok()) return status;
    reference_output = reference_tail.final.get();
    native_output = native_tail.final.get();
  }
  error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return CudaFailure("layer probe synchronize", error);
  std::vector<float> host_reference(kHidden), host_native(kHidden);
  error = cudaMemcpy(host_reference.data(), reference_output, host_reference.size() * sizeof(float),
                     cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return CudaFailure("copy reference layer output", error);
  error = cudaMemcpy(host_native.data(), native_output, host_native.size() * sizeof(float),
                     cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return CudaFailure("copy native layer output", error);

  LayerCheckpointProbeResult result;
  result.layer = geometry.layer;
  result.context_tokens = kContextTokens;
  result.global_attention = geometry.proportional_rope;
  result.reused_k_projection_for_v = geometry.reuse_k_projection_for_v;
  result.includes_mlp = include_mlp;
  result.layer_scalar_applied = include_mlp;
  if (include_mlp) {
    auto input_packed_mismatches = CountMismatchedBytes(
        reference_tail.mlp_input_packed, native_tail.mlp_input_packed);
    auto input_scale_mismatches = CountMismatchedBytes(
        reference_tail.mlp_input_scales, native_tail.mlp_input_scales);
    auto down_packed_mismatches = CountMismatchedBytes(
        reference_tail.down_input_packed, native_tail.down_input_packed);
    auto down_scale_mismatches = CountMismatchedBytes(
        reference_tail.down_input_scales, native_tail.down_input_scales);
    if (!input_packed_mismatches.ok()) return input_packed_mismatches.status();
    if (!input_scale_mismatches.ok()) return input_scale_mismatches.status();
    if (!down_packed_mismatches.ok()) return down_packed_mismatches.status();
    if (!down_scale_mismatches.ok()) return down_scale_mismatches.status();
    result.mlp_input_mismatched_bytes =
        input_packed_mismatches.value() + input_scale_mismatches.value();
    result.down_input_mismatched_bytes =
        down_packed_mismatches.value() + down_scale_mismatches.value();
  }
  double squared_error = 0.0, dot = 0.0, reference_norm = 0.0, native_norm = 0.0;
  for (std::size_t index = 0; index < host_reference.size(); ++index) {
    if (!std::isfinite(host_reference[index]) || !std::isfinite(host_native[index]))
      return Error(StatusCode::kDataLoss, "layer probe produced non-finite output");
    const double difference = static_cast<double>(host_reference[index]) - host_native[index];
    result.reference_native_max_abs =
        std::max(result.reference_native_max_abs, std::fabs(difference));
    squared_error += difference * difference;
    dot += static_cast<double>(host_reference[index]) * host_native[index];
    reference_norm += static_cast<double>(host_reference[index]) * host_reference[index];
    native_norm += static_cast<double>(host_native[index]) * host_native[index];
  }
  result.reference_native_rms = std::sqrt(squared_error / static_cast<double>(kHidden));
  result.reference_native_cosine = dot / std::sqrt(reference_norm * native_norm);
  for (const std::uint64_t element :
       std::array<std::uint64_t, 8>{0, 1, 127, 511, 1023, 2047, 3071, 3839}) {
    result.samples.push_back({element, host_reference[element], host_native[element]});
  }
  result.device_bytes = static_cast<std::uint64_t>(
      device_q.weight.bytes() + device_q.scales.bytes() + device_k.weight.bytes() +
      device_k.scales.bytes() + device_v.weight.bytes() + device_v.scales.bytes() +
      device_o.weight.bytes() + device_o.scales.bytes() + hidden.bytes() + normalized.bytes() +
      activation.bytes() + activation_scale.bytes() + input_norm_weight.bytes() +
      q_norm_weight.bytes() + k_norm_weight.bytes() + post_norm_weight.bytes() +
      device_gate.packed_weight.bytes() + device_gate.weight_scales.bytes() +
      device_up.packed_weight.bytes() + device_up.weight_scales.bytes() +
      device_down.packed_weight.bytes() + device_down.weight_scales.bytes() +
      pre_mlp_norm_weight.bytes() + post_mlp_norm_weight.bytes() +
      layer_scalar_weight.bytes()) +
      PathBytes(reference) + PathBytes(native) + DecoderTailBytes(reference_tail) +
      DecoderTailBytes(native_tail);
  return result;
}

}  // namespace

Result<LayerCheckpointProbeResult> RunLayer0LocalAttentionCheckpointProbe(
    const std::filesystem::path& model_directory) {
  return RunLayerCheckpointProbe(model_directory, kLocalGeometry, false);
}

Result<LayerCheckpointProbeResult> RunLayer5GlobalAttentionCheckpointProbe(
    const std::filesystem::path& model_directory) {
  return RunLayerCheckpointProbe(model_directory, kGlobalGeometry, false);
}

Result<LayerCheckpointProbeResult> RunLayer0DecoderCheckpointProbe(
    const std::filesystem::path& model_directory) {
  return RunLayerCheckpointProbe(model_directory, kLocalGeometry, true);
}

}  // namespace gem16gb::internal

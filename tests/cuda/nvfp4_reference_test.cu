#include "cuda/attention/sm120.h"
#include "cuda/fp8/cutlass_sm120.h"
#include "cuda/fp8/reference.h"
#include "cuda/fp8/sm120.h"
#include "cuda/layer/reference.h"
#include "cuda/mtp/verify.h"
#include "cuda/nvfp4/cutlass_sm120.h"
#include "cuda/nvfp4/reference.h"
#include "cuda/nvfp4/gemv.h"
#include "cuda/nvfp4/sm120.h"
#include "cuda/nvfp4/sm120_layout.h"
#include "cuda/nvfp4/mlp.h"
#include "cuda/sampling/sampling.h"
#include "gem16/fp8.h"
#include "gem16/layer.h"
#include "gem16/nvfp4.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

void Check(bool condition, const char* expression, int line) {
  if (!condition) {
    std::cerr << __FILE__ << ':' << line << ": check failed: " << expression << '\n';
    ++failures;
  }
}

#define CUDA_TEST_CHECK(expression) Check(static_cast<bool>(expression), #expression, __LINE__)

bool CudaOk(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) return true;
  std::cerr << operation << ": " << cudaGetErrorName(error) << ": "
            << cudaGetErrorString(error) << '\n';
  ++failures;
  return false;
}

__global__ void RoundBf16ForComparisonKernel(float* values,
                                              std::uint64_t elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    values[index] = static_cast<float>(__float2bfloat16_rn(values[index]));
  }
}

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t elements) : elements_(elements) {
    if (!CudaOk(cudaMalloc(&data_, elements * sizeof(T)), "cudaMalloc")) data_ = nullptr;
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) (void)cudaFree(data_);
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] T* get() const { return static_cast<T*>(data_); }
  [[nodiscard]] std::size_t bytes() const { return elements_ * sizeof(T); }

 private:
  void* data_ = nullptr;
  std::size_t elements_ = 0;
};

void CheckAttentionMetrics(const std::vector<float>& reference,
                           const std::vector<float>& candidate,
                           const char* label, float maximum_limit,
                           double rms_limit, double cosine_limit) {
  CUDA_TEST_CHECK(reference.size() == candidate.size());
  if (reference.size() != candidate.size() || reference.empty()) return;
  double squared_error = 0.0;
  double reference_squared = 0.0;
  double candidate_squared = 0.0;
  double dot = 0.0;
  float maximum_absolute_error = 0.0F;
  bool finite = true;
  for (std::size_t index = 0; index < reference.size(); ++index) {
    const float reference_value = reference[index];
    const float candidate_value = candidate[index];
    const double difference =
        static_cast<double>(candidate_value) - reference_value;
    finite = finite && std::isfinite(reference_value) &&
             std::isfinite(candidate_value);
    maximum_absolute_error =
        std::max(maximum_absolute_error,
                 static_cast<float>(std::fabs(difference)));
    squared_error += difference * difference;
    reference_squared +=
        static_cast<double>(reference_value) * reference_value;
    candidate_squared +=
        static_cast<double>(candidate_value) * candidate_value;
    dot += static_cast<double>(reference_value) * candidate_value;
  }
  const double rms_error =
      std::sqrt(squared_error / static_cast<double>(reference.size()));
  const double cosine =
      dot / std::sqrt(reference_squared * candidate_squared);
  std::cout << label << ": max_abs=" << maximum_absolute_error
            << " rms=" << rms_error << " cosine=" << cosine << '\n';
  CUDA_TEST_CHECK(finite);
  CUDA_TEST_CHECK(maximum_absolute_error < maximum_limit);
  CUDA_TEST_CHECK(rms_error < rms_limit);
  CUDA_TEST_CHECK(cosine > cosine_limit);
}

void TestCudaIntrinsicConformanceAndProjection() {
  std::array<float, 16> host_activation{};
  for (std::size_t index = 0; index < host_activation.size(); ++index) {
    host_activation[index] =
        gem16::nvfp4::DecodeE2M1(static_cast<std::uint8_t>(index)) / 2.0F;
  }
  const auto host_quantized = gem16::nvfp4::QuantizeActivation(host_activation, 2.0F);
  CUDA_TEST_CHECK(host_quantized.ok());
  if (!host_quantized.ok()) return;

  DeviceBuffer<float> device_activation(host_activation.size());
  DeviceBuffer<std::uint8_t> device_packed(host_activation.size() / 2U);
  DeviceBuffer<std::uint8_t> device_scales(host_activation.size() / 16U);
  if (device_activation.get() == nullptr || device_packed.get() == nullptr ||
      device_scales.get() == nullptr) {
    return;
  }
  if (!CudaOk(cudaMemcpy(device_activation.get(), host_activation.data(),
                         device_activation.bytes(), cudaMemcpyHostToDevice),
              "copy activation to device")) {
    return;
  }

  const gem16::Status quantize_status =
      gem16::internal::LaunchNvfp4ReferenceActivationQuantization(
          device_activation.get(), device_packed.get(), device_scales.get(),
          host_activation.size(), 2.0F, nullptr);
  CUDA_TEST_CHECK(quantize_status.ok());
  if (!quantize_status.ok() || !CudaOk(cudaDeviceSynchronize(), "quantize synchronize")) return;

  std::array<std::uint8_t, 8> gpu_packed{};
  std::array<std::uint8_t, 1> gpu_scales{};
  CUDA_TEST_CHECK(CudaOk(cudaMemcpy(gpu_packed.data(), device_packed.get(), device_packed.bytes(),
                                    cudaMemcpyDeviceToHost),
                             "copy packed activation to host"));
  CUDA_TEST_CHECK(CudaOk(cudaMemcpy(gpu_scales.data(), device_scales.get(), device_scales.bytes(),
                                    cudaMemcpyDeviceToHost),
                             "copy activation scales to host"));
  CUDA_TEST_CHECK(std::equal(gpu_packed.begin(), gpu_packed.end(),
                             host_quantized.value().packed_e2m1.begin()));
  CUDA_TEST_CHECK(gpu_scales[0] == host_quantized.value().block_scales_e4m3fn[0]);

  constexpr std::array<std::uint8_t, 8> weight = {
      0x37U, 0xC1U, 0x53U, 0xA0U, 0xFBU, 0x5DU, 0xADU, 0xFEU,
  };
  constexpr std::array<std::uint8_t, 1> weight_scales = {0x61U};
  DeviceBuffer<std::uint8_t> device_weight(weight.size());
  DeviceBuffer<std::uint8_t> device_weight_scales(weight_scales.size());
  DeviceBuffer<float> device_output(1);
  if (device_weight.get() == nullptr || device_weight_scales.get() == nullptr ||
      device_output.get() == nullptr) {
    return;
  }
  CUDA_TEST_CHECK(CudaOk(cudaMemcpy(device_weight.get(), weight.data(), device_weight.bytes(),
                                    cudaMemcpyHostToDevice),
                             "copy weight to device"));
  CUDA_TEST_CHECK(CudaOk(cudaMemcpy(device_weight_scales.get(), weight_scales.data(),
                                    device_weight_scales.bytes(), cudaMemcpyHostToDevice),
                             "copy weight scales to device"));

  const gem16::Status projection_status = gem16::internal::LaunchNvfp4ReferenceProjection(
      device_packed.get(), device_scales.get(), device_weight.get(), device_weight_scales.get(),
      device_output.get(), 1, 16, 2.0F, 9600.0F, nullptr);
  CUDA_TEST_CHECK(projection_status.ok());
  if (!projection_status.ok() || !CudaOk(cudaDeviceSynchronize(), "projection synchronize")) {
    return;
  }

  float gpu_output = 0.0F;
  CUDA_TEST_CHECK(CudaOk(cudaMemcpy(&gpu_output, device_output.get(), sizeof(gpu_output),
                                    cudaMemcpyDeviceToHost),
                             "copy projection output to host"));
  const auto expected = gem16::nvfp4::ReferenceDotProduct(
      host_quantized.value(), weight, weight_scales, 9600.0F);
  CUDA_TEST_CHECK(expected.ok());
  if (expected.ok()) {
    CUDA_TEST_CHECK(std::fabs(static_cast<double>(gpu_output) - expected.value()) < 1.0e-6);
  }
}

void TestVllmNvfp4QuantizationBoundary() {
  constexpr std::array<float, 16> activation = {
      -0.1708984375F,       0.5703125F,          -0.4375F,
      2.777576446533203e-05F, -3.46875F,          -0.000461578369140625F,
      0.6015625F,          -0.0830078125F,       0.10009765625F,
      0.06884765625F,      0.69921875F,          -0.000652313232421875F,
      -0.875F,             6.866455078125e-05F,  -2.682209014892578e-05F,
      0.8359375F,
  };
  constexpr std::array<std::uint8_t, 8> expected_packed = {
      0x29U, 0x09U, 0x8FU, 0x82U, 0x00U, 0x82U, 0x0BU, 0x38U,
  };
  constexpr std::uint8_t expected_scale = 0x26U;

  DeviceBuffer<float> device_activation(activation.size());
  DeviceBuffer<std::uint8_t> device_packed(expected_packed.size());
  DeviceBuffer<std::uint8_t> device_scale(1);
  if (device_activation.get() == nullptr || device_packed.get() == nullptr ||
      device_scale.get() == nullptr) {
    return;
  }
  if (!CudaOk(cudaMemcpy(device_activation.get(), activation.data(),
                         device_activation.bytes(), cudaMemcpyHostToDevice),
              "copy vLLM NVFP4 boundary activation")) {
    return;
  }
  const auto status =
      gem16::internal::LaunchNvfp4ReferenceActivationQuantization(
          device_activation.get(), device_packed.get(), device_scale.get(),
          activation.size(), 0.375F, nullptr);
  CUDA_TEST_CHECK(status.ok());
  if (!status.ok() ||
      !CudaOk(cudaDeviceSynchronize(), "vLLM NVFP4 boundary synchronize")) {
    return;
  }
  std::array<std::uint8_t, expected_packed.size()> packed{};
  std::uint8_t scale = 0;
  if (!CudaOk(cudaMemcpy(packed.data(), device_packed.get(),
                         device_packed.bytes(), cudaMemcpyDeviceToHost),
              "copy vLLM NVFP4 boundary packed values") ||
      !CudaOk(cudaMemcpy(&scale, device_scale.get(), sizeof(scale),
                         cudaMemcpyDeviceToHost),
              "copy vLLM NVFP4 boundary scale")) {
    return;
  }
  CUDA_TEST_CHECK(packed == expected_packed);
  CUDA_TEST_CHECK(scale == expected_scale);
}

void StoreNibble(std::vector<std::uint8_t>& packed, std::size_t row, std::size_t k,
                 std::size_t packed_row_bytes, std::uint8_t nibble) {
  std::uint8_t& byte = packed[row * packed_row_bytes + k / 2U];
  const unsigned shift = k % 2U == 0U ? 0U : 4U;
  byte = static_cast<std::uint8_t>(byte | static_cast<std::uint8_t>(nibble << shift));
}

void TestDirectSourceSm120Projection() {
  constexpr std::size_t rows = 8;
  constexpr std::size_t k_size = 64;
  constexpr std::size_t tokens = 17;
  constexpr float activation_divisor = 2.0F;
  constexpr float weight_divisor = 4.0F;

  std::array<float, k_size> host_activation{};
  for (std::size_t k = 0; k < k_size; ++k) {
    host_activation[k] =
        gem16::nvfp4::DecodeE2M1(static_cast<std::uint8_t>(k & 0x0FU)) /
        activation_divisor;
  }
  const auto quantized =
      gem16::nvfp4::QuantizeActivation(host_activation, activation_divisor);
  CUDA_TEST_CHECK(quantized.ok());
  if (!quantized.ok()) return;

  std::vector<std::uint8_t> packed_weight(rows * k_size / 2U, 0U);
  std::vector<std::uint8_t> weight_scales(rows * k_size / 16U, 0x38U);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t k = 0; k < k_size; ++k) {
      const auto code = static_cast<std::uint8_t>((row * 3U + k * 5U) & 0x0FU);
      StoreNibble(packed_weight, row, k, k_size / 2U, code);
    }
  }
  const auto scale_layout =
      gem16::internal::PlanSm120Nvfp4SourceLayout(rows, k_size);
  CUDA_TEST_CHECK(scale_layout.ok());
  if (!scale_layout.ok()) return;
  const auto tiled_weight =
      gem16::internal::TileSm120Nvfp4Weights(scale_layout.value(), packed_weight);
  const auto tiled_weight_scales =
      gem16::internal::TileSm120Nvfp4WeightScales(scale_layout.value(), weight_scales);
  CUDA_TEST_CHECK(tiled_weight.ok());
  CUDA_TEST_CHECK(tiled_weight_scales.ok());
  if (!tiled_weight.ok() || !tiled_weight_scales.ok()) return;

  DeviceBuffer<std::uint8_t> device_activation(quantized.value().packed_e2m1.size());
  DeviceBuffer<std::uint8_t> device_activation_scales(
      quantized.value().block_scales_e4m3fn.size());
  DeviceBuffer<std::uint8_t> device_weight(packed_weight.size());
  DeviceBuffer<std::uint8_t> device_tiled_weight(packed_weight.size());
  DeviceBuffer<std::uint8_t> device_weight_scales(weight_scales.size());
  DeviceBuffer<std::uint8_t> device_tiled_weight_scales(weight_scales.size());
  DeviceBuffer<float> device_output(rows);
  DeviceBuffer<float> device_simt_output(rows);
  DeviceBuffer<float> device_fused_gate(rows);
  DeviceBuffer<float> device_fused_up(rows);
  DeviceBuffer<float> device_fused_product(rows);
  DeviceBuffer<std::uint8_t> device_batch_activation(
      tokens * quantized.value().packed_e2m1.size());
  DeviceBuffer<std::uint8_t> device_batch_activation_scales(
      tokens * quantized.value().block_scales_e4m3fn.size());
  DeviceBuffer<float> device_batch_reference(tokens * rows);
  DeviceBuffer<float> device_batch_native(tokens * rows);
  DeviceBuffer<std::uint16_t> device_batch_native_bf16(tokens * rows);
  DeviceBuffer<float> device_batch_product(tokens * rows);
  if (device_activation.get() == nullptr || device_activation_scales.get() == nullptr ||
      device_weight.get() == nullptr || device_tiled_weight.get() == nullptr ||
      device_weight_scales.get() == nullptr ||
      device_tiled_weight_scales.get() == nullptr ||
      device_output.get() == nullptr || device_simt_output.get() == nullptr ||
      device_fused_gate.get() == nullptr || device_fused_up.get() == nullptr ||
      device_fused_product.get() == nullptr ||
      device_batch_activation.get() == nullptr ||
      device_batch_activation_scales.get() == nullptr ||
      device_batch_reference.get() == nullptr || device_batch_native.get() == nullptr ||
      device_batch_native_bf16.get() == nullptr ||
      device_batch_product.get() == nullptr) {
    return;
  }
  if (!CudaOk(cudaMemcpy(device_activation.get(), quantized.value().packed_e2m1.data(),
                         device_activation.bytes(), cudaMemcpyHostToDevice),
              "copy native activation") ||
      !CudaOk(cudaMemcpy(device_activation_scales.get(),
                         quantized.value().block_scales_e4m3fn.data(),
                         device_activation_scales.bytes(), cudaMemcpyHostToDevice),
              "copy native activation scales") ||
      !CudaOk(cudaMemcpy(device_weight.get(), packed_weight.data(), device_weight.bytes(),
                         cudaMemcpyHostToDevice),
              "copy source weights") ||
      !CudaOk(cudaMemcpy(device_tiled_weight.get(), tiled_weight.value().data(),
                         device_tiled_weight.bytes(), cudaMemcpyHostToDevice),
              "copy tiled native weights") ||
      !CudaOk(cudaMemcpy(device_weight_scales.get(), weight_scales.data(),
                         device_weight_scales.bytes(), cudaMemcpyHostToDevice),
              "copy source weight scales") ||
      !CudaOk(cudaMemcpy(device_tiled_weight_scales.get(),
                         tiled_weight_scales.value().data(),
                         device_tiled_weight_scales.bytes(), cudaMemcpyHostToDevice),
              "copy tiled native weight scales")) {
    return;
  }
  for (std::size_t token = 0; token < tokens; ++token) {
    std::vector<std::uint8_t> token_activation =
        quantized.value().packed_e2m1;
    std::rotate(token_activation.begin(),
                token_activation.begin() +
                    static_cast<std::ptrdiff_t>(token % token_activation.size()),
                token_activation.end());
    std::vector<std::uint8_t> token_scales =
        quantized.value().block_scales_e4m3fn;
    const std::uint8_t scale =
        std::array<std::uint8_t, 3>{0x30U, 0x38U, 0x40U}[token % 3U];
    std::fill(token_scales.begin(), token_scales.end(), scale);
    if (!CudaOk(cudaMemcpy(
                    device_batch_activation.get() +
                        token * quantized.value().packed_e2m1.size(),
                    token_activation.data(), token_activation.size(),
                    cudaMemcpyHostToDevice),
                "copy batched native activation") ||
        !CudaOk(cudaMemcpy(
                    device_batch_activation_scales.get() +
                        token * quantized.value().block_scales_e4m3fn.size(),
                    token_scales.data(), token_scales.size(), cudaMemcpyHostToDevice),
                "copy batched native activation scales")) {
      return;
    }
  }

  const gem16::Status status = gem16::internal::LaunchNvfp4Sm120DirectProjection(
      device_activation.get(), device_activation_scales.get(), device_tiled_weight.get(),
      device_tiled_weight_scales.get(), device_output.get(), rows, k_size, activation_divisor,
      weight_divisor, nullptr);
  CUDA_TEST_CHECK(status.ok());
  const gem16::Status simt_status = gem16::internal::LaunchNvfp4SimtGemvProjection(
      device_activation.get(), device_activation_scales.get(), device_weight.get(),
      device_weight_scales.get(), device_simt_output.get(), rows, k_size,
      activation_divisor, weight_divisor, nullptr);
  CUDA_TEST_CHECK(simt_status.ok());
  const gem16::Status fused_status = gem16::internal::LaunchNvfp4Sm120FusedGateUp(
      device_activation.get(), device_activation_scales.get(), device_tiled_weight.get(),
      device_tiled_weight_scales.get(), device_tiled_weight.get(),
      device_tiled_weight_scales.get(),
      device_fused_gate.get(), device_fused_up.get(), device_fused_product.get(), rows,
      k_size, activation_divisor, weight_divisor, activation_divisor, weight_divisor,
      nullptr);
  CUDA_TEST_CHECK(fused_status.ok());
  const auto batch_reference_status =
      gem16::internal::LaunchNvfp4ReferenceProjectionBatch(
          device_batch_activation.get(), device_batch_activation_scales.get(),
          device_weight.get(), device_weight_scales.get(),
          device_batch_reference.get(), tokens, rows, k_size,
          activation_divisor, weight_divisor, nullptr);
  const auto batch_native_status =
      gem16::internal::LaunchNvfp4Sm120DirectProjectionBatch(
          device_batch_activation.get(), device_batch_activation_scales.get(),
          device_tiled_weight.get(), device_tiled_weight_scales.get(),
          device_batch_native.get(), tokens, rows, k_size, activation_divisor,
          weight_divisor, nullptr);
  const auto batch_native_bf16_status =
      gem16::internal::LaunchNvfp4Sm120DirectProjectionBf16Batch(
          device_batch_activation.get(), device_batch_activation_scales.get(),
          device_tiled_weight.get(), device_tiled_weight_scales.get(),
          device_batch_native_bf16.get(), tokens, rows, k_size,
          activation_divisor, weight_divisor, nullptr);
  const auto batch_fused_status =
      gem16::internal::LaunchNvfp4Sm120FusedGateUpBatch(
          device_batch_activation.get(), device_batch_activation_scales.get(),
          device_tiled_weight.get(), device_tiled_weight_scales.get(),
          device_tiled_weight.get(),
          device_tiled_weight_scales.get(), nullptr, nullptr,
          device_batch_product.get(), tokens, rows, k_size, activation_divisor,
          weight_divisor, activation_divisor, weight_divisor, nullptr);
  CUDA_TEST_CHECK(batch_reference_status.ok());
  CUDA_TEST_CHECK(batch_native_status.ok());
  CUDA_TEST_CHECK(batch_native_bf16_status.ok());
  CUDA_TEST_CHECK(batch_fused_status.ok());
  if (!status.ok() || !simt_status.ok() || !fused_status.ok() ||
      !batch_reference_status.ok() || !batch_native_status.ok() ||
      !batch_native_bf16_status.ok() ||
      !batch_fused_status.ok() ||
      !CudaOk(cudaDeviceSynchronize(), "native projection synchronize")) return;

  std::array<float, rows> output{};
  std::array<float, rows> simt_output{};
  std::array<float, rows> fused_gate{};
  std::array<float, rows> fused_up{};
  std::array<float, rows> fused_product{};
  std::array<float, tokens * rows> batch_reference{};
  std::array<float, tokens * rows> batch_native{};
  std::array<std::uint16_t, tokens * rows> batch_native_bf16{};
  std::array<float, tokens * rows> batch_product{};
  if (!CudaOk(cudaMemcpy(output.data(), device_output.get(), device_output.bytes(),
                         cudaMemcpyDeviceToHost),
              "copy native projection output") ||
      !CudaOk(cudaMemcpy(simt_output.data(), device_simt_output.get(),
                         device_simt_output.bytes(), cudaMemcpyDeviceToHost),
              "copy SIMT GEMV projection output") ||
      !CudaOk(cudaMemcpy(fused_gate.data(), device_fused_gate.get(),
                         device_fused_gate.bytes(), cudaMemcpyDeviceToHost),
              "copy fused Gate output") ||
      !CudaOk(cudaMemcpy(fused_up.data(), device_fused_up.get(),
                         device_fused_up.bytes(), cudaMemcpyDeviceToHost),
              "copy fused Up output") ||
      !CudaOk(cudaMemcpy(fused_product.data(), device_fused_product.get(),
                         device_fused_product.bytes(), cudaMemcpyDeviceToHost),
              "copy fused product output") ||
      !CudaOk(cudaMemcpy(batch_reference.data(), device_batch_reference.get(),
                         device_batch_reference.bytes(), cudaMemcpyDeviceToHost),
              "copy batched reference output") ||
      !CudaOk(cudaMemcpy(batch_native.data(), device_batch_native.get(),
                         device_batch_native.bytes(), cudaMemcpyDeviceToHost),
              "copy batched native output") ||
      !CudaOk(cudaMemcpy(batch_native_bf16.data(),
                         device_batch_native_bf16.get(),
                         device_batch_native_bf16.bytes(),
                         cudaMemcpyDeviceToHost),
              "copy batched native BF16 output") ||
      !CudaOk(cudaMemcpy(batch_product.data(), device_batch_product.get(),
                         device_batch_product.bytes(), cudaMemcpyDeviceToHost),
              "copy batched fused output")) {
    return;
  }
  for (std::size_t row = 0; row < rows; ++row) {
    const std::span<const std::uint8_t> weight_row(
        packed_weight.data() + row * k_size / 2U, k_size / 2U);
    const std::span<const std::uint8_t> scale_row(
        weight_scales.data() + row * k_size / 16U, k_size / 16U);
    const auto expected = gem16::nvfp4::ReferenceDotProduct(
        quantized.value(), weight_row, scale_row, weight_divisor);
    CUDA_TEST_CHECK(expected.ok());
    if (expected.ok()) {
      CUDA_TEST_CHECK(std::fabs(static_cast<double>(output[row]) - expected.value()) < 1.0e-5);
      CUDA_TEST_CHECK(
          std::fabs(static_cast<double>(simt_output[row]) - expected.value()) < 1.0e-4);
      const float rounded = static_cast<float>(__float2bfloat16_rn(output[row]));
      const float inner = 0.7978845608028654F *
                          (rounded + 0.044715F * rounded * rounded * rounded);
      const float gelu = static_cast<float>(
          __float2bfloat16_rn(0.5F * rounded * (1.0F + std::tanh(inner))));
      const float product = static_cast<float>(__float2bfloat16_rn(gelu * rounded));
      CUDA_TEST_CHECK(fused_gate[row] == rounded);
      CUDA_TEST_CHECK(fused_up[row] == rounded);
      CUDA_TEST_CHECK(fused_product[row] == product);
      for (std::size_t token = 0; token < tokens; ++token) {
        const float batch_value = batch_reference[token * rows + row];
        const float batch_rounded =
            static_cast<float>(__float2bfloat16_rn(batch_value));
        const float batch_inner =
            0.7978845608028654F *
            (batch_rounded + 0.044715F * batch_rounded * batch_rounded *
                                 batch_rounded);
        const float batch_gelu = static_cast<float>(__float2bfloat16_rn(
            0.5F * batch_rounded * (1.0F + std::tanh(batch_inner))));
        const float batch_expected = static_cast<float>(
            __float2bfloat16_rn(batch_gelu * batch_rounded));
        CUDA_TEST_CHECK(batch_native[token * rows + row] == batch_value);
        CUDA_TEST_CHECK(
            batch_native_bf16[token * rows + row] ==
            __bfloat16_as_ushort(__float2bfloat16_rn(batch_value)));
        CUDA_TEST_CHECK(batch_product[token * rows + row] == batch_expected);
      }
    }
  }
}

float GeluTanhReference(float value) {
  constexpr float square_root_two_over_pi = 0.7978845608028654F;
  constexpr float cubic = 0.044715F;
  return 0.5F * value *
         (1.0F + std::tanh(square_root_two_over_pi *
                           (value + cubic * value * value * value)));
}

float RoundBf16Reference(float value) {
  std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  bits += 0x7FFFU + ((bits >> 16U) & 1U);
  return std::bit_cast<float>(bits & 0xFFFF0000U);
}

void TestCutlassSm120Projection() {
  const auto run_case = [](std::size_t tokens, std::size_t rows,
                           std::size_t k_size, const char* label) {
  constexpr float activation_divisor = 2.0F;
  constexpr float weight_divisor = 4.0F;
  const std::size_t packed_elements = tokens * k_size / 2U;
  const std::size_t activation_scale_elements = tokens * k_size / 16U;
  const std::size_t packed_weight_elements = rows * k_size / 2U;
  const std::size_t weight_scale_elements = rows * k_size / 16U;
  constexpr std::size_t workspace_bytes = 8U * 1024U * 1024U;

  std::vector<std::uint8_t> activation(packed_elements);
  std::vector<std::uint8_t> activation_scales(activation_scale_elements);
  std::vector<std::uint8_t> weight(packed_weight_elements);
  std::vector<std::uint8_t> weight_scales(weight_scale_elements);
  for (std::size_t index = 0; index < activation.size(); ++index) {
    activation[index] = static_cast<std::uint8_t>(
        ((index * 5U + 3U) & 0x0FU) |
        (((index * 11U + 7U) & 0x0FU) << 4U));
  }
  for (std::size_t index = 0; index < weight.size(); ++index) {
    weight[index] = static_cast<std::uint8_t>(
        ((index * 13U + 1U) & 0x0FU) |
        (((index * 7U + 9U) & 0x0FU) << 4U));
  }
  for (std::size_t index = 0; index < activation_scales.size(); ++index) {
    activation_scales[index] =
        std::array<std::uint8_t, 3>{0x30U, 0x38U, 0x40U}[index % 3U];
  }
  for (std::size_t index = 0; index < weight_scales.size(); ++index) {
    weight_scales[index] =
        std::array<std::uint8_t, 3>{0x30U, 0x38U, 0x40U}[(index / 3U) % 3U];
  }

  const auto layout =
      gem16::internal::PlanSm120Nvfp4SourceLayout(rows, k_size);
  CUDA_TEST_CHECK(layout.ok());
  if (!layout.ok()) return;
  const auto tiled_weight =
      gem16::internal::TileSm120Nvfp4Weights(layout.value(), weight);
  const auto tiled_scales =
      gem16::internal::TileSm120Nvfp4WeightScales(layout.value(),
                                                    weight_scales);
  CUDA_TEST_CHECK(tiled_weight.ok());
  CUDA_TEST_CHECK(tiled_scales.ok());
  if (!tiled_weight.ok() || !tiled_scales.ok()) return;

  DeviceBuffer<std::uint8_t> device_activation(activation.size());
  DeviceBuffer<std::uint8_t> device_activation_scales(
      activation_scales.size());
  DeviceBuffer<std::uint8_t> device_interleaved_activation_scales(
      activation_scales.size());
  DeviceBuffer<std::uint8_t> device_weight(weight.size());
  DeviceBuffer<std::uint8_t> device_weight_scales(weight_scales.size());
  DeviceBuffer<std::uint8_t> device_weight_scratch(weight.size());
  DeviceBuffer<std::uint8_t> device_weight_scale_scratch(
      weight_scales.size());
  DeviceBuffer<std::uint8_t> device_workspace(workspace_bytes);
  DeviceBuffer<std::uint16_t> device_reference(tokens * rows);
  DeviceBuffer<std::uint16_t> device_cutlass(tokens * rows);
  if (device_activation.get() == nullptr ||
      device_activation_scales.get() == nullptr ||
      device_interleaved_activation_scales.get() == nullptr ||
      device_weight.get() == nullptr ||
      device_weight_scales.get() == nullptr ||
      device_weight_scratch.get() == nullptr ||
      device_weight_scale_scratch.get() == nullptr ||
      device_workspace.get() == nullptr || device_reference.get() == nullptr ||
      device_cutlass.get() == nullptr) {
    return;
  }
  if (!CudaOk(cudaMemcpy(device_activation.get(), activation.data(),
                         device_activation.bytes(), cudaMemcpyHostToDevice),
              "copy CUTLASS test activation") ||
      !CudaOk(cudaMemcpy(device_activation_scales.get(),
                         activation_scales.data(),
                         device_activation_scales.bytes(),
                         cudaMemcpyHostToDevice),
              "copy CUTLASS test activation scales") ||
      !CudaOk(cudaMemcpy(device_weight.get(), tiled_weight.value().data(),
                         device_weight.bytes(), cudaMemcpyHostToDevice),
              "copy CUTLASS test weight") ||
      !CudaOk(cudaMemcpy(device_weight_scales.get(),
                         tiled_scales.value().data(),
                         device_weight_scales.bytes(),
                         cudaMemcpyHostToDevice),
              "copy CUTLASS test weight scales")) {
    return;
  }

  const auto reference_status =
      gem16::internal::LaunchNvfp4Sm120DirectProjectionBf16Batch(
          device_activation.get(), device_activation_scales.get(),
          device_weight.get(), device_weight_scales.get(),
          device_reference.get(), tokens, rows, k_size, activation_divisor,
          weight_divisor, nullptr);
  const auto interleave_status =
      gem16::internal::LaunchNvfp4CutlassInterleaveActivationScales(
          device_activation_scales.get(),
          device_interleaved_activation_scales.get(), tokens, k_size, nullptr);
  const auto cutlass_status =
      gem16::internal::LaunchNvfp4CutlassProjectionBf16Batch(
          device_activation.get(),
          device_interleaved_activation_scales.get(), device_weight.get(),
          device_weight_scales.get(), device_weight_scratch.get(),
          device_weight_scale_scratch.get(), device_workspace.get(),
          workspace_bytes, device_cutlass.get(), tokens, rows, k_size,
          activation_divisor, weight_divisor, nullptr);
  CUDA_TEST_CHECK(reference_status.ok());
  CUDA_TEST_CHECK(interleave_status.ok());
  CUDA_TEST_CHECK(cutlass_status.ok());
  if (!reference_status.ok() || !interleave_status.ok() ||
      !cutlass_status.ok() ||
      !CudaOk(cudaDeviceSynchronize(), "CUTLASS projection synchronize")) {
    return;
  }

  std::vector<std::uint16_t> reference_bits(tokens * rows);
  std::vector<std::uint16_t> cutlass_bits(tokens * rows);
  if (!CudaOk(cudaMemcpy(reference_bits.data(), device_reference.get(),
                         device_reference.bytes(), cudaMemcpyDeviceToHost),
              "copy CUTLASS reference output") ||
      !CudaOk(cudaMemcpy(cutlass_bits.data(), device_cutlass.get(),
                         device_cutlass.bytes(), cudaMemcpyDeviceToHost),
              "copy CUTLASS output")) {
    return;
  }
  std::vector<float> reference(reference_bits.size());
  std::vector<float> cutlass(cutlass_bits.size());
  std::size_t exact_mismatches = 0U;
  for (std::size_t index = 0; index < reference.size(); ++index) {
    reference[index] =
        std::bit_cast<float>(static_cast<std::uint32_t>(reference_bits[index])
                             << 16U);
    cutlass[index] =
        std::bit_cast<float>(static_cast<std::uint32_t>(cutlass_bits[index])
                             << 16U);
    exact_mismatches += reference_bits[index] != cutlass_bits[index] ? 1U : 0U;
  }
  std::cout << label << " exact BF16 mismatches: " << exact_mismatches << '/'
            << reference.size() << '\n';
  CheckAttentionMetrics(reference, cutlass, label, 0.125F, 0.02, 0.9999);
  };
  run_case(2048U, 128U, 3840U, "CUTLASS NVFP4 Gate/Up projection");
  run_case(128U, 3840U, 15360U, "CUTLASS NVFP4 Down projection");
}

void TestMlpElementwiseBridge() {
  constexpr std::array<float, 9> gate = {
      -4.0F, -1.5F, -0.25F, -0.0F, 0.0F, 0.125F, 0.75F, 2.0F, 5.0F};
  constexpr std::array<float, 9> up = {
      0.5F, -2.0F, 3.0F, 4.0F, -5.0F, 1.25F, -0.75F, 2.5F, -0.125F};
  constexpr std::array<float, 9> residual = {
      1.0F, 0.5F, -0.5F, 2.0F, -2.0F, 0.25F, 0.0F, -1.0F, 3.0F};

  DeviceBuffer<float> device_gate(gate.size());
  DeviceBuffer<float> device_up(up.size());
  DeviceBuffer<float> device_product(gate.size());
  DeviceBuffer<float> device_residual(residual.size());
  DeviceBuffer<float> device_output(gate.size());
  if (device_gate.get() == nullptr || device_up.get() == nullptr ||
      device_product.get() == nullptr || device_residual.get() == nullptr ||
      device_output.get() == nullptr) {
    return;
  }
  if (!CudaOk(cudaMemcpy(device_gate.get(), gate.data(), device_gate.bytes(),
                         cudaMemcpyHostToDevice), "copy Gate") ||
      !CudaOk(cudaMemcpy(device_up.get(), up.data(), device_up.bytes(),
                         cudaMemcpyHostToDevice), "copy Up") ||
      !CudaOk(cudaMemcpy(device_residual.get(), residual.data(), device_residual.bytes(),
                         cudaMemcpyHostToDevice), "copy residual")) {
    return;
  }

  const auto product_status = gem16::internal::LaunchGeluTanhProduct(
      device_gate.get(), device_up.get(), device_product.get(), gate.size(), nullptr);
  CUDA_TEST_CHECK(product_status.ok());
  const auto residual_status = gem16::internal::LaunchAddResidual(
      device_product.get(), device_residual.get(), device_output.get(), gate.size(), nullptr);
  CUDA_TEST_CHECK(residual_status.ok());
  if (!product_status.ok() || !residual_status.ok() ||
      !CudaOk(cudaDeviceSynchronize(), "MLP elementwise synchronize")) {
    return;
  }

  std::array<float, gate.size()> output{};
  if (!CudaOk(cudaMemcpy(output.data(), device_output.get(), device_output.bytes(),
                         cudaMemcpyDeviceToHost), "copy MLP elementwise output")) {
    return;
  }
  for (std::size_t index = 0; index < output.size(); ++index) {
    const float expected =
        RoundBf16Reference(GeluTanhReference(gate[index])) * up[index] + residual[index];
    CUDA_TEST_CHECK(std::fabs(output[index] - expected) < 1.0e-6F);
  }
}

void TestFp8ReferenceAndDirectProjection() {
  constexpr std::size_t rows = 8;
  constexpr std::size_t k_size = 32;
  constexpr std::size_t tokens = 17;
  std::array<float, k_size> host_activation{};
  for (std::size_t index = 0; index < host_activation.size(); ++index) {
    host_activation[index] =
        static_cast<float>(static_cast<int>(index % 13U) - 6) * 0.125F;
  }
  const auto host_quantized = gem16::fp8::QuantizeToken(host_activation);
  CUDA_TEST_CHECK(host_quantized.ok());
  if (!host_quantized.ok()) return;

  std::vector<std::uint8_t> host_weight(rows * k_size);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t k = 0; k < k_size; ++k) {
      const float value = static_cast<float>(static_cast<int>((row * 5U + k * 3U) % 15U) - 7) /
                          4.0F;
      const auto encoded = gem16::fp8::EncodeE4M3Fn(value);
      CUDA_TEST_CHECK(encoded.ok());
      if (!encoded.ok()) return;
      host_weight[row * k_size + k] = encoded.value();
    }
  }
  constexpr std::array<std::uint16_t, rows> host_weight_scales = {
      0x3F80U, 0x3F00U, 0x3FC0U, 0x4000U, 0x3E80U, 0xBF80U, 0x4080U, 0x3F40U};
  // The checkpoint contract requires positive scales; keep the synthetic fixture valid too.
  std::array<std::uint16_t, rows> positive_weight_scales = host_weight_scales;
  positive_weight_scales[5] = 0x3E00U;

  DeviceBuffer<float> device_input(k_size);
  DeviceBuffer<std::uint8_t> device_activation(k_size);
  DeviceBuffer<float> device_activation_scale(1);
  DeviceBuffer<std::uint8_t> device_weight(host_weight.size());
  DeviceBuffer<std::uint16_t> device_weight_scales(rows);
  DeviceBuffer<float> device_reference(rows);
  DeviceBuffer<float> device_native(rows);
  DeviceBuffer<float> device_direct_grouped_q(rows);
  DeviceBuffer<float> device_direct_grouped_k(rows);
  DeviceBuffer<float> device_direct_grouped_v(rows);
  DeviceBuffer<float> device_batch_input(tokens * k_size);
  DeviceBuffer<std::uint8_t> device_batch_activation(tokens * k_size);
  DeviceBuffer<float> device_batch_scales(tokens);
  DeviceBuffer<float> device_batch_reference(tokens * rows);
  DeviceBuffer<float> device_batch_native(tokens * rows);
  DeviceBuffer<float> device_batch_cutlass(tokens * rows);
  DeviceBuffer<std::uint8_t> device_cutlass_workspace(8U * 1024U * 1024U);
  DeviceBuffer<float> device_grouped_q(tokens * rows);
  DeviceBuffer<float> device_grouped_k(tokens * rows);
  DeviceBuffer<float> device_grouped_v(tokens * rows);
  if (device_input.get() == nullptr || device_activation.get() == nullptr ||
      device_activation_scale.get() == nullptr || device_weight.get() == nullptr ||
      device_weight_scales.get() == nullptr || device_reference.get() == nullptr ||
      device_native.get() == nullptr || device_batch_input.get() == nullptr ||
      device_direct_grouped_q.get() == nullptr ||
      device_direct_grouped_k.get() == nullptr ||
      device_direct_grouped_v.get() == nullptr ||
      device_batch_activation.get() == nullptr || device_batch_scales.get() == nullptr ||
      device_batch_reference.get() == nullptr || device_batch_native.get() == nullptr ||
      device_batch_cutlass.get() == nullptr ||
      device_cutlass_workspace.get() == nullptr ||
      device_grouped_q.get() == nullptr || device_grouped_k.get() == nullptr ||
      device_grouped_v.get() == nullptr) {
    return;
  }
  if (!CudaOk(cudaMemcpy(device_input.get(), host_activation.data(), device_input.bytes(),
                         cudaMemcpyHostToDevice), "copy FP8 input") ||
      !CudaOk(cudaMemcpy(device_weight.get(), host_weight.data(), device_weight.bytes(),
                         cudaMemcpyHostToDevice), "copy FP8 weight") ||
      !CudaOk(cudaMemcpy(device_weight_scales.get(), positive_weight_scales.data(),
                         device_weight_scales.bytes(), cudaMemcpyHostToDevice),
              "copy FP8 weight scales")) {
    return;
  }
  for (std::size_t token = 0; token < tokens; ++token) {
    std::array<float, k_size> token_input{};
    const float factor = static_cast<float>(token % 5U + 1U) * 0.25F;
    for (std::size_t index = 0; index < k_size; ++index) {
      token_input[index] =
          host_activation[(index + token) % k_size] * factor;
    }
    if (!CudaOk(cudaMemcpy(device_batch_input.get() + token * k_size,
                           token_input.data(), k_size * sizeof(float),
                           cudaMemcpyHostToDevice), "copy batched FP8 input")) {
      return;
    }
  }

  const auto quantize_status = gem16::internal::LaunchFp8ReferenceTokenQuantization(
      device_input.get(), device_activation.get(), device_activation_scale.get(), k_size, nullptr);
  CUDA_TEST_CHECK(quantize_status.ok());
  if (!quantize_status.ok() || !CudaOk(cudaDeviceSynchronize(), "FP8 quantize synchronize")) {
    return;
  }
  std::array<std::uint8_t, k_size> gpu_activation{};
  float gpu_scale = 0.0F;
  CUDA_TEST_CHECK(CudaOk(cudaMemcpy(gpu_activation.data(), device_activation.get(),
                                    device_activation.bytes(), cudaMemcpyDeviceToHost),
                             "copy FP8 activation"));
  CUDA_TEST_CHECK(CudaOk(cudaMemcpy(&gpu_scale, device_activation_scale.get(), sizeof(float),
                                    cudaMemcpyDeviceToHost),
                             "copy FP8 activation scale"));
  CUDA_TEST_CHECK(gpu_scale == host_quantized.value().scale);
  CUDA_TEST_CHECK(std::equal(gpu_activation.begin(), gpu_activation.end(),
                             host_quantized.value().values_e4m3fn.begin()));

  const auto reference_status = gem16::internal::LaunchFp8ReferenceProjection(
      device_activation.get(), device_activation_scale.get(), device_weight.get(),
      device_weight_scales.get(), device_reference.get(), rows, k_size, nullptr);
  const auto native_status = gem16::internal::LaunchFp8Sm120DirectProjection(
      device_activation.get(), device_activation_scale.get(), device_weight.get(),
      device_weight_scales.get(), device_native.get(), rows, k_size, nullptr);
  const auto direct_grouped_status =
      gem16::internal::LaunchFp8Sm120GroupedQkvProjection(
          device_activation.get(), device_activation_scale.get(),
          device_weight.get(), device_weight_scales.get(),
          device_direct_grouped_q.get(), rows, device_weight.get(),
          device_weight_scales.get(), device_direct_grouped_k.get(), rows,
          device_weight.get(), device_weight_scales.get(),
          device_direct_grouped_v.get(), rows, k_size, nullptr);
  CUDA_TEST_CHECK(reference_status.ok());
  CUDA_TEST_CHECK(native_status.ok());
  CUDA_TEST_CHECK(direct_grouped_status.ok());
  const auto batch_quantize_status =
      gem16::internal::LaunchFp8ReferenceTokenQuantizationBatch(
          device_batch_input.get(), device_batch_activation.get(),
          device_batch_scales.get(), tokens, k_size, nullptr);
  const auto batch_reference_status =
      gem16::internal::LaunchFp8ReferenceProjectionBatch(
          device_batch_activation.get(), device_batch_scales.get(),
          device_weight.get(), device_weight_scales.get(),
          device_batch_reference.get(), tokens, rows, k_size, nullptr);
  const auto batch_native_status =
      gem16::internal::LaunchFp8Sm120DirectProjectionBatch(
          device_batch_activation.get(), device_batch_scales.get(),
          device_weight.get(), device_weight_scales.get(),
          device_batch_native.get(), tokens, rows, k_size, nullptr);
  const auto batch_cutlass_status =
      gem16::internal::LaunchFp8CutlassProjectionBatch(
          device_batch_activation.get(), device_batch_scales.get(),
          device_weight.get(), device_weight_scales.get(),
          device_batch_cutlass.get(), tokens, rows, k_size,
          device_cutlass_workspace.get(), device_cutlass_workspace.bytes(),
          nullptr);
  const auto grouped_native_status =
      gem16::internal::LaunchFp8Sm120GroupedQkvProjectionBatch(
          device_batch_activation.get(), device_batch_scales.get(),
          device_weight.get(), device_weight_scales.get(),
          device_grouped_q.get(), rows, device_weight.get(),
          device_weight_scales.get(), device_grouped_k.get(), rows,
          device_weight.get(), device_weight_scales.get(),
          device_grouped_v.get(), rows, tokens, k_size, nullptr);
  CUDA_TEST_CHECK(batch_quantize_status.ok());
  CUDA_TEST_CHECK(batch_reference_status.ok());
  CUDA_TEST_CHECK(batch_native_status.ok());
  CUDA_TEST_CHECK(batch_cutlass_status.ok());
  CUDA_TEST_CHECK(grouped_native_status.ok());
  if (!reference_status.ok() || !native_status.ok() ||
      !direct_grouped_status.ok() ||
      !batch_quantize_status.ok() || !batch_reference_status.ok() ||
      !batch_native_status.ok() || !batch_cutlass_status.ok() ||
      !grouped_native_status.ok() ||
      !CudaOk(cudaDeviceSynchronize(), "FP8 projection synchronize")) {
    return;
  }
  std::array<float, rows> reference_output{};
  std::array<float, rows> native_output{};
  std::array<float, rows> direct_grouped_q_output{};
  std::array<float, rows> direct_grouped_k_output{};
  std::array<float, rows> direct_grouped_v_output{};
  std::array<float, tokens * rows> batch_reference_output{};
  std::array<float, tokens * rows> batch_native_output{};
  std::array<float, tokens * rows> batch_cutlass_output{};
  std::array<float, tokens * rows> grouped_q_output{};
  std::array<float, tokens * rows> grouped_k_output{};
  std::array<float, tokens * rows> grouped_v_output{};
  if (!CudaOk(cudaMemcpy(reference_output.data(), device_reference.get(), device_reference.bytes(),
                         cudaMemcpyDeviceToHost), "copy FP8 reference output") ||
      !CudaOk(cudaMemcpy(native_output.data(), device_native.get(), device_native.bytes(),
                         cudaMemcpyDeviceToHost), "copy FP8 native output") ||
      !CudaOk(cudaMemcpy(direct_grouped_q_output.data(),
                         device_direct_grouped_q.get(),
                         device_direct_grouped_q.bytes(), cudaMemcpyDeviceToHost),
              "copy direct grouped FP8 Q output") ||
      !CudaOk(cudaMemcpy(direct_grouped_k_output.data(),
                         device_direct_grouped_k.get(),
                         device_direct_grouped_k.bytes(), cudaMemcpyDeviceToHost),
              "copy direct grouped FP8 K output") ||
      !CudaOk(cudaMemcpy(direct_grouped_v_output.data(),
                         device_direct_grouped_v.get(),
                         device_direct_grouped_v.bytes(), cudaMemcpyDeviceToHost),
              "copy direct grouped FP8 V output") ||
      !CudaOk(cudaMemcpy(batch_reference_output.data(),
                         device_batch_reference.get(), device_batch_reference.bytes(),
                         cudaMemcpyDeviceToHost), "copy batched FP8 reference output") ||
      !CudaOk(cudaMemcpy(batch_native_output.data(), device_batch_native.get(),
                         device_batch_native.bytes(), cudaMemcpyDeviceToHost),
              "copy batched FP8 native output") ||
      !CudaOk(cudaMemcpy(batch_cutlass_output.data(), device_batch_cutlass.get(),
                         device_batch_cutlass.bytes(), cudaMemcpyDeviceToHost),
              "copy batched CUTLASS FP8 output") ||
      !CudaOk(cudaMemcpy(grouped_q_output.data(), device_grouped_q.get(),
                         device_grouped_q.bytes(), cudaMemcpyDeviceToHost),
              "copy grouped FP8 Q output") ||
      !CudaOk(cudaMemcpy(grouped_k_output.data(), device_grouped_k.get(),
                         device_grouped_k.bytes(), cudaMemcpyDeviceToHost),
              "copy grouped FP8 K output") ||
      !CudaOk(cudaMemcpy(grouped_v_output.data(), device_grouped_v.get(),
                         device_grouped_v.bytes(), cudaMemcpyDeviceToHost),
              "copy grouped FP8 V output")) {
    return;
  }
  for (std::size_t row = 0; row < rows; ++row) {
    const auto expected = gem16::fp8::ReferenceDotProduct(
        host_quantized.value(),
        std::span<const std::uint8_t>(host_weight.data() + row * k_size, k_size),
        positive_weight_scales[row]);
    CUDA_TEST_CHECK(expected.ok());
    if (expected.ok()) {
      CUDA_TEST_CHECK(std::fabs(static_cast<double>(reference_output[row]) - expected.value()) <
                         1.0e-4);
      CUDA_TEST_CHECK(std::fabs(static_cast<double>(native_output[row]) - expected.value()) <
                         1.0e-4);
      CUDA_TEST_CHECK(direct_grouped_q_output[row] == native_output[row]);
      CUDA_TEST_CHECK(direct_grouped_k_output[row] == native_output[row]);
      CUDA_TEST_CHECK(direct_grouped_v_output[row] == native_output[row]);
      for (std::size_t token = 0; token < tokens; ++token) {
        CUDA_TEST_CHECK(batch_native_output[token * rows + row] ==
                        batch_reference_output[token * rows + row]);
        CUDA_TEST_CHECK(batch_cutlass_output[token * rows + row] ==
                        batch_reference_output[token * rows + row]);
        CUDA_TEST_CHECK(grouped_q_output[token * rows + row] ==
                        batch_reference_output[token * rows + row]);
        CUDA_TEST_CHECK(grouped_k_output[token * rows + row] ==
                        batch_reference_output[token * rows + row]);
        CUDA_TEST_CHECK(grouped_v_output[token * rows + row] ==
                        batch_reference_output[token * rows + row]);
      }
    }
  }
}

void TestFp8CutlassPrefillGeometry() {
  constexpr std::size_t tokens = 128U;
  constexpr std::size_t rows = 4096U;
  constexpr std::size_t k_size = 3840U;
  constexpr std::size_t workspace_bytes = 8U * 1024U * 1024U;
  constexpr std::array<std::uint8_t, 7> fp8_values = {
      0x00U, 0x30U, 0x38U, 0x40U, 0xB0U, 0xB8U, 0xC0U};
  std::vector<std::uint8_t> activation(tokens * k_size);
  std::vector<std::uint8_t> weight(rows * k_size);
  std::vector<float> activation_scales(tokens);
  std::vector<std::uint16_t> weight_scales(rows);
  for (std::size_t index = 0; index < activation.size(); ++index) {
    activation[index] = fp8_values[(index * 5U + index / k_size) %
                                   fp8_values.size()];
  }
  for (std::size_t index = 0; index < weight.size(); ++index) {
    weight[index] =
        fp8_values[(index * 3U + index / k_size) % fp8_values.size()];
  }
  for (std::size_t token = 0; token < tokens; ++token) {
    activation_scales[token] =
        std::array<float, 4>{0.125F, 0.25F, 0.5F, 1.0F}[token % 4U];
  }
  for (std::size_t row = 0; row < rows; ++row) {
    weight_scales[row] =
        std::array<std::uint16_t, 4>{0x3E00U, 0x3E80U, 0x3F00U,
                                     0x3F80U}[row % 4U];
  }

  DeviceBuffer<std::uint8_t> device_activation(activation.size());
  DeviceBuffer<float> device_activation_scales(activation_scales.size());
  DeviceBuffer<std::uint8_t> device_weight(weight.size());
  DeviceBuffer<std::uint16_t> device_weight_scales(weight_scales.size());
  DeviceBuffer<float> device_native(tokens * rows);
  DeviceBuffer<float> device_cutlass(tokens * rows);
  DeviceBuffer<std::uint8_t> device_workspace(workspace_bytes);
  if (device_activation.get() == nullptr ||
      device_activation_scales.get() == nullptr ||
      device_weight.get() == nullptr ||
      device_weight_scales.get() == nullptr ||
      device_native.get() == nullptr || device_cutlass.get() == nullptr ||
      device_workspace.get() == nullptr) {
    return;
  }
  if (!CudaOk(cudaMemcpy(device_activation.get(), activation.data(),
                         device_activation.bytes(), cudaMemcpyHostToDevice),
              "copy real FP8 activation") ||
      !CudaOk(cudaMemcpy(device_activation_scales.get(),
                         activation_scales.data(),
                         device_activation_scales.bytes(),
                         cudaMemcpyHostToDevice),
              "copy real FP8 activation scales") ||
      !CudaOk(cudaMemcpy(device_weight.get(), weight.data(),
                         device_weight.bytes(), cudaMemcpyHostToDevice),
              "copy real FP8 weight") ||
      !CudaOk(cudaMemcpy(device_weight_scales.get(), weight_scales.data(),
                         device_weight_scales.bytes(),
                         cudaMemcpyHostToDevice),
              "copy real FP8 weight scales")) {
    return;
  }
  const auto native_status =
      gem16::internal::LaunchFp8Sm120DirectProjectionBatch(
          device_activation.get(), device_activation_scales.get(),
          device_weight.get(), device_weight_scales.get(), device_native.get(),
          tokens, rows, k_size, nullptr);
  const auto cutlass_status =
      gem16::internal::LaunchFp8CutlassProjectionBatch(
          device_activation.get(), device_activation_scales.get(),
          device_weight.get(), device_weight_scales.get(), device_cutlass.get(),
          tokens, rows, k_size, device_workspace.get(),
          device_workspace.bytes(), nullptr);
  CUDA_TEST_CHECK(native_status.ok());
  CUDA_TEST_CHECK(cutlass_status.ok());
  if (!native_status.ok() || !cutlass_status.ok() ||
      !CudaOk(cudaDeviceSynchronize(),
              "real FP8 CUTLASS projection synchronize")) {
    return;
  }
  std::vector<float> native(tokens * rows);
  std::vector<float> cutlass(tokens * rows);
  if (!CudaOk(cudaMemcpy(native.data(), device_native.get(),
                         device_native.bytes(), cudaMemcpyDeviceToHost),
              "copy real native FP8 output") ||
      !CudaOk(cudaMemcpy(cutlass.data(), device_cutlass.get(),
                         device_cutlass.bytes(), cudaMemcpyDeviceToHost),
              "copy real CUTLASS FP8 output")) {
    return;
  }
  std::size_t exact_mismatches = 0U;
  for (std::size_t index = 0; index < native.size(); ++index) {
    exact_mismatches += native[index] != cutlass[index] ? 1U : 0U;
  }
  std::cout << "CUTLASS FP8 128x4096x3840 exact mismatches: "
            << exact_mismatches << '/' << native.size() << '\n';
  CheckAttentionMetrics(native, cutlass, "CUTLASS FP8 128x4096x3840",
                        0.125F, 0.02, 0.99999);
}

void TestLocalLayerReferenceOperators() {
  constexpr std::size_t query_heads = 4;
  constexpr std::size_t kv_heads = 2;
  constexpr std::size_t head_dimension = 4;
  constexpr std::size_t tokens = 3;
  constexpr std::array<float, query_heads * head_dimension> query = {
      1.0F, 0.0F, 0.5F, -0.5F, 0.0F, 1.0F, -0.5F, 0.5F,
      0.75F, -0.25F, 0.5F, 0.0F, -0.5F, 0.25F, 0.75F, 0.5F};
  constexpr std::array<float, tokens * kv_heads * head_dimension> key_cache = {
      1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
      0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F,
      0.5F, 0.5F, 0.0F, 0.0F, 0.0F, 0.5F, 0.5F, 0.0F};
  constexpr std::array<float, tokens * kv_heads * head_dimension> value_cache = {
      1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F,
      2.0F, 4.0F, 6.0F, 8.0F, 10.0F, 12.0F, 14.0F, 16.0F,
      3.0F, 6.0F, 9.0F, 12.0F, 15.0F, 18.0F, 21.0F, 24.0F};

  const auto host_attention = gem16::layer::LocalAttentionDecode(
      query, key_cache, value_cache, query_heads, kv_heads, head_dimension, tokens);
  CUDA_TEST_CHECK(host_attention.ok());
  if (!host_attention.ok()) return;

  DeviceBuffer<float> device_query(query.size());
  DeviceBuffer<float> device_keys(key_cache.size());
  DeviceBuffer<float> device_values(value_cache.size());
  DeviceBuffer<float> device_scores(query_heads * tokens);
  DeviceBuffer<float> device_output(query.size());
  if (device_query.get() == nullptr || device_keys.get() == nullptr ||
      device_values.get() == nullptr || device_scores.get() == nullptr ||
      device_output.get() == nullptr) return;
  if (!CudaOk(cudaMemcpy(device_query.get(), query.data(), device_query.bytes(), cudaMemcpyHostToDevice),
              "copy layer query") ||
      !CudaOk(cudaMemcpy(device_keys.get(), key_cache.data(), device_keys.bytes(), cudaMemcpyHostToDevice),
              "copy layer keys") ||
      !CudaOk(cudaMemcpy(device_values.get(), value_cache.data(), device_values.bytes(), cudaMemcpyHostToDevice),
              "copy layer values")) return;

  const auto attention_status = gem16::internal::LaunchLocalAttentionDecode(
      device_query.get(), device_keys.get(), device_values.get(), device_scores.get(),
      device_output.get(), query_heads, kv_heads, head_dimension, tokens, nullptr);
  CUDA_TEST_CHECK(attention_status.ok());
  if (!attention_status.ok() || !CudaOk(cudaDeviceSynchronize(), "layer attention synchronize")) return;
  std::array<float, query.size()> gpu_attention{};
  if (!CudaOk(cudaMemcpy(gpu_attention.data(), device_output.get(), device_output.bytes(),
                         cudaMemcpyDeviceToHost), "copy layer attention output")) return;
  for (std::size_t index = 0; index < gpu_attention.size(); ++index) {
    CUDA_TEST_CHECK(std::fabs(gpu_attention[index] - host_attention.value()[index]) < 2.0e-5F);
  }

  constexpr std::array<float, 8> norm_input = {
      1.0F, -2.0F, 3.0F, -4.0F, 0.5F, 1.5F, -0.5F, -1.5F};
  constexpr std::array<float, 4> norm_weight = {1.0F, 0.5F, 2.0F, 1.5F};
  constexpr std::array<std::uint16_t, 4> norm_weight_bf16 = {
      0x3F80U, 0x3F00U, 0x4000U, 0x3FC0U};
  const auto host_norm = gem16::layer::RmsNorm(norm_input, norm_weight, 2, 4, 1.0e-6F);
  CUDA_TEST_CHECK(host_norm.ok());
  DeviceBuffer<float> device_norm_input(norm_input.size());
  DeviceBuffer<std::uint16_t> device_norm_weight(norm_weight_bf16.size());
  DeviceBuffer<float> device_norm_output(norm_input.size());
  DeviceBuffer<float> device_norm_rounded_output(norm_input.size());
  DeviceBuffer<float> device_norm_fused_output(norm_input.size());
  if (!host_norm.ok() || device_norm_input.get() == nullptr || device_norm_weight.get() == nullptr ||
      device_norm_output.get() == nullptr ||
      device_norm_rounded_output.get() == nullptr ||
      device_norm_fused_output.get() == nullptr) return;
  if (!CudaOk(cudaMemcpy(device_norm_input.get(), norm_input.data(), device_norm_input.bytes(), cudaMemcpyHostToDevice),
              "copy norm input") ||
      !CudaOk(cudaMemcpy(device_norm_weight.get(), norm_weight_bf16.data(), device_norm_weight.bytes(),
                         cudaMemcpyHostToDevice), "copy norm weight")) return;
  const auto norm_status = gem16::internal::LaunchRmsNorm(
      device_norm_input.get(), device_norm_weight.get(), device_norm_output.get(), 2, 4, 1.0e-6F, nullptr);
  const auto fused_norm_status = gem16::internal::LaunchRmsNormBf16(
      device_norm_input.get(), device_norm_weight.get(),
      device_norm_fused_output.get(), 2, 4, 1.0e-6F, nullptr);
  if (!CudaOk(cudaMemcpy(device_norm_rounded_output.get(),
                         device_norm_output.get(), device_norm_output.bytes(),
                         cudaMemcpyDeviceToDevice),
              "copy RMSNorm output before BF16 rounding")) return;
  RoundBf16ForComparisonKernel<<<1, 256>>>(device_norm_rounded_output.get(),
                                           norm_input.size());
  CUDA_TEST_CHECK(norm_status.ok());
  CUDA_TEST_CHECK(fused_norm_status.ok());
  if (!norm_status.ok() || !fused_norm_status.ok() ||
      !CudaOk(cudaGetLastError(), "launch comparison BF16 rounding") ||
      !CudaOk(cudaDeviceSynchronize(), "RMSNorm synchronize")) return;
  std::array<float, norm_input.size()> gpu_norm{};
  std::array<float, norm_input.size()> gpu_norm_rounded{};
  std::array<float, norm_input.size()> gpu_norm_fused{};
  if (!CudaOk(cudaMemcpy(gpu_norm.data(), device_norm_output.get(), device_norm_output.bytes(),
                         cudaMemcpyDeviceToHost), "copy norm output") ||
      !CudaOk(cudaMemcpy(gpu_norm_rounded.data(),
                         device_norm_rounded_output.get(),
                         device_norm_rounded_output.bytes(),
                         cudaMemcpyDeviceToHost),
              "copy rounded norm output") ||
      !CudaOk(cudaMemcpy(gpu_norm_fused.data(), device_norm_fused_output.get(),
                         device_norm_fused_output.bytes(), cudaMemcpyDeviceToHost),
              "copy fused norm output")) return;
  for (std::size_t index = 0; index < gpu_norm.size(); ++index) {
    CUDA_TEST_CHECK(std::fabs(gpu_norm[index] - host_norm.value()[index]) < 2.0e-6F);
  }
  CUDA_TEST_CHECK(gpu_norm_rounded == gpu_norm_fused);

  DeviceBuffer<std::uint8_t> device_norm_fp8_reference(norm_input.size());
  DeviceBuffer<std::uint8_t> device_norm_fp8_fused(norm_input.size());
  DeviceBuffer<float> device_norm_fp8_reference_scales(2);
  DeviceBuffer<float> device_norm_fp8_fused_scales(2);
  if (device_norm_fp8_reference.get() == nullptr ||
      device_norm_fp8_fused.get() == nullptr ||
      device_norm_fp8_reference_scales.get() == nullptr ||
      device_norm_fp8_fused_scales.get() == nullptr) return;
  const auto norm_fp8_reference_status =
      gem16::internal::LaunchFp8ReferenceTokenQuantizationBatch(
          device_norm_rounded_output.get(), device_norm_fp8_reference.get(),
          device_norm_fp8_reference_scales.get(), 2, 4, nullptr);
  const auto norm_fp8_fused_status =
      gem16::internal::LaunchRmsNormFp8TokenQuantizationBatch(
          device_norm_input.get(), device_norm_weight.get(),
          device_norm_fp8_fused.get(), device_norm_fp8_fused_scales.get(), 2,
          4, 1.0e-6F, nullptr);
  CUDA_TEST_CHECK(norm_fp8_reference_status.ok());
  CUDA_TEST_CHECK(norm_fp8_fused_status.ok());
  if (!norm_fp8_reference_status.ok() || !norm_fp8_fused_status.ok() ||
      !CudaOk(cudaDeviceSynchronize(), "fused RMSNorm FP8 synchronize")) return;
  std::array<std::uint8_t, norm_input.size()> gpu_norm_fp8_reference{};
  std::array<std::uint8_t, norm_input.size()> gpu_norm_fp8_fused{};
  std::array<float, 2> gpu_norm_fp8_reference_scales{};
  std::array<float, 2> gpu_norm_fp8_fused_scales{};
  if (!CudaOk(cudaMemcpy(gpu_norm_fp8_reference.data(),
                         device_norm_fp8_reference.get(),
                         device_norm_fp8_reference.bytes(),
                         cudaMemcpyDeviceToHost),
              "copy reference RMSNorm FP8 output") ||
      !CudaOk(cudaMemcpy(gpu_norm_fp8_fused.data(),
                         device_norm_fp8_fused.get(),
                         device_norm_fp8_fused.bytes(), cudaMemcpyDeviceToHost),
              "copy fused RMSNorm FP8 output") ||
      !CudaOk(cudaMemcpy(gpu_norm_fp8_reference_scales.data(),
                         device_norm_fp8_reference_scales.get(),
                         device_norm_fp8_reference_scales.bytes(),
                         cudaMemcpyDeviceToHost),
              "copy reference RMSNorm FP8 scales") ||
      !CudaOk(cudaMemcpy(gpu_norm_fp8_fused_scales.data(),
                         device_norm_fp8_fused_scales.get(),
                         device_norm_fp8_fused_scales.bytes(),
                         cudaMemcpyDeviceToHost),
              "copy fused RMSNorm FP8 scales")) return;
  CUDA_TEST_CHECK(gpu_norm_fp8_reference == gpu_norm_fp8_fused);
  CUDA_TEST_CHECK(gpu_norm_fp8_reference_scales == gpu_norm_fp8_fused_scales);

  constexpr std::size_t norm_nvfp4_tokens = 2;
  constexpr std::size_t norm_nvfp4_width = 32;
  constexpr std::size_t norm_nvfp4_elements =
      norm_nvfp4_tokens * norm_nvfp4_width;
  std::array<float, norm_nvfp4_elements> norm_nvfp4_input{};
  std::array<std::uint16_t, norm_nvfp4_width> norm_nvfp4_weight{};
  for (std::size_t index = 0; index < norm_nvfp4_input.size(); ++index) {
    norm_nvfp4_input[index] =
        static_cast<float>(static_cast<int>(index % 23U) - 11) * 0.09375F;
  }
  for (std::size_t index = 0; index < norm_nvfp4_weight.size(); ++index) {
    norm_nvfp4_weight[index] = (index & 1U) == 0U ? 0x3F80U : 0x3F00U;
  }
  DeviceBuffer<float> device_norm_nvfp4_input(norm_nvfp4_elements);
  DeviceBuffer<std::uint16_t> device_norm_nvfp4_weight(norm_nvfp4_width);
  DeviceBuffer<float> device_norm_nvfp4_reference(norm_nvfp4_elements);
  DeviceBuffer<std::uint8_t> device_norm_nvfp4_reference_packed(
      norm_nvfp4_elements / 2U);
  DeviceBuffer<std::uint8_t> device_norm_nvfp4_fused_packed(
      norm_nvfp4_elements / 2U);
  DeviceBuffer<std::uint8_t> device_norm_nvfp4_reference_scales(
      norm_nvfp4_elements / 16U);
  DeviceBuffer<std::uint8_t> device_norm_nvfp4_fused_scales(
      norm_nvfp4_elements / 16U);
  if (device_norm_nvfp4_input.get() == nullptr ||
      device_norm_nvfp4_weight.get() == nullptr ||
      device_norm_nvfp4_reference.get() == nullptr ||
      device_norm_nvfp4_reference_packed.get() == nullptr ||
      device_norm_nvfp4_fused_packed.get() == nullptr ||
      device_norm_nvfp4_reference_scales.get() == nullptr ||
      device_norm_nvfp4_fused_scales.get() == nullptr ||
      !CudaOk(cudaMemcpy(device_norm_nvfp4_input.get(),
                         norm_nvfp4_input.data(),
                         device_norm_nvfp4_input.bytes(),
                         cudaMemcpyHostToDevice),
              "copy RMSNorm NVFP4 input") ||
      !CudaOk(cudaMemcpy(device_norm_nvfp4_weight.get(),
                         norm_nvfp4_weight.data(),
                         device_norm_nvfp4_weight.bytes(),
                         cudaMemcpyHostToDevice),
              "copy RMSNorm NVFP4 weight")) return;
  const auto norm_nvfp4_status = gem16::internal::LaunchRmsNormBf16(
      device_norm_nvfp4_input.get(), device_norm_nvfp4_weight.get(),
      device_norm_nvfp4_reference.get(), norm_nvfp4_tokens,
      norm_nvfp4_width, 1.0e-6F, nullptr);
  const auto norm_nvfp4_reference_status =
      gem16::internal::LaunchNvfp4ReferenceActivationQuantization(
          device_norm_nvfp4_reference.get(),
          device_norm_nvfp4_reference_packed.get(),
          device_norm_nvfp4_reference_scales.get(), norm_nvfp4_elements,
          1.25F, nullptr);
  const auto norm_nvfp4_fused_status =
      gem16::internal::LaunchRmsNormNvfp4ActivationQuantizationBatch(
          device_norm_nvfp4_input.get(), device_norm_nvfp4_weight.get(),
          device_norm_nvfp4_fused_packed.get(),
          device_norm_nvfp4_fused_scales.get(), norm_nvfp4_tokens,
          norm_nvfp4_width, 1.0e-6F, 1.25F, nullptr);
  CUDA_TEST_CHECK(norm_nvfp4_status.ok());
  CUDA_TEST_CHECK(norm_nvfp4_reference_status.ok());
  CUDA_TEST_CHECK(norm_nvfp4_fused_status.ok());
  if (!norm_nvfp4_status.ok() || !norm_nvfp4_reference_status.ok() ||
      !norm_nvfp4_fused_status.ok() ||
      !CudaOk(cudaDeviceSynchronize(), "fused RMSNorm NVFP4 synchronize")) return;
  std::array<std::uint8_t, norm_nvfp4_elements / 2U>
      gpu_norm_nvfp4_reference_packed{};
  std::array<std::uint8_t, norm_nvfp4_elements / 2U>
      gpu_norm_nvfp4_fused_packed{};
  std::array<std::uint8_t, norm_nvfp4_elements / 16U>
      gpu_norm_nvfp4_reference_scales{};
  std::array<std::uint8_t, norm_nvfp4_elements / 16U>
      gpu_norm_nvfp4_fused_scales{};
  if (!CudaOk(cudaMemcpy(gpu_norm_nvfp4_reference_packed.data(),
                         device_norm_nvfp4_reference_packed.get(),
                         device_norm_nvfp4_reference_packed.bytes(),
                         cudaMemcpyDeviceToHost),
              "copy reference RMSNorm NVFP4 packed output") ||
      !CudaOk(cudaMemcpy(gpu_norm_nvfp4_fused_packed.data(),
                         device_norm_nvfp4_fused_packed.get(),
                         device_norm_nvfp4_fused_packed.bytes(),
                         cudaMemcpyDeviceToHost),
              "copy fused RMSNorm NVFP4 packed output") ||
      !CudaOk(cudaMemcpy(gpu_norm_nvfp4_reference_scales.data(),
                         device_norm_nvfp4_reference_scales.get(),
                         device_norm_nvfp4_reference_scales.bytes(),
                         cudaMemcpyDeviceToHost),
              "copy reference RMSNorm NVFP4 scales") ||
      !CudaOk(cudaMemcpy(gpu_norm_nvfp4_fused_scales.data(),
                         device_norm_nvfp4_fused_scales.get(),
                         device_norm_nvfp4_fused_scales.bytes(),
                         cudaMemcpyDeviceToHost),
              "copy fused RMSNorm NVFP4 scales")) return;
  CUDA_TEST_CHECK(gpu_norm_nvfp4_reference_packed ==
                  gpu_norm_nvfp4_fused_packed);
  CUDA_TEST_CHECK(gpu_norm_nvfp4_reference_scales ==
                  gpu_norm_nvfp4_fused_scales);

  constexpr std::size_t fused_gelu_elements = 32;
  std::array<float, fused_gelu_elements> fused_gelu_gate{};
  std::array<float, fused_gelu_elements> fused_gelu_up{};
  std::array<std::uint16_t, fused_gelu_elements> fused_gelu_gate_bf16{};
  std::array<std::uint16_t, fused_gelu_elements> fused_gelu_up_bf16{};
  for (std::size_t index = 0; index < fused_gelu_elements; ++index) {
    fused_gelu_gate[index] =
        static_cast<float>(static_cast<int>(index % 19U) - 9) * 0.15625F;
    fused_gelu_up[index] =
        static_cast<float>(static_cast<int>(index % 13U) - 6) * 0.21875F;
    fused_gelu_gate_bf16[index] =
        __bfloat16_as_ushort(__float2bfloat16_rn(fused_gelu_gate[index]));
    fused_gelu_up_bf16[index] =
        __bfloat16_as_ushort(__float2bfloat16_rn(fused_gelu_up[index]));
  }
  DeviceBuffer<float> device_fused_gelu_gate_reference(fused_gelu_elements);
  DeviceBuffer<float> device_fused_gelu_up_reference(fused_gelu_elements);
  DeviceBuffer<float> device_fused_gelu_gate(fused_gelu_elements);
  DeviceBuffer<float> device_fused_gelu_up(fused_gelu_elements);
  DeviceBuffer<std::uint16_t> device_fused_gelu_gate_bf16(fused_gelu_elements);
  DeviceBuffer<std::uint16_t> device_fused_gelu_up_bf16(fused_gelu_elements);
  DeviceBuffer<float> device_fused_gelu_product(fused_gelu_elements);
  DeviceBuffer<std::uint8_t> device_fused_gelu_reference_packed(
      fused_gelu_elements / 2U);
  DeviceBuffer<std::uint8_t> device_fused_gelu_packed(
      fused_gelu_elements / 2U);
  DeviceBuffer<std::uint8_t> device_fused_gelu_bf16_packed(
      fused_gelu_elements / 2U);
  DeviceBuffer<std::uint8_t> device_fused_gelu_reference_scales(
      fused_gelu_elements / 16U);
  DeviceBuffer<std::uint8_t> device_fused_gelu_scales(
      fused_gelu_elements / 16U);
  DeviceBuffer<std::uint8_t> device_fused_gelu_bf16_scales(
      fused_gelu_elements / 16U);
  if (device_fused_gelu_gate_reference.get() == nullptr ||
      device_fused_gelu_up_reference.get() == nullptr ||
      device_fused_gelu_gate.get() == nullptr ||
      device_fused_gelu_up.get() == nullptr ||
      device_fused_gelu_gate_bf16.get() == nullptr ||
      device_fused_gelu_up_bf16.get() == nullptr ||
      device_fused_gelu_product.get() == nullptr ||
      device_fused_gelu_reference_packed.get() == nullptr ||
      device_fused_gelu_packed.get() == nullptr ||
      device_fused_gelu_bf16_packed.get() == nullptr ||
      device_fused_gelu_reference_scales.get() == nullptr ||
      device_fused_gelu_scales.get() == nullptr ||
      device_fused_gelu_bf16_scales.get() == nullptr ||
      !CudaOk(cudaMemcpy(device_fused_gelu_gate_reference.get(),
                         fused_gelu_gate.data(),
                         device_fused_gelu_gate_reference.bytes(),
                         cudaMemcpyHostToDevice),
              "copy reference fused GELU gate") ||
      !CudaOk(cudaMemcpy(device_fused_gelu_up_reference.get(),
                         fused_gelu_up.data(),
                         device_fused_gelu_up_reference.bytes(),
                         cudaMemcpyHostToDevice),
              "copy reference fused GELU up") ||
      !CudaOk(cudaMemcpy(device_fused_gelu_gate.get(), fused_gelu_gate.data(),
                         device_fused_gelu_gate.bytes(),
                         cudaMemcpyHostToDevice),
              "copy fused GELU gate") ||
      !CudaOk(cudaMemcpy(device_fused_gelu_up.get(), fused_gelu_up.data(),
                         device_fused_gelu_up.bytes(),
                         cudaMemcpyHostToDevice),
              "copy fused GELU up") ||
      !CudaOk(cudaMemcpy(device_fused_gelu_gate_bf16.get(),
                         fused_gelu_gate_bf16.data(),
                         device_fused_gelu_gate_bf16.bytes(),
                         cudaMemcpyHostToDevice),
              "copy fused GELU BF16 gate") ||
      !CudaOk(cudaMemcpy(device_fused_gelu_up_bf16.get(),
                         fused_gelu_up_bf16.data(),
                         device_fused_gelu_up_bf16.bytes(),
                         cudaMemcpyHostToDevice),
              "copy fused GELU BF16 up")) return;
  RoundBf16ForComparisonKernel<<<1, 256>>>(
      device_fused_gelu_gate_reference.get(), fused_gelu_elements);
  RoundBf16ForComparisonKernel<<<1, 256>>>(
      device_fused_gelu_up_reference.get(), fused_gelu_elements);
  const auto gelu_product_status = gem16::internal::LaunchGeluTanhProduct(
      device_fused_gelu_gate_reference.get(),
      device_fused_gelu_up_reference.get(), device_fused_gelu_product.get(),
      fused_gelu_elements, nullptr);
  RoundBf16ForComparisonKernel<<<1, 256>>>(device_fused_gelu_product.get(),
                                           fused_gelu_elements);
  const auto gelu_nvfp4_reference_status =
      gem16::internal::LaunchNvfp4ReferenceActivationQuantization(
          device_fused_gelu_product.get(),
          device_fused_gelu_reference_packed.get(),
          device_fused_gelu_reference_scales.get(), fused_gelu_elements,
          1.125F, nullptr);
  const auto gelu_nvfp4_fused_status =
      gem16::internal::LaunchGatedGeluNvfp4ActivationQuantization(
          device_fused_gelu_gate.get(), device_fused_gelu_up.get(),
          device_fused_gelu_packed.get(), device_fused_gelu_scales.get(),
          fused_gelu_elements, 1.125F, nullptr);
  const auto gelu_nvfp4_bf16_status =
      gem16::internal::LaunchGatedGeluNvfp4ActivationQuantizationBf16(
          device_fused_gelu_gate_bf16.get(),
          device_fused_gelu_up_bf16.get(),
          device_fused_gelu_bf16_packed.get(),
          device_fused_gelu_bf16_scales.get(), fused_gelu_elements, 1.125F,
          nullptr);
  CUDA_TEST_CHECK(gelu_product_status.ok());
  CUDA_TEST_CHECK(gelu_nvfp4_reference_status.ok());
  CUDA_TEST_CHECK(gelu_nvfp4_fused_status.ok());
  CUDA_TEST_CHECK(gelu_nvfp4_bf16_status.ok());
  if (!gelu_product_status.ok() || !gelu_nvfp4_reference_status.ok() ||
      !gelu_nvfp4_fused_status.ok() ||
      !gelu_nvfp4_bf16_status.ok() ||
      !CudaOk(cudaGetLastError(), "launch fused GELU comparison kernels") ||
      !CudaOk(cudaDeviceSynchronize(), "fused GELU NVFP4 synchronize")) return;
  std::array<std::uint8_t, fused_gelu_elements / 2U>
      gpu_fused_gelu_reference_packed{};
  std::array<std::uint8_t, fused_gelu_elements / 2U>
      gpu_fused_gelu_packed{};
  std::array<std::uint8_t, fused_gelu_elements / 2U>
      gpu_fused_gelu_bf16_packed{};
  std::array<std::uint8_t, fused_gelu_elements / 16U>
      gpu_fused_gelu_reference_scales{};
  std::array<std::uint8_t, fused_gelu_elements / 16U>
      gpu_fused_gelu_scales{};
  std::array<std::uint8_t, fused_gelu_elements / 16U>
      gpu_fused_gelu_bf16_scales{};
  if (!CudaOk(cudaMemcpy(gpu_fused_gelu_reference_packed.data(),
                         device_fused_gelu_reference_packed.get(),
                         device_fused_gelu_reference_packed.bytes(),
                         cudaMemcpyDeviceToHost),
              "copy reference fused GELU packed output") ||
      !CudaOk(cudaMemcpy(gpu_fused_gelu_packed.data(),
                         device_fused_gelu_packed.get(),
                         device_fused_gelu_packed.bytes(),
                         cudaMemcpyDeviceToHost),
              "copy fused GELU packed output") ||
      !CudaOk(cudaMemcpy(gpu_fused_gelu_bf16_packed.data(),
                         device_fused_gelu_bf16_packed.get(),
                         device_fused_gelu_bf16_packed.bytes(),
                         cudaMemcpyDeviceToHost),
              "copy fused GELU BF16 packed output") ||
      !CudaOk(cudaMemcpy(gpu_fused_gelu_reference_scales.data(),
                         device_fused_gelu_reference_scales.get(),
                         device_fused_gelu_reference_scales.bytes(),
                         cudaMemcpyDeviceToHost),
              "copy reference fused GELU scales") ||
      !CudaOk(cudaMemcpy(gpu_fused_gelu_scales.data(),
                         device_fused_gelu_scales.get(),
                         device_fused_gelu_scales.bytes(),
                         cudaMemcpyDeviceToHost),
              "copy fused GELU scales") ||
      !CudaOk(cudaMemcpy(gpu_fused_gelu_bf16_scales.data(),
                         device_fused_gelu_bf16_scales.get(),
                         device_fused_gelu_bf16_scales.bytes(),
                         cudaMemcpyDeviceToHost),
              "copy fused GELU BF16 scales")) return;
  CUDA_TEST_CHECK(gpu_fused_gelu_reference_packed ==
                  gpu_fused_gelu_packed);
  CUDA_TEST_CHECK(gpu_fused_gelu_reference_packed ==
                  gpu_fused_gelu_bf16_packed);
  CUDA_TEST_CHECK(gpu_fused_gelu_reference_scales ==
                  gpu_fused_gelu_scales);
  CUDA_TEST_CHECK(gpu_fused_gelu_reference_scales ==
                  gpu_fused_gelu_bf16_scales);

  constexpr std::array<float, norm_input.size()> norm_residual = {
      0.25F, -0.5F, 0.75F, -1.0F, 1.25F, -1.5F, 1.75F, -2.0F};
  std::array<std::uint16_t, norm_input.size()> norm_input_bf16{};
  for (std::size_t index = 0; index < norm_input.size(); ++index) {
    norm_input_bf16[index] = static_cast<std::uint16_t>(
        std::bit_cast<std::uint32_t>(norm_input[index]) >> 16U);
  }
  constexpr std::uint16_t norm_layer_scalar_bf16 = 0x3FC0U;
  DeviceBuffer<std::uint16_t> device_norm_input_bf16(norm_input_bf16.size());
  DeviceBuffer<float> device_norm_residual(norm_residual.size());
  DeviceBuffer<float> device_norm_reference_final(norm_residual.size());
  DeviceBuffer<float> device_norm_fused_final(norm_residual.size());
  DeviceBuffer<float> device_norm_bf16_input_normalized(norm_residual.size());
  DeviceBuffer<float> device_norm_bf16_input_final(norm_residual.size());
  DeviceBuffer<std::uint16_t> device_norm_layer_scalar(1);
  if (device_norm_input_bf16.get() == nullptr ||
      device_norm_residual.get() == nullptr ||
      device_norm_reference_final.get() == nullptr ||
      device_norm_fused_final.get() == nullptr ||
      device_norm_bf16_input_normalized.get() == nullptr ||
      device_norm_bf16_input_final.get() == nullptr ||
      device_norm_layer_scalar.get() == nullptr ||
      !CudaOk(cudaMemcpy(device_norm_input_bf16.get(), norm_input_bf16.data(),
                         device_norm_input_bf16.bytes(),
                         cudaMemcpyHostToDevice),
              "copy BF16 RMSNorm input") ||
      !CudaOk(cudaMemcpy(device_norm_residual.get(), norm_residual.data(),
                         device_norm_residual.bytes(), cudaMemcpyHostToDevice),
              "copy RMSNorm residual") ||
      !CudaOk(cudaMemcpy(device_norm_layer_scalar.get(),
                         &norm_layer_scalar_bf16,
                         sizeof(norm_layer_scalar_bf16),
                         cudaMemcpyHostToDevice),
              "copy RMSNorm layer scalar")) return;
  auto residual_status = gem16::internal::LaunchAddResidual(
      device_norm_rounded_output.get(), device_norm_residual.get(),
      device_norm_reference_final.get(), norm_residual.size(), nullptr);
  RoundBf16ForComparisonKernel<<<1, 256>>>(device_norm_reference_final.get(),
                                           norm_residual.size());
  auto fused_residual_status =
      gem16::internal::LaunchRmsNormResidualBf16(
          device_norm_input.get(), device_norm_weight.get(),
          device_norm_residual.get(), device_norm_fused_output.get(),
          device_norm_fused_final.get(), 2, 4, 1.0e-6F, nullptr, nullptr);
  const auto bf16_input_residual_status =
      gem16::internal::LaunchRmsNormResidualBf16Input(
          device_norm_input_bf16.get(), device_norm_weight.get(),
          device_norm_residual.get(), device_norm_bf16_input_normalized.get(),
          device_norm_bf16_input_final.get(), 2, 4, 1.0e-6F, nullptr,
          nullptr);
  CUDA_TEST_CHECK(residual_status.ok());
  CUDA_TEST_CHECK(fused_residual_status.ok());
  CUDA_TEST_CHECK(bf16_input_residual_status.ok());
  if (!residual_status.ok() || !fused_residual_status.ok() ||
      !bf16_input_residual_status.ok() ||
      !CudaOk(cudaGetLastError(), "launch comparison residual BF16 rounding") ||
      !CudaOk(cudaDeviceSynchronize(), "fused RMSNorm residual synchronize")) return;
  std::array<float, norm_input.size()> gpu_norm_reference_final{};
  std::array<float, norm_input.size()> gpu_norm_fused_final{};
  std::array<float, norm_input.size()> gpu_norm_bf16_input_normalized{};
  std::array<float, norm_input.size()> gpu_norm_bf16_input_final{};
  if (!CudaOk(cudaMemcpy(gpu_norm_fused.data(), device_norm_fused_output.get(),
                         device_norm_fused_output.bytes(), cudaMemcpyDeviceToHost),
              "copy fused RMSNorm residual norm") ||
      !CudaOk(cudaMemcpy(gpu_norm_reference_final.data(),
                         device_norm_reference_final.get(),
                         device_norm_reference_final.bytes(),
                         cudaMemcpyDeviceToHost),
              "copy reference RMSNorm residual output") ||
      !CudaOk(cudaMemcpy(gpu_norm_fused_final.data(),
                         device_norm_fused_final.get(),
                         device_norm_fused_final.bytes(), cudaMemcpyDeviceToHost),
              "copy fused RMSNorm residual output") ||
      !CudaOk(cudaMemcpy(gpu_norm_bf16_input_normalized.data(),
                         device_norm_bf16_input_normalized.get(),
                         device_norm_bf16_input_normalized.bytes(),
                         cudaMemcpyDeviceToHost),
              "copy BF16-input RMSNorm normalized output") ||
      !CudaOk(cudaMemcpy(gpu_norm_bf16_input_final.data(),
                         device_norm_bf16_input_final.get(),
                         device_norm_bf16_input_final.bytes(),
                         cudaMemcpyDeviceToHost),
              "copy BF16-input RMSNorm residual output")) return;
  CUDA_TEST_CHECK(gpu_norm_rounded == gpu_norm_fused);
  CUDA_TEST_CHECK(gpu_norm_reference_final == gpu_norm_fused_final);
  CUDA_TEST_CHECK(gpu_norm_fused == gpu_norm_bf16_input_normalized);
  CUDA_TEST_CHECK(gpu_norm_fused_final == gpu_norm_bf16_input_final);

  const auto scalar_status = gem16::internal::LaunchScale(
      device_norm_reference_final.get(), device_norm_layer_scalar.get(),
      norm_residual.size(), nullptr);
  RoundBf16ForComparisonKernel<<<1, 256>>>(device_norm_reference_final.get(),
                                           norm_residual.size());
  fused_residual_status = gem16::internal::LaunchRmsNormResidualBf16(
      device_norm_input.get(), device_norm_weight.get(),
      device_norm_residual.get(), device_norm_fused_output.get(),
      device_norm_fused_final.get(), 2, 4, 1.0e-6F,
      device_norm_layer_scalar.get(), nullptr);
  CUDA_TEST_CHECK(scalar_status.ok());
  CUDA_TEST_CHECK(fused_residual_status.ok());
  if (!scalar_status.ok() || !fused_residual_status.ok() ||
      !CudaOk(cudaGetLastError(), "launch comparison scalar BF16 rounding") ||
      !CudaOk(cudaDeviceSynchronize(), "fused scaled RMSNorm residual synchronize")) return;
  if (!CudaOk(cudaMemcpy(gpu_norm_reference_final.data(),
                         device_norm_reference_final.get(),
                         device_norm_reference_final.bytes(),
                         cudaMemcpyDeviceToHost),
              "copy reference scaled RMSNorm residual output") ||
      !CudaOk(cudaMemcpy(gpu_norm_fused_final.data(),
                         device_norm_fused_final.get(),
                         device_norm_fused_final.bytes(), cudaMemcpyDeviceToHost),
              "copy fused scaled RMSNorm residual output")) return;
  CUDA_TEST_CHECK(gpu_norm_reference_final == gpu_norm_fused_final);

  std::array<float, 8> host_rope = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
  DeviceBuffer<float> device_rope(host_rope.size());
  if (device_rope.get() == nullptr ||
      !CudaOk(cudaMemcpy(device_rope.get(), host_rope.data(), device_rope.bytes(), cudaMemcpyHostToDevice),
              "copy RoPE input")) return;
  CUDA_TEST_CHECK(gem16::layer::ApplyRotaryEmbedding(host_rope, 1, 8, 8, 37, 10000.0).ok());
  const auto rope_status = gem16::internal::LaunchRotaryEmbedding(
      device_rope.get(), 1, 8, 8, 37, 10000.0, nullptr);
  CUDA_TEST_CHECK(rope_status.ok());
  if (!rope_status.ok() || !CudaOk(cudaDeviceSynchronize(), "RoPE synchronize")) return;
  std::array<float, 8> gpu_rope{};
  if (!CudaOk(cudaMemcpy(gpu_rope.data(), device_rope.get(), device_rope.bytes(), cudaMemcpyDeviceToHost),
              "copy RoPE output")) return;
  for (std::size_t index = 0; index < gpu_rope.size(); ++index) {
    CUDA_TEST_CHECK(std::fabs(gpu_rope[index] - host_rope[index]) < 2.0e-6F);
  }

  std::vector<float> host_proportional_rope(512);
  for (std::size_t index = 0; index < host_proportional_rope.size(); ++index) {
    host_proportional_rope[index] = static_cast<float>(index + 1U) * 0.002F;
  }
  DeviceBuffer<float> device_proportional_rope(host_proportional_rope.size());
  if (device_proportional_rope.get() == nullptr ||
      !CudaOk(cudaMemcpy(device_proportional_rope.get(), host_proportional_rope.data(),
                         device_proportional_rope.bytes(), cudaMemcpyHostToDevice),
              "copy proportional RoPE input")) return;
  CUDA_TEST_CHECK(gem16::layer::ApplyProportionalRotaryEmbedding(
                      host_proportional_rope, 1, 512, 0.25, 31, 1'000'000.0)
                      .ok());
  const auto proportional_status = gem16::internal::LaunchProportionalRotaryEmbedding(
      device_proportional_rope.get(), 1, 512, 0.25, 31, 1'000'000.0, 1.0, nullptr);
  CUDA_TEST_CHECK(proportional_status.ok());
  if (!proportional_status.ok() ||
      !CudaOk(cudaDeviceSynchronize(), "proportional RoPE synchronize")) return;
  std::vector<float> gpu_proportional_rope(host_proportional_rope.size());
  if (!CudaOk(cudaMemcpy(gpu_proportional_rope.data(), device_proportional_rope.get(),
                         device_proportional_rope.bytes(), cudaMemcpyDeviceToHost),
              "copy proportional RoPE output")) return;
  for (std::size_t index = 0; index < gpu_proportional_rope.size(); ++index) {
    CUDA_TEST_CHECK(std::fabs(gpu_proportional_rope[index] -
                              host_proportional_rope[index]) < 2.0e-6F);
  }
}

void TestFusedProjectionRmsNormRotaryBf16Batch() {
  const auto run_case = [](std::uint64_t head_dimension,
                           double rotary_factor, double theta,
                           const char* label) {
    constexpr std::uint64_t tokens = 3;
    constexpr std::uint64_t query_heads = 2;
    constexpr std::uint64_t kv_heads = 1;
    constexpr std::uint64_t start_position = 29;
    const std::size_t query_elements =
        tokens * query_heads * head_dimension;
    const std::size_t key_elements = tokens * kv_heads * head_dimension;
    std::vector<float> query(query_elements);
    std::vector<float> key(key_elements);
    std::vector<std::uint16_t> query_norm(head_dimension);
    std::vector<std::uint16_t> key_norm(head_dimension);
    for (std::size_t index = 0; index < query.size(); ++index) {
      query[index] = static_cast<float>(
          static_cast<int>((index * 17U + 3U) % 61U) - 30) * 0.0173F;
    }
    for (std::size_t index = 0; index < key.size(); ++index) {
      key[index] = static_cast<float>(
          static_cast<int>((index * 13U + 5U) % 53U) - 26) * 0.0197F;
    }
    for (std::size_t index = 0; index < head_dimension; ++index) {
      query_norm[index] = (index % 3U) == 0U ? 0x3F80U : 0x3F00U;
      key_norm[index] = (index % 5U) == 0U ? 0x3FC0U : 0x3F80U;
    }

    DeviceBuffer<float> reference_query(query_elements);
    DeviceBuffer<float> reference_key(key_elements);
    DeviceBuffer<float> candidate_query(query_elements);
    DeviceBuffer<float> candidate_key(key_elements);
    DeviceBuffer<float> reference_query_output(query_elements);
    DeviceBuffer<float> reference_key_output(key_elements);
    DeviceBuffer<float> candidate_query_output(query_elements);
    DeviceBuffer<float> candidate_key_output(key_elements);
    const std::uint64_t rotating_pairs = static_cast<std::uint64_t>(
        rotary_factor * static_cast<double>(head_dimension / 2U));
    DeviceBuffer<float> rotary_cosine(tokens * rotating_pairs);
    DeviceBuffer<float> rotary_sine(tokens * rotating_pairs);
    DeviceBuffer<std::uint16_t> device_query_norm(head_dimension);
    DeviceBuffer<std::uint16_t> device_key_norm(head_dimension);
    if (reference_query.get() == nullptr || reference_key.get() == nullptr ||
        candidate_query.get() == nullptr || candidate_key.get() == nullptr ||
        reference_query_output.get() == nullptr ||
        reference_key_output.get() == nullptr ||
        candidate_query_output.get() == nullptr ||
        candidate_key_output.get() == nullptr ||
        rotary_cosine.get() == nullptr || rotary_sine.get() == nullptr ||
        device_query_norm.get() == nullptr || device_key_norm.get() == nullptr ||
        !CudaOk(cudaMemcpy(reference_query.get(), query.data(),
                           reference_query.bytes(), cudaMemcpyHostToDevice),
                "copy fused Q reference input") ||
        !CudaOk(cudaMemcpy(reference_key.get(), key.data(),
                           reference_key.bytes(), cudaMemcpyHostToDevice),
                "copy fused K reference input") ||
        !CudaOk(cudaMemcpy(candidate_query.get(), query.data(),
                           candidate_query.bytes(), cudaMemcpyHostToDevice),
                "copy fused Q candidate input") ||
        !CudaOk(cudaMemcpy(candidate_key.get(), key.data(),
                           candidate_key.bytes(), cudaMemcpyHostToDevice),
                "copy fused K candidate input") ||
        !CudaOk(cudaMemcpy(device_query_norm.get(), query_norm.data(),
                           device_query_norm.bytes(), cudaMemcpyHostToDevice),
                "copy fused Q norm") ||
        !CudaOk(cudaMemcpy(device_key_norm.get(), key_norm.data(),
                           device_key_norm.bytes(), cudaMemcpyHostToDevice),
                "copy fused K norm")) {
      return;
    }

    RoundBf16ForComparisonKernel<<<
        static_cast<unsigned>((query_elements + 255U) / 256U), 256>>>(
        reference_query.get(), query_elements);
    RoundBf16ForComparisonKernel<<<
        static_cast<unsigned>((key_elements + 255U) / 256U), 256>>>(
        reference_key.get(), key_elements);
    auto status = gem16::internal::LaunchRmsNormBf16(
        reference_query.get(), device_query_norm.get(),
        reference_query_output.get(), tokens * query_heads, head_dimension,
        1.0e-6F, nullptr);
    CUDA_TEST_CHECK(status.ok());
    status = gem16::internal::LaunchRmsNormBf16(
        reference_key.get(), device_key_norm.get(), reference_key_output.get(),
        tokens * kv_heads, head_dimension, 1.0e-6F, nullptr);
    CUDA_TEST_CHECK(status.ok());
    if (rotary_factor == 1.0) {
      status = gem16::internal::LaunchRotaryEmbeddingBatch(
          reference_query_output.get(), tokens, query_heads, head_dimension,
          head_dimension, start_position, theta, nullptr);
      CUDA_TEST_CHECK(status.ok());
      status = gem16::internal::LaunchRotaryEmbeddingBatch(
          reference_key_output.get(), tokens, kv_heads, head_dimension,
          head_dimension, start_position, theta, nullptr);
    } else {
      status = gem16::internal::LaunchProportionalRotaryEmbeddingBatch(
          reference_query_output.get(), tokens, query_heads, head_dimension,
          rotary_factor, start_position, theta, 1.0, nullptr);
      CUDA_TEST_CHECK(status.ok());
      status = gem16::internal::LaunchProportionalRotaryEmbeddingBatch(
          reference_key_output.get(), tokens, kv_heads, head_dimension,
          rotary_factor, start_position, theta, 1.0, nullptr);
    }
    CUDA_TEST_CHECK(status.ok());
    RoundBf16ForComparisonKernel<<<
        static_cast<unsigned>((query_elements + 255U) / 256U), 256>>>(
        reference_query_output.get(), query_elements);
    RoundBf16ForComparisonKernel<<<
        static_cast<unsigned>((key_elements + 255U) / 256U), 256>>>(
        reference_key_output.get(), key_elements);

    const auto table_status =
        gem16::internal::LaunchRotaryEmbeddingTableBatch(
            rotary_cosine.get(), rotary_sine.get(), tokens, rotating_pairs,
            head_dimension, start_position, theta, 1.0, nullptr);
    const auto fused_status =
        gem16::internal::LaunchProjectionRmsNormRotaryBf16Batch(
            candidate_query.get(), device_query_norm.get(),
            candidate_query_output.get(), candidate_key.get(),
            device_key_norm.get(), candidate_key_output.get(),
            rotary_cosine.get(), rotary_sine.get(), tokens, query_heads,
            kv_heads, head_dimension, rotary_factor, 1.0e-6F, nullptr);
    CUDA_TEST_CHECK(table_status.ok());
    CUDA_TEST_CHECK(fused_status.ok());
    if (!status.ok() || !table_status.ok() || !fused_status.ok() ||
        !CudaOk(cudaGetLastError(), "launch fused Q/K comparison kernels") ||
        !CudaOk(cudaDeviceSynchronize(), label)) {
      return;
    }

    std::vector<float> reference_query_host(query_elements);
    std::vector<float> reference_key_host(key_elements);
    std::vector<float> candidate_query_host(query_elements);
    std::vector<float> candidate_key_host(key_elements);
    if (!CudaOk(cudaMemcpy(reference_query_host.data(),
                           reference_query_output.get(),
                           reference_query_output.bytes(),
                           cudaMemcpyDeviceToHost),
                "copy reference fused Q output") ||
        !CudaOk(cudaMemcpy(reference_key_host.data(),
                           reference_key_output.get(),
                           reference_key_output.bytes(), cudaMemcpyDeviceToHost),
                "copy reference fused K output") ||
        !CudaOk(cudaMemcpy(candidate_query_host.data(),
                           candidate_query_output.get(),
                           candidate_query_output.bytes(),
                           cudaMemcpyDeviceToHost),
                "copy candidate fused Q output") ||
        !CudaOk(cudaMemcpy(candidate_key_host.data(),
                           candidate_key_output.get(),
                           candidate_key_output.bytes(), cudaMemcpyDeviceToHost),
                "copy candidate fused K output")) {
      return;
    }
    CUDA_TEST_CHECK(reference_query_host == candidate_query_host);
    CUDA_TEST_CHECK(reference_key_host == candidate_key_host);

    DeviceBuffer<float> controlled_query_output(query_heads * head_dimension);
    DeviceBuffer<float> controlled_key_output(kv_heads * head_dimension);
    DeviceBuffer<float> controlled_cosine((start_position + 1U) * rotating_pairs);
    DeviceBuffer<float> controlled_sine((start_position + 1U) * rotating_pairs);
    DeviceBuffer<gem16::internal::DecodeControl> control(1U);
    gem16::internal::DecodeControl host_control{};
    host_control.position = start_position;
    if (controlled_query_output.get() == nullptr ||
        controlled_key_output.get() == nullptr ||
        controlled_cosine.get() == nullptr || controlled_sine.get() == nullptr ||
        control.get() == nullptr ||
        !CudaOk(cudaMemcpy(control.get(), &host_control, sizeof(host_control),
                           cudaMemcpyHostToDevice),
                "copy controlled fused Q/K state")) {
      return;
    }
    const auto controlled_table_status =
        gem16::internal::LaunchRotaryEmbeddingTableBatch(
            controlled_cosine.get(), controlled_sine.get(), start_position + 1U,
            rotating_pairs, head_dimension, 0U, theta, 1.0, nullptr);
    const auto controlled_status =
        gem16::internal::LaunchProjectionRmsNormRotaryBf16Controlled(
            candidate_query.get(), device_query_norm.get(),
            controlled_query_output.get(), candidate_key.get(),
            device_key_norm.get(), controlled_key_output.get(),
            controlled_cosine.get(), controlled_sine.get(), control.get(),
            query_heads, kv_heads, head_dimension, rotary_factor, 1.0e-6F,
            nullptr);
    CUDA_TEST_CHECK(controlled_table_status.ok());
    CUDA_TEST_CHECK(controlled_status.ok());
    if (!controlled_table_status.ok() || !controlled_status.ok() ||
        !CudaOk(cudaDeviceSynchronize(), "controlled fused Q/K synchronize")) {
      return;
    }
    std::vector<float> controlled_query_host(query_heads * head_dimension);
    std::vector<float> controlled_key_host(kv_heads * head_dimension);
    if (!CudaOk(cudaMemcpy(controlled_query_host.data(),
                           controlled_query_output.get(),
                           controlled_query_output.bytes(),
                           cudaMemcpyDeviceToHost),
                "copy controlled fused Q output") ||
        !CudaOk(cudaMemcpy(controlled_key_host.data(), controlled_key_output.get(),
                           controlled_key_output.bytes(),
                           cudaMemcpyDeviceToHost),
                "copy controlled fused K output")) {
      return;
    }
    CUDA_TEST_CHECK(std::equal(controlled_query_host.begin(),
                               controlled_query_host.end(),
                               candidate_query_host.begin()));
    CUDA_TEST_CHECK(std::equal(controlled_key_host.begin(),
                               controlled_key_host.end(),
                               candidate_key_host.begin()));
  };

  run_case(256U, 1.0, 10000.0, "fused local Q/K RMSNorm RoPE");
  run_case(512U, 0.25, 1000000.0, "fused global Q/K RMSNorm RoPE");
}

void TestPhysicalFp8KvCache() {
  constexpr std::uint64_t query_heads = 2;
  constexpr std::uint64_t kv_heads = 1;
  constexpr std::uint64_t head_dimension = 4;
  constexpr std::uint64_t tokens = 2;
  constexpr std::array<std::uint16_t, 1> key_scale = {0x3F00U};    // 0.5
  constexpr std::array<std::uint16_t, 1> value_scale = {0x3E80U};  // 0.25
  constexpr std::array<float, 8> query = {
      1.0F, 0.5F, -0.5F, 0.25F, -0.5F, 1.0F, 0.25F, -0.25F};
  constexpr std::array<float, 8> keys = {
      0.5F, 1.0F, -0.5F, 0.25F, 1.5F, -1.0F, 0.75F, 0.0F};
  constexpr std::array<float, 8> values = {
      0.25F, 0.5F, -0.25F, 0.125F, 0.75F, -0.5F, 0.375F, 0.0F};

  DeviceBuffer<float> device_query(query.size());
  DeviceBuffer<float> device_keys(head_dimension);
  DeviceBuffer<float> device_values(head_dimension);
  DeviceBuffer<float> device_float_keys(keys.size());
  DeviceBuffer<float> device_float_values(values.size());
  DeviceBuffer<std::uint8_t> device_fp8_keys(keys.size());
  DeviceBuffer<std::uint8_t> device_fp8_values(values.size());
  DeviceBuffer<std::uint16_t> device_key_scale(key_scale.size());
  DeviceBuffer<std::uint16_t> device_value_scale(value_scale.size());
  DeviceBuffer<float> device_scores(query_heads * tokens);
  DeviceBuffer<float> device_fp8_output(query.size());
  DeviceBuffer<float> device_float_output(query.size());
  if (device_query.get() == nullptr || device_keys.get() == nullptr ||
      device_values.get() == nullptr || device_float_keys.get() == nullptr ||
      device_float_values.get() == nullptr || device_fp8_keys.get() == nullptr ||
      device_fp8_values.get() == nullptr || device_key_scale.get() == nullptr ||
      device_value_scale.get() == nullptr || device_scores.get() == nullptr ||
      device_fp8_output.get() == nullptr || device_float_output.get() == nullptr) {
    return;
  }
  if (!CudaOk(cudaMemcpy(device_query.get(), query.data(), device_query.bytes(),
                         cudaMemcpyHostToDevice), "copy FP8-cache query") ||
      !CudaOk(cudaMemcpy(device_float_keys.get(), keys.data(),
                         device_float_keys.bytes(), cudaMemcpyHostToDevice),
              "copy float-cache keys") ||
      !CudaOk(cudaMemcpy(device_float_values.get(), values.data(),
                         device_float_values.bytes(), cudaMemcpyHostToDevice),
              "copy float-cache values") ||
      !CudaOk(cudaMemcpy(device_key_scale.get(), key_scale.data(),
                         device_key_scale.bytes(), cudaMemcpyHostToDevice),
              "copy K cache scale") ||
      !CudaOk(cudaMemcpy(device_value_scale.get(), value_scale.data(),
                         device_value_scale.bytes(), cudaMemcpyHostToDevice),
              "copy V cache scale")) {
    return;
  }
  for (std::uint64_t token = 0; token < tokens; ++token) {
    if (!CudaOk(cudaMemcpy(device_keys.get(),
                           keys.data() + token * head_dimension,
                           device_keys.bytes(), cudaMemcpyHostToDevice),
                "copy FP8-cache K input") ||
        !CudaOk(cudaMemcpy(device_values.get(),
                           values.data() + token * head_dimension,
                           device_values.bytes(), cudaMemcpyHostToDevice),
                "copy FP8-cache V input")) {
      return;
    }
    const auto append = gem16::internal::LaunchAppendKvFp8(
        device_keys.get(), device_values.get(), device_fp8_keys.get(),
        device_fp8_values.get(), device_key_scale.get(),
        device_value_scale.get(), token, kv_heads, head_dimension, nullptr);
    CUDA_TEST_CHECK(append.ok());
    if (!append.ok()) return;
  }
  const auto fp8_attention = gem16::internal::LaunchLocalAttentionDecodeFp8(
      device_query.get(), device_fp8_keys.get(), device_fp8_values.get(),
      device_key_scale.get(), device_value_scale.get(), device_scores.get(),
      device_fp8_output.get(), query_heads, kv_heads, head_dimension, tokens,
      nullptr);
  CUDA_TEST_CHECK(fp8_attention.ok());
  const auto float_attention = gem16::internal::LaunchLocalAttentionDecode(
      device_query.get(), device_float_keys.get(), device_float_values.get(),
      device_scores.get(), device_float_output.get(), query_heads, kv_heads,
      head_dimension, tokens, nullptr);
  CUDA_TEST_CHECK(float_attention.ok());
  if (!fp8_attention.ok() || !float_attention.ok() ||
      !CudaOk(cudaDeviceSynchronize(), "physical FP8 cache synchronize")) {
    return;
  }
  std::array<float, query.size()> fp8_output{};
  std::array<float, query.size()> float_output{};
  if (!CudaOk(cudaMemcpy(fp8_output.data(), device_fp8_output.get(),
                         device_fp8_output.bytes(), cudaMemcpyDeviceToHost),
              "copy FP8-cache output") ||
      !CudaOk(cudaMemcpy(float_output.data(), device_float_output.get(),
                         device_float_output.bytes(), cudaMemcpyDeviceToHost),
              "copy float-cache output")) {
    return;
  }
  for (std::size_t index = 0; index < fp8_output.size(); ++index) {
    CUDA_TEST_CHECK(std::fabs(fp8_output[index] - float_output[index]) <
                    2.0e-6F);
  }
}

void TestWrappedKvRingAttention() {
  constexpr std::uint64_t query_heads = 1;
  constexpr std::uint64_t kv_heads = 1;
  constexpr std::uint64_t head_dimension = 2;
  constexpr std::uint64_t tokens = 3;
  constexpr std::uint64_t first_slot = 1;
  constexpr std::array<float, 2> query = {1.0F, 0.0F};
  constexpr std::array<float, 6> logical_keys = {
      1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F};
  constexpr std::array<float, 6> logical_values = {
      1.0F, 10.0F, 2.0F, 20.0F, 3.0F, 30.0F};
  constexpr std::array<float, 6> physical_keys = {
      1.0F, 1.0F, 1.0F, 0.0F, 0.0F, 1.0F};
  constexpr std::array<float, 6> physical_values = {
      3.0F, 30.0F, 1.0F, 10.0F, 2.0F, 20.0F};
  constexpr std::array<std::uint16_t, 1> unit_scale = {0x3F80U};

  const auto expected = gem16::layer::LocalAttentionDecode(
      query, logical_keys, logical_values, query_heads, kv_heads,
      head_dimension, tokens);
  CUDA_TEST_CHECK(expected.ok());
  if (!expected.ok()) return;

  DeviceBuffer<float> device_query(query.size());
  DeviceBuffer<float> device_keys(physical_keys.size());
  DeviceBuffer<float> device_values(physical_values.size());
  DeviceBuffer<float> device_scores(query_heads * tokens);
  DeviceBuffer<float> device_output(query.size());
  DeviceBuffer<float> controlled_scores(query_heads * tokens);
  DeviceBuffer<float> controlled_output(query.size());
  DeviceBuffer<float> append_key(head_dimension);
  DeviceBuffer<float> append_value(head_dimension);
  DeviceBuffer<std::uint8_t> fp8_keys(physical_keys.size());
  DeviceBuffer<std::uint8_t> fp8_values(physical_values.size());
  DeviceBuffer<std::uint16_t> device_scale(unit_scale.size());
  DeviceBuffer<float> fp8_output(query.size());
  DeviceBuffer<float> controlled_fp8_output(query.size());
  DeviceBuffer<gem16::internal::DecodeControl> device_control(1);
  if (device_query.get() == nullptr || device_keys.get() == nullptr ||
      device_values.get() == nullptr || device_scores.get() == nullptr ||
      device_output.get() == nullptr || controlled_scores.get() == nullptr ||
      controlled_output.get() == nullptr || append_key.get() == nullptr ||
      append_value.get() == nullptr || fp8_keys.get() == nullptr ||
      fp8_values.get() == nullptr || device_scale.get() == nullptr ||
      fp8_output.get() == nullptr || controlled_fp8_output.get() == nullptr ||
      device_control.get() == nullptr) {
    return;
  }
  constexpr gem16::internal::DecodeControl control = {
      .token = 0, .suppressed_token_count = 0, .position = 3};
  if (!CudaOk(cudaMemcpy(device_query.get(), query.data(), device_query.bytes(),
                         cudaMemcpyHostToDevice), "copy ring query") ||
      !CudaOk(cudaMemcpy(device_keys.get(), physical_keys.data(), device_keys.bytes(),
                         cudaMemcpyHostToDevice), "copy ring keys") ||
      !CudaOk(cudaMemcpy(device_values.get(), physical_values.data(),
                         device_values.bytes(), cudaMemcpyHostToDevice),
              "copy ring values") ||
      !CudaOk(cudaMemcpy(device_scale.get(), unit_scale.data(), device_scale.bytes(),
                         cudaMemcpyHostToDevice), "copy ring FP8 scale") ||
      !CudaOk(cudaMemcpy(device_control.get(), &control, sizeof(control),
                         cudaMemcpyHostToDevice), "copy decode control")) {
    return;
  }

  const auto float_status = gem16::internal::LaunchLocalAttentionDecode(
      device_query.get(), device_keys.get(), device_values.get(),
      device_scores.get(), device_output.get(), query_heads, kv_heads,
      head_dimension, tokens, nullptr, tokens, first_slot);
  CUDA_TEST_CHECK(float_status.ok());
  const auto controlled_float_status =
      gem16::internal::LaunchLocalAttentionDecodeControlled(
          device_query.get(), device_keys.get(), device_values.get(),
          controlled_scores.get(), controlled_output.get(), device_control.get(),
          query_heads, kv_heads, head_dimension, tokens, true, nullptr);
  CUDA_TEST_CHECK(controlled_float_status.ok());
  for (std::uint64_t token = 0; token < tokens; ++token) {
    if (!CudaOk(cudaMemcpy(append_key.get(),
                           logical_keys.data() + token * head_dimension,
                           append_key.bytes(), cudaMemcpyHostToDevice),
                "copy ring FP8 key") ||
        !CudaOk(cudaMemcpy(append_value.get(),
                           logical_values.data() + token * head_dimension,
                           append_value.bytes(), cudaMemcpyHostToDevice),
                "copy ring FP8 value")) {
      return;
    }
    const std::uint64_t physical_slot = (first_slot + token) % tokens;
    const auto append_status = gem16::internal::LaunchAppendKvFp8(
        append_key.get(), append_value.get(), fp8_keys.get(), fp8_values.get(),
        device_scale.get(), device_scale.get(), physical_slot, kv_heads,
        head_dimension, nullptr);
    CUDA_TEST_CHECK(append_status.ok());
    if (!append_status.ok()) return;
  }
  const auto fp8_status = gem16::internal::LaunchLocalAttentionDecodeFp8(
      device_query.get(), fp8_keys.get(), fp8_values.get(), device_scale.get(),
      device_scale.get(), device_scores.get(), fp8_output.get(), query_heads,
      kv_heads, head_dimension, tokens, nullptr, tokens, first_slot);
  const auto controlled_fp8_status =
      gem16::internal::LaunchLocalAttentionDecodeFp8Controlled(
          device_query.get(), fp8_keys.get(), fp8_values.get(),
          device_scale.get(), device_scale.get(), controlled_scores.get(),
          controlled_fp8_output.get(), device_control.get(), query_heads,
          kv_heads, head_dimension, tokens, true, nullptr);
  CUDA_TEST_CHECK(fp8_status.ok());
  CUDA_TEST_CHECK(controlled_fp8_status.ok());
  if (!float_status.ok() || !controlled_float_status.ok() ||
      !fp8_status.ok() || !controlled_fp8_status.ok() ||
      !CudaOk(cudaDeviceSynchronize(), "ring attention synchronize")) {
    return;
  }
  std::array<float, query.size()> float_result{};
  std::array<float, query.size()> controlled_float_result{};
  std::array<float, query.size()> fp8_result{};
  std::array<float, query.size()> controlled_fp8_result{};
  if (!CudaOk(cudaMemcpy(float_result.data(), device_output.get(),
                         device_output.bytes(), cudaMemcpyDeviceToHost),
              "copy ring float output") ||
      !CudaOk(cudaMemcpy(controlled_float_result.data(), controlled_output.get(),
                         controlled_output.bytes(), cudaMemcpyDeviceToHost),
              "copy controlled ring float output") ||
      !CudaOk(cudaMemcpy(fp8_result.data(), fp8_output.get(), fp8_output.bytes(),
                         cudaMemcpyDeviceToHost), "copy ring FP8 output") ||
      !CudaOk(cudaMemcpy(controlled_fp8_result.data(),
                         controlled_fp8_output.get(),
                         controlled_fp8_output.bytes(), cudaMemcpyDeviceToHost),
              "copy controlled ring FP8 output")) {
    return;
  }
  for (std::size_t index = 0; index < expected.value().size(); ++index) {
    CUDA_TEST_CHECK(std::fabs(float_result[index] - expected.value()[index]) <
                    2.0e-5F);
    CUDA_TEST_CHECK(controlled_float_result[index] == float_result[index]);
    CUDA_TEST_CHECK(std::fabs(fp8_result[index] - expected.value()[index]) <
                    2.0e-5F);
    CUDA_TEST_CHECK(controlled_fp8_result[index] == fp8_result[index]);
  }
}

void TestCausalPrefillAcrossWrappedRing() {
  constexpr std::uint64_t tokens = 2;
  constexpr std::uint64_t query_heads = 1;
  constexpr std::uint64_t kv_heads = 1;
  constexpr std::uint64_t head_dimension = 2;
  constexpr std::uint64_t capacity = 3;
  constexpr std::uint64_t start_position = 4;
  constexpr std::array<float, 4> queries = {1.0F, 0.0F, 0.5F, 0.5F};
  // Before the chunk: slot 0 = absolute 3, slot 1 = stale absolute 1,
  // slot 2 = absolute 2. Current positions 4 and 5 remain in staging.
  constexpr std::array<float, 6> cache_keys = {
      1.0F, 1.0F, 0.0F, 1.0F, 1.0F, 0.0F};
  constexpr std::array<float, 6> cache_values = {
      4.0F, 40.0F, 2.0F, 20.0F, 3.0F, 30.0F};
  constexpr std::array<float, 4> chunk_keys = {
      2.0F, 0.0F, 0.0F, 2.0F};
  constexpr std::array<float, 4> chunk_values = {
      5.0F, 50.0F, 6.0F, 60.0F};
  constexpr std::array<float, 6> first_logical_keys = {
      1.0F, 0.0F, 1.0F, 1.0F, 2.0F, 0.0F};
  constexpr std::array<float, 6> first_logical_values = {
      3.0F, 30.0F, 4.0F, 40.0F, 5.0F, 50.0F};
  constexpr std::array<float, 6> second_logical_keys = {
      1.0F, 1.0F, 2.0F, 0.0F, 0.0F, 2.0F};
  constexpr std::array<float, 6> second_logical_values = {
      4.0F, 40.0F, 5.0F, 50.0F, 6.0F, 60.0F};
  const auto first_expected = gem16::layer::LocalAttentionDecode(
      std::span<const float>(queries.data(), head_dimension),
      first_logical_keys, first_logical_values, query_heads, kv_heads,
      head_dimension, capacity);
  const auto second_expected = gem16::layer::LocalAttentionDecode(
      std::span<const float>(queries.data() + head_dimension, head_dimension),
      second_logical_keys, second_logical_values, query_heads, kv_heads,
      head_dimension, capacity);
  CUDA_TEST_CHECK(first_expected.ok());
  CUDA_TEST_CHECK(second_expected.ok());
  if (!first_expected.ok() || !second_expected.ok()) return;

  DeviceBuffer<float> device_queries(queries.size());
  DeviceBuffer<float> device_cache_keys(cache_keys.size());
  DeviceBuffer<float> device_cache_values(cache_values.size());
  DeviceBuffer<float> device_chunk_keys(chunk_keys.size());
  DeviceBuffer<float> device_chunk_values(chunk_values.size());
  DeviceBuffer<float> device_scores(tokens * query_heads * capacity);
  DeviceBuffer<float> device_fused_scores(tokens * query_heads * capacity);
  DeviceBuffer<float> device_output(queries.size());
  DeviceBuffer<float> device_fused_output(queries.size());
  if (device_queries.get() == nullptr || device_cache_keys.get() == nullptr ||
      device_cache_values.get() == nullptr || device_chunk_keys.get() == nullptr ||
      device_chunk_values.get() == nullptr || device_scores.get() == nullptr ||
      device_fused_scores.get() == nullptr || device_output.get() == nullptr ||
      device_fused_output.get() == nullptr) {
    return;
  }
  if (!CudaOk(cudaMemcpy(device_queries.get(), queries.data(), device_queries.bytes(),
                         cudaMemcpyHostToDevice), "copy prefill queries") ||
      !CudaOk(cudaMemcpy(device_cache_keys.get(), cache_keys.data(),
                         device_cache_keys.bytes(), cudaMemcpyHostToDevice),
              "copy prefill cache keys") ||
      !CudaOk(cudaMemcpy(device_cache_values.get(), cache_values.data(),
                         device_cache_values.bytes(), cudaMemcpyHostToDevice),
              "copy prefill cache values") ||
      !CudaOk(cudaMemcpy(device_chunk_keys.get(), chunk_keys.data(),
                         device_chunk_keys.bytes(), cudaMemcpyHostToDevice),
              "copy prefill chunk keys") ||
      !CudaOk(cudaMemcpy(device_chunk_values.get(), chunk_values.data(),
                         device_chunk_values.bytes(), cudaMemcpyHostToDevice),
              "copy prefill chunk values")) {
    return;
  }
  const auto attention = gem16::internal::LaunchCausalAttentionPrefill(
      device_queries.get(), device_chunk_keys.get(), device_chunk_values.get(),
      device_cache_keys.get(), device_cache_values.get(), device_scores.get(),
      device_output.get(), start_position, tokens, query_heads, kv_heads,
      head_dimension, capacity, true, nullptr);
  CUDA_TEST_CHECK(attention.ok());
  const auto fused_attention =
      gem16::internal::LaunchFusedCausalAttentionPrefill(
          device_queries.get(), device_chunk_keys.get(),
          device_chunk_values.get(), device_cache_keys.get(),
          device_cache_values.get(), device_fused_scores.get(),
          device_fused_output.get(), start_position, tokens, query_heads,
          kv_heads, head_dimension, capacity, true, nullptr);
  CUDA_TEST_CHECK(fused_attention.ok());
  const auto append = gem16::internal::LaunchAppendKvBatch(
      device_chunk_keys.get(), device_chunk_values.get(), device_cache_keys.get(),
      device_cache_values.get(), start_position, tokens,
      kv_heads * head_dimension, capacity, nullptr);
  CUDA_TEST_CHECK(append.ok());
  if (!attention.ok() || !fused_attention.ok() || !append.ok() ||
      !CudaOk(cudaDeviceSynchronize(), "prefill ring synchronize")) {
    return;
  }
  std::array<float, queries.size()> output{};
  std::array<float, queries.size()> fused_output{};
  std::array<float, cache_keys.size()> appended_keys{};
  if (!CudaOk(cudaMemcpy(output.data(), device_output.get(), device_output.bytes(),
                         cudaMemcpyDeviceToHost), "copy prefill output") ||
      !CudaOk(cudaMemcpy(fused_output.data(), device_fused_output.get(),
                         device_fused_output.bytes(), cudaMemcpyDeviceToHost),
              "copy fused prefill output") ||
      !CudaOk(cudaMemcpy(appended_keys.data(), device_cache_keys.get(),
                         device_cache_keys.bytes(), cudaMemcpyDeviceToHost),
              "copy appended prefill keys")) {
    return;
  }
  for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
    CUDA_TEST_CHECK(std::fabs(output[dimension] -
                              first_expected.value()[dimension]) < 2.0e-5F);
    CUDA_TEST_CHECK(fused_output[dimension] == output[dimension]);
    CUDA_TEST_CHECK(std::fabs(output[head_dimension + dimension] -
                              second_expected.value()[dimension]) < 2.0e-5F);
    CUDA_TEST_CHECK(fused_output[head_dimension + dimension] ==
                    output[head_dimension + dimension]);
  }
  CUDA_TEST_CHECK(appended_keys[2] == chunk_keys[0]);
  CUDA_TEST_CHECK(appended_keys[3] == chunk_keys[1]);
  CUDA_TEST_CHECK(appended_keys[4] == chunk_keys[2]);
  CUDA_TEST_CHECK(appended_keys[5] == chunk_keys[3]);
}

void TestOnlineFp8DecodeAttentionShape(
    std::uint64_t kv_heads, std::uint64_t head_dimension,
    std::uint64_t capacity, std::uint64_t position, bool sliding,
    const char* label) {
  constexpr std::uint64_t query_heads = 16;
  constexpr std::array<std::uint16_t, 1> key_scale = {0x3F00U};
  constexpr std::array<std::uint16_t, 1> value_scale = {0x3F40U};
  const std::uint64_t tokens =
      sliding ? std::min(position + 1U, capacity) : position + 1U;
  const std::uint64_t first_slot =
      sliding && position + 1U > capacity ? (position + 1U) % capacity : 0U;
  std::vector<float> query(query_heads * head_dimension);
  for (std::size_t index = 0; index < query.size(); ++index) {
    query[index] =
        static_cast<float>(static_cast<int>((index * 11U) % 31U) - 15) *
        0.015625F;
  }
  const auto make_cache = [](std::size_t count, std::size_t multiplier) {
    std::vector<std::uint8_t> values(count);
    for (std::size_t index = 0; index < count; ++index) {
      const float value =
          static_cast<float>(
              static_cast<int>((index * multiplier + index / 37U) % 29U) -
              14) *
          0.125F;
      const auto encoded = gem16::fp8::EncodeE4M3Fn(value);
      CUDA_TEST_CHECK(encoded.ok());
      if (!encoded.ok()) return std::vector<std::uint8_t>{};
      values[index] = encoded.value();
    }
    return values;
  };
  const auto keys =
      make_cache(capacity * kv_heads * head_dimension, 7U);
  const auto values =
      make_cache(capacity * kv_heads * head_dimension, 13U);
  if (keys.empty() || values.empty()) return;

  const std::size_t workspace_elements = static_cast<std::size_t>(
      gem16::internal::DecodeAttentionWorkspaceElements(capacity));
  DeviceBuffer<float> device_query(query.size());
  DeviceBuffer<std::uint8_t> device_keys(keys.size());
  DeviceBuffer<std::uint8_t> device_values(values.size());
  DeviceBuffer<std::uint16_t> device_key_scale(key_scale.size());
  DeviceBuffer<std::uint16_t> device_value_scale(value_scale.size());
  DeviceBuffer<float> device_reference_scores(query_heads * tokens);
  DeviceBuffer<float> device_workspace(workspace_elements);
  DeviceBuffer<float> device_reference_output(query.size());
  DeviceBuffer<float> device_online_output(query.size());
  DeviceBuffer<gem16::internal::DecodeControl> device_control(1);
  if (device_query.get() == nullptr || device_keys.get() == nullptr ||
      device_values.get() == nullptr || device_key_scale.get() == nullptr ||
      device_value_scale.get() == nullptr ||
      device_reference_scores.get() == nullptr ||
      device_workspace.get() == nullptr ||
      device_reference_output.get() == nullptr ||
      device_online_output.get() == nullptr ||
      device_control.get() == nullptr) {
    return;
  }
  const gem16::internal::DecodeControl control = {
      .token = 0, .suppressed_token_count = 0, .position = position};
  if (!CudaOk(cudaMemcpy(device_query.get(), query.data(),
                         device_query.bytes(), cudaMemcpyHostToDevice),
              "copy online-decode query") ||
      !CudaOk(cudaMemcpy(device_keys.get(), keys.data(), device_keys.bytes(),
                         cudaMemcpyHostToDevice),
              "copy online-decode keys") ||
      !CudaOk(cudaMemcpy(device_values.get(), values.data(),
                         device_values.bytes(), cudaMemcpyHostToDevice),
              "copy online-decode values") ||
      !CudaOk(cudaMemcpy(device_key_scale.get(), key_scale.data(),
                         device_key_scale.bytes(), cudaMemcpyHostToDevice),
              "copy online-decode key scale") ||
      !CudaOk(cudaMemcpy(device_value_scale.get(), value_scale.data(),
                         device_value_scale.bytes(), cudaMemcpyHostToDevice),
              "copy online-decode value scale") ||
      !CudaOk(cudaMemcpy(device_control.get(), &control, sizeof(control),
                         cudaMemcpyHostToDevice),
              "copy online-decode control")) {
    return;
  }

  const auto reference =
      gem16::internal::LaunchLocalAttentionDecodeFp8(
          device_query.get(), device_keys.get(), device_values.get(),
          device_key_scale.get(), device_value_scale.get(),
          device_reference_scores.get(), device_reference_output.get(),
          query_heads, kv_heads, head_dimension, tokens, nullptr, capacity,
          first_slot);
  const auto online =
      gem16::internal::LaunchOnlineAttentionDecodeFp8Sm120(
          device_query.get(), device_keys.get(), device_values.get(),
          device_key_scale.get(), device_value_scale.get(),
          device_workspace.get(), device_online_output.get(),
          device_control.get(), query_heads, kv_heads, head_dimension,
          capacity, sliding, nullptr);
  CUDA_TEST_CHECK(reference.ok());
  CUDA_TEST_CHECK(online.ok());
  if (!reference.ok() || !online.ok() ||
      !CudaOk(cudaDeviceSynchronize(), "online FP8 decode synchronize")) {
    return;
  }
  std::vector<float> reference_output(query.size());
  std::vector<float> online_output(query.size());
  if (!CudaOk(cudaMemcpy(reference_output.data(),
                         device_reference_output.get(),
                         device_reference_output.bytes(),
                         cudaMemcpyDeviceToHost),
              "copy online-decode reference") ||
      !CudaOk(cudaMemcpy(online_output.data(), device_online_output.get(),
                         device_online_output.bytes(),
                         cudaMemcpyDeviceToHost),
              "copy online-decode output")) {
    return;
  }
  CheckAttentionMetrics(reference_output, online_output, label, 2.0e-3F,
                        3.0e-4, 0.99999);
  std::vector<float> repeated_output(query.size());
  for (int repetition = 0; repetition < 4; ++repetition) {
    const auto repeated =
        gem16::internal::LaunchOnlineAttentionDecodeFp8Sm120(
            device_query.get(), device_keys.get(), device_values.get(),
            device_key_scale.get(), device_value_scale.get(),
            device_workspace.get(), device_online_output.get(),
            device_control.get(), query_heads, kv_heads, head_dimension,
            capacity, sliding, nullptr);
    CUDA_TEST_CHECK(repeated.ok());
    if (!repeated.ok() ||
        !CudaOk(cudaDeviceSynchronize(),
                "repeated online FP8 decode synchronize") ||
        !CudaOk(cudaMemcpy(repeated_output.data(), device_online_output.get(),
                           device_online_output.bytes(),
                           cudaMemcpyDeviceToHost),
                "copy repeated online-decode output")) {
      return;
    }
    CUDA_TEST_CHECK(std::equal(
        online_output.begin(), online_output.end(), repeated_output.begin(),
        [](float left, float right) {
          return std::bit_cast<std::uint32_t>(left) ==
                 std::bit_cast<std::uint32_t>(right);
        }));
  }
}

void TestOnlineFp8DecodeAttention() {
  TestOnlineFp8DecodeAttentionShape(8, 256, 1024, 1100, true,
                                    "online local FP8 decode");
  TestOnlineFp8DecodeAttentionShape(1, 512, 1536, 768, false,
                                    "online global FP8 decode");
  TestOnlineFp8DecodeAttentionShape(1, 512, 16384, 16383, false,
                                    "long online global FP8 decode");
}

void TestVectorizedFp8CausalPrefill() {
  constexpr std::uint64_t tokens = 3;
  constexpr std::uint64_t query_heads = 2;
  constexpr std::uint64_t kv_heads = 1;
  constexpr std::uint64_t head_dimension = 32;
  constexpr std::uint64_t capacity = 4;
  constexpr std::uint64_t start_position = 1;
  constexpr std::uint64_t score_stride = 4;
  constexpr std::array<std::uint16_t, 1> scale = {0x3F00U};  // 0.5

  std::vector<float> queries(tokens * query_heads * head_dimension);
  for (std::size_t index = 0; index < queries.size(); ++index) {
    queries[index] =
        static_cast<float>(static_cast<int>((index * 11U) % 23U) - 11) *
        0.03125F;
  }
  const auto make_fp8 = [](std::size_t count, std::size_t multiplier) {
    std::vector<std::uint8_t> values(count);
    for (std::size_t index = 0; index < count; ++index) {
      const float value =
          static_cast<float>(static_cast<int>((index * multiplier) % 15U) - 7) /
          4.0F;
      const auto encoded = gem16::fp8::EncodeE4M3Fn(value);
      CUDA_TEST_CHECK(encoded.ok());
      if (!encoded.ok()) return std::vector<std::uint8_t>{};
      values[index] = encoded.value();
    }
    return values;
  };
  const auto chunk_keys =
      make_fp8(tokens * kv_heads * head_dimension, 5U);
  const auto chunk_values =
      make_fp8(tokens * kv_heads * head_dimension, 7U);
  const auto cache_keys =
      make_fp8(capacity * kv_heads * head_dimension, 3U);
  const auto cache_values =
      make_fp8(capacity * kv_heads * head_dimension, 13U);
  if (chunk_keys.empty() || chunk_values.empty() || cache_keys.empty() ||
      cache_values.empty()) {
    return;
  }

  DeviceBuffer<float> device_queries(queries.size());
  DeviceBuffer<std::uint8_t> device_chunk_keys(chunk_keys.size());
  DeviceBuffer<std::uint8_t> device_chunk_values(chunk_values.size());
  DeviceBuffer<std::uint8_t> device_cache_keys(cache_keys.size());
  DeviceBuffer<std::uint8_t> device_cache_values(cache_values.size());
  DeviceBuffer<std::uint16_t> device_scale(scale.size());
  DeviceBuffer<float> device_reference_scores(tokens * query_heads * score_stride);
  DeviceBuffer<float> device_fused_scores(tokens * query_heads * score_stride);
  DeviceBuffer<float> device_reference_output(queries.size());
  DeviceBuffer<float> device_fused_output(queries.size());
  if (device_queries.get() == nullptr || device_chunk_keys.get() == nullptr ||
      device_chunk_values.get() == nullptr ||
      device_cache_keys.get() == nullptr ||
      device_cache_values.get() == nullptr || device_scale.get() == nullptr ||
      device_reference_scores.get() == nullptr ||
      device_fused_scores.get() == nullptr ||
      device_reference_output.get() == nullptr ||
      device_fused_output.get() == nullptr) {
    return;
  }
  if (!CudaOk(cudaMemcpy(device_queries.get(), queries.data(),
                         device_queries.bytes(), cudaMemcpyHostToDevice),
              "copy vectorized-prefill queries") ||
      !CudaOk(cudaMemcpy(device_chunk_keys.get(), chunk_keys.data(),
                         device_chunk_keys.bytes(), cudaMemcpyHostToDevice),
              "copy vectorized-prefill chunk keys") ||
      !CudaOk(cudaMemcpy(device_chunk_values.get(), chunk_values.data(),
                         device_chunk_values.bytes(), cudaMemcpyHostToDevice),
              "copy vectorized-prefill chunk values") ||
      !CudaOk(cudaMemcpy(device_cache_keys.get(), cache_keys.data(),
                         device_cache_keys.bytes(), cudaMemcpyHostToDevice),
              "copy vectorized-prefill cache keys") ||
      !CudaOk(cudaMemcpy(device_cache_values.get(), cache_values.data(),
                         device_cache_values.bytes(), cudaMemcpyHostToDevice),
              "copy vectorized-prefill cache values") ||
      !CudaOk(cudaMemcpy(device_scale.get(), scale.data(), device_scale.bytes(),
                         cudaMemcpyHostToDevice),
              "copy vectorized-prefill scale")) {
    return;
  }

  const auto reference = gem16::internal::LaunchCausalAttentionPrefillFp8(
      device_queries.get(), device_chunk_keys.get(),
      device_chunk_values.get(), device_cache_keys.get(),
      device_cache_values.get(), device_scale.get(), device_scale.get(),
      device_reference_scores.get(), device_reference_output.get(),
      start_position, tokens, query_heads, kv_heads, head_dimension, capacity,
      false, nullptr);
  const auto fused = gem16::internal::LaunchFusedCausalAttentionPrefillFp8(
      device_queries.get(), device_chunk_keys.get(),
      device_chunk_values.get(), device_cache_keys.get(),
      device_cache_values.get(), device_scale.get(), device_scale.get(),
      device_fused_scores.get(), device_fused_output.get(), start_position,
      tokens, query_heads, kv_heads, head_dimension, capacity, false, nullptr);
  CUDA_TEST_CHECK(reference.ok());
  CUDA_TEST_CHECK(fused.ok());
  if (!reference.ok() || !fused.ok() ||
      !CudaOk(cudaDeviceSynchronize(), "vectorized FP8 prefill synchronize")) {
    return;
  }
  std::vector<float> reference_output(queries.size());
  std::vector<float> fused_output(queries.size());
  if (!CudaOk(cudaMemcpy(reference_output.data(), device_reference_output.get(),
                         device_reference_output.bytes(), cudaMemcpyDeviceToHost),
              "copy reference vectorized-prefill output") ||
      !CudaOk(cudaMemcpy(fused_output.data(), device_fused_output.get(),
                         device_fused_output.bytes(), cudaMemcpyDeviceToHost),
              "copy fused vectorized-prefill output")) {
    return;
  }
  CUDA_TEST_CHECK(reference_output == fused_output);
}

void TestOnlineLocalFp8CausalPrefill() {
  constexpr std::uint64_t tokens = 64;
  constexpr std::uint64_t query_heads = 16;
  constexpr std::uint64_t kv_heads = 8;
  constexpr std::uint64_t head_dimension = 256;
  constexpr std::uint64_t capacity = 1024;
  constexpr std::array<std::uint16_t, 1> key_scale = {0x3E80U};
  constexpr std::array<std::uint16_t, 1> value_scale = {0x3F00U};

  std::vector<float> queries(tokens * query_heads * head_dimension);
  for (std::size_t index = 0; index < queries.size(); ++index) {
    const float value =
        static_cast<float>(static_cast<int>((index * 17U) % 31U) - 15) /
        64.0F;
    queries[index] = RoundBf16Reference(value);
  }
  const auto make_fp8 = [](std::size_t count, std::size_t multiplier,
                           std::size_t offset) {
    std::vector<std::uint8_t> values(count);
    for (std::size_t index = 0; index < count; ++index) {
      const float value =
          static_cast<float>(
              static_cast<int>(((index + offset) * multiplier) % 31U) - 15) /
          8.0F;
      const auto encoded = gem16::fp8::EncodeE4M3Fn(value);
      CUDA_TEST_CHECK(encoded.ok());
      if (!encoded.ok()) return std::vector<std::uint8_t>{};
      values[index] = encoded.value();
    }
    return values;
  };
  const auto chunk_keys =
      make_fp8(tokens * kv_heads * head_dimension, 5U, 1U);
  const auto chunk_values =
      make_fp8(tokens * kv_heads * head_dimension, 7U, 3U);
  const auto cache_keys =
      make_fp8(capacity * kv_heads * head_dimension, 11U, 5U);
  const auto cache_values =
      make_fp8(capacity * kv_heads * head_dimension, 13U, 7U);
  if (chunk_keys.empty() || chunk_values.empty() || cache_keys.empty() ||
      cache_values.empty()) {
    return;
  }

  DeviceBuffer<float> device_queries(queries.size());
  DeviceBuffer<std::uint8_t> device_chunk_keys(chunk_keys.size());
  DeviceBuffer<std::uint8_t> device_chunk_values(chunk_values.size());
  DeviceBuffer<std::uint8_t> device_cache_keys(cache_keys.size());
  DeviceBuffer<std::uint8_t> device_cache_values(cache_values.size());
  DeviceBuffer<std::uint16_t> device_key_scale(key_scale.size());
  DeviceBuffer<std::uint16_t> device_value_scale(value_scale.size());
  DeviceBuffer<float> device_scores(tokens * query_heads * capacity);
  DeviceBuffer<float> device_reference_output(queries.size());
  DeviceBuffer<float> device_online_output(queries.size());
  if (device_queries.get() == nullptr ||
      device_chunk_keys.get() == nullptr ||
      device_chunk_values.get() == nullptr ||
      device_cache_keys.get() == nullptr ||
      device_cache_values.get() == nullptr ||
      device_key_scale.get() == nullptr ||
      device_value_scale.get() == nullptr || device_scores.get() == nullptr ||
      device_reference_output.get() == nullptr ||
      device_online_output.get() == nullptr) {
    return;
  }
  if (!CudaOk(cudaMemcpy(device_queries.get(), queries.data(),
                         device_queries.bytes(), cudaMemcpyHostToDevice),
              "copy online-prefill queries") ||
      !CudaOk(cudaMemcpy(device_chunk_keys.get(), chunk_keys.data(),
                         device_chunk_keys.bytes(), cudaMemcpyHostToDevice),
              "copy online-prefill chunk keys") ||
      !CudaOk(cudaMemcpy(device_chunk_values.get(), chunk_values.data(),
                         device_chunk_values.bytes(), cudaMemcpyHostToDevice),
              "copy online-prefill chunk values") ||
      !CudaOk(cudaMemcpy(device_cache_keys.get(), cache_keys.data(),
                         device_cache_keys.bytes(), cudaMemcpyHostToDevice),
              "copy online-prefill cache keys") ||
      !CudaOk(cudaMemcpy(device_cache_values.get(), cache_values.data(),
                         device_cache_values.bytes(), cudaMemcpyHostToDevice),
              "copy online-prefill cache values") ||
      !CudaOk(cudaMemcpy(device_key_scale.get(), key_scale.data(),
                         device_key_scale.bytes(), cudaMemcpyHostToDevice),
              "copy online-prefill key scale") ||
      !CudaOk(cudaMemcpy(device_value_scale.get(), value_scale.data(),
                         device_value_scale.bytes(), cudaMemcpyHostToDevice),
              "copy online-prefill value scale")) {
    return;
  }

  const auto run_case = [&](std::uint64_t start_position,
                            const char* label) {
    const auto reference =
        gem16::internal::LaunchFusedCausalAttentionPrefillFp8(
            device_queries.get(), device_chunk_keys.get(),
            device_chunk_values.get(), device_cache_keys.get(),
            device_cache_values.get(), device_key_scale.get(),
            device_value_scale.get(), device_scores.get(),
            device_reference_output.get(), start_position, tokens,
            query_heads, kv_heads, head_dimension, capacity, true, nullptr);
    const auto online =
        gem16::internal::LaunchOnlineCausalAttentionPrefillFp8LocalSm120(
            device_queries.get(), device_chunk_keys.get(),
            device_chunk_values.get(), device_cache_keys.get(),
            device_cache_values.get(), device_key_scale.get(),
            device_value_scale.get(), device_online_output.get(),
            start_position, tokens, query_heads, kv_heads, head_dimension,
            capacity, nullptr);
    CUDA_TEST_CHECK(reference.ok());
    CUDA_TEST_CHECK(online.ok());
    if (!reference.ok() || !online.ok() ||
        !CudaOk(cudaDeviceSynchronize(), label)) {
      return;
    }

    std::vector<float> reference_output(queries.size());
    std::vector<float> online_output(queries.size());
    if (!CudaOk(cudaMemcpy(reference_output.data(),
                           device_reference_output.get(),
                           device_reference_output.bytes(),
                           cudaMemcpyDeviceToHost),
                "copy online-prefill reference output") ||
        !CudaOk(cudaMemcpy(online_output.data(), device_online_output.get(),
                           device_online_output.bytes(),
                           cudaMemcpyDeviceToHost),
                "copy online-prefill tensor-core output")) {
      return;
    }

    const std::string metric_label =
        std::string("online local FP8 prefill ") + label;
    CheckAttentionMetrics(reference_output, online_output,
                          metric_label.c_str(), 1.5e-3F, 3.0e-4, 0.99998);
  };

  run_case(0U, "initial causal window");
  run_case(1100U, "wrapped 1024-token window");
}

void TestOnlineGlobalFp8CausalPrefill() {
  constexpr std::uint64_t tokens = 32;
  constexpr std::uint64_t query_heads = 16;
  constexpr std::uint64_t kv_heads = 1;
  constexpr std::uint64_t head_dimension = 512;
  constexpr std::uint64_t capacity = 128;
  constexpr std::uint64_t start_position = 37;
  constexpr std::uint64_t score_stride = start_position + tokens;
  constexpr std::array<std::uint16_t, 1> key_scale = {0x3E80U};
  constexpr std::array<std::uint16_t, 1> value_scale = {0x3F00U};

  std::vector<float> queries(tokens * query_heads * head_dimension);
  for (std::size_t index = 0; index < queries.size(); ++index) {
    const float value =
        static_cast<float>(static_cast<int>((index * 19U) % 37U) - 18) /
        64.0F;
    queries[index] = RoundBf16Reference(value);
  }
  const auto make_fp8 = [](std::size_t count, std::size_t multiplier,
                           std::size_t offset) {
    std::vector<std::uint8_t> values(count);
    for (std::size_t index = 0; index < count; ++index) {
      const float value =
          static_cast<float>(
              static_cast<int>(((index + offset) * multiplier) % 31U) - 15) /
          8.0F;
      const auto encoded = gem16::fp8::EncodeE4M3Fn(value);
      CUDA_TEST_CHECK(encoded.ok());
      if (!encoded.ok()) return std::vector<std::uint8_t>{};
      values[index] = encoded.value();
    }
    return values;
  };
  const auto chunk_keys = make_fp8(tokens * head_dimension, 5U, 1U);
  const auto chunk_values = make_fp8(tokens * head_dimension, 7U, 3U);
  const auto cache_keys = make_fp8(capacity * head_dimension, 11U, 5U);
  const auto cache_values = make_fp8(capacity * head_dimension, 13U, 7U);
  if (chunk_keys.empty() || chunk_values.empty() || cache_keys.empty() ||
      cache_values.empty()) {
    return;
  }

  DeviceBuffer<float> device_queries(queries.size());
  DeviceBuffer<std::uint8_t> device_chunk_keys(chunk_keys.size());
  DeviceBuffer<std::uint8_t> device_chunk_values(chunk_values.size());
  DeviceBuffer<std::uint8_t> device_cache_keys(cache_keys.size());
  DeviceBuffer<std::uint8_t> device_cache_values(cache_values.size());
  DeviceBuffer<std::uint16_t> device_key_scale(key_scale.size());
  DeviceBuffer<std::uint16_t> device_value_scale(value_scale.size());
  DeviceBuffer<float> device_scores(tokens * query_heads * score_stride);
  DeviceBuffer<float> device_reference_output(queries.size());
  DeviceBuffer<float> device_online_output(queries.size());
  if (device_queries.get() == nullptr ||
      device_chunk_keys.get() == nullptr ||
      device_chunk_values.get() == nullptr ||
      device_cache_keys.get() == nullptr ||
      device_cache_values.get() == nullptr ||
      device_key_scale.get() == nullptr ||
      device_value_scale.get() == nullptr || device_scores.get() == nullptr ||
      device_reference_output.get() == nullptr ||
      device_online_output.get() == nullptr) {
    return;
  }
  if (!CudaOk(cudaMemcpy(device_queries.get(), queries.data(),
                         device_queries.bytes(), cudaMemcpyHostToDevice),
              "copy global online-prefill queries") ||
      !CudaOk(cudaMemcpy(device_chunk_keys.get(), chunk_keys.data(),
                         device_chunk_keys.bytes(), cudaMemcpyHostToDevice),
              "copy global online-prefill chunk keys") ||
      !CudaOk(cudaMemcpy(device_chunk_values.get(), chunk_values.data(),
                         device_chunk_values.bytes(), cudaMemcpyHostToDevice),
              "copy global online-prefill chunk values") ||
      !CudaOk(cudaMemcpy(device_cache_keys.get(), cache_keys.data(),
                         device_cache_keys.bytes(), cudaMemcpyHostToDevice),
              "copy global online-prefill cache keys") ||
      !CudaOk(cudaMemcpy(device_cache_values.get(), cache_values.data(),
                         device_cache_values.bytes(), cudaMemcpyHostToDevice),
              "copy global online-prefill cache values") ||
      !CudaOk(cudaMemcpy(device_key_scale.get(), key_scale.data(),
                         device_key_scale.bytes(), cudaMemcpyHostToDevice),
              "copy global online-prefill key scale") ||
      !CudaOk(cudaMemcpy(device_value_scale.get(), value_scale.data(),
                         device_value_scale.bytes(), cudaMemcpyHostToDevice),
              "copy global online-prefill value scale")) {
    return;
  }

  const auto reference =
      gem16::internal::LaunchFusedCausalAttentionPrefillFp8(
          device_queries.get(), device_chunk_keys.get(),
          device_chunk_values.get(), device_cache_keys.get(),
          device_cache_values.get(), device_key_scale.get(),
          device_value_scale.get(), device_scores.get(),
          device_reference_output.get(), start_position, tokens, query_heads,
          kv_heads, head_dimension, capacity, false, nullptr);
  const auto online =
      gem16::internal::LaunchOnlineCausalAttentionPrefillFp8GlobalSm120(
          device_queries.get(), device_chunk_keys.get(),
          device_chunk_values.get(), device_cache_keys.get(),
          device_cache_values.get(), device_key_scale.get(),
          device_value_scale.get(), device_online_output.get(),
          start_position, tokens, query_heads, kv_heads, head_dimension,
          capacity, nullptr);
  CUDA_TEST_CHECK(reference.ok());
  CUDA_TEST_CHECK(online.ok());
  if (!reference.ok() || !online.ok() ||
      !CudaOk(cudaDeviceSynchronize(), "global online-prefill synchronize")) {
    return;
  }

  std::vector<float> reference_output(queries.size());
  std::vector<float> online_output(queries.size());
  if (!CudaOk(cudaMemcpy(reference_output.data(),
                         device_reference_output.get(),
                         device_reference_output.bytes(),
                         cudaMemcpyDeviceToHost),
              "copy global online-prefill reference output") ||
      !CudaOk(cudaMemcpy(online_output.data(), device_online_output.get(),
                         device_online_output.bytes(),
                         cudaMemcpyDeviceToHost),
              "copy global online-prefill tensor-core output")) {
    return;
  }
  CheckAttentionMetrics(reference_output, online_output,
                        "online global FP8 prefill", 8.0e-4F, 2.0e-4,
                        0.99999);
}

void TestGpuSampling() {
  constexpr std::uint32_t kVocabulary = 8U;
  const std::array<float, kVocabulary> logits =
      {5.0F, 4.0F, 3.0F, 2.0F, 1.0F, -1.0F, -2.0F, -3.0F};
  const std::array<std::uint32_t, 2> history = {0U, 5U};
  const std::array<std::uint32_t, 1> suppressed = {1U};
  DeviceBuffer<float> device_logits(kVocabulary);
  DeviceBuffer<float> adjusted(kVocabulary);
  DeviceBuffer<double> cumulative(kVocabulary);
  DeviceBuffer<std::uint32_t> token_ids(kVocabulary);
  DeviceBuffer<std::uint32_t> sorted_token_ids(kVocabulary);
  DeviceBuffer<std::uint32_t> repetition_mask((kVocabulary + 31U) / 32U);
  DeviceBuffer<std::uint32_t> device_history(history.size());
  DeviceBuffer<std::uint32_t> device_suppressed(suppressed.size());
  DeviceBuffer<std::uint32_t> selected(1U);
  auto workspace_bytes =
      gem16::internal::SamplingWorkspaceBytes(kVocabulary, nullptr);
  CUDA_TEST_CHECK(workspace_bytes.ok());
  if (!workspace_bytes.ok()) return;
  DeviceBuffer<std::uint8_t> workspace(workspace_bytes.value());
  if (!CudaOk(cudaMemset(repetition_mask.get(), 0, repetition_mask.bytes()),
              "clear sampling repetition mask") ||
      !CudaOk(cudaMemcpy(device_history.get(), history.data(), history.size() *
                         sizeof(std::uint32_t), cudaMemcpyHostToDevice),
              "copy sampling history") ||
      !CudaOk(cudaMemcpy(device_suppressed.get(), suppressed.data(),
                         suppressed.size() * sizeof(std::uint32_t),
                         cudaMemcpyHostToDevice),
              "copy sampling suppression")) {
    return;
  }
  auto status = gem16::internal::LaunchMarkRepetitionTokens(
      device_history.get(), history.size(), repetition_mask.get(), nullptr);
  CUDA_TEST_CHECK(status.ok());
  gem16::SamplingOptions options;
  options.enabled = true;
  options.temperature = 1.0F;
  options.top_k = 1U;
  options.repetition_penalty = 2.0F;
  if (!CudaOk(cudaMemcpy(device_logits.get(), logits.data(),
                         device_logits.bytes(), cudaMemcpyHostToDevice),
              "copy synthetic sampling logits")) {
    return;
  }
  status = gem16::internal::LaunchSampleToken(
      device_logits.get(), adjusted.get(), cumulative.get(), token_ids.get(),
      sorted_token_ids.get(), repetition_mask.get(), device_suppressed.get(),
      suppressed.size(), kVocabulary, options, 0U, nullptr, selected.get(),
      workspace.get(), workspace.bytes(), nullptr);
  CUDA_TEST_CHECK(status.ok());
  std::uint32_t host_selected = 0U;
  if (!CudaOk(cudaMemcpy(&host_selected, selected.get(), sizeof(host_selected),
                         cudaMemcpyDeviceToHost),
              "copy synthetic sampled token")) {
    return;
  }
  // Token 0 is reduced by repetition penalty and token 1 is suppressed, so
  // exact top-k=1 must select token 2.
  CUDA_TEST_CHECK(host_selected == 2U);

  options.top_k = 4U;
  options.top_p = 0.9F;
  options.min_p = 0.05F;
  options.seed = 42U;
  if (!CudaOk(cudaMemcpy(device_logits.get(), logits.data(),
                         device_logits.bytes(), cudaMemcpyHostToDevice),
              "restore synthetic sampling logits")) {
    return;
  }
  status = gem16::internal::LaunchSampleToken(
      device_logits.get(), adjusted.get(), cumulative.get(), token_ids.get(),
      sorted_token_ids.get(), repetition_mask.get(), device_suppressed.get(),
      suppressed.size(), kVocabulary, options, 7U, nullptr, selected.get(),
      workspace.get(), workspace.bytes(), nullptr);
  CUDA_TEST_CHECK(status.ok());
  if (!CudaOk(cudaMemcpy(&host_selected, selected.get(), sizeof(host_selected),
                         cudaMemcpyDeviceToHost),
              "copy seeded synthetic sampled token")) {
    return;
  }
  CUDA_TEST_CHECK(host_selected == 2U);

  options.top_k = 0U;
  options.top_p = 1.0F;
  options.min_p = 0.0F;
  options.seed = 0U;
  if (!CudaOk(cudaMemcpy(device_logits.get(), logits.data(),
                         device_logits.bytes(), cudaMemcpyHostToDevice),
              "restore full-vocabulary sampling logits")) {
    return;
  }
  status = gem16::internal::LaunchSampleToken(
      device_logits.get(), adjusted.get(), cumulative.get(), token_ids.get(),
      sorted_token_ids.get(), repetition_mask.get(), device_suppressed.get(),
      suppressed.size(), kVocabulary, options, 0U, nullptr, selected.get(),
      workspace.get(), workspace.bytes(), nullptr);
  CUDA_TEST_CHECK(status.ok());
  if (!CudaOk(cudaMemcpy(&host_selected, selected.get(), sizeof(host_selected),
                         cudaMemcpyDeviceToHost),
              "copy full-vocabulary sampled token")) {
    return;
  }
  CUDA_TEST_CHECK(host_selected == 0U);

  options.top_k = 4U;
  options.top_p = 0.9F;
  options.min_p = 0.05F;
  options.seed = 42U;
  DeviceBuffer<gem16::internal::DecodeControl> control(1U);
  gem16::internal::DecodeControl host_control{};
  host_control.token = 4U;
  host_control.sampling_step = 8U;
  if (!CudaOk(cudaMemcpy(control.get(), &host_control, sizeof(host_control),
                         cudaMemcpyHostToDevice),
              "copy controlled sampling state")) {
    return;
  }
  status = gem16::internal::LaunchMarkControlledRepetitionToken(
      control.get(), repetition_mask.get(), nullptr);
  CUDA_TEST_CHECK(status.ok());
  if (!CudaOk(cudaMemcpy(device_logits.get(), logits.data(),
                         device_logits.bytes(), cudaMemcpyHostToDevice),
              "restore controlled sampling logits")) {
    return;
  }
  status = gem16::internal::LaunchSampleToken(
      device_logits.get(), adjusted.get(), cumulative.get(), token_ids.get(),
      sorted_token_ids.get(), repetition_mask.get(), device_suppressed.get(),
      suppressed.size(), kVocabulary, options, 7U, control.get(),
      selected.get(), workspace.get(), workspace.bytes(), nullptr);
  CUDA_TEST_CHECK(status.ok());
  if (!CudaOk(cudaMemcpy(&host_selected, selected.get(), sizeof(host_selected),
                         cudaMemcpyDeviceToHost),
              "copy controlled sampled token")) {
    return;
  }
  // Device control overrides the captured scalar step. Marking token 4 does
  // not affect the retained top-p set, and pinned step 8 selects token 0.
  CUDA_TEST_CHECK(host_selected == 0U);

  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  if (!CudaOk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "create sampling graph stream")) {
    return;
  }
  host_control.sampling_step = 7U;
  bool graph_ok =
      CudaOk(cudaMemcpy(control.get(), &host_control, sizeof(host_control),
                        cudaMemcpyHostToDevice),
             "reset sampling graph control") &&
      CudaOk(cudaMemcpy(device_logits.get(), logits.data(),
                        device_logits.bytes(), cudaMemcpyHostToDevice),
             "restore sampling graph logits") &&
      CudaOk(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
             "begin sampling graph capture");
  if (graph_ok) {
    status = gem16::internal::LaunchSampleToken(
        device_logits.get(), adjusted.get(), cumulative.get(), token_ids.get(),
        sorted_token_ids.get(), repetition_mask.get(),
        device_suppressed.get(), suppressed.size(), kVocabulary, options, 0U,
        control.get(), selected.get(), workspace.get(), workspace.bytes(),
        stream);
    CUDA_TEST_CHECK(status.ok());
    graph_ok = status.ok() &&
               CudaOk(cudaStreamEndCapture(stream, &graph),
                      "end sampling graph capture") &&
               CudaOk(cudaGraphInstantiate(&executable, graph, nullptr,
                                           nullptr, 0U),
                      "instantiate sampling graph") &&
               CudaOk(cudaGraphLaunch(executable, stream),
                      "launch sampling graph") &&
               CudaOk(cudaStreamSynchronize(stream),
                      "synchronize sampling graph");
  }
  if (graph_ok) {
    CUDA_TEST_CHECK(CudaOk(
        cudaMemcpy(&host_selected, selected.get(), sizeof(host_selected),
                   cudaMemcpyDeviceToHost),
        "copy sampling graph token"));
    CUDA_TEST_CHECK(host_selected == 2U);
  }
  if (executable != nullptr) (void)cudaGraphExecDestroy(executable);
  if (graph != nullptr) (void)cudaGraphDestroy(graph);
  (void)cudaStreamDestroy(stream);
}

void TestMtpDeviceControlTransitions() {
  DeviceBuffer<std::uint32_t> drafts(
      gem16::internal::kMaximumMtpDraftTokens);
  DeviceBuffer<std::uint32_t> verified(
      gem16::internal::kMaximumMtpVerifyTokens);
  DeviceBuffer<std::uint32_t> stop_tokens(1U);
  DeviceBuffer<gem16::internal::MtpGroupTransaction> transaction(1U);
  if (drafts.get() == nullptr || verified.get() == nullptr ||
      stop_tokens.get() == nullptr || transaction.get() == nullptr) {
    return;
  }

  const std::array<std::uint32_t,
                   gem16::internal::kMaximumMtpDraftTokens>
      host_drafts{10U, 20U, 0U, 0U};
  const std::uint32_t stop = 99U;
  if (!CudaOk(cudaMemcpy(drafts.get(), host_drafts.data(), drafts.bytes(),
                         cudaMemcpyHostToDevice),
              "copy MTP transition drafts") ||
      !CudaOk(cudaMemcpy(stop_tokens.get(), &stop, sizeof(stop),
                         cudaMemcpyHostToDevice),
              "copy MTP transition stop token")) {
    return;
  }

  const auto run_case = [
      &](const std::array<std::uint32_t,
                          gem16::internal::kMaximumMtpVerifyTokens>&
              host_verified,
          std::uint32_t stop_count, std::uint32_t expected_accepted,
          std::uint32_t expected_output, bool expected_stopped) {
    gem16::internal::MtpGroupTransaction host_transaction{};
    host_transaction.control.current.input_token = 7U;
    host_transaction.control.current.processed_position = 99U;
    host_transaction.control.current.remaining_output_capacity = 10U;
    host_transaction.control.current.output_write_position = 4U;
    host_transaction.control.next = host_transaction.control.current;
    host_transaction.control.fixed_draft_tokens = 2U;
    if (!CudaOk(cudaMemcpy(verified.get(), host_verified.data(),
                           verified.bytes(), cudaMemcpyHostToDevice),
                "copy MTP transition verified tokens") ||
        !CudaOk(cudaMemcpy(transaction.get(), &host_transaction,
                           sizeof(host_transaction), cudaMemcpyHostToDevice),
                "copy MTP transition control")) {
      return;
    }
    auto status = gem16::internal::LaunchAcceptMtpGroup(
        drafts.get(), verified.get(), 2U, stop_tokens.get(), stop_count,
        &transaction.get()->result, &transaction.get()->control, nullptr);
    CUDA_TEST_CHECK(status.ok());
    if (!status.ok() ||
        !CudaOk(cudaMemcpy(&host_transaction, transaction.get(),
                           sizeof(host_transaction), cudaMemcpyDeviceToHost),
                "copy MTP transition result")) {
      return;
    }
    const auto& result = host_transaction.result;
    const auto& control = host_transaction.control;
    CUDA_TEST_CHECK(result.accepted_count == expected_accepted);
    CUDA_TEST_CHECK(result.output_count == expected_output);
    CUDA_TEST_CHECK(result.stopped == (expected_stopped ? 1U : 0U));
    CUDA_TEST_CHECK(control.transition_valid == 1U);
    CUDA_TEST_CHECK(control.proposal_count == 2U);
    CUDA_TEST_CHECK(control.next.input_token ==
                    host_verified[expected_output - 1U]);
    CUDA_TEST_CHECK(control.next.processed_position ==
                    99U + expected_output);
    CUDA_TEST_CHECK(control.next.remaining_output_capacity ==
                    10U - expected_output);
    CUDA_TEST_CHECK(control.next.output_write_position ==
                    4U + expected_output);
    CUDA_TEST_CHECK(control.next.stopped ==
                    (expected_stopped ? 1U : 0U));
    CUDA_TEST_CHECK(control.next.stop_token ==
                    (expected_stopped ? stop : 0U));
  };

  run_case({30U, 40U, 50U, 0U, 0U}, 0U, 0U, 1U, false);
  run_case({10U, 30U, 50U, 0U, 0U}, 0U, 1U, 2U, false);
  run_case({10U, 20U, 30U, 0U, 0U}, 0U, 2U, 3U, false);
  run_case({10U, 99U, 30U, 0U, 0U}, 1U, 1U, 2U, true);
}

}  // namespace

int main(int argc, char** argv) {
  int device_count = 0;
  if (!CudaOk(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount") || device_count == 0) {
    std::cerr << "CUDA test requires one device\n";
    return 1;
  }
  if (argc == 2 && std::string_view(argv[1]) == "sampling") {
    TestGpuSampling();
    if (failures != 0) {
      std::cerr << failures << " CUDA test assertion(s) failed\n";
      return 1;
    }
    std::cout << "sampling CUDA tests passed\n";
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "online-decode") {
    TestOnlineFp8DecodeAttention();
    if (failures != 0) {
      std::cerr << failures << " CUDA test assertion(s) failed\n";
      return 1;
    }
    std::cout << "online decode CUDA tests passed\n";
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "decode-fusion") {
    TestLocalLayerReferenceOperators();
    TestFusedProjectionRmsNormRotaryBf16Batch();
    if (failures != 0) {
      std::cerr << failures << " CUDA test assertion(s) failed\n";
      return 1;
    }
    std::cout << "decode fusion CUDA tests passed\n";
    return 0;
  }
  TestCudaIntrinsicConformanceAndProjection();
  TestVllmNvfp4QuantizationBoundary();
  TestDirectSourceSm120Projection();
  TestCutlassSm120Projection();
  TestMlpElementwiseBridge();
  TestFp8ReferenceAndDirectProjection();
  TestFp8CutlassPrefillGeometry();
  TestLocalLayerReferenceOperators();
  TestFusedProjectionRmsNormRotaryBf16Batch();
  TestPhysicalFp8KvCache();
  TestWrappedKvRingAttention();
  TestOnlineFp8DecodeAttention();
  TestCausalPrefillAcrossWrappedRing();
  TestVectorizedFp8CausalPrefill();
  TestOnlineLocalFp8CausalPrefill();
  TestOnlineGlobalFp8CausalPrefill();
  TestMtpDeviceControlTransitions();
  TestGpuSampling();
  if (failures != 0) {
    std::cerr << failures << " CUDA test assertion(s) failed\n";
    return 1;
  }
  std::cout << "all CUDA tests passed\n";
  return 0;
}

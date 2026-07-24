#include "cuda/fp8/reference.h"
#include "cuda/fp8/sm120.h"
#include "cuda/layer/reference.h"
#include "cuda/nvfp4/reference.h"
#include "cuda/nvfp4/gemv.h"
#include "cuda/nvfp4/sm120.h"
#include "cuda/nvfp4/mlp.h"
#include "gem16gb/fp8.h"
#include "gem16gb/layer.h"
#include "gem16gb/nvfp4.h"

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

void TestCudaIntrinsicConformanceAndProjection() {
  std::array<float, 16> host_activation{};
  for (std::size_t index = 0; index < host_activation.size(); ++index) {
    host_activation[index] =
        gem16gb::nvfp4::DecodeE2M1(static_cast<std::uint8_t>(index)) / 2.0F;
  }
  const auto host_quantized = gem16gb::nvfp4::QuantizeActivation(host_activation, 2.0F);
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

  const gem16gb::Status quantize_status =
      gem16gb::internal::LaunchNvfp4ReferenceActivationQuantization(
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

  const gem16gb::Status projection_status = gem16gb::internal::LaunchNvfp4ReferenceProjection(
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
  const auto expected = gem16gb::nvfp4::ReferenceDotProduct(
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
      gem16gb::internal::LaunchNvfp4ReferenceActivationQuantization(
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
  constexpr std::size_t tokens = 2;
  constexpr float activation_divisor = 2.0F;
  constexpr float weight_divisor = 4.0F;

  std::array<float, k_size> host_activation{};
  for (std::size_t k = 0; k < k_size; ++k) {
    host_activation[k] =
        gem16gb::nvfp4::DecodeE2M1(static_cast<std::uint8_t>(k & 0x0FU)) /
        activation_divisor;
  }
  const auto quantized =
      gem16gb::nvfp4::QuantizeActivation(host_activation, activation_divisor);
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

  DeviceBuffer<std::uint8_t> device_activation(quantized.value().packed_e2m1.size());
  DeviceBuffer<std::uint8_t> device_activation_scales(
      quantized.value().block_scales_e4m3fn.size());
  DeviceBuffer<std::uint8_t> device_weight(packed_weight.size());
  DeviceBuffer<std::uint8_t> device_weight_scales(weight_scales.size());
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
  DeviceBuffer<float> device_batch_product(tokens * rows);
  if (device_activation.get() == nullptr || device_activation_scales.get() == nullptr ||
      device_weight.get() == nullptr || device_weight_scales.get() == nullptr ||
      device_output.get() == nullptr || device_simt_output.get() == nullptr ||
      device_fused_gate.get() == nullptr || device_fused_up.get() == nullptr ||
      device_fused_product.get() == nullptr ||
      device_batch_activation.get() == nullptr ||
      device_batch_activation_scales.get() == nullptr ||
      device_batch_reference.get() == nullptr || device_batch_native.get() == nullptr ||
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
              "copy native weights") ||
      !CudaOk(cudaMemcpy(device_weight_scales.get(), weight_scales.data(),
                         device_weight_scales.bytes(), cudaMemcpyHostToDevice),
              "copy native weight scales")) {
    return;
  }
  for (std::size_t token = 0; token < tokens; ++token) {
    if (!CudaOk(cudaMemcpy(
                    device_batch_activation.get() +
                        token * quantized.value().packed_e2m1.size(),
                    quantized.value().packed_e2m1.data(),
                    quantized.value().packed_e2m1.size(), cudaMemcpyHostToDevice),
                "copy batched native activation") ||
        !CudaOk(cudaMemcpy(
                    device_batch_activation_scales.get() +
                        token * quantized.value().block_scales_e4m3fn.size(),
                    quantized.value().block_scales_e4m3fn.data(),
                    quantized.value().block_scales_e4m3fn.size(),
                    cudaMemcpyHostToDevice),
                "copy batched native activation scales")) {
      return;
    }
  }

  const gem16gb::Status status = gem16gb::internal::LaunchNvfp4Sm120DirectProjection(
      device_activation.get(), device_activation_scales.get(), device_weight.get(),
      device_weight_scales.get(), device_output.get(), rows, k_size, activation_divisor,
      weight_divisor, nullptr);
  CUDA_TEST_CHECK(status.ok());
  const gem16gb::Status simt_status = gem16gb::internal::LaunchNvfp4SimtGemvProjection(
      device_activation.get(), device_activation_scales.get(), device_weight.get(),
      device_weight_scales.get(), device_simt_output.get(), rows, k_size,
      activation_divisor, weight_divisor, nullptr);
  CUDA_TEST_CHECK(simt_status.ok());
  const gem16gb::Status fused_status = gem16gb::internal::LaunchNvfp4Sm120FusedGateUp(
      device_activation.get(), device_activation_scales.get(), device_weight.get(),
      device_weight_scales.get(), device_weight.get(), device_weight_scales.get(),
      device_fused_gate.get(), device_fused_up.get(), device_fused_product.get(), rows,
      k_size, activation_divisor, weight_divisor, activation_divisor, weight_divisor,
      nullptr);
  CUDA_TEST_CHECK(fused_status.ok());
  const auto batch_reference_status =
      gem16gb::internal::LaunchNvfp4ReferenceProjectionBatch(
          device_batch_activation.get(), device_batch_activation_scales.get(),
          device_weight.get(), device_weight_scales.get(),
          device_batch_reference.get(), tokens, rows, k_size,
          activation_divisor, weight_divisor, nullptr);
  const auto batch_native_status =
      gem16gb::internal::LaunchNvfp4Sm120DirectProjectionBatch(
          device_batch_activation.get(), device_batch_activation_scales.get(),
          device_weight.get(), device_weight_scales.get(),
          device_batch_native.get(), tokens, rows, k_size, activation_divisor,
          weight_divisor, nullptr);
  const auto batch_fused_status =
      gem16gb::internal::LaunchNvfp4Sm120FusedGateUpBatch(
          device_batch_activation.get(), device_batch_activation_scales.get(),
          device_weight.get(), device_weight_scales.get(), device_weight.get(),
          device_weight_scales.get(), nullptr, nullptr,
          device_batch_product.get(), tokens, rows, k_size, activation_divisor,
          weight_divisor, activation_divisor, weight_divisor, nullptr);
  CUDA_TEST_CHECK(batch_reference_status.ok());
  CUDA_TEST_CHECK(batch_native_status.ok());
  CUDA_TEST_CHECK(batch_fused_status.ok());
  if (!status.ok() || !simt_status.ok() || !fused_status.ok() ||
      !batch_reference_status.ok() || !batch_native_status.ok() ||
      !batch_fused_status.ok() ||
      !CudaOk(cudaDeviceSynchronize(), "native projection synchronize")) return;

  std::array<float, rows> output{};
  std::array<float, rows> simt_output{};
  std::array<float, rows> fused_gate{};
  std::array<float, rows> fused_up{};
  std::array<float, rows> fused_product{};
  std::array<float, tokens * rows> batch_reference{};
  std::array<float, tokens * rows> batch_native{};
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
    const auto expected = gem16gb::nvfp4::ReferenceDotProduct(
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
        CUDA_TEST_CHECK(batch_reference[token * rows + row] == output[row]);
        CUDA_TEST_CHECK(batch_native[token * rows + row] == output[row]);
        CUDA_TEST_CHECK(batch_product[token * rows + row] == product);
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

  const auto product_status = gem16gb::internal::LaunchGeluTanhProduct(
      device_gate.get(), device_up.get(), device_product.get(), gate.size(), nullptr);
  CUDA_TEST_CHECK(product_status.ok());
  const auto residual_status = gem16gb::internal::LaunchAddResidual(
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
  constexpr std::size_t tokens = 2;
  std::array<float, k_size> host_activation{};
  for (std::size_t index = 0; index < host_activation.size(); ++index) {
    host_activation[index] =
        static_cast<float>(static_cast<int>(index % 13U) - 6) * 0.125F;
  }
  const auto host_quantized = gem16gb::fp8::QuantizeToken(host_activation);
  CUDA_TEST_CHECK(host_quantized.ok());
  if (!host_quantized.ok()) return;

  std::vector<std::uint8_t> host_weight(rows * k_size);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t k = 0; k < k_size; ++k) {
      const float value = static_cast<float>(static_cast<int>((row * 5U + k * 3U) % 15U) - 7) /
                          4.0F;
      const auto encoded = gem16gb::fp8::EncodeE4M3Fn(value);
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
  DeviceBuffer<float> device_batch_input(tokens * k_size);
  DeviceBuffer<std::uint8_t> device_batch_activation(tokens * k_size);
  DeviceBuffer<float> device_batch_scales(tokens);
  DeviceBuffer<float> device_batch_reference(tokens * rows);
  DeviceBuffer<float> device_batch_native(tokens * rows);
  if (device_input.get() == nullptr || device_activation.get() == nullptr ||
      device_activation_scale.get() == nullptr || device_weight.get() == nullptr ||
      device_weight_scales.get() == nullptr || device_reference.get() == nullptr ||
      device_native.get() == nullptr || device_batch_input.get() == nullptr ||
      device_batch_activation.get() == nullptr || device_batch_scales.get() == nullptr ||
      device_batch_reference.get() == nullptr || device_batch_native.get() == nullptr) {
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
    if (!CudaOk(cudaMemcpy(device_batch_input.get() + token * k_size,
                           host_activation.data(), k_size * sizeof(float),
                           cudaMemcpyHostToDevice), "copy batched FP8 input")) {
      return;
    }
  }

  const auto quantize_status = gem16gb::internal::LaunchFp8ReferenceTokenQuantization(
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

  const auto reference_status = gem16gb::internal::LaunchFp8ReferenceProjection(
      device_activation.get(), device_activation_scale.get(), device_weight.get(),
      device_weight_scales.get(), device_reference.get(), rows, k_size, nullptr);
  const auto native_status = gem16gb::internal::LaunchFp8Sm120DirectProjection(
      device_activation.get(), device_activation_scale.get(), device_weight.get(),
      device_weight_scales.get(), device_native.get(), rows, k_size, nullptr);
  CUDA_TEST_CHECK(reference_status.ok());
  CUDA_TEST_CHECK(native_status.ok());
  const auto batch_quantize_status =
      gem16gb::internal::LaunchFp8ReferenceTokenQuantizationBatch(
          device_batch_input.get(), device_batch_activation.get(),
          device_batch_scales.get(), tokens, k_size, nullptr);
  const auto batch_reference_status =
      gem16gb::internal::LaunchFp8ReferenceProjectionBatch(
          device_batch_activation.get(), device_batch_scales.get(),
          device_weight.get(), device_weight_scales.get(),
          device_batch_reference.get(), tokens, rows, k_size, nullptr);
  const auto batch_native_status =
      gem16gb::internal::LaunchFp8Sm120DirectProjectionBatch(
          device_batch_activation.get(), device_batch_scales.get(),
          device_weight.get(), device_weight_scales.get(),
          device_batch_native.get(), tokens, rows, k_size, nullptr);
  CUDA_TEST_CHECK(batch_quantize_status.ok());
  CUDA_TEST_CHECK(batch_reference_status.ok());
  CUDA_TEST_CHECK(batch_native_status.ok());
  if (!reference_status.ok() || !native_status.ok() ||
      !batch_quantize_status.ok() || !batch_reference_status.ok() ||
      !batch_native_status.ok() ||
      !CudaOk(cudaDeviceSynchronize(), "FP8 projection synchronize")) {
    return;
  }
  std::array<float, rows> reference_output{};
  std::array<float, rows> native_output{};
  std::array<float, tokens * rows> batch_reference_output{};
  std::array<float, tokens * rows> batch_native_output{};
  if (!CudaOk(cudaMemcpy(reference_output.data(), device_reference.get(), device_reference.bytes(),
                         cudaMemcpyDeviceToHost), "copy FP8 reference output") ||
      !CudaOk(cudaMemcpy(native_output.data(), device_native.get(), device_native.bytes(),
                         cudaMemcpyDeviceToHost), "copy FP8 native output") ||
      !CudaOk(cudaMemcpy(batch_reference_output.data(),
                         device_batch_reference.get(), device_batch_reference.bytes(),
                         cudaMemcpyDeviceToHost), "copy batched FP8 reference output") ||
      !CudaOk(cudaMemcpy(batch_native_output.data(), device_batch_native.get(),
                         device_batch_native.bytes(), cudaMemcpyDeviceToHost),
              "copy batched FP8 native output")) {
    return;
  }
  for (std::size_t row = 0; row < rows; ++row) {
    const auto expected = gem16gb::fp8::ReferenceDotProduct(
        host_quantized.value(),
        std::span<const std::uint8_t>(host_weight.data() + row * k_size, k_size),
        positive_weight_scales[row]);
    CUDA_TEST_CHECK(expected.ok());
    if (expected.ok()) {
      CUDA_TEST_CHECK(std::fabs(static_cast<double>(reference_output[row]) - expected.value()) <
                         1.0e-4);
      CUDA_TEST_CHECK(std::fabs(static_cast<double>(native_output[row]) - expected.value()) <
                         1.0e-4);
      for (std::size_t token = 0; token < tokens; ++token) {
        CUDA_TEST_CHECK(batch_reference_output[token * rows + row] ==
                        reference_output[row]);
        CUDA_TEST_CHECK(batch_native_output[token * rows + row] ==
                        native_output[row]);
      }
    }
  }
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

  const auto host_attention = gem16gb::layer::LocalAttentionDecode(
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

  const auto attention_status = gem16gb::internal::LaunchLocalAttentionDecode(
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
  const auto host_norm = gem16gb::layer::RmsNorm(norm_input, norm_weight, 2, 4, 1.0e-6F);
  CUDA_TEST_CHECK(host_norm.ok());
  DeviceBuffer<float> device_norm_input(norm_input.size());
  DeviceBuffer<std::uint16_t> device_norm_weight(norm_weight_bf16.size());
  DeviceBuffer<float> device_norm_output(norm_input.size());
  if (!host_norm.ok() || device_norm_input.get() == nullptr || device_norm_weight.get() == nullptr ||
      device_norm_output.get() == nullptr) return;
  if (!CudaOk(cudaMemcpy(device_norm_input.get(), norm_input.data(), device_norm_input.bytes(), cudaMemcpyHostToDevice),
              "copy norm input") ||
      !CudaOk(cudaMemcpy(device_norm_weight.get(), norm_weight_bf16.data(), device_norm_weight.bytes(),
                         cudaMemcpyHostToDevice), "copy norm weight")) return;
  const auto norm_status = gem16gb::internal::LaunchRmsNorm(
      device_norm_input.get(), device_norm_weight.get(), device_norm_output.get(), 2, 4, 1.0e-6F, nullptr);
  CUDA_TEST_CHECK(norm_status.ok());
  if (!norm_status.ok() || !CudaOk(cudaDeviceSynchronize(), "RMSNorm synchronize")) return;
  std::array<float, norm_input.size()> gpu_norm{};
  if (!CudaOk(cudaMemcpy(gpu_norm.data(), device_norm_output.get(), device_norm_output.bytes(),
                         cudaMemcpyDeviceToHost), "copy norm output")) return;
  for (std::size_t index = 0; index < gpu_norm.size(); ++index) {
    CUDA_TEST_CHECK(std::fabs(gpu_norm[index] - host_norm.value()[index]) < 2.0e-6F);
  }

  std::array<float, 8> host_rope = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
  DeviceBuffer<float> device_rope(host_rope.size());
  if (device_rope.get() == nullptr ||
      !CudaOk(cudaMemcpy(device_rope.get(), host_rope.data(), device_rope.bytes(), cudaMemcpyHostToDevice),
              "copy RoPE input")) return;
  CUDA_TEST_CHECK(gem16gb::layer::ApplyRotaryEmbedding(host_rope, 1, 8, 8, 37, 10000.0).ok());
  const auto rope_status = gem16gb::internal::LaunchRotaryEmbedding(
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
  CUDA_TEST_CHECK(gem16gb::layer::ApplyProportionalRotaryEmbedding(
                      host_proportional_rope, 1, 512, 0.25, 31, 1'000'000.0)
                      .ok());
  const auto proportional_status = gem16gb::internal::LaunchProportionalRotaryEmbedding(
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
    const auto append = gem16gb::internal::LaunchAppendKvFp8(
        device_keys.get(), device_values.get(), device_fp8_keys.get(),
        device_fp8_values.get(), device_key_scale.get(),
        device_value_scale.get(), token, kv_heads, head_dimension, nullptr);
    CUDA_TEST_CHECK(append.ok());
    if (!append.ok()) return;
  }
  const auto fp8_attention = gem16gb::internal::LaunchLocalAttentionDecodeFp8(
      device_query.get(), device_fp8_keys.get(), device_fp8_values.get(),
      device_key_scale.get(), device_value_scale.get(), device_scores.get(),
      device_fp8_output.get(), query_heads, kv_heads, head_dimension, tokens,
      nullptr);
  CUDA_TEST_CHECK(fp8_attention.ok());
  const auto float_attention = gem16gb::internal::LaunchLocalAttentionDecode(
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

  const auto expected = gem16gb::layer::LocalAttentionDecode(
      query, logical_keys, logical_values, query_heads, kv_heads,
      head_dimension, tokens);
  CUDA_TEST_CHECK(expected.ok());
  if (!expected.ok()) return;

  DeviceBuffer<float> device_query(query.size());
  DeviceBuffer<float> device_keys(physical_keys.size());
  DeviceBuffer<float> device_values(physical_values.size());
  DeviceBuffer<float> device_scores(query_heads * tokens);
  DeviceBuffer<float> device_output(query.size());
  DeviceBuffer<float> append_key(head_dimension);
  DeviceBuffer<float> append_value(head_dimension);
  DeviceBuffer<std::uint8_t> fp8_keys(physical_keys.size());
  DeviceBuffer<std::uint8_t> fp8_values(physical_values.size());
  DeviceBuffer<std::uint16_t> device_scale(unit_scale.size());
  DeviceBuffer<float> fp8_output(query.size());
  if (device_query.get() == nullptr || device_keys.get() == nullptr ||
      device_values.get() == nullptr || device_scores.get() == nullptr ||
      device_output.get() == nullptr || append_key.get() == nullptr ||
      append_value.get() == nullptr || fp8_keys.get() == nullptr ||
      fp8_values.get() == nullptr || device_scale.get() == nullptr ||
      fp8_output.get() == nullptr) {
    return;
  }
  if (!CudaOk(cudaMemcpy(device_query.get(), query.data(), device_query.bytes(),
                         cudaMemcpyHostToDevice), "copy ring query") ||
      !CudaOk(cudaMemcpy(device_keys.get(), physical_keys.data(), device_keys.bytes(),
                         cudaMemcpyHostToDevice), "copy ring keys") ||
      !CudaOk(cudaMemcpy(device_values.get(), physical_values.data(),
                         device_values.bytes(), cudaMemcpyHostToDevice),
              "copy ring values") ||
      !CudaOk(cudaMemcpy(device_scale.get(), unit_scale.data(), device_scale.bytes(),
                         cudaMemcpyHostToDevice), "copy ring FP8 scale")) {
    return;
  }

  const auto float_status = gem16gb::internal::LaunchLocalAttentionDecode(
      device_query.get(), device_keys.get(), device_values.get(),
      device_scores.get(), device_output.get(), query_heads, kv_heads,
      head_dimension, tokens, nullptr, tokens, first_slot);
  CUDA_TEST_CHECK(float_status.ok());
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
    const auto append_status = gem16gb::internal::LaunchAppendKvFp8(
        append_key.get(), append_value.get(), fp8_keys.get(), fp8_values.get(),
        device_scale.get(), device_scale.get(), physical_slot, kv_heads,
        head_dimension, nullptr);
    CUDA_TEST_CHECK(append_status.ok());
    if (!append_status.ok()) return;
  }
  const auto fp8_status = gem16gb::internal::LaunchLocalAttentionDecodeFp8(
      device_query.get(), fp8_keys.get(), fp8_values.get(), device_scale.get(),
      device_scale.get(), device_scores.get(), fp8_output.get(), query_heads,
      kv_heads, head_dimension, tokens, nullptr, tokens, first_slot);
  CUDA_TEST_CHECK(fp8_status.ok());
  if (!float_status.ok() || !fp8_status.ok() ||
      !CudaOk(cudaDeviceSynchronize(), "ring attention synchronize")) {
    return;
  }
  std::array<float, query.size()> float_result{};
  std::array<float, query.size()> fp8_result{};
  if (!CudaOk(cudaMemcpy(float_result.data(), device_output.get(),
                         device_output.bytes(), cudaMemcpyDeviceToHost),
              "copy ring float output") ||
      !CudaOk(cudaMemcpy(fp8_result.data(), fp8_output.get(), fp8_output.bytes(),
                         cudaMemcpyDeviceToHost), "copy ring FP8 output")) {
    return;
  }
  for (std::size_t index = 0; index < expected.value().size(); ++index) {
    CUDA_TEST_CHECK(std::fabs(float_result[index] - expected.value()[index]) <
                    2.0e-5F);
    CUDA_TEST_CHECK(std::fabs(fp8_result[index] - expected.value()[index]) <
                    2.0e-5F);
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
  const auto first_expected = gem16gb::layer::LocalAttentionDecode(
      std::span<const float>(queries.data(), head_dimension),
      first_logical_keys, first_logical_values, query_heads, kv_heads,
      head_dimension, capacity);
  const auto second_expected = gem16gb::layer::LocalAttentionDecode(
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
  DeviceBuffer<float> device_output(queries.size());
  if (device_queries.get() == nullptr || device_cache_keys.get() == nullptr ||
      device_cache_values.get() == nullptr || device_chunk_keys.get() == nullptr ||
      device_chunk_values.get() == nullptr || device_scores.get() == nullptr ||
      device_output.get() == nullptr) {
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
  const auto attention = gem16gb::internal::LaunchCausalAttentionPrefill(
      device_queries.get(), device_chunk_keys.get(), device_chunk_values.get(),
      device_cache_keys.get(), device_cache_values.get(), device_scores.get(),
      device_output.get(), start_position, tokens, query_heads, kv_heads,
      head_dimension, capacity, true, nullptr);
  CUDA_TEST_CHECK(attention.ok());
  const auto append = gem16gb::internal::LaunchAppendKvBatch(
      device_chunk_keys.get(), device_chunk_values.get(), device_cache_keys.get(),
      device_cache_values.get(), start_position, tokens,
      kv_heads * head_dimension, capacity, nullptr);
  CUDA_TEST_CHECK(append.ok());
  if (!attention.ok() || !append.ok() ||
      !CudaOk(cudaDeviceSynchronize(), "prefill ring synchronize")) {
    return;
  }
  std::array<float, queries.size()> output{};
  std::array<float, cache_keys.size()> appended_keys{};
  if (!CudaOk(cudaMemcpy(output.data(), device_output.get(), device_output.bytes(),
                         cudaMemcpyDeviceToHost), "copy prefill output") ||
      !CudaOk(cudaMemcpy(appended_keys.data(), device_cache_keys.get(),
                         device_cache_keys.bytes(), cudaMemcpyDeviceToHost),
              "copy appended prefill keys")) {
    return;
  }
  for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
    CUDA_TEST_CHECK(std::fabs(output[dimension] -
                              first_expected.value()[dimension]) < 2.0e-5F);
    CUDA_TEST_CHECK(std::fabs(output[head_dimension + dimension] -
                              second_expected.value()[dimension]) < 2.0e-5F);
  }
  CUDA_TEST_CHECK(appended_keys[2] == chunk_keys[0]);
  CUDA_TEST_CHECK(appended_keys[3] == chunk_keys[1]);
  CUDA_TEST_CHECK(appended_keys[4] == chunk_keys[2]);
  CUDA_TEST_CHECK(appended_keys[5] == chunk_keys[3]);
}

}  // namespace

int main() {
  int device_count = 0;
  if (!CudaOk(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount") || device_count == 0) {
    std::cerr << "CUDA test requires one device\n";
    return 1;
  }
  TestCudaIntrinsicConformanceAndProjection();
  TestVllmNvfp4QuantizationBoundary();
  TestDirectSourceSm120Projection();
  TestMlpElementwiseBridge();
  TestFp8ReferenceAndDirectProjection();
  TestLocalLayerReferenceOperators();
  TestPhysicalFp8KvCache();
  TestWrappedKvRingAttention();
  TestCausalPrefillAcrossWrappedRing();
  if (failures != 0) {
    std::cerr << failures << " CUDA test assertion(s) failed\n";
    return 1;
  }
  std::cout << "all CUDA tests passed\n";
  return 0;
}

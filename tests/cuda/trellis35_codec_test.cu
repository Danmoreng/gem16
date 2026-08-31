#include "trellis35_test_support.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace {

__global__ void ConvertHalf4ToFp8x4LegacyKernel(
    const std::uint16_t* input, std::uint32_t* output, unsigned groups) {
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= groups) return;
  input += static_cast<std::uint64_t>(index) * 4U;
  const half2 first_pair = __halves2half2(__ushort_as_half(input[0]),
                                          __ushort_as_half(input[1]));
  const half2 second_pair = __halves2half2(__ushort_as_half(input[2]),
                                           __ushort_as_half(input[3]));
  const __nv_fp8_e4m3 value0(__half2float(__low2half(first_pair)));
  const __nv_fp8_e4m3 value1(__half2float(__high2half(first_pair)));
  const __nv_fp8_e4m3 value2(__half2float(__low2half(second_pair)));
  const __nv_fp8_e4m3 value3(__half2float(__high2half(second_pair)));
  output[index] = static_cast<std::uint32_t>(value0.__x) |
                  (static_cast<std::uint32_t>(value1.__x) << 8U) |
                  (static_cast<std::uint32_t>(value2.__x) << 16U) |
                  (static_cast<std::uint32_t>(value3.__x) << 24U);
}

__global__ void ConvertHalf4ToFp8x4NativeKernel(
    const std::uint16_t* input, std::uint32_t* output, unsigned groups) {
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= groups) return;
  input += static_cast<std::uint64_t>(index) * 4U;
  const half2 first_pair = __halves2half2(__ushort_as_half(input[0]),
                                          __ushort_as_half(input[1]));
  const half2 second_pair = __halves2half2(__ushort_as_half(input[2]),
                                           __ushort_as_half(input[3]));
  output[index] = __nv_fp8x4_e4m3(first_pair, second_pair).__x;
}

void TestNativeFp8x4ConversionParity() {
  constexpr unsigned kGroups = 1U << 16U;
  constexpr unsigned kThreads = 256U;
  DeviceBuffer<std::uint16_t> input(static_cast<std::uint64_t>(kGroups) * 4U);
  DeviceBuffer<std::uint32_t> legacy(kGroups);
  DeviceBuffer<std::uint32_t> native(kGroups);
  std::vector<std::uint16_t> host_input(input.elements());
  for (unsigned index = 0U; index < kGroups; ++index) {
    host_input[index * 4U] = static_cast<std::uint16_t>(index);
    host_input[index * 4U + 1U] =
        static_cast<std::uint16_t>(index ^ 0xffffU);
    host_input[index * 4U + 2U] = static_cast<std::uint16_t>(
        ((index & 0xffU) << 8U) | (index >> 8U));
    host_input[index * 4U + 3U] =
        static_cast<std::uint16_t>(index * 40503U + 17U);
  }
  if (!Upload(input, host_input, "upload exhaustive FP8x4 half patterns")) {
    return;
  }
  const unsigned blocks = (kGroups + kThreads - 1U) / kThreads;
  ConvertHalf4ToFp8x4LegacyKernel<<<blocks, kThreads>>>(
      input.get(), legacy.get(), kGroups);
  ConvertHalf4ToFp8x4NativeKernel<<<blocks, kThreads>>>(
      input.get(), native.get(), kGroups);
  CHECK(CudaOk(cudaDeviceSynchronize(),
               "synchronize exhaustive native FP8x4 parity"));
  std::vector<std::uint32_t> host_legacy(kGroups);
  std::vector<std::uint32_t> host_native(kGroups);
  CHECK(CudaOk(cudaMemcpy(host_legacy.data(), legacy.get(), legacy.bytes(),
                          cudaMemcpyDeviceToHost),
               "download legacy FP8x4 conversions"));
  CHECK(CudaOk(cudaMemcpy(host_native.data(), native.get(), native.bytes(),
                          cudaMemcpyDeviceToHost),
               "download native FP8x4 conversions"));
  std::uint64_t mismatches = 0U;
  for (unsigned index = 0U; index < kGroups; ++index) {
    mismatches += host_legacy[index] != host_native[index] ? 1U : 0U;
  }
  std::cout << "native Half2-to-FP8x4 exhaustive bit mismatches="
            << mismatches << '\n';
  CHECK(mismatches == 0U);
}

void TestRandomizedProjectionParity() {
  constexpr std::uint64_t kInput = 128U;
  constexpr std::uint64_t kOutput = 128U;
  FamilyStorage family(kInput, kOutput, 19U);
  DeviceBuffer<std::uint32_t> ids(gem16::internal::kTrellis35M1TopK);
  DeviceBuffer<std::uint8_t> activation(
      gem16::internal::kTrellis35M1TopK * kInput);
  DeviceBuffer<float> scales(gem16::internal::kTrellis35M1TopK);
  DeviceBuffer<float> native_output(
      gem16::internal::kTrellis35M1TopK * kOutput);
  DeviceBuffer<float> reference_output(
      gem16::internal::kTrellis35M1TopK * kOutput);
  std::array<std::uint32_t, gem16::internal::kTrellis35M1TopK> host_ids{};
  std::vector<std::uint8_t> host_activation(activation.elements());
  std::vector<float> host_scales(scales.elements());
  for (std::uint32_t slot = 0U; slot < gem16::internal::kTrellis35M1TopK;
       ++slot) {
    host_ids[slot] = slot;
    std::array<float, kInput> token{};
    for (std::uint64_t index = 0U; index < kInput; ++index) {
      token[index] = std::sin(static_cast<float>(index * 17U + slot * 31U) *
                              0.03125F) *
                     0.125F;
    }
    const auto quantized = gem16::fp8::QuantizeToken(token);
    CHECK(quantized.ok());
    if (!quantized.ok()) return;
    host_scales[slot] = quantized.value().scale;
    std::copy(quantized.value().values_e4m3fn.begin(),
              quantized.value().values_e4m3fn.end(),
              host_activation.begin() + slot * kInput);
  }
  if (!Upload(ids, host_ids, "upload parity IDs") ||
      !Upload(activation, host_activation, "upload parity activation") ||
      !Upload(scales, host_scales, "upload parity scales")) {
    return;
  }
  const auto native = gem16::internal::LaunchTrellis35MmaW4A8ProjectionM1(
      activation.get(), scales.get(), family.binding, ids.get(),
      native_output.get(), kInput, kOutput, nullptr);
  CHECK(native.ok());
  for (std::uint32_t slot = 0U; slot < gem16::internal::kTrellis35M1TopK;
       ++slot) {
    const auto reference =
        gem16::internal::LaunchTrellis35ReferenceW4A8ProjectionM1(
            activation.get() + slot * kInput, scales.get() + slot,
            family.binding, host_ids[slot],
            reference_output.get() + slot * kOutput, kInput, kOutput,
            nullptr);
    CHECK(reference.ok());
  }
  CHECK(CudaOk(cudaDeviceSynchronize(), "synchronize projection parity"));
  std::vector<float> host_native(native_output.elements());
  std::vector<float> host_reference(reference_output.elements());
  CHECK(CudaOk(cudaMemcpy(host_native.data(), native_output.get(),
                          native_output.bytes(), cudaMemcpyDeviceToHost),
               "download native projection"));
  CHECK(CudaOk(cudaMemcpy(host_reference.data(), reference_output.get(),
                          reference_output.bytes(), cudaMemcpyDeviceToHost),
               "download reference projection"));
  Compare(host_reference, host_native, 2.0e-4F, 2.0e-5F,
          "randomized mixed-K3/K4 projection parity");
  const std::vector<float> host_oracle = HostProjection(
      MakePayload(3U, kInput, kOutput, 19U),
      MakePayload(4U, kInput, kOutput, 20U), host_activation, host_scales,
      host_ids, kInput, kOutput);
  Compare(host_oracle, host_native, 2.0e-4F, 2.0e-5F,
          "independent CPU mixed-K3/K4 projection parity");
}


}  // namespace

int RunTrellis35CodecTests() {
  TestNativeFp8x4ConversionParity();
  TestRandomizedProjectionParity();
  return failures;
}

#include "trellis35_test_support.h"

namespace {

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
  TestRandomizedProjectionParity();
  return failures;
}

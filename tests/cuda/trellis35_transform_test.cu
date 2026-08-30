#include "trellis35_test_support.h"

namespace {

void TestTransformsAndDownPadding() {
  constexpr std::uint64_t kLogical =
      gem16::internal::kTrellis35ExpertIntermediate;
  constexpr std::uint64_t kPhysical = gem16::internal::kTrellis35DownInput;
  DeviceBuffer<float> input(kLogical);
  DeviceBuffer<std::uint16_t> sidecar(kPhysical);
  DeviceBuffer<float> transformed(kPhysical);
  DeviceBuffer<float> reconstructed(kPhysical);
  std::vector<float> host_input(kLogical);
  std::vector<std::uint16_t> host_sidecar(kPhysical, 0x3c00U);
  for (std::uint64_t index = 0U; index < kLogical; ++index) {
    host_input[index] =
        std::sin(static_cast<float>(index * 29U + 3U) * 0.015625F);
  }
  if (!Upload(input, host_input, "upload transform input") ||
      !Upload(sidecar, host_sidecar, "upload transform sidecar")) {
    return;
  }
  auto status = gem16::internal::LaunchTrellis35InputTransformM1(
      input.get(), sidecar.get(), transformed.get(), kLogical, kPhysical,
      nullptr);
  CHECK(status.ok());
  const auto output_status = gem16::internal::LaunchTrellis35OutputTransformM1(
      transformed.get(), sidecar.get(), reconstructed.get(), kPhysical,
      nullptr);
  CHECK(output_status.ok());
  CHECK(CudaOk(cudaDeviceSynchronize(), "synchronize transform round trip"));
  std::vector<float> host_reconstructed(kPhysical);
  CHECK(CudaOk(cudaMemcpy(host_reconstructed.data(), reconstructed.get(),
                          reconstructed.bytes(), cudaMemcpyDeviceToHost),
               "download transform reconstruction"));
  float maximum_logical_error = 0.0F;
  float maximum_padding = 0.0F;
  for (std::uint64_t index = 0U; index < kLogical; ++index) {
    maximum_logical_error = std::max(
        maximum_logical_error,
        std::fabs(host_reconstructed[index] - host_input[index]));
  }
  for (std::uint64_t index = kLogical; index < kPhysical; ++index) {
    maximum_padding =
        std::max(maximum_padding, std::fabs(host_reconstructed[index]));
  }
  std::cout << "Down 704->768 transform roundtrip max_abs="
            << maximum_logical_error << " padding_max_abs=" << maximum_padding
            << '\n';
  CHECK(maximum_logical_error <= 2.0e-6F);
  CHECK(maximum_padding <= 2.0e-6F);

  constexpr std::uint64_t kFused = gem16::internal::kTrellis35GateUpOutput;
  DeviceBuffer<float> fused_transformed(kFused);
  DeviceBuffer<std::uint16_t> fused_sidecar(kFused);
  DeviceBuffer<float> fused_output(kFused);
  std::vector<float> host_fused(kFused, 0.0F);
  std::vector<std::uint16_t> host_fused_sidecar(kFused, 0x3c00U);
  // Index 750 shares the [640, 768) Hadamard block with both the final Gate
  // values [640, 704) and the first Up values [704, 768).
  host_fused[750] = 1.0F;
  if (!Upload(fused_transformed, host_fused, "upload fused impulse") ||
      !Upload(fused_sidecar, host_fused_sidecar,
              "upload fused output sidecar")) {
    return;
  }
  const auto fused_status = gem16::internal::LaunchTrellis35OutputTransformM1(
      fused_transformed.get(), fused_sidecar.get(), fused_output.get(),
      kFused, nullptr);
  CHECK(fused_status.ok());
  CHECK(CudaOk(cudaDeviceSynchronize(), "synchronize fused output transform"));
  std::vector<float> host_fused_output(kFused);
  CHECK(CudaOk(cudaMemcpy(host_fused_output.data(), fused_output.get(),
                          fused_output.bytes(), cudaMemcpyDeviceToHost),
               "download fused output transform"));
  float gate_energy = 0.0F;
  float up_energy = 0.0F;
  for (std::uint64_t index = 640U; index < 704U; ++index) {
    gate_energy += std::fabs(host_fused_output[index]);
  }
  for (std::uint64_t index = 704U; index < 768U; ++index) {
    up_energy += std::fabs(host_fused_output[index]);
  }
  std::cout << "fused Gate+Up crossing-block gate_l1=" << gate_energy
            << " up_l1=" << up_energy << '\n';
  CHECK(gate_energy > 0.0F && up_energy > 0.0F);
}


}  // namespace

int RunTrellis35TransformTests() {
  TestTransformsAndDownPadding();
  return failures;
}

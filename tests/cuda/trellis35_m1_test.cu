#include "trellis35_test_support.h"

namespace {

void TestSyntheticFullM1() {
  FamilyStorage gate(gem16::internal::kTrellis35GateUpInput,
                     gem16::internal::kTrellis35GateUpOutput, 101U);
  FamilyStorage down(gem16::internal::kTrellis35DownInput,
                     gem16::internal::kTrellis35DownOutput, 211U);
  const gem16::internal::Trellis35DeviceLayerBinding layer{gate.binding,
                                                           down.binding};
  DeviceBuffer<float> input(gem16::internal::kTrellis35GateUpInput);
  DeviceBuffer<std::uint32_t> ids(gem16::internal::kTrellis35M1TopK);
  DeviceBuffer<float> weights(gem16::internal::kTrellis35M1TopK);
  DeviceBuffer<float> rollback_output(gem16::internal::kTrellis35DownOutput);
  DeviceBuffer<float> candidate_output(gem16::internal::kTrellis35DownOutput);
  M1Storage rollback_storage;
  M1Storage candidate_storage;
  std::vector<float> host_input(input.elements());
  for (std::uint64_t index = 0U; index < host_input.size(); ++index) {
    host_input[index] =
        std::sin(static_cast<float>(index) * 0.00390625F) * 1.0e-4F;
  }
  const std::array<std::uint32_t, gem16::internal::kTrellis35M1TopK>
      host_ids{0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U};
  const std::array<float, gem16::internal::kTrellis35M1TopK> host_weights{
      0.19F, 0.17F, 0.15F, 0.13F, 0.11F, 0.10F, 0.08F, 0.07F};
  if (!Upload(input, host_input, "upload synthetic M1 input") ||
      !Upload(ids, host_ids, "upload synthetic M1 IDs") ||
      !Upload(weights, host_weights, "upload synthetic M1 weights")) {
    return;
  }
  const float rollback_latency = RunFullM1(
      layer, input, ids, weights, rollback_storage, rollback_output, 3U,
      false, gem16::internal::Trellis35SmallTransformMode::kWarpH128,
      gem16::internal::Trellis35SmallGeluDownMode::kFusedTransformQuantize,
      gem16::internal::Trellis35M1ProjectionOutputMode::kFusedN128,
      gem16::internal::Trellis35VectorStoreMode::kDisabled);
  const float candidate_latency = RunFullM1(
      layer, input, ids, weights, candidate_storage, candidate_output, 3U,
      true, gem16::internal::Trellis35SmallTransformMode::kWarpH128,
      gem16::internal::Trellis35SmallGeluDownMode::kFusedTransformQuantize,
      gem16::internal::Trellis35M1ProjectionOutputMode::kFusedN128,
      gem16::internal::Trellis35VectorStoreMode::kEnabled);
  auto compare_device = [](const DeviceBuffer<float>& expected,
                           const DeviceBuffer<float>& actual,
                           const char* description) {
    std::vector<float> host_expected(expected.elements());
    std::vector<float> host_actual(actual.elements());
    CHECK(CudaOk(cudaMemcpy(host_expected.data(), expected.get(),
                            expected.bytes(), cudaMemcpyDeviceToHost),
                 "download PFX28-D M1 rollback"));
    CHECK(CudaOk(cudaMemcpy(host_actual.data(), actual.get(), actual.bytes(),
                            cudaMemcpyDeviceToHost),
                 "download PFX28-D M1 candidate"));
    Compare(host_expected, host_actual, 0.0F, 0.0F, description);
  };
  compare_device(rollback_storage.gate_output,
                 candidate_storage.gate_output,
                 "PFX28-D M1 vector Gate+Up parity");
  compare_device(rollback_storage.down_output, candidate_storage.down_output,
                 "PFX28-D M1 vector Down parity");
  compare_device(rollback_output, candidate_output,
                 "PFX28-D M1 vector reduced parity");
  CheckFiniteAndIds(candidate_output, ids, host_ids, "synthetic full M1");
  CheckSlotOrderedReduction(candidate_storage.down_output, host_weights,
                            candidate_output,
                            "synthetic full M1");
  std::cout << "PFX28-D M1 rollback_ms=" << rollback_latency
            << " candidate_ms=" << candidate_latency << '\n';
}


}  // namespace

int RunTrellis35M1Tests() {
  TestSyntheticFullM1();
  return failures;
}

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
  DeviceBuffer<float> output(gem16::internal::kTrellis35DownOutput);
  M1Storage storage;
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
  const float latency =
      RunFullM1(layer, input, ids, weights, storage, output, 3U, true);
  CheckFiniteAndIds(output, ids, host_ids, "synthetic full M1");
  CheckSlotOrderedReduction(storage.down_output, host_weights, output,
                            "synthetic full M1");
  std::cout << "synthetic full M1 latency_ms=" << latency << '\n';
}


}  // namespace

int RunTrellis35M1Tests() {
  TestSyntheticFullM1();
  return failures;
}

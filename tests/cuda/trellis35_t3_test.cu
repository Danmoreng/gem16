#include "trellis35_test_support.h"

namespace {

void TestSyntheticT3() {
  FamilyStorage gate(gem16::internal::kTrellis35GateUpInput,
                     gem16::internal::kTrellis35GateUpOutput, 307U);
  FamilyStorage down(gem16::internal::kTrellis35DownInput,
                     gem16::internal::kTrellis35DownOutput, 401U);
  const gem16::internal::Trellis35DeviceLayerBinding layer{gate.binding,
                                                           down.binding};
  DeviceBuffer<float> input(gem16::internal::kTrellis35T3Rows *
                            gem16::internal::kTrellis35GateUpInput);
  std::vector<float> host_input(input.elements());
  for (unsigned row = 0U; row < gem16::internal::kTrellis35T3Rows; ++row) {
    for (std::uint64_t index = 0U;
         index < gem16::internal::kTrellis35GateUpInput; ++index) {
      host_input[static_cast<std::uint64_t>(row) *
                     gem16::internal::kTrellis35GateUpInput +
                 index] =
          std::sin(static_cast<float>(index * 11U + row * 37U) *
                   0.00390625F) *
          1.0e-4F;
    }
  }
  if (!Upload(input, host_input, "upload synthetic T3 input")) return;
  std::array<float, gem16::internal::kTrellis35T3Assignments> weights{};
  for (unsigned row = 0U; row < gem16::internal::kTrellis35T3Rows; ++row) {
    const std::array<float, gem16::internal::kTrellis35M1TopK> row_weights{
        0.19F, 0.17F, 0.15F, 0.13F, 0.11F, 0.10F, 0.08F, 0.07F};
    std::copy(row_weights.begin(), row_weights.end(),
              weights.begin() + row * gem16::internal::kTrellis35M1TopK);
  }

  std::array<std::uint32_t, gem16::internal::kTrellis35T3Assignments>
      zero_overlap{};
  for (unsigned assignment = 0U; assignment < zero_overlap.size();
       ++assignment) {
    zero_overlap[assignment] = assignment;
  }
  std::array<std::uint32_t, gem16::internal::kTrellis35T3Assignments>
      typical_overlap{0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U,
                      0U, 1U, 2U, 3U, 8U, 9U, 10U, 11U,
                      0U, 1U, 4U, 5U, 12U, 13U, 14U, 15U};
  std::array<std::uint32_t, gem16::internal::kTrellis35T3Assignments>
      maximal_overlap{};
  for (unsigned assignment = 0U; assignment < maximal_overlap.size();
       ++assignment) {
    maximal_overlap[assignment] =
        assignment % gem16::internal::kTrellis35M1TopK;
  }
  (void)RunT3Scenario(layer, input, zero_overlap, weights,
                      "synthetic T3 zero-overlap", false);
  (void)RunT3Scenario(layer, input, typical_overlap, weights,
                      "synthetic T3 typical-overlap", false);
  (void)RunT3Scenario(layer, input, maximal_overlap, weights,
                      "synthetic T3 maximal-overlap", true);
}

void ProfileSyntheticT3(unsigned unique_experts) {
  FamilyStorage gate(gem16::internal::kTrellis35GateUpInput,
                     gem16::internal::kTrellis35GateUpOutput, 503U);
  FamilyStorage down(gem16::internal::kTrellis35DownInput,
                     gem16::internal::kTrellis35DownOutput, 601U);
  const gem16::internal::Trellis35DeviceLayerBinding layer{gate.binding,
                                                           down.binding};
  DeviceBuffer<float> input(gem16::internal::kTrellis35T3Rows *
                            gem16::internal::kTrellis35GateUpInput);
  DeviceBuffer<std::uint32_t> ids(gem16::internal::kTrellis35T3Assignments);
  DeviceBuffer<float> weights(gem16::internal::kTrellis35T3Assignments);
  DeviceBuffer<float> output(gem16::internal::kTrellis35T3Rows *
                             gem16::internal::kTrellis35DownOutput);
  T3Storage storage;
  std::vector<float> host_input(input.elements(), 1.0e-4F);
  std::array<float, gem16::internal::kTrellis35T3Assignments> host_weights{};
  host_weights.fill(0.125F);
  const auto host_ids = MakeT3RoutesWithUnionSize(unique_experts);
  if (!Upload(input, host_input, "upload profile T3 input") ||
      !Upload(ids, host_ids, "upload profile T3 IDs") ||
      !Upload(weights, host_weights, "upload profile T3 weights")) {
    return;
  }
  const auto status = gem16::internal::LaunchTrellis35SelectedExpertsT3(
      input.get(), ids.get(), weights.get(), layer, storage.Bind(),
      output.get(), nullptr);
  CHECK(status.ok());
  CHECK(CudaOk(cudaDeviceSynchronize(), "synchronize profile T3"));
  std::cout << "profile T3 unique_experts=" << unique_experts << '\n';
}


}  // namespace

int RunTrellis35T3Tests() {
  TestSyntheticT3();
  return failures;
}

int ProfileTrellis35T3(unsigned unique_experts) {
  ProfileSyntheticT3(unique_experts);
  return failures;
}

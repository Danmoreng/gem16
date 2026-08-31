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

void TestWp27T3N128InverseMatrix() {
  struct RateCase {
    std::uint16_t forced_rate;
    const char* name;
  };
  constexpr std::array<RateCase, 3> kRates{{{3U, "K3"},
                                            {4U, "K4"},
                                            {0U, "mixed"}}};
  struct ModeCase {
    gem16::internal::Trellis35T3ProjectionOutputMode mode;
    const char* name;
  };
  constexpr std::array<ModeCase, 4> kModes{{
      {gem16::internal::Trellis35T3ProjectionOutputMode::kSeparateN32,
       "off"},
      {gem16::internal::Trellis35T3ProjectionOutputMode::kGateUpFusedN128,
       "gate"},
      {gem16::internal::Trellis35T3ProjectionOutputMode::kDownFusedN128,
       "down"},
      {gem16::internal::Trellis35T3ProjectionOutputMode::kFusedN128, "both"},
  }};

  DeviceBuffer<float> input(gem16::internal::kTrellis35T3Rows *
                            gem16::internal::kTrellis35GateUpInput);
  std::vector<float> host_input(input.elements());
  for (std::uint64_t index = 0U; index < host_input.size(); ++index) {
    host_input[index] =
        std::sin(static_cast<float>(index * 29U + 17U) * 0.00390625F) *
        1.0e-4F;
  }
  CHECK(Upload(input, host_input, "upload WP27 T3 input"));
  const auto host_ids = MakeT3RoutesWithUnionSize(16U);
  std::array<float, gem16::internal::kTrellis35T3Assignments> host_weights{};
  host_weights.fill(0.125F);
  DeviceBuffer<std::uint32_t> ids(gem16::internal::kTrellis35T3Assignments);
  DeviceBuffer<float> weights(gem16::internal::kTrellis35T3Assignments);
  CHECK(Upload(ids, host_ids, "upload WP27 T3 IDs"));
  CHECK(Upload(weights, host_weights, "upload WP27 T3 weights"));

  auto compare_bits = [](const DeviceBuffer<float>& expected,
                         const DeviceBuffer<float>& actual,
                         const std::string& description) {
    CHECK(expected.elements() == actual.elements());
    std::vector<float> host_expected(expected.elements());
    std::vector<float> host_actual(actual.elements());
    CHECK(CudaOk(cudaMemcpy(host_expected.data(), expected.get(),
                            expected.bytes(), cudaMemcpyDeviceToHost),
                 "download WP27 expected values"));
    CHECK(CudaOk(cudaMemcpy(host_actual.data(), actual.get(), actual.bytes(),
                            cudaMemcpyDeviceToHost),
                 "download WP27 actual values"));
    std::uint64_t mismatches = 0U;
    for (std::size_t index = 0U; index < host_expected.size(); ++index) {
      if (std::bit_cast<std::uint32_t>(host_expected[index]) !=
          std::bit_cast<std::uint32_t>(host_actual[index])) {
        ++mismatches;
      }
    }
    std::cout << description << " bit_mismatches=" << mismatches << '\n';
    CHECK(mismatches == 0U);
  };

  for (const auto& rate : kRates) {
    FamilyStorage gate(gem16::internal::kTrellis35GateUpInput,
                       gem16::internal::kTrellis35GateUpOutput, 2701U,
                       rate.forced_rate);
    FamilyStorage down(gem16::internal::kTrellis35DownInput,
                       gem16::internal::kTrellis35DownOutput, 2711U,
                       rate.forced_rate);
    const gem16::internal::Trellis35DeviceLayerBinding layer{gate.binding,
                                                             down.binding};
    T3Storage baseline_storage;
    DeviceBuffer<float> baseline_output(
        gem16::internal::kTrellis35T3Rows *
        gem16::internal::kTrellis35DownOutput);
    (void)RunFullT3(
        layer, input, ids, weights, baseline_storage, baseline_output, 1U,
        false, gem16::internal::Trellis35SmallTransformMode::kWarpH128,
        gem16::internal::Trellis35T3ProjectionMode::kM16,
        gem16::internal::Trellis35SmallGeluDownMode::kFusedTransformQuantize,
        gem16::internal::Trellis35T3ProjectionOutputMode::kSeparateN32);
    for (const auto& mode : kModes) {
      T3Storage rollback_storage;
      T3Storage candidate_storage;
      DeviceBuffer<float> rollback_output(
          gem16::internal::kTrellis35T3Rows *
          gem16::internal::kTrellis35DownOutput);
      DeviceBuffer<float> candidate_output(
          gem16::internal::kTrellis35T3Rows *
          gem16::internal::kTrellis35DownOutput);
      (void)RunFullT3(
          layer, input, ids, weights, rollback_storage, rollback_output, 1U,
          false, gem16::internal::Trellis35SmallTransformMode::kWarpH128,
          gem16::internal::Trellis35T3ProjectionMode::kM16,
          gem16::internal::Trellis35SmallGeluDownMode::
              kFusedTransformQuantize,
          mode.mode, gem16::internal::Trellis35VectorStoreMode::kDisabled);
      (void)RunFullT3(
          layer, input, ids, weights, candidate_storage, candidate_output, 1U,
          mode.mode ==
              gem16::internal::Trellis35T3ProjectionOutputMode::kFusedN128,
          gem16::internal::Trellis35SmallTransformMode::kWarpH128,
          gem16::internal::Trellis35T3ProjectionMode::kM16,
          gem16::internal::Trellis35SmallGeluDownMode::
              kFusedTransformQuantize,
          mode.mode, gem16::internal::Trellis35VectorStoreMode::kEnabled);
      const std::string prefix = std::string("WP27 T3 N128 ") + rate.name +
                                 " mode=" + mode.name;
      compare_bits(rollback_storage.gate_output,
                   candidate_storage.gate_output,
                   prefix + " PFX28-D vector Gate+Up");
      compare_bits(rollback_storage.down_output,
                   candidate_storage.down_output,
                   prefix + " PFX28-D vector Down");
      compare_bits(rollback_output, candidate_output,
                   prefix + " PFX28-D vector reduced");
      compare_bits(baseline_storage.gate_output,
                   candidate_storage.gate_output, prefix + " Gate+Up");
      compare_bits(baseline_storage.down_output,
                   candidate_storage.down_output, prefix + " Down");
      compare_bits(baseline_output, candidate_output, prefix + " reduced");
    }
  }
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

void TestWp20SmallGeluDownMatrix() {
  struct RateCase {
    std::uint16_t forced_rate;
    const char* name;
  };
  constexpr std::array<RateCase, 3> kRates{{{3U, "K3"},
                                            {4U, "K4"},
                                            {0U, "mixed"}}};
  DeviceBuffer<float> input(gem16::internal::kTrellis35T3Rows *
                            gem16::internal::kTrellis35GateUpInput);
  std::vector<float> host_input(input.elements());
  for (std::uint64_t index = 0U; index < host_input.size(); ++index) {
    host_input[index] =
        std::sin(static_cast<float>(index * 13U + 7U) * 0.00390625F) *
        1.0e-4F;
  }
  CHECK(Upload(input, host_input, "upload WP20 GELU/Down input"));
  std::array<float, gem16::internal::kTrellis35T3Assignments> host_weights{};
  host_weights.fill(0.125F);

  for (const auto& rate : kRates) {
    FamilyStorage gate(gem16::internal::kTrellis35GateUpInput,
                       gem16::internal::kTrellis35GateUpOutput, 2003U,
                       rate.forced_rate);
    FamilyStorage down(gem16::internal::kTrellis35DownInput,
                       gem16::internal::kTrellis35DownOutput, 2011U,
                       rate.forced_rate);
    const gem16::internal::Trellis35DeviceLayerBinding layer{gate.binding,
                                                             down.binding};
    for (unsigned unique_experts = gem16::internal::kTrellis35M1TopK;
         unique_experts <= gem16::internal::kTrellis35T3Assignments;
         ++unique_experts) {
      const auto host_ids = MakeT3RoutesWithUnionSize(unique_experts);
      DeviceBuffer<std::uint32_t> ids(
          gem16::internal::kTrellis35T3Assignments);
      DeviceBuffer<float> weights(gem16::internal::kTrellis35T3Assignments);
      DeviceBuffer<float> baseline_output(
          gem16::internal::kTrellis35T3Rows *
          gem16::internal::kTrellis35DownOutput);
      DeviceBuffer<float> fused_output(
          gem16::internal::kTrellis35T3Rows *
          gem16::internal::kTrellis35DownOutput);
      T3Storage baseline_storage;
      T3Storage fused_storage;
      CHECK(Upload(ids, host_ids, "upload WP20 GELU/Down IDs"));
      CHECK(Upload(weights, host_weights, "upload WP20 GELU/Down weights"));
      const float baseline_ms = RunFullT3(
          layer, input, ids, weights, baseline_storage, baseline_output, 10U,
          false);
      const float fused_ms = RunFullT3(
          layer, input, ids, weights, fused_storage, fused_output, 10U,
          rate.forced_rate == 0U && unique_experts == 16U,
          gem16::internal::Trellis35SmallTransformMode::kWarpH128,
          gem16::internal::Trellis35T3ProjectionMode::kM16,
          gem16::internal::Trellis35SmallGeluDownMode::
              kFusedTransformQuantize);
      std::vector<float> host_baseline(baseline_output.elements());
      std::vector<float> host_fused(fused_output.elements());
      CHECK(CudaOk(cudaMemcpy(host_baseline.data(), baseline_output.get(),
                              baseline_output.bytes(),
                              cudaMemcpyDeviceToHost),
                   "download WP20 GELU/Down baseline output"));
      CHECK(CudaOk(cudaMemcpy(host_fused.data(), fused_output.get(),
                              fused_output.bytes(), cudaMemcpyDeviceToHost),
                   "download WP20 GELU/Down fused output"));
      const std::string description =
          std::string("WP20 GELU/Down ") + rate.name + " union=" +
          std::to_string(unique_experts);
      Compare(host_baseline, host_fused, 0.0F, 0.0F,
              description.c_str());
      std::cout << description << " baseline_ms=" << baseline_ms
                << " fused_ms=" << fused_ms << '\n';
    }
  }

  DeviceBuffer<float> m1_input(gem16::internal::kTrellis35GateUpInput);
  DeviceBuffer<std::uint32_t> m1_ids(gem16::internal::kTrellis35M1TopK);
  DeviceBuffer<float> m1_weights(gem16::internal::kTrellis35M1TopK);
  std::array<std::uint32_t, gem16::internal::kTrellis35M1TopK> host_m1_ids{
      0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U};
  std::array<float, gem16::internal::kTrellis35M1TopK> host_m1_weights{};
  host_m1_weights.fill(0.125F);
  std::vector<float> host_m1_input(host_input.begin(),
                                   host_input.begin() + m1_input.elements());
  CHECK(Upload(m1_input, host_m1_input, "upload WP20 M1 GELU/Down input"));
  CHECK(Upload(m1_ids, host_m1_ids, "upload WP20 M1 GELU/Down IDs"));
  CHECK(Upload(m1_weights, host_m1_weights,
               "upload WP20 M1 GELU/Down weights"));
  for (const auto& rate : kRates) {
    FamilyStorage gate(gem16::internal::kTrellis35GateUpInput,
                       gem16::internal::kTrellis35GateUpOutput, 2027U,
                       rate.forced_rate);
    FamilyStorage down(gem16::internal::kTrellis35DownInput,
                       gem16::internal::kTrellis35DownOutput, 2029U,
                       rate.forced_rate);
    const gem16::internal::Trellis35DeviceLayerBinding layer{gate.binding,
                                                             down.binding};
    DeviceBuffer<float> baseline_output(gem16::internal::kTrellis35DownOutput);
    DeviceBuffer<float> fused_output(gem16::internal::kTrellis35DownOutput);
    M1Storage baseline_storage;
    M1Storage fused_storage;
    const float baseline_ms = RunFullM1(
        layer, m1_input, m1_ids, m1_weights, baseline_storage, baseline_output,
        20U, false);
    const float fused_ms = RunFullM1(
        layer, m1_input, m1_ids, m1_weights, fused_storage, fused_output, 20U,
        rate.forced_rate == 0U,
        gem16::internal::Trellis35SmallTransformMode::kWarpH128,
        gem16::internal::Trellis35SmallGeluDownMode::kFusedTransformQuantize);
    std::vector<float> host_baseline(baseline_output.elements());
    std::vector<float> host_fused(fused_output.elements());
    CHECK(CudaOk(cudaMemcpy(host_baseline.data(), baseline_output.get(),
                            baseline_output.bytes(), cudaMemcpyDeviceToHost),
                 "download WP20 M1 GELU/Down baseline output"));
    CHECK(CudaOk(cudaMemcpy(host_fused.data(), fused_output.get(),
                            fused_output.bytes(), cudaMemcpyDeviceToHost),
                 "download WP20 M1 GELU/Down fused output"));
    const std::string description =
        std::string("WP20 M1 GELU/Down ") + rate.name;
    Compare(host_baseline, host_fused, 0.0F, 0.0F, description.c_str());
    std::cout << description << " baseline_ms=" << baseline_ms
              << " fused_ms=" << fused_ms << '\n';
  }
}


}  // namespace

int RunTrellis35T3Tests() {
  TestSyntheticT3();
  TestWp27T3N128InverseMatrix();
  return failures;
}

int ProfileTrellis35T3(unsigned unique_experts) {
  ProfileSyntheticT3(unique_experts);
  return failures;
}

int RunTrellis35Wp20SmallGeluDownMatrix() {
  TestWp20SmallGeluDownMatrix();
  return failures;
}

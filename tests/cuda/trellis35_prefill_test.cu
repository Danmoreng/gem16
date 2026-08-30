#include "trellis35_test_support.h"

namespace {

void TestSyntheticPrefill() {
  auto plan = gem16::internal::BuildGemma4MoePrefillPlan(1024U);
  CHECK(plan.ok());
  if (plan.ok()) {
    CHECK(plan.value().chunk_tokens == 1024U);
    auto region_bytes = [&](std::string_view name) {
      const auto found = std::find_if(
          plan.value().moe_regions.begin(), plan.value().moe_regions.end(),
          [&](const auto& region) { return region.name == name; });
      CHECK(found != plan.value().moe_regions.end());
      return found == plan.value().moe_regions.end() ? 0U : found->bytes;
    };
    constexpr std::uint64_t kPlanTokens = 1024U;
    constexpr std::uint64_t kAssignments =
        kPlanTokens * gem16::internal::kTrellis35M1TopK;
    CHECK(region_bytes("expert_product") ==
          kAssignments * gem16::internal::kTrellis35ExpertIntermediate *
              sizeof(float));
    CHECK(region_bytes("expert_product") >=
          kAssignments * gem16::internal::kTrellis35GateUpInput);
    CHECK(region_bytes("expert_down_partials") ==
          kAssignments * gem16::internal::kTrellis35DownOutput *
              sizeof(float));
    CHECK(region_bytes("shared_product") >=
          kAssignments * 128U * sizeof(float));
    CHECK(region_bytes("shared_output") >=
          kAssignments * gem16::internal::kTrellis35DownInput);
    CHECK(region_bytes("token_nvfp4") -
              kPlanTokens *
                  (gem16::internal::kTrellis35GateUpInput / 2U) >=
          kAssignments * sizeof(float));
  }
  FamilyStorage gate(gem16::internal::kTrellis35GateUpInput,
                     gem16::internal::kTrellis35GateUpOutput, 701U);
  FamilyStorage down(gem16::internal::kTrellis35DownInput,
                     gem16::internal::kTrellis35DownOutput, 809U);
  const gem16::internal::Trellis35DeviceLayerBinding layer{gate.binding,
                                                           down.binding};
  constexpr std::uint64_t kTokens = 4U;
  DeviceBuffer<float> input(kTokens * gem16::internal::kTrellis35GateUpInput);
  std::vector<float> host_input(input.elements());
  for (std::uint64_t index = 0U; index < host_input.size(); ++index) {
    host_input[index] =
        std::sin(static_cast<float>(index * 7U + 3U) * 0.001953125F) *
        1.0e-4F;
  }
  CHECK(Upload(input, host_input, "upload synthetic prefill input"));
  PrefillStorage storage(kTokens);
  const auto routing = MakePrefillRouting(kTokens, SequentialExpertOrder());
  (void)RunPrefillScenario(layer, input, storage, routing,
                           "synthetic W4A8 prefill", true, true);
}

void ProfileSyntheticPrefill(std::uint64_t tokens) {
  FamilyStorage gate(gem16::internal::kTrellis35GateUpInput,
                     gem16::internal::kTrellis35GateUpOutput, 907U);
  FamilyStorage down(gem16::internal::kTrellis35DownInput,
                     gem16::internal::kTrellis35DownOutput, 1009U);
  const gem16::internal::Trellis35DeviceLayerBinding layer{gate.binding,
                                                           down.binding};
  DeviceBuffer<float> input(tokens * gem16::internal::kTrellis35GateUpInput);
  std::vector<float> host_input(input.elements(), 1.0e-4F);
  CHECK(Upload(input, host_input, "upload profile prefill input"));
  PrefillStorage storage(tokens);
  const auto routing = MakePrefillRouting(tokens, SequentialExpertOrder());
  CHECK(UploadPrefillRouting(storage, routing));
  const float latency = RunFullPrefill(layer, input, storage, 1U, false);
  std::cout << "profile W4A8 prefill tokens=" << tokens
            << " assignments=" << tokens * gem16::internal::kTrellis35M1TopK
            << " latency_ms=" << latency << '\n';
}


}  // namespace

int RunTrellis35PrefillTests() {
  TestSyntheticPrefill();
  return failures;
}

int ProfileTrellis35Prefill(std::uint64_t tokens) {
  ProfileSyntheticPrefill(tokens);
  return failures;
}

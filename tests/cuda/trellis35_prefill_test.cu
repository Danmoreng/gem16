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

enum class PrefillRoutingPattern { kUniform, kRealFixture, kOneHot, kLongTail };

const char* RoutingPatternName(PrefillRoutingPattern pattern) {
  switch (pattern) {
    case PrefillRoutingPattern::kUniform:
      return "uniform";
    case PrefillRoutingPattern::kRealFixture:
      return "real_fixture";
    case PrefillRoutingPattern::kOneHot:
      return "one_hot";
    case PrefillRoutingPattern::kLongTail:
      return "long_tail";
  }
  return "unknown";
}

void RebuildPrefillGrouping(PrefillHostRouting& routing) {
  routing.histogram.fill(0U);
  routing.prefix.fill(0U);
  for (const auto& assignment : routing.assignments) {
    ++routing.histogram[assignment.expert_id];
  }
  for (unsigned expert = 0U;
       expert < gem16::internal::kTrellis35ExpertCount; ++expert) {
    routing.prefix[expert + 1U] =
        routing.prefix[expert] + routing.histogram[expert];
  }
  auto cursors = routing.prefix;
  for (std::uint32_t original = 0U; original < routing.assignments.size();
       ++original) {
    const std::uint32_t expert = routing.assignments[original].expert_id;
    const std::uint32_t grouped = cursors[expert]++;
    routing.permutation[grouped] = original;
    routing.inverse[original] = grouped;
  }
}

PrefillHostRouting MakeWp12Routing(std::uint64_t tokens,
                                   PrefillRoutingPattern pattern) {
  static const std::vector<std::uint32_t> kRealFixture{
      3U,  9U,  14U, 22U, 31U, 47U, 55U, 64U,
      70U, 75U, 83U, 91U, 101U, 110U, 117U, 126U};
  PrefillHostRouting routing = MakePrefillRouting(
      tokens, pattern == PrefillRoutingPattern::kRealFixture
                  ? kRealFixture
                  : SequentialExpertOrder());
  if (pattern == PrefillRoutingPattern::kOneHot) {
    for (auto& assignment : routing.assignments) assignment.expert_id = 0U;
    RebuildPrefillGrouping(routing);
  } else if (pattern == PrefillRoutingPattern::kLongTail) {
    for (std::uint64_t original = 0U; original < routing.assignments.size();
         ++original) {
      const unsigned slot = static_cast<unsigned>(
          original % gem16::internal::kTrellis35M1TopK);
      routing.assignments[original].expert_id =
          slot < 4U
              ? 0U
              : static_cast<std::uint16_t>(
                    1U + ((original * 29U + slot * 11U) %
                          (gem16::internal::kTrellis35ExpertCount - 1U)));
    }
    RebuildPrefillGrouping(routing);
  }
  return routing;
}

void RunWp12KernelAb(
    const gem16::internal::Trellis35DeviceLayerBinding& layer,
    DeviceBuffer<float>& input, const PrefillHostRouting& routing,
    std::uint64_t tokens, const char* rate_name,
    PrefillRoutingPattern pattern) {
  PrefillStorage legacy(tokens);
  PrefillStorage grouped(tokens);
  CHECK(UploadPrefillRouting(legacy, routing));
  CHECK(UploadPrefillRouting(grouped, routing));
  const float legacy_ms = RunFullPrefill(
      layer, input, legacy, 1U, false,
      gem16::internal::Trellis35PrefillKernelMode::kLegacyM4,
      gem16::internal::Trellis35PrefillTransformMode::kWarpH128,
      gem16::internal::Trellis35PrefillOutputMode::kLoopN128);
  const float grouped_ms = RunFullPrefill(
      layer, input, grouped, 1U, false,
      gem16::internal::Trellis35PrefillKernelMode::kGroupedM32,
      gem16::internal::Trellis35PrefillTransformMode::kWarpH128,
      gem16::internal::Trellis35PrefillOutputMode::kLoopN128);
  std::vector<float> legacy_experts(legacy.expert_down.elements());
  std::vector<float> grouped_experts(grouped.expert_down.elements());
  std::vector<float> legacy_reduced(legacy.token_hidden.elements());
  std::vector<float> grouped_reduced(grouped.token_hidden.elements());
  CHECK(CudaOk(cudaMemcpy(legacy_experts.data(), legacy.expert_down.get(),
                          legacy.expert_down.bytes(), cudaMemcpyDeviceToHost),
               "download WP12 legacy experts"));
  CHECK(CudaOk(cudaMemcpy(grouped_experts.data(), grouped.expert_down.get(),
                          grouped.expert_down.bytes(), cudaMemcpyDeviceToHost),
               "download WP12 grouped experts"));
  CHECK(CudaOk(cudaMemcpy(legacy_reduced.data(), legacy.token_hidden.get(),
                          legacy.token_hidden.bytes(), cudaMemcpyDeviceToHost),
               "download WP12 legacy reduction"));
  CHECK(CudaOk(cudaMemcpy(grouped_reduced.data(), grouped.token_hidden.get(),
                          grouped.token_hidden.bytes(), cudaMemcpyDeviceToHost),
               "download WP12 grouped reduction"));
  const std::string prefix = std::string("WP12 ") + rate_name + " " +
                             RoutingPatternName(pattern) + " T=" +
                             std::to_string(tokens);
  Compare(legacy_experts, grouped_experts, 0.0F, 0.0F,
          (prefix + " expert parity").c_str());
  Compare(legacy_reduced, grouped_reduced, 0.0F, 0.0F,
          (prefix + " reduction parity").c_str());
  std::cout << prefix << " legacy_ms=" << legacy_ms
            << " grouped_m32_ms=" << grouped_ms << '\n';
}

void TestWp12NumericalMatrix() {
  constexpr std::array<std::uint64_t, 14> kTokens{
      1U, 2U, 3U, 4U, 8U, 15U, 16U, 17U, 31U, 32U, 33U, 128U, 512U,
      1024U};
  constexpr std::array<PrefillRoutingPattern, 4> kPatterns{
      PrefillRoutingPattern::kUniform, PrefillRoutingPattern::kRealFixture,
      PrefillRoutingPattern::kOneHot, PrefillRoutingPattern::kLongTail};
  struct RateCase {
    std::uint16_t forced_rate;
    const char* name;
  };
  constexpr std::array<RateCase, 3> kRates{{{3U, "K3"},
                                            {4U, "K4"},
                                            {0U, "mixed"}}};
  for (const auto& rate : kRates) {
    FamilyStorage gate(gem16::internal::kTrellis35GateUpInput,
                       gem16::internal::kTrellis35GateUpOutput, 1201U,
                       rate.forced_rate);
    FamilyStorage down(gem16::internal::kTrellis35DownInput,
                       gem16::internal::kTrellis35DownOutput, 1301U,
                       rate.forced_rate);
    const gem16::internal::Trellis35DeviceLayerBinding layer{gate.binding,
                                                             down.binding};
    for (const std::uint64_t tokens : kTokens) {
      DeviceBuffer<float> input(
          tokens * gem16::internal::kTrellis35GateUpInput);
      std::vector<float> host_input(input.elements());
      for (std::uint64_t index = 0U; index < host_input.size(); ++index) {
        host_input[index] =
            std::sin(static_cast<float>(index * 17U + tokens * 5U + 3U) *
                     0.001953125F) *
            1.0e-4F;
      }
      CHECK(Upload(input, host_input, "upload WP12 matrix input"));
      for (const PrefillRoutingPattern pattern : kPatterns) {
        const auto routing = MakeWp12Routing(tokens, pattern);
        RunWp12KernelAb(layer, input, routing, tokens, rate.name, pattern);
      }
    }
  }
}

void RunWp14OutputAb(
    const gem16::internal::Trellis35DeviceLayerBinding& layer,
    DeviceBuffer<float>& input, const PrefillHostRouting& routing,
    std::uint64_t tokens, const char* rate_name,
    PrefillRoutingPattern pattern) {
  PrefillStorage loop(tokens);
  PrefillStorage fused(tokens);
  CHECK(UploadPrefillRouting(loop, routing));
  CHECK(UploadPrefillRouting(fused, routing));
  const float loop_ms = RunFullPrefill(
      layer, input, loop, 1U, false,
      gem16::internal::Trellis35PrefillKernelMode::kGroupedM32,
      gem16::internal::Trellis35PrefillTransformMode::kWarpH128,
      gem16::internal::Trellis35PrefillOutputMode::kLoopN128);
  const float fused_ms = RunFullPrefill(
      layer, input, fused, 1U, false,
      gem16::internal::Trellis35PrefillKernelMode::kGroupedM32,
      gem16::internal::Trellis35PrefillTransformMode::kWarpH128,
      gem16::internal::Trellis35PrefillOutputMode::kFusedN128);
  std::vector<float> loop_experts(loop.expert_down.elements());
  std::vector<float> fused_experts(fused.expert_down.elements());
  std::vector<float> loop_reduced(loop.token_hidden.elements());
  std::vector<float> fused_reduced(fused.token_hidden.elements());
  CHECK(CudaOk(cudaMemcpy(loop_experts.data(), loop.expert_down.get(),
                          loop.expert_down.bytes(), cudaMemcpyDeviceToHost),
               "download WP14 loop experts"));
  CHECK(CudaOk(cudaMemcpy(fused_experts.data(), fused.expert_down.get(),
                          fused.expert_down.bytes(), cudaMemcpyDeviceToHost),
               "download WP14 fused experts"));
  CHECK(CudaOk(cudaMemcpy(loop_reduced.data(), loop.token_hidden.get(),
                          loop.token_hidden.bytes(), cudaMemcpyDeviceToHost),
               "download WP14 loop reduction"));
  CHECK(CudaOk(cudaMemcpy(fused_reduced.data(), fused.token_hidden.get(),
                          fused.token_hidden.bytes(), cudaMemcpyDeviceToHost),
               "download WP14 fused reduction"));
  const std::string prefix = std::string("WP14 ") + rate_name + " " +
                             RoutingPatternName(pattern) + " T=" +
                             std::to_string(tokens);
  Compare(loop_experts, fused_experts, 0.0F, 0.0F,
          (prefix + " expert parity").c_str());
  Compare(loop_reduced, fused_reduced, 0.0F, 0.0F,
          (prefix + " reduction parity").c_str());
  std::cout << prefix << " loop_n128_ms=" << loop_ms
            << " fused_n128_ms=" << fused_ms << '\n';
}

void TestWp14OutputMatrix() {
  constexpr std::array<std::uint64_t, 14> kTokens{
      1U, 2U, 3U, 4U, 8U, 15U, 16U, 17U, 31U, 32U, 33U, 128U, 512U,
      1024U};
  constexpr std::array<PrefillRoutingPattern, 4> kPatterns{
      PrefillRoutingPattern::kUniform, PrefillRoutingPattern::kRealFixture,
      PrefillRoutingPattern::kOneHot, PrefillRoutingPattern::kLongTail};
  struct RateCase {
    std::uint16_t forced_rate;
    const char* name;
  };
  constexpr std::array<RateCase, 3> kRates{{{3U, "K3"},
                                            {4U, "K4"},
                                            {0U, "mixed"}}};
  for (const auto& rate : kRates) {
    FamilyStorage gate(gem16::internal::kTrellis35GateUpInput,
                       gem16::internal::kTrellis35GateUpOutput, 1409U,
                       rate.forced_rate);
    FamilyStorage down(gem16::internal::kTrellis35DownInput,
                       gem16::internal::kTrellis35DownOutput, 1451U,
                       rate.forced_rate);
    const gem16::internal::Trellis35DeviceLayerBinding layer{gate.binding,
                                                             down.binding};
    for (const std::uint64_t tokens : kTokens) {
      DeviceBuffer<float> input(
          tokens * gem16::internal::kTrellis35GateUpInput);
      std::vector<float> host_input(input.elements());
      for (std::uint64_t index = 0U; index < host_input.size(); ++index) {
        host_input[index] =
            std::sin(static_cast<float>(index * 23U + tokens * 7U + 5U) *
                     0.001953125F) *
            1.0e-4F;
      }
      CHECK(Upload(input, host_input, "upload WP14 matrix input"));
      for (const PrefillRoutingPattern pattern : kPatterns) {
        const auto routing = MakeWp12Routing(tokens, pattern);
        RunWp14OutputAb(layer, input, routing, tokens, rate.name, pattern);
      }
    }
  }
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

int RunTrellis35Wp12NumericalMatrix() {
  TestWp12NumericalMatrix();
  return failures;
}

int RunTrellis35Wp14OutputMatrix() {
  TestWp14OutputMatrix();
  return failures;
}

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

PrefillHostRouting MakeWp17RowRouting(std::uint32_t target_rows,
                                      PrefillRoutingPattern pattern) {
  constexpr std::uint64_t kTokens = 16U;
  PrefillHostRouting routing =
      MakePrefillRouting(kTokens, SequentialExpertOrder());
  CHECK(target_rows >= 1U && target_rows <= routing.assignments.size());
  static constexpr std::array<std::uint16_t, 16> kRealExperts{
      3U,  9U,  14U, 22U, 31U, 47U, 55U, 64U,
      70U, 75U, 83U, 91U, 101U, 110U, 117U, 126U};
  for (std::uint32_t index = 0U; index < routing.assignments.size(); ++index) {
    std::uint16_t expert = 0U;
    if (index >= target_rows) {
      const std::uint32_t tail = index - target_rows;
      switch (pattern) {
        case PrefillRoutingPattern::kUniform:
          expert = static_cast<std::uint16_t>(1U + tail % 127U);
          break;
        case PrefillRoutingPattern::kRealFixture:
          expert = kRealExperts[tail % kRealExperts.size()];
          break;
        case PrefillRoutingPattern::kOneHot:
          expert = 1U;
          break;
        case PrefillRoutingPattern::kLongTail:
          expert = static_cast<std::uint16_t>(
              1U + ((tail * 29U + (tail % 8U) * 11U) % 127U));
          break;
      }
    }
    routing.assignments[index].expert_id = expert;
  }
  RebuildPrefillGrouping(routing);
  CHECK(routing.histogram[0] == target_rows);
  return routing;
}

void RunWp17M64Ab(
    const gem16::internal::Trellis35DeviceLayerBinding& layer,
    DeviceBuffer<float>& input, const PrefillHostRouting& routing,
    std::uint32_t target_rows, const char* rate_name,
    PrefillRoutingPattern pattern) {
  constexpr std::uint64_t kTokens = 16U;
  PrefillStorage m32(kTokens);
  PrefillStorage m64(kTokens);
  CHECK(UploadPrefillRouting(m32, routing));
  CHECK(UploadPrefillRouting(m64, routing));
  const float m32_ms = RunFullPrefill(
      layer, input, m32, 1U, false,
      gem16::internal::Trellis35PrefillKernelMode::kGroupedM32,
      gem16::internal::Trellis35PrefillTransformMode::kWarpH128,
      gem16::internal::Trellis35PrefillOutputMode::kFusedN128);
  const bool capture = target_rows == 65U &&
                       pattern == PrefillRoutingPattern::kLongTail &&
                       std::string_view(rate_name) == "mixed";
  const float m64_ms = RunFullPrefill(
      layer, input, m64, 1U, capture,
      gem16::internal::Trellis35PrefillKernelMode::kGroupedM64Hybrid,
      gem16::internal::Trellis35PrefillTransformMode::kWarpH128,
      gem16::internal::Trellis35PrefillOutputMode::kFusedN128);
  std::vector<float> m32_experts(m32.expert_down.elements());
  std::vector<float> m64_experts(m64.expert_down.elements());
  std::vector<float> m32_reduced(m32.token_hidden.elements());
  std::vector<float> m64_reduced(m64.token_hidden.elements());
  CHECK(CudaOk(cudaMemcpy(m32_experts.data(), m32.expert_down.get(),
                          m32.expert_down.bytes(), cudaMemcpyDeviceToHost),
               "download WP17 M32 experts"));
  CHECK(CudaOk(cudaMemcpy(m64_experts.data(), m64.expert_down.get(),
                          m64.expert_down.bytes(), cudaMemcpyDeviceToHost),
               "download WP17 M64 experts"));
  CHECK(CudaOk(cudaMemcpy(m32_reduced.data(), m32.token_hidden.get(),
                          m32.token_hidden.bytes(), cudaMemcpyDeviceToHost),
               "download WP17 M32 reduction"));
  CHECK(CudaOk(cudaMemcpy(m64_reduced.data(), m64.token_hidden.get(),
                          m64.token_hidden.bytes(), cudaMemcpyDeviceToHost),
               "download WP17 M64 reduction"));
  const std::string prefix =
      std::string("WP17 ") + rate_name + " " + RoutingPatternName(pattern) +
      " rows=" + std::to_string(target_rows);
  Compare(m32_experts, m64_experts, 0.0F, 0.0F,
          (prefix + " expert parity").c_str());
  Compare(m32_reduced, m64_reduced, 0.0F, 0.0F,
          (prefix + " reduction parity").c_str());
  std::cout << prefix << " m32_ms=" << m32_ms << " m64_ms=" << m64_ms
            << " graph_replay=" << (capture ? "true" : "false") << '\n';
}

void TestWp17M64Matrix() {
  constexpr std::array<std::uint32_t, 22> kRows{
      1U,  2U,  3U,  15U, 16U, 17U, 31U, 32U, 33U,
      47U, 48U, 63U, 64U, 65U, 79U, 80U, 81U, 95U,
      96U, 97U, 127U, 128U};
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
  constexpr std::uint64_t kTokens = 16U;
  DeviceBuffer<float> input(kTokens * gem16::internal::kTrellis35GateUpInput);
  std::vector<float> host_input(input.elements());
  for (std::uint64_t index = 0U; index < host_input.size(); ++index) {
    host_input[index] =
        std::sin(static_cast<float>(index * 31U + 7U) * 0.001953125F) *
        1.0e-4F;
  }
  CHECK(Upload(input, host_input, "upload WP17 matrix input"));
  for (const auto& rate : kRates) {
    FamilyStorage gate(gem16::internal::kTrellis35GateUpInput,
                       gem16::internal::kTrellis35GateUpOutput, 1709U,
                       rate.forced_rate);
    FamilyStorage down(gem16::internal::kTrellis35DownInput,
                       gem16::internal::kTrellis35DownOutput, 1753U,
                       rate.forced_rate);
    const gem16::internal::Trellis35DeviceLayerBinding layer{gate.binding,
                                                             down.binding};
    for (const std::uint32_t rows : kRows) {
      for (const PrefillRoutingPattern pattern : kPatterns) {
        const auto routing = MakeWp17RowRouting(rows, pattern);
        RunWp17M64Ab(layer, input, routing, rows, rate.name, pattern);
      }
    }
  }
}

void TestWp17M64Smoke() {
  constexpr std::uint64_t kTokens = 16U;
  DeviceBuffer<float> input(kTokens * gem16::internal::kTrellis35GateUpInput);
  std::vector<float> host_input(input.elements());
  for (std::uint64_t index = 0U; index < host_input.size(); ++index) {
    host_input[index] =
        std::sin(static_cast<float>(index * 37U + 11U) * 0.001953125F) *
        1.0e-4F;
  }
  CHECK(Upload(input, host_input, "upload WP17 smoke input"));
  FamilyStorage gate(gem16::internal::kTrellis35GateUpInput,
                     gem16::internal::kTrellis35GateUpOutput, 1777U);
  FamilyStorage down(gem16::internal::kTrellis35DownInput,
                     gem16::internal::kTrellis35DownOutput, 1789U);
  const gem16::internal::Trellis35DeviceLayerBinding layer{gate.binding,
                                                           down.binding};
  const auto routing =
      MakeWp17RowRouting(65U, PrefillRoutingPattern::kLongTail);
  RunWp17M64Ab(layer, input, routing, 65U, "mixed",
               PrefillRoutingPattern::kLongTail);
}

std::uint16_t HostBf16Bits(float value) {
  std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

float TimeWp19GeluDown(
    const DeviceBuffer<std::uint16_t>& gate_up,
    DeviceBuffer<std::uint16_t>& product,
    const gem16::internal::Trellis35DeviceFamilyBinding& down,
    const DeviceBuffer<gem16::internal::Gemma4MoePrefillAssignment>&
        assignments,
    DeviceBuffer<std::uint8_t>& output, DeviceBuffer<float>& scales,
    gem16::internal::Trellis35PrefillGeluDownMode mode,
    unsigned iterations) {
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  CHECK(CudaOk(cudaEventCreate(&begin), "create WP19 begin event"));
  CHECK(CudaOk(cudaEventCreate(&end), "create WP19 end event"));
  auto launch = [&]() {
    return gem16::internal::
        LaunchTrellis35GatedGeluDownTransformQuantizeBf16(
            gate_up.get(), product.get(), down, assignments.get(),
            output.get(), scales.get(), assignments.elements(), mode,
            nullptr);
  };
  CHECK(launch().ok());
  CHECK(CudaOk(cudaDeviceSynchronize(), "warm WP19 operator"));
  CHECK(CudaOk(cudaEventRecord(begin), "record WP19 begin event"));
  for (unsigned iteration = 0U; iteration < iterations; ++iteration) {
    CHECK(launch().ok());
  }
  CHECK(CudaOk(cudaEventRecord(end), "record WP19 end event"));
  CHECK(CudaOk(cudaEventSynchronize(end), "synchronize WP19 operator"));
  float milliseconds = 0.0F;
  CHECK(CudaOk(cudaEventElapsedTime(&milliseconds, begin, end),
               "measure WP19 operator"));
  if (begin != nullptr) (void)cudaEventDestroy(begin);
  if (end != nullptr) (void)cudaEventDestroy(end);
  return milliseconds / static_cast<float>(iterations);
}

void TestWp19GeluDownOracle() {
  constexpr std::uint64_t kAssignments = 128U;
  constexpr unsigned kIterations = 50U;
  FamilyStorage down(gem16::internal::kTrellis35DownInput,
                     gem16::internal::kTrellis35DownOutput, 1901U);
  DeviceBuffer<std::uint16_t> gate_up(
      kAssignments * gem16::internal::kTrellis35GateUpOutput);
  DeviceBuffer<std::uint16_t> rollback_product(
      kAssignments * gem16::internal::kTrellis35ExpertIntermediate);
  DeviceBuffer<std::uint16_t> fused_product(
      kAssignments * gem16::internal::kTrellis35ExpertIntermediate);
  DeviceBuffer<gem16::internal::Gemma4MoePrefillAssignment> assignments(
      kAssignments);
  DeviceBuffer<std::uint8_t> rollback_output(
      kAssignments * gem16::internal::kTrellis35DownInput);
  DeviceBuffer<std::uint8_t> fused_output(
      kAssignments * gem16::internal::kTrellis35DownInput);
  DeviceBuffer<float> rollback_scales(kAssignments);
  DeviceBuffer<float> fused_scales(kAssignments);

  std::vector<std::uint16_t> host_gate_up(gate_up.elements());
  for (std::uint64_t assignment = 0U; assignment < kAssignments;
       ++assignment) {
    for (std::uint64_t index = 0U;
         index < gem16::internal::kTrellis35ExpertIntermediate; ++index) {
      const float phase = static_cast<float>(assignment * 704U + index);
      host_gate_up[assignment * gem16::internal::kTrellis35GateUpOutput +
                   index] = HostBf16Bits(std::sin(phase * 0.0078125F) * 3.0F);
      host_gate_up[assignment * gem16::internal::kTrellis35GateUpOutput +
                   gem16::internal::kTrellis35ExpertIntermediate + index] =
          HostBf16Bits(std::cos((phase + 17.0F) * 0.005859375F) * 2.0F);
    }
  }
  std::vector<gem16::internal::Gemma4MoePrefillAssignment>
      host_assignments(kAssignments);
  for (std::uint64_t index = 0U; index < kAssignments; ++index) {
    host_assignments[index] = {
        static_cast<std::uint16_t>(
            (index * 17U + 3U) % gem16::internal::kTrellis35ExpertCount),
        static_cast<std::uint16_t>(index % gem16::internal::kTrellis35M1TopK),
        static_cast<std::uint32_t>(index /
                                   gem16::internal::kTrellis35M1TopK),
        0.125F};
  }
  CHECK(Upload(gate_up, host_gate_up, "upload WP19 Gate+Up BF16"));
  CHECK(Upload(assignments, host_assignments, "upload WP19 assignments"));
  CHECK(CudaOk(cudaMemset(fused_product.get(), 0xa5, fused_product.bytes()),
               "initialize WP19 unused product sentinel"));

  const float rollback_ms = TimeWp19GeluDown(
      gate_up, rollback_product, down.binding, assignments, rollback_output,
      rollback_scales, gem16::internal::Trellis35PrefillGeluDownMode::kTwoKernel,
      kIterations);
  const float fused_ms = TimeWp19GeluDown(
      gate_up, fused_product, down.binding, assignments, fused_output,
      fused_scales,
      gem16::internal::Trellis35PrefillGeluDownMode::kFusedTransformQuantize,
      kIterations);

  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CHECK(CudaOk(cudaStreamCreate(&stream), "create WP19 graph stream"));
  CHECK(CudaOk(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
               "begin WP19 graph capture"));
  CHECK(gem16::internal::LaunchTrellis35GatedGeluDownTransformQuantizeBf16(
            gate_up.get(), fused_product.get(), down.binding,
            assignments.get(), fused_output.get(), fused_scales.get(),
            kAssignments,
            gem16::internal::Trellis35PrefillGeluDownMode::
                kFusedTransformQuantize,
            stream)
            .ok());
  CHECK(CudaOk(cudaStreamEndCapture(stream, &graph),
               "end WP19 graph capture"));
  CHECK(CudaOk(cudaGraphInstantiate(&executable, graph, 0U),
               "instantiate WP19 graph"));
  CHECK(CudaOk(cudaGraphLaunch(executable, stream),
               "launch WP19 graph first"));
  CHECK(CudaOk(cudaGraphLaunch(executable, stream),
               "launch WP19 graph replay"));
  CHECK(CudaOk(cudaStreamSynchronize(stream), "synchronize WP19 graph"));
  if (executable != nullptr) (void)cudaGraphExecDestroy(executable);
  if (graph != nullptr) (void)cudaGraphDestroy(graph);
  if (stream != nullptr) (void)cudaStreamDestroy(stream);

  std::vector<std::uint8_t> host_rollback_output(rollback_output.elements());
  std::vector<std::uint8_t> host_fused_output(fused_output.elements());
  std::vector<float> host_rollback_scales(rollback_scales.elements());
  std::vector<float> host_fused_scales(fused_scales.elements());
  std::vector<std::uint16_t> host_fused_product(fused_product.elements());
  CHECK(CudaOk(cudaMemcpy(host_rollback_output.data(), rollback_output.get(),
                          rollback_output.bytes(), cudaMemcpyDeviceToHost),
               "download WP19 rollback output"));
  CHECK(CudaOk(cudaMemcpy(host_fused_output.data(), fused_output.get(),
                          fused_output.bytes(), cudaMemcpyDeviceToHost),
               "download WP19 fused output"));
  CHECK(CudaOk(cudaMemcpy(host_rollback_scales.data(), rollback_scales.get(),
                          rollback_scales.bytes(), cudaMemcpyDeviceToHost),
               "download WP19 rollback scales"));
  CHECK(CudaOk(cudaMemcpy(host_fused_scales.data(), fused_scales.get(),
                          fused_scales.bytes(), cudaMemcpyDeviceToHost),
               "download WP19 fused scales"));
  CHECK(CudaOk(cudaMemcpy(host_fused_product.data(), fused_product.get(),
                          fused_product.bytes(), cudaMemcpyDeviceToHost),
               "download WP19 unused product sentinel"));
  const auto scale_bits = [](const std::vector<float>& values) {
    std::vector<std::uint32_t> bits(values.size());
    std::transform(values.begin(), values.end(), bits.begin(),
                   [](float value) {
                     return std::bit_cast<std::uint32_t>(value);
                   });
    return bits;
  };
  CHECK(host_rollback_output == host_fused_output);
  CHECK(scale_bits(host_rollback_scales) == scale_bits(host_fused_scales));
  CHECK(std::all_of(host_fused_product.begin(), host_fused_product.end(),
                    [](std::uint16_t value) { return value == 0xa5a5U; }));
  std::cout << "WP19 Gate+Up->Down byte_exact=true scale_exact=true "
               "product_buffer_untouched=true assignments="
            << kAssignments << " rollback_ms=" << rollback_ms
            << " fused_ms=" << fused_ms << '\n';
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

int RunTrellis35Wp17M64Matrix() {
  TestWp17M64Matrix();
  return failures;
}

int RunTrellis35Wp17M64Smoke() {
  TestWp17M64Smoke();
  return failures;
}

int RunTrellis35Wp19GeluDownOracle() {
  TestWp19GeluDownOracle();
  return failures;
}

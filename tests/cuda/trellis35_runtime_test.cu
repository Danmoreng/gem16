#include "trellis35_test_support.h"

namespace {

void ProfileRealPrefill(const std::string& checkpoint, std::uint64_t tokens) {
  auto artifact =
      gem16::internal::Gemma4Moe26BTrellis35DeviceArtifact::Load(checkpoint);
  CHECK(artifact.ok());
  if (!artifact.ok()) {
    std::cerr << artifact.status().message() << '\n';
    return;
  }
  DeviceBuffer<float> input(tokens * gem16::internal::kTrellis35GateUpInput);
  std::vector<float> host_input(input.elements());
  for (std::uint64_t index = 0U; index < host_input.size(); ++index) {
    host_input[index] =
        std::sin(static_cast<float>(index * 19U + 11U) * 0.001953125F) *
        0.03125F;
  }
  CHECK(Upload(input, host_input, "upload real profile prefill input"));
  PrefillStorage storage(tokens);
  const auto routing = MakePrefillRouting(tokens, SequentialExpertOrder());
  CHECK(UploadPrefillRouting(storage, routing));
  const float latency = RunFullPrefill(artifact.value().layers()[0], input,
                                       storage, 1U, false);
  std::cout << "profile real layer-0 W4A8 prefill tokens=" << tokens
            << " assignments=" << tokens * gem16::internal::kTrellis35M1TopK
            << " latency_ms=" << latency
            << " checkpoint_sha256="
            << artifact.value().stats().checkpoint_content_sha256 << '\n';
}

void CompareRealProjection(
    const gem16::internal::Trellis35DeviceFamilyBinding& family,
    std::uint64_t input_elements, std::uint64_t output_elements,
    const std::array<std::uint32_t, gem16::internal::kTrellis35M1TopK>& ids,
    const DeviceBuffer<std::uint8_t>& activation,
    const DeviceBuffer<float>& scales, const DeviceBuffer<float>& native_output,
    const char* description) {
  DeviceBuffer<float> reference(output_elements);
  std::vector<float> host_reference(output_elements);
  std::vector<float> host_native(output_elements);
  for (unsigned slot : {0U, 1U}) {
    const auto status =
        gem16::internal::LaunchTrellis35ReferenceW4A8ProjectionM1(
            activation.get() + slot * input_elements, scales.get() + slot,
            family, ids[slot], reference.get(), input_elements,
            output_elements, nullptr);
    CHECK(status.ok());
    CHECK(CudaOk(cudaDeviceSynchronize(), "synchronize real projection"));
    CHECK(CudaOk(cudaMemcpy(host_reference.data(), reference.get(),
                            reference.bytes(), cudaMemcpyDeviceToHost),
                 "download real reference projection"));
    CHECK(CudaOk(cudaMemcpy(
                     host_native.data(),
                     native_output.get() + slot * output_elements,
                     reference.bytes(), cudaMemcpyDeviceToHost),
                 "download real native projection"));
    const std::string label =
        std::string(description) + " slot=" + std::to_string(slot) +
        " K" + std::to_string(family.rate_map[ids[slot]]);
    Compare(host_reference, host_native, 4.0e-3F, 8.0e-5F, label.c_str());
  }
}

void TestRealCheckpoint(const std::string& checkpoint) {
  const auto format =
      gem16::internal::ResolveValidatedGemma4Moe26BRoutedExpertFormat(
          checkpoint);
  CHECK(format.ok());
  if (!format.ok()) {
    std::cerr << format.status().message() << '\n';
    return;
  }
  CHECK(format.value() ==
        gem16::internal::Gemma4Moe26BRoutedExpertFormat::kTrellis35);
  auto artifact =
      gem16::internal::Gemma4Moe26BTrellis35DeviceArtifact::Load(checkpoint);
  CHECK(artifact.ok());
  if (!artifact.ok()) {
    std::cerr << artifact.status().message() << '\n';
    return;
  }
  const auto& layer = artifact.value().layers()[0];
  std::array<std::uint32_t, gem16::internal::kTrellis35M1TopK> host_ids{};
  std::vector<std::uint32_t> k3;
  std::vector<std::uint32_t> k4;
  for (std::uint32_t expert = 0U;
       expert < gem16::internal::kTrellis35ExpertCount; ++expert) {
    (layer.gate_up.rate_map[expert] == 3U ? k3 : k4).push_back(expert);
  }
  CHECK(!k3.empty() && !k4.empty());
  host_ids[0] = k3[0];
  host_ids[1] = k4[0];
  for (unsigned slot = 2U; slot < host_ids.size(); ++slot) {
    host_ids[slot] = (slot & 1U) == 0U ? k3[slot / 2U] : k4[slot / 2U];
  }
  DeviceBuffer<float> input(gem16::internal::kTrellis35GateUpInput);
  DeviceBuffer<std::uint32_t> ids(gem16::internal::kTrellis35M1TopK);
  DeviceBuffer<float> weights(gem16::internal::kTrellis35M1TopK);
  DeviceBuffer<float> output(gem16::internal::kTrellis35DownOutput);
  M1Storage storage;
  std::vector<float> host_input(input.elements());
  for (std::uint64_t index = 0U; index < host_input.size(); ++index) {
    host_input[index] =
        std::sin(static_cast<float>(index * 13U + 7U) * 0.001953125F) *
        0.03125F;
  }
  const std::array<float, gem16::internal::kTrellis35M1TopK> host_weights{
      0.21F, 0.18F, 0.15F, 0.13F, 0.11F, 0.09F, 0.07F, 0.06F};
  if (!Upload(input, host_input, "upload real M1 input") ||
      !Upload(ids, host_ids, "upload real M1 IDs") ||
      !Upload(weights, host_weights, "upload real M1 weights")) {
    return;
  }
  const float latency =
      RunFullM1(layer, input, ids, weights, storage, output, 5U, false);
  CheckFiniteAndIds(output, ids, host_ids, "real layer-0 full M1");
  CheckSlotOrderedReduction(storage.down_output, host_weights, output,
                            "real layer-0 full M1");
  CompareRealProjection(layer.gate_up, gem16::internal::kTrellis35GateUpInput,
                        gem16::internal::kTrellis35GateUpOutput, host_ids,
                        storage.gate_input_fp8, storage.gate_scales,
                        storage.gate_transformed, "real Gate+Up parity");
  CompareRealProjection(layer.down, gem16::internal::kTrellis35DownInput,
                        gem16::internal::kTrellis35DownOutput, host_ids,
                        storage.down_input_fp8, storage.down_scales,
                        storage.down_transformed, "real Down parity");

  std::uint64_t payload_bytes = 0U;
  for (const std::uint32_t expert : host_ids) {
    payload_bytes +=
        gem16::internal::kTrellis35GateUpInput *
        gem16::internal::kTrellis35GateUpOutput *
        layer.gate_up.rate_map[expert] / 8U;
    payload_bytes += gem16::internal::kTrellis35DownInput *
                     gem16::internal::kTrellis35DownOutput *
                     layer.down.rate_map[expert] / 8U;
  }
  const std::uint64_t sidecar_bytes =
      host_ids.size() *
      (gem16::internal::kTrellis35GateUpInput +
       gem16::internal::kTrellis35GateUpOutput +
       gem16::internal::kTrellis35DownInput +
       gem16::internal::kTrellis35DownOutput) *
      sizeof(std::uint16_t);
  std::cout << "real layer-0 selected_payload_bytes=" << payload_bytes
            << " selected_sidecar_bytes=" << sidecar_bytes
            << " descriptors_read=" << host_ids.size() * 2U
            << " latency_ms=" << latency
            << " checkpoint_sha256="
            << artifact.value().stats().checkpoint_content_sha256 << '\n';

  std::vector<std::uint32_t> expert_order;
  expert_order.reserve(gem16::internal::kTrellis35ExpertCount);
  for (unsigned index = 0U; index < 12U; ++index) {
    expert_order.push_back(k3[index]);
    expert_order.push_back(k4[index]);
  }
  const auto t3_ids = RemapT3Routes(MakeT3RoutesWithUnionSize(16U),
                                    expert_order);
  std::array<float, gem16::internal::kTrellis35T3Assignments> t3_weights{};
  for (unsigned row = 0U; row < gem16::internal::kTrellis35T3Rows; ++row) {
    std::copy(host_weights.begin(), host_weights.end(),
              t3_weights.begin() + row * gem16::internal::kTrellis35M1TopK);
  }
  DeviceBuffer<float> t3_input(gem16::internal::kTrellis35T3Rows *
                               gem16::internal::kTrellis35GateUpInput);
  std::vector<float> host_t3_input(t3_input.elements());
  for (unsigned row = 0U; row < gem16::internal::kTrellis35T3Rows; ++row) {
    for (std::uint64_t index = 0U;
         index < gem16::internal::kTrellis35GateUpInput; ++index) {
      host_t3_input[static_cast<std::uint64_t>(row) *
                        gem16::internal::kTrellis35GateUpInput +
                    index] =
          std::sin(static_cast<float>(index * 13U + row * 41U + 7U) *
                   0.001953125F) *
          0.03125F;
    }
  }
  CHECK(Upload(t3_input, host_t3_input, "upload real T3 input"));
  (void)RunT3Scenario(layer, t3_input, t3_ids, t3_weights,
                      "real layer-0 T3 typical-overlap", true);

  std::array<bool, gem16::internal::kTrellis35ExpertCount> seen{};
  std::uint64_t t3_assignment_payload_bytes = 0U;
  std::uint64_t t3_unique_payload_bytes = 0U;
  for (const std::uint32_t expert : t3_ids) {
    const std::uint64_t expert_bytes =
        gem16::internal::kTrellis35GateUpInput *
            gem16::internal::kTrellis35GateUpOutput *
            layer.gate_up.rate_map[expert] / 8U +
        gem16::internal::kTrellis35DownInput *
            gem16::internal::kTrellis35DownOutput *
            layer.down.rate_map[expert] / 8U;
    t3_assignment_payload_bytes += expert_bytes;
    if (!seen[expert]) {
      seen[expert] = true;
      t3_unique_payload_bytes += expert_bytes;
    }
  }
  std::cout << "real layer-0 T3 assignment_payload_bytes="
            << t3_assignment_payload_bytes
            << " grouped_unique_payload_bytes=" << t3_unique_payload_bytes
            << " payload_bytes_avoided="
            << t3_assignment_payload_bytes - t3_unique_payload_bytes << '\n';
  BenchmarkRetainedOverlapHistogram(layer, t3_input, t3_weights,
                                    expert_order);

  constexpr std::uint64_t kPrefillTokens = 4U;
  DeviceBuffer<float> prefill_input(
      kPrefillTokens * gem16::internal::kTrellis35GateUpInput);
  std::vector<float> host_prefill_input(prefill_input.elements());
  for (std::uint64_t index = 0U; index < host_prefill_input.size(); ++index) {
    host_prefill_input[index] =
        std::sin(static_cast<float>(index * 17U + 5U) * 0.001953125F) *
        0.03125F;
  }
  CHECK(Upload(prefill_input, host_prefill_input,
               "upload real prefill input"));
  PrefillStorage prefill_storage(kPrefillTokens);
  const auto prefill_routing =
      MakePrefillRouting(kPrefillTokens, expert_order);
  (void)RunPrefillScenario(layer, prefill_input, prefill_storage,
                           prefill_routing, "real layer-0 W4A8 prefill",
                           true, true);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 3 && std::string(argv[1]) == "--profile-t3") {
    const unsigned unique_experts =
        static_cast<unsigned>(std::stoul(argv[2]));
    if (unique_experts < gem16::internal::kTrellis35M1TopK ||
        unique_experts > gem16::internal::kTrellis35T3Assignments) {
      std::cerr << "profile T3 union size must be in [8, 24]\n";
      return 2;
    }
    return ProfileTrellis35T3(unique_experts) == 0 ? 0 : 1;
  }
  if (argc == 3 && std::string(argv[1]) == "--profile-prefill") {
    const std::uint64_t tokens = std::stoull(argv[2]);
    if (tokens == 0U || tokens > 1024U) {
      std::cerr << "prefill profile tokens must be in [1, 1024]\n";
      return 2;
    }
    return ProfileTrellis35Prefill(tokens) == 0 ? 0 : 1;
  }
  if (argc == 4 && std::string(argv[1]) == "--profile-prefill-checkpoint") {
    const std::uint64_t tokens = std::stoull(argv[3]);
    if (tokens == 0U || tokens > 1024U) {
      std::cerr << "prefill profile tokens must be in [1, 1024]\n";
      return 2;
    }
    ProfileRealPrefill(argv[2], tokens);
    return failures == 0 ? 0 : 1;
  }
  int suite_failures = 0;
  suite_failures += RunTrellis35CodecTests();
  suite_failures += RunTrellis35TransformTests();
  suite_failures += RunTrellis35M1Tests();
  suite_failures += RunTrellis35T3Tests();
  suite_failures += RunTrellis35PrefillTests();
  if (argc == 3 && std::string(argv[1]) == "--checkpoint") {
    TestRealCheckpoint(argv[2]);
  } else if (argc != 1) {
    std::cerr << "usage: gem16-cuda-trellis35-tests "
                 "[--checkpoint PATH | --profile-t3 UNIQUE | "
                 "--profile-prefill TOKENS | "
                 "--profile-prefill-checkpoint PATH TOKENS]\n";
    return 2;
  }
  suite_failures += failures;
  if (suite_failures == 0) {
    std::cout << "trellis35_m1_test_pass\n";
  }
  return suite_failures == 0 ? 0 : 1;
}

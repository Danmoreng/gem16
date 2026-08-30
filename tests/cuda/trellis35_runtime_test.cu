#include "trellis35_test_support.h"

#include "cuda/fp8/sm120.h"

namespace {

void PrintPrefillScheduleTelemetry(const PrefillHostRouting& routing) {
  std::uint64_t active_experts = 0U;
  std::uint64_t m32_schedule_count = 0U;
  std::uint64_t m64_schedule_count = 0U;
  std::uint64_t m64_full_tiles = 0U;
  std::uint64_t m64_33_63_tails = 0U;
  std::uint64_t m32_le32_tails = 0U;
  std::cout << "profile prefill expert_rows=[";
  bool first = true;
  for (std::uint32_t expert = 0U; expert < routing.histogram.size();
       ++expert) {
    const std::uint32_t rows = routing.histogram[expert];
    if (rows == 0U) continue;
    if (!first) std::cout << ',';
    std::cout << expert << ':' << rows;
    first = false;
    ++active_experts;
    m32_schedule_count += (rows + 31U) / 32U;
    m64_full_tiles += rows / 64U;
    const std::uint32_t remainder = rows % 64U;
    m64_schedule_count += rows / 64U + (remainder > 32U ? 1U : 0U);
    m64_33_63_tails += remainder > 32U ? 1U : 0U;
    m32_le32_tails += remainder != 0U && remainder <= 32U ? 1U : 0U;
  }
  const std::uint64_t assignment_count = routing.assignments.size();
  const std::uint64_t active_expert_upper_bound =
      std::min<std::uint64_t>(assignment_count,
                              gem16::internal::kTrellis35ExpertCount);
  const std::uint64_t m32_launched_blocks =
      (assignment_count + 31U *
                              active_expert_upper_bound) /
      32U;
  const std::uint64_t m64_launched_blocks =
      std::max<std::uint64_t>(
          1U, (assignment_count + 31U * active_expert_upper_bound) / 64U);
  const std::uint64_t m64_parent_launched_blocks =
      (assignment_count + 63U * active_expert_upper_bound) / 64U;
  std::cout << "] assignment_count=" << assignment_count
            << " active_experts=" << active_experts
            << " m32_schedule_count=" << m32_schedule_count
            << " m64_schedule_count=" << m64_schedule_count
            << " m64_full_tiles=" << m64_full_tiles
            << " m64_33_63_tails=" << m64_33_63_tails
            << " m32_le32_tails=" << m32_le32_tails
            << " m32_launched_blocks=" << m32_launched_blocks
            << " m64_parent_launched_blocks=" << m64_parent_launched_blocks
            << " m64_trimmed_launched_blocks=" << m64_launched_blocks << '\n';
}

void ProfileRealPrefill(
    const std::string& checkpoint, std::uint64_t tokens,
    gem16::internal::Trellis35PrefillOutputMode output_mode,
    gem16::internal::Trellis35PrefillKernelMode kernel_mode) {
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
  PrintPrefillScheduleTelemetry(routing);
  CHECK(UploadPrefillRouting(storage, routing));
  const float latency = RunFullPrefill(
      artifact.value().layers()[0], input, storage, 1U, false,
      kernel_mode,
      gem16::internal::Trellis35PrefillTransformMode::kWarpH128, output_mode);
  std::cout << "profile real layer-0 W4A8 prefill tokens=" << tokens
            << " assignments=" << tokens * gem16::internal::kTrellis35M1TopK
            << " kernel_mode="
            << (kernel_mode == gem16::internal::Trellis35PrefillKernelMode::
                                   kGroupedM64Hybrid
                    ? "m64_hybrid"
                    : "m32")
            << " latency_ms=" << latency
            << " checkpoint_sha256="
            << artifact.value().stats().checkpoint_content_sha256 << '\n';
}

template <typename Operation>
float TimeDiagnostic(Operation&& operation, unsigned iterations,
                     const char* description) {
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  CHECK(CudaOk(cudaEventCreate(&begin), "create diagnostic begin event"));
  CHECK(CudaOk(cudaEventCreate(&end), "create diagnostic end event"));
  CHECK(operation().ok());
  CHECK(CudaOk(cudaDeviceSynchronize(), "warm transient slab diagnostic"));
  CHECK(CudaOk(cudaEventRecord(begin), "record diagnostic begin event"));
  for (unsigned iteration = 0U; iteration < iterations; ++iteration) {
    const auto status = operation();
    CHECK(status.ok());
  }
  CHECK(CudaOk(cudaEventRecord(end), "record diagnostic end event"));
  CHECK(CudaOk(cudaEventSynchronize(end), description));
  float milliseconds = 0.0F;
  CHECK(CudaOk(cudaEventElapsedTime(&milliseconds, begin, end),
               "measure transient slab diagnostic"));
  if (begin != nullptr) (void)cudaEventDestroy(begin);
  if (end != nullptr) (void)cudaEventDestroy(end);
  return milliseconds / static_cast<float>(iterations);
}

void ProfileTransientE4M3Slab(const std::string& checkpoint) {
  auto artifact =
      gem16::internal::Gemma4Moe26BTrellis35DeviceArtifact::Load(checkpoint);
  CHECK(artifact.ok());
  if (!artifact.ok()) {
    std::cerr << artifact.status().message() << '\n';
    return;
  }
  const auto& layer = artifact.value().layers()[0];
  constexpr std::uint64_t kSlabRows = 128U;
  constexpr std::uint64_t kMaximumM = 64U;
  constexpr unsigned kIterations = 20U;
  DeviceBuffer<std::uint8_t> slab(
      kSlabRows * gem16::internal::kTrellis35GateUpInput);
  DeviceBuffer<std::uint16_t> slab_scales(kSlabRows);
  DeviceBuffer<float> activation(
      kMaximumM * gem16::internal::kTrellis35GateUpInput);
  DeviceBuffer<std::uint8_t> activation_e4m3(
      kMaximumM * gem16::internal::kTrellis35GateUpInput);
  DeviceBuffer<float> activation_scales(kMaximumM);
  DeviceBuffer<float> output(kMaximumM * kSlabRows);
  DeviceBuffer<float> reference(
      gem16::internal::kTrellis35DownOutput);

  std::vector<float> host_activation(activation.elements());
  for (std::uint64_t index = 0U; index < host_activation.size(); ++index) {
    host_activation[index] =
        std::sin(static_cast<float>(index * 29U + 3U) * 0.001953125F) *
        0.03125F;
  }
  CHECK(Upload(activation, host_activation,
               "upload transient slab activations"));

  const auto profile_family = [&](
                                  const char* family_name,
                                  const gem16::internal::
                                      Trellis35DeviceFamilyBinding& family,
                                  std::uint64_t input_elements,
                                  std::uint64_t output_elements) {
    std::array<std::uint32_t, 2U> representative{};
    for (std::uint32_t expert = 0U;
         expert < gem16::internal::kTrellis35ExpertCount; ++expert) {
      const std::uint16_t rate = family.rate_map[expert];
      if (rate == 3U && family.rate_map[representative[0]] != 3U) {
        representative[0] = expert;
      }
      if (rate == 4U && family.rate_map[representative[1]] != 4U) {
        representative[1] = expert;
      }
    }
    CHECK(family.rate_map[representative[0]] == 3U);
    CHECK(family.rate_map[representative[1]] == 4U);
    for (const std::uint32_t expert : representative) {
      const auto decode = [&]() {
        return gem16::internal::LaunchTrellis35DecodeE4M3SlabDiagnostic(
            family, expert, input_elements, output_elements, 0U, kSlabRows,
            slab.get(), slab_scales.get(), nullptr);
      };
      const float decode_ms =
          TimeDiagnostic(decode, kIterations, "synchronize slab decode");
      for (const std::uint64_t rows : {4U, 16U, 32U, 64U}) {
        auto status = gem16::internal::LaunchFp8ReferenceTokenQuantizationBatch(
            activation.get(), activation_e4m3.get(), activation_scales.get(),
            rows, input_elements, nullptr);
        CHECK(status.ok());
        const auto project = [&]() {
          return gem16::internal::LaunchFp8Sm120DirectProjectionBatch(
              activation_e4m3.get(), activation_scales.get(), slab.get(),
              slab_scales.get(), output.get(), rows, kSlabRows,
              input_elements, nullptr);
        };
        const float projection_ms = TimeDiagnostic(
            project, kIterations, "synchronize decoder-free projection");

        status = gem16::internal::LaunchTrellis35ReferenceW4A8ProjectionM1(
            activation_e4m3.get(), activation_scales.get(), family, expert,
            reference.get(), input_elements, output_elements, nullptr);
        CHECK(status.ok());
        CHECK(CudaOk(cudaDeviceSynchronize(),
                     "synchronize slab correctness reference"));
        std::vector<float> host_slab_output(kSlabRows);
        std::vector<float> host_reference(output_elements);
        CHECK(CudaOk(cudaMemcpy(host_slab_output.data(), output.get(),
                                kSlabRows * sizeof(float),
                                cudaMemcpyDeviceToHost),
                     "download transient slab output"));
        CHECK(CudaOk(cudaMemcpy(host_reference.data(), reference.get(),
                                output_elements * sizeof(float),
                                cudaMemcpyDeviceToHost),
                     "download inline reference output"));
        host_reference.resize(kSlabRows);
        Compare(host_reference, host_slab_output, 4.0e-3F, 8.0e-5F,
                "transient E4M3 slab parity");

        const std::uint64_t slab_bytes = kSlabRows * input_elements;
        const std::uint64_t workspace_bytes =
            slab_bytes + kSlabRows * sizeof(std::uint16_t) +
            rows * input_elements + rows * sizeof(float) +
            rows * kSlabRows * sizeof(float);
        std::cout << "transient_e4m3_slab family=" << family_name
                  << " rate=" << family.rate_map[expert]
                  << " expert=" << expert << " M=" << rows
                  << " N=" << kSlabRows << " K=" << input_elements
                  << " decode_ms=" << decode_ms
                  << " projection_ms=" << projection_ms
                  << " slab_bytes_written="
                  << slab_bytes + kSlabRows * sizeof(std::uint16_t)
                  << " slab_bytes_read="
                  << slab_bytes + kSlabRows * sizeof(std::uint16_t)
                  << " bounded_workspace_bytes=" << workspace_bytes
                  << '\n';
      }
    }
  };
  profile_family("gate_up", layer.gate_up,
                 gem16::internal::kTrellis35GateUpInput,
                 gem16::internal::kTrellis35GateUpOutput);
  profile_family("down", layer.down, gem16::internal::kTrellis35DownInput,
                 gem16::internal::kTrellis35DownOutput);
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
    if (tokens == 0U || tokens > 2048U) {
      std::cerr << "prefill profile tokens must be in [1, 2048]\n";
      return 2;
    }
    return ProfileTrellis35Prefill(tokens) == 0 ? 0 : 1;
  }
  if (argc == 4 &&
      (std::string(argv[1]) == "--profile-prefill-checkpoint" ||
       std::string(argv[1]) == "--profile-prefill-checkpoint-m64" ||
       std::string(argv[1]) == "--profile-prefill-checkpoint-loop")) {
    const std::uint64_t tokens = std::stoull(argv[3]);
    if (tokens == 0U || tokens > 2048U) {
      std::cerr << "prefill profile tokens must be in [1, 2048]\n";
      return 2;
    }
    const auto output_mode =
        std::string(argv[1]) == "--profile-prefill-checkpoint-loop"
            ? gem16::internal::Trellis35PrefillOutputMode::kLoopN128
            : gem16::internal::Trellis35PrefillOutputMode::kFusedN128;
    const auto kernel_mode =
        std::string(argv[1]) == "--profile-prefill-checkpoint-m64"
            ? gem16::internal::Trellis35PrefillKernelMode::kGroupedM64Hybrid
            : gem16::internal::Trellis35PrefillKernelMode::kGroupedM32;
    ProfileRealPrefill(argv[2], tokens, output_mode, kernel_mode);
    return failures == 0 ? 0 : 1;
  }
  if (argc == 3 && std::string(argv[1]) == "--profile-slab-checkpoint") {
    ProfileTransientE4M3Slab(argv[2]);
    return failures == 0 ? 0 : 1;
  }
  if (argc == 2 && std::string(argv[1]) == "--wp12-numerical-matrix") {
    return RunTrellis35Wp12NumericalMatrix() == 0 ? 0 : 1;
  }
  if (argc == 2 && std::string(argv[1]) == "--wp14-output-matrix") {
    return RunTrellis35Wp14OutputMatrix() == 0 ? 0 : 1;
  }
  if (argc == 2 && std::string(argv[1]) == "--wp17-m64-matrix") {
    return RunTrellis35Wp17M64Matrix() == 0 ? 0 : 1;
  }
  if (argc == 2 && std::string(argv[1]) == "--wp17-m64-smoke") {
    return RunTrellis35Wp17M64Smoke() == 0 ? 0 : 1;
  }
  if (argc == 2 && std::string(argv[1]) == "--wp19-gelu-down-oracle") {
    return RunTrellis35Wp19GeluDownOracle() == 0 ? 0 : 1;
  }
  if (argc == 2 && std::string(argv[1]) == "--wp20-small-gelu-down-matrix") {
    return RunTrellis35Wp20SmallGeluDownMatrix() == 0 ? 0 : 1;
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
                 "--profile-prefill-checkpoint PATH TOKENS | "
                 "--profile-prefill-checkpoint-m64 PATH TOKENS | "
                 "--profile-prefill-checkpoint-loop PATH TOKENS | "
                 "--profile-slab-checkpoint PATH | "
                 "--wp12-numerical-matrix | --wp14-output-matrix | "
                 "--wp17-m64-matrix | --wp17-m64-smoke | "
                 "--wp19-gelu-down-oracle | "
                 "--wp20-small-gelu-down-matrix]\n";
    return 2;
  }
  suite_failures += failures;
  if (suite_failures == 0) {
    std::cout << "trellis35_m1_test_pass\n";
  }
  return suite_failures == 0 ? 0 : 1;
}

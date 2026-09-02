#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "cuda/engine/gemma4_26b_vision_artifact.h"
#include "cuda/fp8/reference.h"
#include "cuda/vision/gemma4_26b.h"
#include "gem16/image.h"

namespace {

struct Fixture {
  const char* name;
  std::uint32_t budget;
  std::uint32_t source_width;
  std::uint32_t source_height;
  std::uint32_t processed_width;
  std::uint32_t processed_height;
  std::uint32_t raw_patches;
  std::uint32_t soft_tokens;
};

constexpr std::array<Fixture, 9U> kFixtures{{
    {"budget-70-square.bmp", 70U, 96U, 96U, 384U, 384U, 576U, 64U},
    {"budget-70-wide.bmp", 70U, 100U, 70U, 480U, 336U, 630U, 70U},
    {"budget-70-tall.bmp", 70U, 70U, 100U, 336U, 480U, 630U, 70U},
    {"budget-140-square.bmp", 140U, 96U, 96U, 528U, 528U, 1089U, 121U},
    {"budget-140-wide.bmp", 140U, 70U, 50U, 672U, 480U, 1260U, 140U},
    {"budget-140-tall.bmp", 140U, 50U, 70U, 480U, 672U, 1260U, 140U},
    {"budget-280-square.bmp", 280U, 96U, 96U, 768U, 768U, 2304U, 256U},
    {"budget-280-wide.bmp", 280U, 100U, 70U, 960U, 672U, 2520U, 280U},
    {"budget-280-tall.bmp", 280U, 70U, 100U, 672U, 960U, 2520U, 280U},
}};

int failures = 0;

__global__ void ReferenceGeluProductKernel(const std::uint16_t* gate,
                                           const std::uint16_t* up,
                                           std::uint16_t* product,
                                           std::uint64_t elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const float x =
      static_cast<float>(__ushort_as_bfloat16(gate[index]));
  constexpr float kAlpha = 0.7978845608028654F;
  const float activated =
      0.5F * x *
      (1.0F + tanhf(kAlpha * (x + 0.044715F * x * x * x)));
  product[index] = __bfloat16_as_ushort(__float2bfloat16_rn(
      activated * static_cast<float>(__ushort_as_bfloat16(up[index]))));
}

void Check(bool condition, const std::string& message) {
  if (condition) return;
  ++failures;
  std::cerr << "FAIL: " << message << '\n';
}

bool CudaOk(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) return true;
  Check(false, std::string(operation) + ": " + cudaGetErrorName(error) +
                   ": " + cudaGetErrorString(error));
  return false;
}

gem16::Result<gem16::Gemma4Moe26BVisionImage> LoadFixture(
    const std::filesystem::path& root, const Fixture& fixture,
    gem16::Gemma4Moe26BVisionPreprocessTimings* timings = nullptr) {
  return gem16::LoadGemma4Moe26BVisionImage(
      root / fixture.name,
      gem16::Gemma4Moe26BVisionImageOptions{fixture.budget, timings});
}

void TestPreprocessing(const std::filesystem::path& root) {
  for (const Fixture& fixture : kFixtures) {
    gem16::Gemma4Moe26BVisionPreprocessTimings timings;
    auto image = LoadFixture(root, fixture, &timings);
    Check(image.ok(), std::string(fixture.name) + " loads");
    if (!image.ok()) continue;
    const auto& value = image.value();
    Check(value.source_width == fixture.source_width,
          std::string(fixture.name) + " source width");
    Check(value.source_height == fixture.source_height,
          std::string(fixture.name) + " source height");
    Check(value.processed_width == fixture.processed_width,
          std::string(fixture.name) + " processed width");
    Check(value.processed_height == fixture.processed_height,
          std::string(fixture.name) + " processed height");
    Check(value.raw_patch_count == fixture.raw_patches,
          std::string(fixture.name) + " raw patch count");
    Check(value.soft_token_count == fixture.soft_tokens,
          std::string(fixture.name) + " soft token count");
    Check(value.soft_token_budget == fixture.budget,
          std::string(fixture.name) + " budget");
    Check(value.patches.size() ==
              static_cast<std::size_t>(fixture.raw_patches) * 16U * 16U * 3U,
          std::string(fixture.name) + " patch payload size");
    Check(value.positions.size() ==
              static_cast<std::size_t>(fixture.raw_patches) * 2U,
          std::string(fixture.name) + " position payload size");
    Check(timings.decode_milliseconds >= 0.0 &&
              timings.resize_milliseconds >= 0.0 &&
              timings.patchify_milliseconds >= 0.0 &&
              timings.total_milliseconds > 0.0,
          std::string(fixture.name) + " timing boundaries");
    Check(std::all_of(value.patches.begin(), value.patches.end(),
                      [](float item) {
                        return std::isfinite(item) && item >= 0.0F &&
                               item <= 1.0F;
                      }),
          std::string(fixture.name) + " finite normalized pixels");
  }
}

std::array<const Fixture*, 3U> MaximumFixtures() {
  return {&kFixtures[1], &kFixtures[4], &kFixtures[7]};
}

void CheckLayerTiming(
    const gem16::internal::Gemma4Moe26BVisionLayerTimings& timing,
    std::size_t layer) {
  const std::array<float, 10U> stages{
      timing.input_norm_quant_milliseconds,
      timing.qkv_projection_milliseconds,
      timing.qkv_norm_rope_milliseconds,
      timing.attention_milliseconds,
      timing.output_projection_residual_milliseconds,
      timing.ffn_norm_quant_milliseconds,
      timing.gate_up_milliseconds,
      timing.gelu_milliseconds,
      timing.product_quant_milliseconds,
      timing.down_residual_milliseconds,
  };
  Check(std::all_of(stages.begin(), stages.end(),
                    [](float item) { return std::isfinite(item) && item >= 0.0F; }),
        "finite layer timing " + std::to_string(layer));
}

void TestRuntime(const std::filesystem::path& fixtures,
                 const std::filesystem::path& module) {
  auto artifact =
      gem16::internal::Gemma4Moe26BVisionDeviceArtifact::Load(module);
  Check(artifact.ok(), "real Vision artifact loads");
  if (!artifact.ok()) return;
  constexpr std::array<std::pair<std::uint32_t, std::uint64_t>, 3U>
      kWorkspaceCapacities{{
          {70U, 36380768U},
          {140U, 64372928U},
          {280U, 120356480U},
      }};
  for (const auto& [budget, expected_bytes] : kWorkspaceCapacities) {
    auto sized = gem16::internal::Gemma4Moe26BVisionRuntime::Create(
        artifact.value(), budget);
    Check(sized.ok(), "Vision runtime creates for maximum budget " +
                          std::to_string(budget));
    if (!sized.ok()) continue;
    Check(sized.value().workspace_bytes() == expected_bytes,
          "Vision workspace matches maximum budget " +
              std::to_string(budget));
    Check(sized.value().maximum_soft_token_budget() == budget,
          "Vision maximum budget is reported as " + std::to_string(budget));
    if (budget == 70U) {
      auto image = LoadFixture(fixtures, kFixtures[4]);
      Check(image.ok(), "budget-140 fixture loads for capacity rejection");
      if (image.ok()) {
        const auto& value = image.value();
        const gem16::Gemma4Moe26BVisionInputSegment segment{
            0U, value.soft_token_count, value.soft_token_budget,
            value.raw_patch_count, value.patches, value.positions};
        Check(!sized.value().Encode(segment, nullptr).ok(),
              "budget-70 runtime rejects a budget-140 request");
      }
    }
  }
  Check(!gem16::internal::Gemma4Moe26BVisionRuntime::Create(artifact.value(),
                                                            71U)
             .ok(),
        "unsupported Vision maximum budget is rejected");
  auto runtime =
      gem16::internal::Gemma4Moe26BVisionRuntime::Create(artifact.value());
  Check(runtime.ok(), "Vision runtime creates");
  if (!runtime.ok()) return;
  auto recorder =
      gem16::internal::Gemma4Moe26BVisionTimingRecorder::Create();
  Check(recorder.ok(), "fixed timing recorder creates");
  if (!recorder.ok()) return;
  Check(artifact.value().stats().device_allocations == 1U,
        "Vision weights use one device allocation");
  Check(runtime.value().workspace_bytes() > 0U, "workspace is reported");
  Check(runtime.value().host_pinned_bytes() > 0U,
        "pinned staging is reported");

  cudaStream_t stream = nullptr;
  if (!CudaOk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "create stream")) {
    return;
  }
  for (const Fixture* fixture : MaximumFixtures()) {
    auto image = LoadFixture(fixtures, *fixture);
    Check(image.ok(), std::string(fixture->name) + " reloads for runtime");
    if (!image.ok()) continue;
    const auto& value = image.value();
    const gem16::Gemma4Moe26BVisionInputSegment segment{
        0U, value.soft_token_count, value.soft_token_budget,
        value.raw_patch_count, value.patches, value.positions};
    gem16::Status status = runtime.value().Encode(segment, stream);
    Check(status.ok(), std::string(fixture->name) + " warmup encodes");
    if (!status.ok() ||
        !CudaOk(cudaStreamSynchronize(stream), "synchronize warmup")) {
      continue;
    }
    std::size_t free_before = 0U;
    std::size_t total = 0U;
    if (!CudaOk(cudaMemGetInfo(&free_before, &total), "memory before runs")) {
      continue;
    }
    std::vector<float> reference;
    for (std::uint32_t run = 0U; run < 2U; ++run) {
      (void)(status = runtime.value().Encode(segment, stream, &recorder.value()));
      Check(status.ok(), std::string(fixture->name) + " timed encode");
      gem16::internal::Gemma4Moe26BVisionRuntimeTimings timing;
      if (status.ok()) (void)(status = recorder.value().Resolve(&timing));
      Check(status.ok(), std::string(fixture->name) + " timing resolves");
      if (!status.ok()) break;
      Check(timing.budget == fixture->budget &&
                timing.raw_patch_count == fixture->raw_patches &&
                timing.soft_token_count == fixture->soft_tokens,
            std::string(fixture->name) + " timing geometry");
      Check(std::isfinite(timing.total_gpu_milliseconds) &&
                timing.total_gpu_milliseconds > 0.0F,
            std::string(fixture->name) + " total GPU timing");
      for (std::size_t layer = 0U; layer < timing.layers.size(); ++layer) {
        CheckLayerTiming(timing.layers[layer], layer);
      }
      std::vector<float> output(
          static_cast<std::size_t>(fixture->soft_tokens) * 2816U);
      if (!CudaOk(cudaMemcpy(output.data(), runtime.value().output(),
                             output.size() * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  "copy Vision output")) {
        break;
      }
      Check(std::all_of(output.begin(), output.end(),
                        [](float item) { return std::isfinite(item); }),
            std::string(fixture->name) + " finite output");
      if (run == 0U) {
        reference = std::move(output);
      } else {
        Check(output == reference,
              std::string(fixture->name) + " bitwise repeatability");
      }
    }
    std::size_t free_after = 0U;
    if (CudaOk(cudaMemGetInfo(&free_after, &total), "memory after runs")) {
      Check(free_after == free_before,
            std::string(fixture->name) + " recurring device allocation free");
    }
  }
  (void)cudaStreamDestroy(stream);
}

void TestFusedGeluProductQuantization() {
  constexpr std::uint64_t kTokens = 3U;
  constexpr std::uint64_t kElements = 4304U;
  constexpr std::uint64_t kTotal = kTokens * kElements;
  std::vector<std::uint16_t> gate(kTotal);
  std::vector<std::uint16_t> up(kTotal);
  for (std::uint64_t index = 0U; index < kTotal; ++index) {
    const float gate_value =
        static_cast<float>(static_cast<int>(index % 257U) - 128) / 16.0F;
    const float up_value =
        static_cast<float>(static_cast<int>((index * 17U) % 199U) - 99) /
        32.0F;
    gate[index] =
        __bfloat16_as_ushort(__float2bfloat16_rn(gate_value));
    up[index] = __bfloat16_as_ushort(__float2bfloat16_rn(up_value));
  }
  std::uint16_t* device_gate = nullptr;
  std::uint16_t* device_up = nullptr;
  std::uint16_t* device_product = nullptr;
  std::uint8_t* reference_output = nullptr;
  std::uint8_t* fused_output = nullptr;
  float* reference_scales = nullptr;
  float* fused_scales = nullptr;
  auto allocate = [](auto** pointer, std::size_t bytes) {
    return CudaOk(cudaMalloc(reinterpret_cast<void**>(pointer), bytes),
                  "allocate fused GELU quantization test buffer");
  };
  if (!allocate(&device_gate, kTotal * sizeof(std::uint16_t)) ||
      !allocate(&device_up, kTotal * sizeof(std::uint16_t)) ||
      !allocate(&device_product, kTotal * sizeof(std::uint16_t)) ||
      !allocate(&reference_output, kTotal) ||
      !allocate(&fused_output, kTotal) ||
      !allocate(&reference_scales, kTokens * sizeof(float)) ||
      !allocate(&fused_scales, kTokens * sizeof(float))) {
    return;
  }
  bool ok = CudaOk(cudaMemcpy(device_gate, gate.data(),
                              kTotal * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice),
                   "copy fused GELU test gate") &&
            CudaOk(cudaMemcpy(device_up, up.data(),
                              kTotal * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice),
                   "copy fused GELU test up");
  if (ok) {
    ReferenceGeluProductKernel<<<static_cast<unsigned>((kTotal + 255U) / 256U),
                                 256U>>>(device_gate, device_up,
                                         device_product, kTotal);
    ok = CudaOk(cudaGetLastError(), "launch reference GELU product");
  }
  if (ok) {
    gem16::Status status =
        gem16::internal::LaunchFp8ReferenceTokenQuantizationBf16Batch(
            device_product, reference_output, reference_scales, kTokens,
            kElements, nullptr);
    Check(status.ok(), "reference GELU product quantization launches");
    ok = status.ok();
  }
  if (ok) {
    gem16::Status status = gem16::internal::
        LaunchGemma4Moe26BVisionFusedGeluProductQuantization(
            device_gate, device_up, fused_output, fused_scales, kTokens,
            nullptr);
    Check(status.ok(), "fused GELU product quantization launches");
    ok = status.ok();
  }
  std::vector<std::uint8_t> reference_bytes(kTotal);
  std::vector<std::uint8_t> fused_bytes(kTotal);
  std::vector<float> reference_scale_values(kTokens);
  std::vector<float> fused_scale_values(kTokens);
  if (ok) {
    ok = CudaOk(cudaMemcpy(reference_bytes.data(), reference_output, kTotal,
                           cudaMemcpyDeviceToHost),
                "copy reference E4M3 bytes") &&
         CudaOk(cudaMemcpy(fused_bytes.data(), fused_output, kTotal,
                           cudaMemcpyDeviceToHost),
                "copy fused E4M3 bytes") &&
         CudaOk(cudaMemcpy(reference_scale_values.data(), reference_scales,
                           kTokens * sizeof(float), cudaMemcpyDeviceToHost),
                "copy reference E4M3 scales") &&
         CudaOk(cudaMemcpy(fused_scale_values.data(), fused_scales,
                           kTokens * sizeof(float), cudaMemcpyDeviceToHost),
                "copy fused E4M3 scales");
  }
  if (ok) {
    Check(fused_bytes == reference_bytes,
          "fused GELU product E4M3 bytes are bit identical");
    Check(fused_scale_values == reference_scale_values,
          "fused GELU product scales are bit identical");
  }
  (void)cudaFree(device_gate);
  (void)cudaFree(device_up);
  (void)cudaFree(device_product);
  (void)cudaFree(reference_output);
  (void)cudaFree(fused_output);
  (void)cudaFree(reference_scales);
  (void)cudaFree(fused_scales);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3 && argc != 5) {
    std::cerr << "usage: gem16-cuda-vision26b-tests --fixtures <directory> "
                 "[--vision-module <directory>]\n";
    return 64;
  }
  if (std::string(argv[1]) != "--fixtures") return 64;
  const std::filesystem::path fixtures = argv[2];
  TestPreprocessing(fixtures);
  if (argc == 5) {
    if (std::string(argv[3]) != "--vision-module") return 64;
    TestFusedGeluProductQuantization();
    TestRuntime(fixtures, argv[4]);
  }
  if (failures != 0) {
    std::cerr << failures << " Gemma 4 26B Vision test(s) failed\n";
    return 1;
  }
  std::cout << "Gemma 4 26B Vision tests passed";
  if (argc == 3) std::cout << " (preprocessing fixture contract only)";
  std::cout << '\n';
  return 0;
}

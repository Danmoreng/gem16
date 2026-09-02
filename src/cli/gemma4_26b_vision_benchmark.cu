#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "compiler/sha256.h"
#include "cuda/engine/gemma4_26b_vision_artifact.h"
#include "cuda/vision/gemma4_26b.h"
#include "gem16/image.h"
#include "util/json.h"

namespace {

gem16::Status CudaFailure(const char* operation, cudaError_t error) {
  return gem16::Status(
      gem16::StatusCode::kInternal,
      std::string(operation) + ": " + cudaGetErrorName(error) + ": " +
          cudaGetErrorString(error));
}

bool ParseUnsigned(const char* text, std::uint32_t* value) {
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(text, &end, 10);
  if (text == end || *end != '\0' || parsed > 0xffffffffUL) return false;
  *value = static_cast<std::uint32_t>(parsed);
  return true;
}

double Milliseconds(std::chrono::steady_clock::time_point begin,
                    std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

void PrintLayer(const gem16::internal::Gemma4Moe26BVisionLayerTimings& layer) {
  std::cout << "{\"input_norm_quant_ms\":"
            << layer.input_norm_quant_milliseconds
            << ",\"qkv_projection_ms\":" << layer.qkv_projection_milliseconds
            << ",\"qkv_norm_rope_ms\":" << layer.qkv_norm_rope_milliseconds
            << ",\"attention_ms\":" << layer.attention_milliseconds
            << ",\"output_projection_residual_ms\":"
            << layer.output_projection_residual_milliseconds
            << ",\"ffn_norm_quant_ms\":"
            << layer.ffn_norm_quant_milliseconds
            << ",\"gate_up_ms\":" << layer.gate_up_milliseconds
            << ",\"gelu_ms\":" << layer.gelu_milliseconds
            << ",\"product_quant_ms\":"
            << layer.product_quant_milliseconds
            << ",\"down_residual_ms\":"
            << layer.down_residual_milliseconds << '}';
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4 || argc > 8) {
    std::cerr << "usage: gem16-26b-vision-benchmark <vision-module> <image> "
                 "<70|140|280> [--warmups N] [--repetitions N]\n";
    return 64;
  }
  std::uint32_t budget = 0U;
  std::uint32_t warmups = 1U;
  std::uint32_t repetitions = 3U;
  if (!ParseUnsigned(argv[3], &budget) ||
      (budget != 70U && budget != 140U && budget != 280U)) {
    std::cerr << "error: budget must be 70, 140, or 280\n";
    return 64;
  }
  for (int index = 4; index < argc; index += 2) {
    if (index + 1 >= argc) return 64;
    std::uint32_t value = 0U;
    if (!ParseUnsigned(argv[index + 1], &value)) return 64;
    const std::string option = argv[index];
    if (option == "--warmups") {
      warmups = value;
    } else if (option == "--repetitions") {
      repetitions = value;
    } else {
      return 64;
    }
  }
  if (repetitions == 0U || repetitions > 20U || warmups > 20U) return 64;

  gem16::Gemma4Moe26BVisionPreprocessTimings preprocess;
  const auto preprocess_wall_begin = std::chrono::steady_clock::now();
  auto image = gem16::LoadGemma4Moe26BVisionImage(
      argv[2], gem16::Gemma4Moe26BVisionImageOptions{budget, &preprocess});
  const auto preprocess_wall_end = std::chrono::steady_clock::now();
  if (!image.ok()) {
    std::cerr << "error: " << image.status().message() << '\n';
    return 2;
  }
  const auto load_begin = std::chrono::steady_clock::now();
  auto artifact =
      gem16::internal::Gemma4Moe26BVisionDeviceArtifact::Load(argv[1]);
  if (!artifact.ok()) {
    std::cerr << "error: " << artifact.status().message() << '\n';
    return 2;
  }
  auto runtime =
      gem16::internal::Gemma4Moe26BVisionRuntime::Create(artifact.value());
  if (!runtime.ok()) {
    std::cerr << "error: " << runtime.status().message() << '\n';
    return 2;
  }
  auto recorder =
      gem16::internal::Gemma4Moe26BVisionTimingRecorder::Create();
  if (!recorder.ok()) {
    std::cerr << "error: " << recorder.status().message() << '\n';
    return 2;
  }
  const auto load_end = std::chrono::steady_clock::now();
  cudaStream_t stream = nullptr;
  cudaError_t error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (error != cudaSuccess) {
    std::cerr << "error: " << CudaFailure("create stream", error).message()
              << '\n';
    return 2;
  }
  const auto& value = image.value();
  gem16::Gemma4Moe26BVisionInputSegment segment{
      0U, value.soft_token_count, value.soft_token_budget,
      value.raw_patch_count, value.patches, value.positions};
  gem16::Status status = gem16::Status::Ok();
  for (std::uint32_t warmup = 0U; warmup < warmups; ++warmup) {
    (void)(status = runtime.value().Encode(segment, stream));
    if (!status.ok()) break;
    error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) {
      (void)(status = CudaFailure("synchronize warmup", error));
      break;
    }
  }
  std::size_t free_before = 0U;
  std::size_t total_bytes = 0U;
  if (status.ok()) {
    error = cudaMemGetInfo(&free_before, &total_bytes);
    if (error != cudaSuccess) {
      (void)(status = CudaFailure("query memory before benchmark", error));
    }
  }
  std::vector<gem16::internal::Gemma4Moe26BVisionRuntimeTimings> timings;
  timings.reserve(repetitions);
  std::vector<float> reference;
  std::string output_sha256;
  std::uint64_t repeat_mismatches = 0U;
  for (std::uint32_t repetition = 0U;
       status.ok() && repetition < repetitions; ++repetition) {
    (void)(status = runtime.value().Encode(segment, stream, &recorder.value()));
    gem16::internal::Gemma4Moe26BVisionRuntimeTimings sample;
    if (status.ok()) (void)(status = recorder.value().Resolve(&sample));
    if (!status.ok()) break;
    timings.push_back(sample);
    std::vector<float> output(
        static_cast<std::size_t>(value.soft_token_count) * 2816U);
    error = cudaMemcpy(output.data(), runtime.value().output(),
                       output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
      (void)(status = CudaFailure("copy benchmark output", error));
      break;
    }
    if (repetition == 0U) {
      reference = std::move(output);
      output_sha256 = gem16::compiler::Sha256Hex(
          reference.data(), reference.size() * sizeof(float));
    } else {
      for (std::size_t index = 0U; index < output.size(); ++index) {
        repeat_mismatches += output[index] != reference[index] ? 1U : 0U;
      }
    }
  }
  std::size_t free_after = 0U;
  if (status.ok()) {
    error = cudaMemGetInfo(&free_after, &total_bytes);
    if (error != cudaSuccess) {
      (void)(status = CudaFailure("query memory after benchmark", error));
    }
  }
  (void)cudaStreamDestroy(stream);
  if (!status.ok()) {
    std::cerr << "error: " << status.message() << '\n';
    return 2;
  }

  std::cout << "{\"schema_version\":1,\"fixture\":"
            << gem16::json::Quote(std::filesystem::path(argv[2]).filename().string())
            << ",\"budget\":" << budget
            << ",\"source_width\":" << value.source_width
            << ",\"source_height\":" << value.source_height
            << ",\"processed_width\":" << value.processed_width
            << ",\"processed_height\":" << value.processed_height
            << ",\"raw_patch_count\":" << value.raw_patch_count
            << ",\"soft_token_count\":" << value.soft_token_count
            << ",\"preprocess\":{\"decode_ms\":"
            << preprocess.decode_milliseconds << ",\"resize_ms\":"
            << preprocess.resize_milliseconds << ",\"patchify_ms\":"
            << preprocess.patchify_milliseconds << ",\"total_ms\":"
            << preprocess.total_milliseconds << ",\"file_inclusive_wall_ms\":"
            << Milliseconds(preprocess_wall_begin, preprocess_wall_end)
            << "},\"model_load_ms\":" << Milliseconds(load_begin, load_end)
            << ",\"memory\":{\"vision_weight_bytes\":"
            << artifact.value().arena_bytes() << ",\"vision_workspace_bytes\":"
            << runtime.value().workspace_bytes() << ",\"host_pinned_bytes\":"
            << runtime.value().host_pinned_bytes()
            << ",\"graph_private_bytes\":0,\"free_before_runs\":"
            << free_before << ",\"free_after_runs\":" << free_after
            << ",\"recurring_free_delta\":"
            << static_cast<std::int64_t>(free_after) -
                   static_cast<std::int64_t>(free_before)
            << "},\"repeat_mismatches\":" << repeat_mismatches
            << ",\"output_sha256\":"
            << gem16::json::Quote(output_sha256)
            << ",\"warmups\":" << warmups << ",\"runs\":[";
  for (std::size_t run = 0U; run < timings.size(); ++run) {
    if (run != 0U) std::cout << ',';
    const auto& sample = timings[run];
    std::cout << "{\"upload_ms\":" << sample.upload_milliseconds
              << ",\"patch_project_ms\":"
              << sample.patch_project_milliseconds
              << ",\"position_add_ms\":" << sample.position_add_milliseconds
              << ",\"pool_standardize_ms\":"
              << sample.pool_standardize_milliseconds
              << ",\"final_norm_project_ms\":"
              << sample.final_norm_project_milliseconds
              << ",\"total_gpu_ms\":" << sample.total_gpu_milliseconds
              << ",\"layers\":[";
    for (std::size_t layer = 0U; layer < sample.layers.size(); ++layer) {
      if (layer != 0U) std::cout << ',';
      PrintLayer(sample.layers[layer]);
    }
    std::cout << "]}";
  }
  std::cout << "]}\n";
  return 0;
}

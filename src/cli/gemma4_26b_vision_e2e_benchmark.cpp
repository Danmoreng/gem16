#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "gem16/chat.h"
#include "util/json.h"

namespace {

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

struct Run {
  double ttft_milliseconds = 0.0;
  double generate_wall_milliseconds = 0.0;
  std::uint64_t free_before_bytes = 0U;
  std::uint64_t free_after_bytes = 0U;
  std::uint64_t graph_private_bytes = 0U;
  std::uint64_t prompt_tokens = 0U;
  std::uint32_t first_token = 0U;
};

gem16::Status CudaFailure(const char* operation, cudaError_t error) {
  return gem16::Status(
      gem16::StatusCode::kInternal,
      std::string(operation) + ": " + cudaGetErrorName(error) + ": " +
          cudaGetErrorString(error));
}

gem16::Result<Run> Execute(
    const std::shared_ptr<gem16::ModelRuntime>& runtime,
    const std::filesystem::path& model,
    const std::filesystem::path& vision_model,
    const gem16::Gemma4Moe26BVisionImage& image,
    std::uint32_t context_tokens) {
  auto processor = gem16::GemmaChatProcessor::Load(model);
  if (!processor.ok()) return processor.status();
  gem16::ChatSessionOptions session_options;
  session_options.model_directory = model;
  session_options.vision_model_directory = vision_model;
  session_options.max_context_tokens = context_tokens;
  auto session = gem16::ChatSession::Create(
      runtime, session_options, std::move(processor).value());
  if (!session.ok()) return session.status();

  gem16::GenerationMessage message;
  message.role = "user";
  message.content.push_back(
      gem16::GenerationContentPart::Gemma4Moe26BImage(image));
  message.content.push_back(
      gem16::GenerationContentPart::Text("Describe the image precisely."));
  gem16::ChatGenerationRequest request;
  request.messages.push_back(std::move(message));
  request.max_generated_tokens = 1U;
  request.thinking.effort = gem16::ThinkingEffort::kOff;

  Run run;
  std::size_t total = 0U;
  cudaError_t error = cudaMemGetInfo(&run.free_before_bytes, &total);
  if (error != cudaSuccess) return CudaFailure("memory before image", error);
  const auto begin = std::chrono::steady_clock::now();
  auto generated = session.value().Generate(request);
  const auto end = std::chrono::steady_clock::now();
  if (!generated.ok()) return generated.status();
  error = cudaMemGetInfo(&run.free_after_bytes, &total);
  if (error != cudaSuccess) return CudaFailure("memory after image", error);
  run.generate_wall_milliseconds = Milliseconds(begin, end);
  run.ttft_milliseconds = generated.value().inference.prompt_milliseconds;
  run.graph_private_bytes =
      generated.value().inference.decode_graph_device_bytes;
  run.prompt_tokens = generated.value().prompt_token_ids.size();
  if (!generated.value().inference.output_token_ids.empty()) {
    run.first_token = generated.value().inference.output_token_ids.front();
  }
  return run;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "usage: gem16-26b-vision-e2e-benchmark <model> "
                 "<vision-module> <image> <70|140|280> <warmups> "
                 "<repetitions>\n";
    return 64;
  }
  std::uint32_t budget = 0U;
  std::uint32_t warmups = 0U;
  std::uint32_t repetitions = 0U;
  if (!ParseUnsigned(argv[4], &budget) ||
      !ParseUnsigned(argv[5], &warmups) ||
      !ParseUnsigned(argv[6], &repetitions) ||
      (budget != 70U && budget != 140U && budget != 280U) ||
      warmups > 5U || repetitions == 0U || repetitions > 10U) {
    return 64;
  }
  constexpr std::uint32_t kContextTokens = 32768U;
  gem16::Gemma4Moe26BVisionPreprocessTimings preprocess;
  auto image = gem16::LoadGemma4Moe26BVisionImage(
      argv[3], gem16::Gemma4Moe26BVisionImageOptions{budget, &preprocess});
  if (!image.ok()) {
    std::cerr << "error: " << image.status().message() << '\n';
    return 2;
  }
  const auto load_begin = std::chrono::steady_clock::now();
  auto runtime = gem16::ModelRuntime::Load(
      {argv[1], {}, kContextTokens, 0, true, argv[2]});
  const auto load_end = std::chrono::steady_clock::now();
  if (!runtime.ok()) {
    std::cerr << "error: " << runtime.status().message() << '\n';
    return 2;
  }
  std::size_t free_after_load = 0U;
  std::size_t total_bytes = 0U;
  cudaError_t error = cudaMemGetInfo(&free_after_load, &total_bytes);
  if (error != cudaSuccess) {
    std::cerr << "error: "
              << CudaFailure("memory after load", error).message() << '\n';
    return 2;
  }
  for (std::uint32_t warmup = 0U; warmup < warmups; ++warmup) {
    auto run = Execute(runtime.value(), argv[1], argv[2], image.value(),
                       kContextTokens);
    if (!run.ok()) {
      std::cerr << "error: " << run.status().message() << '\n';
      return 2;
    }
  }
  std::vector<Run> runs;
  runs.reserve(repetitions);
  for (std::uint32_t repetition = 0U; repetition < repetitions;
       ++repetition) {
    auto run = Execute(runtime.value(), argv[1], argv[2], image.value(),
                       kContextTokens);
    if (!run.ok()) {
      std::cerr << "error: " << run.status().message() << '\n';
      return 2;
    }
    runs.push_back(std::move(run).value());
  }
  std::cout << "{\"schema_version\":1,\"fixture\":"
            << gem16::json::Quote(
                   std::filesystem::path(argv[3]).filename().string())
            << ",\"budget\":" << budget
            << ",\"raw_patch_count\":" << image.value().raw_patch_count
            << ",\"soft_token_count\":" << image.value().soft_token_count
            << ",\"preprocess_total_ms\":"
            << preprocess.total_milliseconds
            << ",\"model_load_wall_ms\":"
            << Milliseconds(load_begin, load_end)
            << ",\"memory\":{\"text_weight_bytes\":"
            << runtime.value()->weight_bytes()
            << ",\"vision_weight_bytes\":"
            << runtime.value()->vision_weight_bytes()
            << ",\"assistant_weight_bytes\":"
            << runtime.value()->assistant_weight_bytes()
            << ",\"kv_cache_bytes\":" << runtime.value()->kv_cache_bytes()
            << ",\"workspace_bytes\":" << runtime.value()->workspace_bytes()
            << ",\"free_after_load_bytes\":" << free_after_load
            << "},\"warmups\":" << warmups << ",\"runs\":[";
  for (std::size_t index = 0U; index < runs.size(); ++index) {
    if (index != 0U) std::cout << ',';
    const Run& run = runs[index];
    std::cout << "{\"ttft_ms\":" << run.ttft_milliseconds
              << ",\"generate_wall_ms\":"
              << run.generate_wall_milliseconds
              << ",\"free_before_bytes\":" << run.free_before_bytes
              << ",\"free_after_bytes\":" << run.free_after_bytes
              << ",\"recurring_free_delta\":"
              << static_cast<std::int64_t>(run.free_after_bytes) -
                     static_cast<std::int64_t>(run.free_before_bytes)
              << ",\"graph_private_bytes\":" << run.graph_private_bytes
              << ",\"prompt_tokens\":" << run.prompt_tokens
              << ",\"first_token\":" << run.first_token << '}';
  }
  std::cout << "]}\n";
  return 0;
}

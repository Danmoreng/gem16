#include <cuda_runtime_api.h>

#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "compiler/sha256.h"
#include "gem16/chat.h"
#include "util/json.h"

namespace {

constexpr std::uint32_t kWarmups = 3U;
constexpr std::uint32_t kRetainedPairs = 10U;
constexpr std::uint32_t kContextTokens = 32768U;
constexpr std::string_view kFrozenPrompt =
    "Describe the attached image precisely. Identify the main objects, "
    "their spatial relationships, and any readable text. Then give a concise "
    "caption suitable for a Wikipedia article.";

bool ParseUnsigned(const char* text, std::uint32_t* value) {
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(text, &end, 10);
  if (text == end || *end != '\0' || parsed > 0xffffffffUL) return false;
  *value = static_cast<std::uint32_t>(parsed);
  return true;
}

gem16::Status CudaFailure(const char* operation, cudaError_t error) {
  return gem16::Status(
      gem16::StatusCode::kInternal,
      std::string(operation) + ": " + cudaGetErrorName(error) + ": " +
          cudaGetErrorString(error));
}

struct Run {
  gem16::GreedyInferenceResult inference;
  std::uint64_t prompt_tokens = 0U;
  std::uint64_t device_used_before_bytes = 0U;
  std::uint64_t device_used_after_bytes = 0U;
  std::string output_sha256;
  std::string draft_sha256;
};

gem16::Result<Run> Execute(
    const std::shared_ptr<gem16::ModelRuntime>& runtime,
    const std::filesystem::path& model,
    const std::filesystem::path& assistant,
    const std::filesystem::path& vision,
    const gem16::Gemma4Moe26BVisionImage& image,
    std::uint32_t max_generated_tokens, bool d2) {
  auto processor = gem16::GemmaChatProcessor::Load(model);
  if (!processor.ok()) return processor.status();
  gem16::ChatSessionOptions options;
  options.model_directory = model;
  options.assistant_model_directory = assistant;
  options.vision_model_directory = vision;
  options.max_context_tokens = kContextTokens;
  options.mtp_draft_tokens = d2 ? 2U : 0U;
  options.sampling.enabled = false;
  options.sampling.seed = 0x563134ULL;
  auto session = gem16::ChatSession::Create(
      runtime, options, std::move(processor).value());
  if (!session.ok()) return session.status();

  gem16::GenerationMessage message;
  message.role = "user";
  message.content.push_back(
      gem16::GenerationContentPart::Gemma4Moe26BImage(image));
  message.content.push_back(
      gem16::GenerationContentPart::Text(std::string(kFrozenPrompt)));
  gem16::ChatGenerationRequest request;
  request.messages.push_back(std::move(message));
  request.max_generated_tokens = max_generated_tokens;
  request.thinking.effort = gem16::ThinkingEffort::kOff;

  std::size_t free_before = 0U;
  std::size_t total = 0U;
  cudaError_t error = cudaMemGetInfo(&free_before, &total);
  if (error != cudaSuccess) return CudaFailure("memory before V14 run", error);
  auto generated = session.value().Generate(request);
  if (!generated.ok()) return generated.status();
  std::size_t free_after = 0U;
  error = cudaMemGetInfo(&free_after, &total);
  if (error != cudaSuccess) return CudaFailure("memory after V14 run", error);

  Run run;
  run.inference = std::move(generated.value().inference);
  run.prompt_tokens = generated.value().prompt_token_ids.size();
  run.device_used_before_bytes = total - free_before;
  run.device_used_after_bytes = total - free_after;
  run.output_sha256 = gem16::compiler::Sha256Hex(
      run.inference.output_token_ids.data(),
      run.inference.output_token_ids.size() * sizeof(std::uint32_t));
  run.draft_sha256 = gem16::compiler::Sha256Hex(
      run.inference.mtp_proposed_token_ids.data(),
      run.inference.mtp_proposed_token_ids.size() * sizeof(std::uint32_t));
  return run;
}

void WriteRun(const Run& run) {
  const auto& value = run.inference;
  std::cout << "{\"prompt_tokens\":" << run.prompt_tokens
            << ",\"output_tokens\":" << value.output_token_ids.size()
            << ",\"output_sha256\":"
            << gem16::json::Quote(run.output_sha256)
            << ",\"draft_sha256\":"
            << gem16::json::Quote(run.draft_sha256)
            << ",\"ttft_ms\":" << value.prompt_milliseconds
            << ",\"vision_upload_ms\":"
            << value.vision_upload_milliseconds
            << ",\"vision_tower_ms\":"
            << value.vision_tower_milliseconds
            << ",\"vision_pool_project_ms\":"
            << value.vision_pool_project_milliseconds
            << ",\"text_prefill_ms\":"
            << value.text_prefill_milliseconds
            << ",\"post_first_decode_ms\":" << value.decode_milliseconds
            << ",\"post_first_tokens_per_second\":"
            << value.decode_tokens_per_second
            << ",\"mtp_proposed_tokens\":" << value.mtp_proposed_tokens
            << ",\"mtp_accepted_tokens\":" << value.mtp_accepted_tokens
            << ",\"mtp_rejected_tokens\":" << value.mtp_rejected_tokens
            << ",\"mtp_groups\":" << value.mtp_verification_groups
            << ",\"mtp_ordinary_fallback_tokens\":"
            << value.mtp_ordinary_fallback_tokens
            << ",\"fallback_count\":" << value.fallback_count
            << ",\"token_loop_allocations\":"
            << (value.token_loop_allocations ? "true" : "false")
            << ",\"decode_graph_device_bytes\":"
            << value.decode_graph_device_bytes
            << ",\"kv_cache_bytes\":" << value.kv_cache_bytes
            << ",\"workspace_bytes\":" << value.workspace_bytes
            << ",\"device_used_before_bytes\":"
            << run.device_used_before_bytes
            << ",\"device_used_after_bytes\":"
            << run.device_used_after_bytes << '}';
}

int Fail(const gem16::Status& status) {
  std::cerr << "error: " << status.message() << '\n';
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7 || std::endian::native != std::endian::little) {
    std::cerr << "usage: gem16-26b-vision-v14-benchmark <target> "
                 "<assistant> <vision-module> <image> <70|140|280> "
                 "<max-output-tokens>\n";
    return 64;
  }
  std::uint32_t budget = 0U;
  std::uint32_t max_output_tokens = 0U;
  if (!ParseUnsigned(argv[5], &budget) ||
      !ParseUnsigned(argv[6], &max_output_tokens) ||
      (budget != 70U && budget != 140U && budget != 280U) ||
      max_output_tokens < 8U || max_output_tokens > 512U) {
    return 64;
  }

  gem16::Gemma4Moe26BVisionPreprocessTimings preprocess;
  const auto preprocess_begin = std::chrono::steady_clock::now();
  auto image = gem16::LoadGemma4Moe26BVisionImage(
      argv[4], gem16::Gemma4Moe26BVisionImageOptions{budget, &preprocess});
  const auto preprocess_end = std::chrono::steady_clock::now();
  if (!image.ok()) return Fail(image.status());

  auto runtime = gem16::ModelRuntime::Load(
      {argv[1], argv[2], kContextTokens, 0, true, argv[3]});
  if (!runtime.ok()) return Fail(runtime.status());
  if (!runtime.value()->vision_mtp_supported()) {
    return Fail(gem16::Status(
        gem16::StatusCode::kUnsupported,
        "V14 requires the exact qualified Target+Vision+Assistant profile"));
  }

  const auto execute = [&](bool d2) {
    return Execute(runtime.value(), argv[1], argv[2], argv[3], image.value(),
                   max_output_tokens, d2);
  };
  for (std::uint32_t warmup = 0U; warmup < kWarmups; ++warmup) {
    auto ordinary = execute(false);
    if (!ordinary.ok()) return Fail(ordinary.status());
    auto d2 = execute(true);
    if (!d2.ok()) return Fail(d2.status());
    if (ordinary.value().inference.output_token_ids !=
        d2.value().inference.output_token_ids) {
      return Fail(gem16::Status(
          gem16::StatusCode::kDataLoss,
          "V14 warmup Ordinary and fixed-D2 token streams differ"));
    }
  }

  std::array<Run, kRetainedPairs> ordinary_runs;
  std::array<Run, kRetainedPairs> d2_runs;
  bool streams_identical = true;
  bool no_runtime_fallback = true;
  bool terminal_tail_bounded = true;
  bool no_token_loop_allocations = true;
  for (std::uint32_t index = 0U; index < kRetainedPairs; ++index) {
    auto ordinary = execute(false);
    if (!ordinary.ok()) return Fail(ordinary.status());
    auto d2 = execute(true);
    if (!d2.ok()) return Fail(d2.status());
    ordinary_runs[index] = std::move(ordinary).value();
    d2_runs[index] = std::move(d2).value();
    streams_identical = streams_identical &&
        ordinary_runs[index].inference.output_token_ids ==
            d2_runs[index].inference.output_token_ids;
    // The fixed graph completes an output-cap boundary with at most D2
    // ordinary tail tokens. That bounded terminal condition is not a path or
    // capability fallback; fallback_count records an actual runtime fallback.
    no_runtime_fallback = no_runtime_fallback &&
        d2_runs[index].inference.fallback_count == 0U;
    terminal_tail_bounded = terminal_tail_bounded &&
        d2_runs[index].inference.mtp_ordinary_fallback_tokens <= 2U;
    no_token_loop_allocations = no_token_loop_allocations &&
        !ordinary_runs[index].inference.token_loop_allocations &&
        !d2_runs[index].inference.token_loop_allocations;
  }

  std::cout << "{\"schema_version\":1,\"milestone\":\"V14\""
            << ",\"qualification\":\"exact_trellis35_vision_fixed_d2\""
            << ",\"vision_mtp_supported\":true"
            << ",\"warmups\":" << kWarmups
            << ",\"retained_pairs\":" << kRetainedPairs
            << ",\"budget\":" << budget
            << ",\"max_output_tokens\":" << max_output_tokens
            << ",\"prompt\":" << gem16::json::Quote(kFrozenPrompt)
            << ",\"raw_patch_count\":" << image.value().raw_patch_count
            << ",\"soft_token_count\":" << image.value().soft_token_count
            << ",\"preprocess_wall_ms\":"
            << std::chrono::duration<double, std::milli>(
                   preprocess_end - preprocess_begin).count()
            << ",\"preprocess_total_ms\":"
            << preprocess.total_milliseconds
            << ",\"checks\":{\"streams_identical\":"
            << (streams_identical ? "true" : "false")
            << ",\"no_runtime_fallback\":"
            << (no_runtime_fallback ? "true" : "false")
            << ",\"terminal_ordinary_tail_bounded_to_d2\":"
            << (terminal_tail_bounded ? "true" : "false")
            << ",\"no_token_loop_allocations\":"
            << (no_token_loop_allocations ? "true" : "false")
            << "},\"pairs\":[";
  for (std::uint32_t index = 0U; index < kRetainedPairs; ++index) {
    if (index != 0U) std::cout << ',';
    std::cout << "{\"index\":" << index << ",\"ordinary\":";
    WriteRun(ordinary_runs[index]);
    std::cout << ",\"d2\":";
    WriteRun(d2_runs[index]);
    std::cout << '}';
  }
  std::cout << "]}\n";
  return streams_identical && no_runtime_fallback && terminal_tail_bounded &&
                 no_token_loop_allocations
             ? 0
             : 2;
}

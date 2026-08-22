#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "cuda/engine/gemma4_26b_reference.h"
#include "gem16/status.h"

namespace {

struct Options {
  std::filesystem::path model;
  std::filesystem::path output;
  std::filesystem::path logits;
  std::uint64_t context = 0U;
  std::uint64_t prompt_tokens = 0U;
  int device = 0;
};

bool ParseUnsigned(std::string_view text, std::uint64_t* output) {
  try {
    std::size_t used = 0U;
    const std::uint64_t value = std::stoull(std::string(text), &used);
    if (used != text.size()) return false;
    *output = value;
    return true;
  } catch (...) {
    return false;
  }
}

bool Parse(int argc, char** argv, Options* options) {
  for (int index = 1; index < argc; ++index) {
    if (index + 1 >= argc) return false;
    const std::string_view key(argv[index]);
    const std::string_view value(argv[++index]);
    if (key == "--model") {
      options->model = value;
    } else if (key == "--output") {
      options->output = value;
    } else if (key == "--logits") {
      options->logits = value;
    } else if (key == "--context") {
      if (!ParseUnsigned(value, &options->context)) return false;
    } else if (key == "--prompt-tokens") {
      if (!ParseUnsigned(value, &options->prompt_tokens)) return false;
    } else if (key == "--device") {
      std::uint64_t parsed = 0U;
      if (!ParseUnsigned(value, &parsed) ||
          parsed > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        return false;
      }
      options->device = static_cast<int>(parsed);
    } else {
      return false;
    }
  }
  if (options->context > 1U && options->prompt_tokens == 0U) {
    options->prompt_tokens = options->context - 1U;
  }
  return !options->model.empty() && !options->output.empty() &&
         !options->logits.empty() && options->context > 1U &&
         options->prompt_tokens > 0U &&
         options->prompt_tokens == options->context - 1U;
}

int Fail(std::string_view operation, const gem16::Status& status, int code) {
  std::cerr << operation << ": " << status.message() << '\n';
  return code;
}

int FailCuda(std::string_view operation, cudaError_t error, int code) {
  std::cerr << operation << ": " << cudaGetErrorName(error) << ": "
            << cudaGetErrorString(error) << '\n';
  return code;
}

double Milliseconds(std::chrono::steady_clock::time_point begin,
                    std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!Parse(argc, argv, &options)) {
    std::cerr
        << "usage: gem16-26b-m21-context-driver --model DIR --output JSON "
           "--logits F32LE --context N [--prompt-tokens N-1] [--device N]\n";
    return 2;
  }

  constexpr std::array<std::uint32_t, 20> kChatTokenPattern = {
      2U,     105U,  2364U, 107U, 40654U, 607U, 7121U,
      506U,   3658U, 3730U, 236761U, 106U, 107U, 105U,
      4368U,  107U,  100U,  45518U, 107U, 101U};
  std::vector<std::uint32_t> prompt(
      static_cast<std::size_t>(options.prompt_tokens));
  for (std::size_t index = 0; index < prompt.size(); ++index) {
    prompt[index] = kChatTokenPattern[index % kChatTokenPattern.size()];
  }
  std::vector<float> logits(262144U);

  cudaError_t cuda_status = cudaSetDevice(options.device);
  if (cuda_status != cudaSuccess) {
    return FailCuda("select M21 CUDA device", cuda_status, 3);
  }
  std::size_t free_before = 0U;
  std::size_t total = 0U;
  cuda_status = cudaMemGetInfo(&free_before, &total);
  if (cuda_status != cudaSuccess) {
    return FailCuda("measure M21 memory before engine", cuda_status, 3);
  }

  auto engine = gem16::internal::Gemma4Moe26BReferenceEngine::Create(
      options.model, options.context, options.device,
      gem16::internal::Gemma4Moe26BBackend::kSm120Integrated);
  if (!engine.ok()) return Fail("create M21 engine", engine.status(), 4);
  std::size_t free_after_create = 0U;
  cuda_status = cudaMemGetInfo(&free_after_create, &total);
  if (cuda_status != cudaSuccess) {
    return FailCuda("measure M21 memory after engine", cuda_status, 4);
  }

  const auto prefill_begin = std::chrono::steady_clock::now();
  gem16::Status status = engine.value().PrefillTokens(
      std::span<const std::uint32_t>(prompt));
  if (!status.ok()) return Fail("run M21 real prefill", status, 5);
  auto prefill_prediction = engine.value().Prediction();
  if (!prefill_prediction.ok()) {
    return Fail("read M21 prefill prediction", prefill_prediction.status(), 5);
  }
  const auto prefill_end = std::chrono::steady_clock::now();
  std::size_t free_after_prefill = 0U;
  cuda_status = cudaMemGetInfo(&free_after_prefill, &total);
  if (cuda_status != cudaSuccess) {
    return FailCuda("measure M21 memory after prefill", cuda_status, 5);
  }

  const std::uint32_t decode_token = prefill_prediction.value().token;
  const auto decode_begin = std::chrono::steady_clock::now();
  status = engine.value().ForwardToken(decode_token);
  if (!status.ok()) return Fail("run M21 boundary decode", status, 6);
  auto decode_prediction = engine.value().Prediction();
  if (!decode_prediction.ok()) {
    return Fail("read M21 decode prediction", decode_prediction.status(), 6);
  }
  status = engine.value().CopyLogits(logits);
  if (!status.ok()) return Fail("copy M21 boundary logits", status, 6);
  const auto decode_end = std::chrono::steady_clock::now();
  std::size_t free_after_decode = 0U;
  cuda_status = cudaMemGetInfo(&free_after_decode, &total);
  if (cuda_status != cudaSuccess) {
    return FailCuda("measure M21 memory after decode", cuda_status, 6);
  }

  const std::uint64_t boundary_position = engine.value().position();
  status = engine.value().ForwardToken(decode_prediction.value().token);
  const bool over_limit_rejected =
      !status.ok() && status.code() == gem16::StatusCode::kInvalidArgument &&
      engine.value().position() == boundary_position;
  if (!over_limit_rejected) {
    std::cerr << "M21 over-limit token was not rejected without state change\n";
    return 7;
  }

  std::ofstream binary(options.logits, std::ios::binary | std::ios::trunc);
  binary.write(reinterpret_cast<const char*>(logits.data()),
               static_cast<std::streamsize>(logits.size() * sizeof(float)));
  if (!binary) {
    std::cerr << "cannot write M21 diagnostic logits\n";
    return 8;
  }
  std::ofstream output(options.output, std::ios::binary | std::ios::trunc);
  if (!output) {
    std::cerr << "cannot write M21 diagnostic report\n";
    return 8;
  }
  const std::uint64_t required_margin =
      options.context >= 65536U ? 400U * 1024U * 1024U
                                : 700U * 1024U * 1024U;
  output << std::setprecision(9)
         << "{\"schema_version\":1,\"milestone\":\"M21\","
            "\"qualification_kind\":\"real_synthetic_token_execution\","
            "\"backend\":\"native_sm120_integrated\",\"context_tokens\":"
         << options.context << ",\"prompt_tokens\":" << options.prompt_tokens
         << ",\"final_position\":" << boundary_position
         << ",\"prefill_elapsed_ms\":"
         << Milliseconds(prefill_begin, prefill_end)
         << ",\"decode_elapsed_ms\":"
         << Milliseconds(decode_begin, decode_end)
         << ",\"prefill_prediction_token\":" << decode_token
         << ",\"decode_prediction_token\":"
         << decode_prediction.value().token
         << ",\"all_logits_finite\":"
         << (prefill_prediction.value().all_logits_finite &&
                     decode_prediction.value().all_logits_finite
                 ? "true"
                 : "false")
         << ",\"over_limit_rejected\":"
         << (over_limit_rejected ? "true" : "false")
         << ",\"sliding_cache_capacity\":"
         << engine.value().sliding_cache_capacity()
         << ",\"sliding_ring_wrap_exercised\":"
         << (options.prompt_tokens > engine.value().sliding_cache_capacity()
                 ? "true"
                 : "false")
         << ",\"global_extent_exercised\":"
         << (boundary_position == options.context ? "true" : "false")
         << ",\"prefill_chunk_count\":"
         << engine.value().prefill_chunk_count()
         << ",\"minimum_prefill_chunk_tokens\":"
         << engine.value().minimum_prefill_chunk_tokens()
         << ",\"memory\":{\"visible_total_bytes\":" << total
         << ",\"free_before_engine_bytes\":" << free_before
         << ",\"free_after_create_bytes\":" << free_after_create
         << ",\"free_after_prefill_bytes\":" << free_after_prefill
         << ",\"free_after_decode_bytes\":" << free_after_decode
         << ",\"required_margin_bytes\":" << required_margin
         << ",\"margin_pass\":"
         << (free_after_decode >= required_margin ? "true" : "false")
         << ",\"weight_arena_bytes\":"
         << engine.value().weight_arena_bytes()
         << ",\"kv_cache_bytes\":" << engine.value().kv_cache_bytes()
         << ",\"workspace_bytes\":" << engine.value().workspace_bytes()
         << "}}\n";
  return output ? 0 : 8;
}

#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "cuda/engine/gemma4_26b_reference.h"
#include "gem16/tokenizer.h"

namespace {

struct Options {
  std::filesystem::path model;
  std::filesystem::path tokenizer;
  std::filesystem::path output;
  std::filesystem::path logits;
  std::string prompt = "Reply with exactly OK.";
  std::string continuation = "Reply with exactly OK.";
  std::uint64_t context = 32768U;
  std::uint32_t max_new = 2U;
  int device = 0;
  gem16::internal::Gemma4Moe26BBackend backend =
      gem16::internal::Gemma4Moe26BBackend::kReference;
};

bool ParseUnsigned(std::string_view text, std::uint64_t* output) {
  try {
    std::size_t used = 0U;
    const auto value = std::stoull(std::string(text), &used);
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
    const std::string value(argv[++index]);
    if (key == "--model") options->model = value;
    else if (key == "--tokenizer") options->tokenizer = value;
    else if (key == "--output") options->output = value;
    else if (key == "--logits") options->logits = value;
    else if (key == "--prompt") options->prompt = value;
    else if (key == "--continuation") options->continuation = value;
    else if (key == "--backend") {
      if (value == "reference") {
        options->backend = gem16::internal::Gemma4Moe26BBackend::kReference;
      } else if (value == "sm120") {
        options->backend =
            gem16::internal::Gemma4Moe26BBackend::kSm120MoeHead;
      } else {
        return false;
      }
    }
    else if (key == "--context") {
      if (!ParseUnsigned(value, &options->context)) return false;
    } else if (key == "--max-new") {
      std::uint64_t parsed = 0U;
      if (!ParseUnsigned(value, &parsed) || parsed == 0U || parsed > 16U) {
        return false;
      }
      options->max_new = static_cast<std::uint32_t>(parsed);
    } else if (key == "--device") {
      try {
        std::size_t used = 0U;
        options->device = std::stoi(value, &used);
        if (used != value.size() || options->device < 0) return false;
      } catch (...) { return false; }
    } else return false;
  }
  return !options->model.empty() && !options->output.empty() &&
         !options->logits.empty();
}

template <typename T>
void Array(std::ostream& out, std::span<const T> values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0U) out << ',';
    out << values[i];
  }
  out << ']';
}

struct Capture {
  std::uint64_t position = 0U;
  std::uint32_t layer = 0U;
  std::vector<float> output = std::vector<float>(2816U);
  std::vector<float> probabilities = std::vector<float>(128U);
  std::array<std::uint32_t, 8> ids{};
};

struct RunResult {
  std::vector<std::uint32_t> generated;
  std::uint32_t continuation_prediction = 0U;
  std::uint64_t continuation_start = 0U;
  std::uint64_t continuation_end = 0U;
  bool finite = true;
};

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!Parse(argc, argv, &options)) {
    std::cerr << "usage: gem16-26b-reference --model DIR --output JSON "
                 "--logits F32LE [--tokenizer DIR] [--prompt TEXT] [--continuation TEXT] "
                 "[--context N] [--max-new N] [--device N] "
                 "[--backend reference|sm120]\n";
    return 2;
  }
  if (options.tokenizer.empty()) options.tokenizer = options.model;
  auto tokenizer = gem16::Tokenizer::Load(options.tokenizer / "tokenizer.json");
  if (!tokenizer.ok()) {
    std::cerr << tokenizer.status().message() << '\n';
    return 3;
  }
  const std::string prompt_text =
      "<bos><|turn>user\n" + options.prompt +
      "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
  const std::string continuation_text =
      "\n<|turn>user\n" + options.continuation +
      "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
  auto prompt = tokenizer.value().Encode(prompt_text);
  auto continuation = tokenizer.value().Encode(continuation_text);
  if (!prompt.ok() || !continuation.ok()) {
    std::cerr << "M13 tokenizer/template encoding failed\n";
    return 3;
  }
  auto engine = gem16::internal::Gemma4Moe26BReferenceEngine::Create(
      options.model, options.context, options.device, options.backend);
  if (!engine.ok()) {
    std::cerr << engine.status().message() << '\n';
    return 4;
  }
  const std::array<std::uint32_t, 4> layers{0U, 5U, 6U, 29U};
  std::vector<Capture> captures;
  captures.reserve(8U);
  std::vector<float> logits(262144U);
  std::size_t free_before = 0U, total = 0U;
  if (cudaMemGetInfo(&free_before, &total) != cudaSuccess) return 5;

  auto run = [&](bool capture, RunResult* result) -> gem16::Status {
    auto status = engine.value().Reset();
    if (!status.ok()) return status;
    for (std::size_t position = 0; position < prompt.value().size(); ++position) {
      status = engine.value().ForwardToken(prompt.value()[position]);
      if (!status.ok()) return status;
      auto prediction = engine.value().Prediction();
      if (!prediction.ok()) return prediction.status();
      result->finite = result->finite && prediction.value().all_logits_finite;
      if (capture && (position == 0U || position + 1U == prompt.value().size())) {
        for (const std::uint32_t layer : layers) {
          captures.emplace_back();
          Capture& item = captures.back();
          item.position = position;
          item.layer = layer;
          status = engine.value().CopyLayerOutput(layer, item.output);
          if (status.ok()) {
            status = engine.value().CopyRouterProbabilities(
                layer, item.probabilities);
          }
          if (status.ok()) status = engine.value().CopyRouterTopIds(layer, item.ids);
          if (!status.ok()) return status;
        }
      }
    }
    if (capture) {
      status = engine.value().CopyLogits(logits);
      if (!status.ok()) return status;
    }
    result->generated.reserve(options.max_new);
    for (std::uint32_t step = 0U; step < options.max_new; ++step) {
      auto prediction = engine.value().Prediction();
      if (!prediction.ok()) return prediction.status();
      result->finite = result->finite && prediction.value().all_logits_finite;
      result->generated.push_back(prediction.value().token);
      status = engine.value().ForwardToken(prediction.value().token);
      if (!status.ok()) return status;
    }
    result->continuation_start = engine.value().position();
    for (const std::uint32_t token : continuation.value()) {
      status = engine.value().ForwardToken(token);
      if (!status.ok()) return status;
    }
    result->continuation_end = engine.value().position();
    auto prediction = engine.value().Prediction();
    if (!prediction.ok()) return prediction.status();
    result->finite = result->finite && prediction.value().all_logits_finite;
    result->continuation_prediction = prediction.value().token;
    return gem16::Status::Ok();
  };

  RunResult first, second;
  const auto run_start = std::chrono::steady_clock::now();
  auto status = run(true, &first);
  std::size_t free_after_first = 0U;
  if (status.ok() &&
      cudaMemGetInfo(&free_after_first, &total) != cudaSuccess) {
    status = {gem16::StatusCode::kInternal,
              "cannot measure M13 memory after first warm run"};
  }
  if (status.ok()) status = run(false, &second);
  const auto run_end = std::chrono::steady_clock::now();
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 6;
  }
  std::size_t free_after = 0U;
  if (cudaMemGetInfo(&free_after, &total) != cudaSuccess) return 6;
  std::ofstream binary(options.logits, std::ios::binary | std::ios::trunc);
  binary.write(reinterpret_cast<const char*>(logits.data()),
               static_cast<std::streamsize>(logits.size() * sizeof(float)));
  if (!binary) return 7;
  std::ofstream out(options.output, std::ios::binary | std::ios::trunc);
  if (!out) return 7;
  out << std::setprecision(9)
      << "{\"schema_version\":1,\"milestone\":\""
      << (options.backend ==
                  gem16::internal::Gemma4Moe26BBackend::kSm120MoeHead
              ? "M16"
              : "M13")
      << "\",\"path\":\""
      << (options.backend ==
                  gem16::internal::Gemma4Moe26BBackend::kSm120MoeHead
              ? "native_sm120_moe_and_head"
              : "experimental_reference_only")
      << "\",\"two_run_elapsed_ms\":"
      << std::chrono::duration<double, std::milli>(run_end - run_start).count()
      << ','
      << "\"prompt_token_ids\":";
  Array(out, std::span<const std::uint32_t>(prompt.value()));
  out << ",\"continuation_token_ids\":";
  Array(out, std::span<const std::uint32_t>(continuation.value()));
  out << ",\"first_generated\":";
  Array(out, std::span<const std::uint32_t>(first.generated));
  out << ",\"second_generated\":";
  Array(out, std::span<const std::uint32_t>(second.generated));
  out << ",\"deterministic\":"
      << ((first.generated == second.generated &&
           first.continuation_prediction == second.continuation_prediction)
              ? "true" : "false")
      << ",\"all_logits_finite\":"
      << ((first.finite && second.finite) ? "true" : "false")
      << ",\"continuation\":{\"start_position\":"
      << first.continuation_start << ",\"end_position\":"
      << first.continuation_end << ",\"first_prediction\":"
      << first.continuation_prediction << ",\"second_prediction\":"
      << second.continuation_prediction << "},\"memory\":{"
      << "\"weight_arena_bytes\":" << engine.value().weight_arena_bytes()
      << ",\"kv_cache_bytes\":" << engine.value().kv_cache_bytes()
      << ",\"workspace_bytes\":" << engine.value().workspace_bytes()
      << ",\"free_before_runs_bytes\":" << free_before
      << ",\"free_after_first_run_bytes\":" << free_after_first
      << ",\"free_after_runs_bytes\":" << free_after
      << "},\"captures\":[";
  for (std::size_t i = 0; i < captures.size(); ++i) {
    if (i != 0U) out << ',';
    const Capture& item = captures[i];
    out << "{\"position\":" << item.position << ",\"layer\":"
        << item.layer << ",\"output\":";
    Array(out, std::span<const float>(item.output));
    out << ",\"router_probabilities\":";
    Array(out, std::span<const float>(item.probabilities));
    out << ",\"router_top_ids\":";
    Array(out, std::span<const std::uint32_t>(item.ids));
    out << '}';
  }
  out << "]}\n";
  return out ? 0 : 7;
}

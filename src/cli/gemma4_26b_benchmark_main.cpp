#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
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

#include "compiler/sha256.h"
#include "cuda/engine/gemma4_26b_reference.h"
#include "cuda/moe/router_diagnostic.h"
#include "gem16/sampling.h"
#include "gem16/tokenizer.h"
#include "util/json.h"

namespace {

struct Options {
  std::filesystem::path model;
  std::filesystem::path job;
  std::filesystem::path output;
  std::filesystem::path router_diagnostic_output;
  std::filesystem::path logits_dump;
  std::filesystem::path teacher_forced_reference;
  int device = 0;
  bool tensor_router_selected = true;
  bool router_selection_explicit = false;
};

const gem16::json::Value* Field(const gem16::json::Value& value,
                                std::string_view name) {
  return value.is_object() ? value.find(name) : nullptr;
}

bool Parse(int argc, char** argv, Options* options) {
  for (int index = 1; index < argc; ++index) {
    if (index + 1 >= argc) return false;
    const std::string_view key(argv[index]);
    const std::string value(argv[++index]);
    if (key == "--model") options->model = value;
    else if (key == "--benchmark-job") options->job = value;
    else if (key == "--output") options->output = value;
    else if (key == "--router-diagnostic-output") {
      options->router_diagnostic_output = value;
    }
    else if (key == "--logits-dump") options->logits_dump = value;
    else if (key == "--teacher-forced-reference") {
      options->teacher_forced_reference = value;
    }
    else if (key == "--router-selection") {
      options->router_selection_explicit = true;
      if (value == "exact") options->tensor_router_selected = false;
      else if (value == "tensor-core" || value == "tensor-diagnostic") {
        options->tensor_router_selected = true;
      } else {
        return false;
      }
    }
    else if (key == "--device") {
      try {
        std::size_t used = 0U;
        options->device = std::stoi(value, &used);
        if (used != value.size() || options->device < 0) return false;
      } catch (...) {
        return false;
      }
    } else {
      return false;
    }
  }
  return !options->model.empty() && !options->job.empty() &&
         !options->output.empty();
}

gem16::Result<std::string> ReadRegular(const std::filesystem::path& path,
                                       std::uint64_t limit) {
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  if (error || std::filesystem::is_symlink(status) ||
      !std::filesystem::is_regular_file(status)) {
    return gem16::Status(gem16::StatusCode::kInvalidArgument,
                         "benchmark input is not a regular file: " +
                             path.string());
  }
  const std::uint64_t size = std::filesystem::file_size(path, error);
  if (error || size > limit ||
      size > static_cast<std::uint64_t>(
                 std::numeric_limits<std::size_t>::max())) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         "benchmark input exceeds its size limit: " +
                             path.string());
  }
  std::string result(static_cast<std::size_t>(size), '\0');
  std::ifstream input(path, std::ios::binary);
  input.read(result.data(), static_cast<std::streamsize>(result.size()));
  if (!input || input.gcount() != static_cast<std::streamsize>(result.size())) {
    return gem16::Status(gem16::StatusCode::kIoError,
                         "cannot read benchmark input: " + path.string());
  }
  return result;
}

gem16::Result<std::uint64_t> Unsigned(const gem16::json::Value* value,
                                      std::string_view name) {
  if (value == nullptr || !value->is_integer() || value->as_integer() < 0) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         std::string(name) + " must be unsigned");
  }
  return static_cast<std::uint64_t>(value->as_integer());
}

gem16::Result<std::string> Text(const gem16::json::Value* value,
                                std::string_view name) {
  if (value == nullptr || !value->is_string() || value->as_string().empty()) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         std::string(name) + " must be text");
  }
  return value->as_string();
}

double Milliseconds(std::chrono::steady_clock::time_point begin,
                    std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

std::uint64_t TokenChecksum(std::span<const std::uint32_t> tokens) {
  std::uint64_t value = 14695981039346656037ULL;
  for (const std::uint32_t token : tokens) {
    for (unsigned shift = 0U; shift < 32U; shift += 8U) {
      value ^= static_cast<std::uint8_t>(token >> shift);
      value *= 1099511628211ULL;
    }
  }
  return value;
}

void WriteSampling(std::ostream& output,
                   const gem16::json::Value& sampling) {
  output << gem16::json::Stringify(sampling);
}

int Fail(std::string_view operation, const gem16::Status& status, int code) {
  std::cerr << operation << ": " << status.message() << '\n';
  return code;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!Parse(argc, argv, &options)) {
    std::cerr << "usage: gem16-26b-benchmark --model DIR --benchmark-job JOB.json "
                 "--output SAMPLE.json [--router-diagnostic-output JSON] "
                 "[--router-selection exact|tensor-core] "
                 "[--logits-dump F32LE] "
                 "[--teacher-forced-reference SAMPLE.json] "
                 "[--device N]\n";
    return 2;
  }
  std::error_code output_error;
  if (std::filesystem::exists(options.output, output_error) || output_error) {
    std::cerr << "M20 output already exists or cannot be inspected\n";
    return 2;
  }
  if (!options.router_diagnostic_output.empty() &&
      (std::filesystem::exists(options.router_diagnostic_output,
                               output_error) ||
       output_error)) {
    std::cerr << "router diagnostic output already exists or cannot be inspected\n";
    return 2;
  }
  if (!options.logits_dump.empty() &&
      (std::filesystem::exists(options.logits_dump, output_error) ||
       output_error)) {
    std::cerr << "logits dump already exists or cannot be inspected\n";
    return 2;
  }
  auto job_text = ReadRegular(options.job, 16U * 1024U * 1024U);
  if (!job_text.ok()) return Fail("read M20 job", job_text.status(), 3);
  auto job = gem16::json::Parse(job_text.value());
  if (!job.ok() || !job.value().is_object()) {
    return Fail("parse M20 job",
                job.ok() ? gem16::Status(gem16::StatusCode::kDataLoss,
                                         "M20 job root is not an object")
                         : job.status(),
                3);
  }
  auto schema = Unsigned(Field(job.value(), "schema_version"), "schema_version");
  const auto* scenario = Field(job.value(), "scenario");
  if (!schema.ok() || schema.value() != 1U || scenario == nullptr ||
      !scenario->is_object()) {
    return Fail("validate M20 job",
                gem16::Status(gem16::StatusCode::kDataLoss,
                              "M20 job schema/scenario is invalid"),
                3);
  }
  auto prompt_count = Unsigned(Field(*scenario, "prompt_tokens"),
                               "scenario.prompt_tokens");
  auto output_forwards = Unsigned(Field(*scenario, "output_forwards"),
                                  "scenario.output_forwards");
  auto context = Unsigned(Field(*scenario, "context_tokens"),
                          "scenario.context_tokens");
  auto prompt_path = Text(Field(*scenario, "prompt_manifest_path"),
                          "scenario.prompt_manifest_path");
  auto prompt_hash = Text(Field(*scenario, "prompt_manifest_sha256"),
                          "scenario.prompt_manifest_sha256");
  const auto* sampling_json = Field(*scenario, "sampling");
  constexpr std::uint64_t kMaximumContext = 262144U;
  if (!prompt_count.ok() || !output_forwards.ok() || !context.ok() ||
      !prompt_path.ok() || !prompt_hash.ok() || sampling_json == nullptr ||
      !sampling_json->is_object() || prompt_count.value() == 0U ||
      prompt_count.value() > kMaximumContext || output_forwards.value() < 2U ||
      output_forwards.value() > kMaximumContext || context.value() == 0U ||
      context.value() > kMaximumContext ||
      output_forwards.value() - 1U > context.value() - prompt_count.value()) {
    return Fail("validate M20 scenario",
                gem16::Status(gem16::StatusCode::kDataLoss,
                              "M20 scenario geometry is invalid"),
                3);
  }
  auto prompt_text = ReadRegular(prompt_path.value(), 4U * 1024U * 1024U);
  if (!prompt_text.ok()) return Fail("read M20 prompt", prompt_text.status(), 3);
  if (gem16::compiler::Sha256Hex(prompt_text.value().data(),
                                 prompt_text.value().size()) !=
      prompt_hash.value()) {
    return Fail("bind M20 prompt",
                gem16::Status(gem16::StatusCode::kDataLoss,
                              "M20 prompt manifest hash mismatch"),
                3);
  }
  auto prompt_document = gem16::json::Parse(prompt_text.value());
  const auto* token_array = prompt_document.ok()
                                ? Field(prompt_document.value(), "token_ids")
                                : nullptr;
  if (!prompt_document.ok() || token_array == nullptr ||
      !token_array->is_array() ||
      token_array->as_array().size() != prompt_count.value()) {
    return Fail("parse M20 prompt",
                gem16::Status(gem16::StatusCode::kDataLoss,
                              "M20 prompt token manifest is invalid"),
                3);
  }
  std::vector<std::uint32_t> tokens;
  tokens.reserve(token_array->as_array().size());
  for (const auto& token : token_array->as_array()) {
    auto parsed = Unsigned(&token, "prompt token");
    if (!parsed.ok() || parsed.value() >= 262144U) {
      return Fail("validate M20 prompt token",
                  parsed.ok() ? gem16::Status(
                                    gem16::StatusCode::kDataLoss,
                                    "M20 prompt token exceeds vocabulary")
                              : parsed.status(),
                  3);
    }
    tokens.push_back(static_cast<std::uint32_t>(parsed.value()));
  }
  std::vector<std::uint32_t> forced_output_tokens;
  if (!options.teacher_forced_reference.empty()) {
    auto reference_text =
        ReadRegular(options.teacher_forced_reference, 64U * 1024U * 1024U);
    auto reference =
        reference_text.ok()
            ? gem16::json::Parse(reference_text.value())
            : gem16::Result<gem16::json::Value>(reference_text.status());
    const auto* correctness =
        reference.ok() ? Field(reference.value(), "correctness") : nullptr;
    const auto* reference_tokens =
        correctness != nullptr && correctness->is_object()
            ? Field(*correctness, "output_token_ids")
            : (reference.ok()
                   ? Field(reference.value(), "generated_token_ids")
                   : nullptr);
    if (!reference.ok() || reference_tokens == nullptr ||
        !reference_tokens->is_array() ||
        reference_tokens->as_array().size() != output_forwards.value()) {
      return Fail("read teacher-forced reference",
                  gem16::Status(gem16::StatusCode::kDataLoss,
                                "teacher-forced reference token geometry is invalid"),
                  3);
    }
    forced_output_tokens.reserve(reference_tokens->as_array().size());
    for (const auto& value : reference_tokens->as_array()) {
      auto parsed = Unsigned(&value, "teacher-forced token");
      if (!parsed.ok() || parsed.value() >= 262144U) {
        return Fail("read teacher-forced reference",
                    parsed.ok()
                        ? gem16::Status(gem16::StatusCode::kDataLoss,
                                        "teacher-forced token exceeds vocabulary")
                        : parsed.status(),
                    3);
      }
      forced_output_tokens.push_back(
          static_cast<std::uint32_t>(parsed.value()));
    }
  }

  gem16::SamplingOptions sampling;
  auto mode = Text(Field(*sampling_json, "mode"), "sampling.mode");
  if (!mode.ok() || (mode.value() != "greedy" && mode.value() != "sampled")) {
    return Fail("validate M20 sampling",
                gem16::Status(gem16::StatusCode::kDataLoss,
                              "M20 sampling mode is invalid"),
                3);
  }
  if (mode.value() == "sampled") {
    const auto* temperature = Field(*sampling_json, "temperature");
    const auto* top_p = Field(*sampling_json, "top_p");
    auto top_k = Unsigned(Field(*sampling_json, "top_k"), "sampling.top_k");
    auto seed = Unsigned(Field(*sampling_json, "seed"), "sampling.seed");
    if (temperature == nullptr || !temperature->is_number() ||
        top_p == nullptr || !top_p->is_number() || !top_k.ok() ||
        !seed.ok() || top_k.value() > 262144U) {
      return Fail("validate M20 sampling",
                  gem16::Status(gem16::StatusCode::kDataLoss,
                                "M20 sampled controls are invalid"),
                  3);
    }
    sampling.enabled = true;
    sampling.temperature = static_cast<float>(temperature->as_number());
    sampling.top_p = static_cast<float>(top_p->as_number());
    sampling.top_k = static_cast<std::uint32_t>(top_k.value());
    sampling.seed = seed.value();
  }

  auto lock_text = ReadRegular(
      options.model.parent_path() /
          (options.model.filename().string() + ".lock.json"),
      16U * 1024U * 1024U);
  auto lock = lock_text.ok() ? gem16::json::Parse(lock_text.value())
                             : gem16::Result<gem16::json::Value>(
                                   lock_text.status());
  auto artifact_hash = lock.ok()
                           ? Text(Field(lock.value(), "artifact_content_sha256"),
                                  "artifact_content_sha256")
                           : gem16::Result<std::string>(lock.status());
  auto source_hash = lock.ok()
                         ? Text(Field(lock.value(), "source_lock_sha256"),
                                "source_lock_sha256")
                         : gem16::Result<std::string>(lock.status());
  if (!artifact_hash.ok() || !source_hash.ok()) {
    return Fail("read M20 artifact lock",
                artifact_hash.ok() ? source_hash.status()
                                   : artifact_hash.status(),
                3);
  }
  const std::string lock_hash = gem16::compiler::Sha256Hex(
      lock_text.value().data(), lock_text.value().size());

  cudaError_t cuda_status = cudaSetDevice(options.device);
  if (cuda_status != cudaSuccess) {
    std::cerr << "select M20 device: " << cudaGetErrorString(cuda_status) << '\n';
    return 4;
  }
  std::size_t free_before = 0U;
  std::size_t total = 0U;
  if (cudaMemGetInfo(&free_before, &total) != cudaSuccess) return 4;
  auto engine = gem16::internal::Gemma4Moe26BReferenceEngine::Create(
      options.model, context.value(), options.device,
      gem16::internal::Gemma4Moe26BBackend::kSm120Integrated);
  if (!engine.ok()) return Fail("create M20 engine", engine.status(), 4);
  gem16::Status status =
      engine.value().ConfigureTokenSelection(sampling, {});
  if (!status.ok()) return Fail("configure M20 selection", status, 4);
  status = engine.value().ConfigurePrefillRouter(
      options.tensor_router_selected
          ? gem16::internal::Gemma4MoePrefillRouter::kSm120TensorCore
          : gem16::internal::Gemma4MoePrefillRouter::kSerialExact);
  if (!status.ok()) return Fail("configure M20 prefill router", status, 4);

  std::size_t free_after_create = 0U;
  if (cudaMemGetInfo(&free_after_create, &total) != cudaSuccess) return 4;
  const auto request_begin = std::chrono::steady_clock::now();
  const auto prompt_begin = request_begin;
  if (!options.router_diagnostic_output.empty()) {
    gem16::internal::SetGemma4RouterComparisonEnabled(true);
    status = gem16::internal::ResetGemma4RouterComparison(nullptr);
    if (!status.ok()) return Fail("reset router comparison", status, 5);
  }
  status = engine.value().PrefillTokens(tokens);
  if (!status.ok()) return Fail("run M20 prefill", status, 5);
  // PrefillTokens enqueues work on the engine stream. Prediction is the
  // explicit synchronization and finite-output boundary; measure through it
  // so prompt throughput cannot report host launch latency as GPU execution.
  auto prefill_prediction = engine.value().Prediction();
  if (!prefill_prediction.ok()) {
    return Fail("synchronize M20 prefill", prefill_prediction.status(), 5);
  }
  gem16::Result<gem16::internal::Gemma4RouterComparisonSummary>
      router_comparison = gem16::Status(
          gem16::StatusCode::kUnsupported,
          "router comparison was not requested");
  if (!options.router_diagnostic_output.empty()) {
    router_comparison =
        gem16::internal::CopyGemma4RouterComparison(nullptr);
    gem16::internal::SetGemma4RouterComparisonEnabled(false);
    if (!router_comparison.ok()) {
      return Fail("copy router comparison", router_comparison.status(), 5);
    }
  }
  std::vector<float> dumped_logits;
  if (!options.logits_dump.empty()) {
    dumped_logits.resize(static_cast<std::size_t>(output_forwards.value()) *
                         262144U);
    status = engine.value().CopyLogits(
        std::span<float>(dumped_logits).first(262144U));
    if (!status.ok()) return Fail("copy prefill logits", status, 5);
  }
  const auto prompt_end = std::chrono::steady_clock::now();
  auto selected = engine.value().SelectToken();
  if (!selected.ok()) return Fail("select M20 first token", selected.status(), 5);
  const auto first_token_end = std::chrono::steady_clock::now();
  std::vector<std::uint32_t> output_tokens{selected.value()};
  std::vector<double> intervals;
  intervals.reserve(static_cast<std::size_t>(output_forwards.value() - 1U));
  std::size_t free_after_prefill = 0U;
  if (cudaMemGetInfo(&free_after_prefill, &total) != cudaSuccess) return 5;
  for (std::uint64_t index = 1U; index < output_forwards.value(); ++index) {
    const auto interval_begin = std::chrono::steady_clock::now();
    const std::uint32_t input_token =
        forced_output_tokens.empty()
            ? output_tokens.back()
            : forced_output_tokens[static_cast<std::size_t>(index - 1U)];
    status = engine.value().ForwardToken(input_token);
    if (!status.ok()) return Fail("run M20 decode", status, 6);
    selected = engine.value().SelectToken();
    if (!selected.ok()) return Fail("select M20 decode token", selected.status(), 6);
    if (!options.logits_dump.empty()) {
      status = engine.value().CopyLogits(std::span<float>(dumped_logits)
                                             .subspan(index * 262144U,
                                                      262144U));
      if (!status.ok()) return Fail("copy decode logits", status, 6);
    }
    const auto interval_end = std::chrono::steady_clock::now();
    output_tokens.push_back(selected.value());
    intervals.push_back(Milliseconds(interval_begin, interval_end));
  }
  std::size_t free_after_decode = 0U;
  if (cudaMemGetInfo(&free_after_decode, &total) != cudaSuccess) return 6;
  double decode_ms = 0.0;
  for (const double interval : intervals) decode_ms += interval;
  const std::string output_hash = gem16::compiler::Sha256Hex(
      output_tokens.data(), output_tokens.size() * sizeof(std::uint32_t));
  auto tokenizer = gem16::Tokenizer::Load(options.model / "tokenizer.json");
  if (!tokenizer.ok()) {
    return Fail("load M20 output tokenizer", tokenizer.status(), 6);
  }
  auto decoded_output = tokenizer.value().Decode(output_tokens, false);
  if (!decoded_output.ok()) {
    return Fail("decode M20 output tokens", decoded_output.status(), 6);
  }
  const std::uint64_t used_peak = total -
      std::min({free_before, free_after_create, free_after_prefill,
                free_after_decode});
  const auto evidence = engine.value().execution_evidence();
  if (!evidence.integrated_native_backend || !evidence.decode_graph_ready ||
      evidence.prefill_calls != 1U ||
      evidence.decode_graph_launches != output_forwards.value() - 1U ||
      evidence.token_selections != output_forwards.value() ||
      evidence.fallback_count != 0U ||
      evidence.recurring_allocation_count != 0U) {
    return Fail("validate M20 execution evidence",
                gem16::Status(gem16::StatusCode::kInternal,
                              "M20 engine observations do not match the requested native run"),
                6);
  }
  // Every successful SelectToken call completed Prediction(), which rejects
  // non-finite router values or output logits before incrementing this
  // source-backed counter.
  const bool all_logits_finite =
      evidence.token_selections == output_forwards.value();
  const bool diagnostic_mode = options.router_selection_explicit ||
                               !options.router_diagnostic_output.empty() ||
                               !options.logits_dump.empty() ||
                               !options.teacher_forced_reference.empty();

  std::ofstream output(options.output, std::ios::binary | std::ios::trunc);
  if (!output) return 7;
  if (!options.logits_dump.empty()) {
    std::ofstream logits_output(options.logits_dump,
                                std::ios::binary | std::ios::trunc);
    logits_output.write(
        reinterpret_cast<const char*>(dumped_logits.data()),
        static_cast<std::streamsize>(dumped_logits.size() * sizeof(float)));
    if (!logits_output) return 7;
  }
  output << std::setprecision(12)
         << "{\"schema_version\":1,\"status\":"
         << gem16::json::Quote(diagnostic_mode ? "diagnostic_only" : "ok")
         << ",\"performance_eligible\":"
         << (diagnostic_mode ? "false" : "true") << ",\"model\":{"
            "\"profile\":\"native_sm120_integrated_prefill_decode_head\","
            "\"artifact_content_sha256\":"
         << gem16::json::Quote(artifact_hash.value())
         << ",\"artifact_lock_sha256\":" << gem16::json::Quote(lock_hash)
         << ",\"source_lock_sha256\":"
         << gem16::json::Quote(source_hash.value())
         << "},\"correctness\":{\"all_logits_finite\":"
         << (all_logits_finite ? "true" : "false") << ","
            "\"finite_checks_completed\":"
         << evidence.token_selections << ","
            "\"prompt_manifest_sha256\":"
         << gem16::json::Quote(prompt_hash.value())
         << ",\"output_token_sha256\":" << gem16::json::Quote(output_hash)
         << ",\"teacher_forced_context\":"
         << (forced_output_tokens.empty() ? "false" : "true")
         << ",\"output_checksum\":" << TokenChecksum(output_tokens)
         << ",\"output_token_ids\":[";
  for (std::size_t index = 0U; index < output_tokens.size(); ++index) {
    if (index != 0U) output << ',';
    output << output_tokens[index];
  }
  output << "],\"output_text\":"
         << gem16::json::Quote(decoded_output.value())
         << "},\"runtime_path\":{"
            "\"model_variant\":\"gemma4-26b-a4b\","
            "\"head_format\":\"nvfp4\",\"kv_mode\":\"fp8\","
            "\"backend\":\"sm120\",\"router_selection\":"
         << gem16::json::Quote(evidence.tensor_core_prefill_router
                                   ? "sm120_bf16_tensor_core"
                                   : "serial_exact")
         << ",\"prompt_cache\":false,"
            "\"cpu_weight_offload\":false,\"token_loop_allocations\":"
         << (evidence.recurring_allocation_count == 0U ? "false" : "true")
         << ",\"native_instruction_capability\":true,\"fallback_count\":"
         << evidence.fallback_count << ","
            "\"cuda_graph\":{\"enabled\":"
         << (evidence.decode_graph_ready ? "true" : "false")
         << ","
            "\"first_demotion_reason\":\"none\"},"
            "\"resolved_dispatch\":{"
            "\"attention_prefill\":\"native_fixed_sm120\","
            "\"attention_decode\":\"native_fixed_sm120\","
            "\"moe_decode\":\"native_sm120\","
            "\"moe_prefill\":\"native_grouped_sm120\","
            "\"embedding_head\":\"native_sm120\"},"
            "\"observations\":{\"prefill_calls\":"
         << evidence.prefill_calls << ",\"prefill_chunks\":"
         << evidence.prefill_chunks << ",\"decode_graph_launches\":"
         << evidence.decode_graph_launches << ",\"token_selections\":"
         << evidence.token_selections << ",\"sliding_ring_wraps\":"
         << evidence.sliding_ring_wraps
         << ",\"maximum_global_position_exclusive\":"
         << evidence.maximum_global_position_exclusive
         << ",\"recurring_allocation_count\":"
         << evidence.recurring_allocation_count << "}},"
            "\"performance\":{\"prompt_tokens\":"
         << prompt_count.value() << ",\"output_forwards\":"
         << output_forwards.value() << ",\"sampling\":";
  WriteSampling(output, *sampling_json);
  output << ",\"prompt_ms\":" << Milliseconds(prompt_begin, prompt_end)
         << ",\"ttft_ms\":" << Milliseconds(request_begin, first_token_end)
         << ",\"decode_ms\":" << decode_ms << ",\"decode_tps\":"
         << static_cast<double>(output_forwards.value() - 1U) * 1000.0 /
                decode_ms
         << ",\"itl_ms\":[";
  for (std::size_t index = 0U; index < intervals.size(); ++index) {
    if (index != 0U) output << ',';
    output << intervals[index];
  }
  output << "]},\"memory\":{\"sampled_device_used_bytes\":" << used_peak
         << ",\"margin_bytes\":" << free_after_decode
         << ",\"recurring_allocation_observed\":"
         << (free_after_prefill == free_after_decode ? "false" : "true")
         << "}}\n";
  if (!output) return 7;
  if (!options.router_diagnostic_output.empty()) {
    const auto& comparison = router_comparison.value();
    std::ofstream diagnostic(options.router_diagnostic_output,
                             std::ios::binary | std::ios::trunc);
    if (!diagnostic) return 7;
    diagnostic << std::setprecision(12)
               << "{\"schema_version\":1,\"status\":\"diagnostic_only\","
               << "\"selected_router\":"
               << gem16::json::Quote(options.tensor_router_selected
                                          ? "sm120_bf16_tensor_core"
                                          : "serial_exact")
               << ','
               << "\"comparison_router\":\"sm120_bf16_tensor_core\","
               << "\"cases\":" << comparison.cases
               << ",\"top8_set_matches\":"
               << comparison.top8_set_matches
               << ",\"top8_order_matches\":"
               << comparison.top8_order_matches
               << ",\"changed_top8_slots\":"
               << comparison.changed_top8_slots
               << ",\"flip_cases\":" << comparison.flip_cases
               << ",\"flip_rate\":"
               << (comparison.cases == 0U
                       ? 0.0
                       : static_cast<double>(comparison.flip_cases) /
                             static_cast<double>(comparison.cases))
               << ",\"mean_flip_margin_8_9\":"
               << (comparison.flip_cases == 0U
                       ? 0.0
                       : comparison.flip_margin_sum /
                             static_cast<double>(comparison.flip_cases))
               << ",\"maximum_flip_margin_8_9\":"
               << comparison.maximum_flip_margin_8_9
               << ",\"maximum_tensor_flip_margin_8_9\":"
               << comparison.maximum_tensor_flip_margin_8_9
               << ",\"mean_gating_l1\":"
               << (comparison.cases == 0U
                       ? 0.0
                       : comparison.gating_l1_sum /
                             static_cast<double>(comparison.cases))
               << ",\"maximum_gating_l1\":"
               << comparison.maximum_gating_l1
               << ",\"maximum_logit_absolute_delta\":"
               << comparison.maximum_logit_absolute_delta;
    const auto write_histogram = [&](std::string_view name,
                                     const auto& histogram) {
      diagnostic << ',' << gem16::json::Quote(name) << ":[";
      for (std::size_t index = 0U; index < histogram.size(); ++index) {
        if (index != 0U) diagnostic << ',';
        diagnostic << histogram[index];
      }
      diagnostic << ']';
    };
    write_histogram("flip_margin_histogram",
                    comparison.flip_margin_histogram);
    write_histogram("tensor_flip_margin_histogram",
                    comparison.tensor_flip_margin_histogram);
    write_histogram("serial_margin_histogram",
                    comparison.serial_margin_histogram);
    write_histogram("tensor_margin_histogram",
                    comparison.tensor_margin_histogram);
    diagnostic << "}\n";
    if (!diagnostic) return 7;
  }
  return 0;
}

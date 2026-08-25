#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string_view>
#include <vector>

#include "cuda/engine/gemma4_26b_reference.h"

namespace {

struct Options {
  std::filesystem::path target;
  std::filesystem::path assistant;
  std::filesystem::path output;
  std::filesystem::path oracle_directory;
  std::uint64_t context = 32768U;
  int device = 0;
  bool profile_one_proposal = false;
};

bool Unsigned(std::string_view text, std::uint64_t* value) {
  try {
    std::size_t used = 0;
    const auto parsed = std::stoull(std::string(text), &used);
    if (used != text.size()) return false;
    *value = parsed;
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
    if (key == "--target") options->target = value;
    else if (key == "--assistant") options->assistant = value;
    else if (key == "--output") options->output = value;
    else if (key == "--oracle-dir") options->oracle_directory = value;
    else if (key == "--context") {
      if (!Unsigned(value, &options->context)) return false;
    } else if (key == "--device") {
      std::uint64_t parsed = 0U;
      if (!Unsigned(value, &parsed) || parsed > 1024U) return false;
      options->device = static_cast<int>(parsed);
    } else if (key == "--profile-one-proposal") {
      if (value == "1") options->profile_one_proposal = true;
      else if (value != "0") return false;
    } else {
      return false;
    }
  }
  return !options->target.empty() && !options->assistant.empty() &&
         !options->output.empty() &&
         (options->context == 32768U || options->context == 65536U);
}

int Fail(const gem16::Status& status) {
  std::cerr << status.message() << '\n';
  return 1;
}

template <typename T>
bool Write(const std::filesystem::path& path, std::span<const T> values) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) return false;
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size_bytes()));
  return static_cast<bool>(output);
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!Parse(argc, argv, &options)) {
    std::cerr << "Usage: gem16-26b-assistant-smoke --target DIR "
                 "--assistant DIR --output JSON [--context 32768|65536] "
                 "[--oracle-dir DIR] [--device N] "
                 "[--profile-one-proposal 0|1]\n";
    return 2;
  }
  if (std::filesystem::exists(options.output)) {
    std::cerr << "refusing to overwrite smoke report\n";
    return 2;
  }
  if (!options.oracle_directory.empty() &&
      std::filesystem::exists(options.oracle_directory)) {
    std::cerr << "refusing to overwrite oracle fixture directory\n";
    return 2;
  }
  if (cudaSetDevice(options.device) != cudaSuccess) return 3;
  std::size_t free_before = 0U;
  std::size_t total = 0U;
  if (cudaMemGetInfo(&free_before, &total) != cudaSuccess) return 3;
  auto engine = gem16::internal::Gemma4Moe26BReferenceEngine::Create(
      options.target, options.context, options.device,
      gem16::internal::Gemma4Moe26BBackend::kSm120Integrated);
  if (!engine.ok()) return Fail(engine.status());
  auto status = engine.value().LoadMtpAssistant(options.assistant);
  if (!status.ok()) return Fail(status);
  std::size_t free_loaded = 0U;
  if (cudaMemGetInfo(&free_loaded, &total) != cudaSuccess) return 3;

  if (!options.oracle_directory.empty()) {
    status = engine.value().Reset();
    if (!status.ok()) return Fail(status);
    status = engine.value().ForwardToken(2U);
    if (!status.ok()) return Fail(status);
    auto pending_token = engine.value().SelectToken();
    if (!pending_token.ok()) return Fail(pending_token.status());
    std::array<std::uint32_t, 1> oracle_draft{};
    status = engine.value().GenerateMtpAssistantDraftsForPending(
        pending_token.value(), oracle_draft);
    if (!status.ok()) return Fail(status);
    std::vector<float> concatenated(5632U);
    std::vector<float> logits(262144U);
    std::vector<std::uint8_t> sliding_key(8U * 256U);
    std::vector<std::uint8_t> sliding_value(8U * 256U);
    std::vector<std::uint8_t> full_key(2U * 512U);
    std::vector<std::uint8_t> full_value(2U * 512U);
    std::array<std::uint16_t, 4> scales{};
    status = engine.value().CopyMtpAssistantOracleInputs(
        concatenated, logits, sliding_key, sliding_value, full_key,
        full_value, scales);
    if (!status.ok()) return Fail(status);
    status = engine.value().ForwardToken(pending_token.value());
    if (!status.ok()) return Fail(status);
    auto target_token = engine.value().SelectToken();
    if (!target_token.ok()) return Fail(target_token.status());
    if (!std::filesystem::create_directories(options.oracle_directory) ||
        !Write<float>(options.oracle_directory / "concatenated.f32",
                      concatenated) ||
        !Write<float>(options.oracle_directory / "hybrid-logits.f32", logits) ||
        !Write<std::uint8_t>(options.oracle_directory / "sliding-key.e4m3fn",
                            sliding_key) ||
        !Write<std::uint8_t>(options.oracle_directory / "sliding-value.e4m3fn",
                            sliding_value) ||
        !Write<std::uint8_t>(options.oracle_directory / "full-key.e4m3fn",
                            full_key) ||
        !Write<std::uint8_t>(options.oracle_directory / "full-value.e4m3fn",
                            full_value) ||
        !Write<std::uint16_t>(options.oracle_directory / "kv-scales.bf16bits",
                             scales)) {
      std::cerr << "failed to write M25 oracle fixture\n";
      return 5;
    }
    std::ofstream metadata(options.oracle_directory / "fixture.json",
                           std::ios::binary | std::ios::trunc);
    metadata << "{\n"
             << "  \"schema_version\": 1,\n"
             << "  \"status\": \"diagnostic_oracle_input\",\n"
             << "  \"input_token\": " << pending_token.value() << ",\n"
             << "  \"position\": 0,\n"
             << "  \"tokens\": 1,\n"
             << "  \"target_token\": " << target_token.value() << ",\n"
             << "  \"hybrid_draft_token\": " << oracle_draft[0] << ",\n"
             << "  \"hybrid_target_match\": "
             << (oracle_draft[0] == target_token.value() ? "true" : "false")
             << "\n"
             << "}\n";
    if (!metadata) return 5;
  }

  std::array<std::uint32_t, 2> first{};
  std::array<std::uint32_t, 2> second{};
  const auto started = std::chrono::steady_clock::now();
  for (auto* drafts : {&first, &second}) {
    status = engine.value().Reset();
    if (!status.ok()) return Fail(status);
    status = engine.value().ForwardToken(2U);
    if (!status.ok()) return Fail(status);
    const bool profile = options.profile_one_proposal && drafts == &first;
    if (profile && cudaProfilerStart() != cudaSuccess) return 3;
    status = engine.value().GenerateMtpAssistantDrafts(*drafts);
    if (profile && cudaProfilerStop() != cudaSuccess) return 3;
    if (!status.ok()) return Fail(status);
  }
  const auto finished = std::chrono::steady_clock::now();
  const bool deterministic = first == second;
  if (!deterministic) {
    std::cerr << "M25 Assistant reset determinism failed\n";
    return 4;
  }

  std::ofstream output(options.output, std::ios::binary | std::ios::trunc);
  if (!output) return 5;
  output << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"status\": \"diagnostic_pass_not_acceptance\",\n"
         << "  \"context_tokens\": " << options.context << ",\n"
         << "  \"draft_count\": 2,\n"
         << "  \"draft_tokens\": [" << first[0] << ", " << first[1]
         << "],\n"
         << "  \"reset_deterministic\": true,\n"
         << "  \"target_kv_cache_shared\": true,\n"
         << "  \"assistant_weight_bytes\": "
         << engine.value().mtp_assistant_weight_bytes() << ",\n"
         << "  \"assistant_workspace_bytes\": "
         << engine.value().mtp_assistant_workspace_bytes() << ",\n"
         << "  \"free_before_bytes\": " << free_before << ",\n"
         << "  \"free_loaded_bytes\": " << free_loaded << ",\n"
         << "  \"two_runs_seconds\": "
         << std::chrono::duration<double>(finished - started).count() << "\n"
         << "}\n";
  return output ? 0 : 5;
}

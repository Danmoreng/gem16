#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "gem16/engine.h"

namespace {

constexpr std::uint64_t kMiB = 1024U * 1024U;

struct Options {
  std::filesystem::path model;
  std::filesystem::path assistant;
  std::filesystem::path output;
  std::uint64_t context = 0U;
  std::uint64_t required_reserve_mib = 200U;
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
    } else if (key == "--assistant-model") {
      options->assistant = value;
    } else if (key == "--output") {
      options->output = value;
    } else if (key == "--context") {
      if (!ParseUnsigned(value, &options->context)) return false;
    } else if (key == "--required-reserve-mib") {
      if (!ParseUnsigned(value, &options->required_reserve_mib)) return false;
    } else if (key == "--device") {
      std::uint64_t parsed = 0U;
      if (!ParseUnsigned(value, &parsed) ||
          parsed > static_cast<std::uint64_t>(
                       std::numeric_limits<int>::max())) {
        return false;
      }
      options->device = static_cast<int>(parsed);
    } else {
      return false;
    }
  }
  return !options->model.empty() && !options->assistant.empty() &&
         !options->output.empty() && options->context >= 8U &&
         options->required_reserve_mib > 0U &&
         options->required_reserve_mib <=
             std::numeric_limits<std::uint64_t>::max() / kMiB;
}

int Fail(std::string_view operation, const gem16::Status& status, int code) {
  std::cerr << operation << ": status_code="
            << static_cast<int>(status.code()) << ": " << status.message()
            << '\n';
  return code;
}

int FailCuda(std::string_view operation, cudaError_t error, int code) {
  std::cerr << operation << ": " << cudaGetErrorName(error) << ": "
            << cudaGetErrorString(error) << '\n';
  return code;
}

std::uint64_t TokenChecksum(std::span<const std::uint32_t> tokens) {
  std::uint64_t checksum = 1469598103934665603ULL;
  constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
  for (const std::uint32_t token : tokens) {
    for (unsigned shift = 0U; shift < 32U; shift += 8U) {
      checksum ^= static_cast<std::uint8_t>(token >> shift);
      checksum *= kFnvPrime;
    }
  }
  return checksum;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!Parse(argc, argv, &options)) {
    std::cerr
        << "usage: gem16-26b-m25-context-driver --model DIR "
           "--assistant-model DIR --output JSON --context N "
           "[--required-reserve-mib N] [--device N]\n";
    return 2;
  }

  const cudaError_t selected = cudaSetDevice(options.device);
  if (selected != cudaSuccess) {
    return FailCuda("select M25 CUDA device", selected, 3);
  }
  auto before = gem16::QueryDeviceMemoryInfo();
  if (!before.ok()) {
    return Fail("measure memory before runtime", before.status(), 3);
  }

  gem16::ModelRuntimeOptions runtime_options;
  runtime_options.model_directory = options.model;
  runtime_options.assistant_model_directory = options.assistant;
  runtime_options.max_context_tokens = options.context;
  runtime_options.device = options.device;
  runtime_options.verify_device_image_sha256 = false;
  auto runtime = gem16::ModelRuntime::Load(runtime_options);
  if (!runtime.ok()) return Fail("load M25 runtime", runtime.status(), 4);
  auto after_runtime = gem16::QueryDeviceMemoryInfo();
  if (!after_runtime.ok()) {
    return Fail("measure memory after runtime", after_runtime.status(), 4);
  }

  gem16::ConversationSessionOptions session_options;
  session_options.model_directory = options.model;
  session_options.assistant_model_directory = options.assistant;
  session_options.max_context_tokens = options.context;
  session_options.kv_cache_mode = gem16::KvCacheMode::kCheckpointFp8;
  session_options.mtp_draft_tokens = 2U;
  auto session = gem16::ConversationSession::Create(runtime.value(),
                                                     session_options);
  if (!session.ok()) return Fail("create M25 D2 session", session.status(), 5);
  auto after_session = gem16::QueryDeviceMemoryInfo();
  if (!after_session.ok()) {
    return Fail("measure memory after session", after_session.status(), 5);
  }

  constexpr std::array<std::uint32_t, 20> kChatTokenPattern = {
      2U,     105U,  2364U, 107U, 40654U, 607U, 7121U,
      506U,   3658U, 3730U, 236761U, 106U, 107U, 105U,
      4368U,  107U,  100U,  45518U, 107U, 101U};
  const std::uint64_t prompt_tokens = options.context - 4U;
  std::vector<std::uint32_t> prompt(static_cast<std::size_t>(prompt_tokens));
  for (std::size_t index = 0U; index < prompt.size(); ++index) {
    prompt[index] = kChatTokenPattern[index % kChatTokenPattern.size()];
  }

  auto generated = session.value().Generate(
      std::span<const std::uint32_t>(prompt), 4U);
  if (!generated.ok()) {
    return Fail("run boundary M25 D2 generation", generated.status(), 6);
  }
  auto after_generation = gem16::QueryDeviceMemoryInfo();
  if (!after_generation.ok()) {
    return Fail("measure memory after generation", after_generation.status(),
                6);
  }

  const auto& result = generated.value();
  const std::uint64_t required_reserve = options.required_reserve_mib * kMiB;
  const bool execution_pass =
      result.output_token_ids.size() == 4U && result.mtp_enabled &&
      result.mtp_draft_tokens == 2U && result.mtp_fixed_d2_graph &&
      result.mtp_gpu_chained && result.mtp_d2_groups > 0U &&
      result.mtp_verification_groups > 0U &&
      result.max_context_tokens == options.context &&
      result.prompt_cache_write_tokens == prompt_tokens &&
      result.fallback_count == 0U && !result.token_loop_allocations;
  const bool reserve_pass =
      after_generation.value().free_bytes >= required_reserve;

  std::ofstream output(options.output, std::ios::binary | std::ios::trunc);
  if (!output) {
    std::cerr << "cannot open M25 context report: " << options.output << '\n';
    return 7;
  }
  output << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"milestone\": \"M25-context-diagnostic\",\n"
         << "  \"context_tokens\": " << options.context << ",\n"
         << "  \"prompt_tokens\": " << prompt_tokens << ",\n"
         << "  \"generated_tokens\": " << result.output_token_ids.size()
         << ",\n"
         << "  \"output_token_checksum\": "
         << TokenChecksum(result.output_token_ids) << ",\n"
         << "  \"mtp_draft_tokens\": " << result.mtp_draft_tokens << ",\n"
         << "  \"mtp_d2_groups\": " << result.mtp_d2_groups << ",\n"
         << "  \"mtp_verification_groups\": "
         << result.mtp_verification_groups << ",\n"
         << "  \"mtp_proposed_tokens\": " << result.mtp_proposed_tokens
         << ",\n"
         << "  \"mtp_accepted_tokens\": " << result.mtp_accepted_tokens
         << ",\n"
         << "  \"fallback_count\": " << result.fallback_count << ",\n"
         << "  \"token_loop_allocations\": "
         << (result.token_loop_allocations ? "true" : "false") << ",\n"
         << "  \"model_load_ms\": " << result.model_load_milliseconds
         << ",\n"
         << "  \"prompt_ms\": " << result.prompt_milliseconds << ",\n"
         << "  \"memory\": {\n"
         << "    \"visible_total_bytes\": " << before.value().total_bytes
         << ",\n"
         << "    \"free_before_runtime_bytes\": "
         << before.value().free_bytes << ",\n"
         << "    \"free_after_runtime_bytes\": "
         << after_runtime.value().free_bytes << ",\n"
         << "    \"free_after_session_bytes\": "
         << after_session.value().free_bytes << ",\n"
         << "    \"free_after_generation_bytes\": "
         << after_generation.value().free_bytes << ",\n"
         << "    \"required_reserve_bytes\": " << required_reserve << ",\n"
         << "    \"reserve_pass\": " << (reserve_pass ? "true" : "false")
         << "\n"
         << "  },\n"
         << "  \"execution_pass\": "
         << (execution_pass ? "true" : "false") << ",\n"
         << "  \"passed\": "
         << (execution_pass && reserve_pass ? "true" : "false") << "\n"
         << "}\n";
  if (!output) {
    std::cerr << "failed to write M25 context report: " << options.output
              << '\n';
    return 7;
  }
  return execution_pass && reserve_pass ? 0 : 8;
}

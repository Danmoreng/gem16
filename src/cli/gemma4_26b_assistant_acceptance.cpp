#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

#include <algorithm>
#include <array>
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

#include "cuda/engine/gemma4_26b_reference.h"
#include "gem16/sampling.h"
#include "util/json.h"

namespace {

struct Options {
  std::filesystem::path target;
  std::filesystem::path assistant;
  std::filesystem::path workload;
  std::filesystem::path output;
  std::uint64_t max_output = 64U;
  std::uint32_t drafts = 2U;
  std::uint64_t context = 32768U;
  int device = 0;
  bool profile_one_proposal = false;
  bool fixed_chain = false;
  gem16::internal::Gemma4Moe26BMtpVerifierBackend verifier_backend =
      gem16::internal::Gemma4Moe26BMtpVerifierBackend::kExactDecodeParent;
  std::string verifier_backend_name = "exact-decode-parent";
};

bool ParseUnsigned(std::string_view text, std::uint64_t* value) {
  try {
    std::size_t used = 0U;
    *value = std::stoull(std::string(text), &used);
    return used == text.size();
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
    else if (key == "--workload") options->workload = value;
    else if (key == "--output") options->output = value;
    else if (key == "--max-output") {
      if (!ParseUnsigned(value, &options->max_output)) return false;
    } else if (key == "--drafts") {
      std::uint64_t parsed = 0U;
      if (!ParseUnsigned(value, &parsed) ||
          (parsed != 1U && parsed != 2U && parsed != 4U)) return false;
      options->drafts = static_cast<std::uint32_t>(parsed);
    } else if (key == "--context") {
      if (!ParseUnsigned(value, &options->context)) return false;
    } else if (key == "--device") {
      std::uint64_t parsed = 0U;
      if (!ParseUnsigned(value, &parsed) || parsed > 1024U) return false;
      options->device = static_cast<int>(parsed);
    } else if (key == "--profile-one-proposal") {
      if (value == "1") options->profile_one_proposal = true;
      else if (value != "0") return false;
    } else if (key == "--fixed-chain") {
      if (value == "1") options->fixed_chain = true;
      else if (value != "0") return false;
    } else if (key == "--verifier-backend") {
      if (value == "exact") {
        options->verifier_backend = gem16::internal::
            Gemma4Moe26BMtpVerifierBackend::kExactDecodeParent;
        options->verifier_backend_name = "exact-decode-parent";
      } else if (value == "batch-attention") {
        options->verifier_backend = gem16::internal::
            Gemma4Moe26BMtpVerifierBackend::kBatchedAttention;
        options->verifier_backend_name = "batched-attention";
      } else if (value == "batch-moe") {
        options->verifier_backend = gem16::internal::
            Gemma4Moe26BMtpVerifierBackend::kBatchedMoe;
        options->verifier_backend_name = "batched-moe";
      } else if (value == "exact-shared-batch") {
        options->verifier_backend = gem16::internal::
            Gemma4Moe26BMtpVerifierBackend::kExactSharedBatchedMoe;
        options->verifier_backend_name = "exact-shared-batched-moe";
      } else if (value == "batch-all") {
        options->verifier_backend = gem16::internal::
            Gemma4Moe26BMtpVerifierBackend::kFullyBatched;
        options->verifier_backend_name = "fully-batched";
      } else {
        return false;
      }
    } else {
      return false;
    }
  }
  return !options->target.empty() && !options->assistant.empty() &&
         !options->workload.empty() && !options->output.empty() &&
         options->max_output > 0U && options->max_output <= 4096U &&
         (options->context == 32768U || options->context == 65536U);
}

const gem16::json::Value* Field(const gem16::json::Value& value,
                                std::string_view name) {
  return value.is_object() ? value.find(name) : nullptr;
}

gem16::Result<std::string> ReadRegular(const std::filesystem::path& path) {
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  if (error || std::filesystem::is_symlink(status) ||
      !std::filesystem::is_regular_file(status)) {
    return gem16::Status(gem16::StatusCode::kInvalidArgument,
                         "M25 workload is not a regular file");
  }
  constexpr std::uint64_t kLimit = 8U * 1024U * 1024U;
  const auto size = std::filesystem::file_size(path, error);
  if (error || size == 0U || size > kLimit ||
      size > std::numeric_limits<std::size_t>::max()) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         "M25 workload has an invalid size");
  }
  std::string text(static_cast<std::size_t>(size), '\0');
  std::ifstream input(path, std::ios::binary);
  input.read(text.data(), static_cast<std::streamsize>(text.size()));
  if (!input || input.gcount() != static_cast<std::streamsize>(text.size())) {
    return gem16::Status(gem16::StatusCode::kIoError,
                         "cannot read M25 workload");
  }
  return text;
}

gem16::Result<std::vector<std::uint32_t>> ReadPrompt(
    const std::filesystem::path& path) {
  auto text = ReadRegular(path);
  if (!text.ok()) return text.status();
  auto document = gem16::json::Parse(text.value());
  const auto* prompt = document.ok() ? Field(document.value(), "prompt") : nullptr;
  const auto* tokens = prompt != nullptr ? Field(*prompt, "token_ids") : nullptr;
  if (!document.ok() || tokens == nullptr || !tokens->is_array() ||
      tokens->as_array().empty()) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         "M25 workload token array is invalid");
  }
  std::vector<std::uint32_t> result;
  result.reserve(tokens->as_array().size());
  for (const auto& token : tokens->as_array()) {
    if (!token.is_integer() || token.as_integer() < 0 ||
        token.as_integer() >= 262144) {
      return gem16::Status(gem16::StatusCode::kDataLoss,
                           "M25 workload token exceeds vocabulary");
    }
    result.push_back(static_cast<std::uint32_t>(token.as_integer()));
  }
  return result;
}

int Fail(const gem16::Status& status) {
  std::cerr << status.message() << '\n';
  return 1;
}

struct RunResult {
  std::vector<std::uint32_t> output;
  std::uint64_t proposed = 0U;
  std::uint64_t accepted = 0U;
  std::uint64_t groups = 0U;
  std::uint64_t group_output_tokens = 0U;
  std::uint64_t ordinary_output_tokens = 0U;
  std::uint64_t chain_ordinary_tail_tokens = 0U;
  std::uint64_t chain_non_finite_steps = 0U;
  double group_seconds = 0.0;
  double initial_selection_seconds = 0.0;
  double ordinary_seconds = 0.0;
  double target_seconds = 0.0;
  std::size_t free_loaded = 0U;
  bool group_graph_prepared = false;
  std::uint64_t group_graph_device_bytes = 0U;
  std::uint64_t group_graph_launches = 0U;
  bool chain_graph_prepared = false;
  std::uint64_t chain_graph_device_bytes = 0U;
  std::uint64_t chain_graph_launches = 0U;
  std::vector<float> verification_min_margins;
};

gem16::Result<RunResult> RunHybrid(const Options& options,
                                   std::span<const std::uint32_t> prompt) {
  auto engine = gem16::internal::Gemma4Moe26BReferenceEngine::Create(
      options.target, options.context, options.device,
      gem16::internal::Gemma4Moe26BBackend::kSm120Integrated);
  if (!engine.ok()) return engine.status();
  auto status = engine.value().ConfigureTokenSelection(
      gem16::SamplingOptions{}, std::array<std::uint32_t, 2>{258883U, 258882U});
  if (!status.ok()) return status;
  status = engine.value().LoadMtpAssistant(options.assistant);
  if (!status.ok()) return status;
  status = engine.value().ConfigureMtpVerifierBackend(
      options.verifier_backend);
  if (!status.ok()) return status;
  std::size_t total = 0U;
  RunResult result;
  if (cudaMemGetInfo(&result.free_loaded, &total) != cudaSuccess) {
    return gem16::Status(gem16::StatusCode::kInternal,
                         "cannot measure M25 loaded memory");
  }
  status = engine.value().PrefillTokens(prompt);
  if (!status.ok()) return status;
  const auto decode_begin = std::chrono::steady_clock::now();
  const auto initial_begin = std::chrono::steady_clock::now();
  auto initial = engine.value().SelectToken();
  const auto initial_end = std::chrono::steady_clock::now();
  if (!initial.ok()) return initial.status();
  result.initial_selection_seconds = std::chrono::duration<double>(
      initial_end - initial_begin).count();
  std::uint32_t pending = initial.value();
  result.output.push_back(pending);
  if (options.fixed_chain) {
    if (!engine.value().mtp_chain_graph_prepared(options.drafts)) {
      return gem16::Status(gem16::StatusCode::kUnsupported,
                           "M25 fixed chain has no prepared specialization for the requested D");
    }
    std::vector<std::uint32_t> chained(
        static_cast<std::size_t>(options.max_output - 1U));
    gem16::internal::MtpChainResult chain;
    const auto group_begin = std::chrono::steady_clock::now();
    status = engine.value().RunFixedMtpGraphChain(
        pending, options.drafts, chained, &chain);
    const auto group_end = std::chrono::steady_clock::now();
    if (!status.ok()) return status;
    result.group_seconds = std::chrono::duration<double>(
        group_end - group_begin).count();
    result.proposed = chain.proposed_count;
    result.accepted = chain.accepted_count;
    result.groups = chain.group_count;
    result.group_output_tokens = chain.output_count;
    result.chain_ordinary_tail_tokens = chain.ordinary_tail_count;
    result.chain_non_finite_steps = chain.non_finite_step_count;
    result.output.insert(result.output.end(), chained.begin(), chained.end());
  }
  while (!options.fixed_chain && result.output.size() < options.max_output) {
    const std::uint64_t remaining = options.max_output - result.output.size();
    if (remaining == 1U) {
      const auto ordinary_begin = std::chrono::steady_clock::now();
      status = engine.value().ForwardToken(pending);
      if (!status.ok()) return status;
      auto ordinary = engine.value().SelectToken();
      const auto ordinary_end = std::chrono::steady_clock::now();
      if (!ordinary.ok()) return ordinary.status();
      result.ordinary_seconds += std::chrono::duration<double>(
          ordinary_end - ordinary_begin).count();
      ++result.ordinary_output_tokens;
      pending = ordinary.value();
      result.output.push_back(pending);
      continue;
    }
    const std::uint32_t proposal_count =
        options.drafts >= 4U && remaining >= 5U
            ? 4U
            : (options.drafts >= 2U && remaining >= 3U ? 2U : 1U);
    const bool profile = options.profile_one_proposal && result.groups == 0U;
    if (profile && cudaProfilerStart() != cudaSuccess) {
      return gem16::Status(gem16::StatusCode::kInternal,
                           "cannot start M25 verifier profiler");
    }
    gem16::internal::MtpGroupResult group;
    const auto group_begin = std::chrono::steady_clock::now();
    status = engine.value().RunMtpAssistantGroup(
        pending, proposal_count, &group);
    const auto group_end = std::chrono::steady_clock::now();
    if (profile && cudaProfilerStop() != cudaSuccess) {
      return gem16::Status(gem16::StatusCode::kInternal,
                           "cannot stop M25 verifier profiler");
    }
    if (!status.ok()) return status;
    result.group_seconds += std::chrono::duration<double>(
        group_end - group_begin).count();
    result.proposed += proposal_count;
    result.accepted += group.accepted_count;
    result.verification_min_margins.push_back(
        engine.value().last_mtp_verification_min_margin());
    ++result.groups;
    result.group_output_tokens += group.output_count;
    for (std::uint32_t index = 0U; index < group.output_count; ++index) {
      result.output.push_back(group.verified[index]);
    }
    pending = group.verified[group.output_count - 1U];
  }
  result.target_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - decode_begin).count();
  result.group_graph_prepared =
      engine.value().mtp_group_graph_prepared(options.drafts);
  result.group_graph_device_bytes =
      engine.value().mtp_group_graph_device_bytes();
  result.group_graph_launches = engine.value().mtp_group_graph_launches();
  result.chain_graph_prepared =
      engine.value().mtp_chain_graph_prepared(options.drafts);
  result.chain_graph_device_bytes =
      engine.value().mtp_chain_graph_device_bytes(options.drafts);
  result.chain_graph_launches =
      engine.value().mtp_chain_graph_launches(options.drafts);
  return result;
}

gem16::Result<RunResult> RunOrdinary(const Options& options,
                                     std::span<const std::uint32_t> prompt) {
  auto engine = gem16::internal::Gemma4Moe26BReferenceEngine::Create(
      options.target, options.context, options.device,
      gem16::internal::Gemma4Moe26BBackend::kSm120Integrated);
  if (!engine.ok()) return engine.status();
  auto status = engine.value().ConfigureTokenSelection(
      gem16::SamplingOptions{}, std::array<std::uint32_t, 2>{258883U, 258882U});
  if (!status.ok()) return status;
  status = engine.value().PrefillTokens(prompt);
  if (!status.ok()) return status;
  RunResult result;
  const auto begin = std::chrono::steady_clock::now();
  const auto initial_begin = std::chrono::steady_clock::now();
  auto selected = engine.value().SelectToken();
  const auto initial_end = std::chrono::steady_clock::now();
  if (!selected.ok()) return selected.status();
  result.initial_selection_seconds = std::chrono::duration<double>(
      initial_end - initial_begin).count();
  result.output.push_back(selected.value());
  while (result.output.size() < options.max_output) {
    const auto ordinary_begin = std::chrono::steady_clock::now();
    status = engine.value().ForwardToken(selected.value());
    if (!status.ok()) return status;
    selected = engine.value().SelectToken();
    const auto ordinary_end = std::chrono::steady_clock::now();
    if (!selected.ok()) return selected.status();
    result.ordinary_seconds += std::chrono::duration<double>(
        ordinary_end - ordinary_begin).count();
    ++result.ordinary_output_tokens;
    result.output.push_back(selected.value());
  }
  result.target_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - begin).count();
  return result;
}

void WriteIds(std::ostream& output, std::span<const std::uint32_t> ids) {
  output << '[';
  for (std::size_t index = 0; index < ids.size(); ++index) {
    if (index != 0U) output << ',';
    output << ids[index];
  }
  output << ']';
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!Parse(argc, argv, &options)) {
    std::cerr << "Usage: gem16-26b-assistant-acceptance --target DIR "
                 "--assistant DIR --workload JSON --output JSON "
                 "[--drafts 1|2|4] [--max-output N] "
                 "[--context 32768|65536] [--device N] "
                 "[--profile-one-proposal 0|1] "
                 "[--fixed-chain 0|1] "
                 "[--verifier-backend exact|batch-attention|batch-moe|"
                 "exact-shared-batch|batch-all]\n";
    return 2;
  }
  if (std::filesystem::exists(options.output)) {
    std::cerr << "refusing to overwrite M25 acceptance report\n";
    return 2;
  }
  auto prompt = ReadPrompt(options.workload);
  if (!prompt.ok()) return Fail(prompt.status());
  if (prompt.value().size() + options.max_output > options.context) {
    std::cerr << "M25 workload exceeds requested context\n";
    return 2;
  }
  if (cudaSetDevice(options.device) != cudaSuccess) return 3;
  auto hybrid = RunHybrid(options, prompt.value());
  if (!hybrid.ok()) return Fail(hybrid.status());
  auto ordinary = RunOrdinary(options, prompt.value());
  if (!ordinary.ok()) return Fail(ordinary.status());
  const bool exact = hybrid.value().output == ordinary.value().output;
  if (!exact) {
    const auto mismatch = std::mismatch(hybrid.value().output.begin(),
                                        hybrid.value().output.end(),
                                        ordinary.value().output.begin(),
                                        ordinary.value().output.end());
    const std::size_t index = static_cast<std::size_t>(
        mismatch.first - hybrid.value().output.begin());
    std::cerr << "transactional M25 verifier changed Target output IDs at "
              << index;
    if (mismatch.first != hybrid.value().output.end()) {
      std::cerr << " hybrid=" << *mismatch.first;
    }
    if (mismatch.second != ordinary.value().output.end()) {
      std::cerr << " ordinary=" << *mismatch.second;
    }
    std::cerr << '\n';
    std::cerr << "M25 group minimum margins:";
    for (float margin : hybrid.value().verification_min_margins) {
      std::cerr << ' ' << margin;
    }
    std::cerr << '\n';
    return 4;
  }
  const double acceptance = hybrid.value().proposed == 0U
                                ? 0.0
                                : static_cast<double>(hybrid.value().accepted) /
                                      static_cast<double>(hybrid.value().proposed);
  const double hybrid_seconds = hybrid.value().target_seconds;
  const float minimum_margin = hybrid.value().verification_min_margins.empty()
                                   ? 0.0F
                                   : *std::min_element(
                                         hybrid.value()
                                             .verification_min_margins.begin(),
                                         hybrid.value()
                                             .verification_min_margins.end());
  std::ofstream output(options.output, std::ios::binary | std::ios::trunc);
  output << std::setprecision(12)
         << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"status\": \"diagnostic_transactional_batch_verifier_pass\",\n"
         << "  \"performance_eligible\": false,\n"
         << "  \"reason\": \"single bounded development run; formal warm-up/retained qualification pending\",\n"
         << "  \"context_tokens\": " << options.context << ",\n"
         << "  \"prompt_tokens\": " << prompt.value().size() << ",\n"
         << "  \"output_tokens\": " << hybrid.value().output.size() << ",\n"
         << "  \"draft_length\": " << options.drafts << ",\n"
         << "  \"verifier_backend\": \""
         << options.verifier_backend_name << "\",\n"
         << "  \"fixed_mtp_group_graph_prepared\": "
         << (hybrid.value().group_graph_prepared ? "true" : "false")
         << ",\n"
         << "  \"fixed_mtp_group_graph_device_bytes\": "
         << hybrid.value().group_graph_device_bytes << ",\n"
         << "  \"fixed_mtp_group_graph_launches\": "
         << hybrid.value().group_graph_launches << ",\n"
         << "  \"fixed_mtp_chain_enabled\": "
         << (options.fixed_chain ? "true" : "false") << ",\n"
         << "  \"fixed_mtp_chain_graph_prepared\": "
         << (hybrid.value().chain_graph_prepared ? "true" : "false")
         << ",\n"
         << "  \"fixed_mtp_chain_graph_device_bytes\": "
         << hybrid.value().chain_graph_device_bytes << ",\n"
         << "  \"fixed_mtp_total_graph_device_bytes\": "
         << hybrid.value().group_graph_device_bytes +
                hybrid.value().chain_graph_device_bytes
         << ",\n"
         << "  \"fixed_mtp_chain_graph_launches\": "
         << hybrid.value().chain_graph_launches << ",\n"
         << "  \"groups\": " << hybrid.value().groups << ",\n"
         << "  \"proposed_drafts\": " << hybrid.value().proposed << ",\n"
         << "  \"accepted_drafts\": " << hybrid.value().accepted << ",\n"
         << "  \"rejected_drafts\": "
         << hybrid.value().proposed - hybrid.value().accepted << ",\n"
         << "  \"target_verified_acceptance\": " << acceptance << ",\n"
         << "  \"precision_gate_50_percent_pass\": "
         << (acceptance >= 0.5 ? "true" : "false") << ",\n"
         << "  \"ordinary_target_identity\": true,\n"
         << "  \"minimum_target_top1_margin\": "
         << minimum_margin
         << ",\n"
         << "  \"initial_selection_seconds\": "
         << hybrid.value().initial_selection_seconds << ",\n"
         << "  \"ordinary_tail_tokens\": "
         << hybrid.value().ordinary_output_tokens << ",\n"
         << "  \"chain_ordinary_tail_tokens\": "
         << hybrid.value().chain_ordinary_tail_tokens << ",\n"
         << "  \"chain_non_finite_steps\": "
         << hybrid.value().chain_non_finite_steps << ",\n"
         << "  \"ordinary_tail_seconds\": "
         << hybrid.value().ordinary_seconds << ",\n"
         << "  \"batched_group_output_tokens\": "
         << hybrid.value().group_output_tokens << ",\n"
         << "  \"batched_group_seconds\": "
         << hybrid.value().group_seconds << ",\n"
         << "  \"batched_group_tokens_per_second\": "
         << (hybrid.value().group_seconds > 0.0
                 ? static_cast<double>(hybrid.value().group_output_tokens) /
                       hybrid.value().group_seconds
                 : 0.0)
         << ",\n"
         << "  \"hybrid_post_first_tokens_per_second\": "
         << (hybrid.value().group_seconds + hybrid.value().ordinary_seconds >
                     0.0
                 ? static_cast<double>(hybrid.value().group_output_tokens +
                                       hybrid.value().ordinary_output_tokens) /
                       (hybrid.value().group_seconds +
                        hybrid.value().ordinary_seconds)
                 : 0.0)
         << ",\n"
         << "  \"hybrid_decode_seconds\": "
         << hybrid.value().target_seconds << ",\n"
         << "  \"hybrid_tokens_per_second\": "
         << static_cast<double>(hybrid.value().output.size()) / hybrid_seconds
         << ",\n"
         << "  \"ordinary_tokens_per_second\": "
         << static_cast<double>(ordinary.value().output.size()) /
                ordinary.value().target_seconds
         << ",\n"
         << "  \"ordinary_initial_selection_seconds\": "
         << ordinary.value().initial_selection_seconds << ",\n"
         << "  \"ordinary_post_first_tokens\": "
         << ordinary.value().ordinary_output_tokens << ",\n"
         << "  \"ordinary_post_first_seconds\": "
         << ordinary.value().ordinary_seconds << ",\n"
         << "  \"ordinary_post_first_tokens_per_second\": "
         << (ordinary.value().ordinary_seconds > 0.0
                 ? static_cast<double>(ordinary.value().ordinary_output_tokens) /
                       ordinary.value().ordinary_seconds
                 : 0.0)
         << ",\n"
         << "  \"free_loaded_bytes\": " << hybrid.value().free_loaded << ",\n"
         << "  \"output_token_ids\": ";
  WriteIds(output, hybrid.value().output);
  output << "\n}\n";
  return output ? 0 : 5;
}

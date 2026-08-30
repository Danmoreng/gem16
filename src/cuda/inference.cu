#include "gem16/engine.h"
#include "cuda/engine/gemma4_26b_reference.h"
#include "cuda/engine/inference_engine.h"

#include "cuda/attention/sm120.h"
#include "cuda/engine/target_model.h"
#include "cuda/engine/state_capture.h"
#include "cuda/fp8/cutlass_sm120.h"
#include "cuda/fp8/reference.h"
#include "cuda/fp8/sm120.h"
#include "cuda/layer/reference.h"
#include "cuda/mtp/assistant.h"
#include "cuda/mtp/scheduler.h"
#include "cuda/mtp/verify.h"
#include "cuda/nvfp4/mlp.h"
#include "cuda/nvfp4/reference.h"
#include "cuda/nvfp4/cutlass_sm120.h"
#include "cuda/nvfp4/sm120.h"
#include "cuda/nvfp4/sm120_layout.h"
#include "cuda/output_head.h"
#include "cuda/sampling/sampling.h"
#include "gem16/model.h"
#include "gem16/tokenizer.h"
#include "model/config.h"
#include "model/gemma4_26b_compiled_loader.h"
#include "model/gemma4_26b_trellis35.h"
#include "model/model_variant.h"
#include "platform/mapped_file.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <mutex>
#include <numeric>
#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace gem16 {
namespace {

constexpr std::uint64_t kVocabulary = 262144U;
constexpr std::uint64_t kMaximumContext = 262144U;
constexpr std::uint64_t kSlidingWindow = 1024U;

Status Error(StatusCode code, std::string message) { return Status(code, std::move(message)); }
Status CudaFailure(const char* operation, cudaError_t error) {
  return Error(StatusCode::kInternal, std::string(operation) + ": " +
      cudaGetErrorName(error) + ": " + cudaGetErrorString(error));
}

class NvtxRange {
 public:
  explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
  ~NvtxRange() { nvtxRangePop(); }
};

class PinnedHostAllocation {
 public:
  PinnedHostAllocation() = default;
  PinnedHostAllocation(const PinnedHostAllocation&) = delete;
  PinnedHostAllocation& operator=(const PinnedHostAllocation&) = delete;
  ~PinnedHostAllocation() { if (data_ != nullptr) (void)cudaFreeHost(data_); }
  [[nodiscard]] Status Allocate(std::size_t elements, const char* label) {
    if (data_ != nullptr || elements == 0U || elements > std::numeric_limits<std::size_t>::max() / sizeof(float))
      return Error(StatusCode::kInvalidArgument, std::string("pinned ") + label + " size is invalid");
    const cudaError_t error = cudaHostAlloc(&data_, elements * sizeof(float), cudaHostAllocDefault);
    if (error != cudaSuccess) return CudaFailure(label, error);
    elements_ = elements; return Status::Ok();
  }
  [[nodiscard]] std::span<float> span() const { return {static_cast<float*>(data_), elements_}; }
 private:
  void* data_ = nullptr; std::size_t elements_ = 0U;
};

using MtpGroupResult = internal::MtpGroupResult;
using internal::MakeStateCaptureLayout;
using internal::WriteStateDump;

double Milliseconds(std::chrono::steady_clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

double Percentile(std::vector<double> sorted, double quantile) {
  if (sorted.empty()) return 0.0;
  std::sort(sorted.begin(), sorted.end());
  const double rank = quantile * static_cast<double>(sorted.size() - 1U);
  const std::size_t lower = static_cast<std::size_t>(rank);
  const std::size_t upper = std::min(lower + 1U, sorted.size() - 1U);
  const double fraction = rank - static_cast<double>(lower);
  return sorted[lower] + fraction * (sorted[upper] - sorted[lower]);
}

double StudentTCritical95(std::size_t degrees_of_freedom) {
  constexpr std::array values = {
      0.0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365,
      2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
      2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069,
      2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042};
  return degrees_of_freedom < values.size() ? values[degrees_of_freedom] : 1.96;
}

BenchmarkDistribution Summarize(std::span<const double> samples) {
  BenchmarkDistribution summary;
  summary.sample_count = static_cast<std::uint64_t>(samples.size());
  if (samples.empty()) return summary;
  summary.minimum = *std::min_element(samples.begin(), samples.end());
  summary.maximum = *std::max_element(samples.begin(), samples.end());
  summary.mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                 static_cast<double>(samples.size());
  std::vector<double> values(samples.begin(), samples.end());
  summary.median = Percentile(values, 0.5);
  summary.p95 = Percentile(values, 0.95);
  summary.p99 = Percentile(std::move(values), 0.99);
  if (samples.size() > 1U) {
    double squared_deviation = 0.0;
    for (const double value : samples) {
      const double deviation = value - summary.mean;
      squared_deviation += deviation * deviation;
    }
    summary.standard_deviation =
        std::sqrt(squared_deviation / static_cast<double>(samples.size() - 1U));
    const double margin = StudentTCritical95(samples.size() - 1U) *
                          summary.standard_deviation /
                          std::sqrt(static_cast<double>(samples.size()));
    summary.confidence_95_low = summary.mean - margin;
    summary.confidence_95_high = summary.mean + margin;
  } else {
    summary.confidence_95_low = summary.mean;
    summary.confidence_95_high = summary.mean;
  }
  return summary;
}

std::uint64_t UpdateTokenChecksum(std::uint64_t checksum, std::uint32_t token) {
  constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
  for (unsigned shift = 0; shift < 32U; shift += 8U) {
    checksum ^= static_cast<std::uint8_t>(token >> shift);
    checksum *= kFnvPrime;
  }
  return checksum;
}



}  // namespace

#include "cuda/inference_session.cuh"

Result<GreedyInferenceResult> RunGreedyInference(const GreedyInferenceOptions& options) {
  if (options.model_directory.empty()) {
    return Error(StatusCode::kInvalidArgument, "greedy inference requires --model");
  }
  if (options.input_token_ids.empty()) {
    return Error(StatusCode::kInvalidArgument, "greedy inference requires input token IDs");
  }
  if (options.max_generated_tokens == 0U) {
    return Error(StatusCode::kInvalidArgument, "--max-tokens must be positive");
  }
  auto model_config =
      internal::LoadModelConfig(options.model_directory / "config.json");
  if (!model_config.ok()) return model_config.status();
  if (internal::IsGemma4Moe26BModel(model_config.value())) {
    if (!options.teacher_forced_token_ids.empty() ||
        !options.logits_dump_path.empty() ||
        !options.state_dump_path.empty() ||
        options.state_dump_position.has_value()) {
      return Error(
          StatusCode::kUnsupported,
          "Gemma 4 26B product generation does not expose 12B diagnostic dumps or teacher forcing");
    }
    ConversationSessionOptions session_options;
    session_options.model_directory = options.model_directory;
    session_options.assistant_model_directory =
        options.assistant_model_directory;
    session_options.stop_token_ids = options.stop_token_ids;
    session_options.suppressed_token_ids = options.suppressed_token_ids;
    session_options.max_context_tokens = options.max_context_tokens;
    session_options.kv_cache_mode = options.kv_cache_mode;
    session_options.sampling = options.sampling;
    session_options.mtp_draft_tokens = options.mtp_draft_tokens;
    session_options.mtp_adaptive = options.mtp_adaptive;
    session_options.mtp_router_overlap_diagnostic =
        options.mtp_router_overlap_diagnostic;
    session_options.cuda_profile_phase = options.cuda_profile_phase;
    auto session = ConversationSession::Create(session_options);
    if (!session.ok()) return session.status();
    return session.value().Generate(
        options.input_token_ids, options.max_generated_tokens, {},
        options.generated_token_callback,
        options.generated_token_callback_context);
  }
  const bool teacher_forcing = !options.teacher_forced_token_ids.empty();
  const bool mtp_enabled = options.mtp_draft_tokens != 0U;
  if (mtp_enabled &&
      (options.mtp_draft_tokens != 1U && options.mtp_draft_tokens != 2U &&
       options.mtp_draft_tokens != 4U)) {
    return Error(StatusCode::kInvalidArgument,
                 "active MTP requires draft length 1, 2, or 4");
  }
  if (mtp_enabled && options.assistant_model_directory.empty()) {
    return Error(StatusCode::kInvalidArgument,
                 "active MTP requires --assistant-model");
  }
  if (options.mtp_adaptive && !mtp_enabled) {
    return Error(StatusCode::kInvalidArgument,
                 "--mtp-adaptive requires active MTP");
  }
  if (mtp_enabled &&
      (teacher_forcing || !options.logits_dump_path.empty() ||
       !options.state_dump_path.empty())) {
    return Error(StatusCode::kUnsupported,
                 "the MTP path currently requires generation without diagnostic dumps");
  }
  if (teacher_forcing && options.sampling.enabled) {
    return Error(StatusCode::kInvalidArgument,
                 "teacher-forced diagnostics require greedy prediction");
  }
  const std::uint64_t generation_steps =
      teacher_forcing
          ? static_cast<std::uint64_t>(options.teacher_forced_token_ids.size())
          : options.max_generated_tokens;
  if (options.max_context_tokens == 0U || options.max_context_tokens > kMaximumContext) {
    return Error(StatusCode::kUnsupported,
                 "the hybrid KV cache supports 1..262144 tokens");
  }
  if (options.input_token_ids.size() > options.max_context_tokens ||
      generation_steps - 1U >
          options.max_context_tokens - options.input_token_ids.size()) {
    return Error(StatusCode::kInvalidArgument,
                 "prompt plus generated decode positions exceed --max-context");
  }
  if (options.state_dump_path.empty() !=
      !options.state_dump_position.has_value()) {
    return Error(StatusCode::kInvalidArgument,
                 "--dump-state and --dump-state-position must be used together");
  }
  if (options.state_dump_position.has_value()) {
    const std::uint64_t maximum_forward_position =
        static_cast<std::uint64_t>(options.input_token_ids.size() - 1U) +
        (generation_steps - 1U);
    if (*options.state_dump_position > maximum_forward_position) {
      return Error(StatusCode::kInvalidArgument,
                   "state dump position is outside the requested inference");
    }
  }
  for (const std::uint32_t token : options.input_token_ids) {
    if (token >= kVocabulary) {
      return Error(StatusCode::kInvalidArgument, "input token ID exceeds vocabulary");
    }
  }
  for (const std::uint32_t token : options.teacher_forced_token_ids) {
    if (token >= kVocabulary) {
      return Error(StatusCode::kInvalidArgument,
                   "teacher-forced token ID exceeds vocabulary");
    }
  }
  for (const std::uint32_t token : options.stop_token_ids) {
    if (token >= kVocabulary) {
      return Error(StatusCode::kInvalidArgument, "stop token ID exceeds vocabulary");
    }
  }
  for (const std::uint32_t token : options.suppressed_token_ids) {
    if (token >= kVocabulary) {
      return Error(StatusCode::kInvalidArgument, "suppressed token ID exceeds vocabulary");
    }
  }

  PinnedHostAllocation captured_logits;
  if (!options.logits_dump_path.empty()) {
    if constexpr (std::endian::native != std::endian::little) {
      return Error(StatusCode::kUnsupported,
                   "raw full-logit dumps currently require a little-endian host");
    }
    if (generation_steps >
        std::numeric_limits<std::size_t>::max() / kVocabulary) {
      return Error(StatusCode::kInvalidArgument, "requested logit capture is too large");
    }
    Status status = captured_logits.Allocate(
        static_cast<std::size_t>(generation_steps * kVocabulary),
        "full-logit capture");
    if (!status.ok()) return status;
  }
  PinnedHostAllocation captured_state;
  if (!options.state_dump_path.empty()) {
    Status status = captured_state.Allocate(MakeStateCaptureLayout().elements,
                                            "layer-state capture");
    if (!status.ok()) return status;
  }

  const auto load_start = std::chrono::steady_clock::now();
  InferenceEngine engine;
  Status status = engine.Initialize(options.model_directory,
                                    options.max_context_tokens,
                                    options.kv_cache_mode,
                                    options.sampling,
                                    options.assistant_model_directory,
                                    options.mtp_draft_tokens);
  if (!status.ok()) return status;
  status = engine.SetSuppressedTokens(options.suppressed_token_ids);
  if (!status.ok()) return status;
  status = engine.SetMtpStopTokens(options.stop_token_ids);
  if (!status.ok()) return status;
  const auto load_end = std::chrono::steady_clock::now();

  GreedyInferenceResult result;
  result.output_token_ids.reserve(static_cast<std::size_t>(generation_steps));
  if (mtp_enabled) {
    result.mtp_proposed_token_ids.reserve(
        static_cast<std::size_t>(generation_steps * options.mtp_draft_tokens));
  }
  result.teacher_forced_token_ids = options.teacher_forced_token_ids;
  result.teacher_forcing = teacher_forcing;
  result.kv_cache_mode = options.kv_cache_mode;
  result.sampling = options.sampling;
  result.decode_graphs =
      options.state_dump_path.empty() && options.logits_dump_path.empty();
  result.model_load_milliseconds = Milliseconds(load_end - load_start);
  result.weight_arena_bytes = engine.weight_bytes();
  result.assistant_loaded = engine.assistant_loaded();
  result.assistant_source_bytes = engine.assistant_source_bytes();
  result.assistant_weight_arena_bytes = engine.assistant_weight_bytes();
  result.assistant_device_memory_delta_bytes =
      engine.assistant_device_memory_delta_bytes();
  result.assistant_tensor_count = engine.assistant_tensor_count();
  result.assistant_workspace_bytes = engine.assistant_workspace_bytes();
  result.mtp_enabled = mtp_enabled;
  result.mtp_adaptive = options.mtp_adaptive;
  result.mtp_draft_tokens = options.mtp_draft_tokens;
  result.kv_cache_bytes = engine.cache_bytes();
  result.workspace_bytes = engine.workspace_bytes();
  result.decode_graph_device_bytes = engine.decode_graph_device_bytes();
  result.prefill_chunk_tokens = engine.prefill_chunk_tokens();
  result.max_context_tokens = options.max_context_tokens;
  result.packed_weight_source_layout_direct = false;
  result.token_loop_allocations = false;
  result.benchmark_qualified = false;

  const auto prompt_start = std::chrono::steady_clock::now();
  std::uint32_t next_token = 0;
  bool state_captured = false;
  if (options.state_dump_path.empty()) {
    const std::span<float> prefill_logits =
        captured_logits.span().empty()
            ? std::span<float>()
            : captured_logits.span().first(static_cast<std::size_t>(kVocabulary));
    auto prefilled = engine.Prefill(options.input_token_ids, prefill_logits);
    if (!prefilled.ok()) return prefilled.status();
    next_token = prefilled.value();
  } else for (std::size_t index = 0; index < options.input_token_ids.size(); ++index) {
    const bool select = index + 1U == options.input_token_ids.size();
    const std::span<float> logit_capture =
        select && !captured_logits.span().empty()
            ? captured_logits.span().first(static_cast<std::size_t>(kVocabulary))
            : std::span<float>();
    const bool capture_state =
        options.state_dump_position.has_value() &&
        *options.state_dump_position == index;
    auto forwarded = engine.Forward(
        options.input_token_ids[index], index, select, logit_capture,
        capture_state ? captured_state.span() : std::span<float>());
    if (!forwarded.ok()) return forwarded.status();
    state_captured = state_captured || capture_state;
    if (select) next_token = forwarded.value();
  }
  if (mtp_enabled && options.mtp_draft_tokens == 2U &&
      !options.mtp_adaptive &&
      options.kv_cache_mode == KvCacheMode::kCheckpointFp8 &&
      options.max_context_tokens > kSlidingWindow) {
    status = engine.PrepareFixedD2Graph();
    if (!status.ok()) return status;
  }
  const auto prompt_end = std::chrono::steady_clock::now();
  result.prompt_milliseconds = Milliseconds(prompt_end - prompt_start);
  result.output_token_ids.push_back(next_token);
  if (options.generated_token_callback != nullptr) {
    status = options.generated_token_callback(
        options.generated_token_callback_context, next_token);
    if (!status.ok()) return status;
  }
  if (!teacher_forcing &&
      std::find(options.stop_token_ids.begin(), options.stop_token_ids.end(), next_token) !=
      options.stop_token_ids.end()) {
    result.stopped = true;
    result.stop_token_id = next_token;
  }

  const auto decode_start = std::chrono::steady_clock::now();
  if (mtp_enabled) {
    std::uint64_t processed_position = options.input_token_ids.size() - 1U;
    internal::AdaptiveMtpScheduler adaptive_scheduler(
        options.mtp_draft_tokens, processed_position, options.mtp_adaptive);
    const auto emit_ordinary_token = [&](bool adaptive_fallback) -> Status {
      auto forwarded = engine.Forward(next_token, processed_position + 1U,
                                      true);
      if (!forwarded.ok()) return forwarded.status();
      ++processed_position;
      ++result.mtp_target_forwards;
      ++result.mtp_target_batches;
      if (adaptive_fallback) ++result.mtp_ordinary_fallback_tokens;
      next_token = forwarded.value();
      result.output_token_ids.push_back(next_token);
      if (options.generated_token_callback != nullptr) {
        Status callback_status = options.generated_token_callback(
            options.generated_token_callback_context, next_token);
        if (!callback_status.ok()) return callback_status;
      }
      if (std::find(options.stop_token_ids.begin(),
                    options.stop_token_ids.end(), next_token) !=
          options.stop_token_ids.end()) {
        result.stopped = true;
        result.stop_token_id = next_token;
      }
      return Status::Ok();
    };
    while (result.output_token_ids.size() < generation_steps &&
           !result.stopped) {
      const std::size_t remaining = static_cast<std::size_t>(
          generation_steps - result.output_token_ids.size());
      const bool fixed_d2_chain =
          remaining >= 3U && options.mtp_draft_tokens == 2U &&
          !options.mtp_adaptive &&
          options.kv_cache_mode == KvCacheMode::kCheckpointFp8 &&
          options.max_context_tokens > kSlidingWindow;
      if (fixed_d2_chain) {
        status = engine.PrepareMtpDeviceControl(
            next_token, processed_position, remaining,
            result.output_token_ids.size(), result.stopped,
            result.stop_token_id);
        if (!status.ok()) return status;
        internal::MtpChainResult chain;
        status = engine.ExecuteFixedD2GraphChain(
            &chain, options.generated_token_callback,
            options.generated_token_callback_context);
        if (!status.ok()) return status;
        result.mtp_fixed_d2_graph = true;
        result.mtp_gpu_chained = true;
        result.mtp_verification_groups += chain.group_count;
        result.mtp_d2_groups += chain.group_count;
        result.mtp_proposed_tokens += chain.proposed_count;
        result.mtp_target_forwards += 3U * chain.group_count;
        result.mtp_target_batches += chain.group_count;
        result.mtp_target_forwards += chain.ordinary_tail_count;
        result.mtp_target_batches += chain.ordinary_tail_count;
        result.mtp_accepted_tokens += chain.accepted_count;
        result.mtp_rejected_tokens += chain.rejected_count;
        result.mtp_proposed_token_ids.insert(
            result.mtp_proposed_token_ids.end(),
            engine.mtp_chain_proposals(),
            engine.mtp_chain_proposals() + chain.proposed_count);
        processed_position += chain.output_count;
        const std::uint32_t* chained_outputs = engine.mtp_chain_outputs();
        for (std::uint64_t index = 0U; index < chain.output_count; ++index) {
          next_token = chained_outputs[index];
          result.output_token_ids.push_back(next_token);
        }
        result.stopped = chain.stopped != 0U;
        result.stop_token_id = chain.stop_token;
        status = engine.CheckMtpDeviceControlParity(
            next_token, processed_position,
            generation_steps - result.output_token_ids.size(),
            result.output_token_ids.size(), result.stopped,
            result.stop_token_id);
        if (!status.ok()) return status;
        continue;
      }
      const bool adaptive_fallback =
          adaptive_scheduler.use_ordinary_fallback();
      if (remaining == 1U || adaptive_fallback) {
        status = emit_ordinary_token(adaptive_fallback);
        if (!status.ok()) return status;
        if (adaptive_fallback) adaptive_scheduler.ConsumeOrdinaryFallback();
        continue;
      }
      const std::size_t proposal_count = std::min<std::size_t>(
          adaptive_scheduler.active_drafts(), remaining - 1U);
      status = engine.PrepareMtpDeviceControl(
          next_token, processed_position, remaining,
          result.output_token_ids.size(), result.stopped,
          result.stop_token_id);
      if (!status.ok()) return status;
      MtpGroupResult group;
      if (proposal_count == 2U && options.mtp_draft_tokens == 2U &&
          !options.mtp_adaptive) {
        result.mtp_fixed_d2_graph =
            options.kv_cache_mode == KvCacheMode::kCheckpointFp8 &&
            options.max_context_tokens > kSlidingWindow;
        status = engine.ExecuteFixedD2GraphGroup(
            next_token, processed_position + 1U, &group);
      } else {
        status = engine.GenerateAssistantDraftsDevice(
            next_token, processed_position,
            static_cast<std::uint32_t>(proposal_count));
        if (!status.ok()) return status;
        status = engine.VerifyAcceptCommitAssistantBatch(
            next_token, processed_position + 1U,
            static_cast<std::uint32_t>(proposal_count), &group);
      }
      if (!status.ok()) return status;
      ++result.mtp_verification_groups;
      if (proposal_count == 1U) {
        ++result.mtp_d1_groups;
      } else if (proposal_count == 2U) {
        ++result.mtp_d2_groups;
      } else if (proposal_count == 4U) {
        ++result.mtp_d4_groups;
      }
      result.mtp_proposed_tokens += proposal_count;
      result.mtp_proposed_token_ids.insert(
          result.mtp_proposed_token_ids.end(), group.proposed.begin(),
          group.proposed.begin() + static_cast<std::ptrdiff_t>(proposal_count));
      result.mtp_target_forwards += proposal_count + 1U;
      ++result.mtp_target_batches;
      result.mtp_accepted_tokens += group.accepted_count;
      result.mtp_rejected_tokens += proposal_count - group.accepted_count;
      processed_position += group.output_count;
      for (std::uint32_t index = 0U; index < group.output_count; ++index) {
        next_token = group.verified[index];
        result.output_token_ids.push_back(next_token);
        if (options.generated_token_callback != nullptr) {
          status = options.generated_token_callback(
              options.generated_token_callback_context, next_token);
          if (!status.ok()) return status;
        }
        if (std::find(options.stop_token_ids.begin(),
                      options.stop_token_ids.end(), next_token) !=
            options.stop_token_ids.end()) {
          result.stopped = true;
          result.stop_token_id = next_token;
          break;
        }
      }
      status = engine.CheckMtpDeviceControlParity(
          next_token, processed_position,
          generation_steps - result.output_token_ids.size(),
          result.output_token_ids.size(), result.stopped,
          result.stop_token_id);
      if (!status.ok()) return status;
      if (!result.stopped) {
        adaptive_scheduler.Observe(processed_position,
                                   group.accepted_count);
      }
    }
  } else {
    for (std::uint64_t generated = 1U;
         generated < generation_steps && !result.stopped; ++generated) {
      const std::uint64_t position =
          options.input_token_ids.size() + generated - 1U;
      const std::size_t logit_offset =
          static_cast<std::size_t>(generated * kVocabulary);
      const std::span<float> logit_capture =
          captured_logits.span().empty()
              ? std::span<float>()
              : captured_logits.span().subspan(
                    logit_offset, static_cast<std::size_t>(kVocabulary));
      const bool capture_state =
          options.state_dump_position.has_value() &&
          *options.state_dump_position == position;
      const std::uint32_t input_token =
          teacher_forcing
              ? options.teacher_forced_token_ids[
                    static_cast<std::size_t>(generated - 1U)]
              : next_token;
      auto forwarded = engine.Forward(
          input_token, position, true, logit_capture,
          capture_state ? captured_state.span() : std::span<float>());
      if (!forwarded.ok()) return forwarded.status();
      state_captured = state_captured || capture_state;
      next_token = forwarded.value();
      result.output_token_ids.push_back(next_token);
      if (options.generated_token_callback != nullptr) {
        status = options.generated_token_callback(
            options.generated_token_callback_context, next_token);
        if (!status.ok()) return status;
      }
      if (!teacher_forcing &&
          std::find(options.stop_token_ids.begin(),
                    options.stop_token_ids.end(), next_token) !=
              options.stop_token_ids.end()) {
        result.stopped = true;
        result.stop_token_id = next_token;
      }
    }
  }
  if (teacher_forcing) {
    result.teacher_forced_matches = static_cast<std::uint64_t>(
        std::inner_product(
            result.output_token_ids.begin(), result.output_token_ids.end(),
            result.teacher_forced_token_ids.begin(), std::size_t{0},
            std::plus<>(), std::equal_to<>()));
  }
  const auto decode_end = std::chrono::steady_clock::now();
  result.decode_milliseconds = Milliseconds(decode_end - decode_start);
  const std::uint64_t measured_decode_tokens =
      result.output_token_ids.empty() ? 0U : result.output_token_ids.size() - 1U;
  if (measured_decode_tokens != 0U && result.decode_milliseconds > 0.0) {
    result.decode_tokens_per_second =
        static_cast<double>(measured_decode_tokens) * 1000.0 / result.decode_milliseconds;
  }
  result.decode_graph_device_bytes = engine.decode_graph_device_bytes();
  if (!options.logits_dump_path.empty()) {
    result.logits_dump_steps = result.output_token_ids.size();
    const std::size_t dump_elements =
        static_cast<std::size_t>(result.logits_dump_steps * kVocabulary);
    std::ofstream dump(options.logits_dump_path, std::ios::binary | std::ios::trunc);
    if (!dump) {
      return Error(StatusCode::kIoError, "cannot open full-logit dump");
    }
    dump.write(reinterpret_cast<const char*>(captured_logits.span().data()),
               static_cast<std::streamsize>(dump_elements * sizeof(float)));
    if (!dump) {
      return Error(StatusCode::kIoError, "failed to write full-logit dump");
    }
    result.logits_dumped = true;
  }
  if (!options.state_dump_path.empty()) {
    if (!state_captured) {
      return Error(StatusCode::kInvalidArgument,
                   "generation stopped before the requested state dump position");
    }
    status = WriteStateDump(options.state_dump_path,
                            *options.state_dump_position, options.kv_cache_mode,
                            captured_state.span());
    if (!status.ok()) return status;
    result.state_dumped = true;
    result.state_dump_position = *options.state_dump_position;
  }
  return result;
}

Result<DecodeBenchmarkResult> RunDecodeBenchmark(
    const DecodeBenchmarkOptions& options) {
  if (options.model_directory.empty()) {
    return Error(StatusCode::kInvalidArgument, "decode benchmark requires --model");
  }
  if (options.context_tokens == 0U || options.generated_tokens == 0U ||
      options.warmup_runs == 0U || options.measured_runs == 0U) {
    return Error(StatusCode::kInvalidArgument,
                 "context, tokens, warmups, and repetitions must be positive");
  }
  const std::uint64_t planned_context =
      static_cast<std::uint64_t>(options.context_tokens) + options.generated_tokens;
  if (planned_context > kMaximumContext) {
    return Error(StatusCode::kUnsupported,
                 "the hybrid benchmark cache supports prompt plus decode up to 262144 tokens");
  }

  std::vector<std::uint32_t> prompt(options.context_tokens);
  for (std::size_t index = 0; index < prompt.size(); ++index) {
    const std::uint64_t value = static_cast<std::uint64_t>(options.prompt_seed) +
                                static_cast<std::uint64_t>(index) * 7919U;
    prompt[index] = 1000U + static_cast<std::uint32_t>(value % 9000U);
  }

  const auto load_start = std::chrono::steady_clock::now();
  InferenceEngine engine;
  Status status = engine.Initialize(options.model_directory, planned_context,
                                    options.kv_cache_mode, options.sampling);
  if (!status.ok()) return status;
  const auto load_end = std::chrono::steady_clock::now();

  const auto run_once = [&]() -> Result<DecodeBenchmarkRun> {
    Status reset_status = engine.ResetCache();
    if (!reset_status.ok()) return reset_status;

    DecodeBenchmarkRun run;
    run.inter_token_latency_milliseconds.resize(options.generated_tokens);
    const auto prompt_start = std::chrono::steady_clock::now();
    std::uint32_t next_token = 0U;
    auto prefilled = engine.Prefill(prompt);
    if (!prefilled.ok()) return prefilled.status();
    next_token = prefilled.value();
    const auto prompt_end = std::chrono::steady_clock::now();
    run.prompt_milliseconds = Milliseconds(prompt_end - prompt_start);
    run.first_output_token_id = next_token;
    run.output_token_checksum = UpdateTokenChecksum(14695981039346656037ULL, next_token);

    const auto decode_start = std::chrono::steady_clock::now();
    for (std::uint32_t generated = 0U; generated < options.generated_tokens; ++generated) {
      const std::uint64_t position =
          static_cast<std::uint64_t>(options.context_tokens) + generated;
      const auto token_start = std::chrono::steady_clock::now();
      auto forwarded = engine.Forward(next_token, position, true);
      if (!forwarded.ok()) return forwarded.status();
      const auto token_end = std::chrono::steady_clock::now();
      next_token = forwarded.value();
      run.inter_token_latency_milliseconds[generated] =
          Milliseconds(token_end - token_start);
      run.output_token_checksum = UpdateTokenChecksum(run.output_token_checksum, next_token);
    }
    const auto decode_end = std::chrono::steady_clock::now();
    run.decode_milliseconds = Milliseconds(decode_end - decode_start);
    run.decode_tokens_per_second =
        static_cast<double>(options.generated_tokens) * 1000.0 /
        run.decode_milliseconds;
    run.last_output_token_id = next_token;
    return run;
  };

  for (std::uint32_t warmup = 0U; warmup < options.warmup_runs; ++warmup) {
    auto discarded = run_once();
    if (!discarded.ok()) return discarded.status();
  }

  DecodeBenchmarkResult result;
  result.options = options;
  result.model_load_milliseconds = Milliseconds(load_end - load_start);
  result.weight_arena_bytes = engine.weight_bytes();
  result.kv_cache_bytes = engine.cache_bytes();
  result.workspace_bytes = engine.workspace_bytes();
  result.decode_graph_device_bytes = engine.decode_graph_device_bytes();
  result.prefill_chunk_tokens = engine.prefill_chunk_tokens();
  result.runs.reserve(options.measured_runs);
  std::vector<double> prompt_samples;
  std::vector<double> throughput_samples;
  std::vector<double> latency_samples;
  prompt_samples.reserve(options.measured_runs);
  throughput_samples.reserve(options.measured_runs);
  latency_samples.reserve(static_cast<std::size_t>(options.measured_runs) *
                          options.generated_tokens);
  for (std::uint32_t repetition = 0U; repetition < options.measured_runs; ++repetition) {
    auto measured = run_once();
    if (!measured.ok()) return measured.status();
    prompt_samples.push_back(measured.value().prompt_milliseconds);
    throughput_samples.push_back(measured.value().decode_tokens_per_second);
    latency_samples.insert(latency_samples.end(),
                           measured.value().inter_token_latency_milliseconds.begin(),
                           measured.value().inter_token_latency_milliseconds.end());
    result.runs.push_back(std::move(measured.value()));
  }
  result.prompt_milliseconds = Summarize(prompt_samples);
  result.decode_tokens_per_second = Summarize(throughput_samples);
  result.inter_token_latency_milliseconds = Summarize(latency_samples);
  result.deterministic_outputs = std::all_of(
      result.runs.begin(), result.runs.end(),
      [&result](const DecodeBenchmarkRun& run) {
        return run.output_token_checksum == result.runs.front().output_token_checksum;
      });
  return result;
}


}  // namespace gem16

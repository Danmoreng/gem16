#include "gem16/engine.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <numeric>
#include <ostream>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace gem16 {
namespace {

constexpr std::uint64_t kVocabulary = 262144U;
constexpr std::uint64_t kSlidingWindow = 1024U;

Status Error(StatusCode code, std::string message) {
  return Status(code, std::move(message));
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

void WriteDistributionJson(std::ostream& output,
                           const BenchmarkDistribution& distribution) {
  output << "{\"sample_count\":" << distribution.sample_count
         << ",\"mean\":" << distribution.mean
         << ",\"median\":" << distribution.median
         << ",\"standard_deviation\":" << distribution.standard_deviation
         << ",\"minimum\":" << distribution.minimum
         << ",\"maximum\":" << distribution.maximum
         << ",\"p95\":" << distribution.p95
         << ",\"p99\":" << distribution.p99
         << ",\"confidence_95\":[" << distribution.confidence_95_low << ','
         << distribution.confidence_95_high << "]}";
}

}  // namespace

Status WriteGreedyInferenceJson(const GreedyInferenceResult& result, std::ostream& output) {
  output << "{\n  \"schema_version\": 1,\n"
         << "  \"status\": \"characterization\",\n"
         << "  \"benchmark_qualified\": false,\n"
         << "  \"precision\": \"bf16_state_fp8_attention_nvfp4_mlp\",\n"
         << "  \"projection_path\": \"native_sm120\",\n"
         << "  \"decode_attention_path\": \""
         << (result.kv_cache_mode == KvCacheMode::kCheckpointFp8 &&
                     result.max_context_tokens >= 65536U
                 ? "fp8_online_split_global_gqa_shared_fp8x4"
                 : result.kv_cache_mode == KvCacheMode::kCheckpointFp8 &&
                           result.max_context_tokens >= 16384U
                       ? "fp8_online_split_global_gqa_shared_scalar"
                       : result.kv_cache_mode == KvCacheMode::kCheckpointFp8 &&
                                 result.max_context_tokens > 512U
                             ? "fp8_online_split_gqa"
                       : "score_softmax_value_reference")
         << "\",\n"
         << "  \"kv_cache_mode\": \""
         << (result.kv_cache_mode == KvCacheMode::kCheckpointFp8
                 ? "checkpoint_fp8"
                 : "bf16_correctness")
         << "\",\n"
         << "  \"kv_cache_storage\": \""
         << (result.kv_cache_mode == KvCacheMode::kCheckpointFp8
                 ? "uint8_e4m3fn"
                 : "float32_bf16_semantics")
         << "\",\n"
         << "  \"kv_cache_layout\": \"hybrid_local_ring_global_contiguous\",\n"
         << "  \"local_attention_window\": " << kSlidingWindow << ",\n"
         << "  \"decoding_mode\": \""
         << (result.teacher_forcing
                 ? "teacher_forced"
                 : (result.sampling.enabled ? "sampled" : "greedy"))
         << "\",\n"
         << "  \"sampling\": {\"enabled\":"
         << (result.sampling.enabled ? "true" : "false")
         << ",\"temperature\":" << result.sampling.temperature
         << ",\"top_k\":" << result.sampling.top_k
         << ",\"top_p\":" << result.sampling.top_p
         << ",\"min_p\":" << result.sampling.min_p
         << ",\"repetition_penalty\":"
         << result.sampling.repetition_penalty
         << ",\"seed\":" << result.sampling.seed << "},\n"
         << "  \"fallbacks\": " << result.fallback_count << ",\n"
         << "  \"packed_weight_source_layout_direct\": "
         << (result.packed_weight_source_layout_direct ? "true" : "false") << ",\n"
         << "  \"weight_layout\": \"sm120_row8_k64\",\n"
         << "  \"weight_scale_layout\": \"sm120_row8_k64\",\n"
         << "  \"load_time_weight_swizzle\": true,\n"
         << "  \"load_time_scale_swizzle\": true,\n"
         << "  \"persistent_repack_bytes\": 0,\n"
         << "  \"token_loop_allocations\": "
         << (result.token_loop_allocations ? "true" : "false") << ",\n"
         << "  \"fused_gate_up\": false,\n"
         << "  \"fused_prefill_attention\": true,\n"
         << "  \"fp8_prefill_tile\": \"cutlass_m128n128k64\",\n"
         << "  \"fp8_prefill_output\": \"scaled_bf16\",\n"
         << "  \"nvfp4_gate_up_prefill_tile\": \"cutlass_m128n128k128\",\n"
         << "  \"nvfp4_gate_up_prefill_weight_scratch\": true,\n"
         << "  \"nvfp4_down_prefill_tile\": \"cutlass_m128n128k128\",\n"
         << "  \"fp8_prefill_pipeline_stages\": 0,\n"
         << "  \"fp8_prefill_schedule\": \"cutlass_auto\",\n"
         << "  \"local_prefill_query_heads_per_cta\": 2,\n"
         << "  \"global_prefill_query_heads_per_cta\": 4,\n"
         << "  \"local_prefill_fp8_staging\": \"async_fp8x16_fp8x4_bf16x2\",\n"
         << "  \"global_prefill_fp8_staging\": "
            "\"async_contiguous_fp8x16_fp8x4_bf16x2\",\n"
         << "  \"grouped_qkv_prefill\": false,\n"
         << "  \"grouped_qkv_decode\": true,\n"
         << "  \"fused_rmsnorm_boundaries\": true,\n"
         << "  \"fused_prefill_rmsnorm_fp8_quantization\": true,\n"
         << "  \"fused_prefill_rmsnorm_nvfp4_quantization\": true,\n"
         << "  \"fused_prefill_gated_gelu_nvfp4_quantization\": true,\n"
         << "  \"fused_prefill_qk_rmsnorm_rope\": true,\n"
         << "  \"prefill_rope_table\": \"precomputed_exact_max_context\",\n"
         << "  \"fused_output_head\": true,\n"
         << "  \"decode_graphs\": "
         << (result.decode_graphs ? "true" : "false") << ",\n"
         << "  \"model_load_ms\": " << result.model_load_milliseconds << ",\n"
         << "  \"prompt_ms\": " << result.prompt_milliseconds << ",\n"
         << "  \"decode_ms\": " << result.decode_milliseconds << ",\n"
         << "  \"decode_tokens_per_second\": " << result.decode_tokens_per_second << ",\n"
         << "  \"weight_arena_bytes\": " << result.weight_arena_bytes << ",\n"
         << "  \"assistant\": {\"loaded\":"
         << (result.assistant_loaded ? "true" : "false")
         << ",\"execution_enabled\":"
         << (result.mtp_enabled ? "true" : "false")
         << ",\"tensor_count\":" << result.assistant_tensor_count
         << ",\"source_bytes\":" << result.assistant_source_bytes
         << ",\"arena_bytes\":" << result.assistant_weight_arena_bytes
         << ",\"workspace_bytes\":" << result.assistant_workspace_bytes
         << ",\"attention_path\":\""
         << (!result.mtp_enabled
                 ? "disabled"
                 : result.kv_cache_mode == KvCacheMode::kCheckpointFp8
                       ? "fp8_online_split_long_reference_short"
                       : "bf16_score_softmax_value_reference")
         << "\",\"device_memory_delta_bytes\":"
         << result.assistant_device_memory_delta_bytes << "},\n"
         << "  \"mtp\": {\"enabled\":"
         << (result.mtp_enabled ? "true" : "false")
         << ",\"verification_mode\":\""
         << (result.mtp_enabled ? "batched_exact_target" : "disabled")
         << "\",\"acceptance_path\":\""
         << (result.mtp_enabled ? "gpu_accept_commit" : "disabled")
         << "\",\"device_control\":\""
         << (result.mtp_enabled ? "host_gpu_transition_parity" : "disabled")
         << "\",\"group_execution\":\""
         << (!result.mtp_enabled
                 ? "disabled"
                 : result.mtp_gpu_chained
                       ? "gpu_chained_fixed_d2_conditional_graph"
                 : result.mtp_fixed_d2_graph
                       ? "complete_fixed_d2_cuda_graph_host_replay"
                       : "direct_launch")
         << "\",\"host_synchronizations_per_group\":"
         << (result.mtp_enabled && !result.mtp_gpu_chained ? 1 : 0)
         << ",\"host_synchronizations_per_chain\":"
         << 0
         << ",\"streaming\":\""
         << (result.mtp_gpu_chained
                 ? "mapped_pinned_spsc_ring_256"
                 : "synchronous_group_result")
         << "\",\"short_batch_projection_path\":\""
         << (!result.mtp_enabled
                 ? "disabled"
                 : result.mtp_draft_tokens == 2U
                       ? "decode_order_fp8_t3_shared_qkv_o_nvfp4_down8"
                       : "decode_order_fp8_qkv_nvfp4_down_t_le_5")
         << "\",\"d2_attention_path\":\""
         << (!result.mtp_enabled
                 ? "disabled"
                 : result.mtp_draft_tokens == 2U
                       ? (result.max_context_tokens >= 65536U
                              ? "global_t3_row_gqa_shared_fp8x4_local_serial_exact"
                              : result.max_context_tokens >= 16384U
                                    ? "global_t3_row_gqa_shared_scalar_local_serial_exact"
                                    : "global_t3_shared_kv_local_serial_exact")
                       : "inactive")
         << "\",\"d2_output_head_path\":\""
         << (!result.mtp_enabled
                 ? "disabled"
                 : result.mtp_draft_tokens == 2U
                       ? "fixed_rows_3_exact_softcap"
                       : "inactive")
         << "\",\"adaptive\":"
         << (result.mtp_adaptive ? "true" : "false")
         << ",\"draft_tokens\":" << result.mtp_draft_tokens
         << ",\"d1_groups\":" << result.mtp_d1_groups
         << ",\"d2_groups\":" << result.mtp_d2_groups
         << ",\"d4_groups\":" << result.mtp_d4_groups
         << ",\"ordinary_fallback_tokens\":"
         << result.mtp_ordinary_fallback_tokens
         << ",\"proposed_tokens\":" << result.mtp_proposed_tokens
         << ",\"accepted_tokens\":" << result.mtp_accepted_tokens
         << ",\"rejected_tokens\":" << result.mtp_rejected_tokens
         << ",\"verification_groups\":"
         << result.mtp_verification_groups
         << ",\"target_forwards\":" << result.mtp_target_forwards
         << ",\"target_batches\":" << result.mtp_target_batches
         << ",\"mean_accepted_length\":"
         << (result.mtp_verification_groups == 0U
                 ? 0.0
                 : static_cast<double>(result.mtp_accepted_tokens) /
                       static_cast<double>(result.mtp_verification_groups))
         << ",\"proposed_token_ids\":[";
  for (std::size_t index = 0; index < result.mtp_proposed_token_ids.size();
       ++index) {
    if (index != 0U) output << ',';
    output << result.mtp_proposed_token_ids[index];
  }
  output << "]},\n"
         << "  \"kv_cache_bytes\": " << result.kv_cache_bytes << ",\n"
         << "  \"workspace_bytes\": " << result.workspace_bytes << ",\n"
         << "  \"prefill_chunk_tokens\": " << result.prefill_chunk_tokens << ",\n"
         << "  \"decode_graph_device_bytes\": "
         << result.decode_graph_device_bytes << ",\n"
         << "  \"logits_dumped\": " << (result.logits_dumped ? "true" : "false") << ",\n"
         << "  \"logits_dump_format\": \"raw_float32_little_endian\",\n"
         << "  \"logits_dump_steps\": " << result.logits_dump_steps << ",\n"
         << "  \"logits_dump_vocabulary\": " << kVocabulary << ",\n"
         << "  \"state_dumped\": "
         << (result.state_dumped ? "true" : "false") << ",\n"
         << "  \"state_dump_format\": \"gem16_layer_state_v5\",\n"
         << "  \"state_dump_position\": ";
  if (result.state_dumped) {
    output << result.state_dump_position;
  } else {
    output << "null";
  }
  output << ",\n"
         << "  \"finish_reason\": \"" << (result.stopped ? "stop" : "length") << "\",\n"
         << "  \"stop_token_id\": ";
  if (result.stopped) {
    output << result.stop_token_id;
  } else {
    output << "null";
  }
  output << ",\n"
         << "  \"output_token_ids\": [";
  for (std::size_t index = 0; index < result.output_token_ids.size(); ++index) {
    if (index != 0U) output << ',';
    output << result.output_token_ids[index];
  }
  output << "],\n"
         << "  \"teacher_forced_token_ids\": [";
  for (std::size_t index = 0; index < result.teacher_forced_token_ids.size(); ++index) {
    if (index != 0U) output << ',';
    output << result.teacher_forced_token_ids[index];
  }
  output << "],\n"
         << "  \"teacher_forced_matches\": "
         << result.teacher_forced_matches << "\n}\n";
  return output.good() ? Status::Ok()
                       : Error(StatusCode::kIoError, "failed to write inference JSON");
}

Status WriteDecodeBenchmarkJson(const DecodeBenchmarkResult& result,
                                std::ostream& output) {
  output << std::setprecision(17)
         << "{\"schema_version\":1,\"status\":\"characterization\","
         << "\"benchmark_qualified\":false,\"mode\":\"decode\",\"batch_size\":1,"
         << "\"precision\":\"bf16_state_fp8_attention_nvfp4_mlp\","
         << "\"projection_path\":\"native_sm120\","
         << "\"decode_attention_path\":\""
         << (result.options.kv_cache_mode == KvCacheMode::kCheckpointFp8 &&
                     static_cast<std::uint64_t>(result.options.context_tokens) +
                             result.options.generated_tokens >=
                         65536U
                 ? "fp8_online_split_global_gqa_shared_fp8x4"
                 : result.options.kv_cache_mode ==
                               KvCacheMode::kCheckpointFp8 &&
                           static_cast<std::uint64_t>(
                               result.options.context_tokens) +
                                   result.options.generated_tokens >=
                               16384U
                       ? "fp8_online_split_global_gqa_shared_scalar"
                       : result.options.kv_cache_mode ==
                                     KvCacheMode::kCheckpointFp8 &&
                                 static_cast<std::uint64_t>(
                                     result.options.context_tokens) +
                                         result.options.generated_tokens >
                                     512U
                             ? "fp8_online_split_gqa"
                       : "score_softmax_value_reference")
         << "\",\"kv_cache_mode\":\""
         << (result.options.kv_cache_mode == KvCacheMode::kCheckpointFp8
                 ? "checkpoint_fp8" : "bf16_correctness")
         << "\",\"kv_cache_layout\":\"hybrid_local_ring_global_contiguous"
         << "\",\"fused_gate_up\":false"
         << ",\"fused_prefill_attention\":true"
         << ",\"fp8_prefill_tile\":\"cutlass_m128n128k64\""
         << ",\"fp8_prefill_output\":\"scaled_bf16\""
         << ",\"nvfp4_gate_up_prefill_tile\":\"cutlass_m128n128k128\""
         << ",\"nvfp4_gate_up_prefill_weight_scratch\":true"
         << ",\"nvfp4_down_prefill_tile\":\"cutlass_m128n128k128\""
         << ",\"fp8_prefill_pipeline_stages\":0"
         << ",\"fp8_prefill_schedule\":\"cutlass_auto\""
         << ",\"local_prefill_query_heads_per_cta\":2"
         << ",\"global_prefill_query_heads_per_cta\":4"
         << ",\"local_prefill_fp8_staging\":\"async_fp8x16_fp8x4_bf16x2\""
         << ",\"global_prefill_fp8_staging\":"
            "\"async_contiguous_fp8x16_fp8x4_bf16x2\""
         << ",\"grouped_qkv_prefill\":false"
         << ",\"grouped_qkv_decode\":true"
         << ",\"fused_rmsnorm_boundaries\":true"
         << ",\"fused_prefill_rmsnorm_fp8_quantization\":true"
         << ",\"fused_prefill_rmsnorm_nvfp4_quantization\":true"
         << ",\"fused_prefill_gated_gelu_nvfp4_quantization\":true"
         << ",\"fused_prefill_qk_rmsnorm_rope\":true"
         << ",\"prefill_rope_table\":\"precomputed_exact_max_context\""
         << ",\"fused_output_head\":true"
         << ",\"decode_graphs\":true"
         << ",\"decoding_mode\":\""
         << (result.options.sampling.enabled ? "sampled" : "greedy") << '"'
         << ",\"sampling\":{\"enabled\":"
         << (result.options.sampling.enabled ? "true" : "false")
         << ",\"temperature\":" << result.options.sampling.temperature
         << ",\"top_k\":" << result.options.sampling.top_k
         << ",\"top_p\":" << result.options.sampling.top_p
         << ",\"min_p\":" << result.options.sampling.min_p
         << ",\"repetition_penalty\":"
         << result.options.sampling.repetition_penalty
         << ",\"seed\":" << result.options.sampling.seed << '}'
         << ",\"context_tokens\":" << result.options.context_tokens
         << ",\"generated_tokens\":" << result.options.generated_tokens
         << ",\"warmup_runs\":" << result.options.warmup_runs
         << ",\"measured_runs\":" << result.options.measured_runs
         << ",\"prompt_seed\":" << result.options.prompt_seed
         << ",\"prompt_token_formula\":\"1000+((seed+index*7919)%9000)\","
         << "\"timing_boundary\":\"host_end_to_end_forward_and_gpu_selection\","
         << "\"first_selected_token_excluded_from_decode\":true,"
         << "\"model_loaded_once\":true,\"cache_reset_outside_timing\":true,"
         << "\"prefill_path\":\"native_chunked_sm120\","
         << "\"packed_weight_source_layout_direct\":"
         << (result.packed_weight_source_layout_direct ? "true" : "false")
         << ",\"weight_layout\":\"sm120_row8_k64\""
         << ",\"weight_scale_layout\":\"sm120_row8_k64\""
         << ",\"load_time_weight_swizzle\":true"
         << ",\"load_time_scale_swizzle\":true"
         << ",\"persistent_repack_bytes\":0"
         << ",\"token_loop_allocations\":" << (result.token_loop_allocations ? "true" : "false")
         << ",\"deterministic_outputs\":" << (result.deterministic_outputs ? "true" : "false")
         << ",\"model_load_ms\":" << result.model_load_milliseconds
         << ",\"memory_bytes\":{\"weights\":" << result.weight_arena_bytes
         << ",\"kv_cache\":" << result.kv_cache_bytes
         << ",\"workspace\":" << result.workspace_bytes << "},"
         << "\"prefill_chunk_tokens\":" << result.prefill_chunk_tokens << ','
         << "\"decode_graph_device_bytes\":"
         << result.decode_graph_device_bytes << ','
         << "\"summary\":{\"time_to_first_token_ms\":";
  WriteDistributionJson(output, result.prompt_milliseconds);
  output << ",\"decode_tokens_per_second\":";
  WriteDistributionJson(output, result.decode_tokens_per_second);
  output << ",\"inter_token_latency_ms\":";
  WriteDistributionJson(output, result.inter_token_latency_milliseconds);
  output << "},\"runs\":[";
  for (std::size_t run_index = 0; run_index < result.runs.size(); ++run_index) {
    if (run_index != 0U) output << ',';
    const DecodeBenchmarkRun& run = result.runs[run_index];
    output << "{\"run\":" << run_index
           << ",\"time_to_first_token_ms\":" << run.prompt_milliseconds
           << ",\"decode_ms\":" << run.decode_milliseconds
           << ",\"decode_tokens_per_second\":" << run.decode_tokens_per_second
           << ",\"first_output_token_id\":" << run.first_output_token_id
           << ",\"last_output_token_id\":" << run.last_output_token_id
           << ",\"output_token_checksum\":" << run.output_token_checksum
           << ",\"inter_token_latency_ms\":[";
    for (std::size_t index = 0; index < run.inter_token_latency_milliseconds.size(); ++index) {
      if (index != 0U) output << ',';
      output << run.inter_token_latency_milliseconds[index];
    }
    output << "]}";
  }
  output << "]}\n";
  return output.good() ? Status::Ok()
                       : Error(StatusCode::kIoError,
                               "failed to write decode benchmark JSON");
}

Status WritePrefillBenchmarkJson(const DecodeBenchmarkResult& result,
                                 std::ostream& output) {
  std::vector<double> throughput;
  throughput.reserve(result.runs.size());
  for (const auto& run : result.runs) {
    throughput.push_back(static_cast<double>(result.options.context_tokens) *
                         1000.0 / run.prompt_milliseconds);
  }
  const BenchmarkDistribution throughput_summary = Summarize(throughput);
  output << std::setprecision(17)
         << "{\"schema_version\":1,\"status\":\"characterization\","
         << "\"benchmark_qualified\":false,\"mode\":\"prefill\",\"batch_size\":1,"
         << "\"prompt_tokens\":" << result.options.context_tokens
         << ",\"warmup_runs\":" << result.options.warmup_runs
         << ",\"measured_runs\":" << result.options.measured_runs
         << ",\"prefill_path\":\"native_chunked_sm120\""
         << ",\"fused_prefill_attention\":true"
         << ",\"fp8_prefill_tile\":\"cutlass_m128n128k64\""
         << ",\"fp8_prefill_output\":\"scaled_bf16\""
         << ",\"nvfp4_gate_up_prefill_tile\":\"cutlass_m128n128k128\""
         << ",\"nvfp4_gate_up_prefill_weight_scratch\":true"
         << ",\"nvfp4_down_prefill_tile\":\"cutlass_m128n128k128\""
         << ",\"fp8_prefill_pipeline_stages\":0"
         << ",\"fp8_prefill_schedule\":\"cutlass_auto\""
         << ",\"local_prefill_query_heads_per_cta\":2"
         << ",\"global_prefill_query_heads_per_cta\":4"
         << ",\"local_prefill_fp8_staging\":\"async_fp8x16_fp8x4_bf16x2\""
         << ",\"global_prefill_fp8_staging\":"
            "\"async_contiguous_fp8x16_fp8x4_bf16x2\""
         << ",\"grouped_qkv_prefill\":false"
         << ",\"grouped_qkv_decode\":true"
         << ",\"fused_rmsnorm_boundaries\":true"
         << ",\"fused_prefill_rmsnorm_fp8_quantization\":true"
         << ",\"fused_prefill_rmsnorm_nvfp4_quantization\":true"
         << ",\"fused_prefill_gated_gelu_nvfp4_quantization\":true"
         << ",\"fused_prefill_qk_rmsnorm_rope\":true"
         << ",\"prefill_rope_table\":\"precomputed_exact_max_context\""
         << ",\"decode_graphs\":true"
         << ",\"kv_cache_layout\":\"hybrid_local_ring_global_contiguous\","
         << "\"model_load_ms\":" << result.model_load_milliseconds
         << ",\"memory_bytes\":{\"weights\":" << result.weight_arena_bytes
         << ",\"kv_cache\":" << result.kv_cache_bytes
         << ",\"workspace\":" << result.workspace_bytes << "},"
         << "\"prefill_chunk_tokens\":" << result.prefill_chunk_tokens << ','
         << "\"decode_graph_device_bytes\":"
         << result.decode_graph_device_bytes << ','
         << "\"summary\":{\"prompt_tokens_per_second\":";
  WriteDistributionJson(output, throughput_summary);
  output << ",\"time_to_first_token_ms\":";
  WriteDistributionJson(output, result.prompt_milliseconds);
  output << "},\"runs\":[";
  for (std::size_t index = 0; index < result.runs.size(); ++index) {
    if (index != 0U) output << ',';
    output << "{\"run\":" << index
           << ",\"prompt_ms\":" << result.runs[index].prompt_milliseconds
           << ",\"prompt_tokens_per_second\":" << throughput[index]
           << ",\"first_output_token_id\":"
           << result.runs[index].first_output_token_id << '}';
  }
  output << "]}\n";
  return output.good() ? Status::Ok()
                       : Error(StatusCode::kIoError,
                               "failed to write prefill benchmark JSON");
}

}  // namespace gem16

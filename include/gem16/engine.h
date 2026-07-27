#pragma once

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "gem16/sampling.h"
#include "gem16/status.h"

namespace gem16 {

void PrintKernelCapabilities(std::ostream& output);

enum class KvCacheMode {
  kCheckpointFp8,
  kBf16Correctness,
};

using GeneratedTokenCallback = Status (*)(void* context,
                                          std::uint32_t token_id);

struct GreedyInferenceOptions {
  std::filesystem::path model_directory;
  std::vector<std::uint32_t> input_token_ids;
  // When non-empty, capture one prediction per target and feed the preceding
  // target token into later decode positions. This isolates per-position
  // model drift from autoregressive sequence drift.
  std::vector<std::uint32_t> teacher_forced_token_ids;
  std::vector<std::uint32_t> stop_token_ids;
  std::vector<std::uint32_t> suppressed_token_ids;
  std::filesystem::path logits_dump_path;
  std::filesystem::path state_dump_path;
  std::optional<std::uint64_t> state_dump_position;
  std::uint64_t max_generated_tokens = 1;
  std::uint64_t max_context_tokens = 128;
  KvCacheMode kv_cache_mode = KvCacheMode::kCheckpointFp8;
  SamplingOptions sampling;
  // Optional synchronous observer invoked once for every selected output
  // token, including a stop token. The callback and its context must remain
  // valid for the duration of RunGreedyInference. Benchmark callers leave it
  // null so terminal I/O never enters benchmark timing.
  GeneratedTokenCallback generated_token_callback = nullptr;
  void* generated_token_callback_context = nullptr;
};

struct GreedyInferenceResult {
  std::vector<std::uint32_t> output_token_ids;
  std::vector<std::uint32_t> teacher_forced_token_ids;
  std::uint32_t stop_token_id = 0;
  double model_load_milliseconds = 0.0;
  double prompt_milliseconds = 0.0;
  double decode_milliseconds = 0.0;
  double decode_tokens_per_second = 0.0;
  std::uint64_t weight_arena_bytes = 0;
  std::uint64_t kv_cache_bytes = 0;
  std::uint64_t workspace_bytes = 0;
  std::uint64_t decode_graph_device_bytes = 0;
  std::uint64_t prefill_chunk_tokens = 0;
  std::uint64_t max_context_tokens = 0;
  std::uint64_t fallback_count = 0;
  std::uint64_t logits_dump_steps = 0;
  std::uint64_t teacher_forced_matches = 0;
  std::uint64_t state_dump_position = 0;
  KvCacheMode kv_cache_mode = KvCacheMode::kCheckpointFp8;
  SamplingOptions sampling;
  bool packed_weight_source_layout_direct = false;
  bool token_loop_allocations = false;
  bool benchmark_qualified = false;
  bool stopped = false;
  bool teacher_forcing = false;
  bool decode_graphs = false;
  bool logits_dumped = false;
  bool state_dumped = false;
};

struct ConversationSessionOptions {
  std::filesystem::path model_directory;
  std::vector<std::uint32_t> stop_token_ids;
  std::vector<std::uint32_t> suppressed_token_ids;
  std::uint64_t max_context_tokens = 1024;
  KvCacheMode kv_cache_mode = KvCacheMode::kCheckpointFp8;
  SamplingOptions sampling;
};

// A batch-one conversation owns one resident model, workspace, and KV cache.
// Each turn supplies the fully rendered conversation so the session can prove
// that its existing cache is an exact token prefix before processing only the
// newly appended suffix.
class ConversationSession {
 public:
  ConversationSession(const ConversationSession&) = delete;
  ConversationSession& operator=(const ConversationSession&) = delete;
  ConversationSession(ConversationSession&&) noexcept;
  ConversationSession& operator=(ConversationSession&&) noexcept;
  ~ConversationSession();

  [[nodiscard]] static Result<ConversationSession> Create(
      const ConversationSessionOptions& options);
  [[nodiscard]] Result<GreedyInferenceResult> Generate(
      std::span<const std::uint32_t> full_prompt_token_ids,
      std::uint64_t max_generated_tokens,
      GeneratedTokenCallback generated_token_callback = nullptr,
      void* generated_token_callback_context = nullptr);
  [[nodiscard]] std::uint64_t cached_token_count() const;

 private:
  struct Impl;
  explicit ConversationSession(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

struct BenchmarkDistribution {
  std::uint64_t sample_count = 0;
  double mean = 0.0;
  double median = 0.0;
  double standard_deviation = 0.0;
  double minimum = 0.0;
  double maximum = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
  double confidence_95_low = 0.0;
  double confidence_95_high = 0.0;
};

struct DecodeBenchmarkOptions {
  std::filesystem::path model_directory;
  std::uint32_t context_tokens = 128;
  std::uint32_t generated_tokens = 256;
  std::uint32_t warmup_runs = 3;
  std::uint32_t measured_runs = 10;
  std::uint32_t prompt_seed = 0;
  KvCacheMode kv_cache_mode = KvCacheMode::kCheckpointFp8;
  SamplingOptions sampling;
};

struct DecodeBenchmarkRun {
  double prompt_milliseconds = 0.0;
  double decode_milliseconds = 0.0;
  double decode_tokens_per_second = 0.0;
  std::uint32_t first_output_token_id = 0;
  std::uint32_t last_output_token_id = 0;
  std::uint64_t output_token_checksum = 0;
  std::vector<double> inter_token_latency_milliseconds;
};

struct DecodeBenchmarkResult {
  DecodeBenchmarkOptions options;
  double model_load_milliseconds = 0.0;
  std::uint64_t weight_arena_bytes = 0;
  std::uint64_t kv_cache_bytes = 0;
  std::uint64_t workspace_bytes = 0;
  std::uint64_t decode_graph_device_bytes = 0;
  std::uint64_t prefill_chunk_tokens = 0;
  BenchmarkDistribution prompt_milliseconds;
  BenchmarkDistribution decode_tokens_per_second;
  BenchmarkDistribution inter_token_latency_milliseconds;
  std::vector<DecodeBenchmarkRun> runs;
  bool deterministic_outputs = false;
  bool packed_weight_source_layout_direct = false;
  bool token_loop_allocations = false;
  bool benchmark_qualified = false;
};

// Correctness-first, batch-one CUDA characterization. It accepts already-tokenized input,
// executes every decoder layer, and performs greedy or explicit sampled selection on the GPU.
// The legacy name is retained for API compatibility. The result remains explicitly unqualified
// until prompt-derived hidden-state and full-logit gates pass.
[[nodiscard]] Result<GreedyInferenceResult> RunGreedyInference(
    const GreedyInferenceOptions& options);
[[nodiscard]] Status WriteGreedyInferenceJson(const GreedyInferenceResult& result,
                                              std::ostream& output);

// Persistent-engine, batch-one decode characterization. The prompt token IDs are generated
// deterministically from prompt_seed, the first selected token is untimed for decode, and each
// requested generated token contributes one measured inter-token interval.
[[nodiscard]] Result<DecodeBenchmarkResult> RunDecodeBenchmark(
    const DecodeBenchmarkOptions& options);
[[nodiscard]] Status WriteDecodeBenchmarkJson(const DecodeBenchmarkResult& result,
                                              std::ostream& output);
[[nodiscard]] Status WritePrefillBenchmarkJson(const DecodeBenchmarkResult& result,
                                               std::ostream& output);

}  // namespace gem16

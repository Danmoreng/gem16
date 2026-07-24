#pragma once

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <optional>
#include <span>
#include <vector>

#include "gem16gb/status.h"

namespace gem16gb {

void PrintKernelCapabilities(std::ostream& output);

enum class ProjectionPath {
  kNativeSm120,
  kCudaReference,
};

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
  ProjectionPath projection_path = ProjectionPath::kNativeSm120;
  KvCacheMode kv_cache_mode = KvCacheMode::kCheckpointFp8;
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
  std::uint64_t fallback_count = 0;
  std::uint64_t logits_dump_steps = 0;
  std::uint64_t teacher_forced_matches = 0;
  std::uint64_t state_dump_position = 0;
  ProjectionPath projection_path = ProjectionPath::kNativeSm120;
  KvCacheMode kv_cache_mode = KvCacheMode::kCheckpointFp8;
  bool source_layout_direct = false;
  bool token_loop_allocations = false;
  bool benchmark_qualified = false;
  bool stopped = false;
  bool teacher_forcing = false;
  bool logits_dumped = false;
  bool state_dumped = false;
};

// Correctness-first, batch-one CUDA characterization. It accepts already-tokenized input,
// executes every decoder layer, and performs greedy selection on the GPU. The result remains
// explicitly unqualified until prompt-derived hidden-state and full-logit gates pass.
[[nodiscard]] Result<GreedyInferenceResult> RunGreedyInference(
    const GreedyInferenceOptions& options);
[[nodiscard]] Status WriteGreedyInferenceJson(const GreedyInferenceResult& result,
                                              std::ostream& output);

}  // namespace gem16gb

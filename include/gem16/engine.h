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

enum class CudaProfilePhase {
  kNone,
  kPrefill,
  kDecode,
};

struct DeviceMemoryInfo {
  std::uint64_t free_bytes = 0U;
  std::uint64_t total_bytes = 0U;
};

[[nodiscard]] Result<DeviceMemoryInfo> QueryDeviceMemoryInfo();

using GeneratedTokenCallback = Status (*)(void* context,
                                          std::uint32_t token_id);

struct AudioEmbeddingSegment {
  // Absolute prompt position of the first repeated <|audio|> token.
  std::uint64_t prompt_offset = 0U;
  // Row-major, one 640-sample float waveform frame per audio token.
  std::span<const float> frames;
};

struct VisionEmbeddingSegment {
  std::uint64_t prompt_offset = 0U;
  // Row-major [patch, 6912] and [patch, xy].
  std::span<const float> patches;
  std::span<const std::int32_t> positions;
};

struct Gemma4Moe26BVisionInputSegment {
  // Absolute prompt position of the first repeated <|image|> token.
  std::uint64_t prompt_offset = 0U;
  // Text-side image span after 3x3 spatial pooling.
  std::uint32_t soft_token_count = 0U;
  // Selected fixed padded capacity: 70, 140, or 280 soft tokens.
  std::uint32_t soft_token_budget = 0U;
  // Unpadded teacher-patch rows consumed by the Vision tower.
  std::uint32_t raw_patch_count = 0U;
  // Row-major [raw_patch, 768] and [raw_patch, xy].
  std::span<const float> patches;
  std::span<const std::int32_t> positions;
};

struct ReasoningTokenOptions {
  std::vector<std::uint32_t> channel_open_token_ids;
  std::uint32_t channel_close_token_id = 0U;
  std::uint64_t max_reasoning_tokens = 0U;
  bool enabled = false;
  // Tool-result continuations render the thinking-channel opener into the
  // prompt, so the first generated token is already inside reasoning.
  bool starts_in_reasoning = false;
};

struct GreedyInferenceOptions {
  std::filesystem::path model_directory;
  // Optional official BF16 MTP assistant, loaded and bound into an
  // independent fixed-address arena.
  std::filesystem::path assistant_model_directory;
  // Explicitly selects the Trellis35 Vision profile. File discovery never
  // enables Vision implicitly.
  std::filesystem::path vision_model_directory;
  // Zero keeps the residency-only gate. Active correctness generation
  // supports exactly 1, 2, or 4 assistant drafts per verification group.
  std::uint32_t mtp_draft_tokens = 0;
  // Explicit opt-in for deterministic context/acceptance-based selection up
  // to mtp_draft_tokens, including ordinary decode fallback when proposals do
  // not amortize target verification.
  bool mtp_adaptive = false;
  // Diagnostic-only fixed-D2 router instrumentation. When enabled, one tiny
  // counter kernel records Top-8 expert-set overlap across the three Target
  // verifier rows. The ordinary product graph is unchanged when this is false.
  bool mtp_router_overlap_diagnostic = false;
  CudaProfilePhase cuda_profile_phase = CudaProfilePhase::kNone;
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

struct MtpRouterOverlapDiagnostics {
  std::uint64_t verifier_layer_samples = 0U;
  std::uint64_t routed_assignments = 0U;
  std::uint64_t unique_experts_sum = 0U;
  std::uint64_t row01_intersection_sum = 0U;
  std::uint64_t row02_intersection_sum = 0U;
  std::uint64_t row12_intersection_sum = 0U;
  std::uint64_t triple_intersection_sum = 0U;
  // Index is the union cardinality. Fixed D2/Top-8 uses indices 8..24.
  std::vector<std::uint64_t> union_size_histogram;
  bool enabled = false;
};

struct GreedyInferenceResult {
  std::vector<std::uint32_t> output_token_ids;
  std::vector<std::uint32_t> teacher_forced_token_ids;
  std::vector<std::uint32_t> mtp_proposed_token_ids;
  std::string artifact_profile = "native-checkpoint";
  std::uint32_t stop_token_id = 0;
  double model_load_milliseconds = 0.0;
  double prompt_milliseconds = 0.0;
  double decode_milliseconds = 0.0;
  double decode_tokens_per_second = 0.0;
  double image_decode_milliseconds = 0.0;
  double image_resize_patchify_milliseconds = 0.0;
  double vision_upload_milliseconds = 0.0;
  double vision_tower_milliseconds = 0.0;
  double vision_pool_project_milliseconds = 0.0;
  double text_prefill_milliseconds = 0.0;
  std::uint64_t weight_arena_bytes = 0;
  std::uint64_t assistant_source_bytes = 0;
  std::uint64_t assistant_weight_arena_bytes = 0;
  std::uint64_t assistant_device_memory_delta_bytes = 0;
  std::uint64_t assistant_tensor_count = 0;
  std::uint64_t assistant_workspace_bytes = 0;
  std::uint64_t mtp_proposed_tokens = 0;
  std::uint64_t mtp_accepted_tokens = 0;
  std::uint64_t mtp_rejected_tokens = 0;
  std::uint64_t mtp_verification_groups = 0;
  std::uint64_t mtp_target_forwards = 0;
  std::uint64_t mtp_target_batches = 0;
  std::uint64_t mtp_d1_groups = 0;
  std::uint64_t mtp_d2_groups = 0;
  std::uint64_t mtp_d4_groups = 0;
  std::uint64_t mtp_ordinary_fallback_tokens = 0;
  std::uint64_t reasoning_tokens = 0;
  std::uint64_t reasoning_budget_tokens = 0;
  std::uint64_t reasoning_ordinary_target_tokens = 0;
  std::uint64_t prompt_cached_tokens = 0;
  std::uint64_t prompt_cache_write_tokens = 0;
  std::uint32_t mtp_draft_tokens = 0;
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
  MtpRouterOverlapDiagnostics mtp_router_overlap;
  bool packed_weight_source_layout_direct = false;
  bool assistant_loaded = false;
  bool mtp_enabled = false;
  bool mtp_adaptive = false;
  bool mtp_fixed_d2_graph = false;
  bool mtp_gpu_chained = false;
  bool reasoning_enabled = false;
  bool reasoning_budget_forced = false;
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
  std::filesystem::path assistant_model_directory;
  std::filesystem::path vision_model_directory;
  std::vector<std::uint32_t> stop_token_ids;
  std::vector<std::uint32_t> suppressed_token_ids;
  std::uint64_t max_context_tokens = 1024;
  KvCacheMode kv_cache_mode = KvCacheMode::kCheckpointFp8;
  SamplingOptions sampling;
  std::uint32_t mtp_draft_tokens = 0;
  bool mtp_adaptive = false;
  bool mtp_router_overlap_diagnostic = false;
  CudaProfilePhase cuda_profile_phase = CudaProfilePhase::kNone;
};

struct ModelRuntimeOptions {
  std::filesystem::path model_directory;
  std::filesystem::path assistant_model_directory;
  // The 12B runtime keeps context-owned state in ConversationSession. The
  // text-only 26B product profile owns exactly one execution slot, so its
  // fixed-address arena is admitted together with the resident runtime.
  std::uint64_t max_context_tokens = 1024U;
  int device = 0;
  // Structural mode validates the complete schema, geometry, ranges and
  // allocation plan without rereading every payload byte. Set this for a
  // llama.cpp-like fast server start; the default preserves full identity
  // verification for direct library callers.
  bool verify_device_image_sha256 = true;
  std::filesystem::path vision_model_directory;
};

// Process-wide immutable model residency. A runtime owns exactly one target
// weight arena and, when configured, one assistant weight arena. Conversation
// sessions share these arenas and allocate only mutable session/slot state.
class ModelRuntime {
 public:
  ModelRuntime(const ModelRuntime&) = delete;
  ModelRuntime& operator=(const ModelRuntime&) = delete;
  ~ModelRuntime();

  [[nodiscard]] static Result<std::shared_ptr<ModelRuntime>> Load(
      const ModelRuntimeOptions& options);
  [[nodiscard]] std::uint64_t weight_bytes() const;
  [[nodiscard]] std::uint64_t assistant_weight_bytes() const;
  [[nodiscard]] std::uint64_t assistant_workspace_bytes() const;
  [[nodiscard]] bool assistant_loaded() const;
  [[nodiscard]] std::uint64_t vision_weight_bytes() const;
  [[nodiscard]] std::uint64_t vision_workspace_bytes() const;
  [[nodiscard]] bool vision_module_loaded() const;
  [[nodiscard]] double load_milliseconds() const;
  [[nodiscard]] const char* weight_load_path() const;
  [[nodiscard]] const char* model_variant_name() const;
  [[nodiscard]] const char* selected_native_path() const;
  [[nodiscard]] const char* artifact_profile() const;
  [[nodiscard]] const char* head_format() const;
  [[nodiscard]] const char* artifact_content_sha256() const;
  [[nodiscard]] const char* source_lock_sha256() const;
  [[nodiscard]] const char* compiler_commit() const;
  [[nodiscard]] const char* profile_id() const;
  [[nodiscard]] const char* text_artifact_profile() const;
  [[nodiscard]] const char* vision_artifact_profile() const;
  [[nodiscard]] const char* qualification_state() const;
  [[nodiscard]] bool experimental() const;
  [[nodiscard]] std::uint64_t max_context_tokens() const;
  [[nodiscard]] std::uint64_t default_context_tokens() const;
  [[nodiscard]] std::uint64_t base_max_context_tokens() const;
  [[nodiscard]] bool qualified_64k() const;
  [[nodiscard]] std::uint64_t kv_cache_bytes() const;
  [[nodiscard]] std::uint64_t workspace_bytes() const;
  [[nodiscard]] bool supports_audio() const;
  [[nodiscard]] bool supports_vision() const;
  [[nodiscard]] bool supports_mtp() const;
  // True only for the exact validated 26B Trellis35 Target + FP8 Vision +
  // fixed-D2 Assistant combination. Generic Vision and MTP support must not
  // be combined to infer this capability.
  [[nodiscard]] bool vision_mtp_supported() const;
  [[nodiscard]] std::uint32_t maximum_images() const;
  [[nodiscard]] std::span<const std::uint32_t>
  vision_soft_token_budgets() const;
  // Zero means request-specific and not selected yet.
  [[nodiscard]] std::uint32_t selected_vision_soft_token_budget() const;
  [[nodiscard]] std::uint64_t vision_max_context_tokens() const;
  [[nodiscard]] std::uint32_t maximum_execution_slots() const;

 private:
  struct Impl;
  explicit ModelRuntime(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
  friend class ConversationSession;
};

// A batch-one conversation owns mutable token/KV state and one execution slot.
// The overload taking ModelRuntime shares process-wide immutable weights.
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
  [[nodiscard]] static Result<ConversationSession> Create(
      std::shared_ptr<ModelRuntime> runtime,
      const ConversationSessionOptions& options);
  [[nodiscard]] Result<GreedyInferenceResult> Generate(
      std::span<const std::uint32_t> full_prompt_token_ids,
      std::uint64_t max_generated_tokens,
      const ReasoningTokenOptions& reasoning = {},
      GeneratedTokenCallback generated_token_callback = nullptr,
      void* generated_token_callback_context = nullptr,
      std::span<const AudioEmbeddingSegment> audio_segments = {},
      std::span<const VisionEmbeddingSegment> vision_segments = {},
      std::span<const Gemma4Moe26BVisionInputSegment>
          moe26b_vision_segments = {});
  [[nodiscard]] std::uint64_t cached_token_count() const;
  [[nodiscard]] std::uint64_t reserved_device_bytes() const;
  [[nodiscard]] bool is_poisoned() const;

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

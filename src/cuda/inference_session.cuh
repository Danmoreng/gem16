// Internal implementation fragment included by inference.cu.
// It intentionally remains in the same CUDA translation unit so engine-private
// orchestration and kernel generation are unchanged.

Result<DeviceMemoryInfo> QueryDeviceMemoryInfo() {
  std::size_t free_bytes = 0U;
  std::size_t total_bytes = 0U;
  const cudaError_t error = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (error != cudaSuccess) return CudaFailure("cudaMemGetInfo", error);
  return DeviceMemoryInfo{static_cast<std::uint64_t>(free_bytes),
                          static_cast<std::uint64_t>(total_bytes)};
}

struct ModelRuntime::Impl {
  internal::LoadedTargetModel model;
  internal::AssistantModel assistant;
  std::unique_ptr<internal::Gemma4Moe26BReferenceEngine> moe26b_engine;
  internal::ModelVariant variant = internal::ModelVariant::kUnsupported;
  std::mutex moe26b_slot_mutex;
  std::string artifact_profile = "native-checkpoint";
  std::string head_format = "native-checkpoint";
  std::string artifact_content_sha256;
  std::string source_lock_sha256;
  std::string compiler_commit;
  std::uint64_t max_context_tokens = 0U;
  double load_milliseconds = 0.0;
  bool assistant_loaded = false;
  bool moe26b_slot_leased = false;
};

ModelRuntime::ModelRuntime(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
ModelRuntime::~ModelRuntime() = default;

Result<std::shared_ptr<ModelRuntime>> ModelRuntime::Load(
    const ModelRuntimeOptions& options) {
  if (options.model_directory.empty()) {
    return Error(StatusCode::kInvalidArgument,
                 "model runtime requires --model");
  }
  auto impl = std::make_unique<Impl>();
  const auto load_start = std::chrono::steady_clock::now();
  auto config = internal::LoadModelConfig(options.model_directory / "config.json");
  if (!config.ok()) return config.status();
  impl->variant = internal::ClassifyModelVariant(config.value());
  if (impl->variant == internal::ModelVariant::kGemma4Moe26BA4B) {
    if (!options.assistant_model_directory.empty()) {
      return Error(StatusCode::kUnsupported,
                   "Gemma 4 26B text-only does not support MTP assistant weights");
    }
    if (options.max_context_tokens == 0U) {
      return Error(StatusCode::kInvalidArgument,
                   "Gemma 4 26B requires a positive context capacity");
    }
    auto identity = internal::LoadGemma4Moe26BCompiledIdentity(
        options.model_directory);
    if (!identity.ok()) return identity.status();
    auto engine = internal::Gemma4Moe26BReferenceEngine::Create(
        options.model_directory, options.max_context_tokens, options.device,
        internal::Gemma4Moe26BBackend::kSm120Integrated);
    if (!engine.ok()) return engine.status();
    impl->max_context_tokens = options.max_context_tokens;
    impl->moe26b_engine =
        std::make_unique<internal::Gemma4Moe26BReferenceEngine>(
            std::move(engine).value());
    impl->artifact_profile = std::move(identity.value().artifact_profile);
    impl->head_format = std::move(identity.value().head_format);
    impl->artifact_content_sha256 =
        std::move(identity.value().artifact_content_sha256);
    impl->source_lock_sha256 = std::move(identity.value().source_lock_sha256);
    impl->compiler_commit = std::move(identity.value().compiler_commit);
  } else {
    Status status = impl->model.Load(options.model_directory);
    if (!status.ok()) return status;
    impl->max_context_tokens = kMaximumContext;
    if (!options.assistant_model_directory.empty()) {
      status = impl->assistant.Load(options.assistant_model_directory);
      if (!status.ok()) return status;
      impl->assistant_loaded = true;
    }
  }
  impl->load_milliseconds =
      Milliseconds(std::chrono::steady_clock::now() - load_start);
  return std::shared_ptr<ModelRuntime>(
      new ModelRuntime(std::move(impl)));
}

std::uint64_t ModelRuntime::weight_bytes() const {
  if (impl_ == nullptr) return 0U;
  return impl_->moe26b_engine == nullptr
             ? impl_->model.weight_bytes()
             : impl_->moe26b_engine->weight_arena_bytes();
}
std::uint64_t ModelRuntime::assistant_weight_bytes() const {
  return impl_ == nullptr ? 0U : impl_->assistant.arena_bytes();
}
bool ModelRuntime::assistant_loaded() const {
  return impl_ != nullptr && impl_->assistant_loaded;
}
double ModelRuntime::load_milliseconds() const {
  return impl_ == nullptr ? 0.0 : impl_->load_milliseconds;
}
const char* ModelRuntime::model_variant_name() const {
  return impl_ == nullptr
             ? "unsupported"
             : internal::ModelVariantName(impl_->variant).data();
}
const char* ModelRuntime::selected_native_path() const {
  if (impl_ == nullptr) return "none";
  return impl_->variant == internal::ModelVariant::kGemma4Moe26BA4B
             ? "sm120_integrated_nvfp4_moe_fp8_kv"
             : "gemma4_unified_12b_sm120";
}
const char* ModelRuntime::artifact_profile() const {
  return impl_ == nullptr ? "unsupported" : impl_->artifact_profile.c_str();
}
const char* ModelRuntime::head_format() const {
  return impl_ == nullptr ? "unsupported" : impl_->head_format.c_str();
}
const char* ModelRuntime::artifact_content_sha256() const {
  return impl_ == nullptr ? "" : impl_->artifact_content_sha256.c_str();
}
const char* ModelRuntime::source_lock_sha256() const {
  return impl_ == nullptr ? "" : impl_->source_lock_sha256.c_str();
}
const char* ModelRuntime::compiler_commit() const {
  return impl_ == nullptr ? "" : impl_->compiler_commit.c_str();
}
std::uint64_t ModelRuntime::max_context_tokens() const {
  return impl_ == nullptr ? 0U : impl_->max_context_tokens;
}
std::uint64_t ModelRuntime::kv_cache_bytes() const {
  if (impl_ == nullptr || impl_->moe26b_engine == nullptr) return 0U;
  return impl_->moe26b_engine->kv_cache_bytes();
}
std::uint64_t ModelRuntime::workspace_bytes() const {
  if (impl_ == nullptr || impl_->moe26b_engine == nullptr) return 0U;
  return impl_->moe26b_engine->workspace_bytes();
}
bool ModelRuntime::supports_audio() const {
  return impl_ != nullptr &&
         internal::TraitsForModelVariant(impl_->variant).supports_audio;
}
bool ModelRuntime::supports_vision() const {
  return impl_ != nullptr &&
         internal::TraitsForModelVariant(impl_->variant).supports_vision;
}
bool ModelRuntime::supports_mtp() const {
  return impl_ != nullptr &&
         internal::TraitsForModelVariant(impl_->variant).supports_mtp;
}
std::uint32_t ModelRuntime::maximum_execution_slots() const {
  if (impl_ == nullptr) return 0U;
  return impl_->variant == internal::ModelVariant::kGemma4Moe26BA4B ? 1U
                                                                    : UINT32_MAX;
}

// Host conversation state and device execution resources intentionally have
// distinct owners. This keeps the immutable runtime shareable without making
// RNG/history/KV or CUDA graph state process-global.
struct SessionState {
  std::vector<std::uint32_t> cached_token_ids;
  std::vector<std::uint32_t> stop_token_ids;
  std::uint64_t max_context_tokens = 0U;
  KvCacheMode kv_cache_mode = KvCacheMode::kCheckpointFp8;
  double model_load_milliseconds = 0.0;
  SamplingOptions sampling;
  std::uint32_t mtp_draft_tokens = 0U;
  bool mtp_adaptive = false;
  bool poisoned = false;
};

struct ExecutionSlot {
  InferenceEngine engine;
};

struct ConversationSession::Impl : SessionState, ExecutionSlot {
  std::shared_ptr<ModelRuntime> runtime;
  bool moe26b_slot_lease = false;

  ~Impl() {
    if (!moe26b_slot_lease || runtime == nullptr || runtime->impl_ == nullptr) {
      return;
    }
    std::lock_guard lock(runtime->impl_->moe26b_slot_mutex);
    runtime->impl_->moe26b_slot_leased = false;
  }
};

ConversationSession::ConversationSession(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

ConversationSession::ConversationSession(ConversationSession&&) noexcept =
    default;
ConversationSession& ConversationSession::operator=(
    ConversationSession&&) noexcept = default;
ConversationSession::~ConversationSession() = default;

Result<ConversationSession> ConversationSession::Create(
    const ConversationSessionOptions& options) {
  if (options.model_directory.empty()) {
    return Error(StatusCode::kInvalidArgument,
                 "conversation session requires --model");
  }
  if (options.max_context_tokens == 0U ||
      options.max_context_tokens > kMaximumContext) {
    return Error(StatusCode::kUnsupported,
                 "the hybrid KV cache supports 1..262144 tokens");
  }
  for (const std::uint32_t token : options.stop_token_ids) {
    if (token >= kVocabulary) {
      return Error(StatusCode::kInvalidArgument,
                   "stop token ID exceeds vocabulary");
    }
  }
  for (const std::uint32_t token : options.suppressed_token_ids) {
    if (token >= kVocabulary) {
      return Error(StatusCode::kInvalidArgument,
                   "suppressed token ID exceeds vocabulary");
    }
  }
  const bool mtp_enabled = options.mtp_draft_tokens != 0U;
  if (mtp_enabled && options.mtp_draft_tokens != 1U &&
      options.mtp_draft_tokens != 2U && options.mtp_draft_tokens != 4U) {
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

  auto runtime = ModelRuntime::Load(
      {options.model_directory, options.assistant_model_directory,
       options.max_context_tokens, 0});
  if (!runtime.ok()) return runtime.status();
  return Create(std::move(runtime).value(), options);
}

Result<ConversationSession> ConversationSession::Create(
    std::shared_ptr<ModelRuntime> runtime,
    const ConversationSessionOptions& options) {
  if (runtime == nullptr || runtime->impl_ == nullptr) {
    return Error(StatusCode::kInvalidArgument,
                 "conversation session requires a loaded model runtime");
  }
  const bool moe26b =
      runtime->impl_->variant == internal::ModelVariant::kGemma4Moe26BA4B;
  if (options.max_context_tokens == 0U ||
      options.max_context_tokens > kMaximumContext) {
    return Error(StatusCode::kUnsupported,
                 "the hybrid KV cache supports 1..262144 tokens");
  }
  for (const std::uint32_t token : options.stop_token_ids) {
    if (token >= kVocabulary) {
      return Error(StatusCode::kInvalidArgument,
                   "stop token ID exceeds vocabulary");
    }
  }
  for (const std::uint32_t token : options.suppressed_token_ids) {
    if (token >= kVocabulary) {
      return Error(StatusCode::kInvalidArgument,
                   "suppressed token ID exceeds vocabulary");
    }
  }
  const bool mtp_enabled = options.mtp_draft_tokens != 0U;
  if (moe26b &&
      (mtp_enabled || options.mtp_adaptive || runtime->impl_->assistant_loaded)) {
    return Error(StatusCode::kUnsupported,
                 "Gemma 4 26B text-only does not support MTP");
  }
  if (mtp_enabled && options.mtp_draft_tokens != 1U &&
      options.mtp_draft_tokens != 2U && options.mtp_draft_tokens != 4U) {
    return Error(StatusCode::kInvalidArgument,
                 "active MTP requires draft length 1, 2, or 4");
  }
  if (mtp_enabled && !runtime->impl_->assistant_loaded) {
    return Error(StatusCode::kInvalidArgument,
                 "active MTP requires assistant weights in ModelRuntime");
  }
  if (options.mtp_adaptive && !mtp_enabled) {
    return Error(StatusCode::kInvalidArgument,
                 "--mtp-adaptive requires active MTP");
  }
  if (moe26b) {
    if (options.max_context_tokens != runtime->impl_->max_context_tokens) {
      return Error(
          StatusCode::kInvalidArgument,
          "Gemma 4 26B session context must match its fixed resident runtime arena");
    }
    if (options.kv_cache_mode != KvCacheMode::kCheckpointFp8) {
      return Error(StatusCode::kUnsupported,
                   "Gemma 4 26B currently supports only the compiled FP8 KV cache");
    }
    if (runtime->impl_->moe26b_engine == nullptr) {
      return Error(StatusCode::kInternal,
                   "Gemma 4 26B resident engine is unavailable");
    }
  }

  auto impl = std::make_unique<Impl>();
  impl->runtime = std::move(runtime);
  impl->stop_token_ids = options.stop_token_ids;
  impl->max_context_tokens = options.max_context_tokens;
  impl->kv_cache_mode = options.kv_cache_mode;
  impl->sampling = options.sampling;
  impl->mtp_draft_tokens = options.mtp_draft_tokens;
  impl->mtp_adaptive = options.mtp_adaptive;
  impl->cached_token_ids.reserve(
      static_cast<std::size_t>(options.max_context_tokens));
  if (moe26b) {
    {
      std::lock_guard lock(impl->runtime->impl_->moe26b_slot_mutex);
      if (impl->runtime->impl_->moe26b_slot_leased) {
        return Error(
            StatusCode::kResourceExhausted,
            "Gemma 4 26B supports exactly one resident execution slot");
      }
      impl->runtime->impl_->moe26b_slot_leased = true;
      impl->moe26b_slot_lease = true;
    }
    Status status = impl->runtime->impl_->moe26b_engine->Reset();
    if (!status.ok()) return status;
    status = impl->runtime->impl_->moe26b_engine->ConfigureTokenSelection(
        options.sampling, options.suppressed_token_ids);
    if (!status.ok()) return status;
    impl->model_load_milliseconds = impl->runtime->load_milliseconds();
    return ConversationSession(std::move(impl));
  }
  Status status = impl->engine.InitializeShared(
      impl->runtime->impl_->model,
      impl->runtime->impl_->assistant_loaded
          ? &impl->runtime->impl_->assistant
          : nullptr,
      options.max_context_tokens, options.kv_cache_mode, options.sampling,
      options.mtp_draft_tokens);
  if (!status.ok()) return status;
  status = impl->engine.SetSuppressedTokens(options.suppressed_token_ids);
  if (!status.ok()) return status;
  status = impl->engine.SetMtpStopTokens(options.stop_token_ids);
  if (!status.ok()) return status;
  impl->model_load_milliseconds = impl->runtime->load_milliseconds();
  return ConversationSession(std::move(impl));
}

Result<GreedyInferenceResult> ConversationSession::Generate(
    std::span<const std::uint32_t> full_prompt_token_ids,
    std::uint64_t max_generated_tokens,
    const ReasoningTokenOptions& reasoning,
    GeneratedTokenCallback generated_token_callback,
    void* generated_token_callback_context,
    std::span<const AudioEmbeddingSegment> audio_segments,
    std::span<const VisionEmbeddingSegment> vision_segments) {
  if (impl_ == nullptr) {
    return Error(StatusCode::kInternal,
                 "conversation session was moved from");
  }
  if (impl_->poisoned) {
    return Error(StatusCode::kInternal,
                 "conversation session cannot continue after an inference failure");
  }
  if (full_prompt_token_ids.empty()) {
    return Error(StatusCode::kInvalidArgument,
                 "conversation turn requires prompt token IDs");
  }
  if (max_generated_tokens == 0U) {
    return Error(StatusCode::kInvalidArgument,
                 "--max-tokens must be positive");
  }
  if (reasoning.enabled &&
      (reasoning.channel_open_token_ids.empty() ||
       reasoning.channel_open_token_ids.size() >
           internal::kMaximumThinkingOpenTokens ||
       reasoning.channel_close_token_id >= kVocabulary ||
       reasoning.max_reasoning_tokens == 0U)) {
    return Error(StatusCode::kInvalidArgument,
                 "enabled reasoning requires channel tokens and a positive budget");
  }
  if (full_prompt_token_ids.size() > impl_->max_context_tokens ||
      max_generated_tokens - 1U >
          impl_->max_context_tokens - full_prompt_token_ids.size()) {
    return Error(StatusCode::kInvalidArgument,
                 "conversation prompt plus generated positions exceed --max-context");
  }
  for (const std::uint32_t token : full_prompt_token_ids) {
    if (token >= kVocabulary) {
      return Error(StatusCode::kInvalidArgument,
                   "input token ID exceeds vocabulary");
    }
  }
  const bool moe26b = impl_->runtime != nullptr &&
      impl_->runtime->impl_->variant ==
          internal::ModelVariant::kGemma4Moe26BA4B;
  if (moe26b && (!audio_segments.empty() || !vision_segments.empty())) {
    return Error(
        StatusCode::kUnsupported,
        "Gemma 4 26B is a text-only profile; audio and vision input are unsupported");
  }
  for (const AudioEmbeddingSegment& segment : audio_segments) {
    if (segment.frames.empty() || segment.frames.size() % 640U != 0U) {
      return Error(StatusCode::kInvalidArgument,
                   "audio segment requires complete 640-sample frames");
    }
    const std::uint64_t frame_count = segment.frames.size() / 640U;
    if (frame_count > 750U || segment.prompt_offset >= full_prompt_token_ids.size() ||
        frame_count > full_prompt_token_ids.size() - segment.prompt_offset) {
      return Error(StatusCode::kInvalidArgument,
                   "audio segment does not fit in the rendered prompt");
    }
    for (std::uint64_t frame = 0U; frame < frame_count; ++frame) {
      if (full_prompt_token_ids[segment.prompt_offset + frame] != 258881U) {
        return Error(StatusCode::kInvalidArgument,
                     "audio segment is not aligned with <|audio|> prompt tokens");
      }
    }
  }
  for (const VisionEmbeddingSegment& segment : vision_segments) {
    if (segment.patches.empty() || segment.patches.size() % 6912U != 0U) {
      return Error(StatusCode::kInvalidArgument,
                   "vision segment requires complete 48x48 RGB patches");
    }
    const std::uint64_t patch_count = segment.patches.size() / 6912U;
    if (patch_count == 0U || patch_count > 280U ||
        segment.positions.size() != patch_count * 2U ||
        segment.prompt_offset >= full_prompt_token_ids.size() ||
        patch_count > full_prompt_token_ids.size() - segment.prompt_offset) {
      return Error(StatusCode::kInvalidArgument,
                   "vision segment does not fit in the rendered prompt");
    }
    for (std::uint64_t patch = 0U; patch < patch_count; ++patch) {
      if (full_prompt_token_ids[segment.prompt_offset + patch] != 258880U ||
          segment.positions[patch * 2U] < 0 ||
          segment.positions[patch * 2U + 1U] < 0 ||
          segment.positions[patch * 2U] >= 1120 ||
          segment.positions[patch * 2U + 1U] >= 1120) {
        return Error(StatusCode::kInvalidArgument,
                     "vision segment token or position alignment is invalid");
      }
    }
  }
  if (moe26b) {
    const std::size_t comparable_tokens = std::min(
        impl_->cached_token_ids.size(), full_prompt_token_ids.size());
    const auto mismatch = std::mismatch(
        impl_->cached_token_ids.begin(),
        impl_->cached_token_ids.begin() + comparable_tokens,
        full_prompt_token_ids.begin());
    if (impl_->cached_token_ids.size() > full_prompt_token_ids.size() ||
        mismatch.first != impl_->cached_token_ids.begin() + comparable_tokens) {
      return Error(
          StatusCode::kInvalidArgument,
          "rendered conversation differs from the resident Gemma 4 26B KV-cache prefix");
    }
    const std::size_t prefix_tokens = impl_->cached_token_ids.size();
    const std::span<const std::uint32_t> suffix =
        full_prompt_token_ids.subspan(prefix_tokens);
    if (suffix.empty()) {
      return Error(StatusCode::kInvalidArgument,
                   "conversation turn adds no uncached prompt tokens");
    }

    GreedyInferenceResult result;
    result.output_token_ids.reserve(
        static_cast<std::size_t>(max_generated_tokens));
    result.kv_cache_mode = KvCacheMode::kCheckpointFp8;
    result.sampling = impl_->sampling;
    result.model_load_milliseconds = impl_->model_load_milliseconds;
    result.weight_arena_bytes =
        impl_->runtime->impl_->moe26b_engine->weight_arena_bytes();
    result.reasoning_enabled = reasoning.enabled;
    result.reasoning_budget_tokens = reasoning.max_reasoning_tokens;
    result.prompt_cached_tokens = prefix_tokens;
    result.prompt_cache_write_tokens = suffix.size();
    result.kv_cache_bytes =
        impl_->runtime->impl_->moe26b_engine->kv_cache_bytes();
    result.workspace_bytes =
        impl_->runtime->impl_->moe26b_engine->workspace_bytes();
    result.max_context_tokens = impl_->max_context_tokens;
    result.packed_weight_source_layout_direct = true;
    result.token_loop_allocations = false;
    result.decode_graphs = true;

    const auto prompt_start = std::chrono::steady_clock::now();
    Status status =
        impl_->runtime->impl_->moe26b_engine->PrefillTokens(suffix);
    if (!status.ok()) {
      impl_->poisoned = true;
      return status;
    }
    auto selected = impl_->runtime->impl_->moe26b_engine->SelectToken();
    if (!selected.ok()) {
      impl_->poisoned = true;
      return selected.status();
    }
    result.prompt_milliseconds = Milliseconds(
        std::chrono::steady_clock::now() - prompt_start);
    impl_->cached_token_ids.insert(impl_->cached_token_ids.end(),
                                   suffix.begin(), suffix.end());

    ResponseChannelTracker reasoning_tracker(
        reasoning.channel_open_token_ids, reasoning.channel_close_token_id,
        reasoning.starts_in_reasoning);
    bool reasoning_complete = !reasoning.enabled;
    const auto observe_reasoning_token = [&](std::uint32_t token) {
      if (!reasoning.enabled || reasoning_complete) return;
      const bool was_reasoning = reasoning_tracker.in_reasoning();
      (void)reasoning_tracker.Observe(token);
      result.reasoning_tokens = reasoning_tracker.reasoning_token_count();
      if (was_reasoning && !reasoning_tracker.in_reasoning()) {
        reasoning_complete = true;
      }
    };
    std::uint32_t next_token = selected.value();
    result.output_token_ids.push_back(next_token);
    observe_reasoning_token(next_token);
    if (generated_token_callback != nullptr) {
      status = generated_token_callback(generated_token_callback_context,
                                        next_token);
      if (!status.ok()) {
        impl_->poisoned = true;
        return status;
      }
    }
    if (std::find(impl_->stop_token_ids.begin(), impl_->stop_token_ids.end(),
                  next_token) != impl_->stop_token_ids.end()) {
      result.stopped = true;
      result.stop_token_id = next_token;
    }

    const auto decode_start = std::chrono::steady_clock::now();
    for (std::uint64_t generated = 1U;
         generated < max_generated_tokens && !result.stopped; ++generated) {
      const std::uint32_t input_token = next_token;
      const bool force_reasoning_close =
          !reasoning_complete && reasoning_tracker.in_reasoning() &&
          reasoning_tracker.reasoning_token_count() >=
              reasoning.max_reasoning_tokens;
      status = impl_->runtime->impl_->moe26b_engine->ForwardToken(input_token);
      if (!status.ok()) {
        impl_->poisoned = true;
        return status;
      }
      impl_->cached_token_ids.push_back(input_token);
      selected = impl_->runtime->impl_->moe26b_engine->SelectToken();
      if (!selected.ok()) {
        impl_->poisoned = true;
        return selected.status();
      }
      next_token = force_reasoning_close ? reasoning.channel_close_token_id
                                         : selected.value();
      result.reasoning_budget_forced =
          result.reasoning_budget_forced || force_reasoning_close;
      result.output_token_ids.push_back(next_token);
      observe_reasoning_token(next_token);
      if (generated_token_callback != nullptr) {
        status = generated_token_callback(generated_token_callback_context,
                                          next_token);
        if (!status.ok()) {
          impl_->poisoned = true;
          return status;
        }
      }
      if (std::find(impl_->stop_token_ids.begin(),
                    impl_->stop_token_ids.end(), next_token) !=
          impl_->stop_token_ids.end()) {
        result.stopped = true;
        result.stop_token_id = next_token;
      }
    }
    result.decode_milliseconds = Milliseconds(
        std::chrono::steady_clock::now() - decode_start);
    const std::uint64_t measured_decode_tokens =
        result.output_token_ids.size() - 1U;
    if (measured_decode_tokens != 0U && result.decode_milliseconds > 0.0) {
      result.decode_tokens_per_second =
          static_cast<double>(measured_decode_tokens) * 1000.0 /
          result.decode_milliseconds;
    }
    return result;
  }
  const std::size_t comparable_tokens = std::min(
      impl_->cached_token_ids.size(), full_prompt_token_ids.size());
  const auto mismatch = std::mismatch(
      impl_->cached_token_ids.begin(),
      impl_->cached_token_ids.begin() + comparable_tokens,
      full_prompt_token_ids.begin());
  if (impl_->cached_token_ids.size() > full_prompt_token_ids.size() ||
      mismatch.first !=
          impl_->cached_token_ids.begin() + comparable_tokens) {
    const std::size_t mismatch_index = static_cast<std::size_t>(
        mismatch.first - impl_->cached_token_ids.begin());
    const std::string cached_id =
        mismatch_index < impl_->cached_token_ids.size()
            ? std::to_string(impl_->cached_token_ids[mismatch_index])
            : "end";
    const std::string rendered_id =
        mismatch_index < full_prompt_token_ids.size()
            ? std::to_string(full_prompt_token_ids[mismatch_index])
            : "end";
    return Error(
        StatusCode::kInvalidArgument,
        "rendered conversation differs from the resident KV-cache prefix at token " +
            std::to_string(mismatch_index) + " (cached " + cached_id +
            ", rendered " + rendered_id + ")");
  }
  const std::size_t prefix_tokens = impl_->cached_token_ids.size();
  const std::span<const std::uint32_t> suffix =
      full_prompt_token_ids.subspan(prefix_tokens);
  if (suffix.empty()) {
    return Error(StatusCode::kInvalidArgument,
                 "conversation turn adds no uncached prompt tokens");
  }

  GreedyInferenceResult result;
  result.output_token_ids.reserve(
      static_cast<std::size_t>(max_generated_tokens));
  if (impl_->mtp_draft_tokens != 0U) {
    result.mtp_proposed_token_ids.reserve(static_cast<std::size_t>(
        max_generated_tokens * impl_->mtp_draft_tokens));
  }
  result.kv_cache_mode = impl_->kv_cache_mode;
  result.sampling = impl_->sampling;
  result.decode_graphs = true;
  result.model_load_milliseconds = impl_->model_load_milliseconds;
  result.weight_arena_bytes = impl_->engine.weight_bytes();
  result.assistant_loaded = impl_->engine.assistant_loaded();
  result.assistant_source_bytes = impl_->engine.assistant_source_bytes();
  result.assistant_weight_arena_bytes = impl_->engine.assistant_weight_bytes();
  result.assistant_device_memory_delta_bytes =
      impl_->engine.assistant_device_memory_delta_bytes();
  result.assistant_tensor_count = impl_->engine.assistant_tensor_count();
  result.assistant_workspace_bytes = impl_->engine.assistant_workspace_bytes();
  result.mtp_enabled = impl_->mtp_draft_tokens != 0U;
  result.mtp_adaptive = impl_->mtp_adaptive;
  result.mtp_draft_tokens = impl_->mtp_draft_tokens;
  result.reasoning_enabled = reasoning.enabled;
  result.reasoning_budget_tokens = reasoning.max_reasoning_tokens;
  result.prompt_cached_tokens = prefix_tokens;
  result.prompt_cache_write_tokens = suffix.size();
  result.kv_cache_bytes = impl_->engine.cache_bytes();
  result.workspace_bytes = impl_->engine.workspace_bytes();
  result.decode_graph_device_bytes =
      impl_->engine.decode_graph_device_bytes();
  result.prefill_chunk_tokens = impl_->engine.prefill_chunk_tokens();
  result.max_context_tokens = impl_->max_context_tokens;
  result.packed_weight_source_layout_direct = false;
  result.token_loop_allocations = false;
  result.benchmark_qualified = false;

  const auto prompt_start = std::chrono::steady_clock::now();
  auto prefilled = impl_->engine.PrefillAt(suffix, prefix_tokens, {},
                                           audio_segments, vision_segments);
  if (!prefilled.ok()) {
    impl_->poisoned = true;
    return prefilled.status();
  }
  impl_->cached_token_ids.insert(impl_->cached_token_ids.end(),
                                 suffix.begin(), suffix.end());
  std::uint32_t next_token = prefilled.value();
  result.prompt_milliseconds = Milliseconds(
      std::chrono::steady_clock::now() - prompt_start);
  ResponseChannelTracker reasoning_tracker(
      reasoning.channel_open_token_ids, reasoning.channel_close_token_id,
      reasoning.starts_in_reasoning);
  bool reasoning_started = reasoning.starts_in_reasoning;
  bool reasoning_complete = !reasoning.enabled;
  const auto observe_reasoning_token = [&](std::uint32_t token) {
    if (!reasoning.enabled || reasoning_complete) return;
    const bool was_reasoning = reasoning_tracker.in_reasoning();
    (void)reasoning_tracker.Observe(token);
    if (!was_reasoning && reasoning_tracker.in_reasoning()) {
      reasoning_started = true;
    }
    result.reasoning_tokens = reasoning_tracker.reasoning_token_count();
    if (was_reasoning && !reasoning_tracker.in_reasoning()) {
      reasoning_complete = true;
    }
  };

  result.output_token_ids.push_back(next_token);
  observe_reasoning_token(next_token);
  if (generated_token_callback != nullptr) {
    Status status = generated_token_callback(
        generated_token_callback_context, next_token);
    if (!status.ok()) {
      impl_->poisoned = true;
      return status;
    }
  }
  if (std::find(impl_->stop_token_ids.begin(), impl_->stop_token_ids.end(),
                next_token) != impl_->stop_token_ids.end()) {
    result.stopped = true;
    result.stop_token_id = next_token;
  }

  if (result.mtp_enabled && impl_->mtp_draft_tokens == 2U &&
      !impl_->mtp_adaptive &&
      impl_->kv_cache_mode == KvCacheMode::kCheckpointFp8 &&
      impl_->max_context_tokens > kSlidingWindow) {
    Status status = impl_->engine.PrepareFixedD2Graph();
    if (!status.ok()) {
      impl_->poisoned = true;
      return status;
    }
  }

  const auto decode_start = std::chrono::steady_clock::now();
  if (result.mtp_enabled) {
    std::uint64_t processed_position = impl_->cached_token_ids.size() - 1U;
    const bool reasoning_gpu_chain =
        impl_->mtp_draft_tokens == 2U && !impl_->mtp_adaptive &&
        impl_->kv_cache_mode == KvCacheMode::kCheckpointFp8 &&
        impl_->max_context_tokens > kSlidingWindow;
    while (!reasoning_gpu_chain &&
           result.output_token_ids.size() < max_generated_tokens &&
           !result.stopped && !reasoning_complete) {
      const bool force_close = reasoning_tracker.in_reasoning() &&
                               reasoning_tracker.reasoning_token_count() >=
                                   reasoning.max_reasoning_tokens;
      auto forwarded = impl_->engine.Forward(
          next_token, processed_position + 1U, true);
      if (!forwarded.ok()) {
        impl_->poisoned = true;
        return forwarded.status();
      }
      ++processed_position;
      ++result.mtp_target_forwards;
      ++result.mtp_target_batches;
      ++result.reasoning_ordinary_target_tokens;
      next_token = force_close ? reasoning.channel_close_token_id
                               : forwarded.value();
      result.reasoning_budget_forced =
          result.reasoning_budget_forced || force_close;
      result.output_token_ids.push_back(next_token);
      observe_reasoning_token(next_token);
      if (generated_token_callback != nullptr) {
        Status status = generated_token_callback(
            generated_token_callback_context, next_token);
        if (!status.ok()) {
          impl_->poisoned = true;
          return status;
        }
      }
      if (std::find(impl_->stop_token_ids.begin(),
                    impl_->stop_token_ids.end(), next_token) !=
          impl_->stop_token_ids.end()) {
        result.stopped = true;
        result.stop_token_id = next_token;
      }
    }
    internal::AdaptiveMtpScheduler adaptive_scheduler(
        impl_->mtp_draft_tokens, processed_position, impl_->mtp_adaptive);
    const auto emit_ordinary_token = [&](bool adaptive_fallback) -> Status {
      auto forwarded = impl_->engine.Forward(next_token,
                                             processed_position + 1U, true);
      if (!forwarded.ok()) return forwarded.status();
      ++processed_position;
      ++result.mtp_target_forwards;
      ++result.mtp_target_batches;
      if (adaptive_fallback) ++result.mtp_ordinary_fallback_tokens;
      next_token = forwarded.value();
      result.output_token_ids.push_back(next_token);
      if (generated_token_callback != nullptr) {
        Status callback_status = generated_token_callback(
            generated_token_callback_context, next_token);
        if (!callback_status.ok()) return callback_status;
      }
      if (std::find(impl_->stop_token_ids.begin(),
                    impl_->stop_token_ids.end(), next_token) !=
          impl_->stop_token_ids.end()) {
        result.stopped = true;
        result.stop_token_id = next_token;
      }
      return Status::Ok();
    };
    while (result.output_token_ids.size() < max_generated_tokens &&
           !result.stopped) {
      const std::size_t remaining = static_cast<std::size_t>(
          max_generated_tokens - result.output_token_ids.size());
      const bool fixed_d2_chain =
          remaining >= 3U && impl_->mtp_draft_tokens == 2U &&
          !impl_->mtp_adaptive &&
          impl_->kv_cache_mode == KvCacheMode::kCheckpointFp8 &&
          impl_->max_context_tokens > kSlidingWindow;
      if (fixed_d2_chain) {
        internal::MtpReasoningState device_reasoning{};
        device_reasoning.enabled = reasoning.enabled ? 1U : 0U;
        device_reasoning.started = reasoning_started ? 1U : 0U;
        device_reasoning.complete = reasoning_complete ? 1U : 0U;
        device_reasoning.in_reasoning =
            reasoning_tracker.in_reasoning() ? 1U : 0U;
        device_reasoning.open_match_length = static_cast<std::uint32_t>(
            reasoning_tracker.open_match_length());
        device_reasoning.reasoning_token_count =
            reasoning_tracker.reasoning_token_count();
        device_reasoning.max_reasoning_tokens =
            reasoning.max_reasoning_tokens;
        device_reasoning.close_token_id = reasoning.channel_close_token_id;
        device_reasoning.open_token_count = static_cast<std::uint32_t>(
            reasoning.channel_open_token_ids.size());
        std::copy(reasoning.channel_open_token_ids.begin(),
                  reasoning.channel_open_token_ids.end(),
                  device_reasoning.open_token_ids.begin());
        Status status = impl_->engine.PrepareMtpDeviceControl(
            next_token, processed_position, remaining,
            result.output_token_ids.size(), result.stopped,
            result.stop_token_id, device_reasoning);
        if (!status.ok()) {
          impl_->poisoned = true;
          return status;
        }
        internal::MtpChainResult chain;
        status = impl_->engine.ExecuteFixedD2GraphChain(
            &chain, generated_token_callback,
            generated_token_callback_context);
        if (!status.ok()) {
          impl_->poisoned = true;
          return status;
        }
        result.mtp_fixed_d2_graph = true;
        result.mtp_gpu_chained = true;
        result.mtp_verification_groups += chain.group_count;
        result.mtp_d2_groups += chain.group_count;
        result.mtp_proposed_tokens += chain.proposed_count;
        result.mtp_target_forwards += 3U * chain.group_count;
        result.mtp_target_batches += chain.group_count;
        result.mtp_target_forwards += chain.ordinary_tail_count;
        result.mtp_target_batches += chain.ordinary_tail_count;
        result.reasoning_ordinary_target_tokens +=
            chain.reasoning_ordinary_count;
        result.mtp_accepted_tokens += chain.accepted_count;
        result.mtp_rejected_tokens += chain.rejected_count;
        result.mtp_proposed_token_ids.insert(
            result.mtp_proposed_token_ids.end(),
            impl_->engine.mtp_chain_proposals(),
            impl_->engine.mtp_chain_proposals() + chain.proposed_count);
        processed_position += chain.output_count;
        const std::uint32_t* chained_outputs =
            impl_->engine.mtp_chain_outputs();
        for (std::uint64_t index = 0U; index < chain.output_count; ++index) {
          next_token = chained_outputs[index];
          result.output_token_ids.push_back(next_token);
          observe_reasoning_token(next_token);
        }
        result.reasoning_budget_forced = result.reasoning_budget_forced ||
                                         chain.reasoning_budget_forced != 0U;
        if (result.reasoning_tokens != chain.reasoning_token_count ||
            reasoning_complete != (chain.reasoning_complete != 0U)) {
          impl_->poisoned = true;
          return Error(StatusCode::kInternal,
                       "host reasoning state disagrees with GPU MTP chain");
        }
        result.stopped = chain.stopped != 0U;
        result.stop_token_id = chain.stop_token;
        status = impl_->engine.CheckMtpDeviceControlParity(
            next_token, processed_position,
            max_generated_tokens - result.output_token_ids.size(),
            result.output_token_ids.size(), result.stopped,
            result.stop_token_id);
        if (!status.ok()) {
          impl_->poisoned = true;
          return status;
        }
        continue;
      }
      const bool adaptive_fallback =
          adaptive_scheduler.use_ordinary_fallback();
      if (remaining == 1U || adaptive_fallback) {
        Status status = emit_ordinary_token(adaptive_fallback);
        if (!status.ok()) {
          impl_->poisoned = true;
          return status;
        }
        if (adaptive_fallback) adaptive_scheduler.ConsumeOrdinaryFallback();
        continue;
      }
      const std::size_t proposal_count = std::min<std::size_t>(
          adaptive_scheduler.active_drafts(), remaining - 1U);
      Status status = impl_->engine.PrepareMtpDeviceControl(
          next_token, processed_position, remaining,
          result.output_token_ids.size(), result.stopped,
          result.stop_token_id);
      if (!status.ok()) {
        impl_->poisoned = true;
        return status;
      }
      MtpGroupResult group;
      if (proposal_count == 2U && impl_->mtp_draft_tokens == 2U &&
          !impl_->mtp_adaptive) {
        result.mtp_fixed_d2_graph =
            impl_->kv_cache_mode == KvCacheMode::kCheckpointFp8 &&
            impl_->max_context_tokens > kSlidingWindow;
        status = impl_->engine.ExecuteFixedD2GraphGroup(
            next_token, processed_position + 1U, &group);
      } else {
        status = impl_->engine.GenerateAssistantDraftsDevice(
            next_token, processed_position,
            static_cast<std::uint32_t>(proposal_count));
        if (status.ok()) {
          status = impl_->engine.VerifyAcceptCommitAssistantBatch(
              next_token, processed_position + 1U,
              static_cast<std::uint32_t>(proposal_count), &group);
        }
      }
      if (!status.ok()) {
        impl_->poisoned = true;
        return status;
      }
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
          group.proposed.begin() +
              static_cast<std::ptrdiff_t>(proposal_count));
      result.mtp_target_forwards += proposal_count + 1U;
      ++result.mtp_target_batches;
      result.mtp_accepted_tokens += group.accepted_count;
      result.mtp_rejected_tokens += proposal_count - group.accepted_count;
      processed_position += group.output_count;
      for (std::uint32_t index = 0U; index < group.output_count; ++index) {
        next_token = group.verified[index];
        result.output_token_ids.push_back(next_token);
        if (generated_token_callback != nullptr) {
          status = generated_token_callback(generated_token_callback_context,
                                            next_token);
          if (!status.ok()) {
            impl_->poisoned = true;
            return status;
          }
        }
        if (std::find(impl_->stop_token_ids.begin(),
                      impl_->stop_token_ids.end(), next_token) !=
            impl_->stop_token_ids.end()) {
          result.stopped = true;
          result.stop_token_id = next_token;
          break;
        }
      }
      status = impl_->engine.CheckMtpDeviceControlParity(
          next_token, processed_position,
          max_generated_tokens - result.output_token_ids.size(),
          result.output_token_ids.size(), result.stopped,
          result.stop_token_id);
      if (!status.ok()) {
        impl_->poisoned = true;
        return status;
      }
      if (!result.stopped) {
        adaptive_scheduler.Observe(processed_position,
                                   group.accepted_count);
      }
    }
    if (result.output_token_ids.size() > 1U) {
      impl_->cached_token_ids.insert(
          impl_->cached_token_ids.end(), result.output_token_ids.begin(),
          result.output_token_ids.end() - 1);
    }
  } else for (std::uint64_t generated = 1U;
       generated < max_generated_tokens && !result.stopped; ++generated) {
    const std::uint64_t position = impl_->cached_token_ids.size();
    const std::uint32_t input_token = next_token;
    const bool force_reasoning_close =
        !reasoning_complete && reasoning_tracker.in_reasoning() &&
        reasoning_tracker.reasoning_token_count() >=
            reasoning.max_reasoning_tokens;
    auto forwarded =
        impl_->engine.Forward(input_token, position, true);
    if (!forwarded.ok()) {
      impl_->poisoned = true;
      return forwarded.status();
    }
    impl_->cached_token_ids.push_back(input_token);
    next_token = force_reasoning_close ? reasoning.channel_close_token_id
                                       : forwarded.value();
    result.reasoning_budget_forced =
        result.reasoning_budget_forced || force_reasoning_close;
    result.output_token_ids.push_back(next_token);
    observe_reasoning_token(next_token);
    if (generated_token_callback != nullptr) {
      Status status = generated_token_callback(
          generated_token_callback_context, next_token);
      if (!status.ok()) {
        impl_->poisoned = true;
        return status;
      }
    }
    if (std::find(impl_->stop_token_ids.begin(),
                  impl_->stop_token_ids.end(), next_token) !=
        impl_->stop_token_ids.end()) {
      result.stopped = true;
      result.stop_token_id = next_token;
    }
  }
  result.decode_milliseconds = Milliseconds(
      std::chrono::steady_clock::now() - decode_start);
  const std::uint64_t measured_decode_tokens =
      result.output_token_ids.size() - 1U;
  if (measured_decode_tokens != 0U && result.decode_milliseconds > 0.0) {
    result.decode_tokens_per_second =
        static_cast<double>(measured_decode_tokens) * 1000.0 /
        result.decode_milliseconds;
  }
  return result;
}

std::uint64_t ConversationSession::cached_token_count() const {
  return impl_ == nullptr ? 0U : impl_->cached_token_ids.size();
}

std::uint64_t ConversationSession::reserved_device_bytes() const {
  if (impl_ == nullptr) return 0U;
  if (impl_->moe26b_slot_lease && impl_->runtime != nullptr &&
      impl_->runtime->impl_->moe26b_engine != nullptr) {
    return impl_->runtime->impl_->moe26b_engine->kv_cache_bytes() +
           impl_->runtime->impl_->moe26b_engine->workspace_bytes();
  }
  return impl_->engine.cache_bytes() + impl_->engine.workspace_bytes() +
         impl_->engine.assistant_workspace_bytes() +
         impl_->engine.decode_graph_device_bytes();
}

bool ConversationSession::is_poisoned() const {
  return impl_ == nullptr || impl_->poisoned;
}

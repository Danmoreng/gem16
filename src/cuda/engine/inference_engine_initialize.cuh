  [[nodiscard]] Status Initialize(
      const std::filesystem::path& model_directory,
      std::uint64_t max_context, KvCacheMode kv_cache_mode,
      const SamplingOptions& sampling = {},
      const std::filesystem::path& assistant_model_directory = {},
      std::uint32_t mtp_draft_tokens = 0U) {
    const NvtxRange range("gem16.initialize");
    Status sampling_status = SetSampling(sampling);
    if (!sampling_status.ok()) return sampling_status;
    kv_cache_mode_ = kv_cache_mode;
    max_context_ = max_context;
    const auto chunk =
        ConfiguredPrefillChunkTokensForContext(max_context_, kv_cache_mode_);
    if (!chunk.ok()) return chunk.status();
    prefill_chunk_tokens_ = chunk.value();
    cudaDeviceProp properties{};
    cudaError_t error = cudaGetDeviceProperties(&properties, 0);
    if (error != cudaSuccess) return CudaFailure("cudaGetDeviceProperties", error);
    if (properties.major != 12 || properties.minor != 0) {
      return Error(StatusCode::kUnsupported, "greedy characterization requires SM120");
    }
    error = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (error != cudaSuccess) return CudaFailure("create inference stream", error);

    Status status = model_.Load(model_directory);
    if (!status.ok()) return status;
    if (!assistant_model_directory.empty()) {
      const NvtxRange assistant_range("gem16.initialize.mtp_assistant");
      std::size_t free_before_assistant = 0U;
      std::size_t total_before_assistant = 0U;
      error = cudaMemGetInfo(&free_before_assistant, &total_before_assistant);
      if (error != cudaSuccess) {
        return CudaFailure("measure memory before assistant load", error);
      }
      status = assistant_.Load(assistant_model_directory);
      if (!status.ok()) return status;
      std::size_t free_after_assistant = 0U;
      std::size_t total_after_assistant = 0U;
      error = cudaMemGetInfo(&free_after_assistant, &total_after_assistant);
      if (error != cudaSuccess) {
        return CudaFailure("measure memory after assistant load", error);
      }
      if (total_before_assistant != total_after_assistant) {
        return Error(StatusCode::kInternal,
                     "device total memory changed during assistant load");
      }
      assistant_device_memory_delta_bytes_ =
          free_before_assistant > free_after_assistant
              ? free_before_assistant - free_after_assistant
              : 0U;
      if (mtp_draft_tokens != 0U) {
        status = assistant_.Prepare(max_context_);
        if (!status.ok()) return status;
      }
    }
    return InitializeSlot(mtp_draft_tokens);
  }

  [[nodiscard]] Status InitializeShared(
      const internal::LoadedTargetModel& model,
      const internal::AssistantModel* assistant,
      std::uint64_t max_context, KvCacheMode kv_cache_mode,
      const SamplingOptions& sampling,
      std::uint32_t mtp_draft_tokens) {
    const NvtxRange range("gem16.initialize_shared");
    Status status = SetSampling(sampling);
    if (!status.ok()) return status;
    kv_cache_mode_ = kv_cache_mode;
    max_context_ = max_context;
    const auto chunk =
        ConfiguredPrefillChunkTokensForContext(max_context_, kv_cache_mode_);
    if (!chunk.ok()) return chunk.status();
    prefill_chunk_tokens_ = chunk.value();
    cudaDeviceProp properties{};
    cudaError_t error = cudaGetDeviceProperties(&properties, 0);
    if (error != cudaSuccess) {
      return CudaFailure("cudaGetDeviceProperties", error);
    }
    if (properties.major != 12 || properties.minor != 0) {
      return Error(StatusCode::kUnsupported,
                   "greedy characterization requires SM120");
    }
    error = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (error != cudaSuccess) {
      return CudaFailure("create inference stream", error);
    }
    status = model_.ShareWeightsFrom(model);
    if (!status.ok()) return status;
    if (assistant != nullptr) {
      status = assistant_.ShareWeightsFrom(*assistant);
      if (!status.ok()) return status;
      if (mtp_draft_tokens != 0U) {
        status = assistant_.Prepare(max_context_);
        if (!status.ok()) return status;
      }
    } else if (mtp_draft_tokens != 0U) {
      return Error(StatusCode::kInvalidArgument,
                   "active MTP requires shared assistant weights");
    }
    assistant_device_memory_delta_bytes_ = 0U;
    return InitializeSlot(mtp_draft_tokens);
  }

  [[nodiscard]] Status InitializeSlot(std::uint32_t mtp_draft_tokens) {
    mtp_draft_tokens_ = mtp_draft_tokens;
    const auto math = internal::SelectTwelveBD3Math(
        std::getenv("GEM16_12B_D3_ATTENTION"), mtp_draft_tokens,
        kv_cache_mode_, max_context_, sampling_.enabled);
    if (!math.ok()) return math.status();
    mtp_math_ = math.value();
    fast_d3_attention_ = mtp_math_.optimized;
    if (fast_d3_attention_) {
      const auto prepare_fast = fast_d3_global_.Initialize();
      if (!prepare_fast.ok()) return prepare_fast;
      const auto prepare_local = fast_d3_local_.Initialize();
      if (!prepare_local.ok()) return prepare_local;
    }
    // Ablation controls are read once, before any graph capture. Never inspect
    // environment or create streams/events in recurring inference.
    const auto flag = [](const char* name, bool* value) -> Status {
      const char* text = std::getenv(name);
      if (text == nullptr || std::strcmp(text, "1") == 0) { *value = true; return Status::Ok(); }
      if (std::strcmp(text, "0") == 0) { *value = false; return Status::Ok(); }
      return Error(StatusCode::kInvalidArgument, std::string(name) + " requires0 or1");
    };
    Status status = flag("GEM16_12B_ORDINARY_NORMS", &ordinary_norms_);
    if (!status.ok()) return status;
    status = flag("GEM16_12B_ORDINARY_OVERLAP", &ordinary_overlap_);
    if (!status.ok()) return status;
    status = flag("GEM16_12B_ORDINARY_O", &ordinary_o_);
    if (!status.ok()) return status;
    status = flag("GEM16_12B_ORDINARY_GLOBAL", &ordinary_global_);
    if (!status.ok()) return status;
    ordinary_norms_ = ordinary_norms_ &&
                      kv_cache_mode_ == KvCacheMode::kCheckpointFp8;
    ordinary_overlap_ = ordinary_overlap_ &&
                        kv_cache_mode_ == KvCacheMode::kCheckpointFp8;
    ordinary_o_ = ordinary_o_ &&
                  kv_cache_mode_ == KvCacheMode::kCheckpointFp8;
    ordinary_global_ = ordinary_global_ &&
                       kv_cache_mode_ == KvCacheMode::kCheckpointFp8 &&
                       max_context_ >= 16384U && max_context_ <= 17519U;
    if (ordinary_overlap_) {
      status = ordinary_fork_.Initialize();
      if (!status.ok()) return status;
    }
    cudaError_t error = cudaSuccess;
    status = AllocateCache();
    if (!status.ok()) return status;
    status = AllocateWorkspace();
    if (!status.ok()) return status;
    status = AllocatePrefillWorkspace();
    if (!status.ok()) return status;
    if (mtp_draft_tokens_ != 0U) {
      status = AllocateMtpWorkspace();
      if (!status.ok()) return status;
      constexpr std::size_t kHostTransactionFloats =
          (sizeof(internal::MtpGroupTransaction) + sizeof(float) - 1U) /
          sizeof(float);
      status = mtp_host_result_.Allocate(kHostTransactionFloats,
                                         "MTP group host transaction");
      if (!status.ok()) return status;
      const std::size_t chain_host_bytes =
          sizeof(internal::MtpChainResult) +
          static_cast<std::size_t>(
              (std::max(2U, mtp_draft_tokens_) + 1U) * max_context_) *
              sizeof(std::uint32_t);
      status = mtp_host_chain_.Allocate(
          (chain_host_bytes + sizeof(float) - 1U) / sizeof(float),
          "MTP chain host result");
      if (!status.ok()) return status;
      constexpr std::size_t kStreamingRingFloats =
          (sizeof(internal::MtpStreamingRing) + sizeof(float) - 1U) /
          sizeof(float);
      status = mtp_stream_ring_.Allocate(
          kStreamingRingFloats, "MTP mapped streaming ring",
          cudaHostAllocMapped);
      if (!status.ok()) return status;
    }
    status = internal::LaunchRotaryEmbeddingTableBatch(
        Pointer<float>(prefill_workspace_, prefill_offsets_.local_rope_cosine),
        Pointer<float>(prefill_workspace_, prefill_offsets_.local_rope_sine),
        max_context_, 128U, 256U, 0U, 10000.0, 1.0, stream_);
    if (!status.ok()) return status;
    status = internal::LaunchRotaryEmbeddingTableBatch(
        Pointer<float>(prefill_workspace_, prefill_offsets_.global_rope_cosine),
        Pointer<float>(prefill_workspace_, prefill_offsets_.global_rope_sine),
        max_context_, 64U, 512U, 0U, 1000000.0, 1.0, stream_);
    if (!status.ok()) return status;
    error = cudaStreamSynchronize(stream_);
    if (error != cudaSuccess) {
      return CudaFailure("prepare persistent prefill RoPE tables", error);
    }
    status = decode_host_state_.Allocate(
        (sizeof(HostDecodeState) + sizeof(float) - 1U) / sizeof(float),
        "allocate decode graph host control");
    if (!status.ok()) return status;
    std::size_t free_before = 0U;
    std::size_t total_before = 0U;
    cudaError_t memory_error = cudaMemGetInfo(&free_before, &total_before);
    if (memory_error != cudaSuccess) {
      return CudaFailure("measure memory before decode graph capture",
                         memory_error);
    }
    status = PrepareDecodeGraphs();
    if (!status.ok()) return status;
    std::size_t free_after = 0U;
    std::size_t total_after = 0U;
    memory_error = cudaMemGetInfo(&free_after, &total_after);
    if (memory_error != cudaSuccess) {
      return CudaFailure("measure memory after decode graph capture",
                         memory_error);
    }
    if (total_before != total_after) {
      return Error(StatusCode::kInternal,
                   "device total memory changed during decode graph capture");
    }
    decode_graph_device_bytes_ =
        free_before > free_after ? free_before - free_after : 0U;
    return ResetCache();
  }

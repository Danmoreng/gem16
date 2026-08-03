  [[nodiscard]] Status ResetCache() {
    cudaError_t error = cudaMemsetAsync(
        cache_.data(), 0, static_cast<std::size_t>(cache_.bytes()), stream_);
    if (error != cudaSuccess) return CudaFailure("clear KV cache", error);
    if (sampling_.enabled) {
      error = cudaMemsetAsync(
          Pointer<std::uint32_t>(workspace_, offsets_.repetition_mask), 0,
          static_cast<std::size_t>(kRepetitionMaskWords * sizeof(std::uint32_t)),
          stream_);
      if (error != cudaSuccess) {
        return CudaFailure("clear repetition mask", error);
      }
    }
    sampling_step_ = 0U;
    error = cudaStreamSynchronize(stream_);
    return error == cudaSuccess ? Status::Ok() : CudaFailure("reset KV cache", error);
  }

  [[nodiscard]] Status SetSampling(const SamplingOptions& options) {
    Status status = ValidateSamplingOptions(
        options, static_cast<std::uint32_t>(kVocabulary));
    if (!status.ok()) return status;
    sampling_ = options;
    sampling_step_ = 0U;
    return Status::Ok();
  }

  [[nodiscard]] Status SetSuppressedTokens(std::span<const std::uint32_t> tokens) {
    if (tokens.size() > kMaximumSuppressedTokens) {
      return Error(StatusCode::kUnsupported,
                   "the initial greedy path supports at most 16 suppressed tokens");
    }
    suppressed_token_count_ = static_cast<std::uint32_t>(tokens.size());
    host_decode_state()->control.suppressed_token_count = suppressed_token_count_;
    if (tokens.empty()) return Status::Ok();
    const cudaError_t error = cudaMemcpyAsync(
        Pointer<std::uint32_t>(workspace_, offsets_.suppressed), tokens.data(),
        tokens.size_bytes(), cudaMemcpyHostToDevice, stream_);
    if (error != cudaSuccess) return CudaFailure("copy suppressed token IDs", error);
    const cudaError_t sync_error = cudaStreamSynchronize(stream_);
    return sync_error == cudaSuccess ? Status::Ok()
                                    : CudaFailure("configure suppressed token IDs", sync_error);
  }

  [[nodiscard]] Status SetMtpStopTokens(
      std::span<const std::uint32_t> tokens) {
    if (mtp_draft_tokens_ == 0U) return Status::Ok();
    if (tokens.size() > kMaximumSuppressedTokens) {
      return Error(StatusCode::kUnsupported,
                   "active MTP supports at most 16 stop-token IDs");
    }
    mtp_stop_token_count_ = static_cast<std::uint32_t>(tokens.size());
    if (tokens.empty()) return Status::Ok();
    const cudaError_t error = cudaMemcpyAsync(
        Pointer<std::uint32_t>(mtp_workspace_, mtp_offsets_.stop_tokens),
        tokens.data(), tokens.size_bytes(), cudaMemcpyHostToDevice, stream_);
    if (error != cudaSuccess) return CudaFailure("copy MTP stop-token IDs", error);
    const cudaError_t sync_error = cudaStreamSynchronize(stream_);
    return sync_error == cudaSuccess
               ? Status::Ok()
               : CudaFailure("configure MTP stop-token IDs", sync_error);
  }

 private:
  [[nodiscard]] Status MarkRepetitionToken(std::uint32_t token) {
    if (!sampling_.enabled || sampling_.repetition_penalty == 1.0F) {
      return Status::Ok();
    }
    return internal::LaunchMarkRepetitionToken(
        token, Pointer<std::uint32_t>(workspace_, offsets_.repetition_mask),
        stream_);
  }

  [[nodiscard]] Status SelectSampledTokenWithState(
      float* logits, std::uint32_t* selected,
      std::uint32_t* repetition_mask, std::uint64_t step,
      const internal::DecodeControl* control = nullptr) {
    return internal::LaunchSampleToken(
        logits, Pointer<float>(workspace_, offsets_.sampling_logits),
        Pointer<double>(workspace_, offsets_.sampling_cumulative),
        Pointer<std::uint32_t>(workspace_, offsets_.sampling_token_ids),
        Pointer<std::uint32_t>(workspace_, offsets_.sorted_token_ids),
        repetition_mask,
        Pointer<std::uint32_t>(workspace_, offsets_.suppressed),
        suppressed_token_count_, static_cast<std::uint32_t>(kVocabulary),
        sampling_, step, control, selected,
        Pointer<std::uint8_t>(workspace_, offsets_.sampling_sort_workspace),
        sampling_sort_workspace_bytes_, stream_);
  }

  [[nodiscard]] Status SelectSampledToken(
      float* logits, std::uint32_t* selected,
      const internal::DecodeControl* control = nullptr) {
    return SelectSampledTokenWithState(
        logits, selected,
        Pointer<std::uint32_t>(workspace_, offsets_.repetition_mask),
        control == nullptr ? sampling_step_++ : 0U, control);
  }

  [[nodiscard]] Status AllocateCache() {
    LayoutBuilder layout;
    struct CacheOffsets { std::uint64_t key; std::uint64_t value; };
    std::array<CacheOffsets, kLayers> offsets{};
    for (std::size_t index = 0; index < kLayers; ++index) {
      const auto& layer = model_.layers()[index];
      const std::uint64_t cache_tokens =
          layer.global ? max_context_ : std::min(max_context_, kSlidingWindow);
      Result<std::uint64_t> key =
          kv_cache_mode_ == KvCacheMode::kCheckpointFp8
              ? layout.Add<std::uint8_t>(cache_tokens * layer.kv_elements)
              : layout.Add<float>(cache_tokens * layer.kv_elements);
      Result<std::uint64_t> value =
          kv_cache_mode_ == KvCacheMode::kCheckpointFp8
              ? layout.Add<std::uint8_t>(cache_tokens * layer.kv_elements)
              : layout.Add<float>(cache_tokens * layer.kv_elements);
      if (!key.ok()) return key.status();
      if (!value.ok()) return value.status();
      offsets[index] = {key.value(), value.value()};
    }
    auto size = AlignUp(layout.size(), kAlignment);
    if (!size.ok()) return size.status();
    Status status = cache_.Allocate(
        size.value(), kv_cache_mode_ == KvCacheMode::kCheckpointFp8
                          ? "allocate checkpoint FP8 KV cache"
                          : "allocate BF16-semantics KV cache");
    if (!status.ok()) return status;
    for (std::size_t index = 0; index < kLayers; ++index) {
      if (kv_cache_mode_ == KvCacheMode::kCheckpointFp8) {
        model_.SetLayerFp8Cache(
            index, Pointer<std::uint8_t>(cache_, offsets[index].key),
            Pointer<std::uint8_t>(cache_, offsets[index].value));
      } else {
        model_.SetLayerBf16Cache(
            index, Pointer<float>(cache_, offsets[index].key),
            Pointer<float>(cache_, offsets[index].value));
      }
    }
    return Status::Ok();
  }

  [[nodiscard]] Status AllocateWorkspace() {
    LayoutBuilder layout;
#define GEM16_ADD(field, type, elements)                 \
    do {                                                   \
      auto next = layout.Add<type>(elements);              \
      if (!next.ok()) return next.status();                 \
      offsets_.field = next.value();                        \
    } while (false)
    GEM16_ADD(decode_control, internal::DecodeControl, 1U);
    GEM16_ADD(hidden_a, float, kHidden);
    GEM16_ADD(hidden_b, float, kHidden);
    GEM16_ADD(normalized, float, kHidden);
    GEM16_ADD(fp8_activation, std::uint8_t, kHidden);
    GEM16_ADD(fp8_scale, float, 1U);
    GEM16_ADD(q, float, kQueryHeads * 512U);
    GEM16_ADD(k, float, 8U * 256U);
    GEM16_ADD(v, float, 8U * 256U);
    GEM16_ADD(q_norm, float, kQueryHeads * 512U);
    GEM16_ADD(k_norm, float, 8U * 256U);
    GEM16_ADD(v_norm, float, 8U * 256U);
    GEM16_ADD(scores, float,
                internal::DecodeAttentionWorkspaceElements(max_context_));
    GEM16_ADD(attention, float, kQueryHeads * 512U);
    GEM16_ADD(o_activation, std::uint8_t, kQueryHeads * 512U);
    GEM16_ADD(o_scale, float, 1U);
    GEM16_ADD(projection, float, kHidden);
    GEM16_ADD(post_norm, float, kHidden);
    GEM16_ADD(mlp_packed, std::uint8_t, kHidden / 2U);
    GEM16_ADD(mlp_scales, std::uint8_t, kHidden / 16U);
    GEM16_ADD(gate, float, kIntermediate);
    GEM16_ADD(up, float, kIntermediate);
    GEM16_ADD(product, float, kIntermediate);
    GEM16_ADD(down_packed, std::uint8_t, kIntermediate / 2U);
    GEM16_ADD(down_scales, std::uint8_t, kIntermediate / 16U);
    GEM16_ADD(logits, float, kVocabulary);
    if (sampling_.enabled) {
      GEM16_ADD(sampling_logits, float, kVocabulary);
      GEM16_ADD(sampling_cumulative, double, kVocabulary);
      GEM16_ADD(sampling_token_ids, std::uint32_t, kVocabulary);
      GEM16_ADD(sorted_token_ids, std::uint32_t, kVocabulary);
      auto sort_bytes = internal::SamplingWorkspaceBytes(
          static_cast<std::uint32_t>(kVocabulary), stream_);
      if (!sort_bytes.ok()) return sort_bytes.status();
      sampling_sort_workspace_bytes_ = sort_bytes.value();
      GEM16_ADD(sampling_sort_workspace, std::uint8_t,
                sampling_sort_workspace_bytes_);
      GEM16_ADD(repetition_mask, std::uint32_t, kRepetitionMaskWords);
    }
    GEM16_ADD(output_candidates, ArgmaxValue, kFusedOutputHeadBlocks);
    GEM16_ADD(selected, std::uint32_t, 1U);
    GEM16_ADD(suppressed, std::uint32_t, kMaximumSuppressedTokens);
#undef GEM16_ADD
    auto size = AlignUp(layout.size(), kAlignment);
    if (!size.ok()) return size.status();
    offsets_.total = size.value();
    Status status = workspace_.Allocate(
        size.value(), "allocate inference workspace arena");
    if (!status.ok()) return status;
    if (sampling_.enabled) {
      const cudaError_t error = cudaMemsetAsync(
          Pointer<std::uint32_t>(workspace_, offsets_.repetition_mask), 0,
          static_cast<std::size_t>(kRepetitionMaskWords *
                                   sizeof(std::uint32_t)),
          stream_);
      if (error != cudaSuccess) {
        return CudaFailure("initialize repetition mask", error);
      }
    }
    return Status::Ok();
  }

  [[nodiscard]] Status AllocateMtpWorkspace() {
    LayoutBuilder layout;
    for (std::size_t index = 0; index < model_.layers().size(); ++index) {
      const std::uint64_t elements =
          kMaximumMtpVerifyTokens * model_.layers()[index].kv_elements;
      Result<std::uint64_t> key =
          kv_cache_mode_ == KvCacheMode::kCheckpointFp8
              ? layout.Add<std::uint8_t>(elements)
              : layout.Add<float>(elements);
      Result<std::uint64_t> value =
          kv_cache_mode_ == KvCacheMode::kCheckpointFp8
              ? layout.Add<std::uint8_t>(elements)
              : layout.Add<float>(elements);
      Result<std::uint64_t> backup_key =
          kv_cache_mode_ == KvCacheMode::kCheckpointFp8
              ? layout.Add<std::uint8_t>(elements)
              : layout.Add<float>(elements);
      Result<std::uint64_t> backup_value =
          kv_cache_mode_ == KvCacheMode::kCheckpointFp8
              ? layout.Add<std::uint8_t>(elements)
              : layout.Add<float>(elements);
      if (!key.ok()) return key.status();
      if (!value.ok()) return value.status();
      if (!backup_key.ok()) return backup_key.status();
      if (!backup_value.ok()) return backup_value.status();
      mtp_offsets_.layers[index] = {
          key.value(), value.value(), backup_key.value(), backup_value.value()};
    }
    auto attention_workspace = layout.Add<float>(
        mtp_draft_tokens_ == 0U
            ? 0U
            : (mtp_draft_tokens_ + 1U) *
                  internal::DecodeAttentionWorkspaceElements(max_context_));
    auto candidates = layout.Add<ArgmaxValue>(
        kMaximumMtpVerifyTokens * kFusedOutputHeadBlocks);
    auto selected = layout.Add<std::uint32_t>(kMaximumMtpVerifyTokens);
    auto sampling_logits = layout.Add<float>(
        sampling_.enabled ? kMaximumMtpVerifyTokens * kVocabulary : 0U);
    auto sampling_repetition_masks = layout.Add<std::uint32_t>(
        sampling_.enabled
            ? kMaximumMtpVerifyTokens * kRepetitionMaskWords
            : 0U);
    auto stop_tokens = layout.Add<std::uint32_t>(kMaximumSuppressedTokens);
    auto transaction = layout.Add<internal::MtpGroupTransaction>(1U);
    auto row_controls = layout.Add<internal::DecodeControl>(3U);
    auto chain_result = layout.Add<internal::MtpChainResult>(1U);
    auto chain_outputs = layout.Add<std::uint32_t>(max_context_);
    auto chain_proposals = layout.Add<std::uint32_t>(2U * max_context_);
    auto committed_hidden = layout.Add<float>(kHidden);
    if (!attention_workspace.ok()) return attention_workspace.status();
    if (!candidates.ok()) return candidates.status();
    if (!selected.ok()) return selected.status();
    if (!sampling_logits.ok()) return sampling_logits.status();
    if (!sampling_repetition_masks.ok()) {
      return sampling_repetition_masks.status();
    }
    if (!stop_tokens.ok()) return stop_tokens.status();
    if (!transaction.ok()) return transaction.status();
    if (!row_controls.ok()) return row_controls.status();
    if (!chain_result.ok()) return chain_result.status();
    if (!chain_outputs.ok()) return chain_outputs.status();
    if (!chain_proposals.ok()) return chain_proposals.status();
    if (!committed_hidden.ok()) return committed_hidden.status();
    mtp_offsets_.attention_workspace = attention_workspace.value();
    mtp_offsets_.output_candidates = candidates.value();
    mtp_offsets_.selected = selected.value();
    mtp_offsets_.sampling_logits = sampling_logits.value();
    mtp_offsets_.sampling_repetition_masks =
        sampling_repetition_masks.value();
    mtp_offsets_.stop_tokens = stop_tokens.value();
    mtp_offsets_.transaction = transaction.value();
    mtp_offsets_.row_controls = row_controls.value();
    mtp_offsets_.chain_result = chain_result.value();
    mtp_offsets_.chain_outputs = chain_outputs.value();
    mtp_offsets_.chain_proposals = chain_proposals.value();
    mtp_offsets_.committed_hidden = committed_hidden.value();
    auto size = AlignUp(layout.size(), kAlignment);
    if (!size.ok()) return size.status();
    mtp_offsets_.total = size.value();
    return mtp_workspace_.Allocate(size.value(),
                                   "allocate MTP verification workspace");
  }

  [[nodiscard]] Status AllocatePrefillWorkspace() {
    LayoutBuilder layout;
#define GEM16_PREFILL_ADD(field, type, elements)          \
    do {                                                     \
      auto next = layout.Add<type>(elements);                \
      if (!next.ok()) return next.status();                   \
      prefill_offsets_.field = next.value();                  \
    } while (false)
    const std::uint64_t tokens = prefill_chunk_tokens_;
    constexpr std::uint64_t max_q = kQueryHeads * 512U;
    constexpr std::uint64_t max_kv = 8U * 256U;
    GEM16_PREFILL_ADD(token_ids, std::uint32_t, tokens);
    GEM16_PREFILL_ADD(audio_frames, float, tokens * 640U);
    GEM16_PREFILL_ADD(audio_normalized, float, tokens * 640U);
    GEM16_PREFILL_ADD(vision_patches, float, 280U * 6912U);
    GEM16_PREFILL_ADD(vision_patch_normalized, float, 280U * 6912U);
    GEM16_PREFILL_ADD(vision_hidden_a, float, 280U * kHidden);
    GEM16_PREFILL_ADD(vision_hidden_b, float, 280U * kHidden);
    GEM16_PREFILL_ADD(vision_positions, std::int32_t, 280U * 2U);
    GEM16_PREFILL_ADD(hidden_a, float, tokens * kHidden);
    GEM16_PREFILL_ADD(hidden_b, float, tokens * kHidden);
    GEM16_PREFILL_ADD(normalized, float, tokens * kHidden);
    GEM16_PREFILL_ADD(fp8_activation, std::uint8_t, tokens * max_q);
    GEM16_PREFILL_ADD(fp8_scales, float, tokens);
    GEM16_PREFILL_ADD(q, std::uint16_t, tokens * max_q);
    GEM16_PREFILL_ADD(k, std::uint16_t, tokens * max_kv);
    GEM16_PREFILL_ADD(v, std::uint16_t, tokens * max_kv);
    GEM16_PREFILL_ADD(q_norm, float, tokens * max_q);
    GEM16_PREFILL_ADD(k_norm, float, tokens * max_kv);
    GEM16_PREFILL_ADD(v_norm, float, tokens * max_kv);
    GEM16_PREFILL_ADD(k_fp8, std::uint8_t, tokens * max_kv);
    GEM16_PREFILL_ADD(v_fp8, std::uint8_t, tokens * max_kv);
    if (kv_cache_mode_ == KvCacheMode::kBf16Correctness) {
      GEM16_PREFILL_ADD(scores, float,
                          tokens * kQueryHeads * max_context_);
    }
    if (kv_cache_mode_ == KvCacheMode::kCheckpointFp8) {
      const std::uint64_t attention_bytes = std::max(
          tokens * max_q * sizeof(std::uint16_t),
          kMaximumMtpVerifyTokens * max_q * sizeof(float));
      GEM16_PREFILL_ADD(attention, std::uint8_t, attention_bytes);
    } else {
      GEM16_PREFILL_ADD(attention, float, tokens * max_q);
    }
    GEM16_PREFILL_ADD(o_activation, std::uint8_t, tokens * max_q);
    GEM16_PREFILL_ADD(o_scales, float, tokens);
    GEM16_PREFILL_ADD(projection, std::uint16_t, tokens * kHidden);
    GEM16_PREFILL_ADD(post_norm, float, tokens * kHidden);
    GEM16_PREFILL_ADD(mlp_packed, std::uint8_t, tokens * kHidden / 2U);
    GEM16_PREFILL_ADD(mlp_scales, std::uint8_t, tokens * kHidden / 16U);
    GEM16_PREFILL_ADD(gate, std::uint16_t, tokens * kIntermediate);
    GEM16_PREFILL_ADD(up, std::uint16_t, tokens * kIntermediate);
    GEM16_PREFILL_ADD(down_packed, std::uint8_t, tokens * kIntermediate / 2U);
    GEM16_PREFILL_ADD(down_scales, std::uint8_t, tokens * kIntermediate / 16U);
    constexpr std::uint64_t kCutlassScaleRows = 128U;
    constexpr std::uint64_t kCutlassWorkspaceBytes = 8U * 1024U * 1024U;
    const std::uint64_t cutlass_tokens =
        ((tokens + kCutlassScaleRows - 1U) / kCutlassScaleRows) *
        kCutlassScaleRows;
    GEM16_PREFILL_ADD(cutlass_activation_scales, std::uint8_t,
                        cutlass_tokens * kIntermediate / 16U);
    GEM16_PREFILL_ADD(cutlass_weight, std::uint8_t,
                        kIntermediate * kHidden / 2U);
    GEM16_PREFILL_ADD(cutlass_weight_scales, std::uint8_t,
                        kIntermediate * kHidden / 16U);
    GEM16_PREFILL_ADD(cutlass_workspace, std::uint8_t,
                        kCutlassWorkspaceBytes);
    GEM16_PREFILL_ADD(local_rope_cosine, float, max_context_ * 128U);
    GEM16_PREFILL_ADD(local_rope_sine, float, max_context_ * 128U);
    GEM16_PREFILL_ADD(global_rope_cosine, float, max_context_ * 64U);
    GEM16_PREFILL_ADD(global_rope_sine, float, max_context_ * 64U);
#undef GEM16_PREFILL_ADD
    auto size = AlignUp(layout.size(), kAlignment);
    if (!size.ok()) return size.status();
    return prefill_workspace_.Allocate(size.value(), "allocate native prefill workspace");
  }

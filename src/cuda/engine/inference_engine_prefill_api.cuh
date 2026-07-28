  [[nodiscard]] std::uint64_t cache_bytes() const { return cache_.bytes(); }
  [[nodiscard]] std::uint64_t workspace_bytes() const {
    return workspace_.bytes() + prefill_workspace_.bytes() +
           mtp_workspace_.bytes();
  }
  [[nodiscard]] std::uint64_t decode_graph_device_bytes() const {
    return decode_graph_device_bytes_;
  }
  [[nodiscard]] std::uint64_t prefill_chunk_tokens() const {
    return prefill_chunk_tokens_;
  }

  [[nodiscard]] Result<std::uint32_t> Prefill(
      std::span<const std::uint32_t> token_ids,
      std::span<float> host_logits = {}) {
    return PrefillAt(token_ids, 0U, host_logits);
  }

  [[nodiscard]] Result<std::uint32_t> PrefillAt(
      std::span<const std::uint32_t> token_ids, std::uint64_t start_position,
      std::span<float> host_logits = {}) {
    const NvtxRange range("gem16.prefill");
    if (token_ids.empty() || start_position > max_context_ ||
        token_ids.size() > max_context_ - start_position) {
      return Error(StatusCode::kInvalidArgument, "prefill token extent is invalid");
    }
    if (!host_logits.empty() && host_logits.size() != kVocabulary) {
      return Error(StatusCode::kInternal,
                   "host prefill logit capture span has invalid size");
    }
    std::uint32_t selected_token = 0U;
    for (std::size_t begin = 0; begin < token_ids.size(); begin += prefill_chunk_tokens_) {
      const std::uint64_t tokens = std::min<std::size_t>(
          prefill_chunk_tokens_, token_ids.size() - begin);
      auto* device_tokens = Pointer<std::uint32_t>(prefill_workspace_, prefill_offsets_.token_ids);
      cudaError_t error = cudaMemcpyAsync(
          device_tokens, token_ids.data() + begin,
          static_cast<std::size_t>(tokens * sizeof(std::uint32_t)),
          cudaMemcpyHostToDevice, stream_);
      if (error != cudaSuccess) return CudaFailure("copy prefill token IDs", error);
      Status status;
      if (sampling_.enabled && sampling_.repetition_penalty != 1.0F) {
        status = internal::LaunchMarkRepetitionTokens(
            device_tokens, tokens,
            Pointer<std::uint32_t>(workspace_, offsets_.repetition_mask),
            stream_);
        if (!status.ok()) return status;
      }
      float* hidden = Pointer<float>(prefill_workspace_, prefill_offsets_.hidden_a);
      const std::uint64_t hidden_elements = tokens * kHidden;
      EmbeddingBatchKernel<<<static_cast<unsigned>((hidden_elements + kThreads - 1U) /
                                                   kThreads),
                             kThreads, 0, stream_>>>(
          model_.embedding(), device_tokens, hidden, hidden_elements);
      error = cudaGetLastError();
      if (error != cudaSuccess) return CudaFailure("launch prefill embedding", error);
      for (const auto& layer : model_.layers()) {
        status = RunLayerBatch(layer, start_position + begin, tokens);
        if (!status.ok()) return status;
      }
      float* normalized = Pointer<float>(prefill_workspace_, prefill_offsets_.normalized);
      status = internal::LaunchRmsNormBf16(
          hidden, model_.final_norm(), normalized, tokens, kHidden, kEpsilon, stream_);
      if (!status.ok()) return status;
      if (begin + tokens == token_ids.size()) {
        float* last = normalized + (tokens - 1U) * kHidden;
        latest_target_hidden_ = last;
        float* logits = Pointer<float>(workspace_, offsets_.logits);
        auto* selected = Pointer<std::uint32_t>(workspace_, offsets_.selected);
        if (sampling_.enabled) {
          status = internal::LaunchFusedOutputHeadCandidates(
              model_.embedding(), last,
              Pointer<std::uint32_t>(workspace_, offsets_.suppressed),
              suppressed_token_count_, nullptr,
              Pointer<ArgmaxValue>(workspace_, offsets_.output_candidates),
              logits, stream_);
        } else {
          status = internal::LaunchOutputHeadLogits(model_.embedding(), last,
                                                    logits, stream_);
        }
        if (!status.ok()) return status;
        if (!host_logits.empty()) {
          error = cudaMemcpyAsync(host_logits.data(), logits,
                                  host_logits.size_bytes(),
                                  cudaMemcpyDeviceToHost, stream_);
          if (error != cudaSuccess) {
            return CudaFailure("copy prefill full logits", error);
          }
        }
        if (sampling_.enabled) {
          status = SelectSampledToken(logits, selected);
          if (!status.ok()) return status;
        } else {
          status = internal::LaunchLogitArgmax(
              logits, Pointer<std::uint32_t>(workspace_, offsets_.suppressed),
              suppressed_token_count_, selected, stream_);
          if (!status.ok()) return status;
        }
        error = cudaMemcpyAsync(&selected_token, selected, sizeof(selected_token),
                                cudaMemcpyDeviceToHost, stream_);
        if (error != cudaSuccess) return CudaFailure("copy prefill token", error);
      }
    }
    const cudaError_t error = cudaStreamSynchronize(stream_);
    if (error != cudaSuccess) return CudaFailure("synchronize prefill", error);
    return selected_token;
  }

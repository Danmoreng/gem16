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
      std::span<float> host_logits = {},
      std::span<const AudioEmbeddingSegment> audio_segments = {},
      std::span<const VisionEmbeddingSegment> vision_segments = {}) {
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
    for (std::size_t begin = 0; begin < token_ids.size();) {
      const std::uint64_t maximum_tokens = std::min<std::size_t>(
          prefill_chunk_tokens_, token_ids.size() - begin);
      const std::uint64_t proposed_begin = start_position + begin;
      const std::uint64_t tokens = internal::PlanVisionAwarePrefillChunk(
          proposed_begin, maximum_tokens, vision_segments);
      if (tokens == 0U || tokens > prefill_chunk_tokens_) {
        return Error(StatusCode::kInternal,
                     "vision-aware prefill chunk planning failed");
      }
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
      const bool physical_hidden =
          kv_cache_mode_ == KvCacheMode::kCheckpointFp8;
      auto* hidden_bf16 = Pointer<std::uint16_t>(
          prefill_workspace_, prefill_offsets_.hidden_a);
      float* hidden = reinterpret_cast<float*>(hidden_bf16);
      const std::uint64_t hidden_elements = tokens * kHidden;
      if (physical_hidden) {
        EmbeddingBatchKernel<<<
            static_cast<unsigned>((hidden_elements + kThreads - 1U) /
                                  kThreads),
            kThreads, 0, stream_>>>(model_.embedding(), device_tokens,
                                    hidden_bf16, hidden_elements);
      } else {
        EmbeddingBatchKernel<<<
            static_cast<unsigned>((hidden_elements + kThreads - 1U) /
                                  kThreads),
            kThreads, 0, stream_>>>(model_.embedding(), device_tokens, hidden,
                                    hidden_elements);
      }
      error = cudaGetLastError();
      if (error != cudaSuccess) return CudaFailure("launch prefill embedding", error);
      const std::uint64_t chunk_begin = start_position + begin;
      const std::uint64_t chunk_end = chunk_begin + tokens;
      std::uint64_t vision_begin = 0U;
      std::uint64_t vision_end = 0U;
      for (const AudioEmbeddingSegment& segment : audio_segments) {
        if (segment.frames.empty() || segment.frames.size() % 640U != 0U) {
          return Error(StatusCode::kInvalidArgument,
                       "audio embedding segment has invalid frame geometry");
        }
        const std::uint64_t frame_count = segment.frames.size() / 640U;
        if (frame_count > 750U ||
            segment.prompt_offset > max_context_ ||
            frame_count > max_context_ - segment.prompt_offset) {
          return Error(StatusCode::kInvalidArgument,
                       "audio embedding segment exceeds its supported prompt extent");
        }
        const std::uint64_t segment_end = segment.prompt_offset + frame_count;
        const std::uint64_t overlap_begin =
            std::max(chunk_begin, segment.prompt_offset);
        const std::uint64_t overlap_end = std::min(chunk_end, segment_end);
        if (overlap_begin >= overlap_end) continue;
        const std::uint64_t overlap_frames = overlap_end - overlap_begin;
        const std::uint64_t source_frame = overlap_begin - segment.prompt_offset;
        float* device_audio = Pointer<float>(
            prefill_workspace_, prefill_offsets_.audio_frames);
        error = cudaMemcpyAsync(
            device_audio, segment.frames.data() + source_frame * 640U,
            static_cast<std::size_t>(overlap_frames * 640U * sizeof(float)),
            cudaMemcpyHostToDevice, stream_);
        if (error != cudaSuccess) {
          return CudaFailure("copy audio waveform frames", error);
        }
        float* media_hidden =
            physical_hidden
                ? reinterpret_cast<float*>(Pointer<std::uint16_t>(
                      prefill_workspace_, prefill_offsets_.hidden_b))
                : hidden + (overlap_begin - chunk_begin) * kHidden;
        status = internal::LaunchAudioProjection(
            device_audio,
            Pointer<float>(prefill_workspace_,
                           prefill_offsets_.audio_normalized),
            model_.audio_projection(), media_hidden, overlap_frames, stream_);
        if (!status.ok()) return status;
        if (physical_hidden) {
          status = LaunchStorePhysicalBf16(
              media_hidden,
              hidden_bf16 + (overlap_begin - chunk_begin) * kHidden,
              overlap_frames * kHidden, stream_);
          if (!status.ok()) return status;
        }
      }
      for (const VisionEmbeddingSegment& segment : vision_segments) {
        if (segment.patches.empty() || segment.patches.size() % 6912U != 0U) {
          return Error(StatusCode::kInvalidArgument,
                       "vision embedding segment has invalid patch geometry");
        }
        const std::uint64_t patch_count = segment.patches.size() / 6912U;
        const std::uint64_t segment_end = segment.prompt_offset + patch_count;
        if (segment_end <= chunk_begin || segment.prompt_offset >= chunk_end) {
          continue;
        }
        if (segment.prompt_offset < chunk_begin || segment_end > chunk_end ||
            patch_count > 280U || segment.positions.size() != patch_count * 2U) {
          return Error(StatusCode::kInvalidArgument,
                       "vision embedding segment must fit wholly in one prefill chunk");
        }
        if (vision_end != 0U) {
          return Error(StatusCode::kUnsupported,
                       "one image per prefill chunk is currently supported");
        }
        auto* device_patches = Pointer<float>(
            prefill_workspace_, prefill_offsets_.vision_patches);
        auto* device_positions = Pointer<std::int32_t>(
            prefill_workspace_, prefill_offsets_.vision_positions);
        error = cudaMemcpyAsync(
            device_patches, segment.patches.data(), segment.patches.size_bytes(),
            cudaMemcpyHostToDevice, stream_);
        if (error == cudaSuccess) {
          error = cudaMemcpyAsync(
              device_positions, segment.positions.data(),
              segment.positions.size_bytes(), cudaMemcpyHostToDevice, stream_);
        }
        if (error != cudaSuccess) {
          return CudaFailure("copy vision patches and positions", error);
        }
        float* media_hidden =
            physical_hidden
                ? reinterpret_cast<float*>(Pointer<std::uint16_t>(
                      prefill_workspace_, prefill_offsets_.hidden_b))
                : hidden + (segment.prompt_offset - chunk_begin) * kHidden;
        status = internal::LaunchVisionProjection(
            device_patches, device_positions,
            Pointer<float>(prefill_workspace_,
                           prefill_offsets_.vision_patch_normalized),
            Pointer<float>(prefill_workspace_, prefill_offsets_.vision_hidden_a),
            Pointer<float>(prefill_workspace_, prefill_offsets_.vision_hidden_b),
            model_.vision(), media_hidden, patch_count, stream_);
        if (!status.ok()) return status;
        if (physical_hidden) {
          status = LaunchStorePhysicalBf16(
              media_hidden,
              hidden_bf16 + (segment.prompt_offset - chunk_begin) * kHidden,
              patch_count * kHidden, stream_);
          if (!status.ok()) return status;
        }
        vision_begin = segment.prompt_offset;
        vision_end = segment_end;
      }
      for (const auto& layer : model_.layers()) {
        status = RunLayerBatch(layer, start_position + begin, tokens, kLayers,
                               nullptr, vision_begin, vision_end);
        if (!status.ok()) return status;
      }
      float* normalized = Pointer<float>(prefill_workspace_, prefill_offsets_.normalized);
      status = physical_hidden
                   ? internal::LaunchRmsNormBf16Input(
                         hidden_bf16, model_.final_norm(), normalized, tokens,
                         kHidden, kEpsilon, stream_)
                   : internal::LaunchRmsNormBf16(
                         hidden, model_.final_norm(), normalized, tokens,
                         kHidden, kEpsilon, stream_);
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
      begin += static_cast<std::size_t>(tokens);
    }
    const cudaError_t error = cudaStreamSynchronize(stream_);
    if (error != cudaSuccess) return CudaFailure("synchronize prefill", error);
    return selected_token;
  }

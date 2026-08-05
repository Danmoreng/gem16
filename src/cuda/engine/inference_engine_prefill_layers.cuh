  [[nodiscard]] Status RunLayerBatch(
      const LayerBinding& layer, std::uint64_t start_position,
      std::uint64_t tokens, std::size_t mtp_layer_index = kLayers,
      const internal::DecodeControl* mtp_row_controls = nullptr,
      std::uint64_t vision_begin = 0U, std::uint64_t vision_end = 0U) {
    const bool mtp_verification = mtp_layer_index < kLayers;
    const bool controlled_mtp_d2 = mtp_row_controls != nullptr;
    if (vision_begin < vision_end &&
        kv_cache_mode_ != KvCacheMode::kCheckpointFp8) {
      return Error(StatusCode::kUnsupported,
                   "vision bidirectional prefill currently requires checkpoint FP8 KV cache");
    }
    if (controlled_mtp_d2 &&
        (!mtp_verification || tokens != 3U ||
         kv_cache_mode_ != KvCacheMode::kCheckpointFp8)) {
      return Error(StatusCode::kInvalidArgument,
                   "controlled MTP D2 layer geometry is invalid");
    }
    const NvtxRange range(mtp_verification ? "gem16.mtp.verify.layer"
                                           : "gem16.prefill.layer");
    const bool physical_hidden =
        !mtp_verification &&
        kv_cache_mode_ == KvCacheMode::kCheckpointFp8;
    auto* hidden_a_bf16 = Pointer<std::uint16_t>(
        prefill_workspace_, prefill_offsets_.hidden_a);
    float* hidden_a = reinterpret_cast<float*>(hidden_a_bf16);
    auto* fp8 = Pointer<std::uint8_t>(prefill_workspace_, prefill_offsets_.fp8_activation);
    float* fp8_scales = Pointer<float>(prefill_workspace_, prefill_offsets_.fp8_scales);
    auto* q_bf16 =
        Pointer<std::uint16_t>(prefill_workspace_, prefill_offsets_.q);
    auto* k_bf16 =
        Pointer<std::uint16_t>(prefill_workspace_, prefill_offsets_.k);
    auto* v_bf16 =
        Pointer<std::uint16_t>(prefill_workspace_, prefill_offsets_.v);
    // The fixed-T3 verifier retains the native FP32 projection contract. Its
    // three rows fit inside the larger physical-BF16 prompt regions.
    float* q = reinterpret_cast<float*>(q_bf16);
    float* k = reinterpret_cast<float*>(k_bf16);
    float* v = reinterpret_cast<float*>(v_bf16);
    float* q_norm = Pointer<float>(prefill_workspace_, prefill_offsets_.q_norm);
    float* k_norm = Pointer<float>(prefill_workspace_, prefill_offsets_.k_norm);
    float* v_norm = Pointer<float>(prefill_workspace_, prefill_offsets_.v_norm);
    auto* attention_bf16 = Pointer<std::uint16_t>(
        prefill_workspace_, prefill_offsets_.attention);
    // The fixed-T3 verifier and BF16 correctness mode retain FP32 attention.
    // Their active rows fit in the physical-BF16 production allocation.
    float* attention = reinterpret_cast<float*>(attention_bf16);
    auto* cutlass_workspace = Pointer<std::uint8_t>(
        prefill_workspace_, prefill_offsets_.cutlass_workspace);
    constexpr std::size_t kCutlassWorkspaceBytes = 8U * 1024U * 1024U;
    Status status =
        physical_hidden
            ? internal::LaunchRmsNormFp8TokenQuantizationBf16Batch(
                  hidden_a_bf16, layer.input_norm, fp8, fp8_scales, tokens,
                  kHidden, kEpsilon, stream_)
            : internal::LaunchRmsNormFp8TokenQuantizationBatch(
                  hidden_a, layer.input_norm, fp8, fp8_scales, tokens,
                  kHidden, kEpsilon, stream_);
    if (!status.ok()) return status;
    status = mtp_verification
                 ? internal::LaunchFp8Sm120GroupedQkvProjectionBatch(
                       fp8, fp8_scales, layer.q.weight, layer.q.scales, q,
                       layer.q.rows, layer.k.weight, layer.k.scales, k,
                       layer.k.rows,
                       layer.global ? nullptr : layer.v.weight,
                       layer.global ? nullptr : layer.v.scales,
                       layer.global ? nullptr : v,
                       layer.global ? 0U : layer.v.rows, tokens,
                       layer.q.contracting, stream_)
                 : LaunchFp8QkvProjectionBatch(
                       fp8, fp8_scales, layer.q, q_bf16, layer.k, k_bf16,
                       layer.global ? nullptr : &layer.v, v_bf16, tokens,
                       cutlass_workspace, kCutlassWorkspaceBytes, stream_);
    if (!status.ok()) return status;
    if (layer.global) {
      const std::size_t element_bytes =
          mtp_verification ? sizeof(float) : sizeof(std::uint16_t);
      const cudaError_t error = cudaMemcpyAsync(
          mtp_verification ? static_cast<void*>(v)
                           : static_cast<void*>(v_bf16),
          mtp_verification ? static_cast<const void*>(k)
                           : static_cast<const void*>(k_bf16),
          static_cast<std::size_t>(tokens * layer.kv_elements * element_bytes),
          cudaMemcpyDeviceToDevice, stream_);
      if (error != cudaSuccess) return CudaFailure("reuse batched global K for V", error);
    }
    if (mtp_verification) {
      // The direct verifier projection writes FP32. Restore the ordinary
      // decode projection boundary before scale-free V normalization.
      status = LaunchRoundBf16(v, tokens * layer.kv_elements, stream_);
      if (!status.ok()) return status;
      status = internal::LaunchRmsNormBf16(
          v, nullptr, v_norm, tokens * layer.kv_heads,
          layer.head_dimension, kEpsilon, stream_);
    } else {
      status = internal::LaunchRmsNormBf16Input(
          v_bf16, nullptr, v_norm, tokens * layer.kv_heads,
          layer.head_dimension, kEpsilon, stream_);
    }
    if (!status.ok()) return status;
    if (controlled_mtp_d2) {
      status = internal::LaunchProjectionRmsNormRotaryBf16BatchControlled(
          q, layer.q_norm, q_norm, k, layer.k_norm, k_norm,
          layer.global
              ? Pointer<float>(prefill_workspace_,
                               prefill_offsets_.global_rope_cosine)
              : Pointer<float>(prefill_workspace_,
                               prefill_offsets_.local_rope_cosine),
          layer.global
              ? Pointer<float>(prefill_workspace_,
                               prefill_offsets_.global_rope_sine)
              : Pointer<float>(prefill_workspace_,
                               prefill_offsets_.local_rope_sine),
          mtp_row_controls, tokens, kQueryHeads, layer.kv_heads,
          layer.head_dimension, layer.global ? 0.25 : 1.0, kEpsilon,
          stream_);
    } else if (mtp_verification) {
      status = internal::LaunchProjectionRmsNormRotaryBf16Batch(
          q, layer.q_norm, q_norm, k, layer.k_norm, k_norm,
          layer.global
              ? Pointer<float>(prefill_workspace_,
                               prefill_offsets_.global_rope_cosine) +
                    start_position * layer.head_dimension / 8U
              : Pointer<float>(prefill_workspace_,
                               prefill_offsets_.local_rope_cosine) +
                    start_position * layer.head_dimension / 2U,
          layer.global
              ? Pointer<float>(prefill_workspace_,
                               prefill_offsets_.global_rope_sine) +
                    start_position * layer.head_dimension / 8U
              : Pointer<float>(prefill_workspace_,
                               prefill_offsets_.local_rope_sine) +
                    start_position * layer.head_dimension / 2U,
          tokens, kQueryHeads, layer.kv_heads, layer.head_dimension,
          layer.global ? 0.25 : 1.0, kEpsilon, stream_);
    } else {
      status = internal::LaunchProjectionRmsNormRotaryBf16BatchInput(
          q_bf16, layer.q_norm, q_norm, k_bf16, layer.k_norm, k_norm,
          layer.global
              ? Pointer<float>(prefill_workspace_,
                               prefill_offsets_.global_rope_cosine) +
                    start_position * 64U
              : Pointer<float>(prefill_workspace_,
                               prefill_offsets_.local_rope_cosine) +
                    start_position * 128U,
          layer.global
              ? Pointer<float>(prefill_workspace_,
                               prefill_offsets_.global_rope_sine) +
                    start_position * 64U
              : Pointer<float>(prefill_workspace_,
                               prefill_offsets_.local_rope_sine) +
                    start_position * 128U,
          tokens, kQueryHeads, layer.kv_heads, layer.head_dimension,
          layer.global ? 0.25 : 1.0, kEpsilon, stream_);
    }
    if (!status.ok()) return status;
    const std::uint64_t capacity =
        layer.global ? max_context_ : std::min(max_context_, kSlidingWindow);
    if (kv_cache_mode_ == KvCacheMode::kCheckpointFp8) {
      auto* k_fp8 = Pointer<std::uint8_t>(prefill_workspace_, prefill_offsets_.k_fp8);
      auto* v_fp8 = Pointer<std::uint8_t>(prefill_workspace_, prefill_offsets_.v_fp8);
      status = internal::LaunchQuantizeKvFp8Batch(
          k_norm, v_norm, k_fp8, v_fp8, layer.k_cache_scale,
          layer.v_cache_scale, tokens, layer.kv_elements, stream_);
      if (!status.ok()) return status;
      if (mtp_verification) {
        auto* speculative_key = Pointer<std::uint8_t>(
            mtp_workspace_, mtp_offsets_.layers[mtp_layer_index].key);
        auto* speculative_value = Pointer<std::uint8_t>(
            mtp_workspace_, mtp_offsets_.layers[mtp_layer_index].value);
        const std::size_t bytes = static_cast<std::size_t>(
            tokens * layer.kv_elements * sizeof(std::uint8_t));
        cudaError_t copy_error = cudaMemcpyAsync(
            speculative_key, k_fp8, bytes, cudaMemcpyDeviceToDevice, stream_);
        if (copy_error == cudaSuccess) {
          copy_error = cudaMemcpyAsync(speculative_value, v_fp8, bytes,
                                       cudaMemcpyDeviceToDevice, stream_);
        }
        if (copy_error != cudaSuccess) {
          return CudaFailure("retain speculative FP8 KV", copy_error);
        }
        if (!layer.global) {
          status = controlled_mtp_d2
                       ? internal::LaunchCopyCircularMtpKvFp8ControlledD2(
                             layer.key_cache_fp8, layer.value_cache_fp8,
                             Pointer<std::uint8_t>(
                                 mtp_workspace_,
                                 mtp_offsets_.layers[mtp_layer_index].backup_key),
                             Pointer<std::uint8_t>(
                                 mtp_workspace_,
                                 mtp_offsets_.layers[mtp_layer_index].backup_value),
                             layer.kv_elements, capacity, false,
                             &Pointer<internal::MtpGroupTransaction>(
                                  mtp_workspace_, mtp_offsets_.transaction)
                                  ->control,
                             stream_)
                       : internal::LaunchCopyCircularMtpKvFp8(
                             layer.key_cache_fp8, layer.value_cache_fp8,
                             Pointer<std::uint8_t>(
                                 mtp_workspace_,
                                 mtp_offsets_.layers[mtp_layer_index].backup_key),
                             Pointer<std::uint8_t>(
                                 mtp_workspace_,
                                 mtp_offsets_.layers[mtp_layer_index].backup_value),
                             start_position, tokens, layer.kv_elements,
                             capacity, false, stream_);
          if (!status.ok()) return status;
        }
        if (layer.global) {
          status = controlled_mtp_d2
                       ? internal::LaunchAppendKvFp8BatchControlled(
                             k_fp8, v_fp8, layer.key_cache_fp8,
                             layer.value_cache_fp8, mtp_row_controls, tokens,
                             layer.kv_elements, capacity, stream_)
                       : internal::LaunchAppendKvFp8Batch(
                             k_fp8, v_fp8, layer.key_cache_fp8,
                             layer.value_cache_fp8, start_position, tokens,
                             layer.kv_elements, capacity, stream_);
          if (!status.ok()) return status;
        }
        float* decode_scores = Pointer<float>(workspace_, offsets_.scores);
        auto* direct_control = Pointer<internal::DecodeControl>(
            workspace_, offsets_.decode_control);
        if (!layer.global && controlled_mtp_d2 &&
            capacity == kSlidingWindow &&
            tokens == 3U) {
          status = internal::LaunchOnlineAttentionDecodeFp8LocalD2ControlledSm120(
              q_norm, layer.key_cache_fp8, layer.value_cache_fp8, k_fp8,
              v_fp8, layer.k_cache_scale, layer.v_cache_scale,
              Pointer<float>(mtp_workspace_, mtp_offsets_.attention_workspace),
              attention, mtp_row_controls, capacity, stream_);
          if (!status.ok()) return status;
          status = internal::LaunchAppendKvFp8BatchControlled(
              k_fp8, v_fp8, layer.key_cache_fp8, layer.value_cache_fp8,
              mtp_row_controls, tokens, layer.kv_elements, capacity, stream_);
          if (!status.ok()) return status;
        } else if (layer.global && capacity > 512U && tokens == 3U) {
          status = controlled_mtp_d2
                       ? internal::LaunchOnlineAttentionDecodeFp8GlobalD2ControlledSm120(
                             q_norm, layer.key_cache_fp8,
                             layer.value_cache_fp8, layer.k_cache_scale,
                             layer.v_cache_scale,
                             Pointer<float>(
                                 mtp_workspace_,
                                 mtp_offsets_.attention_workspace),
                             attention, mtp_row_controls, capacity, stream_)
                       : internal::LaunchOnlineAttentionDecodeFp8GlobalD2Sm120(
                             q_norm, layer.key_cache_fp8,
                             layer.value_cache_fp8, layer.k_cache_scale,
                             layer.v_cache_scale,
                             Pointer<float>(
                                 mtp_workspace_,
                                 mtp_offsets_.attention_workspace),
                             attention, start_position, capacity, stream_);
          if (!status.ok()) return status;
        } else {
          for (std::uint64_t row = 0U; row < tokens; ++row) {
            const std::uint64_t position = start_position + row;
            if (!layer.global) {
              status = controlled_mtp_d2
                           ? internal::LaunchAppendKvFp8BatchControlled(
                                 k_fp8 + row * layer.kv_elements,
                                 v_fp8 + row * layer.kv_elements,
                                 layer.key_cache_fp8, layer.value_cache_fp8,
                                 mtp_row_controls + row, 1U,
                                 layer.kv_elements, capacity, stream_)
                           : internal::LaunchAppendKvFp8Batch(
                                 k_fp8 + row * layer.kv_elements,
                                 v_fp8 + row * layer.kv_elements,
                                 layer.key_cache_fp8, layer.value_cache_fp8,
                                 position, 1U, layer.kv_elements, capacity,
                                 stream_);
              if (!status.ok()) return status;
            }
            const std::uint64_t attention_tokens =
                layer.global ? position + 1U
                             : std::min(position + 1U, capacity);
            const std::uint64_t first_slot =
                layer.global || position + 1U <= capacity
                    ? 0U
                    : (position + 1U) % capacity;
            if (capacity <= 512U) {
              status = internal::LaunchLocalAttentionDecodeFp8(
                  q_norm + row * layer.query_elements, layer.key_cache_fp8,
                  layer.value_cache_fp8, layer.k_cache_scale,
                  layer.v_cache_scale, decode_scores,
                  attention + row * layer.query_elements, kQueryHeads,
                  layer.kv_heads, layer.head_dimension, attention_tokens,
                  stream_, capacity, first_slot);
            } else {
              const internal::DecodeControl* attention_control =
                  direct_control;
              if (controlled_mtp_d2) {
                attention_control = mtp_row_controls + row;
              } else {
                status = internal::LaunchSetMtpAttentionPosition(
                    direct_control, position, stream_);
                if (!status.ok()) return status;
              }
              status = internal::LaunchOnlineAttentionDecodeFp8Sm120(
                  q_norm + row * layer.query_elements, layer.key_cache_fp8,
                  layer.value_cache_fp8, layer.k_cache_scale,
                  layer.v_cache_scale, decode_scores,
                  attention + row * layer.query_elements, attention_control,
                  kQueryHeads, layer.kv_heads, layer.head_dimension, capacity,
                  !layer.global, stream_);
            }
            if (!status.ok()) return status;
          }
        }
        if (!layer.global) {
          status = controlled_mtp_d2
                       ? internal::LaunchCopyCircularMtpKvFp8ControlledD2(
                             layer.key_cache_fp8, layer.value_cache_fp8,
                             Pointer<std::uint8_t>(
                                 mtp_workspace_,
                                 mtp_offsets_.layers[mtp_layer_index].backup_key),
                             Pointer<std::uint8_t>(
                                 mtp_workspace_,
                                 mtp_offsets_.layers[mtp_layer_index].backup_value),
                             layer.kv_elements, capacity, true,
                             &Pointer<internal::MtpGroupTransaction>(
                                  mtp_workspace_, mtp_offsets_.transaction)
                                  ->control,
                             stream_)
                       : internal::LaunchCopyCircularMtpKvFp8(
                             layer.key_cache_fp8, layer.value_cache_fp8,
                             Pointer<std::uint8_t>(
                                 mtp_workspace_,
                                 mtp_offsets_.layers[mtp_layer_index].backup_key),
                             Pointer<std::uint8_t>(
                                 mtp_workspace_,
                                 mtp_offsets_.layers[mtp_layer_index].backup_value),
                             start_position, tokens, layer.kv_elements,
                             capacity, true, stream_);
          if (!status.ok()) return status;
        }
      } else {
        status = layer.global
                     ? internal::LaunchOnlineCausalAttentionPrefillFp8GlobalSm120(
                           q_norm, k_fp8, v_fp8, layer.key_cache_fp8,
                           layer.value_cache_fp8, layer.k_cache_scale,
                           layer.v_cache_scale, attention_bf16, start_position,
                           tokens, kQueryHeads, layer.kv_heads,
                           layer.head_dimension, capacity, stream_)
                     : internal::LaunchOnlineCausalAttentionPrefillFp8LocalSm120(
                           q_norm, k_fp8, v_fp8, layer.key_cache_fp8,
                           layer.value_cache_fp8, layer.k_cache_scale,
                           layer.v_cache_scale, attention_bf16, start_position,
                           tokens, kQueryHeads, layer.kv_heads,
                           layer.head_dimension, capacity, stream_,
                           vision_begin, vision_end);
        if (!status.ok()) return status;
        const std::uint64_t commit_offset =
            layer.global || tokens <= capacity ? 0U : tokens - capacity;
        status = internal::LaunchAppendKvFp8Batch(
            k_fp8 + commit_offset * layer.kv_elements,
            v_fp8 + commit_offset * layer.kv_elements, layer.key_cache_fp8,
            layer.value_cache_fp8, start_position + commit_offset,
            tokens - commit_offset, layer.kv_elements, capacity, stream_);
      }
    } else if (mtp_verification) {
      auto* speculative_key = Pointer<float>(
          mtp_workspace_, mtp_offsets_.layers[mtp_layer_index].key);
      auto* speculative_value = Pointer<float>(
          mtp_workspace_, mtp_offsets_.layers[mtp_layer_index].value);
      const std::size_t bytes = static_cast<std::size_t>(
          tokens * layer.kv_elements * sizeof(float));
      cudaError_t copy_error = cudaMemcpyAsync(
          speculative_key, k_norm, bytes, cudaMemcpyDeviceToDevice, stream_);
      if (copy_error == cudaSuccess) {
        copy_error = cudaMemcpyAsync(speculative_value, v_norm, bytes,
                                     cudaMemcpyDeviceToDevice, stream_);
      }
      if (copy_error != cudaSuccess) {
        return CudaFailure("retain speculative BF16 KV", copy_error);
      }
      if (!layer.global) {
        status = internal::LaunchCopyCircularMtpKvBf16(
            layer.key_cache_bf16, layer.value_cache_bf16,
            Pointer<float>(mtp_workspace_,
                           mtp_offsets_.layers[mtp_layer_index].backup_key),
            Pointer<float>(mtp_workspace_,
                           mtp_offsets_.layers[mtp_layer_index].backup_value),
            start_position, tokens, layer.kv_elements, capacity, false,
            stream_);
        if (!status.ok()) return status;
      }
      if (layer.global) {
        status = internal::LaunchAppendKvBatch(
            k_norm, v_norm, layer.key_cache_bf16, layer.value_cache_bf16,
            start_position, tokens, layer.kv_elements, capacity, stream_);
        if (!status.ok()) return status;
      }
      float* decode_scores = Pointer<float>(workspace_, offsets_.scores);
      for (std::uint64_t row = 0U; row < tokens; ++row) {
        const std::uint64_t position = start_position + row;
        if (!layer.global) {
          status = internal::LaunchAppendKvBatch(
              k_norm + row * layer.kv_elements,
              v_norm + row * layer.kv_elements, layer.key_cache_bf16,
              layer.value_cache_bf16, position, 1U, layer.kv_elements,
              capacity, stream_);
          if (!status.ok()) return status;
        }
        const std::uint64_t attention_tokens =
            layer.global ? position + 1U : std::min(position + 1U, capacity);
        const std::uint64_t first_slot =
            layer.global || position + 1U <= capacity
                ? 0U
                : (position + 1U) % capacity;
        status = internal::LaunchLocalAttentionDecode(
            q_norm + row * layer.query_elements, layer.key_cache_bf16,
            layer.value_cache_bf16, decode_scores,
            attention + row * layer.query_elements, kQueryHeads,
            layer.kv_heads, layer.head_dimension, attention_tokens, stream_,
            capacity, first_slot);
        if (!status.ok()) return status;
      }
      if (!layer.global) {
        status = internal::LaunchCopyCircularMtpKvBf16(
            layer.key_cache_bf16, layer.value_cache_bf16,
            Pointer<float>(mtp_workspace_,
                           mtp_offsets_.layers[mtp_layer_index].backup_key),
            Pointer<float>(mtp_workspace_,
                           mtp_offsets_.layers[mtp_layer_index].backup_value),
            start_position, tokens, layer.kv_elements, capacity, true,
            stream_);
        if (!status.ok()) return status;
      }
    } else {
      float* scores =
          Pointer<float>(prefill_workspace_, prefill_offsets_.scores);
      status = internal::LaunchFusedCausalAttentionPrefill(
          q_norm, k_norm, v_norm, layer.key_cache_bf16,
          layer.value_cache_bf16, scores, attention, start_position, tokens,
          kQueryHeads, layer.kv_heads, layer.head_dimension, capacity,
          !layer.global, stream_);
      if (!status.ok()) return status;
      status = internal::LaunchAppendKvBatch(
          k_norm, v_norm, layer.key_cache_bf16, layer.value_cache_bf16,
          start_position, tokens, layer.kv_elements, capacity, stream_);
    }
    if (!status.ok()) return status;
    return LaunchLayerBatchSuffix(layer, tokens, mtp_verification);
  }

  [[nodiscard]] Status LaunchLayerBatchSuffix(
      const LayerBinding& layer, std::uint64_t tokens,
      bool mtp_verification) {
    const bool physical_hidden =
        !mtp_verification &&
        kv_cache_mode_ == KvCacheMode::kCheckpointFp8;
    auto* hidden_a_bf16 = Pointer<std::uint16_t>(
        prefill_workspace_, prefill_offsets_.hidden_a);
    auto* hidden_b_bf16 = Pointer<std::uint16_t>(
        prefill_workspace_, prefill_offsets_.hidden_b);
    float* hidden_a = reinterpret_cast<float*>(hidden_a_bf16);
    float* hidden_b = reinterpret_cast<float*>(hidden_b_bf16);
    auto* attention_bf16 = Pointer<std::uint16_t>(
        prefill_workspace_, prefill_offsets_.attention);
    float* attention = reinterpret_cast<float*>(attention_bf16);
    auto* projection_bf16 = Pointer<std::uint16_t>(
        prefill_workspace_, prefill_offsets_.projection);
    float* projection = reinterpret_cast<float*>(projection_bf16);
    auto* cutlass_workspace = Pointer<std::uint8_t>(
        prefill_workspace_, prefill_offsets_.cutlass_workspace);
    constexpr std::size_t kCutlassWorkspaceBytes = 8U * 1024U * 1024U;
    auto* o_activation = Pointer<std::uint8_t>(prefill_workspace_, prefill_offsets_.o_activation);
    float* o_scales = Pointer<float>(prefill_workspace_, prefill_offsets_.o_scales);
    Status status;
    if (!mtp_verification &&
        kv_cache_mode_ == KvCacheMode::kCheckpointFp8) {
      status = internal::LaunchFp8ReferenceTokenQuantizationBf16Batch(
          attention_bf16, o_activation, o_scales, tokens,
          layer.query_elements, stream_);
    } else {
      status =
          LaunchRoundBf16(attention, tokens * layer.query_elements, stream_);
      if (!status.ok()) return status;
      status = internal::LaunchFp8ReferenceTokenQuantizationBatch(
          attention, o_activation, o_scales, tokens, layer.query_elements,
          stream_);
    }
    if (!status.ok()) return status;
    status = mtp_verification
                 ? internal::LaunchFp8Sm120DirectProjectionBatch(
                       o_activation, o_scales, layer.o.weight, layer.o.scales,
                       projection, tokens, layer.o.rows, layer.o.contracting,
                       stream_)
                 : LaunchFp8ProjectionBatch(
                       o_activation, o_scales, layer.o, projection_bf16, tokens,
                       cutlass_workspace, kCutlassWorkspaceBytes, stream_);
    if (!status.ok()) return status;
    if (mtp_verification) {
      // The direct verifier O projection also writes FP32; ordinary decode
      // closes this projection at BF16 before the residual boundary.
      status = LaunchRoundBf16(projection, tokens * kHidden, stream_);
      if (!status.ok()) return status;
    }
    status = mtp_verification
                 ? internal::LaunchRmsNormResidualBf16(
                       projection, layer.post_attention_norm, hidden_a,
                       nullptr, hidden_b, tokens, kHidden, kEpsilon, nullptr,
                       stream_)
             : physical_hidden
                 ? internal::LaunchRmsNormResidualPhysicalBf16(
                       projection_bf16, layer.post_attention_norm,
                       hidden_a_bf16, hidden_b_bf16, tokens, kHidden,
                       kEpsilon, nullptr, stream_)
                 : internal::LaunchRmsNormResidualBf16Input(
                       projection_bf16, layer.post_attention_norm, hidden_a,
                       nullptr, hidden_b, tokens, kHidden, kEpsilon, nullptr,
                       stream_);
    if (!status.ok()) return status;

    auto* mlp_packed = Pointer<std::uint8_t>(prefill_workspace_, prefill_offsets_.mlp_packed);
    auto* mlp_scales = Pointer<std::uint8_t>(prefill_workspace_, prefill_offsets_.mlp_scales);
    auto* gate = Pointer<std::uint16_t>(prefill_workspace_, prefill_offsets_.gate);
    auto* down_packed = Pointer<std::uint8_t>(
        prefill_workspace_, prefill_offsets_.down_packed);
    auto* down_scales = Pointer<std::uint8_t>(
        prefill_workspace_, prefill_offsets_.down_scales);
    status =
        physical_hidden
            ? internal::LaunchRmsNormNvfp4ActivationQuantizationBf16Batch(
                  hidden_b_bf16, layer.pre_mlp_norm, mlp_packed, mlp_scales,
                  tokens, kHidden, kEpsilon, layer.gate.input_divisor,
                  stream_)
            : internal::LaunchRmsNormNvfp4ActivationQuantizationBatch(
                  hidden_b, layer.pre_mlp_norm, mlp_packed, mlp_scales,
                  tokens, kHidden, kEpsilon, layer.gate.input_divisor,
                  stream_);
    if (!status.ok()) return status;
    auto* cutlass_activation_scales = Pointer<std::uint8_t>(
        prefill_workspace_, prefill_offsets_.cutlass_activation_scales);
    auto* cutlass_product_scales = Pointer<std::uint8_t>(
        prefill_workspace_, prefill_offsets_.cutlass_product_scales);
    auto* cutlass_weight = Pointer<std::uint8_t>(
        prefill_workspace_, prefill_offsets_.cutlass_weight);
    auto* cutlass_weight_scales = Pointer<std::uint8_t>(
        prefill_workspace_, prefill_offsets_.cutlass_weight_scales);
    float* mtp_product = nullptr;
    if (mtp_verification) {
      // The fused native operator preserves the two BF16 projection
      // boundaries and the BF16 GELU/product boundary. Reuse inactive
      // CUTLASS weight scratch only for this transient batch product.
      mtp_product = reinterpret_cast<float*>(cutlass_weight);
      status = internal::LaunchNvfp4Sm120FusedGateUpBatch(
          mlp_packed, mlp_scales, layer.gate.packed_weight,
          layer.gate.scales, layer.up.packed_weight, layer.up.scales, nullptr,
          nullptr, mtp_product, tokens, layer.gate.rows,
          layer.gate.contracting, layer.gate.input_divisor,
          layer.gate.weight_divisor, layer.up.input_divisor,
          layer.up.weight_divisor, stream_);
    } else {
      status = internal::LaunchNvfp4CutlassInterleaveActivationScales(
          mlp_scales, cutlass_activation_scales, tokens, kHidden, stream_);
      if (!status.ok()) return status;
      status = internal::LaunchNvfp4CutlassProjectionBf16Batch(
          mlp_packed, cutlass_activation_scales, layer.gate.packed_weight,
          layer.gate.scales, cutlass_weight, cutlass_weight_scales,
          cutlass_workspace, kCutlassWorkspaceBytes, gate, tokens,
          layer.gate.rows, layer.gate.contracting, layer.gate.input_divisor,
          layer.gate.weight_divisor, stream_);
      if (!status.ok()) return status;
      status = internal::LaunchNvfp4CutlassUpGatedGeluQuantizedBatch(
          mlp_packed, cutlass_activation_scales, layer.up.packed_weight,
          layer.up.scales, cutlass_weight, cutlass_weight_scales,
          cutlass_workspace, kCutlassWorkspaceBytes, gate, down_packed,
          cutlass_product_scales, tokens,
          layer.up.rows, layer.up.contracting, layer.up.input_divisor,
          layer.up.weight_divisor, layer.down.input_divisor,
          layer.down.input_divisor_device, stream_);
    }
    if (!status.ok()) return status;
    status = mtp_verification
                 ? internal::LaunchNvfp4ReferenceActivationQuantization(
                       mtp_product, down_packed, down_scales,
                       tokens * kIntermediate, layer.down.input_divisor,
                       stream_)
                 : Status::Ok();
    if (!status.ok()) return status;
    auto* down_bf16 = projection_bf16;
    if (mtp_verification) {
      status = internal::LaunchNvfp4Sm120DirectProjectionBf16Batch(
          down_packed, down_scales, layer.down.packed_weight,
          layer.down.scales, down_bf16, tokens, layer.down.rows,
          layer.down.contracting, layer.down.input_divisor,
          layer.down.weight_divisor, stream_);
    } else {
      status = internal::LaunchNvfp4CutlassProjectionBf16Batch(
          down_packed, cutlass_product_scales, layer.down.packed_weight,
          layer.down.scales, cutlass_weight, cutlass_weight_scales,
          cutlass_workspace, kCutlassWorkspaceBytes, down_bf16, tokens,
          layer.down.rows, layer.down.contracting, layer.down.input_divisor,
          layer.down.weight_divisor, stream_);
    }
    if (!status.ok()) return status;
    return physical_hidden
               ? internal::LaunchRmsNormResidualPhysicalBf16(
                     down_bf16, layer.post_mlp_norm, hidden_b_bf16,
                     hidden_a_bf16, tokens, kHidden, kEpsilon,
                     layer.layer_scalar, stream_)
               : internal::LaunchRmsNormResidualBf16Input(
                     down_bf16, layer.post_mlp_norm, hidden_b, nullptr,
                     hidden_a, tokens, kHidden, kEpsilon, layer.layer_scalar,
                     stream_);
  }

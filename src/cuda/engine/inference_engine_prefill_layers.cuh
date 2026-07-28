  [[nodiscard]] Status RunLayerBatch(
      const LayerBinding& layer, std::uint64_t start_position,
      std::uint64_t tokens, std::size_t mtp_layer_index = kLayers,
      const internal::DecodeControl* mtp_row_controls = nullptr) {
    const bool mtp_verification = mtp_layer_index < kLayers;
    const bool controlled_mtp_d2 = mtp_row_controls != nullptr;
    if (controlled_mtp_d2 &&
        (!mtp_verification || tokens != 3U ||
         kv_cache_mode_ != KvCacheMode::kCheckpointFp8)) {
      return Error(StatusCode::kInvalidArgument,
                   "controlled MTP D2 layer geometry is invalid");
    }
    const NvtxRange range(mtp_verification ? "gem16.mtp.verify.layer"
                                           : "gem16.prefill.layer");
    float* hidden_a = Pointer<float>(prefill_workspace_, prefill_offsets_.hidden_a);
    float* hidden_b = Pointer<float>(prefill_workspace_, prefill_offsets_.hidden_b);
    auto* fp8 = Pointer<std::uint8_t>(prefill_workspace_, prefill_offsets_.fp8_activation);
    float* fp8_scales = Pointer<float>(prefill_workspace_, prefill_offsets_.fp8_scales);
    float* q = Pointer<float>(prefill_workspace_, prefill_offsets_.q);
    float* k = Pointer<float>(prefill_workspace_, prefill_offsets_.k);
    float* v = Pointer<float>(prefill_workspace_, prefill_offsets_.v);
    float* q_norm = Pointer<float>(prefill_workspace_, prefill_offsets_.q_norm);
    float* k_norm = Pointer<float>(prefill_workspace_, prefill_offsets_.k_norm);
    float* v_norm = Pointer<float>(prefill_workspace_, prefill_offsets_.v_norm);
    float* attention = Pointer<float>(prefill_workspace_, prefill_offsets_.attention);
    float* projection = Pointer<float>(prefill_workspace_, prefill_offsets_.projection);
    auto* cutlass_workspace = Pointer<std::uint8_t>(
        prefill_workspace_, prefill_offsets_.cutlass_workspace);
    constexpr std::size_t kCutlassWorkspaceBytes = 8U * 1024U * 1024U;
    Status status = internal::LaunchRmsNormFp8TokenQuantizationBatch(
        hidden_a, layer.input_norm, fp8, fp8_scales, tokens, kHidden,
        kEpsilon, stream_);
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
                       fp8, fp8_scales, layer.q, q, layer.k, k,
                       layer.global ? nullptr : &layer.v, v, tokens,
                       cutlass_workspace, kCutlassWorkspaceBytes, stream_);
    if (!status.ok()) return status;
    if (layer.global) {
      const cudaError_t error = cudaMemcpyAsync(
          v, k, static_cast<std::size_t>(tokens * layer.kv_elements * sizeof(float)),
          cudaMemcpyDeviceToDevice, stream_);
      if (error != cudaSuccess) return CudaFailure("reuse batched global K for V", error);
    }
    for (const Status next : {
             LaunchRoundBf16(v, tokens * layer.kv_elements, stream_),
             internal::LaunchRmsNormBf16(v, nullptr, v_norm,
                                     tokens * layer.kv_heads, layer.head_dimension,
                                     kEpsilon, stream_)}) {
      if (!next.ok()) return next;
    }
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
    } else {
      status = internal::LaunchProjectionRmsNormRotaryBf16Batch(
          q, layer.q_norm, q_norm, k, layer.k_norm, k_norm,
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
        if (layer.global && capacity > 512U && tokens == 3U) {
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
                           layer.v_cache_scale, attention, start_position,
                           tokens, kQueryHeads, layer.kv_heads,
                           layer.head_dimension, capacity, stream_)
                     : internal::LaunchOnlineCausalAttentionPrefillFp8LocalSm120(
                           q_norm, k_fp8, v_fp8, layer.key_cache_fp8,
                           layer.value_cache_fp8, layer.k_cache_scale,
                           layer.v_cache_scale, attention, start_position,
                           tokens, kQueryHeads, layer.kv_heads,
                           layer.head_dimension, capacity, stream_);
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
    float* hidden_a =
        Pointer<float>(prefill_workspace_, prefill_offsets_.hidden_a);
    float* hidden_b =
        Pointer<float>(prefill_workspace_, prefill_offsets_.hidden_b);
    float* attention =
        Pointer<float>(prefill_workspace_, prefill_offsets_.attention);
    float* projection =
        Pointer<float>(prefill_workspace_, prefill_offsets_.projection);
    auto* cutlass_workspace = Pointer<std::uint8_t>(
        prefill_workspace_, prefill_offsets_.cutlass_workspace);
    constexpr std::size_t kCutlassWorkspaceBytes = 8U * 1024U * 1024U;
    const std::uint64_t hidden_elements = tokens * kHidden;
    Status status =
        LaunchRoundBf16(attention, tokens * layer.query_elements, stream_);
    if (!status.ok()) return status;
    auto* o_activation = Pointer<std::uint8_t>(prefill_workspace_, prefill_offsets_.o_activation);
    float* o_scales = Pointer<float>(prefill_workspace_, prefill_offsets_.o_scales);
    status = internal::LaunchFp8ReferenceTokenQuantizationBatch(
        attention, o_activation, o_scales, tokens, layer.query_elements, stream_);
    if (!status.ok()) return status;
    status = mtp_verification
                 ? internal::LaunchFp8Sm120DirectProjectionBatch(
                       o_activation, o_scales, layer.o.weight, layer.o.scales,
                       projection, tokens, layer.o.rows, layer.o.contracting,
                       stream_)
                 : LaunchFp8ProjectionBatch(
                       o_activation, o_scales, layer.o, projection, tokens,
                       cutlass_workspace, kCutlassWorkspaceBytes, stream_);
    if (!status.ok()) return status;
    status = LaunchRoundBf16(projection, hidden_elements, stream_);
    if (!status.ok()) return status;
    status = internal::LaunchRmsNormResidualBf16(
        projection, layer.post_attention_norm, hidden_a, nullptr, hidden_b,
        tokens, kHidden, kEpsilon, nullptr, stream_);
    if (!status.ok()) return status;

    auto* mlp_packed = Pointer<std::uint8_t>(prefill_workspace_, prefill_offsets_.mlp_packed);
    auto* mlp_scales = Pointer<std::uint8_t>(prefill_workspace_, prefill_offsets_.mlp_scales);
    auto* gate = Pointer<std::uint16_t>(prefill_workspace_, prefill_offsets_.gate);
    auto* up = Pointer<std::uint16_t>(prefill_workspace_, prefill_offsets_.up);
    status = internal::LaunchRmsNormNvfp4ActivationQuantizationBatch(
        hidden_b, layer.pre_mlp_norm, mlp_packed, mlp_scales, tokens, kHidden,
        kEpsilon, layer.gate.input_divisor, stream_);
    if (!status.ok()) return status;
    auto* cutlass_activation_scales = Pointer<std::uint8_t>(
        prefill_workspace_, prefill_offsets_.cutlass_activation_scales);
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
      status = internal::LaunchNvfp4CutlassProjectionBf16Batch(
          mlp_packed, cutlass_activation_scales, layer.up.packed_weight,
          layer.up.scales, cutlass_weight, cutlass_weight_scales,
          cutlass_workspace, kCutlassWorkspaceBytes, up, tokens,
          layer.up.rows, layer.up.contracting, layer.up.input_divisor,
          layer.up.weight_divisor, stream_);
    }
    if (!status.ok()) return status;
    auto* down_packed = Pointer<std::uint8_t>(prefill_workspace_, prefill_offsets_.down_packed);
    auto* down_scales = Pointer<std::uint8_t>(prefill_workspace_, prefill_offsets_.down_scales);
    status = mtp_verification
                 ? internal::LaunchNvfp4ReferenceActivationQuantization(
                       mtp_product, down_packed, down_scales,
                       tokens * kIntermediate, layer.down.input_divisor,
                       stream_)
                 : internal::LaunchGatedGeluNvfp4ActivationQuantizationBf16(
                       gate, up, down_packed, down_scales,
                       tokens * kIntermediate, layer.down.input_divisor,
                       stream_);
    if (!status.ok()) return status;
    auto* down_bf16 = reinterpret_cast<std::uint16_t*>(projection);
    if (mtp_verification) {
      status = internal::LaunchNvfp4Sm120DirectProjectionBf16Batch(
          down_packed, down_scales, layer.down.packed_weight,
          layer.down.scales, down_bf16, tokens, layer.down.rows,
          layer.down.contracting, layer.down.input_divisor,
          layer.down.weight_divisor, stream_);
    } else {
      status = internal::LaunchNvfp4CutlassInterleaveActivationScales(
          down_scales, cutlass_activation_scales, tokens, kIntermediate,
          stream_);
      if (!status.ok()) return status;
      status = internal::LaunchNvfp4CutlassProjectionBf16Batch(
          down_packed, cutlass_activation_scales, layer.down.packed_weight,
          layer.down.scales, cutlass_weight, cutlass_weight_scales,
          cutlass_workspace, kCutlassWorkspaceBytes, down_bf16, tokens,
          layer.down.rows, layer.down.contracting, layer.down.input_divisor,
          layer.down.weight_divisor, stream_);
    }
    if (!status.ok()) return status;
    return internal::LaunchRmsNormResidualBf16Input(
        down_bf16, layer.post_mlp_norm, hidden_b, nullptr, hidden_a, tokens,
        kHidden, kEpsilon, layer.layer_scalar, stream_);
  }

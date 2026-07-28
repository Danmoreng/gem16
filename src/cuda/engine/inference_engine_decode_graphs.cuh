  template <typename Launch>
  [[nodiscard]] Status CaptureDecodeGraph(GraphExecutable& destination,
                                          Launch&& launch,
                                          const char* label) {
    cudaError_t error =
        cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal);
    if (error != cudaSuccess) return CudaFailure(label, error);
    const Status launch_status = launch();
    cudaGraph_t graph = nullptr;
    error = cudaStreamEndCapture(stream_, &graph);
    if (!launch_status.ok()) {
      if (graph != nullptr) (void)cudaGraphDestroy(graph);
      return launch_status;
    }
    if (error != cudaSuccess) {
      if (graph != nullptr) (void)cudaGraphDestroy(graph);
      return CudaFailure(label, error);
    }
    cudaGraphExec_t executable = nullptr;
    error = cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0U);
    const cudaError_t destroy_error = cudaGraphDestroy(graph);
    if (error != cudaSuccess) return CudaFailure(label, error);
    if (destroy_error != cudaSuccess) {
      (void)cudaGraphExecDestroy(executable);
      return CudaFailure(label, destroy_error);
    }
    destination.Adopt(executable);
    return Status::Ok();
  }

  [[nodiscard]] HostDecodeState* host_decode_state() const {
    return reinterpret_cast<HostDecodeState*>(decode_host_state_.span().data());
  }

  [[nodiscard]] Status LaunchControlledDecodeLayer(
      const LayerBinding& layer) {
    Status status = LaunchDecodePrefix(layer, false);
    if (!status.ok()) return status;
    auto* control = Pointer<internal::DecodeControl>(
        workspace_, offsets_.decode_control);
    float* q_norm = Pointer<float>(workspace_, offsets_.q_norm);
    float* k_norm = Pointer<float>(workspace_, offsets_.k_norm);
    float* v_norm = Pointer<float>(workspace_, offsets_.v_norm);
    float* scores = Pointer<float>(workspace_, offsets_.scores);
    float* attention = Pointer<float>(workspace_, offsets_.attention);
    const float* rotary_cosine =
        layer.global
            ? Pointer<float>(prefill_workspace_,
                             prefill_offsets_.global_rope_cosine)
            : Pointer<float>(prefill_workspace_,
                             prefill_offsets_.local_rope_cosine);
    const float* rotary_sine =
        layer.global
            ? Pointer<float>(prefill_workspace_,
                             prefill_offsets_.global_rope_sine)
            : Pointer<float>(prefill_workspace_,
                             prefill_offsets_.local_rope_sine);
    status = internal::LaunchProjectionRmsNormRotaryBf16Controlled(
        Pointer<float>(workspace_, offsets_.q), layer.q_norm, q_norm,
        Pointer<float>(workspace_, offsets_.k), layer.k_norm, k_norm,
        rotary_cosine, rotary_sine, control, kQueryHeads, layer.kv_heads,
        layer.head_dimension, layer.global ? 0.25 : 1.0, kEpsilon, stream_);
    if (!status.ok()) return status;
    const std::uint64_t capacity =
        layer.global ? max_context_ : std::min(max_context_, kSlidingWindow);
    if (kv_cache_mode_ == KvCacheMode::kCheckpointFp8) {
      status = internal::LaunchAppendKvFp8Controlled(
          k_norm, v_norm, layer.key_cache_fp8, layer.value_cache_fp8,
          layer.k_cache_scale, layer.v_cache_scale, control, layer.kv_heads,
          layer.head_dimension, capacity, !layer.global, stream_);
      if (!status.ok()) return status;
      if (capacity <= 512U) {
        status = internal::LaunchLocalAttentionDecodeFp8Controlled(
            q_norm, layer.key_cache_fp8, layer.value_cache_fp8,
            layer.k_cache_scale, layer.v_cache_scale, scores, attention,
            control, kQueryHeads, layer.kv_heads, layer.head_dimension,
            capacity, !layer.global, stream_);
      } else {
        status = internal::LaunchOnlineAttentionDecodeFp8Sm120(
            q_norm, layer.key_cache_fp8, layer.value_cache_fp8,
            layer.k_cache_scale, layer.v_cache_scale, scores, attention,
            control, kQueryHeads, layer.kv_heads, layer.head_dimension,
            capacity, !layer.global, stream_);
      }
    } else {
      status = internal::LaunchAppendKvControlled(
          k_norm, v_norm, layer.key_cache_bf16, layer.value_cache_bf16,
          control, layer.kv_heads, layer.head_dimension, capacity,
          !layer.global, stream_);
      if (!status.ok()) return status;
      status = internal::LaunchLocalAttentionDecodeControlled(
          q_norm, layer.key_cache_bf16, layer.value_cache_bf16, scores,
          attention, control, kQueryHeads, layer.kv_heads,
          layer.head_dimension, capacity, !layer.global, stream_);
    }
    if (!status.ok()) return status;
    return LaunchDecodeSuffix(layer, nullptr, nullptr);
  }

  [[nodiscard]] Status LaunchFullDecodeGraphBody() {
    HostDecodeState* host = host_decode_state();
    auto* control = Pointer<internal::DecodeControl>(
        workspace_, offsets_.decode_control);
    cudaError_t error = cudaMemcpyAsync(
        control, &host->control, sizeof(host->control), cudaMemcpyHostToDevice,
        stream_);
    if (error != cudaSuccess) {
      return CudaFailure("copy decode graph control", error);
    }
    if (sampling_.enabled && sampling_.repetition_penalty != 1.0F) {
      Status mark_status = internal::LaunchMarkControlledRepetitionToken(
          control,
          Pointer<std::uint32_t>(workspace_, offsets_.repetition_mask),
          stream_);
      if (!mark_status.ok()) return mark_status;
    }
    float* hidden = Pointer<float>(workspace_, offsets_.hidden_a);
    ControlledEmbeddingKernel<<<
        static_cast<unsigned>((kHidden + kThreads - 1U) / kThreads), kThreads,
        0, stream_>>>(model_.embedding(), control, hidden);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return CudaFailure("launch controlled embedding", error);
    }
    for (const LayerBinding& layer : model_.layers()) {
      Status status = LaunchControlledDecodeLayer(layer);
      if (!status.ok()) return status;
    }
    float* normalized = Pointer<float>(workspace_, offsets_.normalized);
    Status status = internal::LaunchRmsNormBf16(
        hidden, model_.final_norm(), normalized, 1U, kHidden, kEpsilon, stream_);
    if (!status.ok()) return status;
    auto* selected = Pointer<std::uint32_t>(workspace_, offsets_.selected);
    float* sampling_logits = sampling_.enabled
                                 ? Pointer<float>(workspace_, offsets_.logits)
                                 : nullptr;
    status = internal::LaunchFusedOutputHeadCandidates(
        model_.embedding(), normalized,
        Pointer<std::uint32_t>(workspace_, offsets_.suppressed), 0U, control,
        Pointer<ArgmaxValue>(workspace_, offsets_.output_candidates),
        sampling_logits, stream_);
    if (!status.ok()) return status;
    if (sampling_.enabled) {
      status = SelectSampledToken(sampling_logits, selected, control);
      if (!status.ok()) return status;
    } else {
      status = internal::LaunchOutputHeadCandidateArgmax(
          Pointer<ArgmaxValue>(workspace_, offsets_.output_candidates),
          selected, stream_);
      if (!status.ok()) return status;
    }
    error = cudaMemcpyAsync(&host->selected_token, selected,
                            sizeof(host->selected_token),
                            cudaMemcpyDeviceToHost, stream_);
    return error == cudaSuccess
               ? Status::Ok()
               : CudaFailure("copy controlled selected token", error);
  }

  [[nodiscard]] Status PrepareDecodeGraphs() {
    cudaError_t error = cudaStreamSynchronize(stream_);
    if (error != cudaSuccess) {
      return CudaFailure("synchronize before decode graph capture", error);
    }
    for (std::size_t index = 0; index < model_.layers().size(); ++index) {
      const LayerBinding& layer = model_.layers()[index];
      Status status = CaptureDecodeGraph(
          decode_prefix_graphs_[index],
          [this, &layer]() { return LaunchDecodePrefix(layer); },
          "capture decode prefix graph");
      if (!status.ok()) return status;
      status = CaptureDecodeGraph(
          decode_suffix_graphs_[index],
          [this, &layer]() {
            return LaunchDecodeSuffix(layer, nullptr, nullptr);
          },
          "capture decode suffix graph");
      if (!status.ok()) return status;
    }
    *host_decode_state() = HostDecodeState{};
    return CaptureDecodeGraph(
        full_decode_graph_, [this]() { return LaunchFullDecodeGraphBody(); },
        "capture full decode graph");
  }

  [[nodiscard]] Status LaunchDecodePrefix(const LayerBinding& layer,
                                          bool normalize_query_key = true) {
    float* hidden_a = Pointer<float>(workspace_, offsets_.hidden_a);
    auto* fp8_activation =
        Pointer<std::uint8_t>(workspace_, offsets_.fp8_activation);
    float* fp8_scale = Pointer<float>(workspace_, offsets_.fp8_scale);
    float* q = Pointer<float>(workspace_, offsets_.q);
    float* k = Pointer<float>(workspace_, offsets_.k);
    float* v = Pointer<float>(workspace_, offsets_.v);
    float* q_norm = Pointer<float>(workspace_, offsets_.q_norm);
    float* k_norm = Pointer<float>(workspace_, offsets_.k_norm);
    float* v_norm = Pointer<float>(workspace_, offsets_.v_norm);

    Status status = internal::LaunchRmsNormFp8TokenQuantizationBatch(
        hidden_a, layer.input_norm, fp8_activation, fp8_scale, 1U, kHidden,
        kEpsilon, stream_);
    if (!status.ok()) return status;
    status = LaunchFp8QkvProjection(
        fp8_activation, fp8_scale, layer.q, q, layer.k, k,
        layer.global ? nullptr : &layer.v, v, stream_);
    if (!status.ok()) return status;
    if (layer.global) {
      const cudaError_t error = cudaMemcpyAsync(
          v, k, layer.kv_elements * sizeof(float), cudaMemcpyDeviceToDevice,
          stream_);
      if (error != cudaSuccess) {
        return CudaFailure("reuse global K projection for V", error);
      }
    }
    status = LaunchRoundBf16(v, layer.kv_elements, stream_);
    if (!status.ok()) return status;
    status = internal::LaunchRmsNormBf16(
        v, nullptr, v_norm, layer.kv_heads, layer.head_dimension, kEpsilon,
        stream_);
    if (!status.ok() || !normalize_query_key) return status;
    for (const Status next : {
             LaunchRoundBf16(q, layer.query_elements, stream_),
             LaunchRoundBf16(k, layer.kv_elements, stream_),
             internal::LaunchRmsNormBf16(q, layer.q_norm, q_norm, kQueryHeads,
                                         layer.head_dimension, kEpsilon,
                                         stream_),
             internal::LaunchRmsNormBf16(k, layer.k_norm, k_norm,
                                         layer.kv_heads, layer.head_dimension,
                                         kEpsilon, stream_),
         }) {
      if (!next.ok()) return next;
    }
    return Status::Ok();
  }

  [[nodiscard]] Status LaunchDecodeSuffix(
      const LayerBinding& layer, const LayerStateCapture* capture,
      float* host_state) {
    float* hidden_a = Pointer<float>(workspace_, offsets_.hidden_a);
    float* hidden_b = Pointer<float>(workspace_, offsets_.hidden_b);
    float* normalized = Pointer<float>(workspace_, offsets_.normalized);
    float* attention = Pointer<float>(workspace_, offsets_.attention);
    auto* o_activation =
        Pointer<std::uint8_t>(workspace_, offsets_.o_activation);
    float* o_scale = Pointer<float>(workspace_, offsets_.o_scale);
    float* projection = Pointer<float>(workspace_, offsets_.projection);
    float* post_norm = Pointer<float>(workspace_, offsets_.post_norm);
    auto* mlp_packed =
        Pointer<std::uint8_t>(workspace_, offsets_.mlp_packed);
    auto* mlp_scales =
        Pointer<std::uint8_t>(workspace_, offsets_.mlp_scales);
    float* gate = Pointer<float>(workspace_, offsets_.gate);
    float* up = Pointer<float>(workspace_, offsets_.up);
    float* product = Pointer<float>(workspace_, offsets_.product);
    auto* down_packed =
        Pointer<std::uint8_t>(workspace_, offsets_.down_packed);
    auto* down_scales =
        Pointer<std::uint8_t>(workspace_, offsets_.down_scales);
    const auto capture_values =
        [this, capture, host_state](std::size_t offset, const float* source,
                                    std::size_t elements,
                                    const char* label) -> Status {
      if (capture == nullptr) return Status::Ok();
      const cudaError_t error = cudaMemcpyAsync(
          host_state + offset, source, elements * sizeof(float),
          cudaMemcpyDeviceToHost, stream_);
      return error == cudaSuccess ? Status::Ok() : CudaFailure(label, error);
    };
    const auto capture_hidden =
        [&capture_values](std::size_t offset, const float* source,
                          const char* label) -> Status {
      return capture_values(offset, source, static_cast<std::size_t>(kHidden),
                            label);
    };

    Status status = LaunchRoundBf16(attention, layer.query_elements, stream_);
    if (!status.ok()) return status;
    if (capture != nullptr) {
      const cudaError_t error = cudaMemcpyAsync(
          host_state + capture->attention_context, attention,
          capture->attention_elements * sizeof(float), cudaMemcpyDeviceToHost,
          stream_);
      if (error != cudaSuccess) {
        return CudaFailure("copy attention context state", error);
      }
    }
    status = internal::LaunchFp8ReferenceTokenQuantization(
        attention, o_activation, o_scale, layer.query_elements, stream_);
    if (!status.ok()) return status;
    status = LaunchFp8Projection(o_activation, o_scale, layer.o, projection, stream_);
    if (!status.ok()) return status;
    status = LaunchRoundBf16(projection, kHidden, stream_);
    if (!status.ok()) return status;
    if (capture != nullptr) {
      status = capture_hidden(capture->attention_output, projection,
                              "copy attention output state");
      if (!status.ok()) return status;
    }
    status = internal::LaunchRmsNormResidualBf16(
        projection, layer.post_attention_norm, hidden_a, post_norm, hidden_b,
        1U, kHidden, kEpsilon, nullptr, stream_);
    if (!status.ok()) return status;
    if (capture != nullptr) {
      status = capture_hidden(capture->post_attention_norm, post_norm,
                              "copy post-attention norm state");
      if (!status.ok()) return status;
      status = capture_hidden(capture->post_attention_residual, hidden_b,
                              "copy post-attention residual state");
      if (!status.ok()) return status;
    }
    if (capture == nullptr) {
      status = internal::LaunchRmsNormNvfp4ActivationQuantizationBatch(
          hidden_b, layer.pre_mlp_norm, mlp_packed, mlp_scales, 1U, kHidden,
          kEpsilon, layer.gate.input_divisor, stream_);
    } else {
      status = internal::LaunchRmsNormBf16(
          hidden_b, layer.pre_mlp_norm, normalized, 1U, kHidden, kEpsilon,
          stream_);
      if (!status.ok()) return status;
      status = capture_hidden(capture->pre_feedforward_norm, normalized,
                              "copy pre-feedforward norm state");
      if (!status.ok()) return status;
      status = internal::LaunchNvfp4ReferenceActivationQuantization(
          normalized, mlp_packed, mlp_scales, kHidden,
          layer.gate.input_divisor, stream_);
    }
    if (!status.ok()) return status;
    status = LaunchNvfp4Projection(mlp_packed, mlp_scales, layer.gate, gate,
                                   stream_);
    if (!status.ok()) return status;
    status = LaunchNvfp4Projection(mlp_packed, mlp_scales, layer.up, up,
                                   stream_);
    if (!status.ok()) return status;
    if (capture == nullptr) {
      status = internal::LaunchGatedGeluNvfp4ActivationQuantization(
          gate, up, down_packed, down_scales, kIntermediate,
          layer.down.input_divisor, stream_);
    } else {
      status = LaunchRoundBf16(gate, kIntermediate, stream_);
      if (!status.ok()) return status;
      status = LaunchRoundBf16(up, kIntermediate, stream_);
      if (!status.ok()) return status;
      status = internal::LaunchGeluTanhProduct(gate, up, product, kIntermediate,
                                               stream_);
      if (!status.ok()) return status;
      status = LaunchRoundBf16(product, kIntermediate, stream_);
      if (!status.ok()) return status;
      status = capture_values(capture->gate, gate, kIntermediate,
                              "copy MLP gate state");
      if (!status.ok()) return status;
      status = capture_values(capture->up, up, kIntermediate,
                              "copy MLP up state");
      if (!status.ok()) return status;
      status = capture_values(capture->gelu_product, product, kIntermediate,
                              "copy MLP GELU product state");
      if (!status.ok()) return status;
      status = internal::LaunchNvfp4ReferenceActivationQuantization(
          product, down_packed, down_scales, kIntermediate,
          layer.down.input_divisor, stream_);
    }
    if (!status.ok()) return status;
    status = LaunchNvfp4Projection(down_packed, down_scales, layer.down,
                                   projection, stream_);
    if (!status.ok()) return status;
    status = LaunchRoundBf16(projection, kHidden, stream_);
    if (!status.ok()) return status;
    if (capture != nullptr) {
      status = capture_hidden(capture->mlp_output, projection,
                              "copy MLP output state");
      if (!status.ok()) return status;
    }
    status = internal::LaunchRmsNormResidualBf16(
        projection, layer.post_mlp_norm, hidden_b, post_norm, hidden_a, 1U,
        kHidden, kEpsilon, layer.layer_scalar, stream_);
    if (!status.ok()) return status;
    if (capture != nullptr) {
      status = capture_hidden(capture->post_feedforward_norm, post_norm,
                              "copy post-feedforward norm state");
      if (!status.ok()) return status;
    }
    if (capture != nullptr) {
      status = capture_hidden(capture->hidden, hidden_a,
                              "copy layer hidden state");
      if (!status.ok()) return status;
    }
    return Status::Ok();
  }

  [[nodiscard]] Status RunLayer(std::size_t layer_index,
                                const LayerBinding& layer, std::uint64_t position,
                                const LayerStateCapture* capture,
                                float* host_state) {
    const NvtxRange range("gem16.decode.layer");
    float* q_norm = Pointer<float>(workspace_, offsets_.q_norm);
    float* k_norm = Pointer<float>(workspace_, offsets_.k_norm);
    float* v_norm = Pointer<float>(workspace_, offsets_.v_norm);
    float* scores = Pointer<float>(workspace_, offsets_.scores);
    float* attention = Pointer<float>(workspace_, offsets_.attention);

    Status status;
    if (capture == nullptr) {
      const cudaError_t error =
          cudaGraphLaunch(decode_prefix_graphs_[layer_index].get(), stream_);
      if (error != cudaSuccess) return CudaFailure("launch decode prefix graph", error);
    } else {
      status = LaunchDecodePrefix(layer);
      if (!status.ok()) return status;
    }
    if (layer.global) {
      status = internal::LaunchProportionalRotaryEmbedding(
          q_norm, kQueryHeads, layer.head_dimension, 0.25, position, 1000000.0, 1.0, stream_);
      if (!status.ok()) return status;
      status = internal::LaunchProportionalRotaryEmbedding(
          k_norm, layer.kv_heads, layer.head_dimension, 0.25, position, 1000000.0, 1.0,
          stream_);
    } else {
      status = internal::LaunchRotaryEmbedding(q_norm, kQueryHeads, layer.head_dimension,
                                               layer.head_dimension, position, 10000.0, stream_);
      if (!status.ok()) return status;
      status = internal::LaunchRotaryEmbedding(k_norm, layer.kv_heads, layer.head_dimension,
                                               layer.head_dimension, position, 10000.0, stream_);
    }
    if (!status.ok()) return status;
    status = LaunchRoundBf16(q_norm, layer.query_elements, stream_);
    if (!status.ok()) return status;
    status = LaunchRoundBf16(k_norm, layer.kv_elements, stream_);
    if (!status.ok()) return status;
    if (capture != nullptr) {
      const std::size_t kv_bytes = capture->kv_elements * sizeof(float);
      cudaError_t capture_error = cudaMemcpyAsync(
          host_state + capture->key, k_norm, kv_bytes, cudaMemcpyDeviceToHost,
          stream_);
      if (capture_error != cudaSuccess) {
        return CudaFailure("copy layer K input state", capture_error);
      }
      capture_error = cudaMemcpyAsync(
          host_state + capture->value, v_norm, kv_bytes,
          cudaMemcpyDeviceToHost, stream_);
      if (capture_error != cudaSuccess) {
        return CudaFailure("copy layer V input state", capture_error);
      }
    }
    const std::uint64_t cache_capacity =
        layer.global ? max_context_ : std::min(max_context_, kSlidingWindow);
    const std::uint64_t cache_slot = layer.global ? position : position % cache_capacity;
    const std::uint64_t attention_tokens =
        layer.global ? position + 1U : std::min(position + 1U, cache_capacity);
    const std::uint64_t first_slot =
        layer.global || position + 1U <= cache_capacity
            ? 0U
            : (position + 1U) % cache_capacity;
    if (kv_cache_mode_ == KvCacheMode::kCheckpointFp8) {
      status = internal::LaunchAppendKvFp8(
          k_norm, v_norm, layer.key_cache_fp8, layer.value_cache_fp8,
          layer.k_cache_scale, layer.v_cache_scale, cache_slot, layer.kv_heads,
          layer.head_dimension, stream_);
      if (!status.ok()) return status;
      status = internal::LaunchLocalAttentionDecodeFp8(
          q_norm, layer.key_cache_fp8, layer.value_cache_fp8,
          layer.k_cache_scale, layer.v_cache_scale, scores, attention,
          kQueryHeads, layer.kv_heads, layer.head_dimension, attention_tokens,
          stream_, cache_capacity, first_slot);
    } else {
      status = internal::LaunchAppendKv(
          k_norm, v_norm, layer.key_cache_bf16, layer.value_cache_bf16,
          cache_slot, layer.kv_heads, layer.head_dimension, stream_);
      if (!status.ok()) return status;
      status = internal::LaunchLocalAttentionDecode(
          q_norm, layer.key_cache_bf16, layer.value_cache_bf16, scores,
          attention, kQueryHeads, layer.kv_heads, layer.head_dimension,
          attention_tokens, stream_, cache_capacity, first_slot);
    }
    if (!status.ok()) return status;
    if (capture == nullptr) {
      const cudaError_t error =
          cudaGraphLaunch(decode_suffix_graphs_[layer_index].get(), stream_);
      return error == cudaSuccess
                 ? Status::Ok()
                 : CudaFailure("launch decode suffix graph", error);
    }
    return LaunchDecodeSuffix(layer, capture, host_state);
  }

  internal::LoadedTargetModel model_;
  internal::AssistantModel assistant_;
  DeviceAllocation cache_;
  DeviceAllocation workspace_;
  DeviceAllocation prefill_workspace_;
  DeviceAllocation mtp_workspace_;
  PinnedHostAllocation decode_host_state_;
  PinnedHostAllocation mtp_host_result_;
  WorkspaceOffsets offsets_{};
  PrefillOffsets prefill_offsets_{};
  MtpWorkspaceOffsets mtp_offsets_{};
  std::array<GraphExecutable, kLayers> decode_prefix_graphs_{};
  std::array<GraphExecutable, kLayers> decode_suffix_graphs_{};
  GraphExecutable full_decode_graph_;
  cudaStream_t stream_ = nullptr;
  std::uint64_t max_context_ = 0;
  std::uint64_t prefill_chunk_tokens_ = kMinimumPrefillChunkTokens;
  std::uint64_t decode_graph_device_bytes_ = 0;
  std::uint64_t assistant_device_memory_delta_bytes_ = 0;
  std::uint32_t mtp_draft_tokens_ = 0U;
  const float* latest_target_hidden_ = nullptr;
  std::uint64_t sampling_step_ = 0;
  std::size_t sampling_sort_workspace_bytes_ = 0;
  std::uint32_t suppressed_token_count_ = 0;
  std::uint32_t mtp_stop_token_count_ = 0;
  KvCacheMode kv_cache_mode_ = KvCacheMode::kCheckpointFp8;
  SamplingOptions sampling_{};
};

Status Gemma4Moe26BReferenceEngine::Reset() {
  if (!implementation_) return Invalid("M13 engine is not initialized");
  cudaError_t error = cudaMemsetAsync(implementation_->kv.As<std::byte>(), 0,
                                      implementation_->kv.bytes(),
                                      implementation_->stream);
  if (error != cudaSuccess) return CudaFailure("clear M13 K/V arena", error);
  error = cudaMemsetAsync(implementation_->workspace.As<std::byte>(), 0,
                          implementation_->workspace.bytes(),
                          implementation_->stream);
  if (error != cudaSuccess) return CudaFailure("clear M13 workspace", error);
  if (implementation_->prefill_workspace.bytes() != 0U) {
    error = cudaMemsetAsync(
        implementation_->prefill_workspace.As<std::byte>(), 0,
        implementation_->prefill_workspace.bytes(), implementation_->stream);
    if (error != cudaSuccess) {
      return CudaFailure("clear M17 prefill workspace", error);
    }
  }
  if (implementation_->trellis35_workspace.bytes() != 0U) {
    error = cudaMemsetAsync(
        implementation_->trellis35_workspace.As<std::byte>(), 0,
        implementation_->trellis35_workspace.bytes(),
        implementation_->stream);
    if (error != cudaSuccess) {
      return CudaFailure("clear Trellis35 fixed workspace", error);
    }
  }
  if (implementation_->mtp_verifier.bytes() != 0U) {
    error = cudaMemsetAsync(implementation_->mtp_verifier.As<std::byte>(), 0,
                            implementation_->mtp_verifier.bytes(),
                            implementation_->stream);
    if (error != cudaSuccess) {
      return CudaFailure("clear M25 verifier workspace", error);
    }
  }
  error = cudaStreamSynchronize(implementation_->stream);
  if (error != cudaSuccess) return CudaFailure("synchronize M13 reset", error);
  implementation_->position = 0U;
  implementation_->prefill_chunks = 0U;
  implementation_->minimum_prefill_chunk = 0U;
  implementation_->prefill_calls = 0U;
  implementation_->decode_graph_launches = 0U;
  implementation_->token_selections = 0U;
  implementation_->sliding_ring_wraps = 0U;
  implementation_->maximum_global_position_exclusive = 0U;
  implementation_->fallback_count = 0U;
  implementation_->recurring_allocation_count = 0U;
  implementation_->mtp_group_graph_launches = 0U;
  implementation_->mtp_chain_graph_launches.fill(0U);
  implementation_->last_mtp_verification_min_margin = 0.0F;
  implementation_->sampling_step = 0U;
  // Graph executables remain instantiated across Reset(), so their observed
  // cudaMemGetInfo deltas and captured stop-token count intentionally remain
  // part of the configured runtime. ConversationSession recaptures the graphs
  // when its sampling/stop-token configuration changes.
  implementation_->pending_decode_self_feed = false;
  implementation_->decode_self_feed_valid = false;
  implementation_->has_last_input_token = false;
  implementation_->last_input_token = 0U;
  implementation_->decode_self_feed_token = 0U;
  implementation_->decode_self_feed_position = 0U;
  implementation_->calibration_capture_valid = false;
  return Status::Ok();
}

Status Gemma4Moe26BReferenceEngine::ForwardToken(std::uint32_t token) {
  if (!implementation_) return Invalid("M13 engine is not initialized");
  auto& x = *implementation_;
  x.calibration_capture_valid = false;
  if (token >= kVocabulary || x.position >= x.context) {
    return Invalid("M13 token or position exceeds the fixed contract");
  }
  if (x.sampling.enabled && x.sampling.repetition_penalty != 1.0F) {
    Status marked = LaunchMarkRepetitionToken(
        token, x.repetition_mask, x.stream);
    if (!marked.ok()) return marked;
  }
  if (x.backend == Gemma4Moe26BBackend::kSm120Integrated) {
    if (x.decode_graph == nullptr) {
      return Invalid("M17 decode graph is not initialized");
    }
    const bool use_device_self_feed =
        x.decode_self_feed_valid && x.decode_self_feed_token == token &&
        x.decode_self_feed_position == x.position;
    x.decode_self_feed_valid = false;
    cudaError_t graph_error = cudaSuccess;
    if (!use_device_self_feed) {
      SetDecodeControlKernel<<<1, 1, 0, x.stream>>>(x.decode_control, token,
                                                    x.position);
      graph_error = cudaGetLastError();
    }
    if (graph_error == cudaSuccess) {
      graph_error = cudaGraphLaunch(x.decode_graph, x.stream);
    }
    if (graph_error != cudaSuccess) {
      return CudaFailure("launch M17 decode graph", graph_error);
    }
    const std::uint64_t previous_position = x.position;
    ++x.position;
    ++x.decode_graph_launches;
    x.pending_decode_self_feed = true;
    if (x.sliding_capacity != 0U && previous_position != 0U &&
        previous_position % x.sliding_capacity == 0U) {
      ++x.sliding_ring_wraps;
    }
    x.maximum_global_position_exclusive =
        std::max(x.maximum_global_position_exclusive, x.position);
    x.has_last_input_token = true;
    x.last_input_token = token;
    return Status::Ok();
  }
  constexpr unsigned threads = 256U;
  cudaError_t error = cudaMemsetAsync(x.routing_finite, 1, sizeof(int),
                                      x.stream);
  if (error != cudaSuccess) {
    return CudaFailure("initialize M13 router finite flag", error);
  }
  TiledEmbeddingLookupKernel<<<static_cast<unsigned>((kWidth + threads - 1U) /
                                                       threads),
                               threads, 0, x.stream>>>(
      x.head.packed_e2m1, x.head.scales_e4m3fn,
      x.head.weight_global_divisor, token, x.hidden_a);
  error = cudaGetLastError();
  if (error != cudaSuccess) return CudaFailure("launch M13 embedding", error);

  for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
    Status status = LaunchGemma4Moe26BAttentionReferenceLayer(
        x.hidden_a, x.hidden_b, x.position, x.traits[layer],
        x.attention_weights[layer], x.caches[layer], x.attention_workspace,
        1.0e-6F, x.stream);
    if (!status.ok()) return status;
    status = x.backend != Gemma4Moe26BBackend::kReference
                 ? LaunchGemma4MoeSm120Layer(
                       x.hidden_b, x.hidden_a, x.moe_config,
                       x.moe_weights[layer], x.moe_workspace, x.stream)
                 : LaunchGemma4MoeReferenceLayer(
                       x.hidden_b, x.hidden_a, x.moe_config,
                       x.moe_weights[layer], x.moe_workspace, x.stream);
    if (!status.ok()) return status;
    if (x.calibration_capture_enabled && layer == x.calibration_capture_layer) {
      error = cudaMemcpyAsync(
          x.calibration_gate_up_input.data(), x.moe_workspace.expert_input,
          kWidth * sizeof(float), cudaMemcpyDeviceToHost, x.stream);
      if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            x.calibration_down_inputs.data(), x.moe_workspace.expert_product,
            kTopK * kExpert * sizeof(float), cudaMemcpyDeviceToHost, x.stream);
      }
      if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            x.calibration_expert_ids.data(), x.moe_workspace.top_ids,
            kTopK * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, x.stream);
      }
      if (error != cudaSuccess) {
        return CudaFailure("capture Trellis35 calibration inputs", error);
      }
      x.calibration_capture_valid = true;
    }
    const int capture = CaptureIndex(layer, x.quality_capture_all_layers);
    if (capture >= 0) {
      error = cudaMemcpyAsync(x.layer_captures[static_cast<std::size_t>(capture)],
                              x.hidden_a, kWidth * sizeof(float),
                              cudaMemcpyDeviceToDevice, x.stream);
      if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            x.router_probability_captures[static_cast<std::size_t>(capture)],
            x.moe_workspace.router_probabilities, kExperts * sizeof(float),
            cudaMemcpyDeviceToDevice, x.stream);
      }
      if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            x.router_id_captures[static_cast<std::size_t>(capture)],
            x.moe_workspace.top_ids, kTopK * sizeof(std::uint32_t),
            cudaMemcpyDeviceToDevice, x.stream);
      }
      if (error != cudaSuccess) return CudaFailure("capture M13 layer", error);
    }
  }
  Status status = LaunchRmsNormBf16(x.hidden_a, x.final_norm, x.final_hidden,
                                    1U, kWidth, 1.0e-6F, x.stream);
  if (!status.ok()) return status;
  status = LaunchNvfp4ReferenceActivationQuantization(
      x.final_hidden, x.head_activation, x.head_activation_scales, kWidth,
      x.head.activation_global_divisor, x.stream);
  if (!status.ok()) return status;
  status = x.backend != Gemma4Moe26BBackend::kReference
               ? LaunchNvfp4Sm120DirectProjectionBf16Float(
                     x.head_activation, x.head_activation_scales,
                     x.head.packed_e2m1, x.head.scales_e4m3fn, x.logits,
                     x.head.rows, x.head.columns,
                     x.head.activation_global_divisor,
                     x.head.weight_global_divisor, x.stream)
               : LaunchGemma4MoeTiledNvfp4ReferenceProjection(
                     x.head, x.head_activation, x.head_activation_scales,
                     x.logits, x.stream);
  if (!status.ok()) return status;
  status = LaunchSoftcapArgmax(
      x.logits, x.softcap, x.finite, x.output_candidates,
      x.prediction_token, x.prediction_logit, x.stream);
  if (!status.ok()) return status;
  const std::uint64_t previous_position = x.position;
  ++x.position;
  if (x.sliding_capacity != 0U && previous_position != 0U &&
      previous_position % x.sliding_capacity == 0U) {
    ++x.sliding_ring_wraps;
  }
  x.maximum_global_position_exclusive =
      std::max(x.maximum_global_position_exclusive, x.position);
  x.has_last_input_token = true;
  x.last_input_token = token;
  return Status::Ok();
}

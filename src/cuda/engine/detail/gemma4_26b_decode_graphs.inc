Status Gemma4Moe26BReferenceEngine::Impl::LaunchControlledDecodeBody() {
  if (decode_control == nullptr) {
    return Invalid("M17 decode graph control is not initialized");
  }
  cudaError_t error = cudaMemsetAsync(routing_finite, 1, sizeof(int), stream);
  if (error != cudaSuccess) {
    return CudaFailure("initialize M17 router finite flag", error);
  }
  constexpr unsigned threads = 256U;
  TiledEmbeddingLookupControlledKernel<<<
      static_cast<unsigned>((kWidth + threads - 1U) / threads), threads, 0,
      stream>>>(head.packed_e2m1, head.scales_e4m3fn,
                head.weight_global_divisor, decode_control, hidden_a);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("launch M17 controlled embedding", error);
  }
  const Gemma4Moe26BAttentionLayerTraits* local_trait = nullptr;
  const Gemma4Moe26BAttentionLayerTraits* global_trait = nullptr;
  for (const auto& trait : traits) {
    const bool sliding =
        trait.attention == Gemma4Moe26BAttentionType::kSliding;
    const Gemma4Moe26BAttentionLayerTraits*& profile =
        sliding ? local_trait : global_trait;
    if (profile == nullptr) {
      profile = &trait;
    } else if (profile->head_dimension != trait.head_dimension ||
               profile->rotary_factor != trait.rotary_factor ||
               profile->rope_theta != trait.rope_theta ||
               profile->rope_scaling_factor != trait.rope_scaling_factor) {
      return Invalid("M17 attention classes do not share RoPE profiles");
    }
  }
  if (local_trait == nullptr || global_trait == nullptr) {
    return Invalid("M17 requires local and global RoPE profiles");
  }
  auto prepare_rotary = [&](const Gemma4Moe26BAttentionLayerTraits& trait,
                            float* cosine, float* sine) {
    const std::uint64_t rotating_pairs = static_cast<std::uint64_t>(
        trait.rotary_factor *
        static_cast<double>(trait.head_dimension / 2U));
    return LaunchRotaryEmbeddingTableControlled(
        cosine, sine, rotating_pairs, trait.head_dimension, decode_control,
        trait.rope_theta, trait.rope_scaling_factor, stream);
  };
  Status status = prepare_rotary(*global_trait,
                                 attention_workspace.rotary_cosine,
                                 attention_workspace.rotary_sine);
  if (!status.ok()) return status;
  status = prepare_rotary(*local_trait, local_rotary_cosine,
                          local_rotary_sine);
  if (!status.ok()) return status;
  for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
    auto layer_workspace = attention_workspace;
    if (traits[layer].attention == Gemma4Moe26BAttentionType::kSliding) {
      layer_workspace.rotary_cosine = local_rotary_cosine;
      layer_workspace.rotary_sine = local_rotary_sine;
    }
    status = LaunchGemma4Moe26BAttentionReferenceControlledLayer(
        hidden_a, hidden_b, traits[layer], attention_weights[layer],
        caches[layer], layer_workspace, decode_control, 1.0e-6F, stream,
        true);
    if (!status.ok()) return status;
    status = IsTrellis35RoutedExpertFormat(routed_expert_format)
                 ? LaunchGemma4MoeSm120Trellis35Layer(
                       hidden_b, hidden_a, moe_config, moe_weights[layer],
                       trellis35_artifact.layers()[layer],
                       trellis35_m1_workspace, moe_workspace, stream,
                       shared_moe_stream, shared_moe_fork, shared_moe_join)
                 : LaunchGemma4MoeSm120Layer(
                       hidden_b, hidden_a, moe_config, moe_weights[layer],
                       moe_workspace, stream, shared_moe_stream,
                       shared_moe_fork, shared_moe_join);
    if (!status.ok()) return status;
    const int capture = CaptureIndex(layer, quality_capture_all_layers);
    if (capture >= 0) {
      const std::size_t index = static_cast<std::size_t>(capture);
      error = cudaMemcpyAsync(layer_captures[index], hidden_a,
                              kWidth * sizeof(float),
                              cudaMemcpyDeviceToDevice, stream);
      if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            router_probability_captures[index],
            moe_workspace.router_probabilities, kExperts * sizeof(float),
            cudaMemcpyDeviceToDevice, stream);
      }
      if (error == cudaSuccess) {
        error = cudaMemcpyAsync(router_id_captures[index],
                                moe_workspace.top_ids,
                                kTopK * sizeof(std::uint32_t),
                                cudaMemcpyDeviceToDevice, stream);
      }
      if (error != cudaSuccess) {
        return CudaFailure("capture M17 decode layer", error);
      }
    }
  }
  status = LaunchRmsNormNvfp4ActivationQuantizationBatch(
      hidden_a, final_norm, head_activation, head_activation_scales, 1U,
      kWidth, 1.0e-6F, head.activation_global_divisor, stream);
  if (!status.ok()) return status;
  status = LaunchNvfp4Sm120DirectProjectionBf16Float(
      head_activation, head_activation_scales, head.packed_e2m1,
      head.scales_e4m3fn, logits, head.rows, head.columns,
      head.activation_global_divisor, head.weight_global_divisor, stream);
  if (!status.ok()) return status;
  return LaunchSoftcapArgmax(logits, softcap, finite, output_candidates,
                             prediction_token, prediction_logit, stream,
                             decode_control);
}

Status Gemma4Moe26BReferenceEngine::Impl::PrepareDecodeGraph() {
  cudaError_t error = cudaStreamSynchronize(stream);
  if (error != cudaSuccess) {
    return CudaFailure("synchronize before M17 graph capture", error);
  }
  error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  if (error != cudaSuccess) return CudaFailure("begin M17 graph capture", error);
  const Status body = LaunchControlledDecodeBody();
  cudaGraph_t graph = nullptr;
  error = cudaStreamEndCapture(stream, &graph);
  if (!body.ok()) {
    if (graph != nullptr) (void)cudaGraphDestroy(graph);
    return body;
  }
  if (error != cudaSuccess) {
    if (graph != nullptr) (void)cudaGraphDestroy(graph);
    return CudaFailure("end M17 graph capture", error);
  }
  error = cudaGraphInstantiate(&decode_graph, graph, nullptr, nullptr, 0U);
  const cudaError_t destroy_error = cudaGraphDestroy(graph);
  if (error != cudaSuccess) return CudaFailure("instantiate M17 graph", error);
  if (destroy_error != cudaSuccess) {
    (void)cudaGraphExecDestroy(decode_graph);
    decode_graph = nullptr;
    return CudaFailure("destroy captured M17 graph", destroy_error);
  }
  return Status::Ok();
}

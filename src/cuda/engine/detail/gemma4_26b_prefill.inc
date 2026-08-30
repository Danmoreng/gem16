Status Gemma4Moe26BReferenceEngine::PrefillTokens(
    std::span<const std::uint32_t> tokens) {
  if (!implementation_ || tokens.empty() ||
      implementation_->backend != Gemma4Moe26BBackend::kSm120Integrated ||
      implementation_->prefill_workspace.bytes() == 0U ||
      tokens.size() > implementation_->context - implementation_->position) {
    return Invalid("M17 prefill request exceeds the initialized contract");
  }
  for (const std::uint32_t token : tokens) {
    if (token >= kVocabulary) return Invalid("M17 prefill token is invalid");
  }
  auto& x = *implementation_;
  static const bool trellis35_routing_diagnostic = [] {
    const char* value = std::getenv("GEM16_TRELLIS35_ROUTING_TELEMETRY");
    return value != nullptr && std::string_view(value) == "1";
  }();
  x.pending_decode_self_feed = false;
  x.decode_self_feed_valid = false;
  constexpr unsigned threads = 256U;
  std::size_t consumed = 0U;
  std::uint64_t chunk_index = 0U;
  cudaError_t error =
      cudaMemsetAsync(x.routing_finite, 1, sizeof(int), x.stream);
  if (error != cudaSuccess) {
    return CudaFailure("initialize M17 prefill router finite flag", error);
  }
  const Gemma4Moe26BAttentionLayerTraits* local_trait = nullptr;
  const Gemma4Moe26BAttentionLayerTraits* global_trait = nullptr;
  for (const auto& trait : x.traits) {
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
      return Invalid("M17 prefill attention classes do not share RoPE profiles");
    }
  }
  if (local_trait == nullptr || global_trait == nullptr ||
      x.prefill_local_rotary_cosine == nullptr ||
      x.prefill_local_rotary_sine == nullptr) {
    return Invalid("M17 prefill RoPE profiles are not initialized");
  }
  while (consumed < tokens.size()) {
    const std::uint64_t absolute_chunk_begin = x.position;
    // The 2048-token WP21 candidate is qualified for the prepared global
    // attention extent used by the complete Wikipedia-16K workload. Preserve
    // the established 1024-token long-context path once that fixed prepared
    // extent is exhausted.
    const std::uint64_t prepared_remaining =
        absolute_chunk_begin < kPreparedGlobalPrefillTokens
            ? kPreparedGlobalPrefillTokens - absolute_chunk_begin
            : 0U;
    const std::uint64_t chunk_limit =
        prepared_remaining == 0U
            ? kDefaultPrefillChunkTokens
            : std::min(x.prefill_chunk_tokens, prepared_remaining);
    const std::uint64_t chunk = std::min<std::uint64_t>(
        chunk_limit, tokens.size() - consumed);
    const bool is_last_chunk =
        consumed + static_cast<std::size_t>(chunk) == tokens.size();
    error = cudaStreamSynchronize(x.stream);
    if (error != cudaSuccess) {
      return CudaFailure("synchronize M17 prefill token staging", error);
    }
    std::copy_n(tokens.data() + consumed, static_cast<std::size_t>(chunk),
                x.prefill_host_tokens);
    error = cudaMemcpyAsync(
        x.prefill_tokens, x.prefill_host_tokens,
        chunk * sizeof(std::uint32_t), cudaMemcpyHostToDevice, x.stream);
    if (error == cudaSuccess && x.sampling.enabled &&
        x.sampling.repetition_penalty != 1.0F) {
      Status marked = LaunchMarkRepetitionTokens(
          x.prefill_tokens, chunk, x.repetition_mask, x.stream);
      if (!marked.ok()) return marked;
    }
    if (error != cudaSuccess) return CudaFailure("copy M17 prefill tokens", error);
    const std::uint64_t hidden_elements = chunk * kWidth;
    TiledEmbeddingLookupBatchKernel<<<
        static_cast<unsigned>((hidden_elements + threads - 1U) / threads),
        threads, 0, x.stream>>>(
        x.head.packed_e2m1, x.head.scales_e4m3fn,
        x.head.weight_global_divisor, x.prefill_tokens, x.prefill_hidden_a,
        hidden_elements);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return CudaFailure("launch M17 prefill embedding", error);
    }
    auto prepare_rotary = [&](const Gemma4Moe26BAttentionLayerTraits& trait,
                              float* cosine, float* sine) {
      const std::uint64_t rotating_pairs = static_cast<std::uint64_t>(
          trait.rotary_factor *
          static_cast<double>(trait.head_dimension / 2U));
      return LaunchRotaryEmbeddingTableBatch(
          cosine, sine, chunk, rotating_pairs, trait.head_dimension,
          x.position, trait.rope_theta, trait.rope_scaling_factor, x.stream);
    };
    Status status = prepare_rotary(
        *global_trait, x.prefill_attention_workspace.rotary_cosine,
        x.prefill_attention_workspace.rotary_sine);
    if (!status.ok()) return status;
    status = prepare_rotary(*local_trait, x.prefill_local_rotary_cosine,
                            x.prefill_local_rotary_sine);
    if (!status.ok()) return status;
    for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
      auto layer_workspace = x.prefill_attention_workspace;
      if (x.traits[layer].attention ==
          Gemma4Moe26BAttentionType::kSliding) {
        layer_workspace.rotary_cosine = x.prefill_local_rotary_cosine;
        layer_workspace.rotary_sine = x.prefill_local_rotary_sine;
      }
      status = LaunchGemma4Moe26BAttentionSm120PrefillLayer(
          x.prefill_hidden_a, x.prefill_hidden_b, x.position, chunk,
          x.traits[layer], x.attention_weights[layer], x.caches[layer],
          layer_workspace, 1.0e-6F, x.stream, true);
      if (!status.ok()) return status;
      status = IsTrellis35RoutedExpertFormat(x.routed_expert_format)
                   ? LaunchGemma4MoeSm120Trellis35PrefillLayer(
                         x.prefill_hidden_b, x.prefill_hidden_a, chunk,
                         x.moe_config, x.moe_weights[layer],
                         x.trellis35_artifact.layers()[layer],
                         x.prefill_moe_workspace, x.stream)
                   : LaunchGemma4MoeSm120PrefillLayer(
                         x.prefill_hidden_b, x.prefill_hidden_a, chunk,
                         x.moe_config, x.moe_weights[layer],
                         x.prefill_moe_workspace, x.stream);
      if (!status.ok()) return status;
      if (trellis35_routing_diagnostic &&
          IsTrellis35RoutedExpertFormat(x.routed_expert_format)) {
        std::array<std::uint32_t, kExperts + 1U> host_prefix{};
        std::array<Trellis35ExpertDescriptor, kExperts> host_gate{};
        std::array<Trellis35ExpertDescriptor, kExperts> host_down{};
        error = cudaMemcpyAsync(
            host_prefix.data(), x.prefill_moe_workspace.prefix,
            host_prefix.size() * sizeof(host_prefix[0]),
            cudaMemcpyDeviceToHost, x.stream);
        if (error == cudaSuccess) {
          error = cudaMemcpyAsync(
              host_gate.data(),
              x.trellis35_artifact.layers()[layer].gate_up.descriptors,
              host_gate.size() * sizeof(host_gate[0]), cudaMemcpyDeviceToHost,
              x.stream);
        }
        if (error == cudaSuccess) {
          error = cudaMemcpyAsync(
              host_down.data(),
              x.trellis35_artifact.layers()[layer].down.descriptors,
              host_down.size() * sizeof(host_down[0]), cudaMemcpyDeviceToHost,
              x.stream);
        }
        if (error == cudaSuccess) error = cudaStreamSynchronize(x.stream);
        if (error != cudaSuccess) {
          return CudaFailure("copy WP17 Trellis35 routing telemetry", error);
        }
        std::uint64_t active_experts = 0U;
        std::uint64_t m32_schedules = 0U;
        std::uint64_t m64_schedules = 0U;
        std::uint64_t m64_full_tiles = 0U;
        std::uint64_t m64_33_63_tails = 0U;
        std::uint64_t m32_le32_tails = 0U;
        std::uint64_t gate_k3 = 0U;
        std::uint64_t gate_k4 = 0U;
        std::uint64_t down_k3 = 0U;
        std::uint64_t down_k4 = 0U;
        for (unsigned expert = 0U; expert < kExperts; ++expert) {
          const std::uint32_t rows =
              host_prefix[expert + 1U] - host_prefix[expert];
          if (rows == 0U) continue;
          ++active_experts;
          m32_schedules += (rows + 31U) / 32U;
          m64_full_tiles += rows / 64U;
          const std::uint32_t remainder = rows % 64U;
          m64_schedules +=
              rows / 64U + (remainder > 32U ? 1U : 0U);
          m64_33_63_tails += remainder > 32U ? 1U : 0U;
          m32_le32_tails +=
              remainder != 0U && remainder <= 32U ? 1U : 0U;
          gate_k3 += host_gate[expert].rate_bits == 3U ? 1U : 0U;
          gate_k4 += host_gate[expert].rate_bits == 4U ? 1U : 0U;
          down_k3 += host_down[expert].rate_bits == 3U ? 1U : 0U;
          down_k4 += host_down[expert].rate_bits == 4U ? 1U : 0U;
        }
        std::cerr << "TRELLIS35_WP17_ROUTING {\"chunk\":"
                  << chunk_index << ",\"layer\":" << layer
                  << ",\"tokens\":" << chunk
                  << ",\"assignment_count\":" << chunk * kTopK
                  << ",\"active_experts\":" << active_experts
                  << ",\"m32_schedule_count\":" << m32_schedules
                  << ",\"m64_schedule_count\":" << m64_schedules
                  << ",\"m64_full_tiles\":" << m64_full_tiles
                  << ",\"m64_33_63_tails\":" << m64_33_63_tails
                  << ",\"m32_le32_tails\":" << m32_le32_tails
                  << ",\"gate_up_k3_active\":" << gate_k3
                  << ",\"gate_up_k4_active\":" << gate_k4
                  << ",\"down_k3_active\":" << down_k3
                  << ",\"down_k4_active\":" << down_k4
                  << ",\"rows_per_expert\":[";
        for (unsigned expert = 0U; expert < kExperts; ++expert) {
          if (expert != 0U) std::cerr << ',';
          std::cerr << host_prefix[expert + 1U] - host_prefix[expert];
        }
        std::cerr << "]}\n";
      }
      const int capture = CaptureIndex(layer, x.quality_capture_all_layers);
      if (is_last_chunk && capture >= 0) {
        const std::size_t capture_index = static_cast<std::size_t>(capture);
        error = cudaMemcpyAsync(
            x.layer_captures[capture_index],
            x.prefill_hidden_a + (chunk - 1U) * kWidth,
            kWidth * sizeof(float), cudaMemcpyDeviceToDevice, x.stream);
        if (error == cudaSuccess) {
          error = cudaMemcpyAsync(
              x.router_probability_captures[capture_index],
              x.prefill_moe_workspace.router_probabilities +
                  (chunk - 1U) * kExperts,
              kExperts * sizeof(float), cudaMemcpyDeviceToDevice, x.stream);
        }
        if (error == cudaSuccess) {
          CapturePrefillRouterIdsKernel<<<1, kTopK, 0, x.stream>>>(
              x.prefill_moe_workspace.assignments,
              x.router_id_captures[capture_index], chunk - 1U);
          error = cudaGetLastError();
        }
        if (error != cudaSuccess) {
          return CudaFailure("capture M17 prefill layer", error);
        }
      }
    }
    if (is_last_chunk) {
      float* last_hidden = x.prefill_hidden_a + (chunk - 1U) * kWidth;
      Status status = LaunchRmsNormBf16(
          last_hidden, x.final_norm, x.final_hidden, 1U, kWidth, 1.0e-6F,
          x.stream);
      if (!status.ok()) return status;
      status = LaunchNvfp4ReferenceActivationQuantization(
          x.final_hidden, x.head_activation, x.head_activation_scales, kWidth,
          x.head.activation_global_divisor, x.stream);
      if (!status.ok()) return status;
      status = LaunchNvfp4Sm120DirectProjectionBf16Float(
          x.head_activation, x.head_activation_scales, x.head.packed_e2m1,
          x.head.scales_e4m3fn, x.logits, x.head.rows, x.head.columns,
          x.head.activation_global_divisor, x.head.weight_global_divisor,
          x.stream);
      if (!status.ok()) return status;
      status = LaunchSoftcapArgmax(
          x.logits, x.softcap, x.finite, x.output_candidates,
          x.prediction_token, x.prediction_logit, x.stream);
      if (!status.ok()) return status;
    }
    const std::uint64_t previous_position = x.position;
    x.position += chunk;
    if (x.sliding_capacity != 0U) {
      x.sliding_ring_wraps += x.position / x.sliding_capacity -
                                previous_position / x.sliding_capacity;
    }
    x.maximum_global_position_exclusive =
        std::max(x.maximum_global_position_exclusive, x.position);
    ++x.prefill_chunks;
    x.minimum_prefill_chunk =
        x.minimum_prefill_chunk == 0U
            ? chunk
            : std::min(x.minimum_prefill_chunk, chunk);
    consumed += static_cast<std::size_t>(chunk);
    ++chunk_index;
  }
  ++x.prefill_calls;
  x.has_last_input_token = true;
  x.last_input_token = tokens.back();
  return Status::Ok();
}

Result<Gemma4Moe26BReferencePrediction>
Gemma4Moe26BReferenceEngine::Prediction() {
  if (!implementation_) return Invalid("M13 engine is not initialized");
  auto& x = *implementation_;
  cudaError_t error = cudaMemcpyAsync(
      x.prediction_host_status, x.prediction_device_status,
      sizeof(PredictionStatus), cudaMemcpyDeviceToHost, x.stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(x.stream);
  if (error != cudaSuccess) {
    return CudaFailure("copy compact M20 prediction status", error);
  }
  const PredictionStatus status = *x.prediction_host_status;
  Gemma4Moe26BReferencePrediction result{status.token, status.logit,
                                         status.finite != 0};
  if (status.routing_finite == 0) {
    return Status(StatusCode::kInternal,
                  "Gemma 4 26B router produced a non-finite value");
  }
  if (!result.all_logits_finite) {
    return Status(StatusCode::kInternal,
                  "Gemma 4 26B output logits are non-finite");
  }
  if (x.pending_decode_self_feed) {
    x.pending_decode_self_feed = false;
    x.decode_self_feed_valid = true;
    x.decode_self_feed_token = status.token;
    x.decode_self_feed_position = x.position;
  }
  return result;
}

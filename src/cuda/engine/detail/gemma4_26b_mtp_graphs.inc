Status Gemma4Moe26BReferenceEngine::Impl::SelectMtpVerificationTokens(
    float* batch_logits, const std::uint32_t* verification_inputs,
    DecodeControl* row_controls, std::uint32_t tokens,
    std::uint32_t* selected, bool apply_softcap, int* logits_finite) {
  if (!sampling.enabled) return Status::Ok();
  if (batch_logits == nullptr || verification_inputs == nullptr ||
      row_controls == nullptr || tokens == 0U ||
      tokens > kM25MaximumVerifyTokens || selected == nullptr) {
    return Invalid("M25 sampled verifier buffers are invalid");
  }
  if (apply_softcap && logits_finite == nullptr) {
    return Invalid("M25 sampled verifier finite-state buffer is missing");
  }
  auto* row_masks = mtp_verifier.As<std::uint32_t>(
      mtp_sampling_repetition_masks_offset);
  if (sampling.repetition_penalty != 1.0F) {
    Status status = LaunchBuildSpeculativeRepetitionMasks(
        repetition_mask, verification_inputs, tokens, kRepetitionMaskWords,
        row_masks, stream);
    if (!status.ok()) return status;
  }
  for (std::uint32_t row = 0U; row < tokens; ++row) {
    std::uint32_t* mask = sampling.repetition_penalty == 1.0F
                              ? repetition_mask
                              : row_masks +
                                    static_cast<std::uint64_t>(row) *
                                        kRepetitionMaskWords;
    float* row_logits =
        batch_logits + static_cast<std::uint64_t>(row) * kVocabulary;
    Status status = apply_softcap
                        ? LaunchSampleTokenSoftcapInPlace(
                              row_logits, logits, sampling_logits,
                              sampling_cumulative, sampling_token_ids,
                              sampling_sorted_token_ids, mask,
                              suppressed_token_ids, suppressed_token_count,
                              static_cast<std::uint32_t>(kVocabulary), softcap,
                              logits_finite + row, sampling, 0U,
                              row_controls + row, selected + row,
                              sampling_algorithm_workspace,
                              sampling_algorithm_workspace_bytes, stream)
                        : LaunchSampleTokenFromLogits(
                              row_logits, logits, sampling_logits,
                              sampling_cumulative, sampling_token_ids,
                              sampling_sorted_token_ids, mask,
                              suppressed_token_ids, suppressed_token_count,
                              static_cast<std::uint32_t>(kVocabulary), sampling,
                              0U, row_controls + row, selected + row,
                              sampling_algorithm_workspace,
                              sampling_algorithm_workspace_bytes, stream);
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

Status Gemma4Moe26BReferenceEngine::Impl::LaunchFixedMtpGraphBody(
    std::uint32_t draft_count, bool copy_transaction,
    const std::uint32_t* forced_drafts) {
  const std::uint32_t tokens = draft_count + 1U;
  if ((draft_count != 1U && draft_count != 2U && draft_count != 4U) ||
      mtp_verifier_backend !=
          Gemma4Moe26BMtpVerifierBackend::kExactSharedBatchedMoe ||
      !assistant.prepared() || mtp_verifier.bytes() == 0U ||
      mtp_verifier_host == nullptr) {
    return Invalid("M25 fixed graph body is unavailable for this D/backend");
  }

  auto* transaction =
      mtp_verifier.As<MtpGroupTransaction>(mtp_transaction_offset);
  auto* control = &transaction->control;
  constexpr std::uint32_t kSlidingLayer = 28U;
  constexpr std::uint32_t kFullLayer = 29U;
  const auto make_view = [this](std::uint32_t layer) {
    AssistantSharedKvView view;
    const bool global =
        traits[layer].attention == Gemma4Moe26BAttentionType::kFull;
    view.mode = AssistantKvCacheMode::kCheckpointFp8;
    view.key_fp8 = caches[layer].key;
    view.value_fp8 = caches[layer].value;
    view.key_scale_bf16 = attention_weights[layer].key_cache_scale_bf16;
    view.value_scale_bf16 = attention_weights[layer].value_cache_scale_bf16;
    view.capacity = caches[layer].capacity;
    // Controlled online attention derives the visible prefix/ring slot from
    // MtpDeviceControl. A non-zero fixed extent keeps capture-time validation
    // independent from the first replay position.
    view.tokens = view.capacity;
    view.first_slot = 0U;
    view.kv_heads = traits[layer].kv_heads;
    view.head_dimension = traits[layer].head_dimension;
    return view;
  };
  if (traits[kSlidingLayer].attention !=
          Gemma4Moe26BAttentionType::kSliding ||
      traits[kFullLayer].attention != Gemma4Moe26BAttentionType::kFull) {
    return Status(StatusCode::kInternal,
                  "M25 fixed graph shared-KV layer mapping is invalid");
  }
  Status status;
  const std::uint32_t* drafts = forced_drafts;
  if (drafts == nullptr) {
    Gemma4Moe26BAssistantProposalContext proposal;
    proposal.target_embedding = head;
    proposal.target_hidden = final_hidden;
    proposal.sliding_kv = make_view(kSlidingLayer);
    proposal.full_kv = make_view(kFullLayer);
    {
      const NvtxRange range("gem16.mtp.assistant_proposal");
      status = assistant.GenerateDraftsDevice(
          proposal, draft_count, stream, control);
    }
    if (!status.ok()) return status;
    drafts = assistant.device_draft_tokens();
  }
  if (drafts == nullptr) {
    return Status(StatusCode::kInternal,
                  "M25 fixed graph Assistant drafts are unavailable");
  }

  auto* row_controls =
      mtp_verifier.As<DecodeControl>(mtp_row_controls_offset);
  status = LaunchBuildControlledMtpInputs(
      control, drafts, draft_count, prefill_tokens, row_controls,
      suppressed_token_count, stream);
  if (!status.ok()) return status;
  cudaError_t error = cudaMemsetAsync(routing_finite, 1, sizeof(int), stream);
  if (error != cudaSuccess) {
    return CudaFailure("initialize M25 fixed graph router finite flag", error);
  }
  constexpr unsigned threads = 256U;
  const std::uint64_t hidden_elements = tokens * kWidth;
  TiledEmbeddingLookupBatchKernel<<<
      static_cast<unsigned>((hidden_elements + threads - 1U) / threads),
      threads, 0, stream>>>(
      head.packed_e2m1, head.scales_e4m3fn, head.weight_global_divisor,
      prefill_tokens, prefill_hidden_a, hidden_elements);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("launch M25 fixed graph embedding", error);
  }

  for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
    const auto& trait = traits[layer];
    const std::uint64_t kv_elements =
        trait.kv_heads * trait.head_dimension;
    const auto offsets = mtp_layer_offsets[layer];
    auto* backup_key = mtp_verifier.As<std::uint8_t>(offsets.backup_key);
    auto* backup_value =
        mtp_verifier.As<std::uint8_t>(offsets.backup_value);
    auto* compact_key = mtp_verifier.As<std::uint8_t>(offsets.compact_key);
    auto* compact_value =
        mtp_verifier.As<std::uint8_t>(offsets.compact_value);
    // D2 is the selected fixed-depth candidate. Its three consecutive Target
    // rows share the qualified fixed-row attention body instead of replaying
    // one online-attention launch sequence per row. Keep D1/D4 on their
    // established parent until those non-selected depths are qualified
    // independently.
    const bool shared_fixed_attention = tokens == 3U;
    {
      const NvtxRange range("gem16.mtp.target_attention");
      status = LaunchGemma4Moe26BAttentionSm120MtpFixedLayer(
          prefill_hidden_a, prefill_hidden_b, 0U, trait,
          attention_weights[layer], caches[layer],
          prefill_attention_workspace, prefill_moe_workspace.shared_product,
          row_controls, tokens, shared_fixed_attention, true, true, 1.0e-6F,
          stream, backup_key, backup_value);
    }
    if (!status.ok()) return status;
    auto* overlap = tokens != 3U || mtp_router_overlap.bytes() == 0U
                        ? nullptr
                        : mtp_router_overlap.As<MtpRouterOverlapCounters>();
    if (IsTrellis35RoutedExpertFormat(routed_expert_format) &&
        tokens != kTrellis35T3Rows) {
      return Status(StatusCode::kUnsupported,
                    "Trellis35 fixed verifier supports only D2/T3");
    }
    {
      const NvtxRange range("gem16.mtp.target_moe");
      status = IsTrellis35RoutedExpertFormat(routed_expert_format)
                   ? LaunchGemma4MoeSm120Trellis35T3Layer(
                         prefill_hidden_b, prefill_hidden_a, tokens,
                         moe_config, moe_weights[layer],
                         trellis35_artifact.layers()[layer],
                         trellis35_t3_workspace, prefill_moe_workspace, stream,
                         overlap, shared_moe_stream, shared_moe_fork,
                         shared_moe_join)
                   : LaunchGemma4MoeSm120MtpSharedBatchLayer(
                         prefill_hidden_b, prefill_hidden_a, tokens,
                         moe_config, moe_weights[layer],
                         prefill_moe_workspace, moe_workspace, stream,
                         overlap);
    }
    if (!status.ok()) return status;
    if (mtp_v11_layer_outputs_enabled && tokens == kM25FixedD2Rows) {
      const std::uint64_t capture_offset =
          static_cast<std::uint64_t>(layer) * tokens * kWidth;
      error = cudaMemcpyAsync(
          mtp_verifier.As<float>(mtp_v11_layer_outputs_offset) +
              capture_offset,
          prefill_hidden_a, tokens * kWidth * sizeof(float),
          cudaMemcpyDeviceToDevice, stream);
      if (error != cudaSuccess) {
        return CudaFailure("capture V11 T3 layer output", error);
      }
    }
    {
      const NvtxRange range("gem16.mtp.restore");
      status = LaunchCompactRestoreCircularMtpKvFp8Controlled(
          caches[layer].key, caches[layer].value, compact_key, compact_value,
          backup_key, backup_value, tokens, kv_elements,
          caches[layer].capacity, control, stream);
    }
    if (!status.ok()) return status;
  }

  float* normalized = mtp_verifier.As<float>(mtp_normalized_offset);
  auto* head_activation =
      mtp_verifier.As<std::uint8_t>(mtp_head_activation_offset);
  auto* head_scales =
      mtp_verifier.As<std::uint8_t>(mtp_head_scales_offset);
  float* batch_logits = mtp_verifier.As<float>(mtp_logits_offset);
  auto* selected =
      mtp_verifier.As<std::uint32_t>(mtp_selected_offset);
  auto* second_selected =
      mtp_verifier.As<std::uint32_t>(mtp_second_selected_offset);
  float* top_values = mtp_verifier.As<float>(mtp_top_values_offset);
  float* second_values = mtp_verifier.As<float>(mtp_second_values_offset);
  int* logits_finite = mtp_verifier.As<int>(mtp_finite_offset);
  {
    const NvtxRange range("gem16.mtp.target_head");
    status = LaunchRmsNormBf16Nvfp4ActivationQuantizationBatch(
        prefill_hidden_a, final_norm, normalized, head_activation, head_scales,
        tokens, kWidth, 1.0e-6F, head.activation_global_divisor, stream);
    if (!status.ok()) return status;
    status = LaunchNvfp4Sm120DirectProjectionBf16FloatExactBatch(
        head_activation, head_scales, head.packed_e2m1, head.scales_e4m3fn,
        batch_logits, tokens, head.rows, head.columns,
        head.activation_global_divisor, head.weight_global_divisor, stream);
  }
  if (!status.ok()) return status;
  {
    const NvtxRange selection_range("gem16.mtp.selection_sampling");
    if (!sampling.enabled || copy_transaction) {
      for (std::uint64_t row = 0U; row < tokens; ++row) {
        status = LaunchSoftcapArgmax(
            batch_logits + row * kVocabulary, softcap, logits_finite + row,
            output_candidates, selected + row, top_values + row, stream,
            nullptr, suppressed_token_ids, suppressed_token_count);
        if (!status.ok()) return status;
        if (copy_transaction) {
          status = LaunchSoftcapArgmax(
              batch_logits + row * kVocabulary, softcap,
              logits_finite + row, output_candidates,
              second_selected + row, second_values + row, stream, nullptr,
              suppressed_token_ids, suppressed_token_count, selected + row,
              false);
          if (!status.ok()) return status;
        }
      }
    }
    status = SelectMtpVerificationTokens(
        batch_logits, prefill_tokens, row_controls, tokens, selected,
        sampling.enabled && !copy_transaction, logits_finite);
    if (!status.ok()) return status;
    status = LaunchAcceptMtpGroup(
        drafts, selected, draft_count,
        mtp_verifier.As<std::uint32_t>(mtp_stop_tokens_offset),
        mtp_stop_token_count, &transaction->result, control, stream);
    if (!status.ok()) return status;
    if (sampling.enabled && sampling.repetition_penalty != 1.0F) {
      status = LaunchCommitSpeculativeRepetitionMask(
          mtp_verifier.As<std::uint32_t>(
              mtp_sampling_repetition_masks_offset),
          kRepetitionMaskWords, &transaction->result.output_count,
          repetition_mask, stream);
      if (!status.ok()) return status;
    }
  }
  const NvtxRange commit_range("gem16.mtp.commit");
  std::array<MtpKvCommitLayer, kLayers> commit_layers{};
  for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
    const auto& trait = traits[layer];
    const auto offsets = mtp_layer_offsets[layer];
    commit_layers[layer] = {
        mtp_verifier.As<std::uint8_t>(offsets.compact_key),
        mtp_verifier.As<std::uint8_t>(offsets.compact_value),
        caches[layer].key, caches[layer].value,
        trait.kv_heads * trait.head_dimension, caches[layer].capacity};
  }
  status = LaunchCommitMtpKvFp8ControlledLayers(
      commit_layers.data(), static_cast<std::uint32_t>(commit_layers.size()),
      &transaction->result, control, stream);
  if (!status.ok()) return status;
  status = LaunchCommitMtpHidden(normalized, final_hidden, kWidth,
                                 &transaction->result, stream);
  if (!status.ok()) return status;
  if (copy_transaction) {
    CommitMtpPredictionKernel<true>
        <<<kArgmaxBlocks, kArgmaxThreads, 0, stream>>>(
            batch_logits, logits, &transaction->result, logits_finite,
            routing_finite, prediction_device_status);
  } else {
    CommitMtpPredictionKernel<false><<<1U, 1U, 0, stream>>>(
        batch_logits, logits, &transaction->result, logits_finite,
        routing_finite, prediction_device_status);
  }
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("commit M25 fixed graph prediction", error);
  }

  if (!copy_transaction) {
    LatchMtpGroupNonFiniteKernel<<<1U, 1U, 0, stream>>>(
        logits_finite, tokens, routing_finite,
        mtp_verifier.As<MtpChainResult>(mtp_chain_result_offset));
    error = cudaGetLastError();
    return error == cudaSuccess
               ? Status::Ok()
               : CudaFailure("latch M25 chain verifier finite state", error);
  }

  error = cudaMemcpyAsync(&mtp_verifier_host->transaction, transaction,
                          sizeof(MtpGroupTransaction),
                          cudaMemcpyDeviceToHost, stream);
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(mtp_verifier_host->logits_finite.data(),
                            logits_finite, tokens * sizeof(int),
                            cudaMemcpyDeviceToHost, stream);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(mtp_verifier_host->top_values.data(), top_values,
                            tokens * sizeof(float), cudaMemcpyDeviceToHost,
                            stream);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(mtp_verifier_host->second_values.data(),
                            second_values, tokens * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(&mtp_verifier_host->routing_finite,
                            routing_finite, sizeof(int),
                            cudaMemcpyDeviceToHost, stream);
  }
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("copy M25 fixed graph transaction", error);
}

Status Gemma4Moe26BReferenceEngine::Impl::PrepareFixedMtpGraph(
    std::uint32_t draft_count) {
  if (draft_count == 0U || draft_count > kMaximumMtpDraftTokens) {
    return Invalid("M25 fixed graph draft count is invalid");
  }
  if (mtp_group_graphs[draft_count] != nullptr) return Status::Ok();
  if (draft_count != 1U && draft_count != 2U && draft_count != 4U) {
    return Status(StatusCode::kUnsupported,
                  "M25 fixed graph kernel body is not implemented for this D");
  }
  cudaError_t error = cudaStreamSynchronize(stream);
  if (error != cudaSuccess) {
    return CudaFailure("synchronize before M25 fixed graph capture", error);
  }
  std::size_t free_before = 0U;
  std::size_t total_bytes = 0U;
  error = cudaMemGetInfo(&free_before, &total_bytes);
  if (error != cudaSuccess) {
    return CudaFailure("measure before M25 fixed graph capture", error);
  }
  error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  if (error != cudaSuccess) {
    return CudaFailure("begin M25 fixed graph capture", error);
  }
  const Status body = LaunchFixedMtpGraphBody(draft_count, true);
  cudaGraph_t graph = nullptr;
  error = cudaStreamEndCapture(stream, &graph);
  if (!body.ok()) {
    if (graph != nullptr) (void)cudaGraphDestroy(graph);
    return body;
  }
  if (error != cudaSuccess) {
    if (graph != nullptr) (void)cudaGraphDestroy(graph);
    return CudaFailure("end M25 fixed graph capture", error);
  }
  cudaGraphExec_t executable = nullptr;
  error = cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0U);
  const cudaError_t destroy_error = cudaGraphDestroy(graph);
  if (error != cudaSuccess) {
    return CudaFailure("instantiate M25 fixed graph", error);
  }
  if (destroy_error != cudaSuccess) {
    (void)cudaGraphExecDestroy(executable);
    return CudaFailure("destroy M25 fixed graph source", destroy_error);
  }
  std::size_t free_after = 0U;
  error = cudaMemGetInfo(&free_after, &total_bytes);
  if (error != cudaSuccess) {
    (void)cudaGraphExecDestroy(executable);
    return CudaFailure("measure after M25 fixed graph capture", error);
  }
  const std::uint64_t required_margin =
      context >= 64000U ? 200U * kMiB : 700U * kMiB;
  if (free_after < required_margin) {
    (void)cudaGraphExecDestroy(executable);
    return Status(StatusCode::kResourceExhausted,
                  "M25 fixed graph leaves less than the profile margin");
  }
  mtp_group_graphs[draft_count] = executable;
  if (free_before > free_after) {
    mtp_group_graph_device_bytes += free_before - free_after;
  }
  return Status::Ok();
}

Status Gemma4Moe26BReferenceEngine::Impl::ExecuteFixedMtpGraphGroup(
    std::uint32_t pending_token, std::uint32_t draft_count,
    MtpGroupResult* host_result) {
  if (draft_count == 0U || draft_count > kMaximumMtpDraftTokens ||
      host_result == nullptr || mtp_group_graphs[draft_count] == nullptr) {
    return Invalid("M25 fixed graph execution state is invalid");
  }
  *mtp_verifier_host = {};
  auto& host_control = mtp_verifier_host->transaction.control;
  host_control.current.input_token = pending_token;
  host_control.current.processed_position = position - 1U;
  host_control.current.remaining_output_capacity = context - position;
  host_control.current.output_write_position = 0U;
  host_control.current.sampling_step = sampling_step;
  host_control.next = host_control.current;
  host_control.fixed_draft_tokens = draft_count;
  host_control.sampling_enabled = sampling.enabled ? 1U : 0U;
  auto* transaction =
      mtp_verifier.As<MtpGroupTransaction>(mtp_transaction_offset);
  cudaError_t error = cudaMemcpyAsync(
      transaction, &mtp_verifier_host->transaction,
      sizeof(MtpGroupTransaction), cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess) {
    return CudaFailure("initialize M25 fixed graph transaction", error);
  }
  error = cudaGraphLaunch(mtp_group_graphs[draft_count], stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
  if (error != cudaSuccess) {
    return CudaFailure("execute M25 fixed graph group", error);
  }
  ++mtp_group_graph_launches;

  return CompleteFixedMtpGroup(pending_token, draft_count, host_result);
}

Status Gemma4Moe26BReferenceEngine::Impl::CompleteFixedMtpGroup(
    std::uint32_t pending_token, std::uint32_t draft_count,
    MtpGroupResult* host_result) {
  if (host_result == nullptr || draft_count == 0U ||
      draft_count > kMaximumMtpDraftTokens) {
    return Invalid("M25 fixed group completion state is invalid");
  }
  const std::uint64_t tokens = draft_count + 1U;

  const auto& completed = mtp_verifier_host->transaction;
  if (completed.control.current.input_token != pending_token ||
      completed.control.current.processed_position + 1U != position ||
      completed.control.fixed_draft_tokens != draft_count ||
      completed.control.transition_valid != 1U ||
      completed.control.proposal_count != draft_count ||
      completed.result.output_count == 0U ||
      completed.result.output_count > tokens ||
      completed.control.next.input_token !=
          completed.result.verified[completed.result.output_count - 1U] ||
      completed.control.next.processed_position !=
          completed.control.current.processed_position +
              completed.result.output_count) {
    return Status(StatusCode::kInternal,
                  "M25 fixed graph device transition is inconsistent");
  }
  if (mtp_verifier_host->routing_finite == 0 ||
      std::any_of(mtp_verifier_host->logits_finite.begin(),
                  mtp_verifier_host->logits_finite.begin() + tokens,
                  [](int value) { return value == 0; })) {
    return Status(StatusCode::kInternal,
                  "M25 fixed graph produced non-finite Target values");
  }
  last_mtp_verification_min_margin =
      std::numeric_limits<float>::infinity();
  for (std::uint64_t row = 0U; row < tokens; ++row) {
    last_mtp_verification_min_margin = std::min(
        last_mtp_verification_min_margin,
        mtp_verifier_host->top_values[row] -
            mtp_verifier_host->second_values[row]);
  }

  const std::uint64_t previous_position = position;
  position += completed.result.output_count;
  if (sliding_capacity != 0U) {
    sliding_ring_wraps += position / sliding_capacity -
                          previous_position / sliding_capacity;
  }
  maximum_global_position_exclusive =
      std::max(maximum_global_position_exclusive, position);
  token_selections += completed.result.output_count;
  sampling_step = completed.control.next.sampling_step;
  pending_decode_self_feed = false;
  decode_self_feed_valid = false;
  has_last_input_token = true;
  last_input_token = completed.result.output_count == 1U
                         ? pending_token
                         : completed.result.proposed[
                               completed.result.output_count - 2U];
  *host_result = completed.result;
  return Status::Ok();
}

Status Gemma4Moe26BReferenceEngine::Impl::LaunchFixedMtpOrdinaryTailBody() {
  auto* transaction =
      mtp_verifier.As<MtpGroupTransaction>(mtp_transaction_offset);
  Status status = LaunchInitializeMtpOrdinaryTail(
      transaction, decode_control, suppressed_token_count, stream);
  if (!status.ok()) return status;
  if (sampling.enabled && sampling.repetition_penalty != 1.0F) {
    status = LaunchMarkControlledRepetitionToken(
        decode_control, repetition_mask, stream);
    if (!status.ok()) return status;
  }
  status = LaunchControlledDecodeBody();
  if (!status.ok()) return status;
  // The ordinary graph normally keeps only the fused final activation/head.
  // MTP needs the normalized Target hidden row for the next Assistant input.
  status = LaunchRmsNormBf16(hidden_a, final_norm, final_hidden, 1U, kWidth,
                             1.0e-6F, stream);
  if (!status.ok()) return status;
  LatchMtpOrdinaryNonFiniteKernel<<<1U, 1U, 0, stream>>>(
      prediction_device_status,
      mtp_verifier.As<MtpChainResult>(mtp_chain_result_offset));
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("latch M25 chain ordinary finite state", error);
  }
  const std::uint32_t* selected = prediction_token;
  if (sampling.enabled) {
    status = LaunchSampleToken(
        logits, sampling_logits, sampling_cumulative, sampling_token_ids,
        sampling_sorted_token_ids, repetition_mask, suppressed_token_ids,
        suppressed_token_count, static_cast<std::uint32_t>(kVocabulary),
        sampling, 0U, decode_control, selected_token,
        sampling_algorithm_workspace, sampling_algorithm_workspace_bytes,
        stream);
    if (!status.ok()) return status;
    selected = selected_token;
  }
  return LaunchFinalizeMtpOrdinaryTail(
      selected, mtp_verifier.As<std::uint32_t>(mtp_stop_tokens_offset),
      mtp_stop_token_count, transaction,
      mtp_verifier.As<MtpChainResult>(mtp_chain_result_offset),
      mtp_verifier.As<std::uint32_t>(mtp_chain_outputs_offset),
      mtp_streaming_ring_device, stream);
}

Status Gemma4Moe26BReferenceEngine::Impl::PrepareFixedMtpChainGraph(
    std::uint32_t draft_count) {
  if (draft_count == 0U || draft_count > kMaximumMtpDraftTokens) {
    return Invalid("M25 fixed chain graph draft count is invalid");
  }
  if (mtp_chain_graphs[draft_count] != nullptr) return Status::Ok();
  if ((draft_count != 1U && draft_count != 2U && draft_count != 4U) ||
      mtp_group_graphs[draft_count] == nullptr ||
      mtp_verifier_backend !=
          Gemma4Moe26BMtpVerifierBackend::kExactSharedBatchedMoe) {
    return Status(StatusCode::kUnsupported,
                  "M25 fixed chain graph is unavailable for this D/backend");
  }
  cudaError_t error = cudaStreamSynchronize(stream);
  if (error != cudaSuccess) {
    return CudaFailure("synchronize before M25 chain graph capture", error);
  }
  std::size_t free_before = 0U;
  std::size_t total_bytes = 0U;
  error = cudaMemGetInfo(&free_before, &total_bytes);
  if (error != cudaSuccess) {
    return CudaFailure("measure before M25 chain graph capture", error);
  }

  cudaGraph_t root = nullptr;
  cudaGraph_t route_source = nullptr;
  cudaGraph_t d2_source = nullptr;
  cudaGraph_t ordinary_source = nullptr;
  cudaGraph_t continue_source = nullptr;
  cudaGraphExec_t executable = nullptr;
  const auto cleanup = [&]() {
    if (route_source != nullptr) (void)cudaGraphDestroy(route_source);
    if (d2_source != nullptr) (void)cudaGraphDestroy(d2_source);
    if (ordinary_source != nullptr) (void)cudaGraphDestroy(ordinary_source);
    if (continue_source != nullptr) (void)cudaGraphDestroy(continue_source);
    if (root != nullptr) (void)cudaGraphDestroy(root);
    if (executable != nullptr) (void)cudaGraphExecDestroy(executable);
  };
  error = cudaGraphCreate(&root, 0U);
  if (error != cudaSuccess) {
    return CudaFailure("create M25 fixed chain graph", error);
  }
  cudaGraphConditionalHandle loop_condition = 0U;
  error = cudaGraphConditionalHandleCreate(
      &loop_condition, root, 1U, cudaGraphCondAssignDefault);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("create M25 chain loop condition", error);
  }
  cudaGraphConditionalHandle d2_condition = 0U;
  error = cudaGraphConditionalHandleCreate(
      &d2_condition, root, 0U, cudaGraphCondAssignDefault);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("create M25 chain D2 condition", error);
  }
  cudaGraphConditionalHandle ordinary_condition = 0U;
  error = cudaGraphConditionalHandleCreate(
      &ordinary_condition, root, 0U, cudaGraphCondAssignDefault);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("create M25 chain ordinary condition", error);
  }
  cudaGraphNodeParams loop_parameters{};
  loop_parameters.type = cudaGraphNodeTypeConditional;
  loop_parameters.conditional.handle = loop_condition;
  loop_parameters.conditional.type = cudaGraphCondTypeWhile;
  loop_parameters.conditional.size = 1U;
  cudaGraphNode_t loop_node = nullptr;
  error = cudaGraphAddNode(&loop_node, root, nullptr, nullptr, 0U,
                           &loop_parameters);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("add M25 chain loop", error);
  }

  auto* transaction =
      mtp_verifier.As<MtpGroupTransaction>(mtp_transaction_offset);
  auto* chain_result =
      mtp_verifier.As<MtpChainResult>(mtp_chain_result_offset);
  auto* chain_outputs =
      mtp_verifier.As<std::uint32_t>(mtp_chain_outputs_offset);
  auto* chain_proposals =
      mtp_verifier.As<std::uint32_t>(mtp_chain_proposals_offset);

  error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("begin M25 chain route capture", error);
  }
  Status status = LaunchSelectMtpChainBranch(
      transaction, d2_condition, ordinary_condition, stream);
  error = cudaStreamEndCapture(stream, &route_source);
  if (!status.ok() || error != cudaSuccess) {
    cleanup();
    return !status.ok() ? status
                        : CudaFailure("end M25 chain route capture", error);
  }

  error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("begin M25 chain D2 capture", error);
  }
  status = LaunchFixedMtpGraphBody(draft_count, false);
  if (status.ok()) {
    status = LaunchAdvanceMtpChain(
        transaction, chain_result, chain_outputs, chain_proposals,
        mtp_streaming_ring_device, stream);
  }
  error = cudaStreamEndCapture(stream, &d2_source);
  if (!status.ok() || error != cudaSuccess) {
    cleanup();
    return !status.ok() ? status
                        : CudaFailure("end M25 chain D2 capture", error);
  }

  error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("begin M25 chain ordinary capture", error);
  }
  status = LaunchFixedMtpOrdinaryTailBody();
  error = cudaStreamEndCapture(stream, &ordinary_source);
  if (!status.ok() || error != cudaSuccess) {
    cleanup();
    return !status.ok()
               ? status
               : CudaFailure("end M25 chain ordinary capture", error);
  }

  error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("begin M25 chain continuation capture", error);
  }
  status = LaunchContinueMtpChain(
      transaction, mtp_streaming_ring_device, loop_condition, stream);
  error = cudaStreamEndCapture(stream, &continue_source);
  if (!status.ok() || error != cudaSuccess) {
    cleanup();
    return !status.ok()
               ? status
               : CudaFailure("end M25 chain continuation capture", error);
  }

  cudaGraph_t loop_body = loop_parameters.conditional.phGraph_out[0];
  cudaGraphNode_t route_node = nullptr;
  error = cudaGraphAddChildGraphNode(&route_node, loop_body, nullptr, 0U,
                                     route_source);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("add M25 chain route", error);
  }
  cudaGraphNodeParams d2_parameters{};
  d2_parameters.type = cudaGraphNodeTypeConditional;
  d2_parameters.conditional.handle = d2_condition;
  d2_parameters.conditional.type = cudaGraphCondTypeIf;
  d2_parameters.conditional.size = 1U;
  cudaGraphNode_t d2_node = nullptr;
  error = cudaGraphAddNode(&d2_node, loop_body, &route_node, nullptr, 1U,
                           &d2_parameters);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("add M25 chain D2 branch", error);
  }
  cudaGraphNode_t d2_body_node = nullptr;
  error = cudaGraphAddChildGraphNode(
      &d2_body_node, d2_parameters.conditional.phGraph_out[0], nullptr, 0U,
      d2_source);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("add M25 chain D2 body", error);
  }
  cudaGraphNodeParams ordinary_parameters{};
  ordinary_parameters.type = cudaGraphNodeTypeConditional;
  ordinary_parameters.conditional.handle = ordinary_condition;
  ordinary_parameters.conditional.type = cudaGraphCondTypeIf;
  ordinary_parameters.conditional.size = 1U;
  cudaGraphNode_t ordinary_node = nullptr;
  error = cudaGraphAddNode(&ordinary_node, loop_body, &route_node, nullptr,
                           1U, &ordinary_parameters);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("add M25 chain ordinary branch", error);
  }
  cudaGraphNode_t ordinary_body_node = nullptr;
  error = cudaGraphAddChildGraphNode(
      &ordinary_body_node, ordinary_parameters.conditional.phGraph_out[0],
      nullptr, 0U, ordinary_source);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("add M25 chain ordinary body", error);
  }
  const cudaGraphNode_t branch_nodes[] = {d2_node, ordinary_node};
  cudaGraphNode_t continue_node = nullptr;
  error = cudaGraphAddChildGraphNode(
      &continue_node, loop_body, branch_nodes, 2U, continue_source);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("add M25 chain continuation", error);
  }
  error = cudaGraphInstantiate(&executable, root, nullptr, nullptr, 0U);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("instantiate M25 fixed chain graph", error);
  }
  mtp_chain_graphs[draft_count] = executable;
  executable = nullptr;
  cleanup();

  std::size_t free_after = 0U;
  error = cudaMemGetInfo(&free_after, &total_bytes);
  if (error != cudaSuccess) {
    (void)cudaGraphExecDestroy(mtp_chain_graphs[draft_count]);
    mtp_chain_graphs[draft_count] = nullptr;
    return CudaFailure("measure after M25 chain graph capture", error);
  }
  const std::uint64_t required_margin =
      context >= 64000U ? 200U * kMiB : 700U * kMiB;
  if (free_after < required_margin) {
    (void)cudaGraphExecDestroy(mtp_chain_graphs[draft_count]);
    mtp_chain_graphs[draft_count] = nullptr;
    return Status(StatusCode::kResourceExhausted,
                  "M25 chain graph leaves less than the profile margin");
  }
  if (free_before > free_after) {
    mtp_chain_graph_device_bytes[draft_count] += free_before - free_after;
  }
  return Status::Ok();
}

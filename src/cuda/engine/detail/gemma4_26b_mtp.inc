Status Gemma4Moe26BReferenceEngine::ConfigureTokenSelection(
    const SamplingOptions& options,
    std::span<const std::uint32_t> suppressed_token_ids) {
  if (!implementation_) return Invalid("M22 engine is not initialized");
  Status valid = ValidateSamplingOptions(
      options, static_cast<std::uint32_t>(kVocabulary));
  if (!valid.ok()) return valid;
  if (suppressed_token_ids.size() > kMaximumSuppressedTokens) {
    return Invalid("Gemma 4 26B supports at most 16 suppressed token IDs");
  }
  for (const std::uint32_t token : suppressed_token_ids) {
    if (token >= kVocabulary) {
      return Invalid("Gemma 4 26B suppressed token ID exceeds vocabulary");
    }
  }
  auto& x = *implementation_;
  if (x.position != 0U || x.prefill_calls != 0U ||
      x.decode_graph_launches != 0U) {
    return Invalid("M22 token selection must be configured before execution");
  }
  // Sampling options are captured by value in the fixed MTP graphs. A reused
  // resident runtime may start a new session with different sampling controls,
  // so discard only the MTP executables and recapture them after configuration.
  for (cudaGraphExec_t& graph : x.mtp_group_graphs) {
    if (graph != nullptr) (void)cudaGraphExecDestroy(graph);
    graph = nullptr;
  }
  for (cudaGraphExec_t& graph : x.mtp_chain_graphs) {
    if (graph != nullptr) (void)cudaGraphExecDestroy(graph);
    graph = nullptr;
  }
  x.mtp_group_graph_device_bytes = 0U;
  x.mtp_chain_graph_device_bytes.fill(0U);
  cudaError_t error = cudaMemsetAsync(
      x.repetition_mask, 0,
      static_cast<std::size_t>(kRepetitionMaskWords) * sizeof(std::uint32_t),
      x.stream);
  if (error == cudaSuccess && !suppressed_token_ids.empty()) {
    error = cudaMemcpyAsync(
        x.suppressed_token_ids, suppressed_token_ids.data(),
        suppressed_token_ids.size_bytes(), cudaMemcpyHostToDevice, x.stream);
  }
  if (error == cudaSuccess) error = cudaStreamSynchronize(x.stream);
  if (error != cudaSuccess) {
    return CudaFailure("configure M22 token selection", error);
  }
  x.sampling = options;
  x.sampling_step = 0U;
  x.suppressed_token_count =
      static_cast<std::uint32_t>(suppressed_token_ids.size());
  return Status::Ok();
}

Status Gemma4Moe26BReferenceEngine::ConfigurePrefillRouter(
    Gemma4MoePrefillRouter router) {
  if (!implementation_) return Invalid("M20 engine is not initialized");
  auto& x = *implementation_;
  if (x.position != 0U || x.prefill_calls != 0U ||
      x.decode_graph_launches != 0U) {
    return Invalid("M20 prefill router must be selected before execution");
  }
  if (router != Gemma4MoePrefillRouter::kSerialExact &&
      router != Gemma4MoePrefillRouter::kSm120TensorCore) {
    return Invalid("M20 prefill router selection is invalid");
  }
  if (router == Gemma4MoePrefillRouter::kSm120TensorCore &&
      x.backend != Gemma4Moe26BBackend::kSm120Integrated) {
    return Status(StatusCode::kUnsupported,
                  "SM120 tensor prefill router requires the integrated backend");
  }
  x.moe_config.prefill_router = router;
  return Status::Ok();
}

Status Gemma4Moe26BReferenceEngine::ConfigureMtpVerifierBackend(
    Gemma4Moe26BMtpVerifierBackend backend) {
  if (!implementation_) return Invalid("M25 engine is not initialized");
  auto& x = *implementation_;
  if (!x.assistant.prepared() || x.position != 0U || x.prefill_calls != 0U ||
      x.decode_graph_launches != 0U) {
    return Invalid("M25 verifier backend must be selected before execution");
  }
  x.mtp_verifier_backend = backend;
  if (backend != Gemma4Moe26BMtpVerifierBackend::kExactSharedBatchedMoe) {
    return Status::Ok();
  }
  for (const std::uint32_t draft_count : {1U, 2U, 4U}) {
    if (IsTrellis35RoutedExpertFormat(x.routed_expert_format) &&
        draft_count != 2U) {
      continue;
    }
    Status status = x.PrepareFixedMtpGraph(draft_count);
    if (!status.ok()) return status;
    status = x.PrepareFixedMtpChainGraph(draft_count);
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

Status Gemma4Moe26BReferenceEngine::ConfigureMtpRouterOverlapDiagnostic() {
  if (!implementation_) return Invalid("M25 engine is not initialized");
  auto& x = *implementation_;
  const bool graphs_prepared = std::any_of(
      x.mtp_group_graphs.begin(), x.mtp_group_graphs.end(),
      [](cudaGraphExec_t graph) { return graph != nullptr; });
  if (!x.assistant.prepared() || x.position != 0U || x.prefill_calls != 0U ||
      x.decode_graph_launches != 0U || graphs_prepared ||
      x.mtp_router_overlap.bytes() != 0U) {
    return Invalid(
        "M25 router-overlap diagnostic must be enabled once before graph capture");
  }
  Status status = x.mtp_router_overlap.Allocate(
      sizeof(MtpRouterOverlapCounters),
      "allocate M25 router-overlap diagnostic counters");
  if (!status.ok()) return status;
  return ResetMtpRouterOverlapDiagnostic();
}

Status Gemma4Moe26BReferenceEngine::ResetMtpRouterOverlapDiagnostic() {
  if (!implementation_ || implementation_->mtp_router_overlap.bytes() == 0U) {
    return Invalid("M25 router-overlap diagnostic is not enabled");
  }
  auto& x = *implementation_;
  cudaError_t error = cudaMemsetAsync(
      x.mtp_router_overlap.As<MtpRouterOverlapCounters>(), 0,
      sizeof(MtpRouterOverlapCounters), x.stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(x.stream);
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("reset M25 router-overlap diagnostic", error);
}

Status Gemma4Moe26BReferenceEngine::CopyMtpRouterOverlapDiagnostic(
    MtpRouterOverlapCounters* output) {
  if (!implementation_ || output == nullptr ||
      implementation_->mtp_router_overlap.bytes() == 0U) {
    return Invalid("M25 router-overlap diagnostic output is unavailable");
  }
  auto& x = *implementation_;
  cudaError_t error = cudaMemcpyAsync(
      output, x.mtp_router_overlap.As<MtpRouterOverlapCounters>(),
      sizeof(MtpRouterOverlapCounters), cudaMemcpyDeviceToHost, x.stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(x.stream);
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("copy M25 router-overlap diagnostic", error);
}

Result<std::uint32_t> Gemma4Moe26BReferenceEngine::SelectToken() {
  if (!implementation_) return Invalid("M22 engine is not initialized");
  auto& x = *implementation_;
  const NvtxRange range(x.token_selections == 0U
                            ? "gem16.mtp.initial_target_selection"
                            : "gem16.mtp.selection_sampling");
  ++x.token_selections;
  if (x.sampling.enabled) {
    Status selected = LaunchSampleToken(
        x.logits, x.sampling_logits, x.sampling_cumulative,
        x.sampling_token_ids, x.sampling_sorted_token_ids,
        x.repetition_mask, x.suppressed_token_ids,
        x.suppressed_token_count, static_cast<std::uint32_t>(kVocabulary),
        x.sampling, x.sampling_step++, nullptr, x.selected_token,
        x.sampling_algorithm_workspace,
        x.sampling_algorithm_workspace_bytes, x.stream);
    if (!selected.ok()) return selected;
    CommitSampledSelectionKernel<<<1U, 1U, 0, x.stream>>>(
        x.selected_token, x.prediction_device_status, x.decode_control,
        x.position);
    cudaError_t error = cudaGetLastError();
    if (error == cudaSuccess) {
      error = cudaMemcpyAsync(
          x.prediction_host_status, x.prediction_device_status,
          sizeof(PredictionStatus), cudaMemcpyDeviceToHost, x.stream);
    }
    if (error == cudaSuccess) error = cudaStreamSynchronize(x.stream);
    if (error != cudaSuccess) {
      return CudaFailure("copy sampled M22 prediction status", error);
    }
    const PredictionStatus status = *x.prediction_host_status;
    if (status.routing_finite == 0) {
      return Status(StatusCode::kInternal,
                    "Gemma 4 26B router produced a non-finite value");
    }
    if (status.finite == 0) {
      return Status(StatusCode::kInternal,
                    "Gemma 4 26B output logits are non-finite");
    }
    x.pending_decode_self_feed = false;
    x.decode_self_feed_valid = true;
    x.decode_self_feed_token = status.token;
    x.decode_self_feed_position = x.position;
    return status.token;
  }
  auto prediction = Prediction();
  if (!prediction.ok()) return prediction.status();
  if (x.suppressed_token_count == 0U) {
    return prediction.value().token;
  }
  Status selected = LaunchLogitArgmax(
      x.logits, x.suppressed_token_ids, x.suppressed_token_count,
      x.selected_token, x.stream);
  if (!selected.ok()) return selected;
  std::uint32_t token = 0U;
  cudaError_t error = cudaMemcpyAsync(
      &token, x.selected_token, sizeof(token), cudaMemcpyDeviceToHost,
      x.stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(x.stream);
  if (error != cudaSuccess) {
    return CudaFailure("copy M22 selected token", error);
  }
  return token;
}

Status Gemma4Moe26BReferenceEngine::LoadMtpAssistant(
    const std::filesystem::path& assistant_directory) {
  if (!implementation_ || assistant_directory.empty()) {
    return Invalid("M25 Assistant load request is invalid");
  }
  auto& x = *implementation_;
  if (x.backend != Gemma4Moe26BBackend::kSm120Integrated ||
      x.assistant.loaded()) {
    return Invalid(
        "M25 Assistant requires one unconfigured SM120 integrated engine");
  }
  Gemma4Moe26BAssistantModel candidate;
  Status status = candidate.Load(assistant_directory);
  if (!status.ok()) return status;
  status = candidate.Prepare(x.context);
  if (!status.ok()) return status;

  // The verifier is a separately named fixed region. It holds five Target
  // output rows plus both the overwritten-cache backup and compact
  // speculative K/V rows for every layer. No verifier storage is borrowed
  // from the Assistant, and no allocation occurs after this load boundary.
  LayoutBuilder layout;
  const std::uint64_t normalized =
      layout.Add<float>(kM25MaximumVerifyTokens * kWidth);
  const std::uint64_t head_activation = layout.Add<std::uint8_t>(
      kM25MaximumVerifyTokens * kWidth / 2U);
  const std::uint64_t head_scales = layout.Add<std::uint8_t>(
      kM25MaximumVerifyTokens * kWidth / kNvfp4Block);
  const std::uint64_t verifier_logits =
      layout.Add<float>(kM25MaximumVerifyTokens * kVocabulary);
  const std::uint64_t selected =
      layout.Add<std::uint32_t>(kM25MaximumVerifyTokens);
  const std::uint64_t second_selected =
      layout.Add<std::uint32_t>(kM25MaximumVerifyTokens);
  const std::uint64_t top_values =
      layout.Add<float>(kM25MaximumVerifyTokens);
  const std::uint64_t second_values =
      layout.Add<float>(kM25MaximumVerifyTokens);
  const std::uint64_t finite = layout.Add<int>(kM25MaximumVerifyTokens);
  const std::uint64_t transaction = layout.Add<MtpGroupTransaction>(1U);
  const std::uint64_t row_controls =
      layout.Add<DecodeControl>(kM25MaximumVerifyTokens);
  const std::uint64_t sampling_repetition_masks =
      layout.Add<std::uint32_t>(kM25MaximumVerifyTokens *
                                kRepetitionMaskWords);
  const std::uint64_t stop_tokens =
      layout.Add<std::uint32_t>(kMaximumSuppressedTokens);
  const std::uint64_t chain_result = layout.Add<MtpChainResult>(1U);
  const std::uint64_t chain_outputs =
      layout.Add<std::uint32_t>(x.context);
  const std::uint64_t chain_proposals = layout.Add<std::uint32_t>(
      kMaximumMtpDraftTokens * x.context);
  std::array<MtpVerifierLayerOffsets, kLayers> layer_offsets{};
  for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
    const std::uint64_t elements = kM25MaximumVerifyTokens *
                                   x.traits[layer].kv_heads *
                                   x.traits[layer].head_dimension;
    layer_offsets[layer].backup_key = layout.Add<std::uint8_t>(elements);
    layer_offsets[layer].backup_value = layout.Add<std::uint8_t>(elements);
    layer_offsets[layer].compact_key = layout.Add<std::uint8_t>(elements);
    layer_offsets[layer].compact_value = layout.Add<std::uint8_t>(elements);
  }
  if (layout.bytes == 0U ||
      layout.bytes == std::numeric_limits<std::uint64_t>::max()) {
    return Invalid("M25 verifier workspace layout overflow");
  }
  DeviceBuffer verifier;
  status = verifier.Allocate(layout.bytes, "allocate M25 fixed verifier workspace");
  if (!status.ok()) return status;
  MtpVerifierHostResult* host_result = nullptr;
  auto host_error = cudaMallocHost(&host_result, sizeof(MtpVerifierHostResult));
  if (host_error != cudaSuccess) {
    return CudaFailure("allocate pinned M25 verifier result", host_error);
  }
  std::byte* chain_host = nullptr;
  const std::uint64_t chain_host_bytes =
      sizeof(MtpChainResult) + x.context * sizeof(std::uint32_t);
  host_error = cudaMallocHost(&chain_host,
                              static_cast<std::size_t>(chain_host_bytes));
  if (host_error != cudaSuccess) {
    (void)cudaFreeHost(host_result);
    return CudaFailure("allocate pinned M25 chain result", host_error);
  }
  MtpStreamingRing* streaming_ring_host = nullptr;
  host_error = cudaHostAlloc(&streaming_ring_host, sizeof(MtpStreamingRing),
                             cudaHostAllocMapped);
  if (host_error != cudaSuccess) {
    (void)cudaFreeHost(host_result);
    (void)cudaFreeHost(chain_host);
    return CudaFailure("allocate mapped M25 streaming ring", host_error);
  }
  MtpStreamingRing* streaming_ring_device = nullptr;
  host_error = cudaHostGetDevicePointer(&streaming_ring_device,
                                        streaming_ring_host, 0U);
  if (host_error != cudaSuccess) {
    (void)cudaFreeHost(host_result);
    (void)cudaFreeHost(chain_host);
    (void)cudaFreeHost(streaming_ring_host);
    return CudaFailure("map M25 streaming ring into device space", host_error);
  }
  std::size_t free_bytes = 0U;
  std::size_t total_bytes = 0U;
  const auto measured = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (measured != cudaSuccess) {
    (void)cudaFreeHost(host_result);
    (void)cudaFreeHost(chain_host);
    (void)cudaFreeHost(streaming_ring_host);
    return CudaFailure("measure M25 post-Assistant margin", measured);
  }
  const std::uint64_t required_margin =
      x.context >= 64000U ? 200U * kMiB : 700U * kMiB;
  if (free_bytes < required_margin) {
    (void)cudaFreeHost(host_result);
    (void)cudaFreeHost(chain_host);
    (void)cudaFreeHost(streaming_ring_host);
    return Status(
        StatusCode::kResourceExhausted,
        "M25 Assistant and verifier leave less than the profile margin");
  }
  x.assistant = std::move(candidate);
  x.mtp_verifier = std::move(verifier);
  x.mtp_verifier_host = host_result;
  x.mtp_chain_host = chain_host;
  x.mtp_streaming_ring_host = streaming_ring_host;
  x.mtp_streaming_ring_device = streaming_ring_device;
  x.mtp_normalized_offset = normalized;
  x.mtp_head_activation_offset = head_activation;
  x.mtp_head_scales_offset = head_scales;
  x.mtp_logits_offset = verifier_logits;
  x.mtp_selected_offset = selected;
  x.mtp_second_selected_offset = second_selected;
  x.mtp_top_values_offset = top_values;
  x.mtp_second_values_offset = second_values;
  x.mtp_finite_offset = finite;
  x.mtp_transaction_offset = transaction;
  x.mtp_row_controls_offset = row_controls;
  x.mtp_sampling_repetition_masks_offset = sampling_repetition_masks;
  x.mtp_stop_tokens_offset = stop_tokens;
  x.mtp_chain_result_offset = chain_result;
  x.mtp_chain_outputs_offset = chain_outputs;
  x.mtp_chain_proposals_offset = chain_proposals;
  x.mtp_layer_offsets = layer_offsets;
  return Status::Ok();
}

Status Gemma4Moe26BReferenceEngine::ConfigureMtpStopTokens(
    std::span<const std::uint32_t> stop_token_ids) {
  if (!implementation_) return Invalid("M25 engine is not initialized");
  auto& x = *implementation_;
  if (!x.assistant.prepared() || x.mtp_verifier.bytes() == 0U ||
      x.position != 0U || x.prefill_calls != 0U ||
      x.decode_graph_launches != 0U) {
    return Invalid("M25 stop tokens must be configured before execution");
  }
  if (stop_token_ids.size() > kMaximumSuppressedTokens) {
    return Invalid("Gemma 4 26B MTP supports at most 16 stop-token IDs");
  }
  for (const std::uint32_t token : stop_token_ids) {
    if (token >= kVocabulary) {
      return Invalid("Gemma 4 26B MTP stop-token ID exceeds vocabulary");
    }
  }
  const bool graphs_prepared = std::any_of(
      x.mtp_chain_graphs.begin(), x.mtp_chain_graphs.end(),
      [](cudaGraphExec_t graph) { return graph != nullptr; });
  if (graphs_prepared && stop_token_ids.size() != x.mtp_stop_token_count) {
    return Invalid("M25 stop-token count cannot change after graph capture");
  }
  cudaError_t error = cudaSuccess;
  if (!stop_token_ids.empty()) {
    error = cudaMemcpyAsync(
        x.mtp_verifier.As<std::uint32_t>(x.mtp_stop_tokens_offset),
        stop_token_ids.data(), stop_token_ids.size_bytes(),
        cudaMemcpyHostToDevice, x.stream);
  }
  if (error == cudaSuccess) error = cudaStreamSynchronize(x.stream);
  if (error != cudaSuccess) {
    return CudaFailure("configure M25 stop-token IDs", error);
  }
  x.mtp_stop_token_count =
      static_cast<std::uint32_t>(stop_token_ids.size());
  return Status::Ok();
}

Status Gemma4Moe26BReferenceEngine::GenerateMtpAssistantDrafts(
    std::span<std::uint32_t> draft_token_ids) {
  if (!implementation_ || draft_token_ids.empty() ||
      draft_token_ids.size() > 4U) {
    return Invalid("M25 Assistant draft request must contain one to four slots");
  }
  auto& x = *implementation_;
  if (!x.assistant.prepared() || !x.has_last_input_token || x.position == 0U) {
    return Invalid("M25 Assistant requires one completed target transaction");
  }
  constexpr std::uint32_t kSlidingLayer = 28U;
  constexpr std::uint32_t kFullLayer = 29U;
  const auto make_view = [&x](std::uint32_t layer) {
    AssistantSharedKvView view;
    const bool global =
        x.traits[layer].attention == Gemma4Moe26BAttentionType::kFull;
    view.mode = AssistantKvCacheMode::kCheckpointFp8;
    view.key_fp8 = x.caches[layer].key;
    view.value_fp8 = x.caches[layer].value;
    view.key_scale_bf16 = x.attention_weights[layer].key_cache_scale_bf16;
    view.value_scale_bf16 =
        x.attention_weights[layer].value_cache_scale_bf16;
    view.capacity = x.caches[layer].capacity;
    view.tokens = global ? x.position : std::min(x.position, view.capacity);
    view.first_slot = global || x.position <= view.capacity
                          ? 0U
                          : x.position % view.capacity;
    view.kv_heads = x.traits[layer].kv_heads;
    view.head_dimension = x.traits[layer].head_dimension;
    return view;
  };
  if (x.traits[kSlidingLayer].attention !=
          Gemma4Moe26BAttentionType::kSliding ||
      x.traits[kFullLayer].attention != Gemma4Moe26BAttentionType::kFull) {
    return Status(StatusCode::kInternal,
                  "M25 target shared-KV layer mapping is invalid");
  }
  Gemma4Moe26BAssistantProposalContext context;
  context.target_embedding = x.head;
  context.target_hidden = x.final_hidden;
  context.sliding_kv = make_view(kSlidingLayer);
  context.full_kv = make_view(kFullLayer);
  context.input_token = x.last_input_token;
  context.position = x.position - 1U;
  return x.assistant.GenerateDrafts(context, draft_token_ids, x.stream);
}

Status Gemma4Moe26BReferenceEngine::GenerateMtpAssistantDraftsForPending(
    std::uint32_t pending_token,
    std::span<std::uint32_t> draft_token_ids) {
  if (!implementation_ || draft_token_ids.empty() ||
      draft_token_ids.size() > 4U || pending_token >= kVocabulary) {
    return Invalid("M25 pending-token Assistant request is invalid");
  }
  auto& x = *implementation_;
  if (!x.assistant.prepared() || x.position == 0U) {
    return Invalid("M25 pending-token Assistant requires committed Target state");
  }
  constexpr std::uint32_t kSlidingLayer = 28U;
  constexpr std::uint32_t kFullLayer = 29U;
  const auto make_view = [&x](std::uint32_t layer) {
    AssistantSharedKvView view;
    const bool global =
        x.traits[layer].attention == Gemma4Moe26BAttentionType::kFull;
    view.mode = AssistantKvCacheMode::kCheckpointFp8;
    view.key_fp8 = x.caches[layer].key;
    view.value_fp8 = x.caches[layer].value;
    view.key_scale_bf16 = x.attention_weights[layer].key_cache_scale_bf16;
    view.value_scale_bf16 = x.attention_weights[layer].value_cache_scale_bf16;
    view.capacity = x.caches[layer].capacity;
    view.tokens = global ? x.position : std::min(x.position, view.capacity);
    view.first_slot = global || x.position <= view.capacity
                          ? 0U
                          : x.position % view.capacity;
    view.kv_heads = x.traits[layer].kv_heads;
    view.head_dimension = x.traits[layer].head_dimension;
    return view;
  };
  Gemma4Moe26BAssistantProposalContext context;
  context.target_embedding = x.head;
  context.target_hidden = x.final_hidden;
  context.sliding_kv = make_view(kSlidingLayer);
  context.full_kv = make_view(kFullLayer);
  context.input_token = pending_token;
  context.position = x.position - 1U;
  return x.assistant.GenerateDrafts(context, draft_token_ids, x.stream);
}

Status Gemma4Moe26BReferenceEngine::RunMtpAssistantGroup(
    std::uint32_t pending_token, std::uint32_t proposal_count,
    MtpGroupResult* host_result) {
  if (!implementation_ || host_result == nullptr ||
      (proposal_count != 1U && proposal_count != 2U && proposal_count != 4U)) {
    return Invalid("M25 Target-verification group request is invalid");
  }
  auto& x = *implementation_;
  if (IsTrellis35RoutedExpertFormat(x.routed_expert_format) &&
      proposal_count != 2U) {
    return Status(StatusCode::kUnsupported,
                  "Trellis35 Target verification supports only D2/T3");
  }
  const std::uint64_t tokens = proposal_count + 1U;
  if (!x.assistant.prepared() || x.mtp_verifier.bytes() == 0U ||
      x.mtp_verifier_host == nullptr || x.position == 0U ||
      pending_token >= kVocabulary || x.position >= x.context ||
      tokens > x.context - x.position) {
    return Invalid("M25 Target-verification group exceeds resident state");
  }
  if (x.mtp_verifier_backend ==
          Gemma4Moe26BMtpVerifierBackend::kExactSharedBatchedMoe &&
      (proposal_count == 1U || proposal_count == 2U ||
       proposal_count == 4U)) {
    if (x.mtp_group_graphs[proposal_count] == nullptr) {
      Status prepared = x.PrepareFixedMtpGraph(proposal_count);
      if (!prepared.ok()) return prepared;
    }
    const NvtxRange range("gem16.mtp.pure_t3_verifier_group");
    return x.ExecuteFixedMtpGraphGroup(pending_token, proposal_count,
                                       host_result);
  }

  constexpr std::uint32_t kSlidingLayer = 28U;
  constexpr std::uint32_t kFullLayer = 29U;
  const auto make_view = [&x](std::uint32_t layer) {
    AssistantSharedKvView view;
    const bool global =
        x.traits[layer].attention == Gemma4Moe26BAttentionType::kFull;
    view.mode = AssistantKvCacheMode::kCheckpointFp8;
    view.key_fp8 = x.caches[layer].key;
    view.value_fp8 = x.caches[layer].value;
    view.key_scale_bf16 = x.attention_weights[layer].key_cache_scale_bf16;
    view.value_scale_bf16 = x.attention_weights[layer].value_cache_scale_bf16;
    view.capacity = x.caches[layer].capacity;
    view.tokens = global ? x.position : std::min(x.position, view.capacity);
    view.first_slot = global || x.position <= view.capacity
                          ? 0U
                          : x.position % view.capacity;
    view.kv_heads = x.traits[layer].kv_heads;
    view.head_dimension = x.traits[layer].head_dimension;
    return view;
  };
  if (x.traits[kSlidingLayer].attention !=
          Gemma4Moe26BAttentionType::kSliding ||
      x.traits[kFullLayer].attention != Gemma4Moe26BAttentionType::kFull) {
    return Status(StatusCode::kInternal,
                  "M25 target shared-KV layer mapping is invalid");
  }
  Gemma4Moe26BAssistantProposalContext proposal;
  proposal.target_embedding = x.head;
  proposal.target_hidden = x.final_hidden;
  proposal.sliding_kv = make_view(kSlidingLayer);
  proposal.full_kv = make_view(kFullLayer);
  proposal.input_token = pending_token;
  // This is the qualified 12B Assistant convention: the pending token is
  // combined with the hidden state/KV through the last committed Target row.
  proposal.position = x.position - 1U;
  Status status = x.assistant.GenerateDraftsDevice(
      proposal, proposal_count, x.stream);
  if (!status.ok()) return status;
  const std::uint32_t* drafts = x.assistant.device_draft_tokens();
  if (drafts == nullptr) {
    return Status(StatusCode::kInternal,
                  "M25 Assistant device proposals are unavailable");
  }

  auto* transaction = x.mtp_verifier.As<MtpGroupTransaction>(
      x.mtp_transaction_offset);
  *x.mtp_verifier_host = {};
  auto& control = x.mtp_verifier_host->transaction.control;
  control.current.input_token = pending_token;
  control.current.processed_position = x.position - 1U;
  control.current.remaining_output_capacity = x.context - x.position;
  control.current.output_write_position = 0U;
  control.current.sampling_step = x.sampling_step;
  control.next = control.current;
  control.fixed_draft_tokens = proposal_count;
  control.sampling_enabled = x.sampling.enabled ? 1U : 0U;
  cudaError_t error = cudaMemcpyAsync(
      transaction, &x.mtp_verifier_host->transaction,
      sizeof(MtpGroupTransaction), cudaMemcpyHostToDevice, x.stream);
  if (error != cudaSuccess) {
    return CudaFailure("initialize M25 verifier transaction", error);
  }

  status = LaunchBuildMtpVerificationInputs(
      pending_token, drafts, proposal_count, x.prefill_tokens, x.stream);
  if (!status.ok()) return status;
  error = cudaMemsetAsync(x.routing_finite, 1, sizeof(int), x.stream);
  if (error != cudaSuccess) {
    return CudaFailure("initialize M25 verifier router finite flag", error);
  }
  constexpr unsigned threads = 256U;
  const std::uint64_t hidden_elements = tokens * kWidth;
  TiledEmbeddingLookupBatchKernel<<<
      static_cast<unsigned>((hidden_elements + threads - 1U) / threads),
      threads, 0, x.stream>>>(
      x.head.packed_e2m1, x.head.scales_e4m3fn,
      x.head.weight_global_divisor, x.prefill_tokens, x.prefill_hidden_a,
      hidden_elements);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("launch M25 verifier embedding", error);
  }

  const std::uint64_t start_position = x.position;
  auto* row_controls =
      x.mtp_verifier.As<DecodeControl>(x.mtp_row_controls_offset);
  BuildMtpDecodeControlsKernel<<<1U, static_cast<unsigned>(tokens), 0,
                                 x.stream>>>(
      row_controls, start_position, x.sampling_step, tokens);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("build M25 exact verifier row controls", error);
  }
  const bool batch_attention =
      x.mtp_verifier_backend ==
          Gemma4Moe26BMtpVerifierBackend::kBatchedAttention ||
      x.mtp_verifier_backend ==
          Gemma4Moe26BMtpVerifierBackend::kFullyBatched;
  const bool batch_moe =
      x.mtp_verifier_backend ==
          Gemma4Moe26BMtpVerifierBackend::kBatchedMoe ||
      x.mtp_verifier_backend ==
          Gemma4Moe26BMtpVerifierBackend::kFullyBatched;
  const bool exact_shared_batch_moe =
      x.mtp_verifier_backend ==
      Gemma4Moe26BMtpVerifierBackend::kExactSharedBatchedMoe;
  for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
    const auto& trait = x.traits[layer];
    const std::uint64_t kv_elements =
        trait.kv_heads * trait.head_dimension;
    const auto offsets = x.mtp_layer_offsets[layer];
    auto* backup_key =
        x.mtp_verifier.As<std::uint8_t>(offsets.backup_key);
    auto* backup_value =
        x.mtp_verifier.As<std::uint8_t>(offsets.backup_value);
    auto* compact_key =
        x.mtp_verifier.As<std::uint8_t>(offsets.compact_key);
    auto* compact_value =
        x.mtp_verifier.As<std::uint8_t>(offsets.compact_value);
    status = LaunchCopyCircularMtpKvFp8(
        x.caches[layer].key, x.caches[layer].value, backup_key, backup_value,
        start_position, tokens, kv_elements, x.caches[layer].capacity, false,
        x.stream);
    if (!status.ok()) return status;
    if ((exact_shared_batch_moe || batch_attention) && tokens == 3U) {
      // The fixed-D2 verifier shares the qualified 12B three-row attention
      // machinery. MoE scratch is dead until the attention boundary and is
      // large enough for the split-online partials, so no M25 allocation is
      // introduced here.
      status = LaunchGemma4Moe26BAttentionSm120MtpD2Layer(
          x.prefill_hidden_a, x.prefill_hidden_b, start_position, trait,
          x.attention_weights[layer], x.caches[layer],
          x.prefill_attention_workspace,
          x.prefill_moe_workspace.shared_product, row_controls,
          batch_attention, exact_shared_batch_moe, false, 1.0e-6F,
          x.stream);
      if (!status.ok()) return status;
    } else if (batch_attention) {
      status = LaunchGemma4Moe26BAttentionSm120PrefillLayer(
          x.prefill_hidden_a, x.prefill_hidden_b, start_position, tokens,
          trait, x.attention_weights[layer], x.caches[layer],
          x.prefill_attention_workspace, 1.0e-6F, x.stream);
      if (!status.ok()) return status;
    } else {
      // Layer-major microbatching preserves the frozen ordinary-decode
      // arithmetic for every row. Later rows see earlier speculative K/V at
      // the same layer while all rows reuse one fixed decode workspace.
      for (std::uint64_t row = 0U; row < tokens; ++row) {
        status = LaunchGemma4Moe26BAttentionReferenceControlledLayer(
            x.prefill_hidden_a + row * kWidth,
            x.prefill_hidden_b + row * kWidth, trait,
            x.attention_weights[layer], x.caches[layer],
            x.attention_workspace, row_controls + row, 1.0e-6F, x.stream,
            false);
        if (!status.ok()) return status;
      }
    }
    if (IsTrellis35RoutedExpertFormat(x.routed_expert_format)) {
      if (!exact_shared_batch_moe || tokens != kTrellis35T3Rows) {
        return Status(StatusCode::kUnsupported,
                      "Trellis35 verifier requires exact Fixed-D2/T3");
      }
      status = LaunchGemma4MoeSm120Trellis35T3Layer(
          x.prefill_hidden_b, x.prefill_hidden_a, tokens, x.moe_config,
          x.moe_weights[layer], x.trellis35_artifact.layers()[layer],
          x.trellis35_t3_workspace, x.prefill_moe_workspace, x.stream,
          x.mtp_router_overlap.bytes() == 0U
              ? nullptr
              : x.mtp_router_overlap.As<MtpRouterOverlapCounters>());
      if (!status.ok()) return status;
    } else if (exact_shared_batch_moe) {
      status = LaunchGemma4MoeSm120MtpSharedBatchLayer(
          x.prefill_hidden_b, x.prefill_hidden_a, tokens, x.moe_config,
          x.moe_weights[layer], x.prefill_moe_workspace, x.moe_workspace,
          x.stream,
          tokens != 3U || x.mtp_router_overlap.bytes() == 0U
              ? nullptr
              : x.mtp_router_overlap.As<MtpRouterOverlapCounters>());
      if (!status.ok()) return status;
    } else if (batch_moe) {
      status = LaunchGemma4MoeSm120PrefillLayer(
          x.prefill_hidden_b, x.prefill_hidden_a, tokens, x.moe_config,
          x.moe_weights[layer], x.prefill_moe_workspace, x.stream);
      if (!status.ok()) return status;
    } else {
      for (std::uint64_t row = 0U; row < tokens; ++row) {
      status = LaunchGemma4MoeSm120Layer(
          x.prefill_hidden_b + row * kWidth,
          x.prefill_hidden_a + row * kWidth, x.moe_config,
          x.moe_weights[layer], x.moe_workspace, x.stream,
          x.shared_moe_stream, x.shared_moe_fork, x.shared_moe_join);
      if (!status.ok()) return status;
      }
    }
    status = LaunchCopyCircularMtpKvFp8(
        x.caches[layer].key, x.caches[layer].value, compact_key, compact_value,
        start_position, tokens, kv_elements, x.caches[layer].capacity, false,
        x.stream);
    if (!status.ok()) return status;
    status = LaunchCopyCircularMtpKvFp8(
        x.caches[layer].key, x.caches[layer].value, backup_key, backup_value,
        start_position, tokens, kv_elements, x.caches[layer].capacity, true,
        x.stream);
    if (!status.ok()) return status;
  }

  float* normalized =
      x.mtp_verifier.As<float>(x.mtp_normalized_offset);
  auto* head_activation =
      x.mtp_verifier.As<std::uint8_t>(x.mtp_head_activation_offset);
  auto* head_scales =
      x.mtp_verifier.As<std::uint8_t>(x.mtp_head_scales_offset);
  float* batch_logits = x.mtp_verifier.As<float>(x.mtp_logits_offset);
  auto* selected =
      x.mtp_verifier.As<std::uint32_t>(x.mtp_selected_offset);
  auto* second_selected = x.mtp_verifier.As<std::uint32_t>(
      x.mtp_second_selected_offset);
  float* top_values =
      x.mtp_verifier.As<float>(x.mtp_top_values_offset);
  float* second_values =
      x.mtp_verifier.As<float>(x.mtp_second_values_offset);
  int* finite = x.mtp_verifier.As<int>(x.mtp_finite_offset);
  status = LaunchRmsNormBf16(x.prefill_hidden_a, x.final_norm, normalized,
                             tokens, kWidth, 1.0e-6F, x.stream);
  if (!status.ok()) return status;
  status = LaunchRmsNormNvfp4ActivationQuantizationBatch(
      x.prefill_hidden_a, x.final_norm, head_activation, head_scales, tokens,
      kWidth, 1.0e-6F, x.head.activation_global_divisor, x.stream);
  if (!status.ok()) return status;
  status = LaunchNvfp4Sm120DirectProjectionBf16FloatExactBatch(
      head_activation, head_scales, x.head.packed_e2m1,
      x.head.scales_e4m3fn, batch_logits, tokens, x.head.rows, x.head.columns,
      x.head.activation_global_divisor, x.head.weight_global_divisor,
      x.stream);
  if (!status.ok()) return status;
  for (std::uint64_t row = 0U; row < tokens; ++row) {
    status = LaunchSoftcapArgmax(
        batch_logits + row * kVocabulary, x.softcap, finite + row,
        x.output_candidates, selected + row, top_values + row, x.stream,
        nullptr, x.suppressed_token_ids, x.suppressed_token_count);
    if (!status.ok()) return status;
    status = LaunchSoftcapArgmax(
        batch_logits + row * kVocabulary, x.softcap, finite + row,
        x.output_candidates, second_selected + row, second_values + row,
        x.stream, nullptr, x.suppressed_token_ids, x.suppressed_token_count,
        selected + row, false);
    if (!status.ok()) return status;
  }
  status = x.SelectMtpVerificationTokens(
      batch_logits, x.prefill_tokens, row_controls,
      static_cast<std::uint32_t>(tokens), selected);
  if (!status.ok()) return status;
  status = LaunchAcceptMtpGroup(
      drafts, selected, proposal_count,
      x.mtp_verifier.As<std::uint32_t>(x.mtp_stop_tokens_offset),
      x.mtp_stop_token_count,
      &transaction->result, &transaction->control, x.stream);
  if (!status.ok()) return status;
  if (x.sampling.enabled && x.sampling.repetition_penalty != 1.0F) {
    status = LaunchCommitSpeculativeRepetitionMask(
        x.mtp_verifier.As<std::uint32_t>(
            x.mtp_sampling_repetition_masks_offset),
        kRepetitionMaskWords, &transaction->result.output_count,
        x.repetition_mask, x.stream);
    if (!status.ok()) return status;
  }
  for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
    const auto& trait = x.traits[layer];
    const auto offsets = x.mtp_layer_offsets[layer];
    status = LaunchCommitMtpKvFp8(
        x.mtp_verifier.As<std::uint8_t>(offsets.compact_key),
        x.mtp_verifier.As<std::uint8_t>(offsets.compact_value),
        x.caches[layer].key, x.caches[layer].value, start_position,
        trait.kv_heads * trait.head_dimension, x.caches[layer].capacity,
        &transaction->result, x.stream);
    if (!status.ok()) return status;
  }
  status = LaunchCommitMtpHidden(normalized, x.final_hidden, kWidth,
                                 &transaction->result, x.stream);
  if (!status.ok()) return status;
  CommitMtpPredictionKernel<true>
      <<<kArgmaxBlocks, kArgmaxThreads, 0, x.stream>>>(
          batch_logits, x.logits, &transaction->result, finite,
          x.routing_finite, x.prediction_device_status);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("commit M25 verifier prediction", error);
  }

  error = cudaMemcpyAsync(&x.mtp_verifier_host->transaction, transaction,
                          sizeof(MtpGroupTransaction),
                          cudaMemcpyDeviceToHost, x.stream);
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(x.mtp_verifier_host->logits_finite.data(), finite,
                            tokens * sizeof(int), cudaMemcpyDeviceToHost,
                            x.stream);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(x.mtp_verifier_host->top_values.data(), top_values,
                            tokens * sizeof(float), cudaMemcpyDeviceToHost,
                            x.stream);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(x.mtp_verifier_host->second_values.data(),
                            second_values, tokens * sizeof(float),
                            cudaMemcpyDeviceToHost, x.stream);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(&x.mtp_verifier_host->routing_finite,
                            x.routing_finite, sizeof(int),
                            cudaMemcpyDeviceToHost, x.stream);
  }
  if (error == cudaSuccess) error = cudaStreamSynchronize(x.stream);
  if (error != cudaSuccess) {
    return CudaFailure("synchronize M25 verifier transaction", error);
  }
  const auto& completed = x.mtp_verifier_host->transaction;
  if (completed.control.transition_valid != 1U ||
      completed.control.proposal_count != proposal_count ||
      completed.result.output_count == 0U ||
      completed.result.output_count > tokens ||
      completed.control.next.input_token !=
          completed.result.verified[completed.result.output_count - 1U] ||
      completed.control.next.processed_position !=
          completed.control.current.processed_position +
              completed.result.output_count) {
    return Status(StatusCode::kInternal,
                  "M25 verifier device transition is inconsistent");
  }
  if (x.mtp_verifier_host->routing_finite == 0 ||
      std::any_of(x.mtp_verifier_host->logits_finite.begin(),
                  x.mtp_verifier_host->logits_finite.begin() + tokens,
                  [](int value) { return value == 0; })) {
    return Status(StatusCode::kInternal,
                  "M25 Target verification produced non-finite values");
  }
  x.last_mtp_verification_min_margin =
      std::numeric_limits<float>::infinity();
  for (std::uint64_t row = 0U; row < tokens; ++row) {
    x.last_mtp_verification_min_margin = std::min(
        x.last_mtp_verification_min_margin,
        x.mtp_verifier_host->top_values[row] -
            x.mtp_verifier_host->second_values[row]);
  }

  const std::uint64_t previous_position = x.position;
  x.position += completed.result.output_count;
  if (x.sliding_capacity != 0U) {
    x.sliding_ring_wraps += x.position / x.sliding_capacity -
                            previous_position / x.sliding_capacity;
  }
  x.maximum_global_position_exclusive =
      std::max(x.maximum_global_position_exclusive, x.position);
  x.token_selections += completed.result.output_count;
  if (x.sampling.enabled) {
    x.sampling_step += completed.result.output_count;
  }
  x.pending_decode_self_feed = false;
  x.decode_self_feed_valid = false;
  x.has_last_input_token = true;
  x.last_input_token = completed.result.output_count == 1U
                           ? pending_token
                           : completed.result.proposed[
                                 completed.result.output_count - 2U];
  *host_result = completed.result;
  return Status::Ok();
}

Status Gemma4Moe26BReferenceEngine::RunFixedMtpGraphChain(
    std::uint32_t pending_token, std::uint32_t draft_count,
    std::span<std::uint32_t> output_token_ids,
    MtpChainResult* host_result,
    Status (*generated_token_callback)(void*, std::uint32_t),
    void* generated_token_callback_context) {
  if (!implementation_ || host_result == nullptr ||
      output_token_ids.empty() || pending_token >= kVocabulary ||
      draft_count == 0U || draft_count > kMaximumMtpDraftTokens) {
    return Invalid("M25 fixed graph chain request is invalid");
  }
  auto& x = *implementation_;
  if (x.mtp_chain_graphs[draft_count] == nullptr ||
      !x.assistant.prepared() ||
      x.mtp_streaming_ring_host == nullptr ||
      x.mtp_streaming_ring_device == nullptr ||
      x.position == 0U || x.position >= x.context ||
      output_token_ids.size() > x.context - x.position) {
    return Invalid("M25 fixed graph chain exceeds the resident state");
  }
  *x.mtp_verifier_host = {};
  auto& host_control = x.mtp_verifier_host->transaction.control;
  host_control.current.input_token = pending_token;
  host_control.current.processed_position = x.position - 1U;
  host_control.current.remaining_output_capacity = output_token_ids.size();
  host_control.current.output_write_position = 0U;
  host_control.current.sampling_step = x.sampling_step;
  host_control.next = host_control.current;
  host_control.fixed_draft_tokens = draft_count;
  host_control.sampling_enabled = x.sampling.enabled ? 1U : 0U;
  auto* device_transaction =
      x.mtp_verifier.As<MtpGroupTransaction>(x.mtp_transaction_offset);
  auto* device_result =
      x.mtp_verifier.As<MtpChainResult>(x.mtp_chain_result_offset);
  std::atomic_ref<unsigned long long> producer(
      x.mtp_streaming_ring_host->producer);
  std::atomic_ref<unsigned long long> consumer(
      x.mtp_streaming_ring_host->consumer);
  std::atomic_ref<unsigned long long> cancelled(
      x.mtp_streaming_ring_host->cancelled);
  std::atomic_ref<unsigned long long> backpressure(
      x.mtp_streaming_ring_host->backpressure_events);
  producer.store(0U, std::memory_order_relaxed);
  consumer.store(0U, std::memory_order_relaxed);
  cancelled.store(0U, std::memory_order_relaxed);
  backpressure.store(0U, std::memory_order_relaxed);
  cudaError_t error = cudaMemcpyAsync(
      device_transaction, &x.mtp_verifier_host->transaction,
      sizeof(MtpGroupTransaction), cudaMemcpyHostToDevice, x.stream);
  if (error == cudaSuccess) {
    error = cudaMemsetAsync(device_result, 0, sizeof(MtpChainResult), x.stream);
  }
  if (error == cudaSuccess) {
    error = cudaGraphLaunch(x.mtp_chain_graphs[draft_count], x.stream);
  }
  if (error != cudaSuccess) {
    return CudaFailure("execute M25 fixed graph chain", error);
  }
  Status callback_status = Status::Ok();
  unsigned long long consumed = 0U;
  const auto consume_available = [&]() {
    const unsigned long long published =
        producer.load(std::memory_order_acquire);
    while (consumed < published) {
      const std::uint32_t token = x.mtp_streaming_ring_host->tokens[
          consumed % kMtpStreamingRingCapacity];
      if (generated_token_callback != nullptr && callback_status.ok()) {
        callback_status = generated_token_callback(
            generated_token_callback_context, token);
        if (!callback_status.ok()) {
          cancelled.store(1U, std::memory_order_release);
        }
      }
      ++consumed;
      consumer.store(consumed, std::memory_order_release);
    }
  };
  while (true) {
    consume_available();
    error = cudaStreamQuery(x.stream);
    if (error == cudaSuccess) {
      consume_available();
      break;
    }
    if (error != cudaErrorNotReady) {
      return CudaFailure("poll M25 fixed graph chain", error);
    }
    std::this_thread::yield();
  }
  ++x.mtp_chain_graph_launches[draft_count];

  auto* pinned_result = reinterpret_cast<MtpChainResult*>(x.mtp_chain_host);
  auto* pinned_outputs = reinterpret_cast<std::uint32_t*>(
      x.mtp_chain_host + sizeof(MtpChainResult));
  error = cudaMemcpy(pinned_result, device_result, sizeof(MtpChainResult),
                     cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) {
    error = cudaMemcpy(
        pinned_outputs,
        x.mtp_verifier.As<std::uint32_t>(x.mtp_chain_outputs_offset),
        output_token_ids.size_bytes(), cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(&x.mtp_verifier_host->transaction, device_transaction,
                       sizeof(MtpGroupTransaction), cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) {
    return CudaFailure("copy M25 fixed graph chain result", error);
  }
  const auto& final_control = x.mtp_verifier_host->transaction.control;
  if (pinned_result->output_count == 0U ||
      pinned_result->output_count > output_token_ids.size() ||
      (pinned_result->stopped == 0U &&
       pinned_result->output_count != output_token_ids.size()) ||
      pinned_result->proposed_count !=
          draft_count * pinned_result->group_count ||
      pinned_result->accepted_count + pinned_result->rejected_count !=
          pinned_result->proposed_count ||
      pinned_result->ordinary_tail_count > draft_count ||
      pinned_result->non_finite_step_count != 0U ||
      consumed != pinned_result->output_count ||
      final_control.current.input_token !=
          pinned_outputs[pinned_result->output_count - 1U] ||
      final_control.current.processed_position !=
          x.position - 1U + pinned_result->output_count ||
      final_control.current.remaining_output_capacity !=
          output_token_ids.size() - pinned_result->output_count ||
      final_control.current.output_write_position !=
          pinned_result->output_count) {
    return Status(StatusCode::kInternal,
                  "M25 fixed graph chain result is inconsistent");
  }
  std::copy_n(pinned_outputs, pinned_result->output_count,
              output_token_ids.begin());
  const std::uint64_t previous_position = x.position;
  x.position += pinned_result->output_count;
  if (x.sliding_capacity != 0U) {
    x.sliding_ring_wraps += x.position / x.sliding_capacity -
                            previous_position / x.sliding_capacity;
  }
  x.maximum_global_position_exclusive =
      std::max(x.maximum_global_position_exclusive, x.position);
  x.token_selections += pinned_result->output_count;
  x.sampling_step = final_control.current.sampling_step;
  x.pending_decode_self_feed = false;
  x.decode_self_feed_valid = false;
  x.has_last_input_token = true;
  x.last_input_token = pinned_result->output_count == 1U
                           ? pending_token
                           : output_token_ids[pinned_result->output_count - 2U];
  *host_result = *pinned_result;
  return callback_status;
}

bool Gemma4Moe26BReferenceEngine::mtp_assistant_loaded() const {
  return implementation_ != nullptr && implementation_->assistant.loaded();
}

std::uint64_t Gemma4Moe26BReferenceEngine::mtp_assistant_weight_bytes() const {
  return implementation_ == nullptr ? 0U
                                    : implementation_->assistant.arena_bytes();
}

std::uint64_t
Gemma4Moe26BReferenceEngine::mtp_assistant_workspace_bytes() const {
  return implementation_ == nullptr
             ? 0U
             : implementation_->assistant.workspace_bytes();
}

bool Gemma4Moe26BReferenceEngine::mtp_group_graph_prepared(
    std::uint32_t draft_count) const {
  return implementation_ != nullptr &&
         draft_count <= kMaximumMtpDraftTokens &&
         implementation_->mtp_group_graphs[draft_count] != nullptr;
}

std::uint64_t
Gemma4Moe26BReferenceEngine::mtp_group_graph_device_bytes() const {
  return implementation_ == nullptr
             ? 0U
             : implementation_->mtp_group_graph_device_bytes;
}

std::uint64_t Gemma4Moe26BReferenceEngine::mtp_group_graph_launches() const {
  return implementation_ == nullptr
             ? 0U
             : implementation_->mtp_group_graph_launches;
}

bool Gemma4Moe26BReferenceEngine::mtp_chain_graph_prepared(
    std::uint32_t draft_count) const {
  return implementation_ != nullptr &&
         draft_count <= kMaximumMtpDraftTokens &&
         implementation_->mtp_chain_graphs[draft_count] != nullptr;
}

std::uint64_t
Gemma4Moe26BReferenceEngine::mtp_chain_graph_device_bytes(
    std::uint32_t draft_count) const {
  return implementation_ == nullptr || draft_count > kMaximumMtpDraftTokens
             ? 0U
             : implementation_->mtp_chain_graph_device_bytes[draft_count];
}

std::uint64_t Gemma4Moe26BReferenceEngine::mtp_chain_graph_launches(
    std::uint32_t draft_count) const {
  return implementation_ == nullptr || draft_count > kMaximumMtpDraftTokens
             ? 0U
             : implementation_->mtp_chain_graph_launches[draft_count];
}

float Gemma4Moe26BReferenceEngine::last_mtp_verification_min_margin() const {
  return implementation_ == nullptr
             ? 0.0F
             : implementation_->last_mtp_verification_min_margin;
}

Status Gemma4Moe26BReferenceEngine::CopyMtpAssistantOracleInputs(
    std::span<float> concatenated_input,
    std::span<float> assistant_logits,
    std::span<std::uint8_t> sliding_key,
    std::span<std::uint8_t> sliding_value,
    std::span<std::uint8_t> full_key,
    std::span<std::uint8_t> full_value,
    std::span<std::uint16_t> kv_scale_bf16_bits) {
  if (!implementation_) return Invalid("M25 oracle engine is unavailable");
  auto& x = *implementation_;
  constexpr std::uint32_t kSlidingLayer = 28U;
  constexpr std::uint32_t kFullLayer = 29U;
  const std::uint64_t sliding_tokens =
      std::min(x.position, x.caches[kSlidingLayer].capacity);
  const std::uint64_t sliding_bytes =
      sliding_tokens * x.traits[kSlidingLayer].kv_heads *
      x.traits[kSlidingLayer].head_dimension;
  const std::uint64_t full_bytes =
      x.position * x.traits[kFullLayer].kv_heads *
      x.traits[kFullLayer].head_dimension;
  if (!x.assistant.prepared() || x.position == 0U ||
      x.position > x.caches[kSlidingLayer].capacity ||
      sliding_key.size() != sliding_bytes ||
      sliding_value.size() != sliding_bytes ||
      full_key.size() != full_bytes || full_value.size() != full_bytes ||
      kv_scale_bf16_bits.size() != 4U) {
    return Invalid("M25 Assistant oracle cache geometry is invalid");
  }
  Status status = x.assistant.CopyLastOracleState(
      concatenated_input, assistant_logits, x.stream);
  if (!status.ok()) return status;
  const std::array copies = {
      std::pair{static_cast<void*>(sliding_key.data()),
                static_cast<const void*>(x.caches[kSlidingLayer].key)},
      std::pair{static_cast<void*>(sliding_value.data()),
                static_cast<const void*>(x.caches[kSlidingLayer].value)},
      std::pair{static_cast<void*>(full_key.data()),
                static_cast<const void*>(x.caches[kFullLayer].key)},
      std::pair{static_cast<void*>(full_value.data()),
                static_cast<const void*>(x.caches[kFullLayer].value)},
  };
  const std::array sizes = {sliding_key.size_bytes(),
                            sliding_value.size_bytes(), full_key.size_bytes(),
                            full_value.size_bytes()};
  cudaError_t error = cudaSuccess;
  for (std::size_t index = 0; index < copies.size() &&
                              error == cudaSuccess;
       ++index) {
    error = cudaMemcpyAsync(copies[index].first, copies[index].second,
                            sizes[index], cudaMemcpyDeviceToHost, x.stream);
  }
  const std::array<const std::uint16_t*, 4> scales = {
      x.attention_weights[kSlidingLayer].key_cache_scale_bf16,
      x.attention_weights[kSlidingLayer].value_cache_scale_bf16,
      x.attention_weights[kFullLayer].key_cache_scale_bf16,
      x.attention_weights[kFullLayer].value_cache_scale_bf16,
  };
  for (std::size_t index = 0; index < scales.size() && error == cudaSuccess;
       ++index) {
    error = cudaMemcpyAsync(kv_scale_bf16_bits.data() + index, scales[index],
                            sizeof(std::uint16_t), cudaMemcpyDeviceToHost,
                            x.stream);
  }
  if (error == cudaSuccess) error = cudaStreamSynchronize(x.stream);
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("copy M25 Assistant oracle cache", error);
}

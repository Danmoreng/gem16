  [[nodiscard]] std::uint64_t weight_bytes() const { return model_.weight_bytes(); }
  [[nodiscard]] bool assistant_loaded() const { return assistant_.loaded(); }
  [[nodiscard]] std::uint64_t assistant_source_bytes() const {
    return assistant_.source_bytes();
  }
  [[nodiscard]] std::uint64_t assistant_weight_bytes() const {
    return assistant_.arena_bytes();
  }
  [[nodiscard]] std::uint64_t assistant_device_memory_delta_bytes() const {
    return assistant_device_memory_delta_bytes_;
  }
  [[nodiscard]] std::uint64_t assistant_tensor_count() const {
    return assistant_.tensor_count();
  }
  [[nodiscard]] std::uint64_t assistant_workspace_bytes() const {
    return assistant_.workspace_bytes() + mtp_workspace_.bytes();
  }
  [[nodiscard]] Status GenerateAssistantDraftsDevice(
      std::uint32_t input_token, std::uint64_t processed_position,
      std::uint32_t draft_count) {
    const NvtxRange range("gem16.mtp.propose");
    if (mtp_draft_tokens_ == 0U || !assistant_.prepared() ||
        latest_target_hidden_ == nullptr || draft_count == 0U ||
        draft_count > mtp_draft_tokens_ || processed_position >= max_context_) {
      return Error(StatusCode::kInvalidArgument,
                   "active MTP proposal state is invalid");
    }
    const auto make_view = [this, processed_position](
                               const LayerBinding& layer) {
      internal::AssistantSharedKvView view;
      view.mode = kv_cache_mode_ == KvCacheMode::kCheckpointFp8
                      ? internal::AssistantKvCacheMode::kCheckpointFp8
                      : internal::AssistantKvCacheMode::kBf16;
      view.key_fp8 = layer.key_cache_fp8;
      view.value_fp8 = layer.value_cache_fp8;
      view.key_bf16 = layer.key_cache_bf16;
      view.value_bf16 = layer.value_cache_bf16;
      view.key_scale_bf16 = layer.k_cache_scale;
      view.value_scale_bf16 = layer.v_cache_scale;
      view.capacity = layer.global
                          ? max_context_
                          : std::min(max_context_, kSlidingWindow);
      view.tokens = layer.global
                        ? processed_position + 1U
                        : std::min(processed_position + 1U, view.capacity);
      view.first_slot =
          layer.global || processed_position + 1U <= view.capacity
              ? 0U
              : (processed_position + 1U) % view.capacity;
      view.kv_heads = layer.kv_heads;
      view.head_dimension = layer.head_dimension;
      return view;
    };
    const LayerBinding& sliding = model_.layers()[46U];
    const LayerBinding& full = model_.layers()[47U];
    if (sliding.global || !full.global) {
      return Error(StatusCode::kInternal,
                   "target MTP shared-KV layer mapping is invalid");
    }
    internal::AssistantProposalContext context;
    context.target_embedding = model_.embedding();
    context.target_hidden = latest_target_hidden_;
    context.sliding_kv = make_view(sliding);
    context.full_kv = make_view(full);
    context.input_token = input_token;
    context.position = processed_position;
    return assistant_.GenerateDraftsDevice(context, draft_count, stream_);
  }

  [[nodiscard]] Status PrepareMtpDeviceControl(
      std::uint32_t input_token, std::uint64_t processed_position,
      std::uint64_t remaining_output_capacity,
      std::uint64_t output_write_position, bool stopped,
      std::uint32_t stop_token,
      const internal::MtpReasoningState& reasoning = {}) {
    if (mtp_draft_tokens_ == 0U || remaining_output_capacity == 0U ||
        processed_position >= max_context_ ||
        remaining_output_capacity > max_context_ - processed_position - 1U ||
        reasoning.open_token_count >
            internal::kMaximumThinkingOpenTokens ||
        reasoning.open_match_length > reasoning.open_token_count ||
        (reasoning.enabled != 0U &&
         (reasoning.open_token_count == 0U ||
          reasoning.max_reasoning_tokens == 0U))) {
      return Error(StatusCode::kInvalidArgument,
                   "MTP device-control state is invalid");
    }
    auto* pinned_transaction =
        reinterpret_cast<internal::MtpGroupTransaction*>(
            mtp_host_result_.span().data());
    pinned_transaction->control = {};
    pinned_transaction->control.current.input_token = input_token;
    pinned_transaction->control.current.processed_position =
        processed_position;
    pinned_transaction->control.current.remaining_output_capacity =
        remaining_output_capacity;
    pinned_transaction->control.current.output_write_position =
        output_write_position;
    pinned_transaction->control.current.sampling_step = sampling_step_;
    pinned_transaction->control.current.stopped = stopped ? 1U : 0U;
    pinned_transaction->control.current.stop_token = stop_token;
    pinned_transaction->control.next = pinned_transaction->control.current;
    pinned_transaction->control.reasoning = reasoning;
    pinned_transaction->control.fixed_draft_tokens = mtp_draft_tokens_;
    pinned_transaction->control.sampling_enabled =
        sampling_.enabled ? 1U : 0U;
    auto* device_transaction = Pointer<internal::MtpGroupTransaction>(
        mtp_workspace_, mtp_offsets_.transaction);
    const cudaError_t error = cudaMemcpyAsync(
        &device_transaction->control, &pinned_transaction->control,
        sizeof(internal::MtpDeviceControl), cudaMemcpyHostToDevice, stream_);
    return error == cudaSuccess
               ? Status::Ok()
               : CudaFailure("prepare GPU MTP device control", error);
  }

  [[nodiscard]] Status ValidateCompletedMtpGroup(
      std::uint32_t input_token, std::uint64_t start_position,
      std::uint32_t proposal_count, MtpGroupResult* host_result) const {
    if (host_result == nullptr) {
      return Error(StatusCode::kInvalidArgument,
                   "MTP host result is unavailable");
    }
    const auto* pinned_transaction =
        reinterpret_cast<const internal::MtpGroupTransaction*>(
            mtp_host_result_.span().data());
    const internal::MtpDeviceControl& control =
        pinned_transaction->control;
    const MtpGroupResult& result = pinned_transaction->result;
    if (control.transition_valid != 1U ||
        control.fixed_draft_tokens != mtp_draft_tokens_ ||
        control.proposal_count != proposal_count ||
        control.current.input_token != input_token ||
        control.current.processed_position + 1U != start_position ||
        result.output_count == 0U ||
        control.next.input_token !=
            result.verified[result.output_count - 1U] ||
        control.next.processed_position !=
            control.current.processed_position + result.output_count ||
        control.next.remaining_output_capacity !=
            control.current.remaining_output_capacity - result.output_count ||
        control.next.output_write_position !=
            control.current.output_write_position + result.output_count ||
        control.next.sampling_step !=
            control.current.sampling_step +
                (control.sampling_enabled != 0U ? result.output_count : 0U) ||
        control.next.stopped != result.stopped ||
        control.next.stop_token != result.stop_token) {
      return Error(StatusCode::kInternal,
                   "GPU MTP device-control transition disagrees with host reference");
    }
    *host_result = result;
    return Status::Ok();
  }

  [[nodiscard]] Status VerifyAcceptCommitAssistantBatch(
      std::uint32_t input_token, std::uint64_t start_position,
      std::uint32_t proposal_count, MtpGroupResult* host_result) {
    const NvtxRange range("gem16.mtp.verify_accept_commit");
    const std::uint64_t tokens = proposal_count + 1U;
    if (mtp_draft_tokens_ == 0U || proposal_count == 0U ||
        proposal_count > mtp_draft_tokens_ ||
        tokens > kMaximumMtpVerifyTokens || host_result == nullptr ||
        start_position >= max_context_ ||
        tokens > max_context_ - start_position) {
      return Error(StatusCode::kInvalidArgument,
                   "batched MTP verification extent is invalid");
    }
    const std::uint32_t* device_drafts = assistant_.device_draft_tokens();
    if (device_drafts == nullptr) {
      return Error(StatusCode::kInternal,
                   "assistant device draft storage is unavailable");
    }
    auto* device_tokens =
        Pointer<std::uint32_t>(prefill_workspace_, prefill_offsets_.token_ids);
    Status status = internal::LaunchBuildMtpVerificationInputs(
        input_token, device_drafts, proposal_count, device_tokens, stream_);
    if (!status.ok()) return status;
    cudaError_t error = cudaSuccess;
    float* hidden =
        Pointer<float>(prefill_workspace_, prefill_offsets_.hidden_a);
    const std::uint64_t hidden_elements = tokens * kHidden;
    EmbeddingBatchKernel<<<
        static_cast<unsigned>((hidden_elements + kThreads - 1U) / kThreads),
        kThreads, 0, stream_>>>(model_.embedding(), device_tokens, hidden,
                                hidden_elements);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return CudaFailure("launch MTP verification embedding", error);
    }
    for (std::size_t index = 0; index < model_.layers().size(); ++index) {
      status = RunLayerBatch(model_.layers()[index], start_position, tokens,
                             index);
      if (!status.ok()) return status;
    }
    float* normalized =
        Pointer<float>(prefill_workspace_, prefill_offsets_.normalized);
    status = internal::LaunchRmsNormBf16(
        hidden, model_.final_norm(), normalized, tokens, kHidden, kEpsilon,
        stream_);
    if (!status.ok()) return status;
    auto* candidates = Pointer<ArgmaxValue>(
        mtp_workspace_, mtp_offsets_.output_candidates);
    auto* selected =
        Pointer<std::uint32_t>(mtp_workspace_, mtp_offsets_.selected);
    float* sampling_logits =
        sampling_.enabled
            ? Pointer<float>(mtp_workspace_, mtp_offsets_.sampling_logits)
            : nullptr;
    status = internal::LaunchFusedOutputHeadBatchCandidates(
        model_.embedding(), normalized,
        Pointer<std::uint32_t>(workspace_, offsets_.suppressed),
        suppressed_token_count_, tokens, candidates, sampling_logits, stream_);
    if (!status.ok()) return status;
    if (sampling_.enabled) {
      auto* row_masks = Pointer<std::uint32_t>(
          mtp_workspace_, mtp_offsets_.sampling_repetition_masks);
      status = internal::LaunchBuildSpeculativeRepetitionMasks(
          Pointer<std::uint32_t>(workspace_, offsets_.repetition_mask),
          device_tokens, static_cast<std::uint32_t>(tokens),
          static_cast<std::uint32_t>(kRepetitionMaskWords), row_masks,
          stream_);
      if (!status.ok()) return status;
      for (std::uint64_t row = 0U; row < tokens; ++row) {
        status = SelectSampledTokenWithState(
            sampling_logits + row * kVocabulary, selected + row,
            row_masks + row * kRepetitionMaskWords, sampling_step_ + row);
        if (!status.ok()) return status;
      }
    } else {
      status = internal::LaunchOutputHeadBatchArgmax(candidates, tokens,
                                                     selected, stream_);
      if (!status.ok()) return status;
    }
    auto* device_transaction = Pointer<internal::MtpGroupTransaction>(
        mtp_workspace_, mtp_offsets_.transaction);
    auto* device_result = &device_transaction->result;
    auto* device_control = &device_transaction->control;
    status = internal::LaunchAcceptMtpGroup(
        device_drafts, selected, proposal_count,
        Pointer<std::uint32_t>(mtp_workspace_, mtp_offsets_.stop_tokens),
        mtp_stop_token_count_, device_result, device_control, stream_);
    if (!status.ok()) return status;
    if (sampling_.enabled) {
      status = internal::LaunchCommitSpeculativeRepetitionMask(
          Pointer<std::uint32_t>(
              mtp_workspace_, mtp_offsets_.sampling_repetition_masks),
          static_cast<std::uint32_t>(kRepetitionMaskWords),
          &device_result->output_count,
          Pointer<std::uint32_t>(workspace_, offsets_.repetition_mask),
          stream_);
      if (!status.ok()) return status;
    }
    for (std::size_t index = 0; index < model_.layers().size(); ++index) {
      const LayerBinding& layer = model_.layers()[index];
      const std::uint64_t capacity =
          layer.global ? max_context_ : std::min(max_context_, kSlidingWindow);
      if (kv_cache_mode_ == KvCacheMode::kCheckpointFp8) {
        status = internal::LaunchCommitMtpKvFp8(
            Pointer<std::uint8_t>(mtp_workspace_, mtp_offsets_.layers[index].key),
            Pointer<std::uint8_t>(mtp_workspace_, mtp_offsets_.layers[index].value),
            layer.key_cache_fp8, layer.value_cache_fp8, start_position,
            layer.kv_elements, capacity, device_result, stream_);
      } else {
        status = internal::LaunchCommitMtpKvBf16(
            Pointer<float>(mtp_workspace_, mtp_offsets_.layers[index].key),
            Pointer<float>(mtp_workspace_, mtp_offsets_.layers[index].value),
            layer.key_cache_bf16, layer.value_cache_bf16, start_position,
            layer.kv_elements, capacity, device_result, stream_);
      }
      if (!status.ok()) return status;
    }
    float* committed_hidden =
        Pointer<float>(mtp_workspace_, mtp_offsets_.committed_hidden);
    status = internal::LaunchCommitMtpHidden(
        normalized, committed_hidden, kHidden, device_result, stream_);
    if (!status.ok()) return status;
    latest_target_hidden_ = committed_hidden;
    auto* pinned_transaction =
        reinterpret_cast<internal::MtpGroupTransaction*>(
            mtp_host_result_.span().data());
    error = cudaMemcpyAsync(pinned_transaction, device_transaction,
                            sizeof(internal::MtpGroupTransaction),
                            cudaMemcpyDeviceToHost, stream_);
    if (error != cudaSuccess) {
      return CudaFailure("copy GPU MTP group result", error);
    }
    error = cudaStreamSynchronize(stream_);
    if (error != cudaSuccess) {
      return CudaFailure("synchronize GPU MTP group", error);
    }
    status = ValidateCompletedMtpGroup(input_token, start_position,
                                        proposal_count, host_result);
    if (status.ok() && sampling_.enabled) {
      sampling_step_ += host_result->output_count;
    }
    return status;
  }

  [[nodiscard]] Status LaunchControlledMtpD2GroupBody(
      bool copy_transaction = true) {
    constexpr std::uint32_t kProposalCount = 2U;
    constexpr std::uint64_t kTokens = 3U;
    auto* device_transaction = Pointer<internal::MtpGroupTransaction>(
        mtp_workspace_, mtp_offsets_.transaction);
    auto* device_control = &device_transaction->control;
    auto* row_controls = Pointer<internal::DecodeControl>(
        mtp_workspace_, mtp_offsets_.row_controls);

    const auto make_view = [this](const LayerBinding& layer) {
      internal::AssistantSharedKvView view;
      view.mode = internal::AssistantKvCacheMode::kCheckpointFp8;
      view.key_fp8 = layer.key_cache_fp8;
      view.value_fp8 = layer.value_cache_fp8;
      view.key_scale_bf16 = layer.k_cache_scale;
      view.value_scale_bf16 = layer.v_cache_scale;
      view.capacity =
          layer.global ? max_context_ : std::min(max_context_, kSlidingWindow);
      view.tokens = view.capacity;
      view.first_slot = 0U;
      view.kv_heads = layer.kv_heads;
      view.head_dimension = layer.head_dimension;
      return view;
    };
    const LayerBinding& sliding = model_.layers()[46U];
    const LayerBinding& full = model_.layers()[47U];
    internal::AssistantProposalContext assistant_context;
    assistant_context.target_embedding = model_.embedding();
    assistant_context.target_hidden = latest_target_hidden_;
    assistant_context.sliding_kv = make_view(sliding);
    assistant_context.full_kv = make_view(full);
    Status status = assistant_.GenerateDraftsDevice(
        assistant_context, kProposalCount, stream_, device_control);
    if (!status.ok()) return status;

    const std::uint32_t* device_drafts = assistant_.device_draft_tokens();
    auto* device_tokens =
        Pointer<std::uint32_t>(prefill_workspace_, prefill_offsets_.token_ids);
    status = internal::LaunchBuildControlledMtpD2Inputs(
        device_control, device_drafts, device_tokens, row_controls, stream_);
    if (!status.ok()) return status;
    float* hidden =
        Pointer<float>(prefill_workspace_, prefill_offsets_.hidden_a);
    constexpr std::uint64_t kHiddenElements = kTokens * kHidden;
    EmbeddingBatchKernel<<<
        static_cast<unsigned>((kHiddenElements + kThreads - 1U) / kThreads),
        kThreads, 0, stream_>>>(
        model_.embedding(), device_tokens, hidden, kHiddenElements);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      return CudaFailure("launch controlled MTP D2 embedding", error);
    }
    for (std::size_t index = 0; index < model_.layers().size(); ++index) {
      status = RunLayerBatch(model_.layers()[index], 0U, kTokens, index,
                             row_controls);
      if (!status.ok()) return status;
    }
    float* normalized =
        Pointer<float>(prefill_workspace_, prefill_offsets_.normalized);
    status = internal::LaunchRmsNormBf16(
        hidden, model_.final_norm(), normalized, kTokens, kHidden, kEpsilon,
        stream_);
    if (!status.ok()) return status;
    auto* candidates = Pointer<ArgmaxValue>(
        mtp_workspace_, mtp_offsets_.output_candidates);
    auto* selected =
        Pointer<std::uint32_t>(mtp_workspace_, mtp_offsets_.selected);
    float* sampling_logits =
        sampling_.enabled
            ? Pointer<float>(mtp_workspace_, mtp_offsets_.sampling_logits)
            : nullptr;
    status = internal::LaunchFusedOutputHeadBatchCandidates(
        model_.embedding(), normalized,
        Pointer<std::uint32_t>(workspace_, offsets_.suppressed),
        suppressed_token_count_, kTokens, candidates, sampling_logits,
        stream_);
    if (!status.ok()) return status;
    if (sampling_.enabled) {
      auto* row_masks = Pointer<std::uint32_t>(
          mtp_workspace_, mtp_offsets_.sampling_repetition_masks);
      status = internal::LaunchBuildSpeculativeRepetitionMasks(
          Pointer<std::uint32_t>(workspace_, offsets_.repetition_mask),
          device_tokens, static_cast<std::uint32_t>(kTokens),
          static_cast<std::uint32_t>(kRepetitionMaskWords), row_masks,
          stream_);
      if (!status.ok()) return status;
      for (std::uint64_t row = 0U; row < kTokens; ++row) {
        status = SelectSampledTokenWithState(
            sampling_logits + row * kVocabulary, selected + row,
            row_masks + row * kRepetitionMaskWords, 0U,
            row_controls + row);
        if (!status.ok()) return status;
      }
    } else {
      status = internal::LaunchOutputHeadBatchArgmax(
          candidates, kTokens, selected, stream_);
      if (!status.ok()) return status;
    }
    auto* device_result = &device_transaction->result;
    status = internal::LaunchAcceptMtpGroup(
        device_drafts, selected, kProposalCount,
        Pointer<std::uint32_t>(mtp_workspace_, mtp_offsets_.stop_tokens),
        mtp_stop_token_count_, device_result, device_control, stream_);
    if (!status.ok()) return status;
    if (sampling_.enabled) {
      status = internal::LaunchCommitSpeculativeRepetitionMask(
          Pointer<std::uint32_t>(
              mtp_workspace_, mtp_offsets_.sampling_repetition_masks),
          static_cast<std::uint32_t>(kRepetitionMaskWords),
          &device_result->output_count,
          Pointer<std::uint32_t>(workspace_, offsets_.repetition_mask),
          stream_);
      if (!status.ok()) return status;
    }
    for (std::size_t index = 0; index < model_.layers().size(); ++index) {
      const LayerBinding& layer = model_.layers()[index];
      const std::uint64_t capacity =
          layer.global ? max_context_ : std::min(max_context_, kSlidingWindow);
      status = internal::LaunchCommitMtpKvFp8ControlledD2(
          Pointer<std::uint8_t>(mtp_workspace_, mtp_offsets_.layers[index].key),
          Pointer<std::uint8_t>(mtp_workspace_,
                                mtp_offsets_.layers[index].value),
          layer.key_cache_fp8, layer.value_cache_fp8, layer.kv_elements,
          capacity, device_result, device_control, stream_);
      if (!status.ok()) return status;
    }
    float* committed_hidden =
        Pointer<float>(mtp_workspace_, mtp_offsets_.committed_hidden);
    status = internal::LaunchCommitMtpHidden(
        normalized, committed_hidden, kHidden, device_result, stream_);
    if (!status.ok()) return status;
    if (!copy_transaction) return Status::Ok();
    auto* pinned_transaction = reinterpret_cast<internal::MtpGroupTransaction*>(
        mtp_host_result_.span().data());
    error = cudaMemcpyAsync(
        pinned_transaction, device_transaction,
        sizeof(internal::MtpGroupTransaction), cudaMemcpyDeviceToHost,
        stream_);
    return error == cudaSuccess
               ? Status::Ok()
               : CudaFailure("copy controlled MTP D2 transaction", error);
  }

  [[nodiscard]] Status ExecuteFixedD2GraphGroup(
      std::uint32_t input_token, std::uint64_t start_position,
      MtpGroupResult* host_result) {
    const NvtxRange range("gem16.mtp.fixed_d2_graph");
    if (mtp_draft_tokens_ != 2U || latest_target_hidden_ == nullptr ||
        host_result == nullptr || start_position == 0U ||
        start_position >= max_context_ || 3U > max_context_ - start_position) {
      return Error(StatusCode::kInvalidArgument,
                   "fixed-D2 graph state is invalid");
    }
    if (kv_cache_mode_ != KvCacheMode::kCheckpointFp8 ||
        max_context_ <= kSlidingWindow) {
      Status status = GenerateAssistantDraftsDevice(
          input_token, start_position - 1U, 2U);
      if (!status.ok()) return status;
      return VerifyAcceptCommitAssistantBatch(
          input_token, start_position, 2U, host_result);
    }
    if (mtp_d2_graph_.get() == nullptr) {
      return Error(StatusCode::kInternal,
                   "fixed-D2 graph was not prepared before generation");
    }
    cudaError_t error = cudaGraphLaunch(mtp_d2_graph_.get(), stream_);
    if (error != cudaSuccess) {
      return CudaFailure("launch complete fixed-D2 group graph", error);
    }
    error = cudaStreamSynchronize(stream_);
    if (error != cudaSuccess) {
      return CudaFailure("synchronize complete fixed-D2 group graph", error);
    }
    return ValidateCompletedMtpGroup(input_token, start_position, 2U,
                                     host_result);
  }

  [[nodiscard]] Status PrepareFixedD2Graph() {
    if (mtp_draft_tokens_ != 2U ||
        kv_cache_mode_ != KvCacheMode::kCheckpointFp8 ||
        max_context_ <= kSlidingWindow) {
      return Status::Ok();
    }
    if (latest_target_hidden_ == nullptr) {
      return Error(StatusCode::kInternal,
                   "fixed-D2 graph requires a completed prefill");
    }
    float* committed_hidden =
        Pointer<float>(mtp_workspace_, mtp_offsets_.committed_hidden);
    cudaError_t error = cudaSuccess;
    if (latest_target_hidden_ != committed_hidden) {
      error = cudaMemcpyAsync(
          committed_hidden, latest_target_hidden_, kHidden * sizeof(float),
          cudaMemcpyDeviceToDevice, stream_);
      if (error != cudaSuccess) {
        return CudaFailure("stage initial fixed-D2 target hidden", error);
      }
      error = cudaStreamSynchronize(stream_);
      if (error != cudaSuccess) {
        return CudaFailure("synchronize fixed-D2 target hidden", error);
      }
      latest_target_hidden_ = committed_hidden;
    }
    if (mtp_d2_graph_.get() != nullptr) return Status::Ok();
    std::size_t free_before = 0U;
    std::size_t total_bytes = 0U;
    error = cudaMemGetInfo(&free_before, &total_bytes);
    if (error != cudaSuccess) {
      return CudaFailure("query memory before fixed-D2 graph", error);
    }
    Status status = CaptureDecodeGraph(
        mtp_d2_graph_,
        [this]() { return LaunchControlledMtpD2GroupBody(); },
        "capture complete fixed-D2 group graph");
    if (!status.ok()) return status;
    status = PrepareFixedD2ChainGraph();
    if (!status.ok()) return status;
    std::size_t free_after = 0U;
    error = cudaMemGetInfo(&free_after, &total_bytes);
    if (error != cudaSuccess) {
      return CudaFailure("query memory after fixed-D2 graph", error);
    }
    if (free_before > free_after) {
      decode_graph_device_bytes_ += free_before - free_after;
    }
    return Status::Ok();
  }

  [[nodiscard]] Status PrepareFixedD2ChainGraph() {
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
    cudaError_t error = cudaGraphCreate(&root, 0U);
    if (error != cudaSuccess) {
      return CudaFailure("create fixed-D2 chain graph", error);
    }
    cudaGraphConditionalHandle loop_condition = 0U;
    error = cudaGraphConditionalHandleCreate(
        &loop_condition, root, 1U, cudaGraphCondAssignDefault);
    if (error != cudaSuccess) {
      cleanup();
      return CudaFailure("create fixed-D2 chain condition", error);
    }
    cudaGraphConditionalHandle d2_condition = 0U;
    error = cudaGraphConditionalHandleCreate(
        &d2_condition, root, 0U, cudaGraphCondAssignDefault);
    if (error != cudaSuccess) {
      cleanup();
      return CudaFailure("create fixed-D2 branch condition", error);
    }
    cudaGraphConditionalHandle ordinary_condition = 0U;
    error = cudaGraphConditionalHandleCreate(
        &ordinary_condition, root, 0U, cudaGraphCondAssignDefault);
    if (error != cudaSuccess) {
      cleanup();
      return CudaFailure("create ordinary branch condition", error);
    }
    cudaGraphNodeParams loop_parameters{};
    loop_parameters.type = cudaGraphNodeTypeConditional;
    loop_parameters.conditional.handle = loop_condition;
    loop_parameters.conditional.type = cudaGraphCondTypeWhile;
    loop_parameters.conditional.size = 1U;
    loop_parameters.conditional.ctx = nullptr;
    cudaGraphNode_t loop_node = nullptr;
    error = cudaGraphAddNode(&loop_node, root, nullptr, nullptr, 0U,
                             &loop_parameters);
    if (error != cudaSuccess) {
      cleanup();
      return CudaFailure("add fixed-D2 chain conditional", error);
    }

    auto* transaction = Pointer<internal::MtpGroupTransaction>(
        mtp_workspace_, mtp_offsets_.transaction);
    auto* chain_result = Pointer<internal::MtpChainResult>(
        mtp_workspace_, mtp_offsets_.chain_result);
    auto* streaming_ring = static_cast<internal::MtpStreamingRing*>(
        mtp_stream_ring_.device_data());

    error = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal);
    if (error != cudaSuccess) {
      cleanup();
      return CudaFailure("begin MTP branch-route capture", error);
    }
    Status status = internal::LaunchSelectMtpChainBranch(
        transaction, d2_condition, ordinary_condition, stream_);
    error = cudaStreamEndCapture(stream_, &route_source);
    if (!status.ok() || error != cudaSuccess) {
      cleanup();
      return !status.ok() ? status
                          : CudaFailure("end MTP branch-route capture", error);
    }

    error = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal);
    if (error != cudaSuccess) {
      cleanup();
      return CudaFailure("begin fixed-D2 chain body capture", error);
    }
    status = LaunchControlledMtpD2GroupBody(false);
    if (status.ok()) {
      status = internal::LaunchAdvanceMtpD2Chain(
          transaction, chain_result,
          Pointer<std::uint32_t>(mtp_workspace_, mtp_offsets_.chain_outputs),
          Pointer<std::uint32_t>(mtp_workspace_,
                                 mtp_offsets_.chain_proposals),
          streaming_ring, stream_);
    }
    error = cudaStreamEndCapture(stream_, &d2_source);
    if (!status.ok() || error != cudaSuccess) {
      cleanup();
      return !status.ok()
                 ? status
                 : CudaFailure("end fixed-D2 chain body capture", error);
    }

    error = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal);
    if (error != cudaSuccess) {
      cleanup();
      return CudaFailure("begin MTP ordinary branch capture", error);
    }
    status = LaunchControlledMtpOrdinaryTailBody();
    error = cudaStreamEndCapture(stream_, &ordinary_source);
    if (!status.ok() || error != cudaSuccess) {
      cleanup();
      return !status.ok()
                 ? status
                 : CudaFailure("end MTP ordinary branch capture", error);
    }

    error = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal);
    if (error != cudaSuccess) {
      cleanup();
      return CudaFailure("begin MTP continuation capture", error);
    }
    status = internal::LaunchContinueMtpChain(
        transaction, streaming_ring, loop_condition, stream_);
    error = cudaStreamEndCapture(stream_, &continue_source);
    if (!status.ok() || error != cudaSuccess) {
      cleanup();
      return !status.ok() ? status
                          : CudaFailure("end MTP continuation capture", error);
    }

    cudaGraph_t loop_body = loop_parameters.conditional.phGraph_out[0];
    cudaGraphNode_t route_node = nullptr;
    error = cudaGraphAddChildGraphNode(&route_node, loop_body, nullptr, 0U,
                                       route_source);
    if (error != cudaSuccess) {
      cleanup();
      return CudaFailure("add MTP branch route", error);
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
      return CudaFailure("add MTP D2 branch", error);
    }
    cudaGraphNode_t d2_body_node = nullptr;
    error = cudaGraphAddChildGraphNode(
        &d2_body_node, d2_parameters.conditional.phGraph_out[0], nullptr, 0U,
        d2_source);
    if (error != cudaSuccess) {
      cleanup();
      return CudaFailure("add MTP D2 branch body", error);
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
      return CudaFailure("add MTP ordinary branch", error);
    }
    cudaGraphNode_t ordinary_body_node = nullptr;
    error = cudaGraphAddChildGraphNode(
        &ordinary_body_node, ordinary_parameters.conditional.phGraph_out[0],
        nullptr, 0U, ordinary_source);
    if (error != cudaSuccess) {
      cleanup();
      return CudaFailure("add MTP ordinary branch body", error);
    }

    const cudaGraphNode_t branch_nodes[] = {d2_node, ordinary_node};
    cudaGraphNode_t continue_node = nullptr;
    error = cudaGraphAddChildGraphNode(
        &continue_node, loop_body, branch_nodes, 2U, continue_source);
    if (error != cudaSuccess) {
      cleanup();
      return CudaFailure("add MTP continuation", error);
    }
    error = cudaGraphInstantiate(&executable, root, nullptr, nullptr, 0U);
    if (error != cudaSuccess) {
      cleanup();
      return CudaFailure("instantiate fixed-D2 chain graph", error);
    }
    mtp_d2_chain_graph_.Adopt(executable);
    executable = nullptr;
    cleanup();
    return Status::Ok();
  }

  [[nodiscard]] Status LaunchControlledMtpOrdinaryTailBody() {
    auto* transaction = Pointer<internal::MtpGroupTransaction>(
        mtp_workspace_, mtp_offsets_.transaction);
    auto* decode_control = Pointer<internal::DecodeControl>(
        workspace_, offsets_.decode_control);
    Status status = internal::LaunchInitializeMtpOrdinaryTail(
        transaction, decode_control, suppressed_token_count_, stream_);
    if (!status.ok()) return status;
    if (sampling_.enabled && sampling_.repetition_penalty != 1.0F) {
      status = internal::LaunchMarkControlledRepetitionToken(
          decode_control,
          Pointer<std::uint32_t>(workspace_, offsets_.repetition_mask),
          stream_);
      if (!status.ok()) return status;
    }
    float* hidden = Pointer<float>(workspace_, offsets_.hidden_a);
    ControlledEmbeddingKernel<<<
        static_cast<unsigned>((kHidden + kThreads - 1U) / kThreads),
        kThreads, 0, stream_>>>(model_.embedding(), decode_control, hidden);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      return CudaFailure("launch controlled MTP ordinary-tail embedding",
                         error);
    }
    for (const LayerBinding& layer : model_.layers()) {
      status = LaunchControlledDecodeLayer(layer);
      if (!status.ok()) return status;
    }
    float* normalized = Pointer<float>(workspace_, offsets_.normalized);
    status = internal::LaunchRmsNormBf16(
        hidden, model_.final_norm(), normalized, 1U, kHidden, kEpsilon,
        stream_);
    if (!status.ok()) return status;
    error = cudaMemcpyAsync(
        Pointer<float>(mtp_workspace_, mtp_offsets_.committed_hidden),
        normalized, kHidden * sizeof(float), cudaMemcpyDeviceToDevice,
        stream_);
    if (error != cudaSuccess) {
      return CudaFailure("stage ordinary-tail target hidden", error);
    }
    auto* selected = Pointer<std::uint32_t>(workspace_, offsets_.selected);
    float* sampling_logits =
        sampling_.enabled ? Pointer<float>(workspace_, offsets_.logits)
                          : nullptr;
    status = internal::LaunchFusedOutputHeadCandidates(
        model_.embedding(), normalized,
        Pointer<std::uint32_t>(workspace_, offsets_.suppressed), 0U,
        decode_control,
        Pointer<ArgmaxValue>(workspace_, offsets_.output_candidates),
        sampling_logits, stream_);
    if (!status.ok()) return status;
    if (sampling_.enabled) {
      status = SelectSampledToken(sampling_logits, selected, decode_control);
    } else {
      status = internal::LaunchOutputHeadCandidateArgmax(
          Pointer<ArgmaxValue>(workspace_, offsets_.output_candidates),
          selected, stream_);
    }
    if (!status.ok()) return status;
    return internal::LaunchFinalizeMtpOrdinaryTail(
        selected,
        Pointer<std::uint32_t>(mtp_workspace_, mtp_offsets_.stop_tokens),
        mtp_stop_token_count_, transaction,
        Pointer<internal::MtpChainResult>(
            mtp_workspace_, mtp_offsets_.chain_result),
        Pointer<std::uint32_t>(mtp_workspace_, mtp_offsets_.chain_outputs),
        static_cast<internal::MtpStreamingRing*>(
            mtp_stream_ring_.device_data()),
        stream_);
  }

  [[nodiscard]] Status ExecuteFixedD2GraphChain(
      internal::MtpChainResult* host_result,
      GeneratedTokenCallback callback = nullptr,
      void* callback_context = nullptr) {
    const NvtxRange range("gem16.mtp.fixed_d2_chain");
    if (host_result == nullptr || mtp_d2_chain_graph_.get() == nullptr) {
      return Error(StatusCode::kInvalidArgument,
                   "fixed-D2 chain graph is unavailable");
    }
    auto* device_result = Pointer<internal::MtpChainResult>(
        mtp_workspace_, mtp_offsets_.chain_result);
    auto* streaming_ring = reinterpret_cast<internal::MtpStreamingRing*>(
        mtp_stream_ring_.span().data());
    std::atomic_ref<unsigned long long> producer(streaming_ring->producer);
    std::atomic_ref<unsigned long long> consumer(streaming_ring->consumer);
    std::atomic_ref<unsigned long long> cancelled(streaming_ring->cancelled);
    std::atomic_ref<unsigned long long> backpressure(
        streaming_ring->backpressure_events);
    producer.store(0U, std::memory_order_relaxed);
    consumer.store(0U, std::memory_order_relaxed);
    cancelled.store(0U, std::memory_order_relaxed);
    backpressure.store(0U, std::memory_order_relaxed);
    cudaError_t error = cudaMemsetAsync(
        device_result, 0, sizeof(internal::MtpChainResult), stream_);
    if (error == cudaSuccess) {
      error = cudaGraphLaunch(mtp_d2_chain_graph_.get(), stream_);
    }
    if (error != cudaSuccess) {
      return CudaFailure("execute fixed-D2 chain graph", error);
    }
    Status callback_status = Status::Ok();
    unsigned long long consumed = 0U;
    const auto consume_available = [&]() {
      const unsigned long long published =
          producer.load(std::memory_order_acquire);
      while (consumed < published) {
        const std::uint32_t token = streaming_ring->tokens[
            consumed % internal::kMtpStreamingRingCapacity];
        if (callback != nullptr && callback_status.ok()) {
          callback_status = callback(callback_context, token);
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
      error = cudaStreamQuery(stream_);
      if (error == cudaSuccess) {
        consume_available();
        break;
      }
      if (error != cudaErrorNotReady) {
        return CudaFailure("poll fixed-D2 streaming graph", error);
      }
      std::this_thread::yield();
    }
    auto* host_bytes = reinterpret_cast<std::byte*>(
        mtp_host_chain_.span().data());
    auto* pinned_result =
        reinterpret_cast<internal::MtpChainResult*>(host_bytes);
    error = cudaMemcpy(pinned_result, device_result,
                       sizeof(internal::MtpChainResult),
                       cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
      return CudaFailure("copy fixed-D2 chain result", error);
    }
    if (pinned_result->proposed_count != 2U * pinned_result->group_count ||
        pinned_result->accepted_count + pinned_result->rejected_count !=
            pinned_result->proposed_count ||
        pinned_result->output_count > max_context_ ||
        pinned_result->proposed_count > 2U * max_context_) {
      return Error(StatusCode::kInternal,
                   "fixed-D2 chain result is inconsistent");
    }
    if (callback_status.ok() && consumed != pinned_result->output_count) {
      return Error(StatusCode::kInternal,
                   "fixed-D2 streaming token count is inconsistent");
    }
    auto* pinned_outputs = reinterpret_cast<std::uint32_t*>(
        host_bytes + sizeof(internal::MtpChainResult));
    auto* pinned_proposals = pinned_outputs + max_context_;
    const auto* transaction = reinterpret_cast<const internal::MtpGroupTransaction*>(
        mtp_host_result_.span().data());
    const std::uint64_t output_begin =
        transaction->control.current.output_write_position;
    error = cudaMemcpy(
        pinned_outputs,
        Pointer<std::uint32_t>(mtp_workspace_, mtp_offsets_.chain_outputs) +
            output_begin,
        static_cast<std::size_t>(pinned_result->output_count) *
            sizeof(std::uint32_t),
        cudaMemcpyDeviceToHost);
    if (error == cudaSuccess) {
      error = cudaMemcpy(
          pinned_proposals,
          Pointer<std::uint32_t>(mtp_workspace_, mtp_offsets_.chain_proposals),
          static_cast<std::size_t>(pinned_result->proposed_count) *
              sizeof(std::uint32_t),
          cudaMemcpyDeviceToHost);
    }
    auto* pinned_transaction = reinterpret_cast<internal::MtpGroupTransaction*>(
        mtp_host_result_.span().data());
    if (error == cudaSuccess) {
      error = cudaMemcpy(
          pinned_transaction,
          Pointer<internal::MtpGroupTransaction>(
              mtp_workspace_, mtp_offsets_.transaction),
          sizeof(internal::MtpGroupTransaction), cudaMemcpyDeviceToHost);
    }
    if (error != cudaSuccess) {
      return CudaFailure("copy fixed-D2 chain payload", error);
    }
    const internal::MtpDeviceState& final_state =
        pinned_transaction->control.current;
    if (final_state.output_write_position !=
            output_begin + pinned_result->output_count ||
        final_state.sampling_step !=
            sampling_step_ +
                (sampling_.enabled ? pinned_result->output_count : 0U) ||
        final_state.stopped != pinned_result->stopped ||
        final_state.stop_token != pinned_result->stop_token ||
        pinned_transaction->control.reasoning.reasoning_token_count !=
            pinned_result->reasoning_token_count ||
        pinned_transaction->control.reasoning.complete !=
            pinned_result->reasoning_complete ||
        pinned_transaction->control.reasoning.budget_forced !=
            pinned_result->reasoning_budget_forced) {
      return Error(StatusCode::kInternal,
                   "fixed-D2 chain final state is inconsistent");
    }
    sampling_step_ = final_state.sampling_step;
    *host_result = *pinned_result;
    return callback_status;
  }

  [[nodiscard]] const std::uint32_t* mtp_chain_outputs() const {
    const auto* host_bytes = reinterpret_cast<const std::byte*>(
        mtp_host_chain_.span().data());
    return reinterpret_cast<const std::uint32_t*>(
        host_bytes + sizeof(internal::MtpChainResult));
  }

  [[nodiscard]] const std::uint32_t* mtp_chain_proposals() const {
    return mtp_chain_outputs() + max_context_;
  }

  [[nodiscard]] Status CheckMtpDeviceControlParity(
      std::uint32_t input_token, std::uint64_t processed_position,
      std::uint64_t remaining_output_capacity,
      std::uint64_t output_write_position, bool stopped,
      std::uint32_t stop_token) const {
    const auto* pinned_transaction =
        reinterpret_cast<const internal::MtpGroupTransaction*>(
            mtp_host_result_.span().data());
    const internal::MtpDeviceState& next =
        pinned_transaction->control.next;
    if (next.input_token != input_token ||
        next.processed_position != processed_position ||
        next.remaining_output_capacity != remaining_output_capacity ||
        next.output_write_position != output_write_position ||
        next.sampling_step != sampling_step_ ||
        next.stopped != (stopped ? 1U : 0U) ||
        next.stop_token != stop_token) {
      return Error(StatusCode::kInternal,
                   "host MTP state disagrees with GPU device control");
    }
    return Status::Ok();
  }

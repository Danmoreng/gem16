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
      std::uint32_t stop_token) {
    if (mtp_draft_tokens_ == 0U || remaining_output_capacity == 0U ||
        processed_position >= max_context_ ||
        remaining_output_capacity > max_context_ - processed_position - 1U) {
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
    pinned_transaction->control.current.stopped = stopped ? 1U : 0U;
    pinned_transaction->control.current.stop_token = stop_token;
    pinned_transaction->control.next = pinned_transaction->control.current;
    pinned_transaction->control.fixed_draft_tokens = mtp_draft_tokens_;
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
    status = internal::LaunchFusedOutputHeadBatchCandidates(
        model_.embedding(), normalized,
        Pointer<std::uint32_t>(workspace_, offsets_.suppressed),
        suppressed_token_count_, tokens, candidates, stream_);
    if (!status.ok()) return status;
    status = internal::LaunchOutputHeadBatchArgmax(candidates, tokens,
                                                   selected, stream_);
    if (!status.ok()) return status;
    auto* device_transaction = Pointer<internal::MtpGroupTransaction>(
        mtp_workspace_, mtp_offsets_.transaction);
    auto* device_result = &device_transaction->result;
    auto* device_control = &device_transaction->control;
    status = internal::LaunchAcceptMtpGroup(
        device_drafts, selected, proposal_count,
        Pointer<std::uint32_t>(mtp_workspace_, mtp_offsets_.stop_tokens),
        mtp_stop_token_count_, device_result, device_control, stream_);
    if (!status.ok()) return status;
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
    return ValidateCompletedMtpGroup(input_token, start_position,
                                     proposal_count, host_result);
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
    status = internal::LaunchFusedOutputHeadBatchCandidates(
        model_.embedding(), normalized,
        Pointer<std::uint32_t>(workspace_, offsets_.suppressed),
        suppressed_token_count_, kTokens, candidates, stream_);
    if (!status.ok()) return status;
    status = internal::LaunchOutputHeadBatchArgmax(
        candidates, kTokens, selected, stream_);
    if (!status.ok()) return status;
    auto* device_result = &device_transaction->result;
    status = internal::LaunchAcceptMtpGroup(
        device_drafts, selected, kProposalCount,
        Pointer<std::uint32_t>(mtp_workspace_, mtp_offsets_.stop_tokens),
        mtp_stop_token_count_, device_result, device_control, stream_);
    if (!status.ok()) return status;
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
    cudaGraph_t body_source = nullptr;
    cudaGraphExec_t executable = nullptr;
    const auto cleanup = [&]() {
      if (body_source != nullptr) (void)cudaGraphDestroy(body_source);
      if (root != nullptr) (void)cudaGraphDestroy(root);
      if (executable != nullptr) (void)cudaGraphExecDestroy(executable);
    };
    cudaError_t error = cudaGraphCreate(&root, 0U);
    if (error != cudaSuccess) {
      return CudaFailure("create fixed-D2 chain graph", error);
    }
    cudaGraphConditionalHandle condition = 0U;
    error = cudaGraphConditionalHandleCreate(
        &condition, root, 1U, cudaGraphCondAssignDefault);
    if (error != cudaSuccess) {
      cleanup();
      return CudaFailure("create fixed-D2 chain condition", error);
    }
    cudaGraphNodeParams parameters{};
    parameters.type = cudaGraphNodeTypeConditional;
    parameters.conditional.handle = condition;
    parameters.conditional.type = cudaGraphCondTypeWhile;
    parameters.conditional.size = 1U;
    parameters.conditional.ctx = nullptr;
    cudaGraphNode_t conditional_node = nullptr;
    error = cudaGraphAddNode(&conditional_node, root, nullptr, nullptr, 0U,
                             &parameters);
    if (error != cudaSuccess) {
      cleanup();
      return CudaFailure("add fixed-D2 chain conditional", error);
    }
    error = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal);
    if (error != cudaSuccess) {
      cleanup();
      return CudaFailure("begin fixed-D2 chain body capture", error);
    }
    Status status = LaunchControlledMtpD2GroupBody(false);
    if (status.ok()) {
      status = internal::LaunchAdvanceMtpD2Chain(
          Pointer<internal::MtpGroupTransaction>(
              mtp_workspace_, mtp_offsets_.transaction),
          Pointer<internal::MtpChainResult>(
              mtp_workspace_, mtp_offsets_.chain_result),
          Pointer<std::uint32_t>(mtp_workspace_, mtp_offsets_.chain_outputs),
          Pointer<std::uint32_t>(mtp_workspace_,
                                 mtp_offsets_.chain_proposals),
          condition, stream_);
    }
    error = cudaStreamEndCapture(stream_, &body_source);
    if (!status.ok() || error != cudaSuccess) {
      cleanup();
      return !status.ok()
                 ? status
                 : CudaFailure("end fixed-D2 chain body capture", error);
    }
    cudaGraphNode_t body_node = nullptr;
    error = cudaGraphAddChildGraphNode(
        &body_node, parameters.conditional.phGraph_out[0], nullptr, 0U,
        body_source);
    if (error != cudaSuccess) {
      cleanup();
      return CudaFailure("add fixed-D2 chain body", error);
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

  [[nodiscard]] Status ExecuteFixedD2GraphChain(
      internal::MtpChainResult* host_result) {
    const NvtxRange range("gem16.mtp.fixed_d2_chain");
    if (host_result == nullptr || mtp_d2_chain_graph_.get() == nullptr) {
      return Error(StatusCode::kInvalidArgument,
                   "fixed-D2 chain graph is unavailable");
    }
    auto* device_result = Pointer<internal::MtpChainResult>(
        mtp_workspace_, mtp_offsets_.chain_result);
    cudaError_t error = cudaMemsetAsync(
        device_result, 0, sizeof(internal::MtpChainResult), stream_);
    if (error == cudaSuccess) {
      error = cudaGraphLaunch(mtp_d2_chain_graph_.get(), stream_);
    }
    if (error == cudaSuccess) error = cudaStreamSynchronize(stream_);
    if (error != cudaSuccess) {
      return CudaFailure("execute fixed-D2 chain graph", error);
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
    if (pinned_result->group_count == 0U ||
        pinned_result->proposed_count != 2U * pinned_result->group_count ||
        pinned_result->accepted_count + pinned_result->rejected_count !=
            pinned_result->proposed_count ||
        pinned_result->output_count > max_context_ ||
        pinned_result->proposed_count > 2U * max_context_) {
      return Error(StatusCode::kInternal,
                   "fixed-D2 chain result is inconsistent");
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
        final_state.stopped != pinned_result->stopped ||
        final_state.stop_token != pinned_result->stop_token) {
      return Error(StatusCode::kInternal,
                   "fixed-D2 chain final state is inconsistent");
    }
    *host_result = *pinned_result;
    return Status::Ok();
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
        next.stopped != (stopped ? 1U : 0U) ||
        next.stop_token != stop_token) {
      return Error(StatusCode::kInternal,
                   "host MTP state disagrees with GPU device control");
    }
    return Status::Ok();
  }

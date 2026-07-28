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

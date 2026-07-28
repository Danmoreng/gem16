  [[nodiscard]] Result<std::uint32_t> Forward(
      std::uint32_t token, std::uint64_t position, bool select_token,
      std::span<float> host_logits = {}, std::span<float> host_state = {}) {
    const NvtxRange range("gem16.decode.forward");
    if (token >= kVocabulary || position >= max_context_) {
      return Error(StatusCode::kInvalidArgument, "token or position exceeds inference plan");
    }
    const StateCaptureLayout state_layout = MakeStateCaptureLayout();
    if (!host_state.empty() && host_state.size() != state_layout.elements) {
      return Error(StatusCode::kInternal, "host state capture span has invalid size");
    }
    if (select_token && host_logits.empty() && host_state.empty()) {
      HostDecodeState* host = host_decode_state();
      host->control.token = token;
      host->control.position = position;
      if (sampling_.enabled) host->control.sampling_step = sampling_step_++;
      const cudaError_t launch_error =
          cudaGraphLaunch(full_decode_graph_.get(), stream_);
      if (launch_error != cudaSuccess) {
        return CudaFailure("launch full decode graph", launch_error);
      }
      const cudaError_t sync_error = cudaStreamSynchronize(stream_);
      if (sync_error != cudaSuccess) {
        return CudaFailure("synchronize full decode graph", sync_error);
      }
      latest_target_hidden_ = Pointer<float>(workspace_, offsets_.normalized);
      return host->selected_token;
    }
    Status mark_status = MarkRepetitionToken(token);
    if (!mark_status.ok()) return mark_status;
    float* hidden_a = Pointer<float>(workspace_, offsets_.hidden_a);
    EmbeddingKernel<<<static_cast<unsigned>((kHidden + kThreads - 1U) / kThreads), kThreads,
                      0, stream_>>>(model_.embedding(), token, hidden_a);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return CudaFailure("launch embedding", error);

    for (std::size_t layer_index = 0; layer_index < model_.layers().size();
         ++layer_index) {
      const auto& layer = model_.layers()[layer_index];
      const LayerStateCapture* layer_capture =
          host_state.empty() ? nullptr : &state_layout.layers[layer_index];
      Status status =
          RunLayer(layer_index, layer, position, layer_capture, host_state.data());
      if (!status.ok()) return status;
    }
    float* normalized = Pointer<float>(workspace_, offsets_.normalized);
    Status status = internal::LaunchRmsNormBf16(hidden_a, model_.final_norm(), normalized, 1U,
                                            kHidden, kEpsilon, stream_);
    if (!status.ok()) return status;
    latest_target_hidden_ = normalized;
    if (!select_token) {
      if (!host_state.empty()) {
        error = cudaStreamSynchronize(stream_);
        if (error != cudaSuccess) {
          return CudaFailure("synchronize layer state capture", error);
        }
      }
      return 0U;
    }

    auto* selected = Pointer<std::uint32_t>(workspace_, offsets_.selected);
    float* diagnostic_logits = nullptr;
    if (!host_logits.empty()) {
      if (host_logits.size() != kVocabulary) {
        return Error(StatusCode::kInternal,
                     "host logit capture span has invalid size");
      }
      diagnostic_logits = Pointer<float>(workspace_, offsets_.logits);
    }
    if (sampling_.enabled) {
      diagnostic_logits = Pointer<float>(workspace_, offsets_.logits);
      status = internal::LaunchFusedOutputHeadCandidates(
          model_.embedding(), normalized,
          Pointer<std::uint32_t>(workspace_, offsets_.suppressed),
          suppressed_token_count_, nullptr,
          Pointer<ArgmaxValue>(workspace_, offsets_.output_candidates),
          diagnostic_logits, stream_);
      if (!status.ok()) return status;
      if (!host_logits.empty()) {
        error = cudaMemcpyAsync(host_logits.data(), diagnostic_logits,
                                host_logits.size_bytes(),
                                cudaMemcpyDeviceToHost, stream_);
        if (error != cudaSuccess) {
          return CudaFailure("copy sampled full logits", error);
        }
      }
      status = SelectSampledToken(diagnostic_logits, selected);
      if (!status.ok()) return status;
    } else {
      status = internal::LaunchFusedOutputHeadCandidates(
          model_.embedding(), normalized,
          Pointer<std::uint32_t>(workspace_, offsets_.suppressed),
          suppressed_token_count_, nullptr,
          Pointer<ArgmaxValue>(workspace_, offsets_.output_candidates),
          diagnostic_logits, stream_);
      if (!status.ok()) return status;
      if (diagnostic_logits != nullptr) {
        error = cudaMemcpyAsync(host_logits.data(), diagnostic_logits,
                                host_logits.size_bytes(),
                                cudaMemcpyDeviceToHost, stream_);
        if (error != cudaSuccess) {
          return CudaFailure("copy fused full logits", error);
        }
      }
      status = internal::LaunchOutputHeadCandidateArgmax(
          Pointer<ArgmaxValue>(workspace_, offsets_.output_candidates),
          selected, stream_);
      if (!status.ok()) return status;
    }
    std::uint32_t host_token = 0;
    error = cudaMemcpyAsync(&host_token, selected, sizeof(host_token), cudaMemcpyDeviceToHost,
                            stream_);
    if (error != cudaSuccess) return CudaFailure("copy selected token", error);
    error = cudaStreamSynchronize(stream_);
    if (error != cudaSuccess) return CudaFailure("synchronize selected token", error);
    return host_token;
  }

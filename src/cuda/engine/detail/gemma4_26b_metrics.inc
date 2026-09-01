Status Gemma4Moe26BReferenceEngine::CopyLogits(std::span<float> output) {
  if (!implementation_ || output.size() != kVocabulary) {
    return Invalid("M13 logit destination has the wrong extent");
  }
  cudaError_t error = cudaStreamSynchronize(implementation_->stream);
  if (error == cudaSuccess) {
    error = cudaMemcpy(output.data(), implementation_->logits,
                       output.size_bytes(), cudaMemcpyDeviceToHost);
  }
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("copy M13 logits", error);
}

Status Gemma4Moe26BReferenceEngine::CopyLayerOutput(
    std::uint32_t layer, std::span<float> output) {
  const int index = CaptureIndex(
      layer, implementation_ && implementation_->quality_capture_all_layers);
  if (!implementation_ || index < 0 || output.size() != kWidth) {
    return Invalid("M13 layer capture request is invalid");
  }
  cudaError_t error = cudaStreamSynchronize(implementation_->stream);
  if (error == cudaSuccess) {
    error = cudaMemcpy(output.data(),
                       implementation_->layer_captures[static_cast<std::size_t>(index)],
                       output.size_bytes(), cudaMemcpyDeviceToHost);
  }
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("copy M13 layer output", error);
}

Status Gemma4Moe26BReferenceEngine::CopyRouterProbabilities(
    std::uint32_t layer, std::span<float> output) {
  const int index = CaptureIndex(
      layer, implementation_ && implementation_->quality_capture_all_layers);
  if (!implementation_ || index < 0 || output.size() != kExperts) {
    return Invalid("M13 router probability capture request is invalid");
  }
  cudaError_t error = cudaStreamSynchronize(implementation_->stream);
  if (error == cudaSuccess) {
    error = cudaMemcpy(
        output.data(),
        implementation_->router_probability_captures[static_cast<std::size_t>(index)],
        output.size_bytes(), cudaMemcpyDeviceToHost);
  }
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("copy M13 router probabilities", error);
}

Status Gemma4Moe26BReferenceEngine::CopyRouterTopIds(
    std::uint32_t layer, std::span<std::uint32_t> output) {
  const int index = CaptureIndex(
      layer, implementation_ && implementation_->quality_capture_all_layers);
  if (!implementation_ || index < 0 || output.size() != kTopK) {
    return Invalid("M13 router ID capture request is invalid");
  }
  cudaError_t error = cudaStreamSynchronize(implementation_->stream);
  if (error == cudaSuccess) {
    error = cudaMemcpy(
        output.data(),
        implementation_->router_id_captures[static_cast<std::size_t>(index)],
        output.size_bytes(), cudaMemcpyDeviceToHost);
  }
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("copy M13 router IDs", error);
}

Status Gemma4Moe26BReferenceEngine::ConfigureMoeCalibrationCapture(
    std::uint32_t layer) {
  if (!implementation_ || layer >= kLayers) {
    return Invalid("Trellis35 calibration layer is invalid");
  }
  auto& x = *implementation_;
  if (x.backend != Gemma4Moe26BBackend::kReference) {
    return Invalid("Trellis35 calibration capture requires the reference backend");
  }
  if (x.has_last_input_token || x.position != 0U) {
    return Invalid("Trellis35 calibration capture must be configured before execution");
  }
  try {
    x.calibration_gate_up_input.assign(kWidth, 0.0F);
    x.calibration_down_inputs.assign(kTopK * kExpert, 0.0F);
    x.calibration_expert_ids.assign(kTopK, 0U);
  } catch (const std::bad_alloc&) {
    return Status(StatusCode::kResourceExhausted,
                  "cannot allocate Trellis35 calibration host buffers");
  }
  x.calibration_capture_layer = layer;
  x.calibration_capture_enabled = true;
  x.calibration_capture_valid = false;
  return Status::Ok();
}

Status Gemma4Moe26BReferenceEngine::CopyMoeCalibrationCapture(
    std::span<float> gate_up_input, std::span<float> routed_down_inputs,
    std::span<std::uint32_t> routed_expert_ids) {
  if (!implementation_ || gate_up_input.size() != kWidth ||
      routed_down_inputs.size() != kTopK * kExpert ||
      routed_expert_ids.size() != kTopK) {
    return Invalid("Trellis35 calibration destinations have the wrong extent");
  }
  auto& x = *implementation_;
  if (!x.calibration_capture_enabled || !x.calibration_capture_valid) {
    return Invalid("Trellis35 calibration capture is not available");
  }
  const cudaError_t error = cudaStreamSynchronize(x.stream);
  if (error != cudaSuccess) {
    return CudaFailure("synchronize Trellis35 calibration capture", error);
  }
  std::copy(x.calibration_gate_up_input.begin(),
            x.calibration_gate_up_input.end(), gate_up_input.begin());
  std::copy(x.calibration_down_inputs.begin(), x.calibration_down_inputs.end(),
            routed_down_inputs.begin());
  std::copy(x.calibration_expert_ids.begin(), x.calibration_expert_ids.end(),
            routed_expert_ids.begin());
  return Status::Ok();
}

std::uint64_t Gemma4Moe26BReferenceEngine::position() const {
  return implementation_ ? implementation_->position : 0U;
}
std::uint64_t Gemma4Moe26BReferenceEngine::context_capacity() const {
  return implementation_ ? implementation_->context : 0U;
}
bool Gemma4Moe26BReferenceEngine::vision_module_loaded() const {
  return implementation_ != nullptr &&
         implementation_->vision_artifact.arena() != nullptr;
}
std::uint64_t Gemma4Moe26BReferenceEngine::vision_weight_bytes() const {
  return implementation_ ? implementation_->vision_artifact.arena_bytes() : 0U;
}
std::uint64_t Gemma4Moe26BReferenceEngine::vision_workspace_bytes() const {
  return implementation_ ? implementation_->vision_runtime.workspace_bytes()
                         : 0U;
}

Result<Gemma4Moe26BVisionPhaseTimings>
Gemma4Moe26BReferenceEngine::ResolveVisionPhaseTimings() {
  if (!implementation_ || !implementation_->vision_phase_pending) {
    return Invalid("Vision phase timings are not pending");
  }
  const cudaError_t sync =
      cudaEventSynchronize(implementation_->vision_phase_events[4]);
  if (sync != cudaSuccess) {
    return CudaFailure("synchronize Vision phase timing", sync);
  }
  const auto elapsed = [this](std::size_t begin,
                              std::size_t end) -> Result<double> {
    float milliseconds = 0.0F;
    const cudaError_t error = cudaEventElapsedTime(
        &milliseconds, implementation_->vision_phase_events[begin],
        implementation_->vision_phase_events[end]);
    if (error != cudaSuccess) {
      return CudaFailure("resolve Vision phase timing", error);
    }
    return static_cast<double>(milliseconds);
  };
  Gemma4Moe26BVisionPhaseTimings result;
  auto upload = elapsed(0U, 1U);
  if (!upload.ok()) return upload.status();
  auto tower = elapsed(1U, 2U);
  if (!tower.ok()) return tower.status();
  auto pool_project = elapsed(2U, 3U);
  if (!pool_project.ok()) return pool_project.status();
  auto text_prefill = elapsed(3U, 4U);
  if (!text_prefill.ok()) return text_prefill.status();
  result.upload_milliseconds = upload.value();
  result.tower_milliseconds = tower.value();
  result.pool_project_milliseconds = pool_project.value();
  result.text_prefill_milliseconds = text_prefill.value();
  implementation_->vision_phase_pending = false;
  return result;
}
std::uint64_t Gemma4Moe26BReferenceEngine::weight_arena_bytes() const {
  return implementation_
             ? (IsTrellis35RoutedExpertFormat(
                    implementation_->routed_expert_format)
                    ? implementation_->trellis35_artifact.arena_bytes()
                    : implementation_->artifact.arena_bytes())
             : 0U;
}
const char* Gemma4Moe26BReferenceEngine::weight_load_path() const {
  return implementation_
             ? (IsTrellis35RoutedExpertFormat(
                    implementation_->routed_expert_format)
                    ? implementation_->trellis35_artifact.stats()
                          .load_path.c_str()
                    : implementation_->artifact.stats().load_path.c_str())
             : "none";
}
std::uint64_t Gemma4Moe26BReferenceEngine::kv_cache_bytes() const {
  return implementation_ ? implementation_->kv.bytes() : 0U;
}
std::uint64_t Gemma4Moe26BReferenceEngine::workspace_bytes() const {
  return implementation_ ? implementation_->workspace.bytes() +
                               implementation_->prefill_workspace.bytes() +
                               implementation_->mtp_verifier.bytes() +
                               implementation_->trellis35_workspace.bytes() +
                               implementation_->vision_runtime.workspace_bytes()
                         : 0U;
}
std::uint64_t Gemma4Moe26BReferenceEngine::sliding_cache_capacity() const {
  return implementation_ ? implementation_->sliding_capacity : 0U;
}
std::uint64_t Gemma4Moe26BReferenceEngine::prefill_chunk_count() const {
  return implementation_ ? implementation_->prefill_chunks : 0U;
}
std::uint64_t
Gemma4Moe26BReferenceEngine::minimum_prefill_chunk_tokens() const {
  return implementation_ ? implementation_->minimum_prefill_chunk : 0U;
}
Gemma4Moe26BExecutionEvidence
Gemma4Moe26BReferenceEngine::execution_evidence() const {
  Gemma4Moe26BExecutionEvidence result;
  if (!implementation_) return result;
  result.integrated_native_backend =
      implementation_->backend == Gemma4Moe26BBackend::kSm120Integrated;
  result.decode_graph_ready = implementation_->decode_graph != nullptr;
  result.tensor_core_prefill_router =
      implementation_->moe_config.prefill_router ==
      Gemma4MoePrefillRouter::kSm120TensorCore;
  result.prefill_calls = implementation_->prefill_calls;
  result.prefill_chunks = implementation_->prefill_chunks;
  result.decode_graph_launches = implementation_->decode_graph_launches;
  result.token_selections = implementation_->token_selections;
  result.sliding_ring_wraps = implementation_->sliding_ring_wraps;
  result.maximum_global_position_exclusive =
      implementation_->maximum_global_position_exclusive;
  result.fallback_count = implementation_->fallback_count;
  result.recurring_allocation_count =
      implementation_->recurring_allocation_count;
  return result;
}

}  // namespace gem16::internal

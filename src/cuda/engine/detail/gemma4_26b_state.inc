struct Gemma4Moe26BReferenceEngine::Impl {
  int device = 0;
  std::uint64_t context = 0U;
  std::uint64_t position = 0U;
  std::uint64_t sliding_capacity = 0U;
  std::uint64_t prefill_chunks = 0U;
  std::uint64_t minimum_prefill_chunk = 0U;
  std::uint64_t prefill_calls = 0U;
  std::uint64_t decode_graph_launches = 0U;
  std::uint64_t token_selections = 0U;
  std::uint64_t sliding_ring_wraps = 0U;
  std::uint64_t maximum_global_position_exclusive = 0U;
  std::uint64_t fallback_count = 0U;
  std::uint64_t recurring_allocation_count = 0U;
  bool pending_decode_self_feed = false;
  bool decode_self_feed_valid = false;
  bool has_last_input_token = false;
  std::uint32_t last_input_token = 0U;
  std::uint32_t decode_self_feed_token = 0U;
  std::uint64_t decode_self_feed_position = 0U;
  Gemma4Moe26BBackend backend = Gemma4Moe26BBackend::kReference;
  Gemma4Moe26BMtpVerifierBackend mtp_verifier_backend =
      Gemma4Moe26BMtpVerifierBackend::kExactDecodeParent;
  cudaStream_t stream = nullptr;
  cudaStream_t shared_moe_stream = nullptr;
  cudaEvent_t shared_moe_fork = nullptr;
  cudaEvent_t shared_moe_join = nullptr;
  Gemma4Moe26BDeviceArtifact artifact;
  Gemma4Moe26BTrellis35DeviceArtifact trellis35_artifact;
  Gemma4Moe26BRoutedExpertFormat routed_expert_format =
      Gemma4Moe26BRoutedExpertFormat::kNvfp4;
  Gemma4Moe26BAssistantModel assistant;
  DeviceBuffer mtp_verifier;
  DeviceBuffer mtp_router_overlap;
  MtpVerifierHostResult* mtp_verifier_host = nullptr;
  std::array<cudaGraphExec_t, kMaximumMtpDraftTokens + 1U>
      mtp_group_graphs{};
  std::array<cudaGraphExec_t, kMaximumMtpDraftTokens + 1U>
      mtp_chain_graphs{};
  std::uint64_t mtp_group_graph_device_bytes = 0U;
  std::array<std::uint64_t, kMaximumMtpDraftTokens + 1U>
      mtp_chain_graph_device_bytes{};
  std::uint64_t mtp_group_graph_launches = 0U;
  std::array<std::uint64_t, kMaximumMtpDraftTokens + 1U>
      mtp_chain_graph_launches{};
  std::uint64_t mtp_normalized_offset = 0U;
  std::uint64_t mtp_head_activation_offset = 0U;
  std::uint64_t mtp_head_scales_offset = 0U;
  std::uint64_t mtp_logits_offset = 0U;
  std::uint64_t mtp_selected_offset = 0U;
  std::uint64_t mtp_second_selected_offset = 0U;
  std::uint64_t mtp_top_values_offset = 0U;
  std::uint64_t mtp_second_values_offset = 0U;
  std::uint64_t mtp_finite_offset = 0U;
  std::uint64_t mtp_transaction_offset = 0U;
  std::uint64_t mtp_row_controls_offset = 0U;
  std::uint64_t mtp_sampling_repetition_masks_offset = 0U;
  std::uint64_t mtp_stop_tokens_offset = 0U;
  std::uint64_t mtp_chain_result_offset = 0U;
  std::uint64_t mtp_chain_outputs_offset = 0U;
  std::uint64_t mtp_chain_proposals_offset = 0U;
  std::byte* mtp_chain_host = nullptr;
  MtpStreamingRing* mtp_streaming_ring_host = nullptr;
  MtpStreamingRing* mtp_streaming_ring_device = nullptr;
  std::uint32_t mtp_stop_token_count = 0U;
  std::array<MtpVerifierLayerOffsets, kLayers> mtp_layer_offsets{};
  float last_mtp_verification_min_margin = 0.0F;
  Gemma4Moe26BAttentionTraits traits{};
  std::array<Gemma4Moe26BAttentionReferenceWeights, kLayers> attention_weights{};
  std::array<Gemma4MoeReferenceWeights, kLayers> moe_weights{};
  std::array<Gemma4Moe26BKvCacheView, kLayers> caches{};
  Gemma4Moe26BAttentionReferenceWorkspace attention_workspace{};
  Gemma4MoeReferenceWorkspace moe_workspace{};
  Gemma4MoeReferenceConfig moe_config{kWidth, kShared, kExpert, kExperts,
                                      kTopK, 1.0e-6F};
  Gemma4MoeNvfp4Matrix head{};
  const std::uint16_t* final_norm = nullptr;
  float softcap = 30.0F;
  DeviceBuffer kv;
  DeviceBuffer workspace;
  DeviceBuffer prefill_workspace;
  DeviceBuffer trellis35_workspace;
  Trellis35M1Workspace trellis35_m1_workspace{};
  Trellis35T3Workspace trellis35_t3_workspace{};
  DecodeControl* decode_control = nullptr;
  cudaGraphExec_t decode_graph = nullptr;
  std::uint32_t* prefill_tokens = nullptr;
  std::uint32_t* prefill_host_tokens = nullptr;
  float* prefill_hidden_a = nullptr;
  float* prefill_hidden_b = nullptr;
  Gemma4Moe26BAttentionReferenceWorkspace prefill_attention_workspace{};
  float* prefill_local_rotary_cosine = nullptr;
  float* prefill_local_rotary_sine = nullptr;
  Gemma4MoePrefillWorkspace prefill_moe_workspace{};
  float* hidden_a = nullptr;
  float* hidden_b = nullptr;
  float* final_hidden = nullptr;
  float* local_rotary_cosine = nullptr;
  float* local_rotary_sine = nullptr;
  std::uint8_t* head_activation = nullptr;
  std::uint8_t* head_activation_scales = nullptr;
  float* logits = nullptr;
  PredictionStatus* prediction_device_status = nullptr;
  PredictionStatus* prediction_host_status = nullptr;
  std::uint32_t* prediction_token = nullptr;
  float* prediction_logit = nullptr;
  int* finite = nullptr;
  int* routing_finite = nullptr;
  LogitCandidate* output_candidates = nullptr;
  float* sampling_logits = nullptr;
  double* sampling_cumulative = nullptr;
  std::uint32_t* sampling_token_ids = nullptr;
  std::uint32_t* sampling_sorted_token_ids = nullptr;
  std::uint32_t* repetition_mask = nullptr;
  std::uint32_t* suppressed_token_ids = nullptr;
  std::uint32_t* selected_token = nullptr;
  void* sampling_algorithm_workspace = nullptr;
  std::size_t sampling_algorithm_workspace_bytes = 0U;
  SamplingOptions sampling{};
  std::uint64_t sampling_step = 0U;
  std::uint32_t suppressed_token_count = 0U;
  std::array<float*, kLayers> layer_captures{};
  std::array<float*, kLayers> router_probability_captures{};
  std::array<std::uint32_t*, kLayers> router_id_captures{};
  bool quality_capture_all_layers = false;
  bool calibration_capture_enabled = false;
  bool calibration_capture_valid = false;
  std::uint32_t calibration_capture_layer = 0U;
  std::vector<float> calibration_gate_up_input;
  std::vector<float> calibration_down_inputs;
  std::vector<std::uint32_t> calibration_expert_ids;

  Status LaunchControlledDecodeBody();
  Status PrepareDecodeGraph();
  Status LaunchFixedMtpGraphBody(std::uint32_t draft_count,
                                 bool copy_transaction);
  Status SelectMtpVerificationTokens(
      float* batch_logits, const std::uint32_t* verification_inputs,
      DecodeControl* row_controls, std::uint32_t tokens,
      std::uint32_t* selected, bool apply_softcap = false,
      int* logits_finite = nullptr);
  Status PrepareFixedMtpGraph(std::uint32_t draft_count);
  Status LaunchFixedMtpOrdinaryTailBody();
  Status PrepareFixedMtpChainGraph(std::uint32_t draft_count);
  Status ExecuteFixedMtpGraphGroup(std::uint32_t pending_token,
                                   std::uint32_t draft_count,
                                   MtpGroupResult* host_result);

  ~Impl() {
    if (stream != nullptr) {
      (void)cudaStreamSynchronize(stream);
    }
    if (shared_moe_stream != nullptr) {
      (void)cudaStreamSynchronize(shared_moe_stream);
    }
    if (decode_graph != nullptr) (void)cudaGraphExecDestroy(decode_graph);
    for (cudaGraphExec_t graph : mtp_group_graphs) {
      if (graph != nullptr) (void)cudaGraphExecDestroy(graph);
    }
    for (cudaGraphExec_t graph : mtp_chain_graphs) {
      if (graph != nullptr) (void)cudaGraphExecDestroy(graph);
    }
    if (prefill_host_tokens != nullptr) {
      (void)cudaFreeHost(prefill_host_tokens);
    }
    if (prediction_host_status != nullptr) {
      (void)cudaFreeHost(prediction_host_status);
    }
    if (mtp_verifier_host != nullptr) {
      (void)cudaFreeHost(mtp_verifier_host);
    }
    if (mtp_chain_host != nullptr) (void)cudaFreeHost(mtp_chain_host);
    if (mtp_streaming_ring_host != nullptr) {
      (void)cudaFreeHost(mtp_streaming_ring_host);
    }
    if (stream != nullptr) {
      (void)cudaStreamDestroy(stream);
    }
    if (shared_moe_fork != nullptr) (void)cudaEventDestroy(shared_moe_fork);
    if (shared_moe_join != nullptr) (void)cudaEventDestroy(shared_moe_join);
    if (shared_moe_stream != nullptr) {
      (void)cudaStreamDestroy(shared_moe_stream);
    }
  }
};

Gemma4Moe26BReferenceEngine::Gemma4Moe26BReferenceEngine() = default;
Gemma4Moe26BReferenceEngine::~Gemma4Moe26BReferenceEngine() = default;
Gemma4Moe26BReferenceEngine::Gemma4Moe26BReferenceEngine(
    Gemma4Moe26BReferenceEngine&&) noexcept = default;
Gemma4Moe26BReferenceEngine& Gemma4Moe26BReferenceEngine::operator=(
    Gemma4Moe26BReferenceEngine&&) noexcept = default;
Gemma4Moe26BReferenceEngine::Gemma4Moe26BReferenceEngine(
    std::unique_ptr<Impl> implementation)
    : implementation_(std::move(implementation)) {}

Result<Gemma4Moe26BReferenceEngine> Gemma4Moe26BReferenceEngine::Create(
    const std::filesystem::path& model_directory,
    std::uint64_t context_tokens, int device, Gemma4Moe26BBackend backend,
    bool verify_device_image_sha256, bool quality_capture_all_layers,
    std::optional<Gemma4Moe26BRoutedExpertFormat>
        expected_routed_expert_format) {
  if (context_tokens == 0U || context_tokens > kMaximumContextTokens ||
      device < 0) {
    return Invalid("Gemma 4 26B context must be in [1, 262144]");
  }
  auto routed_expert_format =
      ResolveValidatedGemma4Moe26BRoutedExpertFormat(model_directory);
  if (!routed_expert_format.ok()) return routed_expert_format.status();
  if (expected_routed_expert_format.has_value()) {
    Status expected_status = ValidateGemma4Moe26BRoutedExpertFormat(
        routed_expert_format.value(), expected_routed_expert_format.value());
    if (!expected_status.ok()) return expected_status;
  }
  auto config = LoadModelConfig(model_directory / "config.json");
  if (!config.ok()) return config.status();
  if (context_tokens > config.value().max_positions) {
    return Invalid("Gemma 4 26B context exceeds the model maximum");
  }
  Status valid = ValidateGemma4Moe26BContract(config.value());
  if (!valid.ok()) return valid;
  auto traits = BuildGemma4Moe26BAttentionTraits(config.value());
  if (!traits.ok()) return traits.status();
  cudaError_t error = cudaSetDevice(device);
  if (error != cudaSuccess) return CudaFailure("select M13 CUDA device", error);

  auto impl = std::make_unique<Impl>();
  impl->device = device;
  impl->context = context_tokens;
  impl->backend = backend;
  impl->routed_expert_format = routed_expert_format.value();
  impl->quality_capture_all_layers = quality_capture_all_layers;
  if (IsTrellis35RoutedExpertFormat(routed_expert_format.value())) {
    if (backend != Gemma4Moe26BBackend::kSm120Integrated) {
      return Invalid("Trellis35 requires the SM120 integrated backend");
    }
    auto artifact =
        Gemma4Moe26BTrellis35DeviceArtifact::Load(model_directory);
    if (!artifact.ok()) return artifact.status();
    impl->trellis35_artifact = std::move(artifact).value();
  } else {
    auto manifest = InspectCheckpoint({model_directory, true});
    if (!manifest.ok()) return manifest.status();
    valid = ValidateGemma4Moe26BAttentionBindings(manifest.value().tensors,
                                                  traits.value());
    if (!valid.ok()) return valid;
    auto plan =
        BuildGemma4Moe26BResidencyPlan(manifest.value(), config.value());
    if (!plan.ok()) return plan.status();
    auto artifact = Gemma4Moe26BDeviceArtifact::Load(
        model_directory, manifest.value(), plan.value(),
        verify_device_image_sha256);
    if (!artifact.ok()) return artifact.status();
    impl->artifact = std::move(artifact).value();
  }
  if (backend == Gemma4Moe26BBackend::kSm120Integrated) {
    impl->moe_config.prefill_router =
        Gemma4MoePrefillRouter::kSm120TensorCore;
    impl->moe_config.materialize_native_router_normalized = false;
  }
  impl->traits = traits.value();
  for (const auto& trait : impl->traits) {
    if (trait.attention == Gemma4Moe26BAttentionType::kSliding) {
      if (impl->sliding_capacity == 0U) {
        impl->sliding_capacity = trait.cache_capacity;
      } else if (impl->sliding_capacity != trait.cache_capacity) {
        return Status(StatusCode::kDataLoss,
                      "Gemma 4 26B sliding cache capacities disagree");
      }
    }
  }
  if (impl->sliding_capacity == 0U) {
    return Status(StatusCode::kDataLoss,
                  "Gemma 4 26B has no validated sliding cache layer");
  }
  error = cudaStreamCreateWithFlags(&impl->stream, cudaStreamNonBlocking);
  if (error != cudaSuccess) return CudaFailure("create M13 stream", error);
  if (backend == Gemma4Moe26BBackend::kSm120Integrated) {
    error = cudaStreamCreateWithFlags(&impl->shared_moe_stream,
                                      cudaStreamNonBlocking);
    if (error != cudaSuccess) {
      return CudaFailure("create M20 shared-MoE stream", error);
    }
    error = cudaEventCreateWithFlags(&impl->shared_moe_fork,
                                     cudaEventDisableTiming);
    if (error == cudaSuccess) {
      error = cudaEventCreateWithFlags(&impl->shared_moe_join,
                                       cudaEventDisableTiming);
    }
    if (error != cudaSuccess) {
      return CudaFailure("create M20 shared-MoE events", error);
    }
  }

  for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
    auto attention = IsTrellis35RoutedExpertFormat(routed_expert_format.value())
                         ? BindGemma4Moe26BAttentionReferenceWeights(
                               impl->trellis35_artifact, impl->traits[layer])
                         : BindGemma4Moe26BAttentionReferenceWeights(
                               impl->artifact, impl->traits[layer]);
    if (!attention.ok()) return attention.status();
    impl->attention_weights[layer] = attention.value();
    auto moe = IsTrellis35RoutedExpertFormat(routed_expert_format.value())
                   ? BindGemma4Moe26BReferenceWeights(
                         impl->trellis35_artifact, layer)
                   : BindGemma4Moe26BReferenceWeights(impl->artifact, layer);
    if (!moe.ok()) return moe.status();
    impl->moe_weights[layer] = moe.value();
  }

  auto artifact_pointer = [&](const char* name) {
    return IsTrellis35RoutedExpertFormat(routed_expert_format.value())
               ? impl->trellis35_artifact.NonRoutedPointer(name)
               : impl->artifact.Pointer(name);
  };
  auto head_packed_bytes =
      artifact_pointer("model.language_model.embed_tokens.weight_packed");
  auto head_scales_bytes =
      artifact_pointer("model.language_model.embed_tokens.weight_scale");
  auto final_norm_bytes = artifact_pointer("model.language_model.norm.weight");
  Result<const std::uint8_t*> head_packed =
      head_packed_bytes.ok()
          ? Result<const std::uint8_t*>(reinterpret_cast<const std::uint8_t*>(
                head_packed_bytes.value()))
          : Result<const std::uint8_t*>(head_packed_bytes.status());
  Result<const std::uint8_t*> head_scales =
      head_scales_bytes.ok()
          ? Result<const std::uint8_t*>(reinterpret_cast<const std::uint8_t*>(
                head_scales_bytes.value()))
          : Result<const std::uint8_t*>(head_scales_bytes.status());
  Result<const std::uint16_t*> final_norm =
      final_norm_bytes.ok()
          ? Result<const std::uint16_t*>(
                reinterpret_cast<const std::uint16_t*>(final_norm_bytes.value()))
          : Result<const std::uint16_t*>(final_norm_bytes.status());
  auto head_activation_divisor =
      IsTrellis35RoutedExpertFormat(routed_expert_format.value())
          ? impl->trellis35_artifact.HostFloat32(
                "model.language_model.embed_tokens.input_global_scale")
          : impl->artifact.HostFloat32(
                "model.language_model.embed_tokens.input_global_scale");
  auto head_weight_divisor =
      IsTrellis35RoutedExpertFormat(routed_expert_format.value())
          ? impl->trellis35_artifact.HostFloat32(
                "model.language_model.embed_tokens.weight_global_scale")
          : impl->artifact.HostFloat32(
                "model.language_model.embed_tokens.weight_global_scale");
  if (!head_packed.ok()) return head_packed.status();
  if (!head_scales.ok()) return head_scales.status();
  if (!final_norm.ok()) return final_norm.status();
  if (!head_activation_divisor.ok()) return head_activation_divisor.status();
  if (!head_weight_divisor.ok()) return head_weight_divisor.status();
  impl->head = {head_packed.value(), head_scales.value(), kVocabulary, kWidth,
                head_activation_divisor.value(), head_weight_divisor.value()};
  impl->final_norm = final_norm.value();
  impl->softcap = static_cast<float>(config.value().final_logit_softcap);

  auto kv_bytes = Gemma4Moe26BFp8KvBytes(impl->traits, context_tokens);
  if (!kv_bytes.ok()) return kv_bytes.status();
  valid = impl->kv.Allocate(kv_bytes.value(), "allocate M13 FP8 K/V arena");
  if (!valid.ok()) return valid;
  std::uint64_t kv_offset = 0U;
  for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
    const auto& trait = impl->traits[layer];
    const std::uint64_t capacity =
        trait.attention == Gemma4Moe26BAttentionType::kSliding
            ? trait.cache_capacity
            : context_tokens;
    const std::uint64_t one = capacity * trait.kv_heads * trait.head_dimension;
    impl->caches[layer] = {impl->kv.As<std::uint8_t>(kv_offset),
                           impl->kv.As<std::uint8_t>(kv_offset + one),
                           capacity};
    kv_offset += 2U * one;
  }
  if (kv_offset != kv_bytes.value()) {
    return Status(StatusCode::kInternal,
                  "M13 K/V partition does not match the accepted byte formula");
  }

  LayoutBuilder layout;
  const auto hidden_a = layout.Add<float>(kWidth);
  const auto hidden_b = layout.Add<float>(kWidth);
  const auto final_hidden = layout.Add<float>(kWidth);
  const auto a_input_fp8 = layout.Add<std::uint8_t>(kWidth);
  const auto a_input_scale = layout.Add<float>(1U);
  const auto q_raw = layout.Add<float>(16U * 512U);
  const auto k_raw = layout.Add<float>(8U * 512U);
  const auto v_raw = layout.Add<float>(8U * 512U);
  const auto q_norm = layout.Add<float>(16U * 512U);
  const auto k_norm = layout.Add<float>(8U * 512U);
  const auto v_norm = layout.Add<float>(8U * 512U);
  const auto cosine = layout.Add<float>(256U);
  const auto sine = layout.Add<float>(256U);
  const auto local_cosine = layout.Add<float>(256U);
  const auto local_sine = layout.Add<float>(256U);
  const auto staged_k = layout.Add<std::uint8_t>(8U * 512U);
  const auto staged_v = layout.Add<std::uint8_t>(8U * 512U);
  const std::uint64_t decode_attention_workspace_elements = std::max(
      std::max(16U * context_tokens,
               DecodeAttentionWorkspaceElements(context_tokens)),
      DecodeAttentionWorkspaceElements(impl->sliding_capacity));
  const auto scores =
      layout.Add<float>(decode_attention_workspace_elements);
  const auto attention = layout.Add<float>(16U * 512U);
  const auto output_fp8 = layout.Add<std::uint8_t>(16U * 512U);
  const auto output_scale = layout.Add<float>(1U);
  const auto output_projection = layout.Add<float>(kWidth);
  const auto post_attention = layout.Add<float>(kWidth);

  const auto shared_input = layout.Add<float>(kWidth);
  const auto shared_input_packed = layout.Add<std::uint8_t>(kWidth / 2U);
  const auto shared_input_scales = layout.Add<std::uint8_t>(kWidth / 16U);
  const auto shared_gate = layout.Add<float>(kShared);
  const auto shared_up = layout.Add<float>(kShared);
  const auto shared_product = layout.Add<float>(kShared);
  const auto shared_product_packed = layout.Add<std::uint8_t>(kShared / 2U);
  const auto shared_product_scales = layout.Add<std::uint8_t>(kShared / 16U);
  const auto shared_output = layout.Add<float>(kWidth);
  const auto shared_post = layout.Add<float>(kWidth);
  const auto router_normalized = layout.Add<float>(kWidth);
  const auto router_transformed = layout.Add<float>(kWidth);
  const auto router_logits = layout.Add<float>(kExperts);
  const auto router_probabilities = layout.Add<float>(kExperts);
  const auto top_ids = layout.Add<std::uint32_t>(kTopK);
  const auto top_weights = layout.Add<float>(kTopK);
  const auto expert_input = layout.Add<float>(kWidth);
  const auto expert_input_packed = layout.Add<std::uint8_t>(kWidth / 2U);
  const auto expert_input_scales = layout.Add<std::uint8_t>(kWidth / 16U);
  const auto expert_gate_up = layout.Add<float>(kTopK * 2U * kExpert);
  const auto expert_product = layout.Add<float>(kTopK * kExpert);
  const auto expert_product_packed =
      layout.Add<std::uint8_t>(kTopK * kExpert / 2U);
  const auto expert_product_scales =
      layout.Add<std::uint8_t>(kTopK * kExpert / 16U);
  const auto expert_down = layout.Add<float>(kTopK * kWidth);
  const auto expert_contributions = layout.Add<float>(kTopK * kWidth);
  const auto routed_sum = layout.Add<float>(kWidth);
  const auto routed_post = layout.Add<float>(kWidth);
  const auto combined = layout.Add<float>(kWidth);
  const auto feed_forward = layout.Add<float>(kWidth);
  const auto head_activation = layout.Add<std::uint8_t>(kWidth / 2U);
  const auto head_activation_scale_offset =
      layout.Add<std::uint8_t>(kWidth / 16U);
  const auto logits = layout.Add<float>(kVocabulary);
  const auto prediction_status = layout.Add<PredictionStatus>(1U);
  const auto output_candidates = layout.Add<LogitCandidate>(kArgmaxBlocks);
  const auto decode_control = layout.Add<DecodeControl>(1U);
  auto sampling_workspace_bytes = SamplingWorkspaceBytes(
      static_cast<std::uint32_t>(kVocabulary), impl->stream);
  if (!sampling_workspace_bytes.ok()) {
    return sampling_workspace_bytes.status();
  }
  const auto sampling_logits = layout.Add<float>(kVocabulary);
  const auto sampling_cumulative = layout.Add<double>(kVocabulary);
  const auto sampling_token_ids = layout.Add<std::uint32_t>(kVocabulary);
  const auto sampling_sorted_token_ids =
      layout.Add<std::uint32_t>(kVocabulary);
  const auto repetition_mask =
      layout.Add<std::uint32_t>(kRepetitionMaskWords);
  const auto suppressed_token_ids =
      layout.Add<std::uint32_t>(kMaximumSuppressedTokens);
  const auto selected_token = layout.Add<std::uint32_t>(1U);
  const auto sampling_algorithm_workspace = layout.Add<std::uint8_t>(
      sampling_workspace_bytes.value());
  std::array<std::uint64_t, kLayers> capture_outputs{};
  std::array<std::uint64_t, kLayers> capture_probs{};
  std::array<std::uint64_t, kLayers> capture_ids{};
  for (std::size_t i = 0; i < kLayers; ++i) {
    if (CaptureIndex(static_cast<std::uint32_t>(i),
                     quality_capture_all_layers) < 0) {
      continue;
    }
    capture_outputs[i] = layout.Add<float>(kWidth);
    capture_probs[i] = layout.Add<float>(kExperts);
    capture_ids[i] = layout.Add<std::uint32_t>(kTopK);
  }
  if (layout.bytes == std::numeric_limits<std::uint64_t>::max()) {
    return Invalid("M13 workspace layout overflow");
  }
  valid = impl->workspace.Allocate(layout.bytes, "allocate M13 fixed workspace");
  if (!valid.ok()) return valid;
  auto ptr = [&](std::uint64_t offset) {
    return impl->workspace.As<std::byte>(offset);
  };
  impl->hidden_a = reinterpret_cast<float*>(ptr(hidden_a));
  impl->hidden_b = reinterpret_cast<float*>(ptr(hidden_b));
  impl->final_hidden = reinterpret_cast<float*>(ptr(final_hidden));
  impl->local_rotary_cosine = reinterpret_cast<float*>(ptr(local_cosine));
  impl->local_rotary_sine = reinterpret_cast<float*>(ptr(local_sine));
  impl->attention_workspace = {
      reinterpret_cast<std::uint8_t*>(ptr(a_input_fp8)),
      reinterpret_cast<float*>(ptr(a_input_scale)),
      reinterpret_cast<float*>(ptr(q_raw)), reinterpret_cast<float*>(ptr(k_raw)),
      reinterpret_cast<float*>(ptr(v_raw)), reinterpret_cast<float*>(ptr(q_norm)),
      reinterpret_cast<float*>(ptr(k_norm)), reinterpret_cast<float*>(ptr(v_norm)),
      reinterpret_cast<float*>(ptr(cosine)), reinterpret_cast<float*>(ptr(sine)),
      reinterpret_cast<std::uint8_t*>(ptr(staged_k)),
      reinterpret_cast<std::uint8_t*>(ptr(staged_v)),
      reinterpret_cast<float*>(ptr(scores)),
      decode_attention_workspace_elements,
      reinterpret_cast<float*>(ptr(attention)),
      reinterpret_cast<std::uint8_t*>(ptr(output_fp8)),
      reinterpret_cast<float*>(ptr(output_scale)),
      reinterpret_cast<float*>(ptr(output_projection)),
      reinterpret_cast<float*>(ptr(post_attention)), nullptr, 0U};
  impl->moe_workspace = {
      reinterpret_cast<float*>(ptr(shared_input)),
      reinterpret_cast<std::uint8_t*>(ptr(shared_input_packed)),
      reinterpret_cast<std::uint8_t*>(ptr(shared_input_scales)),
      reinterpret_cast<float*>(ptr(shared_gate)),
      reinterpret_cast<float*>(ptr(shared_up)),
      reinterpret_cast<float*>(ptr(shared_product)),
      reinterpret_cast<std::uint8_t*>(ptr(shared_product_packed)),
      reinterpret_cast<std::uint8_t*>(ptr(shared_product_scales)),
      reinterpret_cast<float*>(ptr(shared_output)),
      reinterpret_cast<float*>(ptr(shared_post)),
      reinterpret_cast<float*>(ptr(router_normalized)),
      reinterpret_cast<float*>(ptr(router_transformed)),
      reinterpret_cast<float*>(ptr(router_logits)),
      reinterpret_cast<float*>(ptr(router_probabilities)),
      reinterpret_cast<std::uint32_t*>(ptr(top_ids)),
      reinterpret_cast<float*>(ptr(top_weights)),
      reinterpret_cast<float*>(ptr(expert_input)),
      reinterpret_cast<std::uint8_t*>(ptr(expert_input_packed)),
      reinterpret_cast<std::uint8_t*>(ptr(expert_input_scales)),
      reinterpret_cast<float*>(ptr(expert_gate_up)),
      reinterpret_cast<float*>(ptr(expert_product)),
      reinterpret_cast<std::uint8_t*>(ptr(expert_product_packed)),
      reinterpret_cast<std::uint8_t*>(ptr(expert_product_scales)),
      reinterpret_cast<float*>(ptr(expert_down)),
      reinterpret_cast<float*>(ptr(expert_contributions)),
      reinterpret_cast<float*>(ptr(routed_sum)),
      reinterpret_cast<float*>(ptr(routed_post)),
      reinterpret_cast<float*>(ptr(combined)),
      reinterpret_cast<float*>(ptr(feed_forward)),
      reinterpret_cast<int*>(
          ptr(prediction_status) +
          offsetof(PredictionStatus, routing_finite))};
  impl->head_activation = reinterpret_cast<std::uint8_t*>(ptr(head_activation));
  impl->head_activation_scales =
      reinterpret_cast<std::uint8_t*>(ptr(head_activation_scale_offset));
  impl->logits = reinterpret_cast<float*>(ptr(logits));
  impl->prediction_device_status =
      reinterpret_cast<PredictionStatus*>(ptr(prediction_status));
  auto* prediction_status_bytes = reinterpret_cast<std::byte*>(
      impl->prediction_device_status);
  impl->prediction_token = reinterpret_cast<std::uint32_t*>(
      prediction_status_bytes + offsetof(PredictionStatus, token));
  impl->prediction_logit = reinterpret_cast<float*>(
      prediction_status_bytes + offsetof(PredictionStatus, logit));
  impl->finite = reinterpret_cast<int*>(
      prediction_status_bytes + offsetof(PredictionStatus, finite));
  impl->routing_finite = reinterpret_cast<int*>(
      prediction_status_bytes + offsetof(PredictionStatus, routing_finite));
  error = cudaMallocHost(&impl->prediction_host_status,
                         sizeof(PredictionStatus));
  if (error != cudaSuccess) {
    return CudaFailure("allocate pinned M20 prediction status", error);
  }
  impl->output_candidates =
      reinterpret_cast<LogitCandidate*>(ptr(output_candidates));
  impl->decode_control = reinterpret_cast<DecodeControl*>(ptr(decode_control));
  impl->sampling_logits = reinterpret_cast<float*>(ptr(sampling_logits));
  impl->sampling_cumulative =
      reinterpret_cast<double*>(ptr(sampling_cumulative));
  impl->sampling_token_ids =
      reinterpret_cast<std::uint32_t*>(ptr(sampling_token_ids));
  impl->sampling_sorted_token_ids =
      reinterpret_cast<std::uint32_t*>(ptr(sampling_sorted_token_ids));
  impl->repetition_mask =
      reinterpret_cast<std::uint32_t*>(ptr(repetition_mask));
  impl->suppressed_token_ids =
      reinterpret_cast<std::uint32_t*>(ptr(suppressed_token_ids));
  impl->selected_token =
      reinterpret_cast<std::uint32_t*>(ptr(selected_token));
  impl->sampling_algorithm_workspace = ptr(sampling_algorithm_workspace);
  impl->sampling_algorithm_workspace_bytes =
      sampling_workspace_bytes.value();
  for (std::size_t i = 0; i < kLayers; ++i) {
    if (CaptureIndex(static_cast<std::uint32_t>(i),
                     quality_capture_all_layers) < 0) {
      continue;
    }
    impl->layer_captures[i] = reinterpret_cast<float*>(ptr(capture_outputs[i]));
    impl->router_probability_captures[i] =
        reinterpret_cast<float*>(ptr(capture_probs[i]));
    impl->router_id_captures[i] =
        reinterpret_cast<std::uint32_t*>(ptr(capture_ids[i]));
  }

  if (backend == Gemma4Moe26BBackend::kSm120Integrated) {
    LayoutBuilder prefill;
    const auto p_tokens = prefill.Add<std::uint32_t>(kPrefillMaxTokens);
    const auto p_hidden_a = prefill.Add<float>(kPrefillMaxTokens * kWidth);
    const auto p_hidden_b = prefill.Add<float>(kPrefillMaxTokens * kWidth);
    // The two RoPE profiles remain live across every attention/MoE pair in a
    // chunk. Keep them in the persistent prefix shared by both phase layouts;
    // placing them in attention-only scratch would let each intervening MoE
    // layer overwrite the tables before the next attention layer consumes
    // them.
    const auto p_global_cosine =
        prefill.Add<float>(kPrefillMaxTokens * 256U);
    const auto p_global_sine =
        prefill.Add<float>(kPrefillMaxTokens * 256U);
    const auto p_local_cosine =
        prefill.Add<float>(kPrefillMaxTokens * 256U);
    const auto p_local_sine =
        prefill.Add<float>(kPrefillMaxTokens * 256U);

    // Attention and MoE execute in-order on impl->stream. Build both phase
    // layouts from the persistent token/hidden prefix so their temporary
    // regions physically alias without overlapping live values.
    // Query-normalized rows stay live through attention. All projection
    // temporaries are dead by then, while prepared global K/V are dead before
    // output quantization. Model those three phases explicitly so the <=16K
    // BF16 staging does not increase the fixed M09 workspace.
    LayoutBuilder attention_common = prefill;
    const auto p_scores =
        attention_common.Add<float>(kPrefillScoreElements);
    const auto p_q_norm = attention_common.Add<float>(
        kPrefillMaxTokens * 16U * 512U);
    LayoutBuilder attention_projection = attention_common;
    const auto p_input_fp8 =
        attention_projection.Add<std::uint8_t>(kPrefillMaxTokens * kWidth);
    const auto p_input_scale =
        attention_projection.Add<float>(kPrefillMaxTokens);
    const auto p_q_raw =
        attention_projection.Add<std::uint16_t>(kPrefillMaxTokens * 16U * 512U);
    const auto p_k_raw =
        attention_projection.Add<std::uint16_t>(kPrefillMaxTokens * 2048U);
    const auto p_v_raw =
        attention_projection.Add<std::uint16_t>(kPrefillMaxTokens * 2048U);
    const auto p_k_norm =
        attention_projection.Add<float>(kPrefillMaxTokens * 2048U);
    const auto p_v_norm =
        attention_projection.Add<float>(kPrefillMaxTokens * 2048U);
    const auto p_staged_k =
        attention_projection.Add<std::uint8_t>(kPrefillMaxTokens * 2048U);
    const auto p_staged_v =
        attention_projection.Add<std::uint8_t>(kPrefillMaxTokens * 2048U);
    const auto p_cutlass_workspace = attention_projection.Add<std::byte>(
        kGemma4Moe26BAttentionCutlassWorkspaceBytes);

    LayoutBuilder attention_prepared = attention_common;
    const auto p_global_key_bf16 = attention_prepared.Add<std::uint16_t>(
        kPreparedGlobalPrefillTokens * 2U * 512U);
    const auto p_global_value_bf16 = attention_prepared.Add<std::uint16_t>(
        kPreparedGlobalPrefillTokens * 2U * 512U);
    const auto p_attention = attention_prepared.Add<std::uint16_t>(
        kPrefillMaxTokens * 16U * 512U);

    LayoutBuilder attention_post = attention_common;
    const auto p_output_fp8 =
        attention_post.Add<std::uint8_t>(kPrefillMaxTokens * 16U * 512U);
    const auto p_output_scale =
        attention_post.Add<float>(kPrefillMaxTokens);
    const auto p_output_projection =
        attention_post.Add<float>(kPrefillMaxTokens * kWidth);
    LayoutBuilder attention = attention_projection;
    attention.bytes = std::max(
        {attention_projection.bytes, attention_prepared.bytes,
         attention_post.bytes});

    LayoutBuilder moe = prefill;
    const auto p_router_logits =
        moe.Add<float>(kPrefillMaxTokens * kExperts);
    const auto p_router_probabilities =
        moe.Add<float>(kPrefillMaxTokens * kExperts);
    const auto p_token_hidden =
        moe.Add<float>(kPrefillMaxTokens * kWidth);
    const auto p_token_packed =
        moe.Add<std::uint8_t>(kPrefillMaxTokens * kWidth / 2U);
    const auto p_token_scales =
        moe.Add<std::uint8_t>(kPrefillMaxTokens * kWidth / 16U);
    const auto p_expert_product = moe.Add<std::uint16_t>(
        kPrefillMaxTokens * kTopK * kExpert);
    const auto p_expert_product_packed = moe.Add<std::uint8_t>(
        kPrefillMaxTokens * kTopK * kExpert / 2U);
    const auto p_expert_product_scales = moe.Add<std::uint8_t>(
        kPrefillMaxTokens * kTopK * kExpert / 16U);
    const auto p_expert_down = moe.Add<std::uint16_t>(
        kPrefillMaxTokens * kTopK * kWidth);
    // Fixed-D2 attention and prefill MoE execute in-order. Alias the dead MoE
    // suffix, but plan the alias with its actual long-context size instead of
    // relying on whatever bytes happen to follow shared_product. The previous
    // implicit alias crossed the prefill allocation at long context and caused
    // the 75K illegal-address boundary.
    LayoutBuilder mtp_d2_attention = moe;
    const auto p_mtp_d2_attention = mtp_d2_attention.Add<float>(
        DecodeAttentionFixedWorkspaceElements(context_tokens,
                                               kM25FixedD2Rows));
    const auto p_shared_product =
        moe.Add<float>(kPrefillMaxTokens * kShared);
    const auto p_shared_product_packed =
        moe.Add<std::uint8_t>(kPrefillMaxTokens * kShared / 2U);
    const auto p_shared_product_scales =
        moe.Add<std::uint8_t>(kPrefillMaxTokens * kShared / 16U);
    const auto p_shared_output =
        moe.Add<float>(kPrefillMaxTokens * kWidth);
    const auto p_reduced_output =
        moe.Add<float>(kPrefillMaxTokens * kWidth);
    const auto p_assignments = moe.Add<Gemma4MoePrefillAssignment>(
        kPrefillMaxTokens * kTopK);
    const auto p_histogram = moe.Add<std::uint32_t>(kExperts);
    const auto p_prefix = moe.Add<std::uint32_t>(kExperts + 1U);
    const auto p_permutation =
        moe.Add<std::uint32_t>(kPrefillMaxTokens * kTopK);
    const auto p_inverse =
        moe.Add<std::uint32_t>(kPrefillMaxTokens * kTopK);
    prefill.bytes =
        std::max({attention.bytes, moe.bytes, mtp_d2_attention.bytes});
    constexpr std::uint64_t kM09MoePrefillCap = 192U * 1024U * 1024U;
    if (p_mtp_d2_attention != p_shared_product ||
        prefill.bytes == std::numeric_limits<std::uint64_t>::max() ||
        prefill.bytes > kM09MoePrefillCap) {
      return Invalid("M17 fixed prefill workspace exceeds the M09 cap");
    }
    valid = impl->prefill_workspace.Allocate(
        prefill.bytes, "allocate M17 fixed prefill workspace");
    if (!valid.ok()) return valid;
    auto pptr = [&](std::uint64_t offset) {
      return impl->prefill_workspace.As<std::byte>(offset);
    };
    impl->prefill_tokens = reinterpret_cast<std::uint32_t*>(pptr(p_tokens));
    error = cudaMallocHost(&impl->prefill_host_tokens,
                           kPrefillMaxTokens * sizeof(std::uint32_t));
    if (error != cudaSuccess) {
      return CudaFailure("allocate M17 pinned prefill tokens", error);
    }
    impl->prefill_hidden_a = reinterpret_cast<float*>(pptr(p_hidden_a));
    impl->prefill_hidden_b = reinterpret_cast<float*>(pptr(p_hidden_b));
    impl->prefill_attention_workspace = {
        reinterpret_cast<std::uint8_t*>(pptr(p_input_fp8)),
        reinterpret_cast<float*>(pptr(p_input_scale)),
        reinterpret_cast<float*>(pptr(p_q_raw)),
        reinterpret_cast<float*>(pptr(p_k_raw)),
        reinterpret_cast<float*>(pptr(p_v_raw)),
        reinterpret_cast<float*>(pptr(p_q_norm)),
        reinterpret_cast<float*>(pptr(p_k_norm)),
        reinterpret_cast<float*>(pptr(p_v_norm)),
        reinterpret_cast<float*>(pptr(p_global_cosine)),
        reinterpret_cast<float*>(pptr(p_global_sine)),
        reinterpret_cast<std::uint8_t*>(pptr(p_staged_k)),
        reinterpret_cast<std::uint8_t*>(pptr(p_staged_v)),
        reinterpret_cast<float*>(pptr(p_scores)), kPrefillScoreElements,
        reinterpret_cast<float*>(pptr(p_attention)),
        reinterpret_cast<std::uint8_t*>(pptr(p_output_fp8)),
        reinterpret_cast<float*>(pptr(p_output_scale)),
        reinterpret_cast<float*>(pptr(p_output_projection)),
        nullptr,
        static_cast<void*>(pptr(p_cutlass_workspace)),
        kGemma4Moe26BAttentionCutlassWorkspaceBytes};
    impl->prefill_attention_workspace.global_key_bf16 =
        reinterpret_cast<std::uint16_t*>(pptr(p_global_key_bf16));
    impl->prefill_attention_workspace.global_value_bf16 =
        reinterpret_cast<std::uint16_t*>(pptr(p_global_value_bf16));
    impl->prefill_attention_workspace.global_bf16_capacity =
        kPreparedGlobalPrefillTokens;
    impl->prefill_local_rotary_cosine =
        reinterpret_cast<float*>(pptr(p_local_cosine));
    impl->prefill_local_rotary_sine =
        reinterpret_cast<float*>(pptr(p_local_sine));
    impl->prefill_moe_workspace = {
        reinterpret_cast<float*>(pptr(p_router_logits)),
        reinterpret_cast<float*>(pptr(p_router_probabilities)),
        reinterpret_cast<float*>(pptr(p_token_hidden)),
        reinterpret_cast<std::uint8_t*>(pptr(p_token_packed)),
        reinterpret_cast<std::uint8_t*>(pptr(p_token_scales)),
        nullptr,
        reinterpret_cast<std::uint8_t*>(pptr(p_expert_product_packed)),
        reinterpret_cast<std::uint8_t*>(pptr(p_expert_product_scales)),
        nullptr,
        reinterpret_cast<float*>(pptr(p_shared_product)),
        reinterpret_cast<std::uint8_t*>(pptr(p_shared_product_packed)),
        reinterpret_cast<std::uint8_t*>(pptr(p_shared_product_scales)),
        reinterpret_cast<float*>(pptr(p_shared_output)),
        reinterpret_cast<float*>(pptr(p_reduced_output)),
        reinterpret_cast<Gemma4MoePrefillAssignment*>(pptr(p_assignments)),
        reinterpret_cast<std::uint32_t*>(pptr(p_histogram)),
        reinterpret_cast<std::uint32_t*>(pptr(p_prefix)),
        reinterpret_cast<std::uint32_t*>(pptr(p_permutation)),
        reinterpret_cast<std::uint32_t*>(pptr(p_inverse)),
        impl->routing_finite};
    impl->prefill_moe_workspace.expert_product_bf16 =
        reinterpret_cast<std::uint16_t*>(pptr(p_expert_product));
    impl->prefill_moe_workspace.expert_down_bf16 =
        reinterpret_cast<std::uint16_t*>(pptr(p_expert_down));
  }

  if (IsTrellis35RoutedExpertFormat(routed_expert_format.value())) {
    LayoutBuilder trellis_layout;
    constexpr std::uint64_t assignments = kTrellis35T3Assignments;
    const auto prefill_activation = trellis_layout.Add<std::uint8_t>(
        kPrefillMaxTokens * kTopK * kTrellis35GateUpInput);
    const auto gate_input = trellis_layout.Add<float>(
        assignments * kTrellis35GateUpInput);
    const auto gate_e4m3 = trellis_layout.Add<std::uint8_t>(
        assignments * kTrellis35GateUpInput);
    const auto gate_scales = trellis_layout.Add<float>(assignments);
    const auto gate_transformed = trellis_layout.Add<float>(
        assignments * kTrellis35GateUpOutput);
    const auto gate_output = trellis_layout.Add<float>(
        assignments * kTrellis35GateUpOutput);
    const auto product = trellis_layout.Add<float>(
        assignments * kTrellis35ExpertIntermediate);
    const auto down_input = trellis_layout.Add<float>(
        assignments * kTrellis35DownInput);
    const auto down_e4m3 = trellis_layout.Add<std::uint8_t>(
        assignments * kTrellis35DownInput);
    const auto down_scales = trellis_layout.Add<float>(assignments);
    const auto down_transformed = trellis_layout.Add<float>(
        assignments * kTrellis35DownOutput);
    const auto down_output = trellis_layout.Add<float>(
        assignments * kTrellis35DownOutput);
    if (trellis_layout.bytes == std::numeric_limits<std::uint64_t>::max()) {
      return Invalid("Trellis35 fixed workspace layout overflow");
    }
    valid = impl->trellis35_workspace.Allocate(
        trellis_layout.bytes, "allocate Trellis35 M1/T3 fixed workspace");
    if (!valid.ok()) return valid;
    auto tptr = [&](std::uint64_t offset) {
      return impl->trellis35_workspace.As<std::byte>(offset);
    };
    impl->trellis35_t3_workspace = {
        reinterpret_cast<float*>(tptr(gate_input)),
        reinterpret_cast<std::uint8_t*>(tptr(gate_e4m3)),
        reinterpret_cast<float*>(tptr(gate_scales)),
        reinterpret_cast<float*>(tptr(gate_transformed)),
        reinterpret_cast<float*>(tptr(gate_output)),
        reinterpret_cast<float*>(tptr(product)),
        reinterpret_cast<float*>(tptr(down_input)),
        reinterpret_cast<std::uint8_t*>(tptr(down_e4m3)),
        reinterpret_cast<float*>(tptr(down_scales)),
        reinterpret_cast<float*>(tptr(down_transformed)),
        reinterpret_cast<float*>(tptr(down_output))};
    impl->prefill_moe_workspace.trellis_activation =
        reinterpret_cast<std::uint8_t*>(tptr(prefill_activation));
    impl->trellis35_m1_workspace = {
        impl->trellis35_t3_workspace.gate_up_input_transformed,
        impl->trellis35_t3_workspace.gate_up_input_e4m3,
        impl->trellis35_t3_workspace.gate_up_input_scales,
        impl->trellis35_t3_workspace.gate_up_transformed_output,
        impl->trellis35_t3_workspace.gate_up_output,
        impl->trellis35_t3_workspace.product,
        impl->trellis35_t3_workspace.down_input_transformed,
        impl->trellis35_t3_workspace.down_input_e4m3,
        impl->trellis35_t3_workspace.down_input_scales,
        impl->trellis35_t3_workspace.down_transformed_output,
        impl->trellis35_t3_workspace.down_output};
  }

  Gemma4Moe26BReferenceEngine engine(std::move(impl));
  valid = engine.Reset();
  if (!valid.ok()) return valid;
  if (backend == Gemma4Moe26BBackend::kSm120Integrated) {
    valid = engine.implementation_->PrepareDecodeGraph();
    if (!valid.ok()) return valid;
  }
  std::size_t free_bytes = 0U;
  std::size_t total_bytes = 0U;
  error = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (error != cudaSuccess) {
    return CudaFailure("measure M21 initialized memory margin", error);
  }
  const std::uint64_t required_margin =
      context_tokens >= 65536U ? kLongContextMargin : kPrimaryContextMargin;
  if (free_bytes < required_margin) {
    const std::uint64_t required_slot =
        engine.weight_arena_bytes() + engine.kv_cache_bytes() +
        engine.workspace_bytes();
    const std::uint64_t shortfall =
        required_margin - static_cast<std::uint64_t>(free_bytes);
    return Status(
        StatusCode::kResourceExhausted,
        "cannot create Gemma 4 26B execution slot: context=" +
            std::to_string(context_tokens) +
            " free=" + std::to_string(free_bytes) +
            " required_slot=" + std::to_string(required_slot) +
            " probe_resident=" +
            std::to_string(engine.weight_arena_bytes()) +
            " required_margin=" + std::to_string(required_margin) +
            " shortfall=" + std::to_string(shortfall));
  }
  return engine;
}

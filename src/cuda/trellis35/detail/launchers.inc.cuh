Status TransformArguments(const float* input, const std::uint16_t* sidecar,
                          float* output, std::uint64_t elements,
                          std::string_view description) {
  if (input == nullptr || sidecar == nullptr || output == nullptr) {
    return Invalid(std::string(description) + " requires non-null pointers");
  }
  if (elements == 0U || elements % 128U != 0U ||
      elements > static_cast<std::uint64_t>(
                     std::numeric_limits<unsigned>::max()) * kThreads) {
    return Invalid(std::string(description) + " extent is invalid");
  }
  return Status::Ok();
}

Status LaunchGroupedT3Projection(
    const std::uint8_t* activation_e4m3, const float* activation_scales,
    const Trellis35DeviceFamilyBinding& family,
    const std::uint32_t* selected_experts, float* transformed_output,
    std::uint64_t input_elements, std::uint64_t output_elements,
    cudaStream_t stream) {
  if (activation_e4m3 == nullptr || activation_scales == nullptr ||
      family.k3_payload_pool == nullptr || family.k4_payload_pool == nullptr ||
      family.descriptors == nullptr || selected_experts == nullptr ||
      transformed_output == nullptr) {
    return Invalid("Trellis35 grouped T3 projection requires non-null pointers");
  }
  if (input_elements == 0U || output_elements == 0U ||
      input_elements % 32U != 0U || output_elements % 8U != 0U ||
      output_elements >
          static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max()) *
              kMmaWarps * 8U) {
    return Invalid("Trellis35 grouped T3 projection geometry is invalid");
  }
  const unsigned blocks = static_cast<unsigned>(
      (output_elements + kMmaWarps * 8U - 1U) / (kMmaWarps * 8U));
  MmaW4A8ProjectionGroupedT3Kernel<<<
      dim3(blocks, kTrellis35T3Assignments), kMmaThreads, 0, stream>>>(
      activation_e4m3, activation_scales, family, selected_experts,
      transformed_output, input_elements, output_elements);
  return CheckLaunch("launch Trellis35 grouped T3 W4A8 projection");
}

Status LaunchPrefillTransformQuantize(
    const float* input, const Trellis35DeviceFamilyBinding& family,
    const Gemma4MoePrefillAssignment* assignments, std::uint8_t* output,
    float* scales, std::uint64_t assignment_count, std::uint64_t tokens,
    std::uint64_t input_stride, std::uint64_t logical_elements,
    std::uint64_t physical_elements, bool token_major_input,
    cudaStream_t stream) {
  if (input == nullptr || family.suh_f16 == nullptr || assignments == nullptr ||
      output == nullptr || scales == nullptr || assignment_count == 0U ||
      tokens == 0U || input_stride == 0U || logical_elements == 0U ||
      logical_elements > physical_elements || physical_elements % 128U != 0U ||
      assignment_count >
          static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max()) ||
      physical_elements >
          static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max()) *
              kThreads) {
    return Invalid("Trellis35 prefill transform/quantize contract is invalid");
  }
  PrefillTransformScaleKernel<<<static_cast<unsigned>(assignment_count),
                                kThreads, 0, stream>>>(
      input, family.suh_f16, assignments, scales, assignment_count, tokens,
      input_stride, logical_elements, physical_elements, token_major_input);
  Status status = CheckLaunch("launch Trellis35 prefill transform scales");
  if (!status.ok()) return status;
  const unsigned blocks = static_cast<unsigned>(
      (physical_elements + kThreads - 1U) / kThreads);
  PrefillTransformQuantizeKernel<<<
      dim3(blocks, static_cast<unsigned>(assignment_count)), kThreads, 0,
      stream>>>(input, family.suh_f16, assignments, scales, output,
                assignment_count, tokens, input_stride, logical_elements,
                physical_elements, token_major_input);
  return CheckLaunch("launch direct Trellis35 prefill E4M3 quantization");
}

Status LaunchPrefillTransformQuantizeBf16(
    const std::uint16_t* input, const Trellis35DeviceFamilyBinding& family,
    const Gemma4MoePrefillAssignment* assignments, std::uint8_t* output,
    float* scales, std::uint64_t assignment_count,
    std::uint64_t input_stride, std::uint64_t logical_elements,
    std::uint64_t physical_elements, cudaStream_t stream) {
  if (input == nullptr || family.suh_f16 == nullptr || assignments == nullptr ||
      output == nullptr || scales == nullptr || assignment_count == 0U ||
      input_stride == 0U || logical_elements == 0U ||
      logical_elements > physical_elements || physical_elements % 128U != 0U ||
      assignment_count >
          static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max()) ||
      physical_elements >
          static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max()) *
              kThreads) {
    return Invalid("Trellis35 BF16 prefill transform/quantize contract is invalid");
  }
  PrefillTransformBf16ScaleKernel<<<static_cast<unsigned>(assignment_count),
                                    kThreads, 0, stream>>>(
      input, family.suh_f16, assignments, scales, assignment_count,
      input_stride, logical_elements, physical_elements);
  Status status = CheckLaunch("launch Trellis35 BF16 prefill transform scales");
  if (!status.ok()) return status;
  const unsigned blocks = static_cast<unsigned>(
      (physical_elements + kThreads - 1U) / kThreads);
  PrefillTransformQuantizeBf16Kernel<<<
      dim3(blocks, static_cast<unsigned>(assignment_count)), kThreads, 0,
      stream>>>(input, family.suh_f16, assignments, scales, output,
                assignment_count, input_stride, logical_elements,
                physical_elements);
  return CheckLaunch("launch direct Trellis35 BF16 prefill E4M3 quantization");
}

Status LaunchPrefillProjectionBlocks(
    const std::uint8_t* activation, const float* activation_scales,
    const Trellis35DeviceFamilyBinding& family,
    const Gemma4MoePrefillAssignment* assignments,
    const std::uint32_t* expert_prefix, const std::uint32_t* schedule_count,
    const std::uint32_t* schedule, const std::uint32_t* permutation,
    float* projection_tile, float* output, std::uint64_t tokens,
    std::uint64_t assignment_count, std::uint64_t input_elements,
    std::uint64_t output_elements, cudaStream_t stream) {
  if (activation == nullptr || activation_scales == nullptr ||
      family.k3_payload_pool == nullptr || family.k4_payload_pool == nullptr ||
      family.descriptors == nullptr || family.svh_f16 == nullptr ||
      assignments == nullptr || expert_prefix == nullptr ||
      schedule_count == nullptr || schedule == nullptr ||
      permutation == nullptr || projection_tile == nullptr ||
      output == nullptr || tokens == 0U || assignment_count == 0U ||
      input_elements == 0U || input_elements % 32U != 0U ||
      output_elements == 0U ||
      output_elements % kPrefillOutputBlock != 0U || tokens > 1024U ||
      assignment_count > 65535U) {
    return Invalid("Trellis35 grouped prefill projection contract is invalid");
  }
  const unsigned schedule_blocks = static_cast<unsigned>(assignment_count);
  const unsigned output_blocks =
      kPrefillOutputBlock / (kMmaWarps * 8U);
  for (std::uint64_t output_offset = 0U; output_offset < output_elements;
       output_offset += kPrefillOutputBlock) {
    MmaW4A8ProjectionGroupedPrefillTileKernel<<<
        dim3(output_blocks, schedule_blocks), kMmaThreads, 0, stream>>>(
        activation, activation_scales, family, expert_prefix, schedule_count,
        schedule, permutation, projection_tile, input_elements,
        output_elements, output_offset);
    Status status =
        CheckLaunch("launch grouped Trellis35 prefill W4A8 output tile");
    if (!status.ok()) return status;
    PrefillOutputTransformTileKernel<<<
        static_cast<unsigned>(assignment_count), kPrefillOutputBlock, 0,
        stream>>>(projection_tile, family.svh_f16, assignments, output,
                  assignment_count, output_elements, output_offset);
    status = CheckLaunch("launch Trellis35 prefill inverse output tile");
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

// Physical-BF16 output can alias the activation buffer. Process output tiles
// from high to low so each projection consumes the complete E4M3 input before
// the final low tile overwrites the aliased prefix.
Status LaunchPrefillProjectionBlocksBf16Reverse(
    const std::uint8_t* activation, const float* activation_scales,
    const Trellis35DeviceFamilyBinding& family,
    const Gemma4MoePrefillAssignment* assignments,
    const std::uint32_t* expert_prefix, const std::uint32_t* schedule_count,
    const std::uint32_t* schedule, const std::uint32_t* permutation,
    float* projection_tile, std::uint16_t* output, std::uint64_t tokens,
    std::uint64_t assignment_count, std::uint64_t input_elements,
    std::uint64_t output_elements, cudaStream_t stream) {
  if (activation == nullptr || activation_scales == nullptr ||
      family.k3_payload_pool == nullptr || family.k4_payload_pool == nullptr ||
      family.descriptors == nullptr || family.svh_f16 == nullptr ||
      assignments == nullptr || expert_prefix == nullptr ||
      schedule_count == nullptr || schedule == nullptr ||
      permutation == nullptr || projection_tile == nullptr ||
      output == nullptr || tokens == 0U || assignment_count == 0U ||
      input_elements == 0U || input_elements % 32U != 0U ||
      output_elements == 0U ||
      output_elements % kPrefillOutputBlock != 0U || tokens > 1024U ||
      assignment_count > 65535U) {
    return Invalid("Trellis35 grouped BF16 prefill projection contract is invalid");
  }
  const unsigned schedule_blocks = static_cast<unsigned>(assignment_count);
  const unsigned output_blocks = kPrefillOutputBlock / (kMmaWarps * 8U);
  for (std::uint64_t output_end = output_elements; output_end != 0U;
       output_end -= kPrefillOutputBlock) {
    const std::uint64_t output_offset = output_end - kPrefillOutputBlock;
    MmaW4A8ProjectionGroupedPrefillTileKernel<<<
        dim3(output_blocks, schedule_blocks), kMmaThreads, 0, stream>>>(
        activation, activation_scales, family, expert_prefix, schedule_count,
        schedule, permutation, projection_tile, input_elements,
        output_elements, output_offset);
    Status status =
        CheckLaunch("launch grouped Trellis35 BF16 prefill W4A8 output tile");
    if (!status.ok()) return status;
    PrefillOutputTransformTileBf16Kernel<<<
        static_cast<unsigned>(assignment_count), kPrefillOutputBlock, 0,
        stream>>>(projection_tile, family.svh_f16, assignments, output,
                  assignment_count, output_elements, output_offset);
    status = CheckLaunch("launch Trellis35 prefill BF16 inverse output tile");
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

}  // namespace

Status LaunchTrellis35InputTransformM1(
    const float* input, const std::uint16_t* suh_f16,
    float* transformed_output, std::uint64_t logical_elements,
    std::uint64_t physical_elements, cudaStream_t stream) {
  Status status = TransformArguments(input, suh_f16, transformed_output,
                                     physical_elements,
                                     "Trellis35 input transform");
  if (!status.ok()) return status;
  if (logical_elements == 0U || logical_elements > physical_elements) {
    return Invalid("Trellis35 logical input extent is invalid");
  }
  const unsigned blocks =
      static_cast<unsigned>((physical_elements + kThreads - 1U) / kThreads);
  InputTransformKernel<<<blocks, kThreads, 0, stream>>>(
      input, suh_f16, transformed_output, logical_elements, physical_elements);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch Trellis35 input transform", error);
}

Status LaunchTrellis35ReferenceW4A8ProjectionM1(
    const std::uint8_t* activation_e4m3, const float* activation_scale,
    const Trellis35DeviceFamilyBinding& family, std::uint32_t expert,
    float* transformed_output, std::uint64_t input_elements,
    std::uint64_t output_elements, cudaStream_t stream) {
  if (activation_e4m3 == nullptr || activation_scale == nullptr ||
      family.k3_payload_pool == nullptr || family.k4_payload_pool == nullptr ||
      family.descriptors == nullptr || transformed_output == nullptr) {
    return Invalid("Trellis35 W4A8 projection requires non-null pointers");
  }
  if (expert >= kTrellis35ExpertCount || input_elements == 0U ||
      output_elements == 0U || input_elements % 16U != 0U ||
      output_elements % 16U != 0U ||
      output_elements > static_cast<std::uint64_t>(
                            std::numeric_limits<unsigned>::max()) * kThreads) {
    return Invalid("Trellis35 W4A8 projection geometry is invalid");
  }
  const unsigned blocks =
      static_cast<unsigned>((output_elements + kThreads - 1U) / kThreads);
  ReferenceW4A8ProjectionKernel<<<blocks, kThreads, 0, stream>>>(
      activation_e4m3, activation_scale, family, expert, transformed_output,
      input_elements, output_elements);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch Trellis35 reference W4A8 projection",
                           error);
}

Status LaunchTrellis35MmaW4A8ProjectionM1(
    const std::uint8_t* activation_e4m3, const float* activation_scales,
    const Trellis35DeviceFamilyBinding& family,
    const std::uint32_t* selected_experts, float* transformed_output,
    std::uint64_t input_elements, std::uint64_t output_elements,
    cudaStream_t stream) {
  if (activation_e4m3 == nullptr || activation_scales == nullptr ||
      family.k3_payload_pool == nullptr || family.k4_payload_pool == nullptr ||
      family.descriptors == nullptr || selected_experts == nullptr ||
      transformed_output == nullptr) {
    return Invalid("Trellis35 MMA W4A8 projection requires non-null pointers");
  }
  if (input_elements == 0U || output_elements == 0U ||
      input_elements % 32U != 0U || output_elements % 8U != 0U ||
      output_elements >
          static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max()) *
              kMmaWarps * 8U) {
    return Invalid("Trellis35 MMA W4A8 projection geometry is invalid");
  }
  const unsigned blocks = static_cast<unsigned>(
      (output_elements + kMmaWarps * 8U - 1U) / (kMmaWarps * 8U));
  MmaW4A8ProjectionSelectedKernel<<<
      dim3(blocks, kTrellis35M1TopK), kMmaThreads, 0, stream>>>(
      activation_e4m3, activation_scales, family, selected_experts,
      transformed_output, input_elements, output_elements);
  return CheckLaunch("launch Trellis35 SM120 MMA W4A8 projection");
}

Status LaunchTrellis35OutputTransformM1(
    const float* transformed_input, const std::uint16_t* svh_f16,
    float* output, std::uint64_t elements, cudaStream_t stream) {
  Status status = TransformArguments(transformed_input, svh_f16, output,
                                     elements,
                                     "Trellis35 output transform");
  if (!status.ok()) return status;
  const unsigned blocks =
      static_cast<unsigned>((elements + kThreads - 1U) / kThreads);
  OutputTransformKernel<<<blocks, kThreads, 0, stream>>>(
      transformed_input, svh_f16, output, elements);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch Trellis35 output transform", error);
}

Status LaunchTrellis35SelectedExpertsM1(
    const float* input, const std::uint32_t* selected_experts,
    const float* route_weights, const Trellis35DeviceLayerBinding& layer,
    const Trellis35M1Workspace& workspace, float* output,
    cudaStream_t stream) {
  if (input == nullptr || selected_experts == nullptr ||
      route_weights == nullptr || output == nullptr ||
      layer.gate_up.k3_payload_pool == nullptr ||
      layer.gate_up.k4_payload_pool == nullptr ||
      layer.gate_up.descriptors == nullptr || layer.gate_up.suh_f16 == nullptr ||
      layer.gate_up.svh_f16 == nullptr ||
      layer.down.k3_payload_pool == nullptr ||
      layer.down.k4_payload_pool == nullptr ||
      layer.down.descriptors == nullptr || layer.down.suh_f16 == nullptr ||
      layer.down.svh_f16 == nullptr ||
      workspace.gate_up_input_transformed == nullptr ||
      workspace.gate_up_input_e4m3 == nullptr ||
      workspace.gate_up_input_scales == nullptr ||
      workspace.gate_up_transformed_output == nullptr ||
      workspace.gate_up_output == nullptr || workspace.product == nullptr ||
      workspace.down_input_transformed == nullptr ||
      workspace.down_input_e4m3 == nullptr ||
      workspace.down_input_scales == nullptr ||
      workspace.down_transformed_output == nullptr ||
      workspace.down_output == nullptr) {
    return Invalid("Trellis35 selected-expert M1 requires complete bindings");
  }

  const dim3 gate_input_blocks(
      static_cast<unsigned>((kTrellis35GateUpInput + kThreads - 1U) /
                            kThreads),
      kTrellis35M1TopK);
  SelectedInputTransformKernel<<<gate_input_blocks, kThreads, 0, stream>>>(
      input, layer.gate_up.suh_f16, selected_experts,
      workspace.gate_up_input_transformed, 0U, kTrellis35GateUpInput,
      kTrellis35GateUpInput);
  Status status = CheckLaunch("launch Trellis35 selected Gate+Up input transform");
  if (!status.ok()) return status;

  status = LaunchFp8ReferenceTokenQuantizationBatch(
      workspace.gate_up_input_transformed, workspace.gate_up_input_e4m3,
      workspace.gate_up_input_scales, kTrellis35M1TopK,
      kTrellis35GateUpInput, stream);
  if (!status.ok()) return status;
  status = LaunchTrellis35MmaW4A8ProjectionM1(
      workspace.gate_up_input_e4m3, workspace.gate_up_input_scales,
      layer.gate_up, selected_experts,
      workspace.gate_up_transformed_output, kTrellis35GateUpInput,
      kTrellis35GateUpOutput, stream);
  if (!status.ok()) return status;

  const dim3 gate_output_blocks(
      static_cast<unsigned>((kTrellis35GateUpOutput + kThreads - 1U) /
                            kThreads),
      kTrellis35M1TopK);
  SelectedOutputTransformKernel<<<gate_output_blocks, kThreads, 0, stream>>>(
      workspace.gate_up_transformed_output, layer.gate_up.svh_f16,
      selected_experts, workspace.gate_up_output, kTrellis35GateUpOutput);
  status = CheckLaunch("launch Trellis35 selected Gate+Up output transform");
  if (!status.ok()) return status;

  const dim3 product_blocks(
      static_cast<unsigned>((kTrellis35ExpertIntermediate + kThreads - 1U) /
                            kThreads),
      kTrellis35M1TopK);
  GatedGeluKernel<<<product_blocks, kThreads, 0, stream>>>(
      workspace.gate_up_output, workspace.product);
  status = CheckLaunch("launch Trellis35 selected gated GELU");
  if (!status.ok()) return status;

  const dim3 down_input_blocks(
      static_cast<unsigned>((kTrellis35DownInput + kThreads - 1U) / kThreads),
      kTrellis35M1TopK);
  SelectedInputTransformKernel<<<down_input_blocks, kThreads, 0, stream>>>(
      workspace.product, layer.down.suh_f16, selected_experts,
      workspace.down_input_transformed, kTrellis35ExpertIntermediate,
      kTrellis35ExpertIntermediate,
      kTrellis35DownInput);
  status = CheckLaunch("launch Trellis35 selected Down input transform");
  if (!status.ok()) return status;

  status = LaunchFp8ReferenceTokenQuantizationBatch(
      workspace.down_input_transformed, workspace.down_input_e4m3,
      workspace.down_input_scales, kTrellis35M1TopK, kTrellis35DownInput,
      stream);
  if (!status.ok()) return status;
  status = LaunchTrellis35MmaW4A8ProjectionM1(
      workspace.down_input_e4m3, workspace.down_input_scales, layer.down,
      selected_experts, workspace.down_transformed_output,
      kTrellis35DownInput, kTrellis35DownOutput, stream);
  if (!status.ok()) return status;

  const dim3 down_output_blocks(
      static_cast<unsigned>((kTrellis35DownOutput + kThreads - 1U) /
                            kThreads),
      kTrellis35M1TopK);
  SelectedOutputTransformKernel<<<down_output_blocks, kThreads, 0, stream>>>(
      workspace.down_transformed_output, layer.down.svh_f16,
      selected_experts, workspace.down_output, kTrellis35DownOutput);
  status = CheckLaunch("launch Trellis35 selected Down output transform");
  if (!status.ok()) return status;

  const unsigned reduction_blocks = static_cast<unsigned>(
      (kTrellis35DownOutput + kThreads - 1U) / kThreads);
  SlotOrderedReductionKernel<<<reduction_blocks, kThreads, 0, stream>>>(
      workspace.down_output, route_weights, output);
  return CheckLaunch("launch Trellis35 slot-ordered reduction");
}

Status LaunchTrellis35SelectedExpertsT3(
    const float* input_rows, const std::uint32_t* selected_experts,
    const float* route_weights, const Trellis35DeviceLayerBinding& layer,
    const Trellis35T3Workspace& workspace, float* output_rows,
    cudaStream_t stream) {
  if (input_rows == nullptr || selected_experts == nullptr ||
      route_weights == nullptr || output_rows == nullptr ||
      layer.gate_up.k3_payload_pool == nullptr ||
      layer.gate_up.k4_payload_pool == nullptr ||
      layer.gate_up.descriptors == nullptr || layer.gate_up.suh_f16 == nullptr ||
      layer.gate_up.svh_f16 == nullptr ||
      layer.down.k3_payload_pool == nullptr ||
      layer.down.k4_payload_pool == nullptr ||
      layer.down.descriptors == nullptr || layer.down.suh_f16 == nullptr ||
      layer.down.svh_f16 == nullptr ||
      workspace.gate_up_input_transformed == nullptr ||
      workspace.gate_up_input_e4m3 == nullptr ||
      workspace.gate_up_input_scales == nullptr ||
      workspace.gate_up_transformed_output == nullptr ||
      workspace.gate_up_output == nullptr || workspace.product == nullptr ||
      workspace.down_input_transformed == nullptr ||
      workspace.down_input_e4m3 == nullptr ||
      workspace.down_input_scales == nullptr ||
      workspace.down_transformed_output == nullptr ||
      workspace.down_output == nullptr) {
    return Invalid("Trellis35 selected-expert T3 requires complete bindings");
  }

  const dim3 gate_input_blocks(
      static_cast<unsigned>((kTrellis35GateUpInput + kThreads - 1U) /
                            kThreads),
      kTrellis35T3Assignments);
  T3InputTransformKernel<<<gate_input_blocks, kThreads, 0, stream>>>(
      input_rows, layer.gate_up.suh_f16, selected_experts,
      workspace.gate_up_input_transformed, kTrellis35GateUpInput,
      kTrellis35M1TopK, kTrellis35GateUpInput, kTrellis35GateUpInput);
  Status status =
      CheckLaunch("launch Trellis35 T3 Gate+Up input transform");
  if (!status.ok()) return status;
  status = LaunchFp8ReferenceTokenQuantizationBatch(
      workspace.gate_up_input_transformed, workspace.gate_up_input_e4m3,
      workspace.gate_up_input_scales, kTrellis35T3Assignments,
      kTrellis35GateUpInput, stream);
  if (!status.ok()) return status;
  status = LaunchGroupedT3Projection(
      workspace.gate_up_input_e4m3, workspace.gate_up_input_scales,
      layer.gate_up, selected_experts,
      workspace.gate_up_transformed_output, kTrellis35GateUpInput,
      kTrellis35GateUpOutput, stream);
  if (!status.ok()) return status;

  const dim3 gate_output_blocks(
      static_cast<unsigned>((kTrellis35GateUpOutput + kThreads - 1U) /
                            kThreads),
      kTrellis35T3Assignments);
  SelectedOutputTransformKernel<<<gate_output_blocks, kThreads, 0, stream>>>(
      workspace.gate_up_transformed_output, layer.gate_up.svh_f16,
      selected_experts, workspace.gate_up_output, kTrellis35GateUpOutput);
  status = CheckLaunch("launch Trellis35 T3 Gate+Up output transform");
  if (!status.ok()) return status;

  const dim3 product_blocks(
      static_cast<unsigned>((kTrellis35ExpertIntermediate + kThreads - 1U) /
                            kThreads),
      kTrellis35T3Assignments);
  GatedGeluKernel<<<product_blocks, kThreads, 0, stream>>>(
      workspace.gate_up_output, workspace.product);
  status = CheckLaunch("launch Trellis35 T3 gated GELU");
  if (!status.ok()) return status;

  const dim3 down_input_blocks(
      static_cast<unsigned>((kTrellis35DownInput + kThreads - 1U) / kThreads),
      kTrellis35T3Assignments);
  T3InputTransformKernel<<<down_input_blocks, kThreads, 0, stream>>>(
      workspace.product, layer.down.suh_f16, selected_experts,
      workspace.down_input_transformed, kTrellis35ExpertIntermediate, 1U,
      kTrellis35ExpertIntermediate, kTrellis35DownInput);
  status = CheckLaunch("launch Trellis35 T3 Down input transform");
  if (!status.ok()) return status;
  status = LaunchFp8ReferenceTokenQuantizationBatch(
      workspace.down_input_transformed, workspace.down_input_e4m3,
      workspace.down_input_scales, kTrellis35T3Assignments,
      kTrellis35DownInput, stream);
  if (!status.ok()) return status;
  status = LaunchGroupedT3Projection(
      workspace.down_input_e4m3, workspace.down_input_scales, layer.down,
      selected_experts, workspace.down_transformed_output,
      kTrellis35DownInput, kTrellis35DownOutput, stream);
  if (!status.ok()) return status;

  const dim3 down_output_blocks(
      static_cast<unsigned>((kTrellis35DownOutput + kThreads - 1U) /
                            kThreads),
      kTrellis35T3Assignments);
  SelectedOutputTransformKernel<<<down_output_blocks, kThreads, 0, stream>>>(
      workspace.down_transformed_output, layer.down.svh_f16,
      selected_experts, workspace.down_output, kTrellis35DownOutput);
  status = CheckLaunch("launch Trellis35 T3 Down output transform");
  if (!status.ok()) return status;

  const unsigned reduction_blocks = static_cast<unsigned>(
      (kTrellis35DownOutput + kThreads - 1U) / kThreads);
  SlotOrderedReductionT3Kernel<<<
      dim3(reduction_blocks, kTrellis35T3Rows), kThreads, 0, stream>>>(
      workspace.down_output, route_weights, output_rows);
  return CheckLaunch("launch Trellis35 T3 slot-ordered reduction");
}

Status LaunchTrellis35PrefillExpertsW4A8(
    const float* normalized_hidden, std::uint64_t tokens,
    const Trellis35DeviceLayerBinding& layer,
    const Gemma4MoePrefillWorkspace& workspace, cudaStream_t stream) {
  const std::uint64_t assignment_count = tokens * kTrellis35M1TopK;
  const bool float_boundaries = workspace.expert_product != nullptr &&
                                workspace.expert_down != nullptr &&
                                workspace.expert_product_bf16 == nullptr &&
                                workspace.expert_down_bf16 == nullptr;
  const bool physical_bf16_boundaries = workspace.expert_product == nullptr &&
                                        workspace.expert_down == nullptr &&
                                        workspace.expert_product_bf16 != nullptr &&
                                        workspace.expert_down_bf16 != nullptr;
  if (normalized_hidden == nullptr || tokens == 0U || tokens > 1024U ||
      assignment_count > 65535U ||
      (!float_boundaries && !physical_bf16_boundaries) ||
      (physical_bf16_boundaries && workspace.trellis_activation == nullptr) ||
      workspace.token_scales == nullptr ||
      workspace.shared_product == nullptr || workspace.shared_output == nullptr ||
      workspace.token_hidden == nullptr || workspace.assignments == nullptr ||
      workspace.prefix == nullptr || workspace.permutation == nullptr ||
      workspace.router_logits == nullptr || workspace.histogram == nullptr) {
    return Invalid("Trellis35 prefill expert workspace is incomplete");
  }
  auto* activation_scales =
      reinterpret_cast<float*>(workspace.token_scales);
  auto* projection_tile = workspace.shared_product;
  auto* schedule = reinterpret_cast<std::uint32_t*>(workspace.router_logits);

  BuildTrellis35PrefillTileScheduleKernel<<<1, 1, 0, stream>>>(
      workspace.prefix, workspace.histogram, schedule);
  Status status = CheckLaunch("build Trellis35 prefill expert tile schedule");
  if (!status.ok()) return status;

  const dim3 product_blocks(
      static_cast<unsigned>((kTrellis35ExpertIntermediate + kThreads - 1U) /
                            kThreads),
      static_cast<unsigned>(assignment_count));
  if (physical_bf16_boundaries) {
    auto* aliased_activation = workspace.trellis_activation;
    status = LaunchPrefillTransformQuantize(
        normalized_hidden, layer.gate_up, workspace.assignments,
        aliased_activation, activation_scales, assignment_count, tokens,
        kTrellis35GateUpInput, kTrellis35GateUpInput,
        kTrellis35GateUpInput, true, stream);
    if (!status.ok()) return status;
    status = LaunchPrefillProjectionBlocksBf16Reverse(
        aliased_activation, activation_scales, layer.gate_up,
        workspace.assignments, workspace.prefix, workspace.histogram,
        schedule, workspace.permutation, projection_tile,
        workspace.expert_down_bf16, tokens, assignment_count,
        kTrellis35GateUpInput, kTrellis35GateUpOutput, stream);
    if (!status.ok()) return status;
    GatedGeluBf16Kernel<<<product_blocks, kThreads, 0, stream>>>(
        workspace.expert_down_bf16, workspace.expert_product_bf16);
    status = CheckLaunch("launch Trellis35 physical-BF16 prefill gated GELU");
    if (!status.ok()) return status;
    status = LaunchPrefillTransformQuantizeBf16(
        workspace.expert_product_bf16, layer.down, workspace.assignments,
        aliased_activation, activation_scales, assignment_count,
        kTrellis35ExpertIntermediate, kTrellis35ExpertIntermediate,
        kTrellis35DownInput, stream);
    if (!status.ok()) return status;
    status = LaunchPrefillProjectionBlocksBf16Reverse(
        aliased_activation, activation_scales, layer.down,
        workspace.assignments, workspace.prefix, workspace.histogram,
        schedule, workspace.permutation, projection_tile,
        workspace.expert_down_bf16, tokens, assignment_count,
        kTrellis35DownInput, kTrellis35DownOutput, stream);
    if (!status.ok()) return status;
    status = LaunchGemma4MoeReduceAssignmentsBf16(
        workspace.expert_down_bf16, workspace.assignments,
        workspace.token_hidden, kTrellis35DownOutput, kTrellis35M1TopK,
        tokens, stream);
  } else {
    auto* gate_activation =
        reinterpret_cast<std::uint8_t*>(workspace.expert_product);
    auto* down_activation =
        reinterpret_cast<std::uint8_t*>(workspace.shared_output);
    status = LaunchPrefillTransformQuantize(
        normalized_hidden, layer.gate_up, workspace.assignments,
        gate_activation, activation_scales, assignment_count, tokens,
        kTrellis35GateUpInput, kTrellis35GateUpInput,
        kTrellis35GateUpInput, true, stream);
    if (!status.ok()) return status;
    status = LaunchPrefillProjectionBlocks(
        gate_activation, activation_scales, layer.gate_up,
        workspace.assignments, workspace.prefix, workspace.histogram,
        schedule, workspace.permutation, projection_tile,
        workspace.expert_down, tokens, assignment_count,
        kTrellis35GateUpInput, kTrellis35GateUpOutput, stream);
    if (!status.ok()) return status;
    GatedGeluKernel<<<product_blocks, kThreads, 0, stream>>>(
        workspace.expert_down, workspace.expert_product);
    status = CheckLaunch("launch Trellis35 prefill gated GELU");
    if (!status.ok()) return status;
    status = LaunchPrefillTransformQuantize(
        workspace.expert_product, layer.down, workspace.assignments,
        down_activation, activation_scales, assignment_count, tokens,
        kTrellis35ExpertIntermediate, kTrellis35ExpertIntermediate,
        kTrellis35DownInput, false, stream);
    if (!status.ok()) return status;
    status = LaunchPrefillProjectionBlocks(
        down_activation, activation_scales, layer.down, workspace.assignments,
        workspace.prefix, workspace.histogram, schedule,
        workspace.permutation, projection_tile, workspace.expert_down, tokens,
        assignment_count, kTrellis35DownInput, kTrellis35DownOutput, stream);
    if (!status.ok()) return status;
    status = LaunchGemma4MoeReduceAssignments(
        workspace.expert_down, workspace.assignments, workspace.token_hidden,
        kTrellis35DownOutput, kTrellis35M1TopK, tokens, stream);
  }
  if (!status.ok()) return status;
  RestoreTrellis35PrefillHistogramZeroKernel<<<1, 1, 0, stream>>>(
      workspace.histogram, workspace.prefix);
  return CheckLaunch("restore Trellis35 prefill expert-zero histogram");
}

}  // namespace gem16::internal

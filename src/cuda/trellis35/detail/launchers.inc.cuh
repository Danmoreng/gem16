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

bool Trellis35PrefillGeluDownFusionEnabled() {
  static const bool enabled = [] {
    const char* value =
        GetEnvironmentVariable("GEM16_TRELLIS35_PREFILL_GELU_DOWN_FUSION");
    return value == nullptr || std::string_view(value) != "0";
  }();
  return enabled;
}

bool Trellis35SmallGeluDownFusionEnabled() {
  static const bool enabled = [] {
    const char* value =
        GetEnvironmentVariable("GEM16_TRELLIS35_SMALL_GELU_DOWN_FUSION");
    return value == nullptr || std::string_view(value) != "0";
  }();
  return enabled;
}

bool Trellis35M1NativeFp8x4Enabled() {
  static const bool enabled = [] {
    const char* value =
        GetEnvironmentVariable("GEM16_TRELLIS35_M1_NATIVE_FP8X4");
    return value == nullptr || std::string_view(value) != "0";
  }();
  return enabled;
}

bool Trellis35T3NativeFp8x4Enabled() {
  static const bool enabled = [] {
    const char* value =
        GetEnvironmentVariable("GEM16_TRELLIS35_T3_NATIVE_FP8X4");
    return value == nullptr || std::string_view(value) != "0";
  }();
  return enabled;
}

bool Trellis35T3VectorStoreEnvironmentEnabled() {
  static const bool enabled = [] {
    const char* value =
        GetEnvironmentVariable("GEM16_TRELLIS35_T3_VECTOR_STORE");
    return value == nullptr || std::string_view(value) != "0";
  }();
  return enabled;
}

bool Trellis35M1VectorStoreEnvironmentEnabled() {
  static const bool enabled = [] {
    const char* value =
        GetEnvironmentVariable("GEM16_TRELLIS35_M1_VECTOR_STORE");
    return value == nullptr || std::string_view(value) != "0";
  }();
  return enabled;
}

Trellis35M1ProjectionOutputMode Trellis35M1ProjectionOutputEnvironmentMode() {
  static const Trellis35M1ProjectionOutputMode mode = [] {
    const char* value =
        GetEnvironmentVariable("GEM16_TRELLIS35_M1_N128_INVERSE");
    if (value == nullptr || std::string_view(value) == "1" ||
        std::string_view(value) == "both") {
      return Trellis35M1ProjectionOutputMode::kFusedN128;
    }
    if (std::string_view(value) == "gate") {
      return Trellis35M1ProjectionOutputMode::kGateUpFusedN128;
    }
    if (std::string_view(value) == "down") {
      return Trellis35M1ProjectionOutputMode::kDownFusedN128;
    }
    return Trellis35M1ProjectionOutputMode::kSeparateN32;
  }();
  return mode;
}

Trellis35T3ProjectionOutputMode Trellis35T3ProjectionOutputEnvironmentMode() {
  static const Trellis35T3ProjectionOutputMode mode = [] {
    const char* value =
        GetEnvironmentVariable("GEM16_TRELLIS35_T3_N128_INVERSE");
    if (value == nullptr || std::string_view(value) == "1" ||
        std::string_view(value) == "both") {
      return Trellis35T3ProjectionOutputMode::kFusedN128;
    }
    if (std::string_view(value) == "gate") {
      return Trellis35T3ProjectionOutputMode::kGateUpFusedN128;
    }
    if (std::string_view(value) == "down") {
      return Trellis35T3ProjectionOutputMode::kDownFusedN128;
    }
    return Trellis35T3ProjectionOutputMode::kSeparateN32;
  }();
  return mode;
}

bool Trellis35PrefillScheduleTrimEnabled() {
  static const bool enabled = [] {
    const char* value =
        GetEnvironmentVariable("GEM16_TRELLIS35_PREFILL_SCHEDULE_TRIM");
    return value == nullptr || std::string_view(value) != "0";
  }();
  return enabled;
}

template <bool NativeFp8x4, bool VectorStore>
void LaunchGroupedT3N128(
    unsigned blocks, const std::uint8_t* activation_e4m3,
    const float* activation_scales,
    const Trellis35DeviceFamilyBinding& family,
    const std::uint32_t* selected_experts, float* output,
    std::uint64_t input_elements, std::uint64_t output_elements,
    cudaStream_t stream) {
  MmaW4A8ProjectionGroupedT3M16N128InverseKernel<NativeFp8x4, VectorStore>
      <<<dim3(blocks, kTrellis35T3Assignments), kMmaN128Threads, 0, stream>>>(
          activation_e4m3, activation_scales, family, selected_experts, output,
          input_elements, output_elements);
}

Status LaunchGroupedT3Projection(
    const std::uint8_t* activation_e4m3, const float* activation_scales,
    const Trellis35DeviceFamilyBinding& family,
    const std::uint32_t* selected_experts, float* output,
    std::uint64_t input_elements, std::uint64_t output_elements,
    Trellis35T3ProjectionMode projection_mode, bool fused_inverse,
    bool vector_store, cudaStream_t stream) {
  if (activation_e4m3 == nullptr || activation_scales == nullptr ||
      family.k3_payload_pool == nullptr || family.k4_payload_pool == nullptr ||
      family.descriptors == nullptr || selected_experts == nullptr ||
      output == nullptr ||
      (projection_mode != Trellis35T3ProjectionMode::kIndependentRows &&
       projection_mode != Trellis35T3ProjectionMode::kM16) ||
      (fused_inverse && family.svh_f16 == nullptr)) {
    return Invalid("Trellis35 grouped T3 projection requires non-null pointers");
  }
  if (input_elements == 0U || output_elements == 0U ||
      input_elements % 32U != 0U || output_elements % 8U != 0U ||
      (fused_inverse && output_elements % 128U != 0U) ||
      (fused_inverse && vector_store &&
       reinterpret_cast<std::uintptr_t>(output) % alignof(float4) != 0U) ||
      output_elements >
          static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max()) *
              kMmaWarps * 8U) {
    return Invalid("Trellis35 grouped T3 projection geometry is invalid");
  }
  if (fused_inverse) {
    if (projection_mode != Trellis35T3ProjectionMode::kM16) {
      return Invalid("Trellis35 fused T3 inverse requires M16 projection");
    }
    const unsigned blocks = static_cast<unsigned>(output_elements / 128U);
    if (Trellis35T3NativeFp8x4Enabled()) {
      if (vector_store) {
        LaunchGroupedT3N128<true, true>(
            blocks, activation_e4m3, activation_scales, family,
            selected_experts, output, input_elements, output_elements, stream);
      } else {
        LaunchGroupedT3N128<true, false>(
            blocks, activation_e4m3, activation_scales, family,
            selected_experts, output, input_elements, output_elements, stream);
      }
    } else {
      if (vector_store) {
        LaunchGroupedT3N128<false, true>(
            blocks, activation_e4m3, activation_scales, family,
            selected_experts, output, input_elements, output_elements, stream);
      } else {
        LaunchGroupedT3N128<false, false>(
            blocks, activation_e4m3, activation_scales, family,
            selected_experts, output, input_elements, output_elements, stream);
      }
    }
    return CheckLaunch(
        "launch Trellis35 grouped T3 M16 N128 projection and inverse");
  }
  const unsigned blocks = static_cast<unsigned>(
      (output_elements + kMmaWarps * 8U - 1U) / (kMmaWarps * 8U));
  if (projection_mode == Trellis35T3ProjectionMode::kM16) {
    if (Trellis35T3NativeFp8x4Enabled()) {
      MmaW4A8ProjectionGroupedT3M16Kernel<true><<<
          dim3(blocks, kTrellis35T3Assignments), kMmaThreads, 0, stream>>>(
          activation_e4m3, activation_scales, family, selected_experts, output,
          input_elements, output_elements);
    } else {
      MmaW4A8ProjectionGroupedT3M16Kernel<false><<<
          dim3(blocks, kTrellis35T3Assignments), kMmaThreads, 0, stream>>>(
          activation_e4m3, activation_scales, family, selected_experts, output,
          input_elements, output_elements);
    }
  } else {
    MmaW4A8ProjectionGroupedT3Kernel<<<
        dim3(blocks, kTrellis35T3Assignments), kMmaThreads, 0, stream>>>(
        activation_e4m3, activation_scales, family, selected_experts,
        output, input_elements, output_elements);
  }
  return CheckLaunch("launch Trellis35 grouped T3 W4A8 projection");
}

Status LaunchPrefillTransformQuantize(
    const float* input, const Trellis35DeviceFamilyBinding& family,
    const Gemma4MoePrefillAssignment* assignments, std::uint8_t* output,
    float* scales, std::uint64_t assignment_count, std::uint64_t tokens,
    std::uint64_t input_stride, std::uint64_t logical_elements,
    std::uint64_t physical_elements, bool token_major_input,
    Trellis35PrefillTransformMode transform_mode, cudaStream_t stream) {
  if (input == nullptr || family.suh_f16 == nullptr || assignments == nullptr ||
      output == nullptr || scales == nullptr || assignment_count == 0U ||
      tokens == 0U || input_stride == 0U || logical_elements == 0U ||
      logical_elements > physical_elements || physical_elements % 128U != 0U ||
      physical_elements > kTrellis35GateUpInput ||
      assignment_count >
          static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max()) ||
      physical_elements >
          static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max()) *
              kThreads) {
    return Invalid("Trellis35 prefill transform/quantize contract is invalid");
  }
  if (transform_mode == Trellis35PrefillTransformMode::kWarpH128) {
    const PrefillFloatInputPolicy policy{input, tokens, input_stride,
                                         token_major_input};
    if (physical_elements == kTrellis35GateUpInput) {
      PrefillTransformQuantizeWarpKernel<kTrellis35GateUpInput><<<
          static_cast<unsigned>(assignment_count), kThreads, 0, stream>>>(
          policy, family.suh_f16, assignments, scales, output,
          assignment_count, logical_elements);
    } else if (physical_elements == kTrellis35DownInput) {
      PrefillTransformQuantizeWarpKernel<kTrellis35DownInput><<<
          static_cast<unsigned>(assignment_count), kThreads, 0, stream>>>(
          policy, family.suh_f16, assignments, scales, output,
          assignment_count, logical_elements);
    } else {
      return Invalid("Trellis35 warp prefill transform extent is unsupported");
    }
    return CheckLaunch(
        "launch warp-fused Trellis35 prefill E4M3 quantization");
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
    std::uint64_t physical_elements,
    Trellis35PrefillTransformMode transform_mode, cudaStream_t stream) {
  if (input == nullptr || family.suh_f16 == nullptr || assignments == nullptr ||
      output == nullptr || scales == nullptr || assignment_count == 0U ||
      input_stride == 0U || logical_elements == 0U ||
      logical_elements > physical_elements || physical_elements % 128U != 0U ||
      physical_elements > kTrellis35GateUpInput ||
      assignment_count >
          static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max()) ||
      physical_elements >
          static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max()) *
              kThreads) {
    return Invalid("Trellis35 BF16 prefill transform/quantize contract is invalid");
  }
  if (transform_mode == Trellis35PrefillTransformMode::kWarpH128) {
    const PrefillBf16InputPolicy policy{input, input_stride};
    if (physical_elements == kTrellis35GateUpInput) {
      PrefillTransformQuantizeWarpKernel<kTrellis35GateUpInput><<<
          static_cast<unsigned>(assignment_count), kThreads, 0, stream>>>(
          policy, family.suh_f16, assignments, scales, output,
          assignment_count, logical_elements);
    } else if (physical_elements == kTrellis35DownInput) {
      PrefillTransformQuantizeWarpKernel<kTrellis35DownInput><<<
          static_cast<unsigned>(assignment_count), kThreads, 0, stream>>>(
          policy, family.suh_f16, assignments, scales, output,
          assignment_count, logical_elements);
    } else {
      return Invalid(
          "Trellis35 BF16 warp prefill transform extent is unsupported");
    }
    return CheckLaunch(
        "launch warp-fused Trellis35 BF16 prefill E4M3 quantization");
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
    std::uint64_t output_elements, Trellis35PrefillKernelMode kernel_mode,
    Trellis35PrefillTransformMode transform_mode,
    Trellis35PrefillOutputMode output_mode, cudaStream_t stream) {
  if (activation == nullptr || activation_scales == nullptr ||
      family.k3_payload_pool == nullptr || family.k4_payload_pool == nullptr ||
      family.descriptors == nullptr || family.svh_f16 == nullptr ||
      assignments == nullptr || expert_prefix == nullptr ||
      schedule_count == nullptr || schedule == nullptr ||
      permutation == nullptr || projection_tile == nullptr ||
      output == nullptr || tokens == 0U || assignment_count == 0U ||
      input_elements == 0U || input_elements % 32U != 0U ||
      output_elements == 0U ||
      output_elements % kPrefillOutputBlock != 0U || tokens > 2048U ||
      assignment_count > 65535U) {
    return Invalid("Trellis35 grouped prefill projection contract is invalid");
  }
  const std::uint64_t active_expert_upper_bound =
      std::min<std::uint64_t>(assignment_count, kTrellis35ExpertCount);
  const unsigned rows_per_tile =
      kernel_mode == Trellis35PrefillKernelMode::kGroupedM32
          ? kPrefillGroupedRowsPerTile
          : kPrefillLegacyRowsPerTile;
  const unsigned schedule_blocks = static_cast<unsigned>(
      (assignment_count + (rows_per_tile - 1U) * active_expert_upper_bound) /
      rows_per_tile);
  const unsigned output_blocks =
      kPrefillOutputBlock / (kMmaWarps * 8U);
  const bool schedule_trim = Trellis35PrefillScheduleTrimEnabled();
  if (output_mode == Trellis35PrefillOutputMode::kFusedN128) {
    if (transform_mode != Trellis35PrefillTransformMode::kWarpH128 ||
        (kernel_mode != Trellis35PrefillKernelMode::kGroupedM32 &&
         kernel_mode != Trellis35PrefillKernelMode::kGroupedM64Hybrid)) {
      return Invalid(
          "Trellis35 fused N128 prefill requires grouped MMA and Warp-H128");
    }
    if (kernel_mode == Trellis35PrefillKernelMode::kGroupedM64Hybrid) {
      const unsigned m64_schedule_blocks = static_cast<unsigned>(
          schedule_trim
              ? std::max<std::uint64_t>(
                    1U, (assignment_count +
                         (kPrefillGroupedRowsPerTile - 1U) *
                             active_expert_upper_bound) /
                            kPrefillM64RowsPerTile)
              : (assignment_count + (kPrefillM64RowsPerTile - 1U) *
                                        active_expert_upper_bound) /
                    kPrefillM64RowsPerTile);
      MmaW4A8ProjectionGroupedPrefillM64N128Kernel<<<
          dim3(static_cast<unsigned>(output_elements / kPrefillOutputBlock),
               m64_schedule_blocks),
          kMmaN128Threads, 0, stream>>>(
          activation, activation_scales, family, assignments, expert_prefix,
          schedule_count, schedule, permutation,
          PrefillFloatOutputPolicy{output}, assignment_count, input_elements,
          output_elements);
      Status status = CheckLaunch(
          "launch M64 fused-N128 Trellis35 prefill projection and inverse");
      if (!status.ok()) return status;
      MmaW4A8ProjectionGroupedPrefillM32N128Kernel<<<
          dim3(static_cast<unsigned>(output_elements / kPrefillOutputBlock),
               static_cast<unsigned>(active_expert_upper_bound)),
          kMmaN128Threads, 0, stream>>>(
          activation, activation_scales, family, assignments, expert_prefix,
          schedule_count, schedule, permutation, 1U, true,
          schedule_trim, PrefillFloatOutputPolicy{output}, assignment_count,
          input_elements, output_elements);
      return CheckLaunch(
          "launch M32-tail fused-N128 Trellis35 prefill projection and inverse");
    }
    MmaW4A8ProjectionGroupedPrefillM32N128Kernel<<<
        dim3(static_cast<unsigned>(output_elements / kPrefillOutputBlock),
             schedule_blocks),
        kMmaN128Threads, 0, stream>>>(
        activation, activation_scales, family, assignments, expert_prefix,
        schedule_count, schedule, permutation, 0U, false,
        schedule_trim, PrefillFloatOutputPolicy{output}, assignment_count,
        input_elements, output_elements);
    return CheckLaunch(
        "launch fused-N128 Trellis35 prefill projection and inverse");
  }
  if (kernel_mode == Trellis35PrefillKernelMode::kGroupedM64Hybrid) {
    return Invalid("Trellis35 M64 prefill requires fused N128 output");
  }
  for (std::uint64_t output_offset = 0U; output_offset < output_elements;
       output_offset += kPrefillOutputBlock) {
    if (kernel_mode == Trellis35PrefillKernelMode::kGroupedM32) {
      MmaW4A8ProjectionGroupedPrefillM32Kernel<<<
          dim3(output_blocks, schedule_blocks), kMmaThreads, 0, stream>>>(
          activation, activation_scales, family, assignments, expert_prefix,
          schedule_count, schedule, permutation, projection_tile,
          assignment_count, input_elements, output_elements, output_offset);
    } else {
      MmaW4A8ProjectionGroupedPrefillTileKernel<<<
          dim3(output_blocks, schedule_blocks), kMmaThreads, 0, stream>>>(
          activation, activation_scales, family, expert_prefix,
          schedule_count, schedule, permutation, projection_tile,
          input_elements, output_elements, output_offset);
    }
    Status status =
        CheckLaunch("launch grouped Trellis35 prefill W4A8 output tile");
    if (!status.ok()) return status;
    if (transform_mode == Trellis35PrefillTransformMode::kWarpH128) {
      const unsigned transform_blocks =
          static_cast<unsigned>((assignment_count + 7U) / 8U);
      PrefillOutputTransformTileWarpKernel<<<transform_blocks, kThreads, 0,
                                             stream>>>(
          projection_tile, family.svh_f16, assignments,
          PrefillFloatOutputPolicy{output}, assignment_count, output_elements,
          output_offset);
    } else {
      PrefillOutputTransformTileKernel<<<
          static_cast<unsigned>(assignment_count), kPrefillOutputBlock, 0,
          stream>>>(projection_tile, family.svh_f16, assignments, output,
                    assignment_count, output_elements, output_offset);
    }
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
    std::uint64_t output_elements, Trellis35PrefillKernelMode kernel_mode,
    Trellis35PrefillTransformMode transform_mode,
    Trellis35PrefillOutputMode output_mode, cudaStream_t stream) {
  if (activation == nullptr || activation_scales == nullptr ||
      family.k3_payload_pool == nullptr || family.k4_payload_pool == nullptr ||
      family.descriptors == nullptr || family.svh_f16 == nullptr ||
      assignments == nullptr || expert_prefix == nullptr ||
      schedule_count == nullptr || schedule == nullptr ||
      permutation == nullptr || projection_tile == nullptr ||
      output == nullptr || tokens == 0U || assignment_count == 0U ||
      input_elements == 0U || input_elements % 32U != 0U ||
      output_elements == 0U ||
      output_elements % kPrefillOutputBlock != 0U || tokens > 2048U ||
      assignment_count > 65535U) {
    return Invalid("Trellis35 grouped BF16 prefill projection contract is invalid");
  }
  const std::uint64_t active_expert_upper_bound =
      std::min<std::uint64_t>(assignment_count, kTrellis35ExpertCount);
  const unsigned rows_per_tile =
      kernel_mode == Trellis35PrefillKernelMode::kGroupedM32
          ? kPrefillGroupedRowsPerTile
          : kPrefillLegacyRowsPerTile;
  const unsigned schedule_blocks = static_cast<unsigned>(
      (assignment_count + (rows_per_tile - 1U) * active_expert_upper_bound) /
      rows_per_tile);
  const unsigned output_blocks = kPrefillOutputBlock / (kMmaWarps * 8U);
  const bool schedule_trim = Trellis35PrefillScheduleTrimEnabled();
  if (output_mode == Trellis35PrefillOutputMode::kFusedN128) {
    if (transform_mode != Trellis35PrefillTransformMode::kWarpH128 ||
        (kernel_mode != Trellis35PrefillKernelMode::kGroupedM32 &&
         kernel_mode != Trellis35PrefillKernelMode::kGroupedM64Hybrid)) {
      return Invalid(
          "Trellis35 fused N128 BF16 prefill requires grouped MMA and Warp-H128");
    }
    if (kernel_mode == Trellis35PrefillKernelMode::kGroupedM64Hybrid) {
      const unsigned m64_schedule_blocks = static_cast<unsigned>(
          schedule_trim
              ? std::max<std::uint64_t>(
                    1U, (assignment_count +
                         (kPrefillGroupedRowsPerTile - 1U) *
                             active_expert_upper_bound) /
                            kPrefillM64RowsPerTile)
              : (assignment_count + (kPrefillM64RowsPerTile - 1U) *
                                        active_expert_upper_bound) /
                    kPrefillM64RowsPerTile);
      MmaW4A8ProjectionGroupedPrefillM64N128Kernel<<<
          dim3(static_cast<unsigned>(output_elements / kPrefillOutputBlock),
               m64_schedule_blocks),
          kMmaN128Threads, 0, stream>>>(
          activation, activation_scales, family, assignments, expert_prefix,
          schedule_count, schedule, permutation,
          PrefillBf16OutputPolicy{output}, assignment_count, input_elements,
          output_elements);
      Status status = CheckLaunch(
          "launch M64 fused-N128 Trellis35 BF16 prefill projection and inverse");
      if (!status.ok()) return status;
      MmaW4A8ProjectionGroupedPrefillM32N128Kernel<<<
          dim3(static_cast<unsigned>(output_elements / kPrefillOutputBlock),
               static_cast<unsigned>(active_expert_upper_bound)),
          kMmaN128Threads, 0, stream>>>(
          activation, activation_scales, family, assignments, expert_prefix,
          schedule_count, schedule, permutation, 1U, true,
          schedule_trim, PrefillBf16OutputPolicy{output}, assignment_count,
          input_elements, output_elements);
      return CheckLaunch(
          "launch M32-tail fused-N128 Trellis35 BF16 prefill projection and inverse");
    }
    MmaW4A8ProjectionGroupedPrefillM32N128Kernel<<<
        dim3(static_cast<unsigned>(output_elements / kPrefillOutputBlock),
             schedule_blocks),
        kMmaN128Threads, 0, stream>>>(
        activation, activation_scales, family, assignments, expert_prefix,
        schedule_count, schedule, permutation, 0U, false,
        schedule_trim, PrefillBf16OutputPolicy{output}, assignment_count,
        input_elements, output_elements);
    return CheckLaunch(
        "launch fused-N128 Trellis35 BF16 prefill projection and inverse");
  }
  if (kernel_mode == Trellis35PrefillKernelMode::kGroupedM64Hybrid) {
    return Invalid("Trellis35 M64 BF16 prefill requires fused N128 output");
  }
  for (std::uint64_t output_end = output_elements; output_end != 0U;
       output_end -= kPrefillOutputBlock) {
    const std::uint64_t output_offset = output_end - kPrefillOutputBlock;
    if (kernel_mode == Trellis35PrefillKernelMode::kGroupedM32) {
      MmaW4A8ProjectionGroupedPrefillM32Kernel<<<
          dim3(output_blocks, schedule_blocks), kMmaThreads, 0, stream>>>(
          activation, activation_scales, family, assignments, expert_prefix,
          schedule_count, schedule, permutation, projection_tile,
          assignment_count, input_elements, output_elements, output_offset);
    } else {
      MmaW4A8ProjectionGroupedPrefillTileKernel<<<
          dim3(output_blocks, schedule_blocks), kMmaThreads, 0, stream>>>(
          activation, activation_scales, family, expert_prefix,
          schedule_count, schedule, permutation, projection_tile,
          input_elements, output_elements, output_offset);
    }
    Status status =
        CheckLaunch("launch grouped Trellis35 BF16 prefill W4A8 output tile");
    if (!status.ok()) return status;
    if (transform_mode == Trellis35PrefillTransformMode::kWarpH128) {
      const unsigned transform_blocks =
          static_cast<unsigned>((assignment_count + 7U) / 8U);
      PrefillOutputTransformTileWarpKernel<<<transform_blocks, kThreads, 0,
                                             stream>>>(
          projection_tile, family.svh_f16, assignments,
          PrefillBf16OutputPolicy{output}, assignment_count, output_elements,
          output_offset);
    } else {
      PrefillOutputTransformTileBf16Kernel<<<
          static_cast<unsigned>(assignment_count), kPrefillOutputBlock, 0,
          stream>>>(projection_tile, family.svh_f16, assignments, output,
                    assignment_count, output_elements, output_offset);
    }
    status = CheckLaunch("launch Trellis35 prefill BF16 inverse output tile");
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

}  // namespace

Status LaunchTrellis35GatedGeluDownTransformQuantizeBf16(
    const std::uint16_t* gate_up_bf16, std::uint16_t* product_bf16,
    const Trellis35DeviceFamilyBinding& down,
    const Gemma4MoePrefillAssignment* assignments,
    std::uint8_t* down_activation_e4m3, float* down_activation_scales,
    std::uint64_t assignment_count, Trellis35PrefillGeluDownMode mode,
    cudaStream_t stream) {
  if (gate_up_bf16 == nullptr || product_bf16 == nullptr ||
      down.suh_f16 == nullptr || assignments == nullptr ||
      down_activation_e4m3 == nullptr || down_activation_scales == nullptr ||
      assignment_count == 0U ||
      assignment_count >
          static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max()) ||
      (mode != Trellis35PrefillGeluDownMode::kTwoKernel &&
       mode !=
           Trellis35PrefillGeluDownMode::kFusedTransformQuantize)) {
    return Invalid(
        "Trellis35 BF16 gated-GELU/Down transform contract is invalid");
  }
  if (mode == Trellis35PrefillGeluDownMode::kFusedTransformQuantize) {
    GatedGeluDownTransformQuantizeBf16WarpKernel<<<
        static_cast<unsigned>(assignment_count), kThreads, 0, stream>>>(
        gate_up_bf16, down.suh_f16, assignments, down_activation_scales,
        down_activation_e4m3, assignment_count);
    return CheckLaunch(
        "launch fused Trellis35 BF16 gated-GELU/Down E4M3 transform");
  }

  const dim3 product_blocks(
      static_cast<unsigned>((kTrellis35ExpertIntermediate + kThreads - 1U) /
                            kThreads),
      static_cast<unsigned>(assignment_count));
  GatedGeluBf16Kernel<<<product_blocks, kThreads, 0, stream>>>(
      gate_up_bf16, product_bf16);
  Status status =
      CheckLaunch("launch rollback Trellis35 physical-BF16 prefill gated GELU");
  if (!status.ok()) return status;
  return LaunchPrefillTransformQuantizeBf16(
      product_bf16, down, assignments, down_activation_e4m3,
      down_activation_scales, assignment_count,
      kTrellis35ExpertIntermediate, kTrellis35ExpertIntermediate,
      kTrellis35DownInput, Trellis35PrefillTransformMode::kWarpH128, stream);
}

Status LaunchTrellis35H128WarpDiagnostic(const float* input, float* output,
                                         std::uint64_t vectors,
                                         cudaStream_t stream) {
  if (input == nullptr || output == nullptr || vectors == 0U ||
      vectors > static_cast<std::uint64_t>(
                    std::numeric_limits<unsigned>::max()) * 8U) {
    return Invalid("Trellis35 H128 warp diagnostic contract is invalid");
  }
  const unsigned blocks = static_cast<unsigned>((vectors + 7U) / 8U);
  H128WarpDiagnosticKernel<<<blocks, kThreads, 0, stream>>>(input, output,
                                                            vectors);
  return CheckLaunch("launch Trellis35 H128 warp diagnostic");
}

Status LaunchTrellis35DecodeE4M3SlabDiagnostic(
    const Trellis35DeviceFamilyBinding& family, std::uint32_t expert,
    std::uint64_t input_elements, std::uint64_t output_elements,
    std::uint64_t output_offset, std::uint64_t slab_rows,
    std::uint8_t* weight_e4m3, std::uint16_t* weight_scales_bf16,
    cudaStream_t stream) {
  if (family.k3_payload_pool == nullptr || family.k4_payload_pool == nullptr ||
      family.descriptors == nullptr || weight_e4m3 == nullptr ||
      weight_scales_bf16 == nullptr || expert >= kTrellis35ExpertCount ||
      input_elements == 0U || input_elements % 32U != 0U ||
      output_elements == 0U || output_elements % 16U != 0U ||
      slab_rows == 0U || slab_rows > 128U || slab_rows % 16U != 0U ||
      output_offset % 16U != 0U || output_offset > output_elements ||
      slab_rows > output_elements - output_offset) {
    return Invalid("Trellis35 transient E4M3 slab contract is invalid");
  }
  const std::uint64_t elements = slab_rows * input_elements;
  if (elements > static_cast<std::uint64_t>(
                     std::numeric_limits<unsigned>::max()) * kThreads) {
    return Invalid("Trellis35 transient E4M3 slab exceeds CUDA grid limits");
  }
  const unsigned blocks =
      static_cast<unsigned>((elements + kThreads - 1U) / kThreads);
  const std::uint16_t rate = family.rate_map[expert];
  if (rate == 3U) {
    DecodeTrellis35E4M3SlabDiagnosticKernel<3><<<blocks, kThreads, 0, stream>>>(
        family, expert, input_elements, output_elements, output_offset,
        slab_rows, weight_e4m3, weight_scales_bf16);
  } else if (rate == 4U) {
    DecodeTrellis35E4M3SlabDiagnosticKernel<4><<<blocks, kThreads, 0, stream>>>(
        family, expert, input_elements, output_elements, output_offset,
        slab_rows, weight_e4m3, weight_scales_bf16);
  } else {
    return Invalid("Trellis35 transient E4M3 slab rate is unsupported");
  }
  return CheckLaunch("launch Trellis35 transient E4M3 slab diagnostic");
}

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

Status LaunchTrellis35MmaW4A8ProjectionM1N128Inverse(
    const std::uint8_t* activation_e4m3, const float* activation_scales,
    const Trellis35DeviceFamilyBinding& family,
    const std::uint32_t* selected_experts, float* output,
    std::uint64_t input_elements, std::uint64_t output_elements,
    bool vector_store, cudaStream_t stream) {
  if (activation_e4m3 == nullptr || activation_scales == nullptr ||
      family.k3_payload_pool == nullptr || family.k4_payload_pool == nullptr ||
      family.descriptors == nullptr || family.svh_f16 == nullptr ||
      selected_experts == nullptr || output == nullptr) {
    return Invalid(
        "Trellis35 M1 N128 inverse projection requires non-null pointers");
  }
  if (input_elements == 0U || input_elements % 32U != 0U ||
      output_elements == 0U || output_elements % 128U != 0U ||
      (vector_store &&
       reinterpret_cast<std::uintptr_t>(output) % alignof(float4) != 0U) ||
      output_elements >
          static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max()) *
              128U) {
    return Invalid("Trellis35 M1 N128 inverse projection geometry is invalid");
  }
  if (Trellis35M1NativeFp8x4Enabled()) {
    if (vector_store) {
      MmaW4A8ProjectionSelectedN128InverseKernel<true, true><<<
          dim3(static_cast<unsigned>(output_elements / 128U),
               kTrellis35M1TopK),
          kMmaN128Threads, 0, stream>>>(
          activation_e4m3, activation_scales, family, selected_experts, output,
          input_elements, output_elements);
    } else {
      MmaW4A8ProjectionSelectedN128InverseKernel<true, false><<<
          dim3(static_cast<unsigned>(output_elements / 128U),
               kTrellis35M1TopK),
          kMmaN128Threads, 0, stream>>>(
          activation_e4m3, activation_scales, family, selected_experts, output,
          input_elements, output_elements);
    }
  } else {
    if (vector_store) {
      MmaW4A8ProjectionSelectedN128InverseKernel<false, true><<<
          dim3(static_cast<unsigned>(output_elements / 128U),
               kTrellis35M1TopK),
          kMmaN128Threads, 0, stream>>>(
          activation_e4m3, activation_scales, family, selected_experts, output,
          input_elements, output_elements);
    } else {
      MmaW4A8ProjectionSelectedN128InverseKernel<false, false><<<
          dim3(static_cast<unsigned>(output_elements / 128U),
               kTrellis35M1TopK),
          kMmaN128Threads, 0, stream>>>(
          activation_e4m3, activation_scales, family, selected_experts, output,
          input_elements, output_elements);
    }
  }
  return CheckLaunch("launch Trellis35 M1 N128 projection and inverse");
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
    cudaStream_t stream, Trellis35SmallTransformMode transform_mode,
    Trellis35SmallGeluDownMode gelu_down_mode,
    Trellis35M1ProjectionOutputMode projection_output_mode,
    Trellis35VectorStoreMode vector_store_mode) {
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
      workspace.down_output == nullptr ||
      (transform_mode != Trellis35SmallTransformMode::kDirectH128 &&
       transform_mode != Trellis35SmallTransformMode::kWarpInputH128 &&
       transform_mode != Trellis35SmallTransformMode::kWarpOutputH128 &&
       transform_mode != Trellis35SmallTransformMode::kWarpH128) ||
      (gelu_down_mode != Trellis35SmallGeluDownMode::kSeparate &&
       gelu_down_mode !=
           Trellis35SmallGeluDownMode::kFusedTransformQuantize) ||
      (projection_output_mode !=
           Trellis35M1ProjectionOutputMode::kEnvironment &&
       projection_output_mode !=
           Trellis35M1ProjectionOutputMode::kSeparateN32 &&
       projection_output_mode !=
           Trellis35M1ProjectionOutputMode::kFusedN128 &&
       projection_output_mode !=
           Trellis35M1ProjectionOutputMode::kGateUpFusedN128 &&
       projection_output_mode !=
           Trellis35M1ProjectionOutputMode::kDownFusedN128) ||
      (vector_store_mode != Trellis35VectorStoreMode::kEnvironment &&
       vector_store_mode != Trellis35VectorStoreMode::kDisabled &&
       vector_store_mode != Trellis35VectorStoreMode::kEnabled)) {
    return Invalid("Trellis35 selected-expert M1 requires complete bindings");
  }

  const bool vector_store =
      vector_store_mode == Trellis35VectorStoreMode::kEnabled ||
      (vector_store_mode == Trellis35VectorStoreMode::kEnvironment &&
       Trellis35M1VectorStoreEnvironmentEnabled());

  const bool warp_input =
      transform_mode == Trellis35SmallTransformMode::kWarpInputH128 ||
      transform_mode == Trellis35SmallTransformMode::kWarpH128;
  const bool warp_output =
      transform_mode == Trellis35SmallTransformMode::kWarpOutputH128 ||
      transform_mode == Trellis35SmallTransformMode::kWarpH128;
  const bool fused_gelu_down =
      gelu_down_mode ==
          Trellis35SmallGeluDownMode::kFusedTransformQuantize &&
      Trellis35SmallGeluDownFusionEnabled();
  if (projection_output_mode ==
      Trellis35M1ProjectionOutputMode::kEnvironment) {
    projection_output_mode = Trellis35M1ProjectionOutputEnvironmentMode();
  }
  const bool fused_gate_n128_inverse =
      warp_output &&
      (projection_output_mode == Trellis35M1ProjectionOutputMode::kFusedN128 ||
       projection_output_mode ==
           Trellis35M1ProjectionOutputMode::kGateUpFusedN128);
  const bool fused_down_n128_inverse =
      warp_output &&
      (projection_output_mode == Trellis35M1ProjectionOutputMode::kFusedN128 ||
       projection_output_mode ==
           Trellis35M1ProjectionOutputMode::kDownFusedN128);
  if (fused_gelu_down && !warp_input) {
    return Invalid("Trellis35 fused small GELU/Down requires warp H128 input");
  }
  if (warp_input) {
    SelectedInputTransformWarpKernel<<<
        dim3(kTrellis35GateUpInput / 128U, 1U), kThreads, 0, stream>>>(
        input, layer.gate_up.suh_f16, selected_experts,
        workspace.gate_up_input_transformed, kTrellis35M1TopK, 0U,
        kTrellis35M1TopK, kTrellis35GateUpInput, kTrellis35GateUpInput);
  } else {
    const dim3 gate_input_blocks(
        static_cast<unsigned>((kTrellis35GateUpInput + kThreads - 1U) /
                              kThreads),
        kTrellis35M1TopK);
    SelectedInputTransformKernel<<<gate_input_blocks, kThreads, 0, stream>>>(
        input, layer.gate_up.suh_f16, selected_experts,
        workspace.gate_up_input_transformed, 0U, kTrellis35GateUpInput,
        kTrellis35GateUpInput);
  }
  Status status = CheckLaunch("launch Trellis35 selected Gate+Up input transform");
  if (!status.ok()) return status;

  status = LaunchFp8ReferenceTokenQuantizationBatch(
      workspace.gate_up_input_transformed, workspace.gate_up_input_e4m3,
      workspace.gate_up_input_scales, kTrellis35M1TopK,
      kTrellis35GateUpInput, stream);
  if (!status.ok()) return status;
  status = fused_gate_n128_inverse
               ? LaunchTrellis35MmaW4A8ProjectionM1N128Inverse(
                     workspace.gate_up_input_e4m3,
                     workspace.gate_up_input_scales, layer.gate_up,
                     selected_experts, workspace.gate_up_output,
                     kTrellis35GateUpInput, kTrellis35GateUpOutput,
                     vector_store, stream)
               : LaunchTrellis35MmaW4A8ProjectionM1(
                     workspace.gate_up_input_e4m3,
                     workspace.gate_up_input_scales, layer.gate_up,
                     selected_experts, workspace.gate_up_transformed_output,
                     kTrellis35GateUpInput, kTrellis35GateUpOutput, stream);
  if (!status.ok()) return status;

  if (!fused_gate_n128_inverse) {
    if (warp_output) {
      SelectedOutputTransformWarpKernel<<<
          dim3(kTrellis35GateUpOutput / 128U, 1U), kThreads, 0, stream>>>(
          workspace.gate_up_transformed_output, layer.gate_up.svh_f16,
          selected_experts, workspace.gate_up_output, kTrellis35M1TopK,
          kTrellis35GateUpOutput);
    } else {
      const dim3 gate_output_blocks(
          static_cast<unsigned>((kTrellis35GateUpOutput + kThreads - 1U) /
                                kThreads),
          kTrellis35M1TopK);
      SelectedOutputTransformKernel<<<gate_output_blocks, kThreads, 0,
                                      stream>>>(
          workspace.gate_up_transformed_output, layer.gate_up.svh_f16,
          selected_experts, workspace.gate_up_output, kTrellis35GateUpOutput);
    }
  }
  status = CheckLaunch("launch Trellis35 selected Gate+Up output transform");
  if (!status.ok()) return status;

  if (fused_gelu_down) {
    GatedGeluDownTransformQuantizeWarpKernel<<<
        kTrellis35M1TopK, kThreads, 0, stream>>>(
        workspace.gate_up_output, layer.down.suh_f16, selected_experts,
        workspace.down_input_scales, workspace.down_input_e4m3,
        kTrellis35M1TopK);
    status = CheckLaunch("launch fused Trellis35 M1 GELU/Down E4M3 transform");
    if (!status.ok()) return status;
  } else {
    const dim3 product_blocks(
        static_cast<unsigned>((kTrellis35ExpertIntermediate + kThreads - 1U) /
                              kThreads),
        kTrellis35M1TopK);
    GatedGeluKernel<<<product_blocks, kThreads, 0, stream>>>(
        workspace.gate_up_output, workspace.product);
    status = CheckLaunch("launch Trellis35 selected gated GELU");
    if (!status.ok()) return status;

    if (warp_input) {
      SelectedInputTransformWarpKernel<<<
          dim3(kTrellis35DownInput / 128U, 1U), kThreads, 0, stream>>>(
          workspace.product, layer.down.suh_f16, selected_experts,
          workspace.down_input_transformed, kTrellis35M1TopK,
          kTrellis35ExpertIntermediate, 1U, kTrellis35ExpertIntermediate,
          kTrellis35DownInput);
    } else {
      const dim3 down_input_blocks(
          static_cast<unsigned>((kTrellis35DownInput + kThreads - 1U) /
                                kThreads),
          kTrellis35M1TopK);
      SelectedInputTransformKernel<<<down_input_blocks, kThreads, 0, stream>>>(
          workspace.product, layer.down.suh_f16, selected_experts,
          workspace.down_input_transformed, kTrellis35ExpertIntermediate,
          kTrellis35ExpertIntermediate, kTrellis35DownInput);
    }
    status = CheckLaunch("launch Trellis35 selected Down input transform");
    if (!status.ok()) return status;

    status = LaunchFp8ReferenceTokenQuantizationBatch(
        workspace.down_input_transformed, workspace.down_input_e4m3,
        workspace.down_input_scales, kTrellis35M1TopK, kTrellis35DownInput,
        stream);
    if (!status.ok()) return status;
  }
  status = fused_down_n128_inverse
               ? LaunchTrellis35MmaW4A8ProjectionM1N128Inverse(
                     workspace.down_input_e4m3, workspace.down_input_scales,
                     layer.down, selected_experts, workspace.down_output,
                     kTrellis35DownInput, kTrellis35DownOutput, vector_store,
                     stream)
               : LaunchTrellis35MmaW4A8ProjectionM1(
                     workspace.down_input_e4m3, workspace.down_input_scales,
                     layer.down, selected_experts,
                     workspace.down_transformed_output, kTrellis35DownInput,
                     kTrellis35DownOutput, stream);
  if (!status.ok()) return status;

  if (!fused_down_n128_inverse) {
    if (warp_output) {
      SelectedOutputTransformWarpKernel<<<
          dim3(kTrellis35DownOutput / 128U, 1U), kThreads, 0, stream>>>(
          workspace.down_transformed_output, layer.down.svh_f16,
          selected_experts, workspace.down_output, kTrellis35M1TopK,
          kTrellis35DownOutput);
    } else {
      const dim3 down_output_blocks(
          static_cast<unsigned>((kTrellis35DownOutput + kThreads - 1U) /
                                kThreads),
          kTrellis35M1TopK);
      SelectedOutputTransformKernel<<<down_output_blocks, kThreads, 0,
                                      stream>>>(
          workspace.down_transformed_output, layer.down.svh_f16,
          selected_experts, workspace.down_output, kTrellis35DownOutput);
    }
  }
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
    cudaStream_t stream, Trellis35SmallTransformMode transform_mode,
    Trellis35T3ProjectionMode projection_mode,
    Trellis35SmallGeluDownMode gelu_down_mode,
    Trellis35T3ProjectionOutputMode projection_output_mode,
    Trellis35VectorStoreMode vector_store_mode) {
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
      workspace.down_output == nullptr ||
      (transform_mode != Trellis35SmallTransformMode::kDirectH128 &&
       transform_mode != Trellis35SmallTransformMode::kWarpInputH128 &&
       transform_mode != Trellis35SmallTransformMode::kWarpOutputH128 &&
       transform_mode != Trellis35SmallTransformMode::kWarpH128) ||
      (projection_mode != Trellis35T3ProjectionMode::kIndependentRows &&
       projection_mode != Trellis35T3ProjectionMode::kM16) ||
      (gelu_down_mode != Trellis35SmallGeluDownMode::kSeparate &&
       gelu_down_mode !=
           Trellis35SmallGeluDownMode::kFusedTransformQuantize) ||
      (projection_output_mode !=
           Trellis35T3ProjectionOutputMode::kEnvironment &&
       projection_output_mode !=
           Trellis35T3ProjectionOutputMode::kSeparateN32 &&
       projection_output_mode !=
           Trellis35T3ProjectionOutputMode::kFusedN128 &&
       projection_output_mode !=
           Trellis35T3ProjectionOutputMode::kGateUpFusedN128 &&
       projection_output_mode !=
           Trellis35T3ProjectionOutputMode::kDownFusedN128) ||
      (vector_store_mode != Trellis35VectorStoreMode::kEnvironment &&
       vector_store_mode != Trellis35VectorStoreMode::kDisabled &&
       vector_store_mode != Trellis35VectorStoreMode::kEnabled)) {
    return Invalid("Trellis35 selected-expert T3 requires complete bindings");
  }

  const bool vector_store =
      vector_store_mode == Trellis35VectorStoreMode::kEnabled ||
      (vector_store_mode == Trellis35VectorStoreMode::kEnvironment &&
       Trellis35T3VectorStoreEnvironmentEnabled());

  const bool warp_input =
      transform_mode == Trellis35SmallTransformMode::kWarpInputH128 ||
      transform_mode == Trellis35SmallTransformMode::kWarpH128;
  const bool warp_output =
      transform_mode == Trellis35SmallTransformMode::kWarpOutputH128 ||
      transform_mode == Trellis35SmallTransformMode::kWarpH128;
  const bool environment_output_mode =
      projection_output_mode ==
      Trellis35T3ProjectionOutputMode::kEnvironment;
  if (environment_output_mode) {
    projection_output_mode = Trellis35T3ProjectionOutputEnvironmentMode();
  }
  if ((projection_mode != Trellis35T3ProjectionMode::kM16 || !warp_output) &&
      projection_output_mode !=
          Trellis35T3ProjectionOutputMode::kSeparateN32) {
    if (!environment_output_mode) {
      return Invalid(
          "Trellis35 T3 N128 inverse requires M16 and Warp-H128 output");
    }
    projection_output_mode =
        Trellis35T3ProjectionOutputMode::kSeparateN32;
  }
  const bool fuse_gate_output =
      projection_output_mode ==
          Trellis35T3ProjectionOutputMode::kFusedN128 ||
      projection_output_mode ==
          Trellis35T3ProjectionOutputMode::kGateUpFusedN128;
  const bool fuse_down_output =
      projection_output_mode ==
          Trellis35T3ProjectionOutputMode::kFusedN128 ||
      projection_output_mode ==
          Trellis35T3ProjectionOutputMode::kDownFusedN128;
  const bool fused_gelu_down =
      gelu_down_mode ==
          Trellis35SmallGeluDownMode::kFusedTransformQuantize &&
      Trellis35SmallGeluDownFusionEnabled();
  if (fused_gelu_down && !warp_input) {
    return Invalid("Trellis35 fused small GELU/Down requires warp H128 input");
  }
  constexpr unsigned kT3AssignmentBlocks =
      (kTrellis35T3Assignments + 7U) / 8U;
  if (warp_input) {
    SelectedInputTransformWarpKernel<<<
        dim3(kTrellis35GateUpInput / 128U, kT3AssignmentBlocks), kThreads, 0,
        stream>>>(input_rows, layer.gate_up.suh_f16, selected_experts,
                  workspace.gate_up_input_transformed,
                  kTrellis35T3Assignments, kTrellis35GateUpInput,
                  kTrellis35M1TopK, kTrellis35GateUpInput,
                  kTrellis35GateUpInput);
  } else {
    const dim3 gate_input_blocks(
        static_cast<unsigned>((kTrellis35GateUpInput + kThreads - 1U) /
                              kThreads),
        kTrellis35T3Assignments);
    T3InputTransformKernel<<<gate_input_blocks, kThreads, 0, stream>>>(
        input_rows, layer.gate_up.suh_f16, selected_experts,
        workspace.gate_up_input_transformed, kTrellis35GateUpInput,
        kTrellis35M1TopK, kTrellis35GateUpInput, kTrellis35GateUpInput);
  }
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
      fuse_gate_output ? workspace.gate_up_output
                       : workspace.gate_up_transformed_output,
      kTrellis35GateUpInput, kTrellis35GateUpOutput, projection_mode,
      fuse_gate_output, vector_store, stream);
  if (!status.ok()) return status;

  if (!fuse_gate_output) {
    if (warp_output) {
      SelectedOutputTransformWarpKernel<<<
          dim3(kTrellis35GateUpOutput / 128U, kT3AssignmentBlocks), kThreads, 0,
          stream>>>(workspace.gate_up_transformed_output,
                    layer.gate_up.svh_f16, selected_experts,
                    workspace.gate_up_output, kTrellis35T3Assignments,
                    kTrellis35GateUpOutput);
    } else {
      const dim3 gate_output_blocks(
          static_cast<unsigned>((kTrellis35GateUpOutput + kThreads - 1U) /
                                kThreads),
          kTrellis35T3Assignments);
      SelectedOutputTransformKernel<<<gate_output_blocks, kThreads, 0,
                                      stream>>>(
          workspace.gate_up_transformed_output, layer.gate_up.svh_f16,
          selected_experts, workspace.gate_up_output,
          kTrellis35GateUpOutput);
    }
    status = CheckLaunch("launch Trellis35 T3 Gate+Up output transform");
    if (!status.ok()) return status;
  }

  if (fused_gelu_down) {
    GatedGeluDownTransformQuantizeWarpKernel<<<
        kTrellis35T3Assignments, kThreads, 0, stream>>>(
        workspace.gate_up_output, layer.down.suh_f16, selected_experts,
        workspace.down_input_scales, workspace.down_input_e4m3,
        kTrellis35T3Assignments);
    status = CheckLaunch("launch fused Trellis35 T3 GELU/Down E4M3 transform");
    if (!status.ok()) return status;
  } else {
    const dim3 product_blocks(
        static_cast<unsigned>((kTrellis35ExpertIntermediate + kThreads - 1U) /
                              kThreads),
        kTrellis35T3Assignments);
    GatedGeluKernel<<<product_blocks, kThreads, 0, stream>>>(
        workspace.gate_up_output, workspace.product);
    status = CheckLaunch("launch Trellis35 T3 gated GELU");
    if (!status.ok()) return status;

    if (warp_input) {
      SelectedInputTransformWarpKernel<<<
          dim3(kTrellis35DownInput / 128U, kT3AssignmentBlocks), kThreads, 0,
          stream>>>(workspace.product, layer.down.suh_f16, selected_experts,
                    workspace.down_input_transformed,
                    kTrellis35T3Assignments, kTrellis35ExpertIntermediate, 1U,
                    kTrellis35ExpertIntermediate, kTrellis35DownInput);
    } else {
      const dim3 down_input_blocks(
          static_cast<unsigned>((kTrellis35DownInput + kThreads - 1U) /
                                kThreads),
          kTrellis35T3Assignments);
      T3InputTransformKernel<<<down_input_blocks, kThreads, 0, stream>>>(
          workspace.product, layer.down.suh_f16, selected_experts,
          workspace.down_input_transformed, kTrellis35ExpertIntermediate, 1U,
          kTrellis35ExpertIntermediate, kTrellis35DownInput);
    }
    status = CheckLaunch("launch Trellis35 T3 Down input transform");
    if (!status.ok()) return status;
    status = LaunchFp8ReferenceTokenQuantizationBatch(
        workspace.down_input_transformed, workspace.down_input_e4m3,
        workspace.down_input_scales, kTrellis35T3Assignments,
        kTrellis35DownInput, stream);
    if (!status.ok()) return status;
  }
  status = LaunchGroupedT3Projection(
      workspace.down_input_e4m3, workspace.down_input_scales, layer.down,
      selected_experts,
      fuse_down_output ? workspace.down_output
                       : workspace.down_transformed_output,
      kTrellis35DownInput, kTrellis35DownOutput, projection_mode,
      fuse_down_output, vector_store, stream);
  if (!status.ok()) return status;

  if (!fuse_down_output) {
    if (warp_output) {
      SelectedOutputTransformWarpKernel<<<
          dim3(kTrellis35DownOutput / 128U, kT3AssignmentBlocks), kThreads, 0,
          stream>>>(workspace.down_transformed_output, layer.down.svh_f16,
                    selected_experts, workspace.down_output,
                    kTrellis35T3Assignments, kTrellis35DownOutput);
    } else {
      const dim3 down_output_blocks(
          static_cast<unsigned>((kTrellis35DownOutput + kThreads - 1U) /
                                kThreads),
          kTrellis35T3Assignments);
      SelectedOutputTransformKernel<<<down_output_blocks, kThreads, 0,
                                      stream>>>(
          workspace.down_transformed_output, layer.down.svh_f16,
          selected_experts, workspace.down_output, kTrellis35DownOutput);
    }
    status = CheckLaunch("launch Trellis35 T3 Down output transform");
    if (!status.ok()) return status;
  }

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
    const Gemma4MoePrefillWorkspace& workspace,
    Trellis35PrefillScheduleMode schedule_mode,
    Trellis35PrefillKernelMode kernel_mode,
    Trellis35PrefillTransformMode transform_mode,
    Trellis35PrefillOutputMode output_mode, cudaStream_t stream) {
  const std::uint64_t assignment_count = tokens * kTrellis35M1TopK;
  const bool float_boundaries = workspace.expert_product != nullptr &&
                                workspace.expert_down != nullptr &&
                                workspace.expert_product_bf16 == nullptr &&
                                workspace.expert_down_bf16 == nullptr;
  const bool physical_bf16_boundaries = workspace.expert_product == nullptr &&
                                        workspace.expert_down == nullptr &&
                                        workspace.expert_product_bf16 != nullptr &&
                                        workspace.expert_down_bf16 != nullptr;
  if (normalized_hidden == nullptr || tokens == 0U || tokens > 2048U ||
      assignment_count > 65535U ||
      (!float_boundaries && !physical_bf16_boundaries) ||
      (physical_bf16_boundaries && workspace.trellis_activation == nullptr) ||
      workspace.token_scales == nullptr ||
      workspace.shared_product == nullptr || workspace.shared_output == nullptr ||
      workspace.token_hidden == nullptr || workspace.assignments == nullptr ||
      workspace.prefix == nullptr || workspace.permutation == nullptr ||
      workspace.router_logits == nullptr || workspace.histogram == nullptr ||
      (output_mode != Trellis35PrefillOutputMode::kLoopN128 &&
       output_mode != Trellis35PrefillOutputMode::kFusedN128)) {
    return Invalid("Trellis35 prefill expert workspace is incomplete");
  }
  if (schedule_mode == Trellis35PrefillScheduleMode::kConsumeM32 &&
      kernel_mode != Trellis35PrefillKernelMode::kGroupedM32) {
    return Invalid("Trellis35 prebuilt prefill schedule requires M32 kernel");
  }
  if ((schedule_mode == Trellis35PrefillScheduleMode::kBuildM64Hybrid) !=
      (kernel_mode == Trellis35PrefillKernelMode::kGroupedM64Hybrid)) {
    return Invalid(
        "Trellis35 M64 hybrid kernel requires its private hybrid schedule");
  }
  auto* activation_scales =
      reinterpret_cast<float*>(workspace.token_scales);
  auto* projection_tile = workspace.shared_product;
  auto* schedule = reinterpret_cast<std::uint32_t*>(workspace.router_logits);

  Status status = Status::Ok();
  if (schedule_mode == Trellis35PrefillScheduleMode::kBuildStandalone) {
    if (kernel_mode == Trellis35PrefillKernelMode::kGroupedM32) {
      BuildTrellis35PrefillTileScheduleKernel<
          kPrefillGroupedRowsPerTile><<<1, 1, 0, stream>>>(
          workspace.prefix, workspace.histogram, schedule);
    } else {
      BuildTrellis35PrefillTileScheduleKernel<
          kPrefillLegacyRowsPerTile><<<1, 1, 0, stream>>>(
          workspace.prefix, workspace.histogram, schedule);
    }
    status = CheckLaunch("build standalone Trellis35 prefill schedule");
    if (!status.ok()) return status;
  } else if (schedule_mode ==
             Trellis35PrefillScheduleMode::kBuildM64Hybrid) {
    BuildTrellis35PrefillM64HybridScheduleKernel<<<1, kThreads, 0, stream>>>(
        workspace.prefix, workspace.histogram, schedule);
    status = CheckLaunch("build Trellis35 M64/M32 hybrid prefill schedule");
    if (!status.ok()) return status;
  }

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
        kTrellis35GateUpInput, true, transform_mode, stream);
    if (!status.ok()) return status;
    status = LaunchPrefillProjectionBlocksBf16Reverse(
        aliased_activation, activation_scales, layer.gate_up,
        workspace.assignments, workspace.prefix, workspace.histogram,
        schedule, workspace.permutation, projection_tile,
        workspace.expert_down_bf16, tokens, assignment_count,
        kTrellis35GateUpInput, kTrellis35GateUpOutput, kernel_mode,
        transform_mode, output_mode, stream);
    if (!status.ok()) return status;
    status = LaunchTrellis35GatedGeluDownTransformQuantizeBf16(
        workspace.expert_down_bf16, workspace.expert_product_bf16,
        layer.down, workspace.assignments, aliased_activation,
        activation_scales, assignment_count,
        Trellis35PrefillGeluDownFusionEnabled()
            ? Trellis35PrefillGeluDownMode::kFusedTransformQuantize
            : Trellis35PrefillGeluDownMode::kTwoKernel,
        stream);
    if (!status.ok()) return status;
    status = LaunchPrefillProjectionBlocksBf16Reverse(
        aliased_activation, activation_scales, layer.down,
        workspace.assignments, workspace.prefix, workspace.histogram,
        schedule, workspace.permutation, projection_tile,
        workspace.expert_down_bf16, tokens, assignment_count,
        kTrellis35DownInput, kTrellis35DownOutput, kernel_mode,
        transform_mode, output_mode, stream);
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
        kTrellis35GateUpInput, true, transform_mode, stream);
    if (!status.ok()) return status;
    status = LaunchPrefillProjectionBlocks(
        gate_activation, activation_scales, layer.gate_up,
        workspace.assignments, workspace.prefix, workspace.histogram,
        schedule, workspace.permutation, projection_tile,
        workspace.expert_down, tokens, assignment_count,
        kTrellis35GateUpInput, kTrellis35GateUpOutput, kernel_mode,
        transform_mode, output_mode, stream);
    if (!status.ok()) return status;
    GatedGeluKernel<<<product_blocks, kThreads, 0, stream>>>(
        workspace.expert_down, workspace.expert_product);
    status = CheckLaunch("launch Trellis35 prefill gated GELU");
    if (!status.ok()) return status;
    status = LaunchPrefillTransformQuantize(
        workspace.expert_product, layer.down, workspace.assignments,
        down_activation, activation_scales, assignment_count, tokens,
        kTrellis35ExpertIntermediate, kTrellis35ExpertIntermediate,
        kTrellis35DownInput, false, transform_mode, stream);
    if (!status.ok()) return status;
    status = LaunchPrefillProjectionBlocks(
        down_activation, activation_scales, layer.down, workspace.assignments,
        workspace.prefix, workspace.histogram, schedule,
        workspace.permutation, projection_tile, workspace.expert_down, tokens,
        assignment_count, kTrellis35DownInput, kTrellis35DownOutput,
        kernel_mode, transform_mode, output_mode, stream);
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

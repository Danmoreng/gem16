__device__ __forceinline__ float PrefillTransformedValue(
    const float* input, const std::uint16_t* suh, std::uint64_t index,
    std::uint64_t logical_elements) {
  const std::uint64_t block = index & ~std::uint64_t{127U};
  const unsigned column = static_cast<unsigned>(index & 127U);
  float accumulator = 0.0F;
#pragma unroll
  for (unsigned row = 0U; row < 128U; ++row) {
    const std::uint64_t source = block + row;
    const float value = source < logical_elements ? input[source] : 0.0F;
    const float sign = (__popc(row & column) & 1U) == 0U ? 1.0F : -1.0F;
    accumulator = fmaf(value * F16(suh + source), sign, accumulator);
  }
  return accumulator * kHadamardScale;
}

__device__ __forceinline__ const float* PrefillAssignmentInput(
    const float* input, const Gemma4MoePrefillAssignment& assignment,
    unsigned assignment_index, std::uint64_t input_stride,
    bool token_major_input, std::uint64_t tokens) {
  if (assignment.expert_id >= kTrellis35ExpertCount ||
      (token_major_input && assignment.token_id >= tokens)) {
    return nullptr;
  }
  const std::uint64_t row =
      token_major_input ? assignment.token_id : assignment_index;
  return input + row * input_stride;
}

__global__ void PrefillTransformScaleKernel(
    const float* input, const std::uint16_t* all_suh,
    const Gemma4MoePrefillAssignment* assignments, float* scales,
    std::uint64_t assignment_count, std::uint64_t tokens,
    std::uint64_t input_stride, std::uint64_t logical_elements,
    std::uint64_t physical_elements, bool token_major_input) {
  const unsigned assignment_index = blockIdx.x;
  if (assignment_index >= assignment_count) return;
  const auto assignment = assignments[assignment_index];
  const float* assignment_input = PrefillAssignmentInput(
      input, assignment, assignment_index, input_stride, token_major_input,
      tokens);
  __shared__ float maxima[kThreads];
  float local_maximum = 0.0F;
  if (assignment_input != nullptr) {
    const std::uint16_t* suh =
        all_suh + static_cast<std::uint64_t>(assignment.expert_id) *
                      physical_elements;
    for (std::uint64_t index = threadIdx.x; index < physical_elements;
         index += blockDim.x) {
      local_maximum =
          fmaxf(local_maximum,
                fabsf(PrefillTransformedValue(assignment_input, suh, index,
                                              logical_elements)));
    }
  }
  maxima[threadIdx.x] = local_maximum;
  __syncthreads();
  for (unsigned stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      maxima[threadIdx.x] =
          fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) {
    scales[assignment_index] =
        maxima[0] == 0.0F ? 1.0F : maxima[0] / kE4M3Maximum;
  }
}

__global__ void PrefillTransformQuantizeKernel(
    const float* input, const std::uint16_t* all_suh,
    const Gemma4MoePrefillAssignment* assignments, const float* scales,
    std::uint8_t* output, std::uint64_t assignment_count,
    std::uint64_t tokens, std::uint64_t input_stride,
    std::uint64_t logical_elements, std::uint64_t physical_elements,
    bool token_major_input) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned assignment_index = blockIdx.y;
  if (index >= physical_elements || assignment_index >= assignment_count) {
    return;
  }
  const auto assignment = assignments[assignment_index];
  const float* assignment_input = PrefillAssignmentInput(
      input, assignment, assignment_index, input_stride, token_major_input,
      tokens);
  if (assignment_input == nullptr) return;
  const std::uint16_t* suh =
      all_suh + static_cast<std::uint64_t>(assignment.expert_id) *
                    physical_elements;
  const float transformed = PrefillTransformedValue(
      assignment_input, suh, index, logical_elements);
  const __nv_fp8_e4m3 encoded(transformed / scales[assignment_index]);
  output[static_cast<std::uint64_t>(assignment_index) * physical_elements +
         index] = encoded.__x;
}

__device__ __forceinline__ float PrefillTransformedBf16Value(
    const std::uint16_t* input, const std::uint16_t* suh,
    std::uint64_t index, std::uint64_t logical_elements) {
  const std::uint64_t block = index & ~std::uint64_t{127U};
  const unsigned column = static_cast<unsigned>(index & 127U);
  float accumulator = 0.0F;
#pragma unroll
  for (unsigned row = 0U; row < 128U; ++row) {
    const std::uint64_t source = block + row;
    const float value = source < logical_elements ? Bf16(input[source]) : 0.0F;
    const float sign = (__popc(row & column) & 1U) == 0U ? 1.0F : -1.0F;
    accumulator = fmaf(value * F16(suh + source), sign, accumulator);
  }
  return accumulator * kHadamardScale;
}

__global__ void PrefillTransformBf16ScaleKernel(
    const std::uint16_t* input, const std::uint16_t* all_suh,
    const Gemma4MoePrefillAssignment* assignments, float* scales,
    std::uint64_t assignment_count, std::uint64_t input_stride,
    std::uint64_t logical_elements, std::uint64_t physical_elements) {
  const unsigned assignment_index = blockIdx.x;
  if (assignment_index >= assignment_count) return;
  const auto assignment = assignments[assignment_index];
  __shared__ float maxima[kThreads];
  float local_maximum = 0.0F;
  if (assignment.expert_id < kTrellis35ExpertCount) {
    const std::uint16_t* assignment_input =
        input + static_cast<std::uint64_t>(assignment_index) * input_stride;
    const std::uint16_t* suh =
        all_suh + static_cast<std::uint64_t>(assignment.expert_id) *
                      physical_elements;
    for (std::uint64_t index = threadIdx.x; index < physical_elements;
         index += blockDim.x) {
      local_maximum = fmaxf(
          local_maximum,
          fabsf(PrefillTransformedBf16Value(
              assignment_input, suh, index, logical_elements)));
    }
  }
  maxima[threadIdx.x] = local_maximum;
  __syncthreads();
  for (unsigned stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      maxima[threadIdx.x] =
          fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) {
    scales[assignment_index] =
        maxima[0] == 0.0F ? 1.0F : maxima[0] / kE4M3Maximum;
  }
}

__global__ void PrefillTransformQuantizeBf16Kernel(
    const std::uint16_t* input, const std::uint16_t* all_suh,
    const Gemma4MoePrefillAssignment* assignments, const float* scales,
    std::uint8_t* output, std::uint64_t assignment_count,
    std::uint64_t input_stride, std::uint64_t logical_elements,
    std::uint64_t physical_elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned assignment_index = blockIdx.y;
  if (index >= physical_elements || assignment_index >= assignment_count) {
    return;
  }
  const auto assignment = assignments[assignment_index];
  if (assignment.expert_id >= kTrellis35ExpertCount) return;
  const std::uint16_t* assignment_input =
      input + static_cast<std::uint64_t>(assignment_index) * input_stride;
  const std::uint16_t* suh =
      all_suh + static_cast<std::uint64_t>(assignment.expert_id) *
                    physical_elements;
  const float transformed = PrefillTransformedBf16Value(
      assignment_input, suh, index, logical_elements);
  const __nv_fp8_e4m3 encoded(transformed / scales[assignment_index]);
  output[static_cast<std::uint64_t>(assignment_index) * physical_elements +
         index] = encoded.__x;
}

__global__ void MmaW4A8ProjectionGroupedPrefillTileKernel(
    const std::uint8_t* activation, const float* activation_scales,
    Trellis35DeviceFamilyBinding family,
    const std::uint32_t* expert_prefix, const std::uint32_t* schedule_count,
    const std::uint32_t* schedule, const std::uint32_t* permutation,
    float* output_tile, std::uint64_t input_elements,
    std::uint64_t output_elements,
    std::uint64_t output_offset) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  const unsigned schedule_index = blockIdx.y;
  if (schedule_index >= schedule_count[0]) return;
  const std::uint32_t packed_tile = schedule[schedule_index];
  const unsigned expert = packed_tile >> 16U;
  const std::uint32_t grouped_begin = packed_tile & 0xffffU;
  const std::uint32_t grouped_end = expert_prefix[expert + 1U];
  if (grouped_begin >= grouped_end) return;
  const unsigned assignment_count = static_cast<unsigned>(
      min(grouped_end - grouped_begin, kPrefillRowsPerTile));
  unsigned assignments[kPrefillRowsPerTile]{};
#pragma unroll
  for (unsigned index = 0U; index < kPrefillRowsPerTile; ++index) {
    if (index < assignment_count) {
      assignments[index] = permutation[grouped_begin + index];
    }
  }

  const unsigned warp = threadIdx.x >> 5U;
  const unsigned lane = threadIdx.x & 31U;
  const std::uint64_t tile_output =
      (static_cast<std::uint64_t>(blockIdx.x) * kMmaWarps + warp) * 8U +
      (lane >> 2U);
  const std::uint64_t source_output = output_offset + tile_output;
  if (tile_output >= kPrefillOutputBlock ||
      source_output >= output_elements) {
    return;
  }
  const Trellis35ExpertDescriptor descriptor = family.descriptors[expert];
  const std::byte* pool = descriptor.rate_bits == 3U
                              ? family.k3_payload_pool
                              : family.k4_payload_pool;
  Fp8Accumulator accumulators[kPrefillRowsPerTile]{};
  if (descriptor.rate_bits == 3U) {
    AccumulateGroupedProjection<3, kPrefillRowsPerTile>(
        activation, assignments, assignment_count, pool,
        descriptor.pool_offset, input_elements, output_elements, source_output,
        accumulators);
  } else {
    AccumulateGroupedProjection<4, kPrefillRowsPerTile>(
        activation, assignments, assignment_count, pool,
        descriptor.pool_offset, input_elements, output_elements, source_output,
        accumulators);
  }
  if (lane < 4U) {
    const std::uint64_t column =
        (static_cast<std::uint64_t>(blockIdx.x) * kMmaWarps + warp) * 8U +
        lane * 2U;
#pragma unroll
    for (unsigned index = 0U; index < kPrefillRowsPerTile; ++index) {
      if (index < assignment_count) {
        const unsigned assignment = assignments[index];
        float* assignment_output =
            output_tile + static_cast<std::uint64_t>(assignment) *
                              kPrefillOutputBlock;
        const float scale = activation_scales[assignment];
        if (column < kPrefillOutputBlock) {
          assignment_output[column] = accumulators[index].x0 * scale;
        }
        if (column + 1U < kPrefillOutputBlock) {
          assignment_output[column + 1U] =
              accumulators[index].x1 * scale;
        }
      }
    }
  }
#else
  (void)activation;
  (void)activation_scales;
  (void)family;
  (void)expert_prefix;
  (void)schedule_count;
  (void)schedule;
  (void)permutation;
  (void)output_tile;
  (void)input_elements;
  (void)output_elements;
  (void)output_offset;
#endif
}

__global__ void BuildTrellis35PrefillTileScheduleKernel(
    const std::uint32_t* prefix, std::uint32_t* tile_count,
    std::uint32_t* schedule) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) return;
  std::uint32_t count = 0U;
  for (std::uint32_t expert = 0U; expert < kTrellis35ExpertCount; ++expert) {
    for (std::uint32_t grouped = prefix[expert];
         grouped < prefix[expert + 1U]; grouped += kPrefillRowsPerTile) {
      schedule[count++] = (expert << 16U) | grouped;
    }
  }
  tile_count[0] = count;
}

__global__ void RestoreTrellis35PrefillHistogramZeroKernel(
    std::uint32_t* histogram, const std::uint32_t* prefix) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    histogram[0] = prefix[1U] - prefix[0U];
  }
}

__global__ void PrefillOutputTransformTileKernel(
    const float* input_tile, const std::uint16_t* all_svh,
    const Gemma4MoePrefillAssignment* assignments, float* output,
    std::uint64_t assignment_count, std::uint64_t output_elements,
    std::uint64_t output_offset) {
  const unsigned assignment_index = blockIdx.x;
  const unsigned column = threadIdx.x;
  if (assignment_index >= assignment_count ||
      column >= kPrefillOutputBlock) {
    return;
  }
  const std::uint32_t expert = assignments[assignment_index].expert_id;
  if (expert >= kTrellis35ExpertCount) return;
  input_tile += static_cast<std::uint64_t>(assignment_index) *
                kPrefillOutputBlock;
  const std::uint16_t* svh =
      all_svh + static_cast<std::uint64_t>(expert) * output_elements +
      output_offset;
  float accumulator = 0.0F;
#pragma unroll
  for (unsigned row = 0U; row < kPrefillOutputBlock; ++row) {
    const float sign = (__popc(row & column) & 1U) == 0U ? 1.0F : -1.0F;
    accumulator = fmaf(input_tile[row], sign, accumulator);
  }
  output[static_cast<std::uint64_t>(assignment_index) * output_elements +
         output_offset + column] =
      accumulator * kHadamardScale * F16(svh + column);
}

__global__ void PrefillOutputTransformTileBf16Kernel(
    const float* input_tile, const std::uint16_t* all_svh,
    const Gemma4MoePrefillAssignment* assignments, std::uint16_t* output,
    std::uint64_t assignment_count, std::uint64_t output_elements,
    std::uint64_t output_offset) {
  const unsigned assignment_index = blockIdx.x;
  const unsigned column = threadIdx.x;
  if (assignment_index >= assignment_count ||
      column >= kPrefillOutputBlock) {
    return;
  }
  const std::uint32_t expert = assignments[assignment_index].expert_id;
  if (expert >= kTrellis35ExpertCount) return;
  input_tile += static_cast<std::uint64_t>(assignment_index) *
                kPrefillOutputBlock;
  const std::uint16_t* svh =
      all_svh + static_cast<std::uint64_t>(expert) * output_elements +
      output_offset;
  float accumulator = 0.0F;
#pragma unroll
  for (unsigned row = 0U; row < kPrefillOutputBlock; ++row) {
    const float sign = (__popc(row & column) & 1U) == 0U ? 1.0F : -1.0F;
    accumulator = fmaf(input_tile[row], sign, accumulator);
  }
  output[static_cast<std::uint64_t>(assignment_index) * output_elements +
         output_offset + column] =
      Bf16Bits(accumulator * kHadamardScale * F16(svh + column));
}

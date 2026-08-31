__global__ void MmaW4A8ProjectionGroupedT3Kernel(
    const std::uint8_t* activation, const float* activation_scales,
    Trellis35DeviceFamilyBinding family,
    const std::uint32_t* selected_experts, float* output,
    std::uint64_t input_elements, std::uint64_t output_elements) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  const unsigned group_candidate = blockIdx.y;
  const std::uint32_t expert = selected_experts[group_candidate];
  if (expert >= kTrellis35ExpertCount) return;
  for (unsigned prior = 0U; prior < group_candidate; ++prior) {
    if (selected_experts[prior] == expert) return;
  }

  unsigned assignments[kTrellis35T3Rows]{};
  unsigned assignment_count = 0U;
#pragma unroll
  for (unsigned row = 0U; row < kTrellis35T3Rows; ++row) {
#pragma unroll
    for (unsigned slot = 0U; slot < kTrellis35M1TopK; ++slot) {
      const unsigned assignment = row * kTrellis35M1TopK + slot;
      if (selected_experts[assignment] == expert) {
        assignments[assignment_count++] = assignment;
        break;
      }
    }
  }
  if (assignment_count == 0U) return;

  const unsigned warp = threadIdx.x >> 5U;
  const unsigned lane = threadIdx.x & 31U;
  const std::uint64_t output_base =
      (static_cast<std::uint64_t>(blockIdx.x) * kMmaWarps + warp) * 8U;
  const std::uint64_t source_output = output_base + (lane >> 2U);
  if (source_output >= output_elements) return;
  const Trellis35ExpertDescriptor descriptor = family.descriptors[expert];
  const std::byte* pool = descriptor.rate_bits == 3U
                              ? family.k3_payload_pool
                              : family.k4_payload_pool;
  Fp8Accumulator accumulators[kTrellis35T3Rows]{};
  if (descriptor.rate_bits == 3U) {
    AccumulateGroupedProjection<3, kTrellis35T3Rows>(
        activation, assignments, assignment_count, pool,
        descriptor.pool_offset, input_elements, output_elements, source_output,
        accumulators);
  } else {
    AccumulateGroupedProjection<4, kTrellis35T3Rows>(
        activation, assignments, assignment_count, pool,
        descriptor.pool_offset, input_elements, output_elements, source_output,
        accumulators);
  }

  if (lane < 4U) {
    const std::uint64_t column = output_base + lane * 2U;
#pragma unroll
    for (unsigned index = 0U; index < kTrellis35T3Rows; ++index) {
      if (index < assignment_count) {
        const unsigned assignment = assignments[index];
        float* assignment_output =
            output + static_cast<std::uint64_t>(assignment) * output_elements;
        const float scale = activation_scales[assignment];
        if (column < output_elements) {
          assignment_output[column] = accumulators[index].x0 * scale;
        }
        if (column + 1U < output_elements) {
          assignment_output[column + 1U] = accumulators[index].x1 * scale;
        }
      }
    }
  }
#else
  (void)activation;
  (void)activation_scales;
  (void)family;
  (void)selected_experts;
  (void)output;
  (void)input_elements;
  (void)output_elements;
#endif
}

template <int Rate, bool NativeFp8x4>
__device__ __forceinline__ void AccumulateGroupedT3M16Direct(
    const std::uint8_t* activation, const unsigned* assignments,
    unsigned assignment_count, const std::byte* pool,
    std::uint32_t pool_offset, std::uint64_t input_elements,
    std::uint64_t output_elements, std::uint64_t source_output,
    Fp8Accumulator& accumulator) {
  constexpr unsigned kPayloadWords = Rate * 256U / 32U;
  const unsigned lane = threadIdx.x & 31U;
  const unsigned group = lane >> 2U;
  const unsigned thread_in_group = lane & 3U;
  const std::uint64_t tile_columns = output_elements / 16U;
  const std::uint64_t output_tile = source_output / 16U;
  const auto* payload = reinterpret_cast<const std::uint32_t*>(
      pool + pool_offset);
  const unsigned iterations = static_cast<unsigned>(input_elements / 32U);
  const unsigned position = TensorCoreIndex(
      thread_in_group * 4U, static_cast<unsigned>(source_output & 15U));

  auto load_word = [&](unsigned input_tile) {
    if (lane >= kPayloadWords) return std::uint32_t{0U};
    const std::uint64_t tile =
        static_cast<std::uint64_t>(input_tile) * tile_columns + output_tile;
    return payload[tile * kPayloadWords + lane];
  };

  for (unsigned iteration = 0U; iteration < iterations; ++iteration) {
    const std::uint32_t lane_word0 = load_word(iteration * 2U);
    const std::uint32_t lane_word1 = load_word(iteration * 2U + 1U);
    const std::uint32_t b0 =
        DecodeLanePayloadSelectedE4M3x4<Rate, NativeFp8x4>(lane_word0,
                                                           position);
    const std::uint32_t b1 =
        DecodeLanePayloadSelectedE4M3x4<Rate, NativeFp8x4>(lane_word1,
                                                           position);
    const std::uint64_t first =
        static_cast<std::uint64_t>(iteration) * 32U + thread_in_group * 4U;
    std::uint32_t a0 = 0U;
    std::uint32_t a1 = 0U;
    std::uint32_t a2 = 0U;
    std::uint32_t a3 = 0U;
    if (group < assignment_count) {
      const std::uint8_t* row =
          activation +
          static_cast<std::uint64_t>(assignments[group]) * input_elements;
      a0 = *reinterpret_cast<const std::uint32_t*>(row + first);
      a2 = *reinterpret_cast<const std::uint32_t*>(row + first + 16U);
    }
    AccumulateFp8M16(a0, a1, a2, a3, b0, b1, accumulator);
  }
}

template <bool NativeFp8x4>
__global__ void MmaW4A8ProjectionGroupedT3M16Kernel(
    const std::uint8_t* activation, const float* activation_scales,
    Trellis35DeviceFamilyBinding family,
    const std::uint32_t* selected_experts, float* output,
    std::uint64_t input_elements, std::uint64_t output_elements) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  const unsigned group_candidate = blockIdx.y;
  const std::uint32_t expert = selected_experts[group_candidate];
  if (expert >= kTrellis35ExpertCount) return;
  for (unsigned prior = 0U; prior < group_candidate; ++prior) {
    if (selected_experts[prior] == expert) return;
  }

  unsigned assignments[kTrellis35T3Rows]{};
  unsigned assignment_count = 0U;
#pragma unroll
  for (unsigned row = 0U; row < kTrellis35T3Rows; ++row) {
#pragma unroll
    for (unsigned slot = 0U; slot < kTrellis35M1TopK; ++slot) {
      const unsigned assignment = row * kTrellis35M1TopK + slot;
      if (selected_experts[assignment] == expert) {
        assignments[assignment_count++] = assignment;
        break;
      }
    }
  }
  if (assignment_count == 0U) return;

  const unsigned warp = threadIdx.x >> 5U;
  const unsigned lane = threadIdx.x & 31U;
  const unsigned group = lane >> 2U;
  const unsigned thread_in_group = lane & 3U;
  const std::uint64_t output_column_base =
      (static_cast<std::uint64_t>(blockIdx.x) * kMmaWarps + warp) * 8U;
  const std::uint64_t source_output = output_column_base + group;
  if (source_output >= output_elements) return;
  const Trellis35ExpertDescriptor descriptor = family.descriptors[expert];
  const std::byte* pool = descriptor.rate_bits == 3U
                              ? family.k3_payload_pool
                              : family.k4_payload_pool;
  Fp8Accumulator accumulator{};
  if (descriptor.rate_bits == 3U) {
    AccumulateGroupedT3M16Direct<3, NativeFp8x4>(
        activation, assignments, assignment_count, pool,
        descriptor.pool_offset, input_elements, output_elements,
        source_output, accumulator);
  } else if (descriptor.rate_bits == 4U) {
    AccumulateGroupedT3M16Direct<4, NativeFp8x4>(
        activation, assignments, assignment_count, pool,
        descriptor.pool_offset, input_elements, output_elements,
        source_output, accumulator);
  } else {
    return;
  }

  const float values[4] = {accumulator.x0, accumulator.x1, accumulator.x2,
                           accumulator.x3};
#pragma unroll
  for (unsigned pair = 0U; pair < 4U; ++pair) {
    const unsigned row = (pair & 2U) == 0U ? group : group + 8U;
    const std::uint32_t assignment =
        row < assignment_count ? assignments[row] : 0xffffffffU;
    const std::uint64_t column =
        output_column_base + thread_in_group * 2U + (pair & 1U);
    if (assignment == 0xffffffffU || column >= output_elements) continue;
    output[static_cast<std::uint64_t>(assignment) * output_elements + column] =
        values[pair] * activation_scales[assignment];
  }
#else
  (void)activation;
  (void)activation_scales;
  (void)family;
  (void)selected_experts;
  (void)output;
  (void)input_elements;
  (void)output_elements;
#endif
}

template <bool NativeFp8x4>
__global__ void MmaW4A8ProjectionGroupedT3M16N128InverseKernel(
    const std::uint8_t* activation, const float* activation_scales,
    Trellis35DeviceFamilyBinding family,
    const std::uint32_t* selected_experts, float* output,
    std::uint64_t input_elements, std::uint64_t output_elements) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  const unsigned group_candidate = blockIdx.y;
  const std::uint32_t expert = selected_experts[group_candidate];
  if (expert >= kTrellis35ExpertCount) return;
  for (unsigned prior = 0U; prior < group_candidate; ++prior) {
    if (selected_experts[prior] == expert) return;
  }

  unsigned assignments[kTrellis35T3Rows]{};
  unsigned assignment_count = 0U;
#pragma unroll
  for (unsigned row = 0U; row < kTrellis35T3Rows; ++row) {
#pragma unroll
    for (unsigned slot = 0U; slot < kTrellis35M1TopK; ++slot) {
      const unsigned assignment = row * kTrellis35M1TopK + slot;
      if (selected_experts[assignment] == expert) {
        assignments[assignment_count++] = assignment;
        break;
      }
    }
  }
  if (assignment_count == 0U) return;

  const unsigned warp = threadIdx.x >> 5U;
  const unsigned lane = threadIdx.x & 31U;
  const unsigned group = lane >> 2U;
  const unsigned thread_in_group = lane & 3U;
  const std::uint64_t output_block =
      static_cast<std::uint64_t>(blockIdx.x) * 128U;
  const std::uint64_t output_column_base = output_block + warp * 8U;
  const std::uint64_t source_output = output_column_base + group;
  const Trellis35ExpertDescriptor descriptor = family.descriptors[expert];
  const std::byte* pool = descriptor.rate_bits == 3U
                              ? family.k3_payload_pool
                              : family.k4_payload_pool;
  Fp8Accumulator accumulator{};
  if (descriptor.rate_bits == 3U) {
    AccumulateGroupedT3M16Direct<3, NativeFp8x4>(
        activation, assignments, assignment_count, pool,
        descriptor.pool_offset, input_elements, output_elements,
        source_output, accumulator);
  } else if (descriptor.rate_bits == 4U) {
    AccumulateGroupedT3M16Direct<4, NativeFp8x4>(
        activation, assignments, assignment_count, pool,
        descriptor.pool_offset, input_elements, output_elements,
        source_output, accumulator);
  } else {
    return;
  }

  __shared__ float transformed[kTrellis35T3Rows][128U];
  const float values[4] = {accumulator.x0, accumulator.x1, accumulator.x2,
                           accumulator.x3};
#pragma unroll
  for (unsigned pair = 0U; pair < 4U; ++pair) {
    const unsigned row = (pair & 2U) == 0U ? group : group + 8U;
    const unsigned column = warp * 8U + thread_in_group * 2U + (pair & 1U);
    if (row < assignment_count) {
      transformed[row][column] =
          values[pair] * activation_scales[assignments[row]];
    }
  }
  __syncthreads();

  if (warp < assignment_count) {
    const unsigned index = lane * 4U;
    float inverse[4] = {transformed[warp][index],
                        transformed[warp][index + 1U],
                        transformed[warp][index + 2U],
                        transformed[warp][index + 3U]};
    H128Warp(inverse);
    const std::uint32_t assignment = assignments[warp];
    const std::uint16_t* svh =
        family.svh_f16 + static_cast<std::uint64_t>(expert) * output_elements +
        output_block + index;
    float* assignment_output =
        output + static_cast<std::uint64_t>(assignment) * output_elements +
        output_block + index;
#pragma unroll
    for (unsigned element = 0U; element < 4U; ++element) {
      assignment_output[element] = inverse[element] * F16(svh + element);
    }
  }
#else
  (void)activation;
  (void)activation_scales;
  (void)family;
  (void)selected_experts;
  (void)output;
  (void)input_elements;
  (void)output_elements;
#endif
}

__global__ void T3InputTransformKernel(
    const float* input, const std::uint16_t* all_suh,
    const std::uint32_t* selected_experts, float* output,
    std::uint64_t input_stride, unsigned assignments_per_input,
    std::uint64_t logical_elements, std::uint64_t physical_elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= physical_elements) return;
  const unsigned assignment = blockIdx.y;
  const std::uint32_t expert = selected_experts[assignment];
  if (expert >= kTrellis35ExpertCount) return;
  input += static_cast<std::uint64_t>(assignment / assignments_per_input) *
           input_stride;
  const std::uint16_t* suh =
      all_suh + static_cast<std::uint64_t>(expert) * physical_elements;
  output += static_cast<std::uint64_t>(assignment) * physical_elements;
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
  output[index] = accumulator * kHadamardScale;
}

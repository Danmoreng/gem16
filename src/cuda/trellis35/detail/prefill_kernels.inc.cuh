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

template <int Rate>
__global__ void DecodeTrellis35E4M3SlabDiagnosticKernel(
    Trellis35DeviceFamilyBinding family, std::uint32_t expert,
    std::uint64_t input_elements, std::uint64_t output_elements,
    std::uint64_t output_offset, std::uint64_t slab_rows,
    std::uint8_t* weight_e4m3, std::uint16_t* weight_scales_bf16) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t slab_elements = slab_rows * input_elements;
  if (index >= slab_elements) return;
  const std::uint64_t row = index / input_elements;
  const std::uint64_t input = index - row * input_elements;
  const Trellis35ExpertDescriptor descriptor = family.descriptors[expert];
  const std::byte* pool =
      Rate == 3 ? family.k3_payload_pool : family.k4_payload_pool;
  const half decoded = DecodeWeight<Rate>(
      pool, descriptor.pool_offset, input_elements, output_elements, input,
      output_offset + row);
  weight_e4m3[index] = __nv_fp8_e4m3(__half2float(decoded)).__x;
  if (input == 0U) weight_scales_bf16[row] = Bf16Bits(1.0F);
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

__device__ __forceinline__ unsigned PrefillSharedAddress(
    const void* pointer) {
  return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void PrefillCopyAsync16(
    void* shared_destination, const void* global_source, int source_bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
               :
               : "r"(PrefillSharedAddress(shared_destination)),
                 "l"(global_source), "r"(source_bytes));
#else
  (void)shared_destination;
  (void)global_source;
  (void)source_bytes;
#endif
}

__device__ __forceinline__ void PrefillCommitAsyncCopies() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  asm volatile("cp.async.commit_group;\n");
#endif
}

__device__ __forceinline__ void PrefillWaitForAsyncCopies() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  asm volatile("cp.async.wait_group 0;\n");
#endif
}

__device__ __forceinline__ void StageGroupedPrefillActivationAsync(
    std::uint32_t* staged_activation, unsigned stage,
    const std::uint8_t* activation, const std::uint32_t* activation_rows,
    std::uint64_t input_elements, std::uint64_t input_offset) {
  constexpr unsigned kWordsPerK32 = 32U / sizeof(std::uint32_t);
  constexpr unsigned kCopiesPerRow = 32U / 16U;
  constexpr unsigned kCopies =
      kPrefillGroupedRowsPerTile * kCopiesPerRow;
  for (unsigned copy = threadIdx.x; copy < kCopies; copy += blockDim.x) {
    const unsigned row = copy / kCopiesPerRow;
    const unsigned chunk = copy % kCopiesPerRow;
    const std::uint32_t activation_row = activation_rows[row];
    const bool valid = activation_row != 0xffffffffU;
    const std::uint8_t* source =
        valid ? activation +
                    static_cast<std::uint64_t>(activation_row) *
                        input_elements +
                    input_offset + chunk * 16U
              : activation;
    PrefillCopyAsync16(
        staged_activation +
            (stage * kPrefillGroupedRowsPerTile + row) * kWordsPerK32 +
            chunk * (16U / sizeof(std::uint32_t)),
        source, valid ? 16 : 0);
  }
}

template <int Rate>
__device__ __forceinline__ void AccumulateGroupedPrefillM32(
    const std::uint8_t* activation, const std::uint32_t* activation_rows,
    const std::byte* pool, std::uint32_t pool_offset,
    std::uint64_t input_elements, std::uint64_t output_elements,
    std::uint64_t source_output, std::uint32_t* staged_activation,
    Fp8Accumulator (&accumulators)[2]) {
  constexpr unsigned kWordsPerK32 = 32U / sizeof(std::uint32_t);
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

  StageGroupedPrefillActivationAsync(staged_activation, 0U, activation,
                                     activation_rows, input_elements, 0U);
  PrefillCommitAsyncCopies();
  PrefillWaitForAsyncCopies();
  __syncthreads();
  for (unsigned iteration = 0U; iteration < iterations; ++iteration) {
    const unsigned stage = iteration & 1U;
    const bool has_next = iteration + 1U < iterations;
    if (has_next) {
      StageGroupedPrefillActivationAsync(
          staged_activation, stage ^ 1U, activation, activation_rows,
          input_elements,
          static_cast<std::uint64_t>(iteration + 1U) * 32U);
      PrefillCommitAsyncCopies();
    }
    const std::uint32_t lane_word0 = load_word(iteration * 2U);
    const std::uint32_t lane_word1 = load_word(iteration * 2U + 1U);
    const std::uint32_t b0 =
        DecodeLanePayloadE4M3x4<Rate>(lane_word0, position);
    const std::uint32_t b1 =
        DecodeLanePayloadE4M3x4<Rate>(lane_word1, position);
    const std::uint32_t* current =
        staged_activation + stage * kPrefillGroupedRowsPerTile * kWordsPerK32;
#pragma unroll
    for (unsigned tile = 0U; tile < 2U; ++tile) {
      const unsigned low_row = tile * 16U + group;
      const unsigned high_row = low_row + 8U;
      const std::uint32_t a0 =
          current[low_row * kWordsPerK32 + thread_in_group];
      const std::uint32_t a1 =
          current[high_row * kWordsPerK32 + thread_in_group];
      const std::uint32_t a2 =
          current[low_row * kWordsPerK32 + thread_in_group + 4U];
      const std::uint32_t a3 =
          current[high_row * kWordsPerK32 + thread_in_group + 4U];
      AccumulateFp8M16(a0, a1, a2, a3, b0, b1, accumulators[tile]);
    }
    if (has_next) {
      PrefillWaitForAsyncCopies();
      __syncthreads();
    }
  }
}

__global__ void MmaW4A8ProjectionGroupedPrefillM32Kernel(
    const std::uint8_t* activation, const float* activation_scales,
    Trellis35DeviceFamilyBinding family,
    const Gemma4MoePrefillAssignment* assignments,
    const std::uint32_t* expert_prefix, const std::uint32_t* schedule_count,
    const std::uint32_t* schedule, const std::uint32_t* permutation,
    float* output_tile, std::uint64_t assignment_count,
    std::uint64_t input_elements, std::uint64_t output_elements,
    std::uint64_t output_offset) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  const unsigned schedule_index = blockIdx.y;
  if (schedule_index >= schedule_count[0]) return;
  const std::uint32_t packed_tile = schedule[schedule_index];
  const unsigned expert = packed_tile >> 16U;
  const std::uint32_t grouped_begin = packed_tile & 0xffffU;
  if (expert >= kTrellis35ExpertCount) return;
  const std::uint32_t grouped_end = expert_prefix[expert + 1U];
  if (grouped_begin >= grouped_end) return;

  __shared__ std::uint32_t activation_rows[kPrefillGroupedRowsPerTile];
  __shared__ alignas(16) std::uint32_t staged_activation
      [2U * kPrefillGroupedRowsPerTile * (32U / sizeof(std::uint32_t))];
  if (threadIdx.x < kPrefillGroupedRowsPerTile) {
    const std::uint32_t grouped = grouped_begin + threadIdx.x;
    std::uint32_t original = 0xffffffffU;
    if (grouped < grouped_end) {
      const std::uint32_t candidate = permutation[grouped];
      if (candidate < assignment_count &&
          assignments[candidate].expert_id == expert) {
        original = candidate;
      }
    }
    activation_rows[threadIdx.x] = original;
  }
  __syncthreads();

  const unsigned warp = threadIdx.x >> 5U;
  const unsigned lane = threadIdx.x & 31U;
  const unsigned group = lane >> 2U;
  const unsigned thread_in_group = lane & 3U;
  const std::uint64_t output_column_base =
      output_offset + static_cast<std::uint64_t>(blockIdx.x) * 32U +
      warp * 8U;
  const std::uint64_t source_output = output_column_base + group;
  const Trellis35ExpertDescriptor descriptor = family.descriptors[expert];
  const std::byte* pool = descriptor.rate_bits == 3U
                              ? family.k3_payload_pool
                              : family.k4_payload_pool;
  Fp8Accumulator accumulators[2]{};
  if (descriptor.rate_bits == 3U) {
    AccumulateGroupedPrefillM32<3>(
        activation, activation_rows, pool, descriptor.pool_offset,
        input_elements, output_elements, source_output, staged_activation,
        accumulators);
  } else if (descriptor.rate_bits == 4U) {
    AccumulateGroupedPrefillM32<4>(
        activation, activation_rows, pool, descriptor.pool_offset,
        input_elements, output_elements, source_output, staged_activation,
        accumulators);
  } else {
    return;
  }

#pragma unroll
  for (unsigned tile = 0U; tile < 2U; ++tile) {
    const float values[4] = {accumulators[tile].x0, accumulators[tile].x1,
                             accumulators[tile].x2, accumulators[tile].x3};
#pragma unroll
    for (unsigned pair = 0U; pair < 4U; ++pair) {
      const unsigned row = tile * 16U +
                           ((pair & 2U) == 0U ? group : group + 8U);
      const std::uint32_t original = activation_rows[row];
      const std::uint64_t column =
          output_column_base + thread_in_group * 2U + (pair & 1U);
      if (original == 0xffffffffU || column >= output_elements) continue;
      output_tile[static_cast<std::uint64_t>(original) *
                      kPrefillOutputBlock +
                  (column - output_offset)] =
          values[pair] * activation_scales[original];
    }
  }
#else
  (void)activation;
  (void)activation_scales;
  (void)family;
  (void)assignments;
  (void)expert_prefix;
  (void)schedule_count;
  (void)schedule;
  (void)permutation;
  (void)output_tile;
  (void)assignment_count;
  (void)input_elements;
  (void)output_elements;
  (void)output_offset;
#endif
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
      min(grouped_end - grouped_begin, kPrefillLegacyRowsPerTile));
  unsigned assignments[kPrefillLegacyRowsPerTile]{};
#pragma unroll
  for (unsigned index = 0U; index < kPrefillLegacyRowsPerTile; ++index) {
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
  Fp8Accumulator accumulators[kPrefillLegacyRowsPerTile]{};
  if (descriptor.rate_bits == 3U) {
    AccumulateGroupedProjection<3, kPrefillLegacyRowsPerTile>(
        activation, assignments, assignment_count, pool,
        descriptor.pool_offset, input_elements, output_elements, source_output,
        accumulators);
  } else {
    AccumulateGroupedProjection<4, kPrefillLegacyRowsPerTile>(
        activation, assignments, assignment_count, pool,
        descriptor.pool_offset, input_elements, output_elements, source_output,
        accumulators);
  }
  if (lane < 4U) {
    const std::uint64_t column =
        (static_cast<std::uint64_t>(blockIdx.x) * kMmaWarps + warp) * 8U +
        lane * 2U;
#pragma unroll
    for (unsigned index = 0U; index < kPrefillLegacyRowsPerTile; ++index) {
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

template <unsigned RowsPerTile>
__global__ void BuildTrellis35PrefillTileScheduleKernel(
    const std::uint32_t* prefix, std::uint32_t* tile_count,
    std::uint32_t* schedule) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) return;
  std::uint32_t count = 0U;
  for (std::uint32_t expert = 0U; expert < kTrellis35ExpertCount; ++expert) {
    for (std::uint32_t grouped = prefix[expert];
         grouped < prefix[expert + 1U];
         grouped += RowsPerTile) {
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

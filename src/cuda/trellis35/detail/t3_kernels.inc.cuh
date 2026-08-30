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

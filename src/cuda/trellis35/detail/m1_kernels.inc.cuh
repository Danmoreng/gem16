__global__ void MmaW4A8ProjectionSelectedKernel(
    const std::uint8_t* activation, const float* activation_scales,
    Trellis35DeviceFamilyBinding family,
    const std::uint32_t* selected_experts, float* output,
    std::uint64_t input_elements, std::uint64_t output_elements) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  const unsigned slot = blockIdx.y;
  const unsigned warp = threadIdx.x >> 5U;
  const unsigned lane = threadIdx.x & 31U;
  const std::uint64_t output_base =
      (static_cast<std::uint64_t>(blockIdx.x) * kMmaWarps + warp) * 8U;
  const std::uint64_t source_output = output_base + (lane >> 2U);
  if (source_output >= output_elements) return;

  const std::uint32_t expert = selected_experts[slot];
  if (expert >= kTrellis35ExpertCount) return;
  const Trellis35ExpertDescriptor descriptor = family.descriptors[expert];
  const std::byte* pool = descriptor.rate_bits == 3U
                              ? family.k3_payload_pool
                              : family.k4_payload_pool;
  activation += static_cast<std::uint64_t>(slot) * input_elements;
  output += static_cast<std::uint64_t>(slot) * output_elements;
  Fp8Accumulator accumulator;
  if (descriptor.rate_bits == 3U) {
    AccumulateSelectedProjectionM1<3>(
        activation, pool, descriptor.pool_offset, input_elements,
        output_elements, source_output, accumulator);
  } else {
    AccumulateSelectedProjectionM1<4>(
        activation, pool, descriptor.pool_offset, input_elements,
        output_elements, source_output, accumulator);
  }

  if (lane < 4U) {
    const std::uint64_t index = output_base + lane * 2U;
    const float scale = activation_scales[slot];
    if (index < output_elements) output[index] = accumulator.x0 * scale;
    if (index + 1U < output_elements) {
      output[index + 1U] = accumulator.x1 * scale;
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

__global__ void InputTransformKernel(const float* input,
                                     const std::uint16_t* suh,
                                     float* output,
                                     std::uint64_t logical_elements,
                                     std::uint64_t physical_elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= physical_elements) return;
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

__global__ void ReferenceW4A8ProjectionKernel(
    const std::uint8_t* activation, const float* activation_scale,
    Trellis35DeviceFamilyBinding family, std::uint32_t expert, float* output,
    std::uint64_t input_elements, std::uint64_t output_elements) {
  const std::uint64_t column =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column >= output_elements) return;
  const Trellis35ExpertDescriptor descriptor = family.descriptors[expert];
  float accumulator = 0.0F;
  for (std::uint64_t row = 0U; row < input_elements; ++row) {
    __nv_fp8_e4m3 activation_value;
    activation_value.__x = activation[row];
    const half decoded =
        descriptor.rate_bits == 3U
            ? DecodeWeight<3>(family.k3_payload_pool, descriptor.pool_offset,
                              input_elements, output_elements, row, column)
            : DecodeWeight<4>(family.k4_payload_pool, descriptor.pool_offset,
                              input_elements, output_elements, row, column);
    const __nv_fp8_e4m3 weight_value(__half2float(decoded));
    accumulator = fmaf(static_cast<float>(activation_value),
                       static_cast<float>(weight_value), accumulator);
  }
  output[column] = accumulator * activation_scale[0];
}

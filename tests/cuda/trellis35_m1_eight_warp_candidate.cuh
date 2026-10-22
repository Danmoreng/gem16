// Unqualified test-only candidate. Included after the existing codec/MMA/H128
// helpers, so numerical boundaries stay shared with the measured parent.
template <int Rate>
__device__ __forceinline__ void AccumulateSelectedPairM1(
    const std::uint8_t* activation, const std::byte* pool,
    std::uint32_t pool_offset, std::uint64_t input_elements,
    std::uint64_t output_elements, std::uint64_t source_output,
    Fp8Accumulator& left, Fp8Accumulator& right) {
  constexpr unsigned kPrefetch = 4U;
  constexpr unsigned kPayloadWords = Rate * 256U / 32U;
  const unsigned lane = threadIdx.x & 31U;
  const unsigned quarter = lane & 3U;
  const auto* payload = reinterpret_cast<const std::uint32_t*>(pool + pool_offset);
  const auto tile_columns = output_elements / 16U;
  const auto output_tile = source_output / 16U;
  const unsigned position = TensorCoreIndex(quarter * 4U, static_cast<unsigned>(source_output & 15U));
  const unsigned iterations = static_cast<unsigned>(input_elements / 32U);
  auto load = [&](unsigned input_tile) {
    if (lane >= kPayloadWords) return std::uint32_t{0U};
    return payload[(static_cast<std::uint64_t>(input_tile) * tile_columns + output_tile) * kPayloadWords + lane];
  };
  std::uint32_t prefetched0[kPrefetch]{}, prefetched1[kPrefetch]{};
#pragma unroll
  for (unsigned depth = 0U; depth < kPrefetch; ++depth) {
    if (depth < iterations) {
      prefetched0[depth] = load(depth * 2U);
      prefetched1[depth] = load(depth * 2U + 1U);
    }
  }
  for (unsigned base = 0U; base < iterations; base += kPrefetch) {
#pragma unroll
    for (unsigned depth = 0U; depth < kPrefetch; ++depth) {
      const unsigned iteration = base + depth;
      if (iteration >= iterations) break;
      const auto word0 = prefetched0[depth], word1 = prefetched1[depth];
      if (iteration + kPrefetch < iterations) {
        prefetched0[depth] = load((iteration + kPrefetch) * 2U);
        prefetched1[depth] = load((iteration + kPrefetch) * 2U + 1U);
      }
      const auto first = static_cast<std::uint64_t>(iteration) * 32U + quarter * 4U;
      const auto a0 = *reinterpret_cast<const std::uint32_t*>(activation + first);
      const auto a1 = *reinterpret_cast<const std::uint32_t*>(activation + first + 16U);
      const auto b0 = DecodeLanePayloadSelectedE4M3x4<Rate, true>(word0, position);
      const auto b1 = DecodeLanePayloadSelectedE4M3x4<Rate, true>(word1, position);
      const auto c0 = DecodeLanePayloadSelectedE4M3x4<Rate, true>(word0, position + 4U);
      const auto c1 = DecodeLanePayloadSelectedE4M3x4<Rate, true>(word1, position + 4U);
      AccumulateFp8(a0, a1, b0, b1, left);
      AccumulateFp8(a0, a1, c0, c1, right);
    }
  }
}

__global__ void MmaW4A8ProjectionSelectedN128EightWarpCandidateKernel(
    const std::uint8_t* activation, const float* activation_scales,
    Trellis35DeviceFamilyBinding family, const std::uint32_t* selected_experts,
    float* output, std::uint64_t input_elements, std::uint64_t output_elements) {
  const unsigned slot = blockIdx.y, warp = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
  const auto output_block = static_cast<std::uint64_t>(blockIdx.x) * 128U;
  const auto source = output_block + warp * 16U + (lane >> 2U);
  const auto expert = selected_experts[slot];
  if (expert >= kTrellis35ExpertCount) return;
  const auto descriptor = family.descriptors[expert];
  const auto* pool = descriptor.rate_bits == 3U ? family.k3_payload_pool : family.k4_payload_pool;
  activation += static_cast<std::uint64_t>(slot) * input_elements;
  output += static_cast<std::uint64_t>(slot) * output_elements;
  Fp8Accumulator left{}, right{};
  if (descriptor.rate_bits == 3U) {
    AccumulateSelectedPairM1<3>(activation, pool, descriptor.pool_offset,
                               input_elements, output_elements, source, left, right);
  } else {
    AccumulateSelectedPairM1<4>(activation, pool, descriptor.pool_offset,
                               input_elements, output_elements, source, left, right);
  }
  __shared__ float transformed[128];
  if (lane < 4U) {
    const unsigned i = warp * 16U + lane * 2U;
    const float scale = activation_scales[slot];
    transformed[i] = left.x0 * scale; transformed[i + 1U] = left.x1 * scale;
    transformed[i + 8U] = right.x0 * scale; transformed[i + 9U] = right.x1 * scale;
  }
  __syncthreads();
  if (warp == 0U) {
    const unsigned i = lane * 4U;
    float values[4]{transformed[i], transformed[i + 1U], transformed[i + 2U], transformed[i + 3U]};
    H128Warp(values);
    const auto* svh = family.svh_f16 + static_cast<std::uint64_t>(expert) * output_elements + output_block + i;
    const float4 packed{values[0] * F16(svh), values[1] * F16(svh + 1U),
                        values[2] * F16(svh + 2U), values[3] * F16(svh + 3U)};
    *reinterpret_cast<float4*>(output + output_block + i) = packed;
  }
}

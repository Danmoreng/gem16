// Diagnostic T3 GateUp fixed resident-payload pipeline; unchanged K order.
// The same 32 paired-MMA positions use the qualified GateUp word-window codec.
template <int Rate, unsigned Depth>
__device__ __forceinline__ void AccumulateGroupedT3GatePipeline(
    const std::uint8_t* activation, const unsigned* assignments,
    unsigned count, const std::byte* pool, std::uint32_t offset,
    std::uint64_t inputs, std::uint64_t outputs, std::uint64_t source,
    Fp8Accumulator& left, Fp8Accumulator& right) {
  constexpr unsigned kWords = Rate * 256U / 32U;
  const unsigned lane = threadIdx.x & 31U, group = lane >> 2U, quarter = lane & 3U;
  const auto* payload = reinterpret_cast<const std::uint32_t*>(pool + offset);
  const auto columns = outputs / 16U, output_tile = source / 16U;
  const unsigned position = TensorCoreIndex(quarter * 4U, static_cast<unsigned>(source & 15U));
  auto load = [&](unsigned tile) {
    if (lane >= kWords) return std::uint32_t{0U};
    return payload[(static_cast<std::uint64_t>(tile) * columns + output_tile) * kWords + lane];
  };
  constexpr unsigned kDepth = Depth;
  const unsigned iterations = static_cast<unsigned>(inputs / 32U);
  std::uint32_t pending0[kDepth]{}, pending1[kDepth]{};
#pragma unroll
  for (unsigned depth = 0U; depth < kDepth; ++depth) {
    if (depth < iterations) {
      pending0[depth] = load(depth * 2U);
      pending1[depth] = load(depth * 2U + 1U);
    }
  }
  for (unsigned base = 0U; base < iterations; base += kDepth) {
#pragma unroll
    for (unsigned depth = 0U; depth < kDepth; ++depth) {
      const unsigned i = base + depth;
      if (i >= iterations) break;
      const auto w0 = pending0[depth], w1 = pending1[depth];
      if (i + kDepth < iterations) {
        pending0[depth] = load((i + kDepth) * 2U);
        pending1[depth] = load((i + kDepth) * 2U + 1U);
      }
      std::uint32_t b0 = 0U, b1 = 0U, c0 = 0U, c1 = 0U;
      DecodeT3GatePairWindow<Rate>(w0, position, b0, c0);
      DecodeT3GatePairWindow<Rate>(w1, position, b1, c1);
      std::uint32_t a0 = 0U, a2 = 0U;
      if (group < count) {
        const auto* row = activation + static_cast<std::uint64_t>(assignments[group]) * inputs;
        const auto first = static_cast<std::uint64_t>(i) * 32U + quarter * 4U;
        a0 = *reinterpret_cast<const std::uint32_t*>(row + first);
        a2 = *reinterpret_cast<const std::uint32_t*>(row + first + 16U);
      }
      AccumulateFp8M16(a0, 0U, a2, 0U, b0, b1, left);
      AccumulateFp8M16(a0, 0U, a2, 0U, c0, c1, right);
    }
  }
}

template <unsigned Depth>
__global__ void MmaW4A8ProjectionGroupedT3GatePipelineCandidateKernel(
    const std::uint8_t* activation, const float* scales,
    Trellis35DeviceFamilyBinding family, const std::uint32_t* ids,
    float* output, std::uint64_t inputs, std::uint64_t outputs) {
  const unsigned candidate = blockIdx.y;
  const unsigned expert = ids[candidate];
  if (expert >= kTrellis35ExpertCount) return;
  for (unsigned prior = 0U; prior < candidate; ++prior) if (ids[prior] == expert) return;
  unsigned assignments[kTrellis35T3Rows]{}, count = 0U;
#pragma unroll
  for (unsigned row = 0U; row < kTrellis35T3Rows; ++row) {
#pragma unroll
    for (unsigned slot = 0U; slot < kTrellis35M1TopK; ++slot) {
      const unsigned assignment = row * kTrellis35M1TopK + slot;
      if (ids[assignment] == expert) { assignments[count++] = assignment; break; }
    }
  }
  const unsigned warp = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
  const unsigned group = lane >> 2U, quarter = lane & 3U;
  const auto block = static_cast<std::uint64_t>(blockIdx.x) * 128U;
  const auto source = block + warp * 16U + group;
  const auto descriptor = family.descriptors[expert];
  const auto* pool = descriptor.rate_bits == 3U ? family.k3_payload_pool : family.k4_payload_pool;
  Fp8Accumulator left{}, right{};
  if (descriptor.rate_bits == 3U) AccumulateGroupedT3GatePipeline<3, Depth>(activation, assignments, count,
      pool, descriptor.pool_offset, inputs, outputs, source, left, right);
  else if (descriptor.rate_bits == 4U) AccumulateGroupedT3GatePipeline<4, Depth>(activation, assignments, count,
      pool, descriptor.pool_offset, inputs, outputs, source, left, right);
  else return;
  __shared__ float transformed[kTrellis35T3Rows][128U];
  const float l[4]{left.x0, left.x1, left.x2, left.x3};
  const float r[4]{right.x0, right.x1, right.x2, right.x3};
#pragma unroll
  for (unsigned pair = 0U; pair < 4U; ++pair) {
    const unsigned row = (pair & 2U) == 0U ? group : group + 8U;
    if (row < count) {
      const unsigned column = warp * 16U + quarter * 2U + (pair & 1U);
      const float scale = scales[assignments[row]];
      transformed[row][column] = l[pair] * scale;
      transformed[row][column + 8U] = r[pair] * scale;
    }
  }
  __syncthreads();
  if (warp < count) {
    const unsigned i = lane * 4U;
    float v[4]{transformed[warp][i], transformed[warp][i + 1U],
               transformed[warp][i + 2U], transformed[warp][i + 3U]};
    H128Warp(v);
    const auto* svh = family.svh_f16 + static_cast<std::uint64_t>(expert) * outputs + block + i;
    const float4 result{v[0] * F16(svh), v[1] * F16(svh + 1U),
                        v[2] * F16(svh + 2U), v[3] * F16(svh + 3U)};
    *reinterpret_cast<float4*>(output + static_cast<std::uint64_t>(assignments[warp]) * outputs + block + i) = result;
  }
}

// One counter-gated register-bounded diagnostic variant.
template <unsigned Depth>
__launch_bounds__(256U, 5) __global__ void MmaW4A8ProjectionGroupedT3GatePipelineBoundedCandidateKernel(
    const std::uint8_t* activation, const float* scales,
    Trellis35DeviceFamilyBinding family, const std::uint32_t* ids,
    float* output, std::uint64_t inputs, std::uint64_t outputs) {
  const unsigned candidate = blockIdx.y;
  const unsigned expert = ids[candidate];
  if (expert >= kTrellis35ExpertCount) return;
  for (unsigned prior = 0U; prior < candidate; ++prior) if (ids[prior] == expert) return;
  unsigned assignments[kTrellis35T3Rows]{}, count = 0U;
#pragma unroll
  for (unsigned row = 0U; row < kTrellis35T3Rows; ++row) {
#pragma unroll
    for (unsigned slot = 0U; slot < kTrellis35M1TopK; ++slot) {
      const unsigned assignment = row * kTrellis35M1TopK + slot;
      if (ids[assignment] == expert) { assignments[count++] = assignment; break; }
    }
  }
  const unsigned warp = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
  const unsigned group = lane >> 2U, quarter = lane & 3U;
  const auto block = static_cast<std::uint64_t>(blockIdx.x) * 128U;
  const auto source = block + warp * 16U + group;
  const auto descriptor = family.descriptors[expert];
  const auto* pool = descriptor.rate_bits == 3U ? family.k3_payload_pool : family.k4_payload_pool;
  Fp8Accumulator left{}, right{};
  if (descriptor.rate_bits == 3U) AccumulateGroupedT3GatePipeline<3, Depth>(activation, assignments, count,
      pool, descriptor.pool_offset, inputs, outputs, source, left, right);
  else if (descriptor.rate_bits == 4U) AccumulateGroupedT3GatePipeline<4, Depth>(activation, assignments, count,
      pool, descriptor.pool_offset, inputs, outputs, source, left, right);
  else return;
  __shared__ float transformed[kTrellis35T3Rows][128U];
  const float l[4]{left.x0, left.x1, left.x2, left.x3};
  const float r[4]{right.x0, right.x1, right.x2, right.x3};
#pragma unroll
  for (unsigned pair = 0U; pair < 4U; ++pair) {
    const unsigned row = (pair & 2U) == 0U ? group : group + 8U;
    if (row < count) {
      const unsigned column = warp * 16U + quarter * 2U + (pair & 1U);
      const float scale = scales[assignments[row]];
      transformed[row][column] = l[pair] * scale;
      transformed[row][column + 8U] = r[pair] * scale;
    }
  }
  __syncthreads();
  if (warp < count) {
    const unsigned i = lane * 4U;
    float v[4]{transformed[warp][i], transformed[warp][i + 1U],
               transformed[warp][i + 2U], transformed[warp][i + 3U]};
    H128Warp(v);
    const auto* svh = family.svh_f16 + static_cast<std::uint64_t>(expert) * outputs + block + i;
    const float4 result{v[0] * F16(svh), v[1] * F16(svh + 1U),
                        v[2] * F16(svh + 2U), v[3] * F16(svh + 3U)};
    *reinterpret_cast<float4*>(output + static_cast<std::uint64_t>(assignments[warp]) * outputs + block + i) = result;
  }
}

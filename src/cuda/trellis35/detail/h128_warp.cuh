// Normalized 128-point Walsh-Hadamard transform. Each warp lane owns four
// contiguous values, so the local H4 resolves bits [1:0] and the shuffle
// stages resolve bits [6:2]. The transform is self-inverse after applying the
// 1/sqrt(128) normalization on both sides.
__device__ __forceinline__ void H128Warp(float (&values)[4]) {
  const float sum01 = values[0] + values[1];
  const float difference01 = values[0] - values[1];
  const float sum23 = values[2] + values[3];
  const float difference23 = values[2] - values[3];
  values[0] = sum01 + sum23;
  values[1] = difference01 + difference23;
  values[2] = sum01 - sum23;
  values[3] = difference01 - difference23;

  const unsigned lane = threadIdx.x & 31U;
#pragma unroll
  for (unsigned mask = 1U; mask < 32U; mask <<= 1U) {
#pragma unroll
    for (unsigned element = 0U; element < 4U; ++element) {
      const float partner =
          __shfl_xor_sync(0xffffffffU, values[element], mask);
      values[element] =
          (lane & mask) == 0U ? values[element] + partner
                              : partner - values[element];
    }
  }
#pragma unroll
  for (unsigned element = 0U; element < 4U; ++element) {
    values[element] *= kHadamardScale;
  }
}

__global__ void H128WarpDiagnosticKernel(const float* input, float* output,
                                         std::uint64_t vectors) {
  const unsigned warp = threadIdx.x >> 5U;
  const unsigned lane = threadIdx.x & 31U;
  const std::uint64_t vector =
      static_cast<std::uint64_t>(blockIdx.x) * 8U + warp;
  if (vector >= vectors) return;
  const std::uint64_t base = vector * 128U + lane * 4U;
  float values[4];
#pragma unroll
  for (unsigned element = 0U; element < 4U; ++element) {
    values[element] = input[base + element];
  }
  H128Warp(values);
#pragma unroll
  for (unsigned element = 0U; element < 4U; ++element) {
    output[base + element] = values[element];
  }
}

struct Fp8Accumulator {
  float x0 = 0.0F;
  float x1 = 0.0F;
  float x2 = 0.0F;
  float x3 = 0.0F;
};

__device__ __forceinline__ void AccumulateFp8(
    std::uint32_t a0, std::uint32_t a1, std::uint32_t b0,
    std::uint32_t b1, Fp8Accumulator& accumulator) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %4, %5, %5}, "
      "{%6, %7}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(accumulator.x0), "+f"(accumulator.x1),
        "+f"(accumulator.x2), "+f"(accumulator.x3)
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
#else
  (void)a0;
  (void)a1;
  (void)b0;
  (void)b1;
  (void)accumulator;
#endif
}

__device__ __forceinline__ void AccumulateFp8M16(
    std::uint32_t a0, std::uint32_t a1, std::uint32_t a2,
    std::uint32_t a3, std::uint32_t b0, std::uint32_t b1,
    Fp8Accumulator& accumulator) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(accumulator.x0), "+f"(accumulator.x1),
        "+f"(accumulator.x2), "+f"(accumulator.x3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
  (void)a0;
  (void)a1;
  (void)a2;
  (void)a3;
  (void)b0;
  (void)b1;
  (void)accumulator;
#endif
}

template <int Rate>
__device__ __forceinline__ void AccumulateSelectedProjectionM1(
    const std::uint8_t* activation, const std::byte* pool,
    std::uint32_t pool_offset, std::uint64_t input_elements,
    std::uint64_t output_elements, std::uint64_t source_output,
    Fp8Accumulator& accumulator) {
  constexpr unsigned kPrefetch = 4U;
  constexpr unsigned kPayloadWords = Rate * 256U / 32U;
  const unsigned lane = threadIdx.x & 31U;
  const std::uint64_t tile_columns = output_elements / 16U;
  const std::uint64_t output_tile = source_output / 16U;
  const auto* payload = reinterpret_cast<const std::uint32_t*>(
      pool + pool_offset);
  const unsigned iterations = static_cast<unsigned>(input_elements / 32U);
  const unsigned k_quarter = lane & 3U;
  const unsigned position = TensorCoreIndex(
      k_quarter * 4U, static_cast<unsigned>(source_output & 15U));

  auto load_word = [&](unsigned input_tile) {
    if (lane >= kPayloadWords) return std::uint32_t{0U};
    const std::uint64_t tile =
        static_cast<std::uint64_t>(input_tile) * tile_columns + output_tile;
    return payload[tile * kPayloadWords + lane];
  };

  std::uint32_t prefetched0[kPrefetch]{};
  std::uint32_t prefetched1[kPrefetch]{};
#pragma unroll
  for (unsigned depth = 0U; depth < kPrefetch; ++depth) {
    if (depth < iterations) {
      prefetched0[depth] = load_word(depth * 2U);
      prefetched1[depth] = load_word(depth * 2U + 1U);
    }
  }

  for (unsigned base = 0U; base < iterations; base += kPrefetch) {
#pragma unroll
    for (unsigned depth = 0U; depth < kPrefetch; ++depth) {
      const unsigned iteration = base + depth;
      if (iteration >= iterations) break;
      const std::uint32_t lane_word0 = prefetched0[depth];
      const std::uint32_t lane_word1 = prefetched1[depth];
      if (iteration + kPrefetch < iterations) {
        prefetched0[depth] = load_word((iteration + kPrefetch) * 2U);
        prefetched1[depth] =
            load_word((iteration + kPrefetch) * 2U + 1U);
      }
      const std::uint64_t first =
          static_cast<std::uint64_t>(iteration) * 32U + k_quarter * 4U;
      const std::uint32_t a0 =
          *reinterpret_cast<const std::uint32_t*>(activation + first);
      const std::uint32_t a1 =
          *reinterpret_cast<const std::uint32_t*>(activation + first + 16U);
      const std::uint32_t b0 =
          DecodeLanePayloadE4M3x4<Rate>(lane_word0, position);
      const std::uint32_t b1 =
          DecodeLanePayloadE4M3x4<Rate>(lane_word1, position);
      AccumulateFp8(a0, a1, b0, b1, accumulator);
    }
  }
}

template <int Rate, unsigned MaximumRows>
__device__ __forceinline__ void AccumulateGroupedProjection(
    const std::uint8_t* activation, const unsigned* assignments,
    unsigned assignment_count, const std::byte* pool,
    std::uint32_t pool_offset, std::uint64_t input_elements,
    std::uint64_t output_elements, std::uint64_t source_output,
    Fp8Accumulator (&accumulators)[MaximumRows]) {
  constexpr unsigned kPrefetch = 4U;
  constexpr unsigned kPayloadWords = Rate * 256U / 32U;
  const unsigned lane = threadIdx.x & 31U;
  const std::uint64_t tile_columns = output_elements / 16U;
  const std::uint64_t output_tile = source_output / 16U;
  const auto* payload = reinterpret_cast<const std::uint32_t*>(
      pool + pool_offset);
  const unsigned iterations = static_cast<unsigned>(input_elements / 32U);
  const unsigned k_quarter = lane & 3U;
  const unsigned position = TensorCoreIndex(
      k_quarter * 4U, static_cast<unsigned>(source_output & 15U));

  auto load_word = [&](unsigned input_tile) {
    if (lane >= kPayloadWords) return std::uint32_t{0U};
    const std::uint64_t tile =
        static_cast<std::uint64_t>(input_tile) * tile_columns + output_tile;
    return payload[tile * kPayloadWords + lane];
  };

  std::uint32_t prefetched0[kPrefetch]{};
  std::uint32_t prefetched1[kPrefetch]{};
#pragma unroll
  for (unsigned depth = 0U; depth < kPrefetch; ++depth) {
    if (depth < iterations) {
      prefetched0[depth] = load_word(depth * 2U);
      prefetched1[depth] = load_word(depth * 2U + 1U);
    }
  }

  for (unsigned base = 0U; base < iterations; base += kPrefetch) {
#pragma unroll
    for (unsigned depth = 0U; depth < kPrefetch; ++depth) {
      const unsigned iteration = base + depth;
      if (iteration >= iterations) break;
      const std::uint32_t lane_word0 = prefetched0[depth];
      const std::uint32_t lane_word1 = prefetched1[depth];
      if (iteration + kPrefetch < iterations) {
        prefetched0[depth] = load_word((iteration + kPrefetch) * 2U);
        prefetched1[depth] =
            load_word((iteration + kPrefetch) * 2U + 1U);
      }
      const std::uint64_t first =
          static_cast<std::uint64_t>(iteration) * 32U + k_quarter * 4U;
      const std::uint32_t b0 =
          DecodeLanePayloadE4M3x4<Rate>(lane_word0, position);
      const std::uint32_t b1 =
          DecodeLanePayloadE4M3x4<Rate>(lane_word1, position);
#pragma unroll
      for (unsigned index = 0U; index < MaximumRows; ++index) {
        if (index < assignment_count) {
          const std::uint8_t* assignment_activation =
              activation +
              static_cast<std::uint64_t>(assignments[index]) * input_elements;
          const std::uint32_t a0 = *reinterpret_cast<const std::uint32_t*>(
              assignment_activation + first);
          const std::uint32_t a1 = *reinterpret_cast<const std::uint32_t*>(
              assignment_activation + first + 16U);
          AccumulateFp8(a0, a1, b0, b1, accumulators[index]);
        }
      }
    }
  }
}

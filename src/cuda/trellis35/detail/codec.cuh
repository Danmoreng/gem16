Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

Status CudaFailure(const char* operation, cudaError_t error) {
  return Status(StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorName(error) +
                    ": " + cudaGetErrorString(error));
}

Status CheckLaunch(const char* operation) {
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : CudaFailure(operation, error);
}

__device__ __forceinline__ float F16(const std::uint16_t* value) {
  return __half2float(__ushort_as_half(*value));
}

__device__ __forceinline__ float Bf16(std::uint16_t value) {
  return static_cast<float>(__ushort_as_bfloat16(value));
}

__device__ __forceinline__ std::uint16_t Bf16Bits(float value) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

__device__ __forceinline__ unsigned TensorCoreIndex(unsigned row,
                                                     unsigned column) {
  const unsigned thread = (column & 7U) * 4U + ((row & 7U) >> 1U);
  const unsigned offset = (column >= 8U ? 4U : 0U) +
                          (row >= 8U ? 2U : 0U) + (row & 1U);
  return thread * 8U + offset;
}

template <int Rate>
__device__ __forceinline__ half DecodeWeight(
    const std::byte* pool, std::uint32_t pool_offset,
    std::uint64_t input_elements, std::uint64_t output_elements,
    std::uint64_t input, std::uint64_t output) {
  const std::uint64_t tile_columns = output_elements / 16U;
  const std::uint64_t tile = (input / 16U) * tile_columns + output / 16U;
  const auto* payload = reinterpret_cast<const std::uint32_t*>(
      pool + pool_offset + tile * 32U * Rate);
  const unsigned position =
      TensorCoreIndex(static_cast<unsigned>(input & 15U),
                      static_cast<unsigned>(output & 15U));

  // The compiler swaps adjacent U16 words so a little-endian U32 load sees
  // the original MSW-first packed branch stream. Extract the complete
  // tail-biting 16-bit state from the two U32 words that straddle its bit
  // window instead of reloading every historical branch independently.
  // This is the bounded cb2 adaptation of ExLlamaV3 exl3_dq.cuh::dq.
  constexpr int kPayloadWords = Rate * 256 / 32;
  const int bit_begin = static_cast<int>(position) * Rate + Rate - 16 +
                        256 * Rate;
  const int bit_end = bit_begin + 16;
  const int first_word = bit_begin / 32;
  const int last_word = (bit_end - 1) / 32;
  const int shift = (last_word + 1) * 32 - bit_end;
  const std::uint32_t first = payload[first_word % kPayloadWords];
  const std::uint32_t last = payload[last_word % kPayloadWords];
  const std::uint32_t state = __funnelshift_r(last, first, shift) & 0xffffU;
  return decode_3inst<2>(state & 0xffffU);
}

template <int Rate>
__device__ __forceinline__ half2 DecodeAdjacentStates(
    std::uint32_t lane_payload_word, unsigned first_position) {
  constexpr int kPayloadWords = Rate * 256 / 32;
  const int bit_begin = static_cast<int>(first_position) * Rate + Rate - 16 +
                        256 * Rate;
  const int bit_end =
      (static_cast<int>(first_position) + 1) * Rate + Rate + 256 * Rate;
  const int first_word = bit_begin / 32;
  const int last_word = (bit_end - 1) / 32;
  const int shift = (last_word + 1) * 32 - bit_end;
  const std::uint32_t first = __shfl_sync(
      0xffffffffU, lane_payload_word, first_word % kPayloadWords);
  const std::uint32_t last = __shfl_sync(
      0xffffffffU, lane_payload_word, last_word % kPayloadWords);
  const std::uint64_t merged =
      (static_cast<std::uint64_t>(first) << 32U) | last;
  const std::uint32_t state0 =
      static_cast<std::uint32_t>(merged >> (shift + Rate)) & 0xffffU;
  const std::uint32_t state1 =
      static_cast<std::uint32_t>(merged >> shift) & 0xffffU;
  return decode_3inst_2<2>(state0, state1);
}

template <int Rate>
__device__ __forceinline__ std::uint32_t DecodeLanePayloadE4M3x4(
    std::uint32_t lane_payload_word, unsigned position) {
  const half2 first_pair =
      DecodeAdjacentStates<Rate>(lane_payload_word, position);
  const half2 second_pair =
      DecodeAdjacentStates<Rate>(lane_payload_word, position + 8U);
  const __nv_fp8_e4m3 value0(__half2float(__low2half(first_pair)));
  const __nv_fp8_e4m3 value1(__half2float(__high2half(first_pair)));
  const __nv_fp8_e4m3 value2(__half2float(__low2half(second_pair)));
  const __nv_fp8_e4m3 value3(__half2float(__high2half(second_pair)));
  const std::uint32_t packed =
      static_cast<std::uint32_t>(value0.__x) |
      (static_cast<std::uint32_t>(value1.__x) << 8U) |
      (static_cast<std::uint32_t>(value2.__x) << 16U) |
      (static_cast<std::uint32_t>(value3.__x) << 24U);
  return packed;
}

template <int Rate>
__device__ __forceinline__ std::uint32_t LoadLanePayloadWord(
    const std::byte* pool, std::uint32_t pool_offset,
    std::uint64_t output_elements, std::uint64_t input,
    std::uint64_t output) {
  constexpr unsigned kPayloadWords = Rate * 256U / 32U;
  const unsigned lane = threadIdx.x & 31U;
  if (lane >= kPayloadWords) return 0U;
  const std::uint64_t tile_columns = output_elements / 16U;
  const std::uint64_t tile =
      (input / 16U) * tile_columns + output / 16U;
  const auto* payload = reinterpret_cast<const std::uint32_t*>(
      pool + pool_offset + tile * 32U * Rate);
  return payload[lane];
}

template <int Rate>
__device__ __forceinline__ std::uint32_t DecodeWeightE4M3x4(
    const std::byte* pool, std::uint32_t pool_offset,
    std::uint64_t input_elements, std::uint64_t output_elements,
    std::uint64_t input, std::uint64_t output) {
  (void)input_elements;
  const std::uint32_t lane_payload_word = LoadLanePayloadWord<Rate>(
      pool, pool_offset, output_elements, input, output);
  const unsigned position =
      TensorCoreIndex(static_cast<unsigned>(input & 15U),
                      static_cast<unsigned>(output & 15U));
  return DecodeLanePayloadE4M3x4<Rate>(lane_payload_word, position);
}

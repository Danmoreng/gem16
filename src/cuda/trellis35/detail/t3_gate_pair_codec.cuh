// Native paired GateUp codec for TensorCoreIndex(quarter * 4, group),
// group in [0,7]. These 32 MMA positions use fixed windows, no dynamic indexing.
template <int Rate>
__device__ __forceinline__ void DecodeT3GatePairWindow(
    std::uint32_t word, unsigned position, std::uint32_t& left,
    std::uint32_t& right) {
  constexpr int kWords = Rate * 8;
  const int first = (static_cast<int>(position) * Rate + Rate - 16 + 256 * Rate) / 32;
  const auto w0 = __shfl_sync(0xffffffffU, word, first % kWords);
  const auto w1 = __shfl_sync(0xffffffffU, word, (first + 1) % kWords);
  const auto w2 = __shfl_sync(0xffffffffU, word, (first + 2) % kWords);
  const bool high = (position & 2U) != 0U;
  std::uint32_t a = 0U, b = 0U, c = 0U, d = 0U;
  if constexpr (Rate == 4) {
    const unsigned shift = high ? 16U : 24U, within = high ? 0U : 8U;
    a = __funnelshift_r(w1, w0, shift);
    c = __funnelshift_r(w2, w1, shift);
    b = w1 >> within; d = w2 >> within;
  } else {
    static_assert(Rate == 3);
    const bool odd = (position & 16U) != 0U;
    const unsigned delta = high ? 6U : 0U;
    a = __funnelshift_r(odd ? w0 : w1, w0, odd ? 10U - delta : 26U - delta);
    b = __funnelshift_r(w1, high && !odd ? w1 : w0,
                        odd ? 30U - delta : high ? 8U : 14U);
    c = __funnelshift_r(high && !odd ? w2 : w1, !high && odd ? w0 : w1,
                        high ? (odd ? 12U : 28U) : (odd ? 18U : 2U));
    d = __funnelshift_r(odd ? w1 : w2, w1, odd ? 6U - delta : 22U - delta);
  }
  const auto ac0 = decode_3inst_2<2>((a >> Rate) & 0xffffU, a & 0xffffU);
  const auto ac1 = decode_3inst_2<2>((c >> Rate) & 0xffffU, c & 0xffffU);
  left = __nv_fp8x4_e4m3(ac0, ac1).__x;
  const auto bd0 = decode_3inst_2<2>((b >> Rate) & 0xffffU, b & 0xffffU);
  const auto bd1 = decode_3inst_2<2>((d >> Rate) & 0xffffU, d & 0xffffU);
  right = __nv_fp8x4_e4m3(bd0, bd1).__x;
}


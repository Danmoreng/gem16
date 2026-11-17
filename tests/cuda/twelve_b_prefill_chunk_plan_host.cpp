#include "cuda/engine/prefill_chunk_plan.h"
#include <array>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string_view>

int main() {
  using gem16::internal::ResolveTwelveBPrefillChunkTokens;
  unsigned failures = 0U;
  const auto check = [&failures](bool condition) { if (!condition) ++failures; };
  check(ResolveTwelveBPrefillChunkTokens(8192U, true, std::nullopt).value() == 8192U);
  check(ResolveTwelveBPrefillChunkTokens(512U, false, std::nullopt).value() == 512U);
  for (const auto text : {"32", "64", "512", "1024", "2048", "4096", "8192"}) {
    const auto result = ResolveTwelveBPrefillChunkTokens(8192U, true, text);
    check(result.ok());
    if (result.ok()) check(result.value() % 32U == 0U && result.value() <= 8192U);
  }
  for (const auto text : {"", "0", "1", "31", "33", "8193", "8224", "-1024", "+1024",
                          " 1024", "1024 ", "1024x", "1.024", "0x400",
                          "18446744073709551615", "18446744073709551616",
                          "0000000000000000000001024"}) {
    check(!ResolveTwelveBPrefillChunkTokens(8192U, true, text).ok());
  }
  check(!ResolveTwelveBPrefillChunkTokens(1024U, true, "2048").ok());
  check(!ResolveTwelveBPrefillChunkTokens(8192U, false, "1024").ok());
  if (failures != 0U) { std::cerr << failures << " failed assertions\n"; return 1; }
  std::cout << "PASS: defaults, supported chunks, malformed/overflow/out-of-budget controls\n";
  return 0;
}

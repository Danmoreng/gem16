#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#if !defined(_WIN32)
#include <dlfcn.h>
#endif

namespace decode_replay {
constexpr std::uint32_t kVocabulary = 262144U;
constexpr std::uint32_t kContext = 17519U;
constexpr std::array<std::uint32_t, 5> kMask{1U, 50U, 106U, 258882U, 258883U};
using Clock = std::chrono::steady_clock;
inline void Require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
inline std::vector<std::uint32_t> ReadTokens(const std::filesystem::path& path,
                                            std::size_t expected) {
  Require(std::filesystem::is_regular_file(std::filesystem::symlink_status(path)),
          "replay input must be a regular non-symlink file");
  const auto bytes = std::filesystem::file_size(path);
  Require(bytes > 0U && bytes <= 2U * 1024U * 1024U, "replay input size exceeds bound");
  std::ifstream file(path, std::ios::binary);
  std::string text(static_cast<std::size_t>(bytes), '\0');
  file.read(text.data(), static_cast<std::streamsize>(bytes));
  Require(file.gcount() == static_cast<std::streamsize>(bytes) &&
              file.peek() == std::char_traits<char>::eof(),
          "short or growing replay input");
  Require(std::filesystem::is_regular_file(std::filesystem::symlink_status(path)),
          "replay input changed to a non-regular file");
  std::vector<std::uint32_t> tokens;
  tokens.reserve(expected);
  const char* current = text.data();
  const char* end = current + text.size();
  while (current != end) {
    while (current != end && (*current == ' ' || *current == '\n' || *current == '\r')) ++current;
    if (current == end) break;
    std::uint32_t value = 0U;
    const auto parsed = std::from_chars(current, end, value);
    Require(parsed.ec == std::errc{} && parsed.ptr != current && value < kVocabulary,
            "invalid replay token");
    Require(tokens.size() < expected, "too many replay tokens");
    tokens.push_back(value);
    current = parsed.ptr;
    if (current != end && *current == ',') {
      ++current;
      while (current != end && (*current == ' ' || *current == '\n' || *current == '\r')) ++current;
      Require(current != end, "trailing replay comma");
    } else {
      Require(current == end || *current == '\n' || *current == '\r' || *current == ' ',
              "invalid replay delimiter");
    }
  }
  Require(tokens.size() == expected, "incorrect replay token count");
  return tokens;
}
inline std::uint32_t Winner(std::span<const float> logits, bool masked = true) {
  Require(logits.size() == kVocabulary, "incorrect replay logit extent");
  std::uint32_t winner = kVocabulary;
  for (std::uint32_t t = 0U; t < kVocabulary; ++t) {
    Require(std::isfinite(logits[t]), "non-finite replay logits");
    if (masked && std::find(kMask.begin(), kMask.end(), t) != kMask.end()) continue;
    if (winner == kVocabulary || logits[t] > logits[winner]) winner = t;
  }
  Require(winner != kVocabulary, "no replay winner");
  return winner;
}
inline std::int64_t Ns(Clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}
inline void LabSave(unsigned phase) {
#if !defined(_WIN32)
  using Save = int (*)(unsigned);
  if (auto save = reinterpret_cast<Save>(dlsym(RTLD_DEFAULT, "gem16_lab_save_phase"))) {
    Require(save(phase) == 0, "operator phase capture failed");
  }
#else
  (void)phase;
#endif
}
inline void LabFinish() {
  const char* directory = std::getenv("GEM16_LAB_OUTPUT");
  if (!directory) return;
#if !defined(_WIN32)
  using Finish = int (*)(const char*);
  auto finish = reinterpret_cast<Finish>(dlsym(RTLD_DEFAULT, "gem16_lab_finish"));
  Require(finish && finish(directory) == 0, "operator export/replay failed");
#else
  throw std::runtime_error("operator lab requires the qualified Linux diagnostic");
#endif
}
struct Row {
  std::uint32_t input = 0U;
  std::uint32_t prediction = 0U;
  std::int64_t diagnostic_ns = 0;
};
inline void WriteRows(const char* engine, std::span<const Row> rows,
                      std::uint32_t initial_prediction) {
  std::cout << "{\"diagnostic_only\":true,\"engine\":\"" << engine
            << "\",\"prompt_tokens\":16384,\"requested_context\":17519,"
               "\"mode\":\"forced_history\","
               "\"timing_scope\":\"diagnostic forward plus engine-specific host selection; not comparable primary throughput\","
               "\"initial_masked_prediction\":" << initial_prediction << ",\"rows\":[";
  for (std::size_t i = 0U; i < rows.size(); ++i) {
    const auto& r = rows[i];
    std::cout << (i ? "," : "") << "{\"input_index\":" << i
              << ",\"context_position\":" << 16384U + i << ",\"forced_token\":"
              << r.input << ",\"masked_prediction\":" << r.prediction
              << ",\"diagnostic_ns\":" << r.diagnostic_ns << '}';
  }
  std::cout << "]}\n";
}
}  // namespace decode_replay

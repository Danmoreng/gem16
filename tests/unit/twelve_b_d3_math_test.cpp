#include "numeric/twelve_b_d3_math.h"
#include "test.h"
#include "util/json.h"

#include <sstream>
#include <string_view>

void RunTwelveBD3MathTests() {
  using gem16::internal::SelectTwelveBD3Math;
  using gem16::KvCacheMode;
  constexpr auto fp8 = KvCacheMode::kCheckpointFp8;
  const auto automatic = SelectTwelveBD3Math(nullptr, 3, fp8, 17519);
  GEM16_CHECK(automatic.ok());
  if (!automatic.ok()) return;
  GEM16_CHECK(automatic.value().optimized);
  GEM16_CHECK(std::string_view(automatic.value().policy) == "auto");
  for (const char* mode : {"optimized", "fp8", "auto"}) {
    const auto selection = SelectTwelveBD3Math(mode, 3, fp8, 17519);
    GEM16_CHECK(selection.ok() && selection.value().optimized);
  }
  const auto exact = SelectTwelveBD3Math("exact", 3, fp8, 17519);
  GEM16_CHECK(exact.ok() && !exact.value().optimized);
  // Unsupported automatic shapes stay on their established native path;
  // explicit optimized requests must fail instead of silently changing math.
  for (unsigned depth : {0U, 1U, 2U, 4U}) {
    const auto native = SelectTwelveBD3Math(nullptr, depth, fp8, 17519);
    GEM16_CHECK(native.ok() && !native.value().optimized);
    GEM16_CHECK(!SelectTwelveBD3Math("optimized", depth, fp8, 17519).ok());
  }
  for (auto context : {5U, 64U, 1024U, 1025U, 2048U, 17518U, 17520U,
                       32768U, 262143U, 262144U}) {
    GEM16_CHECK(SelectTwelveBD3Math(nullptr, 3, fp8, context).value().optimized);
    GEM16_CHECK(SelectTwelveBD3Math("fp8", 3, fp8, context).value().optimized);
  }
  for (auto context : {0U, 4U, 262145U}) {
    GEM16_CHECK(!SelectTwelveBD3Math(nullptr, 3, fp8, context).value().optimized);
    GEM16_CHECK(!SelectTwelveBD3Math("optimized", 3, fp8, context).ok());
  }
  GEM16_CHECK(!SelectTwelveBD3Math(nullptr, 3,
      KvCacheMode::kBf16Correctness, 17519).value().optimized);
  GEM16_CHECK(!SelectTwelveBD3Math("optimized", 3, fp8, 17519, true).ok());
  for (const char* invalid : {"", "fast", "EXACT"})
    GEM16_CHECK(!SelectTwelveBD3Math(invalid, 3, fp8, 17519).ok());

  gem16::GreedyInferenceResult result;
  result.mtp_enabled = true;
  result.mtp_draft_tokens = 3;
  result.experimental_d3_fp8_attention = true;
  result.mtp_math_policy = automatic.value().policy;
  result.mtp_math_selection_reason = automatic.value().reason;
  std::ostringstream output;
  GEM16_CHECK(gem16::WriteGreedyInferenceJson(result, output).ok());
  auto json = gem16::json::Parse(output.str());
  GEM16_CHECK(json.ok());
  if (json.ok()) {
    const auto* mode = json.value().find("mtp_math_mode");
    const auto* quality = json.value().find("mtp_math_quality_state");
    GEM16_CHECK(mode && mode->as_string() == "optimized");
    GEM16_CHECK(quality && quality->as_string() == "initial_quality_screen_only");
  }
  result.experimental_d3_fp8_attention = false;
  result.mtp_math_policy = exact.value().policy;
  result.mtp_math_selection_reason = exact.value().reason;
  std::ostringstream reference;
  GEM16_CHECK(gem16::WriteGreedyInferenceJson(result, reference).ok());
  json = gem16::json::Parse(reference.str());
  GEM16_CHECK(json.ok());
  if (json.ok()) {
    const auto* mode = json.value().find("mtp_math_mode");
    const auto* reason = json.value().find("mtp_math_selection_reason");
    GEM16_CHECK(mode && mode->as_string() == "reference");
    GEM16_CHECK(reason && reason->as_string() == "explicit_reference");
  }
}

#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "gem16/status.h"

namespace gem16::layer {

// Correctness-only host implementation of Gemma 4 RMSNorm. Input contains
// `vectors` consecutive vectors of `width` elements. An empty weight span
// selects the scale-free V normalization used by Gemma 4 attention.
[[nodiscard]] Result<std::vector<float>> RmsNorm(
    std::span<const float> input,
    std::span<const float> weight,
    std::uint64_t vectors,
    std::uint64_t width,
    float epsilon);

// Applies Gemma's split-half rotary embedding in place. `rotary_dimensions`
// may be smaller than head_dimension for partial RoPE.
[[nodiscard]] Status ApplyRotaryEmbedding(
    std::span<float> states,
    std::uint64_t heads,
    std::uint64_t head_dimension,
    std::uint64_t rotary_dimensions,
    std::uint64_t position,
    double theta);

// Gemma 4 full attention uses proportional RoPE: the frequency denominator and
// split-half pairing retain the complete head dimension, while only the first
// `rotary_factor` fraction of pairs receives nonzero frequencies.
[[nodiscard]] Status ApplyProportionalRotaryEmbedding(
    std::span<float> states,
    std::uint64_t heads,
    std::uint64_t head_dimension,
    double rotary_factor,
    std::uint64_t position,
    double theta,
    double scaling_factor = 1.0);

// Batch-one decode attention over an already populated K/V cache. Layouts are
// Q [query_heads, head_dimension] and K/V [tokens, kv_heads, head_dimension].
// The result is concatenated query-head output [query_heads, head_dimension].
[[nodiscard]] Result<std::vector<float>> LocalAttentionDecode(
    std::span<const float> query,
    std::span<const float> key_cache,
    std::span<const float> value_cache,
    std::uint64_t query_heads,
    std::uint64_t kv_heads,
    std::uint64_t head_dimension,
    std::uint64_t tokens);

}  // namespace gem16::layer

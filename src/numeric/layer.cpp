#include "gem16/layer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>

namespace gem16::layer {
namespace {

Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

bool MultiplyFits(std::uint64_t left, std::uint64_t right, std::size_t& result) {
  if (left != 0U && right > std::numeric_limits<std::uint64_t>::max() / left) return false;
  const std::uint64_t product = left * right;
  if (product > std::numeric_limits<std::size_t>::max()) return false;
  result = static_cast<std::size_t>(product);
  return true;
}

}  // namespace

Result<std::vector<float>> RmsNorm(std::span<const float> input,
                                   std::span<const float> weight,
                                   std::uint64_t vectors,
                                   std::uint64_t width,
                                   float epsilon) {
  std::size_t elements = 0;
  if (vectors == 0U || width == 0U || !MultiplyFits(vectors, width, elements) ||
      input.size() != elements) {
    return Invalid("RMSNorm geometry does not match the input extent");
  }
  if (!weight.empty() && weight.size() != static_cast<std::size_t>(width)) {
    return Invalid("RMSNorm weight extent must equal the vector width");
  }
  if (!(epsilon > 0.0F) || !std::isfinite(epsilon)) {
    return Invalid("RMSNorm epsilon must be positive and finite");
  }

  std::vector<float> output(elements);
  for (std::uint64_t vector = 0; vector < vectors; ++vector) {
    const std::size_t base = static_cast<std::size_t>(vector * width);
    double squared_sum = 0.0;
    for (std::uint64_t index = 0; index < width; ++index) {
      const float value = input[base + static_cast<std::size_t>(index)];
      if (!std::isfinite(value)) return Invalid("RMSNorm input must be finite");
      squared_sum += static_cast<double>(value) * static_cast<double>(value);
    }
    const double inverse_rms =
        std::pow(squared_sum / static_cast<double>(width) + static_cast<double>(epsilon), -0.5);
    for (std::uint64_t index = 0; index < width; ++index) {
      const std::size_t offset = base + static_cast<std::size_t>(index);
      const double scale = weight.empty() ? 1.0 : static_cast<double>(weight[index]);
      if (!std::isfinite(scale)) return Invalid("RMSNorm weight must be finite");
      output[offset] = static_cast<float>(static_cast<double>(input[offset]) * inverse_rms * scale);
    }
  }
  return output;
}

Status ApplyRotaryEmbedding(std::span<float> states,
                            std::uint64_t heads,
                            std::uint64_t head_dimension,
                            std::uint64_t rotary_dimensions,
                            std::uint64_t position,
                            double theta) {
  std::size_t elements = 0;
  if (heads == 0U || head_dimension == 0U ||
      !MultiplyFits(heads, head_dimension, elements) || states.size() != elements) {
    return Invalid("RoPE geometry does not match the state extent");
  }
  if (rotary_dimensions == 0U || rotary_dimensions > head_dimension ||
      rotary_dimensions % 2U != 0U) {
    return Invalid("RoPE dimensions must be positive, even, and no larger than the head");
  }
  if (!(theta > 0.0) || !std::isfinite(theta)) {
    return Invalid("RoPE theta must be positive and finite");
  }

  const std::uint64_t half = rotary_dimensions / 2U;
  for (std::uint64_t head = 0; head < heads; ++head) {
    const std::size_t base = static_cast<std::size_t>(head * head_dimension);
    for (std::uint64_t index = 0; index < half; ++index) {
      const double exponent = (2.0 * static_cast<double>(index)) /
                              static_cast<double>(rotary_dimensions);
      const double angle = static_cast<double>(position) / std::pow(theta, exponent);
      const double cosine = std::cos(angle);
      const double sine = std::sin(angle);
      const std::size_t first = base + static_cast<std::size_t>(index);
      const std::size_t second = first + static_cast<std::size_t>(half);
      const double first_value = states[first];
      const double second_value = states[second];
      states[first] = static_cast<float>(first_value * cosine - second_value * sine);
      states[second] = static_cast<float>(second_value * cosine + first_value * sine);
    }
  }
  return Status::Ok();
}

Status ApplyProportionalRotaryEmbedding(std::span<float> states,
                                        std::uint64_t heads,
                                        std::uint64_t head_dimension,
                                        double rotary_factor,
                                        std::uint64_t position,
                                        double theta,
                                        double scaling_factor) {
  std::size_t elements = 0;
  if (heads == 0U || head_dimension == 0U || head_dimension % 2U != 0U ||
      !MultiplyFits(heads, head_dimension, elements) || states.size() != elements) {
    return Invalid("proportional RoPE geometry does not match the state extent");
  }
  if (!(rotary_factor > 0.0) || rotary_factor > 1.0 || !std::isfinite(rotary_factor) ||
      !(theta > 0.0) || !std::isfinite(theta) || !(scaling_factor > 0.0) ||
      !std::isfinite(scaling_factor)) {
    return Invalid("proportional RoPE factors and theta must be positive and finite");
  }
  const std::uint64_t half = head_dimension / 2U;
  const std::uint64_t rotating_pairs =
      static_cast<std::uint64_t>(rotary_factor * static_cast<double>(half));
  if (rotating_pairs == 0U || rotating_pairs > half) {
    return Invalid("proportional RoPE factor selects no valid pairs");
  }
  for (std::uint64_t head = 0; head < heads; ++head) {
    const std::size_t base = static_cast<std::size_t>(head * head_dimension);
    for (std::uint64_t index = 0; index < rotating_pairs; ++index) {
      const double exponent = (2.0 * static_cast<double>(index)) /
                              static_cast<double>(head_dimension);
      const double angle = static_cast<double>(position) /
                           (std::pow(theta, exponent) * scaling_factor);
      const double cosine = std::cos(angle);
      const double sine = std::sin(angle);
      const std::size_t first = base + static_cast<std::size_t>(index);
      const std::size_t second = first + static_cast<std::size_t>(half);
      const double first_value = states[first];
      const double second_value = states[second];
      states[first] = static_cast<float>(first_value * cosine - second_value * sine);
      states[second] = static_cast<float>(second_value * cosine + first_value * sine);
    }
  }
  return Status::Ok();
}

Result<std::vector<float>> LocalAttentionDecode(std::span<const float> query,
                                                std::span<const float> key_cache,
                                                std::span<const float> value_cache,
                                                std::uint64_t query_heads,
                                                std::uint64_t kv_heads,
                                                std::uint64_t head_dimension,
                                                std::uint64_t tokens) {
  std::size_t query_elements = 0;
  std::size_t token_elements = 0;
  std::size_t cache_elements = 0;
  if (query_heads == 0U || kv_heads == 0U || head_dimension == 0U || tokens == 0U ||
      query_heads % kv_heads != 0U ||
      !MultiplyFits(query_heads, head_dimension, query_elements) ||
      !MultiplyFits(kv_heads, head_dimension, token_elements) ||
      !MultiplyFits(tokens, token_elements, cache_elements) || query.size() != query_elements ||
      key_cache.size() != cache_elements || value_cache.size() != cache_elements) {
    return Invalid("local attention geometry or cache extent is invalid");
  }

  std::vector<float> output(query_elements, 0.0F);
  std::vector<double> scores(static_cast<std::size_t>(tokens));
  const std::uint64_t queries_per_kv = query_heads / kv_heads;
  for (std::uint64_t query_head = 0; query_head < query_heads; ++query_head) {
    const std::uint64_t kv_head = query_head / queries_per_kv;
    const std::size_t query_base = static_cast<std::size_t>(query_head * head_dimension);
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::uint64_t token = 0; token < tokens; ++token) {
      const std::size_t cache_base = static_cast<std::size_t>(
          token * token_elements + kv_head * head_dimension);
      double score = 0.0;
      for (std::uint64_t dimension = 0; dimension < head_dimension; ++dimension) {
        score += static_cast<double>(query[query_base + static_cast<std::size_t>(dimension)]) *
                 static_cast<double>(key_cache[cache_base + static_cast<std::size_t>(dimension)]);
      }
      scores[static_cast<std::size_t>(token)] = score;
      maximum = std::max(maximum, score);
    }

    double denominator = 0.0;
    for (double& score : scores) {
      score = std::exp(score - maximum);
      denominator += score;
    }
    if (!(denominator > 0.0) || !std::isfinite(denominator)) {
      return Invalid("local attention softmax normalization is not finite");
    }
    for (std::uint64_t dimension = 0; dimension < head_dimension; ++dimension) {
      double value = 0.0;
      for (std::uint64_t token = 0; token < tokens; ++token) {
        const std::size_t cache_base = static_cast<std::size_t>(
            token * token_elements + kv_head * head_dimension);
        value += scores[static_cast<std::size_t>(token)] / denominator *
                 static_cast<double>(value_cache[cache_base + static_cast<std::size_t>(dimension)]);
      }
      output[query_base + static_cast<std::size_t>(dimension)] = static_cast<float>(value);
    }
  }
  return output;
}

}  // namespace gem16::layer

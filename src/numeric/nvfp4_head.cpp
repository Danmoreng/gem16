#include "gem16/nvfp4_head.h"

#include "gem16/nvfp4.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace gem16::nvfp4 {
namespace {

Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

bool MultiplyFits(std::uint64_t first, std::uint64_t second,
                  std::uint64_t* result) {
  if (second != 0U && first > std::numeric_limits<std::uint64_t>::max() / second) {
    return false;
  }
  *result = first * second;
  return true;
}

constexpr std::size_t kMaximumSuppressedTokens = 16U;

bool IsPositiveFinite(float value) {
  return std::isfinite(value) && value > 0.0F;
}

std::uint8_t LoadNibble(std::span<const std::uint8_t> packed,
                        std::uint64_t index) {
  const auto byte = packed[static_cast<std::size_t>(index / 2U)];
  const unsigned shift = (index & 1U) == 0U ? 0U : 4U;
  return static_cast<std::uint8_t>((byte >> shift) & 0x0FU);
}

// This is the host equivalent of __float2bfloat16_rn for finite values. The
// result is converted back to float so callers can make the precision boundary
// explicit without retaining a second BF16 representation.
float Bf16Rne(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t rounded = bits +
      (0x7FFFU + ((bits >> 16U) & 1U));
  return std::bit_cast<float>(rounded & 0xFFFF0000U);
}

}  // namespace

Result<TiedNvfp4HeadView> TiedNvfp4HeadView::Create(
    std::span<const std::uint8_t> packed_weights,
    std::span<const std::uint8_t> local_scales,
    std::uint64_t vocabulary_size, std::uint64_t hidden_size,
    float weight_global_divisor, float input_global_divisor) {
  if (vocabulary_size == 0U || hidden_size == 0U) {
    return Invalid("NVFP4 tied head dimensions must be nonzero");
  }
  if (vocabulary_size > std::numeric_limits<std::uint32_t>::max()) {
    return Invalid("NVFP4 tied head vocabulary exceeds token ID range");
  }
  if (hidden_size % kBlockElements != 0U) {
    return Invalid("NVFP4 tied head hidden size must be divisible by 16");
  }
  if (vocabulary_size > std::numeric_limits<std::size_t>::max() ||
      hidden_size > std::numeric_limits<std::size_t>::max()) {
    return Invalid("NVFP4 tied head dimensions exceed the host address space");
  }
  if (!IsPositiveFinite(weight_global_divisor) ||
      !IsPositiveFinite(input_global_divisor)) {
    return Invalid("NVFP4 tied head divisors must be positive and finite");
  }

  std::uint64_t elements = 0U;
  std::uint64_t expected_packed = 0U;
  std::uint64_t expected_scales = 0U;
  if (!MultiplyFits(vocabulary_size, hidden_size, &elements) ||
      !MultiplyFits(vocabulary_size, hidden_size / kBlockElements,
                    &expected_scales)) {
    return Invalid("NVFP4 tied head dimensions overflow storage arithmetic");
  }
  expected_packed = elements / kPackedElementsPerByte;
  if (expected_packed > std::numeric_limits<std::size_t>::max() ||
      expected_scales > std::numeric_limits<std::size_t>::max() ||
      packed_weights.size() != static_cast<std::size_t>(expected_packed) ||
      local_scales.size() != static_cast<std::size_t>(expected_scales)) {
    return Invalid("NVFP4 tied head spans do not match matrix dimensions");
  }
  const std::uint64_t blocks_per_row = hidden_size / kBlockElements;
  const std::uint64_t packed_bytes_per_row = hidden_size / kPackedElementsPerByte;
  for (std::uint64_t row = 0U; row < vocabulary_size; ++row) {
    for (std::uint64_t block = 0U; block < blocks_per_row; ++block) {
      const std::uint8_t scale_bits = local_scales[static_cast<std::size_t>(
          row * blocks_per_row + block)];
      // The canonical artifact stores only positive E4M3FN scales. In
      // particular, negative zero (0x80) is not an alternate representation.
      if ((scale_bits & 0x80U) != 0U || !IsFiniteE4M3Fn(scale_bits)) {
        return Invalid("NVFP4 tied head local scales must be finite, positive-sign E4M3FN");
      }
      if (scale_bits == 0U) {
        const std::size_t packed_begin = static_cast<std::size_t>(
            row * packed_bytes_per_row + block * (kBlockElements / kPackedElementsPerByte));
        for (std::size_t index = 0U; index < kBlockElements / kPackedElementsPerByte;
             ++index) {
          if (packed_weights[packed_begin + index] != 0U) {
            return Invalid("NVFP4 tied head zero-scale blocks must have zero payload");
          }
        }
      }
    }
  }
  return TiedNvfp4HeadView(packed_weights, local_scales, vocabulary_size,
                           hidden_size, weight_global_divisor,
                           input_global_divisor);
}

Result<std::vector<float>> TiedNvfp4HeadView::Lookup(std::uint32_t token) const {
  if (static_cast<std::uint64_t>(token) >= vocabulary_size_) {
    return Invalid("NVFP4 tied head token is out of range");
  }

  const std::uint64_t row_offset = static_cast<std::uint64_t>(token) * hidden_size_;
  const std::uint64_t scale_offset =
      static_cast<std::uint64_t>(token) * (hidden_size_ / kBlockElements);
  const float embedding_scale =
      Bf16Rne(std::sqrt(static_cast<float>(hidden_size_)));
  std::vector<float> output(static_cast<std::size_t>(hidden_size_));
  for (std::uint64_t index = 0U; index < hidden_size_; ++index) {
    const std::uint8_t scale_bits = local_scales_[static_cast<std::size_t>(
        scale_offset + index / kBlockElements)];
    if ((scale_bits & 0x80U) != 0U || !IsFiniteE4M3Fn(scale_bits)) {
      return Invalid("NVFP4 tied head lookup encountered an invalid local scale");
    }
    const float decoded = DecodeE2M1(LoadNibble(packed_weights_, row_offset + index)) *
                          DecodeE4M3Fn(scale_bits) / weight_global_divisor_;
    if (!std::isfinite(decoded)) {
      return Status(StatusCode::kDataLoss,
                    "NVFP4 tied head lookup reconstruction is non-finite");
    }
    const float scaled = decoded * embedding_scale;
    if (!std::isfinite(scaled)) {
      return Status(StatusCode::kDataLoss,
                    "NVFP4 tied head embedding scaling produced a non-finite value");
    }
    const float rounded = Bf16Rne(scaled);
    if (!std::isfinite(rounded)) {
      return Status(StatusCode::kDataLoss,
                    "NVFP4 tied head embedding boundary is non-finite");
    }
    output[static_cast<std::size_t>(index)] = rounded;
  }
  return output;
}

Result<TiedNvfp4HeadView::ProjectionResult> TiedNvfp4HeadView::ProjectT1(
    std::span<const float> hidden, const ProjectionOptions& options) const {
  if (hidden.size() != static_cast<std::size_t>(hidden_size_)) {
    return Invalid("NVFP4 tied head hidden vector has the wrong extent");
  }
  for (const float value : hidden) {
    if (!std::isfinite(value)) {
      return Invalid("NVFP4 tied head hidden vector must be finite");
    }
  }
  if (!IsPositiveFinite(options.softcap)) {
    return Invalid("NVFP4 tied head softcap must be positive and finite");
  }

  if (options.suppressed_tokens.size() > kMaximumSuppressedTokens) {
    return Invalid("NVFP4 tied head supports at most 16 suppression tokens");
  }
  for (std::size_t index = 0U; index < options.suppressed_tokens.size(); ++index) {
    const std::uint32_t token = options.suppressed_tokens[index];
    if (static_cast<std::uint64_t>(token) >= vocabulary_size_) {
      return Invalid("NVFP4 tied head suppression token is out of range");
    }
    for (std::size_t previous = 0U; previous < index; ++previous) {
      if (options.suppressed_tokens[previous] == token) {
        return Invalid("NVFP4 tied head suppression tokens must be unique");
      }
    }
  }
  if (options.suppressed_tokens.size() >=
      static_cast<std::size_t>(vocabulary_size_)) {
    return Invalid("NVFP4 tied head cannot suppress every vocabulary token");
  }

  // The runtime's final normalization boundary is physically BF16. Keep this
  // temporary vector reference-only and do not retain a second head/input
  // representation in the view.
  std::vector<float> bf16_hidden(hidden.size());
  for (std::size_t index = 0U; index < hidden.size(); ++index) {
    bf16_hidden[index] = Bf16Rne(hidden[index]);
  }
  const auto activation = QuantizeActivation(bf16_hidden, input_global_divisor_);
  if (!activation.ok()) return activation.status();

  ProjectionResult result;
  if (options.return_diagnostic_logits) {
    result.diagnostic_logits.resize(static_cast<std::size_t>(vocabulary_size_));
  }
  float best_value = -std::numeric_limits<float>::infinity();
  std::uint32_t best_token = 0U;
  bool found = false;
  const std::uint64_t row_packed_bytes = hidden_size_ / kPackedElementsPerByte;
  const std::uint64_t row_scale_bytes = hidden_size_ / kBlockElements;
  for (std::uint64_t token = 0U; token < vocabulary_size_; ++token) {
    const auto packed_begin = packed_weights_.subspan(
        static_cast<std::size_t>(token * row_packed_bytes),
        static_cast<std::size_t>(row_packed_bytes));
    const auto scales_begin = local_scales_.subspan(
        static_cast<std::size_t>(token * row_scale_bytes),
        static_cast<std::size_t>(row_scale_bytes));
    const auto dot = ReferenceDotProduct(activation.value(), packed_begin,
                                         scales_begin, weight_global_divisor_);
    if (!dot.ok()) return dot.status();
    if (!std::isfinite(dot.value())) {
      return Status(StatusCode::kDataLoss,
                    "NVFP4 tied head projection reconstruction is non-finite");
    }
    const float softcapped = static_cast<float>(
        std::tanh(static_cast<float>(dot.value()) / options.softcap) *
        options.softcap);
    if (!std::isfinite(softcapped)) {
      return Status(StatusCode::kDataLoss,
                    "NVFP4 tied head softcap produced a non-finite logit");
    }
    if (options.return_diagnostic_logits) {
      result.diagnostic_logits[static_cast<std::size_t>(token)] = softcapped;
    }
    const auto token32 = static_cast<std::uint32_t>(token);
    bool is_suppressed = false;
    for (const std::uint32_t suppressed : options.suppressed_tokens) {
      if (suppressed == token32) {
        is_suppressed = true;
        break;
      }
    }
    if (is_suppressed) continue;
    if (!found || softcapped > best_value ||
        (softcapped == best_value && token32 < best_token)) {
      found = true;
      best_value = softcapped;
      best_token = token32;
    }
  }

  result.token = best_token;
  result.value = best_value;
  return result;
}

}  // namespace gem16::nvfp4

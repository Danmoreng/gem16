#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cuda/engine/gemma4_26b_trellis35_artifact.h"
#include "cuda/fp8/reference.h"
#include "cuda/moe/prefill.h"
#include "cuda/trellis35/reference.h"
#include "gem16/fp8.h"

namespace {

int failures = 0;

#define CHECK(expression)                                                     \
  do {                                                                        \
    if (!(expression)) {                                                      \
      std::cerr << __FILE__ << ':' << __LINE__ << ": check failed: "         \
                << #expression << '\n';                                       \
      ++failures;                                                             \
    }                                                                         \
  } while (false)

bool CudaOk(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) return true;
  std::cerr << operation << ": " << cudaGetErrorName(error) << ": "
            << cudaGetErrorString(error) << '\n';
  ++failures;
  return false;
}

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::uint64_t elements) : elements_(elements) {
    if (elements_ != 0U) {
      if (!CudaOk(cudaMalloc(&pointer_, elements_ * sizeof(T)), "cudaMalloc")) {
        pointer_ = nullptr;
      }
    }
  }
  ~DeviceBuffer() {
    if (pointer_ != nullptr) (void)cudaFree(pointer_);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() const { return pointer_; }
  std::uint64_t elements() const { return elements_; }
  std::uint64_t bytes() const { return elements_ * sizeof(T); }

 private:
  T* pointer_ = nullptr;
  std::uint64_t elements_ = 0U;
};

template <typename T>
bool Upload(DeviceBuffer<T>& destination, const std::vector<T>& source,
            const char* operation) {
  CHECK(destination.elements() == source.size());
  if (destination.elements() != source.size()) return false;
  return CudaOk(cudaMemcpy(destination.get(), source.data(),
                           destination.bytes(), cudaMemcpyHostToDevice),
                operation);
}

template <typename T, std::size_t N>
bool Upload(DeviceBuffer<T>& destination, const std::array<T, N>& source,
            const char* operation) {
  CHECK(destination.elements() == source.size());
  if (destination.elements() != source.size()) return false;
  return CudaOk(cudaMemcpy(destination.get(), source.data(),
                           destination.bytes(), cudaMemcpyHostToDevice),
                operation);
}

std::vector<std::byte> MakePayload(std::uint16_t rate,
                                   std::uint64_t input_elements,
                                   std::uint64_t output_elements,
                                   std::uint32_t seed) {
  const std::uint64_t tile_count =
      (input_elements / 16U) * (output_elements / 16U);
  std::vector<std::byte> payload(tile_count * 32U * rate);
  for (std::uint64_t tile = 0U; tile < tile_count; ++tile) {
    std::array<std::uint16_t, 64> words{};
    for (unsigned span = 0U; span < 16U; ++span) {
      std::uint64_t bitstream = 0U;
      for (unsigned index = 0U; index < 16U; ++index) {
        const unsigned position = span * 16U + index;
        const std::uint32_t mixed = static_cast<std::uint32_t>(
            tile * 0x9e3779b1ULL + position * 0x85ebca6bULL + seed);
        const unsigned branch =
            (mixed ^ (mixed >> 13U) ^ (mixed >> 23U)) & ((1U << rate) - 1U);
        bitstream = (bitstream << rate) | branch;
      }
      for (unsigned word = 0U; word < rate; ++word) {
        const unsigned shift = 16U * (rate - word - 1U);
        words[span * rate + word] =
            static_cast<std::uint16_t>((bitstream >> shift) & 0xffffU);
      }
    }
    std::byte* tile_output = payload.data() + tile * 32U * rate;
    for (unsigned word = 0U; word < 16U * rate; word += 2U) {
      const std::uint16_t swapped[2] = {words[word + 1U], words[word]};
      std::memcpy(tile_output + word * sizeof(std::uint16_t), swapped,
                  sizeof(swapped));
    }
  }
  return payload;
}

std::uint16_t HostPayloadWord(const std::vector<std::byte>& payload,
                              std::uint64_t tile_offset,
                              unsigned original_word) {
  std::uint16_t value = 0U;
  const std::uint64_t stored_word = original_word ^ 1U;
  std::memcpy(&value,
              payload.data() + tile_offset +
                  stored_word * sizeof(std::uint16_t),
              sizeof(value));
  return value;
}

unsigned HostTensorCoreIndex(unsigned row, unsigned column) {
  const unsigned thread = (column & 7U) * 4U + ((row & 7U) >> 1U);
  const unsigned offset = (column >= 8U ? 4U : 0U) +
                          (row >= 8U ? 2U : 0U) + (row & 1U);
  return thread * 8U + offset;
}

float HostHalf(std::uint16_t bits) {
  const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16U;
  const std::uint32_t exponent = (bits >> 10U) & 0x1fU;
  std::uint32_t fraction = bits & 0x3ffU;
  std::uint32_t value = 0U;
  if (exponent == 0U) {
    if (fraction == 0U) {
      value = sign;
    } else {
      int shift = 0;
      while ((fraction & 0x400U) == 0U) {
        fraction <<= 1U;
        ++shift;
      }
      fraction &= 0x3ffU;
      value = sign | static_cast<std::uint32_t>(113 - shift) << 23U |
              fraction << 13U;
    }
  } else if (exponent == 0x1fU) {
    value = sign | 0x7f800000U | fraction << 13U;
  } else {
    value = sign | (exponent + 112U) << 23U | fraction << 13U;
  }
  return std::bit_cast<float>(value);
}

std::uint16_t HostFloatToHalf(float input) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(input);
  const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16U) & 0x8000U);
  const std::uint32_t magnitude = bits & 0x7fffffffU;
  if (magnitude >= 0x7f800000U) {
    return static_cast<std::uint16_t>(sign | 0x7c00U |
                                      (magnitude > 0x7f800000U ? 0x0200U : 0U));
  }
  const int exponent = static_cast<int>((magnitude >> 23U) & 0xffU) - 127;
  const std::uint32_t significand = (magnitude & 0x7fffffU) | 0x800000U;
  if (exponent < -24) return sign;
  if (exponent < -14) {
    const unsigned shift = static_cast<unsigned>(-exponent - 1);
    std::uint32_t rounded = significand >> shift;
    const std::uint32_t remainder = significand & ((1U << shift) - 1U);
    const std::uint32_t halfway = 1U << (shift - 1U);
    if (remainder > halfway || (remainder == halfway && (rounded & 1U))) {
      ++rounded;
    }
    return static_cast<std::uint16_t>(sign | rounded);
  }
  if (exponent > 15) return static_cast<std::uint16_t>(sign | 0x7c00U);
  std::uint32_t rounded = significand >> 13U;
  const std::uint32_t remainder = significand & 0x1fffU;
  if (remainder > 0x1000U || (remainder == 0x1000U && (rounded & 1U))) {
    ++rounded;
  }
  if (rounded == 0x800U) {
    rounded = 0x400U;
    if (exponent == 15) return static_cast<std::uint16_t>(sign | 0x7c00U);
    return static_cast<std::uint16_t>(sign | (exponent + 16) << 10U);
  }
  return static_cast<std::uint16_t>(
      sign | static_cast<std::uint16_t>(exponent + 15) << 10U |
      static_cast<std::uint16_t>(rounded & 0x3ffU));
}

float HostDecodeMul1(std::uint16_t state) {
  const std::uint32_t value = static_cast<std::uint32_t>(state) * 0x83dcd12dU;
  const std::uint32_t sum =
      0x6400U + (value & 0xffU) + ((value >> 8U) & 0xffU) +
      ((value >> 16U) & 0xffU) + ((value >> 24U) & 0xffU);
  const double decoded =
      static_cast<double>(HostHalf(static_cast<std::uint16_t>(sum))) *
          static_cast<double>(HostHalf(0x1eeeU)) +
      static_cast<double>(HostHalf(0xc931U));
  return HostHalf(HostFloatToHalf(static_cast<float>(decoded)));
}

float HostDecodeWeight(const std::vector<std::byte>& payload,
                       std::uint16_t rate, std::uint64_t input_elements,
                       std::uint64_t output_elements, std::uint64_t input,
                       std::uint64_t output) {
  const std::uint64_t tile_columns = output_elements / 16U;
  const std::uint64_t tile =
      (input / 16U) * tile_columns + output / 16U;
  const std::uint64_t tile_offset = tile * 32U * rate;
  const unsigned position = HostTensorCoreIndex(
      static_cast<unsigned>(input & 15U),
      static_cast<unsigned>(output & 15U));
  const unsigned history_count = (16U + rate - 1U) / rate;
  std::uint32_t state = 0U;
  for (unsigned history = 0U; history < history_count; ++history) {
    const unsigned prior = (position + 256U - history) & 255U;
    const unsigned span = prior >> 4U;
    const unsigned element = prior & 15U;
    std::uint64_t bitstream = 0U;
    for (unsigned word = 0U; word < rate; ++word) {
      bitstream =
          (bitstream << 16U) |
          HostPayloadWord(payload, tile_offset, span * rate + word);
    }
    const unsigned branch = static_cast<unsigned>(
        (bitstream >> (rate * (15U - element))) & ((1U << rate) - 1U));
    state |= branch << (rate * history);
  }
  return HostDecodeMul1(static_cast<std::uint16_t>(state & 0xffffU));
}

std::vector<float> HostProjection(
    const std::vector<std::byte>& k3, const std::vector<std::byte>& k4,
    const std::vector<std::uint8_t>& activation,
    const std::vector<float>& scales,
    const std::array<std::uint32_t, gem16::internal::kTrellis35M1TopK>& ids,
    std::uint64_t input_elements, std::uint64_t output_elements) {
  std::vector<float> output(ids.size() * output_elements);
  for (unsigned slot = 0U; slot < ids.size(); ++slot) {
    const std::uint16_t rate = (ids[slot] & 1U) == 0U ? 3U : 4U;
    const std::vector<std::byte>& payload = rate == 3U ? k3 : k4;
    for (std::uint64_t column = 0U; column < output_elements; ++column) {
      float accumulator = 0.0F;
      for (std::uint64_t row = 0U; row < input_elements; ++row) {
        const float activation_value = gem16::fp8::DecodeE4M3Fn(
            activation[slot * input_elements + row]);
        const auto weight_bits = gem16::fp8::EncodeE4M3Fn(HostDecodeWeight(
            payload, rate, input_elements, output_elements, row, column));
        CHECK(weight_bits.ok());
        if (!weight_bits.ok()) return {};
        const float weight_value =
            gem16::fp8::DecodeE4M3Fn(weight_bits.value());
        accumulator = std::fma(activation_value, weight_value, accumulator);
      }
      output[slot * output_elements + column] = accumulator * scales[slot];
    }
  }
  return output;
}

struct FamilyStorage {
  DeviceBuffer<std::byte> k3;
  DeviceBuffer<std::byte> k4;
  DeviceBuffer<gem16::internal::Trellis35ExpertDescriptor> descriptors;
  DeviceBuffer<std::uint16_t> suh;
  DeviceBuffer<std::uint16_t> svh;
  gem16::internal::Trellis35DeviceFamilyBinding binding;

  FamilyStorage(std::uint64_t input_elements, std::uint64_t output_elements,
                std::uint32_t seed)
      : k3(MakePayload(3U, input_elements, output_elements, seed).size()),
        k4(MakePayload(4U, input_elements, output_elements, seed + 1U).size()),
        descriptors(gem16::internal::kTrellis35ExpertCount),
        suh(gem16::internal::kTrellis35ExpertCount * input_elements),
        svh(gem16::internal::kTrellis35ExpertCount * output_elements) {
    const std::vector<std::byte> host_k3 =
        MakePayload(3U, input_elements, output_elements, seed);
    const std::vector<std::byte> host_k4 =
        MakePayload(4U, input_elements, output_elements, seed + 1U);
    std::vector<gem16::internal::Trellis35ExpertDescriptor> host_descriptors(
        gem16::internal::kTrellis35ExpertCount);
    std::vector<std::uint16_t> host_suh(suh.elements(), 0x3c00U);
    std::vector<std::uint16_t> host_svh(svh.elements(), 0x3c00U);
    for (std::uint32_t expert = 0U;
         expert < gem16::internal::kTrellis35ExpertCount; ++expert) {
      const std::uint16_t rate = (expert & 1U) == 0U ? 3U : 4U;
      host_descriptors[expert] = {0U, rate, 2U};
      binding.rate_map[expert] = rate;
    }
    if (!Upload(k3, host_k3, "upload K3 payload") ||
        !Upload(k4, host_k4, "upload K4 payload") ||
        !Upload(descriptors, host_descriptors, "upload descriptors") ||
        !Upload(suh, host_suh, "upload SUH") ||
        !Upload(svh, host_svh, "upload SVH")) {
      return;
    }
    binding.k3_payload_pool = k3.get();
    binding.k4_payload_pool = k4.get();
    binding.descriptors = descriptors.get();
    binding.suh_f16 = suh.get();
    binding.svh_f16 = svh.get();
  }
};

struct M1Storage {
  DeviceBuffer<float> gate_input{
      gem16::internal::kTrellis35M1TopK *
      gem16::internal::kTrellis35GateUpInput};
  DeviceBuffer<std::uint8_t> gate_input_fp8{
      gem16::internal::kTrellis35M1TopK *
      gem16::internal::kTrellis35GateUpInput};
  DeviceBuffer<float> gate_scales{gem16::internal::kTrellis35M1TopK};
  DeviceBuffer<float> gate_transformed{
      gem16::internal::kTrellis35M1TopK *
      gem16::internal::kTrellis35GateUpOutput};
  DeviceBuffer<float> gate_output{
      gem16::internal::kTrellis35M1TopK *
      gem16::internal::kTrellis35GateUpOutput};
  DeviceBuffer<float> product{
      gem16::internal::kTrellis35M1TopK *
      gem16::internal::kTrellis35ExpertIntermediate};
  DeviceBuffer<float> down_input{
      gem16::internal::kTrellis35M1TopK *
      gem16::internal::kTrellis35DownInput};
  DeviceBuffer<std::uint8_t> down_input_fp8{
      gem16::internal::kTrellis35M1TopK *
      gem16::internal::kTrellis35DownInput};
  DeviceBuffer<float> down_scales{gem16::internal::kTrellis35M1TopK};
  DeviceBuffer<float> down_transformed{
      gem16::internal::kTrellis35M1TopK *
      gem16::internal::kTrellis35DownOutput};
  DeviceBuffer<float> down_output{
      gem16::internal::kTrellis35M1TopK *
      gem16::internal::kTrellis35DownOutput};

  gem16::internal::Trellis35M1Workspace Bind() {
    return {gate_input.get(),       gate_input_fp8.get(), gate_scales.get(),
            gate_transformed.get(), gate_output.get(),    product.get(),
            down_input.get(),       down_input_fp8.get(), down_scales.get(),
            down_transformed.get(), down_output.get()};
  }
};

struct T3Storage {
  DeviceBuffer<float> gate_input{
      gem16::internal::kTrellis35T3Assignments *
      gem16::internal::kTrellis35GateUpInput};
  DeviceBuffer<std::uint8_t> gate_input_fp8{
      gem16::internal::kTrellis35T3Assignments *
      gem16::internal::kTrellis35GateUpInput};
  DeviceBuffer<float> gate_scales{gem16::internal::kTrellis35T3Assignments};
  DeviceBuffer<float> gate_transformed{
      gem16::internal::kTrellis35T3Assignments *
      gem16::internal::kTrellis35GateUpOutput};
  DeviceBuffer<float> gate_output{
      gem16::internal::kTrellis35T3Assignments *
      gem16::internal::kTrellis35GateUpOutput};
  DeviceBuffer<float> product{
      gem16::internal::kTrellis35T3Assignments *
      gem16::internal::kTrellis35ExpertIntermediate};
  DeviceBuffer<float> down_input{
      gem16::internal::kTrellis35T3Assignments *
      gem16::internal::kTrellis35DownInput};
  DeviceBuffer<std::uint8_t> down_input_fp8{
      gem16::internal::kTrellis35T3Assignments *
      gem16::internal::kTrellis35DownInput};
  DeviceBuffer<float> down_scales{gem16::internal::kTrellis35T3Assignments};
  DeviceBuffer<float> down_transformed{
      gem16::internal::kTrellis35T3Assignments *
      gem16::internal::kTrellis35DownOutput};
  DeviceBuffer<float> down_output{
      gem16::internal::kTrellis35T3Assignments *
      gem16::internal::kTrellis35DownOutput};

  gem16::internal::Trellis35T3Workspace Bind() {
    return {gate_input.get(),       gate_input_fp8.get(), gate_scales.get(),
            gate_transformed.get(), gate_output.get(),    product.get(),
            down_input.get(),       down_input_fp8.get(), down_scales.get(),
            down_transformed.get(), down_output.get()};
  }
};

struct PrefillStorage {
  explicit PrefillStorage(std::uint64_t token_count)
      : tokens(token_count),
        token_hidden(tokens * gem16::internal::kTrellis35DownOutput),
        token_packed(tokens *
                     (gem16::internal::kTrellis35GateUpInput / 2U +
                      gem16::internal::kTrellis35GateUpInput / 16U)),
        expert_product(tokens * gem16::internal::kTrellis35M1TopK *
                       gem16::internal::kTrellis35ExpertIntermediate),
        expert_down(tokens * gem16::internal::kTrellis35M1TopK *
                    gem16::internal::kTrellis35DownOutput),
        shared_product(tokens * 2112U),
        shared_output(tokens * gem16::internal::kTrellis35DownOutput),
        schedule(tokens * gem16::internal::kTrellis35M1TopK),
        assignments(tokens * gem16::internal::kTrellis35M1TopK),
        histogram(gem16::internal::kTrellis35ExpertCount),
        prefix(gem16::internal::kTrellis35ExpertCount + 1U),
        permutation(tokens * gem16::internal::kTrellis35M1TopK),
        inverse(tokens * gem16::internal::kTrellis35M1TopK) {}

  gem16::internal::Gemma4MoePrefillWorkspace Bind() {
    gem16::internal::Gemma4MoePrefillWorkspace workspace;
    workspace.token_hidden = token_hidden.get();
    workspace.token_packed = token_packed.get();
    workspace.token_scales =
        token_packed.get() +
        tokens * (gem16::internal::kTrellis35GateUpInput / 2U);
    workspace.expert_product = expert_product.get();
    workspace.expert_down = expert_down.get();
    workspace.shared_product = shared_product.get();
    workspace.shared_output = shared_output.get();
    workspace.router_logits = reinterpret_cast<float*>(schedule.get());
    workspace.assignments = assignments.get();
    workspace.histogram = histogram.get();
    workspace.prefix = prefix.get();
    workspace.permutation = permutation.get();
    workspace.inverse_permutation = inverse.get();
    return workspace;
  }

  std::uint64_t tokens;
  DeviceBuffer<float> token_hidden;
  DeviceBuffer<std::uint8_t> token_packed;
  DeviceBuffer<float> expert_product;
  DeviceBuffer<float> expert_down;
  DeviceBuffer<float> shared_product;
  DeviceBuffer<float> shared_output;
  DeviceBuffer<std::uint32_t> schedule;
  DeviceBuffer<gem16::internal::Gemma4MoePrefillAssignment> assignments;
  DeviceBuffer<std::uint32_t> histogram;
  DeviceBuffer<std::uint32_t> prefix;
  DeviceBuffer<std::uint32_t> permutation;
  DeviceBuffer<std::uint32_t> inverse;
};

void Compare(const std::vector<float>& expected,
             const std::vector<float>& actual, float absolute_tolerance,
             float relative_tolerance, const char* description) {
  CHECK(expected.size() == actual.size());
  float maximum_absolute = 0.0F;
  float maximum_relative = 0.0F;
  std::uint64_t mismatches = 0U;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const float difference = std::fabs(expected[index] - actual[index]);
    const float relative =
        difference / std::max(1.0F, std::fabs(expected[index]));
    maximum_absolute = std::max(maximum_absolute, difference);
    maximum_relative = std::max(maximum_relative, relative);
    if (difference > absolute_tolerance && relative > relative_tolerance) {
      ++mismatches;
    }
  }
  std::cout << description << " max_abs=" << maximum_absolute
            << " max_rel=" << maximum_relative
            << " mismatches=" << mismatches << '\n';
  CHECK(mismatches == 0U);
}

void TestRandomizedProjectionParity() {
  constexpr std::uint64_t kInput = 128U;
  constexpr std::uint64_t kOutput = 128U;
  FamilyStorage family(kInput, kOutput, 19U);
  DeviceBuffer<std::uint32_t> ids(gem16::internal::kTrellis35M1TopK);
  DeviceBuffer<std::uint8_t> activation(
      gem16::internal::kTrellis35M1TopK * kInput);
  DeviceBuffer<float> scales(gem16::internal::kTrellis35M1TopK);
  DeviceBuffer<float> native_output(
      gem16::internal::kTrellis35M1TopK * kOutput);
  DeviceBuffer<float> reference_output(
      gem16::internal::kTrellis35M1TopK * kOutput);
  std::array<std::uint32_t, gem16::internal::kTrellis35M1TopK> host_ids{};
  std::vector<std::uint8_t> host_activation(activation.elements());
  std::vector<float> host_scales(scales.elements());
  for (std::uint32_t slot = 0U; slot < gem16::internal::kTrellis35M1TopK;
       ++slot) {
    host_ids[slot] = slot;
    std::array<float, kInput> token{};
    for (std::uint64_t index = 0U; index < kInput; ++index) {
      token[index] = std::sin(static_cast<float>(index * 17U + slot * 31U) *
                              0.03125F) *
                     0.125F;
    }
    const auto quantized = gem16::fp8::QuantizeToken(token);
    CHECK(quantized.ok());
    if (!quantized.ok()) return;
    host_scales[slot] = quantized.value().scale;
    std::copy(quantized.value().values_e4m3fn.begin(),
              quantized.value().values_e4m3fn.end(),
              host_activation.begin() + slot * kInput);
  }
  if (!Upload(ids, host_ids, "upload parity IDs") ||
      !Upload(activation, host_activation, "upload parity activation") ||
      !Upload(scales, host_scales, "upload parity scales")) {
    return;
  }
  const auto native = gem16::internal::LaunchTrellis35MmaW4A8ProjectionM1(
      activation.get(), scales.get(), family.binding, ids.get(),
      native_output.get(), kInput, kOutput, nullptr);
  CHECK(native.ok());
  for (std::uint32_t slot = 0U; slot < gem16::internal::kTrellis35M1TopK;
       ++slot) {
    const auto reference =
        gem16::internal::LaunchTrellis35ReferenceW4A8ProjectionM1(
            activation.get() + slot * kInput, scales.get() + slot,
            family.binding, host_ids[slot],
            reference_output.get() + slot * kOutput, kInput, kOutput,
            nullptr);
    CHECK(reference.ok());
  }
  CHECK(CudaOk(cudaDeviceSynchronize(), "synchronize projection parity"));
  std::vector<float> host_native(native_output.elements());
  std::vector<float> host_reference(reference_output.elements());
  CHECK(CudaOk(cudaMemcpy(host_native.data(), native_output.get(),
                          native_output.bytes(), cudaMemcpyDeviceToHost),
               "download native projection"));
  CHECK(CudaOk(cudaMemcpy(host_reference.data(), reference_output.get(),
                          reference_output.bytes(), cudaMemcpyDeviceToHost),
               "download reference projection"));
  Compare(host_reference, host_native, 2.0e-4F, 2.0e-5F,
          "randomized mixed-K3/K4 projection parity");
  const std::vector<float> host_oracle = HostProjection(
      MakePayload(3U, kInput, kOutput, 19U),
      MakePayload(4U, kInput, kOutput, 20U), host_activation, host_scales,
      host_ids, kInput, kOutput);
  Compare(host_oracle, host_native, 2.0e-4F, 2.0e-5F,
          "independent CPU mixed-K3/K4 projection parity");
}

void TestTransformsAndDownPadding() {
  constexpr std::uint64_t kLogical =
      gem16::internal::kTrellis35ExpertIntermediate;
  constexpr std::uint64_t kPhysical = gem16::internal::kTrellis35DownInput;
  DeviceBuffer<float> input(kLogical);
  DeviceBuffer<std::uint16_t> sidecar(kPhysical);
  DeviceBuffer<float> transformed(kPhysical);
  DeviceBuffer<float> reconstructed(kPhysical);
  std::vector<float> host_input(kLogical);
  std::vector<std::uint16_t> host_sidecar(kPhysical, 0x3c00U);
  for (std::uint64_t index = 0U; index < kLogical; ++index) {
    host_input[index] =
        std::sin(static_cast<float>(index * 29U + 3U) * 0.015625F);
  }
  if (!Upload(input, host_input, "upload transform input") ||
      !Upload(sidecar, host_sidecar, "upload transform sidecar")) {
    return;
  }
  auto status = gem16::internal::LaunchTrellis35InputTransformM1(
      input.get(), sidecar.get(), transformed.get(), kLogical, kPhysical,
      nullptr);
  CHECK(status.ok());
  status = gem16::internal::LaunchTrellis35OutputTransformM1(
      transformed.get(), sidecar.get(), reconstructed.get(), kPhysical,
      nullptr);
  CHECK(status.ok());
  CHECK(CudaOk(cudaDeviceSynchronize(), "synchronize transform round trip"));
  std::vector<float> host_reconstructed(kPhysical);
  CHECK(CudaOk(cudaMemcpy(host_reconstructed.data(), reconstructed.get(),
                          reconstructed.bytes(), cudaMemcpyDeviceToHost),
               "download transform reconstruction"));
  float maximum_logical_error = 0.0F;
  float maximum_padding = 0.0F;
  for (std::uint64_t index = 0U; index < kLogical; ++index) {
    maximum_logical_error = std::max(
        maximum_logical_error,
        std::fabs(host_reconstructed[index] - host_input[index]));
  }
  for (std::uint64_t index = kLogical; index < kPhysical; ++index) {
    maximum_padding =
        std::max(maximum_padding, std::fabs(host_reconstructed[index]));
  }
  std::cout << "Down 704->768 transform roundtrip max_abs="
            << maximum_logical_error << " padding_max_abs=" << maximum_padding
            << '\n';
  CHECK(maximum_logical_error <= 2.0e-6F);
  CHECK(maximum_padding <= 2.0e-6F);

  constexpr std::uint64_t kFused = gem16::internal::kTrellis35GateUpOutput;
  DeviceBuffer<float> fused_transformed(kFused);
  DeviceBuffer<std::uint16_t> fused_sidecar(kFused);
  DeviceBuffer<float> fused_output(kFused);
  std::vector<float> host_fused(kFused, 0.0F);
  std::vector<std::uint16_t> host_fused_sidecar(kFused, 0x3c00U);
  // Index 750 shares the [640, 768) Hadamard block with both the final Gate
  // values [640, 704) and the first Up values [704, 768).
  host_fused[750] = 1.0F;
  if (!Upload(fused_transformed, host_fused, "upload fused impulse") ||
      !Upload(fused_sidecar, host_fused_sidecar,
              "upload fused output sidecar")) {
    return;
  }
  status = gem16::internal::LaunchTrellis35OutputTransformM1(
      fused_transformed.get(), fused_sidecar.get(), fused_output.get(),
      kFused, nullptr);
  CHECK(status.ok());
  CHECK(CudaOk(cudaDeviceSynchronize(), "synchronize fused output transform"));
  std::vector<float> host_fused_output(kFused);
  CHECK(CudaOk(cudaMemcpy(host_fused_output.data(), fused_output.get(),
                          fused_output.bytes(), cudaMemcpyDeviceToHost),
               "download fused output transform"));
  float gate_energy = 0.0F;
  float up_energy = 0.0F;
  for (std::uint64_t index = 640U; index < 704U; ++index) {
    gate_energy += std::fabs(host_fused_output[index]);
  }
  for (std::uint64_t index = 704U; index < 768U; ++index) {
    up_energy += std::fabs(host_fused_output[index]);
  }
  std::cout << "fused Gate+Up crossing-block gate_l1=" << gate_energy
            << " up_l1=" << up_energy << '\n';
  CHECK(gate_energy > 0.0F && up_energy > 0.0F);
}

float RunFullM1(const gem16::internal::Trellis35DeviceLayerBinding& layer,
                DeviceBuffer<float>& input, DeviceBuffer<std::uint32_t>& ids,
                DeviceBuffer<float>& weights, M1Storage& storage,
                DeviceBuffer<float>& output, unsigned iterations,
                bool capture) {
  const auto workspace = storage.Bind();
  if (capture) {
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    CHECK(CudaOk(cudaStreamCreate(&stream), "create graph stream"));
    CHECK(CudaOk(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                 "begin Trellis35 capture"));
    const auto status = gem16::internal::LaunchTrellis35SelectedExpertsM1(
        input.get(), ids.get(), weights.get(), layer, workspace, output.get(),
        stream);
    CHECK(status.ok());
    CHECK(CudaOk(cudaStreamEndCapture(stream, &graph),
                 "end Trellis35 capture"));
    CHECK(CudaOk(cudaGraphInstantiate(&executable, graph, 0U),
                 "instantiate Trellis35 graph"));
    CHECK(CudaOk(cudaGraphLaunch(executable, stream),
                 "launch Trellis35 graph first"));
    CHECK(CudaOk(cudaGraphLaunch(executable, stream),
                 "launch Trellis35 graph replay"));
    CHECK(CudaOk(cudaStreamSynchronize(stream),
                 "synchronize Trellis35 graph"));
    if (executable != nullptr) (void)cudaGraphExecDestroy(executable);
    if (graph != nullptr) (void)cudaGraphDestroy(graph);
    if (stream != nullptr) (void)cudaStreamDestroy(stream);
  }

  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  CHECK(CudaOk(cudaEventCreate(&begin), "create begin event"));
  CHECK(CudaOk(cudaEventCreate(&end), "create end event"));
  CHECK(CudaOk(cudaEventRecord(begin), "record begin event"));
  for (unsigned iteration = 0U; iteration < iterations; ++iteration) {
    const auto status = gem16::internal::LaunchTrellis35SelectedExpertsM1(
        input.get(), ids.get(), weights.get(), layer, workspace, output.get(),
        nullptr);
    CHECK(status.ok());
  }
  CHECK(CudaOk(cudaEventRecord(end), "record end event"));
  CHECK(CudaOk(cudaEventSynchronize(end), "synchronize end event"));
  float milliseconds = 0.0F;
  CHECK(CudaOk(cudaEventElapsedTime(&milliseconds, begin, end),
               "measure Trellis35 M1"));
  if (begin != nullptr) (void)cudaEventDestroy(begin);
  if (end != nullptr) (void)cudaEventDestroy(end);
  return milliseconds / static_cast<float>(iterations);
}

float RunFullT3(const gem16::internal::Trellis35DeviceLayerBinding& layer,
                DeviceBuffer<float>& input, DeviceBuffer<std::uint32_t>& ids,
                DeviceBuffer<float>& weights, T3Storage& storage,
                DeviceBuffer<float>& output, unsigned iterations,
                bool capture) {
  const auto workspace = storage.Bind();
  if (capture) {
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    CHECK(CudaOk(cudaStreamCreate(&stream), "create T3 graph stream"));
    CHECK(CudaOk(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                 "begin Trellis35 T3 capture"));
    const auto status = gem16::internal::LaunchTrellis35SelectedExpertsT3(
        input.get(), ids.get(), weights.get(), layer, workspace, output.get(),
        stream);
    CHECK(status.ok());
    CHECK(CudaOk(cudaStreamEndCapture(stream, &graph),
                 "end Trellis35 T3 capture"));
    CHECK(CudaOk(cudaGraphInstantiate(&executable, graph, 0U),
                 "instantiate Trellis35 T3 graph"));
    CHECK(CudaOk(cudaGraphLaunch(executable, stream),
                 "launch Trellis35 T3 graph first"));
    CHECK(CudaOk(cudaGraphLaunch(executable, stream),
                 "launch Trellis35 T3 graph replay"));
    CHECK(CudaOk(cudaStreamSynchronize(stream),
                 "synchronize Trellis35 T3 graph"));
    if (executable != nullptr) (void)cudaGraphExecDestroy(executable);
    if (graph != nullptr) (void)cudaGraphDestroy(graph);
    if (stream != nullptr) (void)cudaStreamDestroy(stream);
  }

  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  CHECK(CudaOk(cudaEventCreate(&begin), "create T3 begin event"));
  CHECK(CudaOk(cudaEventCreate(&end), "create T3 end event"));
  CHECK(CudaOk(cudaEventRecord(begin), "record T3 begin event"));
  for (unsigned iteration = 0U; iteration < iterations; ++iteration) {
    const auto status = gem16::internal::LaunchTrellis35SelectedExpertsT3(
        input.get(), ids.get(), weights.get(), layer, workspace, output.get(),
        nullptr);
    CHECK(status.ok());
  }
  CHECK(CudaOk(cudaEventRecord(end), "record T3 end event"));
  CHECK(CudaOk(cudaEventSynchronize(end), "synchronize T3 end event"));
  float milliseconds = 0.0F;
  CHECK(CudaOk(cudaEventElapsedTime(&milliseconds, begin, end),
               "measure Trellis35 T3"));
  if (begin != nullptr) (void)cudaEventDestroy(begin);
  if (end != nullptr) (void)cudaEventDestroy(end);
  return milliseconds / static_cast<float>(iterations);
}

unsigned UniqueExperts(
    const std::array<std::uint32_t,
                     gem16::internal::kTrellis35T3Assignments>& ids) {
  std::array<bool, gem16::internal::kTrellis35ExpertCount> seen{};
  unsigned unique = 0U;
  for (const std::uint32_t expert : ids) {
    if (!seen[expert]) {
      seen[expert] = true;
      ++unique;
    }
  }
  return unique;
}

std::array<std::uint32_t, gem16::internal::kTrellis35T3Assignments>
MakeT3RoutesWithUnionSize(unsigned unique_experts) {
  CHECK(unique_experts >= gem16::internal::kTrellis35M1TopK);
  CHECK(unique_experts <= gem16::internal::kTrellis35T3Assignments);
  std::array<std::uint32_t, gem16::internal::kTrellis35T3Assignments> ids{};
  for (unsigned slot = 0U; slot < gem16::internal::kTrellis35M1TopK;
       ++slot) {
    ids[slot] = slot;
  }
  unsigned next_new = gem16::internal::kTrellis35M1TopK;
  for (unsigned row = 1U; row < gem16::internal::kTrellis35T3Rows; ++row) {
    const unsigned new_in_row =
        std::min<unsigned>(gem16::internal::kTrellis35M1TopK,
                           unique_experts - next_new);
    for (unsigned slot = 0U; slot < new_in_row; ++slot) {
      ids[row * gem16::internal::kTrellis35M1TopK + slot] = next_new++;
    }
    for (unsigned slot = new_in_row;
         slot < gem16::internal::kTrellis35M1TopK; ++slot) {
      ids[row * gem16::internal::kTrellis35M1TopK + slot] =
          slot - new_in_row;
    }
  }
  CHECK(next_new == unique_experts);
  CHECK(UniqueExperts(ids) == unique_experts);
  return ids;
}

std::array<std::uint32_t, gem16::internal::kTrellis35T3Assignments>
RemapT3Routes(
    const std::array<std::uint32_t,
                     gem16::internal::kTrellis35T3Assignments>& routes,
    const std::vector<std::uint32_t>& expert_order) {
  std::array<std::uint32_t, gem16::internal::kTrellis35T3Assignments> result{};
  for (unsigned assignment = 0U; assignment < routes.size(); ++assignment) {
    CHECK(routes[assignment] < expert_order.size());
    result[assignment] = expert_order[routes[assignment]];
  }
  return result;
}

void CheckT3SlotOrderedReduction(
    const DeviceBuffer<float>& expert_output,
    const std::array<float, gem16::internal::kTrellis35T3Assignments>& weights,
    const DeviceBuffer<float>& reduced, const char* description) {
  std::vector<float> host_experts(expert_output.elements());
  std::vector<float> host_reduced(reduced.elements());
  CHECK(CudaOk(cudaMemcpy(host_experts.data(), expert_output.get(),
                          expert_output.bytes(), cudaMemcpyDeviceToHost),
               "download T3 expert slots"));
  CHECK(CudaOk(cudaMemcpy(host_reduced.data(), reduced.get(), reduced.bytes(),
                          cudaMemcpyDeviceToHost),
               "download T3 reduced experts"));
  std::uint64_t mismatches = 0U;
  for (unsigned row = 0U; row < gem16::internal::kTrellis35T3Rows; ++row) {
    for (std::uint64_t index = 0U;
         index < gem16::internal::kTrellis35DownOutput; ++index) {
      float expected = 0.0F;
      for (unsigned slot = 0U; slot < gem16::internal::kTrellis35M1TopK;
           ++slot) {
        const unsigned assignment =
            row * gem16::internal::kTrellis35M1TopK + slot;
        expected = std::fma(
            weights[assignment],
            host_experts[static_cast<std::uint64_t>(assignment) *
                             gem16::internal::kTrellis35DownOutput +
                         index],
            expected);
      }
      const float actual =
          host_reduced[static_cast<std::uint64_t>(row) *
                           gem16::internal::kTrellis35DownOutput +
                       index];
      if (std::bit_cast<std::uint32_t>(expected) !=
          std::bit_cast<std::uint32_t>(actual)) {
        ++mismatches;
      }
    }
  }
  std::cout << description << " T3_slot_order_bit_mismatches=" << mismatches
            << '\n';
  CHECK(mismatches == 0U);
}

float RunT3Scenario(
    const gem16::internal::Trellis35DeviceLayerBinding& layer,
    DeviceBuffer<float>& input,
    const std::array<std::uint32_t,
                     gem16::internal::kTrellis35T3Assignments>& host_ids,
    const std::array<float, gem16::internal::kTrellis35T3Assignments>&
        host_weights,
    const char* description, bool capture) {
  DeviceBuffer<std::uint32_t> ids(gem16::internal::kTrellis35T3Assignments);
  DeviceBuffer<float> weights(gem16::internal::kTrellis35T3Assignments);
  DeviceBuffer<float> output(gem16::internal::kTrellis35T3Rows *
                             gem16::internal::kTrellis35DownOutput);
  DeviceBuffer<float> reference(gem16::internal::kTrellis35T3Rows *
                                gem16::internal::kTrellis35DownOutput);
  T3Storage storage;
  M1Storage reference_storage;
  if (!Upload(ids, host_ids, "upload T3 IDs") ||
      !Upload(weights, host_weights, "upload T3 weights")) {
    return 0.0F;
  }
  const float latency = RunFullT3(layer, input, ids, weights, storage, output,
                                  5U, capture);
  const auto m1_workspace = reference_storage.Bind();
  for (unsigned row = 0U; row < gem16::internal::kTrellis35T3Rows; ++row) {
    const auto status = gem16::internal::LaunchTrellis35SelectedExpertsM1(
        input.get() + static_cast<std::uint64_t>(row) *
                          gem16::internal::kTrellis35GateUpInput,
        ids.get() + row * gem16::internal::kTrellis35M1TopK,
        weights.get() + row * gem16::internal::kTrellis35M1TopK, layer,
        m1_workspace,
        reference.get() + static_cast<std::uint64_t>(row) *
                              gem16::internal::kTrellis35DownOutput,
        nullptr);
    CHECK(status.ok());
  }
  CHECK(CudaOk(cudaDeviceSynchronize(), "synchronize T3 versus M1"));
  std::vector<float> host_output(output.elements());
  std::vector<float> host_reference(reference.elements());
  std::array<std::uint32_t, gem16::internal::kTrellis35T3Assignments>
      unchanged_ids{};
  CHECK(CudaOk(cudaMemcpy(host_output.data(), output.get(), output.bytes(),
                          cudaMemcpyDeviceToHost),
               "download T3 output"));
  CHECK(CudaOk(cudaMemcpy(host_reference.data(), reference.get(),
                          reference.bytes(), cudaMemcpyDeviceToHost),
               "download three-M1 oracle"));
  CHECK(CudaOk(cudaMemcpy(unchanged_ids.data(), ids.get(), ids.bytes(),
                          cudaMemcpyDeviceToHost),
               "download unchanged T3 IDs"));
  CHECK(unchanged_ids == host_ids);
  CHECK(std::all_of(host_output.begin(), host_output.end(),
                    [](float value) { return std::isfinite(value); }));
  Compare(host_reference, host_output, 4.0e-6F, 2.0e-6F, description);
  CheckT3SlotOrderedReduction(storage.down_output, host_weights, output,
                              description);
  std::cout << description << " unique_experts=" << UniqueExperts(host_ids)
            << " latency_ms=" << latency << '\n';
  return latency;
}

void CheckFiniteAndIds(const DeviceBuffer<float>& output,
                       const DeviceBuffer<std::uint32_t>& ids,
                       const std::array<std::uint32_t,
                                        gem16::internal::kTrellis35M1TopK>&
                           expected_ids,
                       const char* description) {
  std::vector<float> host_output(output.elements());
  std::array<std::uint32_t, gem16::internal::kTrellis35M1TopK> host_ids{};
  CHECK(CudaOk(cudaMemcpy(host_output.data(), output.get(), output.bytes(),
                          cudaMemcpyDeviceToHost),
               "download M1 output"));
  CHECK(CudaOk(cudaMemcpy(host_ids.data(), ids.get(), ids.bytes(),
                          cudaMemcpyDeviceToHost),
               "download route IDs"));
  CHECK(host_ids == expected_ids);
  CHECK(std::all_of(host_output.begin(), host_output.end(),
                    [](float value) { return std::isfinite(value); }));
  const auto [minimum, maximum] =
      std::minmax_element(host_output.begin(), host_output.end());
  std::cout << description << " finite=true min=" << *minimum
            << " max=" << *maximum << '\n';
}

void CheckSlotOrderedReduction(
    const DeviceBuffer<float>& expert_output,
    const std::array<float, gem16::internal::kTrellis35M1TopK>& weights,
    const DeviceBuffer<float>& reduced, const char* description) {
  std::vector<float> host_experts(expert_output.elements());
  std::vector<float> host_reduced(reduced.elements());
  CHECK(CudaOk(cudaMemcpy(host_experts.data(), expert_output.get(),
                          expert_output.bytes(), cudaMemcpyDeviceToHost),
               "download expert slots"));
  CHECK(CudaOk(cudaMemcpy(host_reduced.data(), reduced.get(), reduced.bytes(),
                          cudaMemcpyDeviceToHost),
               "download reduced experts"));
  std::uint64_t mismatches = 0U;
  for (std::uint64_t index = 0U; index < host_reduced.size(); ++index) {
    float expected = 0.0F;
    for (unsigned slot = 0U; slot < weights.size(); ++slot) {
      expected = std::fma(
          weights[slot],
          host_experts[static_cast<std::uint64_t>(slot) *
                           gem16::internal::kTrellis35DownOutput +
                       index],
          expected);
    }
    if (std::bit_cast<std::uint32_t>(expected) !=
        std::bit_cast<std::uint32_t>(host_reduced[index])) {
      ++mismatches;
    }
  }
  std::cout << description << " slot_order_bit_mismatches=" << mismatches
            << '\n';
  CHECK(mismatches == 0U);
}

void TestSyntheticFullM1() {
  FamilyStorage gate(gem16::internal::kTrellis35GateUpInput,
                     gem16::internal::kTrellis35GateUpOutput, 101U);
  FamilyStorage down(gem16::internal::kTrellis35DownInput,
                     gem16::internal::kTrellis35DownOutput, 211U);
  const gem16::internal::Trellis35DeviceLayerBinding layer{gate.binding,
                                                           down.binding};
  DeviceBuffer<float> input(gem16::internal::kTrellis35GateUpInput);
  DeviceBuffer<std::uint32_t> ids(gem16::internal::kTrellis35M1TopK);
  DeviceBuffer<float> weights(gem16::internal::kTrellis35M1TopK);
  DeviceBuffer<float> output(gem16::internal::kTrellis35DownOutput);
  M1Storage storage;
  std::vector<float> host_input(input.elements());
  for (std::uint64_t index = 0U; index < host_input.size(); ++index) {
    host_input[index] =
        std::sin(static_cast<float>(index) * 0.00390625F) * 1.0e-4F;
  }
  const std::array<std::uint32_t, gem16::internal::kTrellis35M1TopK>
      host_ids{0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U};
  const std::array<float, gem16::internal::kTrellis35M1TopK> host_weights{
      0.19F, 0.17F, 0.15F, 0.13F, 0.11F, 0.10F, 0.08F, 0.07F};
  if (!Upload(input, host_input, "upload synthetic M1 input") ||
      !Upload(ids, host_ids, "upload synthetic M1 IDs") ||
      !Upload(weights, host_weights, "upload synthetic M1 weights")) {
    return;
  }
  const float latency =
      RunFullM1(layer, input, ids, weights, storage, output, 3U, true);
  CheckFiniteAndIds(output, ids, host_ids, "synthetic full M1");
  CheckSlotOrderedReduction(storage.down_output, host_weights, output,
                            "synthetic full M1");
  std::cout << "synthetic full M1 latency_ms=" << latency << '\n';
}

void TestSyntheticT3() {
  FamilyStorage gate(gem16::internal::kTrellis35GateUpInput,
                     gem16::internal::kTrellis35GateUpOutput, 307U);
  FamilyStorage down(gem16::internal::kTrellis35DownInput,
                     gem16::internal::kTrellis35DownOutput, 401U);
  const gem16::internal::Trellis35DeviceLayerBinding layer{gate.binding,
                                                           down.binding};
  DeviceBuffer<float> input(gem16::internal::kTrellis35T3Rows *
                            gem16::internal::kTrellis35GateUpInput);
  std::vector<float> host_input(input.elements());
  for (unsigned row = 0U; row < gem16::internal::kTrellis35T3Rows; ++row) {
    for (std::uint64_t index = 0U;
         index < gem16::internal::kTrellis35GateUpInput; ++index) {
      host_input[static_cast<std::uint64_t>(row) *
                     gem16::internal::kTrellis35GateUpInput +
                 index] =
          std::sin(static_cast<float>(index * 11U + row * 37U) *
                   0.00390625F) *
          1.0e-4F;
    }
  }
  if (!Upload(input, host_input, "upload synthetic T3 input")) return;
  std::array<float, gem16::internal::kTrellis35T3Assignments> weights{};
  for (unsigned row = 0U; row < gem16::internal::kTrellis35T3Rows; ++row) {
    const std::array<float, gem16::internal::kTrellis35M1TopK> row_weights{
        0.19F, 0.17F, 0.15F, 0.13F, 0.11F, 0.10F, 0.08F, 0.07F};
    std::copy(row_weights.begin(), row_weights.end(),
              weights.begin() + row * gem16::internal::kTrellis35M1TopK);
  }

  std::array<std::uint32_t, gem16::internal::kTrellis35T3Assignments>
      zero_overlap{};
  for (unsigned assignment = 0U; assignment < zero_overlap.size();
       ++assignment) {
    zero_overlap[assignment] = assignment;
  }
  std::array<std::uint32_t, gem16::internal::kTrellis35T3Assignments>
      typical_overlap{0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U,
                      0U, 1U, 2U, 3U, 8U, 9U, 10U, 11U,
                      0U, 1U, 4U, 5U, 12U, 13U, 14U, 15U};
  std::array<std::uint32_t, gem16::internal::kTrellis35T3Assignments>
      maximal_overlap{};
  for (unsigned assignment = 0U; assignment < maximal_overlap.size();
       ++assignment) {
    maximal_overlap[assignment] =
        assignment % gem16::internal::kTrellis35M1TopK;
  }
  (void)RunT3Scenario(layer, input, zero_overlap, weights,
                      "synthetic T3 zero-overlap", false);
  (void)RunT3Scenario(layer, input, typical_overlap, weights,
                      "synthetic T3 typical-overlap", false);
  (void)RunT3Scenario(layer, input, maximal_overlap, weights,
                      "synthetic T3 maximal-overlap", true);
}

void ProfileSyntheticT3(unsigned unique_experts) {
  FamilyStorage gate(gem16::internal::kTrellis35GateUpInput,
                     gem16::internal::kTrellis35GateUpOutput, 503U);
  FamilyStorage down(gem16::internal::kTrellis35DownInput,
                     gem16::internal::kTrellis35DownOutput, 601U);
  const gem16::internal::Trellis35DeviceLayerBinding layer{gate.binding,
                                                           down.binding};
  DeviceBuffer<float> input(gem16::internal::kTrellis35T3Rows *
                            gem16::internal::kTrellis35GateUpInput);
  DeviceBuffer<std::uint32_t> ids(gem16::internal::kTrellis35T3Assignments);
  DeviceBuffer<float> weights(gem16::internal::kTrellis35T3Assignments);
  DeviceBuffer<float> output(gem16::internal::kTrellis35T3Rows *
                             gem16::internal::kTrellis35DownOutput);
  T3Storage storage;
  std::vector<float> host_input(input.elements(), 1.0e-4F);
  std::array<float, gem16::internal::kTrellis35T3Assignments> host_weights{};
  host_weights.fill(0.125F);
  const auto host_ids = MakeT3RoutesWithUnionSize(unique_experts);
  if (!Upload(input, host_input, "upload profile T3 input") ||
      !Upload(ids, host_ids, "upload profile T3 IDs") ||
      !Upload(weights, host_weights, "upload profile T3 weights")) {
    return;
  }
  const auto status = gem16::internal::LaunchTrellis35SelectedExpertsT3(
      input.get(), ids.get(), weights.get(), layer, storage.Bind(),
      output.get(), nullptr);
  CHECK(status.ok());
  CHECK(CudaOk(cudaDeviceSynchronize(), "synchronize profile T3"));
  std::cout << "profile T3 unique_experts=" << unique_experts << '\n';
}

void BenchmarkRetainedOverlapHistogram(
    const gem16::internal::Trellis35DeviceLayerBinding& layer,
    DeviceBuffer<float>& input,
    const std::array<float, gem16::internal::kTrellis35T3Assignments>&
        host_weights,
    const std::vector<std::uint32_t>& expert_order) {
  // Exact union-size histogram from the retained 16K Wikipedia sampled-D2
  // characterization: 11,550 verifier-layer samples, union sizes 8..24.
  constexpr std::array<std::uint64_t, 17> kWikipediaUnionHistogram{
      3U,   53U,  263U, 583U, 976U, 1344U, 1559U, 1649U, 1572U,
      1251U, 1007U, 660U, 402U, 171U, 42U,   14U,   1U};
  constexpr std::uint64_t kSamples = 11550U;
  DeviceBuffer<std::uint32_t> ids(gem16::internal::kTrellis35T3Assignments);
  DeviceBuffer<float> weights(gem16::internal::kTrellis35T3Assignments);
  DeviceBuffer<float> output(gem16::internal::kTrellis35T3Rows *
                             gem16::internal::kTrellis35DownOutput);
  T3Storage storage;
  CHECK(Upload(weights, host_weights, "upload retained-overlap weights"));
  double weighted_latency = 0.0;
  double weighted_union = 0.0;
  std::uint64_t observed_samples = 0U;
  for (unsigned offset = 0U; offset < kWikipediaUnionHistogram.size();
       ++offset) {
    const unsigned unique_experts = 8U + offset;
    const auto host_ids = RemapT3Routes(
        MakeT3RoutesWithUnionSize(unique_experts), expert_order);
    CHECK(Upload(ids, host_ids, "upload retained-overlap IDs"));
    const float latency =
        RunFullT3(layer, input, ids, weights, storage, output, 20U, false);
    const std::uint64_t count = kWikipediaUnionHistogram[offset];
    weighted_latency += static_cast<double>(latency) * count;
    weighted_union += static_cast<double>(unique_experts) * count;
    observed_samples += count;
    std::cout << "retained Wikipedia overlap union=" << unique_experts
              << " samples=" << count << " operator_latency_ms=" << latency
              << '\n';
  }
  CHECK(observed_samples == kSamples);
  const double mean_union = weighted_union / kSamples;
  CHECK(std::fabs(mean_union - 15.175324675324676) < 1.0e-12);
  std::cout << "retained Wikipedia overlap samples=" << observed_samples
            << " mean_unique_experts=" << mean_union
            << " histogram_weighted_operator_latency_ms="
            << weighted_latency / kSamples << '\n';
}

struct PrefillHostRouting {
  std::vector<gem16::internal::Gemma4MoePrefillAssignment> assignments;
  std::array<std::uint32_t, gem16::internal::kTrellis35ExpertCount>
      histogram{};
  std::array<std::uint32_t,
             gem16::internal::kTrellis35ExpertCount + 1U>
      prefix{};
  std::vector<std::uint32_t> permutation;
  std::vector<std::uint32_t> inverse;
};

PrefillHostRouting MakePrefillRouting(
    std::uint64_t tokens, const std::vector<std::uint32_t>& expert_order) {
  const std::uint64_t assignment_count =
      tokens * gem16::internal::kTrellis35M1TopK;
  CHECK(!expert_order.empty());
  CHECK(assignment_count <= std::numeric_limits<std::uint32_t>::max());
  PrefillHostRouting routing;
  routing.assignments.resize(assignment_count);
  routing.permutation.resize(assignment_count);
  routing.inverse.resize(assignment_count);
  const std::array<float, gem16::internal::kTrellis35M1TopK> weights{
      0.21F, 0.18F, 0.15F, 0.13F, 0.11F, 0.09F, 0.07F, 0.06F};
  for (std::uint64_t token = 0U; token < tokens; ++token) {
    for (unsigned slot = 0U; slot < gem16::internal::kTrellis35M1TopK;
         ++slot) {
      const std::uint64_t original =
          token * gem16::internal::kTrellis35M1TopK + slot;
      const std::uint32_t expert =
          expert_order[(token * 3U + slot) % expert_order.size()];
      CHECK(expert < gem16::internal::kTrellis35ExpertCount);
      routing.assignments[original] = {
          static_cast<std::uint16_t>(expert), static_cast<std::uint16_t>(slot),
          static_cast<std::uint32_t>(token), weights[slot]};
      ++routing.histogram[expert];
    }
  }
  for (unsigned expert = 0U;
       expert < gem16::internal::kTrellis35ExpertCount; ++expert) {
    routing.prefix[expert + 1U] =
        routing.prefix[expert] + routing.histogram[expert];
  }
  auto cursors = routing.prefix;
  for (std::uint32_t original = 0U; original < assignment_count; ++original) {
    const std::uint32_t expert = routing.assignments[original].expert_id;
    const std::uint32_t grouped = cursors[expert]++;
    routing.permutation[grouped] = original;
    routing.inverse[original] = grouped;
  }
  CHECK(routing.prefix.back() == assignment_count);
  return routing;
}

bool UploadPrefillRouting(PrefillStorage& storage,
                          const PrefillHostRouting& routing) {
  std::vector<std::uint32_t> prefix(routing.prefix.begin(),
                                    routing.prefix.end());
  std::vector<std::uint32_t> histogram(routing.histogram.begin(),
                                       routing.histogram.end());
  return Upload(storage.assignments, routing.assignments,
                "upload prefill assignments") &&
         Upload(storage.histogram, histogram, "upload prefill histogram") &&
         Upload(storage.prefix, prefix, "upload prefill prefix") &&
         Upload(storage.permutation, routing.permutation,
                "upload prefill permutation") &&
         Upload(storage.inverse, routing.inverse,
                "upload prefill inverse permutation");
}

float RunFullPrefill(
    const gem16::internal::Trellis35DeviceLayerBinding& layer,
    DeviceBuffer<float>& input, PrefillStorage& storage, unsigned iterations,
    bool capture) {
  const auto workspace = storage.Bind();
  if (capture) {
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    CHECK(CudaOk(cudaStreamCreate(&stream), "create prefill graph stream"));
    CHECK(CudaOk(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                 "begin Trellis35 prefill capture"));
    const auto status = gem16::internal::LaunchTrellis35PrefillExpertsW4A8(
        input.get(), storage.tokens, layer, workspace, stream);
    CHECK(status.ok());
    CHECK(CudaOk(cudaStreamEndCapture(stream, &graph),
                 "end Trellis35 prefill capture"));
    CHECK(CudaOk(cudaGraphInstantiate(&executable, graph, 0U),
                 "instantiate Trellis35 prefill graph"));
    CHECK(CudaOk(cudaGraphLaunch(executable, stream),
                 "launch Trellis35 prefill graph first"));
    CHECK(CudaOk(cudaGraphLaunch(executable, stream),
                 "launch Trellis35 prefill graph replay"));
    CHECK(CudaOk(cudaStreamSynchronize(stream),
                 "synchronize Trellis35 prefill graph"));
    if (executable != nullptr) (void)cudaGraphExecDestroy(executable);
    if (graph != nullptr) (void)cudaGraphDestroy(graph);
    if (stream != nullptr) (void)cudaStreamDestroy(stream);
  }
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  CHECK(CudaOk(cudaEventCreate(&begin), "create prefill begin event"));
  CHECK(CudaOk(cudaEventCreate(&end), "create prefill end event"));
  CHECK(CudaOk(cudaEventRecord(begin), "record prefill begin event"));
  for (unsigned iteration = 0U; iteration < iterations; ++iteration) {
    const auto status = gem16::internal::LaunchTrellis35PrefillExpertsW4A8(
        input.get(), storage.tokens, layer, workspace, nullptr);
    CHECK(status.ok());
  }
  CHECK(CudaOk(cudaEventRecord(end), "record prefill end event"));
  CHECK(CudaOk(cudaEventSynchronize(end), "synchronize prefill end event"));
  float milliseconds = 0.0F;
  CHECK(CudaOk(cudaEventElapsedTime(&milliseconds, begin, end),
               "measure Trellis35 prefill"));
  if (begin != nullptr) (void)cudaEventDestroy(begin);
  if (end != nullptr) (void)cudaEventDestroy(end);
  return milliseconds / static_cast<float>(iterations);
}

float RunPrefillScenario(
    const gem16::internal::Trellis35DeviceLayerBinding& layer,
    DeviceBuffer<float>& input, PrefillStorage& storage,
    const PrefillHostRouting& routing, const char* description,
    bool compare_m1, bool capture) {
  CHECK(UploadPrefillRouting(storage, routing));
  const float latency = RunFullPrefill(layer, input, storage, 3U, capture);
  std::vector<float> host_output(storage.token_hidden.elements());
  CHECK(CudaOk(cudaMemcpy(host_output.data(), storage.token_hidden.get(),
                          storage.token_hidden.bytes(),
                          cudaMemcpyDeviceToHost),
               "download prefill reduced output"));
  CHECK(std::all_of(host_output.begin(), host_output.end(),
                    [](float value) { return std::isfinite(value); }));
  std::vector<gem16::internal::Gemma4MoePrefillAssignment>
      unchanged_assignments(storage.assignments.elements());
  std::vector<std::uint32_t> unchanged_permutation(
      storage.permutation.elements());
  std::vector<std::uint32_t> unchanged_prefix(storage.prefix.elements());
  std::vector<std::uint32_t> unchanged_histogram(
      storage.histogram.elements());
  CHECK(CudaOk(cudaMemcpy(unchanged_assignments.data(),
                          storage.assignments.get(), storage.assignments.bytes(),
                          cudaMemcpyDeviceToHost),
               "download unchanged prefill assignments"));
  CHECK(CudaOk(cudaMemcpy(unchanged_permutation.data(),
                          storage.permutation.get(), storage.permutation.bytes(),
                          cudaMemcpyDeviceToHost),
               "download unchanged prefill permutation"));
  CHECK(CudaOk(cudaMemcpy(unchanged_prefix.data(), storage.prefix.get(),
                          storage.prefix.bytes(), cudaMemcpyDeviceToHost),
               "download unchanged prefill prefix"));
  CHECK(CudaOk(cudaMemcpy(unchanged_histogram.data(), storage.histogram.get(),
                          storage.histogram.bytes(), cudaMemcpyDeviceToHost),
               "download restored prefill histogram"));
  CHECK(std::memcmp(unchanged_assignments.data(), routing.assignments.data(),
                    storage.assignments.bytes()) == 0);
  CHECK(unchanged_permutation == routing.permutation);
  CHECK(std::equal(unchanged_prefix.begin(), unchanged_prefix.end(),
                   routing.prefix.begin()));
  CHECK(std::equal(unchanged_histogram.begin(), unchanged_histogram.end(),
                   routing.histogram.begin()));

  if (compare_m1) {
    DeviceBuffer<std::uint32_t> ids(gem16::internal::kTrellis35M1TopK);
    DeviceBuffer<float> weights(gem16::internal::kTrellis35M1TopK);
    DeviceBuffer<float> ignored_output(gem16::internal::kTrellis35DownOutput);
    DeviceBuffer<float> reference_experts(storage.expert_down.elements());
    DeviceBuffer<float> reference_reduced(storage.token_hidden.elements());
    M1Storage m1;
    for (std::uint64_t token = 0U; token < storage.tokens; ++token) {
      std::array<std::uint32_t, gem16::internal::kTrellis35M1TopK> host_ids{};
      std::array<float, gem16::internal::kTrellis35M1TopK> host_weights{};
      for (unsigned slot = 0U; slot < gem16::internal::kTrellis35M1TopK;
           ++slot) {
        const auto& assignment =
            routing.assignments[token * gem16::internal::kTrellis35M1TopK +
                                slot];
        host_ids[slot] = assignment.expert_id;
        host_weights[slot] = assignment.weight;
      }
      CHECK(Upload(ids, host_ids, "upload prefill M1 oracle IDs"));
      CHECK(Upload(weights, host_weights,
                   "upload prefill M1 oracle weights"));
      const auto status = gem16::internal::LaunchTrellis35SelectedExpertsM1(
          input.get() + token * gem16::internal::kTrellis35GateUpInput,
          ids.get(), weights.get(), layer, m1.Bind(), ignored_output.get(),
          nullptr);
      CHECK(status.ok());
      CHECK(CudaOk(cudaMemcpyAsync(
                       reference_experts.get() +
                           token * gem16::internal::kTrellis35M1TopK *
                               gem16::internal::kTrellis35DownOutput,
                       m1.down_output.get(), m1.down_output.bytes(),
                       cudaMemcpyDeviceToDevice),
                   "copy prefill M1 oracle expert outputs"));
    }
    auto status = gem16::internal::LaunchGemma4MoeReduceAssignments(
        reference_experts.get(), storage.assignments.get(),
        reference_reduced.get(), gem16::internal::kTrellis35DownOutput,
        gem16::internal::kTrellis35M1TopK, storage.tokens, nullptr);
    CHECK(status.ok());
    CHECK(CudaOk(cudaDeviceSynchronize(), "synchronize prefill M1 oracle"));
    std::vector<float> host_experts(storage.expert_down.elements());
    std::vector<float> host_reference_experts(reference_experts.elements());
    std::vector<float> host_reference(reference_reduced.elements());
    CHECK(CudaOk(cudaMemcpy(host_experts.data(), storage.expert_down.get(),
                            storage.expert_down.bytes(),
                            cudaMemcpyDeviceToHost),
                 "download prefill expert outputs"));
    CHECK(CudaOk(cudaMemcpy(host_reference_experts.data(),
                            reference_experts.get(), reference_experts.bytes(),
                            cudaMemcpyDeviceToHost),
                 "download prefill oracle expert outputs"));
    CHECK(CudaOk(cudaMemcpy(host_reference.data(), reference_reduced.get(),
                            reference_reduced.bytes(),
                            cudaMemcpyDeviceToHost),
                 "download prefill oracle reduction"));
    const std::string expert_label =
        std::string(description) + " assignment outputs";
    const std::string reduction_label =
        std::string(description) + " slot reduction";
    Compare(host_reference_experts, host_experts, 4.0e-6F, 2.0e-6F,
            expert_label.c_str());
    Compare(host_reference, host_output, 4.0e-6F, 2.0e-6F,
            reduction_label.c_str());
  }
  std::cout << description << " tokens=" << storage.tokens
            << " assignments=" << storage.assignments.elements()
            << " latency_ms=" << latency << '\n';
  return latency;
}

std::vector<std::uint32_t> SequentialExpertOrder() {
  std::vector<std::uint32_t> experts(gem16::internal::kTrellis35ExpertCount);
  for (std::uint32_t expert = 0U; expert < experts.size(); ++expert) {
    experts[expert] = expert;
  }
  return experts;
}

void TestSyntheticPrefill() {
  auto plan = gem16::internal::BuildGemma4MoePrefillPlan(1024U);
  CHECK(plan.ok());
  if (plan.ok()) {
    CHECK(plan.value().chunk_tokens == 1024U);
    auto region_bytes = [&](std::string_view name) {
      const auto found = std::find_if(
          plan.value().moe_regions.begin(), plan.value().moe_regions.end(),
          [&](const auto& region) { return region.name == name; });
      CHECK(found != plan.value().moe_regions.end());
      return found == plan.value().moe_regions.end() ? 0U : found->bytes;
    };
    constexpr std::uint64_t kPlanTokens = 1024U;
    constexpr std::uint64_t kAssignments =
        kPlanTokens * gem16::internal::kTrellis35M1TopK;
    CHECK(region_bytes("expert_product") ==
          kAssignments * gem16::internal::kTrellis35ExpertIntermediate *
              sizeof(float));
    CHECK(region_bytes("expert_product") >=
          kAssignments * gem16::internal::kTrellis35GateUpInput);
    CHECK(region_bytes("expert_down_partials") ==
          kAssignments * gem16::internal::kTrellis35DownOutput *
              sizeof(float));
    CHECK(region_bytes("shared_product") >=
          kAssignments * 128U * sizeof(float));
    CHECK(region_bytes("shared_output") >=
          kAssignments * gem16::internal::kTrellis35DownInput);
    CHECK(region_bytes("token_nvfp4") -
              kPlanTokens *
                  (gem16::internal::kTrellis35GateUpInput / 2U) >=
          kAssignments * sizeof(float));
  }
  FamilyStorage gate(gem16::internal::kTrellis35GateUpInput,
                     gem16::internal::kTrellis35GateUpOutput, 701U);
  FamilyStorage down(gem16::internal::kTrellis35DownInput,
                     gem16::internal::kTrellis35DownOutput, 809U);
  const gem16::internal::Trellis35DeviceLayerBinding layer{gate.binding,
                                                           down.binding};
  constexpr std::uint64_t kTokens = 4U;
  DeviceBuffer<float> input(kTokens * gem16::internal::kTrellis35GateUpInput);
  std::vector<float> host_input(input.elements());
  for (std::uint64_t index = 0U; index < host_input.size(); ++index) {
    host_input[index] =
        std::sin(static_cast<float>(index * 7U + 3U) * 0.001953125F) *
        1.0e-4F;
  }
  CHECK(Upload(input, host_input, "upload synthetic prefill input"));
  PrefillStorage storage(kTokens);
  const auto routing = MakePrefillRouting(kTokens, SequentialExpertOrder());
  (void)RunPrefillScenario(layer, input, storage, routing,
                           "synthetic W4A8 prefill", true, true);
}

void ProfileSyntheticPrefill(std::uint64_t tokens) {
  FamilyStorage gate(gem16::internal::kTrellis35GateUpInput,
                     gem16::internal::kTrellis35GateUpOutput, 907U);
  FamilyStorage down(gem16::internal::kTrellis35DownInput,
                     gem16::internal::kTrellis35DownOutput, 1009U);
  const gem16::internal::Trellis35DeviceLayerBinding layer{gate.binding,
                                                           down.binding};
  DeviceBuffer<float> input(tokens * gem16::internal::kTrellis35GateUpInput);
  std::vector<float> host_input(input.elements(), 1.0e-4F);
  CHECK(Upload(input, host_input, "upload profile prefill input"));
  PrefillStorage storage(tokens);
  const auto routing = MakePrefillRouting(tokens, SequentialExpertOrder());
  CHECK(UploadPrefillRouting(storage, routing));
  const float latency = RunFullPrefill(layer, input, storage, 1U, false);
  std::cout << "profile W4A8 prefill tokens=" << tokens
            << " assignments=" << tokens * gem16::internal::kTrellis35M1TopK
            << " latency_ms=" << latency << '\n';
}

void ProfileRealPrefill(const std::string& checkpoint, std::uint64_t tokens) {
  auto artifact =
      gem16::internal::Gemma4Moe26BTrellis35DeviceArtifact::Load(checkpoint);
  CHECK(artifact.ok());
  if (!artifact.ok()) {
    std::cerr << artifact.status().message() << '\n';
    return;
  }
  DeviceBuffer<float> input(tokens * gem16::internal::kTrellis35GateUpInput);
  std::vector<float> host_input(input.elements());
  for (std::uint64_t index = 0U; index < host_input.size(); ++index) {
    host_input[index] =
        std::sin(static_cast<float>(index * 19U + 11U) * 0.001953125F) *
        0.03125F;
  }
  CHECK(Upload(input, host_input, "upload real profile prefill input"));
  PrefillStorage storage(tokens);
  const auto routing = MakePrefillRouting(tokens, SequentialExpertOrder());
  CHECK(UploadPrefillRouting(storage, routing));
  const float latency = RunFullPrefill(artifact.value().layers()[0], input,
                                       storage, 1U, false);
  std::cout << "profile real layer-0 W4A8 prefill tokens=" << tokens
            << " assignments=" << tokens * gem16::internal::kTrellis35M1TopK
            << " latency_ms=" << latency
            << " checkpoint_sha256="
            << artifact.value().stats().checkpoint_content_sha256 << '\n';
}

void CompareRealProjection(
    const gem16::internal::Trellis35DeviceFamilyBinding& family,
    std::uint64_t input_elements, std::uint64_t output_elements,
    const std::array<std::uint32_t, gem16::internal::kTrellis35M1TopK>& ids,
    const DeviceBuffer<std::uint8_t>& activation,
    const DeviceBuffer<float>& scales, const DeviceBuffer<float>& native_output,
    const char* description) {
  DeviceBuffer<float> reference(output_elements);
  std::vector<float> host_reference(output_elements);
  std::vector<float> host_native(output_elements);
  for (unsigned slot : {0U, 1U}) {
    const auto status =
        gem16::internal::LaunchTrellis35ReferenceW4A8ProjectionM1(
            activation.get() + slot * input_elements, scales.get() + slot,
            family, ids[slot], reference.get(), input_elements,
            output_elements, nullptr);
    CHECK(status.ok());
    CHECK(CudaOk(cudaDeviceSynchronize(), "synchronize real projection"));
    CHECK(CudaOk(cudaMemcpy(host_reference.data(), reference.get(),
                            reference.bytes(), cudaMemcpyDeviceToHost),
                 "download real reference projection"));
    CHECK(CudaOk(cudaMemcpy(
                     host_native.data(),
                     native_output.get() + slot * output_elements,
                     reference.bytes(), cudaMemcpyDeviceToHost),
                 "download real native projection"));
    const std::string label =
        std::string(description) + " slot=" + std::to_string(slot) +
        " K" + std::to_string(family.rate_map[ids[slot]]);
    Compare(host_reference, host_native, 4.0e-3F, 8.0e-5F, label.c_str());
  }
}

void TestRealCheckpoint(const std::string& checkpoint) {
  auto artifact =
      gem16::internal::Gemma4Moe26BTrellis35DeviceArtifact::Load(checkpoint);
  CHECK(artifact.ok());
  if (!artifact.ok()) {
    std::cerr << artifact.status().message() << '\n';
    return;
  }
  const auto& layer = artifact.value().layers()[0];
  std::array<std::uint32_t, gem16::internal::kTrellis35M1TopK> host_ids{};
  std::vector<std::uint32_t> k3;
  std::vector<std::uint32_t> k4;
  for (std::uint32_t expert = 0U;
       expert < gem16::internal::kTrellis35ExpertCount; ++expert) {
    (layer.gate_up.rate_map[expert] == 3U ? k3 : k4).push_back(expert);
  }
  CHECK(!k3.empty() && !k4.empty());
  host_ids[0] = k3[0];
  host_ids[1] = k4[0];
  for (unsigned slot = 2U; slot < host_ids.size(); ++slot) {
    host_ids[slot] = (slot & 1U) == 0U ? k3[slot / 2U] : k4[slot / 2U];
  }
  DeviceBuffer<float> input(gem16::internal::kTrellis35GateUpInput);
  DeviceBuffer<std::uint32_t> ids(gem16::internal::kTrellis35M1TopK);
  DeviceBuffer<float> weights(gem16::internal::kTrellis35M1TopK);
  DeviceBuffer<float> output(gem16::internal::kTrellis35DownOutput);
  M1Storage storage;
  std::vector<float> host_input(input.elements());
  for (std::uint64_t index = 0U; index < host_input.size(); ++index) {
    host_input[index] =
        std::sin(static_cast<float>(index * 13U + 7U) * 0.001953125F) *
        0.03125F;
  }
  const std::array<float, gem16::internal::kTrellis35M1TopK> host_weights{
      0.21F, 0.18F, 0.15F, 0.13F, 0.11F, 0.09F, 0.07F, 0.06F};
  if (!Upload(input, host_input, "upload real M1 input") ||
      !Upload(ids, host_ids, "upload real M1 IDs") ||
      !Upload(weights, host_weights, "upload real M1 weights")) {
    return;
  }
  const float latency =
      RunFullM1(layer, input, ids, weights, storage, output, 5U, false);
  CheckFiniteAndIds(output, ids, host_ids, "real layer-0 full M1");
  CheckSlotOrderedReduction(storage.down_output, host_weights, output,
                            "real layer-0 full M1");
  CompareRealProjection(layer.gate_up, gem16::internal::kTrellis35GateUpInput,
                        gem16::internal::kTrellis35GateUpOutput, host_ids,
                        storage.gate_input_fp8, storage.gate_scales,
                        storage.gate_transformed, "real Gate+Up parity");
  CompareRealProjection(layer.down, gem16::internal::kTrellis35DownInput,
                        gem16::internal::kTrellis35DownOutput, host_ids,
                        storage.down_input_fp8, storage.down_scales,
                        storage.down_transformed, "real Down parity");

  std::uint64_t payload_bytes = 0U;
  for (const std::uint32_t expert : host_ids) {
    payload_bytes +=
        gem16::internal::kTrellis35GateUpInput *
        gem16::internal::kTrellis35GateUpOutput *
        layer.gate_up.rate_map[expert] / 8U;
    payload_bytes += gem16::internal::kTrellis35DownInput *
                     gem16::internal::kTrellis35DownOutput *
                     layer.down.rate_map[expert] / 8U;
  }
  const std::uint64_t sidecar_bytes =
      host_ids.size() *
      (gem16::internal::kTrellis35GateUpInput +
       gem16::internal::kTrellis35GateUpOutput +
       gem16::internal::kTrellis35DownInput +
       gem16::internal::kTrellis35DownOutput) *
      sizeof(std::uint16_t);
  std::cout << "real layer-0 selected_payload_bytes=" << payload_bytes
            << " selected_sidecar_bytes=" << sidecar_bytes
            << " descriptors_read=" << host_ids.size() * 2U
            << " latency_ms=" << latency
            << " checkpoint_sha256="
            << artifact.value().stats().checkpoint_content_sha256 << '\n';

  std::vector<std::uint32_t> expert_order;
  expert_order.reserve(gem16::internal::kTrellis35ExpertCount);
  for (unsigned index = 0U; index < 12U; ++index) {
    expert_order.push_back(k3[index]);
    expert_order.push_back(k4[index]);
  }
  const auto t3_ids = RemapT3Routes(MakeT3RoutesWithUnionSize(16U),
                                    expert_order);
  std::array<float, gem16::internal::kTrellis35T3Assignments> t3_weights{};
  for (unsigned row = 0U; row < gem16::internal::kTrellis35T3Rows; ++row) {
    std::copy(host_weights.begin(), host_weights.end(),
              t3_weights.begin() + row * gem16::internal::kTrellis35M1TopK);
  }
  DeviceBuffer<float> t3_input(gem16::internal::kTrellis35T3Rows *
                               gem16::internal::kTrellis35GateUpInput);
  std::vector<float> host_t3_input(t3_input.elements());
  for (unsigned row = 0U; row < gem16::internal::kTrellis35T3Rows; ++row) {
    for (std::uint64_t index = 0U;
         index < gem16::internal::kTrellis35GateUpInput; ++index) {
      host_t3_input[static_cast<std::uint64_t>(row) *
                        gem16::internal::kTrellis35GateUpInput +
                    index] =
          std::sin(static_cast<float>(index * 13U + row * 41U + 7U) *
                   0.001953125F) *
          0.03125F;
    }
  }
  CHECK(Upload(t3_input, host_t3_input, "upload real T3 input"));
  (void)RunT3Scenario(layer, t3_input, t3_ids, t3_weights,
                      "real layer-0 T3 typical-overlap", true);

  std::array<bool, gem16::internal::kTrellis35ExpertCount> seen{};
  std::uint64_t t3_assignment_payload_bytes = 0U;
  std::uint64_t t3_unique_payload_bytes = 0U;
  for (const std::uint32_t expert : t3_ids) {
    const std::uint64_t expert_bytes =
        gem16::internal::kTrellis35GateUpInput *
            gem16::internal::kTrellis35GateUpOutput *
            layer.gate_up.rate_map[expert] / 8U +
        gem16::internal::kTrellis35DownInput *
            gem16::internal::kTrellis35DownOutput *
            layer.down.rate_map[expert] / 8U;
    t3_assignment_payload_bytes += expert_bytes;
    if (!seen[expert]) {
      seen[expert] = true;
      t3_unique_payload_bytes += expert_bytes;
    }
  }
  std::cout << "real layer-0 T3 assignment_payload_bytes="
            << t3_assignment_payload_bytes
            << " grouped_unique_payload_bytes=" << t3_unique_payload_bytes
            << " payload_bytes_avoided="
            << t3_assignment_payload_bytes - t3_unique_payload_bytes << '\n';
  BenchmarkRetainedOverlapHistogram(layer, t3_input, t3_weights,
                                    expert_order);

  constexpr std::uint64_t kPrefillTokens = 4U;
  DeviceBuffer<float> prefill_input(
      kPrefillTokens * gem16::internal::kTrellis35GateUpInput);
  std::vector<float> host_prefill_input(prefill_input.elements());
  for (std::uint64_t index = 0U; index < host_prefill_input.size(); ++index) {
    host_prefill_input[index] =
        std::sin(static_cast<float>(index * 17U + 5U) * 0.001953125F) *
        0.03125F;
  }
  CHECK(Upload(prefill_input, host_prefill_input,
               "upload real prefill input"));
  PrefillStorage prefill_storage(kPrefillTokens);
  const auto prefill_routing =
      MakePrefillRouting(kPrefillTokens, expert_order);
  (void)RunPrefillScenario(layer, prefill_input, prefill_storage,
                           prefill_routing, "real layer-0 W4A8 prefill",
                           true, true);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 3 && std::string(argv[1]) == "--profile-t3") {
    const unsigned unique_experts =
        static_cast<unsigned>(std::stoul(argv[2]));
    if (unique_experts < gem16::internal::kTrellis35M1TopK ||
        unique_experts > gem16::internal::kTrellis35T3Assignments) {
      std::cerr << "profile T3 union size must be in [8, 24]\n";
      return 2;
    }
    ProfileSyntheticT3(unique_experts);
    return failures == 0 ? 0 : 1;
  }
  if (argc == 3 && std::string(argv[1]) == "--profile-prefill") {
    const std::uint64_t tokens = std::stoull(argv[2]);
    if (tokens == 0U || tokens > 1024U) {
      std::cerr << "prefill profile tokens must be in [1, 1024]\n";
      return 2;
    }
    ProfileSyntheticPrefill(tokens);
    return failures == 0 ? 0 : 1;
  }
  if (argc == 4 && std::string(argv[1]) == "--profile-prefill-checkpoint") {
    const std::uint64_t tokens = std::stoull(argv[3]);
    if (tokens == 0U || tokens > 1024U) {
      std::cerr << "prefill profile tokens must be in [1, 1024]\n";
      return 2;
    }
    ProfileRealPrefill(argv[2], tokens);
    return failures == 0 ? 0 : 1;
  }
  TestRandomizedProjectionParity();
  TestTransformsAndDownPadding();
  TestSyntheticFullM1();
  TestSyntheticT3();
  TestSyntheticPrefill();
  if (argc == 3 && std::string(argv[1]) == "--checkpoint") {
    TestRealCheckpoint(argv[2]);
  } else if (argc != 1) {
    std::cerr << "usage: gem16-cuda-trellis35-tests "
                 "[--checkpoint PATH | --profile-t3 UNIQUE | "
                 "--profile-prefill TOKENS | "
                 "--profile-prefill-checkpoint PATH TOKENS]\n";
    return 2;
  }
  if (failures == 0) {
    std::cout << "trellis35_m1_test_pass\n";
  }
  return failures == 0 ? 0 : 1;
}

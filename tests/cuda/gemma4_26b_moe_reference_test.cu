#include "cuda/moe/reference.h"
#include "cuda/moe/prefill.h"
#include "cuda/nvfp4/reference.h"
#include "cuda/nvfp4/sm120.h"
#include "cuda/nvfp4/sm120_layout.h"
#include "gem16/nvfp4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void Check(bool condition, const char* expression, int line) {
  if (!condition) {
    std::cerr << __FILE__ << ':' << line << ": check failed: " << expression
              << '\n';
    ++failures;
  }
}

#define CHECK(expression) Check(static_cast<bool>(expression), #expression, __LINE__)

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
  explicit DeviceBuffer(std::size_t elements) : elements_(elements) {
    if (!CudaOk(cudaMalloc(&pointer_, elements * sizeof(T)), "cudaMalloc")) {
      pointer_ = nullptr;
    }
  }
  ~DeviceBuffer() {
    if (pointer_ != nullptr) (void)cudaFree(pointer_);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() const { return static_cast<T*>(pointer_); }
  std::size_t bytes() const { return elements_ * sizeof(T); }

 private:
  void* pointer_ = nullptr;
  std::size_t elements_ = 0;
};

std::uint16_t Bf16(float value) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

void TestSelectedExpertSlotBatch() {
  constexpr std::uint64_t kWidth = 64U;
  constexpr std::uint64_t kIntermediate = 64U;
  constexpr std::uint32_t kExperts = 8U;
  constexpr std::uint32_t kTopK = 8U;
  const auto make_expert_matrix = [](std::uint64_t rows,
                                     std::uint64_t columns,
                                     std::uint32_t seed) {
    std::vector<std::uint8_t> result_weights, result_scales;
    for (std::uint32_t expert = 0; expert < kExperts; ++expert) {
      std::vector<std::uint8_t> packed(rows * columns / 2U);
      std::vector<std::uint8_t> scales(rows * columns / 16U, 0x38U);
      for (std::uint64_t index = 0; index < packed.size(); ++index) {
        const std::uint8_t low = static_cast<std::uint8_t>(
            1U + (index * 5U + expert * 3U + seed) % 6U);
        const std::uint8_t high = static_cast<std::uint8_t>(
            8U + (index * 7U + expert * 5U + seed) % 6U);
        packed[index] = static_cast<std::uint8_t>(low | (high << 4U));
      }
      const auto layout =
          gem16::internal::PlanSm120Nvfp4SourceLayout(rows, columns);
      CHECK(layout.ok());
      if (!layout.ok()) return std::pair{result_weights, result_scales};
      const auto tiled = gem16::internal::TileSm120Nvfp4Weights(
          layout.value(), std::span<const std::uint8_t>(packed));
      const auto tiled_scales = gem16::internal::TileSm120Nvfp4WeightScales(
          layout.value(), std::span<const std::uint8_t>(scales));
      CHECK(tiled.ok());
      CHECK(tiled_scales.ok());
      if (!tiled.ok() || !tiled_scales.ok()) {
        return std::pair{result_weights, result_scales};
      }
      result_weights.insert(result_weights.end(), tiled.value().begin(),
                            tiled.value().end());
      result_scales.insert(result_scales.end(), tiled_scales.value().begin(),
                           tiled_scales.value().end());
    }
    return std::pair{result_weights, result_scales};
  };

  std::vector<float> activation(kWidth);
  for (std::uint64_t index = 0; index < kWidth; ++index) {
    activation[index] =
        static_cast<float>(static_cast<int>((index * 11U) % 31U) - 15) /
        16.0F;
  }
  const auto quantized = gem16::nvfp4::QuantizeActivation(activation, 1.0F);
  CHECK(quantized.ok());
  if (!quantized.ok()) return;
  const auto [gate_up_host_weights, gate_up_host_scales] =
      make_expert_matrix(2U * kIntermediate, kWidth, 3U);
  const auto [down_host_weights, down_host_scales] =
      make_expert_matrix(kWidth, kIntermediate, 11U);
  const std::uint64_t gate_up_weight_bytes =
      kExperts * 2U * kIntermediate * kWidth / 2U;
  const std::uint64_t down_weight_bytes =
      kExperts * kWidth * kIntermediate / 2U;
  CHECK(gate_up_host_weights.size() == gate_up_weight_bytes);
  CHECK(gate_up_host_scales.size() == gate_up_weight_bytes / 8U);
  CHECK(down_host_weights.size() == down_weight_bytes);
  CHECK(down_host_scales.size() == down_weight_bytes / 8U);
  if (gate_up_host_weights.size() != gate_up_weight_bytes ||
      gate_up_host_scales.size() != gate_up_weight_bytes / 8U ||
      down_host_weights.size() != down_weight_bytes ||
      down_host_scales.size() != down_weight_bytes / 8U) {
    return;
  }

  DeviceBuffer<std::uint8_t> device_activation(
      quantized.value().packed_e2m1.size());
  DeviceBuffer<std::uint8_t> device_activation_scales(
      quantized.value().block_scales_e4m3fn.size());
  DeviceBuffer<std::uint8_t> gate_up_weights(gate_up_weight_bytes),
      gate_up_scales(gate_up_weight_bytes / 8U),
      down_weights(down_weight_bytes), down_scales(down_weight_bytes / 8U);
  DeviceBuffer<std::uint32_t> selected_ids(kTopK);
  DeviceBuffer<float> sequential_gate_up(kTopK * 2U * kIntermediate),
      batched_gate_up(kTopK * 2U * kIntermediate),
      sequential_product(kTopK * kIntermediate),
      batched_product(kTopK * kIntermediate),
      sequential_down(kTopK * kWidth), batched_down(kTopK * kWidth);
  DeviceBuffer<std::uint8_t> product_packed(kTopK * kIntermediate / 2U),
      product_scales(kTopK * kIntermediate / 16U);
  const std::array<std::uint32_t, kTopK> ids{7U, 1U, 5U, 0U,
                                             3U, 6U, 2U, 4U};
  CHECK(CudaOk(cudaMemcpy(device_activation.get(),
                          quantized.value().packed_e2m1.data(),
                          device_activation.bytes(), cudaMemcpyHostToDevice),
               "copy slot-batch activation"));
  CHECK(CudaOk(cudaMemcpy(device_activation_scales.get(),
                          quantized.value().block_scales_e4m3fn.data(),
                          device_activation_scales.bytes(),
                          cudaMemcpyHostToDevice),
               "copy slot-batch activation scales"));
  CHECK(CudaOk(cudaMemcpy(gate_up_weights.get(), gate_up_host_weights.data(),
                          gate_up_weights.bytes(), cudaMemcpyHostToDevice),
               "copy slot-batch W13 weights"));
  CHECK(CudaOk(cudaMemcpy(gate_up_scales.get(),
                          gate_up_host_scales.data(),
                          gate_up_scales.bytes(), cudaMemcpyHostToDevice),
               "copy slot-batch W13 scales"));
  CHECK(CudaOk(cudaMemcpy(down_weights.get(), down_host_weights.data(),
                          down_weights.bytes(), cudaMemcpyHostToDevice),
               "copy slot-batch W2 weights"));
  CHECK(CudaOk(cudaMemcpy(down_scales.get(),
                          down_host_scales.data(),
                          down_scales.bytes(), cudaMemcpyHostToDevice),
               "copy slot-batch W2 scales"));
  CHECK(CudaOk(cudaMemcpy(selected_ids.get(), ids.data(), selected_ids.bytes(),
                          cudaMemcpyHostToDevice),
               "copy slot-batch selected IDs"));

  for (std::uint32_t slot = 0; slot < kTopK; ++slot) {
    const std::uint64_t product_offset =
        static_cast<std::uint64_t>(slot) * kIntermediate;
    const std::uint64_t gate_offset = 2U * product_offset;
    CHECK(gem16::internal::LaunchNvfp4Sm120SelectedFusedGateUp(
              device_activation.get(), device_activation_scales.get(),
              gate_up_weights.get(), gate_up_scales.get(), selected_ids.get(),
              slot, sequential_gate_up.get() + gate_offset,
              sequential_gate_up.get() + gate_offset + kIntermediate,
              sequential_product.get() + product_offset, kIntermediate,
              kWidth, kExperts, 1.0F, 1.0F, nullptr)
              .ok());
  }
  CHECK(gem16::internal::LaunchNvfp4Sm120SelectedFusedGateUpBatch(
            device_activation.get(), device_activation_scales.get(),
            gate_up_weights.get(), gate_up_scales.get(), selected_ids.get(),
            kTopK, batched_gate_up.get(),
            batched_gate_up.get() + kIntermediate, batched_product.get(),
            kIntermediate, kWidth, kExperts, 1.0F, 1.0F, nullptr)
            .ok());
  CHECK(CudaOk(cudaDeviceSynchronize(), "synchronize slot-batch W13"));
  std::vector<float> sequential_gate_values(kTopK * 2U * kIntermediate),
      batched_gate_values(kTopK * 2U * kIntermediate),
      sequential_product_values(kTopK * kIntermediate),
      batched_product_values(kTopK * kIntermediate);
  CHECK(CudaOk(cudaMemcpy(sequential_gate_values.data(),
                          sequential_gate_up.get(),
                          sequential_gate_up.bytes(), cudaMemcpyDeviceToHost),
               "copy sequential W13"));
  CHECK(CudaOk(cudaMemcpy(batched_gate_values.data(), batched_gate_up.get(),
                          batched_gate_up.bytes(), cudaMemcpyDeviceToHost),
               "copy batched W13"));
  CHECK(CudaOk(cudaMemcpy(sequential_product_values.data(),
                          sequential_product.get(), sequential_product.bytes(),
                          cudaMemcpyDeviceToHost),
               "copy sequential products"));
  CHECK(CudaOk(cudaMemcpy(batched_product_values.data(), batched_product.get(),
                          batched_product.bytes(), cudaMemcpyDeviceToHost),
               "copy batched products"));
  CHECK(sequential_gate_values == batched_gate_values);
  CHECK(sequential_product_values == batched_product_values);

  CHECK(gem16::internal::LaunchNvfp4ReferenceActivationQuantization(
            batched_product.get(), product_packed.get(), product_scales.get(),
            kTopK * kIntermediate, 1.0F, nullptr)
            .ok());
  for (std::uint32_t slot = 0; slot < kTopK; ++slot) {
    CHECK(gem16::internal::LaunchNvfp4Sm120SelectedDirectProjectionBf16Float(
              product_packed.get() + slot * kIntermediate / 2U,
              product_scales.get() + slot * kIntermediate / 16U,
              down_weights.get(), down_scales.get(), selected_ids.get(), slot,
              sequential_down.get() + slot * kWidth, kWidth, kIntermediate,
              kExperts, 1.0F, 1.0F, nullptr)
              .ok());
  }
  CHECK(gem16::internal::
            LaunchNvfp4Sm120SelectedDirectProjectionBf16FloatBatch(
                product_packed.get(), product_scales.get(), down_weights.get(),
                down_scales.get(), selected_ids.get(), kTopK,
                batched_down.get(), kWidth, kIntermediate, kExperts, 1.0F,
                1.0F, nullptr)
            .ok());
  CHECK(CudaOk(cudaDeviceSynchronize(), "synchronize slot-batch W2"));
  std::vector<float> sequential_down_values(kTopK * kWidth),
      batched_down_values(kTopK * kWidth);
  CHECK(CudaOk(cudaMemcpy(sequential_down_values.data(), sequential_down.get(),
                          sequential_down.bytes(), cudaMemcpyDeviceToHost),
               "copy sequential W2"));
  CHECK(CudaOk(cudaMemcpy(batched_down_values.data(), batched_down.get(),
                          batched_down.bytes(), cudaMemcpyDeviceToHost),
               "copy batched W2"));
  CHECK(sequential_down_values == batched_down_values);
}

void TestPhysicalBf16GroupedExpertOperators() {
  constexpr std::uint64_t kWidth = 64U;
  constexpr std::uint64_t kIntermediate = 64U;
  constexpr std::uint32_t kExperts = 2U;
  constexpr std::uint32_t kTopK = 8U;
  constexpr std::uint64_t kTokens = 3U;
  constexpr std::uint64_t kAssignments = kTokens * kTopK;

  const auto make_expert_matrix = [](std::uint64_t rows,
                                     std::uint64_t columns,
                                     std::uint32_t seed) {
    std::vector<std::uint8_t> result_weights, result_scales;
    for (std::uint32_t expert = 0; expert < kExperts; ++expert) {
      std::vector<std::uint8_t> packed(rows * columns / 2U);
      std::vector<std::uint8_t> scales(rows * columns / 16U, 0x38U);
      for (std::uint64_t index = 0; index < packed.size(); ++index) {
        const std::uint8_t low = static_cast<std::uint8_t>(
            1U + (index * 3U + expert * 5U + seed) % 6U);
        const std::uint8_t high = static_cast<std::uint8_t>(
            8U + (index * 7U + expert * 11U + seed) % 6U);
        packed[index] = static_cast<std::uint8_t>(low | (high << 4U));
      }
      const auto layout =
          gem16::internal::PlanSm120Nvfp4SourceLayout(rows, columns);
      CHECK(layout.ok());
      if (!layout.ok()) return std::pair{result_weights, result_scales};
      const auto tiled = gem16::internal::TileSm120Nvfp4Weights(
          layout.value(), std::span<const std::uint8_t>(packed));
      const auto tiled_scales = gem16::internal::TileSm120Nvfp4WeightScales(
          layout.value(), std::span<const std::uint8_t>(scales));
      CHECK(tiled.ok());
      CHECK(tiled_scales.ok());
      if (!tiled.ok() || !tiled_scales.ok()) {
        return std::pair{result_weights, result_scales};
      }
      result_weights.insert(result_weights.end(), tiled.value().begin(),
                            tiled.value().end());
      result_scales.insert(result_scales.end(), tiled_scales.value().begin(),
                           tiled_scales.value().end());
    }
    return std::pair{result_weights, result_scales};
  };

  std::vector<float> activation(kTokens * kWidth);
  for (std::uint64_t token = 0U; token < kTokens; ++token) {
    for (std::uint64_t column = 0U; column < kWidth; ++column) {
      const int centered =
          static_cast<int>((token * 17U + column * 13U) % 37U) - 18;
      activation[token * kWidth + column] =
          static_cast<float>(centered) / 32.0F;
    }
  }
  const auto quantized = gem16::nvfp4::QuantizeActivation(activation, 1.0F);
  CHECK(quantized.ok());
  if (!quantized.ok()) return;
  const auto [gate_up_host_weights, gate_up_host_scales] =
      make_expert_matrix(2U * kIntermediate, kWidth, 3U);
  const auto [down_host_weights, down_host_scales] =
      make_expert_matrix(kWidth, kIntermediate, 19U);
  if (gate_up_host_weights.empty() || gate_up_host_scales.empty() ||
      down_host_weights.empty() || down_host_scales.empty()) {
    return;
  }

  std::vector<gem16::internal::Gemma4MoePrefillAssignment> assignments(
      kAssignments);
  for (std::uint32_t token = 0U; token < kTokens; ++token) {
    for (std::uint32_t slot = 0U; slot < kTopK; ++slot) {
      const std::uint64_t original = token * kTopK + slot;
      assignments[original] = {
          static_cast<std::uint16_t>((token + slot) % kExperts),
          static_cast<std::uint16_t>(slot), token,
          static_cast<float>(slot + 1U) / 36.0F};
    }
  }
  std::array<std::uint32_t, kExperts + 1U> prefix{};
  std::vector<std::uint32_t> permutation;
  permutation.reserve(kAssignments);
  for (std::uint32_t expert = 0U; expert < kExperts; ++expert) {
    prefix[expert] = static_cast<std::uint32_t>(permutation.size());
    for (std::uint32_t original = 0U; original < kAssignments; ++original) {
      if (assignments[original].expert_id == expert) {
        permutation.push_back(original);
      }
    }
  }
  prefix[kExperts] = static_cast<std::uint32_t>(permutation.size());
  CHECK(permutation.size() == kAssignments);
  std::vector<std::uint32_t> tiles;
  for (std::uint32_t expert = 0U; expert < kExperts; ++expert) {
    for (std::uint32_t grouped = prefix[expert];
         grouped < prefix[expert + 1U]; grouped += 16U) {
      tiles.push_back((expert << 16U) | grouped);
    }
  }
  const std::uint32_t tile_count = static_cast<std::uint32_t>(tiles.size());

  DeviceBuffer<std::uint8_t> device_activation(
      quantized.value().packed_e2m1.size());
  DeviceBuffer<std::uint8_t> device_activation_scales(
      quantized.value().block_scales_e4m3fn.size());
  DeviceBuffer<std::uint8_t> gate_up_weights(gate_up_host_weights.size()),
      gate_up_scales(gate_up_host_scales.size()),
      down_weights(down_host_weights.size()),
      down_scales(down_host_scales.size());
  DeviceBuffer<gem16::internal::Gemma4MoePrefillAssignment>
      device_assignments(kAssignments);
  DeviceBuffer<std::uint32_t> device_permutation(kAssignments),
      device_prefix(kExperts + 1U), device_tiles(tiles.size()),
      device_tile_count(1U);
  DeviceBuffer<float> product_float(kAssignments * kIntermediate),
      down_float(kAssignments * kWidth), reduced_float(kTokens * kWidth),
      reduced_bf16(kTokens * kWidth);
  DeviceBuffer<std::uint16_t> product_bf16(kAssignments * kIntermediate),
      down_bf16(kAssignments * kWidth);
  DeviceBuffer<std::uint8_t> product_float_packed(
      kAssignments * kIntermediate / 2U),
      product_bf16_packed(kAssignments * kIntermediate / 2U),
      product_float_scales(kAssignments * kIntermediate / 16U),
      product_bf16_scales(kAssignments * kIntermediate / 16U);

  const auto copy_to_device = [](void* destination, const void* source,
                                 std::size_t bytes, const char* label) {
    CHECK(CudaOk(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice),
                 label));
  };
  copy_to_device(device_activation.get(),
                 quantized.value().packed_e2m1.data(),
                 device_activation.bytes(), "copy physical-BF16 activation");
  copy_to_device(device_activation_scales.get(),
                 quantized.value().block_scales_e4m3fn.data(),
                 device_activation_scales.bytes(),
                 "copy physical-BF16 activation scales");
  copy_to_device(gate_up_weights.get(), gate_up_host_weights.data(),
                 gate_up_weights.bytes(), "copy physical-BF16 W13 weights");
  copy_to_device(gate_up_scales.get(), gate_up_host_scales.data(),
                 gate_up_scales.bytes(), "copy physical-BF16 W13 scales");
  copy_to_device(down_weights.get(), down_host_weights.data(),
                 down_weights.bytes(), "copy physical-BF16 W2 weights");
  copy_to_device(down_scales.get(), down_host_scales.data(),
                 down_scales.bytes(), "copy physical-BF16 W2 scales");
  copy_to_device(device_assignments.get(), assignments.data(),
                 device_assignments.bytes(),
                 "copy physical-BF16 assignments");
  copy_to_device(device_permutation.get(), permutation.data(),
                 device_permutation.bytes(),
                 "copy physical-BF16 permutation");
  copy_to_device(device_prefix.get(), prefix.data(), device_prefix.bytes(),
                 "copy physical-BF16 prefix");
  copy_to_device(device_tiles.get(), tiles.data(), device_tiles.bytes(),
                 "copy physical-BF16 tile schedule");
  copy_to_device(device_tile_count.get(), &tile_count, sizeof(tile_count),
                 "copy physical-BF16 tile count");

  const auto run_float = [&]() {
    CHECK(gem16::internal::LaunchNvfp4Sm120GroupedExpertFusedGateUp(
              device_activation.get(), device_activation_scales.get(),
              gate_up_weights.get(), gate_up_scales.get(),
              device_assignments.get(), device_permutation.get(),
              device_prefix.get(), device_tiles.get(), device_tile_count.get(),
              product_float.get(), kAssignments, kIntermediate, kWidth,
              kExperts, 1.0F, 1.0F, nullptr)
              .ok());
    CHECK(gem16::internal::LaunchNvfp4ReferenceActivationQuantization(
              product_float.get(), product_float_packed.get(),
              product_float_scales.get(), kAssignments * kIntermediate, 1.0F,
              nullptr)
              .ok());
    CHECK(gem16::internal::LaunchNvfp4Sm120GroupedExpertDown(
              product_float_packed.get(), product_float_scales.get(),
              down_weights.get(), down_scales.get(), device_assignments.get(),
              device_permutation.get(), device_prefix.get(), device_tiles.get(),
              device_tile_count.get(), down_float.get(), kAssignments, kWidth,
              kIntermediate, kExperts, 1.0F, 1.0F, nullptr)
              .ok());
    CHECK(gem16::internal::LaunchGemma4MoeReduceAssignments(
              down_float.get(), device_assignments.get(), reduced_float.get(),
              kWidth, kTopK, kTokens, nullptr)
              .ok());
  };
  const auto run_bf16 = [&]() {
    CHECK(gem16::internal::LaunchNvfp4Sm120GroupedExpertFusedGateUpBf16(
              device_activation.get(), device_activation_scales.get(),
              gate_up_weights.get(), gate_up_scales.get(),
              device_assignments.get(), device_permutation.get(),
              device_prefix.get(), device_tiles.get(), device_tile_count.get(),
              product_bf16.get(), kAssignments, kIntermediate, kWidth,
              kExperts, 1.0F, 1.0F, nullptr)
              .ok());
    CHECK(gem16::internal::LaunchNvfp4ReferenceActivationQuantizationBf16(
              product_bf16.get(), product_bf16_packed.get(),
              product_bf16_scales.get(), kAssignments * kIntermediate, 1.0F,
              nullptr)
              .ok());
    CHECK(gem16::internal::LaunchNvfp4Sm120GroupedExpertDownBf16(
              product_bf16_packed.get(), product_bf16_scales.get(),
              down_weights.get(), down_scales.get(), device_assignments.get(),
              device_permutation.get(), device_prefix.get(), device_tiles.get(),
              device_tile_count.get(), down_bf16.get(), kAssignments, kWidth,
              kIntermediate, kExperts, 1.0F, 1.0F, nullptr)
              .ok());
    CHECK(gem16::internal::LaunchGemma4MoeReduceAssignmentsBf16(
              down_bf16.get(), device_assignments.get(), reduced_bf16.get(),
              kWidth, kTopK, kTokens, nullptr)
              .ok());
  };
  run_float();
  run_bf16();
  CHECK(CudaOk(cudaDeviceSynchronize(),
               "synchronize physical-BF16 grouped operators"));

  std::vector<float> host_product_float(kAssignments * kIntermediate),
      host_down_float(kAssignments * kWidth),
      host_reduced_float(kTokens * kWidth),
      host_reduced_bf16(kTokens * kWidth);
  std::vector<std::uint16_t> host_product_bf16(kAssignments * kIntermediate),
      host_down_bf16(kAssignments * kWidth);
  std::vector<std::uint8_t> host_product_float_packed(
      product_float_packed.bytes()),
      host_product_bf16_packed(product_bf16_packed.bytes()),
      host_product_float_scales(product_float_scales.bytes()),
      host_product_bf16_scales(product_bf16_scales.bytes());
  CHECK(CudaOk(cudaMemcpy(host_product_float.data(), product_float.get(),
                          product_float.bytes(), cudaMemcpyDeviceToHost),
               "copy float-container W13 product"));
  CHECK(CudaOk(cudaMemcpy(host_product_bf16.data(), product_bf16.get(),
                          product_bf16.bytes(), cudaMemcpyDeviceToHost),
               "copy physical-BF16 W13 product"));
  CHECK(CudaOk(cudaMemcpy(host_product_float_packed.data(),
                          product_float_packed.get(),
                          product_float_packed.bytes(), cudaMemcpyDeviceToHost),
               "copy float-container product NVFP4"));
  CHECK(CudaOk(cudaMemcpy(host_product_bf16_packed.data(),
                          product_bf16_packed.get(),
                          product_bf16_packed.bytes(), cudaMemcpyDeviceToHost),
               "copy physical-BF16 product NVFP4"));
  CHECK(CudaOk(cudaMemcpy(host_product_float_scales.data(),
                          product_float_scales.get(),
                          product_float_scales.bytes(), cudaMemcpyDeviceToHost),
               "copy float-container product scales"));
  CHECK(CudaOk(cudaMemcpy(host_product_bf16_scales.data(),
                          product_bf16_scales.get(),
                          product_bf16_scales.bytes(), cudaMemcpyDeviceToHost),
               "copy physical-BF16 product scales"));
  CHECK(CudaOk(cudaMemcpy(host_down_float.data(), down_float.get(),
                          down_float.bytes(), cudaMemcpyDeviceToHost),
               "copy float-container W2 output"));
  CHECK(CudaOk(cudaMemcpy(host_down_bf16.data(), down_bf16.get(),
                          down_bf16.bytes(), cudaMemcpyDeviceToHost),
               "copy physical-BF16 W2 output"));
  CHECK(CudaOk(cudaMemcpy(host_reduced_float.data(), reduced_float.get(),
                          reduced_float.bytes(), cudaMemcpyDeviceToHost),
               "copy float-container routed reduction"));
  CHECK(CudaOk(cudaMemcpy(host_reduced_bf16.data(), reduced_bf16.get(),
                          reduced_bf16.bytes(), cudaMemcpyDeviceToHost),
               "copy physical-BF16 routed reduction"));
  for (std::size_t index = 0; index < host_product_float.size(); ++index) {
    CHECK(Bf16(host_product_float[index]) == host_product_bf16[index]);
  }
  CHECK(host_product_float_packed == host_product_bf16_packed);
  CHECK(host_product_float_scales == host_product_bf16_scales);
  for (std::size_t index = 0; index < host_down_float.size(); ++index) {
    CHECK(Bf16(host_down_float[index]) == host_down_bf16[index]);
  }
  CHECK(host_reduced_float == host_reduced_bf16);

  std::size_t free_before = 0U;
  std::size_t total = 0U;
  CHECK(CudaOk(cudaMemGetInfo(&free_before, &total),
               "memory before physical-BF16 operator repeats"));
  for (int repeat = 0; repeat < 4; ++repeat) run_bf16();
  CHECK(CudaOk(cudaDeviceSynchronize(),
               "synchronize physical-BF16 operator repeats"));
  std::size_t free_after = 0U;
  CHECK(CudaOk(cudaMemGetInfo(&free_after, &total),
               "memory after physical-BF16 operator repeats"));
  CHECK(free_before == free_after);
}

void TestFixedAddressMoeReference() {
  constexpr std::uint64_t kWidth = 64;
  constexpr std::uint64_t kShared = 64;
  constexpr std::uint64_t kExpert = 64;
  constexpr std::uint32_t kExperts = 16;
  constexpr std::uint32_t kTopK = 8;

  DeviceBuffer<float> hidden(kWidth), output(kWidth);
  DeviceBuffer<std::uint16_t> norms(5U * kWidth);
  DeviceBuffer<std::uint16_t> router_scale(kWidth);
  DeviceBuffer<std::uint16_t> router_projection(kExperts * kWidth);
  DeviceBuffer<std::uint16_t> expert_scale(kExperts), layer_scalar(1);

  DeviceBuffer<std::uint8_t> shared_gate_weight(kShared * kWidth / 2U);
  DeviceBuffer<std::uint8_t> shared_gate_scales(kShared * kWidth / 16U);
  DeviceBuffer<std::uint8_t> shared_up_weight(kShared * kWidth / 2U);
  DeviceBuffer<std::uint8_t> shared_up_scales(kShared * kWidth / 16U);
  DeviceBuffer<std::uint8_t> shared_down_weight(kWidth * kShared / 2U);
  DeviceBuffer<std::uint8_t> shared_down_scales(kWidth * kShared / 16U);
  DeviceBuffer<std::uint8_t> expert_gate_up_weight(
      kExperts * 2U * kExpert * kWidth / 2U);
  DeviceBuffer<std::uint8_t> expert_gate_up_scales(
      kExperts * 2U * kExpert * kWidth / 16U);
  DeviceBuffer<std::uint8_t> expert_down_weight(
      kExperts * kWidth * kExpert / 2U);
  DeviceBuffer<std::uint8_t> expert_down_scales(
      kExperts * kWidth * kExpert / 16U);

  DeviceBuffer<float> shared_input(kWidth), shared_gate(kShared),
      shared_up(kShared), shared_product(kShared), shared_output(kWidth),
      shared_post(kWidth), router_normalized(kWidth),
      router_transformed(kWidth), router_logits(kExperts),
      router_probabilities(kExperts), expert_input(kWidth),
      expert_gate_up(kTopK * 2U * kExpert),
      expert_product(kTopK * kExpert), expert_down(kTopK * kWidth),
      expert_contributions(kTopK * kWidth), routed_sum(kWidth),
      routed_post(kWidth), combined(kWidth), feed_forward(kWidth);
  DeviceBuffer<std::uint8_t> shared_input_packed(kWidth / 2U),
      shared_input_scales(kWidth / 16U),
      shared_product_packed(kShared / 2U),
      shared_product_scales(kShared / 16U),
      expert_input_packed(kWidth / 2U),
      expert_input_scales(kWidth / 16U),
      expert_product_packed(kTopK * kExpert / 2U),
      expert_product_scales(kTopK * kExpert / 16U);
  DeviceBuffer<std::uint32_t> top_ids(kTopK);
  DeviceBuffer<float> top_weights(kTopK);
  DeviceBuffer<int> routing_finite(1);

  std::vector<float> host_hidden(kWidth);
  for (std::uint64_t index = 0; index < kWidth; ++index) {
    host_hidden[index] = static_cast<float>(__ushort_as_bfloat16(
        Bf16((static_cast<int>(index % 11U) - 5) * 0.125F)));
  }
  std::vector<std::uint16_t> host_norms(5U * kWidth, Bf16(1.0F));
  std::vector<std::uint16_t> host_router_scale(kWidth, Bf16(1.0F));
  std::vector<std::uint16_t> host_expert_scale(kExperts, Bf16(1.0F));
  const std::uint16_t host_scalar = Bf16(0.5F);
  CHECK(CudaOk(cudaMemcpy(hidden.get(), host_hidden.data(), hidden.bytes(),
                          cudaMemcpyHostToDevice),
               "copy hidden"));
  CHECK(CudaOk(cudaMemcpy(norms.get(), host_norms.data(), norms.bytes(),
                          cudaMemcpyHostToDevice),
               "copy norms"));
  CHECK(CudaOk(cudaMemcpy(router_scale.get(), host_router_scale.data(),
                          router_scale.bytes(), cudaMemcpyHostToDevice),
               "copy router scale"));
  CHECK(CudaOk(cudaMemset(router_projection.get(), 0,
                          router_projection.bytes()),
               "clear router projection"));
  CHECK(CudaOk(cudaMemcpy(expert_scale.get(), host_expert_scale.data(),
                          expert_scale.bytes(), cudaMemcpyHostToDevice),
               "copy expert scale"));
  CHECK(CudaOk(cudaMemcpy(layer_scalar.get(), &host_scalar,
                          sizeof(host_scalar), cudaMemcpyHostToDevice),
               "copy layer scalar"));
  for (auto* buffer : {&shared_gate_weight, &shared_gate_scales,
                       &shared_up_weight, &shared_up_scales,
                       &shared_down_weight, &shared_down_scales,
                       &expert_gate_up_weight, &expert_gate_up_scales,
                       &expert_down_weight, &expert_down_scales}) {
    CHECK(CudaOk(cudaMemset(buffer->get(), 0, buffer->bytes()),
                 "clear NVFP4 weights"));
  }

  const auto matrix = [](const std::uint8_t* packed,
                         const std::uint8_t* scales, std::uint64_t rows,
                         std::uint64_t columns) {
    return gem16::internal::Gemma4MoeNvfp4Matrix{
        packed, scales, rows, columns, 1.0F, 1.0F};
  };
  gem16::internal::Gemma4MoeReferenceWeights weights;
  weights.pre_shared_norm_bf16 = norms.get();
  weights.post_shared_norm_bf16 = norms.get() + kWidth;
  weights.pre_expert_norm_bf16 = norms.get() + 2U * kWidth;
  weights.post_expert_norm_bf16 = norms.get() + 3U * kWidth;
  weights.post_combined_norm_bf16 = norms.get() + 4U * kWidth;
  weights.router_scale_bf16 = router_scale.get();
  weights.router_projection_bf16 = router_projection.get();
  weights.per_expert_scale_bf16 = expert_scale.get();
  weights.layer_scalar_bf16 = layer_scalar.get();
  weights.shared_gate = matrix(shared_gate_weight.get(),
                               shared_gate_scales.get(), kShared, kWidth);
  weights.shared_up = matrix(shared_up_weight.get(), shared_up_scales.get(),
                             kShared, kWidth);
  weights.shared_down = matrix(shared_down_weight.get(),
                               shared_down_scales.get(), kWidth, kShared);
  weights.expert_gate_up = matrix(
      expert_gate_up_weight.get(), expert_gate_up_scales.get(),
      kExperts * 2U * kExpert, kWidth);
  weights.expert_down = matrix(expert_down_weight.get(),
                               expert_down_scales.get(), kExperts * kWidth,
                               kExpert);

  gem16::internal::Gemma4MoeReferenceWorkspace workspace{
      shared_input.get(), shared_input_packed.get(), shared_input_scales.get(),
      shared_gate.get(), shared_up.get(), shared_product.get(),
      shared_product_packed.get(), shared_product_scales.get(),
      shared_output.get(), shared_post.get(), router_normalized.get(),
      router_transformed.get(), router_logits.get(),
      router_probabilities.get(), top_ids.get(), top_weights.get(),
      expert_input.get(), expert_input_packed.get(), expert_input_scales.get(),
      expert_gate_up.get(), expert_product.get(), expert_product_packed.get(),
      expert_product_scales.get(), expert_down.get(),
      expert_contributions.get(), routed_sum.get(), routed_post.get(),
      combined.get(), feed_forward.get(), routing_finite.get()};
  const gem16::internal::Gemma4MoeReferenceConfig config{
      kWidth, kShared, kExpert, kExperts, kTopK, 1.0e-6F};

  const auto aliased_status = gem16::internal::LaunchGemma4MoeReferenceLayer(
      hidden.get(), hidden.get(), config, weights, workspace, nullptr);
  CHECK(!aliased_status.ok());

  auto run = [&]() {
    CHECK(CudaOk(cudaMemset(routing_finite.get(), 1, sizeof(int)),
                 "initialize M11 routing finite flag"));
    const auto status = gem16::internal::LaunchGemma4MoeReferenceLayer(
        hidden.get(), output.get(), config, weights, workspace, nullptr);
    CHECK(status.ok());
    CHECK(CudaOk(cudaDeviceSynchronize(), "synchronize M11 reference"));
  };
  run();  // Warm the CUDA runtime before the allocation observation.
  std::size_t free_before = 0U;
  std::size_t total = 0U;

  std::vector<float> first_output(kWidth), repeated_output(kWidth);
  std::vector<std::uint32_t> ids(kTopK);
  std::vector<float> probabilities(kExperts), selected_weights(kTopK);
  std::vector<std::uint8_t> reference_shared_packed(kWidth / 2U),
      reference_shared_scales(kWidth / 16U),
      reference_expert_packed(kWidth / 2U),
      reference_expert_scales(kWidth / 16U);
  std::vector<float> reference_router_normalized(kWidth),
      reference_router_transformed(kWidth);
  run();
  CHECK(CudaOk(cudaMemcpy(first_output.data(), output.get(), output.bytes(),
                          cudaMemcpyDeviceToHost),
               "copy first output"));
  CHECK(CudaOk(cudaMemcpy(ids.data(), top_ids.get(), top_ids.bytes(),
                          cudaMemcpyDeviceToHost),
               "copy IDs"));
  CHECK(CudaOk(cudaMemcpy(probabilities.data(), router_probabilities.get(),
                          router_probabilities.bytes(),
                          cudaMemcpyDeviceToHost),
               "copy probabilities"));
  CHECK(CudaOk(cudaMemcpy(selected_weights.data(), top_weights.get(),
                          top_weights.bytes(), cudaMemcpyDeviceToHost),
               "copy selected weights"));
  CHECK(CudaOk(cudaMemcpy(reference_shared_packed.data(),
                          shared_input_packed.get(),
                          shared_input_packed.bytes(), cudaMemcpyDeviceToHost),
               "copy reference shared-input NVFP4"));
  CHECK(CudaOk(cudaMemcpy(reference_shared_scales.data(),
                          shared_input_scales.get(),
                          shared_input_scales.bytes(), cudaMemcpyDeviceToHost),
               "copy reference shared-input scales"));
  CHECK(CudaOk(cudaMemcpy(reference_expert_packed.data(),
                          expert_input_packed.get(),
                          expert_input_packed.bytes(), cudaMemcpyDeviceToHost),
               "copy reference expert-input NVFP4"));
  CHECK(CudaOk(cudaMemcpy(reference_expert_scales.data(),
                          expert_input_scales.get(),
                          expert_input_scales.bytes(), cudaMemcpyDeviceToHost),
               "copy reference expert-input scales"));
  CHECK(CudaOk(cudaMemcpy(reference_router_normalized.data(),
                          router_normalized.get(), router_normalized.bytes(),
                          cudaMemcpyDeviceToHost),
               "copy reference router-normalized input"));
  CHECK(CudaOk(cudaMemcpy(reference_router_transformed.data(),
                          router_transformed.get(), router_transformed.bytes(),
                          cudaMemcpyDeviceToHost),
               "copy reference router-transformed input"));
  const auto native_status = gem16::internal::LaunchGemma4MoeSm120Layer(
      hidden.get(), output.get(), config, weights, workspace, nullptr);
  CHECK(native_status.ok());
  CHECK(CudaOk(cudaDeviceSynchronize(), "synchronize M14 native MoE"));
  std::vector<float> native_output(kWidth);
  std::vector<std::uint32_t> native_ids(kTopK);
  std::vector<std::uint8_t> native_shared_packed(kWidth / 2U),
      native_shared_scales(kWidth / 16U),
      native_expert_packed(kWidth / 2U),
      native_expert_scales(kWidth / 16U);
  std::vector<float> native_router_normalized(kWidth),
      native_router_transformed(kWidth);
  CHECK(CudaOk(cudaMemcpy(native_output.data(), output.get(), output.bytes(),
                          cudaMemcpyDeviceToHost),
               "copy M14 native output"));
  CHECK(CudaOk(cudaMemcpy(native_ids.data(), top_ids.get(), top_ids.bytes(),
                          cudaMemcpyDeviceToHost),
               "copy M14 native IDs"));
  CHECK(CudaOk(cudaMemcpy(native_shared_packed.data(), shared_input_packed.get(),
                          shared_input_packed.bytes(), cudaMemcpyDeviceToHost),
               "copy M14 shared-input NVFP4"));
  CHECK(CudaOk(cudaMemcpy(native_shared_scales.data(), shared_input_scales.get(),
                          shared_input_scales.bytes(), cudaMemcpyDeviceToHost),
               "copy M14 shared-input scales"));
  CHECK(CudaOk(cudaMemcpy(native_expert_packed.data(), expert_input_packed.get(),
                          expert_input_packed.bytes(), cudaMemcpyDeviceToHost),
               "copy M14 expert-input NVFP4"));
  CHECK(CudaOk(cudaMemcpy(native_expert_scales.data(), expert_input_scales.get(),
                          expert_input_scales.bytes(), cudaMemcpyDeviceToHost),
               "copy M14 expert-input scales"));
  CHECK(CudaOk(cudaMemcpy(native_router_normalized.data(),
                          router_normalized.get(), router_normalized.bytes(),
                          cudaMemcpyDeviceToHost),
               "copy M14 router-normalized input"));
  CHECK(CudaOk(cudaMemcpy(native_router_transformed.data(),
                          router_transformed.get(), router_transformed.bytes(),
                          cudaMemcpyDeviceToHost),
               "copy M14 router-transformed input"));
  CHECK(native_output == first_output);
  CHECK(native_ids == ids);
  CHECK(native_shared_packed == reference_shared_packed);
  CHECK(native_shared_scales == reference_shared_scales);
  CHECK(native_expert_packed == reference_expert_packed);
  CHECK(native_expert_scales == reference_expert_scales);
  CHECK(native_router_normalized == reference_router_normalized);
  CHECK(native_router_transformed == reference_router_transformed);
  cudaStream_t capture_stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  CHECK(CudaOk(cudaStreamCreateWithFlags(&capture_stream,
                                         cudaStreamNonBlocking),
               "create M14 capture stream"));
  CHECK(CudaOk(cudaStreamBeginCapture(capture_stream,
                                      cudaStreamCaptureModeThreadLocal),
               "begin M14 graph capture"));
  CHECK(CudaOk(cudaMemsetAsync(routing_finite.get(), 1, sizeof(int),
                               capture_stream),
               "capture M14 routing finite initialization"));
  const auto capture_status = gem16::internal::LaunchGemma4MoeSm120Layer(
      hidden.get(), output.get(), config, weights, workspace, capture_stream);
  CHECK(capture_status.ok());
  CHECK(CudaOk(cudaStreamEndCapture(capture_stream, &graph),
               "end M14 graph capture"));
  CHECK(CudaOk(cudaGraphInstantiate(&graph_exec, graph, 0),
               "instantiate M14 graph"));
  CHECK(CudaOk(cudaGraphLaunch(graph_exec, capture_stream),
               "launch M14 graph first"));
  CHECK(CudaOk(cudaGraphLaunch(graph_exec, capture_stream),
               "launch M14 graph replay"));
  CHECK(CudaOk(cudaStreamSynchronize(capture_stream),
               "synchronize M14 graph replay"));
  std::vector<float> graph_output(kWidth);
  CHECK(CudaOk(cudaMemcpy(graph_output.data(), output.get(), output.bytes(),
                          cudaMemcpyDeviceToHost),
               "copy M14 graph output"));
  CHECK(graph_output == first_output);
  if (graph_exec != nullptr) (void)cudaGraphExecDestroy(graph_exec);
  if (graph != nullptr) (void)cudaGraphDestroy(graph);
  if (capture_stream != nullptr) (void)cudaStreamDestroy(capture_stream);
  CHECK(CudaOk(cudaMemGetInfo(&free_before, &total), "memory before repeats"));
  for (int repeat = 0; repeat < 4; ++repeat) run();
  CHECK(CudaOk(cudaMemcpy(repeated_output.data(), output.get(), output.bytes(),
                          cudaMemcpyDeviceToHost),
               "copy repeated output"));
  std::size_t free_after = 0U;
  CHECK(CudaOk(cudaMemGetInfo(&free_after, &total), "memory after repeats"));

  CHECK(free_before == free_after);
  CHECK(first_output == repeated_output);
  for (std::uint32_t slot = 0; slot < kTopK; ++slot) {
    CHECK(ids[slot] == slot);  // exact tie: lower expert ID first
    CHECK(std::abs(selected_weights[slot] - 0.125F) < 1.0e-7F);
    CHECK(std::abs(probabilities[slot] - 0.0625F) < 1.0e-7F);
  }
  for (std::uint64_t index = 0; index < kWidth; ++index) {
    const float expected = static_cast<float>(__ushort_as_bfloat16(
        Bf16(host_hidden[index] * 0.5F)));
    CHECK(first_output[index] == expected);
  }

  // Exercise one full 16-assignment expert tile plus a tail, cross the
  // grouping algorithm's 256-assignment chunk boundary, and leave the
  // remaining expert range empty.
  constexpr std::uint64_t kTokens = 33U;
  DeviceBuffer<float> batch_hidden(kTokens * kWidth), batch_output(kTokens * kWidth),
      batch_router_logits(kTokens * kExperts),
      batch_router_probabilities(kTokens * kExperts),
      batch_token_hidden(kTokens * kWidth),
      batch_expert_product(kTokens * kTopK * kExpert),
      batch_expert_down(kTokens * kTopK * kWidth),
      batch_shared_product(kTokens * kShared),
      batch_shared_output(kTokens * kWidth),
      batch_reduced_output(kTokens * kWidth);
  DeviceBuffer<std::uint8_t> batch_token_packed(kTokens * kWidth / 2U),
      batch_token_scales(kTokens * kWidth / 16U),
      batch_expert_product_packed(kTokens * kTopK * kExpert / 2U),
      batch_expert_product_scales(kTokens * kTopK * kExpert / 16U),
      batch_shared_product_packed(kTokens * kShared / 2U),
      batch_shared_product_scales(kTokens * kShared / 16U);
  DeviceBuffer<gem16::internal::Gemma4MoePrefillAssignment> assignments(
      kTokens * kTopK);
  DeviceBuffer<std::uint32_t> histogram(kExperts), prefix(kExperts + 1U),
      permutation(kTokens * kTopK), inverse(kTokens * kTopK);
  DeviceBuffer<int> prefill_routing_finite(1);
  std::vector<float> host_batch_hidden(kTokens * kWidth);
  for (std::uint64_t token = 0; token < kTokens; ++token) {
    std::copy(host_hidden.begin(), host_hidden.end(),
              host_batch_hidden.begin() + token * kWidth);
  }
  CHECK(CudaOk(cudaMemcpy(batch_hidden.get(), host_batch_hidden.data(),
                          batch_hidden.bytes(), cudaMemcpyHostToDevice),
               "copy M15 batch hidden"));
  const gem16::internal::Gemma4MoePrefillWorkspace prefill_workspace{
      batch_router_logits.get(), batch_router_probabilities.get(),
      batch_token_hidden.get(), batch_token_packed.get(),
      batch_token_scales.get(), batch_expert_product.get(),
      batch_expert_product_packed.get(), batch_expert_product_scales.get(),
      batch_expert_down.get(), batch_shared_product.get(),
      batch_shared_product_packed.get(), batch_shared_product_scales.get(),
      batch_shared_output.get(), batch_reduced_output.get(), assignments.get(),
      histogram.get(), prefix.get(), permutation.get(), inverse.get(),
      prefill_routing_finite.get()};
  CHECK(CudaOk(cudaMemset(prefill_routing_finite.get(), 1, sizeof(int)),
               "initialize M15 routing finite flag"));
  const auto prefill_status = gem16::internal::LaunchGemma4MoeSm120PrefillLayer(
      batch_hidden.get(), batch_output.get(), kTokens, config, weights,
      prefill_workspace, nullptr);
  CHECK(prefill_status.ok());
  CHECK(CudaOk(cudaDeviceSynchronize(), "synchronize M15 grouped prefill"));
  std::vector<float> batch_values(kTokens * kWidth);
  std::vector<gem16::internal::Gemma4MoePrefillAssignment> assignment_values(
      kTokens * kTopK);
  std::vector<std::uint32_t> histogram_values(kExperts),
      prefix_values(kExperts + 1U), permutation_values(kTokens * kTopK),
      inverse_values(kTokens * kTopK);
  CHECK(CudaOk(cudaMemcpy(batch_values.data(), batch_output.get(),
                          batch_output.bytes(), cudaMemcpyDeviceToHost),
               "copy M15 output"));
  CHECK(CudaOk(cudaMemcpy(assignment_values.data(), assignments.get(),
                          assignments.bytes(), cudaMemcpyDeviceToHost),
               "copy M15 assignments"));
  CHECK(CudaOk(cudaMemcpy(histogram_values.data(), histogram.get(),
                          histogram.bytes(), cudaMemcpyDeviceToHost),
               "copy M15 histogram"));
  CHECK(CudaOk(cudaMemcpy(prefix_values.data(), prefix.get(), prefix.bytes(),
                          cudaMemcpyDeviceToHost),
               "copy M15 prefix"));
  CHECK(CudaOk(cudaMemcpy(permutation_values.data(), permutation.get(),
                          permutation.bytes(), cudaMemcpyDeviceToHost),
               "copy M15 permutation"));
  CHECK(CudaOk(cudaMemcpy(inverse_values.data(), inverse.get(), inverse.bytes(),
                          cudaMemcpyDeviceToHost),
               "copy M15 inverse"));
  for (std::uint64_t token = 0; token < kTokens; ++token) {
    for (std::uint64_t index = 0; index < kWidth; ++index) {
      CHECK(batch_values[token * kWidth + index] == first_output[index]);
    }
    for (std::uint32_t slot = 0; slot < kTopK; ++slot) {
      const auto& assignment = assignment_values[token * kTopK + slot];
      CHECK(assignment.token_id == token);
      CHECK(assignment.topk_slot == slot);
      CHECK(assignment.expert_id == slot);
      CHECK(std::abs(assignment.weight - 0.125F) < 1.0e-7F);
    }
  }
  for (std::uint32_t expert = 0; expert < kExperts; ++expert) {
    const std::uint32_t selected_experts =
        std::min(expert, static_cast<std::uint32_t>(kTopK));
    CHECK(histogram_values[expert] ==
          (expert < kTopK ? kTokens : 0U));
    CHECK(prefix_values[expert] == selected_experts * kTokens);
    if (expert >= kTopK) continue;
    for (std::uint32_t token = 0; token < kTokens; ++token) {
      const std::uint32_t grouped = expert * kTokens + token;
      const std::uint32_t original = token * kTopK + expert;
      CHECK(permutation_values[grouped] == original);
      CHECK(inverse_values[original] == grouped);
    }
  }
  CHECK(prefix_values[kExperts] == kTokens * kTopK);

  cudaStream_t prefill_capture_stream = nullptr;
  cudaGraph_t prefill_graph = nullptr;
  cudaGraphExec_t prefill_graph_exec = nullptr;
  CHECK(CudaOk(cudaStreamCreateWithFlags(&prefill_capture_stream,
                                         cudaStreamNonBlocking),
               "create M15 capture stream"));
  CHECK(CudaOk(cudaStreamBeginCapture(prefill_capture_stream,
                                      cudaStreamCaptureModeThreadLocal),
               "begin M15 graph capture"));
  CHECK(CudaOk(cudaMemsetAsync(prefill_routing_finite.get(), 1, sizeof(int),
                               prefill_capture_stream),
               "capture M15 routing finite initialization"));
  const auto prefill_capture_status =
      gem16::internal::LaunchGemma4MoeSm120PrefillLayer(
          batch_hidden.get(), batch_output.get(), kTokens, config, weights,
          prefill_workspace, prefill_capture_stream);
  CHECK(prefill_capture_status.ok());
  CHECK(CudaOk(cudaStreamEndCapture(prefill_capture_stream, &prefill_graph),
               "end M15 graph capture"));
  CHECK(CudaOk(cudaGraphInstantiate(&prefill_graph_exec, prefill_graph, 0),
               "instantiate M15 graph"));
  CHECK(CudaOk(cudaGraphLaunch(prefill_graph_exec, prefill_capture_stream),
               "launch M15 graph first"));
  CHECK(CudaOk(cudaGraphLaunch(prefill_graph_exec, prefill_capture_stream),
               "launch M15 graph replay"));
  CHECK(CudaOk(cudaStreamSynchronize(prefill_capture_stream),
               "synchronize M15 graph replay"));
  std::vector<float> prefill_graph_values(kTokens * kWidth);
  CHECK(CudaOk(cudaMemcpy(prefill_graph_values.data(), batch_output.get(),
                          batch_output.bytes(), cudaMemcpyDeviceToHost),
               "copy M15 graph output"));
  CHECK(prefill_graph_values == batch_values);
  if (prefill_graph_exec != nullptr) {
    (void)cudaGraphExecDestroy(prefill_graph_exec);
  }
  if (prefill_graph != nullptr) (void)cudaGraphDestroy(prefill_graph);
  if (prefill_capture_stream != nullptr) {
    (void)cudaStreamDestroy(prefill_capture_stream);
  }
  std::size_t prefill_free_before = 0U;
  CHECK(CudaOk(cudaMemGetInfo(&prefill_free_before, &total),
               "memory before M15 repeats"));
  for (int repeat = 0; repeat < 4; ++repeat) {
    CHECK(CudaOk(cudaMemset(prefill_routing_finite.get(), 1, sizeof(int)),
                 "initialize M15 repeat routing finite flag"));
    const auto repeat_status =
        gem16::internal::LaunchGemma4MoeSm120PrefillLayer(
            batch_hidden.get(), batch_output.get(), kTokens, config, weights,
            prefill_workspace, nullptr);
    CHECK(repeat_status.ok());
  }
  CHECK(CudaOk(cudaDeviceSynchronize(), "synchronize M15 repeats"));
  std::size_t prefill_free_after = 0U;
  CHECK(CudaOk(cudaMemGetInfo(&prefill_free_after, &total),
               "memory after M15 repeats"));
  CHECK(prefill_free_before == prefill_free_after);

  // Compare the coalesced prefill router's BF16-rounded logits against the
  // original serial expert/index order so tiling or vector-load drift cannot
  // hide behind the zero-router determinism fixture above.
  std::vector<std::uint16_t> differential_router(kExperts * kWidth);
  for (std::uint32_t expert = 0; expert < kExperts; ++expert) {
    for (std::uint64_t index = 0; index < kWidth; ++index) {
      const int centered =
          static_cast<int>((index * 17U + expert * 13U) % 31U) - 15;
      differential_router[static_cast<std::uint64_t>(expert) * kWidth +
                          index] = Bf16(static_cast<float>(centered) / 256.0F);
    }
  }
  CHECK(CudaOk(cudaMemcpy(router_projection.get(), differential_router.data(),
                          router_projection.bytes(), cudaMemcpyHostToDevice),
               "copy differential router projection"));
  CHECK(CudaOk(cudaMemset(routing_finite.get(), 1, sizeof(int)),
               "initialize differential router finite flag"));
  const auto differential_status = gem16::internal::LaunchGemma4MoeSm120Layer(
      hidden.get(), output.get(), config, weights, workspace, nullptr);
  CHECK(differential_status.ok());
  CHECK(CudaOk(cudaDeviceSynchronize(), "synchronize differential router"));
  std::vector<float> transformed(kWidth), warp_logits(kExperts);
  std::vector<std::uint32_t> differential_ids(kTopK);
  CHECK(CudaOk(cudaMemcpy(transformed.data(), router_transformed.get(),
                          router_transformed.bytes(), cudaMemcpyDeviceToHost),
               "copy differential router input"));
  CHECK(CudaOk(cudaMemcpy(warp_logits.data(), router_logits.get(),
                          router_logits.bytes(), cudaMemcpyDeviceToHost),
               "copy differential router logits"));
  CHECK(CudaOk(cudaMemcpy(differential_ids.data(), top_ids.get(),
                          top_ids.bytes(), cudaMemcpyDeviceToHost),
               "copy differential router IDs"));
  float maximum_router_error = 0.0F;
  for (std::uint32_t expert = 0; expert < kExperts; ++expert) {
    float serial = 0.0F;
    for (std::uint64_t index = 0; index < kWidth; ++index) {
      const float router_weight = static_cast<float>(__ushort_as_bfloat16(
          differential_router[static_cast<std::uint64_t>(expert) * kWidth +
                              index]));
      serial = std::fma(router_weight, transformed[index], serial);
    }
    const float expected =
        static_cast<float>(__ushort_as_bfloat16(Bf16(serial)));
    maximum_router_error =
        std::max(maximum_router_error, std::abs(warp_logits[expert] - expected));
  }
  std::cout << "warp-router vs serial BF16 max-abs=" << maximum_router_error
            << '\n';
  CHECK(maximum_router_error <= 1.0F / 512.0F);

  CHECK(CudaOk(cudaMemset(prefill_routing_finite.get(), 1, sizeof(int)),
               "initialize differential prefill router finite flag"));
  const auto differential_prefill_status =
      gem16::internal::LaunchGemma4MoeSm120PrefillLayer(
          batch_hidden.get(), batch_output.get(), kTokens, config, weights,
          prefill_workspace, nullptr);
  CHECK(differential_prefill_status.ok());
  CHECK(CudaOk(cudaDeviceSynchronize(),
               "synchronize differential prefill router"));
  CHECK(CudaOk(cudaMemcpy(assignment_values.data(), assignments.get(),
                          assignments.bytes(), cudaMemcpyDeviceToHost),
               "copy differential prefill assignments"));
  for (std::uint64_t token = 0U; token < kTokens; ++token) {
    for (std::uint32_t slot = 0U; slot < kTopK; ++slot) {
      CHECK(assignment_values[token * kTopK + slot].expert_id ==
            differential_ids[slot]);
    }
  }

  // A non-finite router result must be reported without using an invalid
  // expert ID or leaving stale assignments/workspace data behind.
  std::vector<std::uint16_t> nonfinite_router(kExperts * kWidth, Bf16(0.0F));
  nonfinite_router[0] = Bf16(INFINITY);
  CHECK(CudaOk(cudaMemcpy(router_projection.get(), nonfinite_router.data(),
                          router_projection.bytes(), cudaMemcpyHostToDevice),
               "copy non-finite router projection"));
  CHECK(CudaOk(cudaMemset(top_ids.get(), 0xff, top_ids.bytes()),
               "poison decode top IDs"));
  CHECK(CudaOk(cudaMemset(top_weights.get(), 0xff, top_weights.bytes()),
               "poison decode top weights"));
  CHECK(CudaOk(cudaMemset(routing_finite.get(), 1, sizeof(int)),
               "initialize non-finite decode flag"));
  const auto nonfinite_decode_status =
      gem16::internal::LaunchGemma4MoeSm120Layer(
          hidden.get(), output.get(), config, weights, workspace, nullptr);
  CHECK(nonfinite_decode_status.ok());
  CHECK(CudaOk(cudaDeviceSynchronize(), "synchronize non-finite decode"));
  int decode_finite = 1;
  CHECK(CudaOk(cudaMemcpy(&decode_finite, routing_finite.get(), sizeof(int),
                          cudaMemcpyDeviceToHost),
               "copy non-finite decode flag"));
  CHECK(CudaOk(cudaMemcpy(ids.data(), top_ids.get(), top_ids.bytes(),
                          cudaMemcpyDeviceToHost),
               "copy non-finite decode IDs"));
  CHECK(CudaOk(cudaMemcpy(selected_weights.data(), top_weights.get(),
                          top_weights.bytes(), cudaMemcpyDeviceToHost),
               "copy non-finite decode weights"));
  CHECK(decode_finite == 0);
  for (std::uint32_t slot = 0; slot < kTopK; ++slot) {
    CHECK(ids[slot] == 0U);
    CHECK(selected_weights[slot] == 0.0F);
  }

  CHECK(CudaOk(cudaMemset(assignments.get(), 0xff, assignments.bytes()),
               "poison prefill assignments"));
  CHECK(CudaOk(cudaMemset(permutation.get(), 0xff, permutation.bytes()),
               "poison prefill permutation"));
  CHECK(CudaOk(cudaMemset(prefill_routing_finite.get(), 1, sizeof(int)),
               "initialize non-finite prefill flag"));
  const auto nonfinite_prefill_status =
      gem16::internal::LaunchGemma4MoeSm120PrefillLayer(
          batch_hidden.get(), batch_output.get(), kTokens, config, weights,
          prefill_workspace, nullptr);
  CHECK(nonfinite_prefill_status.ok());
  CHECK(CudaOk(cudaDeviceSynchronize(), "synchronize non-finite prefill"));
  int prefill_finite = 1;
  CHECK(CudaOk(cudaMemcpy(&prefill_finite, prefill_routing_finite.get(),
                          sizeof(int), cudaMemcpyDeviceToHost),
               "copy non-finite prefill flag"));
  CHECK(CudaOk(cudaMemcpy(assignment_values.data(), assignments.get(),
                          assignments.bytes(), cudaMemcpyDeviceToHost),
               "copy non-finite prefill assignments"));
  CHECK(CudaOk(cudaMemcpy(permutation_values.data(), permutation.get(),
                          permutation.bytes(), cudaMemcpyDeviceToHost),
               "copy non-finite prefill permutation"));
  CHECK(prefill_finite == 0);
  for (std::uint64_t index = 0; index < kTokens * kTopK; ++index) {
    CHECK(assignment_values[index].token_id < kTokens);
    CHECK(assignment_values[index].topk_slot < kTopK);
    CHECK(assignment_values[index].expert_id < kExperts);
    CHECK(assignment_values[index].weight == 0.0F);
    CHECK(permutation_values[index] < kTokens * kTopK);
  }
}

}  // namespace

int main() {
  int devices = 0;
  if (!CudaOk(cudaGetDeviceCount(&devices), "cudaGetDeviceCount") ||
      devices == 0) {
    return 1;
  }
  TestFixedAddressMoeReference();
  TestSelectedExpertSlotBatch();
  TestPhysicalBf16GroupedExpertOperators();
  if (failures != 0) {
    std::cerr << failures << " M11 CUDA assertion(s) failed\n";
    return 1;
  }
  std::cout << "M11/M14 decode and M15 grouped CUDA MoE tests passed\n";
  return 0;
}

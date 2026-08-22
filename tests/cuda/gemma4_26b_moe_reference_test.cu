#include "cuda/moe/reference.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
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

void TestFixedAddressMoeReference() {
  constexpr std::uint64_t kWidth = 64;
  constexpr std::uint64_t kShared = 64;
  constexpr std::uint64_t kExpert = 64;
  constexpr std::uint32_t kExperts = 8;
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
      combined.get(), feed_forward.get()};
  const gem16::internal::Gemma4MoeReferenceConfig config{
      kWidth, kShared, kExpert, kExperts, kTopK, 1.0e-6F};

  auto run = [&]() {
    const auto status = gem16::internal::LaunchGemma4MoeReferenceLayer(
        hidden.get(), output.get(), config, weights, workspace, nullptr);
    CHECK(status.ok());
    CHECK(CudaOk(cudaDeviceSynchronize(), "synchronize M11 reference"));
  };
  run();  // Warm the CUDA runtime before the allocation observation.
  std::size_t free_before = 0U;
  std::size_t total = 0U;
  CHECK(CudaOk(cudaMemGetInfo(&free_before, &total), "memory before repeats"));

  std::vector<float> first_output(kWidth), repeated_output(kWidth);
  std::vector<std::uint32_t> ids(kTopK);
  std::vector<float> probabilities(kExperts), selected_weights(kTopK);
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
    CHECK(std::abs(probabilities[slot] - 0.125F) < 1.0e-7F);
  }
  for (std::uint64_t index = 0; index < kWidth; ++index) {
    const float expected = static_cast<float>(__ushort_as_bfloat16(
        Bf16(host_hidden[index] * 0.5F)));
    CHECK(first_output[index] == expected);
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
  if (failures != 0) {
    std::cerr << failures << " M11 CUDA assertion(s) failed\n";
    return 1;
  }
  std::cout << "M11 correctness-only CUDA MoE tests passed\n";
  return 0;
}

#include "cuda/attention/gemma4_26b_reference.h"
#include "cuda/attention/sm120.h"
#include "cuda/layer/reference.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
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
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::uint64_t elements) { Allocate(elements); }
  ~DeviceBuffer() {
    if (pointer_ != nullptr) (void)cudaFree(pointer_);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  void Allocate(std::uint64_t elements) {
    elements_ = elements;
    if (!CudaOk(cudaMalloc(&pointer_, elements * sizeof(T)), "cudaMalloc")) {
      pointer_ = nullptr;
    }
  }
  T* get() const { return static_cast<T*>(pointer_); }
  std::uint64_t elements() const { return elements_; }
  std::uint64_t bytes() const { return elements_ * sizeof(T); }

 private:
  void* pointer_ = nullptr;
  std::uint64_t elements_ = 0U;
};

std::uint16_t Bf16(float value) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

std::uint8_t Fp8(float value) {
  const __nv_fp8_e4m3 encoded(value);
  return encoded.__x;
}

void FillBf16(DeviceBuffer<std::uint16_t>* buffer, float value) {
  const std::vector<std::uint16_t> host(buffer->elements(), Bf16(value));
  CHECK(CudaOk(cudaMemcpy(buffer->get(), host.data(), buffer->bytes(),
                          cudaMemcpyHostToDevice),
               "copy BF16 fixture"));
}

struct Case {
  static constexpr std::uint64_t kHidden = 2816U;

  explicit Case(const gem16::internal::Gemma4Moe26BAttentionLayerTraits& value,
                std::uint64_t cache_tokens)
      : traits(value),
        q_elements(traits.query_heads * traits.head_dimension),
        kv_elements(traits.kv_heads * traits.head_dimension),
        hidden(kHidden), output(kHidden),
        input_norm(kHidden), post_norm(kHidden),
        q_norm(traits.head_dimension), k_norm(traits.head_dimension),
        cache_scales(2U), q_weight(q_elements * kHidden),
        q_scale(q_elements), k_weight(kv_elements * kHidden),
        k_scale(kv_elements),
        v_weight(traits.stores_v_projection ? kv_elements * kHidden : 1U),
        v_scale(traits.stores_v_projection ? kv_elements : 1U),
        o_weight(kHidden * q_elements), o_scale(kHidden), input_fp8(kHidden),
        input_scale(1U), q_raw(q_elements), k_raw(kv_elements),
        v_raw(kv_elements), q_normalized(q_elements),
        k_normalized(kv_elements), v_normalized(kv_elements),
        cosine(traits.head_dimension / 2U), sine(traits.head_dimension / 2U),
        staged_k(kv_elements), staged_v(kv_elements),
        scores(traits.query_heads * cache_tokens), attention(q_elements),
        output_fp8(q_elements), dynamic_o_scale(1U),
        output_projection(kHidden), post_attention(kHidden),
        key_cache(cache_tokens * kv_elements),
        value_cache(cache_tokens * kv_elements) {
    std::vector<float> host_hidden(kHidden);
    for (std::uint64_t index = 0; index < kHidden; ++index) {
      host_hidden[index] = static_cast<float>(__ushort_as_bfloat16(
          Bf16(index == 0U ? 2.0F
                           : (static_cast<int>(index % 17U) - 8) * 0.03125F)));
    }
    CHECK(CudaOk(cudaMemcpy(hidden.get(), host_hidden.data(), hidden.bytes(),
                            cudaMemcpyHostToDevice),
                 "copy M12 hidden"));
    FillBf16(&input_norm, 1.0F);
    FillBf16(&post_norm, 1.0F);
    FillBf16(&q_norm, 1.0F);
    FillBf16(&k_norm, 1.0F);
    FillBf16(&cache_scales, 1.0F);
    FillBf16(&q_scale, 1.0F);
    FillBf16(&k_scale, 1.0F);
    FillBf16(&o_scale, 1.0F);
    if (traits.stores_v_projection) FillBf16(&v_scale, 1.0F);
    for (auto* buffer : {&q_weight, &k_weight, &v_weight, &o_weight}) {
      CHECK(CudaOk(cudaMemset(buffer->get(), 0, buffer->bytes()),
                   "clear M12 FP8 weight"));
    }
    const std::uint8_t one = Fp8(1.0F);
    CHECK(CudaOk(cudaMemcpy(q_weight.get(), &one, 1U, cudaMemcpyHostToDevice),
                 "seed Q weight"));
    CHECK(CudaOk(cudaMemcpy(k_weight.get(), &one, 1U, cudaMemcpyHostToDevice),
                 "seed K weight"));
    if (traits.stores_v_projection) {
      const std::uint8_t half = Fp8(0.5F);
      CHECK(CudaOk(cudaMemcpy(v_weight.get(), &half, 1U,
                              cudaMemcpyHostToDevice),
                   "seed V weight"));
    }
    CHECK(CudaOk(cudaMemset(key_cache.get(), 0, key_cache.bytes()),
                 "clear K cache"));
    CHECK(CudaOk(cudaMemset(value_cache.get(), 0, value_cache.bytes()),
                 "clear V cache"));

    weights.input_norm_bf16 = input_norm.get();
    weights.post_attention_norm_bf16 = post_norm.get();
    weights.query_norm_bf16 = q_norm.get();
    weights.key_norm_bf16 = k_norm.get();
    weights.key_cache_scale_bf16 = cache_scales.get();
    weights.value_cache_scale_bf16 = cache_scales.get() + 1U;
    weights.query = {q_weight.get(), q_scale.get(), q_elements, kHidden};
    weights.key = {k_weight.get(), k_scale.get(), kv_elements, kHidden};
    if (traits.stores_v_projection) {
      weights.value = {v_weight.get(), v_scale.get(), kv_elements, kHidden};
    }
    weights.output = {o_weight.get(), o_scale.get(), kHidden, q_elements};
    workspace = {
        input_fp8.get(), input_scale.get(), q_raw.get(), k_raw.get(),
        v_raw.get(), q_normalized.get(), k_normalized.get(),
        v_normalized.get(), cosine.get(), sine.get(), staged_k.get(),
        staged_v.get(), scores.get(), scores.elements(), attention.get(),
        output_fp8.get(), dynamic_o_scale.get(), output_projection.get(),
        post_attention.get()};
    cache = {key_cache.get(), value_cache.get(), cache_tokens};
  }

  gem16::Status Run(std::uint64_t position) {
    return gem16::internal::LaunchGemma4Moe26BAttentionReferenceLayer(
        hidden.get(), output.get(), position, traits, weights, cache, workspace,
        1.0e-6F, nullptr);
  }

  gem16::internal::Gemma4Moe26BAttentionLayerTraits traits;
  std::uint64_t q_elements;
  std::uint64_t kv_elements;
  DeviceBuffer<float> hidden, output;
  DeviceBuffer<std::uint16_t> input_norm, post_norm, q_norm, k_norm,
      cache_scales;
  DeviceBuffer<std::uint8_t> q_weight;
  DeviceBuffer<std::uint16_t> q_scale;
  DeviceBuffer<std::uint8_t> k_weight;
  DeviceBuffer<std::uint16_t> k_scale;
  DeviceBuffer<std::uint8_t> v_weight;
  DeviceBuffer<std::uint16_t> v_scale;
  DeviceBuffer<std::uint8_t> o_weight;
  DeviceBuffer<std::uint16_t> o_scale;
  DeviceBuffer<std::uint8_t> input_fp8;
  DeviceBuffer<float> input_scale, q_raw, k_raw, v_raw, q_normalized,
      k_normalized, v_normalized, cosine, sine;
  DeviceBuffer<std::uint8_t> staged_k, staged_v;
  DeviceBuffer<float> scores, attention;
  DeviceBuffer<std::uint8_t> output_fp8;
  DeviceBuffer<float> dynamic_o_scale, output_projection, post_attention;
  DeviceBuffer<std::uint8_t> key_cache, value_cache;
  gem16::internal::Gemma4Moe26BAttentionReferenceWeights weights;
  gem16::internal::Gemma4Moe26BAttentionReferenceWorkspace workspace;
  gem16::internal::Gemma4Moe26BKvCacheView cache;
};

gem16::internal::Gemma4Moe26BAttentionLayerTraits LocalTraits() {
  return {0U,
          gem16::internal::Gemma4Moe26BAttentionType::kSliding,
          16U,
          8U,
          256U,
          1024U,
          true,
          false,
          gem16::internal::Gemma4Moe26BKvSource::kOwnedProjection,
          0,
          10000.0,
          1.0,
          1.0};
}

gem16::internal::Gemma4Moe26BAttentionLayerTraits GlobalTraits() {
  return {5U,
          gem16::internal::Gemma4Moe26BAttentionType::kFull,
          16U,
          2U,
          512U,
          262144U,
          false,
          true,
          gem16::internal::Gemma4Moe26BKvSource::kOwnedProjection,
          5,
          1000000.0,
          0.25,
          1.0};
}

void CheckCase(Case* test, std::uint64_t first_position,
               std::uint64_t second_position) {
  CHECK(test->Run(first_position).ok());
  CHECK(CudaOk(cudaDeviceSynchronize(), "warm M12 attention"));
  std::size_t free_before = 0U;
  std::size_t total = 0U;
  CHECK(CudaOk(cudaMemGetInfo(&free_before, &total), "M12 memory before"));
  CHECK(test->Run(second_position).ok());
  CHECK(CudaOk(cudaDeviceSynchronize(), "run M12 attention"));
  std::vector<float> first_output(Case::kHidden);
  CHECK(CudaOk(cudaMemcpy(first_output.data(), test->output.get(),
                          test->output.bytes(), cudaMemcpyDeviceToHost),
               "copy M12 output"));
  CHECK(test->Run(second_position).ok());
  CHECK(CudaOk(cudaDeviceSynchronize(), "repeat M12 attention"));
  std::vector<float> repeated_output(Case::kHidden);
  CHECK(CudaOk(cudaMemcpy(repeated_output.data(), test->output.get(),
                          test->output.bytes(), cudaMemcpyDeviceToHost),
               "copy repeated M12 output"));
  std::size_t free_after = 0U;
  CHECK(CudaOk(cudaMemGetInfo(&free_after, &total), "M12 memory after"));
  CHECK(free_before == free_after);
  CHECK(first_output == repeated_output);

  std::vector<std::uint8_t> key(test->kv_elements);
  std::vector<std::uint8_t> value(test->kv_elements);
  const std::uint64_t slot = second_position % test->cache.capacity;
  CHECK(CudaOk(cudaMemcpy(key.data(),
                          test->key_cache.get() + slot * test->kv_elements,
                          key.size(), cudaMemcpyDeviceToHost),
               "copy committed M12 K"));
  CHECK(CudaOk(cudaMemcpy(value.data(),
                          test->value_cache.get() + slot * test->kv_elements,
                          value.size(), cudaMemcpyDeviceToHost),
               "copy committed M12 V"));
  CHECK(key != value);

  auto alias = test->cache;
  alias.value = alias.key;
  CHECK(!gem16::internal::LaunchGemma4Moe26BAttentionReferenceLayer(
             test->hidden.get(), test->output.get(), second_position,
             test->traits, test->weights, alias, test->workspace, 1.0e-6F,
             nullptr)
             .ok());
}

void TestLocalRingAndGlobalAppend() {
  {
    Case local(LocalTraits(), 1024U);
    CheckCase(&local, 1023U, 1024U);
  }
  {
    Case global(GlobalTraits(), 8U);
    CHECK(global.weights.value.weight_e4m3 == nullptr);
    CheckCase(&global, 0U, 1U);
  }
}

void CheckNativeTwoKvHeadGlobalDecode(std::uint64_t capacity,
                                      std::uint64_t position,
                                      const char* tier) {
  constexpr std::uint64_t kQueryHeads = 16U;
  constexpr std::uint64_t kKvHeads = 2U;
  constexpr std::uint64_t kHeadDimension = 512U;
  constexpr std::uint64_t kQueryElements = kQueryHeads * kHeadDimension;
  constexpr std::uint64_t kKvElements = kKvHeads * kHeadDimension;
  DeviceBuffer<float> query(kQueryElements), reference_scores(
      kQueryHeads * (position + 1U)), reference_output(kQueryElements),
      native_workspace(
          gem16::internal::DecodeAttentionWorkspaceElements(capacity)),
      native_output(kQueryElements);
  DeviceBuffer<std::uint8_t> key_cache(capacity * kKvElements),
      value_cache(capacity * kKvElements);
  DeviceBuffer<std::uint16_t> scales(2U);
  DeviceBuffer<gem16::internal::DecodeControl> control(1U);
  std::vector<float> host_query(kQueryElements);
  for (std::uint64_t index = 0; index < host_query.size(); ++index) {
    const int centered = static_cast<int>((index * 17U + 5U) % 31U) - 15;
    host_query[index] = static_cast<float>(__ushort_as_bfloat16(
        Bf16(static_cast<float>(centered) / 32.0F)));
  }
  std::vector<std::uint8_t> host_key(capacity * kKvElements),
      host_value(capacity * kKvElements);
  for (std::uint64_t index = 0; index < host_key.size(); ++index) {
    const int key_value =
        static_cast<int>((index * 13U + index / 512U) % 15U) - 7;
    const int value =
        static_cast<int>((index * 7U + index / 1024U) % 17U) - 8;
    host_key[index] = Fp8(static_cast<float>(key_value) / 16.0F);
    host_value[index] = Fp8(static_cast<float>(value) / 16.0F);
  }
  const std::array<std::uint16_t, 2> host_scales{Bf16(0.75F), Bf16(1.25F)};
  const gem16::internal::DecodeControl host_control{0U, 0U, position,
                                                    position};
  CHECK(CudaOk(cudaMemcpy(query.get(), host_query.data(), query.bytes(),
                          cudaMemcpyHostToDevice),
               "copy M17 two-KV query"));
  CHECK(CudaOk(cudaMemcpy(key_cache.get(), host_key.data(), key_cache.bytes(),
                          cudaMemcpyHostToDevice),
               "copy M17 two-KV key cache"));
  CHECK(CudaOk(cudaMemcpy(value_cache.get(), host_value.data(),
                          value_cache.bytes(), cudaMemcpyHostToDevice),
               "copy M17 two-KV value cache"));
  CHECK(CudaOk(cudaMemcpy(scales.get(), host_scales.data(), scales.bytes(),
                          cudaMemcpyHostToDevice),
               "copy M17 two-KV scales"));
  CHECK(CudaOk(cudaMemcpy(control.get(), &host_control, sizeof(host_control),
                          cudaMemcpyHostToDevice),
               "copy M17 two-KV control"));
  const std::uint8_t* current_key =
      key_cache.get() + position * kKvElements;
  const std::uint8_t* current_value =
      value_cache.get() + position * kKvElements;
  CHECK(gem16::internal::LaunchCausalAttentionPrefillFp8(
            query.get(), current_key, current_value, key_cache.get(),
            value_cache.get(), scales.get(), scales.get() + 1U,
            reference_scores.get(), reference_output.get(), position, 1U,
            kQueryHeads, kKvHeads, kHeadDimension, capacity, false, nullptr)
            .ok());
  CHECK(gem16::internal::LaunchOnlineAttentionDecodeFp8Sm120(
            query.get(), key_cache.get(), value_cache.get(), scales.get(),
            scales.get() + 1U, native_workspace.get(), native_output.get(),
            control.get(), kQueryHeads, kKvHeads, kHeadDimension, capacity,
            false, nullptr)
            .ok());
  CHECK(CudaOk(cudaDeviceSynchronize(), "run M17 two-KV global attention"));
  std::vector<float> reference(kQueryElements), native(kQueryElements);
  CHECK(CudaOk(cudaMemcpy(reference.data(), reference_output.get(),
                          reference_output.bytes(), cudaMemcpyDeviceToHost),
               "copy M17 reference output"));
  CHECK(CudaOk(cudaMemcpy(native.data(), native_output.get(),
                          native_output.bytes(), cudaMemcpyDeviceToHost),
               "copy M17 native output"));
  double error_squared = 0.0, reference_squared = 0.0, dot = 0.0,
         native_squared = 0.0;
  for (std::size_t index = 0; index < reference.size(); ++index) {
    const double difference =
        static_cast<double>(native[index]) - reference[index];
    error_squared += difference * difference;
    reference_squared += static_cast<double>(reference[index]) * reference[index];
    native_squared +=
        static_cast<double>(native[index]) * native[index];
    dot += static_cast<double>(reference[index]) * native[index];
  }
  const double relative_l2 = std::sqrt(error_squared / reference_squared);
  const double cosine = dot / std::sqrt(reference_squared * native_squared);
  std::cout << "M17 QH16/KVH2/D512 " << tier
            << " relative-L2=" << relative_l2 << " cosine=" << cosine
            << '\n';
  CHECK(relative_l2 < 0.01);
  CHECK(cosine > 0.9999);
}

void TestNativeTwoKvHeadGlobalDecode() {
  CheckNativeTwoKvHeadGlobalDecode(1024U, 777U, "scalar");
  CheckNativeTwoKvHeadGlobalDecode(16384U, 12345U, "vectorized");
}

void TestNativeLocalSlidingDecode() {
  constexpr std::uint64_t kQueryHeads = 16U;
  constexpr std::uint64_t kKvHeads = 8U;
  constexpr std::uint64_t kHeadDimension = 256U;
  constexpr std::uint64_t kCapacity = 1024U;
  constexpr std::uint64_t kPosition = 1100U;
  constexpr std::uint64_t kQueryElements = kQueryHeads * kHeadDimension;
  constexpr std::uint64_t kKvElements = kKvHeads * kHeadDimension;
  DeviceBuffer<float> query(kQueryElements),
      reference_scores(kQueryHeads * kCapacity),
      reference_output(kQueryElements),
      native_workspace(
          gem16::internal::DecodeAttentionWorkspaceElements(kCapacity)),
      native_output(kQueryElements);
  DeviceBuffer<std::uint8_t> key_cache(kCapacity * kKvElements),
      value_cache(kCapacity * kKvElements);
  DeviceBuffer<std::uint16_t> scales(2U);
  DeviceBuffer<gem16::internal::DecodeControl> control(1U);
  std::vector<float> host_query(kQueryElements);
  for (std::uint64_t index = 0; index < host_query.size(); ++index) {
    const int centered = static_cast<int>((index * 19U + 3U) % 29U) - 14;
    host_query[index] = static_cast<float>(__ushort_as_bfloat16(
        Bf16(static_cast<float>(centered) / 32.0F)));
  }
  std::vector<std::uint8_t> host_key(kCapacity * kKvElements),
      host_value(kCapacity * kKvElements);
  for (std::uint64_t index = 0; index < host_key.size(); ++index) {
    const int key_value =
        static_cast<int>((index * 11U + index / 256U) % 15U) - 7;
    const int value =
        static_cast<int>((index * 5U + index / 2048U) % 17U) - 8;
    host_key[index] = Fp8(static_cast<float>(key_value) / 16.0F);
    host_value[index] = Fp8(static_cast<float>(value) / 16.0F);
  }
  const std::array<std::uint16_t, 2> host_scales{Bf16(0.75F), Bf16(1.25F)};
  const gem16::internal::DecodeControl host_control{0U, 0U, kPosition,
                                                    kPosition};
  CHECK(CudaOk(cudaMemcpy(query.get(), host_query.data(), query.bytes(),
                          cudaMemcpyHostToDevice),
               "copy M20 local query"));
  CHECK(CudaOk(cudaMemcpy(key_cache.get(), host_key.data(), key_cache.bytes(),
                          cudaMemcpyHostToDevice),
               "copy M20 local key cache"));
  CHECK(CudaOk(cudaMemcpy(value_cache.get(), host_value.data(),
                          value_cache.bytes(), cudaMemcpyHostToDevice),
               "copy M20 local value cache"));
  CHECK(CudaOk(cudaMemcpy(scales.get(), host_scales.data(), scales.bytes(),
                          cudaMemcpyHostToDevice),
               "copy M20 local scales"));
  CHECK(CudaOk(cudaMemcpy(control.get(), &host_control, sizeof(host_control),
                          cudaMemcpyHostToDevice),
               "copy M20 local control"));
  const std::uint64_t slot = kPosition % kCapacity;
  const std::uint8_t* current_key =
      key_cache.get() + slot * kKvElements;
  const std::uint8_t* current_value =
      value_cache.get() + slot * kKvElements;
  CHECK(gem16::internal::LaunchCausalAttentionPrefillFp8(
            query.get(), current_key, current_value, key_cache.get(),
            value_cache.get(), scales.get(), scales.get() + 1U,
            reference_scores.get(), reference_output.get(), kPosition, 1U,
            kQueryHeads, kKvHeads, kHeadDimension, kCapacity, true, nullptr)
            .ok());
  CHECK(gem16::internal::LaunchOnlineAttentionDecodeFp8Sm120(
            query.get(), key_cache.get(), value_cache.get(), scales.get(),
            scales.get() + 1U, native_workspace.get(), native_output.get(),
            control.get(), kQueryHeads, kKvHeads, kHeadDimension, kCapacity,
            true, nullptr)
            .ok());
  CHECK(CudaOk(cudaDeviceSynchronize(), "run M20 local sliding attention"));
  std::vector<float> reference(kQueryElements), native(kQueryElements);
  CHECK(CudaOk(cudaMemcpy(reference.data(), reference_output.get(),
                          reference_output.bytes(), cudaMemcpyDeviceToHost),
               "copy M20 local reference output"));
  CHECK(CudaOk(cudaMemcpy(native.data(), native_output.get(),
                          native_output.bytes(), cudaMemcpyDeviceToHost),
               "copy M20 local native output"));
  double error_squared = 0.0, reference_squared = 0.0, dot = 0.0,
         native_squared = 0.0;
  for (std::size_t index = 0; index < reference.size(); ++index) {
    const double difference =
        static_cast<double>(native[index]) - reference[index];
    error_squared += difference * difference;
    reference_squared += static_cast<double>(reference[index]) * reference[index];
    native_squared += static_cast<double>(native[index]) * native[index];
    dot += static_cast<double>(reference[index]) * native[index];
  }
  const double relative_l2 = std::sqrt(error_squared / reference_squared);
  const double cosine = dot / std::sqrt(reference_squared * native_squared);
  std::cout << "M20 QH16/KVH8/D256 local relative-L2=" << relative_l2
            << " cosine=" << cosine << '\n';
  CHECK(relative_l2 < 0.01);
  CHECK(cosine > 0.9999);
}

}  // namespace

int main() {
  int devices = 0;
  if (!CudaOk(cudaGetDeviceCount(&devices), "cudaGetDeviceCount") ||
      devices == 0) {
    return 1;
  }
  TestLocalRingAndGlobalAppend();
  TestNativeTwoKvHeadGlobalDecode();
  TestNativeLocalSlidingDecode();
  if (failures != 0) {
    std::cerr << failures << " M12 CUDA assertion(s) failed\n";
    return 1;
  }
  std::cout << "M12 26B attention/KV reference tests passed\n";
  return 0;
}

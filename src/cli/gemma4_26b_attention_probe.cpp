#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cuda/attention/gemma4_26b_reference.h"
#include "cuda/engine/gemma4_26b_artifact.h"
#include "gem16/model.h"
#include "model/config.h"
#include "model/gemma4_26b_residency.h"
#include "util/json.h"

namespace {

struct Options {
  std::filesystem::path model;
  std::filesystem::path fixture;
  std::filesystem::path output;
  int device = 0;
};

bool Parse(int argc, char** argv, Options* options) {
  for (int index = 1; index < argc; ++index) {
    if (index + 1 >= argc) return false;
    const std::string_view key(argv[index]);
    const std::string value(argv[++index]);
    if (key == "--model") options->model = value;
    else if (key == "--fixture") options->fixture = value;
    else if (key == "--output") options->output = value;
    else if (key == "--device") {
      try {
        std::size_t used = 0U;
        options->device = std::stoi(value, &used);
        if (used != value.size() || options->device < 0) return false;
      } catch (...) { return false; }
    } else return false;
  }
  return !options->model.empty() && !options->fixture.empty() &&
         !options->output.empty();
}

gem16::Result<std::string> Read(const std::filesystem::path& path) {
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  const auto size = std::filesystem::file_size(path, error);
  if (error || !std::filesystem::is_regular_file(status) ||
      std::filesystem::is_symlink(status) || size == 0U ||
      size > 4U * 1024U * 1024U) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         "M12 fixture is unsafe or outside its 4 MiB bound");
  }
  std::string result(static_cast<std::size_t>(size), '\0');
  std::ifstream input(path, std::ios::binary);
  if (!input.read(result.data(), static_cast<std::streamsize>(result.size()))) {
    return gem16::Status(gem16::StatusCode::kIoError,
                         "cannot read M12 fixture");
  }
  return result;
}

gem16::Result<std::vector<float>> Floats(const gem16::json::Value* value,
                                         std::uint64_t expected) {
  if (value == nullptr || !value->is_array() ||
      value->as_array().size() != expected) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         "M12 fixture array shape mismatch");
  }
  std::vector<float> result;
  result.reserve(static_cast<std::size_t>(expected));
  for (const auto& element : value->as_array()) {
    if (!element.is_number() || !std::isfinite(element.as_number())) {
      return gem16::Status(gem16::StatusCode::kDataLoss,
                           "M12 fixture contains invalid float");
    }
    result.push_back(static_cast<float>(element.as_number()));
  }
  return result;
}

class Buffer {
 public:
  Buffer() = default;
  ~Buffer() { if (data_ != nullptr) (void)cudaFree(data_); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  gem16::Status Allocate(std::uint64_t bytes) {
    if (bytes == 0U || bytes > std::numeric_limits<std::size_t>::max()) {
      return {gem16::StatusCode::kInvalidArgument, "invalid M12 buffer size"};
    }
    const cudaError_t error = cudaMalloc(&data_, static_cast<std::size_t>(bytes));
    if (error != cudaSuccess) {
      return {gem16::StatusCode::kResourceExhausted,
              std::string("allocate M12 buffer: ") + cudaGetErrorString(error)};
    }
    bytes_ = bytes;
    return gem16::Status::Ok();
  }
  template <typename T> T* As() const { return static_cast<T*>(data_); }
  std::uint64_t bytes() const { return bytes_; }
 private:
  void* data_ = nullptr;
  std::uint64_t bytes_ = 0U;
};

template <typename T>
gem16::Status Allocate(Buffer* buffer, std::uint64_t elements) {
  if (elements > std::numeric_limits<std::uint64_t>::max() / sizeof(T)) {
    return {gem16::StatusCode::kInvalidArgument, "M12 buffer overflow"};
  }
  return buffer->Allocate(elements * sizeof(T));
}

struct Buffers {
  Buffer hidden, output, input_fp8, input_scale, q_raw, k_raw, v_raw,
      q_norm, k_norm, v_norm, cosine, sine, staged_k, staged_v, scores,
      attention, output_fp8, dynamic_o_scale, output_projection,
      post_attention, key_cache, value_cache;
};

gem16::Status AllocateBuffers(
    Buffers* b, const gem16::internal::Gemma4Moe26BAttentionLayerTraits& t,
    std::uint64_t capacity) {
  constexpr std::uint64_t hidden = 2816U;
  const std::uint64_t q = t.query_heads * t.head_dimension;
  const std::uint64_t kv = t.kv_heads * t.head_dimension;
  for (const auto& status : {
           Allocate<float>(&b->hidden, hidden), Allocate<float>(&b->output, hidden),
           Allocate<std::uint8_t>(&b->input_fp8, hidden),
           Allocate<float>(&b->input_scale, 1U), Allocate<float>(&b->q_raw, q),
           Allocate<float>(&b->k_raw, kv), Allocate<float>(&b->v_raw, kv),
           Allocate<float>(&b->q_norm, q), Allocate<float>(&b->k_norm, kv),
           Allocate<float>(&b->v_norm, kv),
           Allocate<float>(&b->cosine, t.head_dimension / 2U),
           Allocate<float>(&b->sine, t.head_dimension / 2U),
           Allocate<std::uint8_t>(&b->staged_k, kv),
           Allocate<std::uint8_t>(&b->staged_v, kv),
           Allocate<float>(&b->scores, t.query_heads * capacity),
           Allocate<float>(&b->attention, q),
           Allocate<std::uint8_t>(&b->output_fp8, q),
           Allocate<float>(&b->dynamic_o_scale, 1U),
           Allocate<float>(&b->output_projection, hidden),
           Allocate<float>(&b->post_attention, hidden),
           Allocate<std::uint8_t>(&b->key_cache, capacity * kv),
           Allocate<std::uint8_t>(&b->value_cache, capacity * kv)}) {
    if (!status.ok()) return status;
  }
  return gem16::Status::Ok();
}

struct Metrics { double max_abs = 0.0, relative_l2 = 0.0, cosine = 0.0; };

gem16::Result<Metrics> Compare(const Buffer& actual,
                               const gem16::json::Value* expected_value,
                               std::uint64_t elements) {
  auto expected = Floats(expected_value, elements);
  if (!expected.ok()) return expected.status();
  std::vector<float> host(static_cast<std::size_t>(elements));
  const cudaError_t copied = cudaMemcpy(host.data(), actual.As<float>(),
                                         elements * sizeof(float),
                                         cudaMemcpyDeviceToHost);
  if (copied != cudaSuccess) {
    return gem16::Status(gem16::StatusCode::kInternal,
                         std::string("copy M12 boundary: ") +
                             cudaGetErrorString(copied));
  }
  double error2 = 0.0, reference2 = 0.0, actual2 = 0.0, dot = 0.0;
  Metrics result;
  for (std::size_t i = 0; i < host.size(); ++i) {
    const double a = host[i], e = expected.value()[i], delta = a - e;
    result.max_abs = std::max(result.max_abs, std::abs(delta));
    error2 += delta * delta;
    reference2 += e * e;
    actual2 += a * a;
    dot += a * e;
  }
  result.relative_l2 = std::sqrt(error2 / std::max(reference2, 1.0e-30));
  result.cosine = dot / std::sqrt(std::max(actual2 * reference2, 1.0e-30));
  return result;
}

void WriteMetrics(std::ostream& out, const Metrics& metrics) {
  out << std::setprecision(12) << "{\"max_abs\":" << metrics.max_abs
      << ",\"relative_l2\":" << metrics.relative_l2
      << ",\"cosine\":" << metrics.cosine << '}';
}

int Main(int argc, char** argv) {
  Options options;
  if (!Parse(argc, argv, &options)) return 2;
  if (cudaSetDevice(options.device) != cudaSuccess) return 3;
  auto text = Read(options.fixture);
  if (!text.ok()) { std::cerr << text.status().message() << '\n'; return 3; }
  auto fixture = gem16::json::Parse(text.value());
  if (!fixture.ok()) { std::cerr << fixture.status().message() << '\n'; return 3; }
  const auto* cases = fixture.value().find("cases");
  if (cases == nullptr || !cases->is_array() || cases->as_array().size() != 2U) {
    std::cerr << "invalid M12 fixture cases\n"; return 3;
  }
  auto config = gem16::internal::LoadModelConfig(options.model / "config.json");
  if (!config.ok()) { std::cerr << config.status().message() << '\n'; return 4; }
  auto traits = gem16::internal::BuildGemma4Moe26BAttentionTraits(config.value());
  if (!traits.ok()) { std::cerr << traits.status().message() << '\n'; return 4; }
  auto manifest = gem16::InspectCheckpoint({options.model, true});
  if (!manifest.ok()) { std::cerr << manifest.status().message() << '\n'; return 4; }
  auto plan = gem16::internal::BuildGemma4Moe26BResidencyPlan(
      manifest.value(), config.value());
  if (!plan.ok()) { std::cerr << plan.status().message() << '\n'; return 4; }
  auto artifact = gem16::internal::Gemma4Moe26BDeviceArtifact::Load(
      options.model, manifest.value(), plan.value());
  if (!artifact.ok()) { std::cerr << artifact.status().message() << '\n'; return 5; }

  std::ofstream report(options.output, std::ios::binary | std::ios::trunc);
  if (!report) return 6;
  report << "{\"schema_version\":1,\"milestone\":\"M12\","
         << "\"artifact_arena_bytes\":" << artifact.value().arena_bytes()
         << ",\"cases\":[";
  bool first_case = true;
  for (const auto& case_value : cases->as_array()) {
    const auto* layer_value = case_value.find("layer");
    const auto* position_value = case_value.find("position");
    const auto* hidden_value = case_value.find("hidden_f32");
    const auto* expected = case_value.find("expected");
    if (layer_value == nullptr || !layer_value->is_integer() ||
        position_value == nullptr || !position_value->is_integer() ||
        expected == nullptr || !expected->is_object()) return 7;
    const auto layer = static_cast<std::uint64_t>(layer_value->as_integer());
    const auto position = static_cast<std::uint64_t>(position_value->as_integer());
    if (layer >= traits.value().size() || position != 0U ||
        (layer != 0U && layer != 5U)) return 7;
    const auto& t = traits.value()[layer];
    const std::uint64_t capacity =
        t.attention == gem16::internal::Gemma4Moe26BAttentionType::kSliding
            ? t.cache_capacity : 1U;
    Buffers b;
    auto allocated = AllocateBuffers(&b, t, capacity);
    if (!allocated.ok()) { std::cerr << allocated.message() << '\n'; return 8; }
    auto hidden = Floats(hidden_value, 2816U);
    if (!hidden.ok()) return 7;
    if (cudaMemcpy(b.hidden.As<float>(), hidden.value().data(), b.hidden.bytes(),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemset(b.key_cache.As<std::uint8_t>(), 0, b.key_cache.bytes()) !=
            cudaSuccess ||
        cudaMemset(b.value_cache.As<std::uint8_t>(), 0, b.value_cache.bytes()) !=
            cudaSuccess) return 8;
    auto weights = gem16::internal::BindGemma4Moe26BAttentionReferenceWeights(
        artifact.value(), t);
    if (!weights.ok()) { std::cerr << weights.status().message() << '\n'; return 8; }
    gem16::internal::Gemma4Moe26BAttentionReferenceWorkspace workspace{
        b.input_fp8.As<std::uint8_t>(), b.input_scale.As<float>(),
        b.q_raw.As<float>(), b.k_raw.As<float>(), b.v_raw.As<float>(),
        b.q_norm.As<float>(), b.k_norm.As<float>(), b.v_norm.As<float>(),
        b.cosine.As<float>(), b.sine.As<float>(), b.staged_k.As<std::uint8_t>(),
        b.staged_v.As<std::uint8_t>(), b.scores.As<float>(),
        t.query_heads * capacity, b.attention.As<float>(),
        b.output_fp8.As<std::uint8_t>(), b.dynamic_o_scale.As<float>(),
        b.output_projection.As<float>(), b.post_attention.As<float>()};
    const gem16::internal::Gemma4Moe26BKvCacheView cache{
        b.key_cache.As<std::uint8_t>(), b.value_cache.As<std::uint8_t>(), capacity};
    auto run = [&]() {
      return gem16::internal::LaunchGemma4Moe26BAttentionReferenceLayer(
          b.hidden.As<float>(), b.output.As<float>(), position, t,
          weights.value(), cache, workspace, 1.0e-6F, nullptr);
    };
    auto status = run();
    if (!status.ok() || cudaDeviceSynchronize() != cudaSuccess) return 9;
    std::size_t free_before = 0U, total = 0U;
    if (cudaMemGetInfo(&free_before, &total) != cudaSuccess) return 9;
    std::vector<float> first_output(2816U), repeated_output(2816U);
    for (int repeat = 0; repeat < 4; ++repeat) {
      status = run();
      if (!status.ok()) return 9;
    }
    if (cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(first_output.data(), b.output.As<float>(), b.output.bytes(),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return 9;
    status = run();
    if (!status.ok() || cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(repeated_output.data(), b.output.As<float>(), b.output.bytes(),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return 9;
    std::size_t free_after = 0U;
    if (cudaMemGetInfo(&free_after, &total) != cudaSuccess) return 9;

    const std::uint64_t q = t.query_heads * t.head_dimension;
    const std::uint64_t kv = t.kv_heads * t.head_dimension;
    const std::vector<std::pair<std::string_view, std::pair<const Buffer*, std::uint64_t>>>
        boundaries = {
            {"q_raw", {&b.q_raw, q}}, {"k_raw", {&b.k_raw, kv}},
            {"q_normalized", {&b.q_norm, q}},
            {"k_normalized", {&b.k_norm, kv}},
            {"v_normalized", {&b.v_norm, kv}},
            {"attention", {&b.attention, q}},
            {"output_projection", {&b.output_projection, 2816U}},
            {"post_attention", {&b.post_attention, 2816U}}};
    if (!first_case) report << ',';
    first_case = false;
    report << "{\"layer\":" << layer
           << ",\"attention_type\":\""
           << (t.attention == gem16::internal::Gemma4Moe26BAttentionType::kSliding
                   ? "sliding" : "full")
           << "\",\"stores_v_projection\":"
           << (t.stores_v_projection ? "true" : "false")
           << ",\"reuses_raw_k_for_v\":"
           << (t.reuses_raw_k_for_v ? "true" : "false")
           << ",\"separate_cache_addresses\":"
           << (cache.key != cache.value ? "true" : "false")
           << ",\"repeated_bitwise_identical\":"
           << (first_output == repeated_output ? "true" : "false")
           << ",\"free_before_repeats_bytes\":" << free_before
           << ",\"free_after_repeats_bytes\":" << free_after
           << ",\"metrics\":{";
    bool first_metric = true;
    for (const auto& [name, binding] : boundaries) {
      auto metrics = Compare(*binding.first, expected->find(name), binding.second);
      if (!metrics.ok()) { std::cerr << metrics.status().message() << '\n'; return 10; }
      if (!first_metric) report << ',';
      first_metric = false;
      report << '"' << name << "\":";
      WriteMetrics(report, metrics.value());
    }
    if (t.stores_v_projection) {
      auto metrics = Compare(b.v_raw, expected->find("v_raw"), kv);
      if (!metrics.ok()) return 10;
      report << ",\"v_raw\":";
      WriteMetrics(report, metrics.value());
    }
    report << "}}";
  }
  report << "]}\n";
  return report ? 0 : 11;
}

}  // namespace

int main(int argc, char** argv) { return Main(argc, argv); }

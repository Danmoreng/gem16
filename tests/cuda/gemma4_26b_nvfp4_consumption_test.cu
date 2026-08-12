#include "compiler/nvfp4_batch_encoder.h"
#include "compiler/sha256.h"
#include "cuda/nvfp4/reference.h"
#include "cuda/nvfp4/sm120.h"
#include "cuda/nvfp4/sm120_layout.h"
#include "gem16/nvfp4.h"
#include "util/json.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr int kCTestSkip = 77;
constexpr float kActivationGlobalDivisor = 2.0F;
constexpr float kWeightGlobalDivisor = 4.0F;

int failures = 0;

void Check(bool condition, const char* expression, int line) {
  if (!condition) {
    std::cerr << __FILE__ << ':' << line << ": check failed: " << expression << '\n';
    ++failures;
  }
}

#define NVFP4_TEST_CHECK(expression) Check(static_cast<bool>(expression), #expression, __LINE__)

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
    if (!CudaOk(cudaMalloc(reinterpret_cast<void**>(&data_), bytes()), "cudaMalloc")) {
      data_ = nullptr;
    }
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) (void)cudaFree(data_);
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] T* get() const { return data_; }
  [[nodiscard]] std::size_t bytes() const { return elements_ * sizeof(T); }

 private:
  T* data_ = nullptr;
  std::size_t elements_ = 0;
};

struct NativeMatrix {
  std::string source_name;
  std::size_t rows;
  std::size_t contracting_elements;
  std::vector<std::uint8_t> packed;
  std::vector<std::uint8_t> scales;
  float weight_divisor = 0.0F;
  float input_divisor = 0.0F;
};

std::uint16_t Bf16Bits(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16U) & 1U)) >> 16U);
}

void WriteBytes(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
}

std::vector<std::uint8_t> ReadBytes(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(stream), {}};
}

void WriteZeros(const std::filesystem::path& path, std::size_t bytes) {
  std::ofstream stream(path, std::ios::binary);
  const std::array<char, 1U << 16U> zeros{};
  while (bytes != 0U) {
    const auto count = std::min<std::size_t>(bytes, zeros.size());
    stream.write(zeros.data(), static_cast<std::streamsize>(count));
    bytes -= count;
  }
}

std::vector<std::uint8_t> MakeBf16Source(std::size_t rows, std::size_t columns,
                                         std::uint8_t seed) {
  std::vector<std::uint8_t> bytes(rows * columns * 2U);
  for (std::size_t index = 0; index < rows * columns; ++index) {
    const int centered = static_cast<int>((index * 37U + seed * 19U) % 257U) - 128;
    const float value = static_cast<float>(centered) / 64.0F +
                        static_cast<float>(static_cast<int>(index % 7U) - 3) / 512.0F;
    const std::uint16_t bits = Bf16Bits(value);
    bytes[index * 2U] = static_cast<std::uint8_t>(bits);
    bytes[index * 2U + 1U] = static_cast<std::uint8_t>(bits >> 8U);
  }
  return bytes;
}

std::string HashBytes(const std::vector<std::uint8_t>& bytes) {
  return gem16::compiler::Sha256Hex(bytes.data(), bytes.size());
}

float LoadFloat(const std::vector<std::uint8_t>& bytes) {
  NVFP4_TEST_CHECK(bytes.size() == sizeof(float));
  std::uint32_t bits = static_cast<std::uint32_t>(bytes[0]) |
                       (static_cast<std::uint32_t>(bytes[1]) << 8U) |
                       (static_cast<std::uint32_t>(bytes[2]) << 16U) |
                       (static_cast<std::uint32_t>(bytes[3]) << 24U);
  return std::bit_cast<float>(bits);
}

std::string ComponentOutput(std::string_view component, std::string_view name,
                            const std::filesystem::path& path, std::size_t bytes) {
  return "{\"component\":" + gem16::json::Quote(component) +
         ",\"name\":" + gem16::json::Quote(name) +
         ",\"path\":" + gem16::json::Quote(path.string()) +
         ",\"offset\":0,\"bytes\":" + std::to_string(bytes) + "}";
}

NativeMatrix CompileFixture(const std::filesystem::path& root, std::string source_name,
                            std::size_t rows, std::size_t columns, std::uint8_t seed,
                            std::vector<std::uint8_t>* source_bytes) {
  std::filesystem::create_directories(root);
  const auto source_path = root / "source.bf16";
  *source_bytes = MakeBf16Source(rows, columns, seed);
  WriteBytes(source_path, *source_bytes);
  const auto stem = source_name.ends_with(".weight")
                        ? source_name.substr(0, source_name.size() - 7U)
                        : source_name;
  const auto packed_path = root / "packed.bin";
  const auto scales_path = root / "scales.bin";
  const auto weight_path = root / "weight.bin";
  const auto input_path = root / "input.bin";
  const std::size_t elements = rows * columns;
  WriteZeros(packed_path, elements / 2U);
  WriteZeros(scales_path, elements / 16U);
  WriteZeros(weight_path, sizeof(float));
  WriteZeros(input_path, sizeof(float));
  const std::string operation = "fixture:" + stem;
  const std::string role = source_name.find("gate_up_proj") != std::string::npos
                               ? "routed_expert_gate_up"
                               : source_name.find("experts.down_proj") != std::string::npos
                                     ? "routed_expert_down"
                                     : source_name.find("down_proj") != std::string::npos
                                           ? "shared_mlp_down"
                                           : source_name.find("gate_proj") != std::string::npos
                                                 ? "shared_mlp_gate"
                                                 : "shared_mlp_up";
  const std::string axis = role == "routed_expert_gate_up"
                               ? "expert,gate_then_up,input"
                               : role == "routed_expert_down" ? "expert,output,input"
                                                               : "output,input";
  const std::string runtime = role.starts_with("routed_")
                                  ? "expert_major_sm120_row8_k64" : "sm120_row8_k64";
  const std::string shape = source_name.find("experts.") != std::string::npos
                                ? "[1," + std::to_string(rows) + "," + std::to_string(columns) + "]"
                                : "[" + std::to_string(rows) + "," + std::to_string(columns) + "]";
  const std::size_t packed_bytes = elements / 2U;
  const std::size_t scale_bytes = elements / 16U;
  const auto job_path = root / "job.json";
  std::ofstream job(job_path);
  job << "{\"schema_version\":1,\"protocol\":\"gem16-nvfp4-direct-v1\","
         "\"artifact_profile\":\"nvfp4-experts-partial-v1\",\"scope\":\"fixture\","
         "\"contract_id\":\"gem16.nvfp4_bf16_group16\",\"contract_version\":1,"
         "\"threads\":1,\"operations\":[{\"operation_id\":"
      << gem16::json::Quote(operation) << ",\"source_name\":"
      << gem16::json::Quote(source_name) << ",\"source_path\":"
      << gem16::json::Quote(source_path.string()) << ",\"source_sha256\":"
      << gem16::json::Quote(HashBytes(*source_bytes)) << ",\"source_offset\":0,"
         "\"source_bytes\":"
      << source_bytes->size() << ",\"source_dtype\":\"BF16\",\"logical_shape\":"
      << shape << ",\"rows\":" << rows << ",\"columns\":" << columns
      << ",\"role\":" << gem16::json::Quote(role)
      << ",\"axis_transformation\":" << gem16::json::Quote(axis)
      << ",\"disk_layout\":\"canonical_row_major_low_nibble_first\","
         "\"runtime_layout\":" << gem16::json::Quote(runtime)
      << ",\"packed\":" << ComponentOutput("packed", stem + ".weight_packed", packed_path, packed_bytes)
      << ",\"local_scale\":" << ComponentOutput("local_scale", stem + ".weight_scale", scales_path, scale_bytes)
      << ",\"weight_global\":" << ComponentOutput("weight_global", stem + ".weight_global_scale", weight_path, sizeof(float))
      << ",\"input_global\":" << ComponentOutput("input_global", stem + ".input_global_scale", input_path, sizeof(float))
      << "}]}\n";
  job.close();
  const auto telemetry_path = root / "telemetry.json";
  const auto status = gem16::compiler::EncodeNvfp4JobFile(job_path, telemetry_path);
  NVFP4_TEST_CHECK(status.ok());
  if (!status.ok()) {
    std::cerr << "fixture native compiler failed: " << status.message() << '\n';
    return NativeMatrix{source_name, rows, columns, {}, {}, 0.0F, 0.0F};
  }
  NativeMatrix result{source_name, rows, columns, ReadBytes(packed_path),
                      ReadBytes(scales_path), LoadFloat(ReadBytes(weight_path)),
                      LoadFloat(ReadBytes(input_path))};
  NVFP4_TEST_CHECK(result.packed.size() == packed_bytes);
  NVFP4_TEST_CHECK(result.scales.size() == scale_bytes);
  NVFP4_TEST_CHECK(std::isfinite(result.weight_divisor) && result.weight_divisor > 0.0F);
  NVFP4_TEST_CHECK(result.input_divisor == 1.0F);
  NVFP4_TEST_CHECK(result.weight_divisor != 4.0F);
  const auto telemetry_bytes = ReadBytes(telemetry_path);
  const std::string telemetry_text(telemetry_bytes.begin(), telemetry_bytes.end());
  const auto parsed = gem16::json::Parse(telemetry_text);
  NVFP4_TEST_CHECK(parsed.ok());
  if (parsed.ok()) {
    const auto& root_value = parsed.value();
    NVFP4_TEST_CHECK(root_value.find("protocol") != nullptr &&
                     root_value.find("protocol")->is_string() &&
                     root_value.find("protocol")->as_string() == "gem16-nvfp4-direct-v1");
    NVFP4_TEST_CHECK(root_value.find("scope") != nullptr &&
                     root_value.find("scope")->as_string() == "fixture");
    NVFP4_TEST_CHECK(root_value.find("source_passes") != nullptr &&
                     root_value.find("source_passes")->as_integer() == 2);
    const auto* operations = root_value.find("operations");
    NVFP4_TEST_CHECK(operations != nullptr && operations->is_array() && operations->as_array().size() == 1U);
    if (operations != nullptr && operations->is_array() && !operations->as_array().empty()) {
      const auto& item = operations->as_array()[0];
      NVFP4_TEST_CHECK(item.find("packed_sha256")->as_string() == HashBytes(result.packed));
      NVFP4_TEST_CHECK(item.find("local_scale_sha256")->as_string() == HashBytes(result.scales));
      NVFP4_TEST_CHECK(item.find("weight_global_sha256")->as_string() == HashBytes(ReadBytes(weight_path)));
      NVFP4_TEST_CHECK(item.find("input_global_sha256")->as_string() == HashBytes(ReadBytes(input_path)));
    }
  }
  return result;
}

float RoundBf16(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t rounded = bits + 0x7FFFU + ((bits >> 16U) & 1U);
  return std::bit_cast<float>(rounded & 0xFFFF0000U);
}

float GeluTanh(float value) {
  constexpr float kSqrtTwoOverPi = 0.7978845608028654F;
  constexpr float kCubic = 0.044715F;
  return 0.5F * value *
         (1.0F + std::tanh(kSqrtTwoOverPi * (value + kCubic * value * value * value)));
}

bool CloseEnough(float actual, double expected, double absolute_limit,
                 double relative_limit) {
  if (!std::isfinite(actual) || !std::isfinite(expected)) return false;
  const double difference = std::fabs(static_cast<double>(actual) - expected);
  return difference <= std::max(absolute_limit, std::fabs(expected) * relative_limit);
}

void TestMalformedHostInputs() {
  NVFP4_TEST_CHECK(!gem16::internal::PlanSm120Nvfp4SourceLayout(8U, 48U).ok());
  const auto valid_layout = gem16::internal::PlanSm120Nvfp4SourceLayout(8U, 64U);
  NVFP4_TEST_CHECK(valid_layout.ok());
  if (!valid_layout.ok()) return;

  std::vector<std::uint8_t> packed(valid_layout.value().packed_weight_bytes, 0x12U);
  std::vector<std::uint8_t> scales(valid_layout.value().scale_bytes, 0x38U);
  NVFP4_TEST_CHECK(!gem16::internal::TileSm120Nvfp4Weights(
                              valid_layout.value(), std::span<const std::uint8_t>(packed).first(1U))
                        .ok());
  NVFP4_TEST_CHECK(!gem16::internal::TileSm120Nvfp4WeightScales(
                              valid_layout.value(), std::span<const std::uint8_t>(scales).first(1U))
                        .ok());

  const std::array<float, 16> activation_values = {
      -1.0F, -0.75F, -0.5F, -0.25F, 0.0F, 0.25F, 0.5F, 0.75F,
      1.0F,  1.25F,  1.5F,  1.75F,  2.0F, 2.25F, 2.5F, 2.75F,
  };
  const auto activation = gem16::nvfp4::QuantizeActivation(
      activation_values, kActivationGlobalDivisor);
  NVFP4_TEST_CHECK(activation.ok());
  if (!activation.ok()) return;
  const std::array<std::uint8_t, 8> weight = {0x12U, 0x34U, 0x56U, 0x78U,
                                                0x9AU, 0xBCU, 0xDEU, 0xF0U};
  std::array<std::uint8_t, 1> invalid_scale = {0x7FU};
  NVFP4_TEST_CHECK(!gem16::nvfp4::ReferenceDotProduct(
                              activation.value(), weight, invalid_scale, kWeightGlobalDivisor)
                        .ok());
  NVFP4_TEST_CHECK(!gem16::nvfp4::ReferenceDotProduct(
                              activation.value(), weight, std::array<std::uint8_t, 1>{0x38U}, 0.0F)
                        .ok());
}

std::vector<float> MakeActivation(std::size_t contracting_elements, std::uint8_t seed) {
  std::vector<float> activation(contracting_elements);
  for (std::size_t k = 0; k < contracting_elements; ++k) {
    const int centered = static_cast<int>(
                             (k * 37U + static_cast<std::size_t>(seed) * 19U) % 257U) -
                         128;
    activation[k] = static_cast<float>(centered) / 64.0F +
                    static_cast<float>(static_cast<int>(k % 7U) - 3) / 512.0F;
  }
  return activation;
}

void RunProjection(std::string_view label, std::size_t rows, std::size_t contracting_elements,
                   const std::vector<std::uint8_t>& packed,
                   const std::vector<std::uint8_t>& scales, float weight_divisor,
                   float input_divisor, std::uint8_t seed,
                   const std::vector<std::uint8_t>* up_packed = nullptr,
                   const std::vector<std::uint8_t>* up_scales = nullptr) {
  const auto layout = gem16::internal::PlanSm120Nvfp4SourceLayout(rows, contracting_elements);
  NVFP4_TEST_CHECK(layout.ok());
  if (!layout.ok()) return;
  const std::uint64_t logical_elements = static_cast<std::uint64_t>(rows) * contracting_elements;
  NVFP4_TEST_CHECK(packed.size() == logical_elements / 2U);
  NVFP4_TEST_CHECK(scales.size() == logical_elements / 16U);
  NVFP4_TEST_CHECK(layout.value().packed_weight_bytes == logical_elements / 2U);
  NVFP4_TEST_CHECK(layout.value().scale_bytes == logical_elements / 16U);
  NVFP4_TEST_CHECK(layout.value().persistent_repack_bytes == 0U);
  const auto activation = gem16::nvfp4::QuantizeActivation(
      MakeActivation(contracting_elements, seed), input_divisor);
  NVFP4_TEST_CHECK(activation.ok());
  if (!activation.ok()) return;
  const auto tiled_weight = gem16::internal::TileSm120Nvfp4Weights(layout.value(), packed);
  const auto tiled_scales = gem16::internal::TileSm120Nvfp4WeightScales(layout.value(), scales);
  NVFP4_TEST_CHECK(tiled_weight.ok());
  NVFP4_TEST_CHECK(tiled_scales.ok());
  if (!tiled_weight.ok() || !tiled_scales.ok()) return;
  const bool fused = up_packed != nullptr && up_scales != nullptr;
  NVFP4_TEST_CHECK(!fused || (up_packed->size() == packed.size() && up_scales->size() == scales.size()));
  if (fused && (up_packed->size() != packed.size() || up_scales->size() != scales.size())) return;
  const auto tiled_up_weight = fused
      ? gem16::internal::TileSm120Nvfp4Weights(layout.value(), *up_packed)
      : gem16::Result<std::vector<std::uint8_t>>(gem16::Status(gem16::StatusCode::kInvalidArgument, "unused"));
  const auto tiled_up_scales = fused
      ? gem16::internal::TileSm120Nvfp4WeightScales(layout.value(), *up_scales)
      : gem16::Result<std::vector<std::uint8_t>>(gem16::Status(gem16::StatusCode::kInvalidArgument, "unused"));
  if (fused) {
    NVFP4_TEST_CHECK(tiled_up_weight.ok());
    NVFP4_TEST_CHECK(tiled_up_scales.ok());
    if (!tiled_up_weight.ok() || !tiled_up_scales.ok()) return;
  }

  DeviceBuffer<std::uint8_t> device_activation(activation.value().packed_e2m1.size());
  DeviceBuffer<std::uint8_t> device_activation_scales(activation.value().block_scales_e4m3fn.size());
  DeviceBuffer<std::uint8_t> device_weight(packed.size());
  DeviceBuffer<std::uint8_t> device_scales(scales.size());
  DeviceBuffer<std::uint8_t> device_tiled_weight(tiled_weight.value().size());
  DeviceBuffer<std::uint8_t> device_tiled_scales(tiled_scales.value().size());
  DeviceBuffer<float> device_reference(rows);
  DeviceBuffer<float> device_native(rows);
  std::unique_ptr<DeviceBuffer<std::uint8_t>> device_up;
  std::unique_ptr<DeviceBuffer<std::uint8_t>> device_up_scales;
  std::unique_ptr<DeviceBuffer<std::uint8_t>> device_tiled_up;
  std::unique_ptr<DeviceBuffer<std::uint8_t>> device_tiled_up_scales;
  std::unique_ptr<DeviceBuffer<float>> device_up_reference;
  std::unique_ptr<DeviceBuffer<float>> device_up_native;
  std::unique_ptr<DeviceBuffer<float>> device_fused_gate;
  std::unique_ptr<DeviceBuffer<float>> device_fused_up;
  std::unique_ptr<DeviceBuffer<float>> device_fused_product;
  if (fused) {
    device_up = std::make_unique<DeviceBuffer<std::uint8_t>>(up_packed->size());
    device_up_scales = std::make_unique<DeviceBuffer<std::uint8_t>>(up_scales->size());
    device_tiled_up = std::make_unique<DeviceBuffer<std::uint8_t>>(tiled_up_weight.value().size());
    device_tiled_up_scales = std::make_unique<DeviceBuffer<std::uint8_t>>(tiled_up_scales.value().size());
    device_up_reference = std::make_unique<DeviceBuffer<float>>(rows);
    device_up_native = std::make_unique<DeviceBuffer<float>>(rows);
    device_fused_gate = std::make_unique<DeviceBuffer<float>>(rows);
    device_fused_up = std::make_unique<DeviceBuffer<float>>(rows);
    device_fused_product = std::make_unique<DeviceBuffer<float>>(rows);
  }
  const bool valid = device_activation.get() != nullptr && device_activation_scales.get() != nullptr &&
      device_weight.get() != nullptr && device_scales.get() != nullptr &&
      device_tiled_weight.get() != nullptr && device_tiled_scales.get() != nullptr &&
      device_reference.get() != nullptr && device_native.get() != nullptr &&
      (!fused || (device_up->get() != nullptr && device_up_scales->get() != nullptr &&
                  device_tiled_up->get() != nullptr && device_tiled_up_scales->get() != nullptr &&
                  device_up_reference->get() != nullptr && device_up_native->get() != nullptr &&
                  device_fused_gate->get() != nullptr && device_fused_up->get() != nullptr &&
                  device_fused_product->get() != nullptr));
  NVFP4_TEST_CHECK(valid);
  if (!valid) return;
  if (!CudaOk(cudaMemcpy(device_activation.get(), activation.value().packed_e2m1.data(),
                         device_activation.bytes(), cudaMemcpyHostToDevice), "copy fixture activation") ||
      !CudaOk(cudaMemcpy(device_activation_scales.get(), activation.value().block_scales_e4m3fn.data(),
                         device_activation_scales.bytes(), cudaMemcpyHostToDevice), "copy fixture activation scales") ||
      !CudaOk(cudaMemcpy(device_weight.get(), packed.data(), device_weight.bytes(), cudaMemcpyHostToDevice), "copy compiler packed weights") ||
      !CudaOk(cudaMemcpy(device_scales.get(), scales.data(), device_scales.bytes(), cudaMemcpyHostToDevice), "copy compiler scales") ||
      !CudaOk(cudaMemcpy(device_tiled_weight.get(), tiled_weight.value().data(), device_tiled_weight.bytes(), cudaMemcpyHostToDevice), "copy tiled compiler weights") ||
      !CudaOk(cudaMemcpy(device_tiled_scales.get(), tiled_scales.value().data(), device_tiled_scales.bytes(), cudaMemcpyHostToDevice), "copy tiled compiler scales")) return;
  if (fused &&
      (!CudaOk(cudaMemcpy(device_up->get(), up_packed->data(), device_up->bytes(), cudaMemcpyHostToDevice), "copy compiler Up weights") ||
       !CudaOk(cudaMemcpy(device_up_scales->get(), up_scales->data(), device_up_scales->bytes(), cudaMemcpyHostToDevice), "copy compiler Up scales") ||
       !CudaOk(cudaMemcpy(device_tiled_up->get(), tiled_up_weight.value().data(), device_tiled_up->bytes(), cudaMemcpyHostToDevice), "copy tiled compiler Up weights") ||
       !CudaOk(cudaMemcpy(device_tiled_up_scales->get(), tiled_up_scales.value().data(), device_tiled_up_scales->bytes(), cudaMemcpyHostToDevice), "copy tiled compiler Up scales"))) return;

  const auto reference_status = gem16::internal::LaunchNvfp4ReferenceProjection(
      device_activation.get(), device_activation_scales.get(), device_weight.get(), device_scales.get(),
      device_reference.get(), rows, contracting_elements, input_divisor, weight_divisor, nullptr);
  const auto native_status = gem16::internal::LaunchNvfp4Sm120DirectProjection(
      device_activation.get(), device_activation_scales.get(), device_tiled_weight.get(), device_tiled_scales.get(),
      device_native.get(), rows, contracting_elements, input_divisor, weight_divisor, nullptr);
  NVFP4_TEST_CHECK(reference_status.ok());
  NVFP4_TEST_CHECK(native_status.ok());
  gem16::Status up_reference_status;
  gem16::Status up_native_status;
  gem16::Status fused_status;
  if (fused) {
    const auto up_reference_result = gem16::internal::LaunchNvfp4ReferenceProjection(
        device_activation.get(), device_activation_scales.get(), device_up->get(), device_up_scales->get(),
        device_up_reference->get(), rows, contracting_elements, input_divisor, weight_divisor, nullptr);
    const auto up_native_result = gem16::internal::LaunchNvfp4Sm120DirectProjection(
        device_activation.get(), device_activation_scales.get(), device_tiled_up->get(), device_tiled_up_scales->get(),
        device_up_native->get(), rows, contracting_elements, input_divisor, weight_divisor, nullptr);
    const auto fused_result = gem16::internal::LaunchNvfp4Sm120FusedGateUp(
        device_activation.get(), device_activation_scales.get(), device_tiled_weight.get(), device_tiled_scales.get(),
        device_tiled_up->get(), device_tiled_up_scales->get(), device_fused_gate->get(), device_fused_up->get(),
        device_fused_product->get(), rows, contracting_elements, input_divisor, weight_divisor,
        input_divisor, weight_divisor, nullptr);
    up_reference_status = up_reference_result;
    up_native_status = up_native_result;
    fused_status = fused_result;
    NVFP4_TEST_CHECK(up_reference_status.ok());
    NVFP4_TEST_CHECK(up_native_status.ok());
    NVFP4_TEST_CHECK(fused_status.ok());
  }
  if (!reference_status.ok() || !native_status.ok() ||
      (fused && (!up_reference_status.ok() || !up_native_status.ok() || !fused_status.ok())) ||
      !CudaOk(cudaDeviceSynchronize(), "synchronize compiler consumption")) return;
  std::vector<float> reference(rows), native(rows);
  if (!CudaOk(cudaMemcpy(reference.data(), device_reference.get(), device_reference.bytes(), cudaMemcpyDeviceToHost), "copy reference output") ||
      !CudaOk(cudaMemcpy(native.data(), device_native.get(), device_native.bytes(), cudaMemcpyDeviceToHost), "copy native output")) return;
  std::vector<float> up_reference, up_native, fused_gate, fused_up, fused_product;
  if (fused) {
    up_reference.resize(rows); up_native.resize(rows); fused_gate.resize(rows); fused_up.resize(rows); fused_product.resize(rows);
    if (!CudaOk(cudaMemcpy(up_reference.data(), device_up_reference->get(), device_up_reference->bytes(), cudaMemcpyDeviceToHost), "copy Up reference") ||
        !CudaOk(cudaMemcpy(up_native.data(), device_up_native->get(), device_up_native->bytes(), cudaMemcpyDeviceToHost), "copy Up native") ||
        !CudaOk(cudaMemcpy(fused_gate.data(), device_fused_gate->get(), device_fused_gate->bytes(), cudaMemcpyDeviceToHost), "copy fused Gate") ||
        !CudaOk(cudaMemcpy(fused_up.data(), device_fused_up->get(), device_fused_up->bytes(), cudaMemcpyDeviceToHost), "copy fused Up") ||
        !CudaOk(cudaMemcpy(fused_product.data(), device_fused_product->get(), device_fused_product->bytes(), cudaMemcpyDeviceToHost), "copy fused product")) return;
  }
  const std::size_t row_bytes = contracting_elements / 2U;
  const std::size_t scale_row_bytes = contracting_elements / 16U;
  const std::array<std::size_t, 3> selected_rows = {0U, rows / 2U, rows - 1U};
  for (std::size_t row = 0; row < rows; ++row) {
    NVFP4_TEST_CHECK(std::isfinite(reference[row]) && std::isfinite(native[row]));
    NVFP4_TEST_CHECK(CloseEnough(native[row], reference[row], 2.0e-3, 2.0e-4));
    if (fused) {
      NVFP4_TEST_CHECK(std::isfinite(up_reference[row]) && std::isfinite(up_native[row]) &&
                       std::isfinite(fused_gate[row]) && std::isfinite(fused_up[row]) &&
                       std::isfinite(fused_product[row]));
      NVFP4_TEST_CHECK(CloseEnough(up_native[row], up_reference[row], 2.0e-3, 2.0e-4));
      NVFP4_TEST_CHECK(std::fabs(fused_gate[row] - RoundBf16(native[row])) < 1.0e-6F);
      NVFP4_TEST_CHECK(std::fabs(fused_up[row] - RoundBf16(up_native[row])) < 1.0e-6F);
      NVFP4_TEST_CHECK(std::fabs(fused_product[row] - RoundBf16(RoundBf16(GeluTanh(RoundBf16(native[row]))) * RoundBf16(up_native[row]))) < 2.0e-3F);
    }
  }
  for (const std::size_t row : selected_rows) {
    const auto cpu_reference = gem16::nvfp4::ReferenceDotProduct(
        activation.value(), std::span<const std::uint8_t>(packed.data() + row * row_bytes, row_bytes),
        std::span<const std::uint8_t>(scales.data() + row * scale_row_bytes, scale_row_bytes), weight_divisor);
    NVFP4_TEST_CHECK(cpu_reference.ok());
    if (cpu_reference.ok()) {
      NVFP4_TEST_CHECK(CloseEnough(reference[row], cpu_reference.value(), 2.0e-3, 2.0e-4));
      NVFP4_TEST_CHECK(CloseEnough(native[row], cpu_reference.value(), 2.0e-3, 2.0e-4));
    }
    if (fused) {
      const auto up_cpu = gem16::nvfp4::ReferenceDotProduct(
          activation.value(), std::span<const std::uint8_t>(up_packed->data() + row * row_bytes, row_bytes),
          std::span<const std::uint8_t>(up_scales->data() + row * scale_row_bytes, scale_row_bytes), weight_divisor);
      NVFP4_TEST_CHECK(up_cpu.ok());
      if (up_cpu.ok()) NVFP4_TEST_CHECK(CloseEnough(up_native[row], up_cpu.value(), 2.0e-3, 2.0e-4));
    }
  }
  std::cout << label << ": rows=" << rows << " K=" << contracting_elements
            << " packed_bytes=" << packed.size() << " scale_bytes=" << scales.size()
            << " divisor=" << weight_divisor << " persistent_repack_bytes="
            << layout.value().persistent_repack_bytes << (fused ? " fused_gate_up=true" : "") << '\n';
}

NativeMatrix SliceRows(const NativeMatrix& parent, std::size_t row_start, std::size_t rows) {
  const std::size_t row_bytes = parent.contracting_elements / 2U;
  const std::size_t scale_row_bytes = parent.contracting_elements / 16U;
  NativeMatrix result = parent;
  result.rows = rows;
  result.packed = std::vector<std::uint8_t>(
      parent.packed.begin() + static_cast<std::ptrdiff_t>(row_start * row_bytes),
      parent.packed.begin() + static_cast<std::ptrdiff_t>((row_start + rows) * row_bytes));
  result.scales = std::vector<std::uint8_t>(
      parent.scales.begin() + static_cast<std::ptrdiff_t>(row_start * scale_row_bytes),
      parent.scales.begin() + static_cast<std::ptrdiff_t>((row_start + rows) * scale_row_bytes));
  return result;
}

bool SelectSm120Device() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return false;
  for (int device = 0; device < device_count; ++device) {
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) continue;
    if (properties.major == 12 && properties.minor == 0) {
      return CudaOk(cudaSetDevice(device), "select SM120 device");
    }
  }
  return false;
}

}  // namespace

int main() {
  TestMalformedHostInputs();
  if (failures != 0) {
    std::cerr << failures << " NVFP4 host assertion(s) failed\n";
    return 1;
  }
  if (!SelectSm120Device()) {
    std::cerr << "SKIP: representative NVFP4 consumption requires an SM120 CUDA device\n";
    return kCTestSkip;
  }
  cudaDeviceProp properties{};
  int selected_device = 0;
  if (!CudaOk(cudaGetDevice(&selected_device), "get selected CUDA device") ||
      !CudaOk(cudaGetDeviceProperties(&properties, selected_device),
              "get selected CUDA device properties")) return 1;
  std::cout << "Running representative Gemma 4 26B NVFP4 consumption on "
            << properties.name << " (SM" << properties.major << properties.minor << ")\n";

  const auto root = std::filesystem::temp_directory_path() /
                    ("gem16-m06-cuda-consumption-" +
                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(root);
  std::vector<std::uint8_t> source_bytes;
  const auto shared_gate = CompileFixture(
      root / "shared-gate", "model.language_model.layers.0.mlp.gate_proj.weight",
      2112U, 2816U, 3U, &source_bytes);
  const auto shared_up = CompileFixture(
      root / "shared-up", "model.language_model.layers.0.mlp.up_proj.weight",
      2112U, 2816U, 5U, &source_bytes);
  const auto shared_down = CompileFixture(
      root / "shared-down", "model.language_model.layers.0.mlp.down_proj.weight",
      2816U, 2112U, 7U, &source_bytes);
  const auto routed_gate_up = CompileFixture(
      root / "routed-gate-up", "model.language_model.layers.0.experts.gate_up_proj",
      1408U, 2816U, 11U, &source_bytes);
  const auto routed_down = CompileFixture(
      root / "routed-down", "model.language_model.layers.0.experts.down_proj",
      2816U, 704U, 17U, &source_bytes);
  if (failures != 0) {
    std::filesystem::remove_all(root);
    return 1;
  }

  RunProjection("shared Gate [2112,2816]", shared_gate.rows,
                shared_gate.contracting_elements, shared_gate.packed, shared_gate.scales,
                shared_gate.weight_divisor, shared_gate.input_divisor, 3U);
  RunProjection("shared Up [2112,2816]", shared_up.rows,
                shared_up.contracting_elements, shared_up.packed, shared_up.scales,
                shared_up.weight_divisor, shared_up.input_divisor, 5U);
  RunProjection("shared Down [2816,2112]", shared_down.rows,
                shared_down.contracting_elements, shared_down.packed, shared_down.scales,
                shared_down.weight_divisor, shared_down.input_divisor, 7U);
  const auto gate = SliceRows(routed_gate_up, 0U, 704U);
  const auto up = SliceRows(routed_gate_up, 704U, 704U);
  NVFP4_TEST_CHECK(gate.weight_divisor == up.weight_divisor);
  NVFP4_TEST_CHECK(gate.input_divisor == up.input_divisor);
  RunProjection("routed Gate/Up [704,2816]", gate.rows,
                gate.contracting_elements, gate.packed, gate.scales,
                gate.weight_divisor, gate.input_divisor, 11U, &up.packed, &up.scales);
  RunProjection("routed Down [2816,704]", routed_down.rows,
                routed_down.contracting_elements, routed_down.packed, routed_down.scales,
                routed_down.weight_divisor, routed_down.input_divisor, 17U);
  std::filesystem::remove_all(root);

  if (failures != 0) {
    std::cerr << failures << " NVFP4 consumption assertion(s) failed\n";
    return 1;
  }
  std::cout << "representative Gemma 4 26B NVFP4 consumption tests passed\n";
  return 0;
}

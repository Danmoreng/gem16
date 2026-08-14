#include "compiler/fp8_batch_encoder.h"
#include "compiler/sha256.h"

#include "test.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "util/json.h"

namespace {

std::filesystem::path TestRoot() {
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto root = std::filesystem::temp_directory_path() /
                    ("gem16-fp8-batch-test-" + std::to_string(stamp));
  std::filesystem::create_directories(root);
  return root;
}

void WriteBytes(const std::filesystem::path& path,
                const std::vector<std::uint8_t>& bytes) {
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
}

std::vector<std::uint8_t> ReadBytes(const std::filesystem::path& path);

std::string JobJson(const std::filesystem::path& source, std::uint64_t threads,
                    std::uint64_t rows = 1, std::uint64_t columns = 4,
                    std::uint64_t source_bytes = 8) {
  const auto source_bytes_data = ReadBytes(source);
  const std::string source_hash = gem16::compiler::Sha256Hex(source_bytes_data.data(), source_bytes_data.size());
  const std::string source_text = gem16::json::Quote(source.string());
  return "{\"contract_id\":\"gem16.fp8_attention_rowwise\","
         "\"contract_version\":1,\"matrices\":[{"
         "\"columns\":" + std::to_string(columns) +
         ",\"rows\":" + std::to_string(rows) +
         ",\"scale_bytes\":" + std::to_string(rows * 2) +
         ",\"scale_offset\":" + std::to_string(rows * columns + 0) +
         ",\"scale_output_name\":\"matrix.weight_scale\","
         "\"source_bytes\":" + std::to_string(source_bytes) +
         ",\"source_name\":\"model.language_model.layers.0.self_attn.q_proj.weight\","
         "\"source_sha256\":\"" + source_hash + "\","
         "\"source_offset\":0,\"source_path\":" + source_text +
         ",\"weight_bytes\":" + std::to_string(rows * columns) +
         ",\"weight_offset\":0,\"weight_output_name\":\"matrix.weight\"}],"
         "\"payload_bytes\":" + std::to_string(rows * columns + rows * 2) +
         ",\"schema_version\":1,\"threads\":" + std::to_string(threads) + "}\n";
}

void WriteJob(const std::filesystem::path& path, const std::filesystem::path& source,
              std::uint64_t threads, std::uint64_t rows = 1,
              std::uint64_t columns = 4, std::uint64_t source_bytes = 8) {
  std::ofstream output(path, std::ios::binary);
  const std::string text = JobJson(source, threads, rows, columns, source_bytes);
  output.write(text.data(), static_cast<std::streamsize>(text.size()));
}

std::vector<std::uint8_t> ReadBytes(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(input), {});
}

std::string ReadText(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(input), {});
}

std::string CompareRangeJson(const std::filesystem::path& path, std::uint64_t offset,
                             std::uint64_t bytes) {
  const auto data = ReadBytes(path);
  return "{\"path\":" + gem16::json::Quote(std::filesystem::absolute(path).string()) +
         ",\"offset\":" + std::to_string(offset) +
         ",\"bytes\":" + std::to_string(bytes) +
         ",\"sha256\":\"" +
         gem16::compiler::Sha256Hex(data.data() + offset, bytes) + "\"}";
}

std::string CompareJobJson(const std::filesystem::path& path, std::uint64_t threads) {
  constexpr const char* roles[] = {"q", "k", "o", "v"};
  std::string matrices;
  for (std::size_t index = 0; index < 4U; ++index) {
    if (!matrices.empty()) matrices += ",";
    const auto weight_left = CompareRangeJson(path, 0, 1);
    const auto scale = CompareRangeJson(path, 1, 2);
    const auto weight_right = CompareRangeJson(path, 3, 1);
    const auto layer = index == 3U ? 1U : 0U;
    const std::string name = "model.language_model.layers." + std::to_string(layer) +
                             ".self_attn." + roles[index] + "_proj";
    matrices += "{\"name\":\"" + name +
                "\",\"layer\":" + std::to_string(layer) +
                ",\"role\":\"" + roles[index] +
                "\",\"rows\":1,\"columns\":1,\"left_weight\":" +
                weight_left + ",\"left_scale\":" + scale +
                ",\"right_weight\":" + weight_right +
                ",\"right_scale\":" + scale + "}";
  }
  return "{\"schema_version\":1,\"contract_id\":\"gem16.fp8_attention_compare\","
         "\"contract_version\":1,\"threads\":" + std::to_string(threads) +
         ",\"matrices\":[" + matrices + "]}";
}

std::string ProductionCompareJobJson(const std::filesystem::path& path, std::uint64_t threads,
                                      bool include_global_v, bool malformed_name = false,
                                      bool malformed_shape = false) {
  const auto weight_left = CompareRangeJson(path, 0, 1);
  const auto scale = CompareRangeJson(path, 1, 2);
  const auto weight_right = CompareRangeJson(path, 3, 1);
  std::string matrices;
  std::size_t index = 0;
  for (std::uint64_t layer = 0; layer < 30U; ++layer) {
    const bool global = layer == 5U || layer == 11U || layer == 17U || layer == 23U || layer == 29U;
    constexpr const char* required_roles[] = {"q", "k", "o"};
    for (const char* role : required_roles) {
      if (!matrices.empty()) matrices += ",";
      const std::string name = malformed_name && index == 0U
          ? "malformed" : "model.language_model.layers." + std::to_string(layer) +
              ".self_attn." + role + "_proj.weight";
      const std::uint64_t columns = malformed_shape && index == 0U ? 8193U : 1U;
      matrices += "{\"name\":\"" + name + "\",\"layer\":" + std::to_string(layer) +
                  ",\"role\":\"" + role + "\",\"rows\":1,\"columns\":" +
                  std::to_string(columns) + ",\"left_weight\":" + weight_left +
                  ",\"left_scale\":" + scale + ",\"right_weight\":" + weight_right +
                  ",\"right_scale\":" + scale + "}";
      ++index;
    }
    if ((!global && !(include_global_v && layer == 0U)) || (include_global_v && layer == 5U)) {
      if (!matrices.empty()) matrices += ",";
      const std::string name = "model.language_model.layers." + std::to_string(layer) +
                               ".self_attn.v_proj.weight";
      matrices += "{\"name\":\"" + name + "\",\"layer\":" + std::to_string(layer) +
                  ",\"role\":\"v\",\"rows\":1,\"columns\":1,\"left_weight\":" +
                  weight_left + ",\"left_scale\":" + scale +
                  ",\"right_weight\":" + weight_right + ",\"right_scale\":" + scale + "}";
      ++index;
    }
  }
  return "{\"schema_version\":1,\"contract_id\":\"gem16.fp8_attention_compare\",\"contract_version\":1,\"threads\":" +
         std::to_string(threads) + ",\"matrices\":[" + matrices + "]}";
}

std::string MultiJobJson(const std::filesystem::path& source, std::uint64_t threads) {
  const auto bytes = ReadBytes(source);
  const std::string path = gem16::json::Quote(source.string());
  std::string matrices;
  for (std::size_t index = 0; index < 4U; ++index) {
    const std::string hash = gem16::compiler::Sha256Hex(bytes.data() + index * 2U, 2U);
    if (index != 0U) matrices += ",";
    matrices += "{\"columns\":1,\"rows\":1,\"scale_bytes\":2,\"scale_offset\":" +
                std::to_string(index * 3U + 1U) + ",\"scale_output_name\":\"matrix" +
                std::to_string(index) + ".scale\",\"source_bytes\":2,\"source_name\":\"matrix" +
                std::to_string(index) + "\",\"source_path\":" + path +
                ",\"source_sha256\":\"" + hash + "\",\"source_offset\":" +
                std::to_string(index * 2U) + ",\"weight_bytes\":1,\"weight_offset\":" +
                std::to_string(index * 3U) + ",\"weight_output_name\":\"matrix" +
                std::to_string(index) + ".weight\"}";
  }
  return "{\"contract_id\":\"gem16.fp8_attention_rowwise\",\"contract_version\":1,\"matrices\":[" +
         matrices + "],\"payload_bytes\":12,\"schema_version\":1,\"threads\":" +
         std::to_string(threads) + "}\n";
}

void TestSha256() {
  GEM16_CHECK(gem16::compiler::Sha256Hex("", 0) ==
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  GEM16_CHECK(gem16::compiler::Sha256Hex("abc", 3) ==
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  const std::string text = "abcdefghij";
  gem16::compiler::Sha256 range;
  range.Update(text.data() + 2, 5);
  GEM16_CHECK(range.HexDigest() ==
              "ff7834266e9e68caf1ca05fd2f11d469f6599abab3a62508cb645fde65d30dc3");

  struct Vector {
    std::size_t size;
    const char* digest;
  };
  constexpr Vector vectors[] = {
      {55, "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"},
      {56, "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"},
      {63, "7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34"},
      {64, "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"},
      {65, "635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0"},
  };
  for (const auto& vector : vectors) {
    const std::string repeated(vector.size, 'a');
    GEM16_CHECK(gem16::compiler::Sha256Hex(repeated.data(), repeated.size()) == vector.digest);
  }
  const std::string long_text(257, 'a');
  gem16::compiler::Sha256 incremental;
  for (std::size_t offset = 0; offset < long_text.size();) {
    const std::size_t count = std::min<std::size_t>(17, long_text.size() - offset);
    incremental.Update(long_text.data() + offset, count);
    offset += count;
  }
  GEM16_CHECK(incremental.HexDigest() ==
              "e8d95cc2b4bc198c54b40bd214df958afb65f5e73d2c2eafe0593cf5c635c1f0");
}

void TestCodec() {
  for (std::uint32_t code = 0; code < 256; ++code) {
    if ((code & 0x7FU) == 0x7FU) continue;
    const float decoded = gem16::compiler::DecodeE4M3Fn(static_cast<std::uint8_t>(code));
    const auto encoded = gem16::compiler::EncodeE4M3Fn(decoded);
    GEM16_CHECK(encoded.ok());
    if (encoded.ok()) GEM16_CHECK(encoded.value() == code);
  }
  for (std::uint32_t low = 0; low < 126; ++low) {
    const float midpoint = (gem16::compiler::DecodeE4M3Fn(static_cast<std::uint8_t>(low)) +
                            gem16::compiler::DecodeE4M3Fn(static_cast<std::uint8_t>(low + 1))) /
                           2.0F;
    const auto positive = gem16::compiler::EncodeE4M3Fn(midpoint);
    const auto negative = gem16::compiler::EncodeE4M3Fn(-midpoint);
    const std::uint8_t expected = static_cast<std::uint8_t>((low & 1U) == 0U ? low : low + 1U);
    GEM16_CHECK(positive.ok() && positive.value() == expected);
    GEM16_CHECK(negative.ok() && negative.value() == static_cast<std::uint8_t>(expected | 0x80U));
  }
  GEM16_CHECK(std::isnan(gem16::compiler::DecodeE4M3Fn(0x7FU)));
  GEM16_CHECK(std::isnan(gem16::compiler::DecodeE4M3Fn(0xFFU)));
  GEM16_CHECK(!gem16::compiler::EncodeE4M3Fn(std::numeric_limits<float>::quiet_NaN()).ok());
  GEM16_CHECK(!gem16::compiler::EncodeE4M3Fn(std::numeric_limits<float>::infinity()).ok());
  GEM16_CHECK(gem16::compiler::RoundBf16Rne(1.0F).value() == 0x3F80U);
}

void CheckNativeBuild(const gem16::json::Value& root) {
  const auto* native_build = root.find("native_build");
  GEM16_CHECK(native_build != nullptr && native_build->is_object());
  if (native_build == nullptr || !native_build->is_object()) return;
  constexpr std::string_view fields[] = {
      "compiler_id", "compiler_version", "build_type", "cxx_standard", "system", "processor"};
  GEM16_CHECK(native_build->as_object().size() == std::size(fields));
  for (const auto field : fields) {
    const auto* value = native_build->find(field);
    GEM16_CHECK(value != nullptr && value->is_string() && !value->as_string().empty());
  }
}

void TestBatchAndDeterminism() {
  const auto root = TestRoot();
#ifdef _WIN32
  const auto source = root / "windows-source.bin";
  WriteBytes(source, {0x80, 0x3F});
  const auto job = root / "windows-job.json";
  WriteJob(job, source, 1, 1, 1, 2);
  const auto status = gem16::compiler::EncodeJobFile(job, root / "payload.bin", root / "telemetry.json");
  GEM16_CHECK(status.code() == gem16::StatusCode::kUnsupported);
  std::filesystem::remove_all(root);
  return;
#else
  const auto source = root / "source.bin";
  WriteBytes(source, {0x80, 0x3F, 0x00, 0xC0, 0x00, 0x3F, 0x00, 0x80});
  const auto job1 = root / "job1.json";
  const auto job4 = root / "job4.json";
  WriteJob(job1, source, 1);
  WriteJob(job4, source, 4);
  const auto payload1 = root / "payload1.bin";
  const auto telemetry1 = root / "telemetry1.json";
  const auto payload4 = root / "payload4.bin";
  const auto telemetry4 = root / "telemetry4.json";
  GEM16_CHECK(gem16::compiler::EncodeJobFile(job1, payload1, telemetry1).ok());
  GEM16_CHECK(gem16::compiler::EncodeJobFile(job4, payload4, telemetry4).ok());
  GEM16_CHECK(ReadBytes(payload1) == std::vector<std::uint8_t>({0x76, 0xFE, 0x6E, 0x80, 0x92, 0x3B}));
  GEM16_CHECK(ReadBytes(payload1) == ReadBytes(payload4));

  const auto round_source = root / "round-source.bin";
  WriteBytes(round_source, {0x82, 0x3F, 0x82, 0xBF});
  const auto round_job = root / "round-job.json";
  WriteJob(round_job, round_source, 1, 1, 2, 4);
  const auto round_payload = root / "round-payload.bin";
  GEM16_CHECK(gem16::compiler::EncodeJobFile(round_job, round_payload,
                                             root / "round-telemetry.json").ok());
  GEM16_CHECK(ReadBytes(round_payload) == std::vector<std::uint8_t>({0x7E, 0xFE, 0x15, 0x3B}));

  const auto zero_source = root / "zero-source.bin";
  WriteBytes(zero_source, {0x00, 0x00, 0x00, 0x80});
  const auto zero_job = root / "zero-job.json";
  WriteJob(zero_job, zero_source, 1, 1, 2, 4);
  const auto zero_payload = root / "zero-payload.bin";
  GEM16_CHECK(gem16::compiler::EncodeJobFile(zero_job, zero_payload,
                                             root / "zero-telemetry.json").ok());
  GEM16_CHECK(ReadBytes(zero_payload) == std::vector<std::uint8_t>({0x00, 0x80, 0x80, 0x3F}));

  const auto tiny_source = root / "tiny-source.bin";
  WriteBytes(tiny_source, {0x01, 0x00});
  const auto tiny_job = root / "tiny-job.json";
  WriteJob(tiny_job, tiny_source, 1, 1, 1, 2);
  const auto tiny_payload = root / "tiny-payload.bin";
  GEM16_CHECK(gem16::compiler::EncodeJobFile(tiny_job, tiny_payload,
                                             root / "tiny-telemetry.json").ok());
  GEM16_CHECK(ReadBytes(tiny_payload) == std::vector<std::uint8_t>({0x38, 0x01, 0x00}));
  const std::string report1 = ReadText(telemetry1);
  const std::string report4 = ReadText(telemetry4);
  GEM16_CHECK(report1.find("\"threads\":1") != std::string::npos);
  GEM16_CHECK(report4.find("\"threads\":4") != std::string::npos);

  const auto parallel_source = root / "parallel-source.bin";
  WriteBytes(parallel_source, {0x80, 0x3F, 0x00, 0x40, 0x00, 0x41, 0x00, 0x42});
  const auto parallel_job1 = root / "parallel-job1.json";
  const auto parallel_job4 = root / "parallel-job4.json";
  {
    std::ofstream output(parallel_job1, std::ios::binary);
    output << MultiJobJson(parallel_source, 1);
  }
  {
    std::ofstream output(parallel_job4, std::ios::binary);
    output << MultiJobJson(parallel_source, 4);
  }
  const auto parallel_payload1 = root / "parallel-payload1.bin";
  const auto parallel_payload4 = root / "parallel-payload4.bin";
  const auto parallel_telemetry1 = root / "parallel-telemetry1.json";
  const auto parallel_telemetry4 = root / "parallel-telemetry4.json";
  const auto parallel_status1 = gem16::compiler::EncodeJobFile(parallel_job1, parallel_payload1, parallel_telemetry1);
  const auto parallel_status4 = gem16::compiler::EncodeJobFile(parallel_job4, parallel_payload4, parallel_telemetry4);
  GEM16_CHECK(parallel_status1.ok());
  GEM16_CHECK(parallel_status4.ok());
  GEM16_CHECK(ReadBytes(parallel_payload1) == ReadBytes(parallel_payload4));
  std::string normalized1 = ReadText(parallel_telemetry1);
  std::string normalized4 = ReadText(parallel_telemetry4);
  const auto thread_pos = normalized4.find("\"threads\":4");
  GEM16_CHECK(thread_pos != std::string::npos);
  if (thread_pos != std::string::npos) normalized4.replace(thread_pos, 11, "\"threads\":1");
  GEM16_CHECK(normalized1 == normalized4);
  GEM16_CHECK(normalized1.find("source_sha256") != std::string::npos);
  GEM16_CHECK(normalized1.find("weight_sha256") != std::string::npos);
  GEM16_CHECK(normalized1.find("scale_sha256") != std::string::npos);
  GEM16_CHECK(normalized1.find("\"native_build\"") != std::string::npos);
  GEM16_CHECK(normalized1.find("\"compiler_id\"") != std::string::npos);
  GEM16_CHECK(normalized1.find("\"compiler_version\"") != std::string::npos);
  GEM16_CHECK(normalized1.find("\"build_type\"") != std::string::npos);
  GEM16_CHECK(normalized1.find("\"cxx_standard\":\"20\"") != std::string::npos);
  GEM16_CHECK(normalized1.find("\"system\"") != std::string::npos);
  GEM16_CHECK(normalized1.find("\"processor\"") != std::string::npos);
  const auto telemetry_json = gem16::json::Parse(report1);
  GEM16_CHECK(telemetry_json.ok());
  if (telemetry_json.ok()) CheckNativeBuild(telemetry_json.value());

  // Differential golden from quantize_bf16_row(): the second row's four
  // 2^-27 values contribute one binary64 ulp only when the row sum is closed
  // before matrix aggregation.  A matrix-wide element accumulation yields
  // 1.0 here and violates the versioned row-partial telemetry contract.
  const auto oracle_source = root / "oracle-source.bin";
  WriteBytes(oracle_source, {
      0x80, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x32, 0x00, 0x32, 0x00, 0x32, 0x00, 0x32,
  });
  const auto oracle_job = root / "oracle-job.json";
  WriteJob(oracle_job, oracle_source, 1, 2, 4, 16);
  const auto oracle_payload = root / "oracle-payload.bin";
  const auto oracle_telemetry = root / "oracle-telemetry.json";
  GEM16_CHECK(gem16::compiler::EncodeJobFile(
                  oracle_job, oracle_payload, oracle_telemetry).ok());
  GEM16_CHECK(ReadBytes(oracle_payload) == std::vector<std::uint8_t>({
      0x7E, 0x00, 0x00, 0x00, 0x7E, 0x7E, 0x7E, 0x7E,
      0x12, 0x3B, 0x92, 0x2D,
  }));
  const auto oracle_json = gem16::json::Parse(ReadText(oracle_telemetry));
  GEM16_CHECK(oracle_json.ok());
  if (oracle_json.ok()) {
    const auto* matrices = oracle_json.value().find("matrices");
    GEM16_CHECK(matrices != nullptr && matrices->is_array() &&
                matrices->as_array().size() == 1U);
    if (matrices != nullptr && matrices->is_array() &&
        matrices->as_array().size() == 1U) {
      const auto& matrix = matrices->as_array()[0];
      const auto* source_energy = matrix.find("source_sum_squares");
      const auto* reconstruction_energy =
          matrix.find("reconstruction_sum_squares");
      const auto* dot = matrix.find("source_reconstruction_dot");
      const auto* error_energy = matrix.find("error_sum_squares");
      GEM16_CHECK(source_energy != nullptr && source_energy->is_number() &&
                  source_energy->as_number() == 1.0000000000000002);
      GEM16_CHECK(reconstruction_energy != nullptr &&
                  reconstruction_energy->is_number() &&
                  reconstruction_energy->as_number() == 0.9960975646972658);
      GEM16_CHECK(dot != nullptr && dot->is_number() &&
                  dot->as_number() == 0.9980468750000002);
      GEM16_CHECK(error_energy != nullptr && error_energy->is_number() &&
                  error_energy->as_number() == 3.814697265625001e-06);
    }
  }
  std::filesystem::remove_all(root);
#endif
}

void TestFailureCleanupAndLargestRow() {
  const auto root = TestRoot();
  const auto bad_source = root / "bad.bin";
  WriteBytes(bad_source, {0x80, 0x3F});
  const auto nan_source = root / "nan.bin";
  WriteBytes(nan_source, {0xC1, 0x7F});
  const auto job = root / "bad-job.json";
  WriteJob(job, nan_source, 1, 1, 1, 2);
  const auto payload = root / "bad-payload.bin";
  const auto telemetry = root / "bad-telemetry.json";
  GEM16_CHECK(!gem16::compiler::EncodeJobFile(job, payload, telemetry).ok());
  GEM16_CHECK(!std::filesystem::exists(payload));
  GEM16_CHECK(!std::filesystem::exists(telemetry));

  const auto mismatch_job = root / "mismatch-job.json";
  WriteJob(mismatch_job, bad_source, 1, 1, 1, 2);
  {
    std::string text = ReadText(mismatch_job);
    const auto hash_pos = text.find("source_sha256");
    const auto value_pos = text.find('"', text.find(':', hash_pos) + 1);
    const auto end_pos = text.find('"', value_pos + 1);
    text.replace(value_pos + 1, end_pos - value_pos - 1, 64, '0');
    std::ofstream output(mismatch_job, std::ios::binary);
    output << text;
  }
  const auto mismatch_payload = root / "mismatch-payload.bin";
  const auto mismatch_telemetry = root / "mismatch-telemetry.json";
  GEM16_CHECK(!gem16::compiler::EncodeJobFile(mismatch_job, mismatch_payload, mismatch_telemetry).ok());
  GEM16_CHECK(!std::filesystem::exists(mismatch_payload));
  GEM16_CHECK(!std::filesystem::exists(mismatch_telemetry));

#ifndef _WIN32
  const auto source_link = root / "source-link.bin";
  std::error_code link_error;
  std::filesystem::create_symlink(bad_source, source_link, link_error);
  if (!link_error) {
    const auto link_job = root / "link-job.json";
    WriteJob(link_job, source_link, 1, 1, 1, 2);
    GEM16_CHECK(!gem16::compiler::EncodeJobFile(link_job, root / "link-payload.bin", root / "link-telemetry.json").ok());

    const auto resolved_job = root / "resolved-job.json";
    WriteJob(resolved_job, std::filesystem::canonical(source_link), 1, 1, 1, 2);
    GEM16_CHECK(gem16::compiler::EncodeJobFile(resolved_job, root / "resolved-payload.bin", root / "resolved-telemetry.json").ok());
  }
#endif

  GEM16_CHECK(gem16::compiler::ExitCodeForStatus(gem16::StatusCode::kInvalidArgument) == 2);
  GEM16_CHECK(gem16::compiler::ExitCodeForStatus(gem16::StatusCode::kDataLoss) == 3);
  GEM16_CHECK(gem16::compiler::ExitCodeForStatus(gem16::StatusCode::kInternal) == 4);
  GEM16_CHECK(gem16::compiler::ExitCodeForStatus(gem16::StatusCode::kIoError) == 5);

#ifndef _WIN32
  const auto preexisting_payload = root / "preexisting-payload.bin";
  WriteBytes(preexisting_payload, {});
  GEM16_CHECK(!gem16::compiler::EncodeJobFile(job, preexisting_payload, root / "preexisting-telemetry.json").ok());
  const auto alias_payload = root / "nested" / ".." / "alias.bin";
  GEM16_CHECK(!gem16::compiler::EncodeJobFile(job, alias_payload, root / "alias.bin").ok());

  const auto bool_job = root / "bool-job.json";
  {
    std::string text = JobJson(bad_source, 1, 1, 1, 2);
    const auto position = text.find("\"threads\":1");
    GEM16_CHECK(position != std::string::npos);
    if (position != std::string::npos) text.replace(position, 11, "\"threads\":true");
    std::ofstream output(bool_job, std::ios::binary);
    output << text;
  }
  GEM16_CHECK(!gem16::compiler::EncodeJobFile(bool_job, root / "bool-payload.bin", root / "bool-telemetry.json").ok());

  const auto malformed = root / "malformed-job.json";
  std::string malformed_text = JobJson(bad_source, 1, 1, 1, 2);
  const std::string correct_offset = "\"scale_offset\":1";
  const std::string wrong_offset = "\"scale_offset\":2";
  const auto offset_position = malformed_text.find(correct_offset);
  GEM16_CHECK(offset_position != std::string::npos);
  if (offset_position != std::string::npos) {
    malformed_text.replace(offset_position, correct_offset.size(), wrong_offset);
  }
  {
    std::ofstream output(malformed, std::ios::binary);
    output.write(malformed_text.data(), static_cast<std::streamsize>(malformed_text.size()));
  }
  GEM16_CHECK(!gem16::compiler::EncodeJobFile(
      malformed, root / "malformed-payload.bin", root / "malformed-telemetry.json").ok());

  const auto large_source = root / "large.bin";
  std::vector<std::uint8_t> large(8192 * 2, 0);
  for (std::size_t index = 0; index < 8192; ++index) {
    large[index * 2] = 0x80;
    large[index * 2 + 1] = 0x3F;
  }
  WriteBytes(large_source, large);
  const auto large_job = root / "large-job.json";
  WriteJob(large_job, large_source, 1, 1, 8192, large.size());
  GEM16_CHECK(gem16::compiler::EncodeJobFile(large_job, root / "large-payload.bin",
                                             root / "large-telemetry.json").ok());
#endif
  std::filesystem::remove_all(root);
}

}  // namespace

void TestNativeComparison() {
  const auto root = TestRoot();
#ifdef _WIN32
  // Descriptor-bound native comparison is intentionally Linux-only. Keep the
  // unsupported boundary explicit rather than treating it as a successful
  // comparison on Windows; codec and SHA tests above remain portable.
  const auto status = gem16::compiler::CompareJobFile(root / "missing-job.json", root / "metrics.json");
  GEM16_CHECK(status.code() == gem16::StatusCode::kUnsupported);
  std::filesystem::remove_all(root);
  return;
#else
  const auto source = root / "compare.bin";
  WriteBytes(source, {0x00, 0x80, 0x3F, 0x38});
  const auto job1 = root / "compare-1.json";
  const auto job4 = root / "compare-4.json";
  {
    std::ofstream output(job1, std::ios::binary);
    output << CompareJobJson(source, 1);
  }
  {
    std::ofstream output(job4, std::ios::binary);
    output << CompareJobJson(source, 4);
  }
  const auto metrics1 = root / "metrics-1.json";
  const auto metrics4 = root / "metrics-4.json";
  GEM16_CHECK(gem16::compiler::CompareJobFile(job1, metrics1).ok());
  GEM16_CHECK(gem16::compiler::CompareJobFile(job4, metrics4).ok());
  auto report1 = ReadText(metrics1);
  auto report4 = ReadText(metrics4);
  const auto thread_position = report4.find("\"threads\":4");
  GEM16_CHECK(thread_position != std::string::npos);
  if (thread_position != std::string::npos) report4.replace(thread_position, 11, "\"threads\":1");
  GEM16_CHECK(report1 == report4);
  const auto parsed = gem16::json::Parse(report1);
  GEM16_CHECK(parsed.ok());
  if (parsed.ok()) {
    const auto* matrices = parsed.value().find("matrices");
    GEM16_CHECK(matrices != nullptr && matrices->is_array() && matrices->as_array().size() == 4U);
    if (matrices != nullptr && matrices->is_array() && !matrices->as_array().empty()) {
      const auto& matrix = matrices->as_array().front();
      const auto* mismatches = matrix.find("raw_mismatch_count");
      const auto* perfect = matrix.find("perfect_reconstruction");
      const auto* zero_reference = matrix.find("zero_reference");
      const auto* sqnr = matrix.find("sqnr_db");
      CheckNativeBuild(parsed.value());
      GEM16_CHECK(mismatches != nullptr && mismatches->is_integer() && mismatches->as_integer() == 1);
      GEM16_CHECK(perfect != nullptr && perfect->is_bool() && !perfect->as_bool());
      GEM16_CHECK(zero_reference != nullptr && zero_reference->is_bool() && !zero_reference->as_bool());
      GEM16_CHECK(sqnr != nullptr && sqnr->is_number() && sqnr->as_number() == 0.0);
      const auto* difference = matrix.find("difference_sum_squares");
      GEM16_CHECK(difference != nullptr && difference->is_number() && difference->as_number() == 1.0);
    }
  }

  const auto mismatch_job = root / "compare-mismatch.json";
  std::string mismatch_text = ReadText(job1);
  const auto hash_position = mismatch_text.find("\"sha256\":\"");
  GEM16_CHECK(hash_position != std::string::npos);
  if (hash_position != std::string::npos) {
    const auto begin = hash_position + std::string("\"sha256\":\"").size();
    mismatch_text.replace(begin, 64, 64, '0');
  }
  {
    std::ofstream output(mismatch_job, std::ios::binary);
    output << mismatch_text;
  }
  const auto mismatch_metrics = root / "mismatch-metrics.json";
  const auto mismatch_status = gem16::compiler::CompareJobFile(mismatch_job, mismatch_metrics);
  GEM16_CHECK(mismatch_status.code() == gem16::StatusCode::kDataLoss);
  GEM16_CHECK(!std::filesystem::exists(mismatch_metrics));

  const auto global_job = root / "compare-global-v.json";
  {
    std::ofstream output(global_job, std::ios::binary);
    output << ProductionCompareJobJson(source, 1, true);
  }
  GEM16_CHECK(gem16::compiler::CompareJobFile(global_job, root / "global-metrics.json").code() == gem16::StatusCode::kInvalidArgument);
  const auto malformed_name_job = root / "compare-malformed-name.json";
  {
    std::ofstream output(malformed_name_job, std::ios::binary);
    output << ProductionCompareJobJson(source, 1, false, true);
  }
  GEM16_CHECK(gem16::compiler::CompareJobFile(malformed_name_job, root / "malformed-name-metrics.json").code() == gem16::StatusCode::kInvalidArgument);
  const auto malformed_shape_job = root / "compare-malformed-shape.json";
  {
    std::ofstream output(malformed_shape_job, std::ios::binary);
    output << ProductionCompareJobJson(source, 1, false, false, true);
  }
  GEM16_CHECK(gem16::compiler::CompareJobFile(malformed_shape_job, root / "malformed-shape-metrics.json").code() == gem16::StatusCode::kInvalidArgument);

  WriteBytes(source, {0x7F, 0x80, 0x3F, 0x38});
  const auto nan_job = root / "compare-nan.json";
  {
    std::ofstream output(nan_job, std::ios::binary);
    output << CompareJobJson(source, 1);
  }
  GEM16_CHECK(gem16::compiler::CompareJobFile(nan_job, root / "nan-metrics.json").code() == gem16::StatusCode::kInternal);

#ifndef _WIN32
  const auto symlink = root / "compare-link.bin";
  std::error_code link_error;
  std::filesystem::create_symlink(source, symlink, link_error);
  if (!link_error) {
    const auto symlink_job = root / "compare-symlink.json";
    std::ofstream output(symlink_job, std::ios::binary);
    output << CompareJobJson(symlink, 1);
    output.close();
    GEM16_CHECK(gem16::compiler::CompareJobFile(symlink_job, root / "symlink-metrics.json").code() == gem16::StatusCode::kDataLoss);
  }
#endif
  std::filesystem::remove_all(root);
#endif
}

void RunFp8BatchEncoderTests() {
  TestSha256();
  TestCodec();
  TestBatchAndDeterminism();
  TestFailureCleanupAndLargestRow();
  TestNativeComparison();
}

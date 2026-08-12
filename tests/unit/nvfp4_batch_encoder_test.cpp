#include "compiler/nvfp4_batch_encoder.h"
#include "compiler/sha256.h"
#include "test.h"
#include "util/json.h"

#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {
using Byte = std::uint8_t;

std::filesystem::path MakeRoot() {
  const auto root = std::filesystem::temp_directory_path() /
                    ("gem16-nvfp4-test-" + std::to_string(
                         std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(root);
  return root;
}

void WriteBytes(const std::filesystem::path& path, const std::vector<Byte>& bytes) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
}

void WriteBytes(const std::filesystem::path& path, std::size_t size) {
  WriteBytes(path, std::vector<Byte>(size, 0));
}

std::vector<Byte> ReadBytes(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(stream), {}};
}

std::string Hash(const std::vector<Byte>& bytes) {
  return gem16::compiler::Sha256Hex(bytes.data(), bytes.size());
}

std::string NormalizeThreads(std::string value) {
  for (const auto& marker : {std::string("\"threads\":"),
                             std::string("\"analysis_seconds\":"),
                             std::string("\"conversion_seconds\":")}) {
    const auto begin = value.find(marker);
    if (begin != std::string::npos) {
      const auto end = value.find(',', begin + marker.size());
      value.replace(begin + marker.size(), end - (begin + marker.size()), "1");
    }
  }
  return value;
}

std::string Output(std::string_view component, std::string_view name,
                   const std::filesystem::path& path, std::uint64_t bytes,
                   std::uint64_t offset = 0) {
  return "{\"component\":\"" + std::string(component) + "\",\"name\":\"" +
         std::string(name) + "\",\"path\":" + gem16::json::Quote(path.string()) +
         ",\"offset\":" + std::to_string(offset) + ",\"bytes\":" +
         std::to_string(bytes) + "}";
}

std::string Operation(const std::filesystem::path& source, std::string_view source_name,
                      const std::string& source_hash, const std::filesystem::path& output,
                      std::uint64_t rows) {
  const std::string stem = std::string(source_name);
  const auto packed = output.string() + ".packed";
  const auto scale = output.string() + ".scale";
  const auto weight = output.string() + ".weight";
  const auto input = output.string() + ".input";
  return "{\"operation_id\":\"op:" + stem + "\",\"source_name\":\"" + stem +
         "\",\"source_path\":" + gem16::json::Quote(source.string()) +
         ",\"source_sha256\":\"" + source_hash +
         "\",\"source_offset\":0,\"source_bytes\":" +
         std::to_string(rows * 16 * 2) +
         ",\"source_dtype\":\"BF16\",\"logical_shape\":[" +
         std::to_string(rows) + ",16],\"rows\":" + std::to_string(rows) +
         ",\"columns\":16,\"role\":\"routed_expert\",\"axis_transformation\":\"identity\""
         ",\"disk_layout\":\"canonical_row_major_low_nibble_first\",\"runtime_layout\":\"expert_major_sm120_row8_k64\""
         ",\"packed\":" + Output("packed", stem + ".weight_packed", packed, rows * 8) +
         ",\"local_scale\":" + Output("local_scale", stem + ".weight_scale", scale, rows) +
         ",\"weight_global\":" + Output("weight_global", stem + ".weight_global_scale", weight, 4) +
         ",\"input_global\":" + Output("input_global", stem + ".input_global_scale", input, 4) + "}";
}

std::string SharedOperation(const std::filesystem::path& source,
                            std::string_view source_name, const std::string& source_hash,
                            const std::filesystem::path& output, std::uint64_t source_offset,
                            std::uint64_t output_offset) {
  const std::string stem(source_name);
  return "{\"operation_id\":\"fixture:" + stem + "\",\"source_name\":\"" + stem +
         "\",\"source_path\":" + gem16::json::Quote(source.string()) +
         ",\"source_sha256\":\"" + source_hash +
         "\",\"source_offset\":" + std::to_string(source_offset) +
         ",\"source_bytes\":32,\"source_dtype\":\"BF16\",\"logical_shape\":[1,16],\"rows\":1,\"columns\":16,\"role\":\"routed_expert\",\"axis_transformation\":\"identity\",\"disk_layout\":\"canonical_row_major_low_nibble_first\",\"runtime_layout\":\"expert_major_sm120_row8_k64\",\"packed\":" +
         Output("packed", stem + ".weight_packed", output, 8, output_offset) +
         ",\"local_scale\":" + Output("local_scale", stem + ".weight_scale", output, 1, output_offset + 8) +
         ",\"weight_global\":" + Output("weight_global", stem + ".weight_global_scale", output, 4, output_offset + 9) +
         ",\"input_global\":" + Output("input_global", stem + ".input_global_scale", output, 4, output_offset + 13) + "}";
}

std::string SharedFilesJob(const std::filesystem::path& root, std::uint64_t threads) {
  const auto source = root / "shared-source.bf16";
  const auto output = root / "shared-output.bin";
  const auto bytes = ReadBytes(source);
  const std::vector<Byte> first(bytes.begin(), bytes.begin() + 32);
  const std::vector<Byte> second(bytes.begin() + 32, bytes.end());
  const std::string operations =
      SharedOperation(source, "shared0", Hash(first), output, 0, 0) + "," +
      SharedOperation(source, "shared1", Hash(second), output, 32, 17);
  return "{\"schema_version\":1,\"protocol\":\"gem16-nvfp4-direct-v1\",\"artifact_profile\":\"nvfp4-experts-partial-v1\",\"scope\":\"fixture\",\"contract_id\":\"gem16.nvfp4_bf16_group16\",\"contract_version\":1,\"threads\":" +
         std::to_string(threads) + ",\"operations\":[" + operations + "]}";
}

std::string Job(const std::filesystem::path& root, std::uint64_t threads,
                std::string_view scope = "fixture", bool bad_hash = false) {
  std::string operations;
  for (std::size_t index = 0; index < 4; ++index) {
    const std::string name = "fixture" + std::to_string(index);
    const auto source = root / (name + ".bf16");
    const auto bytes = ReadBytes(source);
    if (!operations.empty()) operations += ",";
    operations += Operation(source, name, bad_hash && index == 0 ? std::string(64, '0') : Hash(bytes),
                             root / name, (index == 2 || index == 3) ? 2 : 1);
  }
  return "{\"schema_version\":1,\"protocol\":\"gem16-nvfp4-direct-v1\","
         "\"artifact_profile\":\"nvfp4-experts-partial-v1\",\"scope\":\"" +
         std::string(scope) + "\",\"contract_id\":\"gem16.nvfp4_bf16_group16\","
         "\"contract_version\":1,\"threads\":" + std::to_string(threads) +
         ",\"operations\":[" + operations + "]}";
}

std::string TiedHeadJob(const std::filesystem::path& root,
                         std::string_view profile = "nvfp4-tied-head-partial-v1",
                         std::string_view scope = "tied_head",
                         std::string_view source_name = "model.language_model.embed_tokens.weight",
                         std::string_view operation_id = "nvfp4-head:model.language_model.embed_tokens",
                         std::uint64_t source_bytes = 1476395008U,
                         std::string_view packed_name = "model.language_model.embed_tokens.weight_packed") {
  const auto source = root / "tied-head.bf16";
  const auto packed = root / "tied-head.packed";
  const auto scale = root / "tied-head.scale";
  const auto weight = root / "tied-head.weight";
  const auto input = root / "tied-head.input";
  const auto output = [&](std::string_view component, std::string_view name,
                          const std::filesystem::path& path, std::uint64_t bytes) {
    return "{\"component\":\"" + std::string(component) +
           "\",\"name\":\"" + std::string(name) +
           "\",\"path\":" + gem16::json::Quote(path.string()) +
           ",\"offset\":0,\"bytes\":" + std::to_string(bytes) + "}";
  };
  return "{\"schema_version\":1,\"protocol\":\"gem16-nvfp4-direct-v1\",\"artifact_profile\":\"" +
      std::string(profile) + "\",\"scope\":\"" + std::string(scope) +
      "\",\"contract_id\":\"gem16.nvfp4_bf16_group16\",\"contract_version\":1,\"threads\":1,\"operations\":[{\"operation_id\":\"" +
      std::string(operation_id) + "\",\"source_name\":\"" + std::string(source_name) +
      "\",\"source_path\":" + gem16::json::Quote(source.string()) +
      ",\"source_sha256\":\"" + std::string(64, '0') +
      "\",\"source_offset\":0,\"source_bytes\":" + std::to_string(source_bytes) +
      ",\"source_dtype\":\"BF16\",\"logical_shape\":[262144,2816],\"rows\":262144,\"columns\":2816,\"role\":\"tied_embedding_and_output\",\"axis_transformation\":\"vocabulary,hidden\",\"disk_layout\":\"canonical_row_major_low_nibble_first\",\"runtime_layout\":\"sm120_row8_k64\",\"packed\":" +
      output("packed", packed_name, packed, 369098752U) + ",\"local_scale\":" +
      output("local_scale", "model.language_model.embed_tokens.weight_scale", scale, 46137344U) +
      ",\"weight_global\":" + output("weight_global", "model.language_model.embed_tokens.weight_global_scale", weight, 4U) +
      ",\"input_global\":" + output("input_global", "model.language_model.embed_tokens.input_global_scale", input, 4U) + "}]}";
}

void PrepareFixture(const std::filesystem::path& root) {
  for (std::size_t index = 0; index < 4; ++index) {
    std::vector<Byte> bytes;
    const std::uint16_t values[] = {
        0x8000, 0x40c0, 0xbfc0, 0x3f80, 0x4000, 0x4040, 0x4080, 0x40c0,
        0xc0c0, 0x3f00, 0x3f80, 0x4000, 0x4040, 0x4080, 0x40c0, 0x0000};
    const std::uint16_t tie_values[] = {
        0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x3f80,
        0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x3f80};
    const std::size_t rows = (index == 2 || index == 3) ? 2 : 1;
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t value = 0; value < 16; ++value) {
        const std::uint16_t encoded =
            index == 1 ? 0 : index == 2 && row == 1 ? 0x3580
                       : index == 3 && row == 0 ? tie_values[value]
                       : index == 3 && row == 1 ? 0x36db : values[value];
        bytes.push_back(static_cast<Byte>(encoded));
        bytes.push_back(static_cast<Byte>(encoded >> 8));
      }
    }
    WriteBytes(root / ("fixture" + std::to_string(index) + ".bf16"), bytes);
    WriteBytes(root / ("fixture" + std::to_string(index) + ".packed"), rows * 8);
    WriteBytes(root / ("fixture" + std::to_string(index) + ".scale"), rows);
    WriteBytes(root / ("fixture" + std::to_string(index) + ".weight"), 4);
    WriteBytes(root / ("fixture" + std::to_string(index) + ".input"), 4);
  }
}

void TestDeterministicFixture() {
  const auto root = MakeRoot();
  PrepareFixture(root);
  std::vector<Byte> expected_packed;
  std::vector<Byte> expected_scale;
  std::string expected_telemetry;
  for (const std::uint64_t threads : {1U, 2U, 4U, 64U}) {
    const auto job = root / ("job-" + std::to_string(threads) + ".json");
    const auto telemetry = root / ("telemetry-" + std::to_string(threads) + ".json");
    std::ofstream stream(job);
    stream << Job(root, threads);
    stream.close();
    const auto status = gem16::compiler::EncodeNvfp4JobFile(job, telemetry);
    GEM16_CHECK(status.ok());
    if (!status.ok()) continue;
    const auto packed = ReadBytes(root / "fixture0.packed");
    const auto scale = ReadBytes(root / "fixture0.scale");
    const auto current_telemetry = ReadBytes(telemetry);
    if (threads == 1) {
      expected_packed = packed;
      expected_scale = scale;
      expected_telemetry.assign(current_telemetry.begin(), current_telemetry.end());
    } else {
      GEM16_CHECK(packed == expected_packed);
      GEM16_CHECK(scale == expected_scale);
      const std::string normalized = NormalizeThreads(
          std::string(current_telemetry.begin(), current_telemetry.end()));
      GEM16_CHECK(normalized == NormalizeThreads(expected_telemetry));
    }
    GEM16_CHECK(!packed.empty() && packed[0] != 0);
    GEM16_CHECK(!scale.empty() && scale[0] != 0);
    if (threads == 1) {
      GEM16_CHECK(expected_telemetry.find("\"code_histogram\":[") != std::string::npos);
      GEM16_CHECK(expected_telemetry.find("\"scale_histogram\":[") != std::string::npos);
      GEM16_CHECK(expected_telemetry.find("\"underflow_blocks\":") != std::string::npos);
      std::vector<Byte> golden = {0x70, 0x2b, 0x54, 0x76, 0x1f, 0x42, 0x65, 0x07};
      golden[0] = 0x78;
      GEM16_CHECK(packed == golden);
      GEM16_CHECK(scale == std::vector<Byte>{0x7e});
      GEM16_CHECK((ReadBytes(root / "fixture0.packed")[0] & 0x0fU) == 8U);
      GEM16_CHECK(ReadBytes(root / "fixture1.packed") == std::vector<Byte>(8, 0));
      GEM16_CHECK(ReadBytes(root / "fixture1.scale") == std::vector<Byte>{0});
      GEM16_CHECK(ReadBytes(root / "fixture2.scale") == std::vector<Byte>({0x7e, 0x00}));
      GEM16_CHECK(ReadBytes(root / "fixture3.scale") == std::vector<Byte>({0x7e, 0x01}));
    }
  }
  std::filesystem::remove_all(root);
}

void TestHashAndFullScopeRejection() {
  const auto root = MakeRoot();
  PrepareFixture(root);
  const auto bad_job = root / "bad.json";
  {
    std::ofstream stream(bad_job);
    stream << Job(root, 1, "fixture", true);
  }
  const auto bad_packed_before = ReadBytes(root / "fixture0.packed");
  const auto bad_scale_before = ReadBytes(root / "fixture0.scale");
  const auto bad_weight_before = ReadBytes(root / "fixture0.weight");
  const auto bad_input_before = ReadBytes(root / "fixture0.input");
  auto status = gem16::compiler::EncodeNvfp4JobFile(bad_job, root / "bad.telemetry");
  GEM16_CHECK(status.code() == gem16::StatusCode::kDataLoss);
  GEM16_CHECK(ReadBytes(root / "fixture0.packed") == bad_packed_before);
  GEM16_CHECK(ReadBytes(root / "fixture0.scale") == bad_scale_before);
  GEM16_CHECK(ReadBytes(root / "fixture0.weight") == bad_weight_before);
  GEM16_CHECK(ReadBytes(root / "fixture0.input") == bad_input_before);
  const auto full_job = root / "full.json";
  {
    std::ofstream stream(full_job);
    stream << Job(root, 1, "full");
  }
  status = gem16::compiler::EncodeNvfp4JobFile(full_job, root / "full.telemetry");
  GEM16_CHECK(status.code() == gem16::StatusCode::kInvalidArgument);
  const auto valid_job = root / "valid.json";
  {
    std::ofstream stream(valid_job);
    stream << Job(root, 1);
  }
  const auto telemetry = root / "valid.telemetry";
  status = gem16::compiler::EncodeNvfp4JobFile(valid_job, telemetry);
  GEM16_CHECK(status.ok());
  status = gem16::compiler::EncodeNvfp4JobFile(valid_job, telemetry);
  GEM16_CHECK(status.code() == gem16::StatusCode::kInvalidArgument);

  // The shared 1 GiB source bound remains in force for M06 fixtures. Keep
  // shape, rows, columns, source bytes, and output sizes mutually consistent
  // so the profile limit, rather than a shape check, is the rejection reason.
  const auto oversized_m06 = root / "oversized-m06.json";
  std::string oversized_text = Job(root, 1);
  const std::string old_shape =
      "\"source_bytes\":32,\"source_dtype\":\"BF16\",\"logical_shape\":[1,16],\"rows\":1,\"columns\":16";
  const std::string new_shape =
      "\"source_bytes\":2147483648,\"source_dtype\":\"BF16\",\"logical_shape\":[1024,65536,16],\"rows\":67108864,\"columns\":16";
  const auto source_bytes_position = oversized_text.find(old_shape);
  GEM16_CHECK(source_bytes_position != std::string::npos);
  if (source_bytes_position != std::string::npos) {
    oversized_text.replace(source_bytes_position, old_shape.size(), new_shape);
    const std::string old_outputs =
        "\"bytes\":8},\"local_scale\":{\"component\":\"local_scale\",\"name\":\"fixture0.weight_scale\",\"path\":";
    const std::string new_outputs =
        "\"bytes\":536870912},\"local_scale\":{\"component\":\"local_scale\",\"name\":\"fixture0.weight_scale\",\"path\":";
    const auto output_position = oversized_text.find(old_outputs);
    GEM16_CHECK(output_position != std::string::npos);
    if (output_position != std::string::npos) {
      oversized_text.replace(output_position, old_outputs.size(), new_outputs);
    }
    const std::string old_local_bytes = "\"offset\":0,\"bytes\":1}";
    const auto local_name_position =
        oversized_text.find("\"name\":\"fixture0.weight_scale\"");
    const auto local_position = local_name_position == std::string::npos
                                    ? std::string::npos
                                    : oversized_text.find(old_local_bytes, local_name_position);
    GEM16_CHECK(local_position != std::string::npos);
    if (local_position != std::string::npos) {
      oversized_text.replace(local_position, old_local_bytes.size(),
                             "\"offset\":0,\"bytes\":67108864}");
    }
  }
  std::ofstream(oversized_m06) << oversized_text;
  status = gem16::compiler::EncodeNvfp4JobFile(oversized_m06, root / "oversized-m06.telemetry");
  GEM16_CHECK(status.code() == gem16::StatusCode::kInvalidArgument);

  // M07's larger bound is reachable only through its canonical identity, and
  // the Debug unit binary must reject it before source/output I/O.
  const auto tied_job = root / "tied-head.json";
  std::ofstream(tied_job) << TiedHeadJob(root);
  status = gem16::compiler::EncodeNvfp4JobFile(tied_job, root / "tied-head.telemetry");
  if (gem16::compiler::Nvfp4BuildSupportsFullJob()) {
    GEM16_CHECK(status.code() == gem16::StatusCode::kIoError);
  } else {
    GEM16_CHECK(status.code() == gem16::StatusCode::kInvalidArgument);
  }
  for (const auto& malformed : {
           TiedHeadJob(root, "nvfp4-experts-partial-v1", "tied_head"),
           TiedHeadJob(root, "nvfp4-tied-head-partial-v1", "fixture"),
           TiedHeadJob(root, "nvfp4-tied-head-partial-v1", "tied_head", "lm_head.weight"),
           TiedHeadJob(root, "nvfp4-tied-head-partial-v1", "tied_head",
                       "model.language_model.embed_tokens.weight", "wrong-operation"),
           TiedHeadJob(root, "nvfp4-tied-head-partial-v1", "tied_head",
                       "model.language_model.embed_tokens.weight",
                       "nvfp4-head:model.language_model.embed_tokens", 1476395007U),
           TiedHeadJob(root, "nvfp4-tied-head-partial-v1", "tied_head",
                       "model.language_model.embed_tokens.weight",
                       "nvfp4-head:model.language_model.embed_tokens", 1476395008U,
                       "lm_head.weight_packed")}) {
    {
      std::ofstream malformed_job(root / "malformed-tied-head.json");
      malformed_job << malformed;
    }
    status = gem16::compiler::EncodeNvfp4JobFile(root / "malformed-tied-head.json",
                                                   root / "malformed-tied-head.telemetry");
    GEM16_CHECK(status.code() == gem16::StatusCode::kInvalidArgument);
  }
  std::filesystem::remove_all(root);
}

void TestSharedFileRanges() {
  const auto root = MakeRoot();
  PrepareFixture(root);
  const auto first = ReadBytes(root / "fixture0.bf16");
  const auto second = ReadBytes(root / "fixture1.bf16");
  std::vector<Byte> source = first;
  source.insert(source.end(), second.begin(), second.end());
  WriteBytes(root / "shared-source.bf16", source);
  WriteBytes(root / "shared-output.bin", 34);
  const auto job = root / "shared.json";
  std::ofstream(job) << SharedFilesJob(root, 2);
  const auto status = gem16::compiler::EncodeNvfp4JobFile(job, root / "shared.telemetry");
  GEM16_CHECK(status.ok());
  const auto output = ReadBytes(root / "shared-output.bin");
  GEM16_CHECK(output.size() == 34U);
  GEM16_CHECK(output[0] != 0U || output[17] != 0U);
  std::filesystem::remove_all(root);
}

void TestRank3Fixture() {
  const auto root = MakeRoot();
  PrepareFixture(root);
  const auto job = root / "rank3.json";
  std::string text = Job(root, 1);
  const std::string rank2 = "\"logical_shape\":[2,16]";
  const auto position = text.find(rank2);
  GEM16_CHECK(position != std::string::npos);
  if (position != std::string::npos) text.replace(position, rank2.size(), "\"logical_shape\":[2,1,16]");
  std::ofstream(job) << text;
  const auto status = gem16::compiler::EncodeNvfp4JobFile(job, root / "rank3.telemetry");
  GEM16_CHECK(status.ok());
  GEM16_CHECK(ReadBytes(root / "fixture2.packed").size() == 16U);
  GEM16_CHECK(ReadBytes(root / "fixture2.scale").size() == 2U);
  std::filesystem::remove_all(root);
}

void TestCodecAndExitHelpers() {
  const auto brute_e2m1 = [](float value) {
    const float magnitude = std::fabs(value);
    if (std::isnan(magnitude)) return static_cast<std::uint8_t>(0U);
    if (magnitude >= 6.0F) {
      return static_cast<std::uint8_t>(7U | (std::signbit(value) ? 8U : 0U));
    }
    std::uint8_t best = 0;
    float best_error = std::numeric_limits<float>::infinity();
    for (std::uint8_t code = 0; code < 8; ++code) {
      const float error = std::fabs(magnitude -
                                    gem16::compiler::DecodeNvfp4E2M1(code));
      if (error < best_error ||
          (error == best_error && (code & 1U) == 0U && (best & 1U) != 0U)) {
        best = code;
        best_error = error;
      }
    }
    return static_cast<std::uint8_t>(best | (std::signbit(value) ? 8U : 0U));
  };
  const auto brute_e4m3 = [](float value) {
    const float magnitude = std::fabs(value);
    if (magnitude >= 448.0F) {
      return static_cast<std::uint8_t>(0x7eU | (std::signbit(value) ? 0x80U : 0U));
    }
    std::uint8_t best = 0;
    float best_error = std::numeric_limits<float>::infinity();
    for (std::uint16_t code = 0; code < 0x7fU; ++code) {
      const float error = std::fabs(magnitude - gem16::compiler::DecodeNvfp4E4M3(
                                                     static_cast<Byte>(code)));
      if (error < best_error ||
          (error == best_error && (code & 1U) == 0U && (best & 1U) != 0U)) {
        best = static_cast<Byte>(code);
        best_error = error;
      }
    }
    return static_cast<std::uint8_t>(best | (std::signbit(value) ? 0x80U : 0U));
  };
  std::uint32_t bits = 0x12345678U;
  for (std::size_t sample = 0; sample < 10000; ++sample) {
    bits = bits * 1664525U + 1013904223U;
    const float value = static_cast<float>(bits & 0xffffU) * (6.0F / 65535.0F);
    GEM16_CHECK(gem16::compiler::EncodeNvfp4E2M1(value) == brute_e2m1(value));
    GEM16_CHECK(gem16::compiler::EncodeNvfp4E4M3(value) == brute_e4m3(value));
  }
  for (std::uint16_t code = 0; code < 7; ++code) {
    const float midpoint = (gem16::compiler::DecodeNvfp4E2M1(static_cast<Byte>(code)) +
                            gem16::compiler::DecodeNvfp4E2M1(static_cast<Byte>(code + 1U))) * 0.5F;
    for (const float value : {midpoint, std::nextafter(midpoint, 0.0F),
                              std::nextafter(midpoint, std::numeric_limits<float>::infinity()),
                              -midpoint}) {
      GEM16_CHECK(gem16::compiler::EncodeNvfp4E2M1(value) == brute_e2m1(value));
    }
  }
  for (std::uint16_t code = 0; code < 0x7eU; ++code) {
    const float midpoint = (gem16::compiler::DecodeNvfp4E4M3(static_cast<Byte>(code)) +
                            gem16::compiler::DecodeNvfp4E4M3(static_cast<Byte>(code + 1U))) * 0.5F;
    for (const float value : {midpoint, std::nextafter(midpoint, 0.0F),
                              std::nextafter(midpoint, std::numeric_limits<float>::infinity()),
                              -midpoint}) {
      GEM16_CHECK(gem16::compiler::EncodeNvfp4E4M3(value) == brute_e4m3(value));
    }
  }
  GEM16_CHECK(std::string(gem16::compiler::Nvfp4BuildInfoJson()).find(
                  "gem16-nvfp4-direct-v1") != std::string::npos);
  for (std::uint16_t code = 0; code < 16; ++code) {
    const float decoded = gem16::compiler::DecodeNvfp4E2M1(static_cast<Byte>(code));
    GEM16_CHECK(gem16::compiler::EncodeNvfp4E2M1(decoded) == code);
  }
  GEM16_CHECK(gem16::compiler::EncodeNvfp4E2M1(-0.0F) == 8U);
  GEM16_CHECK(gem16::compiler::EncodeNvfp4E2M1(0.25F) == 0U);
  GEM16_CHECK(gem16::compiler::EncodeNvfp4E2M1(-0.25F) == 8U);
  GEM16_CHECK(gem16::compiler::EncodeNvfp4E2M1(6.0F) == 7U);
  GEM16_CHECK(gem16::compiler::EncodeNvfp4E2M1(-6.0F) == 15U);
  GEM16_CHECK(gem16::compiler::EncodeNvfp4E2M1(1000.0F) == 7U);
  GEM16_CHECK(gem16::compiler::EncodeNvfp4E2M1(-1000.0F) == 15U);
  GEM16_CHECK(gem16::compiler::EncodeNvfp4E2M1(
                  std::numeric_limits<float>::max()) == 7U);
  GEM16_CHECK(gem16::compiler::EncodeNvfp4E2M1(
                  -std::numeric_limits<float>::max()) == 15U);
  GEM16_CHECK(gem16::compiler::EncodeNvfp4E2M1(
                  std::numeric_limits<float>::infinity()) == 7U);
  GEM16_CHECK(gem16::compiler::EncodeNvfp4E2M1(
                  -std::numeric_limits<float>::infinity()) == 15U);
  GEM16_CHECK(gem16::compiler::EncodeNvfp4E2M1(
                  std::numeric_limits<float>::quiet_NaN()) == 0U);
  for (std::uint16_t code = 0; code < 0x7f; ++code) {
    const auto positive = static_cast<Byte>(code);
    const auto negative = static_cast<Byte>(code | 0x80U);
    GEM16_CHECK(gem16::compiler::EncodeNvfp4E4M3(
                    gem16::compiler::DecodeNvfp4E4M3(positive)) == positive);
    GEM16_CHECK(gem16::compiler::EncodeNvfp4E4M3(
                    gem16::compiler::DecodeNvfp4E4M3(negative)) == negative);
  }
  GEM16_CHECK(gem16::compiler::EncodeNvfp4E4M3(-0.0F) == 0x80U);
  GEM16_CHECK(gem16::compiler::EncodeNvfp4E4M3(448.0F) == 0x7eU);
  GEM16_CHECK(gem16::compiler::EncodeNvfp4E4M3(-448.0F) == 0xfeU);
  GEM16_CHECK(gem16::compiler::ExitCodeForNvfp4Status(
                  gem16::StatusCode::kInvalidArgument) == 2);
  GEM16_CHECK(gem16::compiler::ExitCodeForNvfp4Status(
                  gem16::StatusCode::kDataLoss) == 4);
  GEM16_CHECK(gem16::compiler::ExitCodeForNvfp4Status(
                  gem16::StatusCode::kIoError) == 5);
}

void TestPreflightGuards() {
  const auto root = MakeRoot();
  PrepareFixture(root);

  // A hardlink to another source tensor is legal as a path, but the exact
  // overlapping source range must still be rejected.
  const auto hardlink = root / "fixture1-alias.bf16";
  std::filesystem::create_hard_link(root / "fixture0.bf16", hardlink);
  const auto hardlink_job = root / "hardlink.json";
  {
    std::ofstream stream(hardlink_job);
    stream << Job(root, 1);
  }
  std::string hardlink_text;
  {
    std::ifstream stream(hardlink_job);
    hardlink_text.assign(std::istreambuf_iterator<char>(stream), {});
  }
  const auto original_path = (root / "fixture1.bf16").string();
  const auto alias_path = hardlink.string();
  const auto path_position = hardlink_text.find(original_path);
  GEM16_CHECK(path_position != std::string::npos);
  if (path_position != std::string::npos) hardlink_text.replace(path_position, original_path.size(), alias_path);
  const auto hash0 = Hash(ReadBytes(root / "fixture0.bf16"));
  const auto hash1 = Hash(ReadBytes(root / "fixture1.bf16"));
  const auto hash_position = hardlink_text.find(hash1);
  GEM16_CHECK(hash_position != std::string::npos);
  if (hash_position != std::string::npos) hardlink_text.replace(hash_position, hash1.size(), hash0);
  std::ofstream(hardlink_job) << hardlink_text;
  auto status = gem16::compiler::EncodeNvfp4JobFile(hardlink_job, root / "hardlink.telemetry");
  GEM16_CHECK(status.code() == gem16::StatusCode::kInvalidArgument);

  // Symlink source paths are rejected even when the target is otherwise valid.
  const auto symlink = root / "fixture2-link.bf16";
  std::filesystem::create_symlink(root / "fixture2.bf16", symlink);
  const auto symlink_job = root / "symlink.json";
  {
    std::ofstream stream(symlink_job);
    stream << Job(root, 1);
  }
  std::string symlink_text;
  {
    std::ifstream stream(symlink_job);
    symlink_text.assign(std::istreambuf_iterator<char>(stream), {});
  }
  const auto source2 = (root / "fixture2.bf16").string();
  const auto source2_position = symlink_text.find(source2);
  GEM16_CHECK(source2_position != std::string::npos);
  if (source2_position != std::string::npos) symlink_text.replace(source2_position, source2.size(), symlink.string());
  std::ofstream(symlink_job) << symlink_text;
  status = gem16::compiler::EncodeNvfp4JobFile(symlink_job, root / "symlink.telemetry");
  GEM16_CHECK(status.code() == gem16::StatusCode::kIoError);

  // An output path hardlinked to a source is never an allowed destination.
  const auto alias_job = root / "output-source-alias.json";
  {
    std::ofstream stream(alias_job);
    stream << Job(root, 1);
  }
  std::string alias_text;
  {
    std::ifstream stream(alias_job);
    alias_text.assign(std::istreambuf_iterator<char>(stream), {});
  }
  const auto packed0 = (root / "fixture0.packed").string();
  const auto source0 = (root / "fixture0.bf16").string();
  const auto packed_position = alias_text.find(packed0);
  GEM16_CHECK(packed_position != std::string::npos);
  if (packed_position != std::string::npos) alias_text.replace(packed_position, packed0.size(), source0);
  std::ofstream(alias_job) << alias_text;
  status = gem16::compiler::EncodeNvfp4JobFile(alias_job, root / "output-source-alias.telemetry");
  GEM16_CHECK(status.code() == gem16::StatusCode::kInvalidArgument);

  // Non-finite BF16 is rejected during the source-only pass, before output I/O.
  std::vector<Byte> nonfinite(32, 0);
  nonfinite[0] = 0xc0;
  nonfinite[1] = 0x7f;
  WriteBytes(root / "fixture0.bf16", nonfinite);
  const auto nonfinite_job = root / "nonfinite.json";
  {
    std::ofstream stream(nonfinite_job);
    stream << Job(root, 1);
  }
  const auto packed_before = ReadBytes(root / "fixture0.packed");
  const auto scale_before = ReadBytes(root / "fixture0.scale");
  const auto weight_before = ReadBytes(root / "fixture0.weight");
  const auto input_before = ReadBytes(root / "fixture0.input");
  status = gem16::compiler::EncodeNvfp4JobFile(nonfinite_job, root / "nonfinite.telemetry");
  GEM16_CHECK(status.code() == gem16::StatusCode::kDataLoss);
  GEM16_CHECK(ReadBytes(root / "fixture0.packed") == packed_before);
  GEM16_CHECK(ReadBytes(root / "fixture0.scale") == scale_before);
  GEM16_CHECK(ReadBytes(root / "fixture0.weight") == weight_before);
  GEM16_CHECK(ReadBytes(root / "fixture0.input") == input_before);

  std::filesystem::remove_all(root);
}

void TestMalformedThreadCount() {
  const auto root = MakeRoot();
  PrepareFixture(root);
  const auto job = root / "malformed.json";
  {
    std::ofstream stream(job);
    stream << Job(root, 65);
  }
  const auto status = gem16::compiler::EncodeNvfp4JobFile(job, root / "telemetry.json");
  GEM16_CHECK(status.code() == gem16::StatusCode::kInvalidArgument);
  std::filesystem::remove_all(root);
}
}  // namespace

void RunNvfp4BatchEncoderTests() {
#ifdef _WIN32
  GEM16_CHECK(gem16::compiler::EncodeNvfp4JobFile({}, {}).code() == gem16::StatusCode::kUnsupported);
#else
  TestDeterministicFixture();
  TestHashAndFullScopeRejection();
  TestSharedFileRanges();
  TestRank3Fixture();
  TestCodecAndExitHelpers();
  TestPreflightGuards();
  TestMalformedThreadCount();
#endif
}

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "cuda/engine/gemma4_26b_reference.h"
#include "gem16/tokenizer.h"

namespace {

struct Options {
  std::filesystem::path model;
  std::filesystem::path tokenizer;
  std::filesystem::path output;
  std::filesystem::path token_ids;
  std::string prompt;
  std::uint32_t layer = 0U;
  std::uint64_t context = 4096U;
  int device = 0;
};

bool Unsigned(std::string_view text, std::uint64_t* value) {
  try {
    std::size_t used = 0U;
    *value = std::stoull(std::string(text), &used);
    return used == text.size();
  } catch (...) {
    return false;
  }
}

bool Parse(int argc, char** argv, Options* options) {
  for (int index = 1; index < argc; ++index) {
    if (++index >= argc) return false;
    const std::string_view key(argv[index - 1]);
    const std::string value(argv[index]);
    if (key == "--model") options->model = value;
    else if (key == "--tokenizer") options->tokenizer = value;
    else if (key == "--output") options->output = value;
    else if (key == "--token-ids") options->token_ids = value;
    else if (key == "--prompt") options->prompt = value;
    else if (key == "--layer") {
      std::uint64_t parsed = 0U;
      if (!Unsigned(value, &parsed) || parsed >= 30U) return false;
      options->layer = static_cast<std::uint32_t>(parsed);
    } else if (key == "--context") {
      if (!Unsigned(value, &options->context) || options->context == 0U) return false;
    } else if (key == "--device") {
      std::uint64_t parsed = 0U;
      if (!Unsigned(value, &parsed) || parsed > 63U) return false;
      options->device = static_cast<int>(parsed);
    } else {
      return false;
    }
  }
  return !options->model.empty() && !options->output.empty() &&
         (options->prompt.empty() != options->token_ids.empty());
}

template <typename T>
bool Write(std::ofstream& output, const T& value) {
  output.write(reinterpret_cast<const char*>(&value), sizeof(value));
  return output.good();
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!Parse(argc, argv, &options)) {
    std::cerr << "usage: gem16-trellis35-calibration --model DIR --output BIN "
                 "(--prompt TEXT | --token-ids U32LE) [--tokenizer DIR] "
                 "[--layer N] [--context N] [--device N]\n";
    return 2;
  }
  std::vector<std::uint32_t> token_values;
  if (!options.token_ids.empty()) {
    if constexpr (std::endian::native != std::endian::little) return 3;
    std::error_code error;
    const auto bytes = std::filesystem::file_size(options.token_ids, error);
    if (error || bytes == 0U || bytes % sizeof(std::uint32_t) != 0U ||
        bytes / sizeof(std::uint32_t) > options.context) return 3;
    token_values.resize(bytes / sizeof(std::uint32_t));
    std::ifstream input(options.token_ids, std::ios::binary);
    input.read(reinterpret_cast<char*>(token_values.data()),
               static_cast<std::streamsize>(bytes));
    if (!input || std::any_of(token_values.begin(), token_values.end(),
                              [](std::uint32_t token) { return token >= 262144U; })) return 3;
  } else {
    if (options.tokenizer.empty()) options.tokenizer = options.model;
    auto tokenizer = gem16::Tokenizer::Load(options.tokenizer / "tokenizer.json");
    if (!tokenizer.ok()) {
      std::cerr << tokenizer.status().message() << '\n';
      return 3;
    }
    const std::string text = "<bos><|turn>user\n" + options.prompt +
                             "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
    auto tokens = tokenizer.value().Encode(text);
    if (!tokens.ok()) {
      std::cerr << "Trellis35 calibration prompt tokenization failed\n";
      return 3;
    }
    token_values = std::move(tokens.value());
    if (token_values.empty() || token_values.size() > options.context) return 3;
  }
  if (cudaSetDevice(options.device) != cudaSuccess) return 4;
  auto engine = gem16::internal::Gemma4Moe26BReferenceEngine::Create(
      options.model, options.context, options.device,
      gem16::internal::Gemma4Moe26BBackend::kReference);
  if (!engine.ok()) {
    std::cerr << engine.status().message() << '\n';
    return 4;
  }
  auto status = engine.value().ConfigureMoeCalibrationCapture(options.layer);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 4;
  }
  std::ofstream output(options.output, std::ios::binary | std::ios::trunc);
  if (!output) return 5;
  const std::array<char, 8> magic{'G', '1', '6', 'T', '3', '5', 'C', '1'};
  output.write(magic.data(), magic.size());
  const std::uint32_t version = 1U;
  const std::uint32_t records = static_cast<std::uint32_t>(token_values.size());
  if (!Write(output, version) || !Write(output, options.layer) || !Write(output, records)) return 5;
  std::vector<float> gate_up(2816U);
  std::vector<float> down(8U * 704U);
  std::array<std::uint32_t, 8> ids{};
  for (std::uint32_t position = 0U; position < records; ++position) {
    status = engine.value().ForwardToken(token_values[position]);
    if (status.ok()) {
      status = engine.value().CopyMoeCalibrationCapture(gate_up, down, ids);
    }
    if (!status.ok()) {
      std::cerr << status.message() << '\n';
      return 6;
    }
    if (!Write(output, position)) return 5;
    output.write(reinterpret_cast<const char*>(ids.data()), sizeof(ids));
    output.write(reinterpret_cast<const char*>(gate_up.data()),
                 static_cast<std::streamsize>(gate_up.size() * sizeof(float)));
    output.write(reinterpret_cast<const char*>(down.data()),
                 static_cast<std::streamsize>(down.size() * sizeof(float)));
    if (!output) return 5;
  }
  std::cout << "trellis35_calibration_ok layer=" << options.layer
            << " records=" << records << '\n';
  return 0;
}

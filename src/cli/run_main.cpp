#include <charconv>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "gem16/engine.h"

namespace {

bool ParseUnsigned(std::string_view text, std::uint64_t& value) {
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

bool ParseTokenIds(std::string_view text, std::vector<std::uint32_t>& tokens) {
  if (text.empty()) return false;
  std::size_t begin = 0;
  while (begin < text.size()) {
    const std::size_t comma = text.find(',', begin);
    const std::size_t end = comma == std::string_view::npos ? text.size() : comma;
    std::uint64_t token = 0;
    if (end == begin || !ParseUnsigned(text.substr(begin, end - begin), token) ||
        token > std::numeric_limits<std::uint32_t>::max()) {
      return false;
    }
    tokens.push_back(static_cast<std::uint32_t>(token));
    if (comma == std::string_view::npos) break;
    begin = comma + 1U;
  }
  return !tokens.empty();
}

bool ParseTokenIdsFile(std::string_view path, std::vector<std::uint32_t>& tokens) {
  std::ifstream input(std::string(path), std::ios::binary);
  if (!input) return false;
  const std::string text{std::istreambuf_iterator<char>(input),
                         std::istreambuf_iterator<char>()};
  return !input.bad() && ParseTokenIds(text, tokens);
}

void PrintUsage() {
  std::cout
      << "Usage:\n"
      << "  gem16-run --print-kernel-capabilities\n"
      << "  gem16-run --model <checkpoint> --input-token-ids <id,id,...>\n"
      << "              [--input-token-ids-file <comma-separated-token-file>]\n"
      << "              [--teacher-forced-token-ids <id,id,...>]\n"
      << "              [--stop-token-ids <id,id,...>]\n"
      << "              [--suppress-token-ids <id,id,...>]\n"
      << "              [--dump-logits <raw-f32-path>]\n"
      << "              [--dump-state <path> --dump-state-position N]\n"
      << "              [--kv-cache fp8|bf16]\n"
      << "              [--max-tokens N] [--max-context N] --greedy\n"
      << "\nTeacher forcing uses the forced list length as the number of output\n"
      << "positions and reports the model's greedy prediction at every position.\n"
      << "The inference path is a correctness characterization and is not benchmark-qualified.\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--print-kernel-capabilities") {
    gem16::PrintKernelCapabilities(std::cout);
    return 0;
  }
  if (argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
    PrintUsage();
    return 0;
  }

  gem16::GreedyInferenceOptions options;
  bool greedy = false;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--model" && index + 1 < argc) {
      options.model_directory = argv[++index];
    } else if (argument == "--input-token-ids" && index + 1 < argc) {
      if (!options.input_token_ids.empty() ||
          !ParseTokenIds(argv[++index], options.input_token_ids)) {
        std::cerr << "error: --input-token-ids must be a comma-separated unsigned "
                     "list and cannot be combined with --input-token-ids-file\n";
        return 64;
      }
    } else if (argument == "--input-token-ids-file" && index + 1 < argc) {
      if (!options.input_token_ids.empty() ||
          !ParseTokenIdsFile(argv[++index], options.input_token_ids)) {
        std::cerr << "error: --input-token-ids-file must name a readable "
                     "comma-separated unsigned list and cannot be combined with "
                     "--input-token-ids\n";
        return 64;
      }
    } else if (argument == "--max-tokens" && index + 1 < argc) {
      if (!ParseUnsigned(argv[++index], options.max_generated_tokens)) {
        std::cerr << "error: --max-tokens must be an unsigned integer\n";
        return 64;
      }
    } else if (argument == "--teacher-forced-token-ids" && index + 1 < argc) {
      if (!ParseTokenIds(argv[++index], options.teacher_forced_token_ids)) {
        std::cerr
            << "error: --teacher-forced-token-ids must be a comma-separated unsigned list\n";
        return 64;
      }
    } else if (argument == "--max-context" && index + 1 < argc) {
      if (!ParseUnsigned(argv[++index], options.max_context_tokens)) {
        std::cerr << "error: --max-context must be an unsigned integer\n";
        return 64;
      }
    } else if (argument == "--stop-token-ids" && index + 1 < argc) {
      if (!ParseTokenIds(argv[++index], options.stop_token_ids)) {
        std::cerr << "error: --stop-token-ids must be a comma-separated unsigned list\n";
        return 64;
      }
    } else if (argument == "--suppress-token-ids" && index + 1 < argc) {
      if (!ParseTokenIds(argv[++index], options.suppressed_token_ids)) {
        std::cerr << "error: --suppress-token-ids must be a comma-separated unsigned list\n";
        return 64;
      }
    } else if (argument == "--dump-logits" && index + 1 < argc) {
      options.logits_dump_path = argv[++index];
    } else if (argument == "--dump-state" && index + 1 < argc) {
      options.state_dump_path = argv[++index];
    } else if (argument == "--dump-state-position" && index + 1 < argc) {
      std::uint64_t position = 0;
      if (!ParseUnsigned(argv[++index], position)) {
        std::cerr << "error: --dump-state-position must be an unsigned integer\n";
        return 64;
      }
      options.state_dump_position = position;
    } else if (argument == "--kv-cache" && index + 1 < argc) {
      const std::string_view mode = argv[++index];
      if (mode == "fp8") {
        options.kv_cache_mode = gem16::KvCacheMode::kCheckpointFp8;
      } else if (mode == "bf16") {
        options.kv_cache_mode = gem16::KvCacheMode::kBf16Correctness;
      } else {
        std::cerr << "error: --kv-cache must be fp8 or bf16\n";
        return 64;
      }
    } else if (argument == "--greedy") {
      greedy = true;
    } else {
      std::cerr << "error: unknown or incomplete option: " << argument << '\n';
      return 64;
    }
  }
  if (!greedy) {
    std::cerr << "error: the initial inference path requires explicit --greedy\n";
    return 64;
  }
  auto result = gem16::RunGreedyInference(options);
  if (!result.ok()) {
    std::cerr << "error: " << result.status().message()
              << "; no precision fallback was attempted\n";
    return 2;
  }
  const gem16::Status write = gem16::WriteGreedyInferenceJson(result.value(), std::cout);
  if (!write.ok()) {
    std::cerr << "error: " << write.message() << '\n';
    return 1;
  }
  return 0;
}

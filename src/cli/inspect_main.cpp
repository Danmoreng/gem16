#include <filesystem>
#include <fstream>
#include <iostream>
#include <string_view>

#include "gem16/model.h"

namespace {

void Usage(std::ostream& output) {
  output << "Usage: gem16-inspect --model <checkpoint-dir> [--json <manifest.json>|-] [--validate]\n";
}

}  // namespace

int main(int argc, char** argv) {
  gem16::InspectOptions options;
  std::filesystem::path json_path;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--help" || argument == "-h") {
      Usage(std::cout);
      return 0;
    }
    if (argument == "--validate") {
      options.validate = true;
      continue;
    }
    if ((argument == "--model" || argument == "--json") && index + 1 < argc) {
      const std::filesystem::path value(argv[++index]);
      if (argument == "--model") options.model_directory = value;
      else json_path = value;
      continue;
    }
    std::cerr << "error: unknown or incomplete argument: " << argument << '\n';
    Usage(std::cerr);
    return 2;
  }

  auto manifest = gem16::InspectCheckpoint(options);
  if (!manifest.ok()) {
    std::cerr << "error: " << manifest.status().message() << '\n';
    return 1;
  }
  if (json_path == "-") {
    const auto status = gem16::WriteManifestJson(manifest.value(), std::cout);
    if (!status.ok()) {
      std::cerr << "error: " << status.message() << '\n';
      return 1;
    }
  } else {
    gem16::PrintManifestSummary(manifest.value(), std::cout);
    if (!json_path.empty()) {
      std::ofstream output(json_path, std::ios::binary | std::ios::trunc);
      if (!output) {
        std::cerr << "error: cannot create manifest: " << json_path << '\n';
        return 1;
      }
      const auto status = gem16::WriteManifestJson(manifest.value(), output);
      if (!status.ok()) {
        std::cerr << "error: " << status.message() << '\n';
        return 1;
      }
    }
  }
  return 0;
}

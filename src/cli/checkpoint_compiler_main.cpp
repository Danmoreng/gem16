#include "compiler/nvfp4_batch_encoder.h"

#include <iostream>
#include <string>

namespace {
void Usage() {
  std::cerr << "usage: gem16-checkpoint-compiler --nvfp4-job JOB.json "
               "--telemetry OUTPUT.json\n"
               "       gem16-checkpoint-compiler --build-info\n";
}
}  // namespace

int main(int argc, char** argv) {
  std::string job;
  std::string telemetry;
  bool build_info = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--build-info" && i == 1 && argc == 2) {
      build_info = true;
    } else if (arg == "--nvfp4-job" && i + 1 < argc && job.empty()) {
      job = argv[++i];
    } else if (arg == "--telemetry" && i + 1 < argc && telemetry.empty()) {
      telemetry = argv[++i];
    } else {
      Usage();
      return 2;
    }
  }
  if (build_info) {
    std::cout << gem16::compiler::Nvfp4BuildInfoJson() << '\n';
    return 0;
  }
  if (job.empty() || telemetry.empty() || argc != 5) {
    Usage();
    return 2;
  }
  const auto status = gem16::compiler::EncodeNvfp4JobFile(job, telemetry);
  if (!status.ok()) {
    std::cerr << "native NVFP4 compilation failed: " << status.message() << '\n';
    return gem16::compiler::ExitCodeForNvfp4Status(status.code());
  }
  return 0;
}

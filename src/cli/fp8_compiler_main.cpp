#include "compiler/fp8_batch_encoder.h"

#include <iostream>
#include <string>

namespace {

void PrintUsage() {
  std::cerr << "usage: gem16-fp8-compiler --job JOB.json --payload OUTPUT.bin "
                "--telemetry OUTPUT.json\n"
                "   or: gem16-fp8-compiler --compare-job JOB.json --metrics OUTPUT.json\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string job;
  std::string payload;
  std::string telemetry;
  std::string compare_job;
  std::string metrics;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto require_value = [&](std::string* destination) -> bool {
      if (index + 1 >= argc || argv[index + 1][0] == '\0') return false;
      *destination = argv[++index];
      return true;
    };
    if (argument == "--job") {
      if (!require_value(&job)) {
        PrintUsage();
        return 2;
      }
    } else if (argument == "--compare-job") {
      if (!require_value(&compare_job)) {
        PrintUsage();
        return 2;
      }
    } else if (argument == "--metrics") {
      if (!require_value(&metrics)) {
        PrintUsage();
        return 2;
      }
    } else if (argument == "--payload") {
      if (!require_value(&payload)) {
        PrintUsage();
        return 2;
      }
    } else if (argument == "--telemetry") {
      if (!require_value(&telemetry)) {
        PrintUsage();
        return 2;
      }
    } else {
      std::cerr << "unknown argument: " << argument << '\n';
      PrintUsage();
      return 2;
    }
  }
  const bool encode_mode = !job.empty() || !payload.empty() || !telemetry.empty();
  const bool compare_mode = !compare_job.empty() || !metrics.empty();
  if (encode_mode == compare_mode || (encode_mode &&
      (job.empty() || payload.empty() || telemetry.empty())) ||
      (compare_mode && (compare_job.empty() || metrics.empty()))) {
    PrintUsage();
    return 2;
  }
  const gem16::Status status = encode_mode
      ? gem16::compiler::EncodeJobFile(job, payload, telemetry)
      : gem16::compiler::CompareJobFile(compare_job, metrics);
  if (!status.ok()) {
    std::cerr << (encode_mode ? "native FP8 compilation failed: " :
                                "native FP8 comparison failed: ")
              << status.message() << '\n';
    return gem16::compiler::ExitCodeForStatus(status.code());
  }
  return 0;
}

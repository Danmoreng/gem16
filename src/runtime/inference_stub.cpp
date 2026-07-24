#include "gem16gb/engine.h"

#include <ostream>

namespace gem16gb {

Result<GreedyInferenceResult> RunGreedyInference(const GreedyInferenceOptions&) {
  return Status(StatusCode::kUnsupported,
                "greedy inference requires a CUDA build compiled for SM120a");
}

Status WriteGreedyInferenceJson(const GreedyInferenceResult&, std::ostream&) {
  return Status(StatusCode::kUnsupported,
                "greedy inference JSON requires a CUDA inference result");
}

Result<DecodeBenchmarkResult> RunDecodeBenchmark(const DecodeBenchmarkOptions&) {
  return Status(StatusCode::kUnsupported,
                "decode benchmarking requires a CUDA build compiled for SM120a");
}

Status WriteDecodeBenchmarkJson(const DecodeBenchmarkResult&, std::ostream&) {
  return Status(StatusCode::kUnsupported,
                "decode benchmark JSON requires a CUDA benchmark result");
}

}  // namespace gem16gb

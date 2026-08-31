#include <filesystem>
#include <iostream>

#include "model/gemma4_26b_vision_module.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: gem16-26b-vision-module-probe MODULE\n";
    return 2;
  }
  auto plan = gem16::internal::LoadGemma4Moe26BVisionModulePlan(
      std::filesystem::path(argv[1]));
  if (!plan.ok()) {
    std::cerr << "vision_module_probe_error: " << plan.status().message()
              << '\n';
    return 3;
  }
  std::cout << "vision_module_probe_ok profile="
            << gem16::internal::kGemma4Moe26BVisionProfile
            << " tensors=" << plan.value().tensors.size()
            << " payload_bytes="
            << gem16::internal::kGemma4Moe26BVisionPayloadBytes
            << " sha256=" << plan.value().artifact_sha256 << '\n';
  return 0;
}

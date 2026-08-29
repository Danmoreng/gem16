#include "model_catalog.h"

#include "model_catalog.generated.h"

#include <array>

namespace gem16::studio {
namespace {

constexpr std::array kProfiles{
    ModelProfileCatalog{
        ModelProfile::kGemma4Unified12B,
        "Unified multimodal checkpoint for approximately 16 GB Blackwell GPUs.",
        "Text · Vision · Audio · MTP",
        &generated::kGemma4Unified12BTarget,
        &generated::kGemma4Unified12BAssistant},
    ModelProfileCatalog{
        ModelProfile::kGemma4Moe26BA4B,
        "Text-only A4B mixture-of-experts checkpoint for approximately 16 GB Blackwell GPUs.",
        "Text · Fixed MTP D2 · 86,016 tokens",
        &generated::kGemma4Moe26BA4BTarget,
        &generated::kGemma4Moe26BA4BAssistant},
};

}  // namespace

const ModelProfileCatalog& CatalogForProfile(ModelProfile profile) {
  return profile == ModelProfile::kGemma4Moe26BA4B ? kProfiles[1] : kProfiles[0];
}

std::span<const ModelProfileCatalog> ModelCatalog() { return kProfiles; }

std::filesystem::path RepositoryDirectory(
    const char* repository_name, const std::filesystem::path& hub_root) {
  std::string repository = repository_name;
  const auto separator = repository.find('/');
  if (separator != std::string::npos) repository.replace(separator, 1, "--");
  return hub_root / ("models--" + repository);
}

std::filesystem::path ComponentDirectory(
    const ModelComponentCatalog& component,
    const std::filesystem::path& hub_root) {
  if (component.composed_view) {
    std::string repository = component.repository;
    const auto separator = repository.find('/');
    if (separator != std::string::npos) repository.replace(separator, 1, "--");
    return hub_root / ".gem16/snapshots" /
           (repository + "--" + component.revision);
  }
  return RepositoryDirectory(component.repository, hub_root) / "snapshots" /
         component.revision;
}

std::filesystem::path VerificationMarkerPath(
    const ModelCatalogFile& file, const std::filesystem::path& hub_root) {
  return RepositoryDirectory(file.source_repository, hub_root) /
         ".gem16-verified" / (std::string(file.blob_id) + ".sha256");
}

}  // namespace gem16::studio

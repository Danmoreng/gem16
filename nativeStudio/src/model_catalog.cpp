#include "model_catalog.h"

#include "model_catalog.generated.h"

#include <array>

namespace gem16::studio {
namespace {

constexpr std::array kUnified12BComponents{
    ModelProfileComponent{ModelComponentKind::kTarget,
                          &generated::kGemma4Unified12BTarget, true},
    ModelProfileComponent{ModelComponentKind::kAssistant,
                          &generated::kGemma4Unified12BAssistant, true},
};
constexpr std::array kMoe26BA4BComponents{
    ModelProfileComponent{ModelComponentKind::kTarget,
                          &generated::kGemma4Moe26BA4BTarget, true},
    ModelProfileComponent{ModelComponentKind::kAssistant,
                          &generated::kGemma4Moe26BA4BAssistant, true},
};
constexpr std::array kMoe26BVisionComponents{
    ModelProfileComponent{ModelComponentKind::kTarget,
                          &generated::kGemma4Moe26BTrellis35Target, true},
    ModelProfileComponent{ModelComponentKind::kVision,
                          &generated::kGemma4Moe26BVisionFp8, true},
    ModelProfileComponent{ModelComponentKind::kAssistant,
                          &generated::kGemma4Moe26BA4BAssistant, true},
};

constexpr std::array kProfiles{
    ModelProfileCatalog{
        ModelProfile::kGemma4Unified12B,
        "Unified multimodal checkpoint for approximately 16 GB Blackwell GPUs.",
        "Text · Vision · Audio · MTP", kUnified12BComponents},
    ModelProfileCatalog{
        ModelProfile::kGemma4Moe26BA4B,
        "Text-only A4B mixture-of-experts checkpoint for approximately 16 GB Blackwell GPUs.",
        "Text · Fixed MTP D2 · 86,016 tokens", kMoe26BA4BComponents},
    ModelProfileCatalog{
        ModelProfile::kGemma4Moe26BTrellis35VisionFp8,
        "Trellis35 text checkpoint with the pinned FP8 Vision module for approximately 16 GB Blackwell GPUs.",
        "Text · Vision · Fixed MTP D2 · 229,120 tokens",
        kMoe26BVisionComponents},
};

}  // namespace

const ModelProfileCatalog& CatalogForProfile(ModelProfile profile) {
  return kProfiles[ModelProfileIndex(profile)];
}

std::span<const ModelProfileCatalog> ModelCatalog() { return kProfiles; }

const ModelProfileComponent* ComponentForProfile(
    const ModelProfileCatalog& profile, ModelComponentKind kind) {
  for (const auto& component : profile.components) {
    if (component.kind == kind) return &component;
  }
  return nullptr;
}

const char* ComponentKindLabel(ModelComponentKind kind) {
  switch (kind) {
    case ModelComponentKind::kTarget:
      return "Target";
    case ModelComponentKind::kAssistant:
      return "Assistant";
    case ModelComponentKind::kVision:
      return "Vision";
  }
  return "Unknown";
}

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
    std::string view = repository + "--" + component.revision;
    if (component.composed_view_suffix[0] != '\0') {
      view += "--";
      view += component.composed_view_suffix;
    }
    return hub_root / ".gem16/snapshots" / view;
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

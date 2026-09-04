#pragma once

#include "types.h"

#include <cstdint>
#include <filesystem>
#include <span>

namespace gem16::studio {

struct ModelCatalogFile {
  const char* path;
  std::uint64_t size;
  const char* sha256;
  const char* blob_id;
  const char* source_repository;
  const char* source_revision;
  const char* source_path;
};

struct ModelComponentCatalog {
  const char* id;
  const char* label;
  const char* repository;
  const char* revision;
  std::span<const ModelCatalogFile> files;
  bool composed_view;
  const char* composed_view_suffix;
};

enum class ModelComponentKind { kTarget, kAssistant, kVision };
inline constexpr std::size_t kModelComponentKindCount = 3U;

[[nodiscard]] constexpr std::size_t ModelComponentKindIndex(
    ModelComponentKind kind) {
  switch (kind) {
    case ModelComponentKind::kTarget:
      return 0U;
    case ModelComponentKind::kAssistant:
      return 1U;
    case ModelComponentKind::kVision:
      return 2U;
  }
  return 0U;
}

struct ModelProfileComponent {
  ModelComponentKind kind;
  const ModelComponentCatalog* catalog;
  bool required;
};

struct ModelProfileCatalog {
  ModelProfile profile;
  const char* description;
  const char* capabilities;
  std::span<const ModelProfileComponent> components;
};

[[nodiscard]] const ModelProfileCatalog& CatalogForProfile(ModelProfile profile);
[[nodiscard]] std::span<const ModelProfileCatalog> ModelCatalog();
[[nodiscard]] std::span<const ModelProfile> PublicModelProfiles();
[[nodiscard]] const ModelProfileComponent* ComponentForProfile(
    const ModelProfileCatalog& profile, ModelComponentKind kind);
[[nodiscard]] const char* ComponentKindLabel(ModelComponentKind kind);
[[nodiscard]] std::filesystem::path ComponentDirectory(
    const ModelComponentCatalog& component,
    const std::filesystem::path& hub_root);
[[nodiscard]] std::filesystem::path RepositoryDirectory(
    const char* repository, const std::filesystem::path& hub_root);
[[nodiscard]] std::filesystem::path VerificationMarkerPath(
    const ModelCatalogFile& file, const std::filesystem::path& hub_root);

}  // namespace gem16::studio

#include "test.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include "model/gemma4_26b_trellis35.h"

namespace {

constexpr std::uint64_t Align(std::uint64_t value) {
  return (value + gem16::internal::kTrellis35Alignment - 1U) &
         ~(gem16::internal::kTrellis35Alignment - 1U);
}

void AddRegion(std::uint64_t bytes, gem16::internal::Trellis35Region* region,
               std::uint64_t* cursor) {
  *cursor = Align(*cursor);
  *region = {*cursor, bytes};
  *cursor += bytes;
}

void PlanFamily(bool gate_up, gem16::internal::Trellis35FamilyPlan* family,
                std::uint64_t* cursor) {
  const std::uint64_t rows = gate_up ? 2816U : 768U;
  const std::uint64_t columns = gate_up ? 1408U : 2816U;
  const std::uint64_t suh = gate_up ? 2816U : 768U;
  const std::uint64_t svh = gate_up ? 1408U : 2816U;
  const std::uint64_t coefficients = rows * columns;
  AddRegion(coefficients * 3U / 8U * 64U, &family->k3_payload_pool,
            cursor);
  AddRegion(coefficients * 4U / 8U * 64U, &family->k4_payload_pool,
            cursor);
  AddRegion(128U * 8U, &family->descriptor, cursor);
  AddRegion(128U * suh * 2U, &family->suh, cursor);
  AddRegion(128U * svh * 2U, &family->svh, cursor);
  for (std::size_t expert = 0; expert < family->rate_map.size(); ++expert) {
    family->rate_map[expert] = expert < 64U ? 3U : 4U;
  }
}

void WriteU16(std::array<char, 8>* bytes, std::size_t offset,
              std::uint16_t value) {
  (*bytes)[offset] = static_cast<char>(value & 0xffU);
  (*bytes)[offset + 1U] = static_cast<char>(value >> 8U);
}

void WriteU32(std::array<char, 8>* bytes, std::uint32_t value) {
  WriteU16(bytes, 0U, static_cast<std::uint16_t>(value & 0xffffU));
  WriteU16(bytes, 2U, static_cast<std::uint16_t>(value >> 16U));
}

void WriteDescriptors(std::fstream* output, std::uint64_t rows,
                      std::uint64_t columns,
                      const gem16::internal::Trellis35FamilyPlan& family) {
  std::array<std::uint64_t, 5> next{};
  for (std::size_t expert = 0; expert < family.rate_map.size(); ++expert) {
    const std::uint16_t rate = family.rate_map[expert];
    std::array<char, 8> encoded{};
    WriteU32(&encoded, static_cast<std::uint32_t>(next[rate]));
    WriteU16(&encoded, 4U, rate);
    WriteU16(&encoded, 6U, gem16::internal::kTrellis35CodebookId);
    output->seekp(static_cast<std::streamoff>(family.descriptor.offset +
                                             expert * encoded.size()));
    output->write(encoded.data(), encoded.size());
    next[rate] += rows * columns * rate / 8U;
  }
}

}  // namespace

void RunGemma426BTrellis35Tests() {
  using gem16::internal::Gemma4Moe26BTrellis35EngineDispatchStatus;
  using gem16::internal::Trellis35LayerPlan;
  using gem16::internal::ValidateGemma4Moe26BTrellis35LayerPayload;

  const auto dispatch = Gemma4Moe26BTrellis35EngineDispatchStatus();
  GEM16_CHECK(dispatch.ok());

  const auto root = std::filesystem::temp_directory_path() /
                    "gem16-trellis35-layer-payload-test";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  const auto artifact = root / "layer.bin";
  {
    std::ofstream output(artifact, std::ios::binary);
    output.seekp(static_cast<std::streamoff>(
        gem16::internal::kTrellis35LayerArtifactBytes - 1U));
    output.put('\0');
  }
  Trellis35LayerPlan plan;
  plan.artifact.path = artifact;
  plan.artifact.bytes = gem16::internal::kTrellis35LayerArtifactBytes;
  std::uint64_t cursor = 0U;
  PlanFamily(true, &plan.gate_up, &cursor);
  PlanFamily(false, &plan.down, &cursor);
  GEM16_CHECK(Align(cursor) ==
              gem16::internal::kTrellis35LayerArtifactBytes);
  {
    std::fstream output(artifact,
                        std::ios::binary | std::ios::in | std::ios::out);
    WriteDescriptors(&output, 2816U, 1408U, plan.gate_up);
    WriteDescriptors(&output, 768U, 2816U, plan.down);
  }
  GEM16_CHECK(ValidateGemma4Moe26BTrellis35LayerPayload(&plan).ok());

  {
    std::fstream output(artifact,
                        std::ios::binary | std::ios::in | std::ios::out);
    output.seekp(static_cast<std::streamoff>(plan.gate_up.descriptor.offset +
                                             6U));
    output.put('\0');
    output.put('\0');
  }
  GEM16_CHECK(!ValidateGemma4Moe26BTrellis35LayerPayload(&plan).ok());
  {
    std::fstream output(artifact,
                        std::ios::binary | std::ios::in | std::ios::out);
    output.seekp(static_cast<std::streamoff>(plan.gate_up.descriptor.offset +
                                             6U));
    output.put('\2');
    output.put('\0');
    output.seekp(static_cast<std::streamoff>(plan.down.suh.offset));
    output.put('\0');
    output.put(static_cast<char>(0x7c));
  }
  GEM16_CHECK(!ValidateGemma4Moe26BTrellis35LayerPayload(&plan).ok());
  std::filesystem::remove_all(root);
}

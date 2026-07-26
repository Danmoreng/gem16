#include "test.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "model/safetensors.h"

namespace {

void WriteU64(std::ofstream& output, std::uint64_t value) {
  std::array<char, 8> bytes{};
  for (unsigned index = 0; index < bytes.size(); ++index) {
    bytes[index] = static_cast<char>((value >> (index * 8U)) & 0xFFU);
  }
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void WriteSafetensors(const std::filesystem::path& path, std::string header, std::size_t payload_bytes) {
  while (header.size() % 8U != 0) header.push_back(' ');
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  WriteU64(output, header.size());
  output.write(header.data(), static_cast<std::streamsize>(header.size()));
  std::vector<char> payload(payload_bytes, '\0');
  output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
}

}  // namespace

void RunSafetensorsTests() {
  const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto root = std::filesystem::temp_directory_path() / ("gem16-safetensors-test-" + std::to_string(unique));
  std::error_code error;
  std::filesystem::remove_all(root, error);
  std::filesystem::create_directories(root);

  WriteSafetensors(root / "model.safetensors",
                   R"({"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"BF16","shape":[2],"data_offsets":[4,8]}})", 8);
  auto valid = gem16::internal::LoadSafetensorsDirectory(root);
  GEM16_CHECK(valid.ok());
  if (valid.ok()) {
    GEM16_CHECK(valid.value().size() == 2);
    GEM16_CHECK(valid.value()[0].name == "a");
    GEM16_CHECK(valid.value()[1].length == 4);
  }

  std::u8string unicode_name = u8"checkpoint-";
  unicode_name.push_back(static_cast<char8_t>(0xC3));
  unicode_name.push_back(static_cast<char8_t>(0xA4));
  const auto unicode_root = root / std::filesystem::path(unicode_name);
  std::filesystem::create_directories(unicode_root);
  WriteSafetensors(unicode_root / "model.safetensors", R"({"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}})", 4);
  GEM16_CHECK(gem16::internal::LoadSafetensorsDirectory(unicode_root).ok());

  WriteSafetensors(root / "model.safetensors",
                   R"({"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[4],"data_offsets":[2,6]}})", 6);
  GEM16_CHECK(!gem16::internal::LoadSafetensorsDirectory(root).ok());

  std::filesystem::remove(root / "model.safetensors");
  {
    std::ofstream index(root / "model.safetensors.index.json");
    index << R"({"weight_map":{"a":"../escape.safetensors"}})";
  }
  GEM16_CHECK(!gem16::internal::LoadSafetensorsDirectory(root).ok());

  const auto external = root.parent_path() / (root.filename().string() + "-external.safetensors");
  WriteSafetensors(external, R"({"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}})", 4);
  {
    std::ofstream index(root / "model.safetensors.index.json", std::ios::trunc);
    index << R"({"weight_map":{"a":"linked.safetensors"}})";
  }
  std::filesystem::create_symlink(external, root / "linked.safetensors", error);
  if (!error) {
    GEM16_CHECK(!gem16::internal::LoadSafetensorsDirectory(root).ok());
  } else {
#if defined(_WIN32)
    constexpr int kWindowsPrivilegeNotHeld = 1314;
    GEM16_CHECK(error == std::errc::permission_denied || error == std::errc::operation_not_permitted ||
                  error.value() == kWindowsPrivilegeNotHeld);
#else
    GEM16_CHECK(!error);
#endif
  }

  std::filesystem::remove_all(root, error);
  GEM16_CHECK(!error);
  std::filesystem::remove(external, error);
  GEM16_CHECK(!error);
}

#include "../cuda/decode_replay_support.h"
#include "test.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

namespace {
void TestReplayInputValidation() {
  const auto root = std::filesystem::temp_directory_path() /
      ("gem16-decode-replay-" + std::to_string(
          decode_replay::Clock::now().time_since_epoch().count()));
  GEM16_CHECK(std::filesystem::create_directory(root));
  const auto input = root / "tokens.ids";
  auto fails = [](const auto& path, std::size_t count) {
    try { (void)decode_replay::ReadTokens(path, count); }
    catch (const std::exception&) { return true; }
    return false;
  };
  {
    std::ofstream file(input); file << " 0, 262143\n";
  }
  const auto tokens = decode_replay::ReadTokens(input, 2U);
  GEM16_CHECK(tokens.size() == 2U && tokens[0] == 0U && tokens[1] == 262143U);
  for (const std::string_view text : {
           "", "0,", "0,1,\n", "0,-1", "0,262144", "0,4294967296",
           "0,NaN", "0,1x", "0,1,2", "0"}) {
    { std::ofstream file(input, std::ios::trunc); file << text; }
    GEM16_CHECK(fails(input, 2U));
  }
  {
    std::ofstream file(input, std::ios::trunc);
    file << std::string(2U * 1024U * 1024U + 1U, '0');
  }
  GEM16_CHECK(fails(input, 2U));
  GEM16_CHECK(fails(root, 2U));
  GEM16_CHECK(fails(root / "missing", 2U));
  const auto link = root / "link.ids";
  std::error_code error;
  std::filesystem::create_symlink(input, link, error);
  if (!error) GEM16_CHECK(fails(link, 2U));
  std::filesystem::remove_all(root);
}
}  // namespace

void RunDecodeReplayTests() { TestReplayInputValidation(); }

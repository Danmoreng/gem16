#include "test.h"

#include <string>

#include "util/json.h"

void RunJsonTests() {
  auto parsed = gem16::json::Parse(R"({"name":"gem16","shape":[2,16],"enabled":true,"unicode":"\uD83D\uDE80"})");
  GEM16_CHECK(parsed.ok());
  if (parsed.ok()) {
    GEM16_CHECK(parsed.value().is_object());
    GEM16_CHECK(parsed.value().find("name") != nullptr);
    GEM16_CHECK(parsed.value().find("name")->as_string() == "gem16");
    GEM16_CHECK(parsed.value().find("shape")->as_array()[1].as_integer() == 16);
    GEM16_CHECK(parsed.value().find("unicode")->as_string() == "\xF0\x9F\x9A\x80");
  }

  GEM16_CHECK(!gem16::json::Parse(R"({"duplicate":1,"duplicate":2})").ok());
  const std::string invalid_surrogate{'"', '\\', 'u', 'D', '8', '0', '0', '"'};
  GEM16_CHECK(!gem16::json::Parse(invalid_surrogate).ok());
  GEM16_CHECK(!gem16::json::Parse("[01]").ok());
  GEM16_CHECK(!gem16::json::Parse("{} trailing").ok());
  GEM16_CHECK(!gem16::json::Parse(std::string{"\"\xC0\x80\"", 4}).ok());
  GEM16_CHECK(gem16::json::Escape("a\n\"b") == "a\\n\\\"b");
  if (parsed.ok()) {
    const std::string serialized = gem16::json::Stringify(parsed.value());
    auto round_trip = gem16::json::Parse(serialized);
    GEM16_CHECK(round_trip.ok());
    GEM16_CHECK(serialized.find("\"enabled\":true") != std::string::npos);
  }

  auto large_number = gem16::json::Parse("1000000000000000019884624838656");
  GEM16_CHECK(large_number.ok());
  if (large_number.ok()) {
    GEM16_CHECK(large_number.value().is_number());
    GEM16_CHECK(!large_number.value().is_integer());
    GEM16_CHECK(large_number.value().as_number() > 1.0e29);
  }
}

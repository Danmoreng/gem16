#pragma once

#include <iostream>
#include <string_view>

namespace gem16::test {

inline int failures = 0;

inline void Check(bool condition, std::string_view expression, std::string_view file, int line) {
  if (!condition) {
    std::cerr << file << ':' << line << ": check failed: " << expression << '\n';
    ++failures;
  }
}

}  // namespace gem16::test

#define GEM16_CHECK(expression) ::gem16::test::Check(static_cast<bool>(expression), #expression, __FILE__, __LINE__)


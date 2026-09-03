#include "test.h"

#include <cstdlib>
#include <string_view>

#include "util/environment.h"

void RunEnvironmentTests() {
  constexpr const char* kName = "GEM16_HOST_ENVIRONMENT_TEST_VALUE";
#if defined(_WIN32)
  GEM16_CHECK(_putenv_s(kName, "present") == 0);
#else
  GEM16_CHECK(setenv(kName, "present", 1) == 0);
#endif
  const char* value = gem16::internal::GetEnvironmentVariable(kName);
  GEM16_CHECK(value != nullptr);
  if (value != nullptr) {
    GEM16_CHECK(std::string_view(value) == "present");
  }
#if defined(_WIN32)
  GEM16_CHECK(_putenv_s(kName, "") == 0);
#else
  GEM16_CHECK(unsetenv(kName) == 0);
#endif
  GEM16_CHECK(gem16::internal::GetEnvironmentVariable(kName) == nullptr);
}

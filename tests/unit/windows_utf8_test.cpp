#include "cli/windows_utf8.h"

#include <string>

#include "test.h"

void RunWindowsUtf8Tests() {
  const auto converted =
      gem16gb::cli::WideToUtf8(L"Bitte erz\u00E4hl mir einen Witz \U0001F600");
  GEM16GB_CHECK(converted.has_value());
  if (converted.has_value()) {
    GEM16GB_CHECK(*converted ==
                  "Bitte erz\xC3\xA4" "hl mir einen Witz \xF0\x9F\x98\x80");
  }

  const wchar_t invalid_surrogate[] = {static_cast<wchar_t>(0xD800), L'\0'};
  GEM16GB_CHECK(
      !gem16gb::cli::WideToUtf8(invalid_surrogate).has_value());
}

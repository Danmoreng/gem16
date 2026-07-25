#include <iostream>

#include "test.h"

void RunJsonTests();
void RunLayerTests();
void RunFp8Tests();
void RunMemoryPlanTests();
void RunNvfp4Tests();
void RunSafetensorsTests();
void RunSm120LayoutTests();
#if defined(_WIN32)
void RunWindowsUtf8Tests();
#endif

int main() {
  RunJsonTests();
  RunLayerTests();
  RunFp8Tests();
  RunMemoryPlanTests();
  RunNvfp4Tests();
  RunSafetensorsTests();
  RunSm120LayoutTests();
#if defined(_WIN32)
  RunWindowsUtf8Tests();
#endif
  if (gem16gb::test::failures != 0) {
    std::cerr << gem16gb::test::failures << " test assertion(s) failed\n";
    return 1;
  }
  std::cout << "all host unit tests passed\n";
  return 0;
}


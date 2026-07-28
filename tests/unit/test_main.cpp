#include <iostream>

#include "test.h"

void RunConfigTests();
void RunJsonTests();
void RunLayerTests();
void RunFp8Tests();
void RunMemoryPlanTests();
void RunMtpSchedulerTests();
void RunNvfp4Tests();
void RunSafetensorsTests();
void RunSamplingTests();
void RunSm120LayoutTests();
#if defined(_WIN32)
void RunWindowsUtf8Tests();
#endif

int main() {
  RunConfigTests();
  RunJsonTests();
  RunLayerTests();
  RunFp8Tests();
  RunMemoryPlanTests();
  RunMtpSchedulerTests();
  RunNvfp4Tests();
  RunSafetensorsTests();
  RunSamplingTests();
  RunSm120LayoutTests();
#if defined(_WIN32)
  RunWindowsUtf8Tests();
#endif
  if (gem16::test::failures != 0) {
    std::cerr << gem16::test::failures << " test assertion(s) failed\n";
    return 1;
  }
  std::cout << "all host unit tests passed\n";
  return 0;
}


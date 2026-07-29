#include <iostream>

#include "test.h"

void RunChatTests();
void RunAudioTests();
void RunConfigTests();
void RunJsonTests();
void RunLayerTests();
void RunFp8Tests();
void RunImageTests();
void RunMemoryPlanTests();
void RunMtpSchedulerTests();
void RunNvfp4Tests();
void RunOpenAiChatTests();
void RunSafetensorsTests();
void RunSamplingTests();
void RunSm120LayoutTests();
#if defined(_WIN32)
void RunWindowsUtf8Tests();
#endif

int main() {
  RunAudioTests();
  RunChatTests();
  RunConfigTests();
  RunJsonTests();
  RunLayerTests();
  RunFp8Tests();
  RunImageTests();
  RunMemoryPlanTests();
  RunMtpSchedulerTests();
  RunNvfp4Tests();
  RunOpenAiChatTests();
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

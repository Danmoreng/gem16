#include <iostream>

#include "test.h"

void RunChatTests();
void RunAudioTests();
void RunConfigTests();
void RunJsonTests();
void RunLayerTests();
void RunFp8Tests();
void RunFp8BatchEncoderTests();
void RunGemma426BManifestTests();
void RunGemma426BTrellis35Tests();
void RunGemma426BMoePrefillPlanTests();
void RunImageTests();
void RunMemoryPlanTests();
void RunMtpSchedulerTests();
void RunNvfp4Tests();
void RunNvfp4BatchEncoderTests();
void RunNvfp4HeadTests();
void RunOpenAiChatTests();
void RunObservabilityTests();
void RunRequestQueueTests();
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
  RunFp8BatchEncoderTests();
  RunGemma426BManifestTests();
  RunGemma426BTrellis35Tests();
  RunGemma426BMoePrefillPlanTests();
  RunImageTests();
  RunMemoryPlanTests();
  RunMtpSchedulerTests();
  RunNvfp4Tests();
  RunNvfp4BatchEncoderTests();
  RunNvfp4HeadTests();
  RunOpenAiChatTests();
  RunObservabilityTests();
  RunRequestQueueTests();
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

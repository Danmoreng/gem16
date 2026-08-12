#pragma once

#include <cstdint>
#include <filesystem>

#include "gem16/status.h"

namespace gem16::compiler {

// Public scalar codec oracles used by exhaustive host tests and independent
// control-plane fixtures. They are the exact finite codecs used by the native
// encoder; callers must not use them as a runtime quantization fallback.
[[nodiscard]] float DecodeNvfp4E2M1(std::uint8_t code) noexcept;
[[nodiscard]] std::uint8_t EncodeNvfp4E2M1(float value) noexcept;
[[nodiscard]] float DecodeNvfp4E4M3(std::uint8_t code) noexcept;
[[nodiscard]] std::uint8_t EncodeNvfp4E4M3(float value) noexcept;

// Maps native compiler failures to the documented CLI exit categories.
[[nodiscard]] int ExitCodeForNvfp4Status(StatusCode code) noexcept;
[[nodiscard]] bool Nvfp4BuildSupportsFullJob() noexcept;

// Emits the exact native protocol/build identity without opening a job or
// touching model/output files.
[[nodiscard]] const char* Nvfp4BuildInfoJson() noexcept;

// Direct bounded BF16 -> NVFP4 expert compiler. The job owns all source and
// destination ranges; this function never creates or resizes a shard.
[[nodiscard]] Status EncodeNvfp4JobFile(const std::filesystem::path& job_path,
                                        const std::filesystem::path& telemetry_path);

}  // namespace gem16::compiler

#pragma once

#include <cstdint>
#include <filesystem>

#include "gem16/status.h"

namespace gem16::compiler {

// These helpers are the native M05 arithmetic oracle. They intentionally do
// not share the runtime activation quantizer: checkpoint conversion has a
// different BF16 row-scale and telemetry contract.
[[nodiscard]] float DecodeE4M3Fn(std::uint8_t bits) noexcept;
[[nodiscard]] Result<std::uint8_t> EncodeE4M3Fn(float value);
[[nodiscard]] Result<std::uint16_t> RoundBf16Rne(float value);

// Maps native compiler failures to the documented CLI exit categories. Kept
// in the support library so the mapping is directly unit-testable.
[[nodiscard]] int ExitCodeForStatus(StatusCode code) noexcept;

// Reads and validates one canonical M05 batch job, writes the bounded payload
// bundle and deterministic telemetry, and removes incomplete outputs on any
// failure. The job format is intentionally private to this offline tool; the
// Python orchestration layer is responsible for constructing it from a
// verified compiler plan.
[[nodiscard]] Status EncodeJobFile(const std::filesystem::path& job_path,
                                   const std::filesystem::path& payload_path,
                                   const std::filesystem::path& telemetry_path);

// Compares the four canonical FP8/BF16 ranges in a bounded native pass. The
// compare job is a control-plane protocol emitted from verified Safetensors
// descriptors; this function owns all element-wise work and reductions.
[[nodiscard]] Status CompareJobFile(const std::filesystem::path& job_path,
                                    const std::filesystem::path& metrics_path);

}  // namespace gem16::compiler

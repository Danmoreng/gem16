"""Deterministic BF16-to-FP8 E4M3FN row encoder for M05.

This module is an offline reference encoder.  It deliberately uses only the
Python standard library and keeps the source representation as BF16 bytes so
that the compiler can later replace the row reader without changing the
arithmetic contract.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import math
import struct
from typing import Final

from .common import DataError


FP8_CONTRACT_ID: Final[str] = "gem16.fp8_attention_rowwise"
FP8_CONTRACT_VERSION: Final[int] = 1
E4M3FN_MAX: Final[float] = 448.0
E4M3FN_NAN_CODE: Final[int] = 0x7F
BF16_ONE_BITS: Final[int] = 0x3F80
BF16_MIN_POSITIVE_BITS: Final[int] = 0x0001


def _f32(value: float) -> float:
    """Round a Python number to binary32 using the platform-independent wire form."""
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise DataError(f"value is outside finite binary32 range: {value!r}") from error


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _bits_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def decode_bf16(bits: int) -> float:
    """Decode one little-endian BF16 bit pattern as binary32."""
    if not isinstance(bits, int) or isinstance(bits, bool) or not 0 <= bits <= 0xFFFF:
        raise DataError(f"BF16 bits out of range: {bits!r}")
    return _bits_f32(bits << 16)


def round_binary32_to_bf16_rne(value: float) -> int:
    """Round a finite, non-negative binary32 value to BF16, ties to even."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DataError("BF16 rounding input must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise DataError("BF16 rounding input must be finite and non-negative")
    bits = _f32_bits(value)
    upper = bits >> 16
    lower = bits & 0xFFFF
    if lower > 0x8000 or (lower == 0x8000 and (upper & 1)):
        upper += 1
    # A finite binary32 input can round to BF16 infinity.  The helper exposes
    # that IEEE result; row-scale validation below rejects non-finite scales.
    return upper


def decode_e4m3fn(code: int) -> float:
    """Decode a finite E4M3FN code; NaN encodings are rejected."""
    if not isinstance(code, int) or isinstance(code, bool) or not 0 <= code <= 0xFF:
        raise DataError(f"E4M3FN code out of range: {code!r}")
    if (code & 0x7F) == E4M3FN_NAN_CODE:
        raise DataError(f"E4M3FN NaN code is not finite: 0x{code:02x}")
    magnitude_code = code & 0x7F
    exponent = (magnitude_code >> 3) & 0xF
    mantissa = magnitude_code & 0x7
    if exponent == 0:
        magnitude = math.ldexp(float(mantissa), -9)
    else:
        magnitude = math.ldexp(1.0 + mantissa / 8.0, exponent - 7)
    return -magnitude if code & 0x80 else magnitude


def _build_e4m3_tables() -> tuple[tuple[float, ...], tuple[float, ...]]:
    values = tuple(decode_e4m3fn(code) for code in range(0x7F))
    midpoints = tuple((values[index] + values[index + 1]) / 2.0 for index in range(0x7E))
    return values, midpoints


_E4M3_POSITIVE_VALUES, _E4M3_POSITIVE_MIDPOINTS = _build_e4m3_tables()


def encode_e4m3fn(value: float) -> int:
    """Encode finite binary32 to E4M3FN with RNE and finite saturation."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DataError("E4M3FN input must be numeric")
    value = _f32(float(value))
    if not math.isfinite(value):
        raise DataError("E4M3FN input must be finite")
    negative = math.copysign(1.0, value) < 0.0
    magnitude = abs(value)
    if magnitude >= E4M3FN_MAX:
        code = 0x7E
    else:
        index = bisect_left(_E4M3_POSITIVE_MIDPOINTS, magnitude)
        if index < len(_E4M3_POSITIVE_MIDPOINTS) and magnitude == _E4M3_POSITIVE_MIDPOINTS[index]:
            # At an exact midpoint, the code with an even least significant
            # bit is the selected representable value.
            code = index if (index & 1) == 0 else index + 1
        else:
            code = index
    return code | (0x80 if negative else 0)


@dataclass(frozen=True)
class FP8RowMetrics:
    """Deterministic row reconstruction contributions for compiler telemetry."""

    element_count: int
    source_min: float
    source_max: float
    source_sum_squares: float
    reconstruction_sum_squares: float
    source_reconstruction_dot: float
    error_sum_squares: float
    max_absolute_error: float


@dataclass(frozen=True)
class FP8RowResult:
    weight_bytes: bytes
    scale_bytes: bytes
    scale_bits: int
    scale_value: float
    zero_row: bool
    scale_underflow_clamped: bool
    saturation_count: int
    histogram: tuple[int, ...]
    metrics: FP8RowMetrics


def _decode_bf16_row(source: bytes | bytearray | memoryview) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if not isinstance(source, (bytes, bytearray, memoryview)):
        raise DataError("FP8 source row must be a bytes-like object")
    raw = bytes(source)
    if not raw:
        raise DataError("FP8 source row must be non-empty")
    if len(raw) & 1:
        raise DataError("BF16 source row must have an even byte count")
    bits = tuple(
        struct.unpack_from("<H", raw, offset)[0]
        for offset in range(0, len(raw), 2)
    )
    values = []
    for bit_pattern in bits:
        value = decode_bf16(bit_pattern)
        if not math.isfinite(value):
            raise DataError("BF16 source row must contain only finite values")
        values.append(value)
    return bits, tuple(values)


def quantize_bf16_row(source: bytes | bytearray | memoryview) -> FP8RowResult:
    """Quantize one little-endian BF16 [K] row to FP8 plus one BF16 scale."""
    source_bits, values = _decode_bf16_row(source)
    maximum = max(abs(value) for value in values)
    zero_row = maximum == 0.0
    scale_underflow_clamped = False
    if zero_row:
        scale_bits = BF16_ONE_BITS
    else:
        raw_scale = _f32(maximum / E4M3FN_MAX)
        scale_bits = round_binary32_to_bf16_rne(raw_scale)
        if scale_bits == 0:
            scale_bits = BF16_MIN_POSITIVE_BITS
            scale_underflow_clamped = True
    scale_value = decode_bf16(scale_bits)
    if not math.isfinite(scale_value) or scale_value <= 0.0:
        raise DataError("FP8 row scale must be finite and positive")

    # Conversion is row-local and keyed by source BF16 bits.  The bounded
    # encoder materializes only this row's decoded values, output, and cache;
    # repeated BF16 values avoid repeated codec work.
    cache: dict[int, tuple[int, bool, float]] = {}
    output = bytearray(len(source_bits))
    histogram = [0] * 256
    saturation_count = 0
    source_min = values[0]
    source_max = values[0]
    source_sum_squares = 0.0
    reconstruction_sum_squares = 0.0
    source_reconstruction_dot = 0.0
    error_sum_squares = 0.0
    max_absolute_error = 0.0
    for index, bit_pattern in enumerate(source_bits):
        source_value = values[index]
        cached = cache.get(bit_pattern)
        if cached is None:
            normalized = _f32(source_value / scale_value)
            if not math.isfinite(normalized):
                raise DataError("FP8 row normalization produced a non-finite value")
            magnitude = abs(normalized)
            saturated = magnitude > E4M3FN_MAX
            code = encode_e4m3fn(normalized)
            # Keep reconstruction at the specified binary32 boundary before
            # contributing it to the binary64 telemetry accumulators.
            reconstruction = _f32(decode_e4m3fn(code) * scale_value)
            cached = (code, saturated, reconstruction)
            cache[bit_pattern] = cached
        code, saturated, reconstruction = cached
        output[index] = code
        histogram[code] += 1
        saturation_count += int(saturated)

        source_min = min(source_min, source_value)
        source_max = max(source_max, source_value)
        source_sum_squares += source_value * source_value
        reconstruction_sum_squares += reconstruction * reconstruction
        source_reconstruction_dot += source_value * reconstruction
        error = source_value - reconstruction
        error_sum_squares += error * error
        max_absolute_error = max(max_absolute_error, abs(error))

    return FP8RowResult(
        weight_bytes=bytes(output),
        scale_bytes=struct.pack("<H", scale_bits),
        scale_bits=scale_bits,
        scale_value=scale_value,
        zero_row=zero_row,
        scale_underflow_clamped=scale_underflow_clamped,
        saturation_count=saturation_count,
        histogram=tuple(histogram),
        metrics=FP8RowMetrics(
            element_count=len(source_bits),
            source_min=source_min,
            source_max=source_max,
            source_sum_squares=source_sum_squares,
            reconstruction_sum_squares=reconstruction_sum_squares,
            source_reconstruction_dot=source_reconstruction_dot,
            error_sum_squares=error_sum_squares,
            max_absolute_error=max_absolute_error,
        ),
    )


# Short aliases make the reference routine convenient for compiler adapters.
decode_bf16_bits = decode_bf16
round_bf16_rne = round_binary32_to_bf16_rne
encode_e4m3fn_rne = encode_e4m3fn
quantize_row = quantize_bf16_row

__all__ = [
    "BF16_MIN_POSITIVE_BITS",
    "BF16_ONE_BITS",
    "E4M3FN_MAX",
    "E4M3FN_NAN_CODE",
    "FP8_CONTRACT_ID",
    "FP8_CONTRACT_VERSION",
    "FP8RowMetrics",
    "FP8RowResult",
    "decode_bf16",
    "decode_bf16_bits",
    "decode_e4m3fn",
    "encode_e4m3fn",
    "encode_e4m3fn_rne",
    "quantize_bf16_row",
    "quantize_row",
    "round_bf16_rne",
    "round_binary32_to_bf16_rne",
]

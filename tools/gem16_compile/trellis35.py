"""Hessian-aware offline compiler primitives for GEM16-Trellis35.

This module deliberately remains independent of the qualified NVFP4 compiler.
It consumes floating-point source weights and explicit calibration captures;
it never derives Trellis35 weights from an existing quantized artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import struct
from typing import Sequence

import numpy as np

from .common import InvalidPlanError
from .reader import TensorDescriptor
from .trellis35_quant import (
    TENSOR_CORE_INVERSE_PERMUTATION,
    TENSOR_CORE_PERMUTATION,
    TrellisTileQuantization,
    blockwise_hadamard_left,
    blockwise_hadamard_right,
    pack_trellis_tile,
    quantize_trellis_tile,
    reconstruct_matrix,
)


CODEBOOK_SCALE = 1.24371088
LDL_BLOCK = 16
HADAMARD_BLOCK = 128


@dataclass(frozen=True)
class FinalizedHessian:
    transformed: np.ndarray
    ldl: np.ndarray
    diagonal: np.ndarray
    input_signs: np.ndarray
    sample_count: int
    sigma_reg: float


@dataclass(frozen=True)
class RegularizedWeight:
    transformed: np.ndarray
    suh: np.ndarray
    svh: np.ndarray
    global_scale: float


@dataclass(frozen=True)
class QuantizedMatrix:
    reconstructed: np.ndarray
    encoded_tiles: tuple[tuple[int, ...], ...]
    packed_payload: bytes
    squared_error: float


@dataclass(frozen=True)
class CalibrationCapture:
    layer: int
    positions: np.ndarray
    expert_ids: np.ndarray
    gate_up_inputs: np.ndarray
    routed_down_inputs: np.ndarray


def read_calibration_capture(path: Path, *, maximum_records: int = 4096) -> CalibrationCapture:
    record_bytes = 4 + 8 * 4 + 2816 * 4 + 8 * 704 * 4
    try:
        size = path.stat().st_size
    except OSError as error:
        raise InvalidPlanError(f"cannot inspect Trellis35 calibration capture: {error}") from error
    if size < 20 or size > 20 + maximum_records * record_bytes:
        raise InvalidPlanError("Trellis35 calibration capture exceeds its bounded extent")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise InvalidPlanError(f"cannot read Trellis35 calibration capture: {error}") from error
    if raw[:8] != b"G16T35C1":
        raise InvalidPlanError("Trellis35 calibration capture has the wrong magic")
    version, layer, records = struct.unpack_from("<III", raw, 8)
    if version != 1 or layer >= 30 or records == 0 or records > maximum_records:
        raise InvalidPlanError("Trellis35 calibration capture header is invalid")
    if len(raw) != 20 + records * record_bytes:
        raise InvalidPlanError("Trellis35 calibration capture byte count is inconsistent")
    positions = np.empty(records, dtype=np.uint32)
    ids = np.empty((records, 8), dtype=np.uint32)
    gate = np.empty((records, 2816), dtype=np.float32)
    down = np.empty((records, 8, 704), dtype=np.float32)
    offset = 20
    for record in range(records):
        positions[record] = struct.unpack_from("<I", raw, offset)[0]
        offset += 4
        ids[record] = np.frombuffer(raw, dtype="<u4", count=8, offset=offset)
        offset += 32
        gate[record] = np.frombuffer(raw, dtype="<f4", count=2816, offset=offset)
        offset += 2816 * 4
        down[record] = np.frombuffer(raw, dtype="<f4", count=8 * 704, offset=offset).reshape(8, 704)
        offset += 8 * 704 * 4
    if not np.array_equal(positions, np.arange(records, dtype=np.uint32)):
        raise InvalidPlanError("Trellis35 calibration positions are not canonical")
    if np.any(ids >= 128) or any(len(set(row.tolist())) != 8 for row in ids):
        raise InvalidPlanError("Trellis35 calibration routed expert IDs are invalid")
    if not np.isfinite(gate).all() or not np.isfinite(down).all():
        raise InvalidPlanError("Trellis35 calibration activations are non-finite")
    return CalibrationCapture(int(layer), positions, ids, gate, down)


def combine_calibration_captures(captures: Sequence[CalibrationCapture]) -> CalibrationCapture:
    if not captures or any(capture.layer != captures[0].layer for capture in captures):
        raise InvalidPlanError("Trellis35 calibration captures must be non-empty and share one layer")
    return CalibrationCapture(
        captures[0].layer,
        np.arange(sum(c.positions.size for c in captures), dtype=np.uint32),
        np.concatenate([c.expert_ids for c in captures]),
        np.concatenate([c.gate_up_inputs for c in captures]),
        np.concatenate([c.routed_down_inputs for c in captures]),
    )


def calibration_hessian(capture: CalibrationCapture, family: str, expert: int | None = None) -> tuple[np.ndarray, int]:
    if family == "gate_up" and expert is None:
        values = capture.gate_up_inputs.astype(np.float64)
    elif family == "down" and isinstance(expert, int) and not isinstance(expert, bool) and 0 <= expert < 128:
        selected = capture.expert_ids == expert
        values = capture.routed_down_inputs[selected].astype(np.float64)
    else:
        raise InvalidPlanError("Trellis35 calibration Hessian request is invalid")
    if values.shape[0] == 0:
        raise InvalidPlanError("Trellis35 calibration has no samples for the requested expert")
    return values.T @ values, int(values.shape[0])


def read_source_expert(
    tensor: TensorDescriptor,
    expert_index: int,
    family: str,
) -> np.ndarray:
    """Read one expert directly from a verified BF16 source tensor.

    The returned compiler orientation is ``[in_features, out_features]``.
    Down is zero-padded from 704 to 768 input rows; fused Gate+Up is never
    split or padded.
    """
    expected_shapes = {
        "gate_up": (128, 1408, 2816),
        "down": (128, 2816, 704),
    }
    if family not in expected_shapes:
        raise InvalidPlanError("unsupported Trellis35 expert projection family")
    if tensor.dtype != "BF16" or tensor.shape != expected_shapes[family]:
        raise InvalidPlanError(f"Trellis35 {family} source descriptor has the wrong dtype or shape")
    if isinstance(expert_index, bool) or not isinstance(expert_index, int) or not 0 <= expert_index < 128:
        raise InvalidPlanError("Trellis35 expert index must be in [0, 127]")
    values_per_expert = tensor.shape[1] * tensor.shape[2]
    byte_count = values_per_expert * 2
    offset = tensor.absolute_offset + expert_index * byte_count
    if offset < tensor.absolute_offset or offset + byte_count > tensor.absolute_offset + tensor.byte_length:
        raise InvalidPlanError("Trellis35 expert slice exceeds the verified tensor range")
    try:
        descriptor = os.open(tensor.path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            raw = os.pread(descriptor, byte_count, offset)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise InvalidPlanError(f"cannot read Trellis35 source expert: {error}") from error
    if len(raw) != byte_count:
        raise InvalidPlanError("short read while loading Trellis35 source expert")
    bits = np.frombuffer(raw, dtype="<u2")
    floats = (bits.astype(np.uint32) << 16).view(np.float32)
    source_order = floats.reshape(tensor.shape[1], tensor.shape[2])
    compiler_order = source_order.T.copy()
    if family == "down":
        padded = np.zeros((768, 2816), dtype=np.float32)
        padded[:704] = compiler_order
        compiler_order = padded
    if not np.isfinite(compiler_order).all():
        raise InvalidPlanError("Trellis35 source expert contains non-finite BF16 values")
    return compiler_order


def _matrix(value: object, description: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise InvalidPlanError(f"{description} is not numeric") from error
    if result.ndim != 2 or not np.isfinite(result).all():
        raise InvalidPlanError(f"{description} must be one finite rank-2 matrix")
    return result


def deterministic_signs(length: int, seed: int, domain: str) -> np.ndarray:
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise InvalidPlanError("Trellis35 sign-vector length must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed >= 1 << 64:
        raise InvalidPlanError("Trellis35 seed must be a U64")
    if not isinstance(domain, str) or not domain or len(domain.encode("utf-8")) > 256:
        raise InvalidPlanError("Trellis35 sign domain must be a short non-empty string")
    prefix = seed.to_bytes(8, "little") + domain.encode("utf-8") + b"\0"
    signs = np.empty(length, dtype=np.float64)
    for index in range(length):
        digest = hashlib.sha256(prefix + index.to_bytes(8, "little")).digest()
        signs[index] = 1.0 if digest[0] & 1 else -1.0
    return signs


def block_ldl(hessian: object, block: int = LDL_BLOCK) -> np.ndarray:
    """Return the normalized block-Cholesky factor used by LDLQ."""
    matrix = _matrix(hessian, "Trellis35 Hessian")
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] % block:
        raise InvalidPlanError("Trellis35 Hessian must be square and block-aligned")
    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise InvalidPlanError("Trellis35 damped Hessian is not positive definite") from error
    size = matrix.shape[0]
    for start in range(0, size, block):
        diagonal_block = factor[start:start + block, start:start + block]
        try:
            inverse = np.linalg.inv(diagonal_block)
        except np.linalg.LinAlgError as error:
            raise InvalidPlanError("Trellis35 block-Cholesky diagonal is singular") from error
        factor[:, start:start + block] = factor[:, start:start + block] @ inverse
        factor[start:start + block, start:start + block] = np.eye(block)
    np.fill_diagonal(factor, 0.0)
    return factor


def finalize_hessian(
    capture_sum: object,
    sample_count: int,
    *,
    seed: int,
    domain: str,
    sigma_reg: float = 0.025,
) -> FinalizedHessian:
    matrix = _matrix(capture_sum, "Trellis35 Hessian capture")
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] % HADAMARD_BLOCK:
        raise InvalidPlanError("Trellis35 Hessian dimension must be square and divisible by 128")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise InvalidPlanError("Trellis35 Hessian capture must contain samples")
    if not isinstance(sigma_reg, (int, float)) or not np.isfinite(sigma_reg) or sigma_reg <= 0:
        raise InvalidPlanError("Trellis35 Hessian damping must be finite and positive")
    hessian = matrix / sample_count
    hessian = (hessian + hessian.T) * 0.5
    diagonal = np.diag(hessian).copy()
    diagonal_mean = float(diagonal.mean())
    if not np.isfinite(diagonal_mean) or diagonal_mean < 1e-20:
        raise InvalidPlanError("Trellis35 Hessian capture has insufficient finite energy")
    hessian = hessian.copy()
    hessian.flat[:: hessian.shape[0] + 1] += float(sigma_reg) * diagonal_mean
    signs = deterministic_signs(hessian.shape[0], seed, domain)
    hessian *= signs[None, :]
    hessian = blockwise_hadamard_right(hessian)
    hessian *= signs[:, None]
    hessian = blockwise_hadamard_left(hessian)
    return FinalizedHessian(
        transformed=hessian,
        ldl=block_ldl(hessian),
        diagonal=diagonal,
        input_signs=signs,
        sample_count=sample_count,
        sigma_reg=float(sigma_reg),
    )


def regularize_weight(
    source: object,
    input_signs: Sequence[float],
    *,
    seed: int,
    domain: str,
    global_scale: float = 1.0,
) -> RegularizedWeight:
    weight = _matrix(source, "Trellis35 source weight").copy()
    rows, columns = weight.shape
    signs = np.asarray(input_signs, dtype=np.float64)
    if rows % HADAMARD_BLOCK or columns % HADAMARD_BLOCK or signs.shape != (rows,):
        raise InvalidPlanError("Trellis35 regularization requires 128-aligned dimensions and input signs")
    if not np.isfinite(signs).all() or not np.all(np.abs(signs) == 1.0):
        raise InvalidPlanError("Trellis35 input signs must be finite Rademacher values")
    if not isinstance(global_scale, (int, float)) or not np.isfinite(global_scale) or global_scale <= 0:
        raise InvalidPlanError("Trellis35 global scale must be finite and positive")

    output_rms = np.sqrt(np.mean(np.square(weight), axis=0))
    mean = float(output_rms.mean())
    if mean <= 1e-30:
        output_rms = np.ones(columns, dtype=np.float64)
    else:
        output_rms /= mean
        output_rms[output_rms < 1e-30] = 0.1
    output_signs = deterministic_signs(columns, seed, domain + ":output")
    svh = output_signs * output_rms
    weight /= svh[None, :]
    weight = blockwise_hadamard_right(weight)

    input_rms = np.sqrt(np.mean(np.square(weight), axis=1))
    input_rms[input_rms < 1e-30] = 0.1
    suh = signs * input_rms / -CODEBOOK_SCALE
    weight /= suh[:, None]
    weight = blockwise_hadamard_left(weight)
    weight *= float(global_scale)
    suh /= float(global_scale)
    return RegularizedWeight(weight, suh, svh, float(global_scale))


def quantize_matrix_ldlq(source: object, ldl: object, rate_bits: int) -> QuantizedMatrix:
    weight = _matrix(source, "Trellis35 LDLQ weight")
    factor = _matrix(ldl, "Trellis35 LDLQ factor")
    rows, columns = weight.shape
    if rows % LDL_BLOCK or columns % LDL_BLOCK or factor.shape != (rows, rows):
        raise InvalidPlanError("Trellis35 LDLQ shapes must be 16-aligned and share the input dimension")
    reconstructed = np.zeros_like(weight)
    encoded: list[tuple[int, ...]] = [tuple()] * ((rows // 16) * (columns // 16))
    inverse = np.asarray(TENSOR_CORE_INVERSE_PERMUTATION)
    permutation = np.asarray(TENSOR_CORE_PERMUTATION)
    tile_columns = columns // 16
    for row_end in range(rows, 0, -16):
        row_start = row_end - 16
        error_below = weight[row_end:] - reconstructed[row_end:]
        compensation = factor[row_end:, row_start:row_end].T @ error_below
        block_rows = weight[row_start:row_end] + compensation
        for tile_column in range(tile_columns):
            column_start = tile_column * 16
            tile = block_rows[:, column_start:column_start + 16].reshape(256)
            result: TrellisTileQuantization = quantize_trellis_tile(tuple(tile[permutation]), rate_bits)
            row_major = np.asarray(result.reconstructed, dtype=np.float64)[inverse].reshape(16, 16)
            reconstructed[row_start:row_end, column_start:column_start + 16] = row_major
            encoded[(row_start // 16) * tile_columns + tile_column] = result.encoded
    error = weight - reconstructed
    return QuantizedMatrix(
        reconstructed=reconstructed,
        encoded_tiles=tuple(encoded),
        packed_payload=b"".join(pack_trellis_tile(tile, rate_bits) for tile in encoded),
        squared_error=float(np.sum(np.square(error))),
    )


def proxy_error(source: object, reconstructed: object, hessian: object) -> float:
    weight = _matrix(source, "Trellis35 proxy source")
    quantized = _matrix(reconstructed, "Trellis35 proxy reconstruction")
    matrix = _matrix(hessian, "Trellis35 proxy Hessian")
    if weight.shape != quantized.shape or matrix.shape != (weight.shape[0], weight.shape[0]):
        raise InvalidPlanError("Trellis35 proxy-error shapes are inconsistent")
    error = weight - quantized
    numerator = float(np.sum(error * (matrix @ error)))
    denominator = float(np.sum(weight * (matrix @ weight)))
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        raise InvalidPlanError("Trellis35 proxy-error denominator is not positive and finite")
    return numerator / denominator


def reconstruct_regularized(value: RegularizedWeight) -> np.ndarray:
    return reconstruct_matrix(value.transformed, value.suh, value.svh)

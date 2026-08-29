"""CPU format oracle for GEM16-Trellis35 16x16 trellis tiles.

Selected algorithms are derived from ExLlamaV3 at the revision recorded in
``third_party/exllamav3_quant/PROVENANCE.md`` and are covered by the MIT license
stored beside that record. This module is an offline compiler oracle, not a
runtime dependency.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
import math
import struct

import numpy as np

from .common import InvalidPlanError


TRELLIS_TILE_VALUES = 256
TRELLIS35_RATES = (3, 4)

CODEBOOK_3INST = 0
CODEBOOK_MCG = 1
CODEBOOK_MUL1 = 2
TRELLIS35_CODEBOOK_IDS = (CODEBOOK_3INST, CODEBOOK_MCG, CODEBOOK_MUL1)
TRELLIS35_INITIAL_CODEBOOK = CODEBOOK_MUL1

CODEBOOK_MCG_MULTIPLIER = 0xCBAC1FED
CODEBOOK_MUL1_MULTIPLIER = 0x83DCD12D
CODEBOOK_LOP3_AND_MASK = 0x8FFF8FFF
CODEBOOK_LOP3_XOR_MASK = 0x3B603B60
UINT16_MAX = (1 << 16) - 1
UINT32_MASK = (1 << 32) - 1


def _rate_bits(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in TRELLIS35_RATES:
        raise InvalidPlanError("Trellis35 v1 tile rate must be K3 or K4")
    return value


def tensor_core_permutation() -> tuple[int, ...]:
    """Return the upstream 16x16 row-major to Tensor-Core tile permutation."""
    permutation = [0] * TRELLIS_TILE_VALUES
    for thread in range(32):
        row0 = (thread % 4) * 2
        row1 = row0 + 1
        row2 = row0 + 8
        row3 = row0 + 9
        column0 = thread // 4
        column1 = column0 + 8
        offset = thread * 8
        permutation[offset:offset + 8] = (
            row0 * 16 + column0,
            row1 * 16 + column0,
            row2 * 16 + column0,
            row3 * 16 + column0,
            row0 * 16 + column1,
            row1 * 16 + column1,
            row2 * 16 + column1,
            row3 * 16 + column1,
        )
    if sorted(permutation) != list(range(TRELLIS_TILE_VALUES)):
        raise AssertionError("internal Trellis35 Tensor-Core permutation is not bijective")
    return tuple(permutation)


def inverse_permutation(permutation: Sequence[int]) -> tuple[int, ...]:
    if (
        not isinstance(permutation, Sequence)
        or isinstance(permutation, (str, bytes))
        or len(permutation) != TRELLIS_TILE_VALUES
    ):
        raise InvalidPlanError("Trellis35 permutation must contain 256 entries")
    inverse = [-1] * TRELLIS_TILE_VALUES
    for index, value in enumerate(permutation):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 256:
            raise InvalidPlanError(f"invalid Trellis35 permutation entry {index}")
        if inverse[value] != -1:
            raise InvalidPlanError(f"duplicate Trellis35 permutation value {value}")
        inverse[value] = index
    return tuple(inverse)


TENSOR_CORE_PERMUTATION = tensor_core_permutation()
TENSOR_CORE_INVERSE_PERMUTATION = inverse_permutation(TENSOR_CORE_PERMUTATION)


def _branch_tuple(branches: Sequence[int], rate_bits: int) -> tuple[int, ...]:
    rate = _rate_bits(rate_bits)
    if (
        not isinstance(branches, Sequence)
        or isinstance(branches, (str, bytes))
        or len(branches) != TRELLIS_TILE_VALUES
    ):
        raise InvalidPlanError("Trellis35 branch tile must contain 256 entries")
    maximum = (1 << rate) - 1
    result: list[int] = []
    for index, branch in enumerate(branches):
        if isinstance(branch, bool) or not isinstance(branch, int) or not 0 <= branch <= maximum:
            raise InvalidPlanError(
                f"Trellis35 K{rate} branch {index} must be in [0, {maximum}]"
            )
        result.append(branch)
    return tuple(result)


def encoded_tile_from_branches(branches: Sequence[int], rate_bits: int) -> tuple[int, ...]:
    """Expand 256 cyclic K-bit branches to the 16-bit tail-biting states."""
    rate = _rate_bits(rate_bits)
    values = _branch_tuple(branches, rate)
    history = (16 + rate - 1) // rate
    encoded: list[int] = []
    for index in range(TRELLIS_TILE_VALUES):
        state = 0
        for shift in range(history):
            state |= values[(index - shift) % TRELLIS_TILE_VALUES] << (rate * shift)
        encoded.append(state & UINT16_MAX)
    return tuple(encoded)


def validate_encoded_tile(encoded: Sequence[int], rate_bits: int) -> tuple[int, ...]:
    rate = _rate_bits(rate_bits)
    if (
        not isinstance(encoded, Sequence)
        or isinstance(encoded, (str, bytes))
        or len(encoded) != TRELLIS_TILE_VALUES
    ):
        raise InvalidPlanError("Trellis35 encoded tile must contain 256 entries")
    states: list[int] = []
    for index, value in enumerate(encoded):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= UINT16_MAX:
            raise InvalidPlanError(f"Trellis35 encoded state {index} is not U16")
        states.append(value)
    mask = (1 << rate) - 1
    expected = encoded_tile_from_branches(tuple(value & mask for value in states), rate)
    if tuple(states) != expected:
        raise InvalidPlanError("Trellis35 encoded tile violates the cyclic tail-biting contract")
    return tuple(states)


def pack_trellis_tile(encoded: Sequence[int], rate_bits: int) -> bytes:
    """Pack one valid tail-biting tile into the exact upstream K-bit byte order."""
    rate = _rate_bits(rate_bits)
    states = validate_encoded_tile(encoded, rate)
    mask = (1 << rate) - 1
    words: list[int] = []
    for span in range(16):
        bitstream = 0
        for index in range(16):
            bitstream = (bitstream << rate) | (states[span * 16 + index] & mask)
        for word in range(rate):
            shift = 16 * (rate - word - 1)
            words.append((bitstream >> shift) & UINT16_MAX)
    if len(words) != 16 * rate or len(words) % 2:
        raise AssertionError("internal Trellis35 packed-word count mismatch")
    # CUDA __byte_perm(x, 0, 0x1032) swaps adjacent U16s in every U32.
    swapped: list[int] = []
    for index in range(0, len(words), 2):
        swapped.extend((words[index + 1], words[index]))
    return struct.pack(f"<{len(swapped)}H", *swapped)


def unpack_trellis_tile(payload: bytes | bytearray | memoryview, rate_bits: int) -> tuple[int, ...]:
    """Unpack one tile and reconstruct its full 16-bit cyclic states."""
    rate = _rate_bits(rate_bits)
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise InvalidPlanError("Trellis35 packed tile must be bytes-like")
    raw = bytes(payload)
    expected_bytes = 32 * rate
    if len(raw) != expected_bytes:
        raise InvalidPlanError(
            f"Trellis35 K{rate} packed tile must contain {expected_bytes} bytes"
        )
    swapped = struct.unpack(f"<{16 * rate}H", raw)
    words: list[int] = []
    for index in range(0, len(swapped), 2):
        words.extend((swapped[index + 1], swapped[index]))
    mask = (1 << rate) - 1
    branches: list[int] = []
    for span in range(16):
        bitstream = 0
        for word in words[span * rate:(span + 1) * rate]:
            bitstream = (bitstream << 16) | word
        for index in range(16):
            shift = rate * (15 - index)
            branches.append((bitstream >> shift) & mask)
    return encoded_tile_from_branches(branches, rate)


def _half_bits_to_float(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits & UINT16_MAX))[0]


def _round_to_half_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def _half_add_bits(left: float, right: float) -> int:
    return _round_to_half_bits(left + right)


def _half_fma_bits(left: float, right: float, addend: float) -> int:
    # Binary64 represents products and sums of binary16 inputs exactly here, so
    # this performs the same single final binary16 rounding as CUDA __hfma.
    return _round_to_half_bits(left * right + addend)


def _codebook_lop3(value: int) -> int:
    # PTX lop3.b32(a, b, c, 0x6a) evaluates c ^ (a & b).
    return CODEBOOK_LOP3_XOR_MASK ^ (value & CODEBOOK_LOP3_AND_MASK)


def decode_codebook_half_bits(encoded_state: int, codebook_id: int) -> int:
    """Decode one U16 trellis state to the exact binary16 codebook value."""
    if (
        isinstance(encoded_state, bool)
        or not isinstance(encoded_state, int)
        or not 0 <= encoded_state <= UINT16_MAX
    ):
        raise InvalidPlanError("Trellis35 encoded codebook state must be U16")
    if isinstance(codebook_id, bool) or codebook_id not in TRELLIS35_CODEBOOK_IDS:
        raise InvalidPlanError("unsupported Trellis35 codebook ID")

    if codebook_id == CODEBOOK_3INST:
        value = (encoded_state * 89_226_354 + 64_248_484) & UINT32_MASK
        value = _codebook_lop3(value)
        return _half_add_bits(
            _half_bits_to_float(value),
            _half_bits_to_float(value >> 16),
        )
    if codebook_id == CODEBOOK_MCG:
        value = (encoded_state * CODEBOOK_MCG_MULTIPLIER) & UINT32_MASK
        value = _codebook_lop3(value)
        return _half_add_bits(
            _half_bits_to_float(value),
            _half_bits_to_float(value >> 16),
        )

    value = (encoded_state * CODEBOOK_MUL1_MULTIPLIER) & UINT32_MASK
    byte_sum = sum((value >> shift) & 0xFF for shift in (0, 8, 16, 24))
    accumulator = _half_bits_to_float(0x6400 + byte_sum)
    return _half_fma_bits(
        accumulator,
        _half_bits_to_float(0x1EEE),
        _half_bits_to_float(0xC931),
    )


def decode_codebook(encoded_state: int, codebook_id: int) -> float:
    return _half_bits_to_float(decode_codebook_half_bits(encoded_state, codebook_id))


@lru_cache(maxsize=3)
def _codebook_table(codebook_id: int) -> np.ndarray:
    if codebook_id not in TRELLIS35_CODEBOOK_IDS:
        raise InvalidPlanError("unsupported Trellis35 codebook ID")
    bits = np.fromiter(
        (decode_codebook_half_bits(state, codebook_id) for state in range(1 << 16)),
        dtype=np.uint16,
        count=1 << 16,
    )
    values = bits.view(np.float16)
    values.flags.writeable = False
    return values


@lru_cache(maxsize=6)
def _trellis_edges(rate_bits: int, codebook_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rate = _rate_bits(rate_bits)
    if codebook_id not in TRELLIS35_CODEBOOK_IDS:
        raise InvalidPlanError("unsupported Trellis35 codebook ID")
    edge_count = 1 << (16 - rate)
    labels = 1 << rate
    low_edges = np.arange(edge_count, dtype=np.uint32)
    states = np.empty((labels, edge_count), dtype=np.uint16)
    incoming = np.empty_like(states)
    for label in range(labels):
        state = (label << (16 - rate)) | low_edges
        states[label] = state.astype(np.uint16)
        incoming[label] = (state >> rate).astype(np.uint16)
    decoded = _codebook_table(codebook_id)[states]
    states.flags.writeable = False
    incoming.flags.writeable = False
    decoded.flags.writeable = False
    return states, incoming, decoded


@dataclass(frozen=True)
class TrellisTileQuantization:
    encoded: tuple[int, ...]
    reconstructed: tuple[float, ...]
    squared_error: float


def _cuda_argmin_cost(costs: np.ndarray) -> int:
    """Reproduce the pinned 512-thread strict-compare reduction tie order."""
    minimum = np.min(costs)
    edge_count = int(costs.shape[0])
    for half in range(2):
        for warp in range(16):
            for lane in range(32):
                start = warp * 32 + lane + half * 512
                for edge in range(start, edge_count, 1024):
                    if costs[edge] == minimum:
                        return edge
    raise AssertionError("internal Trellis35 Viterbi argmin is absent")


def quantize_trellis_tile(
    values: Sequence[float],
    rate_bits: int,
    codebook_id: int = TRELLIS35_INITIAL_CODEBOOK,
) -> TrellisTileQuantization:
    """Slow CPU tail-biting Viterbi oracle matching the pinned tile contract."""
    rate = _rate_bits(rate_bits)
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or len(values) != TRELLIS_TILE_VALUES
    ):
        raise InvalidPlanError("Trellis35 quantizer tile must contain 256 values")
    try:
        source = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise InvalidPlanError("Trellis35 quantizer tile contains invalid values") from error
    if source.shape != (TRELLIS_TILE_VALUES,) or not np.isfinite(source).all():
        raise InvalidPlanError("Trellis35 quantizer tile must contain finite scalar values")
    source_half = source.astype(np.float16)
    if not np.isfinite(source_half).all():
        raise InvalidPlanError("Trellis35 quantizer tile exceeds binary16 range")

    states, incoming, decoded = _trellis_edges(rate, codebook_id)
    edge_count = 1 << (16 - rate)
    predecessors = np.empty((TRELLIS_TILE_VALUES, edge_count), dtype=np.uint16)

    def forward(roll: int, pre_state: int) -> None:
        nonlocal predecessors
        costs: np.ndarray | None = None
        reduction_costs: np.ndarray | None = None
        for step in range(TRELLIS_TILE_VALUES):
            position = (step + roll) & 255
            difference = np.subtract(decoded, source_half[position], dtype=np.float16)
            if step == 0:
                candidates = np.multiply(difference, difference, dtype=np.float16)
                if pre_state >= 0:
                    candidates = candidates.copy()
                    candidates[incoming != pre_state] = np.float16(np.inf)
            else:
                # CUDA uses __hfma2(dh, dh, prior): one binary16 rounding
                # after the product and add, not a rounded multiply followed
                # by a separately rounded add.
                candidates = (
                    difference.astype(np.float64) * difference.astype(np.float64)
                    + costs[incoming].astype(np.float64)
                ).astype(np.float16)
                if step == TRELLIS_TILE_VALUES - 1:
                    # The pinned CUDA kernel ping-pongs two cost buffers but
                    # argmin_cost() reads temp_costs_inc after the final swap.
                    # That is the step-254 buffer, while step-255 predecessors
                    # have already been written. Preserve this observable K3
                    # behavior as part of the v1 compiler format contract.
                    reduction_costs = costs
            selected = np.argmin(candidates, axis=0)
            columns = np.arange(edge_count)
            costs = candidates[selected, columns]
            predecessors[position] = incoming[selected, columns]
        forward.reduction_costs = reduction_costs

    forward.reduction_costs = None  # type: ignore[attr-defined]

    def backward(roll: int, write: bool, initial_edge: int) -> tuple[int, tuple[int, ...] | None]:
        edge = initial_edge
        encoded = [0] * TRELLIS_TILE_VALUES if write else None
        for step in range(255, -1, -1):
            position = (step + roll) & 255
            previous = int(predecessors[position, edge])
            if encoded is not None:
                encoded[position] = ((previous << rate) | edge) & UINT16_MAX
            edge = previous
            if not write and position == 0:
                break
        return edge, tuple(encoded) if encoded is not None else None

    forward(128, -1)
    first_costs = forward.reduction_costs  # type: ignore[attr-defined]
    if first_costs is None:
        raise AssertionError("internal Trellis35 Viterbi costs are absent")
    end_edge = _cuda_argmin_cost(first_costs)
    end_state, _unused = backward(128, False, end_edge)
    forward(0, end_state)
    _closed_state, encoded = backward(0, True, end_state)
    if encoded is None:
        raise AssertionError("internal Trellis35 Viterbi output is absent")
    encoded = validate_encoded_tile(encoded, rate)
    reconstructed_half = _codebook_table(codebook_id)[np.asarray(encoded, dtype=np.uint16)]
    reconstructed = reconstructed_half.astype(np.float32)
    error = source.astype(np.float64) - reconstructed.astype(np.float64)
    return TrellisTileQuantization(
        encoded=encoded,
        reconstructed=tuple(float(value) for value in reconstructed),
        squared_error=float(np.dot(error, error)),
    )


@lru_cache(maxsize=1)
def hadamard_128() -> np.ndarray:
    """Normalized Sylvester H128 used by the pinned EXL3 regularizer."""
    matrix = np.ones((1, 1), dtype=np.float64)
    while matrix.shape[0] < 128:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    matrix *= 1.0 / math.sqrt(128.0)
    matrix.flags.writeable = False
    return matrix


def _finite_matrix(value: object, description: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise InvalidPlanError(f"{description} is not a numeric matrix") from error
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise InvalidPlanError(f"{description} must be one finite rank-2 matrix")
    return matrix


def blockwise_hadamard_left(value: object) -> np.ndarray:
    matrix = _finite_matrix(value, "Trellis35 left-Hadamard input")
    rows, columns = matrix.shape
    if rows % 128:
        raise InvalidPlanError("Trellis35 left-Hadamard dimension must be divisible by 128")
    blocks = matrix.reshape(rows // 128, 128, columns)
    return np.einsum("ij,bjk->bik", hadamard_128(), blocks).reshape(rows, columns)


def blockwise_hadamard_right(value: object) -> np.ndarray:
    matrix = _finite_matrix(value, "Trellis35 right-Hadamard input")
    rows, columns = matrix.shape
    if columns % 128:
        raise InvalidPlanError("Trellis35 right-Hadamard dimension must be divisible by 128")
    blocks = matrix.reshape(rows, columns // 128, 128)
    return np.einsum("rbi,ij->rbj", blocks, hadamard_128()).reshape(rows, columns)


def reconstruct_matrix(transformed: object, suh: object, svh: object) -> np.ndarray:
    """Reconstruct diag(suh) H W_hat H diag(svh) in the original basis."""
    matrix = _finite_matrix(transformed, "Trellis35 transformed matrix")
    input_scale = np.asarray(suh, dtype=np.float64)
    output_scale = np.asarray(svh, dtype=np.float64)
    if input_scale.shape != (matrix.shape[0],) or output_scale.shape != (matrix.shape[1],):
        raise InvalidPlanError("Trellis35 reconstruction sidecar shape mismatch")
    if not np.isfinite(input_scale).all() or not np.isfinite(output_scale).all():
        raise InvalidPlanError("Trellis35 reconstruction sidecars must be finite")
    reconstructed = blockwise_hadamard_left(matrix)
    reconstructed *= input_scale[:, None]
    reconstructed = blockwise_hadamard_right(reconstructed)
    reconstructed *= output_scale[None, :]
    return reconstructed


def regularize_matrix(source: object, suh: object, svh: object) -> np.ndarray:
    """Invert the reconstruction contract for nonzero fixture sidecars."""
    matrix = _finite_matrix(source, "Trellis35 source matrix")
    input_scale = np.asarray(suh, dtype=np.float64)
    output_scale = np.asarray(svh, dtype=np.float64)
    if input_scale.shape != (matrix.shape[0],) or output_scale.shape != (matrix.shape[1],):
        raise InvalidPlanError("Trellis35 regularization sidecar shape mismatch")
    if (
        not np.isfinite(input_scale).all()
        or not np.isfinite(output_scale).all()
        or np.any(input_scale == 0.0)
        or np.any(output_scale == 0.0)
    ):
        raise InvalidPlanError("Trellis35 fixture sidecars must be finite and nonzero")
    transformed = matrix / input_scale[:, None]
    transformed = blockwise_hadamard_left(transformed)
    transformed /= output_scale[None, :]
    return blockwise_hadamard_right(transformed)


def inverse_gate_up_output(transformed: object, svh: object) -> tuple[np.ndarray, np.ndarray]:
    """Invert all 1408 outputs before the semantic 704/704 split."""
    values = np.asarray(transformed, dtype=np.float64)
    scales = np.asarray(svh, dtype=np.float64)
    if values.shape != (1408,) or scales.shape != (1408,):
        raise InvalidPlanError("Trellis35 fused Gate+Up output must have 1408 values")
    if not np.isfinite(values).all() or not np.isfinite(scales).all():
        raise InvalidPlanError("Trellis35 fused Gate+Up output and scales must be finite")
    reconstructed = blockwise_hadamard_right(values.reshape(1, 1408)).reshape(1408)
    reconstructed *= scales
    return reconstructed[:704].copy(), reconstructed[704:].copy()


def gelu_tanh_product(gate: object, up: object) -> np.ndarray:
    gate_values = np.asarray(gate, dtype=np.float64)
    up_values = np.asarray(up, dtype=np.float64)
    if gate_values.shape != (704,) or up_values.shape != (704,):
        raise InvalidPlanError("Trellis35 Gate and Up vectors must each contain 704 values")
    inner = math.sqrt(2.0 / math.pi) * (gate_values + 0.044715 * gate_values ** 3)
    return (0.5 * gate_values * (1.0 + np.tanh(inner))) * up_values

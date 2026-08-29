"""Single-layer GEM16-Trellis35 artifact producer.

The producer is intentionally separate from the qualified NVFP4 compiler.  It
uses the slow Python implementation for contracts and transforms, and pinned
native CUDA programs for scale-search tiles and full-matrix LDLQ.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
from typing import BinaryIO, Sequence

import numpy as np

from .common import InvalidPlanError, canonical_json_bytes
from .trellis35 import FinalizedHessian, RegularizedWeight, regularize_weight
from .trellis35_layout import align_up, select_rate_map
from .trellis35_quant import (
    TENSOR_CORE_INVERSE_PERMUTATION,
    TENSOR_CORE_PERMUTATION,
    decode_codebook_half_bits,
)


CODEBOOK_ID = 2
ALIGNMENT = 256


@dataclass(frozen=True)
class NativeCandidate:
    rate_bits: int
    global_scale: float
    proxy_error: float
    payload_path: Path
    payload_bytes: int
    payload_sha256: str
    suh_path: Path
    svh_path: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def pack_encoded_tiles(encoded: object, rate_bits: int) -> bytes:
    """Vectorized exact K3/K4 packer for native U16 encoder output."""
    if rate_bits not in (3, 4):
        raise InvalidPlanError("Trellis35 native pack rate must be K3 or K4")
    states = np.asarray(encoded)
    if states.ndim != 2 or states.shape[1] != 256 or states.dtype != np.uint16:
        raise InvalidPlanError("Trellis35 native states must be U16 [tiles,256]")
    branches = states & np.uint16((1 << rate_bits) - 1)
    expected = np.zeros_like(states)
    for shift in range((16 + rate_bits - 1) // rate_bits):
        expected |= np.roll(branches, shift, axis=1) << np.uint16(rate_bits * shift)
    if not np.array_equal(states, expected):
        raise InvalidPlanError("native Trellis35 output violates tail-biting state recurrence")
    spans = branches.reshape(states.shape[0], 16, 16).astype(np.uint64)
    bitstreams = np.zeros((states.shape[0], 16), dtype=np.uint64)
    for index in range(16):
        bitstreams = (bitstreams << np.uint64(rate_bits)) | spans[:, :, index]
    words = np.empty((states.shape[0], 16, rate_bits), dtype=np.uint16)
    for word in range(rate_bits):
        words[:, :, word] = (
            bitstreams >> np.uint64(16 * (rate_bits - word - 1))
        ).astype(np.uint16)
    flat = words.reshape(states.shape[0], 16 * rate_bits)
    swapped = flat.reshape(states.shape[0], -1, 2)[:, :, ::-1].copy()
    return swapped.astype("<u2", copy=False).tobytes()


@lru_cache(maxsize=1)
def _mul1_codebook() -> np.ndarray:
    bits = np.fromiter(
        (decode_codebook_half_bits(state, CODEBOOK_ID) for state in range(1 << 16)),
        dtype=np.uint16,
        count=1 << 16,
    )
    values = bits.view(np.float16)
    values.flags.writeable = False
    return values


def decode_payload_matrix(payload: bytes, rate_bits: int, rows: int, columns: int) -> np.ndarray:
    """Decode one packed expert payload to its regularized [input,output] matrix."""
    if (
        rate_bits not in (3, 4)
        or rows <= 0
        or columns <= 0
        or rows % 16
        or columns % 16
        or len(payload) != rows * columns * rate_bits // 8
    ):
        raise InvalidPlanError("packed Trellis35 matrix extent is invalid")
    tile_count = rows * columns // 256
    swapped = np.frombuffer(payload, dtype="<u2").reshape(tile_count, 16, rate_bits)
    words = swapped.reshape(tile_count, -1, 2)[:, :, ::-1].reshape(
        tile_count, 16, rate_bits
    )
    bitstreams = np.zeros((tile_count, 16), dtype=np.uint64)
    for word in range(rate_bits):
        bitstreams = (bitstreams << np.uint64(16)) | words[:, :, word]
    branches = np.empty((tile_count, 16, 16), dtype=np.uint16)
    mask = np.uint64((1 << rate_bits) - 1)
    for index in range(16):
        branches[:, :, index] = (
            bitstreams >> np.uint64(rate_bits * (15 - index))
        ).astype(np.uint16) & np.uint16(mask)
    branches = branches.reshape(tile_count, 256)
    states = np.zeros_like(branches)
    for shift in range((16 + rate_bits - 1) // rate_bits):
        states |= np.roll(branches, shift, axis=1) << np.uint16(rate_bits * shift)
    decoded_tc = _mul1_codebook()[states].astype(np.float32)
    row_major_tiles = decoded_tc[:, np.asarray(TENSOR_CORE_INVERSE_PERMUTATION)].reshape(
        rows // 16, columns // 16, 16, 16
    )
    return row_major_tiles.transpose(0, 2, 1, 3).reshape(rows, columns)


def sample_scale_tiles(weight: object, width: int = 3) -> np.ndarray:
    matrix = np.asarray(weight, dtype=np.float32)
    if (
        matrix.ndim != 2
        or matrix.shape[0] % 16
        or matrix.shape[1] % 16
        or not np.isfinite(matrix).all()
        or width <= 0
    ):
        raise InvalidPlanError("Trellis35 scale-search matrix is invalid")
    tiles_k, tiles_n = matrix.shape[0] // 16, matrix.shape[1] // 16
    view = matrix.reshape(tiles_k, 16, tiles_n, 16)
    diagonal_length = max(tiles_k, tiles_n)
    ii = np.repeat(np.arange(diagonal_length), width)
    ww = np.tile(np.arange(width), diagonal_length)
    kk, nn = ii % tiles_k, (ii + ww) % tiles_n
    tile_ms = np.mean(np.square(view, dtype=np.float64), axis=(1, 3)).reshape(-1)
    extreme_count = min(max(8, diagonal_length * width // 16), (tile_ms.size + 1) // 2)
    order = np.argsort(tile_ms, kind="stable")
    low, high = order[:extreme_count], order[-extreme_count:][::-1]
    xk = np.concatenate((high, low)) // tiles_n
    xn = np.concatenate((high, low)) % tiles_n
    tiles = view[np.concatenate((kk, xk)), :, np.concatenate((nn, xn)), :]
    return tiles.reshape(-1, 256)[:, np.asarray(TENSOR_CORE_PERMUTATION)].copy()


class NativeTools:
    def __init__(self, quantize: Path, ldlq: Path) -> None:
        self.quantize = quantize.resolve(strict=True)
        self.ldlq = ldlq.resolve(strict=True)
        if not self.quantize.is_file() or not self.ldlq.is_file():
            raise InvalidPlanError("Trellis35 native tools must be regular files")

    @staticmethod
    def _run(arguments: list[str]) -> None:
        result = subprocess.run(arguments, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            raise InvalidPlanError(
                "native Trellis35 producer failed: "
                + (result.stderr.strip() or result.stdout.strip() or str(result.returncode))
            )

    def quantize_tiles(self, tiles: np.ndarray, rate_bits: int, scratch: Path) -> np.ndarray:
        values = np.asarray(tiles, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 256 or not np.isfinite(values).all():
            raise InvalidPlanError("native Trellis35 tile batch is invalid")
        input_path, output_path, states_path = (
            scratch / "tiles.f32", scratch / "tiles-q.f32", scratch / "tiles.u16"
        )
        values.tofile(input_path)
        self._run([
            str(self.quantize), f"K{rate_bits}", str(values.shape[0]),
            str(input_path), str(output_path), str(states_path),
        ])
        expected = values.size * 4
        if output_path.stat().st_size != expected:
            raise InvalidPlanError("native Trellis35 tile reconstruction has the wrong size")
        reconstructed = np.fromfile(output_path, dtype="<f4").reshape(values.shape)
        if not np.isfinite(reconstructed).all():
            raise InvalidPlanError("native Trellis35 tile reconstruction is non-finite")
        return reconstructed

    def search_global_scale(self, weight: object, rate_bits: int, scratch: Path) -> tuple[float, float]:
        samples = sample_scale_tiles(weight)

        def evaluate(scales: Sequence[float], selected: np.ndarray) -> np.ndarray:
            batch = np.concatenate([selected * np.float32(scale) for scale in scales])
            reconstructed = self.quantize_tiles(batch, rate_bits, scratch)
            count = selected.shape[0]
            errors = []
            for index, scale in enumerate(scales):
                quantized = reconstructed[index * count:(index + 1) * count] / np.float32(scale)
                errors.append(float(np.mean(np.square(quantized - selected, dtype=np.float64))))
            return np.asarray(errors, dtype=np.float64)

        coarse = np.arange(0.1, 2.0, 0.2, dtype=np.float64)
        coarse_error = evaluate(coarse, samples[::3])
        center = float(coarse[int(np.argmin(coarse_error))])
        fine = np.asarray([center + 0.075 * (index - 2) for index in range(5)])
        fine_error = evaluate(fine, samples)
        best = int(np.argmin(fine_error))
        offset = 0.0
        if 0 < best < 4:
            y0, y1, y2 = fine_error[best - 1:best + 2]
            denominator = y0 - 2.0 * y1 + y2
            if denominator > 0:
                offset = float(np.clip(0.5 * (y0 - y2) / denominator, -0.5, 0.5))
        scale = max(float(fine[best]) + offset * 0.075, 0.01)
        return scale, float(fine_error[best])

    def compile_candidate(
        self,
        regularized: RegularizedWeight,
        finalized: FinalizedHessian,
        rate_bits: int,
        output_prefix: Path,
    ) -> NativeCandidate:
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="native-", dir=output_prefix.parent) as directory:
            scratch = Path(directory)
            scale, _sample_mse = self.search_global_scale(
                regularized.transformed, rate_bits, scratch
            )
            weight = (regularized.transformed * scale).astype("<f4")
            suh = (regularized.suh / scale).astype("<f2")
            svh = regularized.svh.astype("<f2")
            if not np.isfinite(suh).all() or not np.isfinite(svh).all():
                raise InvalidPlanError("Trellis35 F16 sidecar conversion overflowed")
            weight_path, ldl_path, hessian_path = (
                scratch / "weight.f32", scratch / "ldl.f32", scratch / "hessian.f32"
            )
            q_path, states_path, metrics_path = (
                scratch / "q.f32", scratch / "states.u16", scratch / "metrics.f32"
            )
            weight.tofile(weight_path)
            finalized.ldl.astype("<f4").tofile(ldl_path)
            finalized.transformed.astype("<f4").tofile(hessian_path)
            self._run([
                str(self.ldlq), f"K{rate_bits}", str(weight.shape[0]), str(weight.shape[1]),
                str(weight_path), str(ldl_path), str(q_path), str(states_path),
                str(hessian_path), str(metrics_path),
            ])
            tile_count = weight.size // 256
            if states_path.stat().st_size != tile_count * 256 * 2:
                raise InvalidPlanError("native Trellis35 LDLQ state output has the wrong size")
            states = np.fromfile(states_path, dtype="<u2").reshape(tile_count, 256)
            payload = pack_encoded_tiles(states, rate_bits)
            expected_payload = weight.size * rate_bits // 8
            if len(payload) != expected_payload:
                raise InvalidPlanError("native Trellis35 packed payload has the wrong size")
            metrics = np.fromfile(metrics_path, dtype="<f4")
            if metrics.shape != (2,) or not np.isfinite(metrics).all() or metrics[0] < 0 or metrics[1] <= 0:
                raise InvalidPlanError("native Trellis35 proxy metrics are invalid")
            payload_path = output_prefix.with_suffix(f".k{rate_bits}.payload")
            suh_path = output_prefix.with_suffix(f".k{rate_bits}.suh.f16")
            svh_path = output_prefix.with_suffix(f".k{rate_bits}.svh.f16")
            payload_path.write_bytes(payload)
            suh.tofile(suh_path)
            svh.tofile(svh_path)
        return NativeCandidate(
            rate_bits=rate_bits,
            global_scale=scale,
            proxy_error=float(metrics[0] / metrics[1]),
            payload_path=payload_path,
            payload_bytes=expected_payload,
            payload_sha256=sha256_file(payload_path),
            suh_path=suh_path,
            svh_path=svh_path,
        )


def write_layer_artifact(
    output_binary: Path,
    output_manifest: Path,
    *,
    layer: int,
    families: dict[str, Sequence[tuple[NativeCandidate, NativeCandidate]]],
    manifest_base: dict[str, object],
) -> dict[str, object]:
    if set(families) != {"gate_up", "down"} or not 0 <= layer < 30:
        raise InvalidPlanError("Trellis35 layer artifact input is invalid")
    rate_maps: dict[str, tuple[int, ...]] = {}
    selected: dict[str, list[NativeCandidate]] = {}
    shapes = {
        "gate_up": (2816, 1408, 2816, 1408),
        "down": (768, 2816, 768, 2816),
    }
    for family, candidates in families.items():
        if len(candidates) != 128:
            raise InvalidPlanError(f"Trellis35 {family} must contain 128 candidate pairs")
        rows, columns, suh_elements, svh_elements = shapes[family]
        for expert, pair in enumerate(candidates):
            if len(pair) != 2 or tuple(candidate.rate_bits for candidate in pair) != (3, 4):
                raise InvalidPlanError(f"Trellis35 {family} expert {expert} candidate rates are invalid")
            for candidate in pair:
                expected_payload = rows * columns * candidate.rate_bits // 8
                if (
                    not np.isfinite(candidate.global_scale)
                    or candidate.global_scale <= 0
                    or not np.isfinite(candidate.proxy_error)
                    or candidate.proxy_error < 0
                    or candidate.payload_bytes != expected_payload
                    or candidate.payload_path.stat().st_size != expected_payload
                    or candidate.suh_path.stat().st_size != suh_elements * 2
                    or candidate.svh_path.stat().st_size != svh_elements * 2
                ):
                    raise InvalidPlanError(
                        f"Trellis35 {family} expert {expert} K{candidate.rate_bits} candidate is invalid"
                    )
        benefits = [pair[0].proxy_error - pair[1].proxy_error for pair in candidates]
        rates = select_rate_map(benefits)
        rate_maps[family] = rates
        selected[family] = [pair[rate - 3] for pair, rate in zip(candidates, rates)]

    regions: dict[str, dict[str, int]] = {}
    descriptors: dict[str, list[tuple[int, int, int]]] = {}
    output_binary.parent.mkdir(parents=True, exist_ok=True)
    with output_binary.open("wb") as stream:
        for family in ("gate_up", "down"):
            descriptors[family] = [(0, 0, 0)] * 128
            for rate in (3, 4):
                _align_stream(stream)
                start = stream.tell()
                local_offset = 0
                for expert, candidate in enumerate(selected[family]):
                    if candidate.rate_bits != rate:
                        continue
                    _copy_file(candidate.payload_path, stream)
                    descriptors[family][expert] = (local_offset, rate, CODEBOOK_ID)
                    local_offset += candidate.payload_bytes
                regions[f"{family}_k{rate}_payload_pool"] = {
                    "offset": start, "bytes": stream.tell() - start,
                }
            _align_stream(stream)
            start = stream.tell()
            for descriptor in descriptors[family]:
                stream.write(struct.pack("<IHH", *descriptor))
            regions[f"{family}_descriptor"] = {"offset": start, "bytes": stream.tell() - start}
            for sidecar in ("suh", "svh"):
                _align_stream(stream)
                start = stream.tell()
                for candidate in selected[family]:
                    _copy_file(candidate.suh_path if sidecar == "suh" else candidate.svh_path, stream)
                regions[f"{family}_{sidecar}"] = {"offset": start, "bytes": stream.tell() - start}
        _align_stream(stream)
    manifest = dict(manifest_base)
    manifest.update({
        "schema_version": 1,
        "format": "GEM16-Trellis35",
        "format_version": 1,
        "layer": layer,
        "trellis_tile": [16, 16],
        "hadamard_block": 128,
        "codebook_id": CODEBOOK_ID,
        "alignment_bytes": ALIGNMENT,
        "rate_maps": {name: list(values) for name, values in rate_maps.items()},
        "regions": regions,
        "artifact": {
            "path": output_binary.name,
            "bytes": output_binary.stat().st_size,
            "sha256": sha256_file(output_binary),
        },
        "gate_up_boundary": 704,
        "gate_up_inverse_before_split": True,
        "padding_contract": {"gate_up": "none", "down": "input_zero_pad_704_to_768"},
        "candidate_proxy": {
            family: [
                {"expert": expert, "k3": pair[0].proxy_error, "k4": pair[1].proxy_error,
                 "benefit": pair[0].proxy_error - pair[1].proxy_error,
                 "selected_rate": rate_maps[family][expert]}
                for expert, pair in enumerate(candidates)
            ]
            for family, candidates in families.items()
        },
    })
    output_manifest.write_bytes(canonical_json_bytes(manifest))
    return manifest


def _align_stream(stream: BinaryIO) -> None:
    aligned = align_up(stream.tell(), ALIGNMENT, "Trellis35 artifact alignment")
    stream.write(bytes(aligned - stream.tell()))


def _copy_file(path: Path, output: BinaryIO) -> None:
    with path.open("rb") as source:
        shutil.copyfileobj(source, output, 4 * 1024 * 1024)

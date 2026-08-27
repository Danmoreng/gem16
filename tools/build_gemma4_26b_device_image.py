#!/usr/bin/env python3
"""Build the immutable SM120 runtime arena image for the accepted M08 artifact.

The source Safetensors remain the immutable, qualified checkpoint.  This tool
creates a deterministic sibling file whose bytes already use the exact device
arena offsets and SM120 Row-8/K-64 layouts consumed by the 26B runtime.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np

try:
    from tools.gem16_compile.common import canonical_json_bytes, load_json
except ModuleNotFoundError:  # Direct execution outside the repository root.
    from gem16_compile.common import canonical_json_bytes, load_json  # type: ignore[no-redef]


ACCEPTED_ARTIFACT_SHA256 = (
    "471805f7dad8abb84300be78b2822a63dcb1d35bff5aa98426a162cc8532ee17"
)
FORMAT = "gem16-sm120-device-image-v1"
IMAGE_NAME = "model.gem16"
ARENA_ALIGNMENT = 256
EXPECTED_TENSORS = 1285
EXPECTED_PAYLOAD_BYTES = 14_696_569_196
MAX_JSON_BYTES = 64 * 1024 * 1024
DEFAULT_STAGING_BYTES = 64 * 1024 * 1024


class DeviceImageError(RuntimeError):
    pass


@dataclass(frozen=True)
class TensorPlan:
    name: str
    shard: str
    source_offset: int
    destination_offset: int
    byte_length: int
    runtime_layout: str
    physical_shape: tuple[int, ...]
    logical_shape: tuple[int, ...]
    source_sha256: str


def _regular_file(path: Path, description: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise DeviceImageError(f"cannot inspect {description} {path}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise DeviceImageError(f"{description} is not a regular non-symlink file: {path}")
    return metadata


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb", buffering=0) as stream:
            while chunk := stream.read(chunk_bytes):
                digest.update(chunk)
    except OSError as error:
        raise DeviceImageError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _safe_leaf(value: object, description: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise DeviceImageError(f"invalid {description}: {value!r}")
    path = Path(value)
    if path.is_absolute() or path.name != value or value in (".", ".."):
        raise DeviceImageError(f"unsafe {description}: {value!r}")
    return value


def _integer(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DeviceImageError(f"invalid nonnegative integer for {description}")
    return value


def _shape(value: object, description: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise DeviceImageError(f"invalid {description}")
    return tuple(_integer(extent, description) for extent in value)


def _align_up(value: int, alignment: int = ARENA_ALIGNMENT) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def _verify_external_lock(model: Path) -> dict[str, Any]:
    lock_path = model.parent / f"{model.name}.lock.json"
    _regular_file(lock_path, "external artifact lock")
    lock = load_json(lock_path, MAX_JSON_BYTES)
    try:
        payload = lock_path.read_bytes()
    except OSError as error:
        raise DeviceImageError(f"cannot read {lock_path}: {error}") from error
    if payload != canonical_json_bytes(lock):
        raise DeviceImageError("external artifact lock is not canonical JSON")
    if (
        lock.get("schema_version") != 1
        or lock.get("artifact_profile") != "sm120-text-hybrid-v1"
        or lock.get("artifact_content_sha256") != ACCEPTED_ARTIFACT_SHA256
        or not isinstance(lock.get("files"), list)
    ):
        raise DeviceImageError("external artifact lock is not the accepted M08 identity")
    content = dict(lock)
    content.pop("schema_version", None)
    recorded = content.pop("artifact_content_sha256", None)
    if hashlib.sha256(canonical_json_bytes(content)).hexdigest() != recorded:
        raise DeviceImageError("external artifact lock aggregate hash mismatch")

    expected_names: set[str] = set()
    for record in lock["files"]:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "size"}:
            raise DeviceImageError("invalid external artifact file record")
        name = _safe_leaf(record.get("path"), "artifact file path")
        if name in expected_names:
            raise DeviceImageError(f"duplicate external artifact file record: {name}")
        expected_names.add(name)
        path = model / name
        metadata = _regular_file(path, "artifact file")
        size = _integer(record.get("size"), "artifact file size")
        expected_hash = record.get("sha256")
        if metadata.st_size != size or not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise DeviceImageError(f"artifact file metadata mismatch: {name}")
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hash:
            raise DeviceImageError(f"artifact file hash mismatch: {name}")

    try:
        actual_names = {
            entry.name
            for entry in model.iterdir()
            if _regular_file(entry, "artifact directory entry")
        }
    except OSError as error:
        raise DeviceImageError(f"cannot enumerate artifact directory: {error}") from error
    if actual_names != expected_names:
        raise DeviceImageError("external lock does not cover the exact artifact directory")
    return lock


def _safetensors_data_base(path: Path) -> int:
    metadata = _regular_file(path, "Safetensors shard")
    try:
        with path.open("rb", buffering=0) as stream:
            prefix = stream.read(8)
    except OSError as error:
        raise DeviceImageError(f"cannot read Safetensors header prefix {path}: {error}") from error
    if len(prefix) != 8:
        raise DeviceImageError(f"short Safetensors header prefix: {path}")
    header_bytes = struct.unpack("<Q", prefix)[0]
    if header_bytes < 2 or header_bytes > 256 * 1024 * 1024:
        raise DeviceImageError(f"invalid Safetensors header length: {path}")
    data_base = 8 + header_bytes
    if data_base > metadata.st_size:
        raise DeviceImageError(f"Safetensors header extends beyond file: {path}")
    return data_base


def _tensor_plan(model: Path, compilation: dict[str, Any]) -> tuple[list[TensorPlan], int]:
    records = compilation.get("tensors")
    if not isinstance(records, list) or len(records) != EXPECTED_TENSORS:
        raise DeviceImageError("M08 compilation tensor count mismatch")
    if (
        compilation.get("artifact_profile") != "sm120-text-hybrid-v1"
        or compilation.get("artifact_status")
        != "m08_complete_runtime_loadable_experimental"
    ):
        raise DeviceImageError("compilation is not the accepted M08 profile")

    data_bases: dict[str, int] = {}
    plan: list[TensorPlan] = []
    cursor = 0
    names: set[str] = set()
    for record in sorted(records, key=lambda item: item.get("output_name", "")):
        if not isinstance(record, dict):
            raise DeviceImageError("invalid compilation tensor record")
        name = record.get("output_name")
        if not isinstance(name, str) or not name or name in names:
            raise DeviceImageError(f"invalid or duplicate output tensor name: {name!r}")
        names.add(name)
        shard = _safe_leaf(record.get("output_shard"), "tensor shard")
        offsets = record.get("output_data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise DeviceImageError(f"invalid output offsets: {name}")
        begin = _integer(offsets[0], f"{name} output begin")
        end = _integer(offsets[1], f"{name} output end")
        byte_length = _integer(record.get("byte_length"), f"{name} byte length")
        if end < begin or end - begin != byte_length or byte_length == 0:
            raise DeviceImageError(f"inconsistent output range: {name}")
        if shard not in data_bases:
            data_bases[shard] = _safetensors_data_base(model / shard)
        source_offset = data_bases[shard] + begin
        shard_size = (model / shard).stat().st_size
        if source_offset > shard_size or byte_length > shard_size - source_offset:
            raise DeviceImageError(f"tensor extends beyond its shard: {name}")
        runtime_layout = record.get("runtime_layout")
        source_hash = record.get("sha256")
        if not isinstance(runtime_layout, str) or not runtime_layout:
            raise DeviceImageError(f"missing runtime layout: {name}")
        if not isinstance(source_hash, str) or len(source_hash) != 64:
            raise DeviceImageError(f"invalid source tensor hash: {name}")
        destination = _align_up(cursor)
        plan.append(
            TensorPlan(
                name=name,
                shard=shard,
                source_offset=source_offset,
                destination_offset=destination,
                byte_length=byte_length,
                runtime_layout=runtime_layout,
                physical_shape=_shape(record.get("physical_shape"), f"{name} physical shape"),
                logical_shape=_shape(record.get("logical_shape"), f"{name} logical shape"),
                source_sha256=source_hash,
            )
        )
        cursor = destination + byte_length
    if sum(tensor.byte_length for tensor in plan) != EXPECTED_PAYLOAD_BYTES:
        raise DeviceImageError("M08 compilation payload byte count mismatch")
    return plan, _align_up(cursor)


def _flattened_geometry(tensor: TensorPlan, scale: bool) -> tuple[int, int]:
    shape = tensor.physical_shape if scale else tensor.logical_shape
    if len(shape) < 2:
        raise DeviceImageError(f"invalid tiled tensor geometry: {tensor.name}")
    rows = math.prod(shape[:-1])
    contracting = shape[-1] * (16 if scale else 1)
    if rows <= 0 or contracting <= 0 or contracting % 64:
        raise DeviceImageError(f"unsupported tiled tensor geometry: {tensor.name}")
    return rows, contracting


def _source_view(model: Path, tensor: TensorPlan) -> np.memmap:
    return np.memmap(
        model / tensor.shard,
        dtype=np.uint8,
        mode="r",
        offset=tensor.source_offset,
        shape=(tensor.byte_length,),
    )


def _write_tiled(
    destination: np.memmap,
    model: Path,
    tensor: TensorPlan,
    staging_bytes: int,
) -> None:
    scale = tensor.runtime_layout.endswith("sm120_row8_group16_e4m3")
    rows, contracting = _flattened_geometry(tensor, scale)
    bytes_per_k_block = 4 if scale else 32
    row_bytes = contracting // (16 if scale else 2)
    if rows * row_bytes != tensor.byte_length:
        raise DeviceImageError(f"tiled tensor byte count mismatch: {tensor.name}")
    k_blocks = contracting // 64
    source = _source_view(model, tensor).reshape(rows, k_blocks, bytes_per_k_block)
    target = destination[
        tensor.destination_offset : tensor.destination_offset + tensor.byte_length
    ]
    full_tile_bytes = 8 * k_blocks * bytes_per_k_block
    tiles_per_chunk = max(1, staging_bytes // max(1, full_tile_bytes))
    full_rows = rows - rows % 8
    for first_row in range(0, full_rows, tiles_per_chunk * 8):
        end_row = min(full_rows, first_row + tiles_per_chunk * 8)
        tile_count = (end_row - first_row) // 8
        tiled = np.ascontiguousarray(
            source[first_row:end_row]
            .reshape(tile_count, 8, k_blocks, bytes_per_k_block)
            .transpose(0, 2, 1, 3)
        ).reshape(-1)
        begin = first_row * k_blocks * bytes_per_k_block
        target[begin : begin + tiled.size] = tiled
    if full_rows != rows:
        tail = np.ascontiguousarray(source[full_rows:rows].transpose(1, 0, 2)).reshape(-1)
        begin = full_rows * k_blocks * bytes_per_k_block
        target[begin : begin + tail.size] = tail


def _write_image(
    model: Path,
    output: Path,
    plan: Iterable[TensorPlan],
    arena_bytes: int,
    staging_bytes: int,
) -> None:
    partial = output.with_name(output.name + ".partial")
    for path in (output, partial):
        if path.exists() or path.is_symlink():
            raise DeviceImageError(f"refusing to overwrite existing output: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with partial.open("xb") as stream:
            stream.truncate(arena_bytes)
            if hasattr(os, "posix_fallocate"):
                try:
                    os.posix_fallocate(stream.fileno(), 0, arena_bytes)
                except OSError:
                    pass
        destination = np.memmap(partial, dtype=np.uint8, mode="r+", shape=(arena_bytes,))
        previous_end = 0
        tensors = list(plan)
        for index, tensor in enumerate(tensors, 1):
            if tensor.destination_offset > previous_end:
                destination[previous_end : tensor.destination_offset] = 0
            source = _source_view(model, tensor)
            actual_source_hash = hashlib.sha256(source).hexdigest()
            if actual_source_hash != tensor.source_sha256:
                raise DeviceImageError(f"source tensor hash mismatch: {tensor.name}")
            if tensor.runtime_layout.endswith("sm120_row8_k64") or tensor.runtime_layout.endswith(
                "sm120_row8_group16_e4m3"
            ):
                _write_tiled(destination, model, tensor, staging_bytes)
            else:
                begin = tensor.destination_offset
                destination[begin : begin + tensor.byte_length] = source
            previous_end = tensor.destination_offset + tensor.byte_length
            if index % 50 == 0 or index == len(tensors):
                print(f"device image: {index}/{len(tensors)} tensors", flush=True)
        if previous_end < arena_bytes:
            destination[previous_end:arena_bytes] = 0
        destination.flush()
        del destination
        os.replace(partial, output)
    except BaseException:
        # Keep a partial file for diagnosis; never promote it to a runtime image.
        raise


def _positive(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--staging-bytes", type=_positive, default=DEFAULT_STAGING_BYTES)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    options = parse_args(arguments)
    model = options.model.resolve(strict=True)
    if not model.is_dir() or model.is_symlink():
        raise DeviceImageError(f"model must be a real directory: {model}")
    output = (
        options.output.resolve(strict=False)
        if options.output is not None
        else model / IMAGE_NAME
    )
    started = time.monotonic()
    _verify_external_lock(model)
    compilation = load_json(model / "gem16_compilation.json", MAX_JSON_BYTES)
    plan, arena_bytes = _tensor_plan(model, compilation)
    print(
        f"building {FORMAT}: tensors={len(plan)} payload_bytes={EXPECTED_PAYLOAD_BYTES} "
        f"arena_bytes={arena_bytes} output={output}",
        flush=True,
    )
    _write_image(model, output, plan, arena_bytes, options.staging_bytes)
    digest = _sha256_file(output)
    elapsed = time.monotonic() - started
    print(f"device_image_sha256={digest}")
    print(f"device_image_bytes={arena_bytes}")
    print(f"elapsed_seconds={elapsed:.3f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DeviceImageError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(4)

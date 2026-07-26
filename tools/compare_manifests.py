#!/usr/bin/env python3
"""Independently compare gem16-inspect output with raw Safetensors headers."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path, PurePath
import struct
import sys
from typing import Any


MAX_HEADER_BYTES = 256 * 1024 * 1024
MAX_INDEX_BYTES = 256 * 1024 * 1024
DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
PHYSICAL_FIELDS = (
    "shape",
    "storage_dtype",
    "byte_offset",
    "byte_length",
    "source_shard",
)


class ManifestError(RuntimeError):
    pass


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path, maximum_bytes: int | None = None) -> Any:
    if maximum_bytes is not None and path.stat().st_size > maximum_bytes:
        raise ManifestError(f"JSON file exceeds safety limit: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream, object_pairs_hook=reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot parse {path}: {error}") from error


def checked_shape(value: Any, tensor_name: str) -> list[int]:
    if not isinstance(value, list):
        raise ManifestError(f"shape is not an array: {tensor_name}")
    shape: list[int] = []
    elements = 1
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
            raise ManifestError(f"invalid shape dimension: {tensor_name}")
        elements *= dimension
        if elements > (1 << 64) - 1:
            raise ManifestError(f"shape product exceeds uint64: {tensor_name}")
        shape.append(dimension)
    return shape


def tensor_alignment(offset: int) -> int:
    for alignment in (4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2):
        if offset % alignment == 0:
            return alignment
    return 1


def read_safetensors_header(path: Path) -> dict[str, dict[str, Any]]:
    try:
        file_size = path.stat().st_size
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ManifestError(f"short Safetensors length prefix: {path}")
            (header_length,) = struct.unpack("<Q", prefix)
            if header_length < 2 or header_length > MAX_HEADER_BYTES or header_length > file_size - 8:
                raise ManifestError(f"invalid Safetensors header length: {path}")
            raw_header = stream.read(header_length)
    except OSError as error:
        raise ManifestError(f"cannot read {path}: {error}") from error

    try:
        header = json.loads(raw_header, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"invalid Safetensors header JSON in {path}: {error}") from error
    if not isinstance(header, dict):
        raise ManifestError(f"Safetensors header root is not an object: {path}")

    data_base = 8 + header_length
    payload_size = file_size - data_base
    result: dict[str, dict[str, Any]] = {}
    intervals: list[tuple[int, int, str]] = []
    for name, metadata in header.items():
        if name == "__metadata__":
            if not isinstance(metadata, dict):
                raise ManifestError(f"__metadata__ is not an object: {path}")
            continue
        if not isinstance(metadata, dict):
            raise ManifestError(f"tensor metadata is not an object: {name}")
        dtype = metadata.get("dtype")
        offsets = metadata.get("data_offsets")
        if dtype not in DTYPE_BYTES:
            raise ManifestError(f"unsupported dtype {dtype!r}: {name}")
        shape = checked_shape(metadata.get("shape"), name)
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in offsets)
        ):
            raise ManifestError(f"invalid data_offsets: {name}")
        begin, end = offsets
        if begin < 0 or end < begin or end > payload_size:
            raise ManifestError(f"out-of-bounds data_offsets: {name}")
        expected_bytes = DTYPE_BYTES[dtype]
        for dimension in shape:
            expected_bytes *= dimension
        if end - begin != expected_bytes:
            raise ManifestError(f"dtype/shape byte mismatch: {name}")
        absolute_offset = data_base + begin
        result[name] = {
            "shape": shape,
            "storage_dtype": dtype,
            "byte_offset": absolute_offset,
            "byte_length": end - begin,
            "alignment": tensor_alignment(absolute_offset),
            "source_shard": path.name,
        }
        if end > begin:
            intervals.append((begin, end, name))
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] < previous[1]:
            raise ManifestError(f"overlapping tensors {previous[2]} and {current[2]} in {path}")
    return result


def discover_safetensors(model_directory: Path) -> tuple[list[Path], dict[str, str] | None]:
    root = model_directory.resolve(strict=True)
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        model_path = (root / "model.safetensors").resolve(strict=True)
        if model_path.parent != root:
            raise ManifestError("model.safetensors resolves outside the checkpoint directory")
        return [model_path], None

    index = load_json(index_path, MAX_INDEX_BYTES)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        raise ManifestError("Safetensors index lacks an object weight_map")
    assignments: dict[str, str] = {}
    shard_names: set[str] = set()
    for tensor, shard_name in weight_map.items():
        if not isinstance(tensor, str) or not isinstance(shard_name, str):
            raise ManifestError("Safetensors index entries must map strings to strings")
        shard_path = PurePath(shard_name)
        if shard_path.is_absolute() or len(shard_path.parts) != 1 or shard_path.suffix != ".safetensors":
            raise ManifestError(f"unsafe shard path: {shard_name}")
        assignments[tensor] = shard_name
        shard_names.add(shard_name)
    shards: list[Path] = []
    for shard_name in sorted(shard_names):
        shard = (root / shard_name).resolve(strict=True)
        if shard.parent != root:
            raise ManifestError(f"shard resolves outside checkpoint directory: {shard_name}")
        shards.append(shard)
    return shards, assignments


def build_reference_manifest(model_directory: Path) -> dict[str, dict[str, Any]]:
    shards, assignments = discover_safetensors(model_directory)
    tensors: dict[str, dict[str, Any]] = {}
    for shard in shards:
        for name, metadata in read_safetensors_header(shard).items():
            if name in tensors:
                raise ManifestError(f"duplicate tensor across shards: {name}")
            if assignments is not None and assignments.get(name) != shard.name:
                raise ManifestError(f"index/shard disagreement: {name}")
            tensors[name] = metadata
    if assignments is not None and set(assignments) != set(tensors):
        raise ManifestError("index and shard tensor names differ")
    return tensors


@dataclass(frozen=True)
class ComparisonReport:
    schema_version: int
    status: str
    checkpoint_revision: str
    tensor_count: int
    tensor_payload_bytes: int
    compared_fields: tuple[str, ...]
    mismatch_count: int
    mismatches: list[str]


def compare_manifests(
    reference: dict[str, dict[str, Any]], engine_document: dict[str, Any], revision: str = "unknown"
) -> ComparisonReport:
    engine_tensors = engine_document.get("tensors")
    if not isinstance(engine_tensors, list):
        raise ManifestError("gem16 manifest has no tensor array")
    engine: dict[str, dict[str, Any]] = {}
    for tensor in engine_tensors:
        if not isinstance(tensor, dict) or not isinstance(tensor.get("name"), str):
            raise ManifestError("gem16 manifest tensor entry is malformed")
        name = tensor["name"]
        if name in engine:
            raise ManifestError(f"duplicate tensor in gem16 manifest: {name}")
        engine[name] = tensor

    mismatches: list[str] = []
    for missing in sorted(set(reference) - set(engine)):
        mismatches.append(f"missing from gem16 manifest: {missing}")
    for extra in sorted(set(engine) - set(reference)):
        mismatches.append(f"extra in gem16 manifest: {extra}")
    for name in sorted(set(reference) & set(engine)):
        for field in PHYSICAL_FIELDS:
            if reference[name][field] != engine[name].get(field):
                mismatches.append(
                    f"{name}.{field}: reference={reference[name][field]!r} gem16={engine[name].get(field)!r}"
                )
        if reference[name]["alignment"] != engine[name].get("alignment"):
            mismatches.append(
                f"{name}.alignment: reference={reference[name]['alignment']!r} "
                f"gem16={engine[name].get('alignment')!r}"
            )

    payload_bytes = sum(tensor["byte_length"] for tensor in reference.values())
    if engine_document.get("total_tensor_bytes") != payload_bytes:
        mismatches.append(
            f"total_tensor_bytes: reference={payload_bytes} gem16={engine_document.get('total_tensor_bytes')!r}"
        )
    return ComparisonReport(
        schema_version=1,
        status="ok" if not mismatches else "mismatch",
        checkpoint_revision=revision,
        tensor_count=len(reference),
        tensor_payload_bytes=payload_bytes,
        compared_fields=PHYSICAL_FIELDS + ("alignment",),
        mismatch_count=len(mismatches),
        mismatches=mismatches,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=Path("models/gemma4-12b-nvfp4.lock.json"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        reference = build_reference_manifest(args.model)
        engine_document = load_json(args.manifest)
        lock = load_json(args.lock)
        revision = lock.get("revision", "unknown") if isinstance(lock, dict) else "unknown"
        report = compare_manifests(reference, engine_document, str(revision))
    except (ManifestError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    serialized = json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())


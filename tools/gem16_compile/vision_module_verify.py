"""Strict offline verifier for the Gemma 4 26B Vision FP8 module."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import struct
from typing import Any

from .common import DataError, MAX_HEADER_BYTES, reject_duplicate_keys
from .reader import TensorDescriptor
from .vision_module import (
    COMPILATION_FILENAME,
    DESCRIPTOR_FILENAME,
    LOCK_FILENAME,
    OUTPUT_PAYLOAD_BYTES,
    OUTPUT_PADDING_BYTES,
    OUTPUT_TENSOR_BYTES,
    OUTPUT_TENSOR_COUNT,
    PROFILE,
    SOURCE_LOCK_PATH,
    TEXT_ARTIFACT_PROFILE,
    TENSOR_ALIGNMENT,
    VISION_FILENAME,
    _file_sha256,
    expected_vision_specs,
    output_plan,
)


REQUIRED_FILES = frozenset({
    VISION_FILENAME, DESCRIPTOR_FILENAME, COMPILATION_FILENAME, LOCK_FILENAME,
})


def _regular(path: Path, description: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as error:
        raise DataError(f"cannot inspect {description}: {error}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise DataError(f"{description} must be a regular non-symlink file")
    return info


def _json(path: Path, maximum: int = 64 * 1024 * 1024) -> dict[str, Any]:
    info = _regular(path, path.name)
    if info.st_size > maximum:
        raise DataError(f"{path.name} exceeds verifier size limit")
    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DataError(f"cannot parse {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise DataError(f"{path.name} root must be an object")
    return value


def _expected_outputs():
    descriptors: dict[str, TensorDescriptor] = {}
    for name, (_role, shape) in expected_vision_specs().items():
        length = 2
        for dimension in shape:
            length *= dimension
        descriptors[name] = TensorDescriptor(
            name, "BF16", shape, "locked", Path("/locked"), 0, 0, length, "0" * 64
        )
    return output_plan(descriptors)


def _verify_header(path: Path) -> tuple[int, dict[str, tuple[int, int]]]:
    size = _regular(path, VISION_FILENAME).st_size
    try:
        with path.open("rb", buffering=0) as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise DataError("Vision module has a short header prefix")
            (header_length,) = struct.unpack("<Q", prefix)
            if header_length < 2 or header_length > MAX_HEADER_BYTES or 8 + header_length > size:
                raise DataError("Vision module header length is invalid")
            raw = stream.read(header_length)
            if len(raw) != header_length:
                raise DataError("Vision module header is truncated")
    except OSError as error:
        raise DataError(f"cannot read Vision module header: {error}") from error
    try:
        header = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise DataError(f"Vision module header JSON is invalid: {error}") from error
    if not isinstance(header, dict):
        raise DataError("Vision module header root must be an object")
    metadata = header.pop("__metadata__", None)
    if metadata != {
        "format": "pt",
        "gem16_artifact": "gemma4-26b-vision-fp8-v1",
        "gem16_capability_profile": PROFILE,
        "gem16_required_text_profile": TEXT_ARTIFACT_PROFILE,
        "gem16_schema_version": "1",
    }:
        raise DataError("Vision module metadata contract mismatch")
    expected = {item.name: item for item in _expected_outputs()}
    if set(header) != set(expected) or len(header) != OUTPUT_TENSOR_COUNT:
        raise DataError("Vision module tensor-name inventory mismatch")
    cursor = 0
    padding = 0
    offsets: dict[str, tuple[int, int]] = {}
    for name, item in sorted(expected.items()):
        aligned = (cursor + TENSOR_ALIGNMENT - 1) & -TENSOR_ALIGNMENT
        padding += aligned - cursor
        cursor = aligned
        record = header[name]
        if not isinstance(record, dict) or set(record) != {"dtype", "shape", "data_offsets"}:
            raise DataError(f"Vision module tensor header schema mismatch: {name}")
        positions = record["data_offsets"]
        if (
            record["dtype"] != item.dtype
            or record["shape"] != list(item.shape)
            or not isinstance(positions, list)
            or positions != [cursor, cursor + item.byte_length]
        ):
            raise DataError(f"Vision module tensor extent mismatch: {name}")
        cursor += item.byte_length
        offsets[name] = (positions[0], positions[1])
    if (
        cursor != OUTPUT_PAYLOAD_BYTES
        or padding != OUTPUT_PADDING_BYTES
        or sum(item.byte_length for item in expected.values()) != OUTPUT_TENSOR_BYTES
        or size != 8 + header_length + cursor
    ):
        raise DataError("Vision module total extent mismatch")
    return 8 + header_length, offsets


def verify_vision_module(directory: Path) -> dict[str, Any]:
    try:
        original = directory.absolute()
        if original.is_symlink():
            raise DataError("Vision module directory must not be a symlink")
        root = original.resolve(strict=True)
    except OSError as error:
        raise DataError(f"cannot resolve Vision module directory: {error}") from error
    if not root.is_dir():
        raise DataError("Vision module path is not a directory")
    entries = {item.name for item in root.iterdir()}
    if entries != REQUIRED_FILES:
        raise DataError(f"Vision module file set mismatch: {sorted(entries)}")

    artifact = root / VISION_FILENAME
    descriptor_path = root / DESCRIPTOR_FILENAME
    compilation_path = root / COMPILATION_FILENAME
    lock_path = root / LOCK_FILENAME
    payload_offset, offsets = _verify_header(artifact)
    artifact_hash = _file_sha256(artifact)
    descriptor_hash = _file_sha256(descriptor_path)
    compilation_hash = _file_sha256(compilation_path)
    descriptor = _json(descriptor_path, 1024 * 1024)
    compilation = _json(compilation_path)
    lock = _json(lock_path, 1024 * 1024)

    if descriptor != {
        "artifact": VISION_FILENAME,
        "artifact_sha256": artifact_hash,
        "artifact_size": artifact.stat().st_size,
        "capability_profile": PROFILE,
        "compilation_manifest": COMPILATION_FILENAME,
        "compilation_manifest_sha256": compilation_hash,
        "enablement": "explicit-profile-selection-only",
        "required_text_artifact_profile": TEXT_ARTIFACT_PROFILE,
        "schema_version": 1,
        "supports": {"audio": False, "image": True, "text": True, "video": False},
    }:
        raise DataError("Vision runtime descriptor contract mismatch")
    expected_lock = {
        "artifact_sha256": artifact_hash,
        "artifact_size": artifact.stat().st_size,
        "capability_profile": PROFILE,
        "compilation_manifest_sha256": compilation_hash,
        "descriptor_sha256": descriptor_hash,
        "required_text_artifact_profile": TEXT_ARTIFACT_PROFILE,
        "schema_version": 1,
        "source_lock_sha256": _file_sha256(SOURCE_LOCK_PATH),
        "source_repository": "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized",
        "source_revision": "f1e06dc520982d9b9edd76859fdb7ab209449949",
    }
    if lock != expected_lock:
        raise DataError("Vision immutable lock contract mismatch")

    artifact_record = compilation.get("artifact")
    if artifact_record != {
        "filename": VISION_FILENAME,
        "format": "gem16-aligned-vision-module-v1",
        "payload_bytes": OUTPUT_PAYLOAD_BYTES,
        "payload_offset": payload_offset,
        "sha256": artifact_hash,
        "size": artifact.stat().st_size,
        "tensor_count": OUTPUT_TENSOR_COUNT,
    }:
        raise DataError("Vision compilation artifact record mismatch")
    if (
        compilation.get("schema_version") != 1
        or compilation.get("contract_id") != "gem16.gemma4_26b_vision_fp8"
        or compilation.get("contract_version") != 1
        or compilation.get("capability_profile") != PROFILE
        or compilation.get("required_text_artifact_profile") != TEXT_ARTIFACT_PROFILE
    ):
        raise DataError("Vision compilation profile contract mismatch")
    source = compilation.get("source")
    if not isinstance(source, dict) or any(source.get(key) != value for key, value in {
        "lock_sha256": expected_lock["source_lock_sha256"],
        "repository": expected_lock["source_repository"],
        "revision": expected_lock["source_revision"],
        "tensor_count": 356,
        "tensor_payload_bytes": 1_145_588_832,
    }.items()):
        raise DataError("Vision compilation source record mismatch")
    tensors = compilation.get("tensors")
    if not isinstance(tensors, list) or len(tensors) != OUTPUT_TENSOR_COUNT:
        raise DataError("Vision compilation tensor inventory is incomplete")
    expected_names = set(offsets)
    names: set[str] = set()
    for record in tensors:
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            raise DataError("Vision compilation tensor record is malformed")
        name = record["name"]
        if name in names or name not in expected_names:
            raise DataError(f"Vision compilation tensor identity mismatch: {name}")
        names.add(name)
        if record.get("data_offsets") != list(offsets[name]):
            raise DataError(f"Vision compilation tensor offset mismatch: {name}")
        for key in ("output_sha256", "source_sha256"):
            value = record.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise DataError(f"Vision compilation tensor hash is invalid: {name}:{key}")
    if names != expected_names:
        raise DataError("Vision compilation tensor names are incomplete")
    return {
        "artifact_sha256": artifact_hash,
        "artifact_size": artifact.stat().st_size,
        "capability_profile": PROFILE,
        "payload_bytes": OUTPUT_PAYLOAD_BYTES,
        "status": "pass",
        "tensor_count": OUTPUT_TENSOR_COUNT,
    }


__all__ = ["verify_vision_module"]

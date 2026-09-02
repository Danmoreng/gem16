#!/usr/bin/env python3
"""Build the single-file Trellis35 SM120 device image v2.

This is an offline packager.  It fully verifies the immutable v1 source,
streams the already GPU-ready bytes into their final arena order, verifies the
new payload with an independent second pass, and only then publishes it.  The
runtime loader intentionally does not repeat the 12.2 GB payload hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
from typing import Any, BinaryIO, Iterable

try:
    from tools.gem16_compile.common import (
        InvalidPlanError,
        canonical_json_bytes,
        load_json,
        write_all,
    )
except ModuleNotFoundError:  # Direct execution with tools/ as sys.path[0].
    from gem16_compile.common import (  # type: ignore[no-redef]
        InvalidPlanError,
        canonical_json_bytes,
        load_json,
        write_all,
    )


FORMAT = "gem16-sm120-trellis35-device-image-v2"
PROFILE = "gem16-trellis35-w4a8-v1"
SOURCE_LOCK = "3d9441fdebef33785e33181c335338208b8bf868cb8c7da692fd9c765cca8230"
NON_ROUTED_BYTES = 1_850_270_720
LAYER_BYTES = 345_147_392
LAYERS = 30
IMAGE_BYTES = NON_ROUTED_BYTES + LAYERS * LAYER_BYTES
ALIGNMENT = 256
DEFAULT_CHUNK_BYTES = 64 * 1024 * 1024
SHA256_LINE = re.compile(r"^([0-9a-f]{64})  ([^\\\0]+)$")
RUNTIME_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "README.md",
    "LICENSE",
    "NOTICE",
)
V1_TOP_LEVEL_FILES = {
    "SHA256SUMS",
    "trellis35-checkpoint.json",
    "trellis35-experts.json",
    "non-routed.gem16",
    "non-routed.json",
    *RUNTIME_FILES,
}

EXPECTED_REGIONS = {
    "gate_up": {
        "k3_payload_pool": (0, 95_158_272),
        "k4_payload_pool": (95_158_272, 126_877_696),
        "descriptor": (222_035_968, 1_024),
        "suh": (222_036_992, 720_896),
        "svh": (222_757_888, 360_448),
    },
    "down": {
        "k3_payload_pool": (223_118_336, 51_904_512),
        "k4_payload_pool": (275_022_848, 69_206_016),
        "descriptor": (344_228_864, 1_024),
        "suh": (344_229_888, 196_608),
        "svh": (344_426_496, 720_896),
    },
}
ELEMENT_BYTES = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1}
LAYOUT_DTYPE = {
    "row_bf16": "BF16",
    "scalar_bf16": "BF16",
    "scalar_f32": "F32",
    "sm120_row8_group16_e4m3": "F8_E4M3",
    "sm120_row8_k64": "U8",
    "source_bf16": "BF16",
    "source_nk_fp8": "F8_E4M3",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    return parser.parse_args()


def sha256_file(path: Path, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def regular_file(path: Path, description: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise InvalidPlanError(f"cannot inspect {description} {path}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise InvalidPlanError(
            f"{description} must be a regular non-symlink file: {path}"
        )
    return metadata


def real_directory(path: Path, description: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise InvalidPlanError(f"cannot inspect {description} {path}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise InvalidPlanError(f"{description} must be a real directory: {path}")
    return path.resolve(strict=True)


def safe_relative(value: object, description: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise InvalidPlanError(f"unsafe {description}: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise InvalidPlanError(f"unsafe {description}: {value!r}")
    return path


def safe_file(root: Path, value: object, description: str) -> Path:
    relative = safe_relative(value, description)
    path = root.joinpath(*relative.parts)
    regular_file(path, description)
    return path


def nonnegative_integer(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidPlanError(f"invalid nonnegative integer for {description}")
    return value


def checked_shape(value: object, description: str) -> list[int]:
    if not isinstance(value, list) or len(value) not in (1, 2):
        raise InvalidPlanError(f"invalid {description}")
    shape: list[int] = []
    product = 1
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise InvalidPlanError(f"invalid {description}")
        if product > IMAGE_BYTES // dimension:
            raise InvalidPlanError(f"overflow in {description}")
        product *= dimension
        shape.append(dimension)
    return shape


def validate_tensor_layout(
    dtype: object,
    runtime_layout: object,
    logical_shape: list[int],
    physical_shape: list[int],
    bytes_: int,
) -> None:
    if (
        not isinstance(dtype, str)
        or not isinstance(runtime_layout, str)
        or LAYOUT_DTYPE.get(runtime_layout) != dtype
        or dtype not in ELEMENT_BYTES
    ):
        raise InvalidPlanError("unsupported v1 non-routed tensor dtype/layout")
    physical_bytes = ELEMENT_BYTES[dtype]
    for dimension in physical_shape:
        physical_bytes *= dimension
    if physical_bytes != bytes_:
        raise InvalidPlanError("v1 non-routed physical shape disagrees with bytes")
    if runtime_layout in ("source_bf16", "row_bf16", "source_nk_fp8"):
        valid = physical_shape == logical_shape
    elif runtime_layout in ("scalar_bf16", "scalar_f32"):
        valid = physical_shape == [1]
    else:
        divisor = 16 if runtime_layout == "sm120_row8_group16_e4m3" else 2
        valid = (
            len(logical_shape) == 2
            and logical_shape[1] % divisor == 0
            and physical_shape == [logical_shape[0], logical_shape[1] // divisor]
        )
    if not valid:
        raise InvalidPlanError("v1 non-routed tensor layout shape is invalid")


def canonical_content_sha256(document: dict[str, Any], field: str) -> str:
    recorded = document.get(field)
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise InvalidPlanError(f"missing {field}")
    content = dict(document)
    del content[field]
    actual = hashlib.sha256(canonical_json_bytes(content)).hexdigest()
    if actual != recorded:
        raise InvalidPlanError(f"{field} mismatch")
    return recorded


def parse_sha256s(source: Path, expected_entries: int = 99) -> dict[str, str]:
    sums_path = source / "SHA256SUMS"
    regular_file(sums_path, "v1 SHA256SUMS")
    try:
        lines = sums_path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise InvalidPlanError(f"cannot read v1 SHA256SUMS: {error}") from error
    if len(lines) != expected_entries:
        raise InvalidPlanError(
            f"v1 SHA256SUMS entry count changed: {len(lines)} != {expected_entries}"
        )
    records: dict[str, str] = {}
    for line in lines:
        match = SHA256_LINE.fullmatch(line)
        if match is None:
            raise InvalidPlanError("invalid v1 SHA256SUMS line")
        digest, text = match.groups()
        relative = safe_relative(text, "v1 SHA256SUMS path")
        normalized = relative.as_posix()
        if normalized in records:
            raise InvalidPlanError(f"duplicate v1 SHA256SUMS path: {normalized}")
        records[normalized] = digest
    return records


def verify_sha256_inventory(source: Path, records: dict[str, str]) -> None:
    expected_files = set(records)
    expected_files.update({"SHA256SUMS", "README.md", "LICENSE", "NOTICE"})
    actual_files: set[str] = set()
    for path in source.rglob("*"):
        relative = path.relative_to(source).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise InvalidPlanError(f"v1 source contains a symlink: {relative}")
        if stat.S_ISREG(metadata.st_mode):
            actual_files.add(relative)
        elif not stat.S_ISDIR(metadata.st_mode):
            raise InvalidPlanError(f"v1 source contains an unsafe entry: {relative}")
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        raise InvalidPlanError(
            f"v1 file inventory changed: missing={missing} extra={extra}"
        )
    for index, (relative, expected) in enumerate(records.items(), 1):
        path = safe_file(source, relative, "v1 locked file")
        if sha256_file(path) != expected:
            raise InvalidPlanError(f"v1 SHA-256 mismatch: {relative}")
        if index % 10 == 0 or index == len(records):
            print(f"trellis35_v1_verify {index}/{len(records)}", flush=True)


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) & -ALIGNMENT


def compact_non_routed(document: dict[str, Any]) -> list[dict[str, Any]]:
    tensors = document.get("tensors")
    if (
        document.get("checkpoint_profile") != PROFILE
        or document.get("bytes") != NON_ROUTED_BYTES
        or document.get("tensor_count") != 1045
        or not isinstance(tensors, list)
        or len(tensors) != 1045
    ):
        raise InvalidPlanError("v1 non-routed manifest contract changed")
    compact: list[dict[str, Any]] = []
    names: set[str] = set()
    cursor = 0
    previous_name = ""
    for record in tensors:
        if not isinstance(record, dict):
            raise InvalidPlanError("v1 non-routed tensor record is invalid")
        name = record.get("name")
        offset = nonnegative_integer(record.get("destination_offset"), "tensor offset")
        bytes_ = nonnegative_integer(record.get("bytes"), "tensor bytes")
        dtype = record.get("storage_dtype")
        runtime_layout = record.get("runtime_layout")
        logical_shape = checked_shape(record.get("logical_shape"), "logical shape")
        physical_shape = checked_shape(record.get("physical_shape"), "physical shape")
        validate_tensor_layout(
            dtype, runtime_layout, logical_shape, physical_shape, bytes_
        )
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or name <= previous_name
            or not isinstance(dtype, str)
            or not dtype
            or bytes_ == 0
            or offset != align(cursor)
            or offset > NON_ROUTED_BYTES
            or bytes_ > NON_ROUTED_BYTES - offset
        ):
            raise InvalidPlanError(f"invalid v1 non-routed tensor plan: {name!r}")
        names.add(name)
        previous_name = name
        cursor = offset + bytes_
        compact.append(
            {
                "name": name,
                "offset": offset,
                "bytes": bytes_,
                "storage_dtype": dtype,
                "runtime_layout": runtime_layout,
                "physical_shape": physical_shape,
                "logical_shape": logical_shape,
            }
        )
    if align(cursor) != NON_ROUTED_BYTES:
        raise InvalidPlanError("v1 non-routed tensor extent is incomplete")
    return compact


def compact_family(document: dict[str, Any], family: str) -> dict[str, Any]:
    rate_maps = document.get("rate_maps")
    regions = document.get("regions")
    if not isinstance(rate_maps, dict) or not isinstance(regions, dict):
        raise InvalidPlanError("v1 layer rate-map or region table is missing")
    rate_map = rate_maps.get(family)
    if (
        not isinstance(rate_map, list)
        or len(rate_map) != 128
        or any(isinstance(rate, bool) or rate not in (3, 4) for rate in rate_map)
        or rate_map.count(3) != 64
        or rate_map.count(4) != 64
    ):
        raise InvalidPlanError(f"v1 {family} rate map is invalid")
    compact_regions: dict[str, dict[str, int]] = {}
    for name, (expected_offset, expected_bytes) in EXPECTED_REGIONS[family].items():
        region = regions.get(f"{family}_{name}")
        if not isinstance(region, dict):
            raise InvalidPlanError(f"v1 {family} region is missing: {name}")
        offset = nonnegative_integer(region.get("offset"), f"{family} {name} offset")
        bytes_ = nonnegative_integer(region.get("bytes"), f"{family} {name} bytes")
        if (offset, bytes_) != (expected_offset, expected_bytes):
            raise InvalidPlanError(f"v1 {family} region changed: {name}")
        compact_regions[name] = {"offset": offset, "bytes": bytes_}
    return {"rate_map": rate_map, "regions": compact_regions}


def validate_v1(source: Path, *, verify_inventory: bool = True) -> dict[str, Any]:
    records = parse_sha256s(source)
    if verify_inventory:
        verify_sha256_inventory(source, records)
    checkpoint = load_json(source / "trellis35-checkpoint.json")
    experts = load_json(source / "trellis35-experts.json")
    non_routed = load_json(source / "non-routed.json")
    source_identity = canonical_content_sha256(
        checkpoint, "checkpoint_content_sha256"
    )
    canonical_content_sha256(experts, "checkpoint_content_sha256")
    if (
        checkpoint.get("checkpoint_profile") != PROFILE
        or checkpoint.get("format") != "GEM16-Trellis35"
        or checkpoint.get("format_version") != 1
        or checkpoint.get("source_lock_sha256") != SOURCE_LOCK
        or checkpoint.get("arena", {}).get("total_bytes") != IMAGE_BYTES
        or checkpoint.get("arena", {}).get("non_routed_bytes") != NON_ROUTED_BYTES
        or checkpoint.get("arena", {}).get("trellis35_routed_expert_bytes")
        != LAYERS * LAYER_BYTES
        or experts.get("checkpoint_profile") != PROFILE
        or experts.get("layer_count") != LAYERS
    ):
        raise InvalidPlanError("v1 Trellis35 checkpoint identity changed")
    non_routed_path = safe_file(
        source, checkpoint.get("non_routed", {}).get("artifact"), "v1 non-routed payload"
    )
    if regular_file(non_routed_path, "v1 non-routed payload").st_size != NON_ROUTED_BYTES:
        raise InvalidPlanError("v1 non-routed payload size changed")
    tensors = compact_non_routed(non_routed)
    checkpoint_layers = checkpoint.get("routed_experts", {}).get("layers")
    expert_layers = experts.get("layers")
    if (
        not isinstance(checkpoint_layers, list)
        or not isinstance(expert_layers, list)
        or len(checkpoint_layers) != LAYERS
        or checkpoint_layers != expert_layers
    ):
        raise InvalidPlanError("v1 layer index changed or disagrees")
    layers: list[dict[str, Any]] = []
    payloads: list[Path] = [non_routed_path]
    for layer, record in enumerate(checkpoint_layers):
        if not isinstance(record, dict) or record.get("layer") != layer:
            raise InvalidPlanError("v1 layer ordering changed")
        artifact = safe_file(source, record.get("artifact"), f"v1 layer {layer} payload")
        manifest_path = safe_file(
            source, record.get("manifest"), f"v1 layer {layer} manifest"
        )
        safe_file(source, record.get("verification"), f"v1 layer {layer} verification")
        if regular_file(artifact, f"v1 layer {layer} payload").st_size != LAYER_BYTES:
            raise InvalidPlanError(f"v1 layer {layer} payload size changed")
        manifest = load_json(manifest_path)
        if (
            manifest.get("schema_version") != 1
            or manifest.get("format_version") != 1
            or manifest.get("checkpoint_profile") != PROFILE
            or manifest.get("source_lock_sha256") != SOURCE_LOCK
            or manifest.get("layer") != layer
            or manifest.get("alignment_bytes") != ALIGNMENT
            or manifest.get("codebook_id") != 2
            or manifest.get("artifact", {}).get("bytes") != LAYER_BYTES
            or manifest.get("artifact", {}).get("sha256") != record.get("artifact_sha256")
        ):
            raise InvalidPlanError(f"v1 layer {layer} manifest contract changed")
        layers.append(
            {
                "layer": layer,
                "arena_offset": NON_ROUTED_BYTES + layer * LAYER_BYTES,
                "gate_up": compact_family(manifest, "gate_up"),
                "down": compact_family(manifest, "down"),
            }
        )
        payloads.append(artifact)
    return {
        "source_checkpoint_content_sha256": source_identity,
        "source_records": records,
        "payloads": payloads,
        "non_routed_tensors": tensors,
        "layers": layers,
    }


def preallocate(stream: BinaryIO, bytes_: int) -> None:
    descriptor = stream.fileno()
    if hasattr(os, "posix_fallocate"):
        try:
            os.posix_fallocate(descriptor, 0, bytes_)
            return
        except OSError:
            pass
    stream.truncate(bytes_)


def stream_concat(
    payloads: Iterable[Path], partial: Path, total_bytes: int,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> str:
    if chunk_bytes <= 0:
        raise InvalidPlanError("chunk bytes must be positive")
    if partial.exists() or partial.is_symlink():
        raise InvalidPlanError(f"refusing to overwrite partial output: {partial}")
    digest = hashlib.sha256()
    written = 0
    with partial.open("xb", buffering=0) as target:
        preallocate(target, total_bytes)
        target.seek(0)
        for index, path in enumerate(payloads):
            with path.open("rb", buffering=0) as source:
                while chunk := source.read(chunk_bytes):
                    if written > total_bytes or len(chunk) > total_bytes - written:
                        raise InvalidPlanError("v1 payloads exceed the v2 image extent")
                    write_all(target, chunk, f"Trellis35 v2 payload {index}")
                    digest.update(chunk)
                    written += len(chunk)
        if written != total_bytes:
            raise InvalidPlanError(
                f"v2 image extent is incomplete: {written} != {total_bytes}"
            )
        target.flush()
        os.fsync(target.fileno())
    first = digest.hexdigest()
    second = sha256_file(partial, chunk_bytes)
    if first != second:
        raise InvalidPlanError("independent v2 image SHA-256 verification failed")
    return second


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_atomic(path: Path, payload: bytes) -> None:
    partial = path.with_name(path.name + ".partial")
    if path.exists() or path.is_symlink() or partial.exists() or partial.is_symlink():
        raise InvalidPlanError(f"refusing to overwrite v2 metadata: {path}")
    with partial.open("xb", buffering=0) as stream:
        write_all(stream, payload, f"v2 metadata {path.name}")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def copy_atomic(source: Path, destination: Path) -> None:
    regular_file(source, f"v1 runtime file {source.name}")
    partial = destination.with_name(destination.name + ".partial")
    if destination.exists() or destination.is_symlink() or partial.exists() or partial.is_symlink():
        raise InvalidPlanError(f"refusing to overwrite runtime file: {destination}")
    with source.open("rb", buffering=0) as input_, partial.open("xb", buffering=0) as output:
        shutil.copyfileobj(input_, output, DEFAULT_CHUNK_BYTES)
        output.flush()
        os.fsync(output.fileno())
    os.replace(partial, destination)


def add_content_hash(document: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(document)
    result[field] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def build(source_path: Path, output_path: Path, chunk_bytes: int) -> dict[str, Any]:
    source = real_directory(source_path, "v1 source")
    if output_path.exists():
        output = real_directory(output_path, "v2 output")
        if any(output.iterdir()):
            raise InvalidPlanError("v2 output directory must be empty")
    else:
        output_path.mkdir(parents=True, exist_ok=False)
        output = output_path.resolve(strict=True)
    if output == source or source in output.parents or output in source.parents:
        raise InvalidPlanError("v1 source and v2 output must be disjoint")
    validated = validate_v1(source)
    image = output / "model.gem16"
    partial = output / "model.gem16.partial"
    image_sha256 = stream_concat(
        validated["payloads"], partial, IMAGE_BYTES, chunk_bytes
    )
    if regular_file(partial, "verified v2 partial image").st_size != IMAGE_BYTES:
        raise InvalidPlanError("verified v2 image size changed before publish")
    os.replace(partial, image)
    fsync_directory(output)

    compilation = add_content_hash(
        {
            "schema_version": 1,
            "format": FORMAT,
            "format_version": 2,
            "artifact_profile": PROFILE,
            "source_checkpoint_content_sha256": validated[
                "source_checkpoint_content_sha256"
            ],
            "arena_bytes": IMAGE_BYTES,
            "non_routed_bytes": NON_ROUTED_BYTES,
            "layer_count": LAYERS,
            "layer_bytes": LAYER_BYTES,
            "non_routed_tensors": validated["non_routed_tensors"],
            "layers": validated["layers"],
        },
        "compilation_content_sha256",
    )
    model = add_content_hash(
        {
            "schema_version": 1,
            "format": FORMAT,
            "format_version": 2,
            "artifact_profile": PROFILE,
            "qualification_state": "production_candidate",
            "runtime_supported": True,
            "runtime_payload_sha256_policy": "verified_at_build_download_install_not_runtime_load",
            "source_lock_sha256": SOURCE_LOCK,
            "source_checkpoint_content_sha256": validated[
                "source_checkpoint_content_sha256"
            ],
            "model_file": "model.gem16",
            "model_bytes": IMAGE_BYTES,
            "model_sha256": image_sha256,
            "arena": {
                "alignment_bytes": ALIGNMENT,
                "non_routed_offset": 0,
                "non_routed_bytes": NON_ROUTED_BYTES,
                "layer_base_offset": NON_ROUTED_BYTES,
                "layer_count": LAYERS,
                "layer_stride_bytes": LAYER_BYTES,
                "total_bytes": IMAGE_BYTES,
            },
        },
        "model_content_sha256",
    )
    compilation_payload = canonical_json_bytes(compilation)
    model_payload = canonical_json_bytes(model)
    write_atomic(output / "gem16_compilation.json", compilation_payload)
    write_atomic(output / "gem16_model.json", model_payload)
    lock = add_content_hash(
        {
            "schema_version": 1,
            "format": FORMAT,
            "format_version": 2,
            "artifact_profile": PROFILE,
            "source_lock_sha256": SOURCE_LOCK,
            "source_checkpoint_content_sha256": validated[
                "source_checkpoint_content_sha256"
            ],
            "runtime_payload_sha256_policy": "verified_at_build_download_install_not_runtime_load",
            "files": [
                {"path": "model.gem16", "size": IMAGE_BYTES, "sha256": image_sha256},
                {
                    "path": "gem16_model.json",
                    "size": len(model_payload),
                    "sha256": hashlib.sha256(model_payload).hexdigest(),
                },
                {
                    "path": "gem16_compilation.json",
                    "size": len(compilation_payload),
                    "sha256": hashlib.sha256(compilation_payload).hexdigest(),
                },
            ],
        },
        "artifact_content_sha256",
    )
    write_atomic(output / "gem16.lock.json", canonical_json_bytes(lock))
    for name in RUNTIME_FILES:
        copy_atomic(source / name, output / name)
    fsync_directory(output)
    return {
        "format": FORMAT,
        "model_bytes": IMAGE_BYTES,
        "model_sha256": image_sha256,
        "source_checkpoint_content_sha256": validated[
            "source_checkpoint_content_sha256"
        ],
        "output": str(output),
    }


def main() -> int:
    args = arguments()
    result = build(args.source, args.output, args.chunk_bytes)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (InvalidPlanError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"trellis35_device_image_error: {error}", file=sys.stderr)
        raise SystemExit(getattr(error, "exit_code", 1))

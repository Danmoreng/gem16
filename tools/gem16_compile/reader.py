"""Strict source-lock and bounded Safetensors readers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import struct
from typing import Any

try:
    from tools.fetch_model import validate_lock
except ModuleNotFoundError:  # Direct tool execution outside repository root.
    from fetch_model import validate_lock

from .common import (
    BoundedWorkspace,
    InvalidPlanError,
    MAX_HEADER_BYTES,
    SourceVerificationError,
    checked_shape,
    load_json,
    reject_duplicate_keys,
    safe_relative_path,
    sha256_bytes,
    tensor_bytes,
)


MAX_INDEX_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class LockedFile:
    relative_path: str
    path: Path
    size: int
    sha256: str


@dataclass(frozen=True)
class VerifiedSource:
    root: Path
    lock_path: Path
    lock_sha256: str
    repository: str
    revision: str
    resolved_at_utc: str
    files: dict[str, LockedFile]


@dataclass(frozen=True)
class TensorDescriptor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    shard: str
    path: Path
    absolute_offset: int
    data_offset: int
    byte_length: int
    shard_sha256: str


def _source_path(root: Path, relative: str) -> Path:
    parsed = safe_relative_path(relative, "source-lock path")
    candidate = root.joinpath(*parsed.parts)
    current = root
    for part in parsed.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise SourceVerificationError(
                f"source-lock path traverses an intermediate symlink: {relative!r}"
            )
    return candidate


def verify_source_lock(
    lock_path: Path, source_root: Path, workspace: BoundedWorkspace
) -> VerifiedSource:
    try:
        root = source_root.resolve(strict=True)
    except OSError as error:
        raise SourceVerificationError(
            f"cannot resolve source directory {source_root}: {error}"
        ) from error
    if not root.is_dir():
        raise SourceVerificationError(f"source path is not a directory: {root}")

    try:
        lock = load_json(lock_path, 4 * 1024 * 1024)
        entries = validate_lock(lock)
    except (InvalidPlanError, ValueError) as error:
        raise SourceVerificationError(f"invalid source lock {lock_path}: {error}") from error

    lock_size = lock_path.stat().st_size
    lock_hash = workspace.hash_range(lock_path, 0, lock_size)
    files: dict[str, LockedFile] = {}
    for entry in entries:
        relative = str(entry["path"])
        path = _source_path(root, relative)
        try:
            if not path.is_file():
                raise SourceVerificationError(
                    f"locked source file is missing: {relative}"
                )
            size = path.stat().st_size
        except OSError as error:
            raise SourceVerificationError(
                f"cannot inspect locked source file {relative}: {error}"
            ) from error
        expected_size = int(entry["size"])
        if size != expected_size:
            raise SourceVerificationError(
                f"locked source size mismatch for {relative}: "
                f"expected {expected_size}, got {size}"
            )
        actual_hash = workspace.hash_range(path, 0, size)
        expected_hash = str(entry["sha256"]).lower()
        if actual_hash != expected_hash:
            raise SourceVerificationError(
                f"locked source SHA-256 mismatch for {relative}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        files[relative] = LockedFile(
            relative_path=relative,
            path=path,
            size=size,
            sha256=actual_hash,
        )
    workspace.check("source-lock verification")
    return VerifiedSource(
        root=root,
        lock_path=lock_path,
        lock_sha256=lock_hash,
        repository=str(lock["repository"]),
        revision=str(lock["revision"]),
        resolved_at_utc=str(lock.get("resolved_at_utc", "")),
        files=files,
    )


def _read_header(file: LockedFile, workspace: BoundedWorkspace) -> dict[str, Any]:
    try:
        with file.path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise SourceVerificationError(
                    f"short Safetensors length prefix: {file.relative_path}"
                )
            (header_length,) = struct.unpack("<Q", prefix)
            if (
                header_length < 2
                or header_length > MAX_HEADER_BYTES
                or header_length > file.size - 8
            ):
                raise SourceVerificationError(
                    f"invalid Safetensors header length: {file.relative_path}"
                )
            if header_length >= workspace.host_memory_cap_bytes:
                raise SourceVerificationError(
                    f"Safetensors header exceeds host-memory cap: {file.relative_path}"
                )
            raw = stream.read(header_length)
            if len(raw) != header_length:
                raise SourceVerificationError(
                    f"short Safetensors header: {file.relative_path}"
                )
    except SourceVerificationError:
        raise
    except OSError as error:
        raise SourceVerificationError(
            f"cannot read Safetensors header {file.relative_path}: {error}"
        ) from error
    try:
        header = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError, InvalidPlanError) as error:
        raise SourceVerificationError(
            f"invalid Safetensors header JSON in {file.relative_path}: {error}"
        ) from error
    if not isinstance(header, dict):
        raise SourceVerificationError(
            f"Safetensors header root is not an object: {file.relative_path}"
        )
    workspace.record_header(header_length, f"parsing {file.relative_path} header")
    return header


def _parse_shard(
    file: LockedFile, workspace: BoundedWorkspace
) -> dict[str, TensorDescriptor]:
    header = _read_header(file, workspace)
    with file.path.open("rb") as stream:
        prefix = stream.read(8)
    (header_length,) = struct.unpack("<Q", prefix)
    data_base = 8 + header_length
    payload_size = file.size - data_base
    result: dict[str, TensorDescriptor] = {}
    intervals: list[tuple[int, int, str]] = []
    for name, metadata in header.items():
        if name == "__metadata__":
            if not isinstance(metadata, dict):
                raise SourceVerificationError(
                    f"Safetensors __metadata__ is not an object: {file.relative_path}"
                )
            continue
        if not isinstance(name, str) or not name or "\x00" in name:
            raise SourceVerificationError(
                f"invalid Safetensors tensor name in {file.relative_path}"
            )
        if not isinstance(metadata, dict):
            raise SourceVerificationError(f"tensor metadata is not an object: {name}")
        dtype = metadata.get("dtype")
        offsets = metadata.get("data_offsets")
        try:
            shape = checked_shape(metadata.get("shape"), f"shape for {name}")
        except InvalidPlanError as error:
            raise SourceVerificationError(str(error)) from error
        if not isinstance(dtype, str):
            raise SourceVerificationError(f"tensor dtype is not a string: {name}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in offsets)
        ):
            raise SourceVerificationError(f"invalid tensor data_offsets: {name}")
        begin, end = offsets
        if begin < 0 or end < begin or end > payload_size:
            raise SourceVerificationError(f"out-of-bounds tensor data_offsets: {name}")
        try:
            expected = tensor_bytes(dtype, shape, name)
        except InvalidPlanError as error:
            raise SourceVerificationError(str(error)) from error
        if end - begin != expected:
            raise SourceVerificationError(
                f"tensor dtype/shape byte mismatch for {name}: "
                f"expected {expected}, got {end - begin}"
            )
        if name in result:
            raise SourceVerificationError(f"duplicate tensor in shard: {name}")
        result[name] = TensorDescriptor(
            name=name,
            dtype=dtype,
            shape=shape,
            shard=file.relative_path,
            path=file.path,
            absolute_offset=data_base + begin,
            data_offset=begin,
            byte_length=end - begin,
            shard_sha256=file.sha256,
        )
        if end > begin:
            intervals.append((begin, end, name))
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] < previous[1]:
            raise SourceVerificationError(
                f"overlapping tensors {previous[2]} and {current[2]} "
                f"in {file.relative_path}"
            )
    workspace.check(f"parsing {file.relative_path} tensors")
    return result


def _load_index(file: LockedFile, workspace: BoundedWorkspace) -> dict[str, str]:
    if file.size > MAX_INDEX_BYTES or file.size >= workspace.host_memory_cap_bytes:
        raise SourceVerificationError(f"Safetensors index exceeds safety cap: {file.path}")
    try:
        raw = file.path.read_bytes()
        index = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError, InvalidPlanError) as error:
        raise SourceVerificationError(f"cannot parse Safetensors index: {error}") from error
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        raise SourceVerificationError("Safetensors index lacks an object weight_map")
    assignments: dict[str, str] = {}
    for tensor, shard in weight_map.items():
        if not isinstance(tensor, str) or not isinstance(shard, str):
            raise SourceVerificationError("Safetensors index must map strings to strings")
        path = PurePosixPath(shard)
        if path.is_absolute() or len(path.parts) != 1 or path.suffix != ".safetensors":
            raise SourceVerificationError(f"unsafe Safetensors shard path: {shard}")
        assignments[tensor] = shard
    workspace.record_header(file.size, "parsing Safetensors index")
    return assignments


def read_source_tensors(
    source: VerifiedSource, workspace: BoundedWorkspace
) -> dict[str, TensorDescriptor]:
    shard_files = {
        relative: file
        for relative, file in source.files.items()
        if PurePosixPath(relative).suffix == ".safetensors"
    }
    if not shard_files:
        raise SourceVerificationError("source lock contains no Safetensors files")
    index_file = source.files.get("model.safetensors.index.json")
    assignments: dict[str, str] | None = None
    if index_file is not None:
        assignments = _load_index(index_file, workspace)
        indexed_shards = set(assignments.values())
        if indexed_shards != set(shard_files):
            raise SourceVerificationError(
                "source lock Safetensors files disagree with index shard set"
            )
    elif len(shard_files) != 1 or "model.safetensors" not in shard_files:
        raise SourceVerificationError(
            "multi-shard source requires model.safetensors.index.json"
        )

    tensors: dict[str, TensorDescriptor] = {}
    for relative in sorted(shard_files):
        for name, descriptor in _parse_shard(
            shard_files[relative], workspace
        ).items():
            if name in tensors:
                raise SourceVerificationError(f"duplicate tensor across shards: {name}")
            if assignments is not None and assignments.get(name) != relative:
                raise SourceVerificationError(f"index/shard disagreement for tensor: {name}")
            tensors[name] = descriptor
    if assignments is not None and set(assignments) != set(tensors):
        raise SourceVerificationError("Safetensors index and shard names differ")
    workspace.check("source tensor discovery")
    return dict(sorted(tensors.items()))


def read_artifact_tensors(
    artifact_root: Path,
    shard_names: tuple[str, ...],
    index_name: str,
    workspace: BoundedWorkspace,
) -> dict[str, TensorDescriptor]:
    try:
        root = artifact_root.resolve(strict=True)
    except OSError as error:
        raise SourceVerificationError(
            f"cannot resolve artifact directory {artifact_root}: {error}"
        ) from error
    if not root.is_dir():
        raise SourceVerificationError(f"artifact path is not a directory: {root}")
    locked_shards: dict[str, LockedFile] = {}
    for name in shard_names:
        parsed = safe_relative_path(name, "artifact shard")
        if len(parsed.parts) != 1 or parsed.suffix != ".safetensors":
            raise SourceVerificationError(f"unsafe artifact shard name: {name}")
        path = root / name
        try:
            if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
                raise SourceVerificationError(f"unsafe artifact shard file: {name}")
            size = path.stat().st_size
        except OSError as error:
            raise SourceVerificationError(
                f"cannot inspect artifact shard {name}: {error}"
            ) from error
        locked_shards[name] = LockedFile(
            relative_path=name,
            path=path,
            size=size,
            sha256=workspace.hash_range(path, 0, size),
        )
    index_path = root / index_name
    try:
        if index_path.is_symlink() or not index_path.is_file():
            raise SourceVerificationError(f"artifact index is missing: {index_name}")
        index_size = index_path.stat().st_size
    except OSError as error:
        raise SourceVerificationError(f"cannot inspect artifact index: {error}") from error
    index_file = LockedFile(
        relative_path=index_name,
        path=index_path,
        size=index_size,
        sha256=workspace.hash_range(index_path, 0, index_size),
    )
    assignments = _load_index(index_file, workspace)
    if set(assignments.values()) != set(shard_names):
        raise SourceVerificationError("artifact index and declared shard set differ")
    tensors: dict[str, TensorDescriptor] = {}
    for name in sorted(shard_names):
        for tensor_name, descriptor in _parse_shard(
            locked_shards[name], workspace
        ).items():
            if tensor_name in tensors:
                raise SourceVerificationError(
                    f"duplicate artifact tensor: {tensor_name}"
                )
            if assignments.get(tensor_name) != name:
                raise SourceVerificationError(
                    f"artifact index/shard disagreement: {tensor_name}"
                )
            tensors[tensor_name] = descriptor
    if set(assignments) != set(tensors):
        raise SourceVerificationError("artifact index and tensor names differ")
    return dict(sorted(tensors.items()))


def tensor_source_identity(tensor: TensorDescriptor) -> dict[str, object]:
    return {
        "shard": tensor.shard,
        "shard_sha256": tensor.shard_sha256,
        "absolute_offset": tensor.absolute_offset,
        "byte_length": tensor.byte_length,
        "range_identity_sha256": sha256_bytes(
            (
                f"{tensor.shard_sha256}:{tensor.absolute_offset}:"
                f"{tensor.byte_length}"
            ).encode("ascii")
        ),
    }

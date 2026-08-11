"""Deterministic Safetensors sharding and atomic artifact publication."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import struct
from typing import Any

from .common import (
    BoundedWorkspace,
    OutputError,
    canonical_json_bytes,
    compact_json_bytes,
    fsync_directory,
    safe_relative_path,
    write_file_atomic,
)
from .encoders import EncoderResult, TensorEncoder
from .plan import QuantizationPlan, TensorCompilePlan
from .reader import TensorDescriptor


@dataclass(frozen=True)
class ShardAssignment:
    name: str
    tensors: tuple[TensorCompilePlan, ...]


@dataclass(frozen=True)
class WrittenTensor:
    plan: TensorCompilePlan
    shard: str
    data_offsets: tuple[int, int]
    source_result: EncoderResult


@dataclass(frozen=True)
class WrittenArtifactPayload:
    tensors: tuple[WrittenTensor, ...]
    shard_names: tuple[str, ...]
    index_name: str
    metadata_names: tuple[str, ...]


def assign_shards(plan: QuantizationPlan) -> tuple[ShardAssignment, ...]:
    groups: list[list[TensorCompilePlan]] = []
    current: list[TensorCompilePlan] = []
    current_bytes = 0
    for tensor in plan.tensors:
        size = tensor.output_bytes
        if current and current_bytes + size > plan.target_shard_bytes:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(tensor)
        current_bytes += size
        if size >= plan.target_shard_bytes:
            groups.append(current)
            current = []
            current_bytes = 0
    if current:
        groups.append(current)
    count = len(groups)
    if count == 0:
        raise OutputError("compiler plan produced no output shards")
    return tuple(
        ShardAssignment(
            name=f"model-{index + 1:05d}-of-{count:05d}.safetensors",
            tensors=tuple(group),
        )
        for index, group in enumerate(groups)
    )


def safetensors_header(
    tensors: tuple[TensorCompilePlan, ...]
) -> tuple[bytes, dict[str, tuple[int, int]]]:
    offset = 0
    metadata: dict[str, Any] = {
        "__metadata__": {
            "format": "pt",
            "gem16_artifact": "m04-synthetic-copy-scaffold",
        }
    }
    offsets: dict[str, tuple[int, int]] = {}
    for tensor in tensors:
        end = offset + tensor.output_bytes
        offsets[tensor.output_name] = (offset, end)
        metadata[tensor.output_name] = {
            "dtype": tensor.output_dtype,
            "shape": list(tensor.physical_shape),
            "data_offsets": [offset, end],
        }
        offset = end
    header = compact_json_bytes(metadata)
    header += b" " * ((-len(header)) % 8)
    return struct.pack("<Q", len(header)) + header, offsets


def _open_exclusive(path: Path):
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        return os.fdopen(descriptor, "wb", buffering=0)
    except OSError as error:
        raise OutputError(f"cannot create compiler output {path}: {error}") from error


def write_shards(
    staging: Path,
    plan: QuantizationPlan,
    source_tensors: dict[str, TensorDescriptor],
    workspace: BoundedWorkspace,
    encoders: dict[str, TensorEncoder],
) -> WrittenArtifactPayload:
    assignments = assign_shards(plan)
    written: list[WrittenTensor] = []
    weight_map: dict[str, str] = {}
    for assignment in assignments:
        header, offsets = safetensors_header(assignment.tensors)
        workspace.record_header(len(header) - 8, f"building {assignment.name} header")
        path = staging / assignment.name
        with _open_exclusive(path) as output:
            output.write(header)
            for tensor in assignment.tensors:
                encoder = encoders.get(tensor.encoder)
                if encoder is None:
                    raise OutputError(f"encoder is not registered: {tensor.encoder}")
                sources = tuple(source_tensors[name] for name in tensor.source_names)
                result = encoder.compile_tensor(tensor, sources, output, workspace)
                if result.output_bytes != tensor.output_bytes:
                    raise OutputError(
                        f"encoder byte count mismatch for {tensor.output_name}: "
                        f"expected {tensor.output_bytes}, got {result.output_bytes}"
                    )
                written.append(
                    WrittenTensor(
                        plan=tensor,
                        shard=assignment.name,
                        data_offsets=offsets[tensor.output_name],
                        source_result=result,
                    )
                )
                weight_map[tensor.output_name] = assignment.name
            output.flush()
            os.fsync(output.fileno())
        workspace.check(f"writing {assignment.name}")

    index_name = "model.safetensors.index.json"
    index = {
        "metadata": {"total_size": plan.output_tensor_bytes},
        "weight_map": dict(sorted(weight_map.items())),
    }
    write_file_atomic(staging / index_name, canonical_json_bytes(index))
    return WrittenArtifactPayload(
        tensors=tuple(written),
        shard_names=tuple(assignment.name for assignment in assignments),
        index_name=index_name,
        metadata_names=(),
    )


def copy_approved_metadata(
    staging: Path,
    plan: QuantizationPlan,
    source_files: dict[str, Any],
    workspace: BoundedWorkspace,
) -> tuple[str, ...]:
    copied: list[str] = []
    for relative in sorted(plan.approved_metadata_files):
        parsed = safe_relative_path(relative, "approved metadata output")
        source = source_files[relative]
        destination = staging.joinpath(*parsed.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with _open_exclusive(destination) as output:
            source_hash, output_hash = workspace.copy_range(
                source.path, 0, source.size, output, track_tensor=False
            )
            output.flush()
            os.fsync(output.fileno())
        if source_hash != source.sha256 or output_hash != source.sha256:
            raise OutputError(f"metadata hash changed while copying: {relative}")
        copied.append(relative)
    return tuple(copied)


def create_staging_directory(output: Path) -> Path:
    if output.exists() or output.is_symlink():
        raise OutputError(f"compiler output already exists: {output}")
    staging = output.with_name(output.name + ".incomplete")
    if staging.exists() or staging.is_symlink():
        raise OutputError(
            f"stale incomplete compiler output exists: {staging}; "
            "M04 uses restart-only recovery"
        )
    try:
        staging.mkdir(mode=0o700, parents=False)
    except OSError as error:
        raise OutputError(f"cannot create staging directory {staging}: {error}") from error
    return staging


def discard_staging(staging: Path) -> None:
    try:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
    except OSError:
        # Preserve the original compiler failure; the visible .incomplete suffix
        # prevents this directory from looking like a valid artifact.
        pass


def publish_staging(staging: Path, output: Path) -> None:
    fsync_directory(staging)
    fsync_directory(staging.parent)
    try:
        os.replace(staging, output)
    except OSError as error:
        raise OutputError(f"cannot atomically publish {output}: {error}") from error
    fsync_directory(output.parent)

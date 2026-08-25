"""Deterministic Safetensors sharding and atomic artifact publication."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import struct
import json
from typing import Any

from .common import (
    BoundedWorkspace,
    DataError,
    OutputError,
    canonical_json_bytes,
    compact_json_bytes,
    fsync_directory,
    safe_relative_path,
    reject_duplicate_keys,
    write_all,
    write_file_atomic,
)
from .encoders import EncoderResult, TensorEncoder
from .native_fp8 import NativeBundle, NativeBundleEncoder
from .plan import QuantizationPlan, TensorCompilePlan
from .profiles import profile_for
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


@dataclass(frozen=True)
class DirectShardLayout:
    """Preallocated canonical shard ranges for the native M06 data plane."""

    assignments: tuple[ShardAssignment, ...]
    shard_names: tuple[str, ...]
    index_name: str
    # output name -> (absolute shard path, absolute file offset)
    outputs: dict[str, tuple[Path, int]]


def compiled_config_bytes(source: bytes, profile: str) -> bytes:
    """Add a versioned Gem16 block without rewriting architecture facts."""
    try:
        document = json.loads(source.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError, DataError, ValueError) as error:
        raise OutputError(f"cannot parse locked compiled config.json: {error}") from error
    if not isinstance(document, dict) or "gem16" in document:
        raise OutputError("compiled source config must be an object without a gem16 block")
    if profile == "sm120-text-hybrid-v1":
        variant = "gemma4-26b-a4b"
        supports_mtp = False
    elif profile == "sm120-mtp-assistant-hybrid-v1":
        variant = "gemma4-26b-a4b-mtp-assistant"
        supports_mtp = True
    else:
        raise OutputError(f"unsupported generated config profile: {profile}")
    document["gem16"] = {
        "schema_version": 1,
        "compiler_data_plane": "native-cpp20",
        "profile": profile,
        "variant": variant,
        "text_only": True,
        "head_format": "nvfp4-group16-divisor-v1",
        "supports_mtp": supports_mtp,
        "supports_vision": False,
        "supports_audio": False,
        "supports_video": False,
    }
    return canonical_json_bytes(document)


def m08_config_bytes(source: bytes) -> bytes:
    """Compatibility wrapper for the accepted M08 config contract."""
    return compiled_config_bytes(source, "sm120-text-hybrid-v1")


class _BoundedRangeWriter:
    """Write-only view over one preallocated tensor range."""

    def __init__(self, stream: Any, offset: int, length: int, description: str):
        self._stream = stream
        self._remaining = length
        self._description = description
        stream.seek(offset)

    def write(self, payload: bytes | bytearray | memoryview) -> int:
        if len(payload) > self._remaining:
            raise OutputError(f"encoder exceeded output range: {self._description}")
        written = self._stream.write(payload)
        if written is None:
            written = len(payload)
        self._remaining -= written
        return written

    def finish(self) -> None:
        if self._remaining != 0:
            raise OutputError(
                f"encoder left {self._remaining} bytes unwritten: {self._description}"
            )


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
    tensors: tuple[TensorCompilePlan, ...],
    artifact_label: str = "m04-synthetic-copy-scaffold",
) -> tuple[bytes, dict[str, tuple[int, int]]]:
    offset = 0
    metadata: dict[str, Any] = {
        "__metadata__": {
            "format": "pt",
            "gem16_artifact": artifact_label,
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
    native_bundle: NativeBundle | None = None,
) -> WrittenArtifactPayload:
    assignments = assign_shards(plan)
    profile = profile_for(plan.artifact_profile, plan.head_format)
    if native_bundle is not None:
        encoders = dict(encoders)
        encoders["fp8-rowwise-weight-v1"] = NativeBundleEncoder(native_bundle, "weight")
        encoders["fp8-rowwise-scale-v1"] = NativeBundleEncoder(native_bundle, "scale")
    written: list[WrittenTensor] = []
    weight_map: dict[str, str] = {}
    for assignment in assignments:
        header, offsets = safetensors_header(
            assignment.tensors, artifact_label=profile.header_label
        )
        workspace.record_header(len(header) - 8, f"building {assignment.name} header")
        path = staging / assignment.name
        with _open_exclusive(path) as output:
            write_all(output, header, f"writing {assignment.name} header")
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


def prepare_direct_shards(
    staging: Path,
    plan: QuantizationPlan,
    workspace: BoundedWorkspace,
) -> DirectShardLayout:
    """Write canonical headers and reserve exact final shard sizes for M06.

    The native encoder may only write the ranges described by the resulting
    layout.  It never creates, truncates, or resizes these files.
    """
    assignments = assign_shards(plan)
    profile = profile_for(plan.artifact_profile, plan.head_format)
    outputs: dict[str, tuple[Path, int]] = {}
    for assignment in assignments:
        header, offsets = safetensors_header(
            assignment.tensors, artifact_label=profile.header_label
        )
        workspace.record_header(len(header) - 8, f"building {assignment.name} header")
        path = staging / assignment.name
        total_size = len(header) + sum(tensor.output_bytes for tensor in assignment.tensors)
        descriptor = -1
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb", buffering=0) as output:
                descriptor = -1
                write_all(output, header, f"writing {assignment.name} header")
                if not hasattr(os, "posix_fallocate"):
                    raise OutputError("M06 requires POSIX disk preallocation")
                try:
                    os.posix_fallocate(output.fileno(), 0, total_size)
                except OSError as error:
                    raise OutputError(
                        f"cannot reserve M06 shard space {path}: {error}"
                    ) from error
                os.ftruncate(output.fileno(), total_size)
                os.fsync(output.fileno())
        except OSError as error:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            raise OutputError(f"cannot preallocate M06 shard {path}: {error}") from error
        for tensor in assignment.tensors:
            begin, _ = offsets[tensor.output_name]
            outputs[tensor.output_name] = (path, len(header) + begin)
    return DirectShardLayout(
        assignments=assignments,
        shard_names=tuple(assignment.name for assignment in assignments),
        index_name="model.safetensors.index.json",
        outputs=outputs,
    )


def finalize_direct_shards(
    staging: Path,
    plan: QuantizationPlan,
    layout: DirectShardLayout,
    native_result: Any,
    workspace: BoundedWorkspace,
) -> WrittenArtifactPayload:
    """Reconcile native direct-range hashes without copying converted payloads."""
    written: list[WrittenTensor] = []
    weight_map: dict[str, str] = {}
    result_by_name = native_result.outputs
    for assignment in layout.assignments:
        path = staging / assignment.name
        header, offsets = safetensors_header(
            assignment.tensors,
            artifact_label=profile_for(plan.artifact_profile, plan.head_format).header_label,
        )
        for tensor in assignment.tensors:
            direct = result_by_name.get(tensor.output_name)
            if direct is None:
                raise DataError(f"native M06 result is missing: {tensor.output_name}")
            shard_path, absolute_offset = layout.outputs[tensor.output_name]
            if shard_path != path or direct.output_bytes != tensor.output_bytes:
                raise DataError(f"native M06 result range mismatch: {tensor.output_name}")
            # The native telemetry hash was computed after the final direct
            # write and the native encoder fsynced every output range.  Reuse
            # it here; canonical artifact verification below remains the
            # mutation-safety boundary and avoids another full payload scan.
            output_hash = direct.output_sha256
            sources = tuple(tensor.source_names)
            written.append(WrittenTensor(
                plan=tensor,
                shard=assignment.name,
                data_offsets=offsets[tensor.output_name],
                source_result=EncoderResult(
                    source_sha256=(direct.source_sha256,),
                    output_sha256=output_hash,
                    output_bytes=tensor.output_bytes,
                    statistics=direct.statistics,
                ),
            ))
            weight_map[tensor.output_name] = assignment.name
        workspace.check(f"reconciling {assignment.name}")
    index = {
        "metadata": {"total_size": plan.output_tensor_bytes},
        "weight_map": dict(sorted(weight_map.items())),
    }
    write_file_atomic(staging / layout.index_name, canonical_json_bytes(index))
    return WrittenArtifactPayload(
        tensors=tuple(written),
        shard_names=layout.shard_names,
        index_name=layout.index_name,
        metadata_names=(),
    )


def finalize_mixed_shards(
    staging: Path,
    plan: QuantizationPlan,
    source_tensors: dict[str, TensorDescriptor],
    layout: DirectShardLayout,
    native_nvfp4_result: Any,
    fp8_bundle: NativeBundle,
    workspace: BoundedWorkspace,
    encoders: dict[str, TensorEncoder],
) -> WrittenArtifactPayload:
    """Fill non-NVFP4 ranges around direct native M08 NVFP4 outputs."""
    registry = dict(encoders)
    registry["fp8-rowwise-weight-v1"] = NativeBundleEncoder(fp8_bundle, "weight")
    registry["fp8-rowwise-scale-v1"] = NativeBundleEncoder(fp8_bundle, "scale")
    nvfp4_encoders = {
        "nvfp4-packed-v1", "nvfp4-local-scale-v1",
        "nvfp4-weight-divisor-v1", "nvfp4-input-divisor-v1",
    }
    written_by_name: dict[str, WrittenTensor] = {}
    assignments_by_name = {
        tensor.output_name: assignment
        for assignment in layout.assignments
        for tensor in assignment.tensors
    }

    for tensor in plan.tensors:
        assignment = assignments_by_name[tensor.output_name]
        _path, absolute_offset = layout.outputs[tensor.output_name]
        header, offsets = safetensors_header(
            assignment.tensors,
            artifact_label=profile_for(plan.artifact_profile, plan.head_format).header_label,
        )
        if tensor.encoder in nvfp4_encoders:
            direct = native_nvfp4_result.outputs.get(tensor.output_name)
            if direct is None or direct.output_bytes != tensor.output_bytes:
                raise DataError(f"native M08 NVFP4 result is missing: {tensor.output_name}")
            result = EncoderResult(
                source_sha256=(direct.source_sha256,),
                output_sha256=direct.output_sha256,
                output_bytes=direct.output_bytes,
                statistics=direct.statistics,
            )
        else:
            encoder = registry.get(tensor.encoder)
            if encoder is None:
                raise OutputError(f"M08 encoder is not registered: {tensor.encoder}")
            sources = tuple(source_tensors[name] for name in tensor.source_names)
            shard_path = staging / assignment.name
            with shard_path.open("r+b", buffering=0) as stream:
                target = _BoundedRangeWriter(
                    stream, absolute_offset, tensor.output_bytes, tensor.output_name
                )
                result = encoder.compile_tensor(
                    tensor, sources, target, workspace  # type: ignore[arg-type]
                )
                target.finish()
                stream.flush()
            if result.output_bytes != tensor.output_bytes:
                raise OutputError(f"M08 encoder byte mismatch: {tensor.output_name}")
        written_by_name[tensor.output_name] = WrittenTensor(
            plan=tensor,
            shard=assignment.name,
            data_offsets=offsets[tensor.output_name],
            source_result=result,
        )

    for name in layout.shard_names:
        with (staging / name).open("rb") as stream:
            os.fsync(stream.fileno())
    weight_map = {
        tensor.output_name: assignments_by_name[tensor.output_name].name
        for tensor in plan.tensors
    }
    write_file_atomic(
        staging / layout.index_name,
        canonical_json_bytes({
            "metadata": {"total_size": plan.output_tensor_bytes},
            "weight_map": dict(sorted(weight_map.items())),
        }),
    )
    return WrittenArtifactPayload(
        tensors=tuple(written_by_name[tensor.output_name] for tensor in plan.tensors),
        shard_names=layout.shard_names,
        index_name=layout.index_name,
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
        if plan.artifact_profile in {
            "sm120-text-hybrid-v1", "sm120-mtp-assistant-hybrid-v1"
        } and relative == "config.json":
            source_payload = source.path.read_bytes()
            if len(source_payload) != source.size:
                raise OutputError("M08 source config size changed while reading")
            payload = compiled_config_bytes(source_payload, plan.artifact_profile)
            with _open_exclusive(destination) as output:
                write_all(output, payload, "writing M08 config.json")
                output.flush()
                os.fsync(output.fileno())
        else:
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

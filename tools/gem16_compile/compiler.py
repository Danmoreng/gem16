"""Deterministic M04 compiler orchestration, provenance, and verification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from .common import (
    BoundedWorkspace,
    DataError,
    InvalidPlanError,
    OutputError,
    ReproducibilityError,
    SourceVerificationError,
    canonical_json_bytes,
    environment_identity,
    git_compiler_identity,
    load_json,
    safe_relative_path,
    sha256_bytes,
    write_file_atomic,
)
from .encoders import TensorEncoder, default_encoder_registry
from .plan import QuantizationPlan, load_quantization_plan, plan_summary
from .reader import (
    TensorDescriptor,
    VerifiedSource,
    read_artifact_tensors,
    read_source_tensors,
    tensor_source_identity,
    verify_source_lock,
)
from .writer import (
    WrittenArtifactPayload,
    assign_shards,
    copy_approved_metadata,
    create_staging_directory,
    discard_staging,
    publish_staging,
    write_shards,
)


COMPILATION_MANIFEST = "gem16_compilation.json"
DEPENDENCIES_LOCK = Path(__file__).with_name("dependencies.lock.json")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CompilerIdentity:
    repository: str
    commit: str
    dirty: bool
    environment: dict[str, str]

    @staticmethod
    def from_repository(repository_root: Path = REPOSITORY_ROOT) -> "CompilerIdentity":
        commit, dirty = git_compiler_identity(repository_root)
        return CompilerIdentity(
            repository="Danmoreng/gem16",
            commit=commit,
            dirty=dirty,
            environment=environment_identity(),
        )


@dataclass(frozen=True)
class CompilerRequest:
    source_lock: Path
    source_directory: Path
    compiler_manifest: Path
    profile: str
    head_format: str
    host_memory_cap_bytes: int
    staging_bytes: int
    shard_size: int | None = None
    threads: int = 1
    reference_platform_strict: bool = False
    dependencies_lock: Path = DEPENDENCIES_LOCK


def _validate_dependencies_lock(path: Path) -> tuple[dict[str, Any], str]:
    document = load_json(path, 1024 * 1024)
    if document.get("schema_version") != 1:
        raise InvalidPlanError("unsupported compiler dependency-lock schema")
    if document.get("runtime") != "python-standard-library-only":
        raise InvalidPlanError("M04 compiler dependencies must be standard-library-only")
    if document.get("external_packages") != []:
        raise InvalidPlanError("M04 compiler dependency lock contains external packages")
    payload = path.read_bytes()
    return document, sha256_bytes(payload)


def _validate_request(request: CompilerRequest) -> None:
    if request.threads != 1:
        raise InvalidPlanError(
            "M04 supports exactly one deterministic compiler thread; "
            "parallel encoders begin in later milestones"
        )
    if request.shard_size is not None and request.shard_size <= 0:
        raise InvalidPlanError("--shard-size must be positive")


def _validate_reference_environment(
    plan: QuantizationPlan, identity: CompilerIdentity
) -> None:
    mismatches = {
        key: {"expected": expected, "actual": identity.environment.get(key)}
        for key, expected in plan.reference_environment.items()
        if identity.environment.get(key) != expected
    }
    if mismatches:
        raise InvalidPlanError(
            f"compiler environment differs from reference policy: {mismatches}"
        )


def _load_request(
    request: CompilerRequest, workspace: BoundedWorkspace
) -> tuple[VerifiedSource, dict[str, TensorDescriptor], QuantizationPlan, str]:
    _validate_request(request)
    _, dependency_hash = _validate_dependencies_lock(request.dependencies_lock)
    # Full source-lock verification deliberately precedes all Safetensors
    # header interpretation and tensor-range access.
    source = verify_source_lock(
        request.source_lock, request.source_directory, workspace
    )
    source_tensors = read_source_tensors(source, workspace)
    plan = load_quantization_plan(
        request.compiler_manifest,
        source,
        source_tensors,
        request.profile,
        request.head_format,
        request.shard_size,
    )
    return source, source_tensors, plan, dependency_hash


def _reverify_source_files(
    request: CompilerRequest,
    expected: VerifiedSource,
    workspace: BoundedWorkspace,
) -> None:
    verified = verify_source_lock(
        request.source_lock, request.source_directory, workspace
    )
    if (
        verified.lock_sha256 != expected.lock_sha256
        or verified.repository != expected.repository
        or verified.revision != expected.revision
        or {
            name: (file.size, file.sha256)
            for name, file in verified.files.items()
        }
        != {
            name: (file.size, file.sha256)
            for name, file in expected.files.items()
        }
    ):
        raise SourceVerificationError(
            "source identity changed during compiler operation"
        )


def _excluded_records(
    plan: QuantizationPlan,
    source_tensors: dict[str, TensorDescriptor],
    workspace: BoundedWorkspace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    family_totals: dict[str, dict[str, Any]] = {
        family: {
            "group": family,
            "reason": "absent from source and excluded by text-only profile",
            "source_tensor_count": 0,
            "source_bytes": 0,
        }
        for family in plan.omitted_families
    }
    for item in sorted(plan.excluded_tensors, key=lambda value: value.source_name):
        tensor = source_tensors[item.source_name]
        payload_hash = workspace.hash_tensor_range(
            tensor.path, tensor.absolute_offset, tensor.byte_length
        )
        record = {
            "source_name": item.source_name,
            "source_dtype": tensor.dtype,
            "source_shape": list(tensor.shape),
            "source_bytes": tensor.byte_length,
            "source_tensor_sha256": payload_hash,
            "source_range": tensor_source_identity(tensor),
            "family": item.family,
            "role": item.role,
            "residency_class": item.residency_class,
            "reason": item.reason,
        }
        records.append(record)
        totals = family_totals[item.family]
        totals["reason"] = item.reason
        totals["source_tensor_count"] += 1
        totals["source_bytes"] += tensor.byte_length
    return records, [family_totals[name] for name in sorted(family_totals)]


def _file_record(
    root: Path, relative: str, kind: str, workspace: BoundedWorkspace
) -> dict[str, Any]:
    parsed = safe_relative_path(relative, "compiled artifact file")
    path = root.joinpath(*parsed.parts)
    try:
        if path.is_symlink() or not path.is_file():
            raise OutputError(f"compiled artifact file is unsafe or missing: {relative}")
        size = path.stat().st_size
    except OSError as error:
        raise OutputError(f"cannot inspect compiled artifact file {relative}: {error}") from error
    return {
        "path": relative,
        "kind": kind,
        "size": size,
        "sha256": workspace.hash_range(path, 0, size),
    }


def _tensor_provenance(
    payload: WrittenArtifactPayload,
    source_tensors: dict[str, TensorDescriptor],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for written in sorted(payload.tensors, key=lambda value: value.plan.output_name):
        plan = written.plan
        sources = [source_tensors[name] for name in plan.source_names]
        if len(sources) != len(written.source_result.source_sha256):
            raise DataError(
                f"encoder source-hash count mismatch: {plan.output_name}"
            )
        source_records = [
            {
                "name": source.name,
                "sha256": source_hash,
                "range": tensor_source_identity(source),
            }
            for source, source_hash in zip(
                sources, written.source_result.source_sha256, strict=True
            )
        ]
        records.append(
            {
                "output_name": plan.output_name,
                "operation_id": plan.operation_id,
                "output_dtype": plan.output_dtype,
                "physical_shape": list(plan.physical_shape),
                "logical_dtype": plan.logical_dtype,
                "logical_shape": list(plan.logical_shape),
                "byte_length": written.source_result.output_bytes,
                "sha256": written.source_result.output_sha256,
                "output_shard": written.shard,
                "output_data_offsets": list(written.data_offsets),
                "sources": source_records,
                "transformation": plan.transformation,
                "transformation_version": plan.transformation_version,
                "axis_transformation": plan.axis_transformation,
                "quantizer_parameters": plan.quantizer_parameters,
                "dequantization_equation": plan.dequantization_equation,
                "role": plan.role,
                "residency_class": plan.residency_class,
                "disk_layout": plan.disk_layout,
                "runtime_layout": plan.runtime_layout,
                "aliased": plan.aliased,
            }
        )
    return records


def _build_compilation_manifest(
    request: CompilerRequest,
    source: VerifiedSource,
    plan: QuantizationPlan,
    dependency_hash: str,
    identity: CompilerIdentity,
    payload: WrittenArtifactPayload,
    source_tensors: dict[str, TensorDescriptor],
    staging: Path,
    workspace: BoundedWorkspace,
) -> dict[str, Any]:
    excluded, omitted_groups = _excluded_records(plan, source_tensors, workspace)
    files: list[dict[str, Any]] = []
    for name in payload.shard_names:
        files.append(_file_record(staging, name, "safetensors_shard", workspace))
    files.append(_file_record(staging, payload.index_name, "safetensors_index", workspace))
    for name in sorted(payload.metadata_names):
        files.append(_file_record(staging, name, "source_metadata_copy", workspace))
    return {
        "schema_version": 1,
        "artifact_profile": plan.artifact_profile,
        "artifact_status": "m04_scaffold_not_runtime_loadable",
        "source": {
            "lock_sha256": source.lock_sha256,
            "repository": source.repository,
            "revision": source.revision,
            "resolved_at_utc": source.resolved_at_utc,
        },
        "compiler": {
            "repository": identity.repository,
            "commit": identity.commit,
            "dirty": identity.dirty,
            "python": identity.environment["python_version"],
            "platform": identity.environment,
            "dependencies_lock_sha256": dependency_hash,
            "implementation": "gem16_compile_m04_v1",
        },
        "plan": {
            "schema_version": plan.schema_version,
            "compiler_manifest_sha256": plan.compiler_manifest_sha256,
            "resolved_plan_sha256": plan.resolved_plan_sha256,
            "source_contract": plan.source_contract,
            "target_shard_bytes": plan.target_shard_bytes,
        },
        "quantization": {
            "profile": plan.artifact_profile,
            "attention": "copy-v1-scaffold",
            "experts": "copy-v1-scaffold",
            "embedding_head": "source-copy-v1",
            "production_quantization_implemented": False,
        },
        "head_format": plan.head_format,
        "text_only": True,
        "omitted_families": list(plan.omitted_families),
        "omitted_tensor_groups": omitted_groups,
        "excluded_tensors": excluded,
        "tensors": _tensor_provenance(payload, source_tensors),
        "files": sorted(files, key=lambda value: value["path"]),
        "file_hash_scope": (
            "all artifact files except gem16_compilation.json; its self-hash "
            "is supplied by the external artifact lock in M08"
        ),
        "byte_totals": {
            "source_tensor_count": len(source_tensors),
            "output_tensor_count": len(plan.tensors),
            "output_tensor_bytes": plan.output_tensor_bytes,
            "excluded_tensor_count": len(excluded),
            "excluded_tensor_bytes": sum(item["source_bytes"] for item in excluded),
        },
        "compiler_settings": {
            "threads": request.threads,
            "host_memory_cap_bytes": request.host_memory_cap_bytes,
            "staging_buffer_bytes": request.staging_bytes,
            "reference_platform_strict": request.reference_platform_strict,
            "resume_policy": "restart-only",
        },
    }


def _write_report(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    if path.exists() or path.is_symlink():
        raise OutputError(f"compiler report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_file_atomic(path, canonical_json_bytes(report))


def _base_report(
    action: str,
    request: CompilerRequest,
    workspace: BoundedWorkspace,
    started: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "milestone": "M04",
        "action": action,
        "status": "pass",
        "artifact_profile": request.profile,
        "duration_seconds": round(time.monotonic() - started, 6),
        "memory": workspace.telemetry(),
    }


def plan_artifact(
    request: CompilerRequest,
    *,
    identity: CompilerIdentity | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    workspace = BoundedWorkspace(
        request.host_memory_cap_bytes, request.staging_bytes
    )
    source, source_tensors, plan, dependency_hash = _load_request(request, workspace)
    compiler_identity = identity or CompilerIdentity.from_repository()
    if request.reference_platform_strict:
        _validate_reference_environment(plan, compiler_identity)
    assignments = assign_shards(plan)
    excluded_bytes = sum(
        source_tensors[item.source_name].byte_length for item in plan.excluded_tensors
    )
    _reverify_source_files(request, source, workspace)
    report = _base_report("plan", request, workspace, started)
    report.update(plan_summary(plan))
    report["status"] = "pass"
    report["source"] = {
        "repository": source.repository,
        "revision": source.revision,
        "lock_sha256": source.lock_sha256,
        "tensor_count": len(source_tensors),
    }
    report["dependencies_lock_sha256"] = dependency_hash
    report["projected_shards"] = [
        {
            "name": assignment.name,
            "tensor_count": len(assignment.tensors),
            "payload_bytes": sum(tensor.output_bytes for tensor in assignment.tensors),
        }
        for assignment in assignments
    ]
    report["excluded_tensor_bytes"] = excluded_bytes
    report["tensor_mapping"] = [
        {
            "operation_id": tensor.operation_id,
            "output_name": tensor.output_name,
            "source_names": list(tensor.source_names),
            "encoder": tensor.encoder,
            "output_dtype": tensor.output_dtype,
            "physical_shape": list(tensor.physical_shape),
            "output_bytes": tensor.output_bytes,
            "role": tensor.role,
            "residency_class": tensor.residency_class,
        }
        for tensor in plan.tensors
    ]
    report["excluded_tensors"] = [
        {
            "source_name": item.source_name,
            "family": item.family,
            "role": item.role,
            "residency_class": item.residency_class,
            "source_bytes": source_tensors[item.source_name].byte_length,
        }
        for item in plan.excluded_tensors
    ]
    report["predicted_incremental_host_bytes"] = (
        request.staging_bytes + workspace.max_header_bytes
    )
    report["memory"] = workspace.telemetry()
    return report


def compile_artifact(
    request: CompilerRequest,
    output: Path,
    *,
    report_path: Path | None = None,
    identity: CompilerIdentity | None = None,
    allow_dirty: bool = False,
    encoders: dict[str, TensorEncoder] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    workspace = BoundedWorkspace(
        request.host_memory_cap_bytes, request.staging_bytes
    )
    source, source_tensors, plan, dependency_hash = _load_request(request, workspace)
    compiler_identity = identity or CompilerIdentity.from_repository()
    if compiler_identity.dirty and not allow_dirty:
        raise InvalidPlanError(
            "release compiler requires a clean repository; "
            "--allow-dirty is diagnostic only"
        )
    if request.reference_platform_strict:
        _validate_reference_environment(plan, compiler_identity)
    output = output.expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = create_staging_directory(output)
    try:
        payload = write_shards(
            staging,
            plan,
            source_tensors,
            workspace,
            encoders or default_encoder_registry(),
        )
        metadata_names = copy_approved_metadata(
            staging, plan, source.files, workspace
        )
        payload = WrittenArtifactPayload(
            tensors=payload.tensors,
            shard_names=payload.shard_names,
            index_name=payload.index_name,
            metadata_names=metadata_names,
        )
        compilation = _build_compilation_manifest(
            request,
            source,
            plan,
            dependency_hash,
            compiler_identity,
            payload,
            source_tensors,
            staging,
            workspace,
        )
        write_file_atomic(
            staging / COMPILATION_MANIFEST, canonical_json_bytes(compilation)
        )
        _verify_loaded_artifact(
            staging,
            source,
            source_tensors,
            plan,
            dependency_hash,
            workspace,
        )
        _reverify_source_files(request, source, workspace)
        publish_staging(staging, output)
    except Exception:
        discard_staging(staging)
        raise

    report = _base_report("compile", request, workspace, started)
    report.update(
        {
            "output": str(output),
            "compiler_commit": compiler_identity.commit,
            "compiler_dirty": compiler_identity.dirty,
            "source_lock_sha256": source.lock_sha256,
            "compiler_manifest_sha256": plan.compiler_manifest_sha256,
            "resolved_plan_sha256": plan.resolved_plan_sha256,
            "compilation_manifest_sha256": workspace.hash_range(
                output / COMPILATION_MANIFEST,
                0,
                (output / COMPILATION_MANIFEST).stat().st_size,
            ),
            "output_tensor_count": len(plan.tensors),
            "output_tensor_bytes": plan.output_tensor_bytes,
            "output_file_count": len(compilation["files"]) + 1,
            "memory": workspace.telemetry(),
        }
    )
    _write_report(report_path, report)
    return report


def _manifest_file_map(compilation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values = compilation.get("files")
    if not isinstance(values, list):
        raise DataError("compilation manifest files must be an array")
    result: dict[str, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, dict) or set(value) != {
            "path",
            "kind",
            "size",
            "sha256",
        }:
            raise DataError("compilation manifest file record must match schema")
        path = value.get("path")
        if not isinstance(path, str):
            raise DataError("compilation manifest file path must be a string")
        safe_relative_path(path, "compilation manifest file")
        if path in result:
            raise DataError(f"duplicate compilation file record: {path}")
        result[path] = value
    return result


def _manifest_tensor_map(compilation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values = compilation.get("tensors")
    if not isinstance(values, list):
        raise DataError("compilation manifest tensors must be an array")
    result: dict[str, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, dict) or not isinstance(value.get("output_name"), str):
            raise DataError("compilation tensor record is malformed")
        name = value["output_name"]
        if name in result:
            raise DataError(f"duplicate compilation tensor record: {name}")
        result[name] = value
    return result


def _actual_artifact_files(root: Path) -> set[str]:
    result: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise DataError(f"compiled artifact contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise DataError(f"compiled artifact contains a non-file: {path}")
        result.add(path.relative_to(root).as_posix())
    return result


def _verify_loaded_artifact(
    artifact: Path,
    source: VerifiedSource,
    source_tensors: dict[str, TensorDescriptor],
    plan: QuantizationPlan,
    dependency_hash: str,
    workspace: BoundedWorkspace,
) -> dict[str, Any]:
    try:
        compilation = load_json(artifact / COMPILATION_MANIFEST, 64 * 1024 * 1024)
    except InvalidPlanError as error:
        raise DataError(f"invalid compilation manifest: {error}") from error
    expected_top_level = {
        "schema_version",
        "artifact_profile",
        "artifact_status",
        "source",
        "compiler",
        "plan",
        "quantization",
        "head_format",
        "text_only",
        "omitted_families",
        "omitted_tensor_groups",
        "excluded_tensors",
        "tensors",
        "files",
        "file_hash_scope",
        "byte_totals",
        "compiler_settings",
    }
    if set(compilation) != expected_top_level:
        raise DataError("gem16_compilation top-level schema mismatch")
    if compilation.get("schema_version") != 1:
        raise DataError("unsupported gem16_compilation schema_version")
    if compilation.get("artifact_profile") != plan.artifact_profile:
        raise DataError("compiled artifact profile differs from compiler plan")
    if compilation.get("artifact_status") != "m04_scaffold_not_runtime_loadable":
        raise DataError("M04 artifact status is missing or incorrect")
    expected_quantization = {
        "profile": plan.artifact_profile,
        "attention": "copy-v1-scaffold",
        "experts": "copy-v1-scaffold",
        "embedding_head": "source-copy-v1",
        "production_quantization_implemented": False,
    }
    if (
        compilation.get("quantization") != expected_quantization
        or compilation.get("head_format") != plan.head_format
        or compilation.get("text_only") is not True
    ):
        raise DataError("M04 scaffold precision/text-only contract mismatch")
    source_record = compilation.get("source")
    if not isinstance(source_record, dict) or source_record != {
        "lock_sha256": source.lock_sha256,
        "repository": source.repository,
        "revision": source.revision,
        "resolved_at_utc": source.resolved_at_utc,
    }:
        raise DataError("compiled artifact source provenance mismatch")
    plan_record = compilation.get("plan")
    if not isinstance(plan_record, dict) or set(plan_record) != {
        "schema_version",
        "compiler_manifest_sha256",
        "resolved_plan_sha256",
        "source_contract",
        "target_shard_bytes",
    } or (
        plan_record.get("schema_version") != plan.schema_version
        or plan_record.get("compiler_manifest_sha256") != plan.compiler_manifest_sha256
        or plan_record.get("resolved_plan_sha256") != plan.resolved_plan_sha256
        or plan_record.get("source_contract") != plan.source_contract
        or plan_record.get("target_shard_bytes") != plan.target_shard_bytes
    ):
        raise DataError("compiled artifact plan provenance mismatch")
    compiler_record = compilation.get("compiler")
    if not isinstance(compiler_record, dict) or set(compiler_record) != {
        "repository",
        "commit",
        "dirty",
        "python",
        "platform",
        "dependencies_lock_sha256",
        "implementation",
    }:
        raise DataError("compiled artifact compiler provenance is missing")
    commit = compiler_record.get("commit")
    platform_record = compiler_record.get("platform")
    required_platform_fields = {
        "system",
        "machine",
        "python_implementation",
        "python_version",
        "python_major_minor",
        "byteorder",
        "locale",
    }
    if (
        compiler_record.get("repository") != "Danmoreng/gem16"
        or not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or not isinstance(compiler_record.get("dirty"), bool)
        or compiler_record.get("dependencies_lock_sha256") != dependency_hash
        or compiler_record.get("implementation") != "gem16_compile_m04_v1"
        or not isinstance(platform_record, dict)
        or set(platform_record) != required_platform_fields
        or not all(isinstance(value, str) and value for value in platform_record.values())
        or compiler_record.get("python") != platform_record.get("python_version")
    ):
        raise DataError("compiled artifact compiler provenance is invalid")

    if compilation.get("file_hash_scope") != (
        "all artifact files except gem16_compilation.json; its self-hash "
        "is supplied by the external artifact lock in M08"
    ):
        raise DataError("compiled artifact file-hash scope is invalid")
    files = _manifest_file_map(compilation)
    expected_files = set(files) | {COMPILATION_MANIFEST}
    actual_files = _actual_artifact_files(artifact)
    if actual_files != expected_files:
        raise DataError(
            f"compiled artifact file set mismatch: "
            f"missing={sorted(expected_files - actual_files)} "
            f"extra={sorted(actual_files - expected_files)}"
        )
    for relative, record in files.items():
        path = artifact.joinpath(*safe_relative_path(relative, "artifact file").parts)
        size = path.stat().st_size
        digest = workspace.hash_range(path, 0, size)
        if record.get("size") != size or record.get("sha256") != digest:
            raise DataError(f"compiled artifact file hash/size mismatch: {relative}")

    shard_names = tuple(
        sorted(
            relative
            for relative, record in files.items()
            if record.get("kind") == "safetensors_shard"
        )
    )
    indexes = [
        relative
        for relative, record in files.items()
        if record.get("kind") == "safetensors_index"
    ]
    if len(indexes) != 1:
        raise DataError("compiled artifact must declare exactly one Safetensors index")
    try:
        output_tensors = read_artifact_tensors(
            artifact, shard_names, indexes[0], workspace
        )
    except SourceVerificationError as error:
        raise DataError(f"invalid compiled Safetensors artifact: {error}") from error
    expected_outputs = {tensor.output_name: tensor for tensor in plan.tensors}
    if set(output_tensors) != set(expected_outputs):
        raise DataError("compiled output tensor names differ from compiler plan")
    manifest_tensors = _manifest_tensor_map(compilation)
    if set(manifest_tensors) != set(expected_outputs):
        raise DataError("compilation provenance tensor names differ from plan")

    for name, expected in expected_outputs.items():
        output = output_tensors[name]
        record = manifest_tensors[name]
        source_records = []
        for source_name in expected.source_names:
            source_tensor = source_tensors[source_name]
            source_hash = workspace.hash_tensor_range(
                source_tensor.path,
                source_tensor.absolute_offset,
                source_tensor.byte_length,
            )
            source_records.append(
                {
                    "name": source_tensor.name,
                    "sha256": source_hash,
                    "range": tensor_source_identity(source_tensor),
                }
            )
        output_hash = workspace.hash_tensor_range(
            output.path, output.absolute_offset, output.byte_length
        )
        expected_record = {
            "output_name": expected.output_name,
            "operation_id": expected.operation_id,
            "output_dtype": expected.output_dtype,
            "physical_shape": list(expected.physical_shape),
            "logical_dtype": expected.logical_dtype,
            "logical_shape": list(expected.logical_shape),
            "byte_length": expected.output_bytes,
            "sha256": output_hash,
            "output_shard": output.shard,
            "output_data_offsets": [output.data_offset, output.data_offset + output.byte_length],
            "sources": source_records,
            "transformation": expected.transformation,
            "transformation_version": expected.transformation_version,
            "axis_transformation": expected.axis_transformation,
            "quantizer_parameters": expected.quantizer_parameters,
            "dequantization_equation": expected.dequantization_equation,
            "role": expected.role,
            "residency_class": expected.residency_class,
            "disk_layout": expected.disk_layout,
            "runtime_layout": expected.runtime_layout,
            "aliased": expected.aliased,
        }
        if output.dtype != expected.output_dtype or output.shape != expected.physical_shape:
            raise DataError(f"compiled output dtype/shape mismatch: {name}")
        if record != expected_record:
            raise DataError(f"compiled tensor provenance mismatch: {name}")

    excluded_values = compilation.get("excluded_tensors")
    if not isinstance(excluded_values, list):
        raise DataError("compiled artifact excluded_tensors must be an array")
    excluded_by_name = {
        value.get("source_name"): value
        for value in excluded_values
        if isinstance(value, dict) and isinstance(value.get("source_name"), str)
    }
    if len(excluded_by_name) != len(excluded_values) or set(excluded_by_name) != {
        item.source_name for item in plan.excluded_tensors
    }:
        raise DataError("compiled artifact exclusion set differs from plan")
    for item in plan.excluded_tensors:
        source_tensor = source_tensors[item.source_name]
        source_hash = workspace.hash_tensor_range(
            source_tensor.path,
            source_tensor.absolute_offset,
            source_tensor.byte_length,
        )
        expected_excluded = {
            "source_name": item.source_name,
            "source_dtype": source_tensor.dtype,
            "source_shape": list(source_tensor.shape),
            "source_bytes": source_tensor.byte_length,
            "source_tensor_sha256": source_hash,
            "source_range": tensor_source_identity(source_tensor),
            "family": item.family,
            "role": item.role,
            "residency_class": item.residency_class,
            "reason": item.reason,
        }
        if excluded_by_name[item.source_name] != expected_excluded:
            raise DataError(f"compiled exclusion provenance mismatch: {item.source_name}")

    for relative in plan.approved_metadata_files:
        source_file = source.files[relative]
        output_file = artifact.joinpath(
            *safe_relative_path(relative, "approved metadata").parts
        )
        output_hash = workspace.hash_range(output_file, 0, output_file.stat().st_size)
        if output_hash != source_file.sha256:
            raise DataError(f"compiled metadata differs from locked source: {relative}")
        record = files.get(relative)
        if record is None or record.get("kind") != "source_metadata_copy":
            raise DataError(f"compiled metadata file record is missing: {relative}")

    index_document = load_json(artifact / indexes[0], 256 * 1024 * 1024)
    metadata = index_document.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("total_size") != plan.output_tensor_bytes:
        raise DataError("compiled Safetensors index total_size mismatch")
    totals = compilation.get("byte_totals")
    expected_totals = {
        "source_tensor_count": len(source_tensors),
        "output_tensor_count": len(plan.tensors),
        "output_tensor_bytes": plan.output_tensor_bytes,
        "excluded_tensor_count": len(plan.excluded_tensors),
        "excluded_tensor_bytes": sum(
            source_tensors[item.source_name].byte_length
            for item in plan.excluded_tensors
        ),
    }
    if totals != expected_totals:
        raise DataError("compiled artifact byte totals do not reconcile")
    if compilation.get("omitted_families") != list(plan.omitted_families):
        raise DataError("compiled artifact omitted_families mismatch")
    expected_groups: dict[str, dict[str, Any]] = {
        family: {
            "group": family,
            "reason": "absent from source and excluded by text-only profile",
            "source_tensor_count": 0,
            "source_bytes": 0,
        }
        for family in plan.omitted_families
    }
    for item in plan.excluded_tensors:
        group = expected_groups[item.family]
        group["reason"] = item.reason
        group["source_tensor_count"] += 1
        group["source_bytes"] += source_tensors[item.source_name].byte_length
    if compilation.get("omitted_tensor_groups") != [
        expected_groups[name] for name in sorted(expected_groups)
    ]:
        raise DataError("compiled artifact omitted tensor groups do not reconcile")
    settings = compilation.get("compiler_settings")
    if not isinstance(settings, dict) or set(settings) != {
        "threads",
        "host_memory_cap_bytes",
        "staging_buffer_bytes",
        "reference_platform_strict",
        "resume_policy",
    } or (
        settings.get("threads") != 1
        or not isinstance(settings.get("host_memory_cap_bytes"), int)
        or settings.get("host_memory_cap_bytes", 0) <= 0
        or not isinstance(settings.get("staging_buffer_bytes"), int)
        or settings.get("staging_buffer_bytes", 0) < 4096
        or not isinstance(settings.get("reference_platform_strict"), bool)
        or settings.get("resume_policy") != "restart-only"
    ):
        raise DataError("compiled artifact compiler settings are invalid")
    workspace.check("compiled artifact verification")
    return compilation


def verify_artifact(
    request: CompilerRequest,
    artifact: Path,
    *,
    report_path: Path | None = None,
    identity: CompilerIdentity | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    workspace = BoundedWorkspace(
        request.host_memory_cap_bytes, request.staging_bytes
    )
    source, source_tensors, plan, dependency_hash = _load_request(request, workspace)
    compiler_identity = identity or CompilerIdentity.from_repository()
    if request.reference_platform_strict:
        _validate_reference_environment(plan, compiler_identity)
    compilation = _verify_loaded_artifact(
        artifact.resolve(strict=True),
        source,
        source_tensors,
        plan,
        dependency_hash,
        workspace,
    )
    _reverify_source_files(request, source, workspace)
    report = _base_report("verify", request, workspace, started)
    report.update(
        {
            "artifact": str(artifact),
            "source_lock_sha256": source.lock_sha256,
            "compiler_manifest_sha256": plan.compiler_manifest_sha256,
            "resolved_plan_sha256": plan.resolved_plan_sha256,
            "recorded_compiler_commit": compilation["compiler"]["commit"],
            "recorded_compiler_dirty": compilation["compiler"]["dirty"],
            "compilation_manifest_sha256": workspace.hash_range(
                artifact / COMPILATION_MANIFEST,
                0,
                (artifact / COMPILATION_MANIFEST).stat().st_size,
            ),
            "output_tensor_count": len(plan.tensors),
            "output_tensor_bytes": plan.output_tensor_bytes,
            "memory": workspace.telemetry(),
        }
    )
    _write_report(report_path, report)
    return report


def _directory_hashes(root: Path) -> dict[str, dict[str, Any]]:
    try:
        resolved = root.resolve(strict=True)
    except OSError as error:
        raise ReproducibilityError(f"cannot resolve artifact {root}: {error}") from error
    if not resolved.is_dir():
        raise ReproducibilityError(f"artifact is not a directory: {root}")
    result: dict[str, dict[str, Any]] = {}
    buffer = bytearray(1024 * 1024)
    view = memoryview(buffer)
    for path in sorted(resolved.rglob("*")):
        if path.is_symlink():
            raise ReproducibilityError(f"artifact contains symlink: {path}")
        if path.is_dir():
            continue
        digest = __import__("hashlib").sha256()
        size = 0
        try:
            with path.open("rb", buffering=0) as stream:
                while True:
                    read = stream.readinto(view)
                    if not read:
                        break
                    digest.update(view[:read])
                    size += read
        except OSError as error:
            raise ReproducibilityError(f"cannot hash {path}: {error}") from error
        result[path.relative_to(resolved).as_posix()] = {
            "size": size,
            "sha256": digest.hexdigest(),
        }
    return result


def compare_reproducibility(
    left: Path, right: Path, *, report_path: Path | None = None
) -> dict[str, Any]:
    started = time.monotonic()
    left_hashes = _directory_hashes(left)
    right_hashes = _directory_hashes(right)
    all_paths = sorted(set(left_hashes) | set(right_hashes))
    mismatches = [
        {
            "path": path,
            "left": left_hashes.get(path),
            "right": right_hashes.get(path),
        }
        for path in all_paths
        if left_hashes.get(path) != right_hashes.get(path)
    ]
    report = {
        "schema_version": 1,
        "milestone": "M04",
        "action": "compare-reproducibility",
        "status": "pass" if not mismatches else "fail",
        "duration_seconds": round(time.monotonic() - started, 6),
        "file_count": len(all_paths),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:1],
        "files": left_hashes if not mismatches else {},
    }
    _write_report(report_path, report)
    if mismatches:
        first = mismatches[0]
        raise ReproducibilityError(
            f"artifact reproducibility mismatch: {first['path']}"
        )
    return report

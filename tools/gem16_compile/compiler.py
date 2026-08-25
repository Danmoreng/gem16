"""Deterministic checkpoint compiler orchestration, provenance, and verification."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import struct
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
from .native_fp8 import (
    NativeBundleEncoder,
    NativeRequest,
    PROTOCOL as M05_PROTOCOL,
    _native_build,
    prepare_native_bundle,
)
from .native_nvfp4 import (
    NativeNvfp4Preflight,
    NativeNvfp4Request,
    PROTOCOL as M06_PROTOCOL,
    preflight_native_nvfp4,
    prepare_native_direct,
)
from .plan import QuantizationPlan, load_quantization_plan, plan_summary
from .profiles import profile_for
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
    safetensors_header,
    write_shards,
    prepare_direct_shards,
    finalize_direct_shards,
    finalize_mixed_shards,
    compiled_config_bytes,
)


COMPILATION_MANIFEST = "gem16_compilation.json"
DEPENDENCIES_LOCK = Path(__file__).with_name("dependencies.lock.json")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COMPLETE_HYBRID_PROFILES = frozenset({
    "sm120-text-hybrid-v1", "sm120-mtp-assistant-hybrid-v1",
})


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
    native_encoder: Path | None = None
    native_fp8_encoder: Path | None = None
    native_nvfp4_encoder: Path | None = None
    artifact_lock: Path | None = None
    native_timeout_seconds: int = 600


def _validate_dependencies_lock(path: Path) -> tuple[dict[str, Any], str]:
    document = load_json(path, 1024 * 1024)
    if document.get("schema_version") != 1:
        raise InvalidPlanError("unsupported compiler dependency-lock schema")
    if document.get("runtime") != "python-standard-library-only":
        raise InvalidPlanError("compiler dependencies must be standard-library-only")
    if document.get("external_packages") != []:
        raise InvalidPlanError("M04 compiler dependency lock contains external packages")
    payload = path.read_bytes()
    return document, sha256_bytes(payload)


def _validate_request(request: CompilerRequest) -> None:
    if request.threads < 1 or request.threads > 64:
        raise InvalidPlanError("--threads must be in the range 1..64")
    if request.profile == "synthetic-copy-v1":
        if request.threads != 1:
            raise InvalidPlanError("M04 supports exactly one deterministic thread")
        if request.native_encoder is not None:
            raise InvalidPlanError("M04 does not accept a native encoder")
    if request.native_timeout_seconds <= 0:
        raise InvalidPlanError("--native-timeout-seconds must be positive")
    if request.shard_size is not None and request.shard_size <= 0:
        raise InvalidPlanError("--shard-size must be positive")


def _plan_with_encoders(
    plan: QuantizationPlan, encoders: frozenset[str]
) -> QuantizationPlan:
    selected = tuple(tensor for tensor in plan.tensors if tensor.encoder in encoders)
    if not selected:
        raise InvalidPlanError(f"compiler plan has no tensors for encoders: {sorted(encoders)}")
    return replace(plan, tensors=selected, excluded_tensors=())


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
    for item in plan.excluded_tensors:
        family_totals.setdefault(
            item.family,
            {
                "group": item.family,
                "reason": item.reason,
                "source_tensor_count": 0,
                "source_bytes": 0,
            },
        )
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
    native_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    excluded, omitted_groups = _excluded_records(plan, source_tensors, workspace)
    profile = profile_for(plan.artifact_profile, plan.head_format)
    files: list[dict[str, Any]] = []
    for name in payload.shard_names:
        files.append(_file_record(staging, name, "safetensors_shard", workspace))
    files.append(_file_record(staging, payload.index_name, "safetensors_index", workspace))
    for name in sorted(payload.metadata_names):
        kind = (
            "generated_config"
            if plan.artifact_profile in COMPLETE_HYBRID_PROFILES and name == "config.json"
            else "source_metadata_copy"
        )
        files.append(_file_record(staging, name, kind, workspace))
    compiler_record: dict[str, Any] = {
        "repository": identity.repository,
        "commit": identity.commit,
        "dirty": identity.dirty,
        "python": identity.environment["python_version"],
        "platform": identity.environment,
        "dependencies_lock_sha256": dependency_hash,
        "implementation": profile.compiler_implementation,
    }
    if plan.artifact_profile in {"fp8-attention-partial-v1", "nvfp4-experts-partial-v1", "nvfp4-tied-head-partial-v1"}:
        if native_identity is None:
            raise InvalidPlanError("native compilation is missing execution identity")
        compiler_record["native_encoder"] = dict(native_identity)
    elif plan.artifact_profile in COMPLETE_HYBRID_PROFILES:
        if native_identity is None or set(native_identity) != {"fp8", "nvfp4"}:
            raise InvalidPlanError("M08 compilation is missing both native identities")
        compiler_record["native_encoders"] = {
            name: dict(value) for name, value in sorted(native_identity.items())
        }
    return {
        "schema_version": 1,
        "artifact_profile": plan.artifact_profile,
        "artifact_status": profile.artifact_status,
        "source": {
            "lock_sha256": source.lock_sha256,
            "repository": source.repository,
            "revision": source.revision,
            "resolved_at_utc": source.resolved_at_utc,
        },
        "compiler": compiler_record,
        "plan": {
            "schema_version": plan.schema_version,
            "compiler_manifest_sha256": plan.compiler_manifest_sha256,
            "resolved_plan_sha256": plan.resolved_plan_sha256,
            "source_contract": plan.source_contract,
            "target_shard_bytes": plan.target_shard_bytes,
        },
        "quantization": profile.quantization,
        "head_format": plan.head_format,
        "text_only": True,
        "omitted_families": list(plan.omitted_families),
        "omitted_tensor_groups": omitted_groups,
        "excluded_tensors": excluded,
        "tensors": _tensor_provenance(payload, source_tensors),
        "files": sorted(files, key=lambda value: value["path"]),
        "file_hash_scope": (
            "all artifact files except gem16_compilation.json; its self-hash "
            "is supplied by the external artifact lock"
            if plan.artifact_profile in COMPLETE_HYBRID_PROFILES
            else "all artifact files except gem16_compilation.json; its self-hash "
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


def _canonical_path_allow_missing(path: Path, description: str) -> Path:
    """Resolve every existing path component without creating the missing tail."""
    candidate = path.expanduser().absolute()
    missing: list[str] = []
    while not candidate.exists() and not candidate.is_symlink():
        parent = candidate.parent
        if parent == candidate:
            raise OutputError(f"cannot resolve {description}: {path}")
        missing.append(candidate.name)
        candidate = parent
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise OutputError(f"cannot resolve {description} {path}: {error}") from error
    for name in reversed(missing):
        resolved = resolved / name
    return resolved


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _preflight_report_path(
    report_path: Path | None, output: Path
) -> Path | None:
    if report_path is None:
        return None
    requested_report = report_path.expanduser().absolute()
    if requested_report.exists() or requested_report.is_symlink():
        raise OutputError(f"compiler report already exists: {requested_report}")

    canonical_report = _canonical_path_allow_missing(
        requested_report, "compiler report path"
    )
    canonical_output = _canonical_path_allow_missing(output, "compiler output path")
    staging = output.with_name(output.name + ".incomplete")
    canonical_staging = _canonical_path_allow_missing(
        staging, "compiler staging output path"
    )
    for candidate, label in (
        (canonical_output, "output"),
        (canonical_staging, "staging output"),
    ):
        if _path_is_within(canonical_report, candidate):
            raise OutputError(
                f"compiler report must not be inside {label}: {requested_report}"
            )

    if canonical_report.exists() or canonical_report.is_symlink():
        raise OutputError(f"compiler report already exists: {requested_report}")
    try:
        canonical_report.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OutputError(
            f"cannot create compiler report directory {canonical_report.parent}: {error}"
        ) from error
    return canonical_report


def _write_report(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    if path.exists() or path.is_symlink():
        raise OutputError(f"compiler report already exists: {path}")
    write_file_atomic(path, canonical_json_bytes(report))


def _external_lock_document(
    artifact: Path,
    compilation: dict[str, Any],
    workspace: BoundedWorkspace,
) -> dict[str, Any]:
    files = []
    for relative in sorted(_actual_artifact_files(artifact)):
        path = artifact.joinpath(*safe_relative_path(relative, "artifact lock file").parts)
        size = path.stat().st_size
        files.append({
            "path": relative,
            "size": size,
            "sha256": workspace.hash_range(path, 0, size),
        })
    source = compilation.get("source")
    compiler = compilation.get("compiler")
    if not isinstance(source, dict) or not isinstance(compiler, dict):
        raise DataError("M08 compilation provenance is incomplete for external lock")
    content = {
        "artifact_profile": compilation.get("artifact_profile"),
        "artifact_status": compilation.get("artifact_status"),
        "source_lock_sha256": source.get("lock_sha256"),
        "compiler_commit": compiler.get("commit"),
        "files": files,
    }
    return {
        "schema_version": 1,
        **content,
        "artifact_content_sha256": sha256_bytes(canonical_json_bytes(content)),
    }


def _verify_external_artifact_lock(
    artifact: Path,
    lock_path: Path,
    compilation: dict[str, Any],
    workspace: BoundedWorkspace,
) -> str:
    try:
        if lock_path.is_symlink() or not lock_path.is_file():
            raise DataError(f"M08 external artifact lock is missing or unsafe: {lock_path}")
        actual = load_json(lock_path, 16 * 1024 * 1024)
    except OSError as error:
        raise DataError(f"cannot inspect M08 external artifact lock: {error}") from error
    expected = _external_lock_document(artifact, compilation, workspace)
    if actual != expected:
        raise DataError("M08 external artifact lock does not bind the artifact")
    payload = canonical_json_bytes(actual)
    if lock_path.read_bytes() != payload:
        raise DataError("M08 external artifact lock is not canonical JSON")
    return sha256_bytes(payload)


def _complete_statistics_record(written: Any) -> dict[str, Any]:
    result = written.source_result
    return {
        "output_name": written.plan.output_name,
        "operation_id": written.plan.operation_id,
        "source_names": list(written.plan.source_names),
        "source_sha256": list(result.source_sha256),
        "logical_dtype": written.plan.logical_dtype,
        "logical_shape": list(written.plan.logical_shape),
        "quantizer_parameters": written.plan.quantizer_parameters,
        "statistics": result.statistics,
    }


def _base_report(
    action: str,
    request: CompilerRequest,
    workspace: BoundedWorkspace,
    started: float,
) -> dict[str, Any]:
    profile = profile_for(request.profile, request.head_format)
    report = {
        "schema_version": 1,
        "milestone": profile.milestone,
        "action": action,
        "status": "pass",
        "artifact_profile": request.profile,
        "duration_seconds": round(time.monotonic() - started, 6),
        "memory": workspace.telemetry(),
    }
    if request.profile == "fp8-attention-partial-v1" and action == "verify":
        report["transformation_recomputed"] = False
    return report


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
    compiler_identity = identity or CompilerIdentity.from_repository()
    if compiler_identity.dirty and not allow_dirty:
        raise InvalidPlanError(
            "release compiler requires a clean repository; "
            "--allow-dirty is diagnostic only"
        )
    native_preflight: NativeNvfp4Preflight | None = None
    if request.profile in COMPLETE_HYBRID_PROFILES and (
        request.native_fp8_encoder is None or request.native_nvfp4_encoder is None
    ):
        raise InvalidPlanError(
            "M08 compilation requires --native-fp8-encoder and --native-nvfp4-encoder"
        )
    if request.profile in {"nvfp4-experts-partial-v1", "nvfp4-tied-head-partial-v1"}:
        if request.native_encoder is None:
            raise InvalidPlanError("M06/M07 compilation requires --native-encoder")
        native_preflight = preflight_native_nvfp4(
            NativeNvfp4Request(
                request.native_encoder,
                request.native_timeout_seconds,
                request.threads,
                request.profile,
            ),
            workspace,
        )
    elif request.profile in COMPLETE_HYBRID_PROFILES:
        assert request.native_nvfp4_encoder is not None
        native_preflight = preflight_native_nvfp4(
            NativeNvfp4Request(
                request.native_nvfp4_encoder,
                request.native_timeout_seconds,
                request.threads,
                request.profile,
            ),
            workspace,
        )
    source, source_tensors, plan, dependency_hash = _load_request(request, workspace)
    if request.reference_platform_strict:
        _validate_reference_environment(plan, compiler_identity)
    output = output.expanduser().absolute()
    report_path = _preflight_report_path(report_path, output)
    artifact_lock_path: Path | None = None
    if request.artifact_lock is not None:
        artifact_lock_path = request.artifact_lock.expanduser().absolute()
        if artifact_lock_path.exists() or artifact_lock_path.is_symlink():
            raise OutputError(f"external artifact lock already exists: {artifact_lock_path}")
        canonical_lock = _canonical_path_allow_missing(
            artifact_lock_path, "external artifact lock"
        )
        canonical_output = _canonical_path_allow_missing(output, "compiler output")
        if _path_is_within(canonical_lock, canonical_output):
            raise OutputError("M08 artifact lock must be external to the artifact directory")
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OutputError(f"cannot create compiler output parent {output.parent}: {error}") from error
    staging = create_staging_directory(output)
    report_written = False
    native_bundle = None
    native_direct = None
    native_identity: dict[str, Any] | None = None
    native_runtime: dict[str, int | float] | None = None
    direct_layout = None
    published = False
    artifact_lock_written = False
    try:
        if plan.artifact_profile == "fp8-attention-partial-v1":
            if request.native_encoder is None:
                raise InvalidPlanError("M05 compilation requires --native-encoder")
            native_bundle = prepare_native_bundle(
                NativeRequest(
                    request.native_encoder,
                    request.native_timeout_seconds,
                    request.threads,
                ),
                plan,
                source_tensors,
                workspace,
                staging,
            )
            native_identity = {
                "protocol": M05_PROTOCOL,
                "sha256": native_bundle.binary_sha256,
                "threads": request.threads,
                "build": dict(native_bundle.native_build),
            }
            native_runtime = {"child_peak_rss_bytes": native_bundle.child_peak_rss_bytes}
        if plan.artifact_profile in {"nvfp4-experts-partial-v1", "nvfp4-tied-head-partial-v1"}:
            if request.native_encoder is None:
                raise InvalidPlanError("M06/M07 compilation requires --native-encoder")
            direct_layout = prepare_direct_shards(staging, plan, workspace)
            native_direct = prepare_native_direct(
                NativeNvfp4Request(
                    request.native_encoder,
                    request.native_timeout_seconds,
                    request.threads,
                    request.profile,
                ), plan, source_tensors, workspace, staging, direct_layout,
                expected_preflight=native_preflight,
            )
            native_identity = {
                "protocol": M06_PROTOCOL,
                "sha256": native_direct.binary_sha256,
                "threads": request.threads,
                "build": dict(native_direct.native_build),
            }
            native_runtime = {
                "child_peak_rss_bytes": native_direct.child_peak_rss_bytes,
                "maximum_source_row_bytes": native_direct.maximum_source_row_bytes,
                "source_passes": native_direct.source_passes,
                "analysis_seconds": native_direct.analysis_seconds,
                "conversion_seconds": native_direct.conversion_seconds,
            }
            payload = finalize_direct_shards(
                staging, plan, direct_layout, native_direct, workspace
            )
        elif plan.artifact_profile in COMPLETE_HYBRID_PROFILES:
            assert request.native_fp8_encoder is not None
            assert request.native_nvfp4_encoder is not None
            fp8_plan = _plan_with_encoders(
                plan, frozenset({"fp8-rowwise-weight-v1", "fp8-rowwise-scale-v1"})
            )
            nvfp4_plan = _plan_with_encoders(
                plan,
                frozenset({
                    "nvfp4-packed-v1", "nvfp4-local-scale-v1",
                    "nvfp4-weight-divisor-v1", "nvfp4-input-divisor-v1",
                }),
            )
            native_bundle = prepare_native_bundle(
                NativeRequest(
                    request.native_fp8_encoder,
                    request.native_timeout_seconds,
                    request.threads,
                ),
                fp8_plan,
                source_tensors,
                workspace,
                staging,
            )
            direct_layout = prepare_direct_shards(staging, plan, workspace)
            native_direct = prepare_native_direct(
                NativeNvfp4Request(
                    request.native_nvfp4_encoder,
                    request.native_timeout_seconds,
                    request.threads,
                    request.profile,
                ),
                nvfp4_plan,
                source_tensors,
                workspace,
                staging,
                direct_layout,
                expected_preflight=native_preflight,
            )
            native_identity = {
                "fp8": {
                    "protocol": M05_PROTOCOL,
                    "sha256": native_bundle.binary_sha256,
                    "threads": request.threads,
                    "build": dict(native_bundle.native_build),
                },
                "nvfp4": {
                    "protocol": M06_PROTOCOL,
                    "sha256": native_direct.binary_sha256,
                    "threads": request.threads,
                    "build": dict(native_direct.native_build),
                },
            }
            native_runtime = {
                "fp8_child_peak_rss_bytes": native_bundle.child_peak_rss_bytes,
                "nvfp4_child_peak_rss_bytes": native_direct.child_peak_rss_bytes,
                "nvfp4_maximum_source_row_bytes": native_direct.maximum_source_row_bytes,
                "nvfp4_source_passes": native_direct.source_passes,
                "nvfp4_analysis_seconds": native_direct.analysis_seconds,
                "nvfp4_conversion_seconds": native_direct.conversion_seconds,
            }
            payload = finalize_mixed_shards(
                staging,
                plan,
                source_tensors,
                direct_layout,
                native_direct,
                native_bundle,
                workspace,
                encoders or default_encoder_registry(),
            )
        else:
            payload = write_shards(
                staging, plan, source_tensors, workspace,
                encoders or default_encoder_registry(), native_bundle=native_bundle,
            )
        if native_bundle is not None:
            native_bundle.cleanup()
            native_bundle = None
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
            native_identity=native_identity,
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
            expected_threads=request.threads if plan.artifact_profile in {"fp8-attention-partial-v1", "nvfp4-experts-partial-v1", "nvfp4-tied-head-partial-v1", *COMPLETE_HYBRID_PROFILES} else 1,
        )
        _reverify_source_files(request, source, workspace)

        report = _base_report("compile", request, workspace, started)
        statistics = [
            _complete_statistics_record(written)
            for written in payload.tensors
            if written.source_result.statistics is not None
        ]
        report.update(
            {
                "output": str(output),
                "compiler_commit": compiler_identity.commit,
                "compiler_dirty": compiler_identity.dirty,
                "source_lock_sha256": source.lock_sha256,
                "compiler_manifest_sha256": plan.compiler_manifest_sha256,
                "resolved_plan_sha256": plan.resolved_plan_sha256,
                "compilation_manifest_sha256": workspace.hash_range(
                    staging / COMPILATION_MANIFEST,
                    0,
                    (staging / COMPILATION_MANIFEST).stat().st_size,
                ),
                "output_tensor_count": len(plan.tensors),
                "output_tensor_bytes": plan.output_tensor_bytes,
                "output_file_count": len(compilation["files"]) + 1,
                "memory": workspace.telemetry(),
            }
        )
        if native_identity is not None:
            if plan.artifact_profile in COMPLETE_HYBRID_PROFILES:
                report["native_encoders"] = native_identity
                report["native_builds"] = {
                    name: dict(value["build"])
                    for name, value in native_identity.items()
                }
            else:
                report["native_encoder"] = native_identity
                report["native_build"] = dict(native_identity["build"])
        if native_runtime is not None:
            report["native_runtime"] = native_runtime
        if statistics:
            report_key = (
                "tensor_statistics"
                if plan.artifact_profile in COMPLETE_HYBRID_PROFILES
                else "fp8_tensor_statistics"
                if plan.artifact_profile == "fp8-attention-partial-v1"
                else "nvfp4_tensor_statistics"
            )
            report[report_key] = statistics
        artifact_lock_payload: bytes | None = None
        if plan.artifact_profile in COMPLETE_HYBRID_PROFILES:
            if artifact_lock_path is None:
                raise InvalidPlanError("M08 external artifact lock path is missing")
            artifact_lock_document = _external_lock_document(
                staging, compilation, workspace
            )
            artifact_lock_payload = canonical_json_bytes(artifact_lock_document)
            report["artifact_lock_sha256"] = sha256_bytes(artifact_lock_payload)
            report["artifact_content_sha256"] = artifact_lock_document[
                "artifact_content_sha256"
            ]
        publish_staging(staging, output)
        published = True
        if artifact_lock_payload is not None:
            assert artifact_lock_path is not None
            artifact_lock_path.parent.mkdir(parents=True, exist_ok=True)
            write_file_atomic(artifact_lock_path, artifact_lock_payload)
            artifact_lock_written = True
        _write_report(report_path, report)
        report_written = report_path is not None
        return report
    except Exception:
        if native_bundle is not None:
            native_bundle.cleanup()
        if not published:
            discard_staging(staging)
        if report_written and report_path is not None:
            try:
                report_path.unlink(missing_ok=True)
            except OSError:
                pass
        if artifact_lock_written and artifact_lock_path is not None and not published:
            try:
                artifact_lock_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


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


def _read_exact_path(path: Path, length: int, description: str) -> bytes:
    if length < 0:
        raise DataError(f"invalid byte length for {description}: {length}")
    try:
        with path.open("rb", buffering=0) as stream:
            chunks: list[bytes] = []
            remaining = length
            while remaining:
                chunk = stream.read(remaining)
                if not chunk:
                    raise DataError(f"short read for {description}")
                chunks.append(chunk)
                remaining -= len(chunk)
            return b"".join(chunks)
    except DataError:
        raise
    except OSError as error:
        raise DataError(f"cannot read {description}: {error}") from error


def _canonical_index_bytes(plan: QuantizationPlan) -> bytes:
    assignments = assign_shards(plan)
    weight_map = {
        tensor.output_name: assignment.name
        for assignment in assignments
        for tensor in assignment.tensors
    }
    return canonical_json_bytes({
        "metadata": {"total_size": plan.output_tensor_bytes},
        "weight_map": dict(sorted(weight_map.items())),
    })


def _validate_canonical_layout(
    artifact: Path,
    plan: QuantizationPlan,
    source: VerifiedSource,
    workspace: BoundedWorkspace,
) -> tuple[dict[str, dict[str, Any]], dict[str, TensorDescriptor], tuple[str, ...], str]:
    assignments = assign_shards(plan)
    profile = profile_for(plan.artifact_profile, plan.head_format)
    expected_paths = {
        assignment.name for assignment in assignments
    } | {"model.safetensors.index.json"} | set(plan.approved_metadata_files)
    actual_paths = _actual_artifact_files(artifact) - {COMPILATION_MANIFEST}
    if actual_paths != expected_paths:
        raise DataError(
            "compiled artifact file set differs from canonical plan: "
            f"missing={sorted(expected_paths - actual_paths)} "
            f"extra={sorted(actual_paths - expected_paths)}"
        )

    expected_records: dict[str, dict[str, Any]] = {}
    for assignment in assignments:
        header, _ = safetensors_header(
            assignment.tensors, artifact_label=profile.header_label
        )
        path = artifact / assignment.name
        header_size = struct.unpack("<Q", _read_exact_path(
            path, 8, f"{assignment.name} header length"
        ))[0]
        if header_size != len(header) - 8:
            raise DataError(f"canonical header length mismatch: {assignment.name}")
        if _read_exact_path(path, len(header), f"{assignment.name} header") != header:
            raise DataError(f"canonical Safetensors header mismatch: {assignment.name}")
        size = path.stat().st_size
        expected_records[assignment.name] = {
            "path": assignment.name,
            "kind": "safetensors_shard",
            "size": size,
            "sha256": workspace.hash_range(path, 0, size),
        }

    index_path = artifact / "model.safetensors.index.json"
    expected_index = _canonical_index_bytes(plan)
    if _read_exact_path(index_path, len(expected_index), "Safetensors index") != expected_index:
        raise DataError("canonical Safetensors index mismatch")
    if index_path.stat().st_size != len(expected_index):
        raise DataError("Safetensors index has trailing bytes")
    expected_records[index_path.name] = {
        "path": index_path.name,
        "kind": "safetensors_index",
        "size": len(expected_index),
        "sha256": workspace.hash_range(index_path, 0, len(expected_index)),
    }
    for relative in plan.approved_metadata_files:
        path = artifact.joinpath(*safe_relative_path(relative, "approved metadata").parts)
        source_file = source.files[relative]
        generated_config = (
            plan.artifact_profile in COMPLETE_HYBRID_PROFILES
            and relative == "config.json"
        )
        expected_payload = (
            compiled_config_bytes(source_file.path.read_bytes(), plan.artifact_profile)
            if generated_config else None
        )
        expected_size = len(expected_payload) if expected_payload is not None else source_file.size
        if path.stat().st_size != expected_size:
            raise DataError(f"compiled metadata size differs from expected: {relative}")
        digest = workspace.hash_range(path, 0, path.stat().st_size)
        expected_digest = (
            sha256_bytes(expected_payload) if expected_payload is not None else source_file.sha256
        )
        if digest != expected_digest:
            raise DataError(f"compiled metadata differs from expected: {relative}")
        expected_records[relative] = {
            "path": relative,
            "kind": "generated_config" if generated_config else "source_metadata_copy",
            "size": expected_size,
            "sha256": expected_digest,
        }
    shard_names = tuple(assignment.name for assignment in assignments)
    try:
        output_tensors = read_artifact_tensors(
            artifact, shard_names, "model.safetensors.index.json", workspace
        )
    except SourceVerificationError as error:
        raise DataError(f"invalid compiled Safetensors artifact: {error}") from error
    return expected_records, output_tensors, shard_names, profile.header_label


def _verify_loaded_artifact(
    artifact: Path,
    source: VerifiedSource,
    source_tensors: dict[str, TensorDescriptor],
    plan: QuantizationPlan,
    dependency_hash: str,
    workspace: BoundedWorkspace,
    *,
    expected_threads: int,
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
    profile = profile_for(plan.artifact_profile, plan.head_format)
    if compilation.get("artifact_status") != profile.artifact_status:
        raise DataError("compiled artifact status is missing or incorrect")
    if (
        compilation.get("quantization") != profile.quantization
        or compilation.get("head_format") != plan.head_format
        or compilation.get("text_only") is not True
    ):
        raise DataError("compiled precision/text-only contract mismatch")
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
    expected_compiler_keys = {
        "repository", "commit", "dirty", "python", "platform",
        "dependencies_lock_sha256", "implementation",
    }
    if plan.artifact_profile in {"fp8-attention-partial-v1", "nvfp4-experts-partial-v1", "nvfp4-tied-head-partial-v1"}:
        expected_compiler_keys.add("native_encoder")
    elif plan.artifact_profile in COMPLETE_HYBRID_PROFILES:
        expected_compiler_keys.add("native_encoders")
    if not isinstance(compiler_record, dict) or set(compiler_record) != expected_compiler_keys:
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
        or compiler_record.get("implementation") != profile.compiler_implementation
        or not isinstance(platform_record, dict)
        or set(platform_record) != required_platform_fields
        or not all(isinstance(value, str) and value for value in platform_record.values())
        or compiler_record.get("python") != platform_record.get("python_version")
    ):
        raise DataError("compiled artifact compiler provenance is invalid")
    if plan.artifact_profile in {"fp8-attention-partial-v1", "nvfp4-experts-partial-v1", "nvfp4-tied-head-partial-v1"}:
        native_record = compiler_record.get("native_encoder")
        native_protocol = M05_PROTOCOL if plan.artifact_profile == "fp8-attention-partial-v1" else M06_PROTOCOL
        milestone = "M05" if plan.artifact_profile == "fp8-attention-partial-v1" else "M07" if plan.artifact_profile == "nvfp4-tied-head-partial-v1" else "M06"
        if (not isinstance(native_record, dict) or set(native_record) != {"protocol", "sha256", "threads", "build"}
            or native_record.get("protocol") != native_protocol
            or native_record.get("threads") != expected_threads
            or not isinstance(native_record.get("sha256"), str) or len(native_record["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in native_record["sha256"])):
            raise DataError(f"compiled native {milestone} encoder provenance is invalid")
        native_build = _native_build(
            native_record.get("build"), f"compiled native {milestone} build provenance"
        )
        if plan.artifact_profile in {"nvfp4-experts-partial-v1", "nvfp4-tied-head-partial-v1"} and native_build.get("build_type") != "Release":
            raise DataError(f"compiled native {milestone} build must be Release")
    elif plan.artifact_profile in COMPLETE_HYBRID_PROFILES:
        native_records = compiler_record.get("native_encoders")
        if not isinstance(native_records, dict) or set(native_records) != {"fp8", "nvfp4"}:
            raise DataError("compiled native M08 encoder provenance is invalid")
        for name, protocol in (("fp8", M05_PROTOCOL), ("nvfp4", M06_PROTOCOL)):
            native_record = native_records[name]
            if (
                not isinstance(native_record, dict)
                or set(native_record) != {"protocol", "sha256", "threads", "build"}
                or native_record.get("protocol") != protocol
                or native_record.get("threads") != expected_threads
                or not isinstance(native_record.get("sha256"), str)
                or len(native_record["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in native_record["sha256"])
            ):
                raise DataError(f"compiled native M08 {name} provenance is invalid")
            native_build = _native_build(
                native_record.get("build"), f"compiled native M08 {name} build provenance"
            )
            if native_build.get("build_type") != "Release":
                raise DataError(f"compiled native M08 {name} build must be Release")

    expected_hash_scope = (
        "all artifact files except gem16_compilation.json; its self-hash "
        "is supplied by the external artifact lock"
        if plan.artifact_profile in COMPLETE_HYBRID_PROFILES
        else "all artifact files except gem16_compilation.json; its self-hash "
             "is supplied by the external artifact lock in M08"
    )
    if compilation.get("file_hash_scope") != expected_hash_scope:
        raise DataError("compiled artifact file-hash scope is invalid")
    expected_file_records, output_tensors, shard_names, _ = _validate_canonical_layout(
        artifact, plan, source, workspace
    )
    files = _manifest_file_map(compilation)
    if files != expected_file_records:
        raise DataError("compiled artifact file records differ from canonical layout")
    expected_outputs = {tensor.output_name: tensor for tensor in plan.tensors}
    if set(output_tensors) != set(expected_outputs):
        raise DataError("compiled output tensor names differ from compiler plan")
    manifest_tensors = _manifest_tensor_map(compilation)
    if set(manifest_tensors) != set(expected_outputs):
        raise DataError("compilation provenance tensor names differ from plan")

    source_hashes: dict[str, str] = {}
    for name, expected in expected_outputs.items():
        output = output_tensors.get(name)
        if output is None:
            raise DataError(f"compiled output tensor is missing: {name}")
        record = manifest_tensors[name]
        if output.dtype != expected.output_dtype or output.shape != expected.physical_shape:
            raise DataError(f"compiled output dtype/shape mismatch: {name}")
        output_hash = workspace.hash_tensor_range(
            output.path, output.absolute_offset, output.byte_length
        )
        if output.byte_length != expected.output_bytes:
            raise DataError(f"compiled output byte count mismatch: {name}")
        source_records = []
        for source_name in expected.source_names:
            source_tensor = source_tensors[source_name]
            source_hash = source_hashes.get(source_name)
            if source_hash is None:
                source_hash = workspace.hash_tensor_range(
                    source_tensor.path,
                    source_tensor.absolute_offset,
                    source_tensor.byte_length,
                )
                source_hashes[source_name] = source_hash
            source_records.append(
                {
                    "name": source_tensor.name,
                    "sha256": source_hash,
                    "range": tensor_source_identity(source_tensor),
                }
            )
        expected_record = {
            "output_name": expected.output_name,
            "operation_id": expected.operation_id,
            "output_dtype": expected.output_dtype,
            "physical_shape": list(expected.physical_shape),
            "logical_dtype": expected.logical_dtype,
            "logical_shape": list(expected.logical_shape),
            "byte_length": output.byte_length,
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
        generated_config = (
            plan.artifact_profile in COMPLETE_HYBRID_PROFILES
            and relative == "config.json"
        )
        expected_hash = (
            sha256_bytes(compiled_config_bytes(
                source_file.path.read_bytes(), plan.artifact_profile
            ))
            if generated_config
            else source_file.sha256
        )
        if output_hash != expected_hash:
            raise DataError(f"compiled metadata differs from expected: {relative}")
        record = files.get(relative)
        expected_kind = "generated_config" if generated_config else "source_metadata_copy"
        if record is None or record.get("kind") != expected_kind:
            raise DataError(f"compiled metadata file record is missing: {relative}")

    index_document = load_json(artifact / "model.safetensors.index.json", 256 * 1024 * 1024)
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
        expected_groups.setdefault(
            item.family,
            {
                "group": item.family,
                "reason": item.reason,
                "source_tensor_count": 0,
                "source_bytes": 0,
            },
        )
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
        (settings.get("threads") != expected_threads)
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
    requested_artifact = artifact.expanduser().absolute()
    if requested_artifact.is_symlink():
        raise SourceVerificationError(
            f"compiled artifact root must not be a symlink: {artifact}"
        )
    try:
        artifact_root = requested_artifact.resolve(strict=True)
    except OSError as error:
        raise SourceVerificationError(
            f"cannot resolve compiled artifact {artifact}: {error}"
        ) from error
    if not artifact_root.is_dir():
        raise SourceVerificationError(
            f"compiled artifact is not a directory: {artifact}"
        )
    report_path = _preflight_report_path(report_path, artifact_root)
    source, source_tensors, plan, dependency_hash = _load_request(request, workspace)
    compiler_identity = identity or CompilerIdentity.from_repository()
    if request.reference_platform_strict:
        _validate_reference_environment(plan, compiler_identity)
    compilation = _verify_loaded_artifact(
        artifact_root,
        source,
        source_tensors,
        plan,
        dependency_hash,
        workspace,
        expected_threads=request.threads if plan.artifact_profile in {"fp8-attention-partial-v1", "nvfp4-experts-partial-v1", "nvfp4-tied-head-partial-v1", *COMPLETE_HYBRID_PROFILES} else 1,
    )
    artifact_lock_sha256: str | None = None
    if plan.artifact_profile in COMPLETE_HYBRID_PROFILES:
        if request.artifact_lock is None:
            raise InvalidPlanError("M08 external artifact lock path is missing")
        artifact_lock_sha256 = _verify_external_artifact_lock(
            artifact_root,
            request.artifact_lock.expanduser().absolute(),
            compilation,
            workspace,
        )
    _reverify_source_files(request, source, workspace)
    report = _base_report("verify", request, workspace, started)
    report.update(
        {
            "artifact": str(artifact_root),
            "source_lock_sha256": source.lock_sha256,
            "compiler_manifest_sha256": plan.compiler_manifest_sha256,
            "resolved_plan_sha256": plan.resolved_plan_sha256,
            "recorded_compiler_commit": compilation["compiler"]["commit"],
            "recorded_compiler_dirty": compilation["compiler"]["dirty"],
            "compilation_manifest_sha256": workspace.hash_range(
                artifact_root / COMPILATION_MANIFEST,
                0,
                (artifact_root / COMPILATION_MANIFEST).stat().st_size,
            ),
            "output_tensor_count": len(plan.tensors),
            "output_tensor_bytes": plan.output_tensor_bytes,
            "memory": workspace.telemetry(),
        }
    )
    if artifact_lock_sha256 is not None:
        report["artifact_lock_sha256"] = artifact_lock_sha256
        report["artifact_content_sha256"] = load_json(
            request.artifact_lock, 16 * 1024 * 1024
        )["artifact_content_sha256"]
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
    milestone = "M04"
    try:
        left_manifest = load_json(left / COMPILATION_MANIFEST, 64 * 1024 * 1024)
        right_manifest = load_json(right / COMPILATION_MANIFEST, 64 * 1024 * 1024)
        if (
            left_manifest.get("artifact_profile") in COMPLETE_HYBRID_PROFILES
            and right_manifest.get("artifact_profile")
            == left_manifest.get("artifact_profile")
        ):
            milestone = (
                "M25" if left_manifest.get("artifact_profile")
                == "sm120-mtp-assistant-hybrid-v1" else "M08"
            )
    except InvalidPlanError:
        # Reproducibility comparison remains useful for deliberately malformed
        # artifacts; schema validation is owned by verify.
        pass
    report = {
        "schema_version": 1,
        "milestone": milestone,
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

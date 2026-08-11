"""Versioned compiler-plan model and exact source-coverage validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import (
    InvalidPlanError,
    canonical_json_bytes,
    checked_shape,
    load_json,
    safe_relative_path,
    sha256_bytes,
    tensor_bytes,
)
from .reader import TensorDescriptor, VerifiedSource


SUPPORTED_PLAN_SCHEMA = 1
SUPPORTED_PROFILE = "synthetic-copy-v1"
SUPPORTED_HEAD_FORMAT = "source"
SUPPORTED_ENCODERS = {"copy-v1"}
EXPECTED_OMITTED_FAMILIES = ("audio", "mtp", "video", "vision")


@dataclass(frozen=True)
class TensorCompilePlan:
    output_name: str
    operation_id: str
    source_names: tuple[str, ...]
    encoder: str
    transformation: str
    transformation_version: int
    output_dtype: str
    physical_shape: tuple[int, ...]
    logical_dtype: str
    logical_shape: tuple[int, ...]
    axis_transformation: str
    quantizer_parameters: dict[str, Any]
    dequantization_equation: str
    role: str
    residency_class: str
    disk_layout: str
    runtime_layout: str
    aliased: bool

    @property
    def output_bytes(self) -> int:
        return tensor_bytes(self.output_dtype, self.physical_shape, self.output_name)


@dataclass(frozen=True)
class ExcludedTensorPlan:
    source_name: str
    family: str
    role: str
    residency_class: str
    reason: str


@dataclass(frozen=True)
class QuantizationPlan:
    schema_version: int
    artifact_profile: str
    head_format: str
    source_contract: str
    compiler_manifest_sha256: str
    resolved_plan_sha256: str
    target_shard_bytes: int
    approved_metadata_files: tuple[str, ...]
    omitted_families: tuple[str, ...]
    tensors: tuple[TensorCompilePlan, ...]
    excluded_tensors: tuple[ExcludedTensorPlan, ...]
    reference_environment: dict[str, str]

    @property
    def output_tensor_bytes(self) -> int:
        return sum(tensor.output_bytes for tensor in self.tensors)


def _required_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise InvalidPlanError(f"{description} must be a non-empty string")
    return value


def _required_integer(value: object, description: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InvalidPlanError(f"{description} must be an integer >= {minimum}")
    return value


def _string_array(value: object, description: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise InvalidPlanError(f"{description} must be an array")
    result = tuple(_required_string(item, description) for item in value)
    if len(set(result)) != len(result):
        raise InvalidPlanError(f"{description} contains duplicates")
    return result


def _tensor_plan(value: object, index: int) -> TensorCompilePlan:
    description = f"compiler tensor plan {index}"
    if not isinstance(value, dict):
        raise InvalidPlanError(f"{description} must be an object")
    output_name = _required_string(value.get("output_name"), f"{description}.output_name")
    source_names = _string_array(value.get("source_names"), f"{description}.source_names")
    if not source_names:
        raise InvalidPlanError(f"{description}.source_names must not be empty")
    encoder = _required_string(value.get("encoder"), f"{description}.encoder")
    if encoder not in SUPPORTED_ENCODERS:
        raise InvalidPlanError(f"unsupported M04 encoder: {encoder}")
    parameters = value.get("quantizer_parameters")
    if not isinstance(parameters, dict):
        raise InvalidPlanError(f"{description}.quantizer_parameters must be an object")
    plan = TensorCompilePlan(
        output_name=output_name,
        operation_id=_required_string(
            value.get("operation_id"), f"{description}.operation_id"
        ),
        source_names=source_names,
        encoder=encoder,
        transformation=_required_string(
            value.get("transformation"), f"{description}.transformation"
        ),
        transformation_version=_required_integer(
            value.get("transformation_version"),
            f"{description}.transformation_version",
            1,
        ),
        output_dtype=_required_string(
            value.get("output_dtype"), f"{description}.output_dtype"
        ),
        physical_shape=checked_shape(
            value.get("physical_shape"), f"{description}.physical_shape"
        ),
        logical_dtype=_required_string(
            value.get("logical_dtype"), f"{description}.logical_dtype"
        ),
        logical_shape=checked_shape(
            value.get("logical_shape"), f"{description}.logical_shape"
        ),
        axis_transformation=_required_string(
            value.get("axis_transformation"),
            f"{description}.axis_transformation",
        ),
        quantizer_parameters=parameters,
        dequantization_equation=_required_string(
            value.get("dequantization_equation"),
            f"{description}.dequantization_equation",
        ),
        role=_required_string(value.get("role"), f"{description}.role"),
        residency_class=_required_string(
            value.get("residency_class"), f"{description}.residency_class"
        ),
        disk_layout=_required_string(
            value.get("disk_layout"), f"{description}.disk_layout"
        ),
        runtime_layout=_required_string(
            value.get("runtime_layout"), f"{description}.runtime_layout"
        ),
        aliased=value.get("aliased") is True,
    )
    if not isinstance(value.get("aliased"), bool):
        raise InvalidPlanError(f"{description}.aliased must be boolean")
    # Force checked output-byte calculation while validating the plan.
    _ = plan.output_bytes
    return plan


def _excluded_plan(value: object, index: int) -> ExcludedTensorPlan:
    description = f"excluded tensor plan {index}"
    if not isinstance(value, dict):
        raise InvalidPlanError(f"{description} must be an object")
    plan = ExcludedTensorPlan(
        source_name=_required_string(
            value.get("source_name"), f"{description}.source_name"
        ),
        family=_required_string(value.get("family"), f"{description}.family"),
        role=_required_string(value.get("role"), f"{description}.role"),
        residency_class=_required_string(
            value.get("residency_class"), f"{description}.residency_class"
        ),
        reason=_required_string(value.get("reason"), f"{description}.reason"),
    )
    if plan.residency_class != "compile_excluded_vision" and plan.family == "vision":
        raise InvalidPlanError("vision exclusion must use compile_excluded_vision")
    return plan


def load_quantization_plan(
    compiler_manifest: Path,
    source: VerifiedSource,
    source_tensors: dict[str, TensorDescriptor],
    profile: str,
    head_format: str,
    shard_size_override: int | None = None,
) -> QuantizationPlan:
    document = load_json(compiler_manifest, 16 * 1024 * 1024)
    raw_bytes = compiler_manifest.read_bytes()
    schema_version = _required_integer(
        document.get("schema_version"), "compiler manifest schema_version", 1
    )
    if schema_version != SUPPORTED_PLAN_SCHEMA:
        raise InvalidPlanError(
            f"unsupported compiler manifest schema_version: {schema_version}"
        )
    artifact_profile = _required_string(
        document.get("artifact_profile"), "compiler manifest artifact_profile"
    )
    if profile != artifact_profile or profile != SUPPORTED_PROFILE:
        raise InvalidPlanError(
            f"unsupported or mismatched M04 profile: cli={profile!r} "
            f"manifest={artifact_profile!r}"
        )
    manifest_head = _required_string(
        document.get("head_format"), "compiler manifest head_format"
    )
    if head_format != manifest_head or head_format != SUPPORTED_HEAD_FORMAT:
        raise InvalidPlanError(
            f"unsupported or mismatched M04 head format: cli={head_format!r} "
            f"manifest={manifest_head!r}"
        )
    expected_source_lock = _required_string(
        document.get("source_lock_sha256"), "compiler manifest source_lock_sha256"
    )
    if expected_source_lock != source.lock_sha256:
        raise InvalidPlanError(
            "compiler manifest is bound to another source lock: "
            f"expected {expected_source_lock}, got {source.lock_sha256}"
        )
    target_shard_bytes = _required_integer(
        document.get("target_shard_bytes"), "compiler manifest target_shard_bytes", 1
    )
    if shard_size_override is not None:
        target_shard_bytes = _required_integer(
            shard_size_override, "--shard-size", 1
        )
    metadata = _string_array(
        document.get("approved_metadata_files"), "approved_metadata_files"
    )
    for relative in metadata:
        safe_relative_path(relative, "approved metadata path")
        if relative not in source.files:
            raise InvalidPlanError(f"approved metadata is not source-locked: {relative}")
        if (
            relative.endswith(".safetensors")
            or relative == "model.safetensors.index.json"
            or relative == "gem16_compilation.json"
        ):
            raise InvalidPlanError(
                f"reserved weight/index/manifest file cannot be metadata: {relative}"
            )
    omitted_families = tuple(
        sorted(_string_array(document.get("omitted_families"), "omitted_families"))
    )
    if omitted_families != EXPECTED_OMITTED_FAMILIES:
        raise InvalidPlanError(
            "M04 omitted_families must be exactly audio, mtp, video, vision"
        )
    tensor_values = document.get("tensors")
    excluded_values = document.get("excluded_tensors")
    if not isinstance(tensor_values, list) or not tensor_values:
        raise InvalidPlanError("compiler manifest tensors must be a non-empty array")
    if not isinstance(excluded_values, list):
        raise InvalidPlanError("compiler manifest excluded_tensors must be an array")
    tensors = tuple(_tensor_plan(value, index) for index, value in enumerate(tensor_values))
    excluded = tuple(
        _excluded_plan(value, index) for index, value in enumerate(excluded_values)
    )
    if tuple(tensor.output_name for tensor in tensors) != tuple(
        sorted(tensor.output_name for tensor in tensors)
    ):
        raise InvalidPlanError("compiler output tensors must be in canonical name order")
    output_names = [tensor.output_name for tensor in tensors]
    if len(set(output_names)) != len(output_names):
        raise InvalidPlanError("compiler manifest contains duplicate output tensor names")

    covered: dict[str, str] = {}
    copy_sources: set[str] = set()
    for tensor in tensors:
        for source_name in tensor.source_names:
            if source_name not in source_tensors:
                raise InvalidPlanError(f"compiler source tensor is absent: {source_name}")
            previous_operation = covered.get(source_name)
            if previous_operation is not None and previous_operation != tensor.operation_id:
                raise InvalidPlanError(
                    f"source tensor is assigned to conflicting operations: {source_name}"
                )
            covered[source_name] = tensor.operation_id
        if tensor.encoder == "copy-v1":
            if len(tensor.source_names) != 1:
                raise InvalidPlanError("copy-v1 requires exactly one source tensor")
            source_name = tensor.source_names[0]
            if source_name in copy_sources:
                raise InvalidPlanError(
                    f"copy-v1 cannot duplicate a source payload: {source_name}"
                )
            copy_sources.add(source_name)
            source_tensor = source_tensors[source_name]
            if (
                tensor.transformation != "identity-copy"
                or tensor.transformation_version != 1
                or tensor.output_dtype != source_tensor.dtype
                or tensor.physical_shape != source_tensor.shape
                or tensor.logical_dtype != source_tensor.dtype
                or tensor.logical_shape != source_tensor.shape
                or tensor.axis_transformation != "identity"
                or tensor.quantizer_parameters
                or tensor.dequantization_equation != "output = source"
            ):
                raise InvalidPlanError(
                    f"copy-v1 semantic contract mismatch: {tensor.output_name}"
                )
            if tensor.output_bytes != source_tensor.byte_length:
                raise InvalidPlanError(
                    f"copy-v1 byte count mismatch: {tensor.output_name}"
                )
    for item in excluded:
        if item.source_name not in source_tensors:
            raise InvalidPlanError(f"excluded source tensor is absent: {item.source_name}")
        if item.source_name in covered:
            raise InvalidPlanError(
                f"source tensor is covered more than once: {item.source_name}"
            )
        covered[item.source_name] = f"excluded:{item.family}"
    if set(covered) != set(source_tensors):
        missing = sorted(set(source_tensors) - set(covered))
        extra = sorted(set(covered) - set(source_tensors))
        raise InvalidPlanError(
            f"compiler source coverage mismatch: missing={missing[:3]} extra={extra[:3]}"
        )

    reference_environment = document.get("reference_environment")
    if not isinstance(reference_environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and value
        for key, value in reference_environment.items()
    ):
        raise InvalidPlanError("reference_environment must map strings to strings")

    source_contract = _required_string(
        document.get("source_contract"), "compiler manifest source_contract"
    )
    compiler_manifest_hash = sha256_bytes(raw_bytes)
    resolved_document = {
        "schema_version": schema_version,
        "artifact_profile": artifact_profile,
        "head_format": manifest_head,
        "source_contract": source_contract,
        "source_lock_sha256": source.lock_sha256,
        "target_shard_bytes": target_shard_bytes,
        "approved_metadata_files": list(metadata),
        "omitted_families": list(omitted_families),
        "tensors": tensor_values,
        "excluded_tensors": excluded_values,
        "reference_environment": reference_environment,
    }
    return QuantizationPlan(
        schema_version=schema_version,
        artifact_profile=artifact_profile,
        head_format=manifest_head,
        source_contract=source_contract,
        compiler_manifest_sha256=compiler_manifest_hash,
        resolved_plan_sha256=sha256_bytes(canonical_json_bytes(resolved_document)),
        target_shard_bytes=target_shard_bytes,
        approved_metadata_files=metadata,
        omitted_families=omitted_families,
        tensors=tensors,
        excluded_tensors=excluded,
        reference_environment=dict(sorted(reference_environment.items())),
    )


def plan_summary(plan: QuantizationPlan) -> dict[str, object]:
    excluded_by_family: dict[str, dict[str, int]] = {}
    for item in plan.excluded_tensors:
        totals = excluded_by_family.setdefault(item.family, {"tensor_count": 0})
        totals["tensor_count"] += 1
    return {
        "schema_version": 1,
        "status": "planned",
        "artifact_profile": plan.artifact_profile,
        "head_format": plan.head_format,
        "compiler_manifest_sha256": plan.compiler_manifest_sha256,
        "resolved_plan_sha256": plan.resolved_plan_sha256,
        "output_tensor_count": len(plan.tensors),
        "output_tensor_bytes": plan.output_tensor_bytes,
        "excluded_tensor_count": len(plan.excluded_tensors),
        "excluded_by_family": dict(sorted(excluded_by_family.items())),
        "target_shard_bytes": plan.target_shard_bytes,
        "encoder_names": sorted({tensor.encoder for tensor in plan.tensors}),
    }

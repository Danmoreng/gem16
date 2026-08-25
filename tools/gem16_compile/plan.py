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
from .profiles import (
    M04_PROFILE,
    M05_APPROVED_SOURCE_LOCKS,
    M05_ATTENTION_TABLE,
    M05_LAYER_COUNT,
    M05_PROFILE,
    M05_DEQUANTIZATION_EQUATION,
    M05_QUANTIZER_PARAMETERS,
    M05_SOURCE_CONTRACT,
    M05_SOURCE_LOCK_SHA256,
    M05_VISION_EXCLUSION_REASON,
    M06_DEFERRED_REASON,
    M06_DEQUANTIZATION_EQUATION,
    M06_PROFILE,
    M06_QUANTIZER_PARAMETERS,
    M06_SOURCE_CONTRACT,
    M06_VISION_EXCLUSION_REASON,
    M06_COMPONENT_LAYOUTS,
    M07_COMPONENT_LAYOUTS,
    M07_DEQUANTIZATION_EQUATION,
    M07_DEFERRED_REASON,
    M07_PROFILE,
    M07_QUANTIZER_PARAMETERS,
    M07_SOURCE_CONTRACT,
    M08_PROFILE,
    M25_PROFILE,
    M25_SOURCE_CONTRACT,
    M25_SOURCE_LOCK_SHA256,
    classify_m05_source,
    m06_component_parameters,
    m06_expected_source_specs,
    m07_component_parameters,
    CompilerProfile,
    profile_for,
)
from .reader import TensorDescriptor, VerifiedSource


SUPPORTED_PLAN_SCHEMA = 1
EXPECTED_OMITTED_FAMILIES = ("audio", "mtp", "video", "vision")
_PLAN_KEYS = frozenset({
    "schema_version", "artifact_profile", "head_format", "source_contract",
    "source_lock_sha256", "target_shard_bytes", "approved_metadata_files",
    "omitted_families", "tensors", "excluded_tensors", "reference_environment",
})
_TENSOR_KEYS = frozenset({
    "output_name", "operation_id", "source_names", "encoder", "transformation",
    "transformation_version", "output_dtype", "physical_shape", "logical_dtype",
    "logical_shape", "axis_transformation", "quantizer_parameters",
    "dequantization_equation", "role", "residency_class", "disk_layout",
    "runtime_layout", "aliased",
})
_EXCLUDED_KEYS = frozenset({"source_name", "family", "role", "residency_class", "reason"})


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


def _tensor_plan(
    value: object, index: int, profile: CompilerProfile
) -> TensorCompilePlan:
    description = f"compiler tensor plan {index}"
    if not isinstance(value, dict):
        raise InvalidPlanError(f"{description} must be an object")
    unknown = set(value) - _TENSOR_KEYS
    if unknown:
        raise InvalidPlanError(f"{description} contains unknown keys: {sorted(unknown)}")
    output_name = _required_string(value.get("output_name"), f"{description}.output_name")
    encoder = _required_string(value.get("encoder"), f"{description}.encoder")
    source_names = _string_array(value.get("source_names"), f"{description}.source_names")
    if not source_names and encoder != "constant-bf16-one-v1":
        raise InvalidPlanError(f"{description}.source_names must not be empty")
    if encoder not in profile.allowed_encoders:
        raise InvalidPlanError(f"unsupported {profile.milestone} encoder: {encoder}")
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
    _ = plan.output_bytes
    return plan


def _excluded_plan(value: object, index: int) -> ExcludedTensorPlan:
    description = f"excluded tensor plan {index}"
    if not isinstance(value, dict):
        raise InvalidPlanError(f"{description} must be an object")
    unknown = set(value) - _EXCLUDED_KEYS
    if unknown:
        raise InvalidPlanError(f"{description} contains unknown keys: {sorted(unknown)}")
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


def _validate_copy_plan(tensor: TensorCompilePlan, source: TensorDescriptor) -> None:
    if (
        tensor.transformation != "identity-copy"
        or tensor.transformation_version != 1
        or tensor.output_dtype != source.dtype
        or tensor.physical_shape != source.shape
        or tensor.logical_dtype != source.dtype
        or tensor.logical_shape != source.shape
        or tensor.axis_transformation != "identity"
        or tensor.quantizer_parameters
        or tensor.dequantization_equation != "output = source"
    ):
        raise InvalidPlanError(f"copy-v1 semantic contract mismatch: {tensor.output_name}")
    if tensor.output_bytes != source.byte_length:
        raise InvalidPlanError(f"copy-v1 byte count mismatch: {tensor.output_name}")


def _validate_m05_pair(
    source_name: str,
    source: TensorDescriptor,
    pair: list[TensorCompilePlan],
) -> None:
    spec = M05_ATTENTION_TABLE.get(source_name)
    if spec is None:
        raise InvalidPlanError(
            "M05 output source is not an approved language attention projection: "
            f"{source_name}"
        )
    if source.dtype != "BF16" or tuple(source.shape) != spec.shape:
        raise InvalidPlanError(
            f"M05 attention source dtype/shape mismatch: {source_name} "
            f"expected BF16 {spec.shape}, got {source.dtype} {source.shape}"
        )
    if source.byte_length != tensor_bytes("BF16", source.shape, source_name):
        raise InvalidPlanError(f"M05 source byte count mismatch: {source_name}")
    if len(pair) != 2:
        raise InvalidPlanError(f"M05 source must have exactly weight and scale outputs: {source_name}")
    rows, columns = source.shape
    stem = source_name.removesuffix(".weight")
    expected_operation = f"fp8-attention:{stem}"
    expected_role = spec.role
    expected_names = {source_name, f"{stem}.weight_scale"}
    if {item.output_name for item in pair} != expected_names:
        raise InvalidPlanError(f"M05 output pair names mismatch: {source_name}")
    if any(item.operation_id != expected_operation for item in pair):
        raise InvalidPlanError(f"M05 operation ID mismatch: {source_name}")
    if any(item.source_names != (source_name,) for item in pair):
        raise InvalidPlanError(f"M05 pair must reference exactly its source: {source_name}")
    for item in pair:
        if (
            item.transformation_version != 1
            or item.axis_transformation != "identity"
            or item.logical_dtype != "BF16"
            or item.aliased
            or item.role != expected_role
            or item.residency_class != "immutable_device_text"
            or item.quantizer_parameters != M05_QUANTIZER_PARAMETERS
            or item.dequantization_equation != M05_DEQUANTIZATION_EQUATION
        ):
            raise InvalidPlanError(f"M05 semantic contract mismatch: {item.output_name}")
    weights = [item for item in pair if item.encoder == "fp8-rowwise-weight-v1"]
    scales = [item for item in pair if item.encoder == "fp8-rowwise-scale-v1"]
    if len(weights) != 1 or len(scales) != 1:
        raise InvalidPlanError(f"M05 pair must contain one weight and one scale encoder: {source_name}")
    weight, scale = weights[0], scales[0]
    if (
        weight.output_name != source_name
        or weight.transformation != "bf16-to-fp8-e4m3fn-rowwise-weight"
        or weight.output_dtype != "F8_E4M3"
        or weight.physical_shape != (rows, columns)
        or weight.logical_shape != (rows, columns)
        or weight.disk_layout != "source_nk_fp8"
        or weight.runtime_layout != "source_nk_fp8"
        or scale.output_name != f"{stem}.weight_scale"
        or scale.transformation != "bf16-to-bf16-rowwise-scale"
        or scale.output_dtype != "BF16"
        or scale.physical_shape != (rows, 1)
        or scale.logical_shape != (rows, 1)
        or scale.disk_layout != "row_bf16"
        or scale.runtime_layout != "row_bf16"
    ):
        raise InvalidPlanError(f"M05 weight/scale tensor shape or layout mismatch: {source_name}")


def _validate_m05_exclusions(
    excluded: tuple[ExcludedTensorPlan, ...],
    source_tensors: dict[str, TensorDescriptor],
) -> None:
    for item in excluded:
        try:
            expected_role = classify_m05_source(item.source_name)
        except ValueError as error:
            raise InvalidPlanError(str(error)) from error
        if item.source_name not in source_tensors:
            raise InvalidPlanError(f"excluded source tensor is absent: {item.source_name}")
        if item.role != expected_role:
            raise InvalidPlanError(
                f"M05 exclusion role mismatch for {item.source_name}: "
                f"expected {expected_role}, got {item.role}"
            )
        is_attention_projection = expected_role in {
            "attention_q_projection",
            "attention_k_projection",
            "attention_v_projection",
            "attention_o_projection",
        }
        if is_attention_projection:
            raise InvalidPlanError(
                f"M05 attention projection cannot be deferred: {item.source_name}"
            )
        is_vision = expected_role.startswith("vision_")
        if is_vision:
            if (
                item.family != "vision"
                or item.residency_class != "compile_excluded_vision"
                or item.reason != M05_VISION_EXCLUSION_REASON
            ):
                raise InvalidPlanError(
                    f"M05 vision exclusion contract mismatch: {item.source_name}"
                )
        elif (
            item.family != "deferred_non_attention"
            or item.residency_class != "m05_deferred_non_attention"
            or item.reason != M05_PROFILE.deferred_reason
        ):
            raise InvalidPlanError(
                f"M05 deferred exclusion contract mismatch: {item.source_name}"
            )


def _validate_m05_source_inventory(
    source_tensors: dict[str, TensorDescriptor],
) -> None:
    actual_attention: set[str] = set()
    for name, source in source_tensors.items():
        try:
            role = classify_m05_source(name)
        except ValueError as error:
            raise InvalidPlanError(str(error)) from error
        if role in {
            "attention_q_projection",
            "attention_k_projection",
            "attention_v_projection",
            "attention_o_projection",
        }:
            actual_attention.add(name)
    expected_attention = set(M05_ATTENTION_TABLE)
    if actual_attention != expected_attention:
        missing = sorted(expected_attention - actual_attention)
        extra = sorted(actual_attention - expected_attention)
        raise InvalidPlanError(
            "M05 source attention table mismatch: "
            f"missing={missing[:3]} extra={extra[:3]}"
        )
    for name, spec in M05_ATTENTION_TABLE.items():
        source = source_tensors[name]
        if source.dtype != "BF16" or tuple(source.shape) != spec.shape:
            raise InvalidPlanError(
                f"M05 source descriptor mismatch: {name}; "
                f"expected BF16 {spec.shape}, got {source.dtype} {source.shape}"
            )
        if source.byte_length != tensor_bytes("BF16", source.shape, name):
            raise InvalidPlanError(f"M05 source byte count mismatch: {name}")


def _validate_m06_source(
    source_name: str,
    source: TensorDescriptor,
    group: list[TensorCompilePlan],
) -> None:
    try:
        role, expected_shape = m06_expected_source_specs()[source_name]
    except (KeyError, ValueError) as error:
        raise InvalidPlanError(
            f"M06 source is not in the frozen source inventory: {source_name}"
        ) from error
    if role not in {
        "shared_mlp_gate", "shared_mlp_up", "shared_mlp_down",
        "routed_expert_gate_up", "routed_expert_down",
    }:
        raise InvalidPlanError(f"M06 source is not an expert/shared tensor: {source_name}")
    if source.dtype != "BF16" or tuple(source.shape) != expected_shape:
        raise InvalidPlanError(
            f"M06 source dtype/shape mismatch: {source_name}; "
            f"expected BF16 {expected_shape}, got {source.dtype} {source.shape}"
        )
    if source.byte_length != tensor_bytes("BF16", source.shape, source_name):
        raise InvalidPlanError(f"M06 source byte count mismatch: {source_name}")
    if len(group) != 4:
        raise InvalidPlanError(f"M06 source must have exactly four outputs: {source_name}")
    stem = source_name.removesuffix(".weight")
    expected = {
        f"{stem}.weight_packed": ("nvfp4-packed-v1", "nvfp4-packed", "U8", tuple(
            list(expected_shape[:-1]) + [expected_shape[-1] // 2]
        )),
        f"{stem}.weight_scale": ("nvfp4-local-scale-v1", "nvfp4-local-scale", "F8_E4M3", tuple(
            list(expected_shape[:-1]) + [expected_shape[-1] // 16]
        )),
        f"{stem}.weight_global_scale": ("nvfp4-weight-divisor-v1", "nvfp4-weight-divisor", "F32", (1,)),
        f"{stem}.input_global_scale": ("nvfp4-input-divisor-v1", "nvfp4-input-divisor", "F32", (1,)),
    }
    if {item.output_name for item in group} != set(expected):
        raise InvalidPlanError(f"M06 output component names mismatch: {source_name}")
    routed = role.startswith("routed_")
    expected_operation = f"nvfp4-experts:{stem}"
    for item in group:
        encoder, transformation, dtype, shape = expected[item.output_name]
        component = item.output_name.rsplit(".", 1)[-1]
        layout = M06_COMPONENT_LAYOUTS[component]
        expected_runtime = layout["runtime_layout_routed" if routed else "runtime_layout_shared"]
        if (
            item.operation_id != expected_operation
            or item.source_names != (source_name,)
            or item.encoder != encoder
            or item.transformation != transformation
            or layout["encoder"] != encoder
            or layout["transformation"] != transformation
            or item.transformation_version != 1
            or item.output_dtype != dtype
            or item.physical_shape != shape
            or item.logical_shape != tuple(expected_shape)
            or item.logical_dtype != "BF16"
            or item.axis_transformation != ("expert,gate_then_up,input" if role == "routed_expert_gate_up" else
                                            "expert,output,input" if role == "routed_expert_down" else "output,input")
            or item.quantizer_parameters != m06_component_parameters(component)
            or item.dequantization_equation != M06_DEQUANTIZATION_EQUATION
            or item.role != role
            or item.residency_class != "immutable_device_text"
            or item.disk_layout != layout["disk_layout"]
            or item.runtime_layout != expected_runtime
            or item.aliased
        ):
            raise InvalidPlanError(f"M06 semantic contract mismatch: {item.output_name}")
    if sum(item.output_bytes for item in group) != (
        tensor_bytes("U8", expected[stem + ".weight_packed"][3], source_name)
        + tensor_bytes("F8_E4M3", expected[stem + ".weight_scale"][3], source_name)
        + 8
    ):
        raise InvalidPlanError(f"M06 output byte count mismatch: {source_name}")


def _validate_m06_exclusions(
    excluded: tuple[ExcludedTensorPlan, ...],
    source_tensors: dict[str, TensorDescriptor],
) -> None:
    for item in excluded:
        if item.source_name not in source_tensors:
            raise InvalidPlanError(f"excluded source tensor is absent: {item.source_name}")
        try:
            expected_role = classify_m05_source(item.source_name)
        except ValueError as error:
            raise InvalidPlanError(str(error)) from error
        if item.role != expected_role:
            raise InvalidPlanError(f"M06 exclusion role mismatch: {item.source_name}")
        if expected_role.startswith("vision_"):
            expected = ("vision", "compile_excluded_vision", M06_VISION_EXCLUSION_REASON)
        else:
            expected = ("deferred_non_expert", "m06_deferred_non_expert", M06_DEFERRED_REASON)
        if (item.family, item.residency_class, item.reason) != expected:
            raise InvalidPlanError(f"M06 exclusion contract mismatch: {item.source_name}")


def _validate_m06_source_inventory(source_tensors: dict[str, TensorDescriptor]) -> None:
    expected = m06_expected_source_specs()
    actual_names = set(source_tensors)
    if actual_names != set(expected):
        missing = sorted(set(expected) - actual_names)
        extra = sorted(actual_names - set(expected))
        raise InvalidPlanError(
            "M06 source inventory mismatch: "
            f"missing={missing[:3]} extra={extra[:3]}"
        )
    for name, (role, shape) in expected.items():
        source = source_tensors[name]
        if source.dtype != "BF16" or tuple(source.shape) != shape:
            raise InvalidPlanError(
                f"M06 source descriptor mismatch: {name}; "
                f"expected BF16 {shape}, got {source.dtype} {source.shape}"
            )
        if source.byte_length != tensor_bytes("BF16", shape, name):
            raise InvalidPlanError(f"M06 source byte count mismatch: {name}")
        try:
            actual_role = classify_m05_source(name)
        except ValueError as error:
            raise InvalidPlanError(str(error)) from error
        if actual_role != role:
            raise InvalidPlanError(f"M06 source role mismatch: {name}")


def _validate_m06_coverage(
    tensors: tuple[TensorCompilePlan, ...],
    excluded: tuple[ExcludedTensorPlan, ...],
    source_tensors: dict[str, TensorDescriptor],
) -> None:
    _validate_m06_source_inventory(source_tensors)
    groups: dict[str, list[TensorCompilePlan]] = {}
    for tensor in tensors:
        if len(tensor.source_names) != 1:
            raise InvalidPlanError("M06 encoders require exactly one source tensor")
        groups.setdefault(tensor.source_names[0], []).append(tensor)
    expert_sources = {
        name for name in source_tensors
        if classify_m05_source(name) in {
            "shared_mlp_gate", "shared_mlp_up", "shared_mlp_down",
            "routed_expert_gate_up", "routed_expert_down",
        }
    }
    if set(groups) != expert_sources:
        raise InvalidPlanError("M06 expert/shared source coverage mismatch")
    for name in sorted(groups):
        _validate_m06_source(name, source_tensors[name], groups[name])
    if len(tensors) != 600:
        raise InvalidPlanError(f"M06 expected 600 output tensors, got {len(tensors)}")
    _validate_m06_exclusions(excluded, source_tensors)


def _validate_m07_coverage(
    tensors: tuple[TensorCompilePlan, ...],
    excluded: tuple[ExcludedTensorPlan, ...],
    source_tensors: dict[str, TensorDescriptor],
) -> None:
    expected_name = "model.language_model.embed_tokens.weight"
    expected_shape = (262144, 2816)
    if set(source_tensors) != set(m06_expected_source_specs()):
        raise InvalidPlanError("M07 requires the complete 1,013-name QAT source inventory")
    # Reuse the frozen M06 inventory descriptors: M07 excludes 1,012 tensors,
    # but must not accept a mutated dtype, shape, or byte length for them.
    _validate_m06_source_inventory(source_tensors)
    source = source_tensors.get(expected_name)
    if source is None or source.dtype != "BF16" or tuple(source.shape) != expected_shape:
        raise InvalidPlanError("M07 tied head source must be BF16 [262144, 2816]")
    if source.byte_length != tensor_bytes("BF16", expected_shape, expected_name):
        raise InvalidPlanError("M07 tied head source byte count mismatch")
    if len(tensors) != 4 or {item.source_names for item in tensors} != {(expected_name,)}:
        raise InvalidPlanError("M07 must compile exactly one tied source into four outputs")
    stem = expected_name.removesuffix(".weight")
    expected = {
        f"{stem}.weight_packed": ("nvfp4-packed-v1", "nvfp4-packed", "U8", (262144, 1408)),
        f"{stem}.weight_scale": ("nvfp4-local-scale-v1", "nvfp4-local-scale", "F8_E4M3", (262144, 176)),
        f"{stem}.weight_global_scale": ("nvfp4-weight-divisor-v1", "nvfp4-weight-divisor", "F32", (1,)),
        f"{stem}.input_global_scale": ("nvfp4-input-divisor-v1", "nvfp4-input-divisor", "F32", (1,)),
    }
    if {item.output_name for item in tensors} != set(expected):
        raise InvalidPlanError("M07 tied head output names mismatch (lm_head duplication is forbidden)")
    for item in tensors:
        encoder, transformation, dtype, shape = expected[item.output_name]
        component = item.output_name.rsplit(".", 1)[-1]
        layout = M07_COMPONENT_LAYOUTS[component]
        if (item.operation_id != "nvfp4-head:model.language_model.embed_tokens"
            or item.source_names != (expected_name,)
            or item.encoder != encoder or item.transformation != transformation
            or item.transformation_version != 1 or item.output_dtype != dtype
            or item.physical_shape != shape or item.logical_shape != expected_shape
            or item.logical_dtype != "BF16" or item.axis_transformation != "vocabulary,hidden"
            or item.quantizer_parameters != m07_component_parameters(component)
            or item.dequantization_equation != M07_DEQUANTIZATION_EQUATION
            or item.role != "tied_embedding_and_output"
            or item.residency_class != "immutable_device_text"
            or item.disk_layout != layout["disk_layout"]
            or item.runtime_layout != layout["runtime_layout_shared"]
            or not item.aliased):
            raise InvalidPlanError(f"M07 tied head semantic contract mismatch: {item.output_name}")
    if sum(item.output_bytes for item in tensors) != 415236104:
        raise InvalidPlanError("M07 tied head output byte count mismatch")
    if len(excluded) != 1012:
        raise InvalidPlanError("M07 requires exactly 1,012 explicit exclusions")
    for item in excluded:
        if item.source_name not in source_tensors or item.source_name == expected_name:
            raise InvalidPlanError(f"M07 invalid excluded source: {item.source_name}")
        role = classify_m05_source(item.source_name)
        if item.role != role:
            raise InvalidPlanError(f"M07 exclusion role mismatch: {item.source_name}")
        if role.startswith("vision_"):
            expected_exclusion = ("vision", "compile_excluded_vision", "text-only Gemma 4 26B profile excludes vision tensors")
        else:
            expected_exclusion = ("deferred_non_head", "m07_deferred_non_head", M07_DEFERRED_REASON)
        if (item.family, item.residency_class, item.reason) != expected_exclusion:
            raise InvalidPlanError(f"M07 exclusion contract mismatch: {item.source_name}")


def _validate_m08_coverage(
    tensors: tuple[TensorCompilePlan, ...],
    excluded: tuple[ExcludedTensorPlan, ...],
    source_tensors: dict[str, TensorDescriptor],
) -> None:
    """Validate the exact complete M08 hybrid mapping."""
    _validate_m06_source_inventory(source_tensors)
    by_source: dict[str, list[TensorCompilePlan]] = {}
    constants: list[TensorCompilePlan] = []
    for tensor in tensors:
        if tensor.encoder == "constant-bf16-one-v1":
            constants.append(tensor)
            continue
        if len(tensor.source_names) != 1:
            raise InvalidPlanError("M08 non-constant encoders require one source tensor")
        by_source.setdefault(tensor.source_names[0], []).append(tensor)

    head_name = "model.language_model.embed_tokens.weight"
    head_group = by_source.get(head_name, [])
    _validate_m07_coverage(
        tuple(head_group),
        tuple(
            ExcludedTensorPlan(
                source_name=name,
                family=("vision" if role.startswith("vision_") else "deferred_non_head"),
                role=role,
                residency_class=("compile_excluded_vision" if role.startswith("vision_") else "m07_deferred_non_head"),
                reason=(M05_VISION_EXCLUSION_REASON if role.startswith("vision_") else M07_DEFERRED_REASON),
            )
            for name, (role, _shape) in m06_expected_source_specs().items()
            if name != head_name
        ),
        source_tensors,
    )

    expert_roles = {
        "shared_mlp_gate", "shared_mlp_up", "shared_mlp_down",
        "routed_expert_gate_up", "routed_expert_down",
    }
    attention_roles = {
        "attention_q_projection", "attention_k_projection",
        "attention_v_projection", "attention_o_projection",
    }
    copied_sources: set[str] = set()
    for source_name, (role, _shape) in m06_expected_source_specs().items():
        if role.startswith("vision_") or source_name == head_name:
            continue
        group = by_source.get(source_name, [])
        if role in expert_roles:
            _validate_m06_source(source_name, source_tensors[source_name], group)
        elif role in attention_roles:
            _validate_m05_pair(source_name, source_tensors[source_name], group)
        else:
            if len(group) != 1 or group[0].encoder != "copy-v1":
                raise InvalidPlanError(
                    f"M08 source tensor must be copied exactly once: {source_name}"
                )
            _validate_copy_plan(group[0], source_tensors[source_name])
            if (
                group[0].role != role
                or group[0].residency_class != "immutable_device_text"
                or group[0].runtime_layout != "source_bf16"
                or group[0].aliased
            ):
                raise InvalidPlanError(f"M08 copy semantic mismatch: {source_name}")
            copied_sources.add(source_name)

    expected_constants = {
        f"model.language_model.layers.{layer}.self_attn.{component}_scale":
            f"attention_{component}_cache_scale"
        for layer in range(M05_LAYER_COUNT)
        for component in ("k", "v")
    }
    if {item.output_name for item in constants} != set(expected_constants):
        raise InvalidPlanError("M08 cache-scale metadata inventory mismatch")
    for item in constants:
        if (
            item.source_names
            or item.operation_id != f"constant-bf16-one:{item.output_name}"
            or item.transformation != "constant-bf16-one"
            or item.transformation_version != 1
            or item.output_dtype != "BF16"
            or item.physical_shape != (1,)
            or item.logical_dtype != "BF16"
            or item.logical_shape != (1,)
            or item.axis_transformation != "scalar"
            or item.quantizer_parameters != {"value": 1.0, "encoding": "BF16-RNE"}
            or item.dequantization_equation != "output = BF16(1.0)"
            or item.role != expected_constants[item.output_name]
            or item.residency_class != "immutable_device_text"
            or item.disk_layout != "scalar_bf16"
            or item.runtime_layout != "scalar_bf16"
            or item.aliased
        ):
            raise InvalidPlanError(
                f"M08 cache-scale metadata contract mismatch: {item.output_name}"
            )

    if len(tensors) != 1285:
        raise InvalidPlanError(f"M08 expected 1,285 output tensors, got {len(tensors)}")
    if len(copied_sources) != 391:
        raise InvalidPlanError(f"M08 expected 391 copied text tensors, got {len(copied_sources)}")
    if len(excluded) != 356:
        raise InvalidPlanError("M08 must exclude exactly the 356 vision tensors")
    for item in excluded:
        role = classify_m05_source(item.source_name)
        if (
            not role.startswith("vision_")
            or item.role != role
            or item.family != "vision"
            or item.residency_class != "compile_excluded_vision"
            or item.reason != M05_VISION_EXCLUSION_REASON
        ):
            raise InvalidPlanError(f"M08 exclusion mismatch: {item.source_name}")


def _m25_source_specs() -> dict[str, tuple[str, tuple[int, ...]]]:
    specs: dict[str, tuple[str, tuple[int, ...]]] = {
        "model.embed_tokens.weight": (
            "tied_embedding_and_output", (262144, 1024)
        ),
        "model.norm.weight": ("final_norm", (1024,)),
        "pre_projection.weight": ("assistant_pre_projection", (1024, 5632)),
        "post_projection.weight": ("assistant_post_projection", (2816, 1024)),
    }
    for layer in range(4):
        prefix = f"model.layers.{layer}."
        q_rows = 8192 if layer == 3 else 4096
        q_norm = 512 if layer == 3 else 256
        specs.update({
            prefix + "input_layernorm.weight": ("input_layer_norm", (1024,)),
            prefix + "layer_scalar": ("layer_scalar", (1,)),
            prefix + "mlp.down_proj.weight": ("assistant_mlp_down", (1024, 8192)),
            prefix + "mlp.gate_proj.weight": ("assistant_mlp_gate", (8192, 1024)),
            prefix + "mlp.up_proj.weight": ("assistant_mlp_up", (8192, 1024)),
            prefix + "post_attention_layernorm.weight": (
                "post_attention_layer_norm", (1024,)
            ),
            prefix + "post_feedforward_layernorm.weight": (
                "post_feed_forward_layer_norm", (1024,)
            ),
            prefix + "pre_feedforward_layernorm.weight": (
                "pre_feed_forward_layer_norm", (1024,)
            ),
            prefix + "self_attn.o_proj.weight": (
                "assistant_attention_o_projection", (1024, q_rows)
            ),
            prefix + "self_attn.q_norm.weight": (
                "assistant_attention_q_norm", (q_norm,)
            ),
            prefix + "self_attn.q_proj.weight": (
                "assistant_attention_q_projection", (q_rows, 1024)
            ),
        })
    return dict(sorted(specs.items()))


def _validate_m25_nvfp4_source(
    source_name: str,
    source: TensorDescriptor,
    group: list[TensorCompilePlan],
    role: str,
    expected_shape: tuple[int, ...],
) -> None:
    if source.dtype != "BF16" or tuple(source.shape) != expected_shape:
        raise InvalidPlanError(
            f"M25 NVFP4 source mismatch: {source_name}; expected BF16 "
            f"{expected_shape}, got {source.dtype} {source.shape}"
        )
    if len(expected_shape) != 2 or expected_shape[-1] % 16 != 0:
        raise InvalidPlanError(f"M25 NVFP4 source shape is unsupported: {source_name}")
    if len(group) != 4:
        raise InvalidPlanError(f"M25 NVFP4 source requires four outputs: {source_name}")
    stem = source_name.removesuffix(".weight")
    expected = {
        f"{stem}.weight_packed": (
            "nvfp4-packed-v1", "nvfp4-packed", "U8",
            expected_shape[:-1] + (expected_shape[-1] // 2,),
        ),
        f"{stem}.weight_scale": (
            "nvfp4-local-scale-v1", "nvfp4-local-scale", "F8_E4M3",
            expected_shape[:-1] + (expected_shape[-1] // 16,),
        ),
        f"{stem}.weight_global_scale": (
            "nvfp4-weight-divisor-v1", "nvfp4-weight-divisor", "F32", (1,)
        ),
        f"{stem}.input_global_scale": (
            "nvfp4-input-divisor-v1", "nvfp4-input-divisor", "F32", (1,)
        ),
    }
    if {item.output_name for item in group} != set(expected):
        raise InvalidPlanError(f"M25 NVFP4 output names mismatch: {source_name}")
    tied = role == "tied_embedding_and_output"
    axis = "vocabulary,hidden" if tied else "output,input"
    for item in group:
        encoder, transformation, dtype, shape = expected[item.output_name]
        component = item.output_name.rsplit(".", 1)[-1]
        layout = M07_COMPONENT_LAYOUTS[component]
        if (
            item.operation_id != f"nvfp4-assistant:{stem}"
            or item.source_names != (source_name,)
            or item.encoder != encoder
            or item.transformation != transformation
            or item.transformation_version != 1
            or item.output_dtype != dtype
            or item.physical_shape != shape
            or item.logical_dtype != "BF16"
            or item.logical_shape != expected_shape
            or item.axis_transformation != axis
            or item.quantizer_parameters != m07_component_parameters(component)
            or item.dequantization_equation != M07_DEQUANTIZATION_EQUATION
            or item.role != role
            or item.residency_class != "immutable_device_mtp_assistant"
            or item.disk_layout != layout["disk_layout"]
            or item.runtime_layout != layout["runtime_layout_shared"]
            or item.aliased != tied
        ):
            raise InvalidPlanError(
                f"M25 NVFP4 semantic contract mismatch: {item.output_name}"
            )


def _validate_m25_fp8_source(
    source_name: str,
    source: TensorDescriptor,
    group: list[TensorCompilePlan],
    role: str,
    expected_shape: tuple[int, ...],
) -> None:
    if source.dtype != "BF16" or tuple(source.shape) != expected_shape:
        raise InvalidPlanError(
            f"M25 FP8 source mismatch: {source_name}; expected BF16 "
            f"{expected_shape}, got {source.dtype} {source.shape}"
        )
    if len(expected_shape) != 2 or len(group) != 2:
        raise InvalidPlanError(f"M25 FP8 source requires two matrix outputs: {source_name}")
    rows, columns = expected_shape
    stem = source_name.removesuffix(".weight")
    expected = {
        source_name: (
            "fp8-rowwise-weight-v1", "bf16-to-fp8-e4m3fn-rowwise-weight",
            "F8_E4M3", (rows, columns), "source_nk_fp8",
        ),
        f"{stem}.weight_scale": (
            "fp8-rowwise-scale-v1", "bf16-to-bf16-rowwise-scale",
            "BF16", (rows, 1), "row_bf16",
        ),
    }
    if {item.output_name for item in group} != set(expected):
        raise InvalidPlanError(f"M25 FP8 output names mismatch: {source_name}")
    for item in group:
        encoder, transformation, dtype, shape, layout = expected[item.output_name]
        if (
            item.operation_id != f"fp8-assistant:{stem}"
            or item.source_names != (source_name,)
            or item.encoder != encoder
            or item.transformation != transformation
            or item.transformation_version != 1
            or item.output_dtype != dtype
            or item.physical_shape != shape
            or item.logical_dtype != "BF16"
            or item.logical_shape != shape
            or item.axis_transformation != "identity"
            or item.quantizer_parameters != M05_QUANTIZER_PARAMETERS
            or item.dequantization_equation != M05_DEQUANTIZATION_EQUATION
            or item.role != role
            or item.residency_class != "immutable_device_mtp_assistant"
            or item.disk_layout != layout
            or item.runtime_layout != layout
            or item.aliased
        ):
            raise InvalidPlanError(
                f"M25 FP8 semantic contract mismatch: {item.output_name}"
            )


def _validate_m25_coverage(
    tensors: tuple[TensorCompilePlan, ...],
    excluded: tuple[ExcludedTensorPlan, ...],
    source_tensors: dict[str, TensorDescriptor],
) -> None:
    specs = _m25_source_specs()
    if set(source_tensors) != set(specs):
        missing = sorted(set(specs) - set(source_tensors))
        extra = sorted(set(source_tensors) - set(specs))
        raise InvalidPlanError(
            f"M25 Assistant source inventory mismatch: missing={missing[:3]} "
            f"extra={extra[:3]}"
        )
    if excluded:
        raise InvalidPlanError("M25 Assistant hybrid artifact excludes no source tensors")
    by_source: dict[str, list[TensorCompilePlan]] = {}
    for tensor in tensors:
        if len(tensor.source_names) != 1:
            raise InvalidPlanError("M25 Assistant encoders require one source tensor")
        by_source.setdefault(tensor.source_names[0], []).append(tensor)
    if set(by_source) != set(specs):
        raise InvalidPlanError("M25 Assistant source coverage is incomplete")
    nvfp4_roles = {
        "tied_embedding_and_output", "assistant_mlp_down",
        "assistant_mlp_gate", "assistant_mlp_up",
    }
    fp8_roles = {
        "assistant_attention_q_projection", "assistant_attention_o_projection",
        "assistant_pre_projection", "assistant_post_projection",
    }
    for name, (role, shape) in specs.items():
        source = source_tensors[name]
        if (
            source.dtype != "BF16"
            or tuple(source.shape) != shape
            or source.byte_length != tensor_bytes("BF16", shape, name)
        ):
            raise InvalidPlanError(f"M25 Assistant source descriptor mismatch: {name}")
        group = by_source[name]
        if role in nvfp4_roles:
            _validate_m25_nvfp4_source(name, source, group, role, shape)
        elif role in fp8_roles:
            _validate_m25_fp8_source(name, source, group, role, shape)
        else:
            if len(group) != 1:
                raise InvalidPlanError(f"M25 BF16 source must be copied once: {name}")
            item = group[0]
            _validate_copy_plan(item, source)
            if (
                item.output_name != name
                or item.operation_id != f"copy-assistant:{name}"
                or item.role != role
                or item.residency_class != "immutable_device_mtp_assistant"
                or item.disk_layout != "source_bf16"
                or item.runtime_layout != "source_bf16"
                or item.aliased
            ):
                raise InvalidPlanError(f"M25 BF16 copy contract mismatch: {name}")
    if len(tensors) != 97:
        raise InvalidPlanError(f"M25 expected 97 output tensors, got {len(tensors)}")
    if sum(item.output_bytes for item in tensors) != 258_306_160:
        raise InvalidPlanError("M25 Assistant hybrid output byte count mismatch")


def _validate_coverage(
    tensors: tuple[TensorCompilePlan, ...],
    excluded: tuple[ExcludedTensorPlan, ...],
    source_tensors: dict[str, TensorDescriptor],
    profile: CompilerProfile,
) -> None:
    covered: dict[str, str] = {}
    for tensor in tensors:
        for source_name in tensor.source_names:
            if source_name not in source_tensors:
                raise InvalidPlanError(
                    f"compiler source tensor is absent: {source_name}"
                )
            previous_operation = covered.get(source_name)
            if previous_operation is not None and previous_operation != tensor.operation_id:
                raise InvalidPlanError(
                    f"source tensor is assigned to conflicting operations: {source_name}"
                )
            covered[source_name] = tensor.operation_id
    if profile.name == M05_PROFILE.name:
        _validate_m05_source_inventory(source_tensors)
        missing_attention = sorted(
            set(M05_ATTENTION_TABLE) - set(covered)
        )
        if missing_attention:
            raise InvalidPlanError(
                "M05 language attention projections must be compiled as FP8 pairs; "
                f"missing={missing_attention[:3]}"
            )
    if profile.name == M04_PROFILE.name:
        copy_sources: set[str] = set()
        for tensor in tensors:
            if tensor.encoder == "copy-v1":
                if len(tensor.source_names) != 1:
                    raise InvalidPlanError("copy-v1 requires exactly one source tensor")
                source_name = tensor.source_names[0]
                if source_name in copy_sources:
                    raise InvalidPlanError(f"copy-v1 cannot duplicate a source payload: {source_name}")
                copy_sources.add(source_name)
                _validate_copy_plan(tensor, source_tensors[source_name])
    elif profile.name == M05_PROFILE.name:
        pairs: dict[str, list[TensorCompilePlan]] = {}
        for tensor in tensors:
            if len(tensor.source_names) != 1:
                raise InvalidPlanError("M05 encoders require exactly one source tensor")
            pairs.setdefault(tensor.source_names[0], []).append(tensor)
        for source_name, pair in pairs.items():
            _validate_m05_pair(source_name, source_tensors[source_name], pair)
        _validate_m05_exclusions(excluded, source_tensors)
    elif profile.name == M06_PROFILE.name:
        _validate_m06_coverage(tensors, excluded, source_tensors)
    elif profile.name == M07_PROFILE.name:
        _validate_m07_coverage(tensors, excluded, source_tensors)
    elif profile.name == M08_PROFILE.name:
        _validate_m08_coverage(tensors, excluded, source_tensors)
    elif profile.name == M25_PROFILE.name:
        _validate_m25_coverage(tensors, excluded, source_tensors)
    for item in excluded:
        if item.source_name not in source_tensors:
            raise InvalidPlanError(f"excluded source tensor is absent: {item.source_name}")
        if item.source_name in covered:
            raise InvalidPlanError(f"source tensor is covered more than once: {item.source_name}")
        covered[item.source_name] = f"excluded:{item.family}"
    if set(covered) != set(source_tensors):
        missing = sorted(set(source_tensors) - set(covered))
        extra = sorted(set(covered) - set(source_tensors))
        raise InvalidPlanError(f"compiler source coverage mismatch: missing={missing[:3]} extra={extra[:3]}")


def load_quantization_plan(
    compiler_manifest: Path,
    source: VerifiedSource,
    source_tensors: dict[str, TensorDescriptor],
    profile: str,
    head_format: str,
    shard_size_override: int | None = None,
) -> QuantizationPlan:
    try:
        contract = profile_for(profile, head_format)
    except ValueError as error:
        raise InvalidPlanError(str(error)) from error
    document = load_json(compiler_manifest, 16 * 1024 * 1024)
    raw_bytes = compiler_manifest.read_bytes()
    if set(document) != _PLAN_KEYS:
        raise InvalidPlanError(
            f"compiler manifest top-level keys mismatch: "
            f"unknown={sorted(set(document) - _PLAN_KEYS)} "
            f"missing={sorted(_PLAN_KEYS - set(document))}"
        )
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
    if profile != artifact_profile:
        raise InvalidPlanError(
            f"unsupported or mismatched compiler profile: cli={profile!r} "
            f"manifest={artifact_profile!r}"
        )
    manifest_head = _required_string(
        document.get("head_format"), "compiler manifest head_format"
    )
    if head_format != manifest_head:
        raise InvalidPlanError(
            f"unsupported or mismatched compiler head format: cli={head_format!r} "
            f"manifest={manifest_head!r}"
        )
    expected_source_lock = _required_string(
        document.get("source_lock_sha256"), "compiler manifest source_lock_sha256"
    )
    if contract.name == M05_PROFILE.name:
        if source.lock_sha256 not in M05_APPROVED_SOURCE_LOCKS:
            raise InvalidPlanError(
                "M05 source lock is not an approved ordinary/QAT BF16 lock: "
                f"{source.lock_sha256}"
            )
        if expected_source_lock not in M05_APPROVED_SOURCE_LOCKS:
            raise InvalidPlanError(
                "M05 compiler manifest does not name an approved source lock"
            )
    elif contract.name in {M06_PROFILE.name, M07_PROFILE.name, M08_PROFILE.name}:
        qat_lock = M05_SOURCE_LOCK_SHA256["qat_bf16"]
        if source.lock_sha256 != qat_lock or expected_source_lock != qat_lock:
            raise InvalidPlanError(f"{contract.milestone} requires the approved QAT BF16 source lock")
    elif contract.name == M25_PROFILE.name:
        if (
            source.lock_sha256 != M25_SOURCE_LOCK_SHA256
            or expected_source_lock != M25_SOURCE_LOCK_SHA256
        ):
            raise InvalidPlanError("M25 requires the approved QAT-Q4_0 Assistant source lock")
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
    expected_omitted = (
        ("audio", "video", "vision")
        if contract.name == M25_PROFILE.name
        else EXPECTED_OMITTED_FAMILIES
    )
    if omitted_families != expected_omitted:
        raise InvalidPlanError(
            f"compiler omitted_families must be exactly {', '.join(expected_omitted)}"
        )
    tensor_values = document.get("tensors")
    excluded_values = document.get("excluded_tensors")
    if not isinstance(tensor_values, list) or not tensor_values:
        raise InvalidPlanError("compiler manifest tensors must be a non-empty array")
    if not isinstance(excluded_values, list):
        raise InvalidPlanError("compiler manifest excluded_tensors must be an array")
    tensors = tuple(
        _tensor_plan(value, index, contract)
        for index, value in enumerate(tensor_values)
    )
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
    _validate_coverage(tensors, excluded, source_tensors, contract)
    reference_environment = document.get("reference_environment")
    if not isinstance(reference_environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and value
        for key, value in reference_environment.items()
    ):
        raise InvalidPlanError("reference_environment must map strings to strings")
    source_contract = _required_string(
        document.get("source_contract"), "compiler manifest source_contract"
    )
    if contract.name == M05_PROFILE.name and source_contract != M05_SOURCE_CONTRACT:
        raise InvalidPlanError(
            "M05 source_contract must be "
            f"{M05_SOURCE_CONTRACT!r}"
        )
    if contract.name == M06_PROFILE.name and source_contract != M06_SOURCE_CONTRACT:
        raise InvalidPlanError(
            "M06 source_contract must be "
            f"{M06_SOURCE_CONTRACT!r}"
        )
    if contract.name == M07_PROFILE.name and source_contract != M07_SOURCE_CONTRACT:
        raise InvalidPlanError(
            "M07 source_contract must be "
            f"{M07_SOURCE_CONTRACT!r}"
        )
    if contract.name == M25_PROFILE.name and source_contract != M25_SOURCE_CONTRACT:
        raise InvalidPlanError(
            f"M25 source_contract must be {M25_SOURCE_CONTRACT!r}"
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

"""Deterministic Gemma 4 26B Vision FP8 sidecar compiler.

The sidecar is a Safetensors-compatible immutable file named ``vision.gem16``.
It is deliberately separate from both the Trellis35 text checkpoint and the
qualified NVFP4 checkpoint.  Runtime capability selection must consume the
explicit descriptor; file existence alone is never a capability signal.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Any

from .common import (
    BoundedWorkspace,
    DataError,
    InvalidPlanError,
    OutputError,
    canonical_json_bytes,
    compact_json_bytes,
    environment_identity,
    git_compiler_identity,
    load_json,
    sha256_bytes,
    write_all,
    write_file_atomic,
)
from .native_fp8 import NativeRequest, prepare_native_bundle
from .plan import QuantizationPlan, TensorCompilePlan
from .profiles import (
    M05_DEQUANTIZATION_EQUATION,
    M05_QUANTIZER_PARAMETERS,
    classify_m05_source,
    m06_expected_source_specs,
)
from .reader import TensorDescriptor, read_source_tensors, verify_source_lock
from .writer import create_staging_directory, discard_staging, publish_staging


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path(__file__).with_name("specs") / "gemma4-26b-vision-fp8-v1.json"
SOURCE_LOCK_PATH = REPOSITORY_ROOT / "models/gemma4-26b-qat-bf16.lock.json"
REFERENCE_LOCK_PATH = REPOSITORY_ROOT / "models/gemma4-26b-reference-sources.lock.json"

PROFILE = "gemma4_26b_trellis35_vision_fp8"
TEXT_ARTIFACT_PROFILE = "gem16-trellis35-w4a8-v1"
VISION_FILENAME = "vision.gem16"
DESCRIPTOR_FILENAME = "gem16_vision.json"
COMPILATION_FILENAME = "vision_compilation.json"
LOCK_FILENAME = "vision.lock.json"
SOURCE_TENSOR_COUNT = 356
SOURCE_TENSOR_BYTES = 1_145_588_832
LINEAR_COUNT = 191
LINEAR_PARAMETERS = 549_070_848
BF16_COPY_COUNT = 165
OUTPUT_TENSOR_COUNT = 547
OUTPUT_TENSOR_BYTES = 597_301_792
OUTPUT_PADDING_BYTES = 11_232
OUTPUT_PAYLOAD_BYTES = 597_313_024
TENSOR_ALIGNMENT = 256


@dataclass(frozen=True)
class VisionOutput:
    name: str
    dtype: str
    shape: tuple[int, ...]
    source_name: str
    role: str
    kind: str
    byte_length: int


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_vision_specs() -> dict[str, tuple[str, tuple[int, ...]]]:
    result = {
        name: (role, shape)
        for name, (role, shape) in m06_expected_source_specs().items()
        if role.startswith("vision_")
    }
    if len(result) != SOURCE_TENSOR_COUNT:
        raise InvalidPlanError(
            f"frozen Vision source table has {len(result)} tensors, expected {SOURCE_TENSOR_COUNT}"
        )
    return dict(sorted(result.items()))


def is_linear_source(name: str, shape: tuple[int, ...]) -> bool:
    if len(shape) != 2:
        return False
    return (
        name == "model.embed_vision.embedding_projection.weight"
        or name == "model.vision_tower.patch_embedder.input_proj.weight"
        or name.endswith(".linear.weight")
    )


def validate_vision_sources(
    source_tensors: dict[str, TensorDescriptor],
) -> dict[str, TensorDescriptor]:
    expected = expected_vision_specs()
    actual_names = {
        name
        for name in source_tensors
        if name == "model.embed_vision.embedding_projection.weight"
        or name.startswith("model.vision_tower.")
    }
    if actual_names != set(expected):
        missing = sorted(set(expected) - actual_names)
        extra = sorted(actual_names - set(expected))
        raise DataError(f"Vision source inventory mismatch: missing={missing} extra={extra}")
    selected: dict[str, TensorDescriptor] = {}
    source_bytes = 0
    linear_count = 0
    linear_parameters = 0
    for name, (role, shape) in expected.items():
        tensor = source_tensors[name]
        if tensor.dtype != "BF16" or tensor.shape != shape:
            raise DataError(
                f"Vision source dtype/shape mismatch for {name}: "
                f"expected BF16 {shape}, got {tensor.dtype} {tensor.shape}"
            )
        if classify_m05_source(name) != role:
            raise DataError(f"Vision source role mismatch: {name}")
        selected[name] = tensor
        source_bytes += tensor.byte_length
        if is_linear_source(name, shape):
            linear_count += 1
            linear_parameters += shape[0] * shape[1]
    if (
        len(selected) != SOURCE_TENSOR_COUNT
        or source_bytes != SOURCE_TENSOR_BYTES
        or linear_count != LINEAR_COUNT
        or linear_parameters != LINEAR_PARAMETERS
    ):
        raise DataError(
            "Vision source totals changed: "
            f"tensors={len(selected)} bytes={source_bytes} "
            f"linears={linear_count} parameters={linear_parameters}"
        )
    return selected


def output_plan(selected: dict[str, TensorDescriptor]) -> tuple[VisionOutput, ...]:
    outputs: list[VisionOutput] = []
    for name, tensor in sorted(selected.items()):
        role = classify_m05_source(name)
        if is_linear_source(name, tensor.shape):
            rows, columns = tensor.shape
            outputs.append(VisionOutput(
                name=name,
                dtype="F8_E4M3",
                shape=(rows, columns),
                source_name=name,
                role=role,
                kind="fp8_weight",
                byte_length=rows * columns,
            ))
            outputs.append(VisionOutput(
                name=name.removesuffix(".weight") + ".weight_scale",
                dtype="BF16",
                shape=(rows, 1),
                source_name=name,
                role=role,
                kind="fp8_scale",
                byte_length=rows * 2,
            ))
        else:
            outputs.append(VisionOutput(
                name=name,
                dtype="BF16",
                shape=tensor.shape,
                source_name=name,
                role=role,
                kind="bf16_copy",
                byte_length=tensor.byte_length,
            ))
    outputs.sort(key=lambda item: item.name)
    if len(outputs) != OUTPUT_TENSOR_COUNT:
        raise DataError(f"Vision output tensor count changed: {len(outputs)}")
    if sum(item.byte_length for item in outputs) != OUTPUT_TENSOR_BYTES:
        raise DataError("Vision output tensor byte total changed")
    if sum(item.kind == "bf16_copy" for item in outputs) != BF16_COPY_COUNT:
        raise DataError("Vision BF16 copy tensor count changed")
    return tuple(outputs)


def fp8_plan(selected: dict[str, TensorDescriptor]) -> QuantizationPlan:
    tensors: list[TensorCompilePlan] = []
    for name, source in sorted(selected.items()):
        if not is_linear_source(name, source.shape):
            continue
        rows, columns = source.shape
        common: dict[str, Any] = {
            "operation_id": f"vision-fp8:{name.removesuffix('.weight')}",
            "source_names": (name,),
            "transformation_version": 1,
            "logical_dtype": "BF16",
            "axis_transformation": "identity",
            "quantizer_parameters": dict(M05_QUANTIZER_PARAMETERS),
            "dequantization_equation": M05_DEQUANTIZATION_EQUATION,
            "role": classify_m05_source(name),
            "residency_class": "immutable_device_vision",
            "aliased": False,
        }
        tensors.append(TensorCompilePlan(
            output_name=name,
            encoder="fp8-rowwise-weight-v1",
            transformation="bf16-to-fp8-e4m3fn-rowwise-weight",
            output_dtype="F8_E4M3",
            physical_shape=(rows, columns),
            logical_shape=(rows, columns),
            disk_layout="source_nk_fp8",
            runtime_layout="source_nk_fp8",
            **common,
        ))
        tensors.append(TensorCompilePlan(
            output_name=name.removesuffix(".weight") + ".weight_scale",
            encoder="fp8-rowwise-scale-v1",
            transformation="bf16-to-bf16-rowwise-scale",
            output_dtype="BF16",
            physical_shape=(rows, 1),
            logical_shape=(rows, 1),
            disk_layout="row_bf16",
            runtime_layout="row_bf16",
            **common,
        ))
    if len(tensors) != LINEAR_COUNT * 2:
        raise DataError(f"Vision FP8 plan has {len(tensors)} outputs")
    return QuantizationPlan(
        schema_version=1,
        artifact_profile=PROFILE,
        head_format="not-applicable",
        source_contract="gemma4-26b-qat-bf16-vision-v1",
        compiler_manifest_sha256="0" * 64,
        resolved_plan_sha256="0" * 64,
        target_shard_bytes=OUTPUT_PAYLOAD_BYTES,
        approved_metadata_files=(),
        omitted_families=("audio", "mtp", "text", "video"),
        tensors=tuple(tensors),
        excluded_tensors=(),
        reference_environment={},
    )


def _artifact_header(outputs: tuple[VisionOutput, ...]) -> tuple[bytes, dict[str, tuple[int, int]]]:
    cursor = 0
    offsets: dict[str, tuple[int, int]] = {}
    header: dict[str, Any] = {
        "__metadata__": {
            "format": "pt",
            "gem16_artifact": "gemma4-26b-vision-fp8-v1",
            "gem16_capability_profile": PROFILE,
            "gem16_required_text_profile": TEXT_ARTIFACT_PROFILE,
            "gem16_schema_version": "1",
        }
    }
    for item in outputs:
        cursor = (cursor + TENSOR_ALIGNMENT - 1) & -TENSOR_ALIGNMENT
        end = cursor + item.byte_length
        offsets[item.name] = (cursor, end)
        header[item.name] = {
            "dtype": item.dtype,
            "shape": list(item.shape),
            "data_offsets": [cursor, end],
        }
        cursor = end
    raw = compact_json_bytes(header)
    raw += b" " * ((-len(raw)) % 8)
    return struct.pack("<Q", len(raw)) + raw, offsets


def _validate_config(source_root: Path) -> dict[str, Any]:
    config = load_json(source_root / "config.json", 1024 * 1024)
    processor = load_json(source_root / "processor_config.json", 1024 * 1024)
    vision = config.get("vision_config")
    image = processor.get("image_processor")
    expected_vision = {
        "default_output_length": 280,
        "dtype": "bfloat16",
        "head_dim": 72,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "num_key_value_heads": 16,
        "patch_size": 16,
        "pooling_kernel_size": 3,
        "position_embedding_size": 10240,
        "rms_norm_eps": 1e-6,
        "standardize": True,
    }
    if not isinstance(vision, dict) or any(vision.get(k) != v for k, v in expected_vision.items()):
        raise DataError("Google QAT Vision config differs from the V00 contract")
    if vision.get("rope_parameters") != {"rope_theta": 100.0, "rope_type": "default"}:
        raise DataError("Google QAT Vision RoPE contract changed")
    expected_image = {
        "do_convert_rgb": True,
        "do_normalize": False,
        "do_rescale": True,
        "do_resize": True,
        "image_seq_length": 280,
        "max_soft_tokens": 280,
        "patch_size": 16,
        "pooling_kernel_size": 3,
        "resample": 3,
        "rescale_factor": 1.0 / 255.0,
    }
    if not isinstance(image, dict) or any(image.get(k) != v for k, v in expected_image.items()):
        raise DataError("Google QAT image processor differs from the V00 contract")
    return {"vision_config": expected_vision, "image_processor": expected_image}


def _copy_artifact(
    path: Path,
    outputs: tuple[VisionOutput, ...],
    selected: dict[str, TensorDescriptor],
    bundle: Any,
    workspace: BoundedWorkspace,
) -> tuple[list[dict[str, Any]], int]:
    header, offsets = _artifact_header(outputs)
    workspace.record_header(len(header) - 8, "building Vision sidecar header")
    records: list[dict[str, Any]] = []
    try:
        with path.open("xb", buffering=0) as stream:
            os.chmod(path, 0o600)
            write_all(stream, header, "Vision sidecar header")
            payload_cursor = 0
            for item in outputs:
                begin, end = offsets[item.name]
                if begin < payload_cursor:
                    raise DataError(f"Vision sidecar output order overlaps: {item.name}")
                if begin != payload_cursor:
                    write_all(
                        stream,
                        bytes(begin - payload_cursor),
                        "Vision sidecar alignment padding",
                    )
                    payload_cursor = begin
                source = selected[item.source_name]
                if item.kind == "bf16_copy":
                    source_hash, output_hash = workspace.copy_range(
                        source.path, source.absolute_offset, source.byte_length, stream
                    )
                    statistics = None
                else:
                    matrix = bundle.matrices[item.name]
                    if item.kind == "fp8_weight":
                        offset, length = matrix.weight_offset, matrix.weight_bytes
                        expected_hash = matrix.telemetry["weight_sha256"]
                    else:
                        offset, length = matrix.scale_offset, matrix.scale_bytes
                        expected_hash = matrix.telemetry["scale_sha256"]
                    source_hash = matrix.source_sha256
                    _unused, output_hash = workspace.copy_range(
                        bundle.payload, offset, length, stream, track_tensor=False
                    )
                    if output_hash != expected_hash:
                        raise DataError(f"native Vision FP8 output hash mismatch: {item.name}")
                    statistics = {
                        "max_absolute_error": matrix.telemetry["max_absolute_error"],
                        "relative_l2_error": (
                            matrix.telemetry["error_sum_squares"]
                            / matrix.telemetry["source_sum_squares"]
                        ) ** 0.5 if matrix.telemetry["source_sum_squares"] else 0.0,
                        "saturation_count": matrix.telemetry["saturation_count"],
                        "zero_rows": matrix.telemetry["zero_rows"],
                    }
                payload_cursor = end
                records.append({
                    "byte_length": item.byte_length,
                    "data_offsets": [begin, end],
                    "dtype": item.dtype,
                    "kind": item.kind,
                    "logical_shape": list(item.shape),
                    "name": item.name,
                    "output_sha256": output_hash,
                    "physical_shape": list(item.shape),
                    "role": item.role,
                    "runtime_layout": "source_nk_fp8" if item.kind == "fp8_weight" else (
                        "row_bf16" if item.kind == "fp8_scale" else "source_bf16"
                    ),
                    "source_name": item.source_name,
                    "source_sha256": source_hash,
                    **({"statistics": statistics} if statistics is not None else {}),
                })
            stream.flush()
            os.fsync(stream.fileno())
    except (OSError, ValueError) as error:
        raise OutputError(f"cannot write Vision sidecar: {error}") from error
    expected_size = len(header) + OUTPUT_PAYLOAD_BYTES
    if path.stat().st_size != expected_size:
        raise DataError(f"Vision sidecar size mismatch: expected {expected_size}, got {path.stat().st_size}")
    return records, len(header)


def compile_vision_sidecar(
    *,
    source_directory: Path,
    output_directory: Path,
    native_encoder: Path,
    source_lock: Path = SOURCE_LOCK_PATH,
    threads: int = 1,
    host_memory_cap_bytes: int = 2 * 1024 * 1024 * 1024,
    staging_bytes: int = 1024 * 1024,
    native_timeout_seconds: int = 1800,
) -> dict[str, Any]:
    source_lock = source_lock.resolve(strict=True)
    if source_lock != SOURCE_LOCK_PATH.resolve(strict=True):
        raise InvalidPlanError(
            "Vision v1 requires the repository-pinned Google QAT BF16 source lock"
        )
    output_directory = output_directory.absolute()
    spec = load_json(SPEC_PATH, 1024 * 1024)
    spec_hash = _file_sha256(SPEC_PATH)
    reference_lock_hash = _file_sha256(REFERENCE_LOCK_PATH)
    workspace = BoundedWorkspace(host_memory_cap_bytes, staging_bytes)
    staging = create_staging_directory(output_directory)
    bundle = None
    try:
        source = verify_source_lock(source_lock, source_directory, workspace)
        expected_source = spec["source"]
        if (
            source.lock_sha256 != expected_source["lock_sha256"]
            or source.repository != expected_source["repository"]
            or source.revision != expected_source["revision"]
        ):
            raise DataError("Vision source is not the pinned Google QAT BF16 checkpoint")
        for filename, key in (
            ("config.json", "config_sha256"),
            ("processor_config.json", "processor_config_sha256"),
            ("model.safetensors.index.json", "index_sha256"),
        ):
            if source.files[filename].sha256 != expected_source[key]:
                raise DataError(f"Vision source metadata hash mismatch: {filename}")
        semantic_contract = _validate_config(source.root)
        source_tensors = read_source_tensors(source, workspace)
        selected = validate_vision_sources(source_tensors)
        outputs = output_plan(selected)
        quantization_plan = fp8_plan(selected)
        bundle = prepare_native_bundle(
            NativeRequest(native_encoder, native_timeout_seconds, threads),
            quantization_plan,
            selected,
            workspace,
            staging,
        )
        artifact_path = staging / VISION_FILENAME
        records, payload_offset = _copy_artifact(
            artifact_path, outputs, selected, bundle, workspace
        )
        artifact_hash = workspace.hash_range(artifact_path, 0, artifact_path.stat().st_size)
        commit, dirty = git_compiler_identity(REPOSITORY_ROOT)
        compilation = {
            "artifact": {
                "filename": VISION_FILENAME,
                "format": spec["artifact"]["format"],
                "payload_bytes": OUTPUT_PAYLOAD_BYTES,
                "payload_offset": payload_offset,
                "sha256": artifact_hash,
                "size": artifact_path.stat().st_size,
                "tensor_count": len(records),
            },
            "capability_profile": PROFILE,
            "compiler": {
                "commit": commit,
                "dirty": dirty,
                "environment": environment_identity(),
                "implementation": "gem16_vision_sidecar_compiler_v1",
                "native_build": bundle.native_build,
                "native_encoder_sha256": bundle.binary_sha256,
                "native_threads": threads,
                "spec_path": SPEC_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
                "spec_sha256": spec_hash,
            },
            "contract_id": spec["contract_id"],
            "contract_version": 1,
            "physical_shapes": spec["physical_shapes"],
            "quantization": {
                **spec["quantization"],
                "bf16_copy_tensor_count": BF16_COPY_COUNT,
                "fp8_linear_count": LINEAR_COUNT,
                "fp8_linear_parameters": LINEAR_PARAMETERS,
                "output_padding_bytes": OUTPUT_PADDING_BYTES,
                "output_payload_bytes": OUTPUT_PAYLOAD_BYTES,
                "output_tensor_bytes": OUTPUT_TENSOR_BYTES,
                "tensor_alignment": TENSOR_ALIGNMENT,
            },
            "reference": {
                "lock_path": REFERENCE_LOCK_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
                "lock_sha256": reference_lock_hash,
                "transformers": spec["transformers_reference"],
            },
            "required_text_artifact_profile": TEXT_ARTIFACT_PROFILE,
            "schema_version": 1,
            "scope": spec["scope"],
            "semantics": semantic_contract,
            "source": {
                "lock_path": source_lock.relative_to(REPOSITORY_ROOT).as_posix(),
                "lock_sha256": source.lock_sha256,
                "repository": source.repository,
                "revision": source.revision,
                "tensor_count": len(selected),
                "tensor_payload_bytes": sum(item.byte_length for item in selected.values()),
            },
            "tensors": records,
            # Persist only deterministic capacity and operation maxima here.
            # Process RSS is useful console diagnostics, but including it in an
            # immutable compilation manifest would make otherwise identical
            # clean builds produce different descriptor and lock hashes.
            "workspace": {
                "host_memory_cap_bytes": workspace.host_memory_cap_bytes,
                "maximum_header_bytes": workspace.max_header_bytes,
                "maximum_mapped_window_bytes": workspace.max_mapped_window_bytes,
                "maximum_tensor_bytes": workspace.max_tensor_bytes,
                "maximum_transform_row_bytes": workspace.maximum_transform_row_bytes,
                "native_maximum_source_row_bytes": bundle.maximum_source_row_bytes,
                "staging_buffer_bytes": workspace.staging_bytes,
            },
        }
        write_file_atomic(staging / COMPILATION_FILENAME, canonical_json_bytes(compilation))
        compilation_hash = _file_sha256(staging / COMPILATION_FILENAME)
        descriptor = {
            "artifact": VISION_FILENAME,
            "artifact_sha256": artifact_hash,
            "artifact_size": artifact_path.stat().st_size,
            "capability_profile": PROFILE,
            "compilation_manifest": COMPILATION_FILENAME,
            "compilation_manifest_sha256": compilation_hash,
            "enablement": "explicit-profile-selection-only",
            "required_text_artifact_profile": TEXT_ARTIFACT_PROFILE,
            "schema_version": 1,
            "supports": {"audio": False, "image": True, "text": True, "video": False},
        }
        write_file_atomic(staging / DESCRIPTOR_FILENAME, canonical_json_bytes(descriptor))
        descriptor_hash = _file_sha256(staging / DESCRIPTOR_FILENAME)
        lock = {
            "artifact_sha256": artifact_hash,
            "artifact_size": artifact_path.stat().st_size,
            "capability_profile": PROFILE,
            "compilation_manifest_sha256": compilation_hash,
            "descriptor_sha256": descriptor_hash,
            "required_text_artifact_profile": TEXT_ARTIFACT_PROFILE,
            "schema_version": 1,
            "source_lock_sha256": source.lock_sha256,
            "source_repository": source.repository,
            "source_revision": source.revision,
        }
        write_file_atomic(staging / LOCK_FILENAME, canonical_json_bytes(lock))
        # The native bundle is compiler scratch, never part of the immutable
        # runtime sidecar.  Remove it while the staging path still exists;
        # after publication its remembered paths would no longer resolve.
        bundle.cleanup()
        bundle = None
        artifact_size = artifact_path.stat().st_size
        publish_staging(staging, output_directory)
        return {
            "artifact_sha256": artifact_hash,
            "artifact_size": artifact_size,
            "output": str(output_directory),
            "payload_bytes": OUTPUT_PAYLOAD_BYTES,
            "tensor_count": len(records),
        }
    except Exception:
        discard_staging(staging)
        raise
    finally:
        if bundle is not None:
            bundle.cleanup()


__all__ = [
    "BF16_COPY_COUNT",
    "COMPILATION_FILENAME",
    "DESCRIPTOR_FILENAME",
    "LINEAR_COUNT",
    "LINEAR_PARAMETERS",
    "LOCK_FILENAME",
    "OUTPUT_PAYLOAD_BYTES",
    "OUTPUT_PADDING_BYTES",
    "OUTPUT_TENSOR_BYTES",
    "OUTPUT_TENSOR_COUNT",
    "PROFILE",
    "SOURCE_TENSOR_BYTES",
    "SOURCE_TENSOR_COUNT",
    "TEXT_ARTIFACT_PROFILE",
    "TENSOR_ALIGNMENT",
    "VISION_FILENAME",
    "compile_vision_sidecar",
    "expected_vision_specs",
    "fp8_plan",
    "is_linear_source",
    "output_plan",
    "validate_vision_sources",
]

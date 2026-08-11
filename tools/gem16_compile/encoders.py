"""Versioned tensor-encoder plugin boundary for the offline compiler."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
import os
from typing import Any, BinaryIO, Iterator, Protocol

from .common import BoundedWorkspace, DataError, tensor_bytes, write_all
from .plan import TensorCompilePlan
from .quantize_fp8 import FP8_CONTRACT_ID, FP8_CONTRACT_VERSION, quantize_bf16_row
from .reader import TensorDescriptor


@dataclass(frozen=True)
class EncoderResult:
    source_sha256: tuple[str, ...]
    output_sha256: str
    output_bytes: int
    statistics: dict[str, object] | None = None


class TensorEncoder(Protocol):
    name: str
    version: int

    def compile_tensor(
        self,
        plan: TensorCompilePlan,
        sources: tuple[TensorDescriptor, ...],
        output: BinaryIO,
        workspace: BoundedWorkspace,
    ) -> EncoderResult: ...


class CopyEncoder:
    name = "copy-v1"
    version = 1

    def compile_tensor(
        self,
        plan: TensorCompilePlan,
        sources: tuple[TensorDescriptor, ...],
        output: BinaryIO,
        workspace: BoundedWorkspace,
    ) -> EncoderResult:
        if len(sources) != 1:
            raise DataError("copy-v1 received more than one source tensor")
        source = sources[0]
        if source.byte_length != plan.output_bytes:
            raise DataError(
                f"copy-v1 byte mismatch for {plan.output_name}: "
                f"source={source.byte_length} output={plan.output_bytes}"
            )
        source_hash, output_hash = workspace.copy_range(
            source.path, source.absolute_offset, source.byte_length, output
        )
        return EncoderResult(
            source_sha256=(source_hash,),
            output_sha256=output_hash,
            output_bytes=source.byte_length,
        )


def _validate_fp8_plan(
    plan: TensorCompilePlan,
    sources: tuple[TensorDescriptor, ...],
    component: str,
) -> tuple[TensorDescriptor, int, int]:
    if len(sources) != 1:
        raise DataError(f"{plan.encoder} received more than one source tensor")
    source = sources[0]
    if source.dtype != "BF16" or len(source.shape) != 2:
        raise DataError(
            f"{plan.encoder} source must be one 2D BF16 tensor: "
            f"dtype={source.dtype!r} shape={source.shape!r}"
        )
    rows, columns = source.shape
    if rows <= 0 or columns <= 0:
        raise DataError(f"{plan.encoder} source shape must be non-empty")
    expected_source_bytes = tensor_bytes("BF16", source.shape, source.name)
    if source.byte_length != expected_source_bytes:
        raise DataError(
            f"{plan.encoder} source byte mismatch: "
            f"expected={expected_source_bytes} got={source.byte_length}"
        )
    expected_dtype = "F8_E4M3" if component == "weight" else "BF16"
    expected_shape = (rows, columns) if component == "weight" else (rows, 1)
    if plan.output_dtype != expected_dtype or plan.physical_shape != expected_shape:
        raise DataError(
            f"{plan.encoder} output plan mismatch: "
            f"expected {expected_dtype} {expected_shape}, got "
            f"{plan.output_dtype} {plan.physical_shape}"
        )
    if plan.output_bytes != tensor_bytes(expected_dtype, expected_shape, plan.output_name):
        raise DataError(f"{plan.encoder} output byte calculation mismatch")
    return source, rows, columns


@contextmanager
def _open_source_rows(
    source: TensorDescriptor,
    workspace: BoundedWorkspace,
    rows: int,
    row_bytes: int,
    operation: str,
) -> Iterator[Iterator[tuple[int, memoryview, Any]]]:
    if workspace.staging_bytes < row_bytes:
        raise DataError(
            f"{operation} requires staging for one source row: "
            f"row={row_bytes} staging={workspace.staging_bytes}"
        )
    if source.absolute_offset < 0 or source.byte_length < 0:
        raise DataError(f"invalid source range for {source.name}")
    try:
        with source.path.open("rb", buffering=0) as stream:
            file_size = os.fstat(stream.fileno()).st_size
            if source.absolute_offset > file_size - source.byte_length:
                raise DataError(
                    f"source range is outside {source.path}: "
                    f"offset={source.absolute_offset} length={source.byte_length} "
                    f"size={file_size}"
                )
            stream.seek(source.absolute_offset)
            digest = hashlib.sha256()
            view = memoryview(workspace.buffer)[:row_bytes]
            try:
                def rows_iter() -> Iterator[tuple[int, memoryview, Any]]:
                    for row in range(rows):
                        workspace.record_transform_row(
                            row_bytes, f"reading {source.name} row {row}"
                        )
                        position = 0
                        while position < row_bytes:
                            try:
                                count = stream.readinto(view[position:])
                            except OSError as error:
                                raise DataError(
                                    f"cannot read source tensor {source.name}: {error}"
                                ) from error
                            if not count:
                                raise DataError(
                                    f"short source row for {source.name}: "
                                    f"expected {row_bytes}, got {position}"
                                )
                            position += count
                        digest.update(view)
                        yield row, view, digest
                        workspace.check(f"encoding {source.name} row {row}")
                yield rows_iter()
            finally:
                view.release()
    except DataError:
        raise
    except (OSError, ValueError) as error:
        raise DataError(f"cannot open/read source tensor {source.name}: {error}") from error


def _weight_statistics(
    rows: int,
    columns: int,
    source_min: float,
    source_max: float,
    source_sum_squares: float,
    reconstruction_sum_squares: float,
    source_reconstruction_dot: float,
    error_sum_squares: float,
    max_absolute_error: float,
    scale_min: float,
    scale_max: float,
    saturation_count: int,
    zero_rows: int,
    underflow_clamped_rows: int,
    histogram: list[int],
) -> dict[str, object]:
    elements = rows * columns
    source_norm = math.sqrt(source_sum_squares)
    reconstruction_norm = math.sqrt(reconstruction_sum_squares)
    error_norm = math.sqrt(error_sum_squares)
    relative_l2 = error_norm / source_norm if source_norm else 0.0
    cosine = (
        source_reconstruction_dot / (source_norm * reconstruction_norm)
        if source_norm and reconstruction_norm
        else 1.0 if source_norm == reconstruction_norm == 0.0 else 0.0
    )
    cosine = max(-1.0, min(1.0, cosine))
    perfect = error_sum_squares == 0.0
    sqnr: float | None
    if perfect or source_sum_squares == 0.0:
        sqnr = None
    else:
        sqnr = 10.0 * math.log10(source_sum_squares / error_sum_squares)
    values: dict[str, object] = {
        "contract_id": FP8_CONTRACT_ID,
        "contract_version": FP8_CONTRACT_VERSION,
        "component": "weight",
        "rows": rows,
        "columns": columns,
        "elements": elements,
        "source_min": source_min,
        "source_max": source_max,
        "source_rms": math.sqrt(source_sum_squares / elements),
        "scale_min": scale_min,
        "scale_max": scale_max,
        "relative_l2_error": relative_l2,
        "cosine_similarity": cosine,
        "max_absolute_error": max_absolute_error,
        "sqnr_db": sqnr,
        "perfect_reconstruction": perfect,
        "saturation_count": saturation_count,
        "saturation_rate": saturation_count / elements,
        "zero_rows": zero_rows,
        "underflow_clamped_rows": underflow_clamped_rows,
        "histogram": histogram,
    }
    for key, value in values.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise DataError(f"non-finite FP8 statistic: {key}")
    return values


class FP8RowwiseWeightEncoder:
    name = "fp8-rowwise-weight-v1"
    version = 1

    def compile_tensor(
        self,
        plan: TensorCompilePlan,
        sources: tuple[TensorDescriptor, ...],
        output: BinaryIO,
        workspace: BoundedWorkspace,
    ) -> EncoderResult:
        source, rows, columns = _validate_fp8_plan(plan, sources, "weight")
        row_bytes = columns * 2
        output_digest = hashlib.sha256()
        source_hash = ""
        source_min = math.inf
        source_max = -math.inf
        source_sum_squares = 0.0
        reconstruction_sum_squares = 0.0
        source_reconstruction_dot = 0.0
        error_sum_squares = 0.0
        max_absolute_error = 0.0
        scale_min = math.inf
        scale_max = -math.inf
        saturation_count = 0
        zero_rows = 0
        underflow_clamped_rows = 0
        histogram = [0] * 256
        with _open_source_rows(
            source, workspace, rows, row_bytes, self.name
        ) as source_rows:
            for row, view, digest in source_rows:
                result = quantize_bf16_row(view)
                payload = result.weight_bytes
                write_all(output, payload, f"{plan.output_name} row {row}")
                output_digest.update(payload)
                source_hash = digest.hexdigest()
                metrics = result.metrics
                source_min = min(source_min, metrics.source_min)
                source_max = max(source_max, metrics.source_max)
                source_sum_squares += metrics.source_sum_squares
                reconstruction_sum_squares += metrics.reconstruction_sum_squares
                source_reconstruction_dot += metrics.source_reconstruction_dot
                error_sum_squares += metrics.error_sum_squares
                max_absolute_error = max(max_absolute_error, metrics.max_absolute_error)
                scale_min = min(scale_min, result.scale_value)
                scale_max = max(scale_max, result.scale_value)
                saturation_count += result.saturation_count
                zero_rows += int(result.zero_row)
                underflow_clamped_rows += int(result.scale_underflow_clamped)
                for code, count in enumerate(result.histogram):
                    histogram[code] += count
                workspace.check(f"writing {plan.output_name} row {row}")
        statistics = _weight_statistics(
            rows, columns, source_min, source_max, source_sum_squares,
            reconstruction_sum_squares, source_reconstruction_dot,
            error_sum_squares, max_absolute_error, scale_min, scale_max,
            saturation_count, zero_rows, underflow_clamped_rows, histogram,
        )
        return EncoderResult(
            source_sha256=(source_hash,),
            output_sha256=output_digest.hexdigest(),
            output_bytes=rows * columns,
            statistics=statistics,
        )


class FP8RowwiseScaleEncoder:
    name = "fp8-rowwise-scale-v1"
    version = 1

    def compile_tensor(
        self,
        plan: TensorCompilePlan,
        sources: tuple[TensorDescriptor, ...],
        output: BinaryIO,
        workspace: BoundedWorkspace,
    ) -> EncoderResult:
        source, rows, columns = _validate_fp8_plan(plan, sources, "scale")
        row_bytes = columns * 2
        output_digest = hashlib.sha256()
        source_hash = ""
        scale_min = math.inf
        scale_max = -math.inf
        zero_rows = 0
        underflow_clamped_rows = 0
        with _open_source_rows(
            source, workspace, rows, row_bytes, self.name
        ) as source_rows:
            for row, view, digest in source_rows:
                result = quantize_bf16_row(view)
                payload = result.scale_bytes
                write_all(output, payload, f"{plan.output_name} row {row}")
                output_digest.update(payload)
                source_hash = digest.hexdigest()
                scale_min = min(scale_min, result.scale_value)
                scale_max = max(scale_max, result.scale_value)
                zero_rows += int(result.zero_row)
                underflow_clamped_rows += int(result.scale_underflow_clamped)
                workspace.check(f"writing {plan.output_name} row {row}")
        statistics = {
            "contract_id": FP8_CONTRACT_ID,
            "contract_version": FP8_CONTRACT_VERSION,
            "component": "scale",
            "rows": rows,
            "columns": 1,
            "scale_min": scale_min,
            "scale_max": scale_max,
            "zero_rows": zero_rows,
            "underflow_clamped_rows": underflow_clamped_rows,
        }
        return EncoderResult(
            source_sha256=(source_hash,),
            output_sha256=output_digest.hexdigest(),
            output_bytes=rows * 2,
            statistics=statistics,
        )


def default_encoder_registry() -> dict[str, TensorEncoder]:
    # M05 production conversion is native-only.  Keep the Python encoders
    # importable as diagnostic/reference implementations, but never register
    # them as a silent compiler fallback.
    encoder = CopyEncoder()
    return {encoder.name: encoder}

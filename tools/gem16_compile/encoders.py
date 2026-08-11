"""Versioned tensor-encoder plugin boundary for the offline compiler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO, Protocol

from .common import BoundedWorkspace, DataError
from .plan import TensorCompilePlan
from .reader import TensorDescriptor


@dataclass(frozen=True)
class EncoderResult:
    source_sha256: tuple[str, ...]
    output_sha256: str
    output_bytes: int


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


def default_encoder_registry() -> dict[str, TensorEncoder]:
    encoder = CopyEncoder()
    return {encoder.name: encoder}

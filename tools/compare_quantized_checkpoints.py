#!/usr/bin/env python3
"""Compare compiled Gemma 4 26B FP8 attention weights with locked Unsloth."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Sequence

try:
    from tools.gem16_compile.common import (
        BoundedWorkspace, CompilerError, InvalidPlanError, OutputError,
        canonical_json_bytes, write_file_atomic,
    )
    from tools.gem16_compile.compiler import CompilerRequest, verify_artifact
    from tools.gem16_compile.fp8_report import compare_attention, write_report
    from tools.gem16_compile.profiles import M05_SOURCE_CONTRACT, M05_SOURCE_LOCK_SHA256
except ModuleNotFoundError:  # Direct execution outside repository root.
    from gem16_compile.common import (  # type: ignore[no-redef]
        BoundedWorkspace, CompilerError, InvalidPlanError, OutputError,
        canonical_json_bytes, write_file_atomic,
    )
    from gem16_compile.compiler import CompilerRequest, verify_artifact  # type: ignore[no-redef]
    from gem16_compile.fp8_report import compare_attention, write_report  # type: ignore[no-redef]
    from gem16_compile.profiles import M05_SOURCE_CONTRACT, M05_SOURCE_LOCK_SHA256  # type: ignore[no-redef]


def positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("attention",), required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--compiled-source-lock", type=Path, required=True)
    parser.add_argument("--compiled-source", type=Path, required=True)
    parser.add_argument("--compiled-plan", type=Path, required=True)
    parser.add_argument("--unsloth-lock", type=Path, required=True)
    parser.add_argument("--unsloth-source", type=Path, required=True)
    parser.add_argument("--max-host-memory", type=positive_integer, required=True)
    parser.add_argument("--staging-bytes", type=positive_integer, default=1024 * 1024)
    parser.add_argument("--native-encoder", type=Path, required=True)
    parser.add_argument("--threads", type=positive_integer, default=1)
    parser.add_argument("--native-timeout-seconds", type=positive_integer, default=600)
    parser.add_argument(
        "--allow-dirty-compiled", action="store_true",
        help="diagnostic only: compare an artifact whose compiler provenance records dirty=true",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compiled-verify-report", type=Path,
        help="retain the single compiler-verification report performed before comparison",
    )
    return parser


def _hash_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise OutputError(f"cannot hash comparison input {path}: {error}") from error


def _reject_output_overlap(output: Path, inputs: Sequence[Path]) -> None:
    candidate = output.expanduser().resolve(strict=False)
    for raw in inputs:
        base = raw.expanduser().resolve(strict=False)
        try:
            candidate.relative_to(base)
            raise OutputError(f"comparison output must not be inside input {raw}: {output}")
        except ValueError:
            pass
        if candidate == base:
            raise OutputError(f"comparison output must not replace input {raw}: {output}")
    if output.exists() or output.is_symlink():
        raise OutputError(f"comparison report already exists: {output}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.family != "attention":  # argparse owns this branch.
            raise InvalidPlanError(f"unsupported comparison family: {args.family}")
        inputs = (
            args.compiled, args.compiled_source, args.unsloth_source,
            args.compiled_source_lock, args.unsloth_lock, args.compiled_plan,
            args.native_encoder,
        )
        _reject_output_overlap(args.output, inputs)
        if args.compiled_verify_report is not None:
            _reject_output_overlap(args.compiled_verify_report, (*inputs, args.output))
        compiled_lock_hash = _hash_file(args.compiled_source_lock)
        if compiled_lock_hash != M05_SOURCE_LOCK_SHA256["ordinary_bf16"]:
            raise InvalidPlanError(
                "comparison requires the approved Ordinary-BF16 source lock, not QAT"
            )
        request = CompilerRequest(
            source_lock=args.compiled_source_lock,
            source_directory=args.compiled_source,
            compiler_manifest=args.compiled_plan,
            profile="fp8-attention-partial-v1",
            head_format="deferred",
            host_memory_cap_bytes=args.max_host_memory,
            staging_bytes=args.staging_bytes,
            threads=args.threads,
            reference_platform_strict=True,
        )
        verified = verify_artifact(request, args.compiled)
        if args.compiled_verify_report is not None:
            args.compiled_verify_report.parent.mkdir(parents=True, exist_ok=True)
            write_file_atomic(
                args.compiled_verify_report, canonical_json_bytes(verified)
            )
        if verified.get("artifact_profile") != "fp8-attention-partial-v1":
            raise InvalidPlanError("compiler verification did not produce the M05 profile")
        report = compare_attention(
            args.compiled,
            args.unsloth_lock,
            args.unsloth_source,
            BoundedWorkspace(args.max_host_memory, args.staging_bytes),
            verified_report=verified,
            production=True,
            native_encoder=args.native_encoder,
            threads=args.threads,
            native_timeout_seconds=args.native_timeout_seconds,
            allow_dirty_compiler=args.allow_dirty_compiled,
        )
        if report["compiled"].get("source_contract") != M05_SOURCE_CONTRACT:
            raise InvalidPlanError("compiled comparison provenance has the wrong source contract")
        write_report(args.output, report)
    except CompilerError as error:
        print(f"error: {error}", file=sys.stderr)
        return error.exit_code
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 4
    print("comparison passed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

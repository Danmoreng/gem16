#!/usr/bin/env python3
"""Compile or verify deterministic Gemma 4 26B derived checkpoint artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

try:
    from tools.gem16_compile.common import (
        CompilerError,
        InvalidPlanError,
        canonical_json_bytes,
        write_file_atomic,
    )
    from tools.gem16_compile.compiler import (
        DEPENDENCIES_LOCK,
        CompilerRequest,
        compare_reproducibility,
        compile_artifact,
        plan_artifact,
        verify_artifact,
    )
    from tools.hf_cache import hub_cache_root, locked_snapshot_path
except ModuleNotFoundError:  # Direct execution from outside the repository root.
    from gem16_compile.common import (  # type: ignore[no-redef]
        CompilerError,
        InvalidPlanError,
        canonical_json_bytes,
        write_file_atomic,
    )
    from gem16_compile.compiler import (  # type: ignore[no-redef]
        DEPENDENCIES_LOCK,
        CompilerRequest,
        compare_reproducibility,
        compile_artifact,
        plan_artifact,
        verify_artifact,
    )
    from hf_cache import hub_cache_root, locked_snapshot_path


def positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def add_source_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-lock", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--source-directory",
        type=Path,
        help="explicit locked snapshot directory; every file is still verified",
    )
    source.add_argument(
        "--source-cache",
        type=Path,
        help="Hugging Face hub cache root containing the locked snapshot",
    )
    parser.add_argument("--compiler-manifest", type=Path, required=True)
    parser.add_argument(
        "--profile",
        required=True,
        help=(
            "versioned compiler profile: synthetic-copy-v1 (M04), "
            "fp8-attention-partial-v1 (M05), nvfp4-experts-partial-v1 (M06), "
            "nvfp4-tied-head-partial-v1 (M07), or sm120-text-hybrid-v1 (M08)"
        ),
    )
    parser.add_argument(
        "--head-format",
        choices=("source", "q4_0", "nvfp4", "deferred"),
        required=True,
        help="profile-bound embedding/head format; M05 uses deferred",
    )
    parser.add_argument("--max-host-memory", type=positive_integer, required=True)
    parser.add_argument("--staging-bytes", type=positive_integer, default=1024 * 1024)
    parser.add_argument("--shard-size", type=positive_integer)
    parser.add_argument(
        "--threads", type=positive_integer, default=None,
        help="worker threads (M04/M05 default: 1; M06/M07 default: min(16, CPU count))",
    )
    parser.add_argument("--dependencies-lock", type=Path, default=DEPENDENCIES_LOCK)
    parser.add_argument("--reference-platform-strict", action="store_true")
    parser.add_argument(
        "--verify-source",
        action="store_true",
        help="retained for CLI compatibility; source verification is always mandatory",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    plan = subparsers.add_parser("plan", help="verify source and emit a dry-run plan")
    add_source_options(plan)
    plan.add_argument("--report", type=Path, required=True)
    plan.add_argument("--dry-run", action="store_true")

    compile_parser = subparsers.add_parser(
        "compile", help="build, verify, and atomically publish an artifact"
    )
    add_source_options(compile_parser)
    compile_parser.add_argument("--output", type=Path, required=True)
    compile_parser.add_argument("--report", type=Path, required=True)
    compile_parser.add_argument(
        "--artifact-lock", type=Path,
        help="required for M08: external immutable derived-artifact lock",
    )
    compile_parser.add_argument("--resume", action="store_true")
    compile_parser.add_argument(
        "--native-encoder", type=Path,
        help="required for native M05/M06/M07 profiles: path to the matching native compiler",
    )
    compile_parser.add_argument(
        "--native-fp8-encoder", type=Path,
        help="required for M08: native FP8 compiler",
    )
    compile_parser.add_argument(
        "--native-nvfp4-encoder", type=Path,
        help="required for M08: native NVFP4 compiler",
    )
    compile_parser.add_argument(
        "--native-timeout-seconds", type=positive_integer, default=None,
        help="native compiler timeout in seconds (M05 default: 600; M06/M07 default: 14400)",
    )
    compile_parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="diagnostic only; release artifacts require a clean compiler tree",
    )

    verify = subparsers.add_parser(
        "verify", help="recompute source, plan, file, and tensor hashes"
    )
    add_source_options(verify)
    verify.add_argument("--model", type=Path, required=True)
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument(
        "--artifact-lock", type=Path,
        help="required for M08: external immutable derived-artifact lock",
    )

    compare = subparsers.add_parser(
        "compare-reproducibility", help="compare every file in two artifacts"
    )
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
    compare.add_argument("--report", type=Path, required=True)
    return parser


def resolve_source(args: argparse.Namespace) -> Path:
    if args.source_directory is not None:
        return args.source_directory
    cache = args.source_cache or hub_cache_root()
    return locked_snapshot_path(args.source_lock, cache)


def request_from_args(args: argparse.Namespace) -> CompilerRequest:
    return CompilerRequest(
        source_lock=args.source_lock,
        source_directory=resolve_source(args),
        compiler_manifest=args.compiler_manifest,
        profile=args.profile,
        head_format=args.head_format,
        host_memory_cap_bytes=args.max_host_memory,
        staging_bytes=args.staging_bytes,
        shard_size=args.shard_size,
        threads=(
            args.threads if args.threads is not None else
            (min(16, os.cpu_count() or 1) if args.profile in {"nvfp4-experts-partial-v1", "nvfp4-tied-head-partial-v1", "sm120-text-hybrid-v1"} else 1)
        ),
        reference_platform_strict=args.reference_platform_strict,
        dependencies_lock=args.dependencies_lock,
        native_encoder=getattr(args, "native_encoder", None),
        native_fp8_encoder=getattr(args, "native_fp8_encoder", None),
        native_nvfp4_encoder=getattr(args, "native_nvfp4_encoder", None),
        artifact_lock=getattr(args, "artifact_lock", None),
        native_timeout_seconds=(
            getattr(args, "native_timeout_seconds", None)
            or (14_400 if args.profile in {"nvfp4-experts-partial-v1", "nvfp4-tied-head-partial-v1", "sm120-text-hybrid-v1"} else 600)
        ),
    )


def write_failure_report(args: argparse.Namespace, error: CompilerError) -> None:
    report = getattr(args, "report", None)
    if report is None or report.exists() or report.is_symlink():
        return
    try:
        report.parent.mkdir(parents=True, exist_ok=True)
        write_file_atomic(
            report,
            canonical_json_bytes(
                {
                    "schema_version": 1,
                    "milestone": (
                        "M08"
                        if getattr(args, "profile", None)
                        == "sm120-text-hybrid-v1"
                        else "M05"
                        if getattr(args, "profile", None)
                        == "fp8-attention-partial-v1"
                        else "M07"
                        if getattr(args, "profile", None)
                        == "nvfp4-tied-head-partial-v1"
                        else "M06"
                        if getattr(args, "profile", None)
                        == "nvfp4-experts-partial-v1"
                        else "M04"
                    ),
                    "action": args.action,
                    "status": "fail",
                    "exit_code": error.exit_code,
                    "error": str(error),
                }
            )
        )
    except (OSError, CompilerError):
        pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.action == "compare-reproducibility":
            print("[compare] hashing both artifacts", file=sys.stderr)
            report = compare_reproducibility(
                args.left, args.right, report_path=args.report
            )
        else:
            request = request_from_args(args)
            if args.action == "plan":
                print("[verify] source lock and files", file=sys.stderr)
                report = plan_artifact(request)
                if args.report.exists() or args.report.is_symlink():
                    raise InvalidPlanError(f"compiler report already exists: {args.report}")
                args.report.parent.mkdir(parents=True, exist_ok=True)
                write_file_atomic(args.report, canonical_json_bytes(report))
            elif args.action == "compile":
                if args.resume:
                    raise InvalidPlanError(
                        "compiler resume is intentionally unsupported; remove only a reviewed "
                        ".incomplete directory and restart"
                    )
                if args.profile in {"nvfp4-experts-partial-v1", "nvfp4-tied-head-partial-v1", "sm120-text-hybrid-v1"}:
                    print(
                        "[preflight] native Release compiler, then source lock and files",
                        file=sys.stderr,
                    )
                else:
                    print("[verify] source lock and files", file=sys.stderr)
                if args.profile == "fp8-attention-partial-v1":
                    print(
                        "[compile] deterministic FP8 attention partial artifact",
                        file=sys.stderr,
                    )
                elif args.profile in {"nvfp4-experts-partial-v1", "nvfp4-tied-head-partial-v1"}:
                    print(
                        "[compile] deterministic NVFP4 partial artifact",
                        file=sys.stderr,
                    )
                elif args.profile == "sm120-text-hybrid-v1":
                    print(
                        "[compile] deterministic complete M08 hybrid artifact",
                        file=sys.stderr,
                    )
                else:
                    print("[compile] deterministic copy scaffold", file=sys.stderr)
                report = compile_artifact(
                    request,
                    args.output,
                    report_path=args.report,
                    allow_dirty=args.allow_dirty,
                )
            elif args.action == "verify":
                print("[verify] source, artifact, provenance, and hashes", file=sys.stderr)
                report = verify_artifact(
                    request, args.model, report_path=args.report
                )
            else:  # pragma: no cover - argparse owns the action set.
                raise InvalidPlanError(f"unknown compiler action: {args.action}")
    except CompilerError as error:
        write_failure_report(args, error)
        print(f"error: {error}", file=sys.stderr)
        return error.exit_code
    except (OSError, ValueError) as error:
        wrapped = InvalidPlanError(str(error))
        write_failure_report(args, wrapped)
        print(f"error: {wrapped}", file=sys.stderr)
        return wrapped.exit_code
    display_report = report
    if report.get("artifact_profile") == "sm120-text-hybrid-v1":
        display_report = {
            key: report[key]
            for key in (
                "schema_version", "milestone", "action", "status", "output",
                "artifact", "source_lock_sha256", "compiler_manifest_sha256",
                "compilation_manifest_sha256", "artifact_lock_sha256",
                "artifact_content_sha256", "output_tensor_count",
                "output_tensor_bytes", "duration_seconds",
            )
            if key in report
        }
    print(json.dumps(display_report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run and reconcile real Gemma 4 26B M21 context executions.

The C++ driver owns the GPU execution and writes its large final-logit sample
to a temporary directory. This control plane retains only hashes and compact
measurements. A context is supported only after two fresh-process runs agree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
import urllib.parse


MIB = 1024 * 1024
MAX_CONTEXT = 262_144


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_state() -> dict[str, Any]:
    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments], text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        commit = git("rev-parse", "HEAD")
        repository = git("config", "--get", "remote.origin.url")
        if "://" in repository:
            parsed = urllib.parse.urlsplit(repository)
            host = parsed.hostname or "redacted"
            if parsed.port is not None:
                host = f"{host}:{parsed.port}"
            repository = urllib.parse.urlunsplit(
                (parsed.scheme, host, parsed.path, parsed.query, parsed.fragment)
            )
        elif "@" in repository and ":" in repository:
            repository = repository.split("@", 1)[1]
        dirty = bool(git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        commit, repository, dirty = "unknown", "unknown", True
    return {"repository": repository, "commit": commit, "dirty": dirty}


def artifact_identity(model: Path) -> dict[str, Any]:
    lock = model.parent / f"{model.name}.lock.json"
    if not lock.is_file() or lock.is_symlink() or lock.stat().st_size > 16 * MIB:
        raise RuntimeError(f"compiled artifact lock is missing or unsafe: {lock}")
    try:
        document = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot parse compiled artifact lock: {error}") from error
    if not isinstance(document, dict):
        raise RuntimeError("compiled artifact lock root is not an object")
    result = {
        "profile": "native_sm120_integrated_prefill_decode_head",
        "artifact_content_sha256": document.get("artifact_content_sha256"),
        "artifact_lock_sha256": sha256(lock),
        "source_lock_sha256": document.get("source_lock_sha256"),
    }
    if any(
        not isinstance(result[field], str)
        or len(result[field]) != 64
        or any(character not in "0123456789abcdef" for character in result[field])
        for field in (
            "artifact_content_sha256",
            "artifact_lock_sha256",
            "source_lock_sha256",
        )
    ):
        raise RuntimeError("compiled artifact lock identity is incomplete")
    return result


def parse_contexts(value: str) -> list[int]:
    try:
        contexts = sorted({int(item.strip()) for item in value.split(",")})
    except ValueError as error:
        raise argparse.ArgumentTypeError("contexts must be comma-separated integers") from error
    if not contexts or any(context <= 1 or context > MAX_CONTEXT for context in contexts):
        raise argparse.ArgumentTypeError(f"contexts must be within 2..{MAX_CONTEXT}")
    return contexts


def expected_margin(context: int) -> int:
    return (400 if context >= 65_536 else 700) * MIB


def driver_command(executable: Path) -> list[str]:
    if executable.suffix.casefold() == ".py":
        return [sys.executable, str(executable)]
    return [str(executable)]


def validate_run(payload: dict[str, Any], context: int) -> list[str]:
    errors: list[str] = []
    memory = payload.get("memory")
    checks = {
        "milestone": payload.get("milestone") == "M21",
        "backend": payload.get("backend") == "native_sm120_integrated",
        "context": payload.get("context_tokens") == context,
        "prompt_extent": payload.get("prompt_tokens") == context - 1,
        "final_position": payload.get("final_position") == context,
        "finite": payload.get("all_logits_finite") is True,
        "over_limit": payload.get("over_limit_rejected") is True,
        "ring_wrap": payload.get("sliding_ring_wrap_exercised") is True,
        "ring_wrap_count": isinstance(payload.get("sliding_ring_wrap_count"), int)
        and payload.get("sliding_ring_wrap_count", 0) > 0,
        "global_extent": payload.get("global_extent_exercised") is True,
        "global_extent_position": payload.get("maximum_global_position_exclusive")
        == context,
        "fallbacks": payload.get("fallback_count") == 0,
        "engine_allocations": payload.get("recurring_allocation_count") == 0,
        "chunks": isinstance(payload.get("prefill_chunk_count"), int)
        and payload.get("prefill_chunk_count", 0) > 0,
        "minimum_chunk": isinstance(payload.get("minimum_prefill_chunk_tokens"), int)
        and payload.get("minimum_prefill_chunk_tokens", 0) > 0,
        "memory_object": isinstance(memory, dict),
    }
    if isinstance(memory, dict):
        checks.update(
            {
                "reported_margin": memory.get("required_margin_bytes")
                == expected_margin(context),
                "margin": memory.get("margin_pass") is True
                and isinstance(memory.get("free_after_decode_bytes"), int)
                and memory["free_after_decode_bytes"] >= expected_margin(context),
                "no_recurring_allocation": memory.get("free_after_prefill_bytes")
                == memory.get("free_after_decode_bytes"),
                "separate_kv": isinstance(memory.get("kv_cache_bytes"), int)
                and memory["kv_cache_bytes"] > 0,
                "fixed_workspace": isinstance(memory.get("workspace_bytes"), int)
                and memory["workspace_bytes"] > 0,
            }
        )
    errors.extend(name for name, passed in checks.items() if not passed)
    return errors


def run_context(
    executable: Path,
    model: Path,
    context: int,
    runs: int,
    device: int,
    timeout: int,
    temporary: Path,
) -> dict[str, Any]:
    retained: list[dict[str, Any]] = []
    for run_index in range(runs):
        report = temporary / f"context-{context}-run-{run_index}.json"
        logits = temporary / f"context-{context}-run-{run_index}.f32le"
        command = [
            *driver_command(executable),
            "--model",
            str(model),
            "--output",
            str(report),
            "--logits",
            str(logits),
            "--context",
            str(context),
            "--prompt-tokens",
            str(context - 1),
            "--device",
            str(device),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout if isinstance(error.stdout, str) else ""
            stderr = error.stderr if isinstance(error.stderr, str) else ""
            retained.append(
                {
                    "run": run_index,
                    "status": "failed",
                    "failure_kind": "timeout",
                    "exit_code": None,
                    "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
                    "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
                    "error_tail": (stderr or stdout)[-1000:],
                }
            )
            break
        (temporary / f"context-{context}-run-{run_index}.stdout.txt").write_text(
            completed.stdout, encoding="utf-8"
        )
        (temporary / f"context-{context}-run-{run_index}.stderr.txt").write_text(
            completed.stderr, encoding="utf-8"
        )
        record: dict[str, Any] = {
            "run": run_index,
            "exit_code": completed.returncode,
            "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
        }
        if completed.returncode != 0 or not report.is_file() or not logits.is_file():
            record["status"] = "failed"
            record["failure_kind"] = (
                "capacity_rejected" if completed.returncode == 20
                else "driver_error"
            )
            record["error_tail"] = (completed.stderr or completed.stdout)[-1000:]
            retained.append(record)
            if record["failure_kind"] == "capacity_rejected":
                continue
            break
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            record.update(
                {
                    "status": "failed",
                    "failure_kind": "malformed_report",
                    "error_tail": str(error)[-1000:],
                }
            )
            retained.append(record)
            break
        if not isinstance(payload, dict):
            record.update(
                {
                    "status": "failed",
                    "failure_kind": "malformed_report",
                    "error_tail": "driver report root is not an object",
                }
            )
            retained.append(record)
            break
        errors = validate_run(payload, context)
        record.update(
            {
                "status": "passed" if not errors else "failed",
                "failure_kind": None if not errors else "validation_error",
                "validation_errors": errors,
                "logits_sha256": sha256(logits),
                "prefill_elapsed_ms": payload.get("prefill_elapsed_ms"),
                "decode_elapsed_ms": payload.get("decode_elapsed_ms"),
                "prefill_prediction_token": payload.get("prefill_prediction_token"),
                "decode_prediction_token": payload.get("decode_prediction_token"),
                "prefill_chunk_count": payload.get("prefill_chunk_count"),
                "minimum_prefill_chunk_tokens": payload.get(
                    "minimum_prefill_chunk_tokens"
                ),
                "sliding_ring_wrap_count": payload.get("sliding_ring_wrap_count"),
                "maximum_global_position_exclusive": payload.get(
                    "maximum_global_position_exclusive"
                ),
                "memory": payload.get("memory"),
            }
        )
        retained.append(record)
        if errors:
            break

    successful = len(retained) == runs and all(
        record.get("status") == "passed" for record in retained
    )
    deterministic = successful and len(
        {
            (
                record["logits_sha256"],
                record["prefill_prediction_token"],
                record["decode_prediction_token"],
            )
            for record in retained
        }
    ) == 1
    capacity_rejected = len(retained) == runs and all(
        record.get("failure_kind") == "capacity_rejected" for record in retained
    )
    return {
        "context_tokens": context,
        "status": (
            "passed" if successful and deterministic
            else "capacity_rejected" if capacity_rejected
            else "failed"
        ),
        "fresh_process_deterministic": deterministic,
        "capacity_rejection_reproducible": capacity_rejected,
        "runs": retained,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-executable", type=Path, required=True)
    parser.add_argument("--toolchain-lock", type=Path, required=True)
    parser.add_argument("--contexts", type=parse_contexts, default=parse_contexts("32768,65536"))
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=21_600)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument(
        "--maximum-gap-tokens", type=int, default=4096,
        help="largest permitted gap between the highest pass and first capacity rejection",
    )
    args = parser.parse_args()
    driver = args.driver.resolve()
    model = args.model.resolve()
    benchmark_executable = args.benchmark_executable.resolve()
    toolchain_lock = args.toolchain_lock.resolve()
    if (args.runs < 2 or args.device < 0 or args.timeout <= 0 or
            args.maximum_gap_tokens <= 0):
        parser.error("runs must be >=2 and device/timeout/gap must be positive")
    if (
        not driver.is_file()
        or not benchmark_executable.is_file()
        or not toolchain_lock.is_file()
        or not model.is_dir()
    ):
        parser.error(
            "driver, benchmark executable and toolchain lock must be files and model must be a directory"
        )
    required_files = [model / "config.json", model / "gem16_compilation.json"]
    if any(not path.is_file() for path in required_files):
        parser.error("model is not a compiled Gemma 4 26B artifact")

    def collect(temporary: Path) -> list[dict[str, Any]]:
        temporary.mkdir(parents=True, exist_ok=True)
        if any(temporary.iterdir()):
            raise RuntimeError(f"refusing to reuse non-empty raw directory: {temporary}")
        return [
            run_context(
                driver,
                model,
                context,
                args.runs,
                args.device,
                args.timeout,
                temporary,
            )
            for context in args.contexts
        ]

    if args.raw_dir is None:
        with tempfile.TemporaryDirectory(prefix="gem16-m21-") as directory:
            contexts = collect(Path(directory))
    else:
        try:
            contexts = collect(args.raw_dir.resolve())
        except (OSError, RuntimeError) as error:
            parser.error(str(error))

    code = repository_state()
    try:
        model_identity = artifact_identity(model)
    except RuntimeError as error:
        parser.error(str(error))
    passed_contexts = [
        item["context_tokens"] for item in contexts if item["status"] == "passed"
    ]
    base_max_context = max(passed_contexts, default=0)
    explicit_64k = next(
        (item["status"] for item in contexts if item["context_tokens"] == 65_536),
        "not_tested",
    )
    capacity_rejections = sorted(
        item["context_tokens"]
        for item in contexts
        if item["status"] == "capacity_rejected"
    )
    unclassified_failures = any(
        item["status"] == "failed"
        for item in contexts
    )
    first_capacity_rejection = next(
        (context for context in capacity_rejections if context > base_max_context),
        None,
    )
    maximum_search_gap = (
        first_capacity_rejection - base_max_context
        if first_capacity_rejection is not None else None
    )
    maximum_search_complete = (
        not unclassified_failures
        and (
            base_max_context == MAX_CONTEXT
            or (
                first_capacity_rejection is not None
                and maximum_search_gap is not None
                and maximum_search_gap <= args.maximum_gap_tokens
            )
        )
    )
    release_32k = any(
        item["context_tokens"] == 32_768 and item["status"] == "passed"
        for item in contexts
    )
    explicit_64k_capacity_rejected = any(
        item["context_tokens"] == 65_536
        and item["status"] == "capacity_rejected"
        for item in contexts
    )
    exit_gate_pass = (
        release_32k
        and (explicit_64k == "passed" or explicit_64k_capacity_rejected)
        and maximum_search_complete
    )
    result = {
        "schema_version": 1,
        "milestone": "M21",
        "status": "qualified" if exit_gate_pass else "incomplete",
        "acceptance": exit_gate_pass,
        "qualification_status": "passed" if exit_gate_pass else "incomplete",
        "code": code,
        "candidate": {
            "model": model_identity,
            "toolchain_lock_sha256": sha256(toolchain_lock),
            "benchmark_binary_sha256": sha256(benchmark_executable),
            "context_driver_sha256": sha256(driver),
        },
        "artifact": {
            "model_directory": str(model),
            "config_sha256": sha256(model / "config.json"),
            "compilation_sha256": sha256(model / "gem16_compilation.json"),
            "driver_sha256": sha256(driver),
        },
        "controls": {
            "fresh_process_runs_per_context": args.runs,
            "contexts_tested": args.contexts,
            "synthetic_token_pattern": "frozen_20_token_chat_pattern",
            "raw_reports_retained": args.raw_dir is not None,
            "raw_evidence": "local_ignored_files_and_compact_hashes"
            if args.raw_dir is not None else "sha256_only",
            "raw_directory": str(args.raw_dir.resolve())
            if args.raw_dir is not None else None,
            "maximum_gap_tokens": args.maximum_gap_tokens,
        },
        "contexts": contexts,
        "release_32k": release_32k,
        "base_64k_result": explicit_64k,
        "base_max_context": base_max_context,
        "first_capacity_rejection": first_capacity_rejection,
        "maximum_search_gap_tokens": maximum_search_gap,
        "unclassified_failures": unclassified_failures,
        "maximum_search_complete": maximum_search_complete,
        "exit_gate_pass": exit_gate_pass,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"M21 qualification {result['qualification_status']}: "
        f"32K={release_32k}, 64K={explicit_64k}, "
        f"base_max_context={base_max_context}, search_complete={maximum_search_complete}"
    )
    return 0 if exit_gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

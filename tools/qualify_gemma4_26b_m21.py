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
import tempfile
from typing import Any


MIB = 1024 * 1024
MAX_CONTEXT = 262_144


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        "global_extent": payload.get("global_extent_exercised") is True,
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
            str(executable),
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
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        record: dict[str, Any] = {
            "run": run_index,
            "exit_code": completed.returncode,
            "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
        }
        if completed.returncode != 0 or not report.is_file() or not logits.is_file():
            record["status"] = "failed"
            record["error_tail"] = (completed.stderr or completed.stdout)[-1000:]
            retained.append(record)
            break
        payload = json.loads(report.read_text(encoding="utf-8"))
        errors = validate_run(payload, context)
        record.update(
            {
                "status": "passed" if not errors else "failed",
                "validation_errors": errors,
                "logits_sha256": sha256(logits),
                "prefill_elapsed_ms": payload["prefill_elapsed_ms"],
                "decode_elapsed_ms": payload["decode_elapsed_ms"],
                "prefill_prediction_token": payload["prefill_prediction_token"],
                "decode_prediction_token": payload["decode_prediction_token"],
                "prefill_chunk_count": payload["prefill_chunk_count"],
                "minimum_prefill_chunk_tokens": payload[
                    "minimum_prefill_chunk_tokens"
                ],
                "memory": payload["memory"],
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
    return {
        "context_tokens": context,
        "status": "passed" if successful and deterministic else "failed",
        "fresh_process_deterministic": deterministic,
        "runs": retained,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=parse_contexts, default=parse_contexts("32768,65536"))
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=21_600)
    args = parser.parse_args()
    driver = args.driver.resolve()
    model = args.model.resolve()
    if args.runs < 2 or args.device < 0 or args.timeout <= 0:
        parser.error("runs must be >=2 and device/timeout must be non-negative")
    if not driver.is_file() or not model.is_dir():
        parser.error("driver must be a file and model must be a directory")
    required_files = [model / "config.json", model / "gem16_compilation.json"]
    if any(not path.is_file() for path in required_files):
        parser.error("model is not a compiled Gemma 4 26B artifact")

    with tempfile.TemporaryDirectory(prefix="gem16-m21-") as directory:
        temporary = Path(directory)
        contexts = [
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

    passed_contexts = [
        item["context_tokens"] for item in contexts if item["status"] == "passed"
    ]
    base_max_context = max(passed_contexts, default=0)
    explicit_64k = next(
        (item["status"] for item in contexts if item["context_tokens"] == 65_536),
        "not_tested",
    )
    tested_failure_above_max = any(
        item["context_tokens"] > base_max_context and item["status"] == "failed"
        for item in contexts
    )
    searched_between_32k_and_64k = any(
        32_768 < item["context_tokens"] < 65_536 for item in contexts
    )
    maximum_search_complete = (
        base_max_context == MAX_CONTEXT
        or (
            tested_failure_above_max
            and (
                explicit_64k == "passed"
                or (explicit_64k == "failed" and searched_between_32k_and_64k)
            )
        )
    )
    release_32k = any(
        item["context_tokens"] == 32_768 and item["status"] == "passed"
        for item in contexts
    )
    exit_gate_pass = release_32k and explicit_64k in {"passed", "failed"} and maximum_search_complete
    result = {
        "schema_version": 1,
        "milestone": "M21",
        "qualification_status": "passed" if exit_gate_pass else "incomplete",
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
            "raw_reports_retained": False,
            "raw_evidence": "sha256_only",
        },
        "contexts": contexts,
        "release_32k": release_32k,
        "base_64k_result": explicit_64k,
        "base_max_context": base_max_context,
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

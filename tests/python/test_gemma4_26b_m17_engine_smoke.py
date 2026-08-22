#!/usr/bin/env python3
"""Optional real-checkpoint M17 lifecycle smoke test.

CTest skips this test when GEM16_26B_COMPILED_MODEL is not set. When present,
the test launches two fresh engine processes and verifies integrated prefill,
decode graph replay, determinism, finite logits, and recurring allocation
stability without retaining the large raw reports.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


SKIP = 77


def fail(message: str) -> None:
    raise RuntimeError(message)


def run_once(executable: Path, model: Path, root: Path, name: str) -> tuple[dict, str]:
    report = root / f"{name}.json"
    logits = root / f"{name}.f32le"
    command = [
        str(executable),
        "--model",
        str(model),
        "--output",
        str(report),
        "--logits",
        str(logits),
        "--backend",
        "sm120",
        "--context",
        "32768",
        "--max-new",
        "2",
    ]
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=300
    )
    if completed.returncode != 0:
        fail(
            f"{name} exited {completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    payload = json.loads(report.read_text(encoding="utf-8"))
    digest = hashlib.sha256(logits.read_bytes()).hexdigest()
    return payload, digest


def validate(payload: dict, name: str) -> None:
    if payload.get("milestone") != "M17":
        fail(f"{name}: wrong milestone")
    if payload.get("path") != "native_sm120_integrated_prefill_decode_head":
        fail(f"{name}: wrong execution path")
    if payload.get("all_logits_finite") is not True:
        fail(f"{name}: non-finite logits")
    if payload.get("deterministic") is not True:
        fail(f"{name}: intra-engine replay is not deterministic")
    if payload.get("full_logits_repeat_equal") is not True:
        fail(f"{name}: intra-engine full-logit replay mismatch")
    if payload.get("first_generated") != payload.get("second_generated"):
        fail(f"{name}: graph replay token mismatch")
    captures = payload.get("captures")
    if not isinstance(captures, list) or len(captures) != 8:
        fail(f"{name}: incomplete layer/router captures")
    memory = payload.get("memory", {})
    if memory.get("free_after_first_run_bytes") != memory.get("free_after_runs_bytes"):
        fail(f"{name}: recurring device allocation observed")
    continuation = payload.get("continuation", {})
    if continuation.get("end_position", 0) <= continuation.get("start_position", 0):
        fail(f"{name}: PrefillTokens did not advance the continuation")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: test_gemma4_26b_m17_engine_smoke.py EXECUTABLE", file=sys.stderr)
        return 2
    model_value = os.environ.get("GEM16_26B_COMPILED_MODEL")
    if not model_value:
        print("SKIP: GEM16_26B_COMPILED_MODEL is not set", file=sys.stderr)
        return SKIP
    model = Path(model_value).resolve()
    executable = Path(sys.argv[1]).resolve()
    if not model.is_dir() or not (model / "gem16_compilation.json").is_file():
        print(f"invalid GEM16_26B_COMPILED_MODEL: {model}", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="gem16-m17-engine-smoke-") as temp:
        root = Path(temp)
        first, first_hash = run_once(executable, model, root, "first")
        second, second_hash = run_once(executable, model, root, "relaunch")
    validate(first, "first")
    validate(second, "relaunch")
    if first.get("first_generated") != second.get("first_generated"):
        fail("fresh engine relaunch token mismatch")
    if first.get("continuation", {}).get("first_prediction") != second.get(
        "continuation", {}
    ).get("first_prediction"):
        fail("fresh engine relaunch continuation mismatch")
    if first_hash != second_hash:
        fail("fresh engine relaunch full-logit mismatch")
    print(
        "M17 engine smoke passed: integrated prefill, graph replay, fresh "
        f"relaunch, fixed memory, logits_sha256={first_hash}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, json.JSONDecodeError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error

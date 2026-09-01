#!/usr/bin/env python3
"""Measure the bounded V19 26B Vision capacity matrix as fresh processes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any


MIB = 1024 * 1024
MAX_CONTEXT = 262_144
DEFAULT_TARGET_CONTEXTS = (32_768, 65_536, 229_376, 229_377, 245_760)
DEFAULT_D2_CONTEXTS = (
    32_768,
    65_536,
    221_184,
    225_280,
    227_328,
    228_352,
    228_864,
    229_120,
    229_376,
    229_377,
)


def parse_contexts(value: str) -> tuple[int, ...]:
    try:
        contexts = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "contexts must be comma-separated integers"
        ) from error
    if not contexts or any(context < 2 or context > MAX_CONTEXT for context in contexts):
        raise argparse.ArgumentTypeError(f"contexts must be within 2..{MAX_CONTEXT}")
    return contexts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chat", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vision-model", type=Path, required=True)
    parser.add_argument("--assistant-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument(
        "--target-contexts",
        type=parse_contexts,
        default=DEFAULT_TARGET_CONTEXTS,
    )
    parser.add_argument(
        "--d2-contexts", type=parse_contexts, default=DEFAULT_D2_CONTEXTS
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * MIB), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_snapshot() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = [field.strip() for field in completed.stdout.splitlines()[0].split(",")]
    if len(fields) != 5:
        raise RuntimeError("unexpected nvidia-smi GPU field count")
    return {
        "name": fields[0],
        "uuid": fields[1],
        "driver_version": fields[2],
        "memory_total_bytes": int(fields[3]) * MIB,
        "memory_free_bytes": int(fields[4]) * MIB,
    }


def process_vram_bytes(pid: int) -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return 0
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == str(pid):
            return int(fields[1]) * MIB
    return 0


def measure(command: list[str], timeout: float) -> dict[str, Any]:
    begin = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak_vram = 0
    while process.poll() is None:
        if time.monotonic() - begin > timeout:
            process.kill()
            stdout, stderr = process.communicate()
            return {
                "status": "timeout",
                "exit_code": process.returncode,
                "elapsed_seconds": time.monotonic() - begin,
                "peak_process_vram_bytes": peak_vram,
                "stdout": stdout.strip(),
                "stderr": stderr.strip(),
            }
        peak_vram = max(peak_vram, process_vram_bytes(process.pid))
        time.sleep(0.05)
    stdout, stderr = process.communicate()
    record: dict[str, Any] = {
        "status": "accepted" if process.returncode == 0 else "rejected",
        "exit_code": process.returncode,
        "elapsed_seconds": time.monotonic() - begin,
        "peak_process_vram_bytes": peak_vram,
    }
    if stdout.strip():
        try:
            record["model_report"] = json.loads(stdout)
        except json.JSONDecodeError:
            record["stdout"] = stdout.strip()
    if stderr.strip():
        record["stderr"] = stderr.strip()
    return record


def write(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_identity(payload: dict[str, Any], expected: dict[str, Any]) -> None:
    for field in (
        "protocol",
        "binary",
        "components",
        "runs_per_context",
        "context_matrix",
    ):
        if payload.get(field) != expected.get(field):
            raise RuntimeError(f"resume identity changed: {field}")


def upgrade_complete_v2_identity(
    payload: dict[str, Any], expected_matrix: dict[str, list[int]]
) -> None:
    if "context_matrix" in payload:
        return
    observed = {
        scenario["name"]: sorted(
            {run["context_tokens"] for run in scenario.get("runs", [])}
        )
        for scenario in payload.get("scenarios", [])
    }
    if observed != expected_matrix or payload.get("completed") is not True:
        raise RuntimeError("legacy V2 result is incomplete or uses another matrix")
    payload["context_matrix"] = expected_matrix


def summarize(scenarios: list[dict[str, Any]], runs_per_context: int) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    accepted = True
    for scenario in scenarios:
        grouped: dict[int, list[dict[str, Any]]] = {}
        for run in scenario["runs"]:
            grouped.setdefault(run["context_tokens"], []).append(run)
        contexts: list[dict[str, Any]] = []
        for context, runs in sorted(grouped.items()):
            statuses = [run["status"] for run in runs]
            classification = (
                statuses[0]
                if len(runs) == runs_per_context and len(set(statuses)) == 1
                else "inconsistent"
            )
            contexts.append(
                {
                    "context_tokens": context,
                    "classification": classification,
                    "admission_headroom_bytes": [
                        run.get("model_report", {}).get("admission_headroom_bytes")
                        for run in runs
                    ],
                    "peak_process_vram_bytes": [
                        run["peak_process_vram_bytes"] for run in runs
                    ],
                }
            )
        repeatable_accepted = [
            item["context_tokens"]
            for item in contexts
            if item["classification"] == "accepted"
        ]
        repeatable_rejected = [
            item["context_tokens"]
            for item in contexts
            if item["classification"] == "rejected"
        ]
        required_contexts_pass = all(
            any(
                item["context_tokens"] == required
                and item["classification"] == "accepted"
                for item in contexts
            )
            for required in (32_768, 65_536)
        )
        first_rejected = min(
            (
                context
                for context in repeatable_rejected
                if not repeatable_accepted or context > max(repeatable_accepted)
            ),
            default=None,
        )
        scenario_pass = (
            required_contexts_pass
            and bool(repeatable_accepted)
            and first_rejected is not None
            and all(item["classification"] != "inconsistent" for item in contexts)
        )
        accepted = accepted and scenario_pass
        summaries.append(
            {
                "name": scenario["name"],
                "passed": scenario_pass,
                "maximum_repeatably_accepted_context": max(repeatable_accepted),
                "first_repeatably_rejected_context_above_maximum": first_rejected,
                "contexts": contexts,
            }
        )
    return {"accepted": accepted, "scenarios": summaries}


def main() -> int:
    args = parse_args()
    if args.runs < 2 or args.runs > 10:
        raise SystemExit("--runs must be within 2..10")
    if args.output.exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite existing result: {args.output}")
    paths = [args.chat, args.model, args.vision_model, args.assistant_model]
    if not args.chat.is_file() or any(not path.is_dir() for path in paths[1:]):
        raise SystemExit("chat executable and all three component directories are required")

    identity = {
        "protocol": "fresh_process_bounded_capacity_v2",
        "binary": {"path": str(args.chat.resolve()), "sha256": sha256(args.chat)},
        "components": {
            "target": str(args.model.resolve()),
            "vision": str(args.vision_model.resolve()),
            "assistant": str(args.assistant_model.resolve()),
        },
        "runs_per_context": args.runs,
        "context_matrix": {
            "target_vision": list(args.target_contexts),
            "target_vision_assistant_d2": list(args.d2_contexts),
        },
    }
    payload: dict[str, Any]
    if args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        upgrade_complete_v2_identity(payload, identity["context_matrix"])
        validate_identity(payload, identity)
    else:
        payload = {
            "schema_version": 2,
            "milestone": "V19",
            **identity,
            "gpu": gpu_snapshot(),
            "scenarios": [],
        }

    previous = {
        (scenario["name"], run["context_tokens"], run["run_index"]): run
        for scenario in payload.get("scenarios", [])
        for run in scenario.get("runs", [])
    }
    common = [
        str(args.chat.resolve()),
        "--model",
        str(args.model.resolve()),
        "--vision-model",
        str(args.vision_model.resolve()),
    ]
    definitions = (
        ("target_vision", [], args.target_contexts),
        (
            "target_vision_assistant_d2",
            [
                "--assistant-model",
                str(args.assistant_model.resolve()),
                "--mtp-draft-tokens",
                "2",
            ],
            args.d2_contexts,
        ),
    )
    scenarios: list[dict[str, Any]] = []
    for name, extra, contexts in definitions:
        runs: list[dict[str, Any]] = []
        for context in contexts:
            for run_index in range(args.runs):
                key = (name, context, run_index)
                if key in previous:
                    runs.append(previous[key])
                    continue
                print(
                    f"measuring {name} at {context}, run {run_index + 1}/{args.runs}",
                    flush=True,
                )
                initial_free = gpu_snapshot()["memory_free_bytes"]
                record = measure(
                    [
                        *common,
                        *extra,
                        "--max-context",
                        str(context),
                        "--print-model-report",
                    ],
                    args.timeout,
                )
                record["context_tokens"] = context
                record["run_index"] = run_index
                record["initial_gpu_free_bytes"] = initial_free
                runs.append(record)
                scenarios.append({"name": name, "runs": runs})
                payload["scenarios"] = scenarios
                write(payload, args.output)
                scenarios.pop()
        scenarios.append({"name": name, "runs": runs})
    payload["scenarios"] = scenarios
    payload["summary"] = summarize(scenarios, args.runs)
    payload["completed"] = True
    write(payload, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

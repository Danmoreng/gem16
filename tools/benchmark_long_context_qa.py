#!/usr/bin/env python3
"""Run one Wikipedia-based QA characterization at a requested long context."""

from __future__ import annotations

try:
    from tools.hf_cache import default_target_model
except ModuleNotFoundError:
    from hf_cache import default_target_model

import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

from qualify_long_context import (
    QualificationError,
    common_prefix,
    common_suffix,
    load_filler,
    query_gpu,
    render_tokens,
    summarize_telemetry,
    token_sha256,
)


QUESTION = (
    "\n\nBeantworte anhand des Wikipedia-Artikels knapp auf Deutsch: "
    "1. Wer prägte den Begriff Artificial Intelligence, wann und bei welcher Veranstaltung? "
    "2. Welchen Test schlug Alan Turing vor? "
    "3. Nenne zwei wichtige Anwendungsfelder sowie zwei Chancen und zwei Risiken von KI."
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=default_target_model())
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--chat-executable", required=True, type=Path)
    parser.add_argument("--filler-workload", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--prompt-tokens", required=True, type=positive_int)
    parser.add_argument("--generated-tokens", type=positive_int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        model = args.model.resolve(strict=True)
        executable = args.executable.resolve(strict=True)
        chat_executable = args.chat_executable.resolve(strict=True)
        filler_workload = args.filler_workload.resolve(strict=True)

        sentinel_a = render_tokens(chat_executable, model, "AAAA")
        sentinel_b = render_tokens(chat_executable, model, "zzzz")
        prefix = common_prefix(sentinel_a, sentinel_b)
        suffix = common_suffix(sentinel_a, sentinel_b, len(prefix))
        rendered_question = render_tokens(chat_executable, model, QUESTION)
        if (
            rendered_question[:len(prefix)] != prefix
            or rendered_question[-len(suffix):] != suffix
        ):
            raise QualificationError("chat-template envelope changed")
        question = rendered_question[len(prefix):-len(suffix)]
        filler = load_filler(filler_workload)
        filler_tokens = args.prompt_tokens - len(prefix) - len(question) - len(suffix)
        if filler_tokens < 1:
            raise QualificationError("prompt is too short for the question")
        prompt = (
            prefix
            + [filler[index % len(filler)] for index in range(filler_tokens)]
            + question
            + suffix
        )
        if len(prompt) != args.prompt_tokens:
            raise QualificationError("constructed prompt has the wrong length")
        max_context = args.prompt_tokens + args.generated_tokens
        if max_context > 262144:
            raise QualificationError("prompt plus generation exceeds 262,144 positions")

        args.prompt_file.parent.mkdir(parents=True, exist_ok=True)
        args.prompt_file.write_text(
            ",".join(str(token) for token in prompt), encoding="ascii"
        )
        command = [
            str(executable),
            "--model", str(model),
            "--input-token-ids-file", str(args.prompt_file),
            "--kv-cache", "fp8",
            "--max-tokens", str(args.generated_tokens),
            "--max-context", str(max_context),
            "--greedy",
        ]
        telemetry_samples: list[dict[str, float]] = []
        stop_monitor = threading.Event()

        def monitor() -> None:
            while not stop_monitor.is_set():
                sample = query_gpu()
                if sample is not None:
                    telemetry_samples.append(sample)
                stop_monitor.wait(0.25)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        started = time.monotonic()
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        wall_seconds = time.monotonic() - started
        stop_monitor.set()
        thread.join(timeout=2.0)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise QualificationError(
                f"gem16 exited with {completed.returncode}: {detail[-4000:]}"
            )
        engine = json.loads(completed.stdout)
        output_tokens = engine.get("output_token_ids")
        if not isinstance(output_tokens, list) or not all(
            isinstance(token, int) for token in output_tokens
        ):
            raise QualificationError("gem16 returned malformed output token IDs")
        prompt_ms = float(engine["prompt_ms"])
        decode_ms = float(engine["decode_ms"])
        checks = {
            "exact_prompt_length": len(prompt) == args.prompt_tokens,
            "full_generation_length": len(output_tokens) == args.generated_tokens,
            "no_fallbacks": engine.get("fallbacks") == 0,
            "no_token_loop_allocations": engine.get("token_loop_allocations") is False,
            "hybrid_cache_layout": (
                engine.get("kv_cache_layout")
                == "hybrid_local_ring_global_contiguous"
            ),
        }
        document: dict[str, Any] = {
            "schema_version": 1,
            "status": "passed" if all(checks.values()) else "failed",
            "scope": "single_run_wikipedia_long_context_qa_characterization",
            "configuration": {
                "prompt_tokens": args.prompt_tokens,
                "generated_tokens": args.generated_tokens,
                "maximum_context_positions": max_context,
                "kv_cache": "checkpoint_fp8",
                "greedy": True,
                "timing_repetitions": 1,
                "question": QUESTION.strip(),
            },
            "prompt": {
                "sha256_u32le": token_sha256(prompt),
                "file": str(args.prompt_file),
                "filler_workload": str(filler_workload),
                "source_article_repetitions": filler_tokens / len(filler),
                "question_token_offset": len(prefix) + filler_tokens,
            },
            "checks": checks,
            "metrics": {
                "wall_seconds": wall_seconds,
                "prompt_ms": prompt_ms,
                "prompt_tokens_per_second": args.prompt_tokens * 1000.0 / prompt_ms,
                "decode_ms": decode_ms,
                "decode_tokens_per_second": (
                    (len(output_tokens) - 1) * 1000.0 / decode_ms
                    if len(output_tokens) > 1 and decode_ms > 0.0
                    else 0.0
                ),
            },
            "telemetry": summarize_telemetry(telemetry_samples),
            "engine_result": engine,
            "command": command,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": document["status"],
            **document["metrics"],
            "peak_memory_used_mib": document["telemetry"]
                .get("memory_used_mib", {}).get("maximum"),
            "workspace_bytes": engine.get("workspace_bytes"),
            "kv_cache_bytes": engine.get("kv_cache_bytes"),
            "output": str(args.output),
        }, indent=2))
        return 0 if document["status"] == "passed" else 1
    except (OSError, ValueError, json.JSONDecodeError, QualificationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

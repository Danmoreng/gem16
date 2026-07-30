#!/usr/bin/env python3
"""Run one mixed 64K retrieval, soak, memory, and runtime qualification."""

from __future__ import annotations

try:
    from tools.hf_cache import default_target_model
except ModuleNotFoundError:
    from hf_cache import default_target_model

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import threading
import time
from typing import Any


class QualificationError(RuntimeError):
    pass


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
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--prompt-tokens", type=positive_int, default=65536)
    parser.add_argument("--generated-tokens", type=positive_int, default=1024)
    return parser.parse_args()


def token_sha256(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        if token < 0 or token > 0xFFFFFFFF:
            raise QualificationError("token ID is outside uint32")
        digest.update(struct.pack("<I", token))
    return digest.hexdigest()


def common_prefix(left: list[int], right: list[int]) -> list[int]:
    length = 0
    while length < min(len(left), len(right)) and left[length] == right[length]:
        length += 1
    return left[:length]


def common_suffix(left: list[int], right: list[int], prefix_length: int) -> list[int]:
    length = 0
    limit = min(len(left), len(right)) - prefix_length
    while length < limit and left[-1 - length] == right[-1 - length]:
        length += 1
    return [] if length == 0 else left[-length:]


def render_tokens(chat_executable: Path, model: Path, message: str) -> list[int]:
    completed = subprocess.run(
        [
            str(chat_executable), "--model", str(model), "--message", message,
            "--render-only", "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise QualificationError(f"chat rendering failed: {detail[-2000:]}")
    document = json.loads(completed.stdout)
    tokens = document.get("prompt_token_ids")
    if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
        raise QualificationError("chat renderer returned malformed token IDs")
    return tokens


def load_filler(path: Path) -> list[int]:
    document = json.loads(path.read_text(encoding="utf-8"))
    prompt = document.get("prompt")
    if not isinstance(prompt, dict):
        raise QualificationError("filler workload has no prompt object")
    tokens = prompt.get("token_ids")
    suffix = prompt.get("preserved_chat_suffix_tokens")
    if (
        not isinstance(tokens, list)
        or not all(isinstance(token, int) and token >= 0 for token in tokens)
        or not isinstance(suffix, int)
        or suffix < 1
        or len(tokens) <= suffix + 4
    ):
        raise QualificationError("filler workload token metadata is malformed")
    return tokens[4:-suffix]


def sequence_index(haystack: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    width = len(needle)
    return next(
        (
            index
            for index in range(len(haystack) - width + 1)
            if haystack[index:index + width] == needle
        ),
        None,
    )


def query_gpu() -> dict[str, float] | None:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,power.draw,clocks.sm,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    fields = [field.strip() for field in completed.stdout.splitlines()[0].split(",")]
    if len(fields) != 4:
        return None
    try:
        values = [float(field) for field in fields]
    except ValueError:
        return None
    return dict(zip(("memory_used_mib", "power_w", "sm_clock_mhz", "temperature_c"), values))


def summarize_telemetry(samples: list[dict[str, float]]) -> dict[str, Any]:
    if not samples:
        return {"available": False}
    result: dict[str, Any] = {"available": True, "sample_count": len(samples)}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        result[key] = {
            "minimum": min(values),
            "maximum": max(values),
            "mean": statistics.fmean(values),
        }
    return result


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
        if len(prefix) < 4 or len(suffix) < 4:
            raise QualificationError("could not derive a stable chat-template envelope")

        def content_tokens(text: str) -> list[int]:
            rendered = render_tokens(chat_executable, model, text)
            if rendered[:len(prefix)] != prefix or rendered[-len(suffix):] != suffix:
                raise QualificationError("chat-template envelope changed while tokenizing fixture")
            return rendered[len(prefix):-len(suffix)]

        needles = [
            ("alpha", "ZEBRA-4821", 0.10),
            ("beta", "COBALT-7395", 0.50),
            ("gamma", "ORCHID-1604", 0.90),
        ]
        statements = {
            name: content_tokens(f"\n\nRetrieval marker {name}: the exact access code is {code}.\n\n")
            for name, code, _ in needles
        }
        question = content_tokens(
            "\n\nList the alpha, beta, and gamma access codes in that order. "
            "Return the three codes exactly."
        )
        answer_variants = {
            name: [content_tokens(code), content_tokens(" " + code)]
            for name, code, _ in needles
        }
        filler = load_filler(filler_workload)
        payload_length = args.prompt_tokens - len(prefix) - len(suffix) - len(question)
        if payload_length < 1 or not filler:
            raise QualificationError("target prompt is too short for the mixed fixture")
        payload = [filler[index % len(filler)] for index in range(payload_length)]
        placements: list[dict[str, Any]] = []
        for name, code, fraction in needles:
            statement = statements[name]
            offset = min(payload_length - len(statement), int(payload_length * fraction))
            payload[offset:offset + len(statement)] = statement
            placements.append({
                "name": name,
                "code": code,
                "target_fraction": fraction,
                "prompt_token_offset": len(prefix) + offset,
                "statement_tokens": len(statement),
            })
        prompt = prefix + payload + question + suffix
        if len(prompt) != args.prompt_tokens:
            raise QualificationError("constructed prompt has the wrong length")

        prompt_path = args.prompt_file or args.output.with_suffix(".tokens")
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(",".join(str(token) for token in prompt), encoding="ascii")
        command = [
            str(executable), "--model", str(model),
            "--input-token-ids-file", str(prompt_path),
            "--kv-cache", "fp8",
            "--max-tokens", str(args.generated_tokens),
            "--max-context", str(args.prompt_tokens + args.generated_tokens),
            "--greedy",
        ]
        samples: list[dict[str, float]] = []
        stop_monitor = threading.Event()

        def monitor() -> None:
            while not stop_monitor.is_set():
                sample = query_gpu()
                if sample is not None:
                    samples.append(sample)
                stop_monitor.wait(0.25)

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        started = time.monotonic()
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        wall_seconds = time.monotonic() - started
        stop_monitor.set()
        monitor_thread.join(timeout=2.0)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise QualificationError(f"gem16 exited with {completed.returncode}: {detail[-4000:]}")
        engine = json.loads(completed.stdout)
        output_tokens = engine.get("output_token_ids")
        if not isinstance(output_tokens, list) or not all(isinstance(token, int) for token in output_tokens):
            raise QualificationError("gem16 returned malformed output token IDs")

        retrieval = {}
        for name, code, _ in needles:
            offsets = [
                offset
                for variant in answer_variants[name]
                if (offset := sequence_index(output_tokens, variant)) is not None
            ]
            retrieval[name] = {
                "code": code,
                "found_in_output_tokens": bool(offsets),
                "first_output_token_offset": min(offsets) if offsets else None,
            }
        retrieval_offsets = [
            retrieval[name]["first_output_token_offset"] for name, _, _ in needles
        ]
        checks = {
            "exact_prompt_length": len(prompt) == args.prompt_tokens,
            "full_generation_length": len(output_tokens) == args.generated_tokens,
            "no_fallbacks": engine.get("fallbacks") == 0,
            "no_token_loop_allocations": engine.get("token_loop_allocations") is False,
            "hybrid_cache_layout": engine.get("kv_cache_layout") == "hybrid_local_ring_global_contiguous",
            "all_needles_retrieved": all(item["found_in_output_tokens"] for item in retrieval.values()),
            "needles_returned_in_requested_order": (
                all(offset is not None for offset in retrieval_offsets)
                and retrieval_offsets == sorted(retrieval_offsets)
            ),
        }
        passed = all(checks.values())
        document = {
            "schema_version": 1,
            "status": "passed" if passed else "failed",
            "scope": "single_run_mixed_64k_qualification",
            "configuration": {
                "prompt_tokens": args.prompt_tokens,
                "generated_tokens": args.generated_tokens,
                "kv_cache": "checkpoint_fp8",
                "greedy": True,
                "fresh_process_runs": 1,
                "timing_repetitions": 1,
            },
            "prompt": {
                "sha256_u32le": token_sha256(prompt),
                "file": str(prompt_path),
                "filler_workload": str(filler_workload),
                "placements": placements,
            },
            "retrieval": retrieval,
            "checks": checks,
            "wall_seconds": wall_seconds,
            "telemetry": summarize_telemetry(samples),
            "engine_result": engine,
            "command": command,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": document["status"],
            "prompt_ms": engine.get("prompt_ms"),
            "decode_ms": engine.get("decode_ms"),
            "generated_tokens": len(output_tokens),
            "retrieval": retrieval,
            "peak_memory_used_mib": document["telemetry"].get("memory_used_mib", {}).get("maximum"),
            "output": str(args.output),
        }, indent=2))
        return 0 if passed else 1
    except (OSError, ValueError, json.JSONDecodeError, QualificationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

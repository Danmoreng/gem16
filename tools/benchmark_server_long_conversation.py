#!/usr/bin/env python3
"""Run a realistic one-slot multimodal 256K sampled-MTP server conversation."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import pathlib
import subprocess
import time
import urllib.error
from typing import Any

from benchmark_server import (
    BenchmarkError,
    TelemetrySampler,
    distribution,
    prometheus_metrics,
    request_json,
    stream_run,
)


DEFAULT_TARGETS = "2048,8192,32768,65536,131072"
FILLER_PARAGRAPH = (
    "Artificial intelligence research studies perception, reasoning, learning, "
    "planning, language, and reliable interaction with people. Practical systems "
    "must balance capability, efficiency, transparency, safety, and careful "
    "evaluation across realistic tasks. Historical progress has alternated between "
    "symbolic methods, statistical learning, neural networks, and hybrid systems. "
)
PROBE_PREFIX = (
    "This is a new short note used to measure incremental prompt processing. "
    "Long conversations require stable retrieval, bounded memory, exact cache "
    "continuation, and responsive generation when old and new modalities are mixed. "
)
PROBE_QUESTIONS = (
    "In one sentence, which number appeared on the sign in the initial image?",
    "Name one cultural subject discussed in the initial audio recording.",
    "Briefly confirm both the sign number and one audio theme.",
    "What detail from the original image remains relevant to this conversation?",
    "What did the original recording say about creation stories or the afterlife?",
    "Answer with the remembered number followed by one remembered audio topic.",
    "Why do the initial visual and spoken details show successful long-context retrieval?",
    "Give a concise factual recap of the two original media inputs.",
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_targets(value: str) -> list[int]:
    try:
        targets = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("targets must be comma-separated integers") from error
    if not targets or any(target < 1 for target in targets):
        raise argparse.ArgumentTypeError("targets must be positive")
    if targets != sorted(set(targets)):
        raise argparse.ArgumentTypeError("targets must be unique and ascending")
    if targets[-1] >= 262144:
        raise argparse.ArgumentTypeError("targets must stay below 262144")
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-executable", required=True, type=pathlib.Path)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--assistant-model", required=True, type=pathlib.Path)
    parser.add_argument("--image", required=True, type=pathlib.Path)
    parser.add_argument("--audio", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--model-name", default="gem16")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=positive_int, default=8080)
    parser.add_argument("--targets", type=parse_targets, default=parse_targets(DEFAULT_TARGETS))
    parser.add_argument("--warmup-turns", type=int, default=1)
    parser.add_argument("--repetitions", type=positive_int, default=3)
    parser.add_argument("--max-output-tokens", type=positive_int, default=128)
    parser.add_argument(
        "--thinking-effort",
        choices=("none", "low", "medium", "high"),
        default="low",
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--telemetry-interval", type=float, default=0.25)
    parser.add_argument("--expected-text", action="append", default=[])
    args = parser.parse_args()
    if args.port > 65535:
        parser.error("port must be in [1, 65535]")
    if args.warmup_turns < 0:
        parser.error("warmup turns must be nonnegative")
    if args.telemetry_interval <= 0.0 or args.timeout <= 0.0:
        parser.error("timeout and telemetry interval must be positive")
    return args


def data_url(path: pathlib.Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    mime = mimetypes.guess_type(path.name)[0]
    if mime not in {"image/png", "image/jpeg", "image/bmp"}:
        raise BenchmarkError(f"unsupported benchmark image format: {path}")
    return f"data:{mime};base64,{encoded}"


def audio_payload(path: pathlib.Path) -> dict[str, str]:
    suffix = path.suffix.lower().removeprefix(".")
    if suffix not in {"wav", "mp3", "flac"}:
        raise BenchmarkError(f"unsupported benchmark audio format: {path}")
    return {
        "format": suffix,
        "data": base64.b64encode(path.read_bytes()).decode("ascii"),
    }


def metric_delta(before: dict[str, float], after: dict[str, float], name: str) -> float:
    return after.get(name, 0.0) - before.get(name, 0.0)


def enrich_engine_metrics(
    run: dict[str, Any], before: dict[str, float], after: dict[str, float]
) -> dict[str, Any]:
    prompt_us = metric_delta(before, after, "gem16_prompt_microseconds_total")
    decode_us = metric_delta(before, after, "gem16_decode_microseconds_total")
    decode_tokens = metric_delta(before, after, "gem16_decode_measured_tokens_total")
    proposed = metric_delta(before, after, "gem16_mtp_proposed_tokens_total")
    accepted = metric_delta(before, after, "gem16_mtp_accepted_tokens_total")
    run["engine_prompt_milliseconds"] = prompt_us / 1000.0
    run["engine_decode_milliseconds"] = decode_us / 1000.0
    run["engine_decode_measured_tokens"] = int(decode_tokens)
    run["engine_decode_tokens_per_second"] = (
        decode_tokens * 1_000_000.0 / decode_us if decode_us > 0.0 else 0.0
    )
    run["incremental_prefill_tokens_per_second"] = (
        float(run["cache_write_tokens"]) * 1_000_000.0 / prompt_us
        if prompt_us > 0.0
        else 0.0
    )
    run["http_overhead_milliseconds"] = max(
        0.0, run["elapsed_seconds"] * 1000.0 - prompt_us / 1000.0 - decode_us / 1000.0
    )
    run["mtp"] = {
        "proposed_tokens": int(proposed),
        "accepted_tokens": int(accepted),
        "rejected_tokens": int(
            metric_delta(before, after, "gem16_mtp_rejected_tokens_total")
        ),
        "verification_groups": int(
            metric_delta(before, after, "gem16_mtp_verification_groups_total")
        ),
        "d1_groups": int(metric_delta(before, after, "gem16_mtp_d1_groups_total")),
        "d2_groups": int(metric_delta(before, after, "gem16_mtp_d2_groups_total")),
        "d4_groups": int(metric_delta(before, after, "gem16_mtp_d4_groups_total")),
        "ordinary_fallback_tokens": int(
            metric_delta(before, after, "gem16_mtp_ordinary_fallback_tokens_total")
        ),
        "acceptance_ratio": accepted / proposed if proposed > 0.0 else 0.0,
    }
    return run


def measured_stream(
    server_url: str,
    responses_url: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    before = prometheus_metrics(f"{server_url}/metrics", timeout)
    run = stream_run(responses_url, payload, timeout)
    after = prometheus_metrics(f"{server_url}/metrics", timeout)
    return enrich_engine_metrics(run, before, after)


def summarize_checkpoint(runs: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "input_tokens",
        "cached_tokens",
        "cache_write_tokens",
        "engine_prompt_milliseconds",
        "incremental_prefill_tokens_per_second",
        "time_to_first_delta_seconds",
        "engine_decode_milliseconds",
        "engine_decode_tokens_per_second",
        "elapsed_seconds",
        "http_overhead_milliseconds",
    )
    summary = {
        field: distribution([float(run[field]) for run in runs])
        for field in fields
    }
    intervals = [
        float(interval)
        for run in runs
        for interval in run["delta_interval_seconds"]
    ]
    if intervals:
        summary["stream_delta_interval_seconds"] = distribution(intervals)
    proposed = sum(run["mtp"]["proposed_tokens"] for run in runs)
    accepted = sum(run["mtp"]["accepted_tokens"] for run in runs)
    summary["mtp"] = {
        "proposed_tokens": proposed,
        "accepted_tokens": accepted,
        "rejected_tokens": sum(run["mtp"]["rejected_tokens"] for run in runs),
        "verification_groups": sum(run["mtp"]["verification_groups"] for run in runs),
        "d1_groups": sum(run["mtp"]["d1_groups"] for run in runs),
        "d2_groups": sum(run["mtp"]["d2_groups"] for run in runs),
        "d4_groups": sum(run["mtp"]["d4_groups"] for run in runs),
        "ordinary_fallback_tokens": sum(
            run["mtp"]["ordinary_fallback_tokens"] for run in runs
        ),
        "acceptance_ratio": accepted / proposed if proposed else 0.0,
    }
    return summary


def filler_text(characters: int, sequence: int) -> str:
    prefix = (
        f"Context archive segment {sequence}. Store these notes for later and "
        "reply with one acknowledgement token.\n"
    )
    repeats = max(1, math_ceil_div(max(1, characters - len(prefix)), len(FILLER_PARAGRAPH)))
    return prefix + (FILLER_PARAGRAPH * repeats)[: max(1, characters - len(prefix))]


def math_ceil_div(left: int, right: int) -> int:
    return (left + right - 1) // right


def fill_to_context(
    server_url: str,
    responses_url: str,
    model: str,
    previous_response_id: str,
    current_tokens: int,
    desired_tokens: int,
    timeout: float,
    sequence_start: int,
) -> tuple[str, int, list[dict[str, Any]], int]:
    fills: list[dict[str, Any]] = []
    sequence = sequence_start
    chars_per_token = 3.0
    while desired_tokens - current_tokens > 128:
        remaining = desired_tokens - current_tokens
        requested_tokens = max(1, remaining - 96)
        characters = max(64, int(requested_tokens * chars_per_token * 0.92))
        text = filler_text(characters, sequence)
        payload = {
            "model": model,
            "previous_response_id": previous_response_id,
            "input": text,
            "max_output_tokens": 1,
            "reasoning": {"effort": "none"},
        }
        metrics_before = prometheus_metrics(
            f"{server_url}/metrics", timeout
        )
        document, elapsed = request_json(responses_url, payload, timeout)
        metrics_after = prometheus_metrics(
            f"{server_url}/metrics", timeout
        )
        usage = document["usage"]
        next_tokens = int(usage["input_tokens"])
        delta = next_tokens - current_tokens
        if delta <= 0:
            raise BenchmarkError("context filler did not extend the resident prefix")
        chars_per_token = max(0.5, min(8.0, len(text) / delta))
        fill_run = {
            "sequence": sequence,
            "requested_characters": len(text),
            "input_tokens": next_tokens,
            "added_input_tokens": delta,
            "output_tokens": int(usage["output_tokens"]),
            "cached_tokens": int(
                usage["input_tokens_details"]["cached_tokens"]
            ),
            "cache_write_tokens": int(
                usage["input_tokens_details"]["cache_write_tokens"]
            ),
            "elapsed_seconds": elapsed,
            "response_id": document["id"],
        }
        enrich_engine_metrics(fill_run, metrics_before, metrics_after)
        fills.append(fill_run)
        previous_response_id = document["id"]
        current_tokens = next_tokens + int(usage["output_tokens"])
        sequence += 1
        if len(fills) > 16:
            raise BenchmarkError("context filler failed to converge")
    return previous_response_id, current_tokens, fills, sequence


def wait_for_server(base_url: str, process: subprocess.Popen[str], timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise BenchmarkError(f"server exited during startup with code {process.returncode}")
        try:
            health, _ = request_json(f"{base_url}/health", timeout=5.0)
            return health
        except (OSError, urllib.error.URLError, BenchmarkError) as error:
            last_error = error
            time.sleep(0.5)
    raise BenchmarkError(f"server did not become healthy: {last_error}")


def validate_health(health: dict[str, Any]) -> None:
    sampling = health.get("sampling") or {}
    expected = {
        "session_limit": 1,
        "max_context_tokens": 262144,
        "mtp_draft_tokens": 2,
        "mtp_adaptive": False,
    }
    for field, value in expected.items():
        if health.get(field) != value:
            raise BenchmarkError(f"server health {field}={health.get(field)!r}, expected {value!r}")
    sampling_expected = {
        "enabled": True,
        "temperature": 1.0,
        "top_k": 64,
        "top_p": 0.95,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "seed": 0,
    }
    for field, value in sampling_expected.items():
        observed = sampling.get(field)
        if isinstance(value, float):
            if observed is None or abs(float(observed) - value) > 1.0e-5:
                raise BenchmarkError(f"sampling {field}={observed!r}, expected {value!r}")
        elif observed != value:
            raise BenchmarkError(f"sampling {field}={observed!r}, expected {value!r}")


def main() -> int:
    args = parse_args()
    for path in (
        args.server_executable,
        args.model,
        args.assistant_model,
        args.image,
        args.audio,
    ):
        if not path.exists():
            raise BenchmarkError(f"required path does not exist: {path}")
    if args.output.exists():
        raise BenchmarkError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    server_log = args.output.with_suffix(".server.log")
    if server_log.exists():
        raise BenchmarkError(f"refusing to overwrite {server_log}")

    command = [
        str(args.server_executable.resolve()),
        "--model", str(args.model.resolve()),
        "--assistant-model", str(args.assistant_model.resolve()),
        "--mtp-draft-tokens", "2",
        "--max-context", "262144",
        "--max-sessions", "1",
        "--model-name", args.model_name,
        "--host", args.host,
        "--port", str(args.port),
        "--kv-cache", "fp8",
    ]
    base_url = f"http://{args.host}:{args.port}"
    responses_url = f"{base_url}/v1/responses"
    telemetry = TelemetrySampler(args.telemetry_interval)
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    with server_log.open("x", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command, stdout=log_file, stderr=subprocess.STDOUT, text=True
        )
        try:
            health = wait_for_server(base_url, process, min(args.timeout, 300.0))
            validate_health(health)
            metrics_before = prometheus_metrics(f"{base_url}/metrics", args.timeout)
            telemetry.start()

            root_payload = {
                "model": args.model_name,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "We are beginning a long conversation. Inspect the image and "
                                    "audio, remember their important details, and briefly identify "
                                    "the sign number plus the main themes in the recording."
                                ),
                            },
                            {"type": "input_image", "image_url": data_url(args.image)},
                            {"type": "input_audio", "input_audio": audio_payload(args.audio)},
                        ],
                    }
                ],
                "max_output_tokens": args.max_output_tokens,
                "reasoning": {"effort": args.thinking_effort},
            }
            root = measured_stream(base_url, responses_url, root_payload, args.timeout)
            previous_response_id = str(root["response_id"])
            current_tokens = int(root["input_tokens"]) + int(root["output_tokens"])
            sequence = 0
            checkpoints: list[dict[str, Any]] = []
            all_fills: list[dict[str, Any]] = []

            for target_index, target in enumerate(args.targets):
                if target <= current_tokens:
                    raise BenchmarkError(
                        f"target {target} is not above current context {current_tokens}"
                    )
                context_before_fill = current_tokens
                reserve = (args.warmup_turns + args.repetitions) * 120
                desired_before_probe = max(current_tokens, target - reserve)
                previous_response_id, current_tokens, fills, sequence = fill_to_context(
                    base_url,
                    responses_url,
                    args.model_name,
                    previous_response_id,
                    current_tokens,
                    desired_before_probe,
                    args.timeout,
                    sequence,
                )
                all_fills.extend(fills)

                warmups: list[dict[str, Any]] = []
                measured: list[dict[str, Any]] = []
                turn_count = args.warmup_turns + args.repetitions
                for turn in range(turn_count):
                    question_index = (
                        target_index * turn_count + turn
                    ) % len(PROBE_QUESTIONS)
                    payload = {
                        "model": args.model_name,
                        "previous_response_id": previous_response_id,
                        "input": (
                            f"Checkpoint {target}, turn {turn + 1}. "
                            f"{PROBE_PREFIX}{PROBE_QUESTIONS[question_index]}"
                        ),
                        "max_output_tokens": args.max_output_tokens,
                        "reasoning": {"effort": args.thinking_effort},
                    }
                    run = measured_stream(base_url, responses_url, payload, args.timeout)
                    run["target_context_tokens"] = target
                    run["turn"] = turn
                    previous_response_id = str(run["response_id"])
                    current_tokens = int(run["input_tokens"]) + int(run["output_tokens"])
                    if turn < args.warmup_turns:
                        warmups.append(run)
                    else:
                        measured.append(run)
                if not any(run["mtp"]["d2_groups"] > 0 for run in measured):
                    raise BenchmarkError(f"MTP D2 did not execute at target {target}")
                checkpoints.append(
                    {
                        "target_context_tokens": target,
                        "context_before_fill_tokens": context_before_fill,
                        "fill_requests": fills,
                        "warmup_runs": warmups,
                        "runs": measured,
                        "summary": summarize_checkpoint(measured),
                    }
                )
                summary = checkpoints[-1]["summary"]
                print(
                    f"context {target}: actual={summary['input_tokens']['median']:.0f} "
                    f"prefill={summary['engine_prompt_milliseconds']['median']:.2f} ms "
                    f"TTFT={summary['time_to_first_delta_seconds']['median'] * 1000.0:.2f} ms "
                    f"decode={summary['engine_decode_tokens_per_second']['median']:.2f} tok/s",
                    flush=True,
                )

            final_retrieval = measured_stream(
                base_url,
                responses_url,
                {
                    "model": args.model_name,
                    "previous_response_id": previous_response_id,
                    "input": (
                        "Final retrieval check: state the number on the original "
                        "image sign and explicitly say whether the original audio "
                        "discussed the afterlife. Answer in one sentence."
                    ),
                    "max_output_tokens": args.max_output_tokens,
                    "reasoning": {"effort": args.thinking_effort},
                },
                args.timeout,
            )
            if final_retrieval["mtp"]["d2_groups"] == 0:
                raise BenchmarkError("MTP D2 did not execute for final retrieval")
            metrics_after = prometheus_metrics(f"{base_url}/metrics", args.timeout)
            gpu_telemetry = telemetry.stop()
            expected_checks = {
                expected: {
                    "root_multimodal": expected.lower()
                    in root["output_text"].lower(),
                    "final_long_context": expected.lower()
                    in final_retrieval["output_text"].lower(),
                }
                for expected in args.expected_text
            }
            try:
                git_sha = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                git_sha = "unknown"
            document = {
                "schema_version": 1,
                "git_sha": git_sha,
                "scope": "one_slot_multimodal_256k_sampled_mtp_d2_conversation",
                "started_utc": started_utc,
                "command": command,
                "configuration": {
                    "maximum_context_tokens": 262144,
                    "session_slots": 1,
                    "kv_cache": "checkpoint_fp8",
                    "sampling": health["sampling"],
                    "mtp_draft_tokens": 2,
                    "mtp_adaptive": False,
                    "targets": args.targets,
                    "warmup_turns": args.warmup_turns,
                    "repetitions": args.repetitions,
                    "max_output_tokens": args.max_output_tokens,
                    "thinking_effort": args.thinking_effort,
                },
                "media": {
                    "image": str(args.image.resolve()),
                    "image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
                    "audio": str(args.audio.resolve()),
                    "audio_sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
                },
                "health": health,
                "root_multimodal_run": root,
                "final_retrieval_run": final_retrieval,
                "context_fill_requests": all_fills,
                "checkpoints": checkpoints,
                "expected_text_checks": expected_checks,
                "gpu_telemetry": gpu_telemetry,
                "metrics_before": metrics_before,
                "metrics_after": metrics_after,
                "metric_counter_deltas": {
                    name: value - metrics_before.get(name, 0.0)
                    for name, value in metrics_after.items()
                    if name.endswith("_total")
                },
                "server_log": str(server_log),
            }
            with args.output.open("x", encoding="utf-8") as output_file:
                json.dump(document, output_file, indent=2, sort_keys=True)
                output_file.write("\n")
            if expected_checks and not all(
                all(stages.values()) for stages in expected_checks.values()
            ):
                raise BenchmarkError(f"expected retrieval checks failed: {expected_checks}")
            print(f"wrote {args.output}", flush=True)
            return 0
        finally:
            telemetry.stop()
            process.terminate()
            try:
                process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BenchmarkError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", flush=True)
        raise SystemExit(2) from error

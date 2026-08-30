#!/usr/bin/env python3
"""Run frozen WP8 sampled-generation or long-context text checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable
import urllib.error
import urllib.request


class BenchmarkError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise BenchmarkError(f"{path} is not a JSON object")
    return document


def file_identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def request_json(
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 3600.0,
) -> tuple[dict[str, Any], float]:
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer EMPTY",
            "Content-Type": "application/json",
        },
        method="GET" if payload is None else "POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            document = json.loads(response.read())
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"request to {url} failed: {error}") from error
    if not isinstance(document, dict):
        raise BenchmarkError(f"response from {url} is not an object")
    return document, time.perf_counter() - started


def api_roots(base_url: str) -> tuple[str, str]:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized, normalized[:-3]
    return normalized + "/v1", normalized


def endpoint_model(v1: str) -> str:
    models, _ = request_json(v1 + "/models", timeout=30.0)
    entries = models.get("data")
    if not isinstance(entries, list) or not entries:
        raise BenchmarkError("endpoint has no advertised model")
    model = entries[0].get("id")
    if not isinstance(model, str) or not model:
        raise BenchmarkError("endpoint model ID is invalid")
    return model


def completion(
    v1: str, model: str, message: str, max_tokens: int, timeout: float
) -> tuple[dict[str, Any], float]:
    response, elapsed = request_json(
        v1 + "/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": max_tokens,
            "stream": False,
            "reasoning_effort": "none",
        },
        timeout,
    )
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise BenchmarkError("completion response must contain one choice")
    content = (choices[0].get("message") or {}).get("content")
    if not isinstance(content, str):
        raise BenchmarkError("completion content is not text")
    # JSON decoding already proves that the response is valid UTF-8.
    return {
        "content": content,
        "finish_reason": choices[0].get("finish_reason"),
        "usage": response.get("usage"),
    }, elapsed


def validate_health(
    health: dict[str, Any], label: str, suite: dict[str, Any]
) -> None:
    expected = suite[label]
    if health.get("status") != "ok":
        raise BenchmarkError("endpoint health is not ok")
    if health.get("artifact_content_sha256") != expected["checkpoint_content_sha256"]:
        raise BenchmarkError(
            "endpoint artifact identity does not match the frozen suite"
        )


def sampled_run(args: argparse.Namespace, suite: dict[str, Any]) -> dict[str, Any]:
    v1, root = api_roots(args.base_url)
    health, _ = request_json(root + "/health", timeout=30.0)
    validate_health(health, args.model_label, suite)
    contract = suite["sampled_generation"]
    sampling = health.get("sampling")
    expected = contract["sampling"]
    if not isinstance(sampling, dict) or not sampling.get("enabled"):
        raise BenchmarkError("sampled check requires endpoint sampling")
    for key in ("temperature", "top_p"):
        if not math.isclose(float(sampling.get(key, math.nan)), expected[key], abs_tol=1e-6):
            raise BenchmarkError(f"endpoint sampling {key} differs from suite")
    if sampling.get("top_k") != expected["top_k"] or sampling.get("seed") != args.seed:
        raise BenchmarkError("endpoint top-k or seed differs from suite invocation")
    if args.seed not in expected["seeds"]:
        raise BenchmarkError("requested seed is not frozen in the suite")
    model = endpoint_model(v1)
    cases: list[dict[str, Any]] = []
    for index, prompt in enumerate(contract["prompts"]):
        first, first_seconds = completion(v1, model, prompt, 256, args.timeout)
        second, second_seconds = completion(v1, model, prompt, 256, args.timeout)
        cases.append(
            {
                "index": index,
                "prompt": prompt,
                "first": first,
                "second": second,
                "first_seconds": first_seconds,
                "second_seconds": second_seconds,
                "nonempty": bool(first["content"] and second["content"]),
                "same_seed_same_runtime_identity": (
                    first["content"] == second["content"]
                    and first["finish_reason"] == second["finish_reason"]
                ),
            }
        )
    passed = all(
        case["nonempty"] and case["same_seed_same_runtime_identity"]
        for case in cases
    )
    return {
        "schema_version": 1,
        "work_packet": "WP8",
        "kind": "sampled_generation",
        "status": "pass" if passed else "fail",
        "model_label": args.model_label,
        "seed": args.seed,
        "health": health,
        "cases": cases,
    }


def render_token_count(chat: Path, model_directory: Path, message: str) -> int:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix="gem16-wp8-message-"
    ) as message_file:
        message_file.write(message)
        message_file.flush()
        completed = subprocess.run(
            [
                str(chat),
                "--model",
                str(model_directory),
                "--message-file",
                message_file.name,
                "--render-only",
                "--json",
                "--thinking-budget",
                "off",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    if completed.returncode != 0:
        raise BenchmarkError(
            "chat render failed: " + completed.stderr.strip()
        )
    document = json.loads(completed.stdout)
    tokens = document.get("prompt_token_ids")
    if not isinstance(tokens, list):
        raise BenchmarkError("chat render did not return prompt token IDs")
    return len(tokens)


def largest_repeat_count(
    target_tokens: int, counter: Callable[[int], int]
) -> tuple[int, int]:
    if target_tokens <= 0:
        raise BenchmarkError("target token count must be positive")
    low = 0
    high = 1
    while counter(high) <= target_tokens:
        low = high
        high *= 2
    while low + 1 < high:
        middle = low + (high - low) // 2
        if counter(middle) <= target_tokens:
            low = middle
        else:
            high = middle
    return low, counter(low)


def retrieval_message(
    contract: dict[str, Any], repeats: int, fraction: float
) -> str:
    before = int(repeats * fraction)
    after = repeats - before
    needle_sentence = contract["needle_sentence"].format(
        needle=contract["needle"]
    )
    return (
        contract["filler"] * before
        + needle_sentence
        + contract["filler"] * after
        + "\n\n"
        + contract["query"]
    )


def retrieval_run(args: argparse.Namespace, suite: dict[str, Any]) -> dict[str, Any]:
    v1, root = api_roots(args.base_url)
    health, _ = request_json(root + "/health", timeout=30.0)
    validate_health(health, args.model_label, suite)
    if (health.get("sampling") or {}).get("enabled"):
        raise BenchmarkError("retrieval check requires a greedy endpoint")
    contract = suite["long_context_retrieval"]
    model = endpoint_model(v1)
    suite_identity = file_identity(args.suite)
    cases: list[dict[str, Any]] = []
    if args.output.is_file():
        prior = load_json(args.output)
        if (
            prior.get("kind") != "long_context_retrieval"
            or prior.get("model_label") != args.model_label
            or prior.get("suite", {}).get("sha256") != suite_identity["sha256"]
            or prior.get("health", {}).get("artifact_content_sha256")
            != health.get("artifact_content_sha256")
        ):
            raise BenchmarkError("existing retrieval journal contract differs")
        if not isinstance(prior.get("cases"), list):
            raise BenchmarkError("existing retrieval journal cases are invalid")
        cases = prior["cases"]

    def save(status: str) -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "work_packet": "WP8",
                    "kind": "long_context_retrieval",
                    "status": status,
                    "model_label": args.model_label,
                    "health": health,
                    "cases": cases,
                    "suite": suite_identity,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    for context_tokens in contract["context_tokens"]:
        for fraction in contract["placement_fractions"]:
            if any(
                case.get("context_capacity") == context_tokens
                and case.get("placement_fraction") == fraction
                for case in cases
            ):
                continue
            target = int(context_tokens) - int(contract["max_output_tokens"]) - 16

            def counter(repeats: int) -> int:
                return render_token_count(
                    args.chat_executable,
                    args.tokenizer_model,
                    retrieval_message(contract, repeats, float(fraction)),
                )

            repeats, prompt_tokens = largest_repeat_count(target, counter)
            message = retrieval_message(contract, repeats, float(fraction))
            result, elapsed = completion(
                v1,
                model,
                message,
                int(contract["max_output_tokens"]),
                args.timeout,
            )
            usage = result.get("usage") or {}
            passed = contract["needle"] in result["content"]
            cases.append(
                {
                    "context_capacity": context_tokens,
                    "placement_fraction": fraction,
                    "filler_repetitions": repeats,
                    "rendered_prompt_tokens": prompt_tokens,
                    "server_prompt_tokens": usage.get("prompt_tokens"),
                    "elapsed_seconds": elapsed,
                    "result": result,
                    "needle_retrieved": passed,
                }
            )
            save("running")
    passed = all(case["needle_retrieved"] for case in cases)
    return {
        "schema_version": 1,
        "work_packet": "WP8",
        "kind": "long_context_retrieval",
        "status": "pass" if passed else "fail",
        "model_label": args.model_label,
        "health": health,
        "cases": cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("sampled", "retrieval"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--suite", required=True, type=Path)
        subparser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
        subparser.add_argument(
            "--model-label", required=True, choices=("candidate", "control")
        )
        subparser.add_argument("--output", required=True, type=Path)
        subparser.add_argument("--timeout", type=float, default=3600.0)
    sampled = subparsers.choices["sampled"]
    sampled.add_argument("--seed", required=True, type=int)
    retrieval = subparsers.choices["retrieval"]
    retrieval.add_argument("--chat-executable", required=True, type=Path)
    retrieval.add_argument("--tokenizer-model", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite = load_json(args.suite)
    if suite.get("status") != "frozen_before_wp8_candidate_results":
        raise BenchmarkError("suite is not frozen")
    result = (
        sampled_run(args, suite)
        if args.command == "sampled"
        else retrieval_run(args, suite)
    )
    result["suite"] = file_identity(args.suite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error

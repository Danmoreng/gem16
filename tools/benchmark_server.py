#!/usr/bin/env python3
"""Benchmark gem16 HTTP root, resident, streaming, and concurrent paths."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import math
import pathlib
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable


class BenchmarkError(RuntimeError):
    pass


class TelemetrySampler:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.samples: list[dict[str, float]] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        summary: dict[str, Any] = {
            "sample_interval_seconds": self.interval,
            "sample_count": len(self.samples),
            "error": self.error,
        }
        for field in ("memory_used_mib", "power_w", "sm_clock_mhz", "temperature_c"):
            values = [sample[field] for sample in self.samples]
            if values:
                summary[field] = distribution(values)
        return summary

    def _run(self) -> None:
        command = [
            "nvidia-smi",
            "--query-gpu=memory.used,power.draw,clocks.sm,temperature.gpu",
            "--format=csv,noheader,nounits",
            "--id=0",
        ]
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(
                    command, text=True, stderr=subprocess.DEVNULL, timeout=5
                ).strip().splitlines()[0]
                values = [float(value.strip()) for value in output.split(",")]
                if len(values) != 4:
                    raise ValueError("unexpected nvidia-smi field count")
                self.samples.append(
                    dict(
                        zip(
                            ("memory_used_mib", "power_w", "sm_clock_mhz", "temperature_c"),
                            values,
                        )
                    )
                )
            except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
                self.error = str(error)
                return
            self._stop.wait(self.interval)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def student_t_95(sample_count: int) -> float:
    # Two-sided 95% critical values for 1..30 degrees of freedom.
    critical = (
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306,
        2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120,
        2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064,
        2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
    )
    if sample_count <= 1:
        return 0.0
    degrees = sample_count - 1
    return critical[degrees - 1] if degrees <= len(critical) else 1.96


def distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise BenchmarkError("cannot summarize an empty sample")
    mean = statistics.fmean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    half_ci = student_t_95(len(values)) * deviation / math.sqrt(len(values))
    return {
        "count": len(values),
        "median": statistics.median(values),
        "mean": mean,
        "standard_deviation": deviation,
        "minimum": min(values),
        "maximum": max(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "confidence_95_low": mean - half_ci,
        "confidence_95_high": mean + half_ci,
    }


def request_text(url: str, timeout: float = 300.0) -> str:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise BenchmarkError(f"HTTP {error.code} from {url}: {detail}") from error


def prometheus_metrics(url: str, timeout: float) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in request_text(url, timeout).splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) == 2:
            values[fields[0]] = float(fields[1])
    return values


def request_json(
    url: str, payload: dict[str, Any] | None = None, timeout: float = 300.0
) -> tuple[dict[str, Any], float]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": "Bearer gem16-benchmark",
            "Content-Type": "application/json",
        },
        method="GET" if payload is None else "POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            document = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise BenchmarkError(f"HTTP {error.code} from {url}: {detail}") from error
    elapsed = time.perf_counter() - started
    return document, elapsed


def response_usage(document: dict[str, Any]) -> tuple[int, int, int, int]:
    usage = document.get("usage") or {}
    details = usage.get("input_tokens_details") or {}
    return (
        int(usage.get("input_tokens", 0)),
        int(usage.get("output_tokens", 0)),
        int(details.get("cached_tokens", 0)),
        int(details.get("cache_write_tokens", 0)),
    )


def response_run(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    document, elapsed = request_json(url, payload)
    input_tokens, output_tokens, cached_tokens, cache_write_tokens = (
        response_usage(document)
    )
    return {
        "response_id": document.get("id"),
        "elapsed_seconds": elapsed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
        "output_tokens_per_second": output_tokens / elapsed if elapsed else 0.0,
    }


def stream_run(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    stream_payload = dict(payload)
    stream_payload["stream"] = True
    request = urllib.request.Request(
        url,
        data=json.dumps(stream_payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer gem16-benchmark",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    started = time.perf_counter()
    first_delta: float | None = None
    delta_times: list[float] = []
    completed: dict[str, Any] | None = None
    stream_error: dict[str, Any] | None = None
    event_counts: dict[str, int] = {}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for encoded_line in response:
                line = encoded_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    continue
                event = json.loads(data)
                event_type = str(event.get("type", "unknown"))
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
                if event_type in {
                    "response.output_text.delta",
                    "response.reasoning_text.delta",
                }:
                    observed_at = time.perf_counter()
                    delta_times.append(observed_at)
                    if first_delta is None:
                        first_delta = observed_at
                if event_type == "response.completed":
                    completed = event.get("response")
                elif event_type == "error":
                    stream_error = event
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise BenchmarkError(f"HTTP {error.code} from {url}: {detail}") from error
    finished = time.perf_counter()
    if completed is None:
        detail = f": {stream_error}" if stream_error is not None else ""
        raise BenchmarkError(f"stream ended without response.completed{detail}")
    if first_delta is None:
        raise BenchmarkError("stream ended without a text or reasoning delta")
    input_tokens, output_tokens, cached_tokens, cache_write_tokens = (
        response_usage(completed)
    )
    elapsed = finished - started
    output_text = "".join(
        content.get("text", "")
        for item in completed.get("output", [])
        if item.get("type") == "message"
        for content in item.get("content", [])
        if content.get("type") == "output_text"
    )
    return {
        "response_id": completed.get("id"),
        "elapsed_seconds": elapsed,
        "time_to_first_delta_seconds": first_delta - started,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
        "output_tokens_per_second": output_tokens / elapsed if elapsed else 0.0,
        "delta_interval_seconds": [
            right - left for left, right in zip(delta_times, delta_times[1:])
        ],
        "output_text": output_text,
        "event_counts": event_counts,
    }


def measured_runs(
    operation: Callable[[], dict[str, Any]], warmups: int, repetitions: int
) -> list[dict[str, Any]]:
    for _ in range(warmups):
        operation()
    return [operation() for _ in range(repetitions)]


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "elapsed_seconds": distribution(
            [float(run["elapsed_seconds"]) for run in runs]
        ),
        "output_tokens_per_second": distribution(
            [float(run["output_tokens_per_second"]) for run in runs]
        ),
        "input_tokens": distribution(
            [float(run["input_tokens"]) for run in runs]
        ),
        "output_tokens": distribution(
            [float(run["output_tokens"]) for run in runs]
        ),
        "cached_tokens": distribution(
            [float(run["cached_tokens"]) for run in runs]
        ),
        "cache_write_tokens": distribution(
            [float(run["cache_write_tokens"]) for run in runs]
        ),
    }
    if "time_to_first_delta_seconds" in runs[0]:
        summary["time_to_first_delta_seconds"] = distribution(
            [float(run["time_to_first_delta_seconds"]) for run in runs]
        )
    if "delta_interval_seconds" in runs[0]:
        intervals = [
            float(interval)
            for run in runs
            for interval in run["delta_interval_seconds"]
        ]
        if intervals:
            summary["delta_interval_seconds"] = distribution(intervals)
    if "aggregate_output_tokens_per_second" in runs[0]:
        summary["aggregate_output_tokens_per_second"] = distribution(
            [float(run["aggregate_output_tokens_per_second"]) for run in runs]
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="gem16")
    parser.add_argument(
        "--scenario",
        choices=("all", "root", "continuation", "stream", "concurrency"),
        default="all",
    )
    parser.add_argument("--prompt", default="Explain why the sky is blue concisely.")
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument("--thinking-effort", default="none")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--telemetry-interval", type=float, default=0.2)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.max_output_tokens < 1 or args.concurrency < 1:
        parser.error("output tokens and concurrency must be positive")
    if args.warmup < 0 or args.repetitions < 1:
        parser.error("warmup must be nonnegative and repetitions positive")
    if args.telemetry_interval <= 0.0:
        parser.error("telemetry interval must be positive")
    if args.output is not None and args.output.exists():
        raise BenchmarkError(f"refusing to overwrite {args.output}")

    base_url = args.base_url.rstrip("/")
    server_url = base_url[:-3] if base_url.endswith("/v1") else base_url
    responses_url = f"{base_url}/responses"
    health, _ = request_json(f"{server_url}/health", timeout=args.timeout)
    metrics_before = prometheus_metrics(
        f"{server_url}/metrics", args.timeout
    )
    if args.scenario in {"all", "concurrency"} and int(
        health.get("session_limit", 0)
    ) < args.concurrency:
        raise BenchmarkError(
            "server session limit is smaller than requested concurrency"
        )

    common = {
        "model": args.model,
        "max_output_tokens": args.max_output_tokens,
        "reasoning": {"effort": args.thinking_effort},
    }
    scenarios = (
        ("root", "continuation", "stream", "concurrency")
        if args.scenario == "all"
        else (args.scenario,)
    )
    results: dict[str, Any] = {}
    telemetry_sampler = TelemetrySampler(args.telemetry_interval)
    telemetry_sampler.start()

    if "root" in scenarios:
        def root_operation() -> dict[str, Any]:
            return response_run(responses_url, {**common, "input": args.prompt})

        runs = measured_runs(root_operation, args.warmup, args.repetitions)
        results["root"] = {"summary": summarize(runs), "runs": runs}

    if "continuation" in scenarios:
        def continuation_operation() -> dict[str, Any]:
            root, _ = request_json(
                responses_url,
                {
                    **common,
                    "input": args.prompt,
                    "max_output_tokens": 1,
                },
                args.timeout,
            )
            return response_run(
                responses_url,
                {
                    **common,
                    "previous_response_id": root["id"],
                    "input": "Continue with one additional concise detail.",
                },
            )

        runs = measured_runs(
            continuation_operation, args.warmup, args.repetitions
        )
        results["continuation"] = {"summary": summarize(runs), "runs": runs}

    if "stream" in scenarios:
        def stream_operation() -> dict[str, Any]:
            return stream_run(
                responses_url, {**common, "input": args.prompt}, args.timeout
            )

        runs = measured_runs(stream_operation, args.warmup, args.repetitions)
        results["stream"] = {"summary": summarize(runs), "runs": runs}

    if "concurrency" in scenarios:
        def concurrency_operation() -> dict[str, Any]:
            roots = []
            for index in range(args.concurrency):
                root, _ = request_json(
                    responses_url,
                    {
                        **common,
                        "input": f"{args.prompt} Request lane {index}.",
                        "max_output_tokens": 1,
                    },
                    args.timeout,
                )
                roots.append(root["id"])
            release = threading.Event()

            def lane(index: int) -> dict[str, Any]:
                release.wait()
                return response_run(
                    responses_url,
                    {
                        **common,
                        "previous_response_id": roots[index],
                        "input": "Continue with one additional concise detail.",
                    },
                )

            started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.concurrency
            ) as executor:
                futures = [executor.submit(lane, index) for index in range(args.concurrency)]
                release.set()
                lanes = [future.result() for future in futures]
            elapsed = time.perf_counter() - started
            output_tokens = sum(int(lane_run["output_tokens"]) for lane_run in lanes)
            return {
                "elapsed_seconds": elapsed,
                "input_tokens": sum(int(lane_run["input_tokens"]) for lane_run in lanes),
                "output_tokens": output_tokens,
                "cached_tokens": sum(int(lane_run["cached_tokens"]) for lane_run in lanes),
                "cache_write_tokens": sum(int(lane_run["cache_write_tokens"]) for lane_run in lanes),
                "output_tokens_per_second": output_tokens / elapsed if elapsed else 0.0,
                "aggregate_output_tokens_per_second": output_tokens / elapsed if elapsed else 0.0,
                "lanes": lanes,
            }

        runs = measured_runs(
            concurrency_operation, args.warmup, args.repetitions
        )
        results["concurrency"] = {"summary": summarize(runs), "runs": runs}

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = "unknown"
    telemetry = telemetry_sampler.stop()
    metrics_after = prometheus_metrics(f"{server_url}/metrics", args.timeout)
    metric_deltas = {
        name: value - metrics_before.get(name, 0.0)
        for name, value in metrics_after.items()
        if name.endswith("_total")
    }
    document = {
        "schema_version": 1,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_sha": git_sha,
        "base_url": base_url,
        "model": args.model,
        "configuration": {
            "scenario": args.scenario,
            "prompt": args.prompt,
            "max_output_tokens": args.max_output_tokens,
            "thinking_effort": args.thinking_effort,
            "concurrency": args.concurrency,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "telemetry_interval_seconds": args.telemetry_interval,
        },
        "health": health,
        "gpu_telemetry": telemetry,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "metric_counter_deltas": metric_deltas,
        "results": results,
    }
    encoded = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with args.output.open("x", encoding="utf-8") as output_file:
                output_file.write(encoded)
        except FileExistsError as error:
            raise BenchmarkError(f"refusing to overwrite {args.output}") from error
    for name, result in results.items():
        summary = result["summary"]
        print(
            f"{name}: median={summary['elapsed_seconds']['median'] * 1000:.2f} ms "
            f"output={summary['output_tokens_per_second']['median']:.2f} tok/s",
            flush=True,
        )
        if "time_to_first_delta_seconds" in summary:
            print(
                "  first_delta="
                f"{summary['time_to_first_delta_seconds']['median'] * 1000:.2f} ms",
                flush=True,
            )
    if args.output is None:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as error:
        print(f"error: {error}", flush=True)
        raise SystemExit(2) from error

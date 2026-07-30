#!/usr/bin/env python3
"""Benchmark direct-checkpoint vLLM prefill and batch-one decode offline.

Run this tool from the pinned reference environment. It passes token IDs
directly, disables prefix caching and detokenization, and uses vLLM request
timestamps to separate time-to-first-token from decode intervals.
"""

from __future__ import annotations

try:
    from tools.hf_cache import default_target_model
except ModuleNotFoundError:
    from hf_cache import default_target_model

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any


T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


class BenchmarkError(RuntimeError):
    pass


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def lengths(value: str) -> list[int]:
    try:
        parsed = [positive_int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("lengths must be comma-separated integers") from error
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("lengths must not contain duplicates")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=default_target_model())
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument(
        "--lock", type=Path, default=Path("models/gemma4-12b-nvfp4.lock.json")
    )
    parser.add_argument("--prefill-lengths", type=lengths, default=[])
    parser.add_argument("--decode-contexts", type=lengths, default=[])
    parser.add_argument("--decode-tokens", type=positive_int, default=256)
    parser.add_argument("--warmups", type=positive_int, default=3)
    parser.add_argument("--repetitions", type=positive_int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--kv-cache-dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=positive_int)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def package_versions(names: tuple[str, ...]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise BenchmarkError(f"required package is not installed: {name}") from error
    return versions


def repository_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise BenchmarkError("cannot determine benchmark source revision") from error
    return {"git_commit": commit, "worktree_dirty_at_start": dirty}


def summarize(samples: list[float]) -> dict[str, Any]:
    if len(samples) < 2:
        raise BenchmarkError("at least two measured repetitions are required")
    mean = statistics.mean(samples)
    standard_deviation = statistics.stdev(samples)
    degrees_freedom = len(samples) - 1
    critical = T_CRITICAL_95.get(degrees_freedom, 1.960)
    half_width = critical * standard_deviation / math.sqrt(len(samples))
    return {
        "sample_count": len(samples),
        "mean": mean,
        "median": statistics.median(samples),
        "standard_deviation": standard_deviation,
        "minimum": min(samples),
        "maximum": max(samples),
        "confidence_interval_95": [mean - half_width, mean + half_width],
        "samples": samples,
    }


def prompt_tokens(count: int, seed: int) -> list[int]:
    # Stay clear of control and modality token ranges while keeping generation
    # deterministic. Dense-model execution does not depend on token semantics.
    return [1000 + ((seed + index * 7919) % 9000) for index in range(count)]


def metric_value(metrics: Any, name: str) -> float:
    if metrics is None or not hasattr(metrics, name):
        raise BenchmarkError(f"vLLM request metrics do not expose {name}")
    value = float(getattr(metrics, name))
    if not math.isfinite(value):
        raise BenchmarkError(f"vLLM request metric {name} is not finite")
    return value


def run_request(
    llm: Any,
    sampling_params_type: Any,
    input_ids: list[int],
    output_tokens: int,
    seed: int,
) -> dict[str, float]:
    sampling = sampling_params_type(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=output_tokens,
        detokenize=False,
        seed=seed,
    )
    started = time.perf_counter()
    request = llm.generate(
        [{"prompt_token_ids": input_ids}], sampling, use_tqdm=False
    )[0]
    wall_seconds = time.perf_counter() - started
    generated = len(request.outputs[0].token_ids)
    if generated != output_tokens:
        raise BenchmarkError(
            f"expected {output_tokens} generated tokens, received {generated}"
        )
    metrics = request.metrics
    first_token_latency = metric_value(metrics, "first_token_latency")
    first_token_ts = metric_value(metrics, "first_token_ts")
    last_token_ts = metric_value(metrics, "last_token_ts")
    decode_seconds = last_token_ts - first_token_ts
    if output_tokens > 1 and decode_seconds <= 0.0:
        raise BenchmarkError("vLLM reported non-positive decode time")
    return {
        "wall_seconds": wall_seconds,
        "first_token_latency_seconds": first_token_latency,
        "decode_seconds": decode_seconds,
    }


def benchmark_prefill(
    llm: Any,
    sampling_params_type: Any,
    token_count: int,
    warmups: int,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    input_ids = prompt_tokens(token_count, seed)
    for _ in range(warmups):
        run_request(llm, sampling_params_type, input_ids, 1, seed)
    samples = [
        run_request(llm, sampling_params_type, input_ids, 1, seed)
        for _ in range(repetitions)
    ]
    ttft = [sample["first_token_latency_seconds"] for sample in samples]
    wall = [sample["wall_seconds"] for sample in samples]
    throughput = [token_count / value for value in ttft]
    return {
        "mode": "prefill",
        "prompt_tokens": token_count,
        "generated_tokens": 1,
        "time_to_first_token_seconds": summarize(ttft),
        "prompt_throughput_tokens_per_second": summarize(throughput),
        "end_to_end_wall_seconds": summarize(wall),
    }


def benchmark_decode(
    llm: Any,
    sampling_params_type: Any,
    context_tokens: int,
    decode_tokens: int,
    warmups: int,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    input_ids = prompt_tokens(context_tokens, seed)
    requested_tokens = decode_tokens + 1
    for _ in range(warmups):
        run_request(llm, sampling_params_type, input_ids, requested_tokens, seed)
    samples = [
        run_request(llm, sampling_params_type, input_ids, requested_tokens, seed)
        for _ in range(repetitions)
    ]
    ttft = [sample["first_token_latency_seconds"] for sample in samples]
    decode_seconds = [sample["decode_seconds"] for sample in samples]
    wall = [sample["wall_seconds"] for sample in samples]
    throughput = [decode_tokens / value for value in decode_seconds]
    inter_token_ms = [1000.0 * value / decode_tokens for value in decode_seconds]
    return {
        "mode": "decode",
        "existing_context_tokens": context_tokens,
        "generated_tokens": decode_tokens,
        "untimed_first_token": 1,
        "time_to_first_token_seconds": summarize(ttft),
        "decode_throughput_tokens_per_second": summarize(throughput),
        "average_inter_token_latency_ms": summarize(inter_token_ms),
        "end_to_end_wall_seconds": summarize(wall),
        "latency_limitation": (
            "request timestamps expose average decode interval, not the per-token latency distribution"
        ),
    }


def main() -> int:
    args = parse_args()
    if not args.prefill_lengths and not args.decode_contexts:
        print("error: at least one benchmark length is required", file=sys.stderr)
        return 2
    if args.repetitions < 2:
        print("error: at least two measured repetitions are required", file=sys.stderr)
        return 2
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        print("error: GPU memory utilization must be in (0, 1]", file=sys.stderr)
        return 2

    try:
        import torch
        from vllm import LLM, SamplingParams

        if not torch.cuda.is_available():
            raise BenchmarkError("CUDA is not available to the vLLM runtime")
        model = args.model.resolve(strict=True)
        lock = json.loads(args.lock.read_text(encoding="utf-8"))
        required_context = max(
            [length + 1 for length in args.prefill_lengths]
            + [length + args.decode_tokens + 1 for length in args.decode_contexts]
        )
        max_model_len = args.max_model_len or required_context
        if max_model_len < required_context:
            raise BenchmarkError(
                f"max model length {max_model_len} is below required {required_context}"
            )

        llm = LLM(
            model=str(model),
            tokenizer=str(model),
            max_model_len=max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            cpu_offload_gb=0,
            enforce_eager=args.enforce_eager,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            kv_cache_dtype=args.kv_cache_dtype,
            max_num_seqs=1,
            disable_log_stats=False,
            seed=args.seed,
            limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0},
        )

        results: list[dict[str, Any]] = []
        for token_count in args.prefill_lengths:
            print(f"benchmarking prefill={token_count}", flush=True)
            results.append(
                benchmark_prefill(
                    llm,
                    SamplingParams,
                    token_count,
                    args.warmups,
                    args.repetitions,
                    args.seed,
                )
            )
        for context_tokens in args.decode_contexts:
            print(f"benchmarking decode_context={context_tokens}", flush=True)
            results.append(
                benchmark_decode(
                    llm,
                    SamplingParams,
                    context_tokens,
                    args.decode_tokens,
                    args.warmups,
                    args.repetitions,
                    args.seed,
                )
            )

        device = torch.cuda.get_device_properties(0)
        document = {
            "schema_version": 1,
            "status": "development_characterization_not_accepted_baseline",
            "benchmark_source": repository_state(),
            "checkpoint": {
                "repository": lock.get("repository"),
                "revision": lock.get("revision"),
                "local_directory_name": model.name,
            },
            "runtime": {
                "python": platform.python_version(),
                "packages": package_versions(
                    (
                        "vllm",
                        "torch",
                        "transformers",
                        "compressed-tensors",
                        "flashinfer-python",
                        "nvidia-cutlass-dsl",
                    )
                ),
                "torch_cuda": torch.version.cuda,
            },
            "hardware": {
                "device_name": device.name,
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "total_memory_bytes": device.total_memory,
            },
            "configuration": {
                "batch_size": 1,
                "max_model_len": max_model_len,
                "kv_cache_dtype": args.kv_cache_dtype,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "cpu_offload_gb": 0,
                "enforce_eager": args.enforce_eager,
                "cuda_graphs_requested": not args.enforce_eager,
                "prefix_caching": False,
                "chunked_prefill": True,
                "text_only": True,
                "detokenize": False,
                "temperature": 0.0,
                "ignore_eos": True,
                "seed": args.seed,
                "warmups": args.warmups,
                "measured_repetitions": args.repetitions,
                "primary_statistic": "median",
            },
            "results": results,
            "limitations": [
                "No power, clock, or thermal time series was captured.",
                "Request metrics provide average decode intervals, not p95/p99 per-token latency.",
                "The direct checkpoint and closest-parity llama.cpp GGUF differ in attention weight precision.",
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(document, indent=2, sort_keys=True) + "\n"
        args.output.write_text(serialized, encoding="utf-8")
        if args.summary_output is not None:
            args.summary_output.parent.mkdir(parents=True, exist_ok=True)
            args.summary_output.write_text(serialized, encoding="utf-8")
        print(f"wrote {args.output} with {len(results)} configurations")
        return 0
    except (BenchmarkError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

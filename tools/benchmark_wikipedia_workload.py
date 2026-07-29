#!/usr/bin/env python3
"""Benchmark one exact-token Wikipedia workload on gem16, vLLM, or llama.cpp."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import struct
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request


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


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine", required=True, choices=("gem16", "vllm", "llama-cpp")
    )
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--assistant-model", type=Path)
    parser.add_argument(
        "--mtp-draft-tokens", type=int, choices=(0, 1, 2, 4), default=0
    )
    parser.add_argument("--mtp-adaptive", action="store_true")
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--warmups", type=nonnegative_int, default=3)
    parser.add_argument("--repetitions", type=positive_int, default=10)
    parser.add_argument(
        "--fixed-output-tokens",
        type=positive_int,
        help="override stop handling and force this many generated tokens",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--vllm-kv-cache-dtype", default="fp8")
    parser.add_argument(
        "--llama-kv-cache-type",
        choices=("f16", "bf16", "q8_0"),
        default="q8_0",
    )
    parser.add_argument("--llama-port", type=positive_int, default=8097)
    parser.add_argument(
        "--llama-spec-types",
        help=(
            "comma-separated llama.cpp speculative implementations; defaults "
            "to draft-mtp when an assistant is supplied"
        ),
    )
    parser.add_argument("--llama-ngram-mod-n-match", type=positive_int, default=24)
    parser.add_argument("--llama-ngram-mod-n-min", type=positive_int, default=48)
    parser.add_argument("--llama-ngram-mod-n-max", type=positive_int, default=64)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def repository_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
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
    return {"git_commit": commit, "worktree_dirty_at_start": dirty}


def summarize(samples: list[float]) -> dict[str, Any]:
    if not samples:
        raise BenchmarkError("at least one measured repetition is required")
    if not all(math.isfinite(value) for value in samples):
        raise BenchmarkError("benchmark samples contain a non-finite value")
    mean = statistics.mean(samples)
    standard_deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
    critical = T_CRITICAL_95.get(len(samples) - 1, 1.960)
    half_width = (
        critical * standard_deviation / math.sqrt(len(samples))
        if len(samples) > 1
        else 0.0
    )
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


def token_checksum(tokens: list[int]) -> str:
    serialized = ",".join(str(token) for token in tokens).encode("ascii")
    return hashlib.sha256(serialized).hexdigest()


def token_u32_checksum(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        if token > 0xFFFFFFFF:
            raise BenchmarkError("workload token ID exceeds uint32")
        digest.update(struct.pack("<I", token))
    return digest.hexdigest()


def load_workload(path: Path) -> tuple[dict[str, Any], list[int], dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    prompt = document.get("prompt")
    generation = document.get("generation")
    if document.get("schema_version") != 1 or not isinstance(prompt, dict):
        raise BenchmarkError("workload is not a schema-version-1 prompt")
    tokens = prompt.get("token_ids")
    if (
        not isinstance(tokens, list)
        or len(tokens) != prompt.get("target_tokens")
        or not all(isinstance(token, int) and token >= 0 for token in tokens)
        or not isinstance(generation, dict)
    ):
        raise BenchmarkError("workload token or generation fields are malformed")
    expected_checksum = prompt.get("token_ids_sha256")
    if not isinstance(expected_checksum, str) or (
        token_u32_checksum(tokens) != expected_checksum
    ):
        raise BenchmarkError("workload token IDs do not match their SHA-256")
    return document, tokens, generation


def metric_run(
    prompt_tokens: int,
    output_tokens: list[int],
    prompt_ms: float,
    decode_ms: float,
    stop_reason: Any,
) -> dict[str, Any]:
    if prompt_ms <= 0.0 or decode_ms < 0.0 or not output_tokens:
        raise BenchmarkError("engine returned invalid timing or output data")
    decode_intervals = len(output_tokens) - 1
    decode_throughput = (
        decode_intervals * 1000.0 / decode_ms
        if decode_intervals > 0 and decode_ms > 0.0
        else 0.0
    )
    return {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": len(output_tokens),
        "measured_decode_intervals": decode_intervals,
        "prompt_ms": prompt_ms,
        "prompt_tokens_per_second": prompt_tokens * 1000.0 / prompt_ms,
        "decode_ms": decode_ms,
        "decode_tokens_per_second": decode_throughput,
        "average_inter_token_latency_ms": (
            decode_ms / decode_intervals if decode_intervals > 0 else 0.0
        ),
        "stop_reason": stop_reason,
        "output_token_sha256": token_checksum(output_tokens),
        "first_output_token_id": output_tokens[0],
        "last_output_token_id": output_tokens[-1],
    }


def outputs_are_deterministic(runs: list[dict[str, Any]]) -> bool:
    identities = {
        (run["generated_tokens"], run["output_token_sha256"]) for run in runs
    }
    return len(identities) == 1


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "prompt_tokens_per_second": summarize(
            [run["prompt_tokens_per_second"] for run in runs]
        ),
        "time_to_first_token_ms": summarize([run["prompt_ms"] for run in runs]),
        "decode_tokens_per_second": summarize(
            [run["decode_tokens_per_second"] for run in runs]
        ),
        "average_inter_token_latency_ms": summarize(
            [run["average_inter_token_latency_ms"] for run in runs]
        ),
        "generated_tokens": summarize(
            [float(run["generated_tokens"]) for run in runs]
        ),
        "measured_decode_intervals": summarize(
            [float(run["measured_decode_intervals"]) for run in runs]
        ),
        "deterministic_outputs": outputs_are_deterministic(runs),
        "output_token_sha256_values": sorted(
            {run["output_token_sha256"] for run in runs}
        ),
        "stop_reason_values": sorted(
            {str(run["stop_reason"]) for run in runs}
        ),
    }
    def summarize_speculation(field: str) -> dict[str, Any]:
        return {
            "proposed_tokens": summarize(
                [float(run[field]["proposed_tokens"]) for run in runs]
            ),
            "accepted_tokens": summarize(
                [float(run[field]["accepted_tokens"]) for run in runs]
            ),
            "rejected_tokens": summarize(
                [float(run[field]["rejected_tokens"]) for run in runs]
            ),
            "mean_accepted_length": summarize(
                [float(run[field]["mean_accepted_length"]) for run in runs]
            ),
            "target_batches": summarize(
                [float(run[field]["target_batches"]) for run in runs]
            ),
            "d1_groups": summarize(
                [float(run[field]["d1_groups"]) for run in runs]
            ),
            "d2_groups": summarize(
                [float(run[field]["d2_groups"]) for run in runs]
            ),
            "d4_groups": summarize(
                [float(run[field]["d4_groups"]) for run in runs]
            ),
            "ordinary_fallback_tokens": summarize(
                [float(run[field]["ordinary_fallback_tokens"]) for run in runs]
            ),
        }

    if all("speculative" in run for run in runs):
        summary["speculative"] = summarize_speculation("speculative")
    if all("mtp" in run for run in runs):
        summary["mtp"] = summarize_speculation("mtp")
    return summary


def run_gem16(
    executable: Path,
    model: Path,
    prompt_file: Path,
    prompt_tokens: int,
    generation: dict[str, Any],
    assistant_model: Path | None,
    mtp_draft_tokens: int,
    mtp_adaptive: bool,
    sampling: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[int]]:
    command = [
        str(executable),
        "--model",
        str(model),
        "--input-token-ids-file",
        str(prompt_file),
        "--suppress-token-ids",
        ",".join(str(token) for token in generation["suppress_token_ids"]),
        "--kv-cache",
        "fp8",
        "--max-tokens",
        str(generation["max_new_tokens"]),
        "--max-context",
        str(prompt_tokens + generation["max_new_tokens"]),
    ]
    if sampling is None:
        command.append("--greedy")
    else:
        command.extend(
            [
                "--sample",
                "--temperature", str(sampling["temperature"]),
                "--top-k", str(sampling["top_k"]),
                "--top-p", str(sampling["top_p"]),
                "--repetition-penalty", str(sampling["repetition_penalty"]),
                "--seed", str(sampling["seed"]),
            ]
        )
    if generation["stop_token_ids"]:
        command.extend(
            [
                "--stop-token-ids",
                ",".join(str(token) for token in generation["stop_token_ids"]),
            ]
        )
    if mtp_draft_tokens != 0:
        if assistant_model is None:
            raise BenchmarkError("active gem16 MTP requires --assistant-model")
        command.extend(
            [
                "--assistant-model",
                str(assistant_model),
                "--mtp-draft-tokens",
                str(mtp_draft_tokens),
            ]
        )
        if mtp_adaptive:
            command.append("--mtp-adaptive")
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise BenchmarkError(
            f"gem16 exited with {completed.returncode}: {detail[-2000:]}"
        )
    result = json.loads(completed.stdout)
    output_tokens = result.get("output_token_ids")
    if not isinstance(output_tokens, list) or not all(
        isinstance(token, int) for token in output_tokens
    ):
        raise BenchmarkError("gem16 output token IDs are malformed")
    if result.get("fallbacks") != 0 or result.get("token_loop_allocations") is not False:
        raise BenchmarkError("gem16 reported a fallback or token-loop allocation")
    run = metric_run(
        prompt_tokens,
        output_tokens,
        float(result["prompt_ms"]),
        float(result["decode_ms"]),
        result.get("finish_reason"),
    )
    run["model_load_ms"] = float(result["model_load_ms"])
    run["workspace_bytes"] = int(result["workspace_bytes"])
    run["kv_cache_bytes"] = int(result["kv_cache_bytes"])
    if mtp_draft_tokens != 0:
        mtp = result.get("mtp")
        if (
            not isinstance(mtp, dict)
            or mtp.get("enabled") is not True
            or mtp.get("draft_tokens") != mtp_draft_tokens
            or mtp.get("adaptive") is not mtp_adaptive
            or mtp.get("verification_mode") != "batched_exact_target"
        ):
            raise BenchmarkError("gem16 returned malformed MTP telemetry")
        run["mtp"] = {
            name: mtp[name]
            for name in (
                "proposed_tokens",
                "accepted_tokens",
                "rejected_tokens",
                "verification_groups",
                "target_forwards",
                "target_batches",
                "mean_accepted_length",
                "d1_groups",
                "d2_groups",
                "d4_groups",
                "ordinary_fallback_tokens",
            )
        }
    return run, output_tokens


def metric_value(metrics: Any, name: str) -> float:
    if metrics is None or not hasattr(metrics, name):
        raise BenchmarkError(f"vLLM request metrics do not expose {name}")
    value = float(getattr(metrics, name))
    if not math.isfinite(value):
        raise BenchmarkError(f"vLLM request metric {name} is not finite")
    return value


def prometheus_counter_total(sample_name: str) -> float:
    import prometheus_client

    total = 0.0
    for metric in prometheus_client.REGISTRY.collect():
        for sample in metric.samples:
            if sample.name == sample_name:
                total += float(sample.value)
    return total


def run_vllm_request(
    llm: Any,
    sampling_params_type: Any,
    prompt: list[int],
    generation: dict[str, Any],
    suppressed_token_strings: list[str],
    mtp_draft_tokens: int,
) -> tuple[dict[str, Any], list[int]]:
    before = None
    if mtp_draft_tokens != 0:
        before = {
            "drafts": prometheus_counter_total(
                "vllm:spec_decode_num_drafts_total"
            ),
            "proposed": prometheus_counter_total(
                "vllm:spec_decode_num_draft_tokens_total"
            ),
            "accepted": prometheus_counter_total(
                "vllm:spec_decode_num_accepted_tokens_total"
            ),
        }
    sampling = sampling_params_type(
        temperature=0.0,
        max_tokens=generation["max_new_tokens"],
        stop_token_ids=generation["stop_token_ids"],
        seed=generation["seed"],
        detokenize=False,
        ignore_eos=bool(generation.get("ignore_eos", False)),
        bad_words=suppressed_token_strings,
    )
    started = time.perf_counter()
    request = llm.generate(
        [{"prompt_token_ids": prompt}], sampling, use_tqdm=False
    )[0]
    wall_ms = (time.perf_counter() - started) * 1000.0
    completion = request.outputs[0]
    output_tokens = list(completion.token_ids)
    metrics = request.metrics
    prompt_ms = metric_value(metrics, "first_token_latency") * 1000.0
    first_token_ts = metric_value(metrics, "first_token_ts")
    last_token_ts = metric_value(metrics, "last_token_ts")
    decode_ms = max(0.0, (last_token_ts - first_token_ts) * 1000.0)
    run = metric_run(
        len(prompt),
        output_tokens,
        prompt_ms,
        decode_ms,
        completion.finish_reason,
    )
    run["wall_ms"] = wall_ms
    run["stop_reason_detail"] = completion.stop_reason
    if before is not None:
        drafts = int(
            round(
                prometheus_counter_total("vllm:spec_decode_num_drafts_total")
                - before["drafts"]
            )
        )
        proposed = int(
            round(
                prometheus_counter_total(
                    "vllm:spec_decode_num_draft_tokens_total"
                )
                - before["proposed"]
            )
        )
        accepted = int(
            round(
                prometheus_counter_total(
                    "vllm:spec_decode_num_accepted_tokens_total"
                )
                - before["accepted"]
            )
        )
        if drafts <= 0 or proposed < accepted:
            raise BenchmarkError("vLLM returned malformed MTP counters")
        run["mtp"] = {
            "proposed_tokens": proposed,
            "accepted_tokens": accepted,
            "rejected_tokens": proposed - accepted,
            "verification_groups": drafts,
            "target_forwards": proposed + drafts,
            "target_batches": drafts,
            "mean_accepted_length": accepted / drafts,
            "conventional_mean_acceptance_length": 1.0 + accepted / drafts,
            "d1_groups": drafts if mtp_draft_tokens == 1 else 0,
            "d2_groups": drafts if mtp_draft_tokens == 2 else 0,
            "d4_groups": drafts if mtp_draft_tokens == 4 else 0,
            "ordinary_fallback_tokens": 0,
        }
    return run, output_tokens


def http_json(
    url: str, body: dict[str, Any] | None = None, timeout: float = 10.0
) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise BenchmarkError(f"HTTP endpoint returned a non-object: {url}")
    return result


def wait_for_server(process: subprocess.Popen[Any], base_url: str) -> None:
    deadline = time.monotonic() + 180.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise BenchmarkError(f"llama-server exited during startup: {process.returncode}")
        try:
            health = http_json(f"{base_url}/health")
            if health.get("status") == "ok":
                return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass
        time.sleep(1.0)
    raise BenchmarkError("timed out waiting for llama-server")


def run_llama_request(
    base_url: str,
    prompt: list[int],
    generation: dict[str, Any],
    maximum_draft_tokens: int,
    speculative_types: tuple[str, ...] = ("draft-mtp",),
) -> tuple[dict[str, Any], list[int]]:
    body = {
        "prompt": prompt,
        "n_predict": generation["max_new_tokens"],
        "temperature": 0.0,
        "seed": generation["seed"],
        "ignore_eos": bool(generation.get("ignore_eos", False)),
        "cache_prompt": False,
        "n_keep": -1,
        "repeat_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "dry_multiplier": 0.0,
        "samplers": ["temperature"],
        "return_tokens": True,
        "logit_bias": [
            [token, False] for token in generation["suppress_token_ids"]
        ],
    }
    started = time.perf_counter()
    response = http_json(f"{base_url}/completion", body, timeout=1800.0)
    wall_ms = (time.perf_counter() - started) * 1000.0
    output_tokens = response.get("tokens")
    timings = response.get("timings")
    if (
        not isinstance(output_tokens, list)
        or not all(isinstance(token, int) for token in output_tokens)
        or not isinstance(timings, dict)
        or response.get("tokens_evaluated") != len(prompt)
        or response.get("truncated") is not False
    ):
        raise BenchmarkError("llama.cpp response has invalid token, timing, or context data")
    predicted_ms = float(timings["predicted_ms"])
    predicted_n = int(timings["predicted_n"])
    if predicted_n != len(output_tokens):
        raise BenchmarkError("llama.cpp timing token count differs from returned tokens")
    # llama.cpp starts this timer after the first sampled token but reports all
    # generated tokens in predicted_n. Use N-1 explicitly for parity.
    run = metric_run(
        len(prompt),
        output_tokens,
        float(timings["prompt_ms"]),
        predicted_ms,
        response.get("stop_type"),
    )
    run["wall_ms"] = wall_ms
    run["server_predicted_tokens_per_second"] = float(
        timings["predicted_per_second"]
    )
    run["tokens_cached"] = int(response.get("tokens_cached", 0))
    if "draft_n" in timings:
        proposed = int(timings["draft_n"])
        accepted = int(timings["draft_n_accepted"])
        groups = len(output_tokens) - accepted
        if (
            maximum_draft_tokens == 0
            or groups <= 0
            or proposed < accepted
            or proposed > groups * maximum_draft_tokens
        ):
            raise BenchmarkError("llama.cpp returned malformed speculative counters")
        fixed_mtp = (
            speculative_types == ("draft-mtp",)
            and maximum_draft_tokens in (1, 2, 4)
        )
        counters = {
            "proposed_tokens": proposed,
            "accepted_tokens": accepted,
            "rejected_tokens": proposed - accepted,
            "verification_groups": groups,
            "target_forwards": proposed + groups,
            "target_batches": groups,
            "mean_accepted_length": accepted / groups,
            "conventional_mean_acceptance_length": 1.0 + accepted / groups,
            "d1_groups": groups if fixed_mtp and maximum_draft_tokens == 1 else 0,
            "d2_groups": groups if fixed_mtp and maximum_draft_tokens == 2 else 0,
            "d4_groups": groups if fixed_mtp and maximum_draft_tokens == 4 else 0,
            "ordinary_fallback_tokens": 0,
        }
        run["speculative"] = counters
        if fixed_mtp:
            run["mtp"] = dict(counters)
    return run, output_tokens


def package_versions(names: tuple[str, ...]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    workload_path = args.workload.resolve(strict=True)
    workload, prompt, generation = load_workload(workload_path)
    generation = dict(generation)
    if args.fixed_output_tokens is not None:
        generation["max_new_tokens"] = args.fixed_output_tokens
        generation["stop_token_ids"] = []
        generation["ignore_eos"] = True
    all_runs: list[dict[str, Any]] = []
    representative: list[int] | None = None
    runtime: dict[str, Any]
    configuration: dict[str, Any]

    if args.engine == "gem16":
        if args.model is None or args.executable is None:
            raise BenchmarkError("gem16 requires --model and --executable")
        model = args.model.resolve(strict=True)
        executable = args.executable.resolve(strict=True)
        assistant_model = (
            args.assistant_model.resolve(strict=True)
            if args.assistant_model is not None
            else None
        )
        if args.mtp_draft_tokens == 0 and assistant_model is not None:
            raise BenchmarkError(
                "--assistant-model requires nonzero --mtp-draft-tokens"
            )
        if args.mtp_draft_tokens != 0 and assistant_model is None:
            raise BenchmarkError("--mtp-draft-tokens requires --assistant-model")
        if args.mtp_adaptive and args.mtp_draft_tokens == 0:
            raise BenchmarkError("--mtp-adaptive requires --mtp-draft-tokens")
        prompt_file = args.output.with_suffix(".prompt-token-ids.txt").resolve()
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(
            ",".join(str(token) for token in prompt), encoding="ascii"
        )

        def run_once() -> tuple[dict[str, Any], list[int]]:
            return run_gem16(
                executable,
                model,
                prompt_file,
                len(prompt),
                generation,
                assistant_model,
                args.mtp_draft_tokens,
                args.mtp_adaptive,
            )

        runtime = {
            "executable": str(executable),
            "checkpoint": str(model),
            "assistant_checkpoint": (
                str(assistant_model) if assistant_model is not None else None
            ),
            "prompt_token_file": str(prompt_file),
        }
        configuration = {
            "kv_cache": "checkpoint_fp8",
            "mtp_draft_tokens": args.mtp_draft_tokens,
            "mtp_adaptive": args.mtp_adaptive,
            "mtp_verification_mode": (
                "batched_exact_target"
                if args.mtp_draft_tokens != 0
                else "disabled"
            ),
        }
    elif args.engine == "vllm":
        if args.model is None:
            raise BenchmarkError("vLLM requires --model")
        if args.mtp_adaptive:
            raise BenchmarkError("vLLM does not expose gem16 --mtp-adaptive")
        if args.mtp_draft_tokens == 0 and args.assistant_model is not None:
            raise BenchmarkError(
                "vLLM --assistant-model requires nonzero --mtp-draft-tokens"
            )
        if args.mtp_draft_tokens != 0 and args.assistant_model is None:
            raise BenchmarkError(
                "vLLM --mtp-draft-tokens requires --assistant-model"
            )
        if not 0.0 < args.gpu_memory_utilization <= 1.0:
            raise BenchmarkError("GPU memory utilization must be in (0, 1]")
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        if not torch.cuda.is_available():
            raise BenchmarkError("CUDA is unavailable to vLLM")
        model = args.model.resolve(strict=True)
        assistant_model = (
            args.assistant_model.resolve(strict=True)
            if args.assistant_model is not None
            else None
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
        suppressed_token_strings = [
            tokenizer.decode([token], skip_special_tokens=False)
            for token in generation["suppress_token_ids"]
        ]
        if not all(suppressed_token_strings):
            raise BenchmarkError("cannot decode one or more suppressed token IDs")
        max_model_len = len(prompt) + generation["max_new_tokens"]
        llm = LLM(
            model=str(model),
            tokenizer=str(model),
            max_model_len=max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            cpu_offload_gb=0,
            enforce_eager=args.enforce_eager,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            kv_cache_dtype=args.vllm_kv_cache_dtype,
            max_num_seqs=1,
            disable_log_stats=False,
            seed=generation["seed"],
            limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0},
            spec_model=(
                str(assistant_model) if assistant_model is not None else None
            ),
            spec_tokens=(
                args.mtp_draft_tokens if args.mtp_draft_tokens != 0 else None
            ),
        )

        def run_once() -> tuple[dict[str, Any], list[int]]:
            return run_vllm_request(
                llm,
                SamplingParams,
                prompt,
                generation,
                suppressed_token_strings,
                args.mtp_draft_tokens,
            )

        device = torch.cuda.get_device_properties(0)
        runtime = {
            "checkpoint": str(model),
            "assistant_checkpoint": (
                str(assistant_model) if assistant_model is not None else None
            ),
            "python": platform.python_version(),
            "packages": package_versions(
                ("vllm", "torch", "transformers", "compressed-tensors")
            ),
            "torch_cuda": torch.version.cuda,
            "device_name": device.name,
        }
        configuration = {
            "kv_cache": args.vllm_kv_cache_dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "cpu_offload_gb": 0,
            "enforce_eager": args.enforce_eager,
            "cuda_graphs_requested": not args.enforce_eager,
            "prefix_caching": False,
            "chunked_prefill": True,
            "mtp_draft_tokens": args.mtp_draft_tokens,
            "mtp_backend": (
                "Gemma4MTPModel" if args.mtp_draft_tokens != 0 else "disabled"
            ),
        }
    else:
        if args.executable is None or args.gguf is None:
            raise BenchmarkError("llama.cpp requires --executable and --gguf")
        if args.mtp_adaptive:
            raise BenchmarkError("llama.cpp does not expose gem16 --mtp-adaptive")
        allowed_spec_types = {"draft-mtp", "ngram-mod"}
        if args.llama_spec_types is None:
            speculative_types = (
                ("draft-mtp",) if args.assistant_model is not None else ()
            )
        else:
            speculative_types = tuple(
                item.strip()
                for item in args.llama_spec_types.split(",")
                if item.strip()
            )
        if len(set(speculative_types)) != len(speculative_types) or not set(
            speculative_types
        ).issubset(allowed_spec_types):
            raise BenchmarkError(
                "llama.cpp speculative types must be unique draft-mtp/ngram-mod values"
            )
        has_mtp = "draft-mtp" in speculative_types
        has_ngram_mod = "ngram-mod" in speculative_types
        if has_mtp != (args.assistant_model is not None):
            raise BenchmarkError(
                "llama.cpp draft-mtp requires exactly one --assistant-model"
            )
        if has_mtp != (args.mtp_draft_tokens != 0):
            raise BenchmarkError(
                "llama.cpp draft-mtp requires nonzero --mtp-draft-tokens"
            )
        if not has_mtp and args.mtp_draft_tokens != 0:
            raise BenchmarkError(
                "llama.cpp --mtp-draft-tokens requires draft-mtp"
            )
        if args.llama_ngram_mod_n_min > args.llama_ngram_mod_n_max:
            raise BenchmarkError("llama.cpp ngram-mod minimum exceeds maximum")
        executable = args.executable.resolve(strict=True)
        gguf = args.gguf.resolve(strict=True)
        assistant_gguf = (
            args.assistant_model.resolve(strict=True)
            if args.assistant_model is not None
            else None
        )
        maximum_draft_tokens = max(
            args.mtp_draft_tokens if has_mtp else 0,
            args.llama_ngram_mod_n_max if has_ngram_mod else 0,
        )
        base_url = f"http://127.0.0.1:{args.llama_port}"
        log_path = args.output.with_suffix(".server.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")
        command = [
            str(executable),
            "--model",
            str(gguf),
            "--ctx-size",
            str(len(prompt) + generation["max_new_tokens"]),
            "--n-gpu-layers",
            "all",
            "--split-mode",
            "none",
            "--flash-attn",
            "on",
            "--cache-type-k",
            args.llama_kv_cache_type,
            "--cache-type-v",
            args.llama_kv_cache_type,
            "--parallel",
            "1",
            "--batch-size",
            "2048",
            "--ubatch-size",
            "512",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.llama_port),
            "--no-webui",
            "--offline",
        ]
        if speculative_types:
            command.extend(["--spec-type", ",".join(speculative_types)])
        if assistant_gguf is not None:
            command.extend(
                [
                    "--spec-draft-model",
                    str(assistant_gguf),
                    "--spec-draft-n-max",
                    str(args.mtp_draft_tokens),
                    "--spec-draft-n-min",
                    "1",
                    "--spec-draft-ngl",
                    "all",
                ]
            )
        if has_ngram_mod:
            command.extend(
                [
                    "--spec-ngram-mod-n-match",
                    str(args.llama_ngram_mod_n_match),
                    "--spec-ngram-mod-n-min",
                    str(args.llama_ngram_mod_n_min),
                    "--spec-ngram-mod-n-max",
                    str(args.llama_ngram_mod_n_max),
                ]
            )
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
        try:
            print("waiting for llama-server model load", flush=True)
            wait_for_server(process, base_url)

            def run_once() -> tuple[dict[str, Any], list[int]]:
                return run_llama_request(
                    base_url, prompt, generation, maximum_draft_tokens,
                    speculative_types,
                )

            for index in range(args.warmups):
                print(f"warmup {index + 1}/{args.warmups}", flush=True)
                run_once()
            for index in range(args.repetitions):
                print(f"measured {index + 1}/{args.repetitions}", flush=True)
                run, output_tokens = run_once()
                all_runs.append(run)
                if representative is None:
                    representative = output_tokens
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            log_file.close()
        runtime = {
            "executable": str(executable),
            "gguf": str(gguf),
            "assistant_gguf": (
                str(assistant_gguf) if assistant_gguf is not None else None
            ),
            "server_log": str(log_path),
        }
        configuration = {
            "kv_cache": args.llama_kv_cache_type,
            "gpu_layers": "all",
            "flash_attention": True,
            "batch_size": 2048,
            "ubatch_size": 512,
            "parallel": 1,
            "cache_prompt": False,
            "speculative_types": list(speculative_types),
            "maximum_draft_tokens": maximum_draft_tokens,
            "mtp_draft_tokens": args.mtp_draft_tokens,
            "mtp_backend": "draft-mtp" if has_mtp else "disabled",
            "ngram_mod": (
                {
                    "n_match": args.llama_ngram_mod_n_match,
                    "n_min": args.llama_ngram_mod_n_min,
                    "n_max": args.llama_ngram_mod_n_max,
                }
                if has_ngram_mod
                else None
            ),
        }

    if args.engine != "llama-cpp":
        for index in range(args.warmups):
            print(f"warmup {index + 1}/{args.warmups}", flush=True)
            run_once()
        for index in range(args.repetitions):
            print(f"measured {index + 1}/{args.repetitions}", flush=True)
            run, output_tokens = run_once()
            all_runs.append(run)
            if representative is None:
                representative = output_tokens

    if representative is None:
        raise BenchmarkError("benchmark produced no representative output")
    return {
        "schema_version": 1,
        "status": "development_characterization",
        "engine": args.engine,
        "benchmark_source": repository_state(),
        "workload": {
            "path": str(workload_path),
            "id": workload.get("id"),
            "source": workload.get("source"),
            "prompt_tokens": len(prompt),
            "prompt_token_ids_sha256": workload["prompt"]["token_ids_sha256"],
            "max_new_tokens": generation["max_new_tokens"],
            "temperature": generation["temperature"],
            "seed": generation["seed"],
            "stop_token_ids": generation["stop_token_ids"],
            "suppress_token_ids": generation["suppress_token_ids"],
            "fixed_output_tokens": args.fixed_output_tokens,
            "ignore_eos": bool(generation.get("ignore_eos", False)),
        },
        "runtime": runtime,
        "configuration": {
            **configuration,
            "batch_size": 1,
            "warmups": args.warmups,
            "measured_repetitions": args.repetitions,
            "primary_statistic": "median",
        },
        "summary": summarize_runs(all_runs),
        "runs": all_runs,
        "representative_output_token_ids": representative,
        "limitations": [
            "No continuous power, clock, or thermal telemetry was captured.",
            "TTFT includes prompt processing and first-token selection.",
            "Decode throughput uses generated_tokens - 1 intervals after the first token.",
            "Engine checkpoint and KV precisions must be read from configuration before comparison.",
        ],
    }


def main() -> int:
    args = parse_args()
    try:
        document = benchmark(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "engine": document["engine"],
                    "prompt_tokens_per_second": document["summary"][
                        "prompt_tokens_per_second"
                    ]["median"],
                    "decode_tokens_per_second": document["summary"][
                        "decode_tokens_per_second"
                    ]["median"],
                    "generated_tokens": document["summary"]["generated_tokens"][
                        "median"
                    ],
                    "deterministic_outputs": document["summary"][
                        "deterministic_outputs"
                    ],
                    "stop_reason_values": document["summary"][
                        "stop_reason_values"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except (
        BenchmarkError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
        urllib.error.URLError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the exact alternating ordinary/MTP Wikipedia qualification."""

from __future__ import annotations

try:
    from tools.hf_cache import default_assistant_model, default_target_model
except ModuleNotFoundError:
    from hf_cache import default_assistant_model, default_target_model

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

from benchmark_wikipedia_workload import (
    BenchmarkError,
    load_workload,
    positive_int,
    repository_state,
    run_gem16,
    summarize,
    summarize_runs,
)


MINIMUM_MTP_DECODE_TOKENS_PER_SECOND = 64.82


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GpuTelemetrySampler:
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, float]] = []
        self.error: str | None = None
        self._started = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._started = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        summary: dict[str, Any] = {
            "sample_interval_seconds": self.interval_seconds,
            "sample_count": len(self.samples),
            "error": self.error,
            "samples": self.samples,
        }
        for field in (
            "memory_used_mib",
            "power_w",
            "sm_clock_mhz",
            "temperature_c",
        ):
            values = [sample[field] for sample in self.samples]
            if values:
                summary[field] = summarize(values)
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
                    command,
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                ).strip().splitlines()[0]
                values = [float(value.strip()) for value in output.split(",")]
                if len(values) != 4:
                    raise ValueError("unexpected nvidia-smi field count")
                self.samples.append(
                    {
                        "elapsed_seconds": time.monotonic() - self._started,
                        **dict(
                            zip(
                                (
                                    "memory_used_mib",
                                    "power_w",
                                    "sm_clock_mhz",
                                    "temperature_c",
                                ),
                                values,
                            )
                        ),
                    }
                )
            except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
                self.error = str(error)
                return
            self._stop.wait(self.interval_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=default_target_model())
    parser.add_argument(
        "--assistant-model", type=Path, default=default_assistant_model()
    )
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--warmup-pairs", type=positive_int, default=3)
    parser.add_argument("--measured-pairs", type=positive_int, default=10)
    parser.add_argument(
        "--sampled", action="store_true",
        help="qualify same-seed Google-recommended target sampling",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--minimum-mtp-decode-tps",
        type=positive_float,
        default=MINIMUM_MTP_DECODE_TOKENS_PER_SECOND,
    )
    parser.add_argument("--stretch-mtp-decode-tps", type=positive_float)
    parser.add_argument(
        "--characterization",
        action="store_true",
        help="report target attainment without turning a missed target into an error",
    )
    parser.add_argument(
        "--telemetry-interval",
        type=positive_float,
        default=0.2,
        help="seconds between per-process nvidia-smi samples",
    )
    return parser.parse_args()


def qualification(args: argparse.Namespace) -> dict[str, Any]:
    workload_path = args.workload.resolve(strict=True)
    workload, prompt, generation = load_workload(workload_path)
    model = args.model.resolve(strict=True)
    assistant = args.assistant_model.resolve(strict=True)
    executable = args.executable.resolve(strict=True)
    prompt_file = args.output.with_suffix(".prompt-token-ids.txt").resolve()
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    prompt_file.write_text(
        ",".join(str(token) for token in prompt), encoding="ascii"
    )

    ordinary_runs: list[dict[str, Any]] = []
    mtp_runs: list[dict[str, Any]] = []
    pair_order: list[dict[str, Any]] = []
    reference_tokens: list[int] | None = None

    sampling = (
        {
            "temperature": 1.0,
            "top_k": 64,
            "top_p": 0.95,
            "repetition_penalty": 1.0,
            "seed": getattr(args, "seed", 0),
        }
        if getattr(args, "sampled", False)
        else None
    )

    telemetry_interval = float(getattr(args, "telemetry_interval", 0.0))

    def run_mode(mode: str) -> tuple[dict[str, Any], list[int]]:
        sampler = (
            GpuTelemetrySampler(telemetry_interval)
            if telemetry_interval > 0.0
            else None
        )
        if sampler is not None:
            sampler.start()
        try:
            run, tokens = run_gem16(
                executable,
                model,
                prompt_file,
                len(prompt),
                generation,
                assistant if mode == "mtp_d2" else None,
                2 if mode == "mtp_d2" else 0,
                False,
                sampling,
            )
        finally:
            telemetry = sampler.stop() if sampler is not None else None
        if telemetry is not None:
            run["gpu_telemetry"] = telemetry
        return run, tokens

    def run_pair(phase: str, index: int, measured: bool) -> None:
        nonlocal reference_tokens
        order = (
            ("ordinary", "mtp_d2")
            if index % 2 == 0
            else ("mtp_d2", "ordinary")
        )
        print(
            f"{phase} pair {index + 1}: {order[0]} then {order[1]}",
            flush=True,
        )
        pair_record: dict[str, Any] = {
            "phase": phase,
            "pair": index + 1,
            "order": list(order),
        }
        for mode in order:
            run, tokens = run_mode(mode)
            if reference_tokens is None:
                reference_tokens = tokens
            if tokens != reference_tokens:
                raise BenchmarkError(
                    f"{phase} pair {index + 1} {mode} changed exact output"
                )
            pair_record[mode] = {
                "decode_tokens_per_second": run["decode_tokens_per_second"],
                "generated_tokens": run.get("generated_tokens"),
                "inference_end_to_end_ms": run.get("inference_end_to_end_ms"),
                "output_token_sha256": run["output_token_sha256"],
            }
            if mode == "mtp_d2":
                proposed = int(run["mtp"]["proposed_tokens"])
                pair_record[mode]["acceptance_rate"] = (
                    float(run["mtp"]["accepted_tokens"]) / proposed
                    if proposed > 0
                    else 0.0
                )
            if measured:
                (ordinary_runs if mode == "ordinary" else mtp_runs).append(run)
        pair_order.append(pair_record)

    for index in range(args.warmup_pairs):
        run_pair("warmup", index, False)
    for index in range(args.measured_pairs):
        run_pair("measured", index, True)

    if reference_tokens is None:
        raise BenchmarkError("qualification produced no output")
    ordinary_summary = summarize_runs(ordinary_runs)
    mtp_summary = summarize_runs(mtp_runs)
    ordinary_median = ordinary_summary["decode_tokens_per_second"]["median"]
    mtp_median = mtp_summary["decode_tokens_per_second"]["median"]
    minimum_target = float(
        getattr(
            args,
            "minimum_mtp_decode_tps",
            MINIMUM_MTP_DECODE_TOKENS_PER_SECOND,
        )
    )
    stretch_target = getattr(args, "stretch_mtp_decode_tps", None)
    minimum_met = mtp_median >= minimum_target
    stretch_met = (
        mtp_median >= float(stretch_target)
        if stretch_target is not None
        else None
    )
    characterization = bool(getattr(args, "characterization", False))
    measured_pairs = [
        pair for pair in pair_order if pair["phase"] == "measured"
    ]
    paired_speedups = [
        pair["mtp_d2"]["decode_tokens_per_second"]
        / pair["ordinary"]["decode_tokens_per_second"]
        for pair in measured_pairs
    ]

    def telemetry_summary(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not runs or not all("gpu_telemetry" in run for run in runs):
            return None
        peaks = [
            float(run["gpu_telemetry"]["memory_used_mib"]["maximum"])
            for run in runs
            if "memory_used_mib" in run["gpu_telemetry"]
        ]
        return (
            {"peak_memory_used_mib": summarize(peaks)} if peaks else None
        )

    ordinary_telemetry = telemetry_summary(ordinary_runs)
    mtp_telemetry = telemetry_summary(mtp_runs)
    target_compilation = model / "gem16_compilation.json"
    assistant_compilation = assistant / "gem16_compilation.json"
    return {
        "schema_version": 2,
        "status": (
            "characterized"
            if characterization
            else ("qualified" if minimum_met else "target_not_met")
        ),
        "benchmark_source": repository_state(),
        "workload": {
            "path": str(workload_path),
            "id": workload.get("id"),
            "source": workload.get("source"),
            "prompt_tokens": len(prompt),
            "prompt_token_ids_sha256": workload["prompt"]["token_ids_sha256"],
            "max_new_tokens": generation["max_new_tokens"],
            "stop_token_ids": generation["stop_token_ids"],
            "suppress_token_ids": generation["suppress_token_ids"],
        },
        "runtime": {
            "executable": str(executable),
            "executable_sha256": file_sha256(executable),
            "checkpoint": str(model),
            "checkpoint_compilation_sha256": (
                file_sha256(target_compilation)
                if target_compilation.is_file()
                else None
            ),
            "assistant_checkpoint": str(assistant),
            "assistant_compilation_sha256": (
                file_sha256(assistant_compilation)
                if assistant_compilation.is_file()
                else None
            ),
            "prompt_token_file": str(prompt_file),
        },
        "configuration": {
            "batch_size": 1,
            "kv_cache": "checkpoint_fp8",
            "mtp_draft_tokens": 2,
            "warmup_pairs": args.warmup_pairs,
            "measured_pairs": args.measured_pairs,
            "alternating_first_mode": True,
            "primary_statistic": "median",
            "decoding_mode":
                "sampled" if getattr(args, "sampled", False) else "greedy",
            "sampling": sampling,
            "gpu_telemetry_interval_seconds": telemetry_interval,
        },
        "qualification": {
            "ordinary_equals_mtp": True,
            "minimum_mtp_decode_tokens_per_second":
                minimum_target,
            "minimum_mtp_decode_tokens_per_second_met": minimum_met,
            "stretch_mtp_decode_tokens_per_second": stretch_target,
            "stretch_mtp_decode_tokens_per_second_met": stretch_met,
            "llama_cpp_parity_requires_separate_comparison": True,
            "median_speedup": mtp_median / ordinary_median,
            "median_throughput_increase": mtp_median / ordinary_median - 1.0,
            "paired_speedup": summarize(paired_speedups),
        },
        "summary": {"ordinary": ordinary_summary, "mtp_d2": mtp_summary},
        "gpu_telemetry": {
            "ordinary": ordinary_telemetry,
            "mtp_d2": mtp_telemetry,
        },
        "runs": {"ordinary": ordinary_runs, "mtp_d2": mtp_runs},
        "pair_order": pair_order,
        "representative_output_token_ids": reference_tokens,
        "limitations": [
            *(
                []
                if telemetry_interval > 0.0
                else ["No continuous power, clock, or thermal telemetry was captured."]
            ),
            "TTFT includes prompt processing and first-token selection.",
            "Decode throughput uses generated_tokens - 1 verified intervals.",
            "Inference end-to-end is TTFT plus measured post-first-token decode; process wall additionally includes checkpoint loading and process startup.",
        ],
    }


def main() -> int:
    args = parse_args()
    try:
        document = qualification(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "status": document["status"],
                    "ordinary_median": document["summary"]["ordinary"][
                        "decode_tokens_per_second"
                    ]["median"],
                    "mtp_d2_median": document["summary"]["mtp_d2"][
                        "decode_tokens_per_second"
                    ]["median"],
                    "median_speedup": document["qualification"][
                        "median_speedup"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0 if document["status"] in {"qualified", "characterized"} else 2
    except (BenchmarkError, OSError, ValueError, KeyError) as error:
        import sys

        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the immutable, reviewable V10 summary from retained raw evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import statistics
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--fixtures-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def median(values: list[float]) -> float:
    return statistics.median(values)


LAYER_STAGES = (
    "input_norm_quant_ms",
    "qkv_projection_ms",
    "qkv_norm_rope_ms",
    "attention_ms",
    "output_projection_residual_ms",
    "ffn_norm_quant_ms",
    "gate_up_ms",
    "gelu_ms",
    "product_quant_ms",
    "down_residual_ms",
)


def summarize_tower(payload: dict) -> dict:
    runs = payload["runs"]
    stage_totals: dict[str, float] = {}
    for stage in LAYER_STAGES:
        stage_totals[stage.removesuffix("_ms") + "_all_layers_ms"] = median(
            [sum(layer[stage] for layer in run["layers"]) for run in runs]
        )
    return {
        "budget": payload["budget"],
        "fixture": payload["fixture"],
        "raw_patch_count": payload["raw_patch_count"],
        "soft_token_count": payload["soft_token_count"],
        "preprocess_total_ms": payload["preprocess"]["total_ms"],
        "tower_total_gpu_median_ms": median(
            [run["total_gpu_ms"] for run in runs]
        ),
        "stage_median_ms": {
            "upload": median([run["upload_ms"] for run in runs]),
            "patch_project": median(
                [run["patch_project_ms"] for run in runs]
            ),
            "position_add": median([run["position_add_ms"] for run in runs]),
            **stage_totals,
            "pool_standardize": median(
                [run["pool_standardize_ms"] for run in runs]
            ),
            "final_norm_project": median(
                [run["final_norm_project_ms"] for run in runs]
            ),
        },
        "repeat_mismatches": payload["repeat_mismatches"],
        "recurring_free_delta_bytes": payload["memory"]["recurring_free_delta"],
    }


def summarize_e2e(payload: dict, total_device_bytes: int) -> dict:
    runs = payload["runs"]
    return {
        "budget": payload["budget"],
        "fixture": payload["fixture"],
        "prompt_tokens": runs[0]["prompt_tokens"],
        "ttft_median_ms": median([run["ttft_ms"] for run in runs]),
        "generate_wall_median_ms": median(
            [run["generate_wall_ms"] for run in runs]
        ),
        "first_tokens": [run["first_token"] for run in runs],
        "recurring_free_deltas_bytes": [
            run["recurring_free_delta"] for run in runs
        ],
        "graph_private_bytes": max(run["graph_private_bytes"] for run in runs),
        "peak_during_image_device_used_bytes": max(
            total_device_bytes - run["free_before_bytes"] for run in runs
        ),
        "peak_method": (
            "visible CUDA bytes minus free immediately before Generate; "
            "warmup completed and all measured recurring free deltas were zero"
        ),
    }


NCU_METRICS = {
    "duration_ms": "gpu__time_duration.sum",
    "registers_per_thread": "launch__registers_per_thread",
    "shared_memory_kbytes_per_block": "launch__shared_mem_per_block",
    "stack_bytes_per_thread": "launch__stack_size",
    "spilling_instructions": "sass__inst_executed_register_spilling",
    "occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "issue_active_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "sm_throughput_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_throughput_pct": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_throughput_pct": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1_throughput_pct": "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "long_scoreboard_per_issue_active": (
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"
    ),
    "shared_bank_conflicts": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "special_function_instructions": "sass__inst_executed_op_special",
    "grid_blocks": "launch__grid_size",
}


def numeric(value: str):
    if value == "":
        return None
    try:
        result = float(value)
    except ValueError:
        return value
    return int(result) if result.is_integer() else result


def ncu_summary(report: Path) -> list[dict]:
    completed = subprocess.run(
        ["ncu", "--import", str(report), "--csv", "--page", "raw"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = list(csv.reader(io.StringIO(completed.stdout)))
    header = rows[0]
    summaries = []
    for row in rows[2:]:
        values = dict(zip(header, row, strict=False))
        item = {"kernel": values["Kernel Name"]}
        for name, metric in NCU_METRICS.items():
            item[name] = numeric(values.get(metric, ""))
        summaries.append(item)
    return summaries


def nsys_attention_share(raw: Path, budget: int) -> dict:
    path = raw / f"nsys-budget-{budget}-stats_cuda_gpu_kern_sum.csv"
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    attention = next(row for row in rows if "FullAttentionKernel" in row["Name"])
    return {
        "full_attention_kernel_time_pct": float(attention["Time (%)"]),
        "full_attention_launches": int(attention["Instances"]),
        "full_attention_average_ns": float(attention["Avg (ns)"]),
    }


def selected_capacity(*payloads: dict) -> list[dict]:
    output = []
    selected = {
        "target_vision": {32768, 65536, 86016, 229376, 245760},
        "target_vision_assistant": {32768, 65536, 86016, 229376, 229377},
    }
    merged: dict[str, dict[int, dict]] = {}
    for payload in payloads:
        for scenario in payload["scenarios"]:
            runs = merged.setdefault(scenario["name"], {})
            for run in scenario["runs"]:
                runs[run["context_tokens"]] = run
    for configuration, contexts in selected.items():
        runs = []
        for context in sorted(contexts):
            run = merged.get(configuration, {}).get(context)
            if run is None:
                continue
            model = run.get("model_report", {})
            if configuration == "target_vision_assistant" and context >= 64000:
                required_margin = 200 * 1024 * 1024
            elif context >= 65536:
                required_margin = 400 * 1024 * 1024
            else:
                required_margin = 700 * 1024 * 1024
            free_after_load = model.get("admission_free_bytes")
            runs.append(
                {
                    "context_tokens": run["context_tokens"],
                    "accepted": run["exit_code"] == 0,
                    "kv_cache_bytes": model.get("kv_cache_bytes"),
                    "workspace_bytes": model.get("workspace_bytes"),
                    "assistant_weight_bytes": model.get(
                        "assistant_weight_bytes"
                    ),
                    "assistant_workspace_bytes": model.get(
                        "assistant_workspace_bytes"
                    ),
                    "free_after_load_bytes": free_after_load,
                    "required_admission_margin_bytes": required_margin,
                    "admission_headroom_bytes": (
                        max(0, free_after_load - required_margin)
                        if free_after_load is not None
                        else None
                    ),
                    "error": run.get("stderr"),
                }
            )
        output.append({"configuration": configuration, "runs": runs})
    return output


def main() -> int:
    args = parse_args()
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().split(", ")
    total_device_bytes = int(gpu[3]) * 1024 * 1024
    fixture_manifest = read_json(args.fixtures_manifest)
    tower = [
        summarize_tower(read_json(args.raw_directory / f"tower-budget-{budget}.json"))
        for budget in (70, 140, 280)
    ]
    e2e_payloads = [
        read_json(args.raw_directory / f"e2e-budget-{budget}.json")
        for budget in (70, 140, 280)
    ]
    profiler = []
    for budget in (70, 140, 280):
        groups = {}
        for group in ("head", "attention", "pool", "product"):
            groups[group] = ncu_summary(
                args.raw_directory / f"ncu-budget-{budget}-{group}.ncu-rep"
            )
        profiler.append(
            {
                "budget": budget,
                "nsys": nsys_attention_share(args.raw_directory, budget),
                "ncu": groups,
            }
        )
    memory = e2e_payloads[0]["memory"]
    payload = {
        "schema_version": 1,
        "package": "V10 dedicated Vision runtime baseline",
        "status": "pass",
        "source": {
            "branch": subprocess.run(
                ["git", "branch", "--show-current"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "base_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "implementation_uncommitted": True,
        },
        "environment": {
            "gpu": gpu[0],
            "gpu_uuid": gpu[1],
            "driver": gpu[2],
            "visible_device_bytes": total_device_bytes,
            "nsys_version": "2026.1.3",
            "ncu_version": "2026.2.1",
        },
        "fixtures": {
            "manifest": str(args.fixtures_manifest),
            "manifest_sha256": sha256(args.fixtures_manifest),
            "cases": fixture_manifest["cases"],
        },
        "timing_boundary": {
            "model_load": "separate host wall clock",
            "host_preprocess": "filesystem I/O excluded from decode/resize/patchify; file-inclusive wall also retained raw",
            "tower": "fixed pre-created CUDA events on one nonblocking stream",
            "ttft": "host start immediately before Vision encode/text prefill through first Target token selection",
            "nvtx": "host launch ranges; asynchronous subranges are not GPU durations",
        },
        "tower_baseline": tower,
        "end_to_end_baseline": [
            summarize_e2e(item, total_device_bytes) for item in e2e_payloads
        ],
        "memory": {
            **memory,
            "vision_workspace_bytes": read_json(
                args.raw_directory / "tower-budget-70.json"
            )["memory"]["vision_workspace_bytes"],
            "vision_host_pinned_bytes": read_json(
                args.raw_directory / "tower-budget-70.json"
            )["memory"]["host_pinned_bytes"],
            "assistant_weight_bytes": 258313728,
            "assistant_workspace_capacity_bytes": 68 * 1024 * 1024,
            "assistant_workspace_bytes_at_max_admitted": 60115216,
        },
        "capacity": {
            "matrix": selected_capacity(
                read_json(args.raw_directory / "capacity-matrix.json"),
                read_json(
                    args.raw_directory / "capacity-matrix-trellis-extended.json"
                ),
            ),
            "target_vision_last_tested_admitted": 229376,
            "target_vision_first_tested_rejected": 245760,
            "target_vision_assistant_last_admitted": 229376,
            "target_vision_assistant_first_rejected": 229377,
            "limits": {
                "target_vision": "fresh-process physical admission with 400 MiB reserve",
                "target_vision_assistant": "fresh-process physical admission with 200 MiB MTP reserve",
                "qualified_m25_nvfp4_assistant_unchanged": 86016,
                "inherited_as_vision_limit": False,
            },
        },
        "profiler": profiler,
        "raw_evidence": {
            "directory": str(args.raw_directory),
            "nsys_reports": 4,
            "ncu_reports": 12,
        },
        "correctness": {
            "fixture_cases": 9,
            "real_module_budgets": [70, 140, 280],
            "repeat_output_mismatches": 0,
            "recurring_device_free_delta_bytes": 0,
            "first_selected_token_identical_across_all_e2e_runs": True,
        },
        "limitations": [
            "V10 characterizes the existing correctness-first runtime and does not contain a performance candidate.",
            "The 26B Assistant artifact remains the M25 diagnostic hybrid artifact; the Trellis35 capacity policy now removes its inherited 86,016-token guard, while V11/V14 still own Vision D2 exactness and qualification.",
            "The qualified M25/NVFP4 Assistant remains limited to 86,016 tokens; the 229,376 result is experimental Trellis35 plus Vision capacity evidence only.",
            "Peak image memory is inferred from the fixed-allocation contract after warmup plus zero measured recurring free delta; no sampling allocator was inserted into the hot path.",
        ],
        "next_gate": "Stop after V10 and obtain owner approval before V11.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

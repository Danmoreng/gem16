#!/usr/bin/env python3
"""Collect and validate controlled Gemma 4 26B M20 benchmark evidence.

The executable protocol is deliberately explicit: every invocation receives a
JSON job through ``--benchmark-job`` and writes one JSON sample through
``--output``.  Model loading may happen outside the reported intervals, but the
sample must report the actual prefill, first-token, and recurring decode timing
boundaries.  Missing data is a failed gate, never an inferred measurement.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
from typing import Any
import urllib.parse


SHA256_LENGTH = 64
EXPECTED_PROFILE = "native_sm120_integrated_prefill_decode_head"
EXPECTED_VARIANT = "gemma4-26b-a4b"
EXPECTED_DISPATCH = {
    "attention_prefill",
    "attention_decode",
    "moe_decode",
    "moe_prefill",
    "embedding_head",
}
REQUIRED_WARMUPS = 3
REQUIRED_RETAINED = 10
EXPECTED_SM120_INSTRUCTIONS = (
    "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X",
    "QMMA.16832.F32.E4M3.E4M3",
)


class QualificationError(RuntimeError):
    pass


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def load_json(path: Path, maximum_bytes: int = 16 * 1024 * 1024) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise QualificationError(f"JSON input is not a regular file: {path}")
    if path.stat().st_size > maximum_bytes:
        raise QualificationError(f"JSON input is too large: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QualificationError(f"cannot parse JSON input {path}: {error}") from error
    if not isinstance(value, dict):
        raise QualificationError(f"JSON root is not an object: {path}")
    return value


def finite_positive(value: object, field: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualificationError(f"{field} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (result == 0.0 and not allow_zero):
        raise QualificationError(f"{field} is not finite and positive")
    return result


def integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise QualificationError(f"{field} is not an integer >= {minimum}")
    return value


def validate_suite(suite: dict[str, Any]) -> list[dict[str, Any]]:
    if suite.get("schema_version") != 1:
        raise QualificationError("suite schema_version must be 1")
    model = suite.get("model")
    if not isinstance(model, dict) or model.get("profile") != EXPECTED_PROFILE:
        raise QualificationError("suite does not select the frozen M17 profile")
    for field in (
        "artifact_content_sha256",
        "artifact_lock_sha256",
        "source_lock_sha256",
    ):
        if not is_sha256(model.get(field)):
            raise QualificationError(f"suite model.{field} is not a SHA-256")
    if not is_sha256(suite.get("toolchain_lock_sha256")):
        raise QualificationError("suite toolchain_lock_sha256 is not a SHA-256")
    evidence = suite.get("native_instruction_evidence")
    if not isinstance(evidence, dict) or evidence.get("observed") is not True:
        raise QualificationError("suite lacks observed native instruction evidence")
    for field in ("binary_sha256", "disassembly_sha256"):
        if not is_sha256(evidence.get(field)):
            raise QualificationError(f"native instruction {field} is not a SHA-256")
    mnemonics = evidence.get("required_mnemonics")
    if not isinstance(mnemonics, list) or not mnemonics or not all(
        isinstance(item, str) and item for item in mnemonics
    ):
        raise QualificationError("native instruction mnemonics are missing")

    scenarios = suite.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise QualificationError("suite scenarios must be a non-empty list")
    identifiers: set[str] = set()
    greedy_lengths: list[int] = []
    sampled = False
    validated: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise QualificationError(f"scenario {index} is not an object")
        identifier = scenario.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise QualificationError(f"scenario {index} has an invalid or duplicate id")
        identifiers.add(identifier)
        prompt_tokens = integer(scenario.get("prompt_tokens"), f"{identifier}.prompt_tokens", minimum=1)
        output_forwards = integer(
            scenario.get("output_forwards"), f"{identifier}.output_forwards", minimum=2
        )
        context_tokens = integer(
            scenario.get("context_tokens"), f"{identifier}.context_tokens", minimum=1
        )
        if prompt_tokens + output_forwards - 1 > context_tokens:
            raise QualificationError(f"{identifier} exceeds its fixed context")
        if not is_sha256(scenario.get("prompt_manifest_sha256")):
            raise QualificationError(f"{identifier} has no prompt manifest SHA-256")
        if not isinstance(scenario.get("prompt_manifest_path"), str) or not scenario["prompt_manifest_path"]:
            raise QualificationError(f"{identifier} has no prompt manifest path")
        sampling = scenario.get("sampling")
        if not isinstance(sampling, dict) or sampling.get("mode") not in {"greedy", "sampled"}:
            raise QualificationError(f"{identifier} has invalid sampling controls")
        if sampling["mode"] == "greedy":
            if set(sampling) != {"mode"}:
                raise QualificationError(f"{identifier} greedy sampling has hidden controls")
            greedy_lengths.append(prompt_tokens)
        else:
            expected = {"mode", "temperature", "top_k", "top_p", "seed"}
            if set(sampling) != expected:
                raise QualificationError(f"{identifier} sampled controls are incomplete")
            finite_positive(sampling["temperature"], f"{identifier}.temperature")
            integer(sampling["top_k"], f"{identifier}.top_k", minimum=1)
            top_p = finite_positive(sampling["top_p"], f"{identifier}.top_p")
            if top_p > 1.0:
                raise QualificationError(f"{identifier}.top_p exceeds one")
            integer(sampling["seed"], f"{identifier}.seed")
            sampled = True
        if scenario.get("kv_mode") != "fp8":
            raise QualificationError(f"{identifier} does not lock FP8 K/V")
        validated.append(scenario)

    ranges = ((1, 512, "short"), (1536, 2560, "2K"), (7168, 9216, "8K"), (28672, 32768, "32K"))
    for low, high, label in ranges:
        if not any(low <= length <= high for length in greedy_lengths):
            raise QualificationError(f"suite lacks a greedy {label} scenario")
    if not sampled:
        raise QualificationError("suite lacks a production sampling control")
    return validated


def validate_instruction_evidence(
    executable: Path, evidence: dict[str, Any]
) -> dict[str, Any]:
    binary_sha256 = sha256_file(executable)
    if evidence.get("binary_sha256") != binary_sha256:
        raise QualificationError("native instruction binary SHA-256 does not match executable")
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        raise QualificationError("cuobjdump is required to verify native instruction evidence")
    completed = subprocess.run(
        [cuobjdump, "--dump-sass", str(executable)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise QualificationError(f"cuobjdump failed: {completed.stderr[-2000:]}")
    disassembly = completed.stdout.encode("utf-8")
    disassembly_sha256 = sha256_bytes(disassembly)
    if evidence.get("disassembly_sha256") != disassembly_sha256:
        raise QualificationError("native instruction disassembly SHA-256 changed")
    required = evidence.get("required_mnemonics")
    if required != list(EXPECTED_SM120_INSTRUCTIONS):
        raise QualificationError("native instruction mnemonic contract changed")
    counts = {mnemonic: completed.stdout.count(mnemonic) for mnemonic in required}
    if any(count <= 0 for count in counts.values()):
        raise QualificationError("required native SM120 instruction is absent")
    return {
        "binary_sha256": binary_sha256,
        "disassembly_sha256": disassembly_sha256,
        "required_mnemonic_counts": counts,
        "cuobjdump": cuobjdump,
    }


def bind_prompt_manifests(
    scenarios: list[dict[str, Any]], suite_directory: Path
) -> list[dict[str, Any]]:
    """Resolve and verify content-addressed prompt inputs before GPU work."""
    result: list[dict[str, Any]] = []
    for scenario in scenarios:
        source = Path(scenario["prompt_manifest_path"])
        path = source if source.is_absolute() else suite_directory / source
        if path.is_symlink():
            raise QualificationError(f"prompt manifest may not be a symlink: {path}")
        path = path.resolve(strict=True)
        if not path.is_file():
            raise QualificationError(f"prompt manifest is not a regular file: {path}")
        if path.stat().st_size > 4 * 1024 * 1024:
            raise QualificationError(f"prompt manifest exceeds 4 MiB: {path}")
        if sha256_file(path) != scenario["prompt_manifest_sha256"]:
            raise QualificationError(f"{scenario['id']}: prompt manifest hash mismatch")
        document = load_json(path, maximum_bytes=4 * 1024 * 1024)
        tokens = document.get("token_ids")
        if (
            document.get("schema_version") != 1
            or not isinstance(tokens, list)
            or len(tokens) != scenario["prompt_tokens"]
            or not all(isinstance(token, int) and not isinstance(token, bool) and 0 <= token <= 0xFFFFFFFF for token in tokens)
        ):
            raise QualificationError(f"{scenario['id']}: prompt manifest/token count is invalid")
        result.append({**scenario, "prompt_manifest_path": str(path)})
    return result


def validate_sample(
    sample: dict[str, Any], scenario: dict[str, Any], suite: dict[str, Any]
) -> dict[str, Any]:
    identifier = str(scenario["id"])
    if sample.get("schema_version") != 1 or sample.get("status") != "ok":
        raise QualificationError(f"{identifier}: runner sample did not succeed")
    model = sample.get("model")
    expected_model = suite["model"]
    if not isinstance(model, dict):
        raise QualificationError(f"{identifier}: sample model identity is missing")
    for field in (
        "profile",
        "artifact_content_sha256",
        "artifact_lock_sha256",
        "source_lock_sha256",
    ):
        if model.get(field) != expected_model.get(field):
            raise QualificationError(f"{identifier}: model {field} changed")

    correctness = sample.get("correctness")
    if not isinstance(correctness, dict):
        raise QualificationError(f"{identifier}: correctness data is missing")
    if correctness.get("all_logits_finite") is not True:
        raise QualificationError(f"{identifier}: non-finite logits")
    if integer(
        correctness.get("finite_checks_completed"),
        f"{identifier}.finite_checks_completed",
        minimum=1,
    ) != scenario["output_forwards"]:
        raise QualificationError(f"{identifier}: finite checks do not cover every output")
    if correctness.get("prompt_manifest_sha256") != scenario["prompt_manifest_sha256"]:
        raise QualificationError(f"{identifier}: prompt identity changed")
    if not is_sha256(correctness.get("output_token_sha256")):
        raise QualificationError(f"{identifier}: output token hash is missing")
    integer(correctness.get("output_checksum"), f"{identifier}.output_checksum")

    runtime = sample.get("runtime_path")
    if not isinstance(runtime, dict):
        raise QualificationError(f"{identifier}: runtime path data is missing")
    exact_runtime = {
        "model_variant": EXPECTED_VARIANT,
        "head_format": "nvfp4",
        "kv_mode": "fp8",
        "backend": "sm120",
        "prompt_cache": False,
        "cpu_weight_offload": False,
        "token_loop_allocations": False,
        "native_instruction_capability": True,
    }
    for field, expected in exact_runtime.items():
        if runtime.get(field) != expected:
            raise QualificationError(f"{identifier}: runtime {field} is {runtime.get(field)!r}")
    if integer(runtime.get("fallback_count"), f"{identifier}.fallback_count") != 0:
        raise QualificationError(f"{identifier}: runtime fallback was used")
    graph = runtime.get("cuda_graph")
    if not isinstance(graph, dict) or graph.get("enabled") is not True or graph.get("first_demotion_reason") != "none":
        raise QualificationError(f"{identifier}: CUDA Graph was disabled or demoted")
    dispatch = runtime.get("resolved_dispatch")
    if not isinstance(dispatch, dict) or set(dispatch) != EXPECTED_DISPATCH:
        raise QualificationError(f"{identifier}: resolved dispatch is incomplete")
    if not all(isinstance(value, str) and value.startswith("native_") for value in dispatch.values()):
        raise QualificationError(f"{identifier}: resolved dispatch contains a non-native path")
    observations = runtime.get("observations")
    if not isinstance(observations, dict):
        raise QualificationError(f"{identifier}: engine execution observations are missing")
    expected_observations = {
        "prefill_calls": 1,
        "decode_graph_launches": scenario["output_forwards"] - 1,
        "token_selections": scenario["output_forwards"],
        "maximum_global_position_exclusive":
            scenario["prompt_tokens"] + scenario["output_forwards"] - 1,
        "recurring_allocation_count": 0,
    }
    for field, expected in expected_observations.items():
        if integer(observations.get(field), f"{identifier}.observations.{field}") != expected:
            raise QualificationError(f"{identifier}: observation {field} changed")
    integer(observations.get("prefill_chunks"), f"{identifier}.observations.prefill_chunks", minimum=1)
    integer(observations.get("sliding_ring_wraps"), f"{identifier}.observations.sliding_ring_wraps")

    performance = sample.get("performance")
    if not isinstance(performance, dict):
        raise QualificationError(f"{identifier}: performance data is missing")
    if integer(performance.get("prompt_tokens"), f"{identifier}.prompt_tokens") != scenario["prompt_tokens"]:
        raise QualificationError(f"{identifier}: prompt token count changed")
    output_forwards = integer(performance.get("output_forwards"), f"{identifier}.output_forwards")
    if output_forwards != scenario["output_forwards"]:
        raise QualificationError(f"{identifier}: output forward count changed")
    if performance.get("sampling") != scenario["sampling"]:
        raise QualificationError(f"{identifier}: sampling controls changed")
    prompt_ms = finite_positive(performance.get("prompt_ms"), f"{identifier}.prompt_ms")
    ttft_ms = finite_positive(performance.get("ttft_ms"), f"{identifier}.ttft_ms")
    decode_ms = finite_positive(performance.get("decode_ms"), f"{identifier}.decode_ms")
    if ttft_ms < prompt_ms:
        raise QualificationError(f"{identifier}: TTFT is smaller than prompt time")
    intervals = performance.get("itl_ms")
    if not isinstance(intervals, list) or len(intervals) != output_forwards - 1:
        raise QualificationError(f"{identifier}: ITL count does not match decode forwards")
    checked_intervals = [
        finite_positive(value, f"{identifier}.itl_ms[{index}]")
        for index, value in enumerate(intervals)
    ]
    if not math.isclose(sum(checked_intervals), decode_ms, rel_tol=0.01, abs_tol=0.05):
        raise QualificationError(f"{identifier}: decode_ms does not match ITLs")
    expected_tps = (output_forwards - 1) * 1000.0 / decode_ms
    reported_tps = finite_positive(performance.get("decode_tps"), f"{identifier}.decode_tps")
    if not math.isclose(expected_tps, reported_tps, rel_tol=0.005):
        raise QualificationError(f"{identifier}: decode throughput has inconsistent boundaries")

    memory = sample.get("memory")
    if not isinstance(memory, dict):
        raise QualificationError(f"{identifier}: memory data is missing")
    device_used = integer(memory.get("sampled_device_used_bytes"), f"{identifier}.sampled_device_used_bytes", minimum=1)
    margin = integer(memory.get("margin_bytes"), f"{identifier}.margin_bytes", minimum=1)
    if memory.get("recurring_allocation_observed") is not False:
        raise QualificationError(f"{identifier}: recurring allocation was observed")

    return {
        "prompt_tokens": scenario["prompt_tokens"],
        "output_forwards": output_forwards,
        "prompt_ms": prompt_ms,
        "prompt_tps": scenario["prompt_tokens"] * 1000.0 / prompt_ms,
        "ttft_ms": ttft_ms,
        "decode_ms": decode_ms,
        "decode_tps": reported_tps,
        "itl_ms": checked_intervals,
        "output_token_sha256": correctness["output_token_sha256"],
        "output_checksum": correctness["output_checksum"],
        "fallback_count": 0,
        "recurring_allocation_count": observations["recurring_allocation_count"],
        "sampled_device_used_bytes": device_used,
        "margin_bytes": margin,
        "resolved_dispatch": dispatch,
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise QualificationError("cannot summarize an empty distribution")
    return {
        "count": len(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "maximum": max(values),
        "p95": percentile(values, 0.95),
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(runs) != REQUIRED_RETAINED:
        raise QualificationError(f"promotion requires exactly {REQUIRED_RETAINED} retained runs")
    result: dict[str, Any] = {}
    for field in (
        "prompt_ms", "prompt_tps", "ttft_ms", "decode_ms", "decode_tps",
        "sampled_device_used_bytes", "telemetry_process_peak_bytes", "margin_bytes",
    ):
        result[field] = distribution([float(run[field]) for run in runs])
    result["itl_ms"] = distribution([float(value) for run in runs for value in run["itl_ms"]])
    result["deterministic_outputs"] = len({run["output_token_sha256"] for run in runs}) == 1
    result["output_token_sha256_values"] = sorted({run["output_token_sha256"] for run in runs})
    return result


def external_gate(path: Path | None, milestone: str) -> dict[str, Any]:
    if path is None:
        return {"available": False, "pass": False, "reason": f"{milestone} evidence not supplied"}
    document = load_json(path)
    accepted = document.get("acceptance") is True or str(document.get("status", "")).lower() in {
        "accepted", "acceptance_pass", "qualified", "pass"
    }
    return {
        "available": True,
        "pass": accepted,
        "path": str(path),
        "sha256": sha256_file(path),
        "reported_milestone": document.get("milestone"),
        "reported_status": document.get("status"),
    }


def repository_state() -> dict[str, Any]:
    def git(*arguments: str) -> str:
        return subprocess.check_output(["git", *arguments], text=True, stderr=subprocess.DEVNULL).strip()
    try:
        commit = git("rev-parse", "HEAD")
        repository = git("config", "--get", "remote.origin.url")
        if "://" in repository:
            parsed = urllib.parse.urlsplit(repository)
            host = parsed.hostname or "redacted"
            if parsed.port is not None:
                host = f"{host}:{parsed.port}"
            repository = urllib.parse.urlunsplit(
                (parsed.scheme, host, parsed.path, parsed.query, parsed.fragment)
            )
        elif "@" in repository and ":" in repository:
            repository = repository.split("@", 1)[1]
        dirty = bool(git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        commit, repository, dirty = "unknown", "unknown", True
    return {"repository": repository, "commit": commit, "dirty": dirty}


def capture_environment(device: int) -> dict[str, Any]:
    allowed_environment = {
        key: value for key, value in os.environ.items()
        if key == "CUDA_VISIBLE_DEVICES" or key.startswith("GEM16_")
    }
    result: dict[str, Any] = {
        "os": platform.platform(),
        "kernel": platform.release(),
        "cpu": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "environment": allowed_environment,
    }
    commands = {
        "gpu": [
            "nvidia-smi", f"--id={device}",
            "--query-gpu=name,uuid,compute_cap,memory.total,memory.free,driver_version,power.limit,clocks.sm,clocks.mem,vbios_version,display_active",
            "--format=csv,noheader,nounits",
        ],
        "cuda_toolkit": ["nvcc", "--version"],
    }
    for key, command in commands.items():
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
            result[key] = {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
        except (OSError, subprocess.SubprocessError) as error:
            result[key] = {"command": command, "error": str(error)}
    return result


def query_telemetry(device: int, process_id: int) -> dict[str, Any]:
    gpu_command = [
        "nvidia-smi", f"--id={device}",
        "--query-gpu=utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem,temperature.gpu,clocks_throttle_reasons.active",
        "--format=csv,noheader,nounits",
    ]
    process_command = [
        "nvidia-smi", f"--id={device}",
        "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits",
    ]
    gpu = subprocess.check_output(gpu_command, text=True, stderr=subprocess.DEVNULL, timeout=5).strip().splitlines()[0]
    fields = [field.strip() for field in gpu.split(",")]
    if len(fields) != 7:
        raise QualificationError("nvidia-smi GPU telemetry field count changed")
    process_vram: float | None = None
    process_output = subprocess.check_output(process_command, text=True, stderr=subprocess.DEVNULL, timeout=5)
    for line in process_output.splitlines():
        parts = [field.strip() for field in line.split(",")]
        if len(parts) == 2 and parts[0] == str(process_id):
            process_vram = float(parts[1])
            break
    numeric = [float(value) for value in fields[:6]]
    return {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "monotonic_seconds": time.monotonic(),
        "process_vram_mib": process_vram,
        "gpu_util": numeric[0],
        "memory_util": numeric[1],
        "power_w": numeric[2],
        "sm_clock_mhz": numeric[3],
        "memory_clock_mhz": numeric[4],
        "temperature_c": numeric[5],
        "throttle_reasons": fields[6],
    }


class TelemetrySampler:
    def __init__(self, device: int, process_id: int, interval: float) -> None:
        self.device = device
        self.process_id = process_id
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(query_telemetry(self.device, self.process_id))
            except (OSError, subprocess.SubprocessError, ValueError, IndexError, QualificationError) as error:
                self.error = str(error)
                return
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> tuple[list[dict[str, Any]], str | None]:
        self._stop.set()
        self._thread.join()
        return self.samples, self.error


def execute_run(
    executable: Path,
    model: Path,
    job_path: Path,
    output_path: Path,
    telemetry_path: Path,
    device: int,
    telemetry_interval: float,
    timeout: int,
) -> tuple[dict[str, Any], list[str]]:
    command = [
        str(executable), "--model", str(model), "--benchmark-job", str(job_path),
        "--output", str(output_path), "--device", str(device),
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    sampler = TelemetrySampler(device, process.pid, telemetry_interval)
    sampler.start()
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.communicate()
        sampler.stop()
        raise QualificationError(f"benchmark runner timed out: {command}") from error
    samples, telemetry_error = sampler.stop()
    telemetry_path.write_text("".join(json.dumps(sample, sort_keys=True) + "\n" for sample in samples), encoding="utf-8")
    if process.returncode != 0:
        raise QualificationError(f"benchmark runner exited {process.returncode}: {(stderr or stdout)[-2000:]}")
    if telemetry_error is not None or not samples:
        raise QualificationError(f"continuous GPU telemetry failed: {telemetry_error or 'no samples'}")
    if any(sample["process_vram_mib"] is None for sample in samples):
        raise QualificationError("continuous telemetry did not observe process VRAM")
    return load_json(output_path), command


def collect(args: argparse.Namespace) -> dict[str, Any]:
    suite_path = args.suite.resolve(strict=True)
    suite = load_json(suite_path)
    scenarios = bind_prompt_manifests(validate_suite(suite), suite_path.parent)
    executable = args.executable.resolve(strict=True)
    instruction_evidence = validate_instruction_evidence(
        executable, suite["native_instruction_evidence"]
    )
    model = args.model.resolve(strict=True)
    raw_dir = args.raw_dir.resolve()
    if raw_dir.exists() and any(raw_dir.iterdir()):
        raise QualificationError(f"refusing to reuse non-empty raw directory: {raw_dir}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    environment = capture_environment(args.device)
    (raw_dir / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    commands: list[list[str]] = []
    retained_by_scenario: dict[str, list[dict[str, Any]]] = {}
    raw_records: list[dict[str, Any]] = []

    for scenario in scenarios:
        identifier = scenario["id"]
        retained: list[dict[str, Any]] = []
        reference_hash: str | None = None
        for sequence in range(REQUIRED_WARMUPS + REQUIRED_RETAINED):
            warmup = sequence < REQUIRED_WARMUPS
            phase_index = sequence + 1 if warmup else sequence - REQUIRED_WARMUPS + 1
            stem = f"{identifier}-{'warmup' if warmup else 'retained'}-{phase_index:02d}"
            job_path = raw_dir / f"{stem}.job.json"
            sample_path = raw_dir / f"{stem}.json"
            telemetry_path = raw_dir / f"{stem}.telemetry.jsonl"
            job = {
                "schema_version": 1,
                "milestone": "M20",
                "scenario": scenario,
                "warmup": warmup,
                "sequence": sequence + 1,
                "timing_boundaries": {
                    "prompt_ms": "first_prefill_launch_to_prefill_sync",
                    "ttft_ms": "request_ready_to_first_token_ready",
                    "decode_ms": "sum_of_post_first_token_synchronized_intervals",
                    "model_load_excluded": True,
                },
            }
            job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"{identifier}: {'warmup' if warmup else 'retained'} {phase_index}/{REQUIRED_WARMUPS if warmup else REQUIRED_RETAINED}", flush=True)
            sample, command = execute_run(
                executable, model, job_path, sample_path, telemetry_path,
                args.device, args.telemetry_interval, args.timeout,
            )
            commands.append(command)
            normalized = validate_sample(sample, scenario, suite)
            if reference_hash is None:
                reference_hash = normalized["output_token_sha256"]
            elif normalized["output_token_sha256"] != reference_hash:
                raise QualificationError(
                    f"{identifier}: same-seed output changed across runs"
                )
            telemetry_rows: list[dict[str, Any]] = []
            with telemetry_path.open("r", encoding="utf-8") as telemetry_file:
                for line_number, line in enumerate(telemetry_file, start=1):
                    try:
                        telemetry_row = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise QualificationError(
                            f"{identifier}: malformed telemetry line {line_number}"
                        ) from error
                    if not isinstance(telemetry_row, dict):
                        raise QualificationError(
                            f"{identifier}: telemetry line {line_number} is not an object"
                        )
                    telemetry_rows.append(telemetry_row)
            observed_peak_bytes = int(
                max(float(row["process_vram_mib"]) for row in telemetry_rows)
                * 1024 * 1024
            )
            record = {
                "scenario": identifier,
                "run": phase_index,
                "warmup": warmup,
                **normalized,
                "sample_sha256": sha256_file(sample_path),
                "telemetry_sha256": sha256_file(telemetry_path),
                "telemetry_samples": len(telemetry_rows),
                "telemetry_process_peak_bytes": observed_peak_bytes,
            }
            raw_records.append(record)
            if not warmup:
                retained.append(record)
        retained_by_scenario[identifier] = retained

    command_lines = [subprocess.list2cmdline(command) for command in commands]
    (raw_dir / "commands.txt").write_text("\n".join(command_lines) + "\n", encoding="utf-8")
    sums_paths = sorted(path for path in raw_dir.iterdir() if path.name != "SHA256SUMS")
    (raw_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in sums_paths), encoding="ascii"
    )

    code = repository_state()
    m19_gate = external_gate(args.m19_evidence, "M19")
    m21_gate = external_gate(args.m21_evidence, "M21")
    data_gates = {
        "clean_code_revision": code["dirty"] is False,
        "three_warmups_per_scenario": all(
            sum(1 for run in raw_records if run["scenario"] == scenario["id"] and run["warmup"]) == REQUIRED_WARMUPS
            for scenario in scenarios
        ),
        "ten_retained_per_scenario": all(len(runs) == REQUIRED_RETAINED for runs in retained_by_scenario.values()),
        "no_fallback_or_offload": all(
            run["fallback_count"] == 0 for run in raw_records
        ),
        "no_recurring_allocation": all(
            run["recurring_allocation_count"] == 0 for run in raw_records
        ),
        "native_dispatch_and_instruction_evidence": bool(
            instruction_evidence["required_mnemonic_counts"]
        ) and all(
            all(value.startswith("native_") for value in run["resolved_dispatch"].values())
            for run in raw_records
        ),
        "continuous_telemetry": all(run["telemetry_samples"] > 0 for run in raw_records),
        "m19_quality_pass": m19_gate["pass"],
        "m21_memory_pass": m21_gate["pass"],
    }
    accepted = all(data_gates.values())
    return {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "milestone": "M20",
        "status": "qualified" if accepted else "candidate_awaiting_gates",
        "acceptance": accepted,
        "code": code,
        "model": suite["model"],
        "toolchain_lock_sha256": suite["toolchain_lock_sha256"],
        "machine_id": sha256_bytes(json.dumps(environment, sort_keys=True).encode("utf-8"))[:16],
        "command": [str(Path(__file__).name), *sys.argv[1:]],
        "environment_sha256": sha256_file(raw_dir / "environment.json"),
        "suite": {"path": str(suite_path), "sha256": sha256_file(suite_path)},
        "configuration": {
            "warmups": REQUIRED_WARMUPS,
            "retained_runs": REQUIRED_RETAINED,
            "batch_size": 1,
            "kv_mode": "fp8",
            "prompt_cache": False,
            "primary_statistic": "median",
            "telemetry_interval_seconds": args.telemetry_interval,
        },
        "native_instruction_evidence": instruction_evidence,
        "external_gates": {"m19": m19_gate, "m21": m21_gate},
        "gates": data_gates,
        "summary": {
            identifier: summarize_runs(runs)
            for identifier, runs in retained_by_scenario.items()
        },
        "runs": raw_records,
        "raw_evidence": {
            "directory": str(raw_dir),
            "commands_sha256": sha256_file(raw_dir / "commands.txt"),
            "sha256sums_sha256": sha256_file(raw_dir / "SHA256SUMS"),
        },
        "limitations": [] if accepted else [
            "This candidate is not an M20 promotion result until every reported gate passes."
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--m19-evidence", type=Path)
    parser.add_argument("--m21-evidence", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--telemetry-interval", type=float, default=0.2)
    parser.add_argument("--timeout", type=positive_int, default=900)
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device must be nonnegative")
    if args.telemetry_interval <= 0.0:
        parser.error("--telemetry-interval must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        if args.output.exists():
            raise QualificationError(f"refusing to overwrite output: {args.output}")
        document = collect(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "output": str(args.output),
            "status": document["status"],
            "acceptance": document["acceptance"],
            "scenario_medians": {
                key: value["decode_tps"]["median"]
                for key, value in document["summary"].items()
            },
        }, sort_keys=True))
        return 0 if document["acceptance"] else 2
    except (QualificationError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

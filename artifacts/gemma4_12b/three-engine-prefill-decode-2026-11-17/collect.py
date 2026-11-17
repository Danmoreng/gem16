"""Collect the native, llama.cpp and vLLM run receipts into portable evidence."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
import struct


ROOT = Path(__file__).resolve().parents[3]
DEST = Path(__file__).resolve().parent
RUN = ROOT / "benchmarks/results/2026-11-17/three-engine-12b-prefill-decode"
HOME = Path.home()
PATH_REPLACEMENTS = (
    (str(RUN), "$RUN"),
    (str(ROOT), "$ROOT"),
    (str(HOME / ".cache/huggingface"), "$HF_CACHE"),
    (str(HOME / "Development/_engines"), "$CENTRAL_ENGINES"),
    ("/tmp/gem16-experiments/three-engine-native", "$NATIVE_RAM_SCRATCH"),
    ("/tmp/gem16-experiments/three-engine-d512-ab", "$D512_RAM_SCRATCH"),
    ("/tmp/gem16-experiments/three-engine-d512-e2e", "$D512_E2E_RAM_SCRATCH"),
    ("/tmp/gem16-experiments/three-engine-release-parent", "$RELEASE_PARENT_RAM_SCRATCH"),
    ("/tmp/gem16-experiments/d512-production-parent", "$D512_RELEASE_RAM_SCRATCH"),
    (str(HOME), "$HOME"),
)


def sanitize(value):
    if isinstance(value, str):
        for source, replacement in PATH_REPLACEMENTS:
            value = value.replace(source, replacement)
        value = re.sub(r"/tmp/gem16-experiments/vllm-three-engine-[a-zA-Z0-9_]+",
                       "$VLLM_RAM_SCRATCH", value)
        value = re.sub(r"/tmp/gem16-experiments/llama-phase-profiles-[a-zA-Z0-9_]+",
                       "$LLAMA_RAM_SCRATCH", value)
        return re.sub(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]+){4}", "<gpu-uuid>", value)
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, dict):
        return {
            sanitize(key): (
                "<redacted>" if re.search(r"(?:^|_)(?:api_key|access_key|token|password|authorization|secret)(?:$|_)", key, re.I)
                and key.startswith(("HF_", "VLLM_", "CUDA_", "FLASHINFER_", "TORCHINDUCTOR_"))
                else sanitize(item)
            )
            for key, item in value.items()
        }
    return value


def load(path):
    return json.loads(path.read_text())


def validate_fixture(manifest):
    fixture = manifest["fixture"]
    source = (ROOT / fixture["source"]).read_bytes()
    assert hashlib.sha256(source).hexdigest() == fixture["source_file_sha256"]
    ids = [int(part) for part in source.decode().strip().split(",")]
    assert len(ids) == 16384 and ids[0] == 2 and ids.count(2) == 1
    for length, expected in fixture["sha256_of_uint32_le_ids_by_length"].items():
        count = int(length)
        body = ids[1:] * ((count - 1) // (len(ids) - 1) + 1)
        binary = struct.pack(f"<{count}I", ids[0], *body[:count - 1])
        assert hashlib.sha256(binary).hexdigest() == expected, length


def median(record, key):
    return record.get(key, {}).get("median")


def native_cells(summary):
    for record in summary["results"]:
        plan = record["plan"]
        yield {
            "engine": "GemSixteen",
            "cohort": record["cohort"],
            "case": record["case"],
            "draft_tokens": plan["mtp_depth"],
            "context_tokens_requested": plan["context_tokens"],
            "prompt_ids": plan["prompt_tokens"],
            "cached_ids": plan["cached_tokens"],
            "prefix_ids_requested": plan["cached_tokens"],
            "fresh_ids": plan["new_prompt_tokens"],
            "samples": record["retained"],
            "first_id_ms": median(record, "ttft_ms"),
            "fresh_ids_per_second_through_first_id": median(record, "actual_fresh_prompt_ids_per_second_at_ttft"),
            "ids_per_second_after_first_id": median(record, "post_first_ids_per_second"),
            "backend_prefill_ms": median(record, "engine_prefill_ms"),
            "backend_decode_ids_per_second": median(record, "engine_post_first_ids_per_second"),
            "accepted_draft_fraction": median(record, "mtp_acceptance_fraction"),
            "verification_groups": median(record, "verification_groups"),
            "post_first_ms_per_recorded_group": median(record, "decode_ms_per_verification_group"),
            "peak_whole_device_mib": record["sampled_gpu"]["whole_request"]["whole_device_peak_mib"],
        }


def llama_cells(summary, manifest, suite):
    for record in summary:
        geometry = record["total"], record["requested_prefix"]
        case_name = next((cell["name"] for cell in manifest["cells"]
                          if (cell["prompt_ids"], cell["prefix_ids"]) == geometry), None)
        if case_name is None:
            continue
        accepted = record.get("server_draft_n_accepted", {}).get("samples", [])
        proposed = record.get("server_draft_n", {}).get("samples", [])
        ratios = [a / p for a, p in zip(accepted, proposed, strict=True) if p]
        actual_cached = record["cached"]
        actual_fresh = record["computed"]
        first = median(record, "http_ttft_ms")
        complete_first = len(record["http_ttft_ms"]["samples"]) == record["n"]
        steps = record.get("observed_verification_steps", [])
        decode = record["http_decode_tps"]["samples"]
        group_ms = (
            [127000 / rate / count for rate, count in zip(decode, steps, strict=True) if count]
            if record["complete_sse"] == record["n"] and steps else []
        )
        yield {
            "engine": "llama.cpp",
            "cohort": record["cohort"],
            "suite": suite,
            "case": case_name,
            "draft_tokens": record["draft"],
            "context_tokens_requested": record["context"],
            "prompt_ids": geometry[0],
            "cached_ids": actual_cached[0] if len(actual_cached) == 1 else actual_cached,
            "prefix_ids_requested": geometry[1],
            "fresh_ids": actual_fresh[0] if len(actual_fresh) == 1 else actual_fresh,
            "samples": record["n"],
            "complete_sse": record["complete_sse"],
            "first_id_ms": first if complete_first else None,
            "fresh_ids_per_second_through_first_id": (
                median(record, "http_prefill_tps")
                if complete_first else None
            ),
            "ids_per_second_after_first_id": (median(record, "http_decode_tps")
                                               if record["complete_sse"] == record["n"] else None),
            "backend_prefill_ms": median(record, "server_prompt_ms"),
            "backend_prefill_ids_per_second": median(record, "engine_prefill_tps"),
            "backend_decode_ids_per_second": median(record, "engine_decode_tps"),
            "accepted_draft_fraction": statistics.median(ratios) if ratios else None,
            "verification_groups": statistics.median(steps) if steps and record["draft"] else None,
            "post_first_ms_per_recorded_group": statistics.median(group_ms) if group_ms else None,
            "peak_whole_device_mib": record["peak_sampled_device_memory_mib"],
        }


def vllm_cell(case, sources, *, primary):
    attempts = list({attempt["name"]: attempt for attempt, _ in sources}.values())
    reference = attempts[0]["config"]["engine"]
    assert all(attempt["config"]["engine"] == reference for attempt in attempts)
    rows = [row for _, row in sources]
    steps = [row["mtp"]["num_spec_steps"] for row in rows]
    group_ms = [127000 / row["decode_tps"] / count
                for row, count in zip(rows, steps, strict=True)]
    peak = max((attempt["gpu"]["device_vram_peak_mib"] for attempt in attempts
                if attempt["gpu"]), default=None)
    return {
        "engine": "vLLM",
        "cohort": " + ".join(attempt["name"] for attempt in attempts),
        "case": case["name"],
        "draft_tokens": reference["spec_tokens"],
        "context_tokens_requested": reference["max_model_len"],
        "max_batched_prompt_ids": reference["max_num_batched_tokens"],
        "gpu_memory_utilization": reference["gpu_memory_utilization"],
        "prompt_ids": case["prompt_ids"],
        "cached_ids": statistics.median(row["cached_tokens"] for row in rows),
        "prefix_ids_requested": case["prefix_ids"],
        "fresh_ids": statistics.median(row["computed_prompt_tokens"] for row in rows),
        "samples": len(rows),
        "first_id_ms": statistics.median(row["ttft_ms"] for row in rows),
        "fresh_ids_per_second_through_first_id": statistics.median(
            row["computed_prompt_tokens_per_sec_stream_ttft"] for row in rows),
        "ids_per_second_after_first_id": statistics.median(row["decode_tps"] for row in rows),
        "backend_prefill_ms": statistics.median(row["engine_scheduled_to_first_ms"] for row in rows),
        "backend_decode_ids_per_second": statistics.median(row["engine_decode_tps"] for row in rows),
        "accepted_draft_fraction": statistics.median(row["mtp"]["draft_acceptance_rate"] for row in rows),
        "verification_groups": statistics.median(steps),
        "post_first_ms_per_recorded_group": statistics.median(group_ms),
        "peak_whole_device_mib": peak,
        "source_runs": [{"attempt": attempt["name"], "index": row["index"]} for attempt, row in sources],
        "primary_comparison": primary,
    }


def vllm_cells(summary, manifest):
    geometry = {cell["name"]: cell for cell in manifest["cells"]}
    attempts = {item["name"]: item for item in summary["attempts"]}
    primary_sources = {}
    for label, record in summary["main"].items():
        selected = []
        for source in record["sources"]:
            attempt = attempts[source["attempt"]]
            row, = [row for row in attempt["cells"][label]["retained"]
                    if row["index"] == source["index"]]
            selected.append((attempt, row))
        primary_sources[label] = {(attempt["name"], row["index"]) for attempt, row in selected}
        result = vllm_cell(geometry[label], selected, primary=True)
        assert result["first_id_ms"] == record["median_ttft_ms"]
        yield result
    for attempt in attempts.values():
        for label, record in attempt["cells"].items():
            own = {(attempt["name"], row["index"]) for row in record["retained"]}
            if label not in geometry or own <= primary_sources.get(label, set()):
                continue
            sources = [(attempt, row) for row in record["retained"]]
            yield vllm_cell(geometry[label], sources, primary=False)


def files_to_collect():
    for path in sorted(RUN.rglob("*")):
        if not path.is_file() or path.stat().st_size == 0:
            continue
        if path.suffix in (".json", ".jsonl", ".csv") or path.name in ("native-identity.txt",
                "device-power-active.txt", "model-sha256.txt", "gguf-sha256.txt", "exit_status.txt",
                "profile-artifact-sha256.txt", "runtime.txt", "profile-cold-16k.log", "profile-warm-130k.log",
                "cold16k-recheck-environment.txt", "pre-production-release-identities.txt"):
            yield path
        diagnostic = {
            "d512-warp-sharing", "d512-e2e", "d512-e2e-cold130k", "d512-e2e-short",
            "d512-e2e-initial-graph-growth",
        }.intersection(path.parts)
        if diagnostic and path.name in (
            "compile-receipt.txt", "run-receipt.txt", "resources.txt", "device-pre.txt", "device-post.txt",
            "memcheck.log", "racecheck.log", "synccheck.log", "pairs.log", "dispatch.h",
            "cold16k.log", "cold130k.log", "warm130k.log", "warm-near-max.log",
            "cold-32.log", "cold-512.log", "cold-4096.log",
        ):
            yield path
        if "d512-production" in path.parts and path.suffix in (".log", ".txt", ".tsv"):
            yield path
        if path.suffix in (".py", ".sh", ".cpp", ".cu") and "__pycache__" not in path.parts:
            yield path


def verify_samples(path, payload, manifest):
    fingerprints = manifest["fixture"]["sha256_of_uint32_le_ids_by_length"]
    if path.parts[0] == "llama" and path.name == "samples.jsonl":
        for row in payload:
            length = row["final"]["tokens_evaluated"]
            expected = fingerprints.get(str(length))
            if expected is not None:
                assert row["prompt_sha256_int32le"] == expected, (path, length)
            assert row["final"]["tokens_predicted"] == (1 if row["kind"] == "primer" else 128), path
    if path.match("vllm/*/raw.jsonl"):
        for row in payload:
            if row["kind"] in ("sample", "primer"):
                expected = fingerprints[str(row["prompt_tokens"])]
                assert row["prompt_ids_sha256_uint32_le"] == expected, path
                assert len(row["output_ids"]) == (1 if row["kind"] == "primer" else 128), path


def pack_files(paths, manifest):
    result = {}
    production_text_receipts = {
        "release-12b-image.json", "release-12b-audio.json", "release-26b-compact-vision.json",
    }
    for path in paths:
        relative = path.relative_to(RUN)
        text = path.read_text()
        if path.suffix == ".jsonl":
            payload = [json.loads(line) for line in text.splitlines()]
            verify_samples(relative, payload, manifest)
        elif path.suffix == ".json" and not (
            "d512-production" in relative.parts and path.name in production_text_receipts
        ):
            payload = json.loads(text)
        else:
            payload = text
        result[str(relative)] = {
            "source_sha256_before_path_redaction": hashlib.sha256(path.read_bytes()).hexdigest(),
            "content": sanitize(payload),
        }
    return result


def llama_startup_receipts():
    patterns = re.compile(
        r"offloaded \d+/\d+ layers to GPU|(?:CPU|CUDA\d+) (?:KV|model|compute) buffer size"
        r"|(?:n_ctx|n_ctx_per_seq|n_batch|n_ubatch|flash_attn)\s*="
        r"|adding speculative implementation|CUDA error|failed to (?:load|allocate)"
    )
    result = {}
    for path in sorted((RUN / "llama").rglob("server.log")):
        lines = path.read_text(errors="replace").splitlines()
        result[str(path.parent.relative_to(RUN / "llama"))] = [
            sanitize(line) for line in lines if patterns.search(line)
        ]
    return result


def vllm_runtime_receipts():
    patterns = re.compile(
        r"GPU KV cache size|Available KV cache memory|CUDA graph pool memory|kv cache group sizes"
        r"|JIT compilation during inference|Using FlashInferCutlassNvFp4|Using .*attention backend"
        r"|\b(?:warmup|retained) \d+ (?:primer|request) \d+"
        r"|No available memory|KV cache.*(?:required|available|estimated)|out of memory",
        re.IGNORECASE,
    )
    result = {}
    for path in sorted((RUN / "vllm").glob("*/*.log")):
        lines = path.read_text(errors="replace").splitlines()
        extract = [line for line in lines if patterns.search(line)]
        if extract:
            result[str(path.relative_to(RUN / "vllm"))] = sanitize(extract)
    return result


def main():
    manifest = load(RUN / "workload.json")
    validate_fixture(manifest)
    result = {"schema": 1, "manifest": manifest, "cells": []}
    diagnostics = {}
    for label, directory in (
        ("operator", "d512-warp-sharing"), ("resident", "d512-e2e"),
        ("resident_cold130k", "d512-e2e-cold130k"),
        ("resident_short", "d512-e2e-short"),
    ):
        summary = RUN / "native" / directory / "summary.json"
        if summary.exists():
            record = load(summary)
            record.pop("equality", None)
            diagnostics[label] = record
    if diagnostics:
        result["d512_warp_sharing"] = diagnostics
    native = RUN / "native/summary.json"
    if native.exists():
        result["cells"].extend(native_cells(load(native)))
    llama_prior = RUN / "llama/short-a9d43ec-summary.json"
    if llama_prior.exists():
        result["cells"].extend(llama_cells(load(llama_prior), manifest, "a9d43ec short validation"))
    campaigns = sorted(
        (RUN / "llama").glob("campaign-*/summary.json"),
        key=lambda path: path.parent.name.rsplit("-", 1)[-1],
    )
    for campaign in campaigns:
        result["cells"].extend(llama_cells(load(campaign), manifest, campaign.parent.name))
    vllm = RUN / "vllm/summary.json"
    if vllm.exists():
        result["cells"].extend(vllm_cells(load(vllm), manifest))
    for cell in result["cells"]:
        if cell["engine"] == "vLLM":
            continue
        if cell["engine"] == "GemSixteen":
            chosen_suite = cell["cohort"] == "resident-library"
        else:
            chosen_suite = bool(campaigns) and cell["suite"] == campaigns[-1].parent.name
        cell["primary_comparison"] = cell["draft_tokens"] == 3 and chosen_suite
    (DEST / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["cells"]:
        columns = list(dict.fromkeys(key for cell in result["cells"] for key in cell))
        with (DEST / "comparison.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(result["cells"])
    payload = {
        "schema": 1,
        "files": pack_files(files_to_collect(), manifest),
        "llama_startup_log_extracts": llama_startup_receipts(),
        "vllm_runtime_log_extracts": vllm_runtime_receipts(),
    }
    with (DEST / "samples-and-receipts.json.gz").open("wb") as destination:
        with gzip.GzipFile(fileobj=destination, mode="wb", filename="", mtime=0) as archive:
            archive.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode() + b"\n")


if __name__ == "__main__":
    main()

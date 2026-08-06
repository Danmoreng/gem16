#!/usr/bin/env python3
"""Capture a deterministic direct Unsloth NVFP4 reference through pinned vLLM."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--software-lock", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    parser.add_argument("--cpu-offload-gb", type=float, default=6.0)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_prompt(corpus: dict[str, Any], prompt_id: str) -> dict[str, Any]:
    records = corpus.get("records")
    if not isinstance(records, list):
        raise ValueError("corpus records must be an array")
    matches = [
        record
        for record in records
        if isinstance(record, dict) and record.get("id") == prompt_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one corpus record for {prompt_id!r}")
    token_ids = matches[0].get("input_token_ids")
    if not isinstance(token_ids, list) or not token_ids or any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0
        for token in token_ids
    ):
        raise ValueError("prompt token IDs are invalid")
    return matches[0]


def reference_version(software_lock: dict[str, Any], name: str) -> str:
    matches = [
        entry
        for entry in software_lock.get("references", [])
        if isinstance(entry, dict) and entry.get("name") == name
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("version"), str):
        raise ValueError(f"software lock has no unique version for {name!r}")
    return matches[0]["version"]


def installed_versions(software_lock: dict[str, Any]) -> dict[str, str]:
    distributions = {
        "vllm": "vllm",
        "transformers": "transformers",
        "pytorch": "torch",
        "compressed-tensors": "compressed-tensors",
        "flashinfer": "flashinfer-python",
    }
    versions: dict[str, str] = {}
    for lock_name, distribution in distributions.items():
        installed = importlib.metadata.version(distribution)
        expected = reference_version(software_lock, lock_name)
        if installed.split("+", 1)[0] != expected:
            raise RuntimeError(
                f"{distribution} version mismatch: expected {expected}, got {installed}"
            )
        versions[distribution] = installed
    return versions


def serialize_logprobs(value: Any) -> list[list[dict[str, Any]]]:
    if not isinstance(value, list):
        raise RuntimeError("vLLM did not return per-position logprobs")
    positions: list[list[dict[str, Any]]] = []
    for position in value:
        if not isinstance(position, dict):
            raise RuntimeError("vLLM returned an invalid logprob position")
        entries = []
        for token_id, candidate in position.items():
            logprob = getattr(candidate, "logprob", None)
            rank = getattr(candidate, "rank", None)
            decoded = getattr(candidate, "decoded_token", None)
            if not isinstance(token_id, int) or not isinstance(logprob, (int, float)):
                raise RuntimeError("vLLM returned an invalid logprob entry")
            entries.append(
                {
                    "token_id": token_id,
                    "logprob": float(logprob),
                    "rank": int(rank) if isinstance(rank, int) else None,
                    "decoded_token": decoded if isinstance(decoded, str) else None,
                }
            )
        entries.sort(
            key=lambda entry: (
                entry["rank"] is None,
                entry["rank"] if entry["rank"] is not None else 1 << 30,
                entry["token_id"],
            )
        )
        positions.append(entries)
    return positions


def repeat_summary(runs: list[dict[str, Any]]) -> dict[str, bool]:
    if len(runs) != 2:
        raise ValueError("exactly two reference runs are required")
    return {
        "token_ids_exact": runs[0].get("token_ids") == runs[1].get("token_ids"),
        "text_exact": runs[0].get("text") == runs[1].get("text"),
        "logprobs_exact": runs[0].get("logprobs") == runs[1].get("logprobs"),
    }


def main() -> int:
    args = parse_args()
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise ValueError("gpu-memory-utilization must be in (0, 1)")
    if args.cpu_offload_gb < 0 or args.cpu_offload_gb > 16:
        raise ValueError("cpu-offload-gb must be in [0, 16]")
    if args.max_model_len < 1 or args.max_model_len > 4096:
        raise ValueError("max-model-len must be in [1, 4096]")
    if args.max_new_tokens < 1 or args.max_new_tokens > 16:
        raise ValueError("max-new-tokens must be in [1, 16]")

    model_lock = json.loads(args.model_lock.read_text(encoding="utf-8"))
    software_lock = json.loads(args.software_lock.read_text(encoding="utf-8"))
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    prompt = find_prompt(corpus, args.prompt_id)
    versions = installed_versions(software_lock)

    import torch
    from vllm import LLM, SamplingParams

    model = args.model.resolve(strict=True)
    llm = LLM(
        model=str(model),
        tokenizer=str(model),
        trust_remote_code=False,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        max_logprobs=20,
        gpu_memory_utilization=args.gpu_memory_utilization,
        cpu_offload_gb=args.cpu_offload_gb,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        seed=0,
        limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0},
        kv_cache_dtype="bfloat16",
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        logprobs=20,
        seed=0,
    )
    def capture_once() -> dict[str, Any]:
        request = llm.generate(
            [{"prompt_token_ids": prompt["input_token_ids"]}],
            sampling,
            use_tqdm=False,
        )[0]
        completion = request.outputs[0]
        return {
            "token_ids": [int(token) for token in completion.token_ids],
            "text": completion.text,
            "finish_reason": completion.finish_reason,
            "logprobs": serialize_logprobs(completion.logprobs),
        }

    warmup = capture_once()
    runs = [capture_once(), capture_once()]
    repeat = repeat_summary(runs)
    token_ids_exact = repeat["token_ids_exact"]
    fully_exact = all(repeat.values())

    result = {
        "schema_version": 1,
        "status": (
            "diagnostic_reference_deterministic"
            if fully_exact
            else "diagnostic_reference_token_deterministic"
            if token_ids_exact
            else "diagnostic_reference_nondeterministic_tokens"
        ),
        "benchmark_qualified": False,
        "model": {
            "repository": model_lock["repository"],
            "revision": model_lock["revision"],
            "source_lock_sha256": file_sha256(args.model_lock),
            "source_format": "compressed-tensors mixed FP8 attention and NVFP4 MLP",
        },
        "software": {
            "lock_sha256": file_sha256(args.software_lock),
            "versions": versions,
            "device": torch.cuda.get_device_name(0),
            "trust_remote_code": False,
        },
        "execution": {
            "max_model_len": args.max_model_len,
            "max_num_seqs": 1,
            "max_logprobs": 20,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "cpu_offload_gb": args.cpu_offload_gb,
            "kv_cache_dtype": "bfloat16",
            "enforce_eager": True,
            "prefix_caching": False,
            "chunked_prefill": True,
            "multimodal_limits": {"image": 0, "audio": 0, "video": 0},
            "temperature": 0.0,
            "seed": 0,
            "max_new_tokens": args.max_new_tokens,
            "warmup_runs": 1,
            "retained_runs": 2,
            "client_process_max_rss_kib": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "performance_eligible": False,
        },
        "prompt": {
            "id": args.prompt_id,
            "corpus_sha256": file_sha256(args.corpus),
            "input_token_ids": prompt["input_token_ids"],
            "input_token_ids_sha256_u32le": prompt[
                "input_token_ids_sha256_u32le"
            ],
        },
        "warmup_run": warmup,
        "repeat": repeat,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        f"wrote {args.output}: token_ids={runs[0]['token_ids']} "
        f"positions={len(runs[0]['logprobs'])} repeat={result['repeat']}"
    )
    # vLLM 0.26.0 exposes shutdown on its pinned V1 EngineCore client rather
    # than on the public LLM facade. Close it explicitly so no worker survives
    # a completed offline capture.
    llm.llm_engine.engine_core.shutdown()
    return 0 if token_ids_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())

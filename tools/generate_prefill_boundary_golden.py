#!/usr/bin/env python3
"""Generate direct-token prefill-boundary fixtures with the pinned vLLM runtime."""

from __future__ import annotations

try:
    from tools.hf_cache import default_target_model
except ModuleNotFoundError:
    from hf_cache import default_target_model

import argparse
from dataclasses import asdict, dataclass
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any


TOKEN_COUNTS = (129, 257)
OUTPUT_TOKENS = 8


class GoldenError(RuntimeError):
    pass


@dataclass(frozen=True)
class LogprobEntry:
    token_id: int
    logprob: float
    rank: int | None
    decoded_token: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=default_target_model())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lock", type=Path, default=Path("models/gemma4-12b-nvfp4.lock.json")
    )
    parser.add_argument("--kv-cache-dtype", choices=("auto", "fp8"), default="fp8")
    return parser.parse_args()


def prompt_tokens(count: int) -> list[int]:
    return [1000 + ((index * 7919) % 9000) for index in range(count)]


def serialize_logprobs(logprobs: Any) -> list[list[dict[str, Any]]]:
    positions: list[list[dict[str, Any]]] = []
    for position in logprobs or []:
        entries = [
            LogprobEntry(
                token_id=int(token_id),
                logprob=float(value.logprob),
                rank=None if value.rank is None else int(value.rank),
                decoded_token=value.decoded_token,
            )
            for token_id, value in position.items()
        ]
        entries.sort(
            key=lambda item: (
                item.rank is None,
                item.rank if item.rank is not None else sys.maxsize,
                item.token_id,
            )
        )
        positions.append([asdict(entry) for entry in entries])
    return positions


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise GoldenError(f"required package is not installed: {name}") from error


def main() -> int:
    args = parse_args()
    try:
        import torch
        from vllm import LLM, SamplingParams

        if not torch.cuda.is_available():
            raise GoldenError("CUDA is not available to the reference runtime")
        model = args.model.resolve(strict=True)
        lock = json.loads(args.lock.read_text(encoding="utf-8"))
        maximum_context = max(TOKEN_COUNTS) + OUTPUT_TOKENS
        llm = LLM(
            model=str(model),
            tokenizer=str(model),
            max_model_len=maximum_context,
            max_logprobs=20,
            gpu_memory_utilization=0.90,
            cpu_offload_gb=0,
            enforce_eager=True,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            seed=0,
            limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0},
            kv_cache_dtype=args.kv_cache_dtype,
        )
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=OUTPUT_TOKENS,
            logprobs=20,
            seed=0,
            ignore_eos=True,
        )
        cases = []
        for count in TOKEN_COUNTS:
            input_ids = prompt_tokens(count)
            request = llm.generate(
                [{"prompt_token_ids": input_ids}], sampling, use_tqdm=False
            )[0]
            completion = request.outputs[0]
            cases.append(
                {
                    "prompt_tokens": count,
                    "prompt_token_ids": input_ids,
                    "output_token_ids": list(completion.token_ids),
                    "cumulative_logprob": completion.cumulative_logprob,
                    "top_logprobs": serialize_logprobs(completion.logprobs),
                }
            )
        device = torch.cuda.get_device_properties(0)
        document = {
            "schema_version": 1,
            "checkpoint": {
                "repository": lock.get("repository"),
                "revision": lock.get("revision"),
                "local_directory_name": model.name,
            },
            "reference_runtime": {
                "python": platform.python_version(),
                "vllm": package_version("vllm"),
                "torch": package_version("torch"),
                "torch_cuda": torch.version.cuda,
                "device_name": device.name,
                "device_compute_capability": list(
                    torch.cuda.get_device_capability(0)
                ),
            },
            "execution": {
                "batch_size": 1,
                "temperature": 0.0,
                "seed": 0,
                "max_model_len": maximum_context,
                "max_tokens": OUTPUT_TOKENS,
                "kv_cache_dtype": args.kv_cache_dtype,
                "enforce_eager": True,
                "enable_prefix_caching": False,
                "enable_chunked_prefill": True,
            },
            "cases": cases,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(document, sort_keys=True))
        return 0
    except (GoldenError, json.JSONDecodeError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

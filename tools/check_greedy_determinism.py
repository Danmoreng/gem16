#!/usr/bin/env python3
"""Check exact greedy token identity across fresh gem16 processes."""

from __future__ import annotations

try:
    from tools.hf_cache import default_target_model
except ModuleNotFoundError:
    from hf_cache import default_target_model

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
from typing import Any


class DeterminismError(RuntimeError):
    pass


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=default_target_model())
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--logits-directory", type=Path)
    parser.add_argument("--prompt-tokens", type=positive_int, default=512)
    parser.add_argument("--generated-tokens", type=positive_int, default=256)
    parser.add_argument("--repetitions", type=positive_int, default=5)
    parser.add_argument("--prompt-seed", type=int, default=0)
    parser.add_argument("--max-context", type=positive_int)
    parser.add_argument("--require-deterministic", action="store_true")
    return parser.parse_args()


def token_sha256(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        if token < 0 or token > 0xFFFFFFFF:
            raise DeterminismError("token ID is outside uint32")
        digest.update(struct.pack("<I", token))
    return digest.hexdigest()


def load_prompt(
    workload_path: Path | None, prompt_tokens: int, prompt_seed: int
) -> tuple[list[int], list[int], dict[str, Any]]:
    if workload_path is None:
        prompt = [
            1000 + ((prompt_seed + index * 7919) % 9000)
            for index in range(prompt_tokens)
        ]
        return prompt, [], {
            "kind": "synthetic",
            "formula": "1000+((seed+index*7919)%9000)",
            "seed": prompt_seed,
        }

    path = workload_path.resolve(strict=True)
    workload = json.loads(path.read_text(encoding="utf-8"))
    prompt_document = workload.get("prompt")
    generation = workload.get("generation")
    if not isinstance(prompt_document, dict) or not isinstance(generation, dict):
        raise DeterminismError("workload has no prompt or generation object")
    source_tokens = prompt_document.get("token_ids")
    suffix_tokens = prompt_document.get("preserved_chat_suffix_tokens")
    suppress_tokens = generation.get("suppress_token_ids", [])
    if (
        not isinstance(source_tokens, list)
        or not all(isinstance(token, int) for token in source_tokens)
        or not isinstance(suffix_tokens, int)
        or suffix_tokens < 1
        or suffix_tokens >= prompt_tokens
        or prompt_tokens > len(source_tokens)
        or not isinstance(suppress_tokens, list)
        or not all(isinstance(token, int) for token in suppress_tokens)
    ):
        raise DeterminismError("workload cannot produce the requested short prompt")
    prompt = (
        source_tokens[: prompt_tokens - suffix_tokens]
        + source_tokens[-suffix_tokens:]
    )
    return prompt, suppress_tokens, {
        "kind": "workload_prefix_with_preserved_suffix",
        "workload": str(path),
        "workload_id": workload.get("id"),
        "source_prompt_sha256": prompt_document.get("token_ids_sha256"),
        "preserved_suffix_tokens": suffix_tokens,
    }


def first_divergence(reference: list[int], candidate: list[int]) -> int | None:
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if left != right:
            return index
    if len(reference) != len(candidate):
        return min(len(reference), len(candidate))
    return None


def run_once(
    executable: Path,
    model: Path,
    prompt: list[int],
    suppress_tokens: list[int],
    generated_tokens: int,
    max_context: int,
    logits_path: Path | None,
) -> tuple[dict[str, Any], list[int]]:
    command = [
        str(executable),
        "--model",
        str(model),
        "--input-token-ids",
        ",".join(str(token) for token in prompt),
    ]
    if suppress_tokens:
        command.extend([
            "--suppress-token-ids",
            ",".join(str(token) for token in suppress_tokens),
        ])
    if logits_path is not None:
        command.extend(["--dump-logits", str(logits_path)])
    command.extend(
        [
            "--kv-cache",
            "fp8",
            "--max-tokens",
            str(generated_tokens),
            "--max-context",
            str(max_context),
            "--greedy",
        ]
    )
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise DeterminismError(
            f"gem16 exited with {completed.returncode}: {detail[-2000:]}"
        )
    result = json.loads(completed.stdout)
    tokens = result.get("output_token_ids")
    if (
        not isinstance(tokens, list)
        or len(tokens) != generated_tokens
        or not all(isinstance(token, int) for token in tokens)
    ):
        raise DeterminismError("gem16 returned malformed or short output")
    if result.get("fallbacks") != 0 or result.get("token_loop_allocations") is not False:
        raise DeterminismError("gem16 reported a fallback or token-loop allocation")
    return result, tokens


def main() -> int:
    args = parse_args()
    try:
        model = args.model.resolve(strict=True)
        executable = args.executable.resolve(strict=True)
        prompt, suppress_tokens, prompt_source = load_prompt(
            args.workload, args.prompt_tokens, args.prompt_seed
        )
        required_context = len(prompt) + args.generated_tokens - 1
        max_context = (
            required_context if args.max_context is None else args.max_context
        )
        if max_context < required_context:
            raise DeterminismError(
                "--max-context is smaller than prompt plus decode positions"
            )
        logits_directory = args.logits_directory
        if logits_directory is not None:
            logits_directory.mkdir(parents=True, exist_ok=True)
        runs: list[dict[str, Any]] = []
        reference: list[int] | None = None
        for index in range(args.repetitions):
            result, tokens = run_once(
                executable,
                model,
                prompt,
                suppress_tokens,
                args.generated_tokens,
                max_context,
                (
                    logits_directory / f"run-{index}.f32"
                    if logits_directory is not None
                    else None
                ),
            )
            if reference is None:
                reference = tokens
            divergence = first_divergence(reference, tokens)
            run = {
                "run": index,
                "output_token_sha256": token_sha256(tokens),
                "first_divergence_from_run_0": divergence,
                "first_output_token_id": tokens[0],
                "last_output_token_id": tokens[-1],
                "model_load_ms": result.get("model_load_ms"),
                "prompt_ms": result.get("prompt_ms"),
                "decode_ms": result.get("decode_ms"),
            }
            if logits_directory is not None:
                logits_path = logits_directory / f"run-{index}.f32"
                run["logits_sha256"] = hashlib.sha256(
                    logits_path.read_bytes()
                ).hexdigest()
            runs.append(run)
            print(
                f"run {index + 1}/{args.repetitions}: "
                f"{run['output_token_sha256']} divergence={divergence}",
                flush=True,
            )

        hashes = sorted({run["output_token_sha256"] for run in runs})
        deterministic = len(hashes) == 1
        document = {
            "schema_version": 1,
            "status": "correctness_gate",
            "engine": "gem16",
            "execution_scope": "fresh_process_per_run",
            "configuration": {
                "kv_cache": "checkpoint_fp8",
                "greedy": True,
                "prompt_tokens": len(prompt),
                "generated_tokens": args.generated_tokens,
                "repetitions": args.repetitions,
                "max_context": max_context,
                "normal_eos_stopping": False,
            },
            "prompt": {
                **prompt_source,
                "token_ids_sha256": token_sha256(prompt),
            },
            "deterministic_outputs": deterministic,
            "unique_output_hashes": hashes,
            "runs": runs,
            "reference_output_token_ids": reference,
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(document, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(
            json.dumps(
                {
                    "deterministic_outputs": deterministic,
                    "unique_output_hashes": len(hashes),
                    "prompt_tokens": len(prompt),
                    "generated_tokens": args.generated_tokens,
                },
                sort_keys=True,
            )
        )
        if args.require_deterministic and not deterministic:
            return 2
        return 0
    except (
        DeterminismError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

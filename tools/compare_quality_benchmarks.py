#!/usr/bin/env python3
"""Compare two gem16 quality-benchmark result files sample by sample."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any


class ComparisonError(RuntimeError):
    pass


def load(path: pathlib.Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ComparisonError(f"{path} is not a schema-version-1 quality result")
    if document.get("status") != "complete":
        raise ComparisonError(f"{path} is not a complete quality result")
    return document


def scored_samples(document: dict[str, Any]) -> dict[tuple[str, int], bool]:
    samples: dict[tuple[str, int], bool] = {}
    examples = document.get("examples")
    if not isinstance(examples, list):
        raise ComparisonError("quality result has no examples")
    for example in examples:
        identifier = example.get("id")
        rows = example.get("samples")
        if not isinstance(identifier, str) or not isinstance(rows, list):
            raise ComparisonError("quality result contains a malformed example")
        for index, sample in enumerate(rows):
            score = sample.get("score")
            if not isinstance(score, (int, float)) or score not in (0, 1, 0.0, 1.0):
                raise ComparisonError(f"{identifier} sample {index} has a non-binary score")
            key = (identifier, index)
            if key in samples:
                raise ComparisonError(f"duplicate sample {key}")
            samples[key] = bool(score)
    return samples


def exact_mcnemar_p(reference_only: int, candidate_only: int) -> float:
    discordant = reference_only + candidate_only
    if discordant == 0:
        return 1.0
    tail = min(reference_only, candidate_only)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1))
    return min(1.0, 2.0 * probability / (2**discordant))


def compare(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if reference.get("benchmark") != candidate.get("benchmark"):
        raise ComparisonError("benchmark names differ")
    reference_protocol = reference.get("protocol")
    candidate_protocol = candidate.get("protocol")
    protocol_fields = (
        "reasoning",
        "generation",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "max_tokens",
        "repeats",
    )
    mismatches = [
        field
        for field in protocol_fields
        if (reference_protocol or {}).get(field) != (candidate_protocol or {}).get(field)
    ]
    if mismatches:
        raise ComparisonError("protocol fields differ: " + ", ".join(mismatches))

    reference_scores = scored_samples(reference)
    candidate_scores = scored_samples(candidate)
    if reference_scores.keys() != candidate_scores.keys():
        missing = sorted(reference_scores.keys() - candidate_scores.keys())[:5]
        extra = sorted(candidate_scores.keys() - reference_scores.keys())[:5]
        raise ComparisonError(f"sample identities differ; missing={missing}, extra={extra}")

    both_correct = reference_only = candidate_only = both_wrong = 0
    changed = []
    for key in sorted(reference_scores):
        ref = reference_scores[key]
        cand = candidate_scores[key]
        if ref and cand:
            both_correct += 1
        elif ref:
            reference_only += 1
            changed.append({"id": key[0], "repeat": key[1], "change": "regression"})
        elif cand:
            candidate_only += 1
            changed.append({"id": key[0], "repeat": key[1], "change": "improvement"})
        else:
            both_wrong += 1

    count = len(reference_scores)
    reference_accuracy = (both_correct + reference_only) / count
    candidate_accuracy = (both_correct + candidate_only) / count
    return {
        "schema_version": 1,
        "status": "complete",
        "benchmark": reference["benchmark"],
        "reference": {
            "backend": (reference.get("endpoint") or {}).get("backend"),
            "model": (reference.get("endpoint") or {}).get("model"),
            "accuracy": reference_accuracy,
        },
        "candidate": {
            "backend": (candidate.get("endpoint") or {}).get("backend"),
            "model": (candidate.get("endpoint") or {}).get("model"),
            "accuracy": candidate_accuracy,
        },
        "sample_count": count,
        "absolute_delta": candidate_accuracy - reference_accuracy,
        "relative_retention": (
            candidate_accuracy / reference_accuracy if reference_accuracy > 0 else None
        ),
        "paired": {
            "both_correct": both_correct,
            "reference_only_correct": reference_only,
            "candidate_only_correct": candidate_only,
            "both_wrong": both_wrong,
            "mcnemar_exact_two_sided_p": exact_mcnemar_p(
                reference_only, candidate_only
            ),
        },
        "changed_samples": changed,
        "protocol": reference_protocol,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=pathlib.Path, required=True)
    parser.add_argument("--candidate", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ComparisonError(f"refusing to overwrite {args.output}")
        result = compare(load(args.reference), load(args.candidate))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(
            f"{result['benchmark']}: reference={result['reference']['accuracy'] * 100:.2f}% "
            f"candidate={result['candidate']['accuracy'] * 100:.2f}% "
            f"delta={result['absolute_delta'] * 100:+.2f} pp"
        )
        return 0
    except (ComparisonError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

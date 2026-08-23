#!/usr/bin/env python3
"""Qualify the frozen Gemma 4 26B M17 profile on M19 held-out evidence.

This command is deliberately an evidence verifier and summarizer, not a model
runner.  The candidate and official Google QAT Q4_0 runs write raw reports below
``artifacts/raw``; this tool verifies their identities, recomputes aggregate
gates, and emits one compact auditable M19 result.  Missing evidence is a
blocker, never an implicit pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLDS = ROOT / "artifacts/m19/thresholds.json"
DEFAULT_CORPUS = ROOT / "benchmarks/corpora/gemma4_26b/test.json"
DEFAULT_CORPUS_LOCK = ROOT / "benchmarks/corpora/gemma4_26b/splits.lock.json"
DEFAULT_M17_ACCEPTANCE = ROOT / "artifacts/m17/acceptance.json"
DEFAULT_M17_CLOSURE = ROOT / "artifacts/m17/closure-hardening.json"
DEFAULT_Q4_LOCK = ROOT / "models/gemma4-26b-qat-q4_0.lock.json"
MAX_JSON_BYTES = 512 * 1024 * 1024


class QualificationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_file(path: Path, *, max_bytes: int = MAX_JSON_BYTES) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise QualificationError(f"cannot stat {path}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise QualificationError(f"refusing symbolic-link input: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise QualificationError(f"input is not a regular file: {path}")
    if metadata.st_size > max_bytes:
        raise QualificationError(
            f"input exceeds {max_bytes} bytes: {path} ({metadata.st_size})"
        )
    return path


def load_json(path: Path) -> dict[str, Any]:
    regular_file(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QualificationError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(document, dict):
        raise QualificationError(f"JSON root is not an object: {path}")
    return document


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualificationError(f"{label} is not a number")
    result = float(value)
    if not math.isfinite(result):
        raise QualificationError(f"{label} is not finite")
    return result


def bounded_fraction(value: Any, label: str) -> float:
    result = finite_number(value, label)
    if result < 0.0 or result > 1.0:
        raise QualificationError(f"{label} is outside [0, 1]")
    return result


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise QualificationError(f"{label} mismatch: {actual!r} != {expected!r}")


def git_revision() -> str | None:
    override = os.environ.get("GEM16_QUALIFICATION_REVISION")
    if override:
        return override
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def validate_static_identity(
    thresholds_path: Path,
    corpus_path: Path,
    corpus_lock_path: Path,
    artifact_lock_path: Path,
    q4_lock_path: Path,
    m17_acceptance_path: Path,
    m17_closure_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    thresholds = load_json(thresholds_path)
    require_equal(thresholds.get("schema_version"), 1, "threshold schema")
    require_equal(thresholds.get("milestone"), "M19", "threshold milestone")
    require_equal(
        thresholds.get("status"),
        "frozen_before_q4_heldout_execution",
        "threshold freeze status",
    )
    identity = thresholds.get("identity")
    heldout = thresholds.get("heldout")
    if not isinstance(identity, dict) or not isinstance(heldout, dict):
        raise QualificationError("threshold identity/heldout contract is missing")

    require_equal(sha256(corpus_path), heldout.get("corpus_sha256"), "corpus hash")
    require_equal(
        sha256(corpus_lock_path), heldout.get("split_lock_sha256"), "split lock hash"
    )
    require_equal(
        sha256(artifact_lock_path),
        identity.get("artifact_lock_sha256"),
        "artifact lock hash",
    )
    require_equal(
        sha256(q4_lock_path),
        identity.get("reference_q4_lock_sha256"),
        "official Q4 lock hash",
    )
    require_equal(
        sha256(m17_acceptance_path),
        identity.get("m17_acceptance_sha256"),
        "M17 acceptance hash",
    )
    require_equal(
        sha256(m17_closure_path),
        identity.get("m17_closure_sha256"),
        "M17 closure hash",
    )

    corpus = load_json(corpus_path)
    split_lock = load_json(corpus_lock_path)
    artifact_lock = load_json(artifact_lock_path)
    q4_lock = load_json(q4_lock_path)
    acceptance = load_json(m17_acceptance_path)
    closure = load_json(m17_closure_path)
    require_equal(corpus.get("schema_version"), 1, "corpus schema")
    require_equal(corpus.get("split"), heldout.get("split"), "corpus split")
    require_equal(corpus.get("policy"), heldout.get("policy"), "corpus policy")
    require_equal(split_lock.get("status"), "frozen", "split lock status")
    require_equal(
        (split_lock.get("split_audit") or {}).get("status"),
        "pass",
        "split disjointness audit",
    )
    require_equal(
        (split_lock.get("split_audit") or {}).get(
            "document_or_token_span_overlap_count"
        ),
        0,
        "split overlap count",
    )

    records = corpus.get("records")
    if not isinstance(records, list):
        raise QualificationError("held-out corpus has no record list")
    require_equal(len(records), heldout.get("required_record_count"), "record count")
    categories = sorted(record.get("category") for record in records)
    require_equal(categories, heldout.get("required_categories"), "held-out categories")
    identifiers = [record.get("id") for record in records]
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise QualificationError("held-out corpus contains an invalid record id")
    if len(set(identifiers)) != len(identifiers):
        raise QualificationError("held-out corpus contains duplicate record ids")

    for source, label in ((artifact_lock, "artifact lock"), (acceptance, "M17"), (closure, "M17 closure")):
        require_equal(
            source.get("qualified_artifact_content_sha256")
            if label != "artifact lock"
            else source.get("artifact_content_sha256"),
            identity.get("artifact_content_sha256"),
            f"{label} artifact content hash",
        )
    require_equal(
        artifact_lock.get("artifact_profile"),
        identity.get("artifact_profile"),
        "artifact profile",
    )
    require_equal(
        artifact_lock.get("source_lock_sha256"),
        identity.get("source_lock_sha256"),
        "artifact source lock",
    )
    require_equal(
        q4_lock.get("revision"),
        "d1c082be9cf3c8a514acf63b8761f4b41935842e",
        "official Q4 revision",
    )
    q4_files = {
        row.get("path"): row
        for row in q4_lock.get("files", [])
        if isinstance(row, dict)
    }
    require_equal(
        (q4_files.get("gemma-4-26B_q4_0-it.gguf") or {}).get("sha256"),
        identity.get("reference_q4_gguf_sha256"),
        "official Q4 GGUF hash",
    )
    require_equal(acceptance.get("profile"), identity.get("runtime_profile"), "M17 profile")
    require_equal(closure.get("profile"), identity.get("runtime_profile"), "closure profile")

    locked_files = {
        row.get("path"): row
        for row in artifact_lock.get("files", [])
        if isinstance(row, dict)
    }
    expected_files = {
        "config.json": "config_sha256",
        "generation_config.json": "generation_config_sha256",
        "tokenizer.json": "tokenizer_sha256",
        "tokenizer_config.json": "tokenizer_config_sha256",
        "chat_template.jinja": "chat_template_sha256",
        "gem16_compilation.json": "compilation_manifest_sha256",
    }
    for filename, identity_key in expected_files.items():
        require_equal(
            (locked_files.get(filename) or {}).get("sha256"),
            identity.get(identity_key),
            f"locked {filename} hash",
        )
    return thresholds, corpus, {
        "artifact_lock": artifact_lock,
        "q4_lock": q4_lock,
        "m17_acceptance": acceptance,
        "m17_closure": closure,
    }


def validate_numerical(
    path: Path,
    thresholds: dict[str, Any],
    corpus: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    report = load_json(path)
    identity = thresholds["identity"]
    heldout = thresholds["heldout"]
    limits = thresholds["thresholds"]
    require_equal(report.get("schema_version"), 1, "numerical report schema")
    require_equal(report.get("kind"), "gemma4_26b_m19_numerical", "numerical report kind")
    require_equal(report.get("status"), "complete", "numerical report status")
    require_equal(
        report.get("artifact_content_sha256"),
        identity["artifact_content_sha256"],
        "numerical artifact hash",
    )
    require_equal(report.get("runtime_profile"), identity["runtime_profile"], "numerical profile")
    require_equal(report.get("corpus_sha256"), heldout["corpus_sha256"], "numerical corpus hash")
    reference = report.get("reference")
    if not isinstance(reference, dict):
        raise QualificationError("numerical report has no official Q4 reference identity")
    require_equal(reference.get("kind"), "official_google_qat_q4_0", "numerical reference kind")
    require_equal(
        reference.get("q4_lock_sha256"),
        identity["reference_q4_lock_sha256"],
        "numerical Q4 lock",
    )
    require_equal(
        reference.get("gguf_sha256"),
        identity["reference_q4_gguf_sha256"],
        "numerical Q4 GGUF",
    )
    require_equal(
        reference.get("llama_cpp_revision"),
        identity["reference_llama_cpp_revision"],
        "numerical llama.cpp revision",
    )

    expected = {record["id"]: record for record in corpus["records"]}
    rows = report.get("records")
    if not isinstance(rows, list):
        raise QualificationError("numerical report has no records")
    actual = {row.get("id"): row for row in rows if isinstance(row, dict)}
    if len(actual) != len(rows):
        raise QualificationError("numerical report has duplicate or malformed records")
    require_equal(sorted(actual), sorted(expected), "numerical record identities")

    total_tokens = 0
    weighted_logprob_delta = 0.0
    weighted_compared_fraction = 0.0
    all_finite = True
    min_top5 = 1.0
    min_top1 = 1.0
    max_record_logprob_delta = 0.0
    per_record: list[dict[str, Any]] = []
    for identifier, source in expected.items():
        row = actual[identifier]
        require_equal(row.get("category"), source.get("category"), f"{identifier} category")
        require_equal(
            row.get("input_token_ids_sha256_u32le"),
            source.get("input_token_ids_sha256_u32le"),
            f"{identifier} input token hash",
        )
        target_hash = row.get("target_token_ids_sha256_u32le")
        if not isinstance(target_hash, str) or len(target_hash) != 64:
            raise QualificationError(f"{identifier} has no frozen target-token hash")
        require_equal(
            row.get("target_token_source"),
            "official_q4_greedy_seed_0",
            f"{identifier} target-token source",
        )
        for evidence_hash in (
            "q4_reference_capture_sha256",
            "candidate_capture_sha256",
        ):
            value = row.get(evidence_hash)
            if not isinstance(value, str) or len(value) != 64:
                raise QualificationError(f"{identifier} has no {evidence_hash}")
        finite = row.get("all_logits_finite") is True
        all_finite = all_finite and finite
        teacher = row.get("teacher_forced")
        if not isinstance(teacher, dict):
            raise QualificationError(f"{identifier} has no teacher-forced metrics")
        count = teacher.get("scored_token_count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise QualificationError(f"{identifier} has invalid scored_token_count")
        top5 = bounded_fraction(
            teacher.get("q4_reference_token_candidate_top5_fraction"),
            f"{identifier} top5",
        )
        top1 = bounded_fraction(teacher.get("top1_agreement_fraction"), f"{identifier} top1")
        compared_fraction = bounded_fraction(
            teacher.get("selected_logprob_compared_fraction"),
            f"{identifier} selected-logprob coverage",
        )
        mean_logprob_delta = finite_number(
            teacher.get("mean_selected_logprob_absolute_delta"),
            f"{identifier} mean selected-logprob delta",
        )
        max_logprob_delta = finite_number(
            teacher.get("maximum_selected_logprob_absolute_delta"),
            f"{identifier} maximum selected-logprob delta",
        )
        if min(mean_logprob_delta, max_logprob_delta) < 0.0:
            raise QualificationError(f"{identifier} contains a negative logprob delta")
        total_tokens += count
        weighted_logprob_delta += mean_logprob_delta * count
        weighted_compared_fraction += compared_fraction * count
        category = str(source["category"])
        min_top5 = min(min_top5, top5)
        min_top1 = min(min_top1, top1)
        max_record_logprob_delta = max(max_record_logprob_delta, max_logprob_delta)
        per_record.append(
            {
                "id": identifier,
                "category": category,
                "scored_token_count": count,
                "q4_reference_token_candidate_top5_fraction": top5,
                "top1_agreement_fraction": top1,
                "selected_logprob_compared_fraction": compared_fraction,
                "mean_selected_logprob_absolute_delta": mean_logprob_delta,
                "maximum_selected_logprob_absolute_delta": max_logprob_delta,
                "all_logits_finite": finite,
            }
        )

    weighted_logprob_delta /= total_tokens
    weighted_compared_fraction /= total_tokens
    gates = {
        "all_logits_finite": all_finite,
        "q4_reference_token_candidate_top5": min_top5
        >= limits["teacher_forced_q4_reference_token_candidate_top5_fraction_min"],
        "top1_agreement": min_top1
        >= limits["teacher_forced_q4_top1_agreement_fraction_min"],
        "selected_logprob_coverage": weighted_compared_fraction
        >= limits["teacher_forced_selected_logprob_compared_fraction_min"],
        "mean_selected_logprob_delta": weighted_logprob_delta
        <= limits["teacher_forced_mean_selected_logprob_absolute_delta_max"],
        "record_selected_logprob_delta": max_record_logprob_delta
        <= limits["teacher_forced_record_selected_logprob_absolute_delta_max"],
    }
    summary = {
        "raw_report": {"bytes": path.stat().st_size, "sha256": sha256(path)},
        "scored_token_count": total_tokens,
        "weighted_mean_selected_logprob_absolute_delta": weighted_logprob_delta,
        "weighted_selected_logprob_compared_fraction": weighted_compared_fraction,
        "maximum_record_selected_logprob_absolute_delta": max_record_logprob_delta,
        "minimum_q4_reference_token_candidate_top5_fraction": min_top5,
        "minimum_top1_agreement_fraction": min_top1,
        "records": per_record,
    }
    return gates, summary


def score_map(document: dict[str, Any], label: str) -> dict[tuple[str, int], bool]:
    if document.get("schema_version") != 1 or document.get("status") != "complete":
        raise QualificationError(f"{label} is not a complete schema-version-1 task result")
    examples = document.get("examples")
    if not isinstance(examples, list) or not examples:
        raise QualificationError(f"{label} has no examples")
    result: dict[tuple[str, int], bool] = {}
    for example in examples:
        if not isinstance(example, dict) or not isinstance(example.get("id"), str):
            raise QualificationError(f"{label} contains a malformed example")
        samples = example.get("samples")
        if not isinstance(samples, list) or not samples:
            raise QualificationError(f"{label} example has no samples")
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict) or sample.get("score") not in (0, 1, 0.0, 1.0):
                raise QualificationError(f"{label} contains a non-binary score")
            key = (example["id"], index)
            if key in result:
                raise QualificationError(f"{label} contains duplicate sample {key}")
            result[key] = bool(sample["score"])
    return result


def invalid_fraction(document: dict[str, Any]) -> float:
    samples = [
        sample
        for example in document.get("examples", [])
        for sample in example.get("samples", [])
    ]
    invalid = sum(
        sample.get("finish_reason") in ("error", "invalid_format")
        or sample.get("extracted") is None
        for sample in samples
    )
    return invalid / len(samples)


def validate_tasks(
    manifest_path: Path,
    thresholds: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    manifest = load_json(manifest_path)
    identity = thresholds["identity"]
    evaluator = thresholds["evaluator"]
    limits = thresholds["thresholds"]
    require_equal(manifest.get("schema_version"), 1, "task manifest schema")
    require_equal(manifest.get("kind"), "gemma4_26b_m19_task_pairs", "task manifest kind")
    require_equal(manifest.get("status"), "complete", "task manifest status")
    require_equal(manifest.get("artifact_content_sha256"), identity["artifact_content_sha256"], "task artifact hash")
    require_equal(
        manifest.get("reference_q4_lock_sha256"),
        identity["reference_q4_lock_sha256"],
        "task official-Q4 source lock",
    )
    require_equal(manifest.get("sgl_eval_commit"), evaluator["sgl_eval_commit"], "task evaluator revision")
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list):
        raise QualificationError("task manifest has no pairs")
    by_name = {row.get("benchmark"): row for row in pairs if isinstance(row, dict)}
    if len(by_name) != len(pairs):
        raise QualificationError("task manifest has duplicate or malformed benchmark pairs")
    require_equal(sorted(by_name), sorted(evaluator["required_task_benchmarks"]), "task benchmark set")

    gates: dict[str, bool] = {}
    summaries: list[dict[str, Any]] = []
    for benchmark in evaluator["required_task_benchmarks"]:
        pair = by_name[benchmark]
        reference_path = Path(pair.get("reference_path", ""))
        candidate_path = Path(pair.get("candidate_path", ""))
        if not reference_path.is_absolute():
            reference_path = manifest_path.parent / reference_path
        if not candidate_path.is_absolute():
            candidate_path = manifest_path.parent / candidate_path
        regular_file(reference_path)
        regular_file(candidate_path)
        require_equal(sha256(reference_path), pair.get("reference_sha256"), f"{benchmark} reference hash")
        require_equal(sha256(candidate_path), pair.get("candidate_sha256"), f"{benchmark} candidate hash")
        reference = load_json(reference_path)
        candidate = load_json(candidate_path)
        require_equal(reference.get("benchmark"), benchmark, f"{benchmark} reference name")
        require_equal(candidate.get("benchmark"), benchmark, f"{benchmark} candidate name")
        require_equal(pair.get("reference_q4_lock_sha256"), identity["reference_q4_lock_sha256"], f"{benchmark} reference identity")
        require_equal(pair.get("candidate_artifact_content_sha256"), identity["artifact_content_sha256"], f"{benchmark} candidate identity")
        require_equal(
            (reference.get("endpoint") or {}).get("backend"),
            "openai",
            f"{benchmark} reference backend",
        )
        require_equal(
            (candidate.get("endpoint") or {}).get("backend"),
            "gem16",
            f"{benchmark} candidate backend",
        )
        expected_dataset = evaluator["datasets"][benchmark]
        for document, label in ((reference, "reference"), (candidate, "candidate")):
            source = document.get("benchmark_source") or {}
            require_equal(
                (source.get("sgl_eval") or {}).get("commit"),
                evaluator["sgl_eval_commit"],
                f"{benchmark} {label} evaluator revision",
            )
            observed_dataset = source.get("dataset") or {}
            if "sha256" in expected_dataset:
                require_equal(
                    observed_dataset.get("sha256"),
                    expected_dataset["sha256"],
                    f"{benchmark} {label} dataset hash",
                )
            require_equal(
                document.get("completed_examples"),
                expected_dataset["examples"],
                f"{benchmark} {label} completed examples",
            )
        require_equal(reference.get("protocol"), candidate.get("protocol"), f"{benchmark} protocols")
        protocol = reference.get("protocol") or {}
        require_equal(protocol.get("generation"), "checkpoint", f"{benchmark} generation")
        require_equal(protocol.get("repeats"), 1, f"{benchmark} repeats")
        expected_reasoning = "none" if benchmark == "gsm8k" else "high"
        require_equal(protocol.get("reasoning"), expected_reasoning, f"{benchmark} reasoning")
        require_equal(protocol.get("temperature"), 1.0, f"{benchmark} temperature")
        require_equal(protocol.get("top_p"), 0.95, f"{benchmark} top-p")
        require_equal(protocol.get("top_k"), 64, f"{benchmark} top-k")
        require_equal(protocol.get("seed"), 0, f"{benchmark} seed")
        require_equal(
            protocol.get("max_tokens"),
            512 if benchmark == "gsm8k" else 16384,
            f"{benchmark} max tokens",
        )
        reference_scores = score_map(reference, f"{benchmark} reference")
        candidate_scores = score_map(candidate, f"{benchmark} candidate")
        require_equal(reference_scores.keys(), candidate_scores.keys(), f"{benchmark} sample identities")
        count = len(reference_scores)
        reference_accuracy = sum(reference_scores.values()) / count
        candidate_accuracy = sum(candidate_scores.values()) / count
        delta = candidate_accuracy - reference_accuracy
        retention = candidate_accuracy / reference_accuracy if reference_accuracy else None
        candidate_invalid = invalid_fraction(candidate)
        gate = (
            delta >= limits["task_absolute_delta_min"]
            and retention is not None
            and retention >= limits["task_relative_retention_min"]
            and candidate_invalid <= limits["task_invalid_response_fraction_max"]
        )
        gates[f"task_{benchmark}"] = gate
        summaries.append(
            {
                "benchmark": benchmark,
                "sample_count": count,
                "reference_accuracy": reference_accuracy,
                "candidate_accuracy": candidate_accuracy,
                "absolute_delta": delta,
                "relative_retention": retention,
                "candidate_invalid_response_fraction": candidate_invalid,
                "reference": {"bytes": reference_path.stat().st_size, "sha256": sha256(reference_path)},
                "candidate": {"bytes": candidate_path.stat().st_size, "sha256": sha256(candidate_path)},
            }
        )
    return gates, {"manifest_sha256": sha256(manifest_path), "benchmarks": summaries}


def validate_prose(
    path: Path,
    thresholds: dict[str, Any],
    corpus: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    report = load_json(path)
    identity = thresholds["identity"]
    heldout = thresholds["heldout"]
    limits = thresholds["thresholds"]
    require_equal(report.get("schema_version"), 1, "prose report schema")
    require_equal(report.get("kind"), "gemma4_26b_m19_blind_prose_review", "prose report kind")
    require_equal(report.get("status"), "complete", "prose report status")
    require_equal(report.get("artifact_content_sha256"), identity["artifact_content_sha256"], "prose artifact hash")
    require_equal(report.get("corpus_sha256"), heldout["corpus_sha256"], "prose corpus hash")
    reviewers = report.get("independent_reviewer_count")
    if isinstance(reviewers, bool) or not isinstance(reviewers, int) or reviewers < 0:
        raise QualificationError("prose reviewer count is invalid")
    expected = {record["id"]: record for record in corpus["records"]}
    rows = report.get("records")
    if not isinstance(rows, list):
        raise QualificationError("prose report has no records")
    actual = {row.get("id"): row for row in rows if isinstance(row, dict)}
    if len(actual) != len(rows):
        raise QualificationError("prose report has duplicate or malformed records")
    require_equal(sorted(actual), sorted(expected), "prose record identities")
    candidate_total = reference_total = maximum_total = 0.0
    invalid_count = 0
    category_deltas: dict[str, float] = {}
    summaries = []
    for identifier, source in expected.items():
        row = actual[identifier]
        require_equal(row.get("category"), source.get("category"), f"{identifier} prose category")
        require_equal(row.get("input_token_ids_sha256_u32le"), source.get("input_token_ids_sha256_u32le"), f"{identifier} prose input hash")
        for key in ("candidate_response_sha256", "q4_response_sha256", "rubric_sha256"):
            value = row.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise QualificationError(f"{identifier} has invalid {key}")
        candidate_score = finite_number(row.get("candidate_score"), f"{identifier} candidate score")
        reference_score = finite_number(row.get("q4_score"), f"{identifier} reference score")
        maximum_score = finite_number(row.get("maximum_score"), f"{identifier} maximum score")
        if maximum_score <= 0 or not (0 <= candidate_score <= maximum_score) or not (0 <= reference_score <= maximum_score):
            raise QualificationError(f"{identifier} prose scores are outside the rubric range")
        invalid = row.get("invalid_response") is True
        invalid_count += int(invalid)
        candidate_total += candidate_score
        reference_total += reference_score
        maximum_total += maximum_score
        delta = candidate_score - reference_score
        category_deltas[str(source["category"])] = delta
        summaries.append({"id": identifier, "category": source["category"], "candidate_score": candidate_score, "q4_score": reference_score, "maximum_score": maximum_score, "score_delta": delta, "invalid_response": invalid})
    retention = candidate_total / reference_total if reference_total else None
    gates = {
        "prose_independent_review": reviewers >= limits["prose_minimum_independent_reviewers"],
        "prose_relative_retention": retention is not None and retention >= limits["prose_relative_retention_min"],
        "prose_per_category": min(category_deltas.values()) >= limits["prose_per_category_score_delta_min"],
        "prose_valid_responses": invalid_count <= limits["prose_invalid_response_count_max"],
    }
    return gates, {
        "raw_report": {"bytes": path.stat().st_size, "sha256": sha256(path)},
        "independent_reviewer_count": reviewers,
        "candidate_score": candidate_total,
        "q4_score": reference_total,
        "maximum_score": maximum_total,
        "relative_retention": retention,
        "invalid_response_count": invalid_count,
        "category_score_delta": category_deltas,
        "records": summaries,
    }


def qualify(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    thresholds, corpus, static = validate_static_identity(
        args.thresholds,
        args.corpus,
        args.corpus_lock,
        args.artifact_lock,
        args.q4_lock,
        args.m17_acceptance,
        args.m17_closure,
    )
    blockers: list[str] = []
    gates: dict[str, bool] = {
        "frozen_artifact_and_profile_identity": True,
        "frozen_heldout_corpus_and_disjointness": True,
        "thresholds_frozen_before_execution": True,
    }
    evidence: dict[str, Any] = {
        "thresholds": {"bytes": args.thresholds.stat().st_size, "sha256": sha256(args.thresholds)},
        "corpus": {"bytes": args.corpus.stat().st_size, "sha256": sha256(args.corpus)},
        "corpus_lock": {"bytes": args.corpus_lock.stat().st_size, "sha256": sha256(args.corpus_lock)},
        "artifact_lock": {"bytes": args.artifact_lock.stat().st_size, "sha256": sha256(args.artifact_lock)},
        "q4_lock": {"bytes": args.q4_lock.stat().st_size, "sha256": sha256(args.q4_lock)},
        "m17_acceptance": {"bytes": args.m17_acceptance.stat().st_size, "sha256": sha256(args.m17_acceptance)},
        "m17_closure": {"bytes": args.m17_closure.stat().st_size, "sha256": sha256(args.m17_closure)},
    }
    if args.numerical_report is None:
        blockers.append("held-out official-Q4/candidate teacher-forcing report is missing")
    else:
        numerical_gates, numerical = validate_numerical(args.numerical_report, thresholds, corpus)
        gates.update(numerical_gates)
        evidence["numerical"] = numerical
    if args.task_manifest is None:
        blockers.append("paired GSM8K/GPQA/AIME26 task manifest is missing")
    else:
        task_gates, tasks = validate_tasks(args.task_manifest, thresholds)
        gates.update(task_gates)
        evidence["tasks"] = tasks
    if args.prose_report is None:
        blockers.append("blind per-category prose review is missing")
    else:
        prose_gates, prose = validate_prose(args.prose_report, thresholds, corpus)
        gates.update(prose_gates)
        evidence["prose"] = prose

    failed = sorted(name for name, passed in gates.items() if not passed)
    if blockers:
        status = "blocked_evidence_pending"
        decision = "pending"
        code = 2
    elif failed:
        status = "qualification_fail"
        decision = "diagnose_m18_triggered"
        code = 1
    else:
        status = "qualification_pass"
        decision = "accept_frozen_m17_profile"
        code = 0
    result = {
        "schema_version": 1,
        "milestone": "M19",
        "status": status,
        "decision": decision,
        "acceptance": code == 0,
        "code_revision": git_revision(),
        "artifact_content_sha256": thresholds["identity"]["artifact_content_sha256"],
        "runtime_profile": thresholds["identity"]["runtime_profile"],
        "gates": gates,
        "failed_gates": failed,
        "blockers": blockers,
        "thresholds": thresholds["thresholds"],
        "evidence": evidence,
        "claims": (
            ["Frozen M17 text-only profile passed the specified held-out numerical, paired-task, and blind prose gates."]
            if code == 0
            else ["No production-quality claim is supported while M19 is pending or failed."]
        ),
        "limitations": [
            "Vision and audio are outside the 26B profile.",
            "Task and prose scores are paired against Google's exact official QAT Q4_0 GGUF in pinned llama.cpp; public model-card scores remain non-protocol external context.",
            "Raw responses, per-token metrics, and captures remain ignored under artifacts/raw; this compact record retains their exact hashes.",
        ],
        "static_identity": {
            "artifact_profile": static["artifact_lock"]["artifact_profile"],
            "m17_status": static["m17_acceptance"]["status"],
            "m17_closure_status": static["m17_closure"]["status"],
        },
    }
    return code, result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    result.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    result.add_argument("--corpus-lock", type=Path, default=DEFAULT_CORPUS_LOCK)
    result.add_argument("--artifact-lock", type=Path, required=True)
    result.add_argument("--q4-lock", type=Path, default=DEFAULT_Q4_LOCK)
    result.add_argument("--m17-acceptance", type=Path, default=DEFAULT_M17_ACCEPTANCE)
    result.add_argument("--m17-closure", type=Path, default=DEFAULT_M17_CLOSURE)
    result.add_argument("--numerical-report", type=Path)
    result.add_argument("--task-manifest", type=Path)
    result.add_argument("--prose-report", type=Path)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.output.exists():
            raise QualificationError(f"refusing to overwrite {args.output}")
        code, result = qualify(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"M19 {result['status']}: {args.output}")
        for blocker in result["blockers"]:
            print(f"blocker: {blocker}", file=sys.stderr)
        for gate in result["failed_gates"]:
            print(f"failed gate: {gate}", file=sys.stderr)
        return code
    except (QualificationError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 64


if __name__ == "__main__":
    raise SystemExit(main())

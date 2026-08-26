#!/usr/bin/env python3
"""Run pinned task-quality benchmarks against an OpenAI-compatible endpoint.

Scoring, prompts, and answer extraction come from the exact sgl-eval revision
pinned in tools/requirements-quality.txt. This wrapper adds gem16's strict API
mapping, immutable dataset revisions, endpoint validation, and one portable JSON
result containing every scored response.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import pathlib
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

SGL_EVAL_COMMIT = "fe0df4731526a73e915bedc1e2750ecb79274996"
GSM8K_REPOSITORY_COMMIT = "3101c7d5072418e28b9008a6636bde82a006892c"
GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    f"{GSM8K_REPOSITORY_COMMIT}/grade_school_math/data/test.jsonl"
)
GSM8K_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
GPQA_URL = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
GPQA_SHA256 = "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305"


class BenchmarkError(RuntimeError):
    pass


@dataclass(frozen=True)
class BenchmarkSpec:
    examples: int
    default_reasoning: str
    default_max_tokens: int
    estimated_output_tokens: int
    published_score: float | None
    published_source: str
    dataset: dict[str, Any]


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "gsm8k": BenchmarkSpec(
        examples=1319,
        default_reasoning="none",
        default_max_tokens=512,
        estimated_output_tokens=192,
        published_score=0.9612,
        published_source=(
            "AxionML/Gemma-4-12B-NVFP4 revision "
            "eb0838da6da0faec05a518573041c0ebc68583f7; different MLP-only "
            "NVFP4 checkpoint, greedy SGLang evaluation"
        ),
        dataset={
            "repository": "openai/grade-school-math",
            "revision": GSM8K_REPOSITORY_COMMIT,
            "path": "grade_school_math/data/test.jsonl",
            "sha256": GSM8K_SHA256,
        },
    ),
    "gpqa": BenchmarkSpec(
        examples=198,
        default_reasoning="high",
        default_max_tokens=16384,
        estimated_output_tokens=8192,
        published_score=0.788,
        published_source=(
            "google/gemma-4-12B-it revision "
            "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7; official model score "
            "with thinking, exact internal protocol not published"
        ),
        dataset={
            "source": "OpenAI simple-evals GPQA Diamond CSV",
            "url": GPQA_URL,
            "sha256": GPQA_SHA256,
            "rows": 198,
            "choice_shuffle_seed": 42,
        },
    ),
    "aime26": BenchmarkSpec(
        examples=30,
        default_reasoning="high",
        default_max_tokens=16384,
        estimated_output_tokens=8192,
        published_score=0.775,
        published_source=(
            "google/gemma-4-12B-it revision "
            "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7; official no-tools "
            "model score with thinking, exact internal protocol not published"
        ),
        dataset={
            "source": "bundled in pinned sgl-eval/NeMo-Skills snapshot",
            "sgl_eval_revision": SGL_EVAL_COMMIT,
        },
    ),
}


def normalize_base_url(value: str) -> str:
    normalized = value.rstrip("/")
    return normalized if normalized.endswith("/v1") else normalized + "/v1"


def api_root(base_url: str) -> str:
    normalized = normalize_base_url(base_url)
    return normalized[:-3] if normalized.endswith("/v1") else normalized


def request_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    api_key: str = "EMPTY",
    timeout: float = 3600.0,
    attempts: int = 4,
) -> tuple[dict[str, Any], float]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="GET" if payload is None else "POST",
    )
    for attempt in range(attempts):
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                document = json.loads(response.read())
            if not isinstance(document, dict):
                raise BenchmarkError(f"HTTP response from {url} is not a JSON object")
            return document, time.perf_counter() - started
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if error.code < 500 and error.code != 429:
                raise BenchmarkError(f"HTTP {error.code} from {url}: {detail}") from error
            last_error = f"HTTP {error.code} from {url}: {detail}"
        except (OSError, ValueError, json.JSONDecodeError) as error:
            last_error = f"request to {url} failed: {error}"
        if attempt + 1 < attempts:
            time.sleep(2**attempt)
    raise BenchmarkError(last_error)


def endpoint_model(base_url: str, api_key: str, requested: str | None) -> str:
    if requested:
        return requested
    document, _ = request_json(
        normalize_base_url(base_url) + "/models", api_key=api_key, timeout=30.0
    )
    models = document.get("data")
    if not isinstance(models, list) or not models or not isinstance(models[0], dict):
        raise BenchmarkError("endpoint /models response does not contain a model")
    model = models[0].get("id")
    if not isinstance(model, str) or not model:
        raise BenchmarkError("endpoint /models response contains an invalid model id")
    return model


def validate_gem16_endpoint(
    base_url: str, generation: str, seed: int | None
) -> dict[str, Any]:
    health, _ = request_json(api_root(base_url) + "/health", timeout=30.0)
    if health.get("status") != "ok":
        raise BenchmarkError("gem16 /health is not ok")
    sampling = health.get("sampling")
    if not isinstance(sampling, dict):
        raise BenchmarkError("gem16 /health does not expose sampling configuration")
    enabled = bool(sampling.get("enabled"))
    if generation == "greedy" and enabled:
        raise BenchmarkError(
            "quality run requests greedy generation, but gem16-server uses sampling; restart it with --greedy"
        )
    if generation == "checkpoint":
        expected = {"temperature": 1.0, "top_k": 64, "top_p": 0.95}
        if not enabled or any(
            not math.isclose(float(sampling.get(key, math.nan)), value, abs_tol=1e-6)
            for key, value in expected.items()
        ):
            raise BenchmarkError(
                "checkpoint generation requires gem16-server sampling temperature=1, top-k=64, top-p=0.95"
            )
        if seed is not None and sampling.get("seed") != seed:
            raise BenchmarkError(
                f"gem16-server sampling seed is {sampling.get('seed')!r}, but the run requests {seed}"
            )
    return health


def generation_parameters(generation: str) -> tuple[float, float, int | None]:
    if generation == "greedy":
        return 0.0, 0.95, None
    return 1.0, 0.95, 64


def build_request_payload(
    *,
    backend: str,
    model: str,
    messages: list[dict[str, Any]],
    reasoning: str,
    generation: str,
    max_tokens: int,
    seed: int | None,
) -> dict[str, Any]:
    temperature, top_p, top_k = generation_parameters(generation)
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if backend == "gem16":
        # gem16 deliberately fixes sampling at session creation. The health
        # check above proves that the server process matches this run, while
        # reasoning remains a supported per-request control.
        payload["reasoning_effort"] = reasoning
        return payload
    payload["temperature"] = temperature
    payload["top_p"] = top_p
    if seed is not None:
        payload["seed"] = seed
    extra: dict[str, Any] = {
        "chat_template_kwargs": {"enable_thinking": reasoning != "none"}
    }
    if top_k is not None:
        extra["top_k"] = top_k
    payload.update(extra)
    return payload


def installed_sgl_eval_revision() -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution("sgl-eval")
    except importlib.metadata.PackageNotFoundError as error:
        raise BenchmarkError(
            "sgl-eval is not installed; create a Python 3.10-3.13 environment and run "
            "pip install -r tools/requirements-quality.txt"
        ) from error
    direct_text = distribution.read_text("direct_url.json")
    if not direct_text:
        raise BenchmarkError("sgl-eval installation has no direct_url.json provenance")
    direct = json.loads(direct_text)
    commit = (direct.get("vcs_info") or {}).get("commit_id")
    if commit != SGL_EVAL_COMMIT:
        raise BenchmarkError(
            f"sgl-eval revision is {commit!r}, expected pinned {SGL_EVAL_COMMIT}"
        )
    return {"version": distribution.version, "commit": commit, "direct_url": direct}


def cached_download(url: str, sha256: str, name: str) -> pathlib.Path:
    cache = pathlib.Path.home() / ".cache" / "gem16" / "quality"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / name
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=120.0) as response:
            temporary.write_bytes(response.read())
        temporary.replace(path)
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != sha256:
        raise BenchmarkError(
            f"dataset checksum mismatch for {path}: expected {sha256}, observed {observed}"
        )
    return path


def _numeric_answer(text: str) -> int | float:
    value = float(text.replace(",", "").strip())
    return int(value) if value.is_integer() else value


def gsm8k_loader(example_type: Any) -> Callable[[int | None], list[Any]]:
    path = cached_download(GSM8K_URL, GSM8K_SHA256, "gsm8k-test-3101c7d5.jsonl")

    def load(num_examples: int | None) -> list[Any]:
        examples = []
        with path.open(encoding="utf-8") as source:
            for index, line in enumerate(source):
                row = json.loads(line)
                solution, answer = row["answer"].split("####")
                examples.append(
                    example_type(
                        id=f"gsm8k-{index}",
                        inputs={"problem": row["question"]},
                        target=_numeric_answer(answer),
                        meta={"reference_solution": re.sub(r"<<.*?>>", "", solution)},
                    )
                )
                if num_examples is not None and len(examples) >= num_examples:
                    break
        return examples

    return load


def _clean_gpqa(value: Any) -> str:
    if value is None:
        return " "
    return " ".join(str(value).replace(" [title]", ". ").strip().split())


def gpqa_loader(example_type: Any) -> Callable[[int | None], list[Any]]:
    path = cached_download(GPQA_URL, GPQA_SHA256, "gpqa-diamond-41d1213c.csv")

    def load(num_examples: int | None) -> list[Any]:
        rng = random.Random(42)
        examples = []
        with path.open(encoding="utf-8-sig", newline="") as source:
            dataset = csv.DictReader(source)
            for index, row in enumerate(dataset):
                correct = _clean_gpqa(row["Correct Answer"])
                choices = [
                    _clean_gpqa(row["Incorrect Answer 1"]),
                    _clean_gpqa(row["Incorrect Answer 2"]),
                    _clean_gpqa(row["Incorrect Answer 3"]),
                    correct,
                ]
                rng.shuffle(choices)
                expected = "ABCD"[choices.index(correct)]
                options = "\n".join(
                    f"{letter}) {choice}"
                    for letter, choice in zip("ABCD", choices)
                )
                question = _clean_gpqa(row["Question"])
                examples.append(
                    example_type(
                        id=str(row.get("Record ID") or f"gpqa-diamond-{index}"),
                        inputs={"problem": f"{question}\n\n{options}"},
                        target=expected,
                        meta={
                            "explanation": _clean_gpqa(row.get("Explanation")),
                            "subdomain": row.get("Subdomain"),
                            "choice_shuffle_seed": 42,
                        },
                    )
                )
                if num_examples is not None and len(examples) >= num_examples:
                    break
        if len(examples) != min(num_examples or 198, 198):
            raise BenchmarkError(f"GPQA dataset produced only {len(examples)} examples")
        return examples

    return load


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def estimate_seconds(
    examples: int,
    repeats: int,
    average_output_tokens: int,
    tokens_per_second: float,
    per_request_overhead_seconds: float,
) -> float:
    if min(examples, repeats, average_output_tokens) <= 0 or tokens_per_second <= 0:
        raise BenchmarkError("runtime estimate values must be positive")
    requests = examples * repeats
    return requests * (
        average_output_tokens / tokens_per_second + per_request_overhead_seconds
    )


def write_json_atomic(path: pathlib.Path, document: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def quality_loader(benchmark: str, example_type: Any) -> Callable[[int | None], list[Any]]:
    if benchmark == "gsm8k":
        return gsm8k_loader(example_type)
    if benchmark == "gpqa":
        return gpqa_loader(example_type)
    if benchmark == "aime26":
        try:
            from sgl_eval.evals._loader import load_bundled
        except ImportError as error:
            raise BenchmarkError(f"cannot load bundled AIME26 data: {error}") from error
        return load_bundled("aime26")
    raise BenchmarkError(f"no resumable loader for benchmark {benchmark!r}")


def resume_identity(
    args: argparse.Namespace,
    *,
    model: str,
    provenance: dict[str, Any],
    examples: list[Any],
) -> dict[str, Any]:
    return {
        "benchmark": args.benchmark,
        "backend": args.backend,
        "model": model,
        "sgl_eval_commit": provenance["commit"],
        "dataset": BENCHMARKS[args.benchmark].dataset,
        "protocol": {
            "reasoning": args.reasoning,
            "generation": args.generation,
            "temperature": generation_parameters(args.generation)[0],
            "top_p": generation_parameters(args.generation)[1],
            "top_k": generation_parameters(args.generation)[2],
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "repeats": args.repeats,
        },
        "planned_example_ids": [example.id for example in examples],
    }


def initialize_resume_state(
    state_path: pathlib.Path,
    identity: dict[str, Any],
) -> dict[str, Any]:
    if state_path.exists():
        if state_path.is_symlink() or not state_path.is_file():
            raise BenchmarkError(f"resume state is not a regular file: {state_path}")
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise BenchmarkError(f"cannot read resume state {state_path}: {error}") from error
        if not isinstance(state, dict) or state.get("schema_version") != 1:
            raise BenchmarkError(f"invalid resume state: {state_path}")
        if state.get("identity") != identity:
            raise BenchmarkError(
                "resume contract differs from the original run; use the original "
                "model, dataset, protocol, and example count or select a new output path"
            )
        return state
    state = {
        "schema_version": 1,
        "status": "in_progress",
        "created_at_unix": time.time(),
        "identity": identity,
    }
    write_json_atomic(state_path, state)
    return state


def prediction_row(
    example: Any,
    sample: Any,
    score: float,
    extracted: str | None,
) -> dict[str, Any]:
    completion = sample.completion_tokens or 0
    reasoning = sample.reasoning_tokens or 0
    row: dict[str, Any] = {
        "id": example.id,
        "expected_answer": str(example.target),
        "num_generated_tokens": completion,
        "num_prompt_tokens": sample.prompt_tokens,
        "num_reasoning_tokens": reasoning,
        "num_answer_tokens": max(completion - reasoning, 0),
        "finish_reason": sample.finish_reason,
        "problem": example.inputs.get("problem", ""),
        "generation": sample.text,
        "predicted_answer": extracted,
        "symbolic_correct": bool(score),
    }
    if sample.generation_start_time is not None:
        row["generation_start_time"] = sample.generation_start_time
    if sample.generation_end_time is not None:
        row["generation_end_time"] = sample.generation_end_time
    return row


def load_prediction_rows(
    path: pathlib.Path,
    examples: list[Any],
) -> dict[str, dict[str, Any]]:
    expected = {example.id: example for example in examples}
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    if path.is_symlink() or not path.is_file():
        raise BenchmarkError(f"prediction journal is not a regular file: {path}")
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if len(line) > 16 * 1024 * 1024:
                raise BenchmarkError(f"prediction journal line {line_number} is too large")
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise BenchmarkError(
                    f"prediction journal has invalid JSON on line {line_number}: {error}"
                ) from error
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise BenchmarkError(
                    f"prediction journal has a malformed row on line {line_number}"
                )
            identifier = row["id"]
            if identifier in rows:
                raise BenchmarkError(f"prediction journal contains duplicate id {identifier!r}")
            example = expected.get(identifier)
            if example is None:
                raise BenchmarkError(f"prediction journal contains unexpected id {identifier!r}")
            if row.get("problem") != example.inputs.get("problem", ""):
                raise BenchmarkError(f"prediction journal problem changed for {identifier!r}")
            if row.get("expected_answer") != str(example.target):
                raise BenchmarkError(f"prediction journal target changed for {identifier!r}")
            if row.get("symbolic_correct") not in (True, False):
                raise BenchmarkError(f"prediction journal score is invalid for {identifier!r}")
            rows[identifier] = row
    return rows


class ResumablePredictionsWriter:
    def __init__(
        self,
        path: pathlib.Path,
        examples: list[Any],
    ) -> None:
        self._path = path
        self._completed = set(load_prediction_rows(path, examples))
        self._file = path.open("a", encoding="utf-8")

    def __call__(
        self,
        example: Any,
        repeat: int,
        sample: Any,
        score: float,
        extracted: str | None,
    ) -> None:
        if repeat != 0:
            raise BenchmarkError("resumable quality runs support exactly one repeat")
        if example.id in self._completed:
            raise BenchmarkError(f"refusing duplicate resumed sample {example.id!r}")
        row = prediction_row(example, sample, score, extracted)
        self._file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())
        self._completed.add(example.id)

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


def serialize_prediction_rows(
    examples: list[Any],
    rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for example in examples:
        row = rows.get(example.id)
        if row is None:
            continue
        started = row.get("generation_start_time")
        ended = row.get("generation_end_time")
        serialized.append(
            {
                "id": example.id,
                "inputs": example.inputs,
                "target": example.target,
                "meta": example.meta,
                "samples": [
                    {
                        "text": row.get("generation", ""),
                        "score": 1.0 if row["symbolic_correct"] else 0.0,
                        "extracted": row.get("predicted_answer"),
                        "prompt_tokens": row.get("num_prompt_tokens"),
                        "completion_tokens": row.get("num_generated_tokens"),
                        "reasoning_tokens": row.get("num_reasoning_tokens"),
                        "finish_reason": row.get("finish_reason"),
                        "generation_seconds": (
                            ended - started
                            if isinstance(started, (int, float))
                            and isinstance(ended, (int, float))
                            else None
                        ),
                    }
                ],
            }
        )
    return serialized


def resumed_aggregate(rows: dict[str, dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"score": 0.0}
    result: dict[str, float] = {
        "score": sum(bool(row["symbolic_correct"]) for row in rows.values())
        / len(rows)
    }
    reasons = [row.get("finish_reason") for row in rows.values() if row.get("finish_reason")]
    if reasons:
        count = len(reasons)
        result.update(
            {
                "stop_rate": sum(value == "stop" for value in reasons) / count,
                "truncated_rate": sum(value == "length" for value in reasons) / count,
                "error_rate": sum(value not in ("stop", "length") for value in reasons)
                / count,
            }
        )
    return result


def serialize_result(result: Any) -> list[dict[str, Any]]:
    serialized = []
    for example_result in result.per_example:
        serialized.append(
            {
                "id": example_result.example.id,
                "inputs": example_result.example.inputs,
                "target": example_result.example.target,
                "meta": example_result.example.meta,
                "samples": [
                    {
                        "text": sample.text,
                        "score": score,
                        "extracted": extracted,
                        "prompt_tokens": sample.prompt_tokens,
                        "completion_tokens": sample.completion_tokens,
                        "reasoning_tokens": sample.reasoning_tokens,
                        "finish_reason": sample.finish_reason,
                        "generation_seconds": (
                            sample.generation_end_time - sample.generation_start_time
                            if sample.generation_start_time is not None
                            and sample.generation_end_time is not None
                            else None
                        ),
                    }
                    for sample, score, extracted in zip(
                        example_result.samples,
                        example_result.scores,
                        example_result.extracted,
                    )
                ],
            }
        )
    return serialized


def run_resumable(
    args: argparse.Namespace,
    *,
    provenance: dict[str, Any],
    spec_config: BenchmarkSpec,
    model: str,
    health: dict[str, Any] | None,
    example_type: Any,
    benchmark_spec: Any,
    generation: Any,
    sampler: Any,
) -> dict[str, Any]:
    raw_dir = args.output.parent / f"{args.output.stem}.samples"
    if raw_dir.exists() and (raw_dir.is_symlink() or not raw_dir.is_dir()):
        raise BenchmarkError(f"sample path is not a regular directory: {raw_dir}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    state_path = raw_dir / "resume-state.json"
    journal_path = raw_dir / "output-rs0.jsonl"
    if journal_path.exists() and not state_path.exists():
        raise BenchmarkError(
            f"prediction journal exists without its resume contract: {journal_path}"
        )

    loader = quality_loader(args.benchmark, example_type)
    examples = loader(args.num_examples)
    identifiers = [example.id for example in examples]
    if len(set(identifiers)) != len(identifiers):
        raise BenchmarkError("benchmark loader produced duplicate example ids")
    identity = resume_identity(
        args,
        model=model,
        provenance=provenance,
        examples=examples,
    )
    state = initialize_resume_state(state_path, identity)
    before = load_prediction_rows(journal_path, examples)
    remaining = [example for example in examples if example.id not in before]
    if before:
        print(
            f"Resuming {args.benchmark}: {len(before)} complete, "
            f"{len(remaining)} remaining",
            file=sys.stderr,
        )

    if remaining:
        writer = ResumablePredictionsWriter(journal_path, examples)
        try:
            benchmark_spec.run(
                sampler=sampler,
                gen=generation,
                n_repeats=1,
                num_examples=None,
                num_threads=1,
                predictions_writer=writer,
                load_examples=lambda _limit: remaining,
            )
        finally:
            writer.close()

    rows = load_prediction_rows(journal_path, examples)
    completed = time.time()
    status = "complete" if len(rows) == len(examples) else "partial"
    state.update(
        {
            "status": status,
            "completed_example_count": len(rows),
            "updated_at_unix": completed,
        }
    )
    write_json_atomic(state_path, state)

    generation_seconds = sum(
        max(0.0, float(row["generation_end_time"]) - float(row["generation_start_time"]))
        for row in rows.values()
        if isinstance(row.get("generation_start_time"), (int, float))
        and isinstance(row.get("generation_end_time"), (int, float))
    )
    prompt_tokens = sum(
        int(row.get("num_prompt_tokens") or 0) for row in rows.values()
    )
    completion_tokens = sum(
        int(row.get("num_generated_tokens") or 0) for row in rows.values()
    )
    output = {
        "schema_version": 1,
        "status": status,
        "benchmark": args.benchmark,
        "benchmark_source": {
            "gem16_commit": git_revision(),
            "sgl_eval": provenance,
            "dataset": spec_config.dataset,
        },
        "endpoint": {
            "backend": args.backend,
            "base_url": normalize_base_url(args.base_url),
            "model": model,
            "gem16_health": health,
        },
        "protocol": {
            **identity["protocol"],
            "threads": 1,
        },
        "published_reference": {
            "score": spec_config.published_score,
            "source": spec_config.published_source,
            "comparison_is_exact_protocol": False,
        },
        "started_at_unix": state["created_at_unix"],
        "completed_at_unix": completed,
        "latency_seconds": generation_seconds,
        "planned_examples": len(examples),
        "completed_examples": len(rows),
        "total_prompt_tokens": prompt_tokens,
        "total_completion_tokens": completion_tokens,
        "output_tokens_per_second": (
            completion_tokens / generation_seconds if generation_seconds > 0 else 0.0
        ),
        "aggregate": resumed_aggregate(rows),
        "examples": serialize_prediction_rows(examples, rows),
        "resume": {
            "enabled": True,
            "journal": journal_path.name,
            "state": state_path.name,
            "completed_before_attempt": len(before),
            "generated_this_attempt": len(rows) - len(before),
            "remaining": len(examples) - len(rows),
        },
    }
    write_json_atomic(args.output, output)
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    provenance = installed_sgl_eval_revision()
    try:
        from sgl_eval.predictions import PredictionsWriter
        from sgl_eval.registry import get
        from sgl_eval.types import Example, GenConfig, Sample
    except ImportError as error:
        raise BenchmarkError(f"cannot import pinned sgl-eval: {error}") from error

    spec_config = BENCHMARKS[args.benchmark]
    model = endpoint_model(args.base_url, args.api_key, args.model)
    health = (
        validate_gem16_endpoint(args.base_url, args.generation, args.seed)
        if args.backend == "gem16"
        else None
    )

    class EndpointSampler:
        def __call__(
            self, messages: list[dict[str, Any]], _gen: Any = None
        ) -> Any:
            payload = build_request_payload(
                backend=args.backend,
                model=model,
                messages=messages,
                reasoning=args.reasoning,
                generation=args.generation,
                max_tokens=args.max_tokens,
                seed=args.seed,
            )
            document, elapsed = request_json(
                normalize_base_url(args.base_url) + "/chat/completions",
                payload=payload,
                api_key=args.api_key,
                timeout=args.timeout,
            )
            choices = document.get("choices")
            if not isinstance(choices, list) or not choices:
                raise BenchmarkError("chat completion response contains no choice")
            choice = choices[0]
            message = choice.get("message") or {}
            text = message.get("content") or ""
            if not isinstance(text, str):
                raise BenchmarkError("chat completion content is not a string")
            usage = document.get("usage") or {}
            details = usage.get("completion_tokens_details") or {}
            ended = time.time()
            return Sample(
                text=text,
                completion_tokens=usage.get("completion_tokens"),
                prompt_tokens=usage.get("prompt_tokens"),
                reasoning_tokens=details.get("reasoning_tokens"),
                finish_reason=choice.get("finish_reason"),
                generation_start_time=ended - elapsed,
                generation_end_time=ended,
            )

    benchmark_spec = get(args.benchmark)
    generation = GenConfig(
        temperature=generation_parameters(args.generation)[0],
        top_p=generation_parameters(args.generation)[1],
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning,
        chat_template_kwargs={"thinking": args.reasoning != "none"},
        seed=args.seed,
    )
    if args.resume:
        return run_resumable(
            args,
            provenance=provenance,
            spec_config=spec_config,
            model=model,
            health=health,
            example_type=Example,
            benchmark_spec=benchmark_spec,
            generation=generation,
            sampler=EndpointSampler(),
        )

    loader = None
    if args.benchmark == "gsm8k":
        loader = gsm8k_loader(Example)
    elif args.benchmark == "gpqa":
        loader = gpqa_loader(Example)

    raw_dir = args.output.parent / f"{args.output.stem}.samples"
    if raw_dir.exists():
        raise BenchmarkError(f"refusing to overwrite sample directory {raw_dir}")
    raw_dir.mkdir(parents=True)
    writer = PredictionsWriter(raw_dir, args.repeats)
    started = time.time()
    try:
        result = benchmark_spec.run(
            sampler=EndpointSampler(),
            gen=generation,
            n_repeats=args.repeats,
            num_examples=args.num_examples,
            num_threads=args.threads,
            predictions_writer=writer,
            load_examples=loader,
        )
    finally:
        writer.close()
    completed = time.time()
    output = {
        "schema_version": 1,
        "status": "partial" if result.partial else "complete",
        "benchmark": args.benchmark,
        "benchmark_source": {
            "gem16_commit": git_revision(),
            "sgl_eval": provenance,
            "dataset": spec_config.dataset,
        },
        "endpoint": {
            "backend": args.backend,
            "base_url": normalize_base_url(args.base_url),
            "model": model,
            "gem16_health": health,
        },
        "protocol": {
            "reasoning": args.reasoning,
            "generation": args.generation,
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "top_k": generation_parameters(args.generation)[2],
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "repeats": args.repeats,
            "threads": args.threads,
        },
        "published_reference": {
            "score": spec_config.published_score,
            "source": spec_config.published_source,
            "comparison_is_exact_protocol": False,
        },
        "started_at_unix": started,
        "completed_at_unix": completed,
        "latency_seconds": result.latency,
        "planned_examples": result.planned_examples,
        "completed_examples": result.num_examples,
        "total_prompt_tokens": result.total_prompt_tokens,
        "total_completion_tokens": result.total_completion_tokens,
        "output_tokens_per_second": result.output_throughput,
        "aggregate": result.aggregate,
        "examples": serialize_result(result),
    }
    write_json_atomic(args.output, output)
    return output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--benchmark", choices=sorted(BENCHMARKS), required=True)
    result.add_argument("--backend", choices=("gem16", "openai"), default="gem16")
    result.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    result.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    result.add_argument("--model")
    result.add_argument("--output", type=pathlib.Path, required=True)
    result.add_argument(
        "--resume",
        action="store_true",
        help=(
            "journal each completed sample and resume the same output path; "
            "requires --repeats 1 --threads 1"
        ),
    )
    result.add_argument("--num-examples", type=int)
    result.add_argument("--repeats", type=int, default=1)
    result.add_argument("--threads", type=int, default=1)
    result.add_argument("--reasoning", choices=("none", "low", "medium", "high"))
    result.add_argument(
        "--generation", choices=("greedy", "checkpoint"), default="checkpoint"
    )
    result.add_argument("--max-tokens", type=int)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--timeout", type=float, default=3600.0)
    result.add_argument("--estimate-only", action="store_true")
    result.add_argument("--average-output-tokens", type=int)
    result.add_argument("--tokens-per-second", type=float, default=35.0)
    result.add_argument("--per-request-overhead-seconds", type=float, default=0.5)
    return result


def main() -> int:
    args = parser().parse_args()
    spec = BENCHMARKS[args.benchmark]
    if args.num_examples is not None and args.num_examples <= 0:
        print("error: --num-examples must be positive", file=sys.stderr)
        return 64
    if args.repeats <= 0 or args.threads <= 0:
        print("error: --repeats and --threads must be positive", file=sys.stderr)
        return 64
    if args.resume and (args.repeats != 1 or args.threads != 1):
        print(
            "error: --resume currently requires --repeats 1 --threads 1",
            file=sys.stderr,
        )
        return 64
    if args.max_tokens is not None and args.max_tokens <= 0:
        print("error: --max-tokens must be positive", file=sys.stderr)
        return 64
    if args.average_output_tokens is not None and args.average_output_tokens <= 0:
        print("error: --average-output-tokens must be positive", file=sys.stderr)
        return 64
    args.reasoning = args.reasoning or spec.default_reasoning
    args.max_tokens = args.max_tokens or spec.default_max_tokens
    args.average_output_tokens = (
        args.average_output_tokens or spec.estimated_output_tokens
    )
    planned = min(args.num_examples or spec.examples, spec.examples)
    try:
        estimate = estimate_seconds(
            planned,
            args.repeats,
            args.average_output_tokens,
            args.tokens_per_second,
            args.per_request_overhead_seconds,
        )
        print(
            f"Estimated wall time: {estimate / 3600.0:.2f} h for {planned} examples x "
            f"{args.repeats} repeat(s), assuming {args.average_output_tokens} output tokens "
            f"at {args.tokens_per_second:g} tok/s plus "
            f"{args.per_request_overhead_seconds:g}s/request",
            file=sys.stderr,
        )
        if args.estimate_only:
            return 0
        if args.output.exists() and not args.resume:
            raise BenchmarkError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = run(args)
        score = float(result["aggregate"].get("score", 0.0))
        print(
            f"{args.benchmark}: {score * 100.0:.2f}% over "
            f"{result['completed_examples']} examples; wrote {args.output}"
        )
        return 0 if result["status"] == "complete" else 2
    except (BenchmarkError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

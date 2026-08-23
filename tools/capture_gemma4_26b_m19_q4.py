#!/usr/bin/env python3
"""Capture the frozen M19 held-out suite from Google's official QAT Q4_0 GGUF."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any


EXPECTED_REVISION = "0b14b87d7c20cb753b94b96854dd7b45306fc696"
EXPECTED_GGUF_SHA256 = "3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d"


class CaptureError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise CaptureError(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CaptureError(f"JSON root is not an object: {path}")
    return value


def request_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise CaptureError("llama.cpp returned a non-object response")
    return value


def wait_ready(endpoint: str, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise CaptureError(f"llama.cpp exited during startup with {process.returncode}")
        try:
            if request_json(endpoint + "/health").get("status") == "ok":
                return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass
        time.sleep(0.25)
    raise CaptureError("llama.cpp did not become healthy within 120 seconds")


def validate_lock(lock: dict[str, Any], gguf: Path) -> None:
    if lock.get("revision") != "d1c082be9cf3c8a514acf63b8761f4b41935842e":
        raise CaptureError("official Google Q4 revision changed")
    files = {
        row.get("path"): row
        for row in lock.get("files", [])
        if isinstance(row, dict)
    }
    expected = files.get("gemma-4-26B_q4_0-it.gguf") or {}
    if expected.get("sha256") != EXPECTED_GGUF_SHA256:
        raise CaptureError("official Google Q4 lock has the wrong GGUF hash")
    if gguf.stat().st_size != expected.get("size") or sha256_file(gguf) != EXPECTED_GGUF_SHA256:
        raise CaptureError("local official Google Q4 GGUF does not match its lock")


def collect(args: argparse.Namespace) -> dict[str, Any]:
    executable = args.executable.resolve(strict=True)
    gguf = args.gguf.resolve(strict=True)
    corpus_path = args.corpus.resolve(strict=True)
    lock_path = args.q4_lock.resolve(strict=True)
    corpus = load_object(corpus_path)
    lock = load_object(lock_path)
    validate_lock(lock, gguf)
    records = corpus.get("records")
    if corpus.get("schema_version") != 1 or corpus.get("split") != "test" or not isinstance(records, list):
        raise CaptureError("M19 corpus contract changed")

    version = subprocess.run(
        [str(executable), "--version"], capture_output=True, text=True,
        check=False, timeout=30,
    )
    if version.returncode != 0 or EXPECTED_REVISION[:9] not in (version.stdout + version.stderr):
        raise CaptureError("llama.cpp executable is not the pinned b10240 revision")

    endpoint = f"http://127.0.0.1:{args.port}"
    command = [
        str(executable), "-m", str(gguf), "-ngl", "99", "-ot",
        "token_embd.weight=CUDA0", "-fa", "on", "-ctk", "q8_0",
        "-ctv", "q8_0", "-c", str(args.context), "-np", "1",
        "--host", "127.0.0.1", "--port", str(args.port),
    ]
    args.server_log.parent.mkdir(parents=True, exist_ok=True)
    with args.server_log.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        try:
            wait_ready(endpoint, process)
            captures: list[dict[str, Any]] = []
            for index, source in enumerate(records, start=1):
                identifier = source.get("id")
                tokens = source.get("input_token_ids")
                if not isinstance(identifier, str) or not isinstance(tokens, list):
                    raise CaptureError("M19 corpus record is malformed")
                print(f"[{index}/{len(records)}] official Q4 {identifier}", flush=True)
                response = request_json(
                    endpoint + "/completion",
                    {
                        "prompt": tokens,
                        "n_predict": args.max_tokens,
                        "temperature": 0.0,
                        "seed": 0,
                        "n_probs": args.top_logprobs,
                        "cache_prompt": False,
                        "stream": False,
                    },
                )
                probabilities = response.get("completion_probabilities")
                if not isinstance(probabilities, list) or not probabilities:
                    raise CaptureError(f"{identifier}: no completion probabilities")
                steps: list[dict[str, Any]] = []
                for step_index, probability in enumerate(probabilities):
                    if not isinstance(probability, dict) or not isinstance(probability.get("id"), int):
                        raise CaptureError(f"{identifier}: malformed probability step")
                    top = probability.get("top_logprobs")
                    if not isinstance(top, list) or not top:
                        raise CaptureError(f"{identifier}: missing top-logprobs")
                    steps.append({
                        "index": step_index,
                        "token_id": probability["id"],
                        "top_logprobs": [
                            {"token_id": row.get("id"), "logprob": row.get("logprob")}
                            for row in top if isinstance(row, dict)
                        ],
                    })
                capture = {
                    "id": identifier,
                    "category": source.get("category"),
                    "input_token_ids_sha256_u32le": source.get("input_token_ids_sha256_u32le"),
                    "prompt_tokens": len(tokens),
                    "text": response.get("content"),
                    "finish_reason": "stop" if response.get("stopped_eos") else "length",
                    "steps": steps,
                }
                capture["record_sha256"] = canonical_sha256(capture)
                captures.append(capture)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    return {
        "schema_version": 1,
        "kind": "gemma4_26b_m19_official_q4_capture",
        "status": "complete",
        "corpus_sha256": sha256_file(corpus_path),
        "reference": {
            "kind": "official_google_qat_q4_0",
            "q4_lock_sha256": sha256_file(lock_path),
            "gguf_sha256": EXPECTED_GGUF_SHA256,
            "llama_cpp_revision": EXPECTED_REVISION,
        },
        "execution": {
            "context_tokens": args.context,
            "maximum_generated_tokens": args.max_tokens,
            "temperature": 0.0,
            "seed": 0,
            "top_logprobs": args.top_logprobs,
            "all_layers_on_gpu": True,
            "kv_cache": "q8_0",
        },
        "records": captures,
        "server_log_sha256": sha256_file(args.server_log),
        "command": command,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--q4-lock", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--port", type=int, default=18091)
    args = parser.parse_args()
    if args.output.exists() or args.server_log.exists():
        parser.error("output and server log must not already exist")
    if args.context < 1 or args.max_tokens < 1 or args.top_logprobs < 5:
        parser.error("context/tokens/top-logprobs are invalid")
    return args


def main() -> int:
    args = parse_args()
    try:
        document = collect(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output), "records": len(document["records"])}))
        return 0
    except (CaptureError, OSError, ValueError, json.JSONDecodeError, urllib.error.URLError) as error:
        print(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the deterministic V19 bounded Vision quality matrix."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any
import urllib.error
import urllib.request


MAX_IMAGE_BYTES = 16 * 1024 * 1024
EXPECTED_BUDGETS = (70, 140, 280)


class QualificationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_json(
    url: str, payload: dict[str, Any] | None = None, timeout: float = 300.0
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": "Bearer gem16-v19",
            "Content-Type": "application/json",
        },
        method="GET" if body is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            document = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise QualificationError(f"HTTP {error.code} from {url}: {detail}") from error
    if not isinstance(document, dict):
        raise QualificationError(f"response from {url} is not a JSON object")
    return document


def metrics(url: str, timeout: float) -> dict[str, float]:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        lines = response.read().decode("utf-8").splitlines()
    result: dict[str, float] = {}
    for line in lines:
        fields = line.split()
        if len(fields) == 2 and not line.startswith("#"):
            result[fields[0]] = float(fields[1])
    return result


def wait_for_server(url: str, process: subprocess.Popen[Any], timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise QualificationError(f"server exited during startup: {process.returncode}")
        try:
            return request_json(f"{url}/health", timeout=5.0)
        except (OSError, urllib.error.URLError, QualificationError) as error:
            last_error = error
            time.sleep(0.5)
    raise QualificationError(f"server did not become healthy: {last_error}")


def load_suite(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = path.resolve().parent
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        document.get("schema_version") != 1
        or tuple(document.get("budgets", [])) != EXPECTED_BUDGETS
        or not isinstance(document.get("assets"), list)
        or not document["assets"]
    ):
        raise QualificationError("invalid V19 suite root")
    assets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in document["assets"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise QualificationError("invalid V19 asset entry")
        if entry["id"] in seen:
            raise QualificationError(f"duplicate V19 asset id: {entry['id']}")
        seen.add(entry["id"])
        image = (root / entry.get("path", "")).resolve()
        try:
            image.relative_to(root)
        except ValueError as error:
            raise QualificationError(f"asset escapes suite: {entry['id']}") from error
        if (
            not image.is_file()
            or image.is_symlink()
            or image.stat().st_size > MAX_IMAGE_BYTES
            or sha256(image) != entry.get("sha256")
        ):
            raise QualificationError(f"unsafe or mismatched asset: {entry['id']}")
        if not isinstance(entry.get("prompt"), str) or not entry["prompt"]:
            raise QualificationError(f"asset prompt is missing: {entry['id']}")
        checks = entry.get("checks")
        if not isinstance(checks, list) or not checks:
            raise QualificationError(f"asset checks are missing: {entry['id']}")
        for check in checks:
            required = check.get("required_budgets") if isinstance(check, dict) else None
            terms = check.get("terms") if isinstance(check, dict) else None
            if (
                not isinstance(required, list)
                or any(budget not in EXPECTED_BUDGETS for budget in required)
                or not isinstance(terms, list)
                or not terms
                or any(not isinstance(term, str) or not term for term in terms)
            ):
                raise QualificationError(f"invalid check in asset: {entry['id']}")
        assets.append({**entry, "resolved_path": image})
    return document, assets


def normalized(value: str) -> str:
    return " ".join(value.casefold().replace("–", "-").replace("—", "-").split())


def check_output(output: str, check: dict[str, Any]) -> bool:
    text = normalized(output)
    terms_match = all(normalized(term) in text for term in check["terms"])
    alternatives = check.get("alternatives", [])
    alternatives_match = any(
        all(normalized(term) in text for term in alternative)
        if isinstance(alternative, list)
        else normalized(alternative) in text
        for alternative in alternatives
    )
    return terms_match or alternatives_match


def response_text(document: dict[str, Any]) -> str:
    try:
        content = document["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise QualificationError("chat response has no assistant content") from error
    if not isinstance(content, str):
        raise QualificationError("assistant content is not text")
    return content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vision-model", type=Path, required=True)
    parser.add_argument("--assistant-model", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18089)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing result: {args.output}")
    if args.runs < 2 or args.runs > 5 or not 1 <= args.port <= 65535:
        raise SystemExit("--runs must be within 2..5 and --port must be valid")
    if not args.server.is_file() or any(
        not path.is_dir()
        for path in (args.model, args.vision_model, args.assistant_model)
    ):
        raise SystemExit("server executable and all component directories are required")
    suite, assets = load_suite(args.suite)
    base_url = f"http://127.0.0.1:{args.port}"
    command = [
        str(args.server.resolve()),
        "--model",
        str(args.model.resolve()),
        "--vision-model",
        str(args.vision_model.resolve()),
        "--assistant-model",
        str(args.assistant_model.resolve()),
        "--mtp-draft-tokens",
        "2",
        "--max-context",
        "32768",
        "--max-sessions",
        "1",
        "--model-name",
        "gem16-v19",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--greedy",
        "--log-level",
        "warning",
    ]
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as server_log:
        process = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT)
        try:
            health = wait_for_server(base_url, process, args.timeout)
            capabilities = health.get("capabilities")
            if (
                not isinstance(capabilities, dict)
                or capabilities.get("vision_mtp") is not True
            ):
                raise QualificationError("server did not expose qualified Vision+D2")
            before = metrics(f"{base_url}/metrics", 10.0)
            records: list[dict[str, Any]] = []
            for asset in assets:
                encoded = base64.b64encode(asset["resolved_path"].read_bytes()).decode("ascii")
                for budget in EXPECTED_BUDGETS:
                    print(f"quality {asset['id']} budget {budget}", flush=True)
                    outputs: list[str] = []
                    usages: list[dict[str, Any]] = []
                    for _ in range(args.runs):
                        response = request_json(
                            f"{base_url}/v1/chat/completions",
                            {
                                "model": "gem16-v19",
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": "data:image/png;base64," + encoded
                                                },
                                            },
                                            {"type": "text", "text": asset["prompt"]},
                                        ],
                                    }
                                ],
                                "max_tokens": 160,
                                "vision_soft_token_budget": budget,
                            },
                            args.timeout,
                        )
                        outputs.append(response_text(response))
                        usages.append(response.get("usage", {}))
                    checks = [
                        {
                            "id": check["id"],
                            "required": budget in check["required_budgets"],
                            "passed": check_output(outputs[0], check),
                        }
                        for check in asset["checks"]
                    ]
                    records.append(
                        {
                            "asset": asset["id"],
                            "geometry": asset["geometry"],
                            "categories": asset["categories"],
                            "budget": budget,
                            "output": outputs[0],
                            "output_sha256": hashlib.sha256(outputs[0].encode()).hexdigest(),
                            "deterministic": len(set(outputs)) == 1,
                            "checks": checks,
                            "usage": usages,
                        }
                    )
            final_health = request_json(f"{base_url}/health", timeout=10.0)
            after = metrics(f"{base_url}/metrics", 10.0)
        finally:
            process.terminate()
            try:
                process.wait(timeout=20.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10.0)
            server_log.seek(0)
            log_tail = server_log.read()[-4000:]

    counter_names = (
        "gem16_fallback_total",
        "gem16_token_loop_allocation_total",
        "gem16_vision_failures_total",
        "gem16_vision_d2_rejections_total",
    )
    counter_deltas = {
        name: after.get(name, 0.0) - before.get(name, 0.0) for name in counter_names
    }
    expected_per_budget = len(assets) * args.runs
    budget_deltas = {
        budget: after.get(f"gem16_vision_budget_{budget}_total", 0.0)
        - before.get(f"gem16_vision_budget_{budget}_total", 0.0)
        for budget in EXPECTED_BUDGETS
    }
    vision_requests = after.get("gem16_vision_requests_total", 0.0) - before.get(
        "gem16_vision_requests_total", 0.0
    )
    required_checks_pass = all(
        check["passed"]
        for record in records
        for check in record["checks"]
        if check["required"]
    )
    accepted = (
        required_checks_pass
        and all(record["deterministic"] for record in records)
        and all(value == 0.0 for value in counter_deltas.values())
        and all(value == expected_per_budget for value in budget_deltas.values())
        and vision_requests == len(assets) * len(EXPECTED_BUDGETS) * args.runs
        and final_health.get("selected_vision_soft_token_budget") == 280
    )
    payload = {
        "schema_version": 1,
        "milestone": "V19",
        "qualification": "bounded_vision_quality",
        "accepted": accepted,
        "binary": {"path": str(args.server.resolve()), "sha256": sha256(args.server)},
        "suite": {
            "path": str(args.suite.resolve()),
            "sha256": sha256(args.suite),
            "name": suite["name"],
        },
        "health": health,
        "final_health": final_health,
        "runs_per_case": args.runs,
        "records": records,
        "counter_deltas": counter_deltas,
        "budget_counter_deltas": budget_deltas,
        "vision_request_delta": vision_requests,
        "required_checks_pass": required_checks_pass,
        "server_log_tail": log_tail,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}; accepted={accepted}")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())

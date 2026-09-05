#!/usr/bin/env python3
"""Bounded live regression for HTTP policy, cache reconciliation and admission."""

import argparse
import concurrent.futures
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid


def run(base, samples):
    def request(path, body=None, headers=None):
        h = {"Content-Type": "application/json", **(headers or {})}
        req = urllib.request.Request(
            base + path, None if body is None else json.dumps(body).encode(), h
        )
        begin = time.monotonic()
        try:
            response = urllib.request.urlopen(req, timeout=60)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            data = json.loads(response.read())
            result = (response.status, data, dict(response.headers))
        samples.append(
            {
                "path": path,
                "status": result[0],
                "seconds": time.monotonic() - begin,
                "response": data,
                "headers": result[2],
            }
        )
        return result

    assert request("/health", headers={"Host": "attacker.example"})[0] == 400
    assert request("/health", headers={"Origin": "https://attacker.example"})[0] == 400
    body = {
        "model": "gem16",
        "messages": [{"role": "user", "content": "Say hello."}],
        "reasoning_effort": "none",
        "max_completion_tokens": 32,
    }
    # The production binary must not expose the separate fault-test executable's hooks.
    assert request("/v1/chat/completions", body,
                   {"X-Gem16-Test-Fault": "acquired:exception"})[0] == 200
    assert (
        request("/v1/chat/completions", body, {"Content-Type": "text/plain"})[0] == 400
    )
    affinity = {"session_id": uuid.uuid4().hex}
    status, first, _ = request("/v1/chat/completions", body, affinity)
    assert status == 200
    body["messages"].append(first["choices"][0]["message"])
    body["messages"].append({"role": "user", "content": "Say goodbye."})
    status, warm, _ = request("/v1/chat/completions", body, affinity)
    assert status == 200 and warm["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
    # Compacted/forked full history must rebuild, never apply the old KV prefix.
    body["messages"] = [
        {"role": "user", "content": "This is a compacted summary. Say ready."}
    ]
    status, reset, headers = request("/v1/chat/completions", body, affinity)
    assert (
        status == 200 and reset["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    )
    assert headers.get("X-Gem16-Cache-Reset") == "history_or_tools_changed"
    assert (
        request(
            "/v1/chat/completions",
            body,
            {**affinity, "X-Gem16-Session-Id": "different"},
        )[0]
        == 400
    )
    body["tools"] = [
        {
            "type": "function",
            "function": {"name": "probe", "parameters": {"type": "object"}},
        }
    ]
    assert (
        request("/v1/chat/completions", body, affinity)[2].get("X-Gem16-Cache-Reset")
        == "history_or_tools_changed"
    )
    body.pop("tools")
    invalid = {
        **body,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "probe",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "$defs": {"x": {"$ref": "#/$defs/x"}},
                        "$ref": "#/$defs/x",
                    },
                },
            }
        ],
    }
    assert request("/v1/chat/completions", invalid)[0] == 400
    response_body = {
        "model": "gem16",
        "input": "Say hello.",
        "reasoning": {"effort": "none"},
        "max_output_tokens": 32,
    }
    status, output, _ = request("/v1/responses", response_body)
    assert status == 200
    response_body["input"] = [
        {"role": "user", "content": "Say hello."},
        *output["output"],
        {"role": "user", "content": "Say goodbye."},
    ]
    assert request("/v1/responses", response_body)[0] == 200
    # Run against a server configured with --max-queued-requests 2.
    long_body = {
        **body,
        "messages": [
            {
                "role": "user",
                "content": "Count from 1 to 1000, one integer per line. Do not stop early.",
            }
        ],
        "max_completion_tokens": 512,
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as workers:
        futures = [
            workers.submit(request, "/v1/chat/completions", long_body)
            for _ in range(16)
        ]
        time.sleep(0.1)
        begin = time.monotonic()
        assert request("/health")[0] == 200
        health_seconds = time.monotonic() - begin
        assert health_seconds < 1.0, health_seconds
        statuses = [future.result()[0] for future in futures]
    assert 503 in statuses, statuses
    assert request("/health")[1]["request_queue_active"] == 0
    assert request("/v1/chat/completions", body)[0] == 200
    return {
        "status": "passed",
        "scope": "bounded live host/API checks; no long-prefill cancellation latency claim",
        "health_under_saturation_seconds": health_seconds,
        "samples": samples,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as output:
        samples = []
        try:
            result = run(args.base_url, samples)
        except Exception as error:
            json.dump(
                {"status": "failed", "error": repr(error), "samples": samples},
                output,
                indent=2,
            )
            raise
        json.dump(result, output, indent=2)
        print(
            json.dumps(
                {key: value for key, value in result.items() if key != "samples"}
            )
        )

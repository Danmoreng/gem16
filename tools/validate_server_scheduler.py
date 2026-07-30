#!/usr/bin/env python3
"""Validate gem16 multi-session scheduling with the official OpenAI SDK."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata
import json
import time
import urllib.request

import openai
from openai import OpenAI


PINNED_OPENAI_VERSION = "2.50.0"


def metrics(base_url: str) -> dict[str, float]:
    root = base_url.removesuffix("/v1")
    with urllib.request.urlopen(f"{root}/metrics", timeout=10) as response:
        text = response.read().decode("utf-8")
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, value = line.split(maxsplit=1)
        values[name] = float(value)
    return values


def create(client: OpenAI, model: str, prompt: str, tokens: int = 24):
    return client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=tokens,
        reasoning={"effort": "none"},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="gem16")
    args = parser.parse_args()

    installed = importlib.metadata.version("openai")
    if installed != PINNED_OPENAI_VERSION:
        raise RuntimeError(
            f"openai-python {PINNED_OPENAI_VERSION} required, got {installed}"
        )
    client = OpenAI(base_url=args.base_url, api_key="gem16-local", timeout=120)

    first_a = create(client, args.model, "Reply with exactly: session alpha")
    first_b = create(client, args.model, "Reply with exactly: session beta")
    second_a = client.responses.create(
        model=args.model,
        previous_response_id=first_a.id,
        input="Now reply with exactly: alpha continued",
        max_output_tokens=24,
        reasoning={"effort": "none"},
    )
    safe_error_rejected = False
    try:
        client.responses.create(
            model=args.model,
            previous_response_id=second_a.id,
            input="This unsupported option must not poison the session.",
            parallel_tool_calls=False,
            max_output_tokens=8,
            reasoning={"effort": "none"},
        )
    except openai.BadRequestError:
        safe_error_rejected = True
    if not safe_error_rejected:
        raise RuntimeError("unsupported generation option was not rejected")
    third_a = client.responses.create(
        model=args.model,
        previous_response_id=second_a.id,
        input="Reply with exactly: alpha cache survived",
        max_output_tokens=24,
        reasoning={"effort": "none"},
    )
    first_c = create(client, args.model, "Reply with exactly: session gamma")

    stale_evicted = False
    try:
        client.responses.create(
            model=args.model,
            previous_response_id=first_b.id,
            input="This evicted chain must fail.",
            max_output_tokens=8,
        )
    except openai.NotFoundError:
        stale_evicted = True
    if not stale_evicted:
        raise RuntimeError("least-recently-used response chain was not evicted")

    def concurrent_root(label: str) -> str:
        local = OpenAI(
            base_url=args.base_url, api_key="gem16-local", timeout=120
        )
        response = create(
            local,
            args.model,
            f"Write four short sentences about {label}.",
            64,
        )
        return response.id

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(concurrent_root, "Berlin"),
            executor.submit(concurrent_root, "CUDA"),
        ]
        concurrent_ids = [future.result() for future in futures]
    if len(set(concurrent_ids)) != 2:
        raise RuntimeError("concurrent sessions did not receive unique IDs")

    cancellation_id = ""
    cancellation_event = ""
    single_flight_rejected = False
    with client.responses.stream(
        model=args.model,
        input="Count upward slowly from one and do not stop early.",
        max_output_tokens=512,
        reasoning={"effort": "none"},
    ) as stream:
        for event in stream:
            if event.type == "response.created":
                cancellation_id = event.response.id
                try:
                    client.responses.create(
                        model=args.model,
                        previous_response_id=cancellation_id,
                        input="A concurrent continuation must be rejected.",
                        max_output_tokens=8,
                    )
                except openai.APIStatusError as error:
                    single_flight_rejected = error.status_code == 503
                client.responses.cancel(cancellation_id)
            elif event.type == "error":
                cancellation_event = event.type
                break
    if not cancellation_id:
        raise RuntimeError("stream did not publish a cancellable response ID")
    if not single_flight_rejected:
        raise RuntimeError("concurrent request on one session was not rejected")

    deadline = time.monotonic() + 10
    observed = metrics(args.base_url)
    while (
        observed.get("gem16_cancellations_observed_total", 0) < 1
        and time.monotonic() < deadline
    ):
        time.sleep(0.1)
        observed = metrics(args.base_url)

    required = {
        "gem16_session_limit": 2,
        # A cancelled generation is deliberately discarded because its KV
        # cache can be partially advanced and therefore unsafe to reuse.
        "gem16_resident_sessions": 1,
    }
    for name, expected in required.items():
        if observed.get(name) != expected:
            raise RuntimeError(f"{name}={observed.get(name)}, expected {expected}")
    if observed.get("gem16_sessions_created_total", 0) < 6:
        raise RuntimeError("session creation metric did not advance")
    if observed.get("gem16_sessions_evicted_total", 0) < 4:
        raise RuntimeError("session eviction metric did not advance")
    if observed.get("gem16_cancellations_requested_total", 0) < 1:
        raise RuntimeError("cancellation request metric did not advance")
    if observed.get("gem16_cancellations_observed_total", 0) < 1:
        raise RuntimeError("generation did not observe cancellation")
    if observed.get("gem16_active_requests", -1) != 0:
        raise RuntimeError("active request metric did not return to zero")

    print(
        json.dumps(
            {
                "status": "ok",
                "openai_sdk": installed,
                "response_chains": [first_a.id, first_b.id, first_c.id],
                "continued_response": third_a.id,
                "safe_error_retained_cache": safe_error_rejected,
                "stale_evicted": stale_evicted,
                "concurrent_response_ids": concurrent_ids,
                "cancelled_response_id": cancellation_id,
                "cancellation_event": cancellation_event,
                "single_flight_rejected": single_flight_rejected,
                "metrics": observed,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

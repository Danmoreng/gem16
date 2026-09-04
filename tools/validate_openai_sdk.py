#!/usr/bin/env python3
"""Live official-SDK conformance probes; no OpenAI-hosted service is contacted."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import urllib.request
import urllib.error
from typing import Any

import openai
from openai import OpenAI

TOOL = {
    "type": "function",
    "name": "lookup",
    "description": "Look up the exact marker for a key. Never guess a marker.",
    "parameters": {
        "type": "object",
        "properties": {"key": {"type": "string", "enum": ["alpha", "beta"]}},
        "required": ["key"],
        "additionalProperties": False,
    },
    "strict": True,
}
PROMPT = "Use lookup for alpha and beta, one call per key. Both lookups are independent. Then report both exact markers."
RESULTS = {"alpha": "München-α-731", "beta": "東京-β-942"}


def chat(client: OpenAI, model: str, history: list, stream: bool, **extra: Any) -> dict:
    params = dict(
        model=model,
        messages=history,
        reasoning_effort="none",
        max_completion_tokens=256,
        **extra,
    )
    if not stream:
        response = client.chat.completions.create(**params)
        assert (
            response._request_id and response.usage and response.usage.total_tokens > 0
        )
        return {
            "message": response.choices[0].message.model_dump(exclude_none=True),
            "finish": response.choices[0].finish_reason,
            "usage": response.usage.model_dump(),
        }
    parts, calls, finish, usage = [], {}, None, None
    with client.chat.completions.create(
        **params, stream=True, stream_options={"include_usage": True}
    ) as chunks:
        for chunk in chunks:
            if chunk.usage:
                usage = chunk.usage.model_dump()
            for choice in chunk.choices:
                if choice.finish_reason:
                    finish = choice.finish_reason
                if choice.delta.content:
                    parts.append(choice.delta.content)
                for item in choice.delta.tool_calls or []:
                    call = calls.setdefault(
                        item.index,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if item.id:
                        call["id"] = item.id
                    if item.function:
                        call["function"]["name"] += item.function.name or ""
                        call["function"]["arguments"] += item.function.arguments or ""
    assert finish and usage and usage["total_tokens"] > 0
    message = {"role": "assistant", "content": "".join(parts) or None}
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    return {"message": message, "finish": finish, "usage": usage}


def responses(client: OpenAI, stream: bool, **params: Any):
    if not stream:
        r = client.responses.create(**params)
        assert r._request_id
    else:
        events, text = [], []
        with client.responses.stream(**params) as s:
            for event in s:
                events.append(event)
                if event.type == "response.output_text.delta":
                    text.append(event.delta)
            # openai-python 2.50.0 accumulates completed responses only.
            # Incomplete is a valid typed terminal event, not a completed response.
            r = (
                events[-1].response
                if events[-1].type == "response.incomplete"
                else s.get_final_response()
            )
        seq = [e.sequence_number for e in events]
        assert seq == sorted(set(seq)), seq
        assert events[0].type == "response.created"
        assert events[-1].type == "response." + r.status
        assert "".join(text) == r.output_text
        assert all(e.response.id == r.id for e in events if hasattr(e, "response"))
    assert r.usage and r.usage.total_tokens > 0
    assert (r.completed_at is None) == (r.status != "completed")
    return r


def tool_loop(
    client: OpenAI, model: str, api: str, stream: bool, multi: bool = True
) -> dict:
    history = [
        {
            "role": "user",
            "content": PROMPT
            if multi
            else "First use lookup for alpha. Only after receiving its result, use lookup for beta. Then report both exact markers.",
        }
    ]
    seen, response_ids, parallel = set(), [], False
    tools = (
        [TOOL]
        if api == "responses"
        else [
            {
                "type": "function",
                "function": {k: v for k, v in TOOL.items() if k != "type"},
            }
        ]
    )
    previous = None
    for turn in range(5):
        if api == "chat":
            r = chat(
                client, model, history, stream, tools=tools, parallel_tool_calls=True
            )
            message = r["message"]
            history.append(message)
            calls = message.get("tool_calls", [])
            answer = message.get("content") or ""
            normalized = [
                (c["id"], c["function"]["name"], c["function"]["arguments"])
                for c in calls
            ]
            if calls:
                assert r["finish"] == "tool_calls"
            else:
                assert r["finish"] == "stop", r
        else:
            params = dict(
                model=model,
                input=history,
                tools=tools,
                parallel_tool_calls=True,
                reasoning={"effort": "none"},
                max_output_tokens=256,
            )
            if previous:
                params["previous_response_id"] = previous
            r = responses(client, stream, **params)
            assert r.status == "completed", r
            response_ids.append(r.id)
            previous = r.id
            normalized = [
                (c.call_id, c.name, c.arguments)
                for c in r.output
                if c.type == "function_call"
            ]
            answer = r.output_text
            history = []
        parallel |= len(normalized) > 1
        if not normalized:
            assert seen == set(RESULTS), seen
            if not multi:
                assert turn >= 2, "sequential tool rounds were not observed"
            assert all(marker in answer for marker in RESULTS.values()), answer
            if api == "responses":
                try:
                    client.responses.create(
                        model=model,
                        previous_response_id=response_ids[0],
                        input="Reject this stale branch.",
                    )
                except openai.NotFoundError:
                    pass
                else:
                    raise AssertionError("stale response accepted")
            return {
                "turns": turn + 1,
                "parallel_calls_observed": parallel,
                "keys": sorted(seen),
                "answer": answer,
            }
        assert len({c[0] for c in normalized}) == len(normalized)
        for call_id, name, arguments in normalized:
            arg = json.loads(arguments)
            assert (
                call_id
                and name == "lookup"
                and set(arg) == {"key"}
                and arg["key"] in RESULTS
            )
            key = arg["key"]
            seen.add(key)
            # Long tool result with Unicode; the final sentinel is the only requested fact.
            result = json.dumps(
                {"notes": ["irrelevant padding"] * 128, "marker": RESULTS[key]},
                ensure_ascii=False,
            )
            if api == "chat":
                history.append(
                    {"role": "tool", "tool_call_id": call_id, "content": result}
                )
            else:
                history.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result,
                    }
                )
    raise AssertionError("tool loop exceeded five turns")


def parallel_history(client: OpenAI, model: str, api: str, stream: bool) -> dict:
    """Protocol fixture: two pending calls/results, independent of model call scheduling."""
    calls = [
        {
            "id": "call_" + key,
            "type": "function",
            "function": {"name": "lookup", "arguments": json.dumps({"key": key})},
        }
        for key in RESULTS
    ]
    if api == "chat":
        history = [
            {"role": "user", "content": PROMPT},
            {"role": "assistant", "content": None, "tool_calls": calls},
        ]
        history.extend(
            {"role": "tool", "tool_call_id": c["id"], "content": RESULTS[key]}
            for c, key in zip(calls, RESULTS)
        )
        tools = [
            {
                "type": "function",
                "function": {k: v for k, v in TOOL.items() if k != "type"},
            }
        ]
        r = chat(client, model, history, stream, tools=tools, parallel_tool_calls=True)
        assert r["finish"] == "stop"
        answer = r["message"]["content"]
    else:
        history = [{"role": "user", "content": PROMPT}]
        history.extend(
            {"type": "function_call", "call_id": c["id"], **c["function"]}
            for c in calls
        )
        history.extend(
            {"type": "function_call_output", "call_id": c["id"], "output": RESULTS[key]}
            for c, key in zip(calls, RESULTS)
        )
        r = responses(
            client,
            stream,
            model=model,
            input=history,
            tools=[TOOL],
            reasoning={"effort": "none"},
            max_output_tokens=256,
        )
        assert r.status == "completed"
        answer = r.output_text
    assert all(marker in answer for marker in RESULTS.values()), answer
    return {"fixture_call_ids": [c["id"] for c in calls], "answer": answer}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    p.add_argument("--model", default="gem16")
    args = p.parse_args()
    assert openai.__version__ == "2.50.0", openai.__version__
    client = OpenAI(
        base_url=args.base_url,
        api_key="gem16-local-validation",
        max_retries=0,
        timeout=120,
    )
    cases = []

    def run(name, fn):
        try:
            cases.append({"name": name, "status": "passed", "detail": fn()})
        except Exception as exc:
            cases.append({"name": name, "status": "failed", "error": str(exc)})

    for api in ["chat", "responses"]:
        for stream in [False, True]:
            run(
                f"{api}-tools-{stream}",
                lambda api=api, stream=stream: tool_loop(
                    client, args.model, api, stream
                ),
            )
            run(
                f"{api}-sequential-tools-{stream}",
                lambda api=api, stream=stream: tool_loop(
                    client, args.model, api, stream, False
                ),
            )

            run(
                f"{api}-parallel-history-{stream}",
                lambda api=api, stream=stream: parallel_history(
                    client, args.model, api, stream
                ),
            )

            def limit(api=api, stream=stream):
                if api == "chat":
                    # Explicitly override the helper's regular output cap through a direct SDK call.
                    r = client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "user", "content": "Write a long story."}],
                        max_completion_tokens=1,
                        reasoning_effort="none",
                        stream=stream,
                    )
                    if stream:
                        with r:
                            finishes = [
                                c.finish_reason
                                for chunk in r
                                for c in chunk.choices
                                if c.finish_reason
                            ]
                        assert finishes == ["length"], finishes
                    else:
                        assert r.choices[0].finish_reason == "length"
                else:
                    r = responses(
                        client,
                        stream,
                        model=args.model,
                        input="Write a long story.",
                        max_output_tokens=1,
                        reasoning={"effort": "none"},
                    )
                    assert (
                        r.status == "incomplete"
                        and r.incomplete_details.reason == "max_output_tokens"
                    )
                return {
                    "expected": "length" if api == "chat" else "response.incomplete"
                }

            run(f"{api}-output-limit-{stream}", limit)
    for api in ["chat", "responses"]:

        def reject(api=api):
            try:
                if api == "chat":
                    client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "user", "content": "Hello"}],
                        temperature=0.5,
                    )
                else:
                    client.responses.create(
                        model=args.model, input="Hello", store=False
                    )
            except openai.BadRequestError as exc:
                assert exc.request_id and isinstance(exc.body, dict)
                return {"http_status": exc.status_code, "error": exc.body}
            raise AssertionError("unsupported field accepted")

        run(f"{api}-unsupported", reject)

    def metrics():
        with urllib.request.urlopen(
            args.base_url.removesuffix("/v1") + "/metrics", timeout=5
        ) as r:
            return {
                parts[0]: float(parts[1])
                for line in r.read().decode().splitlines()
                if line and not line.startswith("#") and len(parts := line.split()) == 2
            }

    def transport_error():
        request = urllib.request.Request(
            args.base_url + "/responses",
            data=b"{broken",
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(request, timeout=5)
        except urllib.error.HTTPError as e:
            assert e.code == 400 and e.headers.get("X-Request-Id")
            assert "error" in json.load(e)
            return {"http_status": e.code}
        raise AssertionError("malformed JSON accepted")

    run("malformed-json", transport_error)

    def overflow():
        try:
            client.responses.create(
                model=args.model,
                input="word " * 20000,
                max_output_tokens=1,
                reasoning={"effort": "none"},
            )
        except openai.BadRequestError as e:
            assert "context capacity" in str(e)
            return {"http_status": e.status_code}
        raise AssertionError(
            "expected context overflow; run this matrix at 16,384 context"
        )

    run("context-overflow", overflow)

    def cancel_or_disconnect(disconnect):
        counter = (
            "gem16_client_disconnects_total"
            if disconnect
            else "gem16_cancellations_observed_total"
        )
        before = metrics()[counter]
        response_id = None
        saw_delta = False
        terminal = None
        with client.responses.stream(
            model=args.model,
            input="Count upward from one to one thousand, writing each number.",
            max_output_tokens=2048,
            reasoning={"effort": "none"},
        ) as stream:
            for event in stream:
                if event.type == "response.created":
                    response_id = event.response.id
                elif event.type == "response.output_text.delta" and not saw_delta:
                    saw_delta = True
                    if disconnect:
                        break
                    client.responses.cancel(response_id)
                elif event.type == "error":
                    terminal = event.type
                    break
        assert response_id and saw_delta
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            observed = metrics()
            if observed[counter] > before and observed["gem16_active_requests"] == 0:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("generation did not observe cancellation/disconnect")
        if not disconnect:
            assert terminal == "error"
        recovery = client.responses.create(
            model=args.model,
            input="Reply with OK.",
            max_output_tokens=24,
            reasoning={"effort": "none"},
        )
        assert recovery.status == "completed" and recovery.output_text
        return {"cancellation_observed": True, "recovery_completed": True}

    run("cancel-and-recover", lambda: cancel_or_disconnect(False))
    run("disconnect-and-recover", lambda: cancel_or_disconnect(True))
    report = {
        "sdk": "openai-python",
        "version": openai.__version__,
        "platform": platform.system(),
        "cases": cases,
        "status": "passed" if all(c["status"] == "passed" for c in cases) else "failed",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

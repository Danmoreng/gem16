#!/usr/bin/env python3
"""Run a real function-tool loop through gem16 with the official OpenAI SDK."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from typing import Any

from openai import OpenAI


PINNED_OPENAI_VERSION = "2.50.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="gem16")
    parser.add_argument("--no-stream", action="store_true")
    return parser.parse_args()


def tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country",
                    }
                },
                "required": ["location"],
                "additionalProperties": False,
            },
        },
    }


def streamed_completion(
    client: OpenAI, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> tuple[dict[str, Any], str]:
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        reasoning_effort="none",
        max_completion_tokens=96,
        stream=True,
        stream_options={"include_usage": True},
    )
    content_parts: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    finish_reason = ""
    usage_seen = False
    for chunk in stream:
        if chunk.usage is not None:
            usage_seen = True
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.finish_reason is not None:
            finish_reason = choice.finish_reason
        if choice.delta.content:
            content_parts.append(choice.delta.content)
        for delta in choice.delta.tool_calls or []:
            call = calls.setdefault(
                delta.index,
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            if delta.id:
                call["id"] = delta.id
            if delta.function is not None:
                if delta.function.name:
                    call["function"]["name"] = delta.function.name
                if delta.function.arguments:
                    call["function"]["arguments"] += delta.function.arguments
    if not usage_seen:
        raise AssertionError("stream did not contain the requested usage chunk")
    content = "".join(content_parts)
    assistant: dict[str, Any] = {
        "role": "assistant",
        "content": content or None,
    }
    if calls:
        assistant["tool_calls"] = [calls[index] for index in sorted(calls)]
    return assistant, finish_reason


def nonstream_completion(
    client: OpenAI, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> tuple[dict[str, Any], str]:
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        reasoning_effort="none",
        max_completion_tokens=96,
    )
    choice = completion.choices[0]
    return choice.message.model_dump(exclude_none=True), choice.finish_reason


def main() -> int:
    args = parse_args()
    installed = importlib.metadata.version("openai")
    if installed != PINNED_OPENAI_VERSION:
        print(
            f"error: openai=={PINNED_OPENAI_VERSION} is required, found {installed}",
            file=sys.stderr,
        )
        return 2
    client = OpenAI(base_url=args.base_url, api_key="gem16-local-test")
    tools = [tool_definition()]
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": "What is the current weather in Berlin? Use the weather tool.",
        }
    ]
    complete = nonstream_completion if args.no_stream else streamed_completion
    assistant, first_finish = complete(client, args.model, messages, tools)
    calls = assistant.get("tool_calls") or []
    if first_finish != "tool_calls" or len(calls) != 1:
        raise AssertionError(
            f"expected one tool call, got finish={first_finish!r}, calls={calls!r}"
        )
    call = calls[0]
    if call["function"]["name"] != "get_weather":
        raise AssertionError(f"unexpected function: {call!r}")
    arguments = json.loads(call["function"]["arguments"])
    if "berlin" not in arguments.get("location", "").lower():
        raise AssertionError(f"tool arguments are not grounded in Berlin: {arguments!r}")

    messages.append(assistant)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "content": "Sunny, 25 C",
        }
    )
    final_assistant, second_finish = complete(client, args.model, messages, tools)
    answer = final_assistant.get("content") or ""
    if second_finish != "stop":
        raise AssertionError(f"final response did not stop normally: {second_finish!r}")
    normalized = answer.lower().replace("°", "")
    if "25" not in normalized or "sun" not in normalized:
        raise AssertionError(f"final answer did not use the tool result: {answer!r}")

    print(
        json.dumps(
            {
                "status": "ok",
                "openai_sdk": installed,
                "stream": not args.no_stream,
                "tool_call": call,
                "final_answer": answer,
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

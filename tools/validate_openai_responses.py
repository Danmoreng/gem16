#!/usr/bin/env python3
"""Validate gem16 Responses API state and tool continuation via openai-python."""

from __future__ import annotations

import argparse
import json

import openai
from openai import OpenAI


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="gem16")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="gem16-local")
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]

    event_types: list[str] = []
    with client.responses.stream(
        model=args.model,
        input="What is the weather in Berlin? Use the weather tool.",
        tools=tools,
        max_output_tokens=96,
        reasoning={"effort": "none"},
    ) as stream:
        for event in stream:
            event_types.append(event.type)
        first = stream.get_final_response()

    calls = [item for item in first.output if item.type == "function_call"]
    if len(calls) != 1:
        raise RuntimeError(f"expected one function call, got {len(calls)}")
    call = calls[0]
    arguments = json.loads(call.arguments)
    if call.name != "get_weather" or "Berlin" not in arguments.get(
        "location", ""
    ):
        raise RuntimeError(f"unexpected function call: {call}")

    final = client.responses.create(
        model=args.model,
        previous_response_id=first.id,
        input=[
            {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": "Sunny, 25 C",
            }
        ],
        tools=tools,
        max_output_tokens=96,
        reasoning={"effort": "none"},
    )
    if "25" not in final.output_text and "sun" not in final.output_text.lower():
        raise RuntimeError(f"final answer is not grounded: {final.output_text!r}")
    required_events = {
        "response.created",
        "response.output_item.added",
        "response.output_item.done",
        "response.completed",
    }
    if not required_events.issubset(event_types):
        raise RuntimeError(f"missing stream events: {required_events - set(event_types)}")
    stale_rejected = False
    try:
        client.responses.create(
            model=args.model,
            previous_response_id=first.id,
            input="This branch must be rejected.",
        )
    except openai.NotFoundError:
        stale_rejected = True
    if not stale_rejected:
        raise RuntimeError("stale previous_response_id was not rejected")
    print(
        json.dumps(
            {
                "status": "ok",
                "openai_sdk": openai.__version__,
                "first_response_id": first.id,
                "call_id": call.call_id,
                "events": event_types,
                "final_response_id": final.id,
                "final_answer": final.output_text,
                "stale_previous_response_rejected": stale_rejected,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

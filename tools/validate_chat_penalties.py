#!/usr/bin/env python3
"""Bounded local Chat penalty, resident-cache and ordinary/D2 evidence.

Runs only already installed immutable profiles. Each output directory must be new.
The baseline mode records zero-penalty timing/output before native changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from urllib import request
from urllib.error import HTTPError
from threading import Thread, Event

from hf_cache import default_target_model, default_assistant_model, locked_snapshot_path

ROOT = Path(__file__).resolve().parents[1]


def api(base, path, payload=None, session=None):
    headers = {"Content-Type": "application/json"}
    if session:
        headers["X-Gem16-Session-Id"] = session
    body = None if payload is None else json.dumps(payload).encode()
    started = time.perf_counter()
    with request.urlopen(request.Request(base + path, body, headers), timeout=120) as r:
        value = json.load(r)
        response_headers = dict(r.headers)
    return value, time.perf_counter() - started, response_headers


def streamed_chat(base, payload):
    body = json.dumps({**payload, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    with request.urlopen(request.Request(base + "/v1/chat/completions", body,
                         {"Content-Type": "application/json"}), timeout=120) as response:
        assert "text/event-stream" in response.headers.get("Content-Type", "")
        events = [line[6:].strip() for line in response.read().decode().splitlines()
                  if line.startswith("data: ")]
    assert events and events[-1] == "[DONE]", events
    parsed = [json.loads(event) for event in events[:-1]]
    assert any(event.get("usage") for event in parsed), parsed
    assert any(choice.get("finish_reason") for event in parsed
               for choice in event.get("choices", [])), parsed
    return parsed


def lifecycle_checks(base):
    history = [{"role": "user", "content": "Use the add tool to calculate 2 plus 3, then tell me the result."}]
    tools = [{"type": "function", "function": {"name": "add", "description": "Add two numbers. Use this tool for arithmetic.",
        "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                       "required": ["a", "b"], "additionalProperties": False}}}]
    first, _, _ = api(base, "/v1/chat/completions", {
        "model": "gem16", "messages": history, "tools": tools, "max_completion_tokens": 96,
        "reasoning_effort": "none", "frequency_penalty": 0.5, "presence_penalty": 0.25}, "penalty-tools")
    assistant = first["choices"][0]["message"]
    calls = assistant.get("tool_calls", [])
    assert calls, first
    history.append(assistant)
    for call in calls:
        assert call["function"]["name"] == "add", call
        arguments = json.loads(call["function"]["arguments"])
        history.append({"role": "tool", "tool_call_id": call["id"],
                        "content": str(arguments["a"] + arguments["b"])})
    second, _, headers = api(base, "/v1/chat/completions", {
        "model": "gem16", "messages": history, "tools": tools, "max_completion_tokens": 48,
        "reasoning_effort": "none", "frequency_penalty": 0, "presence_penalty": -0.25}, "penalty-tools")
    assert "X-Gem16-Cache-Reset" not in headers and second["usage"]["prompt_tokens_details"]["cached_tokens"] > 0, second
    body = json.dumps({"model": "gem16", "messages": [{"role": "user", "content": "Explain rain in great detail for 1000 words."}],
                       "max_completion_tokens": 256, "stream": True,
                       "reasoning_effort": "none", "frequency_penalty": 0.5}).encode()
    with request.urlopen(request.Request(base + "/v1/chat/completions", body,
                         {"Content-Type": "application/json"}), timeout=120) as response:
        while True:
            line = response.readline().decode()
            assert line, "stream ended before cancellation fixture"
            if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                event = json.loads(line[6:])
                if any(choice.get("delta", {}).get("content") for choice in event.get("choices", [])):
                    break
    # Closing after actual content exercises disconnect cancellation. A following
    # generation must acquire the slot and complete, without restarting the server.
    recovered, _, _ = api(base, "/v1/chat/completions", {
        "model": "gem16", "messages": [{"role": "user", "content": "Say hello."}],
        "max_completion_tokens": 16, "reasoning_effort": "none"})
    return {"tool_call": first, "tool_result": second, "post_disconnect_response": recovered}


def observe_gpu(stop, samples):
    while not stop.is_set():
        try:
            sample = subprocess.check_output([
                "nvidia-smi", "--query-gpu=memory.used,power.draw,clocks.sm,temperature.gpu",
                "--format=csv,noheader,nounits"], text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0).strip()
            samples.append({"monotonic": time.monotonic(), "memory_power_clock_temperature": sample})
        except Exception as error:
            samples.append({"error": str(error)})
        stop.wait(0.2)


def profile_arguments(profile, d2):
    if profile == "12b":
        target, assistant, vision = default_target_model(), default_assistant_model(), None
    else:
        target = locked_snapshot_path(ROOT / "models" / (
            "gemma4-26b-trellis35-target.lock.json" if profile == "compact"
            else "gemma4-26b-gem16-target.lock.json"))
        assistant = locked_snapshot_path(ROOT / "models/gemma4-26b-gem16-assistant.lock.json")
        vision = locked_snapshot_path(ROOT / "models/gemma4-26b-vision-fp8.lock.json") if profile == "compact" else None
    for directory in (target, assistant if d2 else None, vision):
        if directory is not None and not directory.is_dir():
            raise RuntimeError(f"installed model missing: {directory}")
    result = ["--model", str(target)]
    if vision:
        result += ["--vision-model", str(vision)]
    if d2:
        result += ["--assistant-model", str(assistant), "--mtp-draft-tokens", "2"]
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--lifecycle", action="store_true")
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--source-patch", type=Path)
    p.add_argument("--profiles", nargs="+", choices=["12b", "compact", "nvfp4"], default=["12b", "compact", "nvfp4"])
    args = p.parse_args()
    if not 1 <= args.warmups <= 20 or not 1 <= args.samples <= 100:
        p.error("warmups must be 1..20 and samples 1..100")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "platform": platform.platform(), "python": sys.version,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_diff_sha256": hashlib.sha256(args.source_patch.read_bytes() if args.source_patch else subprocess.check_output(["git", "diff", "--binary", "--no-ext-diff"], cwd=ROOT)).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(args.server.read_bytes()).hexdigest(),
        "locks": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT / "models").glob("*.lock.json")},
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip(),
        "scope": f"{args.warmups} warmups + {args.samples} measured sequential HTTP nonstream requests, 96 output tokens; HTTP end-to-end timing",
        "runs": [],
    }
    for profile in args.profiles:
        for d2 in (False, True):
            label = f"{profile}-{'d2' if d2 else 'ordinary'}"
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            base = f"http://127.0.0.1:{port}"
            command = [str(args.server.resolve()), *profile_arguments(profile, d2),
                       "--host", "127.0.0.1", "--port", str(port), "--model-name", "gem16",
                       "--max-context", "4096", "--max-sessions", "1", "--seed", "42"]
            if args.greedy:
                command.append("--greedy")
            current = {"profile": label, "command": command, "samples": [], "gpu_observations": []}
            stop_monitor = Event()
            monitor = Thread(target=observe_gpu, args=(stop_monitor, current["gpu_observations"]), daemon=True)
            monitor.start()
            with (args.output_dir / f"{label}.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=log, stderr=log,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
                try:
                    deadline = time.monotonic() + 240
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(f"{label}: server exited {process.returncode}; see log")
                        try:
                            current["health"] = api(base, "/health")[0]
                            if current["health"].get("status") == "ok":
                                break
                        except Exception:
                            pass
                        if time.monotonic() > deadline:
                            raise RuntimeError(f"{label}: readiness timeout")
                        time.sleep(0.2)
                    for index in range(args.warmups + args.samples):
                        payload = {"model": "gem16", "messages": [{"role": "user", "content":
                            "Write a detailed 1000-word explanation of how rain forms. Begin immediately and use ordinary prose."}],
                            "max_completion_tokens": 96, "reasoning_effort": "none",
                            "frequency_penalty": 0, "presence_penalty": 0}
                        response, elapsed, headers = api(base, "/v1/chat/completions", payload)
                        current["samples"].append({"warmup": index < args.warmups, "seconds": elapsed,
                                                  "response": response, "headers": headers})
                    if args.greedy:
                        current["greedy_rejections"] = []
                        for field in ("frequency_penalty", "presence_penalty"):
                            try:
                                api(base, "/v1/chat/completions", {
                                    "model": "gem16", "messages": [{"role": "user", "content": "Hello"}],
                                    "stream": True, field: 0.5})
                                raise AssertionError("greedy accepted an active penalty")
                            except HTTPError as error:
                                content_type = error.headers.get("Content-Type", "")
                                body = json.load(error)
                                assert error.code == 400 and "application/json" in content_type, body
                                assert "penalties require sampling" in body["error"]["message"], body
                                current["greedy_rejections"].append({"field": field, "body": body})
                    if not args.baseline and not args.greedy:
                        history = [{"role": "user", "content": "Write the word rain repeatedly for 40 words."}]
                        current["turns"] = []
                        for frequency, presence in ((0.5, 0), (0.5, 0), (0, -0.5), (0, 0)):
                            response, elapsed, headers = api(base, "/v1/chat/completions", {
                                "model": "gem16", "messages": history, "max_completion_tokens": 48,
                                "reasoning_effort": "none", "frequency_penalty": frequency,
                                "presence_penalty": presence}, session="penalty-continuation")
                            assert "X-Gem16-Cache-Reset" not in headers, headers
                            if current["turns"]:
                                assert response["usage"]["prompt_tokens_details"]["cached_tokens"] > 0, response
                            current["turns"].append({"frequency": frequency, "presence": presence,
                                                     "response": response, "headers": headers, "seconds": elapsed})
                            history += [response["choices"][0]["message"], {"role": "user", "content": "Continue for another 40 words."}]
                        current["streaming_penalty_events"] = streamed_chat(base, {
                            "model": "gem16", "messages": [{"role": "user", "content": "Explain rain in one sentence."}],
                            "reasoning_effort": "low", "max_completion_tokens": 64,
                            "frequency_penalty": 0.5, "presence_penalty": 0.25})
                    if args.lifecycle:
                        current["lifecycle"] = lifecycle_checks(base)
                    current["memory_after_requests"] = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"], text=True).strip()
                    current["final_health"] = api(base, "/health")[0]
                finally:
                    stop_monitor.set()
                    monitor.join(timeout=5)
                    current["sampled_peak_gpu_memory_mib"] = max(
                        (float(sample["memory_power_clock_temperature"].split(",")[0])
                         for sample in current["gpu_observations"]
                         if "memory_power_clock_temperature" in sample), default=None)
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=15)
            report["runs"].append(current)
            (args.output_dir / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"{label}: passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

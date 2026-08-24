#!/usr/bin/env python3
"""Optional real-model product acceptance for M22 and protected 12B behavior."""

from __future__ import annotations

import base64
import contextlib
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib import error, request


SKIP = 77
M08_ARTIFACT_SHA256 = (
    "471805f7dad8abb84300be78b2822a63dcb1d35bff5aa98426a162cc8532ee17"
)
M08_SOURCE_LOCK_SHA256 = (
    "3d9441fdebef33785e33181c335338208b8bf868cb8c7da692fd9c765cca8230"
)
M08_COMPILER_COMMIT = "f433358b8e2c1250b95801fc898faee4fcedcbe5"


class ProductFailure(RuntimeError):
    pass


def fail(message: str) -> None:
    raise ProductFailure(message)


def run(command: list[str], timeout: int = 600) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=timeout
    )
    if completed.returncode != 0:
        fail(
            f"command exited {completed.returncode}: {' '.join(command)}\n"
            f"stdout: {completed.stdout.strip()}\n"
            f"stderr: {completed.stderr.strip()}"
        )
    return completed


def parse_json_output(completed: subprocess.CompletedProcess[str], label: str) -> dict[str, Any]:
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        fail(f"{label} emitted invalid JSON: {exc}: {completed.stdout!r}")
    if not isinstance(value, dict):
        fail(f"{label} JSON root is not an object")
    return value


def evidence_directory() -> Path | None:
    value = os.environ.get("GEM16_M22_RAW_DIR")
    if not value:
        return None
    directory = Path(value).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_evidence(name: str, payload: dict[str, Any]) -> None:
    directory = evidence_directory()
    if directory is None:
        return
    (directory / name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def http_call(
    port: int,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> tuple[int, bytes, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    merged = {"Accept": "application/json"}
    if body is not None:
        merged["Content-Type"] = "application/json"
    if headers:
        merged.update(headers)
    outgoing = request.Request(
        f"http://127.0.0.1:{port}{path}", data=body, headers=merged, method=method
    )
    try:
        with request.urlopen(outgoing, timeout=timeout) as response:
            return response.status, response.read(), response.headers
    except error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers


def json_call(*args: Any, **kwargs: Any) -> tuple[int, dict[str, Any], Any]:
    status, body, headers = http_call(*args, **kwargs)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        fail(f"HTTP {status} returned invalid JSON: {exc}: {body!r}")
    if not isinstance(payload, dict):
        fail(f"HTTP {status} JSON root is not an object")
    return status, payload, headers


class Server:
    def __init__(
        self, executable: Path, model: Path, context: int, sessions: int
    ) -> None:
        self.port = free_port()
        self.process = subprocess.Popen(
            [
                str(executable),
                "--model",
                str(model),
                "--model-name",
                "m22-test",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--max-context",
                str(context),
                "--max-sessions",
                str(sessions),
                "--greedy",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def wait(self, timeout: float = 360.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                fail(
                    f"server exited {self.process.returncode} before health\n"
                    f"stdout: {stdout}\nstderr: {stderr}"
                )
            try:
                status, payload, _ = json_call(
                    self.port, "GET", "/health", timeout=1.0
                )
                if status == 200:
                    return payload
            except (OSError, ProductFailure):
                pass
            time.sleep(0.25)
        fail("server did not become healthy before timeout")

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.communicate(timeout=10)

    def __enter__(self) -> "Server":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def metric_value(text: str, name: str) -> int:
    prefix = name + " "
    values = [line[len(prefix) :] for line in text.splitlines() if line.startswith(prefix)]
    if len(values) != 1:
        fail(f"metric {name} is missing or duplicated")
    try:
        return int(values[0])
    except ValueError as exc:
        fail(f"metric {name} is not an integer: {values[0]!r}")
        raise AssertionError from exc


def tiny_bmp_data_url() -> str:
    pixel = bytes((255, 0, 0, 0))
    file_size = 14 + 40 + len(pixel)
    header = struct.pack("<2sIHHI", b"BM", file_size, 0, 0, 54)
    dib = struct.pack(
        "<IIIHHIIIIII", 40, 1, 1, 1, 24, 0, len(pixel), 2835, 2835, 0, 0
    )
    return "data:image/bmp;base64," + base64.b64encode(header + dib + pixel).decode()


def assistant_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        fail("chat completion does not contain one choice")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        fail("chat completion has no assistant text")
    return {"role": "assistant", "content": message["content"]}


def run_26b(driver: Path, chat: Path, server_executable: Path) -> int:
    model_value = os.environ.get("GEM16_26B_COMPILED_MODEL")
    if not model_value:
        print("SKIP: GEM16_26B_COMPILED_MODEL is not set", file=sys.stderr)
        return SKIP
    model = Path(model_value).resolve()
    if not model.is_dir() or not (model / "gem16_compilation.json").is_file():
        fail(f"invalid GEM16_26B_COMPILED_MODEL: {model}")

    report = parse_json_output(
        run(
            [
                str(chat),
                "--model",
                str(model),
                "--max-context",
                "32768",
                "--print-model-report",
            ]
        ),
        "chat model report",
    )
    expected_report = {
        "model_variant": "gemma4_moe_26b_a4b",
        "artifact_profile": "sm120-text-hybrid-v1",
        "head_format": "nvfp4",
        "artifact_content_sha256": M08_ARTIFACT_SHA256,
        "source_lock_sha256": M08_SOURCE_LOCK_SHA256,
        "compiler_commit": M08_COMPILER_COMMIT,
        "native_path": "sm120_integrated_nvfp4_moe_bf16_tensor_router_fp8_kv",
        "text_only": True,
        "supports_mtp": False,
        "default_context": 32768,
        "qualified_64k": False,
        "base_max_context": 32768,
        "mtp_max_context": None,
    }
    for key, expected in expected_report.items():
        if report.get(key) != expected:
            fail(f"chat model report {key}={report.get(key)!r}, expected {expected!r}")
    for key in ("resident_weight_bytes", "kv_cache_bytes", "workspace_bytes"):
        if not isinstance(report.get(key), int) or report[key] <= 0:
            fail(f"chat model report has invalid {key}")
    if report.get("required_admission_margin_bytes") != 700 * 1024 * 1024:
        fail("chat model report has the wrong 32K reserve")
    if report.get("admission_free_bytes", 0) < report["required_admission_margin_bytes"]:
        fail("chat model report does not satisfy its admission reserve")

    one_shot = parse_json_output(
        run(
            [
                str(chat),
                "--model",
                str(model),
                "--message",
                "Reply with one short word.",
                "--max-tokens",
                "1",
                "--max-context",
                "32768",
                "--thinking-budget",
                "off",
                "--greedy",
                "--json",
            ]
        ),
        "26B chat one-shot",
    )
    if not isinstance(one_shot.get("output_token_ids"), list) or len(
        one_shot["output_token_ids"]
    ) != 1:
        fail("26B chat one-shot did not produce exactly one token")
    media_rejection = subprocess.run(
        [
            str(chat),
            "--model",
            str(model),
            "--message",
            "describe",
            "--image",
            "/definitely/not/read.bmp",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if media_rejection.returncode == 0 or "text-only" not in media_rejection.stderr:
        fail("26B chat CLI did not reject media before reading the file")

    raw_directory = evidence_directory()
    temp_context: Any = (
        contextlib.nullcontext(raw_directory)
        if raw_directory is not None
        else tempfile.TemporaryDirectory(prefix="gem16-m22-product-")
    )
    with temp_context as temp_value:
        temp = Path(temp_value)
        driver_reports: list[dict[str, Any]] = []
        for name in ("first", "relaunch"):
            output = temp / f"{name}.json"
            run(
                [
                    str(driver),
                    "--model",
                    str(model),
                    "--output",
                    str(output),
                    "--context",
                    "32768",
                ]
            )
            driver_reports.append(json.loads(output.read_text(encoding="utf-8")))
        for payload in driver_reports:
            if payload.get("passed") is not True:
                fail("M22 direct product driver did not pass")
            for key in (
                "second_slot_rejected",
                "unsupported_mtp_rejected",
                "unsupported_media_rejected",
                "unsupported_vision_rejected",
                "prefix_mismatch_rejected",
                "cancelled_session_poisoned",
                "post_cancellation_relaunch_passed",
            ):
                if payload.get(key) is not True:
                    fail(f"M22 direct product driver did not prove {key}")
        for key in (
            "first_output_token_ids",
            "first_output_token_checksum",
            "second_output_token_ids",
            "second_output_token_checksum",
            "relaunch_output_token_ids",
        ):
            if driver_reports[0].get(key) != driver_reports[1].get(key):
                fail(f"fresh M22 driver processes disagree on {key}")

    with Server(server_executable, model, 32768, 1) as server:
        health = server.wait()
        for key, expected in expected_report.items():
            if key == "supports_mtp":
                capabilities = health.get("capabilities")
                actual = (
                    capabilities.get("mtp")
                    if isinstance(capabilities, dict)
                    else None
                )
                server_key = "capabilities.mtp"
            else:
                server_key = "weight_profile" if key == "artifact_profile" else key
                actual = health.get(server_key)
            if actual != expected:
                fail(f"server health {server_key}={actual!r}, expected {expected!r}")
        if health.get("session_limit") != 1:
            fail("26B server did not enforce one resident session")
        if health.get("required_admission_margin_bytes") != 700 * 1024 * 1024:
            fail("26B server health has wrong 32K admission reserve")
        if health.get("admission_free_bytes", 0) < health[
            "required_admission_margin_bytes"
        ]:
            fail("26B server health does not satisfy its reserve")

        first_messages = [{"role": "user", "content": "Reply with one word."}]
        status, first, headers = json_call(
            server.port,
            "POST",
            "/v1/chat/completions",
            {
                "model": "m22-test",
                "messages": first_messages,
                "max_completion_tokens": 2,
                "reasoning_effort": "none",
            },
            {"X-Gem16-Session-Id": "resident-a"},
            timeout=180.0,
        )
        if status != 200 or headers.get("X-Gem16-Session-Id") != "resident-a":
            fail(f"26B first server turn failed with HTTP {status}")
        messages = first_messages + [assistant_message(first)] + [
            {"role": "user", "content": "Now answer with a different word."}
        ]
        status, second, _ = json_call(
            server.port,
            "POST",
            "/v1/chat/completions",
            {
                "model": "m22-test",
                "messages": messages,
                "max_completion_tokens": 2,
                "reasoning_effort": "none",
            },
            {"X-Gem16-Session-Id": "resident-a"},
            timeout=180.0,
        )
        if status != 200:
            fail(f"26B resident continuation failed with HTTP {status}")
        messages += [assistant_message(second)]
        media_messages = messages + [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": tiny_bmp_data_url()}},
                ],
            }
        ]
        status, media, _ = json_call(
            server.port,
            "POST",
            "/v1/chat/completions",
            {
                "model": "m22-test",
                "messages": media_messages,
                "max_completion_tokens": 1,
                "reasoning_effort": "none",
            },
            {"X-Gem16-Session-Id": "resident-a"},
            timeout=30.0,
        )
        if status != 400 or media.get("error", {}).get("type") != "unsupported_feature":
            fail(f"26B media API rejection is imprecise: HTTP {status} {media}")
        status, second_slot, _ = json_call(
            server.port,
            "POST",
            "/v1/chat/completions",
            {
                "model": "m22-test",
                "messages": [{"role": "user", "content": "new session"}],
                "max_completion_tokens": 1,
                "reasoning_effort": "none",
            },
            {"X-Gem16-Session-Id": "resident-b"},
            timeout=30.0,
        )
        if status != 503 or second_slot.get("error", {}).get("type") != "resource_exhausted":
            fail(f"26B second server session was not rejected: HTTP {status} {second_slot}")
        metrics_status, metrics_bytes, _ = http_call(
            server.port, "GET", "/metrics", timeout=10.0
        )
        metrics = metrics_bytes.decode("utf-8")
        if metrics_status != 200:
            fail("metrics endpoint failed")
        if metric_value(metrics, "gem16_unsupported_feature_total") < 1:
            fail("unsupported feature counter did not increment")
        if metric_value(metrics, "gem16_resource_exhaustion_total") < 1:
            fail("resource exhaustion counter did not increment")
        for name in (
            "gem16_fallback_total",
            "gem16_model_validation_failure_total",
            "gem16_token_loop_allocation_total",
        ):
            if metric_value(metrics, name) != 0:
                fail(f"{name} changed during successful product generation")

    write_evidence(
        "m22-26b-product.json",
        {
            "schema_version": 1,
            "passed": True,
            "model_report": report,
            "chat_one_shot": one_shot,
            "driver_reports": driver_reports,
            "server_health": health,
            "server_first": first,
            "server_continuation": second,
            "media_rejection": media,
            "second_slot_rejection": second_slot,
            "metrics": {
                name: metric_value(metrics, name)
                for name in (
                    "gem16_fallback_total",
                    "gem16_resource_exhaustion_total",
                    "gem16_unsupported_feature_total",
                    "gem16_model_validation_failure_total",
                    "gem16_token_loop_allocation_total",
                )
            },
        },
    )

    print("M22 26B product acceptance passed")
    return 0


def select_exact_blue(golden: Path) -> tuple[list[int], list[int]]:
    document = json.loads(golden.read_text(encoding="utf-8"))
    for prompt in document.get("prompts", []):
        if isinstance(prompt, dict) and prompt.get("id") == "exact_blue_no_thinking":
            return prompt["prompt_token_ids"], prompt["output_token_ids"]
    fail("12B golden does not contain exact_blue_no_thinking")
    raise AssertionError


def run_12b(run_executable: Path, server_executable: Path, golden: Path) -> int:
    model_value = os.environ.get("GEM16_12B_MODEL")
    if not model_value:
        print("SKIP: GEM16_12B_MODEL is not set", file=sys.stderr)
        return SKIP
    model = Path(model_value).resolve()
    if not model.is_dir():
        fail(f"invalid GEM16_12B_MODEL: {model}")
    input_ids, expected = select_exact_blue(golden)
    generation = json.loads((model / "generation_config.json").read_text(encoding="utf-8"))
    stops = generation.get("eos_token_id")
    if isinstance(stops, int):
        stops = [stops]
    suppressed = generation.get("suppress_tokens", [])
    command = [
        str(run_executable),
        "--model",
        str(model),
        "--input-token-ids",
        ",".join(map(str, input_ids)),
        "--max-tokens",
        str(len(expected)),
        "--max-context",
        str(len(input_ids) + len(expected)),
        "--greedy",
        "--stop-token-ids",
        ",".join(map(str, stops)),
    ]
    if suppressed:
        command += ["--suppress-token-ids", ",".join(map(str, suppressed))]
    exact = parse_json_output(run(command), "12B exact-blue")
    if exact.get("output_token_ids") != expected or exact.get("fallbacks") != 0:
        fail(
            f"12B exact-blue changed: expected {expected}, got "
            f"{exact.get('output_token_ids')} with fallbacks={exact.get('fallbacks')}"
        )

    with Server(server_executable, model, 512, 2) as server:
        health = server.wait()
        if health.get("session_limit") != 2:
            fail("12B server did not retain two-slot behavior")
        first_messages = [{"role": "user", "content": "Reply briefly."}]
        status, first, _ = json_call(
            server.port,
            "POST",
            "/v1/chat/completions",
            {"model": "m22-test", "messages": first_messages, "max_completion_tokens": 1},
            {"X-Gem16-Session-Id": "slot-a"},
            timeout=180.0,
        )
        if status != 200:
            fail(f"12B slot A failed with HTTP {status}")
        status, _, _ = json_call(
            server.port,
            "POST",
            "/v1/chat/completions",
            {
                "model": "m22-test",
                "messages": [{"role": "user", "content": "Second slot."}],
                "max_completion_tokens": 1,
            },
            {"X-Gem16-Session-Id": "slot-b"},
            timeout=180.0,
        )
        if status != 200:
            fail(f"12B slot B failed with HTTP {status}")
        continuation = first_messages + [assistant_message(first)] + [
            {"role": "user", "content": "Continue."}
        ]
        status, _, _ = json_call(
            server.port,
            "POST",
            "/v1/chat/completions",
            {"model": "m22-test", "messages": continuation, "max_completion_tokens": 1},
            {"X-Gem16-Session-Id": "slot-a"},
            timeout=180.0,
        )
        if status != 200:
            fail(f"12B resident continuation failed with HTTP {status}")
        metrics_status, metrics_bytes, _ = http_call(
            server.port, "GET", "/metrics", timeout=10.0
        )
        if metrics_status != 200:
            fail("12B metrics endpoint failed")
        metrics = metrics_bytes.decode("utf-8")
        if metric_value(metrics, "gem16_fallback_total") != 0:
            fail("12B server used a fallback")
        if metric_value(metrics, "gem16_token_loop_allocation_total") != 0:
            fail("12B server reported token-loop allocation")

    write_evidence(
        "m22-12b-product.json",
        {
            "schema_version": 1,
            "passed": True,
            "expected_output_token_ids": expected,
            "run_output": exact,
            "server_health": health,
            "metrics": {
                "gem16_fallback_total": metric_value(
                    metrics, "gem16_fallback_total"
                ),
                "gem16_token_loop_allocation_total": metric_value(
                    metrics, "gem16_token_loop_allocation_total"
                ),
            },
            "two_resident_sessions": True,
            "resident_continuation": True,
        },
    )

    print(f"M22 protected 12B regression passed: output={expected}, two slots")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: test_gemma4_26b_m22_product.py 26b|12b ...", file=sys.stderr)
        return 2
    mode = sys.argv[1]
    if mode == "26b" and len(sys.argv) == 5:
        return run_26b(*(Path(value).resolve() for value in sys.argv[2:5]))
    if mode == "12b" and len(sys.argv) == 5:
        return run_12b(*(Path(value).resolve() for value in sys.argv[2:5]))
    print("invalid M22 product test arguments", file=sys.stderr)
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ProductFailure,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        subprocess.TimeoutExpired,
    ) as exc:
        print(f"M22 product test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

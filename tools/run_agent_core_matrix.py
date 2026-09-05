#!/usr/bin/env python3
"""Serialize two public-profile SDK/agent probes on one local 16 GB GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import urllib.request

from hf_cache import default_target_model, default_assistant_model, locked_snapshot_path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def fetch(url: str):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--server", type=Path, required=True)
    p.add_argument("--pi-cli", type=Path, required=True)
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory; never overwrites prior evidence",
    )
    p.add_argument("--port", type=int, default=18081)
    args = p.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    server = args.server.resolve()
    pi = args.pi_cli.resolve()
    if not server.is_file() or not pi.is_file():
        p.error("server and Pi CLI must exist")
    # Refuse an occupied listener before launching or attributing another server's results.
    import socket

    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.port))
    base = f"http://127.0.0.1:{args.port}"

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    source_files = [
        "src/model/tokenizer.cpp",
        "src/runtime/chat.cpp",
        "src/runtime/tool_call_parser.cpp",
        "src/cli/server_main.cpp",
        "src/server/session_pool.cpp",
        "src/server/session_wait.h",
        "src/server/request_queue.cpp",
        "src/server/http_policy.h",
        "src/model/image.cpp",
        "src/model/image_decode_budget.h",
        "tools/check_server_hardening.py",
        "src/server/openai_chat.cpp",
        "tests/unit/openai_chat_test.cpp",
        "src/model/tokenizer_config.h",
        "tests/unit/tokenizer_test.cpp",
        "tools/validate_openai_sdk.py",
        "tools/validate_external_agent.py",
        "tools/run_agent_core_matrix.py",
        "tools/openai-sdk/validate.ts",
        "tools/openai-sdk/package-lock.json",
        "tools/pi-agent/package-lock.json",
        "tools/requirements-openai-sdk.txt",
        "toolchains/blackwell16gb.lock",
    ]
    report = {
        "scope": "Linux/Windows local development matrix; not full Agent Core release qualification",
        "platform": platform.platform(),
        "python": sys.version,
        "node": subprocess.check_output(["node", "--version"], text=True).strip(),
        "git_revision": git("rev-parse", "HEAD"),
        "dirty_status": git("status", "--short"),
        "source_sha256": {f: digest(ROOT / f) for f in source_files},
        "server_sha256": digest(server),
        "model_locks": {
            f.name: digest(f) for f in sorted((ROOT / "models").glob("*.lock.json"))
        },
        "profiles": [],
    }
    if shutil.which("nvidia-smi"):
        report["gpu"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    for profile in ["12b-unified", "26b-compact-vision"]:
        out = args.output_dir / profile
        out.mkdir()
        command = [
            str(server),
            "--model-name",
            "gem16",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--max-context",
            "16384",
            "--max-sessions",
            "1",
            "--mtp-draft-tokens",
            "2",
            "--max-queued-requests",
            "2",
        ]
        if profile == "12b-unified":
            command += [
                "--model",
                str(default_target_model()),
                "--assistant-model",
                str(default_assistant_model()),
            ]
        else:
            for flag, name in [
                ("--model", "trellis35-target"),
                ("--vision-model", "vision-fp8"),
                ("--assistant-model", "gem16-assistant"),
            ]:
                command += [
                    flag,
                    str(
                        locked_snapshot_path(
                            ROOT / "models" / f"gemma4-26b-{name}.lock.json"
                        )
                    ),
                ]
            command += ["--vision-max-soft-token-budget", "280"]
        row = {"profile": profile, "server_command": command, "checks": []}
        report["profiles"].append(row)
        with (out / "server.txt").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
            try:
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"{profile} startup exited {process.returncode}"
                        )
                    try:
                        health = fetch(base + "/health")
                        if health["status"] == "ok":
                            break
                    except OSError:
                        pass
                    time.sleep(0.25)
                else:
                    raise RuntimeError("startup timeout")
                # Runtime profile labels are preserved verbatim in evidence; inspect capabilities too.
                assert (
                    health["capabilities"]["text"]
                    and health["decode_mode"] == "fixed-d2"
                )
                assert health["model_variant"] == (
                    "gemma4_unified_12b"
                    if profile == "12b-unified"
                    else "gemma4_moe_26b_a4b"
                )
                row["health"] = health
                tsx = ROOT / "tools/openai-sdk/node_modules/tsx/dist/cli.mjs"
                checks = [
                    (
                        "python",
                        [
                            sys.executable,
                            str(ROOT / "tools/validate_openai_sdk.py"),
                            "--base-url",
                            base + "/v1",
                        ],
                    ),
                    (
                        "typescript",
                        [
                            "node",
                            str(tsx),
                            str(ROOT / "tools/openai-sdk/validate.ts"),
                            base + "/v1",
                        ],
                    ),
                    (
                        "external-agent",
                        [
                            sys.executable,
                            str(ROOT / "tools/validate_external_agent.py"),
                            "--base-url",
                            base + "/v1",
                            "--pi-cli",
                            str(pi),
                            "--output-dir",
                            str(out / "pi"),
                        ],
                    ),
                ]
                checks += [
                    (
                        "hardening",
                        [
                            sys.executable,
                            str(ROOT / "tools/check_server_hardening.py"),
                            "--base-url",
                            base,
                            "--output",
                            str(out / "hardening-result.json"),
                        ],
                    ),
                    (
                        "multi-image",
                        [
                            sys.executable,
                            str(ROOT / "tools/check_multi_image_conversation.py"),
                            "--base-url",
                            base,
                            "--output",
                            str(out / "multi-image-result.json"),
                        ]
                        + (["--compact"] if profile == "26b-compact-vision" else []),
                    ),
                ]
                for name, cmd in checks:
                    print(f"{profile}: {name}", flush=True)
                    with (
                        (out / f"{name}.json").open("w", encoding="utf-8") as output,
                        (out / f"{name}.stderr.txt").open(
                            "w", encoding="utf-8"
                        ) as error,
                    ):
                        result = subprocess.run(
                            cmd, cwd=ROOT, stdout=output, stderr=error, timeout=360
                        )
                    row["checks"].append(
                        {"name": name, "command": cmd, "exit_code": result.returncode}
                    )
                with urllib.request.urlopen(base + "/metrics", timeout=5) as response:
                    (out / "metrics.txt").write_bytes(response.read())
            except Exception as exc:
                row["error"] = str(exc)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                row["shutdown_exit_code"] = process.returncode
        row["status"] = (
            "passed"
            if "error" not in row
            and len(row["checks"]) == 5
            and all(c["exit_code"] == 0 for c in row["checks"])
            else "failed"
        )
    report["status"] = (
        "passed"
        if all(r["status"] == "passed" for r in report["profiles"])
        else "failed"
    )
    report["files"] = {
        str(f.relative_to(args.output_dir)): digest(f)
        for f in sorted(args.output_dir.rglob("*"))
        if f.is_file()
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(report["status"])
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

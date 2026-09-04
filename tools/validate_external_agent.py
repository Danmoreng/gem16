#!/usr/bin/env python3
"""Run an unmodified, pinned Pi agent against a disposable coding fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

PI_VERSION = "0.85.0"
SOURCE = "def add(a, b):\n    return a - b\n"
TEST = """import unittest
from arithmetic import add

class AdditionTest(unittest.TestCase):
    def test_signed_addition(self):
        self.assertEqual(add(7, 5), 12)
        self.assertEqual(add(-4, 3), -1)

if __name__ == "__main__":
    unittest.main()
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    p.add_argument("--model", default="gem16")
    p.add_argument(
        "--pi-cli",
        type=Path,
        required=True,
        help="Installed @earendil-works/pi-coding-agent/dist/cli.js",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New evidence directory; must not exist",
    )
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cli = args.pi_cli.resolve()
    version = subprocess.check_output(
        ["node", str(cli), "--version"], text=True, stderr=subprocess.STDOUT
    ).strip()
    if version != PI_VERSION:
        raise RuntimeError(f"expected Pi {PI_VERSION}, found {version}")
    with tempfile.TemporaryDirectory(prefix="gem16-agent-fixture-") as temporary:
        root = Path(temporary)
        project = root / "project"
        config = root / "config"
        project.mkdir()
        config.mkdir()
        (project / "arithmetic.py").write_text(SOURCE)
        (project / "test_arithmetic.py").write_text(TEST)
        before = subprocess.run(
            [sys.executable, "-m", "unittest", "-v"],
            cwd=project,
            capture_output=True,
            text=True,
        )
        assert before.returncode != 0, "fixture must fail before editing"
        model_config = {
            "providers": {
                "gem16": {
                    "baseUrl": args.base_url,
                    "api": "openai-completions",
                    "apiKey": "gem16-local-validation",
                    "compat": {
                        "supportsStore": False,
                        "supportsDeveloperRole": True,
                        "supportsReasoningEffort": True,
                        "supportsUsageInStreaming": True,
                        "supportsStrictMode": False,
                        "supportsLongCacheRetention": False,
                    },
                    "models": [
                        {
                            "id": args.model,
                            "name": "gem16 local",
                            "reasoning": True,
                            "thinkingLevelMap": {"off": "none"},
                            "input": ["text"],
                            "contextWindow": 16384,
                            "maxTokens": 2048,
                            "cost": {
                                "input": 0,
                                "output": 0,
                                "cacheRead": 0,
                                "cacheWrite": 0,
                            },
                        }
                    ],
                }
            }
        }
        (config / "models.json").write_text(json.dumps(model_config, indent=2))
        (config / "settings.json").write_text(
            json.dumps({"compaction": {"enabled": False}})
        )
        # Use the interpreter that owns this run on both Windows and Linux.
        # JSON quoting also protects Windows interpreter paths containing spaces.
        check_command = json.dumps(sys.executable) + " -m unittest -v"
        prompt = (
            "Work only in this temporary project. Read arithmetic.py and test_arithmetic.py with the read tool. "
            "Fix add using the edit tool. Do not change the test. Use bash to run "
            + check_command
            + ". Report the verified result."
        )
        command = [
            "node",
            str(cli),
            "--provider",
            "gem16",
            "--model",
            args.model,
            "--thinking",
            "off",
            "--mode",
            "json",
            "--print",
            "--no-session",
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-themes",
            "--no-context-files",
            "--tools",
            "read,edit,bash",
            prompt,
        ]
        log = args.output_dir / "transcript.jsonl"
        with log.open("w", encoding="utf-8") as out:
            run = subprocess.run(
                command,
                cwd=project,
                env={**os.environ, "PI_CODING_AGENT_DIR": str(config)},
                stdout=out,
                stderr=subprocess.STDOUT,
                timeout=180,
            )
        events = []
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
        messages = [e["message"] for e in events if e.get("type") == "message_end"]
        calls = [
            c
            for m in messages
            if m.get("role") == "assistant"
            for c in m.get("content", [])
            if c.get("type") == "toolCall"
        ]
        results = [m for m in messages if m.get("role") == "toolResult"]
        errors = [
            m.get("errorMessage", "agent error")
            for m in messages
            if m.get("stopReason") in ("error", "aborted")
        ]
        after = subprocess.run(
            [sys.executable, "-m", "unittest", "-v"],
            cwd=project,
            capture_output=True,
            text=True,
        )
        edited = (project / "arithmetic.py").read_text()
        checks = {
            "agent_exit_zero": run.returncode == 0,
            "no_agent_errors": not errors,
            "read_both_files": all(
                any(
                    c["name"] == "read"
                    and Path(c["arguments"].get("path", "")).name == f
                    for c in calls
                )
                for f in ("arithmetic.py", "test_arithmetic.py")
            ),
            "edit_tool_used": any(
                c["name"] == "edit"
                and Path(c["arguments"].get("path", "")).name == "arithmetic.py"
                for c in calls
            ),
            "agent_ran_check": any(
                c["name"] == "bash"
                and "-m unittest" in c["arguments"].get("command", "")
                for c in calls
            ),
            "successful_check_result": any(
                m.get("toolName") == "bash"
                and not m.get("isError")
                and "OK" in "".join(c.get("text", "") for c in m.get("content", []))
                for m in results
            ),
            "source_changed": edited != SOURCE,
            "test_unchanged": (project / "test_arithmetic.py").read_text() == TEST,
            "independent_check_passed": after.returncode == 0,
            "final_answer": bool(
                messages
                and messages[-1].get("role") == "assistant"
                and messages[-1].get("stopReason") == "stop"
            ),
        }
        for name, text in [
            ("before.py", SOURCE),
            ("after.py", edited),
            ("test_arithmetic.py", TEST),
            ("check-before.txt", before.stdout + before.stderr),
            ("check-after.txt", after.stdout + after.stderr),
        ]:
            (args.output_dir / name).write_text(text, encoding="utf-8")
        (args.output_dir / "models.json").write_text(json.dumps(model_config, indent=2))
        report = {
            "agent": "@earendil-works/pi-coding-agent",
            "version": version,
            "platform": platform.system(),
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "errors": errors,
            "tool_calls": calls,
            "command": command,
            "reasoning": "none",
            "transport": "Chat Completions full history",
            "transcript_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        }
        (args.output_dir / "result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

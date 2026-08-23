#!/usr/bin/env python3
"""Host tests for the compact M21 qualification reconciler."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/qualify_gemma4_26b_m21.py"
SPEC = importlib.util.spec_from_file_location("qualify_m21", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


FAKE_DRIVER = r'''#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model")
parser.add_argument("--output", type=Path)
parser.add_argument("--logits", type=Path)
parser.add_argument("--context", type=int)
parser.add_argument("--prompt-tokens", type=int)
parser.add_argument("--device")
args = parser.parse_args()
if args.context > 65_536:
    raise SystemExit(20)
margin = (400 if args.context >= 65_536 else 700) * 1024 * 1024
free = margin + 1024 * 1024
payload = {
    "schema_version": 1,
    "milestone": "M21",
    "backend": "native_sm120_integrated",
    "context_tokens": args.context,
    "prompt_tokens": args.prompt_tokens,
    "final_position": args.context,
    "prefill_elapsed_ms": 1.0,
    "decode_elapsed_ms": 2.0,
    "prefill_prediction_token": 17,
    "decode_prediction_token": 23,
    "all_logits_finite": True,
    "over_limit_rejected": True,
    "sliding_ring_wrap_exercised": True,
    "sliding_ring_wrap_count": max(1, args.context // 1024),
    "global_extent_exercised": True,
    "maximum_global_position_exclusive": args.context,
    "fallback_count": 0,
    "recurring_allocation_count": 0,
    "prefill_chunk_count": 2,
    "minimum_prefill_chunk_tokens": 64,
    "memory": {
        "required_margin_bytes": margin,
        "margin_pass": True,
        "free_after_prefill_bytes": free,
        "free_after_decode_bytes": free,
        "kv_cache_bytes": 123,
        "workspace_bytes": 456,
    },
}
args.output.write_text(json.dumps(payload), encoding="utf-8")
args.logits.write_bytes(args.context.to_bytes(4, "little"))
'''


class M21QualificationTest(unittest.TestCase):
    def test_context_parser_and_margins(self) -> None:
        self.assertEqual(MODULE.parse_contexts("65536,32768,65536"), [32768, 65536])
        self.assertEqual(MODULE.expected_margin(32768), 700 * 1024 * 1024)
        self.assertEqual(MODULE.expected_margin(65536), 400 * 1024 * 1024)
        self.assertEqual(
            MODULE.driver_command(Path("driver.py")),
            [sys.executable, "driver.py"],
        )
        self.assertEqual(MODULE.driver_command(Path("driver.exe")), ["driver.exe"])

    def test_two_process_reconciliation_and_maximum_boundary(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gem16-m21-host-test-") as temp:
            root = Path(temp)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            (model / "gem16_compilation.json").write_text("{}\n", encoding="utf-8")
            driver = root / "fake_driver.py"
            driver.write_text(FAKE_DRIVER, encoding="utf-8")
            driver.chmod(driver.stat().st_mode | stat.S_IXUSR)
            output = root / "acceptance.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "--driver",
                    str(driver),
                    "--model",
                    str(model),
                    "--output",
                    str(output),
                    "--contexts",
                    "32768,65536,69632",
                    "--runs",
                    "2",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(result["release_32k"])
            self.assertEqual(result["base_64k_result"], "passed")
            self.assertEqual(result["base_max_context"], 65536)
            self.assertTrue(result["maximum_search_complete"])
            self.assertTrue(result["exit_gate_pass"])
            self.assertEqual(
                [item["status"] for item in result["contexts"]],
                ["passed", "passed", "capacity_rejected"],
            )
            self.assertEqual(result["first_capacity_rejection"], 69632)
            self.assertEqual(result["maximum_search_gap_tokens"], 4096)
            self.assertFalse(result["unclassified_failures"])

    def test_driver_error_does_not_define_capacity_boundary(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gem16-m21-host-test-") as temp:
            root = Path(temp)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            (model / "gem16_compilation.json").write_text("{}\n", encoding="utf-8")
            driver = root / "fake_driver.py"
            driver.write_text(
                FAKE_DRIVER.replace("raise SystemExit(20)", "raise SystemExit(5)"),
                encoding="utf-8",
            )
            driver.chmod(driver.stat().st_mode | stat.S_IXUSR)
            output = root / "acceptance.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "--driver", str(driver),
                    "--model", str(model),
                    "--output", str(output),
                    "--contexts", "32768,65536,69632",
                    "--runs", "2",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 1, completed.stderr)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(result["maximum_search_complete"])
            self.assertTrue(result["unclassified_failures"])
            self.assertFalse(result["exit_gate_pass"])


if __name__ == "__main__":
    unittest.main()

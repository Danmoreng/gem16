#!/usr/bin/env python3
"""Host tests for the V19 fresh-process capacity qualifier."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/qualify_gemma4_26b_vision_v19_capacity.py"
SPEC = importlib.util.spec_from_file_location("qualify_vision_v19_capacity", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VisionV19CapacityTest(unittest.TestCase):
    def test_context_parser_is_bounded_sorted_and_unique(self) -> None:
        self.assertEqual(MODULE.parse_contexts("65536,32768,65536"), (32768, 65536))
        for value in ("", "1", "262145", "32768,bad"):
            with self.assertRaises(argparse.ArgumentTypeError):
                MODULE.parse_contexts(value)

    def test_resume_identity_rejects_candidate_drift(self) -> None:
        identity = {
            "protocol": "fresh_process_bounded_capacity_v2",
            "binary": {"path": "/binary", "sha256": "a" * 64},
            "components": {
                "target": "/target",
                "vision": "/vision",
                "assistant": "/assistant",
            },
            "runs_per_context": 2,
            "context_matrix": {
                "target_vision": [32768, 65536],
                "target_vision_assistant_d2": [32768, 65536],
            },
        }
        MODULE.validate_identity(dict(identity), identity)
        changed = dict(identity)
        changed["runs_per_context"] = 3
        with self.assertRaisesRegex(RuntimeError, "runs_per_context"):
            MODULE.validate_identity(changed, identity)

    def test_summary_requires_repeatable_acceptance_and_rejection(self) -> None:
        def run(context: int, status: str, headroom: int | None = None) -> dict:
            return {
                "context_tokens": context,
                "status": status,
                "peak_process_vram_bytes": 123,
                "model_report": {"admission_headroom_bytes": headroom},
            }

        result = MODULE.summarize(
            [
                {
                    "name": "target_vision",
                    "runs": [
                        run(32768, "accepted", 10),
                        run(32768, "accepted", 11),
                        run(65536, "accepted", 8),
                        run(65536, "accepted", 9),
                        run(69632, "rejected"),
                        run(69632, "rejected"),
                    ],
                }
            ],
            2,
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(
            result["scenarios"][0]["maximum_repeatably_accepted_context"],
            65536,
        )
        self.assertEqual(
            result["scenarios"][0][
                "first_repeatably_rejected_context_above_maximum"
            ],
            69632,
        )


if __name__ == "__main__":
    unittest.main()

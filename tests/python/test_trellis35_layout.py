from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest

from tools.gem16_compile.common import UINT64_MAX, InvalidPlanError
from tools.gem16_compile.trellis35_layout import (
    align_up,
    checked_mul,
    estimate_trellis35_layout,
    select_rate_map,
    validate_rate_map,
)
from tools.generate_gemma4_26b_trellis35_plan import (
    OUTPUT_ESTIMATE,
    OUTPUT_SPEC,
    ROOT,
    generated_documents,
)


class Trellis35LayoutTest(unittest.TestCase):
    def test_checked_outputs_are_current(self) -> None:
        documents = generated_documents()
        self.assertEqual(documents[OUTPUT_SPEC], OUTPUT_SPEC.read_bytes())
        self.assertEqual(documents[OUTPUT_ESTIMATE], OUTPUT_ESTIMATE.read_bytes())
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/generate_gemma4_26b_trellis35_plan.py"), "--check"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_exact_shapes_rate_and_byte_accounting(self) -> None:
        spec = json.loads(OUTPUT_SPEC.read_text(encoding="utf-8"))
        report = json.loads(OUTPUT_ESTIMATE.read_text(encoding="utf-8"))
        estimate = report["estimate"]
        self.assertEqual(spec["families"]["gate_up"]["logical_shape"], [128, 1408, 2816])
        self.assertEqual(spec["families"]["gate_up"]["physical_shape"], [128, 1408, 2816])
        self.assertEqual(spec["families"]["gate_up"]["padding"], "none")
        self.assertTrue(spec["families"]["gate_up"]["gate_up_inverse_before_split"])
        self.assertEqual(spec["families"]["down"]["logical_shape"], [128, 2816, 704])
        self.assertEqual(spec["families"]["down"]["physical_shape"], [128, 2816, 768])
        self.assertEqual(estimate["logical_expert_coefficients"], 22_837_985_280)
        self.assertEqual(estimate["encoded_expert_coefficients"], 23_530_045_440)
        self.assertEqual(estimate["trellis_payload_bytes"], 10_294_394_880)
        self.assertEqual(estimate["suh_bytes"], 27_525_120)
        self.assertEqual(estimate["svh_bytes"], 32_440_320)
        self.assertEqual(estimate["rate_descriptor_bytes"], 61_440)
        self.assertEqual(estimate["codebook_bytes"], 0)
        self.assertEqual(estimate["alignment_bytes"], 0)
        self.assertEqual(estimate["total_expert_bytes"], 10_354_421_760)
        self.assertEqual(estimate["total_target_arena_bytes"], 12_204_692_480)
        self.assertEqual(estimate["saving_vs_locked_nvfp4_bytes"], 2_491_975_680)
        self.assertEqual(estimate["payload_bpw_encoded"], 3.5)
        self.assertGreater(estimate["effective_expert_bpw"], 3.5)
        self.assertLess(estimate["total_target_arena_bytes"] / (1 << 30), 11.370)
        self.assertGreater(estimate["saving_vs_locked_nvfp4_bytes"] / (1 << 30), 2.317)

    def test_rate_selection_is_deterministic_and_ties_use_expert_id(self) -> None:
        benefits = [1.0] * 128
        rate_map = select_rate_map(benefits)
        self.assertEqual(rate_map[:64], (4,) * 64)
        self.assertEqual(rate_map[64:], (3,) * 64)
        self.assertEqual(validate_rate_map(rate_map), rate_map)
        estimate = estimate_trellis35_layout(
            baseline_arena_bytes=14_696_668_160,
            baseline_gate_up_bytes=8_564_259_840,
            baseline_down_bytes=4_282_137_600,
            rate_map=rate_map,
        )
        self.assertEqual(estimate["payload_bpw_encoded"], 3.5)

    def test_malformed_rate_maps_and_proxy_benefits_fail_closed(self) -> None:
        with self.assertRaises(InvalidPlanError):
            validate_rate_map(None)  # type: ignore[arg-type]
        for rate_map in (
            [3] * 127,
            [3] * 128,
            [3] * 63 + [4] * 65,
            [3] * 64 + [5] * 64,
            [3] * 64 + [True] * 64,
        ):
            with self.subTest(rate_map_length=len(rate_map)):
                with self.assertRaises(InvalidPlanError):
                    validate_rate_map(rate_map)
        with self.assertRaises(InvalidPlanError):
            select_rate_map([1.0] * 63 + [0.0] * 65)
        with self.assertRaises(InvalidPlanError):
            select_rate_map([1.0] * 127 + [float("nan")])

    def test_checked_arithmetic_rejects_overflow_and_bad_alignment(self) -> None:
        with self.assertRaises(InvalidPlanError):
            checked_mul(UINT64_MAX, 2, description="test product")
        with self.assertRaises(InvalidPlanError):
            align_up(UINT64_MAX, 256, "test alignment")
        with self.assertRaises(InvalidPlanError):
            align_up(1, 3, "test alignment")

    def test_spec_hash_and_source_identity_are_bound(self) -> None:
        report = json.loads(OUTPUT_ESTIMATE.read_text(encoding="utf-8"))
        self.assertEqual(
            report["format_spec"]["sha256"],
            hashlib.sha256(OUTPUT_SPEC.read_bytes()).hexdigest(),
        )
        self.assertEqual(report["source"], json.loads(OUTPUT_SPEC.read_text())["source"])
        self.assertEqual(report["rate_map_status"], "deferred_until_wp2_proxy_errors")
        self.assertIn("not a compiled artifact", report["limitations"][0])


if __name__ == "__main__":
    unittest.main()

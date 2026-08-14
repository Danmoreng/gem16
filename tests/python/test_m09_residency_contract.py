"""M09 compact residency-evidence contract tests."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "artifacts/m09/diagnostic-summary.json"
ACCEPTANCE = ROOT / "artifacts/m09/acceptance.json"


class M09ResidencyContractTest(unittest.TestCase):
    def test_compact_diagnostic_records_real_residency_without_acceptance(self) -> None:
        summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
        self.assertEqual(summary["milestone"], "M09")
        self.assertEqual(summary["status"], "diagnostic_pass_not_acceptance")
        self.assertFalse(summary["acceptance"])
        self.assertTrue(summary["code_revision"].endswith("-dirty"))

        artifact = summary["artifact_residency"]
        self.assertEqual(artifact["tensor_count"], 1_285)
        self.assertEqual(artifact["shard_count"], 16)
        self.assertEqual(artifact["payload_bytes"], 14_696_569_196)
        self.assertEqual(len(artifact["artifact_content_sha256"]), 64)
        self.assertEqual(len(artifact["artifact_lock_sha256"]), 64)
        self.assertEqual(artifact["immutable_weight_arena_bytes"], 14_696_668_160)
        self.assertTrue(artifact["one_device_weight_arena"])
        self.assertEqual(artifact["persistent_device_repack_bytes"], 0)
        self.assertLessEqual(artifact["host_staging_peak_bytes"], 4 * 1024 * 1024)
        self.assertEqual(sum(artifact["uploaded_tensors"].values()), 1_285)

        slot = summary["one_slot"]
        self.assertEqual(slot["context_tokens"], 32_768)
        self.assertGreaterEqual(slot["final_free_bytes"], 700 * 1024 * 1024)
        self.assertTrue(slot["pass"])

        feasibility = summary["context_feasibility"]
        self.assertEqual(feasibility["64k"]["status"], "admitted")
        self.assertFalse(feasibility["64k"]["partial_allocation"])
        self.assertTrue(feasibility["64k"]["allocations_complete"])
        self.assertEqual(feasibility["64k"]["margin_shortfall_bytes"], 0)
        self.assertGreaterEqual(
            feasibility["64k"]["final_free_bytes"], 400 * 1024 * 1024
        )
        candidate = feasibility["measured_candidate"]
        self.assertEqual(candidate["context_tokens"], 65_536)
        self.assertGreaterEqual(candidate["final_free_bytes"], 400 * 1024 * 1024)
        self.assertFalse(candidate["advertised"])

        self.assertEqual(summary["second_slot"]["cuda_allocation_delta_bytes"], 0)
        self.assertFalse(summary["second_slot"]["partial_allocation"])
        self.assertEqual(summary["protected_12b"]["status"], "pass")
        self.assertTrue(all(summary["gates"].values()))

    def test_compact_acceptance_binds_clean_implementation_commit(self) -> None:
        summary = json.loads(ACCEPTANCE.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "acceptance_pass")
        self.assertTrue(summary["acceptance"])
        self.assertEqual(
            summary["code_revision"],
            "6c3b9e456bc7fed68e2e90a51ba20c1c895fd085",
        )
        self.assertEqual(summary["context_feasibility"]["64k"]["status"], "admitted")
        self.assertGreaterEqual(
            summary["context_feasibility"]["64k"]["final_free_bytes"],
            400 * 1024 * 1024,
        )
        self.assertTrue(all(summary["gates"].values()))


if __name__ == "__main__":
    unittest.main()

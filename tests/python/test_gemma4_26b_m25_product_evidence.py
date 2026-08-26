import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "artifacts" / "m25" / "sampled-mtp-product.json"


class Gemma426BM25ProductEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_sampled_mtp_is_target_identical_for_bounded_controls(self):
        record = self.record
        self.assertEqual(record["status"], "passed")
        self.assertIn("not formal M25 acceptance", record["scope"])
        self.assertEqual(
            record["implementation_commit"],
            "c4ead1dc2b74f2b2cffe38599ff21703fba55a6f",
        )
        self.assertEqual(
            record["target"]["artifact_content_sha256"],
            "471805f7dad8abb84300be78b2822a63dcb1d35bff5aa98426a162cc8532ee17",
        )
        self.assertEqual(
            record["assistant"]["artifact_content_sha256"],
            "978f5e3804dd08b8ea5551883811e0bf6737a23ae5ae3dcfa0ef71dc4ffe532b",
        )
        differentials = record["sampling_differentials"]
        self.assertEqual(len(differentials), 2)
        self.assertEqual({case["sampling"]["seed"] for case in differentials}, {0, 42})
        self.assertEqual(
            {case["sampling"]["repetition_penalty"] for case in differentials},
            {1.0, 1.1},
        )
        for case in differentials:
            self.assertTrue(case["exact_match"])
            self.assertTrue(case["ordinary_output_token_ids"])
            self.assertEqual(
                case["ordinary_output_token_ids"], case["mtp_output_token_ids"]
            )
            counters = case["mtp"]
            self.assertEqual(counters["draft_tokens"], 2)
            self.assertEqual(
                counters["accepted_tokens"] + counters["rejected_tokens"],
                counters["proposed_tokens"],
            )
            self.assertGreater(counters["verification_groups"], 0)

    def test_server_smoke_binds_text_only_sampled_mtp_contract(self):
        product = self.record["server_product"]
        self.assertEqual(product["model_variant"], "gemma4_moe_26b_a4b")
        self.assertTrue(product["text_only"])
        self.assertTrue(product["supports_mtp"])
        self.assertEqual(product["mtp_draft_tokens"], 2)
        self.assertFalse(product["mtp_adaptive"])
        self.assertEqual(product["max_context_tokens"], 32768)
        self.assertEqual(product["resident_session_limit"], 1)
        self.assertGreaterEqual(
            product["admission_free_bytes"],
            product["required_admission_margin_bytes"],
        )
        self.assertGreater(product["continuation_cached_tokens"], 0)
        self.assertGreater(product["mtp_proposed_tokens"], 0)
        self.assertGreater(product["mtp_accepted_tokens"], 0)
        self.assertEqual(product["fallback_total"], 0)
        self.assertEqual(product["token_loop_allocation_total"], 0)
        limitations = " ".join(self.record["limitations"])
        self.assertIn("not accepted", limitations)
        self.assertIn("text-only", limitations)


if __name__ == "__main__":
    unittest.main()

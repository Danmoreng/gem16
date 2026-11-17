from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from analyze_mtp_group_cost import analyze, group_facts, proposal_groups


class GroupCostTest(unittest.TestCase):
    def setUp(self):
        self.engine = {"output_token_ids": [99, 3, 4, 8, 5, 6, 7],
                       "mtp": {"verification_groups": 3, "proposed_tokens": 6,
                               "accepted_tokens": 3, "ordinary_fallback_tokens": 0,
                               "draft_tokens": 2, "adaptive": False,
                               "group_execution": "gpu_chained_fixed_d2_conditional_graph"}}
        self.proposals = {"captured": True, "copy_error": 0,
                          "proposed_token_ids": [1, 2, 4, 9, 5, 6],
                          "group_count": 3, "proposed_count": 6, "accepted_count": 3,
                          "ordinary_tail_count": 0, "output_count": 6}

    def test_proposals_derive_all_acceptance_cases_without_inventing_routes(self):
        rows, provenance = group_facts(self.engine, 100, proposals=self.proposals)
        self.assertEqual([r["accepted"] for r in rows], [0, 1, 2])
        self.assertEqual([r["position"] for r in rows], [100, 101, 103])
        self.assertTrue(all(r["expert_unions_by_layer"] is None for r in rows))
        self.assertIn("derived", provenance["acceptance_position_depth"])

    def test_missing_observation_leaves_joint_data_unknown(self):
        rows, provenance = group_facts(self.engine, 100)
        self.assertTrue(all(r["accepted"] is None and r["depth"] is None and r["position"] is None for r in rows))
        kernels = []
        for i in range(3):
            kernels += [{"name": "InitializeControlledAssistantKernel", "start": i * 10,
                         "end": i * 10 + 2, "category": "control"},
                        {"name": "AdvanceMtpD2ChainKernel", "start": i * 10 + 5,
                         "end": i * 10 + 8, "category": "control"}]
        result = analyze({"window_ns": [0, 30]}, kernels, rows, provenance)
        self.assertIsNone(result["by_accepted"])
        self.assertEqual(result["groups"][0]["kernel_busy_union_ms"], 5e-6)

    def test_rejects_missing_continuation_and_inconsistent_totals(self):
        bad = {**self.engine, "output_token_ids": self.engine["output_token_ids"][:-1]}
        with self.assertRaisesRegex(ValueError, "continuation"):
            proposal_groups({**self.proposals, "output_count": 5}, bad, 100)
        bad = {**self.proposals, "proposed_token_ids": [1, 2]}
        with self.assertRaisesRegex(ValueError, "depth-two"):
            proposal_groups(bad, self.engine, 100)
        # The same average proposal count under maximum4 can mix depths.
        bad = {**self.engine, "mtp": {**self.engine["mtp"], "draft_tokens": 4}}
        with self.assertRaisesRegex(ValueError, "fixed-D2 chain"):
            proposal_groups(self.proposals, bad, 100)
        with self.assertRaisesRegex(ValueError, "counter mismatch"):
            proposal_groups({**self.proposals, "group_count": 2}, self.engine, 100)


if __name__ == "__main__":
    unittest.main()

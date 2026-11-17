import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "validate_mtp_groups", Path(__file__).resolve().parents[2] / "tools/validate_mtp_groups.py")
groups = importlib.util.module_from_spec(spec)
spec.loader.exec_module(groups)


def fixture():
    rows = []
    for i, (position, proposals, outputs, accepted) in enumerate([
        (10, [99, 100], [1], 0), (11, [2, 99], [2, 3], 1), (13, [4, 5], [4, 5, 6], 2)
    ]):
        rows.append(dict(group=i, position=position, proposals=proposals, outputs=outputs,
                         accepted=accepted, depth=2, output_count=len(outputs),
                         layer_expert_ids=[list(range(8))*3 for _ in range(30)]))
    return dict(captured=True, fault=0, chain_group_count=3, groups=rows), {
        "mtp": dict(verification_groups=3, proposed_tokens=6, accepted_tokens=3,
                    ordinary_fallback_tokens=0), "output_token_ids": list(range(7))}


class ValidateMtpGroupsTest(unittest.TestCase):
    def test_all_acceptance_cases_and_union(self):
        observation, engine = fixture()
        original = copy.deepcopy(observation)
        result = groups.validate(observation, engine, 10)
        self.assertEqual(observation, original)
        self.assertEqual(result["observed_acceptance_histogram_0_1_2"], [1, 1, 1])
        self.assertEqual(result["observed_depth_histogram"], {2: 3})
        self.assertEqual(result["derived_expert_union_histogram"], {8: 90})

    def test_rejects_corrupted_position_depth_output_route_and_totals(self):
        original, engine = fixture()
        for field, value in [("position", 9), ("depth", 1), ("outputs", [99]),
                             ("layer_expert_ids", []), ("accepted", 1)]:
            observation = copy.deepcopy(original)
            observation["groups"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                groups.validate(observation, engine, 10)
        observation = copy.deepcopy(original)
        observation["groups"][0]["layer_expert_ids"][0][0] = 128
        with self.assertRaises(ValueError):
            groups.validate(observation, engine, 10)


if __name__ == "__main__":
    unittest.main()

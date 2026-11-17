import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "llama_speculative_accounting",
    Path(__file__).resolve().parents[2] / "tools/llama_speculative_accounting.py")
accounting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(accounting)


def line(b, g, a, drafts, accepted_groups, proposed, accepted):
    return (f"#calls(b,g,a) = {b} {g} {a}, #gen drafts = {drafts}, "
            f"#acc drafts = {accepted_groups}, #gen tokens = {proposed}, #acc tokens = {accepted}")


class LlamaSpeculativeAccountingTest(unittest.TestCase):
    def test_mixed_depth_and_acceptance_are_derived_not_observed(self):
        log = "\n".join(line(i, 497*i, 497*i, 497*i, 367*i, 993*i, 637*i)
                        for i in range(1, 14))
        rows = accounting.request_deltas(log)
        self.assertEqual(len(rows), 13)
        for row in rows:
            self.assertEqual(row["derived"]["acceptance_histogram_0_1_2"], [130, 97, 270])
            self.assertEqual(row["derived"]["depth_histogram"], {"1": 1, "2": 496})

    def test_no_drafts_is_unknown_depth(self):
        row = accounting.request_deltas(line(1, 10, 10, 0, 0, 0, 0))[0]
        self.assertIsNone(row["derived"]["depth_histogram"])

    def test_rejects_resets_missing_requests_and_inconsistent_acceptance(self):
        for log in [line(2, 5, 5, 5, 2, 10, 3),
                    line(1, 5, 5, 5, 1, 10, 4),
                    line(1, 5, 6, 5, 2, 10, 3),
                    line(1, 5, 5, 5, 2, 10, 3) + "\n" + line(1, 5, 5, 5, 2, 10, 3)]:
            with self.subTest(log=log), self.assertRaises(ValueError):
                accounting.request_deltas(log)

    def test_missing_counters_stays_empty(self):
        self.assertEqual(accounting.request_deltas("unrelated log"), [])


if __name__ == "__main__":
    unittest.main()

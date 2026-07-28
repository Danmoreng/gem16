import argparse
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import qualify_mtp


class QualifyMtpTest(unittest.TestCase):
    def test_pair_order_alternates_and_requires_exact_output(self):
        tokens = [7, 8, 9]

        def fake_run(*args, **kwargs):
            is_mtp = args[6] != 0
            run = {
                "prompt_tokens_per_second": 100.0,
                "prompt_ms": 10.0,
                "decode_tokens_per_second": 55.0 if is_mtp else 40.0,
                "average_inter_token_latency_ms": 20.0,
                "generated_tokens": 3,
                "measured_decode_intervals": 2,
                "stop_reason": "stop",
                "output_token_sha256": "same",
            }
            if is_mtp:
                run["mtp"] = {
                    "proposed_tokens": 2,
                    "accepted_tokens": 1,
                    "rejected_tokens": 1,
                    "mean_accepted_length": 1.0,
                    "target_batches": 1,
                    "d1_groups": 0,
                    "d2_groups": 1,
                    "d4_groups": 0,
                    "ordinary_fallback_tokens": 0,
                }
            return run, tokens

        args = argparse.Namespace(
            workload=Path("workload.json"),
            output=Path("result.json"),
            model=Path("model"),
            assistant_model=Path("assistant"),
            executable=Path("gem16-run"),
            warmup_pairs=1,
            measured_pairs=2,
        )
        workload = {
            "id": "test",
            "source": {},
            "prompt": {"token_ids_sha256": "prompt-hash"},
        }
        generation = {
            "max_new_tokens": 3,
            "stop_token_ids": [1],
            "suppress_token_ids": [],
        }
        resolved = Path(__file__).resolve()
        with (
            mock.patch.object(Path, "resolve", return_value=resolved),
            mock.patch.object(Path, "write_text"),
            mock.patch.object(Path, "mkdir"),
            mock.patch.object(
                qualify_mtp, "load_workload", return_value=(workload, [2], generation)
            ),
            mock.patch.object(qualify_mtp, "run_gem16", side_effect=fake_run),
            mock.patch.object(
                qualify_mtp,
                "repository_state",
                return_value={"git_commit": "abc", "worktree_dirty_at_start": False},
            ),
        ):
            result = qualify_mtp.qualification(args)

        self.assertEqual(result["status"], "qualified")
        self.assertEqual(
            [pair["order"] for pair in result["pair_order"]],
            [
                ["ordinary", "mtp_d2"],
                ["ordinary", "mtp_d2"],
                ["mtp_d2", "ordinary"],
            ],
        )
        self.assertTrue(result["qualification"]["ordinary_equals_mtp"])
        self.assertAlmostEqual(result["qualification"]["median_speedup"], 1.375)

    def test_changed_output_fails_qualification(self):
        call_count = 0

        def fake_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            run = {
                "decode_tokens_per_second": 50.0,
                "output_token_sha256": f"hash-{call_count}",
            }
            return run, [7, 8, call_count]

        args = argparse.Namespace(
            workload=Path("workload.json"),
            output=Path("result.json"),
            model=Path("model"),
            assistant_model=Path("assistant"),
            executable=Path("gem16-run"),
            warmup_pairs=1,
            measured_pairs=1,
        )
        workload = {
            "id": "test",
            "source": {},
            "prompt": {"token_ids_sha256": "prompt-hash"},
        }
        generation = {
            "max_new_tokens": 3,
            "stop_token_ids": [1],
            "suppress_token_ids": [],
        }
        resolved = Path(__file__).resolve()
        with (
            mock.patch.object(Path, "resolve", return_value=resolved),
            mock.patch.object(Path, "write_text"),
            mock.patch.object(Path, "mkdir"),
            mock.patch.object(
                qualify_mtp, "load_workload", return_value=(workload, [2], generation)
            ),
            mock.patch.object(qualify_mtp, "run_gem16", side_effect=fake_run),
        ):
            with self.assertRaisesRegex(
                qualify_mtp.BenchmarkError, "changed exact output"
            ):
                qualify_mtp.qualification(args)


if __name__ == "__main__":
    unittest.main()

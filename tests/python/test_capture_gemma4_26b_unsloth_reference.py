from __future__ import annotations

import unittest

from tools.capture_gemma4_26b_unsloth_reference import (
    find_prompt,
    process_max_rss_kib,
    reference_version,
    repeat_summary,
    serialize_logprobs,
)


class Candidate:
    def __init__(self, logprob: float, rank: int, decoded_token: str) -> None:
        self.logprob = logprob
        self.rank = rank
        self.decoded_token = decoded_token


class CaptureGemma426BUnslothReferenceTest(unittest.TestCase):
    def test_process_rss_is_optional_on_unsupported_hosts(self) -> None:
        value = process_max_rss_kib()
        self.assertTrue(value is None or value >= 0)

    def test_find_prompt_requires_one_valid_token_sequence(self) -> None:
        corpus = {
            "records": [
                {
                    "id": "tiny",
                    "input_token_ids": [2, 3],
                    "input_token_ids_sha256_u32le": "a" * 64,
                }
            ]
        }
        self.assertEqual(find_prompt(corpus, "tiny")["input_token_ids"], [2, 3])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            find_prompt(corpus, "missing")

    def test_reference_version_requires_unique_locked_version(self) -> None:
        lock = {"references": [{"name": "vllm", "version": "0.26.0"}]}
        self.assertEqual(reference_version(lock, "vllm"), "0.26.0")
        with self.assertRaisesRegex(ValueError, "no unique version"):
            reference_version(lock, "flashinfer")

    def test_repeat_distinguishes_tokens_from_logprob_drift(self) -> None:
        runs = [
            {"token_ids": [7], "text": "x", "logprobs": [[{"logprob": -1.0}]]},
            {"token_ids": [7], "text": "x", "logprobs": [[{"logprob": -1.1}]]},
        ]
        self.assertEqual(
            repeat_summary(runs),
            {
                "token_ids_exact": True,
                "text_exact": True,
                "logprobs_exact": False,
            },
        )

    def test_logprobs_are_sorted_by_rank_and_token_id(self) -> None:
        serialized = serialize_logprobs(
            [
                {
                    9: Candidate(-2.0, 2, "nine"),
                    7: Candidate(-1.0, 1, "seven"),
                }
            ]
        )
        self.assertEqual([entry["token_id"] for entry in serialized[0]], [7, 9])
        self.assertEqual(serialized[0][0]["decoded_token"], "seven")


if __name__ == "__main__":
    unittest.main()

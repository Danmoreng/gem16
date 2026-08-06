from __future__ import annotations

import unittest

from tools.freeze_gemma4_26b_corpus import (
    audit_disjoint,
    expanded_user,
    token_sha256,
)


class FreezeGemma426BCorpusTest(unittest.TestCase):
    def test_repeat_source_is_bounded_and_deterministic(self) -> None:
        text, source = expanded_user(
            {"id": "repeat", "user_repeat": {"text": "abc", "count": 3}}
        )
        self.assertEqual(text, "abcabcabc")
        self.assertEqual(source, {"kind": "repeat", "text": "abc", "count": 3})
        with self.assertRaisesRegex(ValueError, "invalid repeat count"):
            expanded_user(
                {"id": "bad", "user_repeat": {"text": "abc", "count": 100_001}}
            )

    def test_token_hash_is_uint32_little_endian(self) -> None:
        self.assertEqual(
            token_sha256([1, 256]),
            "242045e2f1bb37769b514f182fd91b3d215324cb57f187cced1e9c62921dbac3",
        )
        with self.assertRaisesRegex(ValueError, "invalid token ID"):
            token_sha256([-1])

    def test_split_audit_rejects_cross_split_document_or_token_overlap(self) -> None:
        clean = {
            "calibration": [
                {"expanded_user_sha256": "a", "input_token_ids_sha256_u32le": "b"}
            ],
            "development": [
                {"expanded_user_sha256": "c", "input_token_ids_sha256_u32le": "d"}
            ],
            "test": [
                {"expanded_user_sha256": "e", "input_token_ids_sha256_u32le": "f"}
            ],
        }
        self.assertEqual(audit_disjoint(clean)["status"], "pass")
        clean["test"][0]["input_token_ids_sha256_u32le"] = "b"
        audit = audit_disjoint(clean)
        self.assertEqual(audit["status"], "fail")
        self.assertEqual(audit["document_or_token_span_overlap_count"], 1)


if __name__ == "__main__":
    unittest.main()

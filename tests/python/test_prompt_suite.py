from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools" / "generate_golden.py"
SPEC = importlib.util.spec_from_file_location("generate_golden", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
generate_golden = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = generate_golden
SPEC.loader.exec_module(generate_golden)


class PromptSuiteTest(unittest.TestCase):
    def test_repository_suite_is_valid_and_unique(self) -> None:
        prompts = generate_golden.load_prompts(
            ROOT / "benchmarks/prompts/correctness-v1.json"
        )
        self.assertEqual(len(prompts), 12)
        self.assertEqual(len({prompt["id"] for prompt in prompts}), len(prompts))
        self.assertTrue(any(prompt["enable_thinking"] for prompt in prompts))
        self.assertTrue(any(len(prompt["messages"]) > 1 for prompt in prompts))

    def test_generated_cache_fixtures_match_the_suite(self) -> None:
        source = generate_golden.load_prompts(
            ROOT / "benchmarks/prompts/correctness-v1.json"
        )
        expected_ids = [prompt["id"] for prompt in source]
        for filename_suffix, cache_dtype in (
            ("fp8", "fp8"),
            ("bf16", "bfloat16"),
        ):
            fixture = json.loads(
                (
                    ROOT
                    / "tests/golden"
                    / f"vllm-gemma4-12b-nvfp4-correctness-v1-{filename_suffix}.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(fixture["execution"]["kv_cache_dtype"], cache_dtype)
            self.assertEqual(
                [prompt["id"] for prompt in fixture["prompts"]], expected_ids
            )
            for prompt in fixture["prompts"]:
                self.assertEqual(
                    len(prompt["output_token_ids"]), len(prompt["top_logprobs"])
                )
                for selected, entries in zip(
                    prompt["output_token_ids"], prompt["top_logprobs"]
                ):
                    selected_entry = next(
                        entry for entry in entries if entry["token_id"] == selected
                    )
                    self.assertEqual(
                        selected_entry["logprob"],
                        entries[0]["logprob"],
                        "the selected greedy token may be rank 2 only on an exact tie",
                    )

    def test_duplicate_ids_are_rejected(self) -> None:
        document = {
            "schema_version": 1,
            "prompts": [
                {
                    "id": "duplicate",
                    "messages": [{"role": "user", "content": "one"}],
                    "enable_thinking": False,
                },
                {
                    "id": "duplicate",
                    "messages": [{"role": "user", "content": "two"}],
                    "enable_thinking": False,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(generate_golden.GoldenError):
                generate_golden.load_prompts(path)


if __name__ == "__main__":
    unittest.main()

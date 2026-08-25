#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.qualify_gemma4_26b_m23 import QualificationFailure, qualify


class M23QualificationTest(unittest.TestCase):
    def arguments(self, directory: Path) -> argparse.Namespace:
        artifact = "a" * 64
        source = "s" * 64
        artifact_lock = "l" * 64
        toolchain = "t" * 64
        binary_hash = "b" * 64
        output_hash = "o" * 64
        m20 = {
            "acceptance": True,
            "model": {
                "artifact_content_sha256": artifact,
                "source_lock_sha256": source,
                "artifact_lock_sha256": artifact_lock,
            },
            "toolchain_lock_sha256": toolchain,
            "native_instruction_evidence": {"binary_sha256": binary_hash},
            "promotion": {
                "scenario": "fixture",
                "prompt_tps_median": 6501.0,
                "decode_tps_median": 151.0,
            },
            "configuration": {"warmups": 3, "retained_runs": 10},
            "summary": {
                "fixture": {
                    "prompt_tps": {"count": 10},
                    "decode_tps": {"count": 10},
                }
            },
            "runs": [
                {
                    "output_token_sha256": output_hash,
                    "fallback_count": 0,
                    "recurring_allocation_count": 0,
                }
            ] * 13,
        }
        m21 = {
            "acceptance": True,
            "base_64k_result": "passed",
            "base_max_context": 98304,
            "first_capacity_rejection": 102400,
            "candidate": {
                "artifact_content_sha256": artifact,
                "source_lock_sha256": source,
                "artifact_lock_sha256": artifact_lock,
                "toolchain_lock_sha256": toolchain,
                "benchmark_binary_sha256": binary_hash,
                "context_driver_sha256": "c" * 64,
            },
        }
        m22 = {
            "status": "accepted",
            "qualified_artifact_content_sha256": artifact,
            "source_lock_sha256": source,
        }
        driver = {
            "passed": True,
            "artifact_content_sha256": artifact,
            "source_lock_sha256": source,
            "compiler_commit": "f" * 40,
            "native_path": "native_fixture",
            "head_format": "nvfp4",
        }
        capability = {
            "default_context": 32768,
            "qualified_64k": True,
            "base_max_context": 98304,
            "mtp_max_context": None,
        }
        product_26b = {
            "passed": True,
            "model_report": capability,
            "server_health": capability,
        }
        product_12b = {
            "passed": True,
            "expected_output_token_ids": [1, 2],
            "run_output": {"output_token_ids": [1, 2], "fallbacks": 0},
        }

        def write(name: str, value: object) -> Path:
            path = directory / name
            path.write_text(json.dumps(value), encoding="utf-8")
            return path

        binary = directory / "binary"
        binary.write_bytes(b"fixture")
        binary_hash_actual = hashlib.sha256(b"fixture").hexdigest()
        product_26b["qualified_binaries"] = {
            "gem16_chat_sha256": binary_hash_actual,
            "gem16_server_sha256": binary_hash_actual,
            "m22_product_driver_sha256": binary_hash_actual,
        }
        product_12b["qualified_binaries"] = {
            "gem16_server_sha256": binary_hash_actual,
            "protected_12b_runner_sha256": binary_hash_actual,
        }
        return argparse.Namespace(
            m20=write("m20.json", m20),
            m21=write("m21.json", m21),
            m22=write("m22.json", m22),
            product_26b=write("product-26b.json", product_26b),
            product_12b=write("product-12b.json", product_12b),
            first_driver=write("first.json", driver),
            relaunch_driver=write("relaunch.json", driver),
            chat=binary,
            server=binary,
            product_driver=binary,
            run_binary=binary,
            implementation_revision="79072d43375bb0e96c63ceaadbfda578b15f5a17",
            accepted_on="2026-08-25",
            output_token_sha256=output_hash,
            output=directory / "acceptance.json",
        )

    def test_real_evidence_reconciles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = qualify(self.arguments(Path(temporary)))
        self.assertEqual(result["status"], "accepted_technical_target")
        self.assertEqual(
            result["capability_statement"]["base_max_context_tokens"], 98304
        )
        self.assertFalse(
            result["capability_statement"]["shipping_or_production_quality_claim"]
        )

    def test_artifact_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            args = self.arguments(directory)
            payload = json.loads(args.first_driver.read_text(encoding="utf-8"))
            payload["artifact_content_sha256"] = "0" * 64
            path = directory / "first.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            args.first_driver = path
            with self.assertRaisesRegex(QualificationFailure, "artifact hashes"):
                qualify(args)

    def test_other_contract_mismatches_fail_closed(self) -> None:
        cases = (
            ("source", "m21", lambda value: value["candidate"].update(source_lock_sha256="x" * 64), "source locks"),
            ("toolchain", "m21", lambda value: value["candidate"].update(toolchain_lock_sha256="x" * 64), "toolchain locks"),
            ("output", "m20", lambda value: value["runs"][0].update(output_token_sha256="x" * 64), "output-token hash"),
            ("fallback", "m20", lambda value: value["runs"][0].update(fallback_count=1), "fallback"),
            ("protocol", "m20", lambda value: value["configuration"].update(retained_runs=9), "3-warm-up/10-retained"),
            ("12b output", "product-12b", lambda value: value["run_output"].update(output_token_ids=[9]), "12B output"),
            ("maximum", "product-26b", lambda value: value["model_report"].update(base_max_context=65536), "wrong base maximum"),
            ("binary", "product-26b", lambda value: value["qualified_binaries"].update(gem16_chat_sha256="x" * 64), "not bound"),
            ("relaunch", "relaunch", lambda value: value.update(extra=True), "fresh product driver"),
        )
        for label, stem, mutate, message in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                args = self.arguments(directory)
                attributes = {
                    "m20": "m20",
                    "m21": "m21",
                    "product-12b": "product_12b",
                    "product-26b": "product_26b",
                    "relaunch": "relaunch_driver",
                }
                path = getattr(args, attributes[stem])
                value = json.loads(path.read_text(encoding="utf-8"))
                mutate(value)
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaisesRegex(QualificationFailure, message):
                    qualify(args)

    def test_malformed_json_root_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            args = self.arguments(directory)
            args.m22.write_text("[]\n", encoding="utf-8")
            with self.assertRaisesRegex(QualificationFailure, "root is not an object"):
                qualify(args)


if __name__ == "__main__":
    unittest.main()

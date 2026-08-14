from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from tools.gem16_compile.common import BoundedWorkspace, DataError, SourceVerificationError, canonical_json_bytes
from tools.gem16_compile.fp8_report import (
    _attention_pairs, _pearson, compare_attention, write_report,
)
from tools.gem16_compile.native_fp8_compare import _read_metrics
from tools.gem16_compile.profiles import M05_ATTENTION_TABLE, M05_DEQUANTIZATION_EQUATION
from tools.gem16_compile.reader import TensorDescriptor


GLOBAL_LAYERS = {5, 11, 17, 23, 29}
ROLES = ("q", "k", "o")


def assert_json_schema(value, schema, root, path="$", test_case=None):
    """Small dependency-free validator for the retained comparison contract."""
    if "$ref" in schema:
        target = root
        for component in schema["$ref"].removeprefix("#/").split("/"):
            target = target[component]
        return assert_json_schema(value, target, root, path, test_case)
    if "const" in schema and value != schema["const"]:
        raise AssertionError(f"{path}: expected {schema['const']!r}, got {value!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise AssertionError(f"{path}: value is not in enum")
    expected_types = schema.get("type")
    if expected_types is not None:
        if not isinstance(expected_types, list):
            expected_types = [expected_types]
        def matches(kind):
            return {
                "object": isinstance(value, dict),
                "array": isinstance(value, list),
                "string": isinstance(value, str),
                "integer": isinstance(value, int) and not isinstance(value, bool),
                "number": isinstance(value, (int, float)) and not isinstance(value, bool),
                "boolean": isinstance(value, bool),
                "null": value is None,
            }[kind]
        if not any(matches(kind) for kind in expected_types):
            raise AssertionError(f"{path}: expected {expected_types}, got {type(value).__name__}")
    if isinstance(value, dict):
        for name in schema.get("required", ()):
            if name not in value:
                raise AssertionError(f"{path}: missing required {name}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            unknown = set(value) - set(properties)
            if unknown:
                raise AssertionError(f"{path}: unknown properties {sorted(unknown)}")
        for name, child in properties.items():
            if name in value:
                assert_json_schema(value[name], child, root, f"{path}.{name}", test_case)
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", len(value)):
            raise AssertionError(f"{path}: array length outside schema bounds")
        for index, child in enumerate(schema.get("prefixItems", ())):
            if index < len(value):
                assert_json_schema(value[index], child, root, f"{path}[{index}]", test_case)
        if "items" in schema:
            for index, item in enumerate(value):
                assert_json_schema(item, schema["items"], root, f"{path}[{index}]", test_case)
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise AssertionError(f"{path}: string shorter than schema minimum")
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            raise AssertionError(f"{path}: string does not match schema pattern")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise AssertionError(f"{path}: below schema minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise AssertionError(f"{path}: above schema maximum")


def tensor_names(include_global_v: bool = False):
    for layer in range(30):
        roles = ROLES if layer in GLOBAL_LAYERS else ROLES + ("v",)
        if include_global_v and layer == 5:
            roles = roles + ("v",)
        for role in roles:
            stem = f"model.language_model.layers.{layer}.self_attn.{role}_proj"
            yield stem + ".weight"
            yield stem + ".weight_scale"


def write_safetensors(path: Path, payloads: dict[str, tuple[str, list[int], bytes]]) -> None:
    offset = 0
    metadata: dict[str, object] = {"__metadata__": {"format": "pt"}}
    for name, (dtype, shape, payload) in payloads.items():
        metadata[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(payload)]}
        offset += len(payload)
    header = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode()
    header += b" " * ((-len(header)) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"".join(item[2] for item in payloads.values()))


def make_payloads(*, bad_dtype: bool = False, nan_code: bool = False, zero_scale: bool = False,
                  nonfinite_scale: bool = False, include_global_v: bool = False, changed: bool = False):
    payloads: dict[str, tuple[str, list[int], bytes]] = {}
    for index, name in enumerate(tensor_names(include_global_v)):
        if name.endswith(".weight"):
            value = 0x7F if nan_code and index == 0 else (0x39 if changed and index == 0 else 0x38)
            payloads[name] = ("F16" if bad_dtype and index == 0 else "F8_E4M3", [1, 2], bytes((value, value)))
        else:
            value = (0x7F80 if nonfinite_scale and index == 1 else
                     (0x0000 if zero_scale and index == 1 else
                      (0x4000 if changed and index == 1 else 0x3F80)))
            payloads[name] = ("BF16", [1, 1], struct.pack("<H", value))
    return payloads


def make_lock(root: Path, source_file: Path) -> Path:
    digest = hashlib.sha256(source_file.read_bytes()).hexdigest()
    lock = {
        "schema_version": 1,
        "repository": "test/example",
        "revision": "0123456789abcdef0123456789abcdef01234567",
        "files": [{"path": "model.safetensors", "size": source_file.stat().st_size,
                   "sha256": digest, "git_oid": "abcdef0123456789abcdef0123456789abcdef01"}],
    }
    lock_path = root / "lock.json"
    lock_path.write_bytes(canonical_json_bytes(lock))
    return lock_path


def make_artifact(root: Path, payloads: dict[str, tuple[str, list[int], bytes]]) -> Path:
    artifact = root / "compiled"
    artifact.mkdir()
    shard = artifact / "model-00001-of-00001.safetensors"
    write_safetensors(shard, payloads)
    index = artifact / "model.safetensors.index.json"
    index.write_bytes(canonical_json_bytes({"metadata": {"total_size": sum(len(v[2]) for v in payloads.values())},
                                             "weight_map": {name: shard.name for name in payloads}}))
    files = []
    for path, kind in ((shard, "safetensors_shard"), (index, "safetensors_index")):
        files.append({"path": path.name, "kind": kind, "size": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {
        "schema_version": 1, "artifact_profile": "fp8-attention-partial-v1",
        "artifact_status": "m05_fp8_attention_partial_not_runtime_loadable",
        "source": {},
        "compiler": {"implementation": "gem16_compile_m05_v1"},
        "plan": {},
        "quantization": {
            "profile": "fp8-attention-partial-v1",
            "attention": "fp8-per-output-row-v1",
            "experts": "deferred-to-m06",
            "embedding_head": "deferred-to-m07",
            "production_quantization_implemented": False,
        },
        "head_format": "deferred", "text_only": True,
        "omitted_families": ["audio", "mtp", "video", "vision"],
        "omitted_tensor_groups": [], "excluded_tensors": [],
        "files": files,
        "tensors": [{"output_name": name} for name in payloads],
        "file_hash_scope": "test",
        "byte_totals": {},
        "compiler_settings": {},
    }
    (artifact / "gem16_compilation.json").write_bytes(canonical_json_bytes(manifest))
    return artifact


class FP8ComparisonTest(unittest.TestCase):
    def fixture(self, **kwargs):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        source = root / "unsloth"
        source.mkdir()
        payload_options = dict(kwargs)
        payload_options.pop("changed_artifact", None)
        payloads = make_payloads(**payload_options)
        source_file = source / "model.safetensors"
        write_safetensors(source_file, payloads)
        lock = make_lock(root, source_file)
        artifact_payloads = make_payloads(changed=kwargs.get("changed_artifact", False))
        artifact = make_artifact(root, artifact_payloads)
        return temp, root, source, lock, artifact

    def run_compare(self, source, lock, artifact, staging=4096):
        workspace = BoundedWorkspace(64 * 1024 * 1024, staging)
        return compare_attention(artifact, lock, source, workspace)

    def test_exact_and_deterministic_report_is_bounded(self):
        temp, root, source, lock, artifact = self.fixture()
        self.addCleanup(temp.cleanup)
        first = self.run_compare(source, lock, artifact)
        second = self.run_compare(source, lock, artifact)
        first_semantic = dict(first); first_semantic.pop("memory")
        second_semantic = dict(second); second_semantic.pop("memory")
        self.assertEqual(first_semantic, second_semantic)
        self.assertEqual(first["aggregates"]["matrix_count"], 115)
        self.assertEqual(first["aggregates"]["raw_mismatch_count"], 0)
        self.assertEqual(first["aggregates"]["scale_mismatch_count"], 0)
        self.assertEqual(first["memory"]["staging_buffer_bytes"], 4096)
        self.assertLessEqual(first["memory"]["maximum_transform_row_bytes"], 4096)
        self.assertTrue(all(item["reconstruction"]["perfect_reconstruction"] for item in first["matrices"]))
        self.assertTrue(all(item["reconstruction"]["sqnr_db"] is None for item in first["matrices"]))

    def test_code_and_scale_mismatch_metrics(self):
        temp, root, source, lock, artifact = self.fixture(changed_artifact=True)
        self.addCleanup(temp.cleanup)
        report = self.run_compare(source, lock, artifact)
        self.assertEqual(report["aggregates"]["raw_mismatch_count"], 2)
        self.assertEqual(report["aggregates"]["scale_mismatch_count"], 1)
        first = next(item for item in report["matrices"] if item["name"].endswith("q_proj"))
        self.assertEqual(first["weight"]["raw_mismatch_count"], 2)
        self.assertEqual(first["scale"]["mismatch_count"], 1)
        self.assertGreater(first["reconstruction"]["max_absolute_error"], 0.0)
        self.assertIsNotNone(first["reconstruction"]["sqnr_db"])

    def test_rejects_global_v_dtype_nan_and_zero_scale(self):
        cases = ({"include_global_v": True}, {"bad_dtype": True}, {"nan_code": True},
                 {"zero_scale": True}, {"nonfinite_scale": True})
        for options in cases:
            with self.subTest(options=options):
                temp, root, source, lock, artifact = self.fixture(**options)
                self.addCleanup(temp.cleanup)
                with self.assertRaises((DataError, SourceVerificationError)):
                    self.run_compare(source, lock, artifact)

    def test_rejects_corrupt_lock_artifact_symlink_extra_and_existing_output(self):
        temp, root, source, lock, artifact = self.fixture()
        self.addCleanup(temp.cleanup)
        source_file = source / "model.safetensors"
        source_file.write_bytes(source_file.read_bytes() + b"x")
        with self.assertRaises(SourceVerificationError):
            self.run_compare(source, lock, artifact)

        temp2, root2, source2, lock2, artifact2 = self.fixture()
        self.addCleanup(temp2.cleanup)
        shard = artifact2 / "model-00001-of-00001.safetensors"
        shard.write_bytes(shard.read_bytes() + b"x")
        with self.assertRaises(DataError):
            self.run_compare(source2, lock2, artifact2)

        temp3, root3, source3, lock3, artifact3 = self.fixture()
        self.addCleanup(temp3.cleanup)
        (artifact3 / "extra").write_bytes(b"extra")
        with self.assertRaises(DataError):
            self.run_compare(source3, lock3, artifact3)
        temp4, root4, source4, lock4, artifact4 = self.fixture()
        self.addCleanup(temp4.cleanup)
        if hasattr(os, "symlink"):
            os.symlink(artifact4 / "model-00001-of-00001.safetensors", artifact4 / "link")
            with self.assertRaises(SourceVerificationError):
                self.run_compare(source4, lock4, artifact4)
        output = root3 / "report.json"
        output.write_bytes(b"existing")
        with self.assertRaises(Exception):
            write_report(output, {"status": "pass"})

    def test_production_shape_table_and_constant_scale_correlation(self):
        descriptors = {}
        for name, spec in M05_ATTENTION_TABLE.items():
            stem = name.removesuffix(".weight")
            descriptors[name] = TensorDescriptor(
                name, "F8_E4M3", spec.shape, "model.safetensors", Path("/tmp/model"),
                0, 0, spec.shape[0] * spec.shape[1], "0" * 64,
            )
            scale_name = stem + ".weight_scale"
            descriptors[scale_name] = TensorDescriptor(
                scale_name, "BF16", (spec.shape[0], 1), "model.safetensors", Path("/tmp/model"),
                0, 0, spec.shape[0] * 2, "0" * 64,
            )
        pairs = _attention_pairs(descriptors, "synthetic", production=True)
        self.assertEqual(len(pairs), 115)
        self.assertEqual(sum(pair.weight.byte_length for pair in pairs.values()), 1_110_179_840)
        self.assertEqual(sum(pair.scale.byte_length for pair in pairs.values()), 670_720)
        self.assertEqual(_pearson(4, 4.0, 8.0, 4.0, 16.0, 8.0, True), 1.0)
        self.assertEqual(_pearson(4, 4.0, 8.0, 4.0, 16.0, 8.0, False), 0.0)

    def test_comparison_cli_propagates_threads_to_verify_and_native_runner(self):
        import tools.compare_quantized_checkpoints as cli
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for name in ("compiled", "compiled_source", "compiled_plan", "unsloth", "unsloth_lock", "native"):
                path = root / name
                if name in {"compiled", "compiled_source", "unsloth"}:
                    path.mkdir()
                else:
                    path.write_bytes(b"input")
                paths[name] = path
            output = root / "report.json"
            lock_digest = hashlib.sha256(b"input").hexdigest()
            fake_report = {"compiled": {"source_contract": "gemma4-26b-source-bf16-attention-v1"}}
            with patch.object(cli, "M05_SOURCE_LOCK_SHA256", {"ordinary_bf16": lock_digest}), \
                 patch.object(cli, "verify_artifact", return_value={"artifact_profile": "fp8-attention-partial-v1"}) as verify, \
                 patch.object(cli, "compare_attention", return_value=fake_report) as compare, \
                 patch.object(cli, "write_report"):
                result = cli.main([
                    "--family", "attention", "--compiled", str(paths["compiled"]),
                    "--compiled-source-lock", str(paths["compiled_plan"]),
                    "--compiled-source", str(paths["compiled_source"]),
                    "--compiled-plan", str(paths["compiled_plan"]),
                    "--unsloth-lock", str(paths["unsloth_lock"]),
                    "--unsloth-source", str(paths["unsloth"]),
                    "--max-host-memory", "67108864", "--native-encoder", str(paths["native"]),
                    "--threads", "4", "--output", str(output),
                ])
            self.assertEqual(result, 0)
            self.assertEqual(verify.call_args.args[0].threads, 4)
            self.assertEqual(compare.call_args.kwargs["threads"], 4)

    def test_native_metrics_reconcile_nested_job_range_hashes(self):
        digest = "a" * 64
        build = {"compiler_id": "test", "compiler_version": "1", "build_type": "test", "cxx_standard": "20", "system": "Linux", "processor": "x86_64"}
        expected = {
            "threads": 4,
            "matrices": [{
                "name": "m", "layer": 0, "role": "q", "rows": 1, "columns": 1,
                "left_weight": {"sha256": digest}, "right_weight": {"sha256": digest},
                "left_scale": {"sha256": digest}, "right_scale": {"sha256": digest},
            }],
        }
        metric = {
            "name": "m", "layer": 0, "role": "q", "rows": 1, "columns": 1, "elements": 1,
            "left_weight_sha256": digest, "right_weight_sha256": digest,
            "left_scale_sha256": digest, "right_scale_sha256": digest,
            "raw_mismatch_count": 0, "left_endpoint_7e": 0, "left_endpoint_fe": 0,
            "right_endpoint_7e": 0, "right_endpoint_fe": 0, "left_nan_count": 0,
            "right_nan_count": 0, "scale_mismatch_count": 0,
            "left_scale_min": 1.0, "left_scale_max": 1.0, "right_scale_min": 1.0,
            "right_scale_max": 1.0, "left_scale_sum": 1.0, "right_scale_sum": 1.0,
            "left_scale_sum_squares": 1.0, "right_scale_sum_squares": 1.0,
            "scale_difference_sum_squares": 0.0, "scale_dot": 1.0,
            "scale_relative_l2": None, "scale_pearson_correlation": 1.0,
            "left_min": 0.0, "left_max": 0.0, "left_sum_squares": 0.0,
            "right_min": 0.0, "right_max": 0.0, "right_sum_squares": 0.0,
            "difference_sum_squares": 0.0, "reconstruction_dot": 0.0,
            "reconstruction_relative_l2": None, "cosine_similarity": 1.0,
            "max_absolute_error": 0.0, "sqnr_db": None,
            "perfect_reconstruction": True, "zero_reference": True,
        }
        document = {"schema_version": 1, "contract_id": "gem16.fp8_attention_compare",
                    "contract_version": 1, "threads": 4, "maximum_chunk_bytes": 1,
                    "native_build": build, "matrices": [metric]}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.json"
            path.write_bytes(canonical_json_bytes(document))
            parsed = _read_metrics(path, expected)
        self.assertEqual(parsed["matrices"][0]["left_weight_sha256"], digest)

    def test_schema_is_strict_machine_readable_json(self):
        schema = json.loads(Path("benchmarks/goldens/gemma4_26b/fp8/comparison-schema-v1.json").read_text())
        self.assertEqual(schema["properties"]["artifact_profile"]["const"], "fp8-attention-partial-v1")
        self.assertEqual(schema["properties"]["matrices"]["maxItems"], 115)
        self.assertEqual(schema["$defs"]["contract"]["properties"]["operator_output_comparison"]["const"],
                         "not_performed_in_M05_weight_comparison")
        self.assertEqual(schema["$defs"]["contract"]["properties"]["native_comparator_build"]["$ref"],
                         "#/$defs/nativeBuild")

    def test_compact_acceptance_retains_clean_report_contract(self):
        schema = json.loads(Path("benchmarks/goldens/gemma4_26b/fp8/comparison-schema-v1.json").read_text())
        acceptance = json.loads(Path("artifacts/m05/acceptance.json").read_text())
        raw_index = json.loads(Path("artifacts/raw-evidence-index.json").read_text())
        report = acceptance["ordinary_vs_unsloth"]
        indexed = {
            item["original_path"]: item
            for item in raw_index["raw_reports"]
        }["artifacts/m05/ordinary-vs-unsloth-fp8.json"]
        self.assertEqual(indexed["sha256"], report["sha256"])
        self.assertEqual(indexed["size"], 366830)
        self.assertEqual(acceptance["implementation"]["compiler_dirty"], False)
        self.assertEqual(
            schema["$defs"]["contract"]["properties"]["dequantization_equation"]["const"],
            M05_DEQUANTIZATION_EQUATION,
        )
        self.assertEqual(report["quality_claim"], "none; stored weights and row scales only")


if __name__ == "__main__":
    unittest.main()

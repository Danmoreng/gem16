from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest

from tools.gem16_compile.common import BoundedWorkspace, OutputError
from tools.gem16_compile.native_nvfp4 import (
    NativeNvfp4Request,
    _regular_executable,
    prepare_native_direct,
)
from tools.gem16_compile.plan import QuantizationPlan, TensorCompilePlan
from tools.gem16_compile.reader import TensorDescriptor
from tools.gem16_compile.writer import finalize_direct_shards, prepare_direct_shards


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "build/Linux/host-debug/bin/gem16-checkpoint-compiler"


@unittest.skipUnless(NATIVE.is_file(), "host native M06 compiler is not built")
class NativeNvfp4AdapterTest(unittest.TestCase):
    @staticmethod
    def make_fixture(root: Path) -> tuple[QuantizationPlan, dict[str, TensorDescriptor]]:
        source_name = "model.language_model.layers.0.experts.down_proj"
        source = root / "source.bf16"
        values = [0x3F80, 0xC000, 0x4000, 0x0000] * 4
        payload = b"".join(value.to_bytes(2, "little") for value in values)
        source.write_bytes(payload)
        common = dict(
            operation_id=f"fixture:{source_name}",
            source_names=(source_name,),
            transformation="nvfp4",
            transformation_version=1,
            logical_dtype="BF16",
            logical_shape=(1, 16),
            axis_transformation="identity",
            quantizer_parameters={},
            dequantization_equation="W = fp4 * local / divisor",
            role="routed_expert_down",
            residency_class="immutable_device_text",
            disk_layout="canonical_row_major_low_nibble_first",
            runtime_layout="expert_major_sm120_row8_k64",
            aliased=False,
        )
        components = (
            ("nvfp4-input-divisor-v1", "input_global_scale", "F32", (1,)),
            ("nvfp4-local-scale-v1", "weight_scale", "F8_E4M3", (1, 1)),
            ("nvfp4-packed-v1", "weight_packed", "U8", (1, 8)),
            ("nvfp4-weight-divisor-v1", "weight_global_scale", "F32", (1,)),
        )
        tensors = tuple(sorted([
            TensorCompilePlan(
                output_name=f"{source_name}.{suffix}",
                encoder=encoder,
                output_dtype=dtype,
                physical_shape=shape,
                **common,
            )
            for encoder, suffix, dtype, shape in components
        ], key=lambda tensor: tensor.output_name))
        plan = QuantizationPlan(
            schema_version=1,
            artifact_profile="nvfp4-experts-partial-v1",
            head_format="deferred",
            source_contract="fixture",
            compiler_manifest_sha256="a" * 64,
            resolved_plan_sha256="b" * 64,
            target_shard_bytes=1024,
            approved_metadata_files=(),
            omitted_families=("audio", "mtp", "video", "vision"),
            tensors=tensors,
            excluded_tensors=(),
            reference_environment={},
        )
        descriptor = TensorDescriptor(
            name=source_name,
            dtype="BF16",
            shape=(1, 16),
            shard=source.name,
            path=source,
            absolute_offset=0,
            data_offset=0,
            byte_length=len(payload),
            shard_sha256=hashlib.sha256(payload).hexdigest(),
        )
        return plan, {source_name: descriptor}

    def test_native_writes_only_prepared_ranges_and_finalizes_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, sources = self.make_fixture(root)
            staging = root / "artifact.incomplete"
            staging.mkdir()
            workspace = BoundedWorkspace(512 * 1024 * 1024, 4096)
            layout = prepare_direct_shards(staging, plan, workspace)
            shard = staging / layout.shard_names[0]
            before = shard.read_bytes()
            result = prepare_native_direct(
                NativeNvfp4Request(NATIVE, timeout_seconds=20, threads=2),
                plan, sources, workspace, staging, layout,
            )
            payload = finalize_direct_shards(staging, plan, layout, result, workspace)
            self.assertEqual(len(payload.tensors), 4)
            self.assertEqual(shard.read_bytes()[: len(before[:8])], before[:8])
            self.assertTrue((staging / layout.index_name).is_file())
            self.assertTrue(all(t.source_result.output_sha256 for t in payload.tensors))
            self.assertEqual(result.threads, 2)
            self.assertEqual(result.source_passes, 2)

    def test_verified_snapshot_leaf_symlink_resolves_before_native_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, sources = self.make_fixture(root)
            source = next(iter(sources.values()))
            blob = root / "blob.bf16"
            source.path.rename(blob)
            source.path.symlink_to(blob)
            sources = {
                source.name: TensorDescriptor(
                    name=source.name,
                    dtype=source.dtype,
                    shape=source.shape,
                    shard=source.shard,
                    path=source.path,
                    absolute_offset=source.absolute_offset,
                    data_offset=source.data_offset,
                    byte_length=source.byte_length,
                    shard_sha256=source.shard_sha256,
                )
            }
            staging = root / "artifact.incomplete"
            staging.mkdir()
            workspace = BoundedWorkspace(512 * 1024 * 1024, 4096)
            layout = prepare_direct_shards(staging, plan, workspace)
            result = prepare_native_direct(
                NativeNvfp4Request(NATIVE, timeout_seconds=20, threads=2),
                plan, sources, workspace, staging, layout,
            )
            payload = finalize_direct_shards(staging, plan, layout, result, workspace)
            self.assertEqual(len(payload.tensors), 4)

    def test_build_info_is_strict_and_reports_release_identity(self) -> None:
        result = subprocess.run(
            [str(NATIVE), "--build-info"], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0)
        document = __import__("json").loads(result.stdout)
        self.assertEqual(document["protocol"], "gem16-nvfp4-direct-v1")
        self.assertIn(document["native_build"]["build_type"], {"Debug", "Release"})
        mixed = subprocess.run(
            [str(NATIVE), "--build-info", "--telemetry", "ignored"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(mixed.returncode, 2)

    def test_symlink_native_executable_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            link = root / "native-link"
            try:
                link.symlink_to(NATIVE)
            except OSError as error:
                self.skipTest(f"symbolic links unavailable: {error}")
            with self.assertRaises(OutputError):
                _regular_executable(link)


if __name__ == "__main__":
    unittest.main()

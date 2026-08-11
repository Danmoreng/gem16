from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest

from tools.gem16_compile.common import BoundedWorkspace, OutputError, SourceVerificationError
from tools.gem16_compile.native_fp8 import (
    NativeBundleEncoder,
    NativeRequest,
    prepare_native_bundle,
)
from tools.gem16_compile.plan import QuantizationPlan, TensorCompilePlan
from tools.gem16_compile.reader import TensorDescriptor


@unittest.skipUnless(os.name == "posix", "native M05 adapter is Linux/POSIX-only")
class NativeFp8AdapterTest(unittest.TestCase):
    @staticmethod
    def _plan() -> QuantizationPlan:
        common = dict(
            operation_id="fp8-attention:test.weight",
            source_names=("test.weight",),
            transformation="fp8-rowwise",
            transformation_version=1,
            logical_dtype="BF16",
            axis_transformation="identity",
            quantizer_parameters={"contract_id": "gem16.fp8_attention_rowwise", "contract_version": 1},
            dequantization_equation="bf16 ~= fp8 * bf16_scale",
            role="attention_q_projection",
            residency_class="immutable_device_text",
            disk_layout="row-major",
            runtime_layout="row-major",
            aliased=False,
        )
        weight = TensorCompilePlan(
            output_name="test.weight", encoder="fp8-rowwise-weight-v1",
            output_dtype="F8_E4M3", physical_shape=(1, 4), logical_shape=(1, 4), **common,
        )
        scale = TensorCompilePlan(
            output_name="test.weight_scale", encoder="fp8-rowwise-scale-v1",
            output_dtype="BF16", physical_shape=(1, 1), logical_shape=(1, 1), **common,
        )
        return QuantizationPlan(
            schema_version=1, artifact_profile="fp8-attention-partial-v1",
            head_format="deferred", source_contract="test",
            compiler_manifest_sha256="a" * 64, resolved_plan_sha256="b" * 64,
            target_shard_bytes=1024, approved_metadata_files=(),
            omitted_families=("audio", "mtp", "video", "vision"),
            tensors=(weight, scale), excluded_tensors=(), reference_environment={},
        )

    @staticmethod
    def _fake_encoder(path: Path) -> None:
        path.write_text(
            "#!/usr/bin/env python3\n"
            "import hashlib, json, pathlib, sys\n"
            "args = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n"
            "job = json.loads(pathlib.Path(args['--job']).read_text())\n"
            "m = job['matrices'][0]\n"
            "source = pathlib.Path(m['source_path']).read_bytes()[m['source_offset']:m['source_offset'] + m['source_bytes']]\n"
            "weight = bytes([0, 0, 0, 0])\n"
            "scale = bytes([0, 0])\n"
            "pathlib.Path(args['--payload']).write_bytes(weight + scale)\n"
            "telemetry = {'schema_version': 1, 'contract_id': 'gem16.fp8_attention_rowwise', 'contract_version': 1, 'payload_bytes': 6, 'threads': job['threads'], 'maximum_source_row_bytes': 8, 'native_build': {'compiler_id': 'test', 'compiler_version': '1', 'build_type': 'test', 'cxx_standard': '20', 'system': 'Linux', 'processor': 'x86_64'}, 'matrices': [{\n"
            "'source_name': m['source_name'], 'weight_output_name': m['weight_output_name'], 'scale_output_name': m['scale_output_name'], 'rows': 1, 'columns': 4, 'elements': 4,\n"
            "'source_sha256': hashlib.sha256(source).hexdigest(), 'weight_sha256': hashlib.sha256(weight).hexdigest(), 'scale_sha256': hashlib.sha256(scale).hexdigest(),\n"
            "'source_min': 0.0, 'source_max': 1.0, 'source_sum_squares': 1.0, 'reconstruction_sum_squares': 0.0, 'source_reconstruction_dot': 0.0, 'error_sum_squares': 1.0, 'max_absolute_error': 1.0, 'scale_min': 1.0, 'scale_max': 1.0, 'saturation_count': 0, 'zero_rows': 0, 'underflow_clamped_rows': 0, 'histogram': [4] + [0] * 255}]}\n"
            "pathlib.Path(args['--telemetry']).write_text(json.dumps(telemetry))\n",
            encoding="utf-8",
        )
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    def test_stages_once_and_reconciles_paired_hashes_for_threads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.bin"
            source = bytes.fromhex("803f00c0003f0080")
            source_path.write_bytes(source)
            executable = root / "fake_encoder.py"
            self._fake_encoder(executable)
            descriptor = TensorDescriptor(
                name="test.weight", dtype="BF16", shape=(1, 4), shard="source.bin",
                path=source_path, absolute_offset=0, data_offset=0,
                byte_length=len(source), shard_sha256=hashlib.sha256(source).hexdigest(),
            )
            plan = self._plan()
            for threads in (1, 4):
                staging = root / f"staging-{threads}"
                staging.mkdir(mode=0o700)
                workspace = BoundedWorkspace(512 * 1024 * 1024, 4096)
                bundle = prepare_native_bundle(
                    NativeRequest(executable, timeout_seconds=10, threads=threads),
                    plan, {descriptor.name: descriptor}, workspace, staging,
                )
                self.assertEqual(bundle.binary_sha256, hashlib.sha256(executable.read_bytes()).hexdigest())
                self.assertEqual(bundle.threads, threads)
                self.assertEqual(bundle.matrices["test.weight"].source_sha256, hashlib.sha256(source).hexdigest())
                output = __import__("io").BytesIO()
                NativeBundleEncoder(bundle, "weight").compile_tensor(
                    plan.tensors[0], (descriptor,), output, workspace,
                )
                self.assertEqual(output.getvalue(), b"\0\0\0\0")
                bundle.cleanup()
                self.assertFalse((staging / ".m05_fp8_encoder").exists())

    def test_timeout_terminates_native_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.bin"
            source_path.write_bytes(bytes.fromhex("803f00c0003f0080"))
            executable = root / "timeout_encoder.py"
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "import subprocess, sys, time\n"
                "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                "time.sleep(60)\n",
                encoding="utf-8",
            )
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
            descriptor = TensorDescriptor(
                name="test.weight", dtype="BF16", shape=(1, 4), shard="source.bin",
                path=source_path, absolute_offset=0, data_offset=0,
                byte_length=8, shard_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
            )
            staging = root / "staging"
            staging.mkdir(mode=0o700)
            workspace = BoundedWorkspace(512 * 1024 * 1024, 4096)
            with self.assertRaises(OutputError):
                prepare_native_bundle(
                    NativeRequest(executable, timeout_seconds=1),
                    self._plan(), {descriptor.name: descriptor}, workspace, staging,
                )
            self.assertFalse((staging / ".m05_fp8_encoder").exists())
            self.assertFalse((staging / ".m05_fp8_payload.bin").exists())
            self.assertFalse((staging / ".m05_fp8_telemetry.json").exists())

    def test_successful_leader_with_live_descendant_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.bin"
            source_path.write_bytes(bytes.fromhex("803f00c0003f0080"))
            executable = root / "leader_exit_encoder.py"
            child_pid_path = root / "child.pid"
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib, subprocess, sys\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import os,time; os.close(1); os.close(2); time.sleep(60)'])\n"
                f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid))\n"
                "sys.exit(0)\n",
                encoding="utf-8",
            )
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
            descriptor = TensorDescriptor(
                name="test.weight", dtype="BF16", shape=(1, 4), shard="source.bin",
                path=source_path, absolute_offset=0, data_offset=0,
                byte_length=8, shard_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
            )
            staging = root / "staging"
            staging.mkdir(mode=0o700)
            workspace = BoundedWorkspace(512 * 1024 * 1024, 4096)
            with self.assertRaises(OutputError):
                prepare_native_bundle(
                    NativeRequest(executable, timeout_seconds=5), self._plan(),
                    {descriptor.name: descriptor}, workspace, staging,
                )
            self.assertFalse((staging / ".m05_fp8_encoder").exists())
            child_pid = int(child_pid_path.read_text(encoding="ascii"))
            self.assertFalse((Path("/proc") / str(child_pid)).exists())

    def test_native_source_exit_code_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.bin"
            source_path.write_bytes(bytes.fromhex("803f00c0003f0080"))
            executable = root / "source_error.py"
            executable.write_text("#!/usr/bin/env python3\nraise SystemExit(3)\n", encoding="utf-8")
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
            descriptor = TensorDescriptor(
                name="test.weight", dtype="BF16", shape=(1, 4), shard="source.bin",
                path=source_path, absolute_offset=0, data_offset=0,
                byte_length=8, shard_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
            )
            staging = root / "staging"
            staging.mkdir(mode=0o700)
            workspace = BoundedWorkspace(512 * 1024 * 1024, 4096)
            with self.assertRaises(SourceVerificationError):
                prepare_native_bundle(
                    NativeRequest(executable), self._plan(),
                    {descriptor.name: descriptor}, workspace, staging,
                )

    def test_invalid_native_numeric_telemetry_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.bin"
            source_path.write_bytes(bytes.fromhex("803f00c0003f0080"))
            executable = root / "fake_encoder.py"
            self._fake_encoder(executable)
            executable.write_text(
                executable.read_text(encoding="utf-8").replace(
                    "'source_sum_squares': 1.0", "'source_sum_squares': -1.0"
                ),
                encoding="utf-8",
            )
            descriptor = TensorDescriptor(
                name="test.weight", dtype="BF16", shape=(1, 4), shard="source.bin",
                path=source_path, absolute_offset=0, data_offset=0,
                byte_length=8, shard_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
            )
            staging = root / "staging"
            staging.mkdir(mode=0o700)
            workspace = BoundedWorkspace(512 * 1024 * 1024, 4096)
            from tools.gem16_compile.common import DataError
            with self.assertRaises(DataError):
                prepare_native_bundle(
                    NativeRequest(executable), self._plan(),
                    {descriptor.name: descriptor}, workspace, staging,
                )

    def test_symlink_executable_is_rejected_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.write_bytes(b"not executable")
            link = root / "link"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("symbolic links unavailable")
            with self.assertRaises(OutputError):
                from tools.gem16_compile.native_fp8 import _regular_executable
                _regular_executable(link)


if __name__ == "__main__":
    unittest.main()

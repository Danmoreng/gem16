"""Bounded CLI contract checks for the M07 actual-artifact diagnostic."""
from __future__ import annotations

from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def _binary() -> Path | None:
    for preset in ("host-debug", "host-release", "host-sanitize"):
        candidate = ROOT / "build" / "Linux" / preset / "bin" / "gem16-nvfp4-head-diagnostic"
        if candidate.is_file():
            return candidate
    return None


@unittest.skipUnless(_binary() is not None, "host diagnostic executable is not built")
class M07Nvfp4HeadDiagnosticTest(unittest.TestCase):
    def test_rejects_missing_model_before_report_creation(self) -> None:
        binary = _binary()
        assert binary is not None
        with tempfile.TemporaryDirectory(prefix="gem16-m07-diag-") as directory:
            output = Path(directory) / "report.json"
            result = subprocess.run(
                [str(binary), "--model", str(ROOT / "does-not-exist"), "--output", str(output)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse(output.exists())

    def test_rejects_duplicate_arguments(self) -> None:
        binary = _binary()
        assert binary is not None
        with tempfile.TemporaryDirectory(prefix="gem16-m07-diag-") as directory:
            result = subprocess.run(
                [
                    str(binary),
                    "--model", str(ROOT / "does-not-exist"),
                    "--model", str(ROOT / "does-not-exist"),
                    "--output", str(Path(directory) / "report.json"),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("duplicate argument", result.stderr)

    def test_rejects_m06_artifact_before_report_creation(self) -> None:
        binary = _binary()
        assert binary is not None
        artifact = Path("/tmp/m06-release-81055eb48e05321481a8b63dd0dc5e7e017a7c00/qat-artifact")
        if not artifact.is_dir():
            self.skipTest("M06 external artifact is unavailable")
        with tempfile.TemporaryDirectory(prefix="gem16-m07-diag-") as directory:
            output = Path(directory) / "report.json"
            result = subprocess.run(
                [str(binary), "--model", str(artifact), "--output", str(output)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())

    def test_rejects_report_inside_artifact(self) -> None:
        binary = _binary()
        assert binary is not None
        artifact = Path("/tmp/m06-release-81055eb48e05321481a8b63dd0dc5e7e017a7c00/qat-artifact")
        if not artifact.is_dir():
            self.skipTest("M06 external artifact is unavailable")
        fd, output_name = tempfile.mkstemp(prefix="gem16-m07-report-", dir=artifact)
        os.close(fd)
        Path(output_name).unlink()
        output = Path(output_name)
        result = subprocess.run(
            [str(binary), "--model", str(artifact), "--output", str(output)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

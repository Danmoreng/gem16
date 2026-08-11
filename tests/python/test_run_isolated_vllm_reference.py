from __future__ import annotations

import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/run_isolated_vllm_reference.sh"


class RunIsolatedVllmReferenceTest(unittest.TestCase):
    def test_script_records_fixed_linux_policy(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        for required in (
            "MemoryHigh=40G",
            "MemoryMax=45G",
            "OOMPolicy=kill",
            "KillMode=control-group",
            "MAX_JOBS=4",
            "FLASHINFER_NVCC_THREADS=1",
        ):
            self.assertIn(required, source)

    @unittest.skipUnless(os.name == "posix", "Linux systemd wrapper")
    def test_shell_syntax_is_valid(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    @unittest.skipUnless(os.name == "posix", "Linux systemd wrapper")
    def test_dry_run_records_fixed_limits_and_preserves_command(self) -> None:
        result = subprocess.run(
            [
                "bash",
                str(SCRIPT),
                "--dry-run",
                "--",
                "/usr/bin/printf",
                "%s",
                "argument with spaces",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("MemoryHigh:40G MemoryMax:45G", result.stdout)
        self.assertIn("MAX_JOBS:4 FLASHINFER_NVCC_THREADS:1", result.stdout)
        self.assertIn("OOMPolicy=kill", result.stdout)
        self.assertIn("KillMode=control-group", result.stdout)
        self.assertIn("/usr/bin/printf", result.stdout)
        self.assertIn("argument\\ with\\ spaces", result.stdout)

    @unittest.skipUnless(os.name == "posix", "Linux systemd wrapper")
    def test_command_separator_is_required(self) -> None:
        result = subprocess.run(
            ["bash", str(SCRIPT), "/usr/bin/true"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("expected -- before COMMAND", result.stderr)


if __name__ == "__main__":
    unittest.main()

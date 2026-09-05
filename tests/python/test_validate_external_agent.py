import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from tools.validate_external_agent import run_agent


class AgentProcessTests(unittest.TestCase):
    def test_timeout_stops_tool_child_before_fixture_cleanup(self):
        with tempfile.TemporaryDirectory(prefix="gem16-agent-cleanup-test-") as temporary:
            root = Path(temporary)
            marker = root / "child-progress.txt"
            child = (
                "import time\n"
                "from pathlib import Path\n"
                f"p = Path({str(marker)!r})\n"
                "while True:\n"
                " p.write_text(str(time.monotonic_ns()))\n"
                " time.sleep(0.05)\n"
            )
            parent = (
                "import subprocess, sys, time\n"
                f"subprocess.Popen([sys.executable, '-c', {child!r}])\n"
                "time.sleep(60)\n"
            )
            with (root / "output.txt").open("w") as output:
                with self.assertRaises(subprocess.TimeoutExpired):
                    run_agent(
                        [sys.executable, "-c", parent], cwd=root, env=os.environ.copy(),
                        stdout=output, timeout=2,
                    )
            self.assertTrue(marker.exists(), "tool child must actually have started")
            progress = marker.read_text()
            time.sleep(0.25)
            self.assertEqual(marker.read_text(), progress, "tool child survived timeout")

    def test_success_returns_exit_status(self):
        with tempfile.TemporaryDirectory(prefix="gem16-agent-exit-test-") as temporary:
            with (Path(temporary) / "output.txt").open("w") as output:
                result = run_agent(
                    [sys.executable, "-c", "raise SystemExit(7)"], cwd=temporary,
                    env=os.environ.copy(), stdout=output, timeout=10,
                )
            self.assertEqual(result.returncode, 7)


if __name__ == "__main__":
    unittest.main()

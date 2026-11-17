from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from write_decode_ap1_evidence import sanitize, technical_environment


class PublicationTest(unittest.TestCase):
    def test_removes_personal_identifiers_without_changing_samples_or_technical_facts(self):
        raw = {"path": "/home/alice/project/run", "pid": "1234", "hostname": "my-pc",
               "/home/alice/lib": {"uuid": "private", "sha256": "abc"},
               "message": "nsys-alice on my-pc GPU-12345678-1234-1234-1234-123456789abc",
               "technical": {"gpu_model": "GPU X", "driver": "123.45", "vram": 16384},
               "samples": [139.1234, 140.5678], "unknown": None}
        replacements = [("/home/alice/project", "${REPO_ROOT}"),
                        ("/home/alice", "${HOME}"), ("my-pc", "${HOST}"), ("alice", "${USER}")]
        result = sanitize(raw, replacements)
        self.assertNotIn("pid", result)
        self.assertNotIn("hostname", result)
        self.assertEqual(result["${HOME}/lib"], {"sha256": "abc"})
        self.assertEqual(result["path"], "${REPO_ROOT}/run")
        self.assertEqual(result["samples"], raw["samples"])
        self.assertEqual(result["technical"], raw["technical"])
        self.assertIsNone(result["unknown"])
        self.assertEqual(result["message"], "nsys-${USER} on ${HOST} ${GPU_UUID}")

    def test_environment_allowlist_excludes_raw_machine_identifiers(self):
        raw = {"cpu": {"stdout": "Model name: CPU X\n"},
               "gpu": {"stdout": "Product Name : GPU X\nDriver Version : 1.2\nTotal : 16 GiB\nUUID : SECRET\n"},
               "platform_profile": {"stdout": "max-power\n"}, "powerd": {"stdout": "active\n"}}
        for key in ("nvcc", "cmake", "gcc", "nsys", "ncu"):
            raw[key] = {"stdout": key + " version"}
        result = technical_environment(raw)
        self.assertEqual(result["driver_version"], "1.2")
        self.assertEqual(result["gpu_memory_total"], "16 GiB")
        self.assertNotIn("SECRET", str(result))


if __name__ == "__main__":
    unittest.main()

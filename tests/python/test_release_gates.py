import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from verify_release_gates import verify, GATES, PLATFORMS
from package_server import sha256, package


class ReleaseGatesTest(unittest.TestCase):
    def test_integrity_and_missing_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commit = "a" * 40
            artifact = root / "artifact"
            artifact.write_bytes(b"binary")
            raw = root / "raw"
            raw.write_text("actual command and output fixture")
            manifest = {
                "commit": commit,
                "version": "test",
                "dirty": False,
                "platforms": {},
            }
            for platform in PLATFORMS:
                entry = {
                    "artifact": "artifact",
                    "sha256": sha256(artifact),
                    "gates": {},
                }
                for gate in GATES:
                    evidence = root / f"{platform}-{gate}.json"
                    evidence.write_text(
                        json.dumps(
                            {
                                "commit": commit,
                                "version": "test",
                                "platform": platform,
                                "gate": gate,
                                "status": "passed",
                                "artifact_sha256": sha256(artifact),
                                "samples": [{"path": "raw", "sha256": sha256(raw)}],
                            }
                        )
                    )
                    entry["gates"][gate] = {
                        "path": evidence.name,
                        "sha256": sha256(evidence),
                    }
                manifest["platforms"][platform] = entry
            path = root / "release.json"
            path.write_text(json.dumps(manifest))
            verify(path, commit, "test")
            with self.assertRaises(ValueError):
                verify(path, "b" * 40, "test")
            with self.assertRaises(ValueError):
                verify(path, commit, "wrong")
            artifact.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                verify(path, commit, "test")
            artifact.write_bytes(b"binary")
            raw.write_text("tampered")
            with self.assertRaises(ValueError):
                verify(path, commit, "test")
            raw.write_text("actual command and output fixture")
            manifest["platforms"]["windows-x64"]["gates"].pop("gpu_compact_vision")
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                verify(path, commit, "test")

    def test_fresh_headless_stage_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "gem16-server"
            binary.write_bytes(b"candidate binary fixture")
            output = root / "packages"
            output.mkdir()
            (output / "stale-secret.txt").write_text("must not be packaged")
            version = (ROOT / "VERSION").read_text().strip()

            def command(arguments, **kwargs):
                if arguments[-1] == "--version":
                    return f"gem16-server {version}"
                if arguments[1:3] == ["rev-parse", "HEAD"]:
                    return "a" * 40
                return b""

            with mock.patch(
                "package_server.subprocess.check_output", side_effect=command
            ):
                archive = package(binary, output, "linux-x64")
                # A second run must still use only its fresh allowlist.
                package(binary, output, "linux-x64")
            with tarfile.open(archive) as package_tar:
                names = package_tar.getnames()
                self.assertFalse(any("stale-secret" in name for name in names))
                manifest_name = next(
                    name for name in names if name.endswith("/manifest.json")
                )
                manifest = json.load(package_tar.extractfile(manifest_name))
                prefix = manifest_name.removesuffix("manifest.json")
                self.assertIn("tools/fetch_model.py", manifest["files"])
                self.assertIn("README.txt", manifest["files"])
                for name, digest in manifest["files"].items():
                    import hashlib

                    self.assertEqual(
                        hashlib.sha256(
                            package_tar.extractfile(prefix + name).read()
                        ).hexdigest(),
                        digest,
                    )


if __name__ == "__main__":
    unittest.main()

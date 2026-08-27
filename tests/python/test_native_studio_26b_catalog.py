import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "nativeStudio/src/model_manager.cpp"


class NativeStudio26BCatalogTest(unittest.TestCase):
    def test_compiled_catalog_matches_remote_locks(self):
        text = SOURCE.read_text(encoding="utf-8")
        blocks = {
            "target": text.split("constexpr std::array kTargetFiles{", 1)[1]
            .split("};", 1)[0],
            "assistant": text.split("constexpr std::array kAssistantFiles{", 1)[1]
            .split("};", 1)[0],
        }
        pattern = re.compile(
            r'LockedFile\{"([^"]+)",\s*(\d+)(?:ULL)?,\s*'
            r'"([0-9a-f]{64})",\s*"([0-9a-f]{40}|[0-9a-f]{64})"\}'
        )
        for kind, lock_name in (
            ("target", "gemma4-26b-gem16-target.lock.json"),
            ("assistant", "gemma4-26b-gem16-assistant.lock.json"),
        ):
            lock = json.loads((ROOT / "models" / lock_name).read_text())
            actual = {
                path: {
                    "size": int(size),
                    "sha256": sha256,
                    "blob_id": blob_id,
                }
                for path, size, sha256, blob_id in pattern.findall(blocks[kind])
            }
            expected = {
                item["path"]: {
                    "size": item["size"],
                    "sha256": item["sha256"],
                    "blob_id": item.get("lfs_oid") or item["git_oid"],
                }
                for item in lock["files"]
            }
            self.assertEqual(expected, actual)
            self.assertIn(lock["repository"], text)
            self.assertIn(lock["revision"], text)


if __name__ == "__main__":
    unittest.main()

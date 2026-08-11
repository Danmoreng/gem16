from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from tools.inventory_safetensors import make_inventory


class InventorySafetensorsTest(unittest.TestCase):
    def test_inventory_reads_header_without_loading_tensor_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            header = json.dumps(
                {"tensor": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}},
                separators=(",", ":"),
            ).encode("utf-8")
            payload = struct.pack("<Q", len(header)) + header + b"data"
            model_file = model / "model.safetensors"
            model_file.write_bytes(payload)
            lock_path = root / "model.lock.json"
            lock_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "repository": "owner/model",
                        "revision": "1" * 40,
                        "files": [
                            {
                                "path": "model.safetensors",
                                "size": len(payload),
                                "sha256": hashlib.sha256(payload).hexdigest(),
                                "git_oid": "2" * 40,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            inventory = make_inventory(model, lock_path, "fixture")
            self.assertEqual(inventory["checkpoint"]["tensor_count"], 1)
            self.assertEqual(inventory["checkpoint"]["tensor_payload_bytes"], 4)
            self.assertEqual(
                inventory["tensors"],
                [
                    {
                        "name": "tensor",
                        "dtype": "U8",
                        "shape": [4],
                        "bytes": 4,
                        "shard": "model.safetensors",
                        "absolute_offset": 8 + len(header),
                        "aliases": [],
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()

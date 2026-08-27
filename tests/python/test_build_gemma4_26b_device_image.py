import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.build_gemma4_26b_device_image import TensorPlan, _align_up, _write_tiled


def reference_tiling(source: bytes, rows: int, k_blocks: int, block_bytes: int) -> bytes:
    result = bytearray()
    row_bytes = k_blocks * block_bytes
    for first_row in range(0, rows, 8):
        tile_rows = min(8, rows - first_row)
        for k_block in range(k_blocks):
            for row in range(tile_rows):
                begin = (first_row + row) * row_bytes + k_block * block_bytes
                result.extend(source[begin : begin + block_bytes])
    return bytes(result)


class DeviceImageTilingTest(unittest.TestCase):
    def run_case(self, runtime_layout: str, rows: int, contracting: int) -> None:
        scale = runtime_layout.endswith("group16_e4m3")
        block_bytes = 4 if scale else 32
        k_blocks = contracting // 64
        byte_length = rows * k_blocks * block_bytes
        source = bytes((index * 37 + 11) & 0xFF for index in range(byte_length))
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            shard = directory / "model.safetensors"
            shard.write_bytes(source)
            output = directory / "image.bin"
            output.write_bytes(bytes(byte_length))
            destination = np.memmap(output, dtype=np.uint8, mode="r+", shape=(byte_length,))
            physical_shape = (
                (rows, contracting // 16) if scale else (rows, contracting // 2)
            )
            tensor = TensorPlan(
                name="weight_scale" if scale else "weight_packed",
                shard=shard.name,
                source_offset=0,
                destination_offset=0,
                byte_length=byte_length,
                runtime_layout=runtime_layout,
                physical_shape=physical_shape,
                logical_shape=(rows, contracting),
                source_sha256="unused",
            )
            _write_tiled(destination, directory, tensor, staging_bytes=257)
            destination.flush()
            del destination
            self.assertEqual(
                output.read_bytes(),
                reference_tiling(source, rows, k_blocks, block_bytes),
            )

    def test_packed_weight_full_and_tail_tiles(self) -> None:
        self.run_case("expert_major_sm120_row8_k64", rows=17, contracting=128)

    def test_scale_full_and_tail_tiles(self) -> None:
        self.run_case(
            "expert_major_sm120_row8_group16_e4m3", rows=17, contracting=128
        )

    def test_arena_alignment(self) -> None:
        self.assertEqual(_align_up(0), 0)
        self.assertEqual(_align_up(1), 256)
        self.assertEqual(_align_up(256), 256)
        self.assertEqual(_align_up(257), 512)


if __name__ == "__main__":
    unittest.main()

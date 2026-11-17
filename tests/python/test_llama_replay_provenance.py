import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
try:
    spec = importlib.util.spec_from_file_location("run_llama_decode_replay", TOOLS / "run_llama_decode_replay.py")
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
finally:
    sys.path.remove(str(TOOLS))


def encoded(path):
    return "GEM16_REPLAY_MODULE_HEX=" + str(path).encode().hex()


class LlamaReplayProvenanceTest(unittest.TestCase):
    def test_hashes_actual_central_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            paths = [directory / "libllama.so", directory / "libggml-cuda.so"]
            for p in paths:
                p.write_bytes(b"fixture")
            result = replay.loaded_modules("\n".join(map(encoded, paths)), directory)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[str(paths[0])]["bytes"], 7)

    def test_rejects_noncentral_module_even_with_central_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            central = directory / "central"
            central.mkdir()
            wrong = directory / "libllama.so"
            wrong.touch()
            with self.assertRaisesRegex(ValueError, "noncentral"):
                replay.loaded_modules(encoded(wrong), central)

    def test_requires_actual_cuda_and_llama_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "actually loaded"):
                replay.loaded_modules("", Path(tmp))
            with self.assertRaisesRegex(ValueError, "absolute"):
                replay.loaded_modules(encoded(Path("libllama.so")), Path(tmp))


if __name__ == "__main__":
    unittest.main()

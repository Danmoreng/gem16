import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('experiment_scratch', ROOT/'tools/experiment_scratch.py')
scratch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scratch)

class ExperimentScratchTests(unittest.TestCase):
    def test_refuses_ssd_before_creating_directory(self):
        with tempfile.TemporaryDirectory() as parent:
            target = Path(parent)/'must-not-exist'
            with patch.object(scratch.subprocess,'check_output',return_value='ext4\n'):
                with self.assertRaisesRegex(ValueError,'must be tmpfs'):
                    scratch.check_ram_storage(target)
            self.assertFalse(target.exists())

    def test_keeps_ram_reserve(self):
        with patch.object(scratch.subprocess,'check_output',return_value='tmpfs\n'), \
             patch.object(scratch.shutil,'disk_usage',return_value=type('Usage',(),{'free':32*scratch.GIB})()), \
             patch.object(scratch,'_available_ram',return_value=11*scratch.GIB):
            with self.assertRaisesRegex(ValueError,'available RAM'):
                scratch.check_ram_storage(Path('/tmp'),expected_gib=4)

    def test_keeps_tmpfs_reserve(self):
        with patch.object(scratch.subprocess,'check_output',return_value='tmpfs\n'), \
             patch.object(scratch.shutil,'disk_usage',return_value=type('Usage',(),{'free':4*scratch.GIB})()):
            with self.assertRaisesRegex(ValueError,'tmpfs space'):
                scratch.check_ram_storage(Path('/tmp'),expected_gib=4)

    def test_rejects_oversize_and_path_labels(self):
        for size in [0,-1,float('nan'),17]:
            with self.assertRaises(ValueError):scratch.check_ram_storage(Path('/tmp'),size)
        with self.assertRaises(ValueError):
            with scratch.experiment_scratch('../outside'):pass

    def test_context_cleans_up_when_analysis_fails(self):
        with tempfile.TemporaryDirectory() as parent, \
             patch.dict(os.environ,{'GEM16_EXPERIMENT_TMPDIR':parent}), \
             patch.object(scratch,'check_ram_storage'):
            with self.assertRaisesRegex(RuntimeError,'analysis failed'):
                with scratch.experiment_scratch('test') as directory:
                    (directory/'trace.sqlite').write_bytes(b'temporary')
                    raise RuntimeError('analysis failed')
            self.assertFalse(directory.exists())

    def test_cli_replaces_placeholder_without_shell_and_returns_child_status(self):
        with tempfile.TemporaryDirectory() as parent, \
             patch.dict(os.environ,{'GEM16_EXPERIMENT_TMPDIR':parent}), \
             patch.object(scratch,'check_ram_storage'), \
             patch.object(sys,'argv',['scratch','--','python','{scratch}/file;literal']):
            def child(command,env,check):
                directory=Path(env['GEM16_EXPERIMENT_DIR'])
                self.assertEqual(env['TMPDIR'],str(directory))
                self.assertEqual(command,['python',str(directory)+'/file;literal'])
                return subprocess.CompletedProcess(command,7)
            with patch.object(scratch.subprocess,'run',side_effect=child):
                self.assertEqual(scratch.main(),7)
            self.assertFalse(any(Path(parent).iterdir()))

if __name__ == '__main__':unittest.main()

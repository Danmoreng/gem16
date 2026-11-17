#!/usr/bin/env python3
"""Run diagnostic work in checked Linux tmpfs; keep durable results elsewhere.

API: with experiment_scratch('ordinary-o', expected_gib=4) as directory: ...
CLI: python tools/experiment_scratch.py --label ordinary-o --expected-gib 4 --
     nsys export --type sqlite --output '{scratch}/trace.sqlite' trace.nsys-rep
The command runs without a shell; {scratch} arguments are replaced literally.
Only the helper-owned directory is removed after the child exits.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

GIB = 1024 ** 3
DEFAULT_ROOT = Path('/tmp/gem16-experiments')
MIN_AVAILABLE_RAM_GIB = 8

def _available_ram() -> int:
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        values[key] = int(value.split()[0]) * 1024
    return values['MemAvailable']

def check_ram_storage(path: Path, expected_gib: float = 4) -> None:
    if not math.isfinite(expected_gib) or not 0 < expected_gib <= 16:
        raise ValueError('expected scratch size must be positive and at most16GiB')
    path = path.absolute()
    ancestor = path
    while not ancestor.exists():
        ancestor = ancestor.parent
    filesystems = subprocess.check_output(
        ['findmnt', '--noheadings', '--output', 'FSTYPE', '--target', str(ancestor)],
        text=True).split()
    if not filesystems or any(fs != 'tmpfs' for fs in filesystems):
        raise ValueError(f'experiment scratch must be tmpfs, got{filesystems} at{ancestor}')
    expected = int(expected_gib * GIB)
    if shutil.disk_usage(ancestor).free < expected + GIB:
        raise ValueError('insufficient tmpfs space: expected scratch plus1GiB reserve required')
    if _available_ram() < expected + MIN_AVAILABLE_RAM_GIB * GIB:
        raise ValueError('insufficient available RAM: expected scratch plus8GiB reserve required')

@contextmanager
def experiment_scratch(label: str, expected_gib: float = 4):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}', label):
        raise ValueError('scratch label must use1..64 letters, digits, hyphens or underscores')
    root = Path(os.environ.get('GEM16_EXPERIMENT_TMPDIR', str(DEFAULT_ROOT)))
    check_ram_storage(root, expected_gib)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix=label+'-', dir=root) as directory:
        yield Path(directory)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', default='diagnostic')
    parser.add_argument('--expected-gib', type=float, default=4)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if not command:
        parser.error('a child command after -- is required')
    try:
        with experiment_scratch(args.label, args.expected_gib) as directory:
            env = dict(os.environ, TMPDIR=str(directory), GEM16_EXPERIMENT_DIR=str(directory))
            command = [part.replace('{scratch}', str(directory)) for part in command]
            print(f'RAM scratch:{directory}; planned{args.expected_gib:g}GiB; automatic cleanup', file=sys.stderr)
            return subprocess.run(command, env=env, check=False).returncode
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f'experiment scratch refused:{error}', file=sys.stderr)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export retained benchmark/qualification argv as a sanitized linked supplement."""
import argparse
import getpass
import json
from pathlib import Path
import socket

from write_decode_ap1_evidence import identity, sanitize

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--raw", type=Path, required=True)
parser.add_argument("--record", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
record = {"schema_version": 1, "measurement_record": identity(args.record.resolve()),
          "commands": {p.name: {"argv": json.loads(p.read_text()), "raw": identity(p)}
                       for p in sorted(args.raw.resolve().glob("*.command.json"))},
          "generator": identity(Path(__file__).resolve())}
record = sanitize(record, [(str(ROOT), "${REPO_ROOT}"), (str(Path.home()), "${HOME}"),
                           (socket.gethostname(), "${HOST}"), (getpass.getuser(), "${USER}")])
with args.output.open("x") as stream:
    json.dump(record, stream, indent=2)
    stream.write("\n")
print(args.output)

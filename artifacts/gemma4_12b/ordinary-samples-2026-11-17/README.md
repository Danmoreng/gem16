# Retained Ordinary evidence

Portable copies of the new2026-11-17 Ordinary primary samples, telemetry,
commands, qualification logs and compressed trace exports. Source paths in
immutable JSON/logs identify the original result directories. Equivalent
files are retained here by result-directory name. Large executable/Nsight
report derivatives remain in the original ignored result roots; compressed
SQLite/torch exports and their hashes are included here. Real norm fixtures
are copied byte-for-byte; screen metadata omits old timing samples.

Replay example from the repository root:

```sh
python tests/cuda/twelve_b_ordinary_actual_norms.py artifacts/gemma4_12b/ordinary-samples-2026-11-17/norm-fixtures /path/to/actual-norm-lab.so --output /tmp/ordinary-norm-check.json
```

The source/test build recipes and full numerical/dispatch limitations are in
`docs/evidence/gemma4_12b/ordinary-optimization-2026-11-17.md`.

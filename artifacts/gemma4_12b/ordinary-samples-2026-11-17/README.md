# Retained Ordinary evidence

The real norm inputs below are used by the CUDA replay. The [original run directory](https://github.com/Danmoreng/gem16/tree/9f7f7a6555ac169a9e49580f11824eef0e55fd85/artifacts/gemma4_12b/ordinary-samples-2026-11-17)
contains the primary samples, command lines, hashes, logs and compressed profiler traces.

Replay example from the repository root:

```sh
python tests/cuda/twelve_b_ordinary_actual_norms.py artifacts/gemma4_12b/ordinary-samples-2026-11-17/norm-fixtures /path/to/actual-norm-lab.so --output /tmp/ordinary-norm-check.json
```

The source/test build recipes and full numerical/dispatch limitations are in
`docs/evidence/gemma4_12b/ordinary-optimization-2026-11-17.md`.

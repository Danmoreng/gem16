# Agent Core development evidence

The [2026-09-04 Linux matrix](2026-09-04-linux-verified/result.json) passed
22 Python and 18 TypeScript cases per public profile plus an unmodified Pi
coding workflow per profile. Each profile directory retains SDK results,
server logs, final metrics, the Pi transcript and independently checked edits.

The run used the CUDA server built from parent `73ab37c` plus the recorded source
changes. Exact source, binary and model-lock hashes are in `result.json`; the
parent revision alone does not identify the tested implementation. The enclosing
commit records those source changes. No checkpoint, sampling default or kernel
was changed. Final fallback and token-loop allocation counters were zero, and
both owned server processes stopped with exit code zero.

Additional checks for this implementation:

- `cmake --build --preset host-debug --parallel 4 && ctest --preset host-debug`: 5/5 passed.
- `cmake --build --preset host-sanitize --parallel 4 && ctest --preset host-sanitize`: 5/5 passed (ASan/UBSan).
- `python3 -m unittest discover -s tests/python`: 329 tests, 2 skipped, passed.
- `npm --prefix tools/openai-sdk run check`: passed.
- Both npm dependency audits: zero reported vulnerabilities at acquisition time.
- The sequential-tool golden text matches Jinja 3.1.6 rendering of the pinned
  12B, Compact Vision Target and internal NVFP4 chat templates.

These checks do not establish full release qualification, GPU numerical parity
for every workload, Windows live behavior, queue saturation, restart recovery,
or a coding-quality benchmark. See the [compatibility guide](../../docs/AGENT_COMPATIBILITY.md)
for the exact tested subset and reproducible commands.

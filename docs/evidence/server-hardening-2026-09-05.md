# Server hardening checkpoint — 2026-09-05

This is development evidence, not a two-platform release approval. The active
remaining work is tracked once in [ROADMAP.md](../ROADMAP.md), C00–C09.

## Implemented boundary

The server now bounds schema evaluation, preserves exact integer comparisons,
uses exception-safe session leases/reservations, bounds admission waits, checks
local HTTP Host/Origin/content type, and limits cumulative image decoding and
preparation. Responses output objects can be replayed directly through the pinned
OpenAI SDK. Pi session affinity reuses KV state; changed histories/tools explicitly
rebuild the slot. Adjacent user messages from Pi compaction are joined in order.
No model precision, kernel, weight lock or sampling semantics were changed.

Public Compact Vision defaults are Linux 220,000 and Windows 170,000 tokens;
the long-context reserve remains 200 MiB. Windows values reflect owner evidence,
not a new Windows run. Server/CLI is the primary entry; Studio remains optional.

## Retained evidence

Raw logs and results are under
[artifacts/server-hardening/2026-09-05](../../artifacts/server-hardening/2026-09-05/).

- `agent-matrix-final`: both public fixed-D2 profiles passed Python SDK, TypeScript
  SDK, unmodified Pi, HTTP/cache/saturation checks and multi-image checks. Binary
  SHA-256: `1aa6fb8c113e37da5395f3caa0e52c8e57c748646064d0d01afb2750c9d98231`.
  This run precedes the final schema work-accounting expansion; it must not be
  represented as validation of a later binary.
- `compaction-normalized-*`: real Pi RPC manual compaction, remembered codeword
  and new session passed for both public profiles. Earlier failures are retained.
- `additional-profiles`: Compact Vision D2 started at 220,000 context and answered
  a short prompt. This is an admission smoke, not a long-context stability test.
- `additional-profiles-runtime-view-nvfp4`: internal NVFP4 D2 short smoke passed
  using verified same-filesystem hardlinks and excluding the packaging-only
  `gem16_components.json`. Original cache files and loader validation were not
  changed. Earlier shared-root/symlink/descriptor failures remain recorded;
  acquisition layout cleanup remains open.
- `final-host-checks`: host, ASan/leak checks, Studio host tests, two Python
  release-gate tests, shell syntax and diff checks passed at that checkpoint.
- `commit-host-checks`: repeats host/sanitizer/package-gate checks after the final
  schema budget expansion and oversized BMP-header regression. Consult each
  recorded exit code rather than assuming an incomplete run passed.
- `package-smoke.json`: 17 packaged file hashes, unpacked binary version and
  fetch help passed on the build machine. This is not a clean-machine test.

Representative commands: `cmake --build build/Linux/host-debug --target
 gem16-unit-tests -j4`, `build/Linux/host-debug/bin/gem16-unit-tests`,
`ASAN_OPTIONS=detect_leaks=1 build/Linux/host-sanitize/bin/gem16-unit-tests`,
`ctest --test-dir build/native-studio -R gem16-native-studio-host --output-on-failure`,
`python3 -m unittest discover -s tests/python -p test_release_gates.py -v`.
The GPU matrix uses `tools/run_agent_core_matrix.py` with the pinned SDK environment
and pinned Pi CLI; exact commands/configuration and raw outcomes live in the logs.

## Remaining qualification

Pool waits now share the admission deadline and observe disconnect/draining. Long-prefill
abort, systematic exception injection, peak-RSS media stress, historical-image CPU
reuse, full Pi fork/resume/automatic-compaction coverage, sampling/reasoning/tool
constraints, Windows GPU and clean-machine tests remain open. The package verifier
is available, but no publish workflow or release is approved by these results.

## Follow-up: bounded resident-session waits

`session-wait-host` records host and ASan/leak regressions, including expired
immediate admission, disconnected pool wait without a condition notification,
deadline expiry, normal wakeup and draining. Both HTTP APIs propagate one budget
through queued creation, named-session waits and Responses-chain waits. No CUDA
kernel or prefill chunking changes are included. `session-wait-matrix` records the
rebuilt candidate and exact source hashes, including `session_wait.h`; its per-profile
outcomes are authoritative for this follow-up rather than the earlier binary.

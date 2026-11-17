# 12B MTP with optimized Ordinary only — 2026-11-17

The owner clarifies that existing optimized MTP verification and Assistant must
remain unchanged. Only Ordinary single-row kernels and their activation inside
MTP are in scope. This supersedes MTP_INTEGRATED_2026_11_17.md entirely. The
unwanted row/Assistant ports are removed from runtime source and archived as a
rejected diagnostic patch; their slower measurements are historical only.

Restore the measured pre-port source exactly, then remove only the four startup
MTP exclusions on Ordinary norm/overlap/O/global. Preserve all existing native
D2/D3/D4 verifier specializations, full conditional graphs and Assistant code.
Rebuild run/chat/server/unit. Measure the final binary at each depth with one
warmup and three retained requests. Preserve exact output IDs, timing, model,
precision, cache and resident weights. Existing unchanged Ordinary numerical/
sanitizer evidence applies; check actual dispatch in a bounded MTP graph trace.
No further MTP kernel work, new commit or push.

Owner update 2026-11-17: commit and push this completed, qualified state are
now explicitly authorized; this supersedes the prior no-commit/push restriction.

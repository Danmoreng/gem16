# Secure12B gains, then compare Ordinary kernels —2026-11-17

The owner explicitly authorizes first committing the completed12B benefits,
their tests and evidence. This supersedes the earlier task entries' no-commit
restriction for the completed local12B work only. Do not push, switch branches,
commit unrelated older work or promote a release. Include the required source
and generator dependencies so the tested runtime is reproducible from Git.
Preserve unrelated dirty files and all historical measurements.

After the commit, investigate12B Ordinary Decode on native gem16, the verified
current central llama.cpp CUDA build, and the installed vLLM reference. This
supersedes the prior MTP-only measurement scope. Use no speculative decoding,
one warmup plus three retained requests per engine, the existing16K/1135
fixture, explicit matching timing/context/batch/sampling controls, resident
weights and peak VRAM telemetry. Disclose native-format/output differences;
no quality-equivalence claim or timing from profiler runs.

Capture a bounded warmed Ordinary GPU kernel window for all three engines,
record actual dispatch/graph/API geometry, compare per-family costs and gaps,
and inspect installed/current source for the observed paths. Identify concrete
12B opportunities and numerical/cache boundaries. This request is measurement
and analysis, not an open-ended production-kernel tuning campaign. Any needed
diagnostic instrumentation stays separate from primary samples.

Raw root: benchmarks/results/2026-11-17/ordinary-kernel-comparison/.
Serialize GPU jobs, use /tmp for bulky temporary profiling exports, retain
losslessly compressed traces/hashes, and respect Berlin22:45/22:55/23:00.

## Owner extension after baseline analysis

The owner now explicitly requests implementing both measured kernel improvements
and overlap of independent Ordinary graph work. This supersedes the measurement-only
scope above for a narrow12B T1 slice. Keep the committed baseline executable and
samples unchanged. Qualify fixed-width norm/quant boundaries bit-exactly, test graph
fork/join dependencies and allocations, then measure kernel-only, overlap-only and
combined with one warmup+three retained requests. Do not promote an unqualified or
slower combination; report it honestly. No26B tuning or additional commit/push is
implicitly authorized by this extension.

## Owner checkpoint and next slice

The owner explicitly requests committing the completed Ordinary kernel/overlap
changes and their evidence before any further runtime modification. This
supersedes the no-additional-commit restriction for that completed slice only.
No push or branch change. The next authorized slice is12B Ordinary FP8 O
projection and global attention, with exact numerical/cache/format boundaries,
operator/sanitizer/dispatch evidence and a final one-warmup/three-retained
comparison against the newly committed51.6977tok/s parent.26B and other12B
paths retain their behavior. Failed candidates remain diagnostic.

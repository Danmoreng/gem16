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

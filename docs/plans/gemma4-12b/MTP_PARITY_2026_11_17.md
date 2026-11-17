# 12B D2/D3/D4 optimization parity —2026-11-17

The owner explicitly requests carrying the existing D3 optimizations to D2
and D4 and measuring all three again. This supersedes the prior T4-only
packing scope and the reuse-only D2 timing restriction for this changed
12B implementation. No26B implementation/checks/cohorts, new kernel/tactic
search, default-depth change, precision/layout/cache/sampling change,
commit, push, branch switch or release. Preserve unrelated dirty work.

Audit existing graph, fixed-row attention, head, post-attention fusion,
fixed Down and Gate/Up parity. Extend the exact same native M16/K32 FP8
packing to the known12B three-/four-/five-row QKV/O shapes. Keep all other
shape branches unchanged. Each depth retains preallocated GPU control,
transactional acceptance/KV, reasoning/stop/tail and streaming behavior.
Carry the existing T4/T5 first-window-crossing correction to12B T3 as well,
using a dedicated12B dispatch so other model paths remain unchanged.

Pin parent source/binaries, compare pinned-weight operator bytes and
independent ordinary-attention boundary behavior, full1135 output IDs,
all proposals and whole KV state. Cover12B text/media/BF16/cached/sampled
chat, scoped sanitizers and actual graph/instruction dispatch. Keep the
inherited Assistant PreProjection full conditional synccheck limitation
explicit. No false-positive or full-model sync/race pass is claimed.

Measure all D2/D3/D4 on one final binary, each1warmup+3retained requests,
with the same16K/1135 fixture, timing boundary and200ms NVML telemetry.
No profiler/interposer in primary timing. GPU jobs remain serialized and
bounded under the Berlin shutdown deadline. Raw root:
benchmarks/results/2026-11-17/12b-mtp-parity/.

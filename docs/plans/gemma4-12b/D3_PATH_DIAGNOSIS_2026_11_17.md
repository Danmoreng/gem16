# 12B D3 path diagnosis — 2026-11-17

The owner requests a focused gem16/vLLM MTP-D3 kernel/order/time comparison,
including whether each MTP path preserves its own Ordinary output and which
remaining changes could speed up gem16 without changing output. This supersedes
the closed-archive no-new-work boundary only for this bounded diagnosis.
No new runtime optimization, precision/layout change, other model, D2/D4 sweep,
commit or push is part of this investigation. Do not reuse speculative extra
MTP kernel ports that were explicitly rejected in the preceding work.

Reuse fresh native primary evidence; refresh vLLM D3 at one warmup plus three
retained requests only if needed to establish current protocol/output identity.
Capture one bounded diagnostic window per engine, attribute Target/Assistant/
head/acceptance/control, matrix shapes, graph boundaries, kernel/device interval
unions and host gaps. Do not add overlapping budgets or equate profiler rates
with primary throughput. Compare model/toolchain identities and own-Ordinary
output IDs, and distinguish numerical variants from algorithmic speculation.
All large traces/SQL exports use checked RAM scratch, saving compact evidence
before automatic cleanup. Read current exact installed vLLM code, plus primary
official documentation; rank only measured gaps with output-preservation limits.

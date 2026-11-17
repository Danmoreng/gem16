> Historical task contract, closed or superseded. Current disposition is in
> [the archive index](../../evidence/benchmark-archive-2026-11-17.md).
> This file does not authorize a new experiment.

# AP1: current decode measurement basis

Owner request: 2026-11-04. Parent: `66b74d79977c4fb44d0b2e7eef451f7196037c5a`.
This bounded measurement task supersedes the tuning freeze only for AP1 diagnostics.
AP2 and production runtime changes are not authorized by this entry. No commit,
push or release is part of AP1.

## Required work

1. Verify Pro's source findings against the parent: R3 QKV grid/staging,
   one-row global attention normalization, last Assistant feedback, and the
   attention postnorm reduction. Distinguish source facts from measured costs.
2. Fix future helper telemetry: separate configuration, observed and derived
   speculation work; represent unavailable data as null, including fallback and
   depth distributions. Retain the N-1 throughput boundary. Test mixed depths,
   missing counters, ordinary fallbacks, malformed counters and prefill-only.
3. Replace the replay driver's historical engine label with verified source,
   binary and loaded-library identities; diagnostic timing remains diagnostic.
4. Add a maintained offline analyzer with full-instance attention attribution,
   interval unions and explicit unknowns. Check overlap accounting, both merge
   dimensions, partial intervals and source-defined MoE fork/join fixtures.
   Historical artifacts/analyzers remain immutable.
5. Freeze native code/model/binary/toolchain and verified current central llama
   identities. Run ordinary then D2, each native/llama/llama/native, three warmups
   and ten retained per cohort. Record all samples, outputs, counters, allocation
   evidence and 200ms telemetry. Fail on native output/proposal drift, offload,
   mixed identities or unverifiable upstream. Do not attach an observer.
6. Separately capture complete native decode traces, group acceptance/depth/
   position/proposal observations and actual operator inputs for AP2–AP5.
   Quantify observer perturbation and never promote diagnostic rates. Correlate
   group costs with acceptance where supported; missing fields stay unknown.
7. Publish a sanitized new evidence record, README pointer and AP2 go/no-go with
   limitations. No runtime speedup is claimed by AP1.

## Gates and stopping rules

1135 output tokens / 1134 decode intervals; identical native greedy fixture,
precision, cache/context, sampling, resident arenas and timing boundaries.
No recurring production allocation, fallback, repack or altered dispatch.
Account busy union + idle = window; exclusive + overlap = busy union. Costs
from different captures are not added. Do not interpret sync API time as extra
CPU budget. Full-model groups and output-work differences are disclosed.

Source/host changes require targeted host fixtures; test-only CUDA observers
require output/proposal parity and memory checking. Production `src/` remains
unchanged. Guard long operations against the 22:45 wrap/22:55 save/23:00 shutdown.
Retain failed attempts; never overwrite old samples or retry until favorable.

Local raw directory: `benchmarks/results/2026-11-04/66b74d7/ap1-measurement/`.

Status: complete. All nine measurement gates passed; 48 targeted host fixtures
pass. The input observer's full-model memcheck timeout remains explicitly
unqualified; its exact copy/marker kernels and slot arithmetic passed a focused
actual-input memory check. Production source remains unchanged.

The [evidence report](../../evidence/gemma4_26b/decode-ap1-measurement-2026-11-04.md)
and [sanitized record](../../../artifacts/trellis35/decode-ap1-measurement-2026-11-04.json)
retain both complete baseline campaigns, current source/model/library identities,
cost unions, same-capture group attribution, actual operator inputs/counters and
all limitations. Current medians are native139.497/198.193 versus
llama143.945/202.026 tok/s Ordinary/D2 at official master9e2a04a.
AP2 receives a Go for one bounded flat-QKV-grid trial; it was not implemented
within AP1. No commit or push was performed.

# T3 GateUp register pipeline — rejected, 2026-10-24

Goal Step2 investigates a fixed resident-payload register pipeline after accepted
M1 GateUp `e0c7fe5`. The accepted native engine and its measured rates remain
unchanged. Only diagnostic lab code is retained; no production speed claim.

## Matching evidence and bounded variants

Fresh actual-production GateUp profiles at late layer29 and early layer5 report
47 registers, long-scoreboard ratios 0.8167 / 0.9589 and short-scoreboard ratios
0.1809 / 0.1993 per active issue. This motivates a bounded payload scheduling
experiment, not a causal wall-time budget or a broad prefetch reopening.

Both depths preserve the accepted paired codec windows, exact K order, MMA,
scales, H128, expert-row mapping and vector stores. All 24 real GateUp/Down
projections per variant and 2,097,152 scoped codec lane cases remain bit exact.
Four layers and three context positions give twelve GateUp measurements:

| Diagnostic variant | Worst operator throughput change | Median | Best |
|---|---:|---:|---:|
| Depth2 | -4.74% | +8.11% | +23.45% |
| Depth4 | -3.28% | +9.28% | +23.07% |
| Depth2, register-bounded | -8.27% | -3.47% | -1.85% |

The predeclared conservative gate requires every sampled GateUp case to improve
by more than 2% before full qualification. Neither ordinary pipeline passes.
Their positive medians are not promoted as whole-model gains. No full-model
candidate benchmark was run, so these results do not prove an aggregate loss
or rule out a different explicitly qualified performance policy.

## Countercase and one final refinement

The phase1/layer15 union24 case compares actual production parent to the
separately compiled depth4 diagnostic. Register usage rises 47->63 and the
theoretical register block limit falls 5->4. Long-scoreboard ratio falls
1.1063->0.5881, yet NCU duration rises 84.576->85.824 us. Better individual stall
metrics do not establish a net gain; occupancy alone is not a causal proof.

This finding justifies one final depth2 diagnostic with
`__launch_bounds__(256,5)`. It compiles to 48 registers but loses in every
sampled case. The bounded shader reports stack16/shared2560/local0 bytes in
the standalone lab. These are diagnostic resources, not a production build's
resources. No further register/depth sweep or runtime promotion follows.

## Method, integrity and evidence

Each isolated graph sample contains sixteen calls, with three warmups and ten
alternating samples. NCU uses three explicit real-input warmups, cache-control
none and clock-control none. Counter values retain units; ratios are not bounded
wall-time shares. Diagnostic replay timings are excluded from primary throughput,
and inconsistent cache percentages above100% are excluded from interpretation.

The current built executable and all accepted production source identities match
the Step1 qualification. Formats, precision, sampling, cache, context, graph/ring
safety and protected paths are unchanged. The rejected candidates receive no
full-model sanitizer/peak-VRAM qualification because they are not promoted. No
new llama comparison, Windows/media/release qualification or push is claimed.

Raw root: `benchmarks/results/2026-10-24/e0c7fe5/t3-gate-pipeline/`.
Record: `artifacts/trellis35/t3-gate-pipeline-rejected-2026-10-24.json`.
Original and bounded pre-build snapshots, compiled modules, exact commands,
counter units, real inputs/outputs, all samples and SHA-256 manifest are retained.

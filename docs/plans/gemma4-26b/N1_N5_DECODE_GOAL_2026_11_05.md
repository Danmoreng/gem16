> Historical task contract, closed or superseded. Current disposition is in
> [the archive index](../../evidence/benchmark-archive-2026-11-17.md).
> This file does not authorize a new experiment.

# Five decode experiments after AP5

The owner authorized the five experiments on 2026-11-05 in the order below.
Each starts from the last qualified parent. The initial source checkout is
`66b74d79977c4fb44d0b2e7eef451f7196037c5a`, with the qualified executable SHA-256
`02b65a2eafdcc1c7cd26c6013007226207a76226d8d99021159546e5c90651a8`.
Existing local AP1–AP5 changes and evidence are separate work.

| Order | Scope | Weighted screen threshold | Primary |
|---|---|---:|---|
| N1 | Two group-wide RoPE tables for the Trellis35 fixed-D2 Target | 30 µs per D2 group | D2 |
| N3 | Two warps per CTA for the Trellis35 Ordinary router projection | 20 µs per Ordinary interval | Ordinary |
| N2 | Offline byte permutation of Target FP8 O weights for paired 64-bit loads | 60 µs per D2 group | D2 |
| N4 | Restore only the rejected KV suffix after fixed-D2 acceptance | 30 µs per D2 group | D2 |
| N5 | One 64-KiB read-only GPU codebook table for T3 GateUp | 60 µs per D2 group | D2 |

Use one candidate per package. First prove actual dispatch, bit equality,
bounded memory and measured savings with representative real cases. Keep raw
measurements, source identities and scoped limitations. Candidate promotion
requires unprofiled end-to-end ABBA with the same AP1 workload and measurements.
The primary lower 95% bound must be positive in both orders. Secondary requires
both nonnegative points and lower bounds above −0.20%, or a stricter active
rule. Run protected profile regressions and operator/lifecycle checks applicable
to the affected paths. Restore any rejected or incomplete runtime candidate.
Keep positive and negative evidence separate from the qualified runtime.
Production, packaging and Git publication are separate decisions.

N1 uses the actual early, middle and late 30-layer D2 epilogue inputs. Measure
30 parent RoPE/epilogue pairs with their original PDL dependency against two
prepared tables plus 30 normally ordered epilogues. Compare all consumed
normalized Q and staged KV bytes, plus table bytes for both profiles. The
isolated graph omits the surrounding projections, attention and MoE. A passing
screen needs an integrated dispatch and end-to-end result. Candidate tables
would use the existing persistent prefill prefix, with lifetime, sentinel,
Prefill and TTFT checks before adoption.

N1 also uses the same ABBA cohorts for prefill throughput and TTFT. The
predeclared practical non-regression tolerance for those two noisier metrics
is −1% at the lower 95% bound in each order. This does not change the decode
throughput gates above.

## Progress

N1's isolated screen passed at +223.98 µs per group across three real
positions. The candidate production trace confirms 15,060 → 1,004 table
launches and 15,060 epilogues. All other kernel resource/dispatch shapes,
1,135 outputs, 1,004 proposals and 90 actual layer boundary captures match.
Scoped offline memcheck and synccheck passed. Unprofiled D2 passed both
orders (+1.225% AB, +1.645% BA) including prefill and TTFT. Ordinary failed
the nonnegative-point requirement at −0.0731% AB and −0.0518% BA. The two
95% intervals include zero and their lower bounds exceed −0.20%. The
predeclared rule therefore rejects promotion despite the positive D2
measurement. Acceptance-only lifecycle and protected profile work stopped
at the failed secondary gate. Raw evidence:
`benchmarks/results/2026-11-05/66b74d7/n1-rope`.

The exact qualified parent is restored before N3. N3 reproduced all 90
captured projection outputs bitwise, with identical 36 registers and no local
spill storage. Its two-warp candidate costs 3.72 µs more per isolated
30-layer Ordinary interval. All three windows lose and the 20 µs screen
fails. No integrated fork/join timing is claimed.

N2 emits an actual 64-bit weight load and reproduces 12 actual O projections
bitwise after correcting the stream ordering of the test uploads. The weighted
gain is 0.95 µs per D2 group against a 60 µs gate. No format migration follows.

N4 restores the same entire cache and sentinel bytes for real acceptance 0/1/2
and nine extra boundary cases. The screen saves 29.83 µs per D2 group after
acceptance weighting, below the 30 µs gate. This is an isolated result.

N5 exactly reproduces all 65,536 codebook states and 12 real GateUp vectors.
The average case median rises from 62.52 to 197.01 µs per projection. It fails
the 60 µs screen.

All five experiments are complete. The qualified executable and tracked
runtime match the original parent. The
[report](../../evidence/gemma4_26b/n1-n5-decode-2026-11-05.md) links the
sanitized machine-readable evidence and raw hashes. No measurement job remains.

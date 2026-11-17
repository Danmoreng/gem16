# AP5 attention postnorm exact warp tail — 2026-11-04

**Rejected at the predeclared operator gate; production was never changed.**
Geometric-mean warm operator change is-0.0303%,
95%bootstrap[-0.0699%,+0.0177%]. Both required screen conditions fail.
[Complete record](../../../artifacts/trellis35/attention-postnorm-warp-2026-11-04.json) retains
raw samples, input/source/binary identities, commands and instruction evidence.

| Actual case | Rows | Operator throughput change |
|---|---:|---:|
| 1: phase0-target-postnorm-r1 | 1 | -0.0774% |
| 2: phase1-target-postnorm-r1 | 1 | -0.2755% |
| 3: phase2-target-postnorm-r1 | 1 | -0.0388% |
| 4: phase0-target-postnorm-r3 | 3 | +0.1188% |
| 5: phase1-target-postnorm-r1 | 1 | -0.1176% |
| 6: phase1-target-postnorm-r3 | 3 | +0.1404% |
| 7: phase2-target-postnorm-r1 | 1 | -0.0346% |
| 8: phase2-target-postnorm-r3 | 3 | +0.0432% |

The derivative preserves the exact128/64/32CTA tree and16/8/4/2/1warp-zero
tail with explicit synchronization after every stage and a final CTA barrier.
SASS confirms CTA barrier sites9to5. Both kernels use28registers,1024bytes
static shared storage and no local allocation/spills; NCU captures are separate
diagnostics. Fewer barrier sites alone did not yield a winning actual-input screen.

All8real R1/R3cases reproduce consumed outputs bit-for-bit, including optional
normalized-output pointer presence. Scalar and output/input/residual-alias cases,
finite extremes and null weights agree. Memcheck/synccheck report0errors. This
does not claim every possible pointer overlap or NaN/Inf coverage.

Timing uses128launch warm CUDAgraphs,3warmups+10retained alternating samples.
Every case median and the aggregate lower95bound must be positive. All samples
remain; no favorable rerun or R3-only candidate is selected. Production/E2E/
protected-profile gates are deliberately unrun after screen failure. No new
model throughput, peakVRAM, cross-engine superiority, Windows or release claim.

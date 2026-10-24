# T3 GateUp matched-build audit — 2026-10-24

**No production promotion.** The earlier
[non-RDC diagnostic record](t3-gate-pipeline-rejected-2026-10-24.md) remains
unchanged. This final audit closes its explicitly disclosed build-mode limit:
the three existing current-source variants are rebuilt with the production
`compute_120a/sm_120a`, `-O3`, C++20 and `-rdc=true` settings. Runtime parent is
accepted f6de38e; main 8046990 only adds the rejected head diagnostics.
[Full record](../../../artifacts/trellis35/t3-gate-rdc-audit-2026-10-24.json).
Raw: `benchmarks/results/2026-10-24/8046990/t3-gate-rdc-audit/`.

| Existing variant | Minimum | Median | Maximum throughput change |
|---|---:|---:|---:|
| Depth 2 | -0.764% | -0.155% | +0.659% |
| Depth 4 | -4.170% | +8.592% | +21.285% |
| Depth 2, register bounded | -6.809% | -5.213% | -4.358% |

Each variant retains 24 actual GateUp/Down projection cases at three controlled
positions and four layers, all output bits exact, plus 2,097,152 exact codec lane
cases. The table uses the 12 GateUp cases, each with three warmups and ten retained
interleaved CUDA-graph samples (16 repeated projections/sample). Immutable
weights are borrowed, mutable inputs/IDs/expected outputs are snapshotted; no
observer is part of a primary benchmark. Depth2/4 and the single old register
bound are the complete bounded set; no extra staging/depth trial is added.

All three fail the original every-GateUp-case>2% gate, so no full-model candidate
gain/loss or candidate memory qualification is claimed. A positive median for
depth 4 does not justify retaining its losses. This matched-build result confirms
the rejection for these sources/settings; it does not overwrite earlier data or
claim compilation flags alone explain all differences. The initial source
snapshot predates appending the bounded kernel; original generic kernel bodies
remain unchanged. The current whole-module source and exact flags are pinned.

Fresh actual function-handle counters at the same layer 15/union 24 case show
parent 47 registers/block limit 5, depth 2 48/5 and depth 4 63/4. Long-scoreboard per
active issue is 1.0556/0.8416/0.6535; eligible warps per scheduler cycle remain
2.809/2.915/2.719. These are diagnostic, cache/clock-uncontrolled counter captures;
their single-kernel wall times are not the standalone screen or a same-clock
causal gain. SASS/resource dumps and raw counter values/units are retained.

Only the unrelated old attention link derivative is omitted, replaced by a
visible `cudaErrorNotSupported` link stub. It is never called in T3 mode; any
accidental use fails the diagnostic. Attention is not re-benchmarked or silently
substituted. Source and both stub/module hashes, all actual inputs/outputs,
commands and timings are in the immutable record. No runtime source changes,
new profile/precision/format, sampling/cache change, repack or full-model claim.

Reproduction: recorded `screen.py`, then `counters.py`, using the frozen current
parent driver and exact source snapshots. The result is diagnostic only and
does not require repeating unrelated product/12B tests when no production code
changes. The accepted M1 GateUp, attention and FP8 QKV/O gains remain intact.

"""Render the bounded production-build-match audit of the three existing T3 variants."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
d=json.loads((ROOT/'artifacts/trellis35/t3-gate-rdc-audit-2026-10-24.json').read_text());assert d['status']=='rejected_operator_regressions_no_production_change'
names={'2':'Depth 2','4':'Depth 4','0':'Depth 2, register bounded'}
rows=[f"| {names[k]} | {100*v['min_gain']:+.3f}% | {100*v['median_gain']:+.3f}% | {100*v['max_gain']:+.3f}% |" for k,v in d['screens'].items()]
table='| Existing variant | Minimum | Median | Maximum throughput change |\n|---|---:|---:|---:|\n'+'\n'.join(rows)+'\n'
report=f'''# T3 GateUp matched-build audit — 2026-10-24

**No production promotion.** The earlier
[non-RDC diagnostic record](t3-gate-pipeline-rejected-2026-10-24.md) remains
unchanged. This final audit closes its explicitly disclosed build-mode limit:
the three existing current-source variants are rebuilt with the production
`compute_120a/sm_120a`, `-O3`, C++20 and `-rdc=true` settings. Runtime parent is
accepted f6de38e; main8046990 only adds the rejected head diagnostics.
[Full record](../../../artifacts/trellis35/t3-gate-rdc-audit-2026-10-24.json).
Raw: `benchmarks/results/2026-10-24/8046990/t3-gate-rdc-audit/`.

{table}
Each variant retains 24 actual GateUp/Down projection cases at three controlled
positions and four layers, all output bits exact, plus2,097,152 exact codec lane
cases. The table uses the12 GateUp cases, each with three warmups and ten retained
interleaved CUDA-graph samples (16 repeated projections/sample). Immutable
weights are borrowed, mutable inputs/IDs/expected outputs are snapshotted; no
observer is part of a primary benchmark. Depth2/4 and the single old register
bound are the complete bounded set; no extra staging/depth trial is added.

All three fail the original every-GateUp-case>2% gate, so no full-model candidate
gain/loss or candidate memory qualification is claimed. A positive median for
depth4 does not justify retaining its losses. This matched-build result confirms
the rejection for these sources/settings; it does not overwrite earlier data or
claim compilation flags alone explain all differences. The initial source
snapshot predates appending the bounded kernel; original generic kernel bodies
remain unchanged. The current whole-module source and exact flags are pinned.

Fresh actual function-handle counters at the same layer15/union24 case show
parent47registers/block limit5, depth2 48/5 and depth4 63/4. Long-scoreboard per
active issue is1.0556/0.8416/0.6535; eligible warps per scheduler cycle remain
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
'''.replace('main8046990','main 8046990').replace('plus2,097,152','plus 2,097,152').replace('the12','the 12').replace('depth4','depth 4').replace('depth2','depth 2').replace('parent47registers/block limit5','parent 47 registers/block limit 5').replace('layer15/union24','layer 15/union 24').replace('is1.0556','is 1.0556')
(ROOT/'docs/evidence/gemma4_26b/t3-gate-rdc-audit-2026-10-24.md').write_text(report)
def update(path,start,end,anchor,block):
 text=path.read_text()
 if start in text:
  a=text.index(start);b=text.index(end,a)+len(end);text=text[:a]+text[b:].lstrip('\n')
 assert anchor in text;text=text.replace(anchor,block+anchor,1);path.write_text(text)
start='<!-- T3-GATE-RDC-AUDIT-2026-10-24 -->';end='<!-- /T3-GATE-RDC-AUDIT-2026-10-24 -->'
update(ROOT/'docs/PERFORMANCE.md',start,end,'<!-- NVFP4-HEADS-2026-10-24 -->',f'''{start}
## 2026-10-24 T3 GateUp production-build audit

The three existing variants also fail their all-case gate with production-matched
RDC=true; no production change. All24 actual outputs per variant and2,097,152
codec lanes are exact. These are operator diagnostics, not full-model gain/loss.

{table}
[Matched-build evidence](evidence/gemma4_26b/t3-gate-rdc-audit-2026-10-24.md).
{end}

''')
start='<!-- T3-GATE-RDC-RESULT-2026-10-24 -->';end='<!-- /T3-GATE-RDC-RESULT-2026-10-24 -->'
update(ROOT/'README.md',start,end,'Earlier public-profile measurements',f'''{start}
A final [T3 GateUp audit with production-matched RDC builds](docs/evidence/gemma4_26b/t3-gate-rdc-audit-2026-10-24.md)
also rejects all three existing pipeline variants: exact outputs, but no consistent
operator gain. The accepted engine implementation and performance baseline remain unchanged.
{end}

''')

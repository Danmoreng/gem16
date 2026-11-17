# AP3 R1 global attention normalization — 2026-11-04

**Rejected by the predeclared gate; exact parent runtime restored.** The [complete record](../../../artifacts/trellis35/r1-global-warp-normalization-2026-11-04.json)
retains samples, commands, source/model/binary identities, operator/dispatch and
memory/protected evidence, limitations and private-raw hashes.

| Ordinary order | Candidate throughput change | 95% bootstrap interval |
|---|---:|---:|
| AB | +0.0783% | [+0.0330%, +0.1023%] |
| BA | +0.0768% | [-0.1123%, +0.2338%] |
| pooled | +0.0834% | [+0.0378%, +0.0999%] |

Only non-batch R1 global26B normalization changes. Each warp normalizes its own
query head using the accepted exact four32-token-chunk/two-level reduction tree.
QK/PV, FP8 conversion,16-key staging, splits/merge and batch/fixed-R3 remain.
The11actual Target/Assistant cases reproduce partial/LSE/final-merge bytes and
improve1.87% in warm alternating operator replay. Registers remain56 with no
spills; shared storage falls32bytes. Detailed NCU source/scoreboard/resource
captures are retained. Diagnostic rates are not model throughput claims.

372counterfactual boundary cases cover all128tails, short/inactive/direct splits,
independent K/V scales,16K and capacity-end positions. Memcheck/synccheck pass.
Nine production host/CUDA tests,12Bproduct, parent/candidate12B/internalNVFP4
vectors and a real CompactVision scene pass. Full16K-context memory check uses
8outputs; complete1135output/1004proposal parity is a separate trace.

Primary Ordinary requires positive lower95bounds in both orders; secondary D2
requires each point>=0 and lower95>-0.20%. Primary failure stops before secondary.
Each mode uses parent/candidate/candidate/parent,3warmups+10retained,16,384prompt,
1,135outputs/1,134intervals,context17,519,greedy0,FP8KV,fixed masks,cold cache and
resident GPU weights, no observer/profiler. All samples stay retained; clocks
are firmware managed and sampled with200msNVML plus1Hzprocess monitoring.
No new cross-engine, Windows, complete media or context-capacity claim is made.

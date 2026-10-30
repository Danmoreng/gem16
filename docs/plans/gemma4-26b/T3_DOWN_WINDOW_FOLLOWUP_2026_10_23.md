# T3 Down paired-window follow-up — complete, 2026-10-23

Owner asks to continue toward the pinned llama reference after accepted GateUp
`629bbac`. Main starts clean; the frozen parent binary matches that qualification.
Raw root: `benchmarks/results/2026-10-23/629bbac/t3-down-window/`.

Smallest candidate: use the already bit-qualified fixed-window codec at the
same 32 MMA positions in T3 Down, preserving its accepted eight-warp geometry,
K accumulation, scale/reduction/H128 and output stores. First screen all real
projections with diagnostic-only code and compare warm actual-input counters.
Do not infer gain from GateUp alone. Promote only with exact operator/reference,
memory, protected-profile and full ordinary/D2 evidence. Retain positive and
negative results; no push or release. README must reflect any accepted result.

Berlin deadline: begin wrapping at22:45, save/stop all owned jobs by22:55.
Do not launch qualification that cannot finish safely before22:55. If needed,
leave the accepted parent production kernel and document the diagnostic state.

22:20: direct window transfer is bit exact over all24 real projections and the
2,097,152 primitive cases, but Down throughput spans -1.10% to+15.07%, median
+2.89%. The predeclared all-case screen fails; nothing is promoted. Actual
production layer0 Down counters show long-scoreboard0.9556 versus short0.1742
per active issue,56registers and a four-block register occupancy limit. This
justifies testing one fixed four-iteration register-pipeline refinement using
the already existing M1 pattern. New pre-build snapshots must distinguish it
from the rejected direct-transfer binary. Counter/profile ratios are diagnostic,
not bounded time shares. Current production/build remain accepted parent.

22:26: the refined fixed-four-iteration candidate is exact and faster on all12
Down cases (3.50–16.63%, median5.54%). A tentative native/vector Down-only
selector is built; GateUp/M1 are unchanged. Sources frozen in candidate-sources.
Seven CUDA/host checks pass33.96s, targeted memcheck0errors, native8/64 and
protected12B/NVFP4 ordinary/D2 and explicit-fixture12B product pass. Actual
production shader63registers, stack/local0, shared1536, nativeE4M3/QMMA.
Full-D2 memcheck is running; no primary speed claim or production commit yet.
Prepared six-cohort3+10 AB/BA D2 plus ordinary comparison and full trace/proposal/
forced checks. Finish or restore parent and leave a precise note by22:55.


22:35: full 16K/context17519/eight-output D2 memcheck passes with0errors in238.20s.
First frozen3+10 pair: parent193.417825 -> candidate194.854073tok/s (+0.7426%).
All1135IDs match and D2 stays502groups/632accepted. Reverse pair and ordinary
non-regression remain running, full trace/proposals/forced gates pending. No
accepted production commit or README rate update until all required gates pass.


22:41: reverse pair also improves:193.499346 ->194.715722tok/s (+0.6286%).
All40retained D2 runs exact1135IDs, acceptance/arenas unchanged. Ordinary pair
is running, last cohort expected to finish before22:47. Then full1135trace,
proposal/forced checks and documentation/commit only; no next experiment tonight.
All required work must finish/save/stop by22:55. Positive D2 remains conditional
on the pending gates; raw/continuation paths above permit a precise restart.


## Final state 22:48 CEST

Accepted native/vector T3 Down window plus fixed-four-iteration pipeline.
D2 AB/BA: 193.417825->194.854073 and
193.499346->194.715722tok/s (+0.7426% / +0.6286%).
Ordinary adjacent137.671900->137.658531
(-0.0097%). All required numerical/memory/protected/model/dispatch/proposal/
forced-acceptance gates pass. All60retained1135vectors exact,502groups/632accepted,
no loop allocations/fallback/repack, unchanged arenas/VRAM. Full trace and earlier
countercases retained. README and PERFORMANCE.md updated, historical rows kept.

Evidence: docs/evidence/gemma4_26b/t3-down-window-2026-10-23.md and
artifacts/trellis35/t3-down-window-2026-10-23.json. Raw root above. Accepted executable
built and frozen. Commit narrowly under standing authorization, no push/release.
No next experiment tonight; stop owned jobs and preserve this qualified parent.

Next bounded investigation: M1 GateUp paired-word windows, since ordinary GateUp
still uses16warps. Screen former countercases, preserve its existing payload
pipeline and numerical boundaries, then run separate ordinary AB/BA plus D2 tail
non-regression. No automatic transfer or new speed claim. After reaching a
candidate lead, refresh the direct same-machine llama comparison before claiming
superiority; the pinned196.55/138.31 rows are not new/latest measurements.

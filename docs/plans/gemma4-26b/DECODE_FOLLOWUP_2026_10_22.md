# Decode follow-up — complete, 2026-10-22

Owner authorization: implement five experiments carefully; keep and commit correct
production changes only with reproducible end-to-end benefit. No push requested.
The bounded tuning exception is recorded in `docs/ACTIVE_DECISIONS.md`.

## Final state before the daily shutdown

All five experiments are complete. Production source is the accepted `0dd1240`
runtime, rebuilt after rejecting the last candidate. No benchmark remains running.

1. **Kept, `08d4741`.** Allowed-top1 shortcut: ordinary +4.45% in both 3+10
   AB/BA comparisons, exact 1,135 outputs, unchanged VRAM and D2. Finite/router
   error checks and blocked-winner fallback remain effective.
2. **Diagnostic infrastructure retained.** Native/llama 2x2 forced-history replay
   exactly reproduces both own 1,135-token vectors; cross-history successor
   differences are 102 native-on-llama and 88 llama-on-native. Forced D2 accepts
   0/1/2, rollback/commit and continuation pass; early/middle/late windows run.
   Actual M1/T3 inputs: layers 0/5/15/29, three positions, 24 projection cases
   each; bit-exact results, own routes/K3/K4/expert unions. Four real-input NCU
   profiles and raw counters retained. Input parsing is bounded and unit-tested.
3. **Kept, `0dd1240`.** M1 Down eight warps with paired N8 fragments:
   ordinary 134.976 -> 137.695 and 134.761 -> 137.642 tok/s (+2.01–2.14%).
   Adjacent D2 candidate 172.407 tok/s, no claimed D2 improvement. Same tokens
   and VRAM. GateUp stays unchanged after its isolated regression. Actual
   dispatch, native E4M3/QMMA instructions, sanitizer and protected tests pass.
4. **Rejected.** Ordinary KVH2 32-key staging is bit exact but takes roughly
   119 versus 104 us at all three positions. Cold NCU: 197.504 -> 226.016 us;
   shared memory 20,544 -> 36,928 bytes; achieved occupancy 52.17 -> 30.69%.
   Ordinary remains 16 keys; actual Fixed-T3 already uses 32. No production patch.
5. **Rejected.** Combined GateUp transform/FP8 fusion is bit exact and faster
   in isolated operator graphs, but ordinary AB/BA drops 138.079 -> 136.030
   and 137.711 -> 135.947 tok/s (-1.48/-1.28%). D2 drops 172.344 -> 170.451
   (-1.10%, adjacent pair). All sixty primary runs have the same 1,135 IDs and
   no loop allocation/fallback. Candidate sources/binaries are archived; four
   owned production/test files were restored exactly to `0dd1240` at 22:44.

Final restored build: `cmake --build build/Linux/blackwell-release --parallel 4`
passed (93 build steps, no stderr). Final checks:
`ctest --test-dir build/Linux/blackwell-release --output-on-failure -R
'^gem16-(unit|cuda-trellis35)$'`: 2/2 pass in 1.26 s. Earlier broad CUDA, native
memcheck, 12B product, numerical, selection fault and protected 12B/NVFP4 checks
are in the individual production records. `git diff --check` passes.

## Evidence and continuation

Reports: `docs/evidence/gemma4_26b/allowed-top1-2026-10-22.md`,
`m1-eight-down-2026-10-22.md`, `decode-followup-diagnostics-2026-10-22.md`,
`gate-input-fusion-2026-10-22.md`; matching JSON in `artifacts/trellis35/`.
Raw roots: `benchmarks/results/2026-10-22/c0a807af/{decode-cost-audit,
allowed-top1,operator-replay,m1-eight-down,gate-input-fusion}/`.
The final supporting commit preserves tools and evidence, not rejected production
kernels. No push. Ordinary is close to the pinned llama reference (138.312 tok/s),
not a demonstrated win; native D2 remains about 172 versus llama 196.551 tok/s.
Formats differ; cross-engine quality equivalence is not established.

No required step remains for this Goal. Further work needs a new bounded slice:
profile the fused full-run regression before another fusion, and use the retained
controlled D2 histories to isolate verification cost. Do not infer whole-run
bottlenecks from four sampled operators or cold NCU replay alone.

NCU progress interleaves with long JSON and is excluded from parity gates;
unprofiled full replay and byte checks pass. Setup failures (full-SWA C API OOM,
verifier/Assistant setup order, copied executable permission, CUDA 13 modern launch
hook) are retained and excluded. No older llama binary, offload, precision change,
extra model weight copy, relaxed tolerance or unqualified release claim.

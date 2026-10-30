# Controlled decode replay and real-input diagnostics — 2026-10-22

Steps 2 and 4 of the owner-authorized decode follow-up. Production changes are
limited to the independently qualified allowed-top1 (`08d4741`) and M1 Down
(`0dd1240`) improvements. No push, model format change or quality-equivalence claim.

## Controlled history

The test-only native driver and independently built llama C-API driver use the
same 16,384 prompt IDs and 1,135 generated IDs, with 1,134 forced successor rows.
Both engines exactly reproduce their own recorded vector. Native on llama history
has 102 differing successor predictions; llama on native history has 88. Router
choices and activations remain each model's own: cross-format divergence is real,
and the replay does not force common experts. Both exercise forced D2 acceptance
0/1/2, rollback/commit and exact continuation, plus early/middle/late windows.
Pinned llama revision is `23b0202a189c44a54625aadcb37a946dd1d6278d` (b11552).
No older llama binary was run. Native frozen replay drivers predate the M1 change.

The bounded CSV reader rejects malformed, overflowing, oversized, symlink and
nonregular inputs. It is a diagnostic-only input surface. Native driver is built
by CMake; llama and operator observers have explicit commands in the evidence.
Linux-only observer code is never loaded by normal inference.

## Actual operator inputs

GPU-kernel snapshots reserve a fixed diagnostic arena; they do not copy model
weights. Layers 0/5/15/29 at three decode positions supply 24 projection cases
per M1/T3 mode. All reference/candidate projections are bit exact. Actual K3/K4
counts and expert unions are retained. Four real-input Nsight Compute profiles
cover M1/T3 GateUp/Down; register counts are 55/42, with zero local spills.
Warm graph replay uses 16 operations/sample, three warmups and ten alternating
samples. Cold-cache profiler replay is a distinct diagnostic timing boundary.

## KVH2 staging experiment: rejected

Ordinary attention currently stages 16 keys; Fixed-T3 already stages 32. Parent
counters (19.09% DRAM active, barrier/short-scoreboard waits and sampled PCs)
justify an isolated experiment, not a causal bottleneck claim. An offline-built
32-key derivative is bit exact for partial outputs and LSE at positions
16,384/16,951/17,517, but takes approximately 119 us versus 104 us (13% less
throughput). Cold profiler replay also worsens, 197.504 to 226.016 us. Static
shared memory rises 20,544 to 36,928 bytes and achieved occupancy falls 52.17%
to 30.69%, with 56 registers and zero spills in both. Production remains unchanged.

## Validation and limitations

Final commands: `cmake --build build/Linux/blackwell-release --parallel 4`, then
`ctest --test-dir build/Linux/blackwell-release --output-on-failure -R
'^gem16-(unit|cuda-trellis35)$'`. Final bounded-reader host checks are recorded
separately from earlier GPU-driver replay qualification. Broader CUDA, sanitizer,
12B product and 12B/NVFP4 regressions accompany the production changes.

Raw commands, complete replay rows, operator bytes/timings, source snapshots,
resource counters and SHA-256 identities are in
`artifacts/trellis35/decode-followup-diagnostics-2026-10-22.json`.
Raw root: `benchmarks/results/2026-10-22/c0a807af/operator-replay/`.

Profiler stdout interleaves progress with long JSON; those streams are not parity
gates. Separate unprofiled replays and byte checks establish correctness. Initial
full-SWA C-API OOM, verifier setup-order error, executable permission and CUDA 13
launch-hook failures are preserved and excluded. Diagnostic timing does not replace
primary tok/s measurements. No Windows or additional release qualification.

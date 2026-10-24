# Native NVFP4 output heads — 2026-10-24

**Rejected after full qualification and end-to-end comparison.** Frozen parent `f6de38e58b090a8f4f542b8acfa22a54098e614a`. The
[record](../../../artifacts/trellis35/nvfp4-output-heads-2026-10-24.json) contains
exact source/binary/model/toolchain identities, commands, all raw samples and
hashes, negative attempts, process/clock/power/VRAM telemetry and native shaders.
Raw: `benchmarks/results/2026-10-24/f6de38e/output-heads/`.

## Whole-model result

| Mode | Parent tok/s | Candidate tok/s | Change | Bootstrap 95% |
|---|---:|---:|---:|---|
| Fixed-D2 AB | 195.967 | 196.290 | +0.1647% | [+0.0218%, +0.3158%] |
| Fixed-D2 BA | 195.881 | 196.220 | +0.1728% | [-0.0018%, +0.3470%] |
| Ordinary adjacent | 139.027 | 139.169 | +0.1024% | [+0.0604%, +0.2185%] |

The predeclared reproducible end-to-end gate is not met. The production Target trial is removed and the accepted parent runtime/source is restored; the Assistant variant never entered production. Correct operator results alone do not justify keeping this change.

Each cohort uses three discarded warmups and ten retained process runs with
16,384 prompt IDs, 1,135 output IDs / 1,134 decode intervals, context 17,519,
batch one, greedy/mask and FP8 KV; fixed-D2 retains draft depth two. Both orders
are measured under the same firmware max-power/Dynamic Boost policy on the RTX
5080 Laptop, driver 615.71.09 / CUDA 13.4.92. Bootstrap: 10,000 independent resamples,
seed 20261031; intervals are descriptive for these fixed cohorts. No outlier is
removed. All 60 retained output vectors are exact; all 40 D2 runs retain 502 groups,
632 accepted / 1,004 proposed. All precision/cache/weight/KV/workspace/repack/graph
accounting, zero fallbacks and zero token-loop allocations are unchanged.

Sampled device-total peaks: 12,455MiB ordinary /
12,725MiB D2. Whole-process high-util clocks: parent-a: 2565 MHz, candidate-b: 2572 MHz, candidate-b2: 2572 MHz, parent-a2: 2565 MHz.
Clocks are not fixed.200 ms samples include loading/prefill/warmup and are not
instantaneous allocation peaks or isolated decode clocks.1 Hz pmon retains the
observed compute and graphics processes. Different clocks preclude a pure
same-clock instruction-cycle claim. Ordinary also has a small positive point
estimate even though its head implementation is unchanged; the seven-MHz
whole-process clock difference appears in both ordinary and D2 pairs. This
further limits attribution of the tiny point estimates to head packing. No
new llama lead is established here.

## Actual heads and bounded variants

These heads use native NVFP4/E2M1 with UE4M3 block scales, not the BF16 reference
head or the FP8 QKV kernels. Target vocab 262,144/K 2,816 emits three BF16-rounded
float logit rows with divisor4,736. Assistant vocab 262,144/K 1,024 emits all 4,096
64-token candidate value/token pairs, with its existing 258883/258882 masks,
finite check, BF16 boundary and lower-token tie rule. Target softcap/masks and
all vocabulary/sampling/cache/acceptance behavior stay unchanged.

Actual parent Target M1 has 54 registers and 93.63% peak DRAM utilization; it is left
unchanged. Target T3 has 106 registers, a register block limit 4 and 91.55% DRAM.
The bounded candidate packs the three real A rows into one M16 block-scaled MMA
using the existing matrix SF_A map: lane0 of each quad supplies its lower-M8 row,
lane1 the unused upper-M8 row. Unused rows/scales are zero. Per-row K64 order,
weight layout/traffic, scale selectors and BF16/divisor epilogue remain exact.
Actual candidate 64 registers and block limit 8, zero stack/local/shared; native
`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` SASS is retained. Repeated-operator throughput improves only
0.184–0.214% across the three positions (aggregate bootstrap lower bound>0).
Sampled diagnostic DRAM becomes 94.11%; counters use uncontrolled clocks/caches.
This is consistent with bandwidth dominance for these shapes, not a universal
hardware limit or proof of exclusive causality. Three-to-one instruction work
and fewer registers do not reduce the dominant weight bytes.

Assistant adjacent-N8 pairing shares A/SF_A across two ordered MMA chains per
warp, preserving every original64-token candidate partition. All six real cases
are exact but lose throughput by approximately 0.01–0.85% (median 0.154% loss).
Registers rise 56→95 and register block limit 4→2; DRAM stays near 95%. This variant
is rejected at the operator gate. No full-model gain/loss is attributed to it.
There is no further pipeline/staging/depth experiment in this scope.

## Correctness, memory and retained failures

The first observer uses forced verifier windows; those do not execute Assistant
proposals and leave its diagnostic slots unexecuted. Its replay failure is not
a runtime head regression. The initial older-driver M1 counter and both failed
Assistant captures are retained and excluded from the final current-parent basis.
The new test-only `d2-head-windows` invokes actual Assistant groups at controlled
Target positions 16,384/16,951/17,512. Three baseline M1 cases and all nine actual
D2 cases replay byte-exactly; actual production Target inputs/scales/logits and
all Assistant candidate bytes match the frozen parent. The observer has a fixed
96 MiB arena for mutable snapshots; immutable resident weights are borrowed. It
is absent from all primary runs.

Initial `-arch=sm_120a` shorthand emits an extra generic compute_120 PTX target;
ptxas rejects native block-scale MMA there. Failed build/log/source remain.
V2 explicitly matches production compute_120a/sm_120a and RDC=true; no fallback
precision or kernel is used.42 guarded synthetic cases vary native FP4 bytes,
per-row/per-K scales, zeros, partial Target warps, divisor overflow, masks and
ties. All output bytes/canaries match original kernels; memcheck/synccheck report
zero errors. The production trial passes seven CUDA/host tests, the recognized
fixed-T3 focused memcheck, native 8/64 ordinary/D2, protected 12B / internal NVFP4
64-token vectors identical to the accepted parent, explicit 12B product fixtures
and full 16K D2 memcheck 238.78 s/zero errors. No tolerance is weakened.

Full 1,135-output traces retain 502 Target head calls and 1,004 Assistant candidate
calls with exact IDs, all draft proposals and forced 0/1/2 acceptance/rollback
continuations. Actual shader resources and instructions are captured before any
rejection/restore. Source scope is only the exact Target R3 vocab/K selector;
ordinary, Assistant, legacy and other shapes retain their original implementation.
The record's final source/binary checks prove the selected retention or rollback.
For the rejected trial, all runtime source bytes are restored exactly. A strict
whole-file hash comparison finds only four changed nvcc temporary-PID characters
in non-loadable `.strtab`, outside all PT_LOAD segments. Read-only ELF inspection
proves every other byte and all loaded CPU code/data and CUDA PTX/SASS/fatbins
identical to the accepted parent. Both full binary hashes and segment hashes are
retained; no binary is patched. The initial verifier assumes only one sm120
filename across the module; its corrected version isolates precisely the one
changed string, without excluding any other bytes.

Offline reproduction starts with `current_parent_counters.py` (new actual head
windows), `screen_v2.py`, `boundaries.py`, `candidate_counters.py`, `promote.py`,
`qualify.py`, `finish.py` (actual shader/full comparisons/traces/proposals), the
recorded `decision_v2.py` restore/`verify_restored_elf_v2.py` and `make_record_v2.py`. Failed/old scripts are kept for
provenance, not to be substituted for the corrected recipe. Generated derivatives
come from immutable source snapshots via `tests/cuda/build_nvfp4_head_candidates.py`;
production startup never compiles or repacks. The record lists every command.
No new full model-payload hash, Windows/image/release qualification is claimed.

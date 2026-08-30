# Codex master instruction – Gem16 Trellis35 WP10–WP16

You are continuing work in the Gem16 repository after Trellis35 WP9.

## Source boundary

Expected starting branch and commit:

```text
branch: codex/gemma4-26b-trellis35-w4a8
commit: fbc0121ad6c699d85d9d9e7792083e11744b2eba
```

Create the next working branch only if the owner has not already done so:

```text
codex/gemma4-26b-trellis35-perf2
```

Do not push, merge, rebase, reset, stash, change unrelated files, or rewrite accepted evidence unless explicitly authorized.

Before every work packet, read:

```text
AGENTS.md
docs/ACTIVE_DECISIONS.md
docs/plans/gemma4-26b/GEM16_TRELLIS35_W4A8_DISCOVERY_PLAN_v2.md
docs/plans/gemma4-26b/CHATGPT_PRO_TRELLIS35_PERFORMANCE_REVIEW_2026-08-30.md
GEM16_TRELLIS35_POST_WP9_PERFORMANCE_ARCHITECTURE_REVIEW.md
```

## Permanent owner requirements

Two model paths must coexist:

```text
NVFP4 profile:
  existing fastest qualified path
  must remain supported and regression protected

Trellis35 profile:
  one persistent mixed-K3/K4 routed-expert payload
  lower VRAM / longer-context path
  remains unqualified and separately reported
```

Trellis35 must never replace, rename, mutate, or silently become the existing NVFP4 artifact.

Do not introduce:

- a persistent decoded E4M3 expert copy;
- a persistent NVFP4 expert fallback beside Trellis;
- CPU expert streaming;
- runtime JIT;
- token-loop allocation;
- runtime model quantization/repacking;
- silent precision, kernel, context or format fallback;
- changed prompts, sampling, output length, KV format or timing boundaries.

Do not run WP8B long quality evaluation in this wave. Run only proportional numerical and bounded WP8A-style checks after arithmetic changes.

Current comparison facts:

```text
Trellis35 Target arena:     12,204,692,480 bytes
saving vs NVFP4:             2,491,975,680 bytes / 2.3208 GiB
payload rate:                3.5 bpw

Trellis35:
  16K prefill:              ~951.1 tok/s
  Ordinary:                 ~119.4 tok/s
  T3 verifier diagnostic:   ~168.5 tok/s
  short total D2 diagnostic: ~85.2 tok/s

NVFP4 references:
  sampled prefill:          ~6965.6 tok/s
  Ordinary:                 ~148.4 tok/s
  Fixed-D2:                 ~203.8 tok/s
```

The 85.2 and 203.8 timing boundaries are not directly comparable. Preserve and report both total-request and post-first boundaries.

---

# WP10 – Explicit format coexistence and zero-codegen source split

## Goal

Harden NVFP4/Trellis coexistence and split the large Trellis/engine/test files without changing generated kernels or arithmetic.

## Required changes

### 1. Explicit routed-expert format

Add a narrow enum, for example:

```cpp
enum class Gemma4Moe26BRoutedExpertFormat : std::uint8_t {
  kNvfp4,
  kTrellis35,
};
```

Suggested location:

```text
src/cuda/engine/gemma4_26b_routed_expert_format.h
```

Replace format-owning `bool trellis35` state with the enum. Local compile-time booleans may remain only where they describe kernel templates, not artifact identity.

Resolve the format from validated artifact metadata and compare it with the expected model profile/catalog entry.

Fail closed on an ambiguous or mixed directory.

Add tests that prove:

```text
NVFP4 directory -> NVFP4 artifact only
Trellis directory -> Trellis artifact only
mixed families -> rejected
profile/metadata disagreement -> rejected
```

Do not load both routed expert banks.

### 2. Split Trellis source while preserving one CUDA TU

Keep `src/cuda/trellis35/reference.cu` as a thin aggregator initially.

Create responsibility shards such as:

```text
src/cuda/trellis35/detail/codec.cuh
src/cuda/trellis35/detail/mma_w4a8.cuh
src/cuda/trellis35/detail/transform_common.cuh
src/cuda/trellis35/detail/m1_kernels.inc.cuh
src/cuda/trellis35/detail/t3_kernels.inc.cuh
src/cuda/trellis35/detail/prefill_kernels.inc.cuh
src/cuda/trellis35/detail/launchers.inc.cuh
```

Move code without editing arithmetic, qualifiers, unroll pragmas, declaration order or launch geometry.

### 3. Split engine orchestration

Keep one translation unit initially. Use implementation shards, for example:

```text
src/cuda/engine/detail/gemma4_26b_state.inc
src/cuda/engine/detail/gemma4_26b_create.inc
src/cuda/engine/detail/gemma4_26b_decode.inc
src/cuda/engine/detail/gemma4_26b_prefill.inc
src/cuda/engine/detail/gemma4_26b_mtp.inc
src/cuda/engine/detail/gemma4_26b_metrics.inc
```

Do not introduce a virtual runtime framework.

### 4. Split Trellis tests

Build one existing test executable from multiple source files:

```text
tests/cuda/trellis35_codec_test.cu
tests/cuda/trellis35_transform_test.cu
tests/cuda/trellis35_m1_test.cu
tests/cuda/trellis35_t3_test.cu
tests/cuda/trellis35_prefill_test.cu
tests/cuda/trellis35_runtime_test.cu
tests/cuda/trellis35_test_support.h
```

## Acceptance

- exact current Trellis output hashes;
- exact NVFP4 product regression;
- unchanged device arena;
- normalized SASS/cubin identity for current Trellis hot kernels;
- unchanged registers/shared memory;
- current benchmarks within 1% or recorded run noise;
- ambiguous artifact test fails before CUDA allocation;
- no new allocation or fallback.

Write:

```text
artifacts/trellis35/wp10-structure-coexistence.json
```

Do not proceed to arithmetic changes until WP10 passes.

---

# WP11 – Fresh instrumentation and zero-numerics cleanup

## Goal

Profile the actual post-WP9 implementation and remove work that has no numerical effect.

## 1. Fresh Nsight Compute

Profile current:

```text
MmaW4A8ProjectionGroupedPrefillTileKernel
```

at T=512 using the real mixed-K3/K4 artifact/routing fixture.

Record at minimum:

- duration;
- registers/thread;
- local memory loads/stores;
- stack;
- spills;
- achieved/theoretical occupancy;
- executed integer, FP, control and tensor instructions;
- tensor active cycles;
- issue-slot utilization;
- DRAM/L2/L1 traffic and hit rates;
- warp stall reasons;
- waves/SM.

Do not reuse WP6 NCU as post-WP9 evidence.

## 2. Schedule telemetry and safe launch bound

Record per layer/family:

```text
assignment_count
active_experts
actual schedule_count
launched schedule_blocks
rows-per-expert histogram
tail-size histogram
```

For the current rows=4 kernel, replace:

```cpp
schedule_blocks = assignment_count;
```

with a checked safe upper bound:

```text
q = min(active_experts, assignment_count)
max_tiles = floor((assignment_count + 3*q) / 4)
```

The device `schedule_count` guard remains.

Accept only with bit-identical output and a non-regressing full-prefill measurement.

## 3. Trellis-specific M1/T3 input boundary

Current native boundary performs dead routed-expert NVFP4 quantization and then separately computes Trellis BF16 expert input.

Add a Trellis-specific boundary that performs exactly once:

```text
RMS reduction
Shared-MLP NVFP4 activation
Router normalized/transformed data
physical BF16 Trellis expert input
```

Do not produce routed-expert FP4 bytes/scales.

Preserve the exact reduction order and BF16 rounding. If output is not bit-identical to the current Trellis path, reject this candidate.

Reuse preallocated Trellis workspace if a separate temporary destination is needed. No new device allocation.

## 4. Bounded transient E4M3 probe

Add a diagnostic operator, not a default engine path:

```text
Trellis -> E4M3 N128 slab -> decoder-free full-M W4A8
```

Measure Gate+Up and Down at M=4/16/32/64.

Report bytes written/read and bounded workspace.

## 5. D2 NVTX ranges

Add ranges around:

```text
initial Target selection
Assistant proposal
Target attention
Target shared MoE
Target routed MoE
Target head
selection/sampling
KV backup
tentative append
restore
commit
```

Report complete request, post-first, and pure T3 verifier separately.

## Acceptance

- no numerical changes for accepted cleanup;
- no new allocation;
- no NVFP4 SASS change;
- no performance regression outside measured noise;
- complete artifacts:

```text
artifacts/trellis35/wp11-post-wp9-ncu.json
artifacts/trellis35/wp11-schedule-telemetry.json
artifacts/trellis35/wp11-input-boundary.json
artifacts/trellis35/wp11-transient-e4m3-probe.json
artifacts/trellis35/wp11-d2-breakdown.json
```

---

# WP12 – True M16/M32 grouped Trellis W4A8 prefill

## Goal

Replace the current four independent broadcast-row MMAs with real M16 tiles.

This is the primary performance packet.

## Current defect to remove

Current `AccumulateFp8` supplies:

```text
{a0, a0, a1, a1}
```

to `mma.sync.m16n8k32` and stores separate accumulators for each of four assignments.

Do not repeat the failed rows=8 design. It increased independent M1 accumulator state rather than populating the M16 operand.

## Required kernel

Add a new kernel, provisionally:

```text
MmaW4A8ProjectionGroupedPrefillM32Kernel
```

First geometry:

```text
4 warps / CTA
N32 / CTA
2 x M16 assignment tiles
K32
```

Reuse:

```text
src/cuda/fp8/sm120.cu::Sm120MatrixProjectionKernel
```

for full E4M3 A-fragment/shared-memory design, and:

```text
src/cuda/nvfp4/sm120.cu::Sm120GroupedExpertMatrixKernel
```

for 32-assignment expert scheduling and permutation semantics.

Do not modify these qualified kernels merely to share code unless extraction is proven codegen-neutral.

## Implementation requirements

1. Consume the existing 32-row `BuildExpertTileScheduleKernel` result from `src/cuda/moe/prefill.cu`.
2. Remove the Trellis-specific rows=4 schedule from the new path.
3. Stage up to 32 assignment activations:
   - two M16 E4M3 tiles;
   - K32 fragments;
   - zero fill invalid tails;
   - double buffer with `cp.async`.
4. Use full A registers `a0,a1,a2,a3`.
5. Each warp owns N8 and decodes one B fragment per K32.
6. Feed the same B fragment to both M16 MMAs.
7. Keep K3/K4 branch uniform per CTA and outside K loop.
8. Apply the correct per-assignment activation scale when storing each D-fragment row.
9. Preserve permutation and original assignment identity.
10. Initially retain the current output inverse transform, so only projection math changes.
11. No complete decoded matrix in global memory.

## Numerical matrix

Test:

```text
families: Gate+Up, Down
rates: K3, K4, mixed
T: 1, 2, 3, 4, 8, 15, 16, 17, 31, 32, 33, 128, 512, 1024
routing:
  uniform
  real fixture
  one hot expert
  long tails
```

Require current independent oracle thresholds. Prefer bit-identical transformed projection output. If not bit-identical, stop and explain the changed operation order before proposing a tolerance.

## Engineering rejection gate

This is not a product threshold.

Default-enable the candidate only if:

- no local-memory spills;
- transformed projection correctness passes;
- T=512 projection stage is at least 1.5x faster than WP11 current;
- full 512 prefill improves at least 10%;
- small/tail shapes have no catastrophic regression.

Otherwise retain the old kernel as the Trellis experimental fallback, record NCU, and use the transient probe to decide the next design.

## Evidence

```text
artifacts/trellis35/wp12-m32-operator.json
artifacts/trellis35/wp12-m32-ncu.txt/json
artifacts/trellis35/wp12-prefill-ab.json
artifacts/trellis35/wp12-sass.txt
```

---

# WP13 – Warp H128 and fused transform/scale/quantize

## Goal

Remove repeated direct 128-FMA transforms.

## Implement

Create an independently tested warp primitive:

```text
H128Warp
```

with:

```text
32 lanes
4 values/lane
local H4
five shuffle-xor stages
normalization
```

Input prefill kernel:

```text
one CTA per assignment
8 warps

load + SUH
FWHT once
store values in shared
amax reduction
scale
E4M3 quantize from shared
```

Support current float-container input and physical BF16 input through explicit policy/template types.

Approximate shared requirements:

```text
Gate+Up input: 2816 floats ~= 11 KiB
Down input:     768 floats ~= 3 KiB
```

Implement a fast inverse H128 kernel, one warp per assignment/H128 block or another measured equivalent.

## Precision requirements

Preserve explicit physical BF16 rounding at:

```text
Gate+Up inverse -> BF16 -> GELU
GELU product -> BF16 -> Down transform
Down inverse -> BF16 -> route reduction
```

A fused kernel may keep data in registers/shared only if it explicitly rounds at the same boundary.

Because FWHT changes association, define numerical thresholds before measuring. Run bounded WP8A-style checks only.

## Acceptance

- exhaustive transform oracle;
- no spill/local memory;
- combined transform GPU time at least 2x lower than WP12 baseline;
- end-to-end prefill improves;
- no route/top-k changes outside predeclared numerical contract;
- no new allocation.

Write:

```text
artifacts/trellis35/wp13-fwht.json
```

---

# WP14 – Profile-selected decoder/MMA and launch consolidation

Do not choose the mechanism before reading WP12/WP13 NCU.

## Candidate A – producer/consumer warp specialization

Use only if decoder/ALU instructions remain dominant.

Prototype:

```text
producer warp(s):
  load Trellis words
  decode K3/K4 to packed E4M3 B tile
  write double-buffered shared B

consumer warps:
  load M16 A
  issue W4A8 MMA for two M16 tiles
```

Benchmark producer count 1 versus 2. Reject if producer serialization or shared-memory occupancy loses end-to-end.

## Candidate B – transient E4M3 slabs

Use only if WP11 probe beats optimized inline decode.

No persistent copy. Price workspace against long-context capacity.

## Candidate C – output launch consolidation

Current host loop creates 990 projection and 990 inverse-transform launches over 30 layers.

Investigate only after WP12/WP13:

- flatten output blocks into a grid dimension;
- enlarge only bounded reusable scratch;
- or use an N128 CTA/shared epilogue if registers permit;
- optionally fuse Gate+Up inverse + explicit BF16 round + GELU;
- optionally fuse Down inverse + explicit BF16 round, then preserve slot-order reduction.

Each candidate must be isolated and independently revertible.

## Acceptance

- no persistent weight duplication;
- exact workspace bytes reported;
- context-capacity impact reported;
- physical BF16 boundaries preserved;
- no spills/deadlock;
- full-prefill win, not microbenchmark-only win.

Write one artifact per attempted candidate, including rejected results.

---

# WP15 – Ordinary and Fixed-D2 recovery

## Goal

Apply the proven transform/boundary improvements to small-M paths and obtain timing-comparable D2 evidence.

## Required sequence

1. Run a canonical repeated Trellis Ordinary/D2 performance panel with the same prompt, output, sampling and timing definitions as the NVFP4 reference.
2. Report:
   - total request;
   - initial selection;
   - post-first steady state;
   - pure T3 verifier.
3. Adopt the exact Trellis-specific input boundary from WP11.
4. Port H128Warp to M1/T3.
5. Profile the T3 projection after transform changes.
6. If MMA issue/register overhead is still material, implement one real M16 A tile per unique expert with up to three valid rows and zero-filled remaining rows.
7. Preserve unique-expert weight reuse and route-overlap behavior.
8. Perform additional chain fusion only when NVTX shows a dominant surrounding stage.

## Correctness gates

- Ordinary output token identity;
- Target verifier identity;
- identical accepted/rejected drafts on the fixed trajectory;
- same slot-ordered reduction;
- same KV backup/tentative/restore/commit behavior;
- zero non-finite;
- zero fallback;
- zero token-loop allocation;
- CUDA graph replay;
- memcheck/initcheck/racecheck as applicable.

## Evidence

```text
artifacts/trellis35/wp15-ordinary-panel.json
artifacts/trellis35/wp15-d2-panel.json
artifacts/trellis35/wp15-d2-nvtx-breakdown.json
artifacts/trellis35/wp15-t3-ncu.json
```

Do not call a single short run a product median.

---

# WP16 – Architecture decision only

After WP12–WP15, write:

```text
docs/plans/gemma4-26b/TRELLIS35_W4A8_OR_W4A4_DECISION.md
```

Choose among:

```text
A. continue W4A8 optimization
B. add a prefill-only transient E4M3 backend
C. research Trellis -> native W4A4 for large M
D. stop Trellis performance work
```

## W4A4 admission conditions

Do not begin W4A4 merely because NVFP4 is faster.

Require evidence that after M32, FWHT and decoder scheduling:

- W4A8 tensor compute is a material limiting stage;
- decoder-free or transient W4A8 exposes a compute ceiling;
- Tensor Core utilization is high enough that FP4 throughput can matter;
- expected benefit justifies new E2M1/E4M3 scale numerics and sidecars.

A W4A4 design must define:

- E2M1 reconstruction;
- E4M3 K/16 scales;
- payload/sidecar byte delta;
- second quantization error;
- runtime decoder cost;
- one persistent representation only.

WP16 must not implement W4A4 without a new owner instruction.

---

# Ideas forbidden in this wave

- W4A16;
- exact-704 Down;
- Vision/MMProject;
- WP8B long quality suite;
- persistent decoded expert copy;
- broad refactor of qualified NVFP4/attention kernels;
- rows=8 with eight independent M1 accumulators;
- TMA-only rewrite before M32 evidence;
- changed BF16 round points;
- changed activation scale semantics;
- changed prompt/KV/sampling/timing boundary;
- virtual calls or generic dynamic kernel framework in the token hot path.

---

# Reporting standard for every packet

Report:

```text
files changed
exact commands
tests and results
source/model/toolchain/GPU identities
numerical boundary
kernel launch identity
registers/shared/stack/local/spills
Nsight measurements
full-path adjacent benchmark
device allocation delta
limitations
accepted/rejected decision
```

Never overwrite prior evidence. Add a new artifact and preserve rejected experiments.

Stop and report rather than guessing if an optimization requires changing existing NVFP4 semantics or keeping two persistent routed-expert formats resident.

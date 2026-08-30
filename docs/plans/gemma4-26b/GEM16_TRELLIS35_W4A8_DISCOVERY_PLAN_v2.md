# Gem16 – Trellis35/W4A8 Discovery Plan v2

**Supersedes:** the prior EXL3 discovery plan from this review
**Repository basis:** `main@6938ebea1d87bda7114cf1b9149f7f8a62bc4559`
**Source checkpoint:** `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized@f1e06dc520982d9b9edd76859fdb7ab209449949`
**Purpose:** build a Gem16-native, EXL3-derived routed-expert format that saves roughly 2.3 GiB at 3.5 bits/weight while preserving the existing qualified NVFP4 profile as a completely separate rollback path.

> Terminology: this plan uses **3.5 bpw = bits per weight**, not bits per machine word.

---

## 1. New owner decision

This is no longer a compatibility experiment with upstream ExLlamaV3 artifacts.

The new target is a **Gem16-specific trellis format and compiler**, derived from EXL3/QTIP-style ideas and allowed to reuse/port MIT-licensed ExLlamaV3 quantizer code, but deliberately optimized around Gemma 4 26B A4B and Gem16's current kernels, memory ownership, fused expert layout and fixed-D2 execution.

Working name:

```text
GEM16-Trellis35 v1
```

or, internally:

```text
trellis35
```

Do **not** call the on-disk format "EXL3" unless it remains byte-compatible with upstream EXL3. The likely outcome is algorithmically EXL3-derived but Gem16-specific.

The qualified NVFP4 Target remains untouched and is the rollback/control arm.

---

## 2. Hard scope of the first discovery

### In scope

Only routed expert weights change:

```text
model.language_model.layers.*.experts.gate_up_proj
model.language_model.layers.*.experts.down_proj
```

The first discovery must eventually cover all three execution modes:

```text
1. Ordinary decode, M=1
2. Fixed-D2 verifier, T=3 / three Target rows
3. Prefill, routed expert W4A8
```

The storage target from the first usable artifact is:

```text
3.5 trellis payload bpw
```

with real effective bpw, sidecars and alignment reported separately.

### Explicitly out of scope

- Vision/MMProject integration
- 26B multimodal execution
- Changing FP8 attention
- Changing FP8 KV
- Changing Shared MLP NVFP4
- Changing tied Head/Embedding NVFP4
- Changing Router/Norm precision
- Assistant re-quantization
- Product/profile promotion
- Native W4A4 trellis compute in the first implementation
- Exact-704 Down tail in the first implementation

Vision is postponed until the text-only Trellis35 path is functionally and numerically stable.

---

## 3. Why the format is Gem16-specific from day one

The current source already stores routed Gate+Up fused:

```text
[128 experts, 1408 output, 2816 input]
logical order = expert, gate_then_up, input
```

The new compiler must quantize each expert's Gate+Up as **one 2816 x 1408 logical linear**.

This gives:

```text
2816 = 22 * 128
1408 = 11 * 128
```

so the normal 128-wide EXL3-style transforms require **zero padding**.

Do not split Gate and Up into two 704-wide matrices during compilation.

### Coupled Gate+Up output transform

The 704/704 semantic boundary is not aligned to a 128-wide Hadamard block. Therefore the format contract is:

1. quantize the full fused 1408-output matrix;
2. allow the 128-wide output transform to cross the Gate/Up boundary;
3. at runtime reconstruct/invert the full 1408 transformed output;
4. only then split `[0:704]` and `[704:1408]`;
5. apply Gemma GELU(gate) * up.

This is intentional and must be tested explicitly. It is not assumed numerically equivalent to separately quantizing Gate and Up.

### Down v1

Down remains:

```text
logical: 704 input x 2816 output
physical trellis input: 768
```

for v1.

The first implementation therefore retains 704→768 padding only on Down. A later exact-704 transform/tail is a separate optimization.

---

## 4. Expected memory target

Current aligned Target arena:

```text
14,696,668,160 bytes
13.687 GiB
```

Current routed experts:

```text
Gate/Up: 8,564,259,840 bytes
Down:    4,282,137,600 bytes
Total:  12,846,397,440 bytes
```

For fused Gate+Up, padded Down, 3.5 trellis payload bpw, FP16 transform sidecars and a small descriptor/alignment reserve:

```text
estimated new Target arena ≈ 11.370 GiB
estimated saving          ≈  2.317 GiB
```

This is the primary discovery target.

The compiler and final artifact must publish:

```text
logical_expert_coefficients
encoded_expert_coefficients
trellis_payload_bytes
suh_bytes
svh_bytes
rate_descriptor_bytes
codebook_bytes
alignment_bytes
total_expert_bytes
total_target_arena_bytes
effective_expert_bpw
effective_whole_model_bpw
saving_vs_locked_nvfp4_bytes
```

No theoretical bpw claim is accepted without these exact byte counts.

---

## 5. Bit allocation: deterministic K3/K4 at exactly 3.5 payload bpw

The first format supports:

```text
K3
K4
```

per expert matrix.

There are two matrix families per layer:

```text
128 fused Gate+Up experts
128 Down experts
```

To make the first artifact simple, deterministic and runtime-friendly:

```text
per layer:
  Gate+Up: exactly 64 experts K3, 64 experts K4
  Down:    exactly 64 experts K3, 64 experts K4
```

This produces exactly 3.5 payload bpw independently for each family and layer.

### Which experts receive K4?

For every candidate matrix, run both K3 and K4 quantization during offline compilation and measure the existing EXL3-style Hessian/proxy objective.

Define:

```text
benefit = proxy_error_K3 - proxy_error_K4
```

For each layer and projection family, promote the 64 experts with highest positive benefit to K4; the rest are K3.

Persist the exact per-layer maps in the compilation manifest.

This is preferred over "first 64 experts" or random allocation because it provides a deterministic quality-aware 3.5-bpw budget without requiring the still-experimental sparse-model recipe optimizer.

Later versions may allow a global weighted budget or K2/K5, but not v1.

---

## 6. Compiler architecture

The new artifact is built **directly from the immutable Google QAT-BF16 source**.

Never do:

```text
BF16 -> existing NVFP4 -> trellis
```

The two artifacts must be sibling derivations:

```text
Google QAT BF16
  ├── current Gem16 FP8/NVFP4 compiler
  └── new Gem16 Trellis35 compiler
```

### 6.1 Reuse of ExLlamaV3

Pin the exact upstream source used for provenance:

```text
turboderp-org/exllamav3
0c49587a7c235e6303a6bbedc8b665272ad3a2ea
MIT
```

Gem16 may copy/port the minimum required quantizer pieces, for example:

- Hessian capture/finalization concepts
- block LDL decomposition
- LDLQ recursion
- randomized signs/scales
- 128-wide Hadamard regularization
- 16x16 tile permutation
- trellis encoder
- trellis packing
- proxy error calculation
- MCG/MUL1 codebook primitives as reference candidates

If substantial MIT code is copied, preserve the upstream copyright and MIT license in the vendored subtree and record the pinned source revision.

Suggested location:

```text
third_party/exllamav3_quant/
  LICENSE
  PROVENANCE.md
  ...
```

Do not add ExLlamaV3 as a runtime dependency.

### 6.2 Gem16-owned producer

Suggested files:

```text
tools/gem16_compile/trellis35.py
tools/gem16_compile/trellis35_quant.py
tools/gem16_compile/trellis35_layout.py
tools/generate_gemma4_26b_trellis35_plan.py
tools/gem16_compile/specs/trellis35-experts-v1.json
```

The producer must emit the final Gem16 runtime layout directly. There must be no model-sized startup repack.

Offline use of Python/PyTorch/CUDA is acceptable for the discovery compiler if it is pinned and reproducible. Runtime remains C++/CUDA with no PyTorch, JIT or dynamic quantization.

---

## 7. Trellis35 on-disk/runtime format v1

Per layer, store separate compact pools for K3/K4 and the two projection families.

Conceptual layout:

```text
layer L:
  gate_up_k3_payload_pool
  gate_up_k4_payload_pool
  down_k3_payload_pool
  down_k4_payload_pool

  gate_up_descriptor[128]
  down_descriptor[128]

  gate_up_suh[128][2816]
  gate_up_svh[128][1408]

  down_suh[128][768]
  down_svh[128][2816]
```

Descriptor:

```text
struct Trellis35ExpertDesc {
    uint32_t pool_offset;
    uint16_t rate_bits;      // 3 or 4
    uint16_t codebook_id;
};
```

Use 64-bit checked arithmetic for file/device offsets even if the descriptor stores a narrower validated local offset.

The manifest must record:

```text
format_version
producer_revision
source_lock_sha256
source_tensor
source_slice
logical_shape
physical_shape
logical_axis_order
trellis_tile = 16x16
hadamard_block = 128
rate_map
codebook_id
pool_offsets
padding_contract
gate_up_boundary = 704
gate_up_inverse_before_split = true
```

---

## 8. W4A8 is the first compute target

Do **not** spend the first runtime milestone implementing W4A16.

The first production-shaped runtime experiment is:

```text
Trellis35 storage (~3.5 bpw)
      ↓
inline trellis decode
      ↓
E4M3 weight operand
+
BF16/FP16 activation
      ↓
MXFP8/E4M3 activation quantization
      ↓
SM120 FP8 Tensor Core MMA
```

The important contract is:

```text
storage ~3.5 bpw
compute W4A8
```

not "weights are literally stored as four-bit E4M3."

### 8.1 Two decoder stages inside the same W4A8 program

To de-risk implementation while still skipping W4A16 compute:

**Reference W4A8 decoder**

```text
trellis -> reconstructed FP16 register values -> E4M3 pack -> FP8 MMA
```

This is allowed only as an operator bring-up/oracle path.

**Optimized W4A8 decoder**

```text
trellis/codebook -> packed E4M3 words directly
```

or the nearest equivalent lookup/bit-manipulation path that avoids scalar FP16 reconstruction.

The optimized decoder is required before full-model performance optimization.

If standard MUL1/MCG reconstruction maps poorly to E4M3, introduce a Gem16 E4M3-native trellis codebook as a format v2 experiment rather than forcing an inaccurate conversion silently.

---

## 9. Runtime implementation order

### WP0 – Branch and frozen baselines

Recommended branch:

```text
exp/gemma4-26b-trellis35-w4a8
```

Base:

```text
6938ebea1d87bda7114cf1b9149f7f8a62bc4559
```

Before changing CUDA, record the locked comparison rows:

```text
sampled Prefill median ≈ 6965.631 tok/s
Fixed-D2 decode median ≈ 203.842 tok/s
Ordinary decode median ≈ 148.439 tok/s
Target arena = 14,696,668,160 bytes
Target-only context = 98,304
Fixed-D2 context = 86,016
```

Performance is observational during discovery, not a pass/fail gate.

### WP1 – Format/byte estimator

Implement the exact fused-GateUp 3.5-bpw plan without generating weights yet.

Required tests prove:

```text
Gate+Up physical dimension = 2816 x 1408, no padding
Down physical dimension    = 768 x 2816
K3/K4 maps yield exactly 3.5 payload bpw
all pools/sidecars/alignment fit expected bytes
estimated whole Target ≈ 11.370 GiB
saving ≈ 2.317 GiB
```

### WP2 – Gem16 compiler and CPU/PyTorch oracle

Port/vendor the required quantizer logic and generate one layer first.

Compiler steps per expert:

```text
source BF16 matrix
 -> float working matrix
 -> Hessian-aware regularization
 -> input transform/scales
 -> output transform/scales
 -> K3 candidate
 -> K4 candidate
 -> proxy errors
 -> rate selection
 -> packed Trellis35 payload
```

For fused Gate+Up, reconstruct the full 1408 output and verify:

```text
inverse transform
split 704/704
GELU(gate) * up
```

against an independent high-precision reference.

Only after a complete single-layer artifact passes should the compiler generate all 30 layers.

### WP3 – Device loader, but no engine dispatch yet

Add Trellis35 manifest/loader support behind the new profile.

Requirements:

```text
one immutable device representation
no NVFP4 expert copy
strict source/format/rate/offset validation
all pointers fixed at initialization
no token-loop allocation
```

At this stage full inference may still reject the profile as "kernel not implemented."

### WP4 – Ordinary decode M=1 W4A8

This is the first actual runtime milestone.

Create a Gem16-native mixed-K3/K4 selected-expert kernel:

```text
8 selected experts
one cooperative/fused launch where practical
Gate+Up trellis decode
W4A8 Gate+Up GEMM/GEMV
inverse 1408 output transform
split 704/704
GELU multiply
Down input transform + pad to 768
W4A8 Down
inverse output transform
slot-ordered route weighting/reduction
```

Do not reconstruct a complete expert matrix to global memory.

K3/K4 choice is runtime descriptor data; avoid one Python/C++ launch per bitrate tier.

Required evidence:

- randomized operator parity
- real layer parity
- finite output
- route IDs and slot order unchanged
- CUDA graph replay
- memcheck/initcheck/racecheck
- exact bytes read from only selected experts
- profiler counters
- latency characterization

**No performance pass/fail threshold yet.**

If M=1 is 2x slower, record it and profile it; do not abandon the branch solely for that reason.

### WP5 – Fixed-D2 T=3 W4A8

This is mandatory for the discovery, not optional future work.

Do not implement it as three independent M=1 calls.

Treat the verifier as:

```text
3 Target rows
x 8 selected experts
= up to 24 routed assignments
```

Build a dedicated T3 path that:

- preserves three independent accumulator rows;
- groups/reuses an expert weight tile when the same expert occurs in multiple rows;
- keeps each row's exact Top-8 slot order and route weight;
- performs Gate+Up and Down transforms once per needed operand/tile;
- keeps current Fixed-D2 commit/rollback semantics unchanged;
- remains CUDA-graph capturable.

Benchmark:

```text
route overlap 0%
typical real overlap
maximal synthetic overlap
```

and the retained Wikipedia D2 workload.

Again, characterize performance; do not gate early discovery on a fixed percentage.

### WP6 – Prefill W4A8

Skip W4A16.

Reuse the existing:

```text
Gemma4MoePrefillPlan
routing
assignment list
permutation
workspace ownership
chunking
```

Replace only routed-expert compute.

Target execution:

```text
BF16 hidden
 -> input trellis transform
 -> MXFP8/E4M3 quantization
 -> mixed K3/K4 Trellis decode to E4M3
 -> FP8 Tensor Core Gate+Up
 -> inverse transform + GELU product
 -> Down transform + FP8 quantization
 -> FP8 Tensor Core Down
 -> route reduction
```

The first full-prefill implementation is allowed to be dramatically slower than NVFP4.

A result such as:

```text
~3000 prompt tok/s vs ~7000 baseline
```

is still a valid discovery result if correctness, memory and profiling are solid.

Mandatory output is a bottleneck decomposition:

```text
trellis global loads
decode instructions
activation transforms
activation quantization
FP8 MMA
route packing
Gate/Up inverse transform + GELU
Down transform
reduction
register count
spill bytes
shared memory
occupancy
DRAM throughput
Tensor Core utilization
```

Optimization follows evidence rather than an arbitrary early gate.

### WP7 – Full text-only model characterization

Run the complete model with the new artifact.

Report:

```text
actual device arena
actual peak VRAM
effective bpw
32K / 64K / 98K / larger-context admission
Ordinary decode
Fixed-D2 decode
prefill
TTFT
D2 acceptance
no recurring allocations/fallbacks
```

Do not attempt vision yet.

### WP8 – Quality characterization

Because both weight quantization and W4A8 activation precision differ from the current W4A4 path, separate the causes where possible.

Required:

```text
per-layer numerical differential
router Top-8 drift
full-logit KL/NLL
top-1 agreement
GSM8K
AIME
GPQA subset/full depending available harness
long-context retrieval
sampled generation
Fixed-D2 Target identity/acceptance behavior
```

Freeze prompts/seeds before inspecting failures.

#### Owner-directed staging (2026-08-30)

The quality contract was frozen before candidate inspection, but the first
correct full-model implementation is intentionally too slow for the complete
suite to be an efficient pre-optimization gate.  WP8 is therefore split
without changing the frozen prompts, seeds, datasets, or thresholds:

```text
WP8A before performance work:
  all-layer numerical/router/logit differential
  sampled-generation determinism for candidate and control
  16K retrieval at 10% / 50% / 90% needle placement

WP8B after the first performance pass:
  64K / 98K retrieval
  GSM8K / AIME / GPQA
  full Fixed-D2 identity and acceptance characterization
```

WP8A is a bounded sanity/diagnostic gate, not completion of the original WP8
quality scope and not product qualification.  The already frozen full suite is
retained unchanged for WP8B.  The owner explicitly superseded the original
execution order after the three 16K retrieval cases because running the large
suite at roughly 212 prompt tok/s and 38–40 decode tok/s would consume many
hours on an implementation already known to need optimization.

### WP9 — first performance pass (complete 2026-08-30)

The first evidence-led optimization pass replaced repeated branch-history
loads with direct two-word tail-biting state extraction, paired state decode,
warp-shared payload words, and four-deep payload prefetch.  It preserved the
single mixed-K3/K4 persistent artifact and the W4A8 compute contract.

On the bounded 512 x 16 workload, prompt time fell from 2449.37 ms to
506.74 ms and ordinary decode rose from 39.08 to 123.44 tok/s with the exact
same output-token hash.  On the real 16K x 64 workload, the path reached
951.09 prefill tok/s and 119.44 ordinary decode tok/s.  A bounded Fixed-D2 T3
run passed exact ordinary-Target identity and the frozen 50% precision gate,
accepted 38/50 drafts, reached 168.51 tok/s inside the batched verifier, and
85.16 tok/s end to end.

The final profiler still attributes 63.3% of GPU time to grouped Trellis
prefill projection and another 21.0% to activation/output transforms.  An
eight-row reuse candidate regressed prompt time by 23.1% and was reverted.
The next performance design should therefore address large-M reconstruction
amortization and transform fusion rather than merely increasing per-thread
row accumulators.  Full evidence is retained in
`artifacts/trellis35/wp9-runtime-decoder-optimization.json`.

Per the owner decision, WP8B remains deferred until another reviewed
performance plan materially improves the current path.  A dedicated external
review request is retained in
`docs/plans/gemma4-26b/CHATGPT_PRO_TRELLIS35_PERFORMANCE_REVIEW_2026-08-30.md`.

---

## 10. Discovery performance policy

There is deliberately **no early hard performance gate**.

The branch has two different definitions of success.

### Discovery success

- exact reproducible 3.5-bpw artifact exists;
- real Target arena drops by about 2.3 GiB;
- Ordinary M=1 works;
- Fixed-D2 T3 works;
- Prefill W4A8 works;
- outputs pass defined numerical/quality checks;
- no duplicate persistent expert layout;
- all paths are profiled well enough to explain the slowdown.

### Later optimization success

The long-term engineering objective is to recover as much of the current:

```text
~6966 prompt tok/s
~148 ordinary decode tok/s
~204 Fixed-D2 tok/s
```

as practical.

Only after the first correct full model exists should a later owner decision define promotion thresholds.

---

## 11. W4A4 is the second-generation performance experiment

Once W4A8 is understood, add a separate compute backend:

```text
large M:
Trellis35
 -> inline E2M1 values + E4M3 K/16 scales
 -> existing/native SM120 block-scaled W4A4 MMA
```

while keeping:

```text
M=1 and possibly T3:
W4A8
```

if W4A8 proves numerically attractive there.

The key invariant is that W4A4 must consume the **same Trellis35 persistent payload**.

Never keep:

```text
Trellis experts + full NVFP4 experts
```

simultaneously merely to choose a compute path.

Possible final dispatcher:

```cpp
if (mode == ordinary_decode)
    trellis35_w4a8_m1(...);
else if (mode == fixed_d2_t3)
    trellis35_w4a8_t3(...);
else if (M >= crossover_m && trellis35_w4a4_supported)
    trellis35_w4a4_prefill(...);
else
    trellis35_w4a8_prefill(...);
```

The crossover is measured, not hardcoded from theory.

---

## 12. Deferred exact-704 optimization

After the v1 path is stable, investigate replacing Down's:

```text
704 -> 768
```

with a native exact-704 transform, likely a 128-block body plus a 64-wide tail/alternative transform.

At 3.5 bpw this can recover roughly another 0.28 GiB beyond the initial ~2.317 GiB saving.

It is intentionally not part of first discovery because it changes both quantizer numerics and decoder geometry while the larger fused-GateUp gain is already available.

---

## 13. Files likely to change

### Compiler/format

```text
tools/gem16_compile/profiles.py
tools/gem16_compile/plan.py
tools/gem16_compile/compiler.py
tools/gem16_compile/writer.py
tools/gem16_compile/schemas/compiler-plan.schema.json
tools/gem16_compile/schemas/gem16-compilation.schema.json
tools/gem16_compile/specs/trellis35-experts-v1.json
tools/gem16_compile/trellis35*.py
tools/generate_gemma4_26b_trellis35_plan.py
third_party/exllamav3_quant/*
```

### Model/artifact

```text
src/model/gemma4_26b_manifest.*
src/model/gemma4_26b_compiled_loader.*
src/model/gemma4_26b_device_image.*
src/model/gemma4_26b_residency.*
src/cuda/engine/gemma4_26b_artifact.*
```

### CUDA

Suggested new namespace rather than mixing into current NVFP4 files:

```text
src/cuda/trellis35/format.h
src/cuda/trellis35/decode.cuh
src/cuda/trellis35/transform.cuh
src/cuda/trellis35/w4a8_mma.cuh
src/cuda/trellis35/decode_m1.cu
src/cuda/trellis35/decode_t3.cu
src/cuda/trellis35/prefill.cu
src/cuda/moe/gemma4_26b_trellis35.*
```

Keep `src/cuda/nvfp4/*` unchanged except for truly shared helper extraction justified by tests.

---

## 14. Test matrix Codex must build as it goes

### Host/compiler

- deterministic K3/K4 allocation
- repeat build byte identity
- fused Gate+Up no-padding assertion
- Down 768 padding assertion
- malformed rate map
- invalid pool offset
- duplicate/overlap
- truncated payload
- wrong source lock
- wrong codebook ID
- integer overflow
- manifest/device-image round trip

### CUDA unit/operator

- trellis window extraction
- K3 decoder
- K4 decoder
- mixed descriptor decoder
- transforms
- E4M3 packing
- FP8 MMA fragments
- Gate+Up inverse-before-split
- GELU product
- Down padded input
- route weighting/reduction
- M1
- T3
- prefill tails

### Runtime

- fixed addresses
- CUDA graph capture/replay
- zero replay allocation
- cancellation/rollback unchanged
- capacity rejection
- old NVFP4 profile bitwise/regression protected
- 12B protected

---

## 15. Codex working rules

1. Read `AGENTS.md` and `docs/ACTIVE_DECISIONS.md` before each work package.
2. Treat Trellis35 as an unqualified experimental profile.
3. Never alter accepted NVFP4 tensor semantics to "make room" for the experiment.
4. Never keep a model-sized NVFP4 expert fallback alongside Trellis35.
5. No runtime JIT or runtime requantization.
6. Preserve exact test prompts, sampling and timing boundaries in A/B results.
7. Record every rejected performance experiment rather than silently replacing the baseline.
8. Every copied upstream source file must retain license/provenance.
9. Prefer a correct, profiled slow W4A8 implementation over an opaque fast result.
10. Vision work is forbidden in this branch until the text-only discovery report is complete.

---

## 16. Expected final discovery report

The branch is ready for an owner decision when it can produce one document containing:

```text
A. exact source/compiler/toolchain identities
B. exact artifact byte accounting
C. actual RTX 5080 device arena / peak VRAM
D. effective bpw
E. Ordinary M1 correctness + latency
F. Fixed-D2 T3 correctness + latency
G. Prefill W4A8 correctness + throughput
H. detailed profiler attribution
I. quality/KLD/task comparison
J. context capacity measurements
K. list of unresolved optimization opportunities
L. recommendation:
   - stop
   - optimize W4A8
   - implement Trellis->W4A4
   - later explore exact704
```

The discovery itself does not need to prove product-level performance parity. Its job is to establish whether the ~2.3 GiB memory gain is real, what it costs in W4A8 performance/quality, and which low-level bottlenecks must be solved next.

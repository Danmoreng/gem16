# Codex Master Instruction — Gem16 26B Vision Productization V09–V20

## Starting point

Use the current clean branch/commit:

```text
branch: codex/gemma4-26b-vision-fp8
commit: 109a4be444322a3921a520f6f911ac00ddfcb92f
```

Create a new branch only when explicitly authorized, suggested:

```text
codex/gemma4-26b-vision-product
```

Read before every package:

```text
AGENTS.md
docs/ACTIVE_DECISIONS.md
REVIEW_REQUEST.md
docs/plans/gemma4-26b/V00_VISION_PROFILE_AND_MODULE_CONTRACT.md
GEM16_26B_VISION_POST_109A4BE4_PRODUCTIZATION_REVIEW.md
```

Do not push, merge, rebase, reset, or alter unrelated files without owner instruction.

---

# Permanent product constraints

Preserve all of these paths:

```text
12B qualified multimodal
26B qualified NVFP4 text
26B experimental Trellis35 text
26B experimental Trellis35 + FP8 Vision
```

Vision v1 remains bound to:

```text
profile:
  gemma4_26b_trellis35_vision_fp8

text artifact:
  gem16-trellis35-w4a8-v1

Vision source:
  google/gemma-4-26B-A4B-it-qat-q4_0-unquantized
  f1e06dc520982d9b9edd76859fdb7ab209449949

oracle:
  Transformers 5.14.1
  a08ace4bbd97e721c98751deec37d87b026acadc
```

Never introduce:

- runtime quantization or model repack;
- CPU Weight offload or expert streaming;
- duplicate persistent Weight representations;
- token/image-loop `cudaMalloc`;
- unbounded graph caches;
- silent precision, format, kernel, context, image-budget, MTP, or sampling fallback;
- inherited text-only context qualification for the Vision profile;
- a Vision capability enabled only because a file exists;
- a changed physical BF16/FP32 boundary to gain speed.

The current user-visible Vision+D2 fail-close remains active until V14 is accepted.

---

# V09 — Ordinary multimodal semantic closure

## Purpose

Make the current Ordinary 26B image path match the pinned Gemma 4 text-side Vision semantics before profiling or D2 diagnosis.

## V09-A — local image-block attention

### Current issue

`PrefillTokensWithVision` inserts the image embeddings, but:

```text
LaunchGemma4Moe26BAttentionSm120PrefillLayer
```

does not receive the image range. Local layers therefore use ordinary causal local attention.

### Required changes

Touchpoints:

```text
src/cuda/engine/detail/gemma4_26b_prefill.inc
src/cuda/attention/gemma4_26b_reference.h
src/cuda/attention/gemma4_26b_reference.cu
src/cuda/attention/prefill_local_sm120.cu
tests/cuda/
```

Extend the 26B prefill API with:

```cpp
std::uint64_t vision_begin = 0U,
std::uint64_t vision_end = 0U
```

Contract:

```text
sliding/local layers:
  local sliding mask AND (causal OR same Vision block)

global layers:
  remain causal
  receive an empty Vision range
```

Reuse the existing Vision-aware local SM120 kernel. Do not create a different mask formula.

The image range is absolute in the prompt. Convert/check it consistently at the wrapper boundary.

## V09-B — image-span-aware chunk plan

The current 2,048/1,024 chunking must not split:

```text
[vision_begin, vision_end)
```

Build a bounded plan before the loop.

Required cases:

- shorten the chunk immediately before the image if needed;
- put the complete image span in one chunk;
- never exceed current maximum chunk size;
- continue normal chunking afterward.

No host synchronization may be added to the normal per-layer path.

Tests at offsets:

```text
0
1
1023
1024
1940
2047
2048
4095
16105
16383
16384
```

plus:

```text
fully cached image span
fully uncached image span
cache prefix that splits image span -> reject
```

## V09-C — canonical positions

In `Gemma4Moe26BVisionRuntime::Encode`, require exact row-major positions:

```cpp
expected_x = patch % grid_width;
expected_y = patch / grid_width;
```

Validate uniqueness implicitly through exact sequence.

Also prove:

```text
soft_token_count =
  (grid_width / 3) * (grid_height / 3)
```

## V09-D — padded versus unpadded oracle

The pinned processor pads to the selected maximum patch count and masks padding. Current Gem16 executes only real rows.

Create a bounded comparison for budgets:

```text
70 / 140 / 280
```

and aspect ratios:

```text
square
wide
tall
minimum narrow valid
```

Capture:

```text
patch embedding valid rows
layer 0 / 13 / 26
pooled rows
final 2816 image embeddings
```

Decision:

- if equivalent under a predeclared boundary, keep unpadded execution and fix the inaccurate header comment;
- otherwise implement padding plus attention mask exactly.

Do not tune the tolerance after seeing results.

## V09 acceptance

- local/global masks match the pinned oracle;
- no image span is split across a chunk;
- all position-contract tests pass;
- padded/unpadded decision is documented;
- deterministic Ordinary output on frozen images;
- text-only Trellis and NVFP4 hashes unchanged;
- no device allocation delta;
- D2 still fail-closed.

For the NVFP4 hash gate, use the current qualified published 26B Target
`danmoreng/gemma-4-26B-A4B-it-GEM16` at revision
`63508b5826527484e707b4b46e2eacf077cf2b35` in its
`sm120-device-image-v1` layout. The original 16-shard M08 working artifact is
historical/deprecated and is not a runtime regression target. Do not search
for, reconstruct, or recompile that old layout merely to satisfy this gate.
M08 identity remains provenance; current device-image dispatch and output
identity are the test boundary.

Write:

```text
artifacts/vision/v09-ordinary-semantic-closure.json
```

Stop after V09 if the correct mask changes the current expected image outputs; record the new baseline rather than hiding it.

---

# V10 — dedicated Vision test, benchmark, capacity, and profiler baseline

## Purpose

The current archive has no reliable Vision performance evidence. Create it before optimization.

## V10-A — test target

Add:

```text
gem16-cuda-vision26b-tests
```

Suggested source split:

```text
tests/cuda/vision26b_artifact_test.cu
tests/cuda/vision26b_preprocess_test.cpp
tests/cuda/vision26b_patch_test.cu
tests/cuda/vision26b_attention_test.cu
tests/cuda/vision26b_mlp_test.cu
tests/cuda/vision26b_pool_project_test.cu
tests/cuda/vision26b_text_integration_test.cu
tests/cuda/vision26b_test_support.*
```

Keep `src/cuda/vision/gemma4_26b.cu` in one TU initially; split into include shards only if V13 makes it unwieldy.

## V10-B — frozen fixtures

Create or lock fixtures for:

```text
70 budget / maximum valid grid
140 budget / maximum valid grid
280 budget / maximum valid grid
square / wide / tall
```

Record original image SHA-256, processed dimensions, positions, raw/soft counts.

Do not include copyrighted external images without permission; generated geometric/text/chart fixtures are preferred.

## V10-C — timing and NVTX

Add stage timing/NVTX:

```text
decode
resize
patchify
upload
patch project
position add

per layer:
  input norm/quant
  QKV projection
  QKV norm/RoPE
  attention
  O projection/residual
  FFN norm/quant
  Gate/Up
  GELU
  product quant
  Down/residual

pool
standardize
final norm/project
text embedding insertion
text prefill
TTFT
```

Timing boundaries must be documented. Model loading is separate.

## V10-D — profiler

For 70/140/280, retain:

```text
NSYS
NCU for HeadNormRopeKernel
NCU for FullAttentionKernel
NCU for PoolStandardizeKernel
NCU for GELU/product quant boundary
```

Record:

```text
duration
registers
shared
stack/local/spills
occupancy
issue slots
SM/DRAM/L2/L1
long scoreboard
math/special-function instructions
bank conflicts
launch count
```

## V10-E — memory/capacity

Report:

```text
Vision weights
Vision workspace
graph-private bytes
host pinned bytes
text weights
Assistant weights
KV
free after load
peak during image
```

Matrix:

```text
Target + Vision
Target + Vision + Assistant
32K
64K
86,016 diagnostic
first rejected context
```

Do not publish inherited text limits.

## V10-F — refresh stale evidence

Replace the diagnostic limitation statements only in a new immutable file:

```text
artifacts/vision/v10-runtime-baseline.json
```

Never rewrite the historical V01/V02 diagnostic.

---

# V11 — Vision Fixed-D2 exactness laboratory

## Purpose

Diagnose the exploratory mismatch without enabling D2 in normal requests.

## V11-A — diagnostic gate

Add an explicit local-only switch:

```text
GEM16_VISION_D2_DIAGNOSTIC=1
```

Requirements:

- disabled by default;
- not settable by model metadata or API request;
- logs a warning;
- excludes the run from product claims.

The normal user-facing path continues to return `vision_mtp_unqualified`.

## V11-B — frozen trajectory manifest

Freeze:

```text
image SHA-256
image budget
processed grid
rendered prompt token SHA-256
image begin/end
sampling
Target/Assistant/Vision hashes
binary hash
```

Start greedy.

## V11-C — post-prefill state capture

Capture before first Assistant proposal:

```text
position
first Target selected token
final_hidden bits/hash
local/global KV bytes/hash
ring positions
decode control
prediction status
sampling step
repetition/suppression state
```

Compare MTP-off and diagnostic-MTP-on runs up to that point.

## V11-D — forced proposals

Build a diagnostic method that feeds the next two exact sequential Ordinary Target tokens as proposals.

This isolates:

```text
Target T3 verifier
selection
KV tentative/restore/commit
```

from Assistant proposals.

No product API.

## V11-E — per-row/per-layer differential

Compare each T3 row against three sequential Ordinary Target forwards.

Capture first divergence at:

```text
input
attention
shared MLP
routed MLP
layer output
final norm
head
```

Include local and global layer classes separately.

## V11-F — transaction differential

Before/after each group compare:

```text
KV backup
tentative append
accepted prefix
rejected restore
committed position
pending token
```

## V11-G — real Assistant

Only after forced proposals are exact:

- enable actual Assistant;
- compare proposal context;
- verify acceptance;
- require final token identity.

## Matrix

```text
budget 70/140/280
image at start/middle/chunk boundary
short context
>16K boundary
local ring wrap
first turn
cached continuation
greedy
fixed-seed sampled
cancellation
```

## Acceptance

V11 may conclude with a diagnosed defect; it need not enable D2.

Write:

```text
artifacts/vision/v11-d2-exactness.json
```

Do not proceed to V14 until:

```text
forced proposals exact
real Assistant final stream exact
per-group KV/position exact
```

---

# V12 — low-risk Vision performance wave

Run only after V09/V10.

Every candidate uses same-binary rollback and frozen inputs.

## V12-A — direct row-major 3×3 pooling

Replace the O(soft × hidden × raw) coordinate scan with exactly nine direct loads per output after V09 row-major validation.

Preserve summation order and all BF16/FP32 boundaries.

Rollback:

```text
GEM16_VISION_POOL=scan|direct
```

Gate:

- byte-identical pool output;
- byte-identical final embeddings;
- full tower/TTFT gain;
- no allocation.

## V12-B — precomputed 2D RoPE

Build bounded BF16 cosine/sine tables once per image.

Preferred shape:

```text
x positions × 18 frequencies × {cos,sin}
y positions × 18 frequencies × {cos,sin}
```

Use exact current formula and BF16 rounding.

Rollback:

```text
GEM16_VISION_ROPE=transcendental|table
```

Gate:

- bit-identical Q/K normalized+rotated values;
- head-72 exact;
- special-function instruction reduction;
- full-tower gain;
- table bytes included in workspace.

## V12-C — GELU product + BF16-round + FP8 quant

Fuse:

```text
GELU-tanh Gate×Up
BF16 product
amax
E4M3 quant
```

Preserve the physical BF16 product.

Rollback:

```text
GEM16_VISION_FFN_QUANT=split|fused
```

Gate:

- E4M3 bytes/scales exact;
- full output exact;
- no workspace growth;
- full-tower gain.

## V12-D — budget-sized workspace

Make the selected budget a startup/profile property.

Allocate fixed capacity for:

```text
70 -> 630 raw
140 -> 1260 raw
280 -> 2520 raw
```

Expected planning sizes from the current layout:

```text
70:   36,290,048 bytes
140:  64,191,488 bytes
280: 119,993,600 bytes
```

These must be recomputed by code and verified.

Changing budget requires a controlled runtime/profile reconfiguration, not per-request allocation.

## V12-E — input staging only if measured

Current `Encode` synchronizes the stream before filling pinned input.

Only if V10 shows material host/input cost:

- use two pinned slots;
- event-based reuse;
- no global stream synchronize;
- no unbounded buffers.

---

# V13 — tiled K/V-reuse Vision attention

## Purpose

Replace the correctness-first per-query full-memory scan while preserving the exact two-pass softmax boundary.

## Current normative boundaries

Keep:

```text
16 heads
head dimension 72
non-causal Vision attention
Q/K BF16 after norm+RoPE
V BF16 after scale-free norm
FP32 max and denominator
probability rounded to BF16
BF16 probability × BF16 V accumulated in FP32
BF16 output
```

## First kernel family

Test:

```text
Q rows per CTA: 4 and 8
K/V tile:       32 and 64
one head per CTA
```

A Q4/K32 design:

```text
128 threads / four query warps
shared K[32][72] BF16
shared V[32][72] BF16
~9 KiB logical K/V shared
```

Each warp owns one query and consumes source tokens in the same increasing order as the reference.

Two passes remain:

1. max/denominator;
2. score recomputation, BF16 probability, V accumulation.

Do not introduce a quadratic score slab.

## Optimization sequence

1. synchronous shared staging;
2. prove exactness;
3. `cp.async` double buffering;
4. choose Q4/Q8 and K32/K64 from NCU.

Do not start with a one-pass online softmax.

## Tests

```text
tokens: 9, 18, 63, 64, 65, 630, 1260, 2520
head: all 16
position patterns: square/wide/tall
tail 72
```

Compare:

```text
attention output bits
layer output
final image embeddings
```

## Performance gate

No invented fixed target.

Retain a candidate only if:

- no spills;
- material global K/V traffic reduction;
- reference-compatible output;
- full 27-layer tower and TTFT improve;
- workspace remains bounded.

Write:

```text
artifacts/vision/v13-tiled-attention.json
```

---

# V14 — Fixed-D2 enablement and performance

Begin only after V11 exactness passes.

## Runtime behavior

For one uncached image:

```text
Vision tower once
correct multimodal Target prefill
first Target selection
normal Fixed-D2 generation
```

The tower must not run per proposal or verifier group.

## Capability

Only the exact accepted profile sets:

```text
vision_mtp_supported = true
```

All other combinations fail closed.

## Benchmark

Frozen image+Wikipedia/text prompt:

```text
3 warm-ups
10 retained Ordinary/D2 pairs
same seed/sampling/output cap
```

Report separately:

```text
image preprocess
tower
text prefill
TTFT
post-first Ordinary decode
post-first Fixed-D2
acceptance
groups
fallback
peak VRAM
```

Correctness:

```text
final streams identical
accepted/rejected trajectory frozen
KV transaction exact
no fallback
no allocation
```

---

# V15 — runtime/server profile and observability closure

## Composite identity

Add explicit fields:

```text
profile_id
text_artifact_profile
vision_artifact_profile
experimental
```

## Capabilities

Add to engine/runtime/health:

```text
vision_module_loaded
supports_vision
vision_mtp_supported
maximum_images
vision_soft_token_budgets
selected_vision_soft_token_budget
vision_weight_bytes
vision_workspace_bytes
vision_max_context_tokens
qualification_state
```

Do not expose text-only context qualification as Vision qualification.

## Server validation

Validate both:

```text
GenerationContentKind::kImage
GenerationContentKind::kGemma4Moe26BImage
```

Reject more than one image before preprocessing.

Explicit error codes:

```text
vision_module_not_loaded
vision_profile_required
vision_multiple_images_unsupported
vision_mtp_unqualified
vision_budget_unsupported
vision_context_unqualified
vision_artifact_mismatch
```

## Metrics

Add timings/counters listed in the review.

Update OpenAI Chat and Responses paths equally.

Tests:

- internal parsed kind;
- remote image URL/data path as supported by current API contract;
- unsupported combinations;
- health JSON;
- Prometheus semantics;
- cancellation.

---

# V16 — sidecar publication and component catalog

## Goal

Make the exact Vision sidecar installable without duplicating target blobs.

Tasks:

1. create immutable Vision repository/package or approved component source;
2. include:
   - `vision.gem16`
   - `gem16_vision.json`
   - `vision_compilation.json`
   - `vision.lock.json`
   - README/LICENSE/NOTICE;
3. pin exact revision and file hashes;
4. update the generated model catalog;
5. publish the Trellis Target component if it is not already an installable product component;
6. define Target/Assistant/Vision compatibility;
7. verify download/resume/hash/hardlink behavior;
8. no duplicate private blob store.

The Vision profile must require Target+Vision; Assistant is optional until V14 and then selected by capability.

---

# V17 — Native Studio profile/component/settings integration

## Profile enum

Add an explicit entry:

```cpp
kGemma4Moe26BTrellis35VisionFp8
```

Do not modify the meaning of `kGemma4Moe26BA4B`.

## Catalog data model

Replace fixed Target/Assistant assumptions with a bounded component span:

```cpp
enum class ModelComponentKind {
  kTarget,
  kAssistant,
  kVision,
};

struct ModelProfileComponent {
  ModelComponentKind kind;
  const ModelComponentCatalog* catalog;
  bool required;
};
```

Remove fixed array-of-two and ternary profile indexing. Use enum-index helper or vector/map with exhaustive switches.

## ServerConfig/settings

Add:

```text
vision_directory
vision_soft_token_budget
```

Add MTP control driven by live `vision_mtp_supported`.

Persist and migrate old settings.

## Command

`BuildServerCommand` adds:

```text
--vision-model
```

only for the explicit Vision profile.

## Install UI

Show component state independently:

```text
Target
Vision
Assistant
```

Ready state must reflect required components.

Storage preflight includes all missing blobs.

---

# V18 — Native Studio image/D2 user experience

## Attachment gating

For the explicit live-compatible Vision profile:

```text
one image allowed
audio rejected
document behavior unchanged
```

Remove the blanket 26B image rejection only for that profile.

Show:

```text
image preview
processed budget 70/140/280
estimated soft tokens
experimental badge
remove/retry
```

Reject a second image locally.

## D2

Until V14:

- disable MTP for image request;
- explain that Vision+D2 is unqualified;
- do not silently downgrade explicit D2 to Ordinary.

After V14:

- enable only if server health says `vision_mtp_supported=true`.

## External server

Compare profile/capabilities from health. Never assume local settings match the process.

## Tests

- profile switching;
- restart persistence;
- command line;
- missing module;
- wrong sidecar;
- one/two images;
- D2 disabled/enabled states;
- external server mismatch;
- streaming/cancellation.

---

# V19 — capacity, bounded quality, cross-platform and lifecycle gates

## Capacity

Fresh-process measurements for:

```text
Target + Vision
Target + Vision + Assistant
budget 70/140/280
32K
64K
candidate higher contexts
first rejected context
```

Report:

```text
weights
workspace
graphs
KV
peak
free reserve
```

Do not inherit 86,016 or 98,304.

## Quality

Bounded first gate:

- image description;
- OCR;
- chart;
- document page;
- counting;
- spatial relations;
- colors;
- small details;
- wide/tall/square;
- budgets 70/140/280.

Compare against the pinned BF16 oracle where executable.

Metrics:

```text
final image embedding cosine/relative L2
text logit KL/top-1
deterministic answer checks
paired blind qualitative review where appropriate
```

## Lifecycle/product

- Windows and Linux;
- clean machine;
- install/download/resume;
- server start/stop;
- repeated images;
- cached conversation continuation;
- cancellation during tower and text decode;
- queueing;
- no allocation growth;
- sidecar corruption/mismatch;
- 12B/NVFP4/Trellis-text regression.

---

# V20 — experimental product freeze

Freeze only when:

```text
V09 semantics accepted
V10 evidence complete
V11/V14 D2 exact and enabled, or explicitly excluded from profile claim
V12/V13 selected performance path accepted
V15 capabilities truthful
V16 package/catalog accepted
V17/V18 Studio complete
V19 capacity/quality/platform gates accepted
```

Initial claim:

```text
Experimental Gemma 4 26B Trellis35 + FP8 Vision
one image
batch one
selected image budget
measured context limit
bounded image-quality evidence
```

Do not claim broad multimodal production quality without a later larger suite.

---

# Reporting standard

Every package writes a new immutable artifact and reports:

```text
commit
files
commands
source/model/artifact/binary hashes
GPU/driver/CUDA
input fixture identities
numerical boundary
SASS
register/shared/stack/local/spills
NSYS/NCU
Same-Binary A/B
stage and full-path timing
VRAM/allocation delta
fallback count
accepted or rolled back
limitations
```

Never overwrite historical evidence.

---

# First Codex assignment

Start with **V09 only**.

Do not profile or optimize the Vision tower until:

- local image-block attention is wired correctly;
- image spans cannot be split across prompt chunks;
- positions are canonical;
- padded/unpadded equivalence is resolved;
- text-only regressions pass.

At the end of V09, stop and present the semantic evidence before beginning V10.

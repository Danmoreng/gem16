# Gem16 Gemma 4 26B Vision – Post-109a4be4 Productization Review

**Review basis:** `codex/gemma4-26b-vision-fp8@109a4be444322a3921a520f6f911ac00ddfcb92f`  
**Archive:** `gem16-chatgpt-review-2026-09-01-109a4be4.zip`  
**Date:** 2026-09-01  
**Review type:** static source review plus compact artifact/evidence review  
**Not available in the archive:** compiled binaries, model payloads, image fixtures, raw Vision performance captures, Nsight databases, or the exploratory Vision+D2 mismatch trace

---

## 1. Executive verdict

The branch has completed substantially more than an initial Vision proof of concept:

- a separately locked and validated FP8/BF16 Vision sidecar exists;
- the sidecar has one resident GPU allocation and no runtime weight repack;
- the 26B image processor, raw 16×16 patch representation, 27-layer Vision tower, pooling, projector, and text-embedding insertion are implemented;
- CLI/server plumbing already accepts `--vision-model`;
- Ordinary image generation reaches the text engine;
- the qualified 12B and 26B NVFP4 product paths remain separate;
- Fixed-D2 image requests remain deliberately fail-closed.

However, the current branch is **not yet ready to be presented as a product Vision profile**. There are four major reasons:

1. **The 26B text-side local-attention mask does not yet receive the image-span semantics required by the pinned Gemma 4 oracle.** The image embeddings are overlaid, but the normal causal 26B prefill attention is still invoked without a Vision range.
2. **The normal 2,048/1,024-token chunk planner can split a 280-token image span across chunks.** The existing local Vision-aware kernel requires the complete Vision range to be contained in the current chunk.
3. **Fixed-D2 exactness has not been diagnosed.** The archive states that one exploratory run diverged from Ordinary, but it contains no raw mismatch trace, state hashes, first divergent token, or layer boundary.
4. **Native Studio still models only the old 12B and 26B text product profiles.** It has no Vision component path, no `--vision-model` wiring, no Vision capability fields, and explicitly rejects image attachments for 26B.

There is also no trustworthy Vision performance baseline in the archive. The current CUDA code contains several obvious high-cost reference implementations, especially:

- quadratic two-pass full attention with no cross-query K/V reuse;
- trigonometric 2D RoPE recomputed in Q and K for all 27 layers;
- pooling that scans every raw patch for every pooled output element;
- five separate major linear launches per Vision layer around attention/MLP boundaries;
- no Vision CUDA graph;
- a fixed maximum 114.4 MiB workspace even for 70- or 140-token image budgets.

The correct next sequence is therefore:

```text
V09  Ordinary multimodal semantic closure
V10  dedicated tests, profiler and capacity baseline
V11  exactness-first Fixed-D2 diagnosis
V12  low-risk Vision speedups
V13  tiled bidirectional Vision attention
V14  Fixed-D2 product enablement and timing
V15  runtime/server capability and observability closure
V16  sidecar publication/catalog/component model
V17  Native Studio model/profile/settings integration
V18  Native Studio image and D2 UX
V19  capacity, packaging and bounded quality qualification
V20  experimental product freeze
```

**Do not start with performance tuning.** First close the text-side multimodal semantics, because otherwise profiling and D2 debugging would optimize a path whose model behavior is not yet the pinned Gemma 4 behavior.

---

# 2. Demonstrated current state

## 2.1 Profile and artifact boundary

Vision v1 is explicitly bound to:

```text
profile:
  gemma4_26b_trellis35_vision_fp8

required text artifact:
  gem16-trellis35-w4a8-v1

source:
  google/gemma-4-26B-A4B-it-qat-q4_0-unquantized
  f1e06dc520982d9b9edd76859fdb7ab209449949

semantic oracle:
  Transformers 5.14.1
  a08ace4bbd97e721c98751deec37d87b026acadc
```

The sidecar is not intended to extend the qualified NVFP4 profile or every 26B checkpoint. File presence alone is not meant to enable the capability.

## 2.2 Vision artifact

The compact diagnostic reports:

```text
source tensors:              356 BF16 tensors
source payload:              1,145,588,832 bytes

compiled tensors:            547
tensor bytes:                597,301,792
zero alignment padding:          11,232
uploaded payload extent:     597,313,024
container bytes:             597,390,648

device allocations:          1
standalone CUDA delta:       597,688,320 bytes
text + Vision free after
resident weights:          3,622,633,472 bytes
```

The format contains:

```text
191 large two-dimensional linears:
  FP8 E4M3FN weights
  BF16 per-output-row scales

165 remaining tensors:
  source BF16
```

The artifact is already close to the intended approximately 570 MiB static Vision-weight target.

## 2.3 Implemented runtime sequence

`src/cuda/vision/gemma4_26b.cu` currently implements:

```text
host RGB/resized image
  -> real 16×16×3 patches
  -> 2 × (pixel - 0.5)
  -> FP8 patch projection 768 -> 1152
  -> BF16 x/y position embeddings

27 ×:
  RMSNorm + FP8 activation quantization
  Q projection
  K projection
  V projection
  Q/K learned RMSNorm + 2D RoPE
  V scale-free RMSNorm
  bidirectional full Vision attention
  O projection + residual/norm
  Gate projection
  Up projection
  GELU-tanh Gate×Up
  FP8 quantization
  Down projection + residual/norm

  -> 3×3 spatial pooling
  -> sqrt(1152) scaling
  -> standardization
  -> scale-free RMSNorm
  -> FP8 projector 1152 -> 2816
  -> float text embeddings
```

`PrefillTokensWithVision` encodes the image, inserts the resulting 2,816-wide embeddings at the `<|image|>` placeholder rows, and then runs the normal text prefill.

## 2.4 Current Fixed-D2 policy

The session currently rejects a request that contains an uncached 26B image while MTP is enabled:

```text
Gemma 4 26B Vision v1 requires Ordinary decoding
```

This is an appropriate temporary fail-closed policy.

The MTP Assistant itself does not need a second Vision tower. Its proposal context consumes the Target's final hidden state and Target KV views. Therefore the architectural goal is:

```text
Vision tower once
  -> exact Target multimodal prefill state
  -> normal Assistant proposals from Target state/KV
  -> exact Target T3 verification
```

The unresolved problem is the exact Target state/verifier boundary, not necessarily a missing Assistant-side image encoder.

---

# 3. P0 semantic findings

## 3.1 Text-side Vision attention semantics are incomplete

### Pinned oracle behavior

For Gemma 4, the text model's local/sliding layers use a special image-block mask:

```text
local mask =
  sliding-window mask
  AND
  (causal relationship OR same Vision block)
```

This makes the image placeholder span bidirectional inside the local window. Global layers remain causal.

### Current 26B behavior

The 26B path:

1. inserts Vision embeddings into `prefill_hidden_a`;
2. invokes `LaunchGemma4Moe26BAttentionSm120PrefillLayer`;
3. that API does not accept `vision_begin`/`vision_end`;
4. local attention is therefore invoked as normal causal local prefill.

The repository already has the required Vision-aware local-attention machinery for the existing path:

```text
LaunchOnlineCausalAttentionPrefillFp8LocalSm120(
  ...,
  vision_begin,
  vision_end
)
```

The 26B wrapper simply does not expose or pass those values.

### Consequence

Ordinary image output may look plausible, but this branch has not yet demonstrated that text-side image-token attention matches the pinned Gemma 4 semantics.

This is a **product blocker and a D2-debug blocker**.

### Required fix

Extend only the 26B prefill boundary:

```cpp
LaunchGemma4Moe26BAttentionSm120PrefillLayer(
    ...,
    bool rotary_prepared,
    uint64_t vision_begin,
    uint64_t vision_end);
```

Then:

```text
sliding/local layer:
  pass the chunk-relative absolute Vision range

global layer:
  force vision_begin == vision_end == 0
  preserve causal global attention
```

Do not change decode attention. Once the image prompt has been prefetched, generated text uses normal causal decode.

---

## 3.2 The image span may be split by prompt chunks

The current prompt planner uses 2,048-token chunks inside the prepared first 16K and 1,024-token chunks afterwards.

`PrefillTokensWithVision` overlays whichever subset of the image span overlaps the current chunk. That means a 280-token image span beginning near a chunk boundary can be divided between two prefill launches.

The local Vision-aware kernel validates that:

```text
vision_end <= start_position + tokens
```

and its semantics assume the image block is represented as one range inside that launch.

### Required chunk planner

Before entering the loop, build a bounded chunk plan that guarantees:

```text
no chunk boundary occurs strictly inside
[image_begin, image_end)
```

Allowed strategy:

- before the image: shorten the preceding chunk so it ends exactly at `image_begin`;
- image-containing chunk: include the complete image span and surrounding text up to the normal limit;
- after the image: resume normal 2,048/1,024 chunking.

Because the image span is at most 280 tokens, this does not require a larger maximum chunk.

### Mandatory boundary cases

Test image starts at least at:

```text
0
1
1023
1024
1940
2047
2048
4095
prepared 16K boundary - 279
prepared 16K boundary - 1
prepared 16K boundary
cached-prefix boundary
```

Also test:

- image span wholly cached;
- wholly uncached;
- cache prefix attempting to split the image span, which must remain rejected.

---

## 3.3 Position validation is too weak for pooling assumptions

`Gemma4Moe26BVisionRuntime::Encode` currently proves only that:

```text
max_x + 1 and max_y + 1 form a 3-divisible rectangular area
grid_width × grid_height == raw_patch_count
```

It does not prove:

- positions are unique;
- every coordinate appears exactly once;
- positions are row-major;
- no coordinate is duplicated while another is absent.

This matters because the most attractive pooling optimization, and several possible attention/position optimizations, require a canonical row-major grid.

### Required validation

On the host, require for every patch index:

```cpp
expected_x = patch % grid_width;
expected_y = patch / grid_width;
position == {expected_x, expected_y};
```

This is O(raw_patch_count), occurs once per image, and removes ambiguity.

Also validate:

```text
grid_width % 3 == 0
grid_height % 3 == 0
raw_patch_count == grid_width × grid_height
soft_token_count == (grid_width / 3) × (grid_height / 3)
```

---

## 3.4 Padded versus unpadded Vision execution lacks a formal proof

The pinned processor pads:

```text
pixel patches -> selected max patch count
positions -> (-1, -1)
```

and the pinned model masks padded rows in the Vision encoder and strips invalid pooled outputs.

The current Gem16 host payload contains only real rows. The header says the CUDA tower supplies fixed 2,520-row zero/-1 padding, but the current CUDA runtime actually launches only over `raw_patch_count`.

This may be numerically equivalent for valid rows because masked padding does not participate in attention. The archive does not contain a proof.

### Required bounded oracle

For budgets 70, 140, and 280, and several aspect ratios:

```text
A. pinned padded+masked oracle
B. Gem16 unpadded execution
```

Compare:

- patch projection valid rows;
- layer 0/13/26 hidden rows;
- pooled valid rows;
- final 2,816-wide image embeddings.

If equivalent within the predeclared boundary, retain the faster unpadded contract and correct the misleading header comment. Otherwise implement explicit padding and attention masking.

Do not silently assume equivalence.

---

# 4. Fixed-D2 exactness-first diagnosis

## 4.1 What the archive supports

Demonstrated:

- text-only Trellis35 Fixed-D2 is stable;
- Vision Ordinary reaches generation;
- an exploratory Vision+D2 run produced a token stream different from Ordinary;
- product behavior remains fail-closed.

Not present:

- the image/prompt used;
- first divergent token;
- logits;
- accepted/rejected drafts;
- Target hidden/KV hashes;
- position/control snapshots;
- exact executable and graph identity for the mismatching run.

No definitive root cause can be assigned from this archive.

## 4.2 Likely boundary classes — hypotheses only

Ranked hypotheses:

1. **Ordinary multimodal prefill itself is not yet using the pinned local image mask.**
2. Image-span chunking produces a different resident Target state.
3. The pending selected token or `position` handed to the first Assistant/D2 group is off by one after multimodal prefill.
4. T3 Target verification differs from three sequential Ordinary Target forwards at a local/global attention or KV commit boundary.
5. Fixed-D2 backup/tentative/restore/commit does not preserve the exact post-Vision cache state.
6. The exploratory comparison used different sampling or timing boundaries.

The first two must be fixed before interpreting any D2 result.

## 4.3 Smallest exactness-first sequence

### Stage D0 — preserve product fail-close

Keep the existing user-visible rejection.

Add a clearly named diagnostic-only switch, disabled by default and excluded from performance/product claims:

```text
GEM16_VISION_D2_DIAGNOSTIC=1
```

It must be impossible to enable silently through a model file or request field.

### Stage D1 — freeze one trajectory

Use one immutable image and prompt. Record:

```text
encoded image SHA-256
processed dimensions
raw grid
raw patch count
soft-token count
rendered prompt token SHA-256
image begin/end
sampling configuration
Target/Assistant/Vision artifact hashes
executable hash
```

First run greedy, then same-seed sampled only after greedy closes.

### Stage D2 — capture post-prefill Target state

Immediately after Ordinary Vision prefill and first selection, capture:

```text
position
prefill call count
first selected token
final_hidden bits/hash
local layer KV bytes/hash
global layer KV bytes/hash
ring cursor / absolute positions
pending decode control
selected-token control
repetition/suppression state
```

Compare a Vision request run with MTP disabled and diagnostic MTP enabled up to the exact point before the first proposal. They must be identical.

### Stage D3 — forced proposals

Do not initially use real Assistant proposals.

Feed D2 with the exact next two Ordinary Target tokens from a frozen sequential run. This isolates:

```text
Target T3 verifier
KV speculative transaction
selection/commit
```

from Assistant quality and acceptance.

### Stage D4 — T3 versus sequential Target

For each forced group, compare:

```text
T3 row 0 ↔ Ordinary forward pending token
T3 row 1 ↔ next sequential Ordinary forward
T3 row 2 ↔ next sequential Ordinary forward
```

Capture the first divergence after every layer:

```text
input hidden
local/global attention output
shared MLP
routed MLP
layer output
final norm
head logits / selected token
```

Binary-search the first mismatching layer/operator.

### Stage D5 — cache transaction

Before and after every group compare:

```text
pre-group KV
tentative KV
accepted prefix
restored rejected suffix
committed position
next pending token
```

Include local circular-cache wrap and global cache.

### Stage D6 — real Assistant proposals

Only after forced proposals are exact:

- enable normal Assistant proposals;
- compare Assistant context inputs;
- verify accepted/rejected decisions;
- require final token identity.

### Stage D7 — matrix

Run at least:

```text
budget:          70 / 140 / 280
image placement: prompt start / middle / near chunk boundary
context:         short / >16K prepared boundary / local-ring boundary
session:         first turn / cached continuation
sampling:        greedy / fixed seed sampled
failure:         cancellation during Vision prefill and during D2
```

### Product enable gate

Enable Vision+D2 only when:

```text
same final tokens
same Target selections
same accepted/rejected decisions for frozen trajectory
same committed KV/position after every group
no fallback
no token-loop allocation
Vision tower executes once per uncached image, never per D2 group
```

---

# 5. Vision performance review

## 5.1 Evidence gap

The archive has no Vision stage timings or Nsight captures. Therefore all gain estimates below are hypotheses until V10 establishes a baseline.

The first profiler must emit at least:

```text
host decode
resize
patchify
host-to-device upload
patch projection
position add
per-layer:
  input norm/quant
  Q/K/V GEMMs
  Q/K/V norm+RoPE
  attention pass 1
  attention pass 2/value
  O GEMM/residual
  FFN norm/quant
  Gate/Up GEMMs
  GELU
  product quant
  Down GEMM/residual
pool
standardize
final norm/quant
projector
embedding insertion
text prefill
TTFT
```

Measure raw patch geometries associated with 70, 140 and 280 soft-token budgets.

---

## 5.2 Ranked optimization candidates

| Rank | Candidate | Expected local gain | Risk | Complexity | VRAM effect |
|---:|---|---|---|---|---|
| 1 | direct row-major 3×3 pooling | very high for pool kernel | low | low | neutral |
| 2 | precomputed 2D RoPE tables | high for Q/K norm+RoPE | low–medium | medium | small bounded workspace |
| 3 | multi-query tiled K/V-shared Vision attention | highest tower-wide potential | high | high | bounded shared memory, no score slab |
| 4 | fuse GELU product + BF16 round + FP8 quantization | moderate | low–medium | medium | can remove one transient pass |
| 5 | Vision CUDA graphs by fixed budget/grid | moderate launch reduction | medium | medium | graph-private bytes |
| 6 | grouped/fused QKV and Gate+Up projections | moderate | medium–high | high/artifact v2 possible | neutral or small |
| 7 | asynchronous double-buffered input staging | small | low | low–medium | one extra host/device input buffer |
| 8 | budget-sized workspaces | capacity gain, not compute | low | medium | saves ~53–80 MiB for smaller budgets |

---

## 5.3 Candidate P1 — direct 3×3 pooling

### Current implementation

For every:

```text
soft token × 1,152 channels
```

the kernel scans all raw tokens and checks their coordinates.

At 280 soft tokens and 2,520 raw patches:

```text
280 × 1,152 × 2,520
= 812,851,200 coordinate tests
```

Only nine raw patches contribute to each pooled token.

### Replacement

After exact row-major validation:

```text
pooled_x = pooled % pooled_width
pooled_y = pooled / pooled_width

raw base:
  x0 = pooled_x × 3
  y0 = pooled_y × 3

sum in the exact current order:
  (y0+0,x0+0), (y0+0,x0+1), (y0+0,x0+2),
  (y0+1,x0+0), ...
  (y0+2,x0+2)
```

Preserve:

```text
sum / 9
BF16 rounding
sqrt(1152) in FP32
subtract BF16 bias
multiply BF16 scale
BF16 rounding
```

This changes O(soft × hidden × raw) to O(soft × hidden × 9).

### Gate

- byte-identical pooled/standardized output;
- no change in final image embeddings;
- no additional allocation;
- same-binary A/B for 70/140/280;
- retain only on full Vision-tower/TTFT gain, not microkernel gain alone.

This is the best first performance candidate after semantic closure.

---

## 5.4 Candidate P2 — precompute 2D RoPE

### Current implementation

`HeadNormRopeKernel` evaluates for Q and K, in every one of 27 layers:

```text
powf(...)
cosf(...)
sinf(...)
```

for head channels and patch positions.

The same position/frequency values are reused across:

```text
all 16 heads
Q and K
all 27 layers
```

### Replacement

Create a bounded per-image table after positions upload:

```text
x_cos[x][18]
x_sin[x][18]
y_cos[y][18]
y_sin[y][18]
```

or a row-wise equivalent if that is faster.

Match the oracle boundary by rounding cosine and sine to BF16 exactly once, as the current kernel does.

Then fuse Q/K norm and table-based RoPE, or at least replace transcendentals with loads.

### Gate

- exhaustive coordinate/frequency comparison;
- Q/K output bit identity;
- head-72 tail exact;
- table bytes reported;
- no runtime allocation;
- NCU reduction in special-function instructions;
- full tower and TTFT gain.

---

## 5.5 Candidate P3 — tiled bidirectional Vision attention

### Current implementation

One warp owns one query/head row. It:

1. scans every source token to compute running max and denominator;
2. scans every source token again;
3. recomputes every Q·K score;
4. rounds probability to BF16;
5. accumulates probability×V.

K and V are reread independently for each query.

At the largest input, this is the likely dominant tower cost, but V10 must prove it.

### First optimized design

Keep the exact two-pass algorithm and per-query source order.

A CTA handles:

```text
one head
Q query rows: 4 or 8
K/V tile:     32 or 64 source rows
```

Example Q4/K32:

```text
128 threads = 4 query warps

Shared:
  K[32][72] BF16
  V[32][72] BF16
  approximately 9 KiB total

Each warp:
  owns one query
  walks sources in exactly ascending order
  uses shared K/V loaded once for four queries
```

Q8/K32 uses 256 threads and approximately the same K/V shared tile, with more reuse.

Do not initially change:

- Q/K dot-product order within a query;
- WarpSum topology;
- two-pass softmax;
- FP32 max/denominator;
- BF16 probability boundary;
- V accumulation order.

That gives a plausible route to bit identity while reducing K/V global traffic by roughly the query-group reuse factor.

### Later candidates only after the first design

- `cp.async` double buffering;
- larger K tiles;
- persistent head/query scheduling;
- online one-pass variants.

A one-pass online softmax is not an initial candidate because it would change the normative BF16-probability boundary and reduction order.

### Gate

- head-level exhaustive oracle for small and maximum shapes;
- layer 0/13/26 exact or predeclared tight boundary;
- no score slab;
- no spills;
- NCU memory/stall/occupancy data;
- full-tower and TTFT gain;
- image task sanity.

---

## 5.6 Candidate P4 — fuse GELU and product quantization

Current path:

```text
Gate GEMM -> BF16
Up GEMM   -> BF16
GELU Gate×Up kernel -> BF16 product
separate BF16 -> FP8 quantization
Down GEMM
```

Replacement:

```text
read BF16 Gate/Up
GELU-tanh Gate×Up
explicit BF16 product rounding
reduce amax
quantize rounded BF16 product to E4M3
```

The exact physical BF16 product boundary must remain.

This removes one full intermediate read/write and one launch.

---

## 5.7 Candidate P5 — graph capture

The Vision tower has a fixed 27-layer structure. After correctness and kernel stabilization, capture one graph per admitted physical image geometry or, preferably, per product budget if the padded/masked contract is adopted.

Do not create an unbounded graph cache.

Allowed v1 graph set:

```text
70
140
280
```

or a bounded set of validated raw-grid shapes.

Report graph-private device bytes and include them in context admission.

---

## 5.8 Budget-sized workspace

The current runtime always allocates the maximum workspace:

```text
119,993,600 bytes
114.43 MiB
```

Static layout estimates for smaller fixed budgets are:

| Soft-token budget | Maximum raw rows | Estimated workspace |
|---:|---:|---:|
| 70 | 630 | 36,290,048 B = 34.61 MiB |
| 140 | 1,260 | 64,191,488 B = 61.22 MiB |
| 280 | 2,520 | 119,993,600 B = 114.43 MiB |

Potential capacity savings relative to 280:

```text
70:  ~79.8 MiB
140: ~53.2 MiB
```

For a 16 GB GPU, this is worth exposing as a **startup/profile setting**, not an allocation changed per request. Changing the budget may require restarting/rebinding the server.

---

# 6. Runtime and API product gaps

## 6.1 Composite profile identity

Current runtime values remain largely text-artifact identities even when Vision is loaded.

Add explicit composite identity:

```text
profile_id:
  gemma4_26b_trellis35_vision_fp8

text_artifact_profile:
  gem16-trellis35-w4a8-v1

vision_artifact_profile:
  gemma4_26b_vision_fp8_v1

experimental:
  true
```

Do not infer this solely from file presence.

## 6.2 Capability matrix

`supports_vision()` and `supports_mtp()` are currently independent booleans. This can advertise both Vision and MTP even though their combination is intentionally unsupported.

Add:

```text
vision_module_loaded
supports_vision
vision_mtp_supported
supports_multiple_images
maximum_images
vision_soft_token_budgets
selected_vision_soft_token_budget
vision_weight_bytes
vision_workspace_bytes
vision_context_limit
profile_qualification
```

For the current branch:

```text
supports_vision = true
vision_mtp_supported = false
maximum_images = 1
experimental = true
```

Only V14 may set `vision_mtp_supported = true`.

## 6.3 Context values must not be inherited

Current generic 26B methods still expose text-only values such as:

```text
base_max_context = 98,304
qualified_64k = true
```

The Vision profile has not qualified those limits.

Introduce profile-specific:

```text
max_context_tokens
base_max_context_tokens
mtp_max_context_tokens
vision_max_context_tokens
required_reserve_bytes
```

Until V19, publish only measured values or an explicit unqualified/unknown status.

## 6.4 Error codes

Current unsupported-feature mapping can still emit a `gemma4_26b_text_only` code even for a Vision profile whose unsupported feature is specifically Vision+D2 or multiple images.

Use explicit codes:

```text
vision_module_not_loaded
vision_profile_required
vision_multiple_images_unsupported
vision_mtp_unqualified
vision_budget_unsupported
vision_context_unqualified
vision_artifact_mismatch
```

## 6.5 Metrics

Expose separate timings:

```text
image_decode_ms
image_resize_patchify_ms
vision_upload_ms
vision_tower_ms
vision_pool_project_ms
text_prefill_ms
ttft_ms
decode_ms
```

Counters:

```text
vision_requests
vision_failures
vision_d2_rejections
vision_budget_70/140/280
vision_artifact_validation_failures
```

Never fold the Vision tower into generic text-prefill throughput without also reporting it separately.

---

# 7. Native Studio review

## 7.1 Current limitations

Native Studio currently has:

```cpp
enum class ModelProfile {
  kGemma4Unified12B,
  kGemma4Moe26BA4B
};
```

`ServerConfig` has Target and Assistant directories but no Vision directory.

The catalog/profile model contains only:

```text
target
assistant
```

The manager stores a fixed array of two profiles and uses ternary indexing.

`BuildServerCommand` does not add `--vision-model`.

`HealthSnapshot` has no Vision fields.

The chat UI explicitly rejects image/audio attachments whenever the selected profile is 26B.

The media and OpenAI request plumbing already support image bytes and previews, so the missing work is primarily profile/component/capability integration rather than a new image widget.

## 7.2 Recommended profile model

Add an explicit third profile:

```cpp
kGemma4Moe26BTrellis35VisionFp8
```

Do not silently convert the existing qualified 26B profile into the Vision profile.

Suggested labels:

```text
Gemma 4 26B A4B – Fast NVFP4
Gemma 4 26B A4B – Compact Trellis35 + Vision
```

If the Trellis text-only profile is also exposed separately, make it a fourth explicit catalog entry rather than a hidden mode.

## 7.3 Component model

Replace fixed `target`/`assistant` assumptions with a small bounded component list:

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

A profile then declares:

```text
12B:
  Target required
  Assistant according to existing product contract

26B NVFP4:
  Target required
  Assistant required when MTP selected

26B Trellis Vision:
  Trellis Target required
  Vision required
  Assistant optional until Vision+D2 qualified
```

Do not duplicate Hub blobs. Build deterministic component views from verified blobs/locks.

## 7.4 Settings and server command

Add:

```text
vision_directory
vision_soft_token_budget = 70|140|280
enable_vision_mtp
```

Persist them with the profile.

`BuildServerCommand` must append:

```text
--vision-model <directory>
```

only for the explicit Vision profile.

While `vision_mtp_supported == false`:

- force `mtp_draft_tokens = 0` for image requests;
- preferably disable the MTP toggle with a clear explanation;
- do not silently run Ordinary if the user explicitly selected D2.

After V14, enable D2 only when live health confirms support.

## 7.5 Attachment gating

For the explicit Vision profile:

```text
image: allowed
maximum images: 1
audio: rejected
document: existing text extraction only
```

Before upload/generation:

- validate image type and byte limit;
- show selected 70/140/280 budget;
- show estimated image-token use;
- reject a second image before server work;
- preserve preview and remove/retry controls.

## 7.6 Health matching

Studio should not trust local profile selection alone. It must compare the running server's health:

```text
profile_id
supports_vision
vision_module_loaded
vision_mtp_supported
vision_soft_token_budget
max_images
context limit
experimental
```

If an external server lacks the expected module or profile, image send must remain disabled.

## 7.7 Disclosure

The model card in Studio should show:

```text
Experimental
One image per conversation
FP8 Vision sidecar
Trellis35 text weights
Vision budget 70/140/280
Vision+D2 status
qualified context limit
```

Do not label this as the same qualified 26B NVFP4 profile.

---

# 8. Security and integrity observations

## 8.1 Image identity

The image structs currently store a 64-bit source fingerprint. If that fingerprint is used to decide whether a resident image representation can be reused, collision resistance matters.

For product hardening, use:

```text
SHA-256 of original encoded bytes
```

or a strong truncated digest plus byte length, and keep the original filename out of identity.

Because this code is shared with 12B, make this a separate hardening change with regressions.

## 8.2 Position contract

Canonical row-major position validation also prevents malicious duplicate-coordinate inputs from forcing pathological pooling semantics.

## 8.3 Sidecar integrity

Require all four components:

```text
vision.gem16
gem16_vision.json
vision_compilation.json
vision.lock.json
```

and verify:

- Target profile binding;
- source revision;
- full tensor inventory;
- payload bounds/overlaps;
- alignment;
- exact file hashes according to selected integrity mode;
- no symlink escape;
- no unindexed payload region except explicit aligned gaps.

---

# 9. Testing and evidence gaps

Current archive evidence is insufficient for product promotion because it lacks:

- a dedicated CUDA Vision CTest target;
- real image fixtures;
- per-stage Vision timings;
- NCU/NSYS reports;
- Vision output comparison against the pinned oracle;
- text-side local image-mask tests;
- chunk-boundary tests;
- Vision+D2 mismatch trace;
- Vision+Assistant capacity matrix;
- context qualification;
- Studio integration tests;
- Windows/Linux product smoke;
- a current artifact diagnostic reflecting the now-implemented runtime.

Create:

```text
gem16-cuda-vision26b-tests
gem16-26b-vision-product
gem16-26b-vision-d2-diagnostic
gem16-26b-vision-benchmark
```

and store immutable compact evidence plus raw captures outside the source archive as usual.

---

# 10. Prioritized work packages

## V09 — Ordinary semantic closure

**Goal:** make Ordinary multimodal prefill match the pinned model contract.

Tasks:

1. Add image-range support to the 26B local prefill attention wrapper.
2. Keep global layers causal.
3. Add an image-span-aware chunk planner.
4. Require canonical row-major positions.
5. Prove padded/masked versus unpadded equivalence.
6. Add exact text-only regression hashes.
7. Keep D2 fail-closed.

Acceptance:

```text
pinned mask semantics
all boundary offsets
no split image span
selected ordinary image outputs deterministic
no text-only change
no allocation delta
```

## V10 — Vision baseline and qualification harness

**Goal:** create the evidence needed to optimize or promote.

Tasks:

- dedicated CUDA tests;
- frozen 70/140/280 fixtures;
- stage timers and NVTX;
- NSYS/NCU;
- peak VRAM;
- context matrix;
- regenerate the stale V01/V02 artifact diagnostic.

No optimization is accepted without V10.

## V11 — D2 exactness laboratory

Follow the D0–D7 sequence above.

Product remains fail-closed.

## V12 — low-risk performance wave

In this order:

1. direct 3×3 pooling;
2. precomputed 2D RoPE;
3. GELU+product-quant fusion;
4. asynchronous input staging if profiler-relevant;
5. budget-sized workspaces.

Each candidate gets a same-binary rollback.

## V13 — tiled Vision attention

Implement Q4/Q8 K/V-shared two-pass attention while retaining exact source order and BF16 probability semantics.

This is the likely largest speedup.

## V14 — Vision+D2 product enablement

Only after V11 exactness passes:

- remove the user-visible fail-close for the qualified combination;
- ensure tower runs once;
- 3W10 Ordinary/D2 image benchmark;
- no fallback;
- capability bit `vision_mtp_supported=true`.

## V15 — runtime/server product contract

- composite profile identity;
- capability matrix;
- profile-specific context values;
- explicit errors;
- metrics;
- one-image early rejection.

## V16 — publication and catalog

- publish/pin the Trellis Target if not already a product component;
- publish/pin Vision sidecar;
- generated component catalog;
- no duplicate blobs;
- license/notice/provenance;
- anonymous or authenticated download contract as appropriate.

## V17 — Native Studio profile/component integration

- third explicit profile;
- generalized bounded component list;
- settings;
- `--vision-model`;
- install/verify/remove;
- health matching.

## V18 — Native Studio chat UX and D2

- image attachment allowed only for live-compatible Vision profile;
- one-image limit;
- budget UI;
- D2 disabled or enabled from live capability;
- experimental disclosure;
- restart/recovery behavior.

## V19 — capacity, quality and cross-platform product gates

- 32K/64K and measured upper boundaries;
- Target+Vision and Target+Vision+Assistant;
- reserve and first rejection;
- bounded image suite;
- Windows/Linux clean machine;
- cancellation/streaming/session continuation.

## V20 — experimental product freeze

Freeze only after all mandatory gates.

The claim should initially be:

```text
Experimental Gemma 4 26B Trellis35 + FP8 Vision,
one image, batch one, measured context limit,
bounded image-quality evidence.
```

Not a general production-quality multimodal claim.

---

# 11. Stop and rollback rules

Stop and report rather than relaxing semantics if:

- the correct text image mask changes expected outputs;
- padded/unpadded equivalence does not hold;
- D2 cannot reproduce sequential Target state;
- an optimization changes the BF16 probability or residual boundary;
- Vision requires a second persistent Weight representation;
- a context limit is inherited rather than measured;
- Studio would need to silently disable a selected feature;
- a performance candidate wins only by changing image budget or image geometry.

Every performance candidate must be independently reversible and preserve the reference path until the final freeze.

---

# 12. Final recommendation

The project should not move directly from the current branch to UI polish. The most efficient route to a real product is:

```text
first:
  V09 semantics
  V10 evidence
  V11 D2 exactness

then:
  V12/V13 performance
  V14 D2 enablement

then:
  V15/V16 API and publication
  V17/V18 Studio
  V19/V20 qualification and freeze
```

The current Vision implementation is a strong vertical slice, but the text-side image mask and chunking are more important than any kernel optimization. Once those are corrected, the largest likely performance wins are direct pooling, precomputed RoPE, and especially tiled K/V-reuse attention.

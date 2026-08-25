# Gemma 4 26B A4B maximum-performance execution plan

Status: active engineering plan after implementation commit `4b7c2f3`
Primary hardware: NVIDIA GeForce RTX 5080 Laptop GPU, 16 GB, native `sm_120a` build
Primary promotion row: `wikipedia-real-16k64-greedy`
Plan inputs: `CODEX_GEMMA4_26B_MAX_PERFORMANCE_PLAN.md`, `chatgptpro_plan.md`, current source, accepted evidence, and active project policy

## 1. Purpose and authority

This document consolidates the two German maximum-performance proposals into one executable English plan. It is a
work plan, not a new owner decision. Conflicts are resolved in this order:

1. repository `AGENTS.md` and permanent runtime/integrity rules;
2. `docs/ACTIVE_DECISIONS.md`;
3. current source, tests, and accepted evidence;
4. `ACTIVE_CONTRACT.md`, `FAST_TRACK_STATUS.json`, and the active milestone;
5. this plan;
6. the two source proposals and historical planning documents.

The program objective is the fastest correct engine that can be sustained on the target hardware, not merely a
threshold pass. Formal M20 still uses at least **6,000 prompt token/s** and **150 ordinary decode token/s** on the
exact canonical row, with **6,500 prompt token/s** reported as its competitive stretch. The current candidate has
already passed both prompt thresholds. M20 therefore closes once the 150 ordinary-decode result is reached and the
existing formal qualification protocol confirms the complete row. Performance work then continues as a clearly
labelled post-M20 maximum-performance campaign toward 155/160/165+ ordinary decode and 6,750/7,000/7,250/7,500+
prefill, unless the owner freezes or reprioritizes the track. M20 closure is a milestone boundary, not a stop rule.

The maximum-performance campaign stops only when at least one of these conditions is true:

1. profiling shows that the relevant path is near the practical hardware limit;
2. several independently implemented variants fail to deliver a reproducible gain;
3. the remaining gain requires an unaccepted quality, semantic, memory, or product trade-off;
4. the owner explicitly freezes, stops, or reprioritizes the track.

## 2. Current authoritative baseline

Implementation baseline:

- branch: `feat/gemma4-26b`;
- commit: `4b7c2f3` (`Promote SM120 tensor-core prefill router`);
- integrated 26B prefill router: `sm120_bf16_tensor_core`;
- explicit rollback: `serial_exact` selected before execution;
- one fully resident target-weight representation;
- fixed-address FP8 K/V and workspaces;
- whole-model ordinary decode CUDA Graph;
- no recurring inference allocation, fallback, prompt cache, MTP, or CPU weight offload.

Adjacent development measurements:

| Metric | Current mean | M20 gate | Status |
|---|---:|---:|---|
| Prompt throughput | 6,574.164 token/s | 6,000 token/s | pass |
| Prompt stretch | 6,574.164 token/s | 6,500 token/s | pass |
| Ordinary decode | 139.054 token/s | 150 token/s | open |
| Ordinary token latency | 7.191 ms | 6.667 ms | save about 0.525 ms/token |

The current Tensor-Core-router output-token SHA-256 is
`c750d0b33f8eb4a8103299875886e51ab144d874cafe8cea77b0cfd99d2aedaf`. The explicit serial-router rollback produces
`d8f0aa61ba66750bda8582861d3f42ef7022917e75b6abd3b6fac4e66ef75e89`. Exact candidates relative to the current
default must preserve `c750d0…`; numerically bounded candidates must declare a new hash, retain a rollback, and pass
the applicable numerical gate.

## 3. Non-negotiable invariants

- Preserve the 12B production path and run its regression whenever shared runtime/CUDA code changes.
- Keep target weights fully GPU-resident. CPU offload and expert streaming are diagnostics only.
- Keep one persistent device representation per weight and retain physical tied-weight aliasing.
- Do not allocate, read files, JIT-compile, repack weights, or grow containers in recurring inference.
- Keep FP8 K and V caches physically distinct. `attention_k_eq_v` permits reuse or aliasing of the **raw projection
  scratch only**; learned K normalization plus RoPE and V normalization produce distinct cache values.
- Keep the exact prompt, 64 output forwards, greedy sampling, FP8 KV, batch size, cache state, and timing boundaries.
- Do not use MTP, prefix caching, context reduction, precision substitution, or altered synchronization boundaries to
  satisfy the ordinary M20 gate.
- Do not enable global `--use_fast_math`. Approximate operations require a kernel-local, explicitly bounded candidate.
- Keep every risky path selectable before graph capture, visibly reported, and paired with an explicit rollback.
- Prefer static graph variants over a recurring branch in the token hot path.
- Preserve deterministic expert-slot ordering and top-k tie-breaking unless a numerically bounded candidate explicitly
  owns the difference.
- Retain separate 32K/64K and base/MTP memory gates. No speedup may silently consume the required safety margin.
- Do not silently reduce requested context when enabling sampling, MTP, or another product mode.
- Do not perform generic cleanup unless it is required by the isolated performance mechanism under test.

Bounded CPU or CPU/GPU/disk-offload execution of the pinned 49 GB QAT-BF16 checkpoint is allowed for small Golden
Gates and targeted numerical diagnosis under an explicit memory budget. It is performance-ineligible. Broad M19
quality runs remain deferred.

## 4. Experiment and evidence protocol

### 4.1 Isolation

Change one performance mechanism per candidate. A candidate record must name:

- base commit and dirty/untracked preservation;
- hypothesis and expected latency/kernel effect;
- files changed;
- selector and rollback;
- exact versus numerically bounded class;
- build, operator, sanitizer, and end-to-end commands;
- raw samples and compact hashes;
- output hash and numerical metrics;
- dispatch/instruction facts;
- registers, shared memory, spills, occupancy, and relevant bandwidth;
- peak VRAM, free margin, fallback count, and allocation count;
- promotion or rejection with reason.

Do not overwrite older evidence. Raw reports remain under ignored `artifacts/raw/`; tracked evidence is compact and
hash-addressed.

### 4.2 Screening versus formal promotion

Screening is intentionally bounded:

1. one complete warm-up;
2. at least two adjacent retained runs for a large, stable effect;
3. three alternating parent/candidate A/B pairs for small or thermally sensitive effects;
4. an operator/kernel measurement plus the full canonical engine row;
5. clocks, power state, thermals, driver, binary hash, and model hash recorded when they could explain the result.

A candidate normally advances only if it gives at least one of:

- reproducible full-engine improvement of at least 0.5%;
- at least 20 microseconds/token decode reduction;
- a necessary memory reduction for an approved context/MTP profile;
- a clearly measured kernel win whose full-engine contribution is temporarily masked by a known adjacent bottleneck.

A full candidate promotion should normally deliver at least 1% retained median improvement. Smaller gains may be
combined only after each mechanism has been independently proven and documented. Substantial candidates receive an
independent read-only review before commit.

Formal M20 promotion runs exactly three warm-ups and ten retained measurements and uses the median. That expensive
protocol runs on the final frozen candidate, not on every experiment. It also records distribution, power, clocks,
thermals, peak VRAM, dispatch, and matching M21 evidence.

### 4.3 Profilers

- Use Nsight Systems for Graph nodes, CPU gaps, copy nodes, synchronization, stream overlap, and kernel attribution.
- Use Nsight Compute only on isolated representative kernels for DRAM/L2 throughput, Tensor-Core activity, stalls,
  occupancy, registers, shared memory, and spills.
- Never publish a profiled run as an absolute throughput result.

## 5. Correctness classes

### E — exact

Required for scheduling, aliasing, launch, and fusion changes that preserve arithmetic:

- current default output hash `c750d0…`;
- bitwise operator boundaries where a direct oracle exists;
- unchanged tie-breaking and expert-slot order;
- unchanged finite-error behavior;
- relevant boundary and ring-wrap tests;
- protected 12B regression for shared code.

### N — numerically bounded

Required for Tensor-Core reassociation, reduced-precision staging, selected-logit softmax, KV4, and similar changes:

- explicit exact rollback;
- max-absolute, RMS, relative-L2, and cosine metrics at operator/layer boundaries;
- attention score/LSE and layer-output checks;
- layers 0/5/6/29 and boundary positions relevant to the changed operator;
- both physical global KV heads and local ring-wrap cases;
- Top-1/Top-5 stability, teacher-forced KL/NLL, and bounded generation;
- the existing QAT-BF16 Golden Gate when routing or model-wide numerics change;
- memcheck, racecheck, and initcheck.

Tolerances are not weakened to rescue a candidate.

### S — lossless sampling/speculation

- ordinary and speculative execution emit the same target tokens under matched deterministic controls;
- sampling retains seed, RNG-step, penalty, suppression, and stop semantics;
- tentative KV, hidden state, position, ring state, RNG, and repetition state commit transactionally;
- only target-verified tokens become visible output.

## 6. Known completed work and dead ends

Do not repeat these as new ideas:

- native `sm_120a` compilation is already active;
- physical-BF16 prefill intermediates and the 1,024-token chunk are already promoted;
- prepared global prefill K/V staging through 16K is already promoted;
- grouped-expert activation staging, K64 double buffering, and two-row expert work are already promoted;
- the BF16 Tensor-Core prefill router is already the integrated default and passed its bounded Golden Gate;
- global fast math is rejected;
- a shared 256-token global softmax changes semantics and is not the proposed two-independent-split design;
- eight NVFP4 warps per block were slower;
- the historical fast W13 two-row artifact had a wrong/degenerate output and is not evidence; only the idea may be
  independently reimplemented against current semantics;
- packing four Prediction D2H values alone is not a primary hypothesis; revisit the host tail only together with
  control/self-feed and Graph-node removal;
- CPU weight offload, expert streaming, prompt-cache accounting, and MTP cannot satisfy ordinary M20;
- broad prefill Graph capture is not reopened without fresh launch-dominance evidence.

## 7. Mandatory path to the open M20 decode gate

### W0 — exact prefill tail requested by the owner

#### P01: finalize only the last prompt chunk

Status: retained on 2026-08-25 as an exact safety and redundant-work cleanup; compact evidence is
`../../../artifacts/m20/optimization-final-chunk-only.json`.

Before P01, `PrefillTokens()` performed layer captures, final RMSNorm, tied head projection, softcap, and argmax for every
1,024-token chunk, although only the final prompt position is externally visible.

Implement `is_last_chunk = consumed + chunk == tokens.size()` and execute only on the last chunk:

- the layer/router captures that are intended to represent the final prompt token;
- final RMSNorm;
- output head;
- softcap/argmax.

Keep KV writes, position/ring accounting, router-finite checking, and all state required by the next chunk unchanged.
Require the same final prediction, captures, output hash, finite behavior, memory facts, and fewer launches/head-weight
reads. This is the first implementation slice after this plan.

#### P02: remove chunk staging barriers safely

Status: P02b was measured separately on P01, was neutral at +0.08%, and was fully reverted. Do not retry a staging
variant without fresh profiler evidence of a material host gap.

Do not delete `cudaStreamSynchronize` while reusing one pinned host slot. Test the simplest safe design first:

- allocate a context-sized pinned token staging region at engine creation;
- copy the complete request into disjoint pinned addresses before enqueueing;
- keep the existing single device token buffer and same-stream H2D/compute order;
- enqueue each chunk H2D from a disjoint host range;
- perform no allocation inside `PrefillTokens()`.

Alternative only if needed: two pinned slots plus per-slot copy-completion events. A copy stream and two device token
buffers are later candidates only if profiling predicts useful H2D/compute overlap. Require hash identity, host-buffer
lifetime proof, initcheck/racecheck, and no recurring allocation.

P01 and P02 are isolated candidates and are measured separately.

### W1 — fresh decode cost model

#### D00: profile the current `7f745f5` candidate

Status: complete on 2026-08-25. Compact attribution is
`artifacts/m20/decode-attribution-d00.json`. The 16K Graph contains 697 kernel, 17 D2D-copy and 2 memset nodes.
Global split plus merge accounts for 635.988 instrumented microseconds per token and is the largest
context-dependent exact target. The special KVH2 kernel uses 56 registers/thread, 12,352 bytes static shared memory,
66.67% theoretical occupancy and 51.67% achieved occupancy. D03a is selected next under a strict four-block
register-residency and zero-spill boundary.

Build a current microsecond/token table for:

- 25 local and 5 global attention layers;
- Q/K projection, K/V normalization, quantization, and cache append;
- split attention and merge;
- shared MoE, router projection/top-8, routed W13, activation, W2, and reduction;
- head plus softcap/argmax;
- diagnostic copies, control nodes, Graph replay, and host tail.

Use short 512 and 2K contexts to estimate constant cost and 8K/16K to isolate long-context cost. Run 32K/64K only
when the active context milestone or a specific long-context hypothesis requires them. Record Graph-node counts and
the five largest kernel groups with registers/shared/occupancy.

Exit: a checked-in compact attribution that selects the next decode candidate. The current evidence suggests global
16K attention is the leading context-dependent cost, but the fresh profile decides.

### W2 — exact global-attention scheduling

#### D03a: two independent 128-token splits per CTA

Status: rejected and fully reverted on 2026-08-25. Two exact end-to-end samples regressed from the 138.746-token/s
parent mean to 137.489 token/s. The special global kernel grew from 56 to 80 registers/thread and from 595.574 to
691.700 instrumented microseconds despite reducing the 16K grid from 258 to 130 CTAs. D03b is therefore skipped.
Negative evidence is `artifacts/m20/optimization-global-paired-splits-rejected.json`; do not retry sequential split
pairing without a different live-range design and fresh occupancy proof.

In `SplitOnlineDecodeAttentionFp8GlobalKvh2Kernel`, map one CTA to two adjacent existing 128-token splits while
preserving:

- two independent online-softmax states;
- two independent score/partial-output/LSE destinations;
- the existing FMA and merge order inside each split;
- the same physical KV-head mapping and tail behavior.

Share only query loading, CTA setup, invariant scale/address work, and resources proven safe. This is not a 256-token
softmax. Add operator coverage at 127/128/129, 255/256/257, final-tail positions, both global KV heads, and full 16K.
Expose a pre-capture selector during development; remove or freeze it only after promotion. Record CTA count, query
bytes, registers, shared memory, spills, occupancy, kernel time, merge time, hash, and full-engine delta.

#### D03b: four independent splits per CTA

Test only if D03a wins and occupancy/register/shared-memory evidence leaves room. Reject long-running CTAs that reduce
wave utilization.

#### D04: global K/V ping-pong staging

Test separately from D03:

1. preload tile 0;
2. compute tile `i` while loading tile `i+1` into the alternate shared buffer;
3. preserve the exact QK, softmax, and PV order;
4. compare K-only, V-only, and combined double buffering;
5. compare cache policy and prefetch distance only after the basic pipeline wins.

Reject on spills, harmful occupancy loss, or less than the screening threshold.

### W3 — exact Graph and small-kernel reduction

#### D01: alias raw K/V projection scratch, never final caches

For layers with `reuses_raw_k_for_v`, allow `value_raw == key_raw` through the raw-projection handoff and remove the
K-to-V D2D copy. K normalization/RoPE and V normalization must still write distinct destinations, and FP8 K/V caches
remain physically distinct. Update validation and workspace lifetime proof accordingly.

Follow-on D01b may fuse K/V normalization, quantization, and distinct cache writes only if it reproduces the current
rounding order exactly. Measure removed Graph nodes, workspace bytes, and decode/prefill effect.

#### D02: separate production and diagnostic decode graphs

Capture static variants before execution:

- production graph without layer hidden/router/top-ID captures;
- diagnostic graph retaining the existing captures.

Evaluate separately, not as a bundle:

- sticky finite flags initialized once per request and cleared only on failure;
- combined local/global RoPE preparation;
- product-only direct expert reduction without a materialized contribution matrix;
- omission of diagnostic buffers from production memory profiles.

#### D05: exact Router Top-8

First preserve the current probability calculation and renormalization. Compare:

- current eight selection rounds with 128 threads;
- four warp-local 32-element sorts plus a deterministic 32-candidate merge;
- one block-wide deterministic sort.

Tie-breaking remains higher value first, then lower expert ID. Only after an exact winner exists may an N-class
candidate select Top-8 directly from logits, exponentiate only winners, and omit full probabilities in production.

### W4 — MoE, head, and host-tail completion

#### D09: M=1 NVFP4 MoE

Profile routed/shared Gate-Up and Down separately. Reimplement current-semantics candidates one at a time:

- reuse one activation/scale load across multiple output-row tiles;
- tune Down projection similarly while preserving BF16 epilogue and slot reduction order;
- compare overlap schedules for shared and routed branches;
- avoid materializing expert contributions only in a separately selected production graph;
- consider weight-stationary/persistent CTA work only after simpler candidates.

Do not use the invalid historical W13 result as a baseline. Require current hash identity for E-class scheduling.

#### D10: fused greedy head candidates

For ordinary greedy, test an NVFP4 head epilogue that emits per-CTA finite candidate `(rounded_logit, token_id)`
without writing the complete 262,144-float logit buffer. Preserve the current BF16 projection rounding, softcap
evaluation, and token-ID tie-break. Keep the full-logit path for sampling, suppression, `CopyLogits()`, and diagnosis.

Measure head traffic, candidate reduction, Graph nodes, and full-engine effect. Do not assume argmax-before-softcap is
bit-identical merely because softcap is monotonic.

#### D11: device self-feed and compact host status

Treat this as one coherent Graph/control experiment rather than repeating a copy-only microchange:

- feed the selected device token directly to the next embedding lookup;
- update `DecodeControl` on device at Graph end;
- remove per-replay control H2D when safe;
- write token, logit, and sticky error/stop state to one initialized pinned host status;
- keep exactly one externally required event/wait per streamed token.

A later hostless multi-token loop changes the synchronization/timing boundary and must be reported as a separate
maximum-throughput mode; it cannot retroactively satisfy ordinary M20.

### W5 — numerically bounded attention only if exact work is insufficient

#### D06: global Tensor-Core QK

Extract/reuse the proven SM120 BF16 MMA and shared-layout helpers. Build QK only first:

- stage FP8 K and convert to BF16;
- provide Q in the matching BF16 layout;
- compute a `16 tokens x 8 query heads` score tile over dimension 512;
- retain current softmax, LSE, merge, and FP32 PV initially;
- keep scalar exact rollback.

This is N-class because BF16 MMA changes accumulation. Gate scores, LSE, partial/final attention output, layer output,
Top-k, teacher-forced metrics, generation, sanitizers, and end-to-end speed.

#### D07: Tensor-Core PV and online fusion

Proceed only after a winning D06 QK-only candidate. Test PV separately before combined online QK/PV fusion. More
aggressive hierarchical merge, FP8 MMA, or changed online reduction order remain separate N-class experiments.

#### D08: local attention

Tune local 1,024-window attention only after global work. Consider independent split sharing, ping-pong K/V, contiguous
versus wrapped-ring specializations, and finally QK/PV MMA. Cover 1,023/1,024/1,025 and later wrap positions.

## 8. Additional prefill work after P01/P02

Prefill already passes both M20 prompt gates, so these candidates must not delay the ordinary 150 decode gate unless
they are small exact wins or needed for memory/context.

### P03: hoist RoPE preparation

Prepare the local and global RoPE tables once per prompt chunk instead of once per layer. Use the existing exact table
kernel and fixed workspace regions; combine both profiles in one kernel only as a follow-up candidate.

### P04: raw K/V scratch reuse and distinct cache-write fusion

Apply the same raw scratch rule as D01. Do not alias final caches. Evaluate exact fused normalization/quantization and
distinct FP8 cache writes after the raw-copy removal is independently measured.

### P05: chunk planning, not a repeat of physical BF16

Physical BF16 and 1,024-token chunks are complete. Only sweep 512/768/1,024/1,536/2,048 after memory and profile
preflight. A larger chunk must preserve the 192 MiB prefill cap or receive a new explicit contract; it must not reduce
the 64K reserve silently.

### P06: grouped-MoE scheduling

Record tokens-per-expert and padding per layer/chunk, then test deterministic shape buckets, GPU task lists, and
persistent work queues. Reuse activation/scale tiles across output rows, and consider TMA only after the existing
`cp.async` pipeline is profiled. Preserve stable assignment and slot reduction order.

### P07: Tensor-Core router epilogue fusion

Sweep token tile/pipeline shapes and then test projection plus local Top-8/assignment epilogue. Omitting full
probabilities is N-class and retains the serial exact rollback plus the current QAT-BF16 Golden Gate.

### P08/P09: attention and boundary fusion

Only after fresh profiles:

- tune local/global query and key tiles, ping-pong staging, and cache policy;
- consider TMA/cluster/DSM only as isolated Blackwell candidates;
- fuse norm/quant/residual/projection boundaries only while preserving documented BF16 rounding points;
- preserve the prepared-global path where it is faster and valid.

Prefix caching is a separate product feature and never a raw-prefill result.

## 9. Post-M20 product-performance tracks

These tracks are intentionally subordinate to the open ordinary decode gate.

### Sampling S00-S05

1. Freeze Golden vectors for greedy, suppression, temperature, top-k, top-p, min-p, penalties, seeds, EOS, ties,
   NaN, and Inf.
2. Capture sampling-specific graphs so `Prediction()` is not called before device sampling.
3. Build deterministic top-k fast paths for `k <= 8/32/64` without a full vocabulary sort.
4. Retain a full-sort oracle and exact fallback for unrestricted top-p/min-p.
5. Allow the head to emit candidates directly only for bounded top-k configurations.
6. Qualify same-seed output, no allocations/fallback, and separately reported sampling throughput.

### MTP M00-M07

1. Lock and inventory the official 26B assistant without executing repository code.
2. Implement BF16 as the oracle, then compare FP8/NVFP4/hybrid assistant formats by acceptance and effective speed.
3. Build target verification for T=2/3/5 with real multi-row attention, grouped MoE, and head reuse.
4. Reuse the proven 12B transactional-state architecture for KV, hidden, position, RNG, repetition, and stop state.
5. Capture fixed ordinary/D1/D2/D4 graphs and select depth using measured device-side acceptance/cost with hysteresis.
6. Qualify greedy identity first, sampled lossless speculation second.
7. Promote MTP only if it gives at least 10% retained effective-throughput gain and passes its independent context
   memory gates. MTP never satisfies the ordinary 150 gate.

### Context C00-C04

- Use fixed initialization profiles such as speed-16K, balanced-64K, and later measured long/MTP profiles.
- Build a workspace lifetime matrix and alias only proven disjoint regions.
- Keep separate base and MTP maximum-context claims.
- Test global K8/V4 before K4/V4 only after stable attention and under an N-class quality gate.
- Tune long-context split/merge/pipeline parameters per context tier; do not advertise allocation-only maxima.

### Later research X01-X03

Prompt lookup/N-gram speculation may reuse the multi-row target verifier with little extra VRAM. DFlash and
EAGLE/P-EAGLE-like approaches require a locked compatible drafter, explicit training/licensing/reproducibility review,
and a complete VRAM/context comparison against native MTP. They are not part of current M20/M23 completion.

## 10. Execution order

Use this order unless a fresh profile provides contrary evidence:

1. **P01** last-chunk-only captures/head/argmax;
2. **P02** safe non-aliasing pinned prompt staging;
3. **D00** current decode attribution;
4. **D03a**, then D03b only if justified;
5. **D04** global K/V double buffering;
6. **D01**, **D02**, and **D05** as isolated exact Graph/small-kernel candidates;
7. **D09**, **D10**, and coherent **D11**;
8. **D06/D07/D08** only if exact candidates do not reach the desired decode result;
9. additional **P03-P09** work without delaying the open ordinary gate;
10. once 150 ordinary decode is reached, freeze that candidate, run matching real M21 context qualification, and
    complete formal M20 3/10;
11. record M20 complete, then continue ordinary and prefill optimization on top of the M20 checkpoint toward the
    next staged targets and practical local optimum;
12. perform the technical M23 base freeze only when the owner accepts the post-M20 base checkpoint, with M19 visibly
    deferred;
13. continue sampling, MTP, longer-context optimization, and optional research tracks under their separate records.

After every accepted candidate, reset the measured parent. Commit only isolated, reviewed, documented improvements
under the current owner authorization; do not push unless the owner explicitly requests it.

## 11. Immediate handoff

P01 is retained, P02 and D03a are rejected, D03b is skipped and D00 is complete. The next isolated mechanism is
**D04**, independent asynchronous K/V double buffering inside the existing one-split KVH2 CTA:

1. keep one split per CTA and preserve all current score, softmax, PV and merge arithmetic;
2. test K-only and V-only ping-pong staging independently before combining them;
3. keep the current 56-register/four-block residency and reject spills or harmful shared-memory occupancy loss;
4. retain the existing long-context and final-tail differential gates, then measure isolated kernel and full-engine
   decode;
5. if no D04 subcandidate wins, proceed to the larger constant-cost D09/D10 targets identified by D00.

## 12. Complete candidate inventory

This inventory is the exhaustive technical catalog from both owner-supplied German plans. The earlier sections define
priority and active execution; this section preserves every proposed experiment, variant, measurement, dependency,
and kill criterion. Inclusion is not automatic authorization to violate a higher-precedence contract. Items marked
complete, corrected, conditional, or rejected remain documented so that they are not accidentally rediscovered.

### 12.1 Extended performance targets and budgets

Ordinary decode begins at 139.054 token/s, or about 7.19145 ms/token:

| Target | Latency | Required saving | Relative throughput gain |
|---:|---:|---:|---:|
| 150 token/s | 6.66667 ms | 0.52478 ms/token | 7.87% |
| 155 token/s | 6.45161 ms | 0.73984 ms/token | 11.47% |
| 160 token/s | 6.25000 ms | 0.94145 ms/token | 15.06% |
| 165 token/s | 6.06061 ms | 1.13084 ms/token | 18.66% |
| 175 token/s | 5.71429 ms | 1.47716 ms/token | 25.85% |

Decode objectives are staged at 150, 155, 160, and 165+ ordinary token/s. The 150 gate closes M20 after formal
confirmation; higher stages belong to the continuing maximum-performance campaign.

Prefill begins at 6,574.164 token/s:

| Target | Required additional throughput |
|---:|---:|
| 6,750 token/s | 2.67% |
| 7,000 token/s | 6.48% |
| 7,250 token/s | 10.28% |
| 7,500 token/s | 14.08% |

Later mode objectives, reported separately from raw ordinary execution:

- bounded GPU sampling with `top_k <= 64`: aim for no more than 5% loss versus ordinary greedy;
- unrestricted top-p sampling: aim for no more than 10% loss versus ordinary greedy;
- MTP default threshold: at least 10% effective verified-throughput gain;
- MTP staged targets: investigate 180 effective token/s first and then 200+ without treating either as a promise;
- base context: 64K required, 96K desired, 128K experimental stretch;
- MTP maximum context: measured and published independently from base context.

### 12.2 D00 — full decode baseline and cost model

Measure contexts 128, 512, 2K, 4K, 8K, 16K, 32K, and 64K when the current run scope permits. The initial active
screen may stop at 16K; longer contexts are collected for M21 or a context-dependent hypothesis. Attribute:

- 25 local and 5 global attention layers separately;
- input normalization and FP8 activation quantization;
- Q/K/V and O projections;
- K/V normalization, RoPE, FP8 quantization, and cache append;
- local/global split attention and merge;
- shared MoE, router transform/projection, softmax, Top-8, W13, activation, W2, weighted reduction, and post norms;
- output head, softcap, candidate reduction, and argmax;
- Graph nodes, capture copies, H2D/D2D/D2H operations, control kernels, host gaps, and final synchronization.

Record microseconds/token and percentage of total latency per group, Graph-node count, context slope, and
register/shared/occupancy/spill facts for the five largest kernels. Separate constant T=1 cost from context-linear KV
cost. Nsight Systems owns end-to-end attribution; Nsight Compute is restricted to representative kernels.

### 12.3 D01 — exact raw K=V projection handoff

#### D01a: raw scratch alias

- verify `reuses_raw_k_for_v` from the locked model traits and loader transformation;
- let the raw V projection view alias raw K only until K and V post-processing diverge;
- remove the per-layer K-to-V D2D copy and redundant raw-V workspace;
- keep query, K-normalized/RoPE, V-normalized, staged FP8 K/V, and final K/V caches distinct;
- prove that no consumer writes raw K before the V normalization read completes;
- update validation so cache aliasing remains rejected.

#### D01b: direct distinct cache fill

- fuse the common raw handoff, K normalization/RoPE, V normalization, quantization, and cache append only where the
  existing BF16 and FP8 rounding order can be reproduced;
- consider writing the distinct K and V cache destinations from one controlled kernel;
- eliminate staged K/V buffers only after a lifetime and Graph-capture proof;
- cover both global and sliding layers, ring wrap, cache scales, and boundary positions.

Acceptance: current hash, bitwise K/V boundary outputs, no cache alias, no 12B regression, sanitizer-clean execution,
removed nodes/workspace, and at least a measured full-engine or enabling structural gain.

### 12.4 D02 — production graph separated from diagnostics

Capture before execution:

- `production_decode_graph`: no layer hidden-state, router-probability, or Top-ID captures;
- `diagnostic_decode_graph`: current capture behavior and `Copy*` observability.

Measure these subcandidates independently:

1. remove the twelve recurring D2D capture nodes from the production token Graph;
2. initialize `routing_finite` and `finite` as sticky request state and only clear on a detected error;
3. generate local and global RoPE tables in one exact kernel or prepare them once when valid;
4. bypass the product `expert_contributions` matrix and reduce slots directly in fixed order;
5. omit diagnostic capture buffers from production memory profiles;
6. remove unnecessary control/status nodes without changing the one-token external contract.

No subcandidate is bundled until its isolated result is known.

### 12.5 D03 — exact global-attention CTA structure

#### D03a: two independent 128-token splits per CTA

- retain two separate maximum, denominator, inverse-sum, partial output, and LSE states;
- retain the current per-split QK FMA, exponentiation, PV, and merge order;
- load each query once and reuse CTA setup, scales, and address arithmetic;
- write the two original partial/LSE indices so the merge kernel is unchanged;
- cover split tails, both physical global KV heads, and direct/split variants.

This must never become one 256-token softmax. Measure query-load bytes, CTA count, kernel/merge time, registers,
shared memory, occupancy, spills, exact hash, and full-engine decode.

#### D03b: four independent splits per CTA

Proceed only if D03a wins. At 16K this may reduce the global grid to about 64 CTAs per layer, which may improve wave
shape or create long under-occupied CTAs. The profile decides.

### 12.6 D04 — global K/V asynchronous double buffering

Implement a ping-pong shared-memory pipeline while preserving the arithmetic sequence:

- D04a: double-buffer K staging only;
- D04b: double-buffer V staging only;
- D04c: double-buffer both K and V;
- D04d: combine the accepted buffer design with D03a only after both isolated results;
- D04e: compare existing `cp.async` caching with a 16-byte L1-bypass/`cg` streaming policy;
- D04f: compare prefetch distance one versus two groups.

Pipeline contract:

1. preload tile zero;
2. wait for tile zero;
3. compute tile `i` while loading tile `i+1` into the alternate buffer;
4. use the appropriate `wait_group` and synchronization before buffer reuse;
5. handle prologue, tail, and epilogue explicitly.

Reject candidates that spill, materially lower occupancy, or fail the full-engine screening threshold.

### 12.7 D05 — Router Top-8 and selected-logit routing

#### Exact candidates

- D05a: 128 threads with the existing eight selection rounds;
- D05b: each of four warps sorts its 32 `(probability, expert_id)` pairs once, emits local Top-8, and warp zero
  deterministically merges 32 candidates;
- D05c: one block-wide deterministic 128-element bitonic sort;
- preserve current maximum, exponential, global softmax sum, selected-weight renormalization, output slots, and tie
  rule: higher value first, then lower expert ID.

#### D05n: numerically bounded selected-logit path

- choose Top-8 directly from logits;
- exponentiate only the eight winners;
- renormalize only those eight values;
- omit the 128-value probability buffer in the production graph;
- retain full probabilities in the diagnostic graph.

The common softmax constant cancels mathematically after Top-8 renormalization, but floating-point results can differ;
D05n is N-class and must retain the exact path.

### 12.8 D06 — global Tensor-Core QK

Use the existing SM120 BF16 MMA/`ldmatrix` and shared-layout knowledge. Per physical global KV head:

```text
K tile: 16 tokens x 512 dimensions
Q tile: 512 dimensions x 8 query heads
score: 16 tokens x 8 query heads
candidate MMA: m16n8k16 across 32 K steps
```

Variants:

- D06a: QK-only BF16 MMA, current FP32 softmax and PV;
- D06b: tune Q/K staging, swizzle, and warp specialization;
- D06c: one versus two QK consumer warps;
- D06d: combine only accepted D03/D04 scheduling with accepted QK.

Stage FP8 K to BF16 and Q to the matching layout. Compare score matrix, max/RMS/cosine, LSE, partial output, merged
attention, layer output, near ties, and 16K end-to-end tokens. Keep the scalar exact rollback.

### 12.9 D07 — Tensor-Core PV and fused online decode

Candidate PV mapping:

```text
V transpose: 16 output dimensions x 16 tokens
P: 16 tokens x 8 heads
O: 16 output dimensions x 8 heads
candidate MMA: m16n8k16
```

Variants:

- D07a: accepted QK MMA + current softmax + PV MMA;
- D07b: tile-wise online softmax without retaining all 128 scores;
- D07c: process K and V in one staged pipeline cycle;
- D07d: hierarchical two-/four-split merge within a CTA using an explicitly defined merge order;
- D07e: aggressive FP8-MMA QK/PV experiment.

QK and PV are measured separately. D07b-e are N-class because they alter reduction or quantization order.

### 12.10 D08 — local decode attention

After global attention:

1. share one query load across two independent local 128-token splits;
2. double-buffer local K and V;
3. specialize contiguous versus two-part wrapped-ring ranges;
4. bypass merge when the active local range needs only one split;
5. sweep CTA/warp shape separately for context classes 128, 512, and 1,024;
6. test BF16 Tensor-Core QK only if the remaining profile justifies it;
7. test PV MMA only after a winning local QK candidate.

Cover positions 1,023/1,024/1,025 and later 16K wrap transitions. Do not assume the global shape transfers to local
attention.

### 12.11 D09 — decode MoE and NVFP4 M=1

#### D09a: current-semantics W13 row reuse

- independently rebuild the historical two-output-row idea against the current kernel;
- reuse an activation and activation-scale load across adjacent output-row tiles;
- preserve the K-loop, MMA accumulation, BF16 product boundary, and output order;
- require the current `c750d0…` hash.

#### D09b: Down projection

- process adjacent output-row tiles per warp where profitable;
- reuse the expert activation/product and scales;
- apply router weight only after the same BF16 output rounding point;
- preserve deterministic slot-ordered accumulation.

#### D09c: shared/routed stream schedule

Compare independently:

- current full overlap;
- overlap only the router with shared Gate-Up;
- overlap shared Gate-Up but serialize Down phases;
- fully serialized execution;
- CUDA stream-priority variants.

Two simultaneous memory-bandwidth-heavy NVFP4 kernels may contend; less overlap can be faster.

#### D09d: production path without contribution writes

Avoid writing an `8 x 2816` contribution matrix when the product graph can reduce each slot directly in the same
fixed order. Keep the materialized matrix in the diagnostic path.

#### D09e: persistent/weight-stationary kernel

Only after profiling:

- let a CTA process multiple row tiles for one selected expert;
- stage activation once;
- double-buffer weight tiles;
- compare TMA against `cp.async`;
- test producer/consumer warp specialization;
- do not repeat the already-losing eight-warps-per-block design unchanged.

### 12.12 D10 — output head, softcap, and argmax

Variants:

- D10a: current full logits plus current softcap/argmax;
- D10b: the NVFP4 head writes one finite local candidate per CTA without the full-logit write;
- D10c: fuse or minimize the final candidate reduction;
- retain full logits for sampling, suppression, `CopyLogits()`, and diagnostics.

The candidate path must use the same BF16 projection rounding, current softcap evaluation, finite handling, and
token-ID tie rule. Softcap monotonicity alone is insufficient because rounding may create or remove ties.

### 12.13 D11 — device self-feed and host tail

#### D11a: coherent one-token product path

- feed selected token ID directly to the next embedding lookup;
- advance device position and ring state at Graph end;
- remove the per-replay `SetDecodeControl` H2D/kernel when safe;
- write token, selected logit, finite/routing-finite, and stop/error state into one fixed pinned or mapped host status;
- overlap host streaming with already-independent GPU work where API semantics permit;
- retain one externally required event/wait per visible streamed token.

The already-neutral four-copy packing idea is not repeated alone. Its value is reconsidered only with control H2D and
Graph-node removal.

#### D11b: hostless multi-token scheduler

Post-M20 research:

- conditional/loop Graphs or device-routed child Graphs;
- execute multiple tokens without a host replay decision;
- write verified tokens to a pinned ring while the host streams asynchronously;
- maintain stop state on device;
- make lookahead KV/position state rollback-capable.

This changes the ordinary per-token synchronization boundary and receives a separate maximum-throughput record.

### 12.14 P00 — prefill roofline and layer attribution

For the 16K row measure:

- 30 input-normalization/QKV projection groups;
- local/global RoPE preparation;
- local and global attention separately;
- O projection and residual boundaries;
- shared MoE;
- Tensor-Core router projection, assignments, and expert scheduling;
- routed Gate-Up and Down;
- assignment sort/permutation and deterministic reduction;
- activation quantization, BF16 round passes, scale interleaving, and W2 partial traffic;
- host chunk gaps, final head, and final synchronization.

Record Tensor-Core duty cycle, DRAM/L2 bandwidth, expert-token histograms, expert padding, CTA occupancy, launch gaps,
and per-chunk tail effects. No large prefill-GEMM or attention rewrite starts without this attribution.

### 12.15 P01 — final work only after the last chunk

Status: implemented and retained on 2026-08-25 as an exact safety/redundant-work cleanup. Three canonical samples
measure a +0.25% mean prompt gain with the same output hash; see
`../../../artifacts/m20/optimization-final-chunk-only.json`.

The implemented condition is:

```text
is_last_chunk = consumed + chunk == tokens.size()
```

Only on `is_last_chunk` execute final-prompt captures, final RMSNorm, output head, softcap, and argmax. Do not skip
per-chunk KV/cache updates, router finite detection, position/ring accounting, or state needed by the next chunk.
Require identical final captures, prediction, output hash, continuation behavior, and allocation evidence. Measure
removed head-weight traffic and launches.

### 12.16 P02 — safe prompt staging without whole-stream chunk barriers

Status: P02b measured neutral at +0.08% over P01 on three canonical samples and was fully reverted on 2026-08-25.
Do not retry without profiler evidence of a material host gap.

Variants:

- P02a: two pinned 1,024-token host slots with a copy-completion event per slot;
- P02b: one pinned host region sized to the initialized maximum context, with disjoint chunk source ranges;
- P02c: host ring plus copy stream, events, and two device token buffers;
- optional device-resident full prompt-ID buffer when memory and API lifetime justify it.

Recommended first candidate is P02b because at 262K it costs about 1 MiB of pinned host memory and keeps the existing
same-stream device-buffer ordering. P02a is the fallback if full-context pinning is undesirable. P02c is attempted
only if Nsight shows enough H2D cost to overlap. Prove source lifetime, device-buffer reuse order, event dependencies,
no `PrefillTokens()` allocation, racecheck/initcheck, hash identity, and reduced host gaps.

### 12.17 P03 — RoPE once per profile and chunk

- generate the local table once per chunk;
- generate the global table once per chunk;
- keep fixed, separate workspace regions and pass a prepared-table flag/pointer to each layer;
- use the current table math and rounding;
- measure removing roughly 28 redundant table launches per chunk;
- follow up with one combined local/global table kernel only after the exact hoist wins;
- consider a persistent full-context table only after an explicit memory/speed trade-off;
- do not recompute directly in each consumer unless it is measured faster.

### 12.18 P04 — prefill raw K/V reuse and direct distinct staging

- reuse or alias raw K as the V input for `attention_k_eq_v` layers;
- remove K-to-V raw D2D copies;
- keep learned K normalization/RoPE and V normalization separate;
- keep final FP8 K and V caches physically distinct;
- then test one exact kernel for K/V normalization, quantization, and distinct cache writes;
- preserve prepared-global persistent cache contents and all continuation semantics.

At 30 layers times 16 chunks, even small exact node/copy removals may matter.

### 12.19 P05 — physical BF16 audit and chunk sweep

Physical-BF16 hidden intermediates and T=1,024 are already promoted. Preserve the source proposals as an audit and
follow-up sweep:

1. label every residual, norm, activation, and layer-output rounding boundary;
2. verify that every float buffer claimed to carry only BF16 values is physically BF16 or has a justified consumer;
3. convert remaining transport-only float storage to BF16 without introducing a new rounding point;
4. sweep 512, 768, 1,024, 1,536, and 2,048 token chunks;
5. record VRAM peak, expert distribution, tail chunk, Tensor-Core utilization, attention behavior, and host staging;
6. reject larger chunks that violate the 192 MiB prefill workspace contract or required 64K reserve without a new
   explicit owner decision.

### 12.20 P06 — grouped MoE, expert locality, and work stealing

Candidate sequence:

1. record `tokens_per_expert` for every layer/chunk;
2. record padding, idle lanes, and bytes per selected expert;
3. select deterministic count buckets such as `M <= 16`, `17-32`, `33-64`, and `>64`;
4. generate a GPU task list `(expert, start, count, tile_shape)`;
5. compare static grids against a persistent CTA work queue/work stealing;
6. reuse activation and scale tiles across several output-row tiles;
7. double-buffer weight tiles;
8. compare TMA against the current `cp.async` staging only after the simple pipeline is measured;
9. fuse the Down epilogue with router weighting while preserving the current BF16 boundary;
10. preserve deterministic token/slot output placement and reduction.

Additional proposals from the second plan:

- stable sorting/grouping by expert;
- specialized small-expert paths for sparse counts;
- larger Tensor-Core paths for heavily populated experts;
- persistent scheduling tuned separately for prefill and multi-row verification.

Unordered/atomic accumulation that removes slot buffers is an N-class late experiment, never an exact shortcut.

### 12.21 P07 — Tensor-Core router fusion

Test independently:

- token tiles 8, 16, and 32;
- warp count and pipeline stages;
- input normalization plus router scaling/transform fusion;
- projection plus local Top-8 epilogue;
- selected values only in production;
- expert-count/histogram generation in the same epilogue;
- full logits/probabilities only in the diagnostic path.

The current Tensor-Core router already changed accumulation and has an exact serial rollback. Any further selected-logit
or probability omission is N-class and repeats the four-prompt/14-row QAT-BF16 gate plus full-model qualification.

### 12.22 P08 — local/global prefill attention staging

Preserve the accepted prepared-global BF16 K/V path through 16K while testing:

1. query/key tile-size sweeps;
2. K/V double buffering and prefetch distance;
3. TMA for large regular BF16 tiles;
4. cluster/DSM multicast of one K/V tile to multiple query CTAs;
5. shared-memory swizzles and bank-conflict removal;
6. register/spill control in the 1,024-token chunk;
7. prepared-global versus direct FP8 dequantization on current hardware;
8. a local-attention path specialized exactly for the 1,024 window;
9. separate tail-free full 1,024 chunks from small continuation/tail chunks;
10. multiple query rows per CTA or persistent key-outer/query-inner loop order if profiling still shows redundant K/V
    dequantization;
11. online-softmax rescaling only under exact/N-class gates appropriate to the changed order.

TMA/cluster work follows simple double buffering. Broad prefill Graph capture is reopened only if a new profile shows
launch dominance; fixed-shape capture, chunk replay, or two-stream pipelining remain catalogued but conditional on
workspace and dependency evidence.

### 12.23 P09 — projection, norm, residual, and quantization fusion

Measure one boundary at a time:

1. Q/K normalization plus RoPE and V normalization from the shared raw K input;
2. K/V normalization directly into distinct FP8 cache writes;
3. attention-output quantization in the O-projection prologue;
4. O-projection epilogue plus residual plus RMS sum-of-squares;
5. MoE Down epilogue plus weighted contribution;
6. post-shared/post-routed norm sums produced alongside the preceding epilogue;
7. residual plus next RMSNorm;
8. final RMSNorm plus output head;
9. router RMSNorm/scale plus router projection;
10. scale interleave and activation-quant passes fused into adjacent producers/consumers.

Every candidate documents the old and new BF16 rounding location and preserves it for E-class promotion.

### 12.24 P10 — optional product prefix cache

Prefix caching is separate from raw prefill:

- key entries by model/artifact, template, exact token prefix, KV precision, and RoPE profile;
- bound residency explicitly;
- never use the cache in M20 raw-prefill results;
- report cache-hit throughput and memory separately;
- use it for repeated system prompts/product workloads, not to replace engine optimization.

### 12.25 S00 — freeze the 26B sampling contract

Create Golden vectors for:

- greedy and suppression-only;
- temperature;
- top-k 1/8/16/32/64/128;
- top-p 0.8/0.9/0.95/1.0;
- min-p;
- repetition, presence, and frequency penalties supported by the product contract;
- combinations of penalties, suppression, top-k, top-p, and min-p;
- fixed equal/different seeds and sampling steps;
- EOS/stop cases, ties, NaN, and Inf.

The existing GPU sampling implementation is the token/RNG semantic oracle until a more independent exact reference
is added. Compare against the protected 12B implementation where contracts match, but do not assume architecture
identity.

### 12.26 S01 — sampling inside static decode graphs

Prepare before execution:

```text
greedy-candidate graph
suppressed-greedy graph
sample-full-sort graph
sample-topk-fast graph
```

Sampling Graph order:

1. head/logits or candidate epilogue;
2. softcap, finite detection, penalties, suppression, and temperature;
3. candidate selection/sort;
4. sampling with the current RNG sequence;
5. selected token written to persistent device state;
6. repetition state and sampling step updated on device;
7. next embedding consumes the device token;
8. host receives only compact token/stop/error status.

Do not call `Prediction()` before sampling in the hot path.

### 12.27 S02 — specialized small top-k

For `k <= 64`, avoid a full sort over 262,144 vocabulary entries:

- fuse preparation of penalized/suppressed/temperature-scaled logits;
- generate local warp/block Top-k candidates;
- merge hierarchically with stable token-ID ties;
- materialize and exponentiate only K candidates;
- preserve the current SplitMix64 or active RNG sequence exactly;
- specialize at least K=1, K<=8, K<=32, and K<=64;
- route K=1 to the exact greedy candidate path;
- keep generic full-sort fallback.

### 12.28 S03 — unrestricted top-p and min-p

Retain an exact CUB full-sort oracle. Then test:

- radix-select or histogram buckets for the high-logit region;
- block-wise exponential mass scans;
- an adaptive candidate buffer expanded until top-p mass is provably covered;
- combined top-k/top-p preparation;
- exact full-sort fallback whenever the candidate mass is insufficient.

The fast path must reproduce same-seed token sequences; it may never silently approximate the distribution.

### 12.29 S04 — head-to-sampling candidates

For bounded top-k configurations:

- apply softcap, penalties, suppression, and temperature in the NVFP4 head epilogue;
- emit local Top-k candidates instead of full logits;
- merge only candidates;
- retain full logits for unrestricted top-p/min-p, diagnostics, and `CopyLogits()`;
- read the repetition mask and up to the supported suppression-ID limit directly in the epilogue.

### 12.30 S05 — sampling qualification

Measure 16K+256 as well as shorter interactive runs:

- top-k 32/top-p 0.95 interactive profile;
- unrestricted top-p;
- repeated same-seed identity;
- many-seed statistical sanity checks;
- no invalid probabilities, allocation, or fallback;
- no host decision in the token loop;
- per-token streaming retained;
- throughput with and without host streaming reported separately;
- greedy fast path unchanged.

### 12.31 M00 — 26B assistant asset and architecture audit

Candidate source: `google/gemma-4-26B-A4B-it-assistant`. Lock without executing model-repository code:

- exact repository revision and SHA-256;
- configuration and complete tensor inventory;
- tokenizer/vocabulary compatibility;
- four-layer assistant architecture;
- target backbone hidden size 2,816 and assistant hidden size 1,024;
- intermediate size 8,192;
- three sliding plus one full-attention layer;
- 16 query heads, 8 local KV heads, and 2 global KV heads;
- four target-KV shared layers and their exact mapping;
- maximum position/configured context;
- target-last-hidden/preprojection interface;
- tied embedding/head without a second physical payload;
- BF16 payload and precise memory model.

The source plans estimate about 845,713,928 BF16 payload bytes, but the locked inventory is authoritative. Do not use
`trust_remote_code`.

### 12.32 M01 — assistant compiler and precision matrix

Build BF16 first as the numerical oracle, then compare:

- FP8 weights with BF16 or FP8 activations;
- NVFP4 for large MLP/projection matrices with BF16 norms/small tensors;
- Q4 only if an existing trusted decoder can be reused;
- hybrid FP8 attention plus NVFP4 MLP.

Ideal payload estimates from 845,713,928 BF16 bytes are about 806.5 MiB BF16, 403.3 MiB FP8 before metadata, and
201.6 MiB at four bits before scales/alignment. Select production precision by draft acceptance, proposal latency,
accepted tokens per target call, effective verified throughput, and VRAM—not logit error or draft token/s alone.
Assistant tokens may differ across quantizations; target output must not.

### 12.33 M02 — target verification for T=2/3/5

Draft depths generally map to target rows:

```text
D1 -> T=2
D2 -> T=3
D4 -> T=5
```

Build real multi-row target kernels:

1. batch FP8 Q/K/V projections over T;
2. local/global multi-row attention;
3. Tensor-Core router over all verifier rows;
4. group `T x 8` expert slots by expert;
5. reuse expert weights across rows and repeated experts;
6. restore outputs to deterministic `(row, slot)` positions;
7. reduce target slots in fixed order per row;
8. load NVFP4 head weights once for 2/3/5 rows;
9. commit only target-accepted rows.

Record `unique_experts_per_layer_per_group`, repeated slots, bytes loaded per verified row, and target batch cost. A
T=1 loop over verifier rows is not an acceptable optimized verifier.

### 12.34 M03 — transactional state

Adapt the proven 12B architecture from `src/cuda/mtp/assistant.cu`, `verify.cu`,
`inference_engine_mtp.cuh`, and `docs/MTP.md`:

- tentative global/local KV rows;
- tentative hidden states;
- accepted prefix length;
- position and ring wrap;
- RNG and sampling step;
- repetition state;
- stop/reasoning/response state;
- atomic commit of accepted rows;
- ordinary target bonus token on rejection;
- no host decision between draft and verification.

Test acceptance 0/1/K, stop during draft, stop at bonus, context end, and local ring wrap.

### 12.35 M04 — static graphs and adaptive draft depth

Prepare fixed Graphs for ordinary, D1, D2, and D4. Measure fixed depths before adaptive control. Maintain a device-side
EWMA per context tier for accepted prefix, all-accepted rate, rejection position, measured group cost, and remaining
output. Estimate:

```text
time_per_emitted_token(K) =
  (proposal_cost(K) + verify_cost(K) + commit_cost(K)) /
  expected_emitted_tokens(K)
```

Select among `{0,1,2,4}` with hysteresis. Google's increase-on-full-accept/decrease-on-reject heuristic is an initial
reference; final thresholds come from RTX 5080 measurements. Prefer device-resident selection without per-group CPU
roundtrips.

### 12.36 M05 — greedy and sampled MTP

Greedy accepts a draft token exactly when it equals the target argmax for that verifier row. Ordinary and MTP output
must be identical.

Sampled MTP first follows the protected 12B lossless contract:

- target performs its ordinary GPU sampling for each row with identical seed/step semantics;
- accept a draft token when it equals the target sample;
- resulting output equals ordinary same-seed target output;
- rejected rows roll back RNG/repetition/state transactionally.

Standard p/q residual rejection sampling is a later separate research path requiring correct target and assistant
probabilities and additional memory.

### 12.37 M06 — MTP sweep and kill criteria

For D1/D2/D3/D4 where implemented, plus adaptive, measure at 512, 2K, 16K, 32K, and compatible 64K:

- proposal, target verification, commit, and total group milliseconds;
- proposed, accepted, and rejected tokens;
- mean accepted prefix and all-accepted rate;
- target batches;
- effective verified token/s;
- peak VRAM, free margin, and maximum context;
- thermal/power behavior.

Default promotion requires at least 10% retained median gain at 16K, target-identical output, no allocation/fallback,
and valid 32K memory. If a quantized assistant plus specialized T=2/3/5 verifier plus adaptive depth still yields no
more than about 2% or regresses, do not enable MTP by default. Optional mode claims remain honest.

### 12.38 M07 — assistant and verifier fusion

After functional MTP:

- fuse target-last-hidden, token embedding, and assistant preprojection;
- perform selective target-embedding lookup only for draft IDs;
- implement Q-only assistant attention directly over shared target KV;
- reuse local/global assistant RoPE tables;
- capture the four assistant layers in fixed Graph paths;
- fuse assistant Gate-Up/Down for the selected FP8/NVFP4 format;
- use head candidates rather than full logits for greedy drafts;
- capture D2/D4 proposal Graphs separately;
- group verifier routes by expert across rows;
- specialize multi-row head for 2/3/5 rows;
- keep proposed tokens device-resident and stream only verified output.

### 12.39 C00 — exact memory baseline

Current fixed target-weight arena:

```text
14,696,668,160 bytes (about 14,015.83 MiB)
```

Repository FP8-KV approximation:

```text
local fixed after 1,024 tokens: about 100 MiB
global: 10,240 bytes per context token
```

Approximate FP8-KV payload:

| Context | Payload |
|---:|---:|
| 32K | 420 MiB |
| 64K | 740 MiB |
| 96K | 1,060 MiB |
| 128K | 1,380 MiB |

These are planning estimates; direct `cudaMemGetInfo` and execution evidence own qualification. The source analysis
estimates that 96K needs roughly 260 MiB less workspace/Graph/side residency to retain a 400 MiB reserve.

### 12.40 C01 — fixed initialization profiles

Candidate profiles:

```text
speed-16k
balanced-64k
long-96k
long-128k-experimental
mtp-32k
mtp-long
```

Each selects fixed arena offsets and Graphs at initialization. Potential profile-specific omissions/choices:

- diagnostic captures only in diagnostic profiles;
- sampling arrays/CUB workspace only when sampling is enabled;
- full logits only for sampling/CopyLogits/diagnosis;
- prepared-global BF16 staging only in the speed profile when profitable;
- T=1,024 prefill for speed and smaller fixed chunks for long context;
- alias prefill and MTP verifier/assistant workspaces only when lifetimes are disjoint;
- allocate MTP transaction state only in MTP profiles.

No profile may silently reduce the requested context.

### 12.41 C02 — workspace lifetime aliasing

Build a matrix for:

```text
load/init
prefill projection
prefill attention
prefill MoE
decode attention
decode MoE
head/sampling
MTP assistant
MTP target verify
diagnostics
```

Alias only proven non-overlapping lifetimes. Targets:

- 96K base with at least 400 MiB reserve;
- 32K MTP with at least 700 MiB reserve;
- 64K MTP with at least 500 MiB reserve or an explicit clear failure.

The source plan notes that an approximately 781 MiB current 32K base reserve would leave only about 81 MiB for MTP
under the 700 MiB rule, while even ideal four-bit assistant weights are about 202 MiB before metadata/workspace.
Therefore base/workspace savings of at least roughly 121 MiB plus verifier reserve are likely needed. Re-measure on
the final candidate rather than treating this estimate as acceptance evidence.

### 12.42 C03 — global KV compression

Only global layers are candidates; local KV is already bounded by the 1,024 window. Test after attention stabilizes:

- K8/V4 first because K is usually more sensitive;
- K4/V4 second;
- per-head, per-128-token-block, and per-16-token-tile scales;
- dequantization fused into attention staging;
- compatibility with accepted QK/PV pipelines.

At 96K, planning estimates are about 960 MiB global FP8, 720 MiB K8/V4 before scales (about 240 MiB saved), and 480
MiB K4/V4 before scales (about 480 MiB saved). Measure quality, 16K/64K/96K decode, scale storage, dequant cost,
Tensor-Core compatibility, margin, and maximum context. Keep FP8 rollback.

### 12.43 C04 — context-tier kernels

Do not reuse one 16K kernel blindly at 64K/96K/128K. Tune per tier:

- splits per CTA;
- hierarchical merge;
- K/V ping-pong staging;
- direct compressed-KV dequantization;
- context selector before graph capture;
- 16K/32K/64K/96K/128K-specific launch plans.

Publish base and MTP maxima separately.

### 12.44 X01 — prompt lookup/N-gram speculation

Low-VRAM, target-verified product mode:

- fixed GPU hash/suffix structure over existing context;
- D2/D4 proposals for repetition, code, lists, and copied text;
- reuse the same multi-row target verifier as MTP;
- ordinary fallback on no match;
- separate workload and throughput record, never the general Wikipedia ordinary number.

### 12.45 X02 — DFlash

DFlash requires a compatible block-diffusion drafter and target-feature contract; DiffusionGemma is not automatically
compatible with an autoregressive 26B target. After native MTP:

1. find and lock a compatible Gemma 4 26B DFlash drafter;
2. otherwise evaluate training data, cost, license, and reproducibility;
3. model exact weights/KV/workspace/context residency;
4. define target-hidden interface;
5. test block sizes 4/8/16;
6. reuse target verification;
7. compare effective speed, acceptance, and lost context against MTP.

Kill on no compatible locked drafter, unreasonable training burden, or loss of required context without a decisive
speed advantage.

### 12.46 X03 — EAGLE/P-EAGLE and trained speculators

Investigate only if official/native MTP is exhausted, MoE verifier reuse is insufficient, a compatible trained
speculator promises materially higher acceptance, and the VRAM/context budget fits. Do not build a generic
speculator framework; keep the implementation model- and hardware-specific.

## 13. Execution waves beyond the immediate queue

### Wave 1 — exact isolated foundation

- P01 final-chunk-only work;
- P02 safe staging;
- P03 RoPE hoist;
- D00 current profile;
- D01 raw K/V handoff;
- D02 production/diagnostic Graph separation;
- M00 assistant asset/memory audit may proceed as documentation/fixture work only.

### Wave 2 — ordinary decode through 150 and beyond

- D03 two/four independent splits;
- D04 double buffering;
- D05 exact Router Top-8;
- D09 exact M=1 MoE candidates;
- D10 greedy head candidates;
- coherent D11 host/control tail.

### Wave 3 — numerically bounded main levers

- D06 Tensor-Core QK;
- D07 Tensor-Core PV/online fusion;
- D08 local attention MMA;
- D05n selected-logit routing;
- later D09 pipeline/accumulation experiments.

### Wave 4 — prefill through 7K and beyond

- P04 exact raw K/V/direct staging;
- P05 chunk sweep;
- P06 expert buckets/work queue;
- P07 router epilogue fusion;
- P08 attention TMA/cluster only after simpler pipelines;
- P09 exact boundary fusion.

### Wave 5 — real GPU sampling

- S00 Golden contract;
- S01 Graph integration;
- S02 small top-k;
- S04 direct candidates;
- S03 unrestricted top-p optimization;
- S05 qualification.

### Wave 6 — MTP

M00/M01 can begin earlier without base-runtime overlap. Runtime integration follows a strong frozen ordinary target:

- M02 target T=2/3/5;
- M03 transaction state;
- M04 Graph/adaptive depth;
- M05 sampling;
- M06 sweep;
- M07 optimization.

### Wave 7 — context and research

- C01/C02 fixed profiles/lifetime aliasing;
- C03 KV4;
- C04 context-tier kernels;
- X01 prompt lookup;
- X02 DFlash;
- X03 other trained speculators.

## 14. Acceptance matrix

| Area | Required correctness | Required performance evidence | Required memory evidence |
|---|---|---|---|
| Exact ordinary | `c750d0…`, bitwise operator gates | bounded A/B, then formal M20 when frozen | no unexplained gate loss |
| Numeric ordinary | N-class Golden Gate and rollback | reproducible full-engine win | peak and free margin |
| Prefill | final token/captures/KV/continuation correct | exact raw 16K prefill | 64K support not silently lost |
| Sampling | same-seed versus oracle | separate sampling token/s | sampling-only workspace/profile |
| MTP | ordinary-identical target output | effective verified token/s | 32K >=700 MiB; 64K >=500 MiB or fail |
| KV4 | quality plus FP8 rollback | 16K/64K/96K decode | maximum context and margin |
| DFlash/N-gram | target verified | compared against ordinary and MTP | context loss disclosed |

## 15. Complete source-plan coverage matrix

Every source section is mapped below. “Corrected” means the proposal is retained with a contract fix rather than
silently omitted.

### 15.1 `CODEX_GEMMA4_26B_MAX_PERFORMANCE_PLAN.md`

| Source section | English destination | Coverage note |
|---|---|---|
| Mission and operating rules | Sections 1-4 | complete; M20 versus post-M20 stop rule updated by owner |
| Baseline/latency/targets | Sections 2 and 12.1 | complete, including 150/155/160/165/175 and 6,750-7,500 |
| Measurement/evidence | Section 4 | complete, including A/B, 3/10, profiler, artifact fields |
| Correctness E/N/S | Section 5 | complete |
| D00 | 7/W1 and 12.2 | complete |
| D01 | 7/W3 and 12.3 | corrected: raw scratch may alias; physical K/V caches may not |
| D02 | 7/W3 and 12.4 | complete, including four subcandidates |
| D03 | 7/W2 and 12.5 | complete, including 2/4 splits and explicit non-256-softmax rule |
| D04 | 7/W2 and 12.6 | complete, variants a-f |
| D05 | 7/W3 and 12.7 | complete, exact a-c plus numeric selected-logit path |
| D06 | 7/W5 and 12.8 | complete, QK-only through combined scheduling |
| D07 | 7/W5 and 12.9 | complete, variants a-e |
| D08 | 7/W5 and 12.10 | complete, all local/window/ring candidates |
| D09a-e | 7/W4 and 12.11 | complete, including stream schedules and persistent path |
| D10 | 7/W4 and 12.12 | complete, variants a-c and full-logit fallback |
| D11/D11b | 7/W4 and 12.13 | complete; copy-only experiment not repeated |
| P00 | 12.14 | complete |
| P01 | 7/W0 and 12.15 | complete, immediate slice |
| P02 | 7/W0 and 12.16 | complete, variants a-c |
| P03 | 8 and 12.17 | complete |
| P04 | 8 and 12.18 | corrected distinct final caches |
| P05 | 8 and 12.19 | complete; physical BF16/T1024 marked done, sweep retained |
| P06 | 8 and 12.20 | complete, all ten experiments plus locality variants |
| P07 | 8 and 12.21 | complete |
| P08 | 8 and 12.22 | complete, including TMA/cluster/tail/persistent ideas |
| P09 | 8 and 12.23 | complete, all fusion boundaries |
| P10 | 8 and 12.24 | complete, separate product mode |
| S00-S05 | 9 and 12.25-12.30 | complete |
| M00-M07 | 9 and 12.31-12.38 | complete, including asset facts, T rows, adaptive cost, kill gate |
| C00-C04 | 9 and 12.39-12.43 | complete, including memory estimates/profiles/KV4/context tiers |
| X01-X03 | 9 and 12.44-12.46 | complete |
| Execution waves | Sections 10 and 13 | complete; P01/P02 moved first by owner |
| Known dead ends | Section 6 | complete |
| Acceptance matrix | Section 14 | complete |
| First concrete task | Section 11 | complete; owner-selected P01 precedes D03a |

### 15.2 `chatgptpro_plan.md`

| Source section | English destination | Coverage note |
|---|---|---|
| Mission/rules/budgets | Sections 1-4 and 12.1 | complete; 49 GB rule corrected to current owner decision |
| D00-D11 | Sections 7 and 12.2-12.13 | complete, including D01a/b, D03a/b, D05a/b, and device self-feed |
| P00-P10 | Sections 8 and 12.14-12.24 | complete |
| S00-S05 | Sections 9 and 12.25-12.30 | complete |
| M00-M07 | Sections 9 and 12.31-12.38 | complete |
| C00-C04 | Sections 9 and 12.39-12.43 | complete |
| Later speculation | Sections 9 and 12.44-12.46 | complete |
| Recommended waves | Sections 10 and 13 | complete |
| First D03a slice | Sections 7/W2 and 12.5 | retained; P01/P02 execute first per latest owner ordering |

No proposal from either source plan is intentionally omitted. Where a source proposal conflicts with accepted model
semantics or active policy, this plan records the corrected version and the reason rather than copying an unsafe or
stale instruction literally.

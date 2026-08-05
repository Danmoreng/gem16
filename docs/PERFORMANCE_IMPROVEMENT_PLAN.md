# Gemma 4 12B final performance sprint

Status: S00-S08 and the Windows regression are complete; the optimization sequence is closed; the direct
all-regions memory-reserve record remains a repository closure gate

Primary platform: Linux x86-64, NVIDIA GeForce RTX 5080 Laptop GPU, Blackwell SM120, firmware profile
`max-power`

Primary model: `unsloth/gemma-4-12b-it-NVFP4` at
`b1f649734b34aa5575b03d186abd1b9be3d0d5c4`

Assistant model: `google/gemma-4-12B-it-assistant` at
`364bd03c9952e5b7da73665ee30c9eccfc408345`

Plan revision: 1, 2026-08-04

## 1. Purpose and authority

Run one bounded 12B optimization sprint before beginning Gemma 4 26B A4B implementation. The sprint targets the
remaining measured prefill gap to direct-load vLLM and a small number of exact, source-visible decode inefficiencies.
It does not reopen completed 12B feature work or authorize a broad runtime rewrite.

This document supersedes the execution order in [PREFILL_OPTIMIZATION_PLAN.md](PREFILL_OPTIMIZATION_PLAN.md).
That document remains historical evidence for previously promoted and rejected prefill work. Permanent correctness,
benchmark, memory, and development rules remain in [AGENTS.md](../AGENTS.md),
[BENCHMARKING.md](BENCHMARKING.md), [CORRECTNESS.md](CORRECTNESS.md),
[MEMORY.md](MEMORY.md), and [DEVELOPMENT.md](DEVELOPMENT.md).

The sprint ends before 26B M00 begins. A candidate not admitted by this plan requires a new profile and an explicit
plan amendment; static plausibility alone is insufficient.

## 2. Known baseline and objective

The retained 2026-08-03 Linux comparison used batch one, the exact 16,384-token Wikipedia prompt, 1,135 fixed
output positions, fixed D2, three warm-ups, and ten measured runs:

| Engine | Prefill tok/s | TTFT | Effective D2 tok/s | ITL | Sampled peak VRAM |
|---|---:|---:|---:|---:|---:|
| vLLM 0.26.0 | **6,247.55** | **2,622.47 ms** | 81.95 | 12.202 ms | 15,465 MiB |
| **gem16 `8e86cb38`** | 5,863.59 | 2,794.19 ms | **89.58** | **11.163 ms** | 11,867 MiB |
| llama.cpp b10240 | 3,922.61 | 4,176.81 ms | 82.88 | 12.065 ms | 10,631 MiB |

The measured gem16 prefill deficit to vLLM is 6.15%. Gem16 already leads both retained external D2 rows. External
outputs and formats differ, so these are controlled performance results rather than semantic or format parity.

### Primary objective

Meet or exceed the freshly rerun direct-load vLLM prefill median on the fixed 16K workload while retaining gem16's
current model semantics, deterministic output, MTP exactness, memory contract, and decode performance.

### Secondary objectives

- Reduce ordinary and fixed-D2 long-context latency only through exact Target-preserving changes.
- Recover prefill workspace for larger prompt tiles and future model work.
- Leave one production implementation per qualified geometry; remove losing experimental paths.

### Completion or stop conditions

The sprint closes when the first applicable condition is met:

1. a final controlled 3/10 run meets or exceeds the current vLLM prefill row and passes every gate;
2. the mandatory low-risk sequence and at most two profile-admitted deeper candidates are exhausted without parity;
3. two consecutive admitted kernel candidates fail their end-to-end promotion gate after the mandatory sequence;
4. a correctness, quality, or 700 MiB memory-reserve gate blocks further work.

A closed sprint records the measured remaining gap rather than weakening a gate. The 26B track resumes after the
closure record is written.

## 3. Evidence classification

The ideas that motivated this plan were static code-review hypotheses. They are not performance results. This plan
uses three evidence labels:

- **Measured:** retained raw benchmark or profiler output from the reference GPU.
- **Source-confirmed:** the current repository performs the described operation, but its cost is not yet isolated.
- **Hypothesis:** a proposed replacement that must pass an admission profile before implementation.

Current source-confirmed opportunities are:

- the ordinary, global-batch, and local-D2 attention merge kernels recompute the same split `expf` weights in every
  output-dimension loop;
- prompt prefill runs final RMSNorm over every row of every chunk although only the last row of the final chunk is
  consumed;
- production MLP prefill writes compact activation scales and launches a separate CUTLASS scale-interleave kernel;
- production prompt Q/K/V projection invokes the CUTLASS FP8 projection separately for Q, K, and optional V;
- prompt Q/K/V normalized buffers are FP32 allocations containing BF16-rounded values, followed by a separate K/V
  FP8 quantization pass;
- every NVFP4 CUTLASS projection prepares a complete temporary weight and scale layout before GEMM;
- global D512 prefill CTAs independently load the shared K/V head;
- the long-context global GQA decode kernel uses one K/V staging buffer and serializes normalization across 16 query
  heads.

Only the first three are mandatory implementation candidates. The remaining items require a fresh profile gate.

## 4. Non-negotiable constraints

Every candidate must preserve:

- the pinned checkpoint, tokenizer, template, generation controls, prompt IDs, and output count;
- direct mixed FP8/NVFP4 execution and the sole persistent SM120 Row8/K64 Target weight layout;
- checkpoint-FP8 K/V semantics for production results;
- exact ordinary/MTP Target identity inside gem16;
- no CPU weight offload and no silent precision, kernel, graph, or semantic fallback;
- no allocation, filesystem access, JIT, graph capture, or pageable host allocation in the token loop;
- fixed execution-plan addresses after initialization;
- at least 700 MiB directly measured unused CUDA-visible memory on the reference machine;
- batch-one timing boundaries and all disclosures required by `docs/BENCHMARKING.md`.

No candidate may add a permanent second weight layout. A load-time exact transformation into the sole final layout
remains permitted. Temporary benchmark selectors must not survive as public runtime modes; the winner becomes the
single production dispatch.

## 5. Sprint workflow

### 5.1 Parent discipline

For every candidate:

1. start from the latest promoted sprint commit and a clean worktree;
2. record the full parent SHA, model locks, toolchain, GPU state, and baseline result directory;
3. capture the parent profile that admits the candidate;
4. implement only one hypothesis;
5. run focused correctness before broad correctness;
6. run a short rejection screen;
7. if positive, run an adjacent parent/candidate 3-warm-up/10-measurement qualification;
8. retain compiler, SASS, register, spill, shared-memory, and allocator evidence;
9. promote only after the applicable exit gate passes;
10. remove a losing implementation completely and record it in `PERFORMANCE_LEDGER.md`.

Coding agents create commits only after explicit owner authorization. Final evidence must still identify exact source
by commit or retained patch hash; dirty-worktree status is always disclosed.

### 5.2 Result layout

Store results under a unique path:

```text
benchmarks/results/<date>/<git-sha>/blackwell16gb-linux-maxpower-12b-sprint/
  S00-baseline/
  S01-merge-weights/
  S02-final-rmsnorm/
  S03-direct-cutlass-scales/
  S04-chunk-sweep/
  S05-reprofile/
  S06-<admitted-candidate>/
  S08-final/
```

Each candidate directory contains commands, system and model identity, raw JSON, summary JSON, continuous GPU
telemetry, allocator accounting, and relevant `nsys/` or `ncu/` reports. Never overwrite an earlier run.

### 5.3 Default statistical decision

A speed candidate is promotable when:

- the affected end-to-end median improves;
- the result is supported by the 3/10 distributions and 95% confidence intervals;
- no required neighboring workload regresses materially;
- clocks and thermals do not explain the difference;
- correctness and memory gates pass.

If a small apparent win has overlapping intervals, increase to 30 measured runs. Reject it if the larger run remains
inconclusive. Do not promote on a best run or microbenchmark alone.

A memory-foundation candidate may be promoted without a throughput win only when it removes at least 100 MiB of
named arena space, changes no arithmetic, and its repeated throughput remains statistically neutral. Any measurable
slowdown rejects the foundation.

## 6. Standard build, test, and benchmark commands

Initialize the reference variables once:

```bash
MODEL="$(python -c 'from tools.hf_cache import default_target_model; print(default_target_model())')"
ASSISTANT="$(python -c 'from tools.hf_cache import default_assistant_model; print(default_assistant_model())')"
RUN=build/Linux/blackwell-release/bin/gem16-run
BENCH=build/Linux/blackwell-release/bin/gem16-bench
WORKLOAD=benchmarks/prompts/wikipedia-summary-16k.json
```

Build and run the normal suites:

```bash
cmake --preset host-debug
cmake --build --preset host-debug -j
ctest --preset host-debug --output-on-failure

cmake --preset blackwell-release
cmake --build --preset blackwell-release -j
ctest --preset blackwell-release --output-on-failure
```

Run the primary gem16 workload:

```bash
python tools/benchmark_wikipedia_workload.py \
  --engine gem16 \
  --workload "$WORKLOAD" \
  --output <unique-result>/gem16.json \
  --model "$MODEL" \
  --assistant-model "$ASSISTANT" \
  --mtp-draft-tokens 2 \
  --fixed-output-tokens 1135 \
  --warmups 3 --repetitions 10 \
  --executable "$RUN"
```

Run the prefill screening matrix:

```bash
for context in 128 512 2048 8192 16384; do
  "$BENCH" prefill --model "$MODEL" --context "$context" \
    --kv-cache fp8 --warmups 3 --repetitions 10 \
    > "<unique-result>/prefill-${context}.json"
done
```

Run the ordinary decode matrix for decode-affecting candidates:

```bash
for context in 128 2048 8192 16384 32768 65536; do
  "$BENCH" decode --model "$MODEL" --context "$context" --tokens 256 \
    --kv-cache fp8 --warmups 3 --repetitions 10 \
    > "<unique-result>/decode-${context}.json"
done
```

The final cross-engine run is executed only for S00 refresh and S08 closure:

```bash
systemd-run --user --scope -p MemoryMax=48G -p MemorySwapMax=0 \
  ./scripts/benchmark-cross-engine-mtp.sh \
  --output <unique-result>/cross-engine
```

Do not use the full external matrix to screen every code edit.

## 7. Correctness and resource gates

### 7.1 Gates required for every production change

- host-debug and Blackwell CTest pass;
- no new warnings, CUDA launch failures, sanitizer errors, or fallback counters;
- `tools/validate_inference.py` passes after intentional dispatch metadata is updated;
- `tools/validate_prefill_boundaries.py` passes for prefill-affecting changes;
- exact-blue and the retained 12-prompt teacher-forced metrics do not regress;
- fixed Wikipedia output remains deterministic;
- ordinary and D2 output IDs remain identical inside gem16;
- no recurring allocation is added;
- named workspace, graph-private memory, sampled peak VRAM, and direct `cudaMemGetInfo` reserve are recorded.

Validators that pin the old production dispatch, such as the 8,192-token chunk or separate Q/K/V metadata, must be
updated in the same change only after the replacement is qualified. Do not weaken numerical assertions merely to
accept a candidate.

### 7.2 Additional CUDA-kernel gates

- focused synthetic and real-shape operator comparison;
- Compute Sanitizer memcheck and racecheck for the changed kernel family;
- representative SASS instruction confirmation;
- registers, static/dynamic shared memory, stack frame, local memory, spills, and launch geometry;
- an Nsight timeline proving that the intended old work disappeared;
- Nsight Compute counters relevant to the hypothesis.

### 7.3 Additional memory gates

- checked arena arithmetic and alignment;
- initialization succeeds with Target, assistant, selected context, graphs, media regions, and sampling workspace;
- directly measured free memory after initialization and at sampled peak remains at least 700 MiB;
- no persistent duplicate layout or undisclosed CUDA-private growth.

## 8. Milestone overview

| ID | Candidate | Type | Admission | Required outcome |
|---|---|---|---|---|
| S00 | Refresh parent and profiles | mandatory | none | clean reproducible baseline |
| S01 | Cache attention-merge split weights | mandatory | source-confirmed | exact decode/MTP win or removal |
| S02 | Final RMSNorm on the final row only | mandatory | source-confirmed | about 120 MiB removed, neutral or faster |
| S03 | Write MLP scales directly in CUTLASS layout | mandatory | source-confirmed | remove interleave launches and win or remove |
| S04 | Sweep 8K/12K/16K prefill chunks | mandatory after S02/S03 | memory preflight | select one winner or retain 8K |
| S05 | Reprofile the promoted stack | mandatory | S01-S04 complete | ranked measured bottlenecks |
| S06 | Medium candidate | conditional | S05 thresholds | one measured winner or documented rejection |
| S07 | Deep kernel candidate | conditional | S05 and sprint budget | one measured winner or documented rejection |
| S08 | Final qualification and close | mandatory | candidate work closed | final cross-engine result and hand-off to 26B |

S06 and S07 together may admit at most two candidates unless the owner explicitly extends the sprint.

## 9. S00 — refresh the exact parent

### Tasks

1. Synchronize the intended parent with `origin/main` without losing local work.
2. Record `git status`, full SHA, model locks, toolchain lock, driver, CUDA, VBIOS, power profile, clocks, thermals,
   display use, and CUDA-visible memory.
3. Build host-debug and Blackwell release from scratch.
4. Run the complete host and CUDA test suites.
5. Run the exact 16K gem16 workload and the prefill/decode matrices.
6. Capture one Nsight Systems profile of 16K prefill and one of ordinary plus D2 long-context decode.
7. Capture targeted Nsight Compute reports for:
   - the three attention merge kernels;
   - global GQA split attention;
   - MLP activation quantization and scale interleave;
   - NVFP4 weight preparation and GEMM;
   - FP8 Q/K/V GEMMs;
   - local and global prefill attention.
8. Run the controlled external comparison once to refresh the sprint target.

### Deliverable

`S00-baseline/summary.md` contains a phase-time table, launch counts, memory accounting, exact output hashes, and a
ranked list of bottlenecks. It separates split, merge, projection, attention, output-head, and host/API time.

### Exit gate

No implementation starts until S00 is reproducible and the new profile confirms that the mandatory source-visible
work still executes in production.

## 10. S01 — cache attention-merge split weights

### Scope

Modify only:

- `MergeOnlineDecodeAttentionKernel`;
- `MergeOnlineDecodeAttentionGlobalBatchKernel`;
- `MergeOnlineDecodeAttentionFp8LocalD2Kernel`;
- their focused tests and launch metadata.

The kernels currently compute each `expf(partial_lse - maximum)` once for the denominator and again for every
output dimension. At 16K, the source-level upper count is approximately 2.75 million repeated merge exponentials
per ordinary token; a three-row D2 Target batch scales this work by about three. This is an operation-count
hypothesis until SASS and profiling confirm the executed cost.

### Implementation contract

1. Allocate a separate shared array for split weights; do not alias it unsafely with the reduction scratch.
2. Have the same thread-to-split mapping compute the same `expf` values used by the denominator.
3. Store unnormalized weights, then reuse them in the dimension loop.
4. Preserve the existing denominator reduction tree, split order, `fmaf` order, reciprocal, and final multiply.
5. Support local and global maximum split counts; 512 FP32 global weights require 2 KiB shared memory.
6. Keep ordinary, batched global, and local-D2 behavior consistent.

### Required tests

- valid split counts at 1, local maximum, 16K, 64K, and 262,144 positions;
- repeated launches and racecheck;
- bit-identical ordinary and D2 operator output where the arithmetic contract predicts identity;
- fixed 16K ordinary/MTP ID identity;
- SASS confirms the repeated dimension-loop exponentials are gone.

### Performance gate

Run ordinary decode at 8K/16K/32K/64K and fixed 16K D2. Promote only if at least one representative end-to-end
long-context metric improves with statistical support and no matrix point regresses materially. A kernel-only SFU
reduction with no end-to-end win is removed.

## 11. S02 — compute final prompt RMSNorm for one row

### Scope

Update `InferenceEngine::Impl::PrefillAt` and prefill arena planning.

Production prefill currently normalizes every row after every chunk. Only the last row of the final chunk feeds the
first output head and `latest_target_hidden_`. MTP verification independently needs at most
`kMaximumMtpVerifyTokens == 5` normalized rows.

### Implementation contract

1. Do not launch final RMSNorm for non-final prompt chunks.
2. On the final chunk, normalize only `hidden[tokens - 1]` into normalized row zero.
3. Point `latest_target_hidden_` to normalized row zero.
4. Retain the existing MTP path, which may normalize up to five rows.
5. Size `prefill_offsets_.normalized` for five FP32 rows, not `prefill_chunk_tokens_` rows.
6. Keep diagnostic full-state/logit paths explicit; do not silently route them through an incomplete buffer.

At an 8,192-token chunk, the current allocation is 125,829,120 bytes. Five rows require 76,800 bytes, so the named
arena reduction should be 125,752,320 bytes before alignment effects.

### Required tests

- one-token, exact chunk, chunk-plus-one, and multi-chunk prompts;
- first output logits and token unchanged;
- `latest_target_hidden_` valid through ordinary and MTP continuation;
- D1/D2/D4 verification still handles up to five normalized rows;
- no use-after-overwrite across resident turns;
- allocator report proves the expected named reduction.

### Promotion gate

Promote as a memory foundation if at least 100 MiB is removed, all correctness gates pass, and repeated prefill and
D2 results are neutral or faster. Otherwise remove the change.

## 12. S03 — generate CUTLASS activation scales directly

### Scope

Production MLP prefill currently performs:

```text
RMSNorm + NVFP4 activation quantization
  -> compact E4M3 scales
  -> InterleaveActivationScalesKernel
  -> CUTLASS Gate/Up
```

Add a production-only quantizer output mode that writes the exact final CUTLASS scale layout directly.

### Implementation contract

1. Share one tested `InterleavedScaleOffset` contract between the quantizer and the existing reference transform.
2. Preserve E2M1 activation bytes and E4M3 scale words exactly.
3. Preserve the compact layout for direct T=1/T≤5 and MTP verifier kernels.
4. In production prefill, write directly to `cutlass_activation_scales` and skip
   `LaunchNvfp4CutlassInterleaveActivationScales`.
5. Shrink the compact `mlp_scales` prefill region to the maximum five-row verifier requirement rather than deleting
   it.
6. Do not change Down-product scale generation, which already writes its production layout directly.

For an 8K chunk, the compact scale payload is 1,966,080 bytes. The current two-chunk 16K path launches the
interleave 96 times. These are source counts, not assumed DRAM traffic.

### Required tests

- byte identity against compact-plus-interleave for row counts 1, 5, 127, 128, 129, 8,192, and a legal tail;
- real Layer-0 and representative late-layer Gate/Up outputs;
- graph capture and MTP verifier behavior unchanged;
- register/spill comparison for both FP32-input and physical-BF16-input quantizers;
- workspace report confirms the compact production-scale region is reduced.

### Performance gate

Promote only if the interleave launches disappear from Nsight and repeated 8K/16K prefill improves with statistical
support, or if a qualifying memory-foundation reduction is achieved with neutral performance.

## 13. S04 — prefill chunk sweep

### Admission

Run only after S02 and S03 are resolved. Perform a dry-run arena calculation for 8,192, 12,288, and 16,384 tokens
before allocating or benchmarking. Reject any candidate that cannot retain 700 MiB directly measured free memory
with Target, assistant, K/V, sampling, graphs, media regions, and the selected 16K workload resident.

### Method

1. Build separately identifiable candidates for chunk sizes 8,192, 12,288, and 16,384.
2. Use a temporary source/build-time selector only for the experiment; do not add a public runtime flag.
3. Run synthetic 128/512/2K/8K/16K prefill and the exact Wikipedia 16K prompt.
4. Keep the assistant resident and capture continuous memory, clocks, power, and thermals.
5. Record chunk count, CUTLASS weight-preparation count, scale-interleave count, phase time, and TTFT.
6. Run chunk-boundary correctness at `chunk - 1`, `chunk`, and `chunk + 1`, plus repeated image/audio prompts whose
   media spans must remain inside valid chunks.
7. Select one checkpoint-FP8 production constant. Remove experimental selectors and other variants.

### Promotion gate

The winner must improve the exact 16K end-to-end prefill distribution, retain all output and media-prefix gates, and
meet the memory reserve. If neither 12K nor 16K wins, retain 8K and record the negative result.

## 14. S05 — reprofile and admit deeper work

After S01-S04, collect a fresh 16K profile. Publish this table before selecting S06/S07:

| Family | Total time | Share | Launches | DRAM bytes | L2 hit rate | Dominant stalls |
|---|---:|---:|---:|---:|---:|---|
| FP8 Q/K/V/O projection | | | | | | |
| Q/K/V normalization and K/V quantization | | | | | | |
| NVFP4 activation quantization | | | | | | |
| NVFP4 weight preparation | | | | | | |
| NVFP4 GEMM | | | | | | |
| local prefill attention | | | | | | |
| global D512 prefill attention | | | | | | |
| final norm/output head | | | | | | |
| host/API gaps | | | | | | |

Admit at most two of the following candidates. A candidate must account for at least 5% of primary-workload time or
have a profile-backed recoverable bound of at least 1% end to end. For latency-hiding candidates, Nsight Compute
must identify the matching stall class. Reprofile after every promotion because shares and Amdahl bounds change.

## 15. Conditional medium candidates

### S06A — one combined Q/K/V CUTLASS projection

Admission: FP8 Q/K/V projection remains a leading family and profiles show repeated activation reads or launch/
prologue cost.

Implementation:

- arrange Q/K/V weight rows and channel scales contiguously in the sole final Target arena;
- local layers use one `N=8192` projection; global layers use one `N=8704` Q/K projection;
- expose Q, K, and V outputs as offsets into one contiguous output region;
- preserve global raw-K reuse for V while keeping K normalization/RoPE and V normalization separate;
- retain direct T=1/T≤5 bindings as interior pointers into the same immutable allocation;
- do not create a second persistent Q/K/V copy.

Gate: exact projection bytes and scales, unchanged final logits, lower projection family time, and an end-to-end
prefill win. If larger-N scheduling or loader locality loses, revert completely.

### S06B — physical BF16 Q and direct FP8 K/V boundaries

Admission: normalization/quantization traffic remains material after S06A is independently resolved.

Implementation:

- store normalized Q physically as BF16 and add BF16-input local/global attention paths;
- compute K RMSNorm plus RoPE in FP32, round to BF16 exactly, reload that BF16 value in FP32, divide by the
  checkpoint K scale, and write E4M3 directly;
- compute V RMSNorm with the same BF16-round/reload/divide/E4M3 sequence;
- keep MTP verification and BF16 correctness mode on their existing FP32 paths;
- remove full-token FP32 K/V normalized regions from production and retain only verifier-sized storage.

The current 8K maximum-shape allocations contain 256 MiB Q norm plus 64 MiB each for K and V. The expected
production reduction is about 256 MiB: half of Q plus both K/V FP32 regions. Prove exact FP8 cache bytes before any
performance claim.

Gate: byte-identical Q boundary and K/V cache payload, attention output within existing qualified semantics,
material traffic reduction, and an end-to-end win or qualifying memory-foundation result.

### S06C — reduce global GQA normalization barriers

Admission: the global split kernel remains limited by barrier or synchronization stalls.

Step 1 removes only the explicit per-head barrier after `DecodeBlockSum` when racecheck and dependency analysis prove
the next reduction barrier safely publishes `inverse_sum_shared`. Step 2, attempted only if Step 1 wins but leaves
barrier pressure, normalizes 2 or 4 heads per synchronization phase while preserving each head's shuffle and
addition order.

Gate: racecheck, bit-identical partial LSE/output where expected, register/shared-memory report, and long-context
ordinary/D2 end-to-end improvement. Test 2 and 4 heads separately; retain only one winner.

### S06D — double-buffer global GQA K/V staging

Admission: Nsight Compute reports material long-scoreboard or memory-dependency stalls in the global GQA split
kernel after merge and normalization work.

Implementation keeps the 16-token tile, 512-token split, head assignment, score order, reduction tree, and merge
geometry. Add one alternate raw-FP8 shared tile, issue the next `cp.async` group while computing the current tile,
then wait and swap. Reject if occupancy, barriers, or register pressure erase the overlap.

Gate: no output-ID change at 16K/32K/64K, no race, reduced admitted stall/time, and an ordinary plus D2 long-context
win.

### S06E — active-context decode graph tiers

Admission: a real product session configured for 64K/128K/262K spends at least 1% end to end launching split CTAs
that immediately return at a much smaller active position. This candidate is not admitted by the fixed 16K
benchmark alone.

Capture a bounded set of split tiers while retaining one physical K/V capacity. Ordinary decode may select the tier
from the host-known position; chained D2 requires a device conditional route. Account every extra graph-private byte
and retain fixed addresses. Promote only if representative large-capacity/small-active-context sessions win and the
700 MiB reserve remains intact.

## 16. Conditional deep candidates

### S07A — remove full NVFP4 weight preparation

Admission: cumulative `PrepareWeightKernel` plus related scratch traffic remains at least 5% of 16K prefill or has a
profile-backed end-to-end bound above 1%.

Do not repeat the rejected layer-major chunk traversal. Instead:

1. keep the persistent decode-optimized Row8/K64 codes and scales as the only model layout;
2. prototype Gate only with a CuTe/CUTLASS B-layout adapter that stages the required source tile directly into the
   MMA shared-memory form;
3. compare against current Prepare+CUTLASS on synthetic and real Gate shapes;
4. promote Gate before extending to Up;
5. treat Down separately because its orientation and contracting dimension differ;
6. remove global weight scratch only after all production projections no longer use it.

Gate: exact code/scale interpretation, no persistent copy, no hot-kernel spills without evidence, lower complete
projection time, and an end-to-end win. Keep the existing preparation route as a test oracle, never as a silent
production fallback.

### S07B — share global D512 prefill K/V across a cluster

Admission: global D512 attention remains at least 15% of primary prefill and profiles attribute material bytes or
conversion time to repeated K/V loading across query-head CTAs.

The current kernel is already close to the SM120 per-block register and shared-memory limits. Begin with a resource
model and a two-CTA cluster prototype that preserves four query heads per CTA and shares a leader-loaded raw-FP8
K/V tile through distributed shared memory. This uses two clusters to cover all 16 heads but halves duplicated K/V
loading. Only test eight heads per CTA if the four-head cluster proves that remaining duplication dominates and the
register model is viable.

Use leader-load plus distributed shared memory before trying TMA multicast; do not assume multicast is optimized
for SM120. Preserve the 16-key online-softmax order, FP32 maximum/sum/output accumulation, causal semantics, and
D512 output decomposition. Standard FMHA is not a drop-in replacement for the current D512 PV geometry.

Gate: cluster capability check, complete operator correctness, no scheduling deadlock, measured occupancy and
traffic reduction, and a statistically supported 16K end-to-end gain. Remove the prototype if cluster scheduling or
DSM cost loses.

### S07C — output-head work

Admission: only if the refreshed profile places final norm/output head among the top two recoverable costs.

Do not remove softcap from the primary path based only on monotonicity. FP32 `tanhf` can collapse distinct raw logits
into a tie whose lowest-token rule is observable. Any raw-top candidate must prove an exact fallback condition and
full-vocabulary equivalence. Weight bandwidth is likely dominant, so no implementation begins without profiling.

## 17. Explicitly closed directions

Do not repeat these without a materially new profile-backed hypothesis:

- prefill CUDA Graphs;
- one CTA for the complete local 1,024-token ring;
- eight global query heads per ordinary D512 CTA without a new cluster-sharing design;
- global split size 384;
- unguarded FP8x4 global variants at 16K;
- 32-key global prefill or approximate exponential;
- narrow N64 CUTLASS prefill variants;
- 512-thread or generic Gate/Up fusion;
- layer-major full-model chunk traversal for weight-repack reuse;
- permanent second weight layouts;
- compact T=3 Q/K/V tile-list variants already screened;
- more decode graph/node fusion without a measured host or launch gap.

The ledger, not source comments, is authoritative for the exact prior result and rejection reason.

## 18. S08 — final qualification and sprint closure

### Required final runs

1. Complete host-debug and Blackwell CTest.
2. Targeted Compute Sanitizer memcheck/racecheck for every promoted kernel family.
3. Full prefill matrix at 128/512/2K/8K/16K.
4. Ordinary decode matrix at 128/2K/8K/16K/32K/64K.
5. Exact Wikipedia 16K ordinary and fixed-D2 3/10 qualification.
6. Sampled-MTP identity and resident two-turn chat regression.
7. Long-context deterministic checksum at 16K and at least one 64K run.
8. Final controlled gem16/vLLM/llama.cpp comparison.
9. Windows host/CUDA regression and one adjacent Windows 16K characterization for cross-platform confidence.
10. Direct memory-reserve measurement with Target, assistant, final chunk plan, graphs, media regions, and sampling
    workspace resident.

### Required documentation

- append every promoted and rejected candidate to `PERFORMANCE_LEDGER.md`;
- add ADRs for any changed execution layout, arithmetic boundary, or final stop decision;
- update `ARCHITECTURE.md`, `MEMORY.md`, `CORRECTNESS.md`, `BENCHMARKING.md`, and dispatch validators when behavior
  changes;
- update `ROADMAP.md` with the final 12B result and reactivate 26B M00;
- mark this plan complete with exact winning commits and result paths;
- mark `PREFILL_OPTIMIZATION_PLAN.md` historical.

### Definition of done

The sprint is done only when the final tree contains no losing candidate, no unreported selector, no silent fallback,
no stale validator expectation, and no unexplained numerical or memory delta. The closure summary states one of:

- **Parity achieved:** gem16 meets or exceeds the fresh direct-load vLLM prefill row under the controlled method;
- **Bounded improvement:** gem16 improves but a measured residual gap remains after the stop rule;
- **No promotion:** mandatory candidates did not survive end-to-end gates and the original production path remains.

In all cases, raw evidence is retained. The owner selected the bounded
[Linux short-context ordinary-decode plan](SHORT_CONTEXT_DECODE_OPTIMIZATION_PLAN.md) as the next measured 12B task
before Gemma 4 26B A4B M00.

## References

- NVIDIA CUDA Graphs: <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html>
- NVIDIA Blackwell Tuning Guide: <https://docs.nvidia.com/cuda/blackwell-tuning-guide/>
- NVIDIA CUDA thread-block clusters and distributed shared memory:
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-clusters>
- CUTLASS Blackwell functionality:
  <https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html>

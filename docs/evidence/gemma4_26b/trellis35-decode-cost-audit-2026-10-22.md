# Trellis35 decode cost audit — 2026-10-22

The five agreed audit steps are complete: pinned baseline, short probes, full decode accounting,
targeted operator/selection/llama dispatch inspection, and a concrete first patch definition.
No production optimization was implemented. The first proposed change is an allowed-winner
shortcut in `Gemma4Moe26BReferenceEngine::SelectToken`, supported by observation of all 1,135 output tokens.
The earlier D2 graph/ring fixes remain unchanged. No commit or push was made for this audit.

Machine-readable evidence: `artifacts/trellis35/decode-cost-audit-2026-10-22.json`.
It includes the ten raw samples per cohort, token vectors, model/binary identities, diagnostic
source code, references to exact command records, and a SHA-256 manifest of local raw evidence.
Raw root: `benchmarks/results/2026-10-22/c0a807af/decode-cost-audit/`.
Large profiler reports, SQLite databases and the saved parent binary remain local and ignored.

## 1. Pinned environment and fresh baseline

- Native source: `c0a807afcbcd2528f304bada513afd11d1e9c979`, clean at audit start.
- Native rebuild: `cmake --build build/Linux/blackwell-release --parallel 4`, passed.
  Saved parent: raw-root `gem16-run-baseline`, SHA-256
  `256600faa5e3b94750da4f32265a7bdc45e588cbaf6f7e571debc0e62ffaabae`.
- llama.cpp: the previously refreshed **b11552**, commit
  `23b0202a189c44a54625aadcb37a946dd1d6278d`; executable and all ten shared libraries
  match the October 11 campaign. This audit did not update that pin to a later revision.
  No old llama binary was executed.
- RTX 5080 Laptop, 16,303 MiB; GPU UUID `GPU-93070293-2184-eb69-555b-1d856a4fdbb8`;
  driver 615.71.09, CUDA 13.4.92, GCC 16.2.1, Linux 7.2.9.
  Nsight Systems 2026.3.2.476 and Nsight Compute 2026.3.1.
- Max-power firmware profile, `nvidia-powerd` active, no unrelated CUDA process at checks.
  CPU governor was observed as powersave; settings were not changed.
- Target and assistant Trellis model manifests were verified with `tools/fetch_model.py --verify-only`.
  Q4_0 target and draft GGUF contents were rehashed against their pins. Their SHA-256 values are
  `3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d` and
  `7272d97595f0d4c74bd7b623492b7dbdaafd8b7c72f329a8270ba4eca68f768a`.

Every primary cohort used three warmups and ten retained repetitions, 16,384 identical prompt
token IDs, 1,135 forced greedy outputs and 1,134 measured decode intervals, batch one,
suppression IDs `[1,50,106,258882,258883]`, cold prompt cache, requested context 17,519
(llama allocated 17,664), and resident GPU weights. Native FP8 KV and llama Q8_0 KV are
explicitly different. Native uses fresh processes; llama uses a resident server with a cold
prompt for each request. Loading/plan construction are excluded, retaining the prior harness
timing boundaries. The harness command records include the remaining batch/thread/embedding
settings. Independent NVML telemetry was sampled every 200 ms; each cohort started at <=50 C.

| Cohort, in execution order | Prefill median tok/s | Decode median tok/s | Decode mean 95% CI | Sampled peak VRAM MiB |
|---|---:|---:|---:|---:|
| Trellis ordinary, initial | 5301.00 | 124.546 | 124.517–124.624 | 12455 |
| llama ordinary | 4608.71 | 138.312 | 137.941–138.611 | 15201 |
| Trellis D2 | 5640.55 | 171.745 | 171.644–171.910 | 12725 |
| llama D2 | 4455.50 | 196.551 | 196.370–196.908 | 15561 |
| Trellis ordinary, independent repeat | 5646.99 | 129.002 | 128.987–129.026 | 12455 |

Both native ordinary cohorts are retained. Their active telemetry SM-clock medians were
2,497 and 2,655 MHz respectively; sampled peak powers were 156.62 and 178.05 W.
The repeat reproduces the earlier approximately 129 tok/s result. Within-cohort confidence
intervals do not capture this between-campaign power-state variation. These observations
do not establish a source regression or isolate clock causality. Telemetry includes startup
and warmup and is not a decode-only clock measurement; sampled VRAM is device-wide.

Each cohort is deterministic and reproduces its corresponding prior output vector. Native
ordinary, repeat and D2 outputs match exactly. Cross-engine output/quality equivalence is
not established; llama ordinary and D2 themselves diverge at output index 101.
Different formats, assistants, KV and output sequences limit cross-engine causal attribution.

## 2. Short probes and protected boundaries

Ordinary/D2 probes at 8 and 64 output tokens kept full 17,519 context capacity and matched
their original fixture prefixes. Profiled/unprofiled 8-token and full 1,135-token native
outputs match. All native primary runs report zero fallbacks and no token-loop allocations.

Protected 12B and internal 26B NVFP4 profiles were each run ordinary and D2 for 64 tokens:
exact token equality, zero fallbacks and no token-loop allocations. The existing verified
NVFP4 runtime view was reused; precision and strict loader behavior were not changed.

## 3. Full timeline and dependency accounting

Native captures use `nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none
--cuda-graph-trace=node:nvtx-precapture --capture-range=cudaProfilerApi
--capture-range-end=stop` with `gem16-run --cuda-profile-phase decode`.
Every full native profile executes all 1,135 outputs. Captures, source-scope mappings,
kernel CSVs and executable analysis scripts are retained. Conditional graph node mappings
are calibrated against source anchors and checked for all 30 target layers; assumptions
are recorded in each `.cost.json` rather than treating flattened node IDs as source IDs.

| Diagnostic full trace | Ordinary | D2 |
|---|---:|---:|
| Decode wall ms | 8943.040 | 6670.158 |
| GPU busy interval union ms | 8697.845 | 6598.798 |
| Cross-category overlap ms | 714.201 | 450.731 |
| Unclassified kernel time | 0 | 0 |
| Idle time outside recorded host APIs ms | 1.703 | 70.095 |
| Wall assigned to GPU or known host API state | 99.981% | 98.949% |

GPU kernels are charged by interval union; host synchronization wait is not added again to
GPU work. Inclusive kernel sums and overlap remain separate. This coverage is an accounting
result, not a causal critical-path proof for the entire engine.

For MoE, `critical_moe.py` additionally reconstructs the source fork/join:
input norms -> {shared MLP, routed experts} -> postnorm/residual. The routed branch finishes
last in all 34,020 ordinary and 15,060 D2 blocks. Its work is charged once, including work
overlapping the shorter shared branch; dependency/scheduling gaps remain visible.
Sequential winning-branch operations and completion before the join are asserted.

| Observed MoE terminal-path component | Ordinary ms/output interval | D2 ms/verification group |
|---|---:|---:|
| Input norms/router transform | 0.18912 | 0.25294 |
| Router projection/top-k | 0.42338 | 0.45904 |
| GateUp input transform | 0.05998 | 0.07199 |
| Activation quantization | 0.09737 | 0.09453 |
| GateUp codec/MMA/inverse | 1.27305 | 3.24172 |
| GELU/down transform/quantization | 0.07107 | 0.06981 |
| Down codec/MMA/inverse | 0.67675 | 1.61327 |
| Expert reduction | 0.03088 | 0.03314 |
| Postnorm/residual | 0.29041 | 0.29854 |
| Observed dependency/scheduling gaps | 0.06869 | 0.05608 |
| Total observed MoE span | 3.18072 | 6.19107 |

D2 groups emit a variable number of tokens; those columns have different denominators
(1,134 ordinary intervals versus 502 D2 groups). They are not per-token speed comparisons.
Outside MoE, ordinary QKV/O, local attention, global attention and target head kernel sums
are 1804.652, 875.444, 638.844 and 561.062 ms. D2 additionally has assistant projections,
attention and head sums of 298.778, 190.162 and 183.288 ms. These sums must not be added
to overlapping MoE sums to invent a wall-time total.

Eight-unit windows were selected from the full traces, without reducing context capacity:
ordinary unit starts 0/567/1126 have 7.8793/7.8835/7.9035 ms per interval;
D2 group starts 0/251/494 have 13.0499/13.3033/13.2802 ms per group. Group input positions
are retained in the group-case record, including the final window near the full context.
Contention, branch choice and scheduling must be remeasured after any optimization.

## 4. Targeted selection, D2, operators and llama dispatch

### Ordinary selection

`src/cuda/engine/detail/gemma4_26b_mtp.inc:215` calls `Prediction()` before handling the
suppression mask. The existing no-mask shortcut is at line 263. With the benchmark mask,
it launches `ArgmaxKernel` from `src/cuda/output_head.cu`, one CTA of 256 threads scanning
262,144 logits, then performs another token readback and synchronization.

Across 1,134 profiled decode intervals, that extra GPU kernel has median 361.562 us and
mean 360.753 us. From completion of the first status transfer to selection return, the
mean interval is 374.785 us; the non-argmax tail contributes mean 14.032 us. The mandatory
first `Prediction()` GPU wait is excluded from this proposed saving.

A separate read-only observer of the engine's existing DtoH transfers and synchronization
completions records **1,135/1,135 unmasked winners already allowed**, identical to selected
tokens and the primary fixture. It adds no GPU work or inference-loop file I/O. A separate
one-token correctness probe adds the original winner 61684 to the mask and observes
the valid replacement 119308. That changed-mask probe is excluded from performance.

### D2 acceptance and router union

The full run has 502 verification groups, accepted-draft histogram **[138,96,268]** for
0/1/2 accepted drafts, 632 accepted out of 1,004 proposals, and no ordinary tail.
Those groups emit 1,134 outputs after the initial token. The proposal observer reads the
existing proposal buffer only after the chain completes, with preallocated host storage
and no graph mutation. Its run and the existing router-overlap diagnostic are excluded
from primary timing and reproduce all primary native tokens.

The 15,060 layer samples contain 361,440 routed assignments and 227,685 unique experts:
mean union 15.1185, potential assignment reuse 133,755. Exact proposals, positions,
accepted counts, outputs and union histogram are retained. This motivates the T3 union-16
resource probe; reuse opportunity alone is not a measured speedup.

### Standalone NCU resources

Existing Trellis fixtures were profiled at native M1 and T3 unions 8/16/24, two projection
kernels each. M1 is selected by the actual `<bool1,bool1>` native specialization; no
fictional `--profile-m1` switch is used. Sections include LaunchStats, Occupancy,
SchedulerStats, MemoryWorkloadAnalysis, SpeedOfLight and InstructionStats, with additional
M1/union-16 WarpStateStats and explicit DRAM counters. Local spilling requests are zero.

The additional M1 GateUp capture uses 55 registers/thread, 46.842% achieved occupancy,
63.183% issue activity and 1.881 eligible warps. T3 union-16 GateUp uses 42 registers,
59.457% occupancy, 61.090% issue activity and 1.668 eligible warps. M1 GateUp/Down
kernel replay times are 76.448/38.560 us; union-16 times are 156.512/78.720 us.
These are synthetic fixtures under kernel replay, `--cache-control all` and default NCU
clock controls, not real-activation end-to-end costs or whole-model DRAM traffic.
WarpStateStats ratios retain their native units. Queried average-warp-latency `.pct`
fields are not interpreted as bounded percentage shares.

### Executed llama decode path

The pinned server is traced after two full cold-prompt warmups. A client NVTX range begins
after the first nonempty streaming response chunk and ends on completion. Analysis filters
that conservative decode-only window; no prefill dispatch is extrapolated into decode.
A partial initial assistant step may precede the marker. All 1,135 outputs match the
corresponding primary llama cohort. Streaming, profiling and transport timings are diagnostic.

- MoE weighted reduction, `topk_moe_cuda<128,false>` and scale-enabled router RMSNorm
  execute 34,020 times ordinary and 14,760 times D2.
- D2 uses `mul_mat_vec_q_moe<ggml_type2,int2,bool0>` 29,520 times. Source verifies that
  `int2` is **rows per block**, and `bool0` means **has_fusion=false**. Separate GELU
  kernels execute. The Q4_0 GGUF already has concatenated `ffn_gate_up_exps.weight`;
  this does not imply the extra Gate/GEGLU epilogue fusion suggested in the Pro analysis.
- Ordinary local attention uses Q8_0 vector attention, global attention FP16 MMA;
  actual Q8 KV-to-FP16 conversion executes. D2 local/global target attention uses
  the observed FP16 MMA specializations; assistant vector attention is also present.
- The safe decode window still contains 5 ordinary / 11 D2 capture-and-update pairs,
  with zero graph instantiations. Graph launches plus updates account for 1,134 ordinary
  steps and 1,476 D2 steps (3 x 492). Graph caching does not eliminate every recapture.

These dispatch facts refine the hypotheses; they do not attribute the historical llama
speed increase to particular commits. Old llama was not rerun. No cross-engine forced
replay or per-layer real-activation K3/K4 counter export was added. Those remain necessary
before selecting a later projection/kernel redesign, but not for the selection shortcut.

## 5. First patch definition and acceptance gates

The smallest first experiment changes only the non-sampling branch of
`src/cuda/engine/detail/gemma4_26b_mtp.inc`:

1. Call `Prediction()` exactly as today, retaining finite/router checks and its public result.
2. Test whether its winner occurs in the **active prefix** of
   `configured_suppressed_token_ids`, bounded by `suppressed_token_count`.
3. Return an allowed winner immediately. If it is blocked, execute the existing masked
   argmax/readback/control behavior unchanged.

The host array is fixed and zero-padded; searching its entire capacity would incorrectly
treat token 0 as suppressed. For finite post-softcap values and the existing lowest-ID
tie rule, removing other candidates cannot change an already allowed global winner.
No new kernel, arena field, device status ABI, allocation or changed self-feed semantics
is required. Sampling, penalties, `Prediction()` behavior and the blocked-winner fallback
remain outside this first slice. This patch is **defined, not implemented or benchmarked**.

The observed avoidable interval is about **0.375 ms/output**. The repeated native baseline
needs about **0.522 ms/output** to reach this llama ordinary median. Pure subtraction
would give about 135.6 tok/s, but that is an illustrative upper-budget calculation,
not a measured result or guarantee: profiling, host membership checks and power states
prevent transferring the interval directly. This first patch alone may not close the gap.
The larger dual-winner softcap fusion is deferred: this fixture already has no blocked
winners, and the remaining non-argmax tail is only about 14 us.

Before accepting a candidate:

- Cover empty/nonempty masks, allowed/blocked winners, multiple suppressed maxima,
  ties after softcap, allowed token 0 despite padding, repeated selection and
  `Prediction()`/self-feed ordering. Preserve nonfinite rejection and sampling/penalty paths.
- Require 8/64/full-1,135 native ordinary/D2 identity, 12B and internal NVFP4 regressions,
  zero fallbacks/loop allocations and proportional sanitizer checks.
- Prove the extra argmax/readback disappears for allowed winners and remains for blocked
  winners. Run paired short screens, then 3+10 AB/BA campaigns with comparable clocks;
  retain both independent confirmations and all raw telemetry.
- Use the saved parent binary. Change one factor, disclose timing/format boundaries,
  remeasure overlap/dependencies, and record new evidence without replacing this baseline.

## Completed checks and diagnostic setup limitations

The following exact CTest selectors passed under `ctest --test-dir
build/Linux/blackwell-release --output-on-failure -R ...`:

| Selector | Result |
|---|---|
| `^gem16-cuda-(conditional-graph\|trellis35\|attention-reference\|nvfp4-consumption)$` | 4/4 passed, 5.03 s |
| `^gem16-cuda$` | passed, 30.76 s |
| `^gem16-(unit\|responses-allocation)$` | 2/2 passed, 0.82 s |
| `^gem16-12b-m22-product$` with explicit `GEM16_12B_MODEL` fixture | passed, 7.78 s |

`python -m unittest tests.python.test_benchmark_wikipedia_workload
tests.python.test_benchmark_gemma4_26b_wp8_text` passed all 18 tests. Seven additional
diagnostic-accounting tests pass: overlap, same-category parallelism, host waits, clipping,
routed work overlapping shared work, shared-branch wins, and rejected invalid dependencies.
`compute-sanitizer --tool memcheck --error-exitcode 99` reports **zero errors** for the
conditional graph fixture, streaming ring fixture, and native Trellis D2 with the original
16K prompt, full 17,519 context and eight outputs (247.94 s for the full-model check).

An initial 12B test skipped because its fixture environment was absent; only the explicit
qualified rerun counts above. NCU's `.ncu-repz` extension, a SQLite mtime check and an
observer's initial assumption that a nonzero finite flag must literally equal 1 were
corrected in diagnostic setup. Original logs remain; invalid observer metadata is excluded.
There was no corresponding engine failure. No new Windows, media or broad quality
qualification is claimed. Runtime source is unchanged; only this report and the audit
record are added to the repository worktree.

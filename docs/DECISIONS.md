# Decisions

## 2026-07-29: Qualify sampled MTP chat before multimodal work

Date: 2026-07-29
Decision: Make sampled MTP in resident `gem16-chat` the next product gate, ahead of multimodal expansion and any
N-Gram proposer. The first qualified mode uses the target checkpoint's recommended `temperature=1.0`, `top_k=64`,
and `top_p=0.95` settings, seeded GPU sampling, fixed D2 target verification, asynchronous token streaming, and
resident multi-turn state. Assistant outputs remain deterministic proposals. For a fixed seed, each verifier row
uses the same target sampling step and committed repetition history as ordinary sampled decode; only the longest
proposal prefix equal to those target-selected tokens may be accepted. A mismatch emits the target sample and
invalid later rows are discarded. This seed-identical target-sampling contract is the first implementation; do not
reuse greedy argmax acceptance or claim standard probability-ratio speculative sampling without assistant
probabilities and independent distribution evidence.
Context: Greedy fixed-D2 MTP and its mapped-pinned streaming graph are qualified, while chat currently rejects MTP
whenever sampling is enabled. The target sampling plan already implements the checkpoint's recommended controls,
full-history repetition handling, suppression, deterministic RNG, and a whole-model CUDA Graph. The assistant only
materializes argmax proposal IDs, not proposal probabilities, so the usual `min(1,p/q)` rejection sampler cannot be
implemented correctly by relabeling the existing path. The user explicitly prioritized production sampled MTP
chat, a real multi-turn model test, and Linux graph qualification before multimodal work.
Alternatives: Keep MTP greedy-only; accept assistant drafts using greedy target IDs and sample only on mismatch;
implement probability-ratio speculative sampling without assistant distributions; or start multimodal work first.
The first misses the intended Gemma chat behavior, the next two change the requested target distribution, and the
last conflicts with the new product priority.
Consequences: Sampled verification needs full target logits for each fixed verifier row, row-specific repetition
state, monotonically advancing target sampling steps, transactional commit of the accepted repetition state, and
sample-aware device control for GPU chaining. Qualification compares ordinary and MTP outputs exactly for each
seed over a multi-seed suite, then covers resident multi-turn continuation, stop/tail positions, ring wrap,
streaming order, allocation accounting, and Linux 3-warm-up/10-run performance. MTP may be slower or accept fewer
proposals under sampling; telemetry must disclose this and ordinary sampled fallback remains required. Chat should
use checkpoint generation defaults only after they are parsed and validated rather than silently hard-coded.
Implementation may split internal `.cuh` fragments while retaining one CUDA translation unit. The source-size goal
remains below 1,000 lines where practical; files up to 2,000 lines are acceptable when a functional split would
harm locality or generated code, and files above 2,000 require an explicit follow-up.
Evidence: The pinned target `generation_config.json` and model card both specify sampling with temperature 1.0,
top-k 64, and top-p 0.95. The existing sampler keys SplitMix64 by seed and output step, and the verifier already
retains transactional target K/V and hidden state. The assistant currently exposes only device draft token IDs.

## 2026-07-28: Reuse exact greedy MTP in resident chat sessions

Date: 2026-07-28
Decision: Let `ConversationSession` optionally own the pinned official assistant and expose
`--assistant-model`, `--mtp-draft-tokens 1|2|4`, and `--mtp-adaptive` through `gem16-chat`. Fixed D2 reuses the
qualified GPU-chained graph and mapped-pinned callback ring. `--stats` makes the active mode, effective decode
throughput, draft counters, verifier groups, and GPU chaining visible per turn. MTP chat remains greedy-only.
Context: One-shot `gem16-run` already proved exact target verification, GPU acceptance/commit, stop/tail handling,
and asynchronous streaming. Conversation sessions already retain exact target KV and validate that every rendered
turn extends the materialized token prefix. The missing boundary was assistant ownership and MTP scheduling after
`PrefillAt` on a later suffix.
Alternatives: Reload through `RunGreedyInference` for every message; expose only one-shot MTP through the chat
frontend; or silently fall back to ordinary chat. Reloading discards resident weights and KV, one-shot execution is
not live multi-turn chat, and silent fallback would make MTP testing misleading.
Consequences: Target and assistant weights, graph allocations, and cache remain resident for the session. At turn
completion the host prefix records all generated IDs except the final not-yet-forwarded ID, exactly matching the
committed target KV state. One-shot JSON reports MTP counters, while interactive `--stats` reports them per turn.
Sampling with MTP fails before model loading. Ordinary and sampled chat behavior is unchanged.

## 2026-07-29: Complete the GPU-controlled decode graph before multimodal expansion

Date: 2026-07-29
Decision: Make a fully GPU-controlled greedy MTP decode graph with asynchronous token streaming the next
architectural milestone before multimodal expansion. Deliver it incrementally: first add a device-resident MTP
control record while retaining the current host loop and checking host/device state parity; then capture one
complete fixed-D2 group; then chain fixed-D2 groups on the GPU; then add stop and tail handling; then add a
preallocated asynchronous GPU-to-host streaming ring; finally add adaptive D1/D2/ordinary conditional branches.
After that fixed graph is stable, evaluate an optional device-resident `ngram-mod` proposer as a higher-priority
conditional branch that can skip MTP on a measured hit and reuse the same exact target verifier. No step may change
target arithmetic, kernel order within a row, acceptance, committed KV state, or ordinary/MTP greedy identity
merely to make capture easier.
Context: Active D2 already keeps both assistant proposals, batched target verification, GPU acceptance, and KV
commit device-resident, but copies one compact `MtpGroupResult` to pinned memory and synchronizes the compute
stream after every group. Host code then updates the next token, processed position, stop state, callbacks, and
adaptive scheduler before launching another group. Fixed arenas, stable addresses, ordinary decode's existing
device control record, and GPU acceptance make removal of this final per-group control roundtrip feasible. The
user selected host-independent decode and streaming as the long-term direction and requested small independently
validated steps. The current exact 47.432 tok/s 3-warm-up/5-run characterization remains below the 50 tok/s gate;
graph work is an architectural objective and not a guaranteed performance claim.
Alternatives: Begin multimodal work now; capture only a verifier suffix; build one monolithic persistent kernel;
or remove the host synchronization without moving variable token/position/stop state to the GPU. Multimodal work
would postpone a foundational decode boundary, the suffix graph was already exact but neutral, a monolithic
kernel is premature, and asynchronous host launches alone cannot safely resolve variable acceptance.
Consequences: Multimodal implementation remains queued until the fixed-D2 GPU loop and nonblocking streaming
boundary are complete or a documented blocker changes this decision. Phase one must introduce `MtpDeviceControl`
with GPU/host parity assertions while preserving the current production scheduling. Later graph phases use fixed
arena addresses and conditional graph execution where supported by the pinned toolchain. Streaming uses a
preallocated single-producer/single-consumer ring: GPU compute publishes verified target tokens, a host poller
invokes callbacks, and the compute stream does not wait on the host during normal operation. Backpressure, EOS,
maximum length, D1/ordinary tail execution, context limits, and output ordering must be explicit. Every phase must
pass the fixed 1,135-ID hash and 632/372 counters, ring-wrap and stop tests, allocation checks, resource accounting,
and before/after profiling; only a complete 3/10 run may qualify a performance headline. Phase one remains the
narrow MTP control implementation; it must not add an unused generic proposer framework. If N-Gram later passes its
hit-rate and end-to-end gates, evolve only the common token/position/draft-count prefix into a measured
multi-proposer control and keep N-Gram's fixed hash storage in a separate arena region.
Evidence: Current `GenerateAssistantDraftsDevice` has no intermediate host synchronization, while
`VerifyAcceptCommitAssistantBatch` ends in one D2H result copy and `cudaStreamSynchronize`; ordinary decode already
uses a device `DecodeControl`; the exact suffix-graph experiment and current Nsight profiles bound both feasibility
and performance uncertainty. Current llama.cpp Wikipedia screens show that conservative N-Gram configurations
produce no drafts and aggressive short matches reduce MTP throughput, so optional N-Gram integration requires
fresh gem16-specific evidence rather than assumed benefit.

## 2026-07-29: Reopen exact MTP for structural verifier work

Date: 2026-07-29
Decision: Reopen exact D2 optimization at the user's explicit direction, while retaining the binding requirement that
MTP emit exactly the ordinary greedy token sequence. Restrict the reopened work to structural, profile-supported
changes with enough modeled headroom to reach 50.0 tok/s; do not investigate or modify llama.cpp, weaken target
arithmetic merely to copy another runtime, or proceed toward 55 tok/s before 50 qualifies. The first candidates are
a three-row BF16 Tensor-Core target output head and a local-attention verifier that reads tentative K/V directly
and reuses historical K/V across rows without changing causal visibility.
Context: The retained exact result is 46.422 tok/s and needs approximately 3.5 ms less per D2 group to reach the
45.18 ms/group gate. The last profile assigns roughly 4.8 ms/group to the target head and 6.9 ms/group to local
attention; unlike the rejected CTA-size probes, either structural candidate can plausibly cover a material part of
the gap. The user explicitly chose continued gem16 work and ordinary/MTP identity over llama.cpp investigation.
Alternatives: Keep the sprint closed and move to multimodal; study llama.cpp's divergent verifier; repeat small
projection geometry probes; or accept numerically different MTP output. These conflict with the current product
direction, lack modeled headroom, or violate exact speculation.
Consequences: The closure decision immediately below remains the provenance of the 46.422 result but no longer
blocks bounded structural experiments. Every candidate must retain the 1,135-token SHA-256
`43bc3380fc1cce5182a679fa3a340c04bcc79c52e73d5102ec1f737f57d0a1e1`, the 632/372 acceptance counters, zero
fallback/allocation, and acceptable kernel resources. A result at or above 50.0 tok/s still requires the full
alternating 3-warm-up/10-run qualification before any 55 tok/s work.
Evidence: User direction in the active development session, the clean-head exact D2 profile, and the retained
Wikipedia characterization at `fffefcb`.

## 2026-07-29: Close the exact MTP verifier sprint below the 50 tok/s gate

Date: 2026-07-29
Decision: Retain the exact three-row global-attention, FP8 Q/K/V/O, output-head, and NVFP4 Down improvements, but
close the bounded verifier sprint without claiming qualification. The full Wikipedia candidate reaches 46.422
effective tok/s and remains below the binding 50.0 tok/s minimum. Do not continue toward 55 tok/s or adopt
numerically different causal-prefill verification. Move the active roadmap gate to multimodal expansion.
Context: The retained Direct-O candidate reached 44.347 tok/s. The final combination preserves all 1,135 ordinary
IDs, the fixed output hash, 632 accepted and 372 rejected drafts over 502 groups, and zero fallback/allocation.
Profiles still attribute the remaining time broadly across exact attention, NVFP4 projections, FP8 projections,
the BF16 target head, and the official BF16 assistant; no remaining bounded candidate covers the required group
reduction. The tested causal-prefill route is faster but changes output and remains invalid.
Alternatives: Run a 3/10 qualification below the gate; continue unbounded tuning; weaken exactness; or claim the
external vLLM result. These violate the explicit qualification threshold, bounded-sprint policy, ordinary/MTP
identity, or external-characterization disclosure.
Consequences: 46.422 tok/s is a one-run exact characterization, not a qualified headline. The retained kernels
remain observable and tested, while rejected local-attention, grouping, CTA, assistant-head, and raw-logit
experiments are absent. Future MTP work requires a new written, profile-supported decision after the multimodal
milestone rather than silently reopening this sprint.
Evidence: Clean-head and final Nsight profiles, the full exact Wikipedia candidate, CTest, kernel resource reports,
and the 2026-07-29 performance-ledger entry.

## 2026-07-29: Set the exact MTP minimum at 50 tok/s and stretch target at 55 tok/s

Date: 2026-07-29
Decision: Replace the former 60 tok/s stretch gate with a qualified minimum of 50.0 effective target-verified tok/s
and a stretch target of 55.0 tok/s on the fixed 16K Wikipedia workload. Qualification still requires exact ordinary
identity over all 1,135 output IDs, three alternating warm-up pairs, ten alternating measured pairs, no fallback or
token-loop allocation, and complete memory/resource evidence. Reaching 50 permits only further bounded, material,
exact candidates toward 55; it does not reopen unrestricted text optimization.
Context: Current gem16 D2 is 42.639 tok/s with 52.98 ms per verifier group and mean accepted drafts 1.259. At the
same acceptance, 50 and 55 tok/s require at most 45.18 and 41.07 ms/group, reductions of 14.7% and 22.5%. Current
llama.cpp reaches 48.38 tok/s in the controlled fixed-1,135-token D2 screen, so 50 tok/s exceeds that observed path;
55 tok/s provides a clearer margin and also exceeds its 50.21 tok/s stop-terminated characterization. External MTP
outputs remain non-exact and therefore provide performance bounds rather than correctness evidence.
Alternatives: Retain 60 tok/s as the minimum practical objective; accept the existing 42.639 tok/s result; or define
success against llama.cpp's stop-terminated run despite different outputs. The first demands an unnecessary 28.9%
group-latency reduction, while the latter two do not satisfy the stated competitive objective or benchmark parity.
Consequences: The exact-verifier sprint first targets 45.18 ms/group. If 50 tok/s qualifies, profile-proven exact
work may continue toward 41.07 ms/group and 55 tok/s. If bounded candidates cannot reach 50, retain 42.639 as a
correct characterization but mark the MTP performance target unmet rather than claiming success.
Evidence: Qualified gem16 Wikipedia result, fixed-length current llama.cpp D2 characterization, and the external
MTP feasibility summaries in `benchmarks/baselines/{llama_cpp,vllm}/mtp-characterization.json`.

## 2026-07-29: Use external MTP only as a hardware bound and allow one final exact-verifier sprint

Date: 2026-07-29
Decision: Treat patched graph-vLLM and current llama.cpp Gemma 4 MTP as non-exact performance characterizations,
not baselines. The 60 tok/s threshold in this entry is superseded by the 50 minimum/55 stretch decision above.
Use vLLM's fixed-length 35.75 ms/group result as evidence that 60 effective tok/s is physically
possible at gem16's acceptance, and permit one bounded profile-driven sprint on gem16's exact target verifier.
Do not adopt either runtime's numerically different batched target semantics. If the sprint cannot materially
close gem16's 52.98-to-37.65 ms/group requirement, end text optimization and proceed to multimodal.
Context: Exact gem16 D2 reaches 42.639 tok/s at mean accepted drafts 1.259. Patched vLLM reaches 57.390 tok/s over
3 warm-ups/10 runs and 57.363 tok/s at fixed 1,135-token length despite lower acceptance, because its verifier
groups take 35.75 ms. Current llama.cpp reaches 48.38 fixed-length D2 tok/s. vLLM ordinary/MTP first differ at
fixed index 2; llama.cpp first differs at index 133 and also uses BF16-mapped attention plus Q8_0 KV.
Alternatives: Declare 60 impossible from gem16's current cost; copy vLLM's faster numerical route; fully qualify
llama.cpp as parity; or optimize indefinitely. The first ignores measured hardware headroom, the next two violate
exactness/format disclosure, and the last violates the bounded optimization policy.
Consequences: The remaining performance objective is verifier-group latency, not acceptance or host scheduling.
A successful candidate still requires exact 1,135-token identity and normal 3/10 qualification. External numbers
cannot become correctness evidence or headline cross-engine speedups.
Evidence: `benchmarks/baselines/vllm/mtp-characterization.json`,
`benchmarks/baselines/llama_cpp/mtp-characterization.json`, direct official-assistant conversion, target-Layer-46/47
shared-KV logs, fixed-output screens, vLLM 3/10 results, and continuous vLLM telemetry.

## 2026-07-28: Use GPU MTP transactions and decode-sized verifier projections

Date: 2026-07-28
Decision: Keep assistant drafts on device, perform acceptance/stop handling and tentative K/V/hidden commit on GPU,
and return one compact pinned group result. For verifier T≤5, dispatch FP8 Q/K/V through the latency-oriented
decode-order MMA and NVFP4 Down through one unstaged 16-token/four-warp tile. Keep exact CUTLASS O, the existing
fused Gate/Up, and split-online attention. Make context/acceptance adaptation explicit through `--mtp-adaptive`;
leave explicit D1/D2/D4 unchanged. Do not retain the tested verifier-suffix CUDA Graph.
Context: The restored 16K D2 path measured 35.184 tok/s. Nsight showed that host waits represented GPU work and
that projection kernels designed for 128-token prefill tiles dominated the three-row verifier. It also showed that
GPU acceptance was primarily a prerequisite for future scheduling rather than the main bottleneck.
Alternatives: Keep host draft/acceptance transactions; use CUTLASS Q/K/V or NVFP4 Down; retain suffix graphs;
change attention reduction order; or force one draft length globally. These respectively preserve two syncs,
violate long-context exactness, add memory without speed, risk greedy drift, or lose workload-dependent fallback.
Consequences: Qualified Wikipedia 16K D2 rises to 42.639 median tok/s versus 31.798 ordinary (1.341x), with all
1,135 IDs exact in ten alternating measured pairs. Workspace adds 15,360 bytes at context 128; sampled 16K peak is
10,838 MiB. One host synchronization remains per group for callbacks and variable output length. Full controlled
MTP graph capture remains deferred because the exact suffix graph did not improve throughput.
Evidence: 3 alternating warm-up pairs, 10 alternating measured pairs, Nsight scoped kernel attribution, resource
capture with no per-thread local memory, native QMMA/OMMA SASS, Transformers drafts, D1/D2/D4 FP8 identity, BF16,
ring-wrap, stop/adaptive tests, CTest, memcheck, allocation-boundary query, and raw telemetry.

## 2026-07-28: Preserve direct target Q/K/V and use split-online assistant attention at long context

Date: 2026-07-28
Decision: MTP target verification must use decode-order direct grouped FP8 Q/K/V; CUTLASS remains enabled only for
its byte-qualified O batch. For FP8 assistant cache capacities above 512, use the qualified split-online decode
attention kernel with the constant proposal-group position. Keep the reference score path for short FP8 and BF16.
Context: Wikipedia 16K MTP was deterministic but diverged from ordinary at output index 68. One-run D2 isolation
showed that direct Q/K/V restores all 1,135 ordinary IDs, while CUTLASS O remains exact. At long context, assistant
materialized-score attention then dominated proposal cost.
Alternatives: Accept a deterministic batched-target sequence; disable all CUTLASS FP8 verification; retain scalar
assistant attention; or share global Target K/V loads across verification rows. The first violates exactness, the
second discards an exact O win, the third leaves measured cost, and the fourth was exact but slower.
Consequences: Full Wikipedia D2 output is exact and one no-warm-up characterization reaches 35.184 tok/s, +10.7%
against the retained ordinary median but not statistically qualified. Assistant workspace grows by 24,576 bytes at
context 128. Runtime metadata identifies the assistant attention route.
Evidence: Full 1,135-token comparison, D2 Transformers fixture, FP8 ring-wrap memcheck, CTest, one-run candidate
ledger, and rejected 34.767 tok/s global multi-row target-attention result.

## 2026-07-27: Reuse exact fused NVFP4 Gate/Up and FP8 CUTLASS batch projections for MTP

Date: 2026-07-27
Decision: In fixed-shape MTP target verification, keep recursive assistant selected tokens on device within each
draft group, use the existing FP8 CUTLASS batch projection plan, and use the exact native fused Gate/Up/GELU batch
operator followed by exact NVFP4 quantization. Keep the decode-specialized attention verifier and direct NVFP4
Down projection.
Context: The first exact batched verifier trailed ordinary decode. Nsight attributed its largest removable target
costs to separate NVFP4 Gate/Up and FP8 projection work. The existing fused Gate/Up test proves its BF16 boundaries
and product, while direct FP8/NVFP4 alternatives provide a correctness control.
Alternatives: Use CUTLASS for all NVFP4 projections; replace attention with causal prefill attention; or retain
separate Gate/Up. Full NVFP4 CUTLASS changes the natural greedy sequence. Causal-prefill attention reaches 55.06
tok/s but first diverges at output step 15. Separate Gate/Up leaves a measured win unused.
Consequences: A context-512 natural workload improves from 35.270 ordinary to 42.897 MTP tok/s (+21.6%) at mean
accepted length 1.886, while retaining exact target output in short, natural, FP8/BF16, and ring-wrap checks.
This is a conditional workload win only; graph capture, GPU acceptance, and more acceptance-sensitive workload
measurement remain open before a 60 tok/s claim.
Evidence: 3 warm-ups plus 10 alternating runs, Nsight kernel attribution, `tools/validate_mtp.py`, CTest, FP8/BF16
ring-wrap equivalence, and active MTP memcheck. Raw results are retained locally under the ignored result path.

## 2026-07-27: Use fixed-shape batched target verification before graph optimization

Date: 2026-07-27
Decision: Replace serial MTP verification with one fixed-shape causal target batch over `[input, drafts]`, retain
per-layer tentative K/V rows in a separate fixed arena, and commit only the host-confirmed accepted prefix plus the
first mismatch. Use direct batch-native FP8/NVFP4 projections and per-row decode attention to preserve the
qualified batch-one numerical boundaries. Label the path `batched_exact_target`.
Context: Serial verification structurally could not improve target-forward count. Full causal prefill attention
could not be used directly because its target cache writes must remain transactional and its output values differ
from the exact decode route. The bounded maximum draft length is four, so temporary K/V and candidate storage are
small and fixed.
Alternatives: Mutate the live cache with no rollback; copy the entire cache; use a prefill-only semantic route; or
wait for a full graph implementation. These respectively corrupt rejected drafts, waste long-context memory,
weaken output equivalence, or delay measuring the actual target-batch bottleneck.
Consequences: MTP exactly preserves ordinary greedy outputs for tested FP8/BF16 contexts and local-ring wrap while
reducing target launch groups. Host token comparison and commit remain synchronization points, and the current
implementation is not promoted as a speedup because its natural-prompt result still trails ordinary decode.
Evidence: 16-token FP8 draft lengths 1/2/4 and BF16 outputs equal ordinary output. A 256-token natural chat prompt
also retains exact output with D4, mean acceptance 1.89, and 35.44 versus 36.20 ordinary tok/s in one profiling
characterization. Nsight attributes the current verifier primarily to direct target projections and batched output
selection; it guides, but does not qualify, the next optimization.

## 2026-07-27: Qualify MTP assistant execution with serial exact target verification first

Date: 2026-07-27
Decision: Activate the complete BF16 assistant at draft lengths 1, 2, and 4 against target Layers 46/47, but first
verify proposals with one ordinary target forward per emitted token. Emit only the target prediction at each
position and label the route `serial_exact_correctness`; do not claim speculative speedup until batched target
verification and GPU-side acceptance are implemented.
Context: The official assistant architecture and recurrent constant-position semantics require independent
operator validation before cache transactions and batched verification are allowed to affect multiple positions.
Serial verification exercises the real assistant, acceptance, rejection, target cache progression, stop handling,
and both cache precisions while making output equivalence structurally explicit.
Alternatives: Implement batched verification and cache rollback in the first assistant execution change; trust
vLLM output without an independent fixture; or count assistant proposals as output. These combine too many cache
and numerical changes, depend on a runtime with a blocked graph path, or violate benchmark semantics.
Consequences: Correctness MTP performs no token-loop allocation and reports complete acceptance telemetry, but it
cannot provide a throughput speedup because every emitted post-prefill token still requires a target forward.
Sampling, chat, and diagnostic dumps reject active MTP for now. The ordinary decode graph remains mandatory.
Evidence: `tools/validate_mtp.py` matches four recurrent Transformers drafts exactly; draft lengths 1/2/4 retain
the ordinary 16-token FP8 output, BF16 output also matches, local-ring wrap and stop handling pass, and active
memcheck reports zero errors.

## 2026-07-27: Use Google's separate BF16 12B assistant for first MTP support

Date: 2026-07-27
Decision: Pin `google/gemma-4-12B-it-assistant` separately and implement its published BF16 Safetensors payload
before considering community FP8/NVFP4 conversions. Share the target KV states required by its Q-only layers, but
retain the assistant's own 1,024-dimensional tied LM head.
Context: The pinned Unsloth target contains no MTP tensors. Google's official assistant is a four-layer
`gemma4_unified_assistant` with a 3,840-dimensional target interface, 1,024-dimensional internal state, and an
806.54 MiB BF16 payload. Its exact vocabulary mapping matches the target. vLLM loads and wires the pair directly,
confirming checkpoint compatibility; natural-prompt acceptance is promising but strongly workload-dependent.
Alternatives: Treat MTP as embedded target weights; use a community quantized assistant first; quantize the
assistant ourselves; or skip MTP. The first is contradicted by the complete target manifest, the next two weaken
provenance and first-result quality, and the memory estimate does not justify skipping the official model.
Consequences: MTP is an optional second checkpoint and adds about 807 MiB persistent payload but no independent KV
cache. The maximum-context target peak plus payload is about 13,051 MiB before MTP-specific graph/workspace costs,
leaving about 3,252 MiB on the reference GPU. Ordinary decode remains mandatory because poor acceptance can make
MTP slower. Direct-load, exact verification, acceptance, incremental-VRAM, and effective-throughput gates apply.
Evidence: `docs/MTP.md`, the pinned 48-tensor assistant header, lock-file hashes, successful vLLM eager execution,
and vLLM's target Layer 46/47 cross-model KV mapping.

## 2026-07-27: Use one mixed run for the bounded 64K text gate

Date: 2026-07-27
Decision: Close the practical 64K text gate with one exact 65,536-token mixed prompt containing retrieval markers
at 10%, 50%, and 90%, followed by a forced 1,024-token greedy soak. Collect cache/allocation assertions and sampled
GPU resource telemetry in the same run. Do not present its timings as a statistically qualified benchmark.
Context: The prior synthetic 64K characterization already covered repeated prefill/decode timing, while repeating
another 64K workload ten times would add substantial wall time without being needed for a new performance claim.
The remaining question was whether meaningful distant information survives while the production cache executes a
long generation beyond the 64K boundary.
Alternatives: Repeat the full performance policy; run separate retrieval, soak, memory, and allocation jobs; or
skip long-context quality. The first is disproportionate without a performance promotion, the second repeats the
36-second prefill and model load for every check, and the third leaves the active correctness gate open.
Consequences: `tools/qualify_long_context.py` is a bounded correctness/soak harness, not a replacement for the
3-warm-up/10-run benchmark contract. It preserves one raw engine result locally under the ignored results tree.
A future performance headline or regression claim still requires the normal repetition and telemetry policy.
Evidence: The first run returned all three exact codes in order at output offsets 0, 8, and 17, completed all 1,024
requested output tokens with zero fallbacks and no token-loop allocation, and used the hybrid local-ring/global
cache. Peak sampled GPU memory was 10,418 MiB. Prefill/decode took 36,293.4/39,610.3 ms in this single run.

## 2026-07-27: Fuse exact decode boundaries and controlled Q/K processing

Date: 2026-07-27
Decision: Reuse the exact RMSNorm/FP8, RMSNorm/NVFP4, and Gate/Up/GELU/NVFP4 production boundaries in ordinary
decode. Replace separate Q/K projection rounding, per-head RMSNorm, controlled RoPE, and final rounding with one
whole-model-graph kernel that reads the dynamic position from `DecodeControl` and indexes the immutable exact RoPE
tables. Retain the unfused route for diagnostic hidden-state capture and CUDA comparison, not as a production
selector.
Context: A fresh 8K graph-node profile attributed substantial latency to 624 pointwise kernel nodes per token that
repeated already-qualified boundaries or separately launched Q/K processing. Direct projection, attention, and the
tied output head were bandwidth-dominant and offered no similarly low-risk change. The retained vLLM result was
38.056 tok/s versus gem16's 32.586 tok/s at the same 8K matrix point, subject to the documented timing caveats.
Alternatives: Retune NVFP4 block warps; combine attention residual and the next MLP quantization; change projection
rounding inside MMA epilogues; or retain all pointwise launches. Two/eight NVFP4 warps per block were neutral, the
combined residual candidate regressed 1.7%, and rounded projection stores had already regressed. All rejected code
was removed.
Consequences: Production greedy and sampled graphs execute 964 rather than 1,588 kernel nodes per token in the
profiled 8K plan. Persistent weights, KV layout, reusable workspace, quantized bytes, and token-loop allocation
behavior are unchanged. The new Q/K kernel uses 24 registers, 2,048 bytes shared memory, and zero stack/local
memory. Diagnostic forwards retain their prior intermediate values and ordering.
Evidence: Final 3-warm-up/10-run medians are 34.446/34.257/33.545 tok/s at context 128/2,048/8,192. The final 8K
result is 2.10% above the nearby 32.853 tok/s parent and reaches 88.1% of retained vLLM. Graph-node profiling reduces
summed kernel time from 31.517 to 30.447 ms/token despite slower candidate-profile clocks. Qualification added a
required barrier before reusing the RMSNorm/FP8 shared reduction array; final timings include that correctness fix. CTest, exact-blue,
129/257 boundaries, the 12-prompt teacher-forced suite, sampled CPU/GPU validation, targeted memcheck/racecheck,
and deterministic decode checks pass. Peak process VRAM is 9,852 MiB and no CUDA allocation occurs in the token
loop.

## 2026-07-27: Keep greedy unchanged and use exact sorted GPU sampling

Date: 2026-07-27
Decision: Sampling is explicit rather than implicit in temperature defaults. Disabled sampling keeps the existing
fused greedy CUDA Graph and workspace. Enabled sampling applies sign-aware full-history repetition penalty,
suppression, temperature, descending radix sort, top-k, min-p relative to the maximum probability, then top-p,
and draws from a SplitMix64 stream keyed by seed and output step. All recurring state and sort scratch is
preallocated; only the selected token returns to the host.
Context: Phase-one requires temperature, top-k, top-p, repetition penalty, deterministic seeded RNG, and min-p,
without weakening greedy correctness or the no-allocation token loop. Exact top-p over the 262,144-token vocabulary
requires more than the greedy plan's one candidate per block.
Alternatives: Approximate vocabulary filtering; CPU sampling after a full-logit transfer; mutate the greedy graph;
or use a persistent converted candidate format. Approximation and CPU selection violate the primary quality/hot
path contract, while changing greedy would add regression risk. A parallel prefix selector is deferred because the
initial final scan is only about 7 microseconds at top-k 64.
Consequences: Sampling adds 7,408,128 workspace bytes at context 128, including an in-place double-precision
probability scan; repetition history uses an atomic bitset so duplicate prompt tokens cannot create byte-write
races. The filter order and RNG mapping are public
reproducibility contracts. Conversation sessions retain RNG step and repetition history across turns. The sampled
whole-model graph reads its changing step from the copied device control; diagnostic captures retain the direct
path. Greedy output IDs, graph replay, and workspace remain unchanged.
Evidence: The checkpoint-backed CPU oracle, synthetic CUDA operator and graph-replay tests, repeated seeded runs,
top-k-1/greedy equivalence, CTest, and chat/run integration pass. The initial direct-path 3-warm-up/10-run result was
32.596 sampled tok/s versus 32.819 greedy tok/s. After whole-model graph capture and atomic repetition bitset,
final nearby 3/10 runs measure 32.989 top-k-64 sampled and 33.003 unfiltered full-vocabulary sampled versus 32.839
greedy tok/s, all treated as performance parity within run variance. Nsight records one graph
launch per post-prefill token and no token-loop allocation.

## 2026-07-27: Index the contiguous global KV cache directly during prefill

Date: 2026-07-27
Decision: Use each absolute key position directly when global-prefill attention stages older K/V from its
contiguous cache. Remove `cache_capacity` and the redundant runtime modulo from the internal global staging helper;
retain modulo addressing exclusively in the local circular cache. Runtime metadata reports
`global_prefill_fp8_staging` as `async_contiguous_fp8x16_fp8x4_bf16x2`.
Context: After vectorizing local staging, global attention was the largest individual 8K kernel family. The global
helper inherited ring-style indexing even though its launch contract proves `start_position + tokens <=
cache_capacity` and global K/V storage is contiguous. Consequently every valid older `absolute_key` is already
strictly below capacity, making `absolute_key % cache_capacity` identical to `absolute_key` while still requiring
runtime integer division in every staged vector.
Alternatives: Retain defensive modulo; specialize only power-of-two capacities; or change global cache layout. The
first preserves provably dead work, the second would constrain valid context plans, and the third is unnecessary.
A separate paired-BF16 global-query-store experiment was also exact but lost 0.42% in an adjacent 8K comparison
and was removed.
Consequences: Cache bounds validation, allocation, physical layout, K/V bytes, attention arithmetic, shared memory,
and decode are unchanged. The local ring retains its required modulo. The global kernel retains 254 registers,
99,328 reported shared bytes, and zero stack/local memory.
Evidence: Two adjacent parent/candidate 3-warm-up/10-run pairs combine to 3,801.98 versus 3,827.47 tok/s median at
8K (+0.67%), with median TTFT 2,154.66 versus 2,140.32 ms (-0.67%). Nsight reduces global-attention time from
1.03955 to 0.96426 seconds (-7.24%) across two profiled prefills and total kernel time from 4.44462 to 4.39667
seconds (-1.08%). CTest, exact-blue, 129/257 boundaries, and the 12-prompt teacher-forced suite pass unchanged;
an 8K decode regression retains one checksum.

## 2026-07-27: Vectorize local-prefill FP8 staging

Date: 2026-07-27
Decision: Load local-attention K/V rows in aligned 16-byte vectors, decode each packed word with E4M3x4 conversion,
and store paired BF16 values into the existing swizzled shared-memory operands. Preserve the QK, online-softmax,
PV, masking, and accumulation order. Runtime metadata reports `local_prefill_fp8_staging` as
`fp8x16_fp8x4_bf16x2`.
Context: A fresh Linux 8K profile at `304a113` attributed 20.9% of GPU kernel time to local prefill attention. Its
staging loop loaded eight bytes at a time, extracted individual FP8 bytes, and performed scalar BF16 stores for
both K and V in every tile. Global attention had already demonstrated that wider loads and FP8x4 conversion reduce
this overhead without changing the attention algorithm.
Alternatives: Keep scalar extraction; add raw-FP8 asynchronous ping-pong buffers like the global kernel; or change
local attention tile geometry. Scalar staging leaves a measured bottleneck. Async staging requires a larger
scheduling and shared-memory change, while the exact vector conversion isolates the currently proven opportunity.
Tile changes would alter more arithmetic and scheduling at once.
Consequences: The kernel retains 254 registers, zero stack/local memory, and 66,560 bytes reported shared memory.
Persistent arenas, reusable workspace, K/V format, attention reduction order, and decode are unchanged. Product
rows are naturally 256-byte aligned and each staged dimension advances by 16 bytes. Validators require the new
runtime metadata so the selected path remains observable.
Evidence: Against the fresh same-session baseline, 3-warm-up/10-run median prefill changes by
-1.23%/+3.99%/+2.67%/+3.84% at 128/512/2,048/8,192 tokens; the short points show strong clock-related bimodality,
while the stable 8K confidence intervals do not overlap. At 8K, TTFT falls from 2,229.39 to 2,147.04 ms and Nsight
reduces local-attention kernel time from 0.96996 to 0.70919 seconds (-26.88%) across two measured prefills; total
profiled kernel time falls from 4.64848 to 4.44462 seconds. CTest, exact-blue, 129/257 prefill boundaries, and the
12-prompt teacher-forced suite pass; the suite retains 121/127 Top-1 and 127/127 Top-5/Top-20. A 1-warm-up/3-run
8K decode check retains one checksum at 33.01 tok/s.

## 2026-07-26: Extend the 12B engine through its native Unified multimodal path

Date: 2026-07-26
Decision: Add image and audio input by consuming the pinned Gemma 4 12B Unified checkpoint's encoder-free BF16
modality projections directly, then add video as sampled frames over the qualified image path. Preserve explicit
text-only residency, reuse the existing mixed FP8/NVFP4 transformer and generated-token decode graph, implement
vision's blockwise bidirectional sliding-attention semantics during prefill, and bind resident cached prefixes to
canonical media identity as well as token IDs. Treat [the multimodal expansion plan](MULTIMODAL.md) as the ordered
implementation and qualification contract.
Context: The current engine deliberately excludes 104,759,808 bytes of modality tensors, accepts text token IDs,
and validates resident cache reuse by token prefix. The checkpoint already contains a 4,915,200-byte direct audio
projection and 99,844,608 bytes of vision projection/embedding tensors, with no separate encoder. Placeholder token
IDs alone cannot identify image/audio-derived K/V, and causal-only prefill would violate
`use_bidirectional_attention="vision"`.
Alternatives: Move to the 26B A4B model; attach separate vision/audio encoders; preprocess media in Python at
runtime; requantize the BF16 modality tensors; make all runs load modality weights; or treat projected media as
ordinary causal text embeddings. These alternatives respectively weaken the 16 GB target or audio capability,
duplicate model components, violate the native C++ runtime boundary, alter the checkpoint without evidence,
regress text-only residency, or change model semantics.
Consequences: Multimodal work remains after the text correctness/performance gates. It requires strict
`processor_config.json` parsing, ordered content parts, media preprocessing, BF16 modality operators,
vision-aware local prefill attention, media-safe session identity, new memory regions, and independent quality and
performance fixtures. Text-only execution must load no modality weights and retain its current transformer and
decode behavior. Multimodal support cannot be advertised as implemented until the plan's gates pass.
Evidence: The locked manifest contains 9,200,026,528 text-only bytes and 104,759,808 skipped modality bytes.
`config.json` declares image/audio/video placeholders, a 1,120-entry two-axis vision position table, direct
640-to-3,840 audio projection, and vision-only bidirectional attention. `processor_config.json` declares 16 kHz,
640 samples per audio token, a 750-token audio limit, 16-pixel image patches, 3x3 spatial merging, and a 280-token
default image budget.

## 2026-07-26: Pipeline global-prefill FP8 staging inside the existing 96 KiB tile

Date: 2026-07-26
Decision: Keep the vectorized global-prefill attention arithmetic, but split its shared K/V operand storage into
one overlaid BF16 operand tile and two raw-FP8 ping-pong tiles. Use aligned 16-byte `cp.async` copies so the current
V load overlaps QK/online softmax and the next K load overlaps PV, then perform the existing E4M3x4-to-paired-BF16
conversion before each operand is consumed. Runtime metadata reports `global_prefill_fp8_staging` as
`async_fp8x16_fp8x4_bf16x2`.
Context: After vectorized staging, global attention remained the largest attention kernel at 22.8% of an adjacent
8K profile. K and V occupied separate 16 KiB BF16 tiles even though QK and PV never consume them concurrently,
leaving enough shared memory to retain raw FP8 for the following operand without increasing the 96 KiB allocation.
Alternatives: Allocate larger independent BF16 K/V plus ping-pong storage; change the MMA or online-softmax order;
or keep synchronous staging. The first exceeds the SM120 opt-in shared-memory budget, the second adds unnecessary
correctness risk, and the third leaves global-memory latency exposed.
Consequences: The persistent arenas, reusable workspace, 96 KiB per-CTA shared-memory footprint, physical FP8 KV
cache, MMA order, and online-softmax reduction order are unchanged. The local-prefill and all decode paths are
unchanged.
Evidence: Nsight measures the global kernel from 1.03791 to 0.98382 seconds across two 8K prefill executions
(-5.21%). Under 3 warm-ups and 10 measured runs, median 8K prefill improves from 3,683.18 to 3,704.64 tok/s
(+0.58%) and TTFT from 2,224.17 to 2,211.28 ms (-0.58%), with a candidate 95% throughput CI of
`[3,692.82, 3,709.30]`. An exploratory 16K run improves from 3,095.57 to 3,154.66 tok/s (+1.91%). The global
operator retains max absolute error 0.000538, RMS error 0.000126, and cosine similarity 0.999997. CTest,
exact-blue, both vLLM boundary Top-1 cases, and teacher-forced 121/127 Top-1 plus 127/127 Top-5/Top-20 pass.
Three restored-path 8K decode runs retain the existing checksum at 33.236 tok/s.

## 2026-07-26: Reject Tensor-Core attention for single-token FP8 decode

Date: 2026-07-26
Decision: Retain the online scalar-FMA split-GQA decode-attention kernel over the physical FP8 KV cache. Do not
ship either the BF16 or TF32 Tensor-Core decode prototypes.
Context: Decode attention is an obvious remaining Tensor-Core candidate, but its M dimension is one. WMMA therefore
pads one live query row to a 16-row tile, adds shared staging and synchronization, and reduces the number of
resident CTAs. The long global layers amortize these costs better than local layers, so full and global-only
variants were measured separately.
Alternatives: BF16 WMMA for QK/PV; TF32 WMMA for QK/PV with FP32 accumulation; Tensor Cores only for global layers;
or smaller split groups. BF16 exceeded the CUDA operator tolerance (local max absolute error 0.000158; long-global
cosine 0.999962). TF32 passed the operator gates, including long-global max absolute error 0.00000551, but every
end-to-end variant regressed.
Consequences: No experimental Tensor-Core decode code or selector remains. Decode continues to read E4M3 K/V
directly, uses FP32 online-softmax state, performs no token-loop allocation, and retains the deterministic output
checksum.
Evidence: Against the 33.349 tok/s 8K/256 scalar baseline, TF32 local+global reaches 28.797 tok/s (-13.65%),
global-only with split groups of 16 reaches 32.232 tok/s (-3.35%), and global-only with groups of 8 reaches
32.292 tok/s (-3.17%). After removing the prototypes, a 1-warm-up/3-measured 8K/256 run reaches 33.236 tok/s with
p50/p95/p99 latency 30.054/30.682/31.045 ms and the unchanged checksum `14820510372112584179`; the difference from
the baseline is within run-to-run variation.

## 2026-07-26: Vectorize global-attention FP8 staging

Date: 2026-07-26
Decision: Stage global-attention K/V in aligned 16-byte vectors, convert each packed word through E4M3x4, and
write paired BF16 values into the existing shared-memory tiles. Preserve the QK, online-softmax, and PV arithmetic
and retain the local-attention path unchanged. Runtime metadata originally reported `global_prefill_fp8_staging`
as `fp8x4_bf16x2`; the subsequent pipelining decision supersedes that staging label.
Context: After CUTLASS covered all prompt projections, an adjacent 8K profile attributed 1.287 seconds, or 27.1%
of kernel time, to the global-attention kernel. Its scalar byte extraction and scalar BF16 stores were repeated
for every staged K/V tile.
Alternatives: Share QK/softmax work between output-half warps; process two heads per CTA; retain scalar staging;
or use only 8-byte vector loads. Shared reduction state made global attention 1.48% slower, the 128-thread
two-head CTA reduced end-to-end prefill to 2,912 tok/s, and the 8-byte variant produced only a noisy sub-percent
end-to-end gain. All three rejected implementations were removed.
Consequences: The persistent weights, KV cache, reusable workspace, shared-memory footprint, MMA order, and
online-softmax reduction order are unchanged. The wider access is valid because global head rows are 512-byte
aligned and the staged dimension advances in 16-byte units. Decode and local attention are unchanged.
Evidence: Nsight measures the global kernel from 1.2867 to 1.0379 seconds across two 8K prefill executions
(-19.34%) and profiled end-to-end throughput from 3,420.02 to 3,587.25 tok/s (+4.89%). Under 3 warm-ups and 10
measured runs, median 8K prefill improves from 3,539.55 to 3,683.18 tok/s (+4.06%) and TTFT from 2,314.42 to
2,224.17 ms (-3.90%), with a 95% throughput CI of `[3,680.21, 3,692.22]`. The global CUDA operator retains max
absolute error 0.000538, RMS error 0.000126, and cosine similarity 0.999997. CTest, exact-blue, vLLM boundary
Top-1 at 129/257, and teacher-forced 121/127 Top-1 plus 127/127 Top-5/Top-20 pass. Three 8K decode runs retain one
checksum and a 33.349 tok/s median.

## 2026-07-26: Use CUTLASS SM120 for FP8 prefill projections

Date: 2026-07-26
Decision: Run prompt Q, K, optional V, and O projections through CUTLASS 4.5.2 SM120 128x128x64
warp-specialized FP8 GEMMs with FP32 accumulation/output. Consume checkpoint `[N,K]` weight bytes directly as
CUTLASS column-major B, then apply each token's FP32 activation scale and each output channel's BF16 weight scale
in a separate device kernel using the original left-to-right multiplication order. Retain the grouped native
direct-source path for token-at-a-time decode.
Context: After CUTLASS NVFP4 covered the full MLP, FP8 Q/K/V/O was the largest remaining projection family.
The source-layout M128xN64xK64 kernel improved substantially over the original direct path, but still managed
tiling, staging, and scheduling manually. The checkpoint's FP8 weight ordering already matches a regular GEMM B
operand, so CUTLASS requires no per-layer transform.
Alternatives: Continue tuning the native CTA; keep grouped Q/K/V in one launch; fuse per-row/per-column scaling
into a custom CUTLASS epilogue; or retain a raw unscaled FP32 buffer. The first left confirmed end-to-end
throughput unused, the grouped launch was slower than independently scheduled GEMMs, and epilogue fusion is
deferred until it proves a further gain without changing scaling order. The existing projection buffers hold the
raw accumulators transiently, so no new arena is necessary.
Consequences: Runtime metadata reports `cutlass_m128n128k64`, `cutlass_auto`, and no grouped prefill Q/K/V.
Persistent weights, KV storage, reusable workspace (673,808,384 bytes at 8K), and
`persistent_repack_bytes=0` remain unchanged.
Evidence: A real 128x4,096x3,840 fixture is bit-exact across 524,288 FP32 outputs versus the prior kernel. Under
3 warm-ups and 10 measured 8K runs, median prefill improves from 2,910.53 to 3,539.55 tok/s (+21.61%) and TTFT
from 2,814.61 to 2,314.42 ms (-17.77%), with a 95% throughput CI of `[3,528.28, 3,543.69]`. CUDA tests,
exact-blue, vLLM boundary Top-1 at 129/257, and teacher-forced 121/127 Top-1 plus 127/127 Top-5/Top-20 pass.
Three 8K decode runs retain one checksum and a 33.073 tok/s median.

## 2026-07-26: Extend CUTLASS block-scaled GEMM to Down prefill

Date: 2026-07-26
Decision: Run Down through the same CUTLASS 4.5.2 SM120 128x128x128 block-scaled GEMM used by Gate and Up.
Interleave the newly quantized Down-input scales for its 15,360-element contracting dimension, transform the
active Down weight into the existing preallocated CUTLASS scratch, write BF16 directly into the projection
buffer, and consume that buffer in the fused residual/RMSNorm boundary. Keep the native Row8/K64 path for
token-at-a-time decode.
Context: After promoting Gate/Up, adjacent Nsight attribution placed native Down at 16.1% of 8K profiled kernel
time, versus 7.8% for both CUTLASS Gate and Up together. The real Down shape therefore had enough work to amortize
the in-arena layout conversion, while the decode-optimal persistent layout still could not be replaced globally.
Alternatives: Retain native Down; keep a persistent second CUTLASS layout; or convert Down back through FP32 before
the next boundary. The first leaves a measured bottleneck, the second violates the single-copy contract, and the
third adds traffic with no numerical benefit.
Consequences: The weight, weight-scale, and CUTLASS workspace scratch already fit Down. Only the padded
activation-scale view grows by 1,474,560 bytes, taking the 8K reusable workspace from 672,333,824 to 673,808,384
bytes. Persistent weights remain 9,200,135,680 bytes, `persistent_repack_bytes` remains zero, and decode is
unchanged.
Evidence: The real 128x3,840x15,360 Down fixture has zero BF16 mismatches across 491,520 outputs. Under 3 warm-ups
and 10 measured 8K runs, the adjacent median improves from 2,594.28 to 2,910.53 tok/s (+12.19%) and TTFT from
3,157.72 to 2,814.61 ms (-10.87%), with a 95% throughput CI of `[2,904.41, 2,915.00]`. CUDA tests, exact-blue,
vLLM boundary Top-1 at 129/257, and teacher-forced 121/127 Top-1 plus 127/127 Top-5/Top-20 pass. Three 8K decode
runs retain one checksum and a 32.661 tok/s median.

## 2026-07-26: Use CUTLASS SM120 block-scaled GEMM for Gate/Up prefill

Date: 2026-07-26
Decision: Run the large Gate and Up prompt projections through the pinned CUTLASS 4.5.2 SM120
128x128x128 block-scaled persistent GEMM with BF16 output. Interleave activation scales once per layer and
transform one active projection at a time from the sole persistent Row8/K64 allocation into preallocated prompt
scratch. Reuse that scratch for Up immediately after Gate. Retain the native Row8/K64 path for Down and all
token-at-a-time decode.
Context: After grouped online attention, 2K chunks, and M128 FP8 projections, projections accounted for 65% of
the 8K prefill profile. Isolated CUTLASS runs at the real Gate/Up and Down geometries showed substantially more
headroom than further raster-order or L2-persistence changes to the native CTA. The persistent Row8/K64 layout is
still the measured decode winner, so replacing it globally would trade away a higher-priority path.
Alternatives: Retain the native prefill CTA; keep a second persistent CUTLASS weight layout; transform every
projection including Down; or restore row-major weights globally. The first leaves confirmed prompt throughput
unused, the second violates the single-copy memory contract, Down did not yet have an end-to-end promotion result,
and the fourth regresses the qualified decode layout.
Consequences: Gate and Up each pay an in-timing device layout conversion but use a TMA/warp-specialized
block-scaled Tensor-Core GEMM afterward. One 29,491,200-byte weight scratch, 3,686,400-byte weight-scale scratch,
padded activation scales, and an 8 MiB CUTLASS workspace increase the 8K reusable workspace from 630,276,096 to
672,333,824 bytes. The 9,200,135,680-byte persistent weight arena, KV cache, decode implementation, checkpoint
bytes, and `persistent_repack_bytes=0` remain unchanged. Runtime JSON records the distinct Gate/Up and Down plans.
Evidence: Under 3 warm-ups and 10 measured runs, 8K median prefill improves from 2,135.93 to 2,584.77 tok/s
(+21.0%) and TTFT from 3,835.33 to 3,169.34 ms (-17.4%); 2K improves from approximately 2,428 to 2,984.77 tok/s
(+22.9%). A real-geometry 2,048x128x3,840 CUDA fixture has zero differing BF16 outputs against the native kernel.
CTest, exact-blue `[9503,106]`, vLLM boundary Top-1 at 129/257, and teacher-forced 118/127 Top-1,
126/127 Top-5, and 127/127 Top-20 pass unchanged. Five 8K decode runs remain internally deterministic and median
decode throughput changes by -0.63% in a short regression run, consistent with an untouched decode path and
measurement noise.

## 2026-07-26: Fence shared online-decode reduction results before reuse

Date: 2026-07-26
Decision: In both online-decode block reductions, copy the final shared maximum or sum into a thread-local register,
then execute a second block barrier before returning. Keep the selected split/merge algorithm, FP8 KV format,
fixed graph geometry, and accumulation order unchanged. Add repeated local/global/16K operator output checks and a
fresh-process 512+256 greedy determinism gate.
Context: The shared 16K Wikipedia benchmark produced ten different nominally greedy gem16 outputs while vLLM
and llama.cpp were stable. A 512+256 prompt was stable, but a 16K+64 reproducer diverged as early as output step 22.
Prefill logits were bit-identical and the controlled reference-attention path was deterministic, isolating the
problem to online decode attention. Racecheck then reported Read/Write hazards in both Split and Merge.
Alternatives: Accept numerically plausible greedy drift; disable CUDA Graphs; clear the partial workspace on every
layer; or fall back to the score-matrix attention path. Drift violates the correctness contract, the direct
controlled path reproduced the issue without graphs, workspace clearing did not help, and the reference path
would discard the promoted long-context performance.
Consequences: Two additional `__syncthreads()` operations protect each shared reduction result before the same
array is reused. There is no allocation, format, kernel-grid, cache, or arithmetic-order change. The new tool can
force full-length generation without EOS and exits non-zero when `--require-deterministic` observes multiple
hashes.
Evidence: Targeted Racecheck changes from reported Split/Merge hazards and corrupted instrumented output to zero
hazards. Repeated operator outputs are bit-identical; the 16K global fixture retains maximum absolute error
`1.04308e-7`, RMS `2.19933e-8`, and cosine 1.0. Five fresh-process 16K+64 production-graph runs share hash
`0b373ccd...e43d52`; five 512+256 runs share `8cc1cc48...6fce2`. CTest and exact-blue `[9503,106]` pass.

## 2026-07-26: Group prefill GQA heads, widen FP8 M, and use 2K FP8 chunks

Date: 2026-07-26
Decision: Make each local prefill-attention CTA process the two query heads sharing one KV head and each global
CTA process four query heads sharing the sole KV head. Stage K/V once for that group while retaining independent
per-head FP32 online-softmax state and the existing MMA order. Widen the two-stage FP8 projection CTA from
M64xN64xK64 to M128xN64xK64 so each weight fragment serves eight M16 tiles. Use 2,048-token chunks for
checkpoint-FP8 prefill; after local attention, commit only the newest 1,024-token suffix to its ring. Keep BF16
correctness prefill capped at 1,024 tokens.
Context: The 8K Nsight baseline attributed 52.6% of GPU time to attention. Its one-head CTAs redundantly staged
the same local K/V twice and the same global K/V sixteen times. After grouping, projections became the dominant
cost, while a 1,024-token chunk still repeated every layer launch group eight times at 8K.
Alternatives: Group two instead of four global heads; leave local heads independent; retain M64 FP8 projection;
or make a local chunk wider than its ring and commit all positions modulo the ring. Two global heads left measured
performance unused. Independent local heads repeat K/V traffic. M64 loses the adjacent A/B. Committing more than
one ring concurrently creates modulo-aliasing writes and is rejected.
Consequences: Local/global attention CTAs use 128/256 threads and group 2/4 query heads. Their static shared-memory
allocations including toolchain overhead are 66,560/99,328 bytes; both use 254 registers with zero stack/local
memory. The FP8 projection uses 96 registers and 25,600 bytes shared with zero stack/local memory. The 8K prefill
workspace grows from 322,457,600 to 630,276,096 bytes, remaining below the 1 GiB activation-arena target. No
persistent allocation or checkpoint representation changes. Runtime JSON and validators require the M128 tile,
head grouping, and 2K checkpoint-FP8 chunk.
Evidence: Under 3 warm-ups and 10 measured runs, 8K reaches 2,138.50 tok/s with 95% CI
`[2,134.61, 2,144.24]` and 3,830.72 ms median TTFT. The preceding 1K plan measured 2,107.04 tok/s under the same
policy; the Row8/K64 starting point was 1,560.23 tok/s and 5,250.50 ms TTFT. Final improvement over that starting
point is +37.1% throughput and -27.0% TTFT. Nsight reduces global/local attention from 4.440/1.242 seconds to
1.207/0.892 seconds across two 8K prefills; projections are now 65.0% of kernel time. CTest, exact-blue
`[9503,106]`, vLLM boundary rank 1 at 129/257, and teacher-forced 118/127 Top-1 plus 126/127 Top-5 all pass.
The 8K decode regression retains checksum `17504476492555856403` in all runs and reaches 33.676 tok/s median.

## 2026-07-26: Tile packed NVFP4 weights into the sole final SM120 allocation

Date: 2026-07-26
Decision: Transform every manifest-classified `NVFP4_PACKED` tensor at load time from checkpoint row-major order
to `[row tile 8][K64 block][row][32 packed E2M1 bytes]`. Use the existing matching Row8/K64 scale order for both
T=1 and batch SM120 kernels. Stream bounded transformed windows directly into the final arena, retain no raw GPU
copy, expose no runtime selector, and preserve source order only in reference/SIMT probes.
Context: After attention and grouped Q/K/V improvements, NVFP4 Gate/Up/Down still consumed about 10.56 ms of an
8K decode forward. Each output warp loaded eight rows whose K64 fragments were separated by a full source-row
stride even though the corresponding scale words had already been tiled contiguously.
Alternatives: Keep source-row weights; retain source plus a decode-only tiled copy; tile only Gate/Up; or perform
layout conversion inside each kernel. These respectively leave measured decode speed unused, violate the 16 GB
single-copy contract, split the MLP layout, or repeat address/data movement in the hot path.
Consequences: Every packed nibble, E4M3 scale byte, global divisor, and MMA accumulation order is unchanged.
Decode and prefill share one runtime layout. A reusable host staging vector is bounded at 4 MiB, the persistent
weight arena stays 9,200,135,680 bytes, and `persistent_repack_bytes` remains zero. Model-load timing includes the
CPU transformation and direct-to-final-allocation transfers.
Evidence: Host mapping tests cover K blocks, row tiles, and tail rows. CUDA native projections match the
source-layout reference, and the complete Layer-0 MLP has maximum absolute difference 0 and cosine 1. Exact-blue
remains `[9503,106]`; 129/257-token prefill retains vLLM Top-1 rank 1; the teacher-forced suite remains 118/127
Top-1 and 126/127 Top-5; and CTest passes. FP8-KV 8K decode improves from 31.604 to 33.143 tok/s median (+4.87%)
under the same 1-warm-up/3-run policy.

## 2026-07-26: Split and merge long-context FP8 decode attention

Date: 2026-07-26
Decision: For checkpoint-FP8 plans above 512 positions, replace the materialized score matrix and separate
softmax/value kernels with shape-specific split attention inside the complete decode graph. Group the two local
queries per KV head and four global queries at a time, compute normalized partial outputs with FP32 log-sum-exp
state, and merge token splits in a second kernel. Use 256-token local splits and 512-token global splits. Retain the
prior score/softmax/value path for plans through 512 positions and for explicit BF16 K/V correctness mode.
Context: The prior controlled decode path serially reduced D256/D512 QK inside one thread, wrote every score to
global memory, reread it for softmax, and scanned the cache again for PV. Its cost grew sharply at 8K even though
vLLM remained nearly context-flat. A single CTA per query with token-serial online softmax would remove score
traffic but expose too little parallelism, especially for the eight global layers.
Alternatives: Use one CTA per query; retain separate score and value kernels while only parallelizing QK; use one
fixed split size for both Gemma geometries; or select a plan dynamically in the token loop. These respectively
underfill the GPU, retain avoidable traffic, ignore distinct D256/D512 work, or weaken deterministic CUDA Graph
execution.
Consequences: Decode workspace stores normalized split outputs and LSE values rather than only scores, but remains
approximately the previous global score size plus small LSE storage. Kernel grids and addresses are fixed during
graph capture; there are no token-loop allocations or host decisions. JSON reports `fp8_online_split_gqa` versus
`score_softmax_value_reference`. The remaining 8K gap is now primarily projection and output-head work.
Evidence: Product-shape CUDA comparisons have maximum absolute error `3.73e-8` local and `1.86e-7` global with
cosine 1.0. All host/CUDA tests and exact-blue generation pass. On the Linux RTX 5080 Laptop, context-8K decode is
30.02 tok/s with 33.21/35.19/36.65 ms p50/p95/p99 over 3 warm-ups and 10 measured runs; all runs share checksum
`17504476492555856403`. The node-level Nsight trace measures about 5.1 ms/token for all 48 decode-attention layers
and merges. The short-path regression check is 32.13 tok/s at context 128.

## 2026-07-26: Compose Unsloth weights with Google's current tokenizer metadata

Date: 2026-07-26
Decision: Keep every weight, quantization, vocabulary, generation, and chat-template artifact at the locked
Unsloth revision, but source `tokenizer_config.json` from official Google Gemma 4 12B IT commit
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`. Represent this as a per-file immutable source in lock schema v2.
Parse and validate Google's token roles and response template at engine startup, use its thinking/content
delimiters for visible response extraction, and retain `generation_config.json` as the authoritative stop-ID list.
Context: The Unsloth snapshot predates Google's response template and aliases `eos_token` to the turn-end token.
Google now distinguishes `<eos>` from `<turn|>` and publishes structured response boundaries. Merely copying the
new file into one local checkpoint would make provenance irreproducible, while merely locking it without reading
it would leave engine behavior unchanged.
Alternatives: Continue using Unsloth metadata; follow Google's `main`; silently overlay a bundled runtime file; or
replace the complete checkpoint with Google's BF16 source. Those choices respectively retain stale semantics,
lose reproducibility, hide the effective checkpoint composition, or discard the selected NVFP4/FP8 weights.
Consequences: `tools/fetch_model.py` resolves file-specific sources and resumes only partial files whose URL, size,
and digest identity matches. A downloaded checkpoint directory remains directly loadable and contains one
canonical `tokenizer_config.json`. Google's approximately 1e30 tokenizer-length sentinel is accepted as generic
JSON metadata but never replaces the 262,144-position model contract. Generated tool calls still fail visibly
until the phase-one runtime deliberately supports them.
Evidence: The official file is locked at 3,089 bytes with SHA-256
`a62f4e85a47c0c136edaaa3a4f591fd6783717299a9def47e5ad03a49f6a5eb9`. Host C++ tests cover the large JSON number,
Google schema validation, response extraction, and rejection of the old EOS alias. Python tests cover per-file
source selection, safe replacement, and identity-bound resume. The fully materialized local snapshot passes lock
verification, `gem16-inspect --validate`, and native chat render-only loading.

## 2026-07-26: Make interactive chat a resident exact-token session

Date: 2026-07-26
Decision: Create one engine for the lifetime of `gem16-chat`, retain its weights, arenas, CUDA Graphs, and hybrid
KV cache across turns, and batch-prefill only tokens appended after the materialized cache prefix. Preserve the
original generated token IDs at the CLI boundary; do not reconstruct prior assistant output through text
decode/re-encode. Require every submitted prompt to extend the cached token prefix exactly and fail visibly on a
mismatch. This is the sole interactive path and has no cache/reload selector.
Context: The former loop called `RunGreedyInference` for every user message. It reloaded roughly 9 GB of weights,
cleared the cache, rendered the entire history, and recomputed every prior token. A first resident prototype proved
why text is not a valid cache identity: decoded `blau` re-encoded without the generated channel tokens and failed
the exact prefix gate. Continuing from the actual autoregressive token sequence is both cheaper and faithful to
the state that produced the answer.
Alternatives: Reload and prefill the complete conversation; compare only decoded text; silently reset when token
prefixes differ; or add a user-selectable session mode. Full replay discards the available KV state, decoded text
does not uniquely identify BPE tokens, silent reset hides a performance and semantic change, and a selector would
retain an inferior interactive path.
Consequences: Initial prompt processing is unchanged. Each later turn pre-fills the preserved final assistant token
when a turn ended at the length limit, followed by the newly tokenized exact Gemma turn delimiter, user content,
and generation header. Existing conversation K/V remains in the local rings and global contiguous cache. The
session pre-reserves host token bookkeeping through `--max-context`; generation retains the no-allocation token
loop. A failed inference poisons the session because partially written KV state cannot be rolled back safely.
Evidence: The release and host builds pass, CTest passes both host and CUDA suites, and a real resident GPU smoke
test completed three dependent turns (`blau`, recall `blau`, translate `blue`) after a single model load. Both
continuations passed the exact token-prefix check and returned without another weight load or full-history prefill.
A separate two-turn run with `--max-tokens 1` validates the pending, not-yet-materialized final assistant token;
a two-turn thinking-template run validates the same prefix continuation with generated channel tokens.

## 2026-07-25: Precompute exact RoPE and fuse the full Q/K normalization boundary

Date: 2026-07-25
Decision: Generate local D256 and proportional global D512 cosine/sine tables once during engine initialization for
every position in the planned context. Make one CTA per token/head preserve projection BF16 rounding, the original
256-thread RMSNorm reduction, normalized BF16 rounding, RoPE, and post-RoPE BF16 rounding. Group Q and K blocks in
one launch without sharing reduction state. Make this the sole prefill path and expose no selector.
Context: After the first boundary promotion, the old RoPE kernels still consumed 50.58 ms per 512-token prefill
and repeated identical double-precision `pow`/`cos`/`sin` work for every head in every layer. Together with Q/K
rounding and normalization they required eight launches per layer.
Alternatives: Merely group Q/K launches; share trigonometry inside each layer; use approximate float intrinsics;
retain runtime variants; or store no table. The first fused prototype was exact but only won long prompts and kept
the expensive trig stackframe in the hot kernel. An earlier approximate sharing probe changed logits. Persistent
tables preserve the exact float values, remove hot-path trigonometry, and cost only 1,536 bytes per planned token.
Consequences: Prefill executes one 35-register, 3,072-byte-shared, zero-stack/local kernel instead of eight launches
per layer. The initialization-only table kernel retains the exact double expressions and is included in model-load
time. At context 2,048 the workspace grows by 3,147,264 bytes and measured process peak grows 4 MiB to 9,586 MiB.
Runtime JSON and validators require the fused kernel and `precomputed_exact_max_context` table.
Evidence: Against detached `ccbe4ed`, adjacent 3/30 medians improve 15.17%/15.14% at 128/512 with non-overlapping
mean 95% intervals; 2,048 improves 16.76% in the required 3/10 run. Nsight reduces launches from 1,300 to 964 and
GPU kernel time from 250.03 to 208.01 ms per 512-token prefill. Local/global CUDA outputs are bit-identical to the
eight-kernel oracle. CTest, exact-blue, vLLM boundaries 129/257, and every aggregate metric in the
12-prompt/127-position teacher-forced suite remain unchanged.

## 2026-07-25: Fuse exact prefill normalization and MLP quantization boundaries

Date: 2026-07-25
Decision: Replace the production prefill sequences at RMSNorm/FP8 quantization, RMSNorm/NVFP4 quantization,
post-projection norm/residual/optional layer scale, and Gate/Up/GELU-tanh/NVFP4 quantization with shape-specific
fused kernels. Preserve every prescribed BF16 cast and quantized payload exactly. Make the fused implementation
the sole production path; retain the former sequence only as a CUDA test oracle and expose no selector.
Context: After the projection phases, context-512 prefill still launched 2,165 kernels. Repeated standalone
rounding, RMSNorm, GELU, and quantization kernels dominated launch-heavy residual work even though their producer
and consumer share the same token geometry.
Alternatives: Keep the separate sequence; combine Gate and Up projections; share Q/K RoPE trigonometric arithmetic;
or retain runtime A/B modes. Separate launches leave a measured end-to-end gain unused. Combined projections had
lost the prior Linux A/B. Sharing RoPE arithmetic was re-tested and rejected because it changed boundary logits.
Runtime variants violate the single-winner policy.
Consequences: Gate and Up projections remain separate, but their exact BF16 outputs feed one GELU/product/NVFP4
kernel. Normalization fusions retain BF16 rounding before residual and optional layer scaling. Runtime JSON and
validators require all four fused families. Launches fall to 1,300 per context-512 prefill; arenas and persistent
checkpoint storage are unchanged. The selected kernels use at most 40 registers and 3,072 bytes shared memory and
have zero stack/local memory.
Evidence: Against detached `bdb1294`, final 3/10 medians improve from 1,540.69/1,748.61/1,563.23 to
1,664.23/1,914.76/1,722.95 tok/s at 128/512/2,048 (`+8.0%/+9.5%/+10.2%`). Nsight measures 40.0% fewer launches
and 8.73% less total kernel time at 512. CTest, exact-blue, vLLM boundaries 129/257, and all aggregate metrics in
the 12-prompt/127-position teacher-forced suite remain unchanged. Peak process VRAM is 9,582 MiB.

## 2026-07-25: Pipeline source-layout FP8 prefill and group Q/K/V

Date: 2026-07-25
Decision: Replace the 32-token-per-warp FP8 batch projection with one 256-thread M64xN64xK64 CTA. Double-buffer
exact source-layout activation and weight bytes in shared memory with `cp.async`, reuse every weight fragment over
four M16 MMA tiles, and group local Q/K/V or global Q/K through one binding-dimension launch. Make this the only
FP8 prefill projection path; keep the latency-oriented T=1 decode kernel separate and expose no selector.
Context: At context 512, the prior direct matrix kernel consumed 114.09 ms per prefill and launched 184 times.
Increasing per-warp M reuse alone was neutral or regressive because it did not address redundant, weakly
coalesced operand traffic across warps. The CTA tile lets eight warps share staged operands and overlap the next
K64 slice, while grouping removes otherwise independent launches without changing any projection arithmetic.
Alternatives: Retain two M16 tiles per warp; use four/eight tiles without CTA staging; preserve grouped and
ungrouped runtime variants; repack FP8 checkpoint weights. The M64/M128 warp-only candidates lost at short context,
parallel variants violate the single-winner policy, and persistent repacking is unnecessary.
Consequences: FP8 checkpoint bytes, per-token scales, per-channel BF16 scales, and FP32 K order are unchanged.
The kernel uses 60 registers and 17,408 bytes static shared memory with zero stack/local memory. FP8 launches fall
from 184 to 96 per context-512 prefill; no persistent allocation, arena size, or token-loop allocation changes.
Runtime JSON and validators require `m64n64k64`, two pipeline stages, and grouped Q/K/V.
Evidence: Against exact commit `6005921`, final 3/10 medians rise from 1,348.97/1,460.83/1,358.11 to
1,583.23/1,769.04/1,562.05 tok/s at 128/512/2,048 (`+17.4%/+21.1%/+15.0%`). Nsight reduces FP8 projection time
from 114.09 to 62.21 ms per context-512 prefill (`-45.5%`). CTest, the exact grouped-Q/K/V operator fixture,
exact-blue, vLLM boundaries 129/257, and all 127 teacher-forced positions preserve their prior results.

## 2026-07-25: Tile exact NVFP4 weight scales into the final allocation

Date: 2026-07-25
Decision: Keep packed E2M1 weights in checkpoint row-major order, but reorder every manifest-classified
`NVFP4_LOCAL_SCALE_E4M3` tensor at load time to `[row tile 8][K64 block][row][4 scale bytes]`. Stream each bounded
host tensor directly into its final arena address. Make this the only native SM120 scale layout for decode and
prefill; expose no selector and retain source order only in correctness/SIMT probes.
Context: Each output warp needs one four-byte scale vector for each of eight rows. Source row-major order places
those words at large row strides. The tiled order places them in one 32-byte region and lets decode use a constant
32-byte K-step instead of repeated strided 64-bit address construction.
Alternatives: Keep direct scale order; repack packed weights too; retain both device layouts; expose an opt-in.
Direct scales leave measured performance unused. Packed-weight repacking has no evidence, while duplicate layouts
and switches violate the memory and single-winner contracts.
Consequences: Every quantized value, scale byte, global divisor, and FP32 K accumulation remains unchanged. The
weight arena remains 9,200,135,680 bytes; the largest transient host vector is 3,686,400 bytes and exists only
during model load. Prefill uses 128 registers/10,240 shared bytes and decode uses 40 registers; both report zero
stack/local memory. Runtime JSON distinguishes direct packed weights from the mandatory scale layout.
Evidence: Against detached `e17049b`, 128/512/2,048-token prefill medians improve by 1.10%/1.43%/3.19%. Nsight
reduces NVFP4 time by 4.61% and all GPU-operation time by 2.22% at 512. A short context-128 decode rises from
25.54 to 31.63 tok/s with identical checksum; the complete Layer-0 MLP falls from 0.480 to 0.260 ms. CTest,
exact-blue, 129/257 vLLM boundaries, Layer-0, and all 127 teacher-forced positions preserve their prior metrics.
Evidence is retained under
`benchmarks/results/2026-07-25/e17049b-worktree/blackwell16gb-linux-nvfp4-scale-tile/`.

## 2026-07-25: Pipeline NVFP4 activation staging with two cp.async buffers

Date: 2026-07-25
Decision: Make the production M128xN64 NVFP4 prefill CTA double-buffer its exact packed activation bytes and E4M3
activation scales. Use 16-byte and 4-byte `cp.async` transfers with zero fill for token tails, and issue the next
K64 stage while the current stage feeds eight native block-scaled OMMAs. Replace the synchronous stage directly.
Context: CTA-wide activation reuse removed more than half of NVFP4 time, but every K64 iteration still synchronously
loaded and stored its shared tile before MMA work could begin. The independent K64 accumulation provides a natural
ping-pong boundary without changing arithmetic.
Alternatives: Retain synchronous staging; add a runtime pipeline selector; use an N128 CTA; prefetch weights only
in registers. N128 loses about 2.5% end to end and was removed. Register-only weight prefetch was neutral. A public
selector conflicts with the single-winner plan.
Consequences: Static shared memory rises from 5,632 to 10,240 bytes and registers from 123 to 124; stack/local
memory remain zero. SASS contains the expected `LDGSTS` asynchronous copies. Arena, checkpoint layout, launch
count, weight loads, token-tail values, and FP32 K accumulation are unchanged.
Evidence: A neighboring 30-run context-512 comparison raises mean/median throughput by 1.92%/1.71%. Paired
throughput differences have a 95% interval of +10.68 to +43.69 tok/s (`t=3.37`, 29 degrees of freedom). At 2,048
tokens the 3/10 median rises from 1,307.64 to 1,329.51 tok/s with non-overlapping confidence intervals. Nsight
reduces NVFP4 time by 16.9% and projected total GPU time by 7.5%. All fixed correctness metrics remain identical.
Evidence is retained under
`benchmarks/results/2026-07-25/d8b73ce-worktree/blackwell16gb-linux-nvfp4-async-pipeline/`.

## 2026-07-25: Reuse each NVFP4 activation K64 slice across an M128xN64 CTA

Date: 2026-07-25
Decision: Make the sole production NVFP4 prefill kernel use eight-warp, 256-thread M128xN64xK64 CTAs. Retain the
existing M128 per-warp accumulator stack and direct N8 packed-weight fragments, but cooperatively stage the exact
M128xK64 packed activation bytes and E4M3 activation-scale words once in shared memory for all eight output warps.
Context: After the M128 register-reuse promotion, Nsight still attributed about 47% of projected prefill GPU time
to NVFP4. Every output warp independently issued the same activation loads and address arithmetic while only its
N8 weight fragment differed.
Alternatives: Prefetch the next weight fragment in registers; extend each warp to M256; retain four-warp N32
CTAs; add a selectable CTA size; decode or repack weights. Adjacent 3/10 measurement found register prefetch
neutral/slightly slower and it was removed. M256 spills. Permanent geometry choices conflict with the one-winner
plan, and weight conversion is unnecessary before exact activation reuse is exhausted.
Consequences: The kernel uses 123 registers, 5,632 static shared bytes, zero stack/local memory, and no new arena
or persistent allocation. Packed checkpoint weights and scales remain unchanged. K64 and FP32 accumulation order
within every M16N8 result is identical. No runtime selector or fallback is added.
Evidence: Against a separately built `2366c03` reference, 3/10 median prefill improves by 26.0%/29.8%/26.7% at
128/512/2,048 tokens. The 512 and 2,048 confidence intervals do not overlap. Nsight reduces Gate+Up, Down, and
total NVFP4 time by 49.4%, 54.4%, and 51.2%. Release CTest, exact-blue, vLLM 129/257 boundary logits, and every
teacher-forced aggregate remain unchanged. Evidence is retained under
`benchmarks/results/2026-07-25/abb430c-worktree/blackwell16gb-linux-nvfp4-cta-m128n64/`.

## 2026-07-25: Vectorize checkpoint-FP8 key reads without reordering QK

Date: 2026-07-25
Decision: Make fused checkpoint-FP8 prefill attention load aligned key rows in 16-byte vectors, extract their FP8
bytes in increasing dimension order, and retain the existing serial FP32 FMA accumulation. Use the scalar loop only
for internal geometries whose row address or extent is not 16-byte aligned; product model shapes always satisfy the
wide-load invariant. This is deterministic geometry handling, not a runtime performance option.
Context: Linux Nsight Systems attributed 41.3% of context-512 projected GPU time to the fused attention kernel.
Each score thread consumed a contiguous FP8 key row one byte at a time; the compiler could not combine those loads
across the loop-carried FMA dependency.
Alternatives: Parallelize each QK dot product across a warp; introduce an approximate or tensor-core attention
route; retain scalar loads. The warp prototype was faster but changed the first generated token on 129/257-token
synthetic prompts and was discarded. Wider loads obtain a larger gain without changing arithmetic.
Consequences: The production FP8 kernel uses 48 registers, 3 KiB shared memory, and no stack/local memory. Score
storage, softmax, value accumulation, cache semantics, arena sizes, and launch count are unchanged. A true online
FlashAttention design remains the next architectural opportunity but must establish its own numerical evidence.
Evidence: Against a separately built `c0c9b42` reference at context 512 with 3 warm-ups and 10 runs, throughput
improves from 603.42 to 698.25 tok/s (+15.72%) and TTFT falls from 848.50 to 733.27 ms (-13.58%). Nsight measures
705.49 to 399.53 ms (-43.37%) fused-attention time. A 32-dimensional FP8 fused/reference fixture is bit-identical;
exact-blue, exact 129/257-token eight-step sequences, and release unit/CUDA tests pass.

## 2026-07-25: Reuse FP8 weights across two prefill token tiles

Date: 2026-07-25
Decision: Assign two consecutive 16-token MMA tiles to each production FP8 batch-projection warp, reusing the
loaded Q/K/V/O weight fragment for both. Replace the one-tile mapping directly without a selector.
Context: After the NVFP4 promotion, FP8 attention projections still consumed 16.0% of context-512 projected GPU
time. Their source-layout `m16n8k32` mapping had the same adjacent-token weight reload as the old NVFP4 kernel.
Alternatives: Retain one tile per warp; add a configurable tile count; defer all projection work until a complete
asynchronous pipeline exists. The first wastes measured bandwidth, the second violates the single-winner plan, and
the third leaves an independently validated gain unused.
Consequences: Accumulator count grows while each tile retains its original K-order. `cuobjdump` reports 56
registers and zero stack/local memory. Source layout, scales, arena sizes, and tail masking are unchanged.
Evidence: Against a separately built `b032e6f` binary at context 512 with 3 warm-ups and 10 measured runs, median
throughput improves from 587.87 to 605.33 tok/s (+2.97%) and TTFT falls from 870.95 to 845.82 ms (-2.89%). Nsight
Systems measures 280.18 to 245.73 ms (-12.30%) for FP8 projection kernels across two prefill executions. Exact-blue,
exact 129/257-token eight-step sequences, and release unit/CUDA tests pass.

## 2026-07-25: Reuse NVFP4 weights across two prefill token tiles

Date: 2026-07-25
Decision: Make the production NVFP4 batch projection assign two consecutive 16-token MMA tiles to each warp. Load
each source-layout Gate, Up, or Down weight fragment and scale once per contracting block, then issue one MMA for
each token tile with independent activation fragments, scales, and accumulators. This replaces the prior one-tile
kernel directly and adds no runtime selector.
Context: After widening prefill chunks, Nsight Systems attributed 35.5% of projected context-512 GPU time to NVFP4
projections. The old mapping made separate warps reread identical weight fragments for adjacent token tiles even
though batch-one prefill has abundant token-parallel work.
Alternatives: Keep one tile per warp; add a user-selectable tile count; change the K-dimension accumulation order;
stage a larger weight tile in shared memory. A permanent selector conflicts with the single-winner execution plan,
changing accumulation order weakens the numerical comparison, and shared staging is not justified before measuring
this register-only reuse.
Consequences: Each warp carries twice as many output accumulators. The unfused and fused forms use 72 and 80
registers respectively, but `cuobjdump` reports zero stack and local memory. Arena sizes, source weight layout,
global/local scaling, FP32 accumulation order within each tile, and tail masking are unchanged.
Evidence: A separately built `8f05333` reference and the promoted kernel were measured with 3 warm-ups and 10 runs
at context 512. Median throughput changes from 542.58 to 587.68 tok/s (+8.31%); TTFT changes from 943.64 to
871.23 ms (-7.67%). Nsight Systems reports 669.99 to 547.20 ms (-18.33%) for NVFP4 projection kernels across two
prefill executions. Exact eight-token sequences match at prompt lengths 129 and 257, the exact-blue fixture passes,
and all release unit/CUDA tests pass.

## 2026-07-25: Promote measured winners and remove production optimization switches

Date: 2026-07-25
Decision: Expose one production execution plan: native SM120 projections, context-budgeted chunked prefill (128 tokens
by default), fused causal
prefill attention, separate Gate/Up/GELU, complete decode graphs, and fused warp-row output reduction. Remove the
six public projection/prefill/fusion/graph A/B switches and their option fields. Keep slower and reference
implementations callable only from dedicated tests and characterization probes.
Context: The CLI had accumulated six optimization switch families and 68 source/documentation occurrences. This
made a validated fast path look optional and allowed ordinary runs to select known-slower plans. Gate/Up was the
only unresolved choice because a 1.7% isolated kernel gain had not survived Windows end-to-end measurement.
Alternatives: Keep every A/B switch indefinitely; enable every fusion; delete reference kernels as well as runtime
dispatch; introduce an automatic tuner. Persistent switches weaken the product contract, while deleting probes
would weaken correctness evidence and an automatic tuner would make plans non-deterministic.
Consequences: Product CLIs have no optimization opt-ins or opt-outs. A diagnostic logits/state request may still
use its required capture mechanics, and explicit BF16 K/V remains a labeled numerical correctness mode rather than
a performance implementation choice. New implementations must beat the current plan with correctness evidence
before replacing it; they are characterized through tests/probes rather than shipped as permanent toggles.
Evidence: Existing gates establish fused prefill attention (+5.6%), complete decode graphs (+31% over direct in the
same Windows characterization), and fused output reduction (+0.75%) with matching token checks. Linux Gate/Up A/B
at commit `960528d` selected separate operations: Prefill 128 was 573.32 versus 527.53 tok/s, Prefill 512 was 441.73
versus 410.16 tok/s, and context-128 Decode was 26.08 versus 25.86 tok/s. All compared runs retained deterministic
checksums.

## 2026-07-25: Use warp-row candidate reduction for greedy decode output

Date: 2026-07-25
Decision: Make the decode-only tied-BF16 output path evaluate one vocabulary row per warp, execute eight rows per
block, apply the configured softcap to every row, and retain one candidate per block before a final GPU reduction.
Keep the original full-logit output head and separate argmax for internal arithmetic and performance probes. The
later production-path consolidation removes its public runtime switch.
Context: The original head launched one 256-thread block for each of 262,144 rows, wrote every float32 logit, and
then scanned that array in a separate argmax kernel. An initially fused implementation preserved the exact
256-thread addition order but serialized multiple rows per block and was slower. Row-per-warp execution exposes
enough parallel rows while avoiding the full-logit write/read in ordinary greedy decode.
Alternatives: Retain only the full-logit path; accept the slower exact-order fused implementation; skip the monotonic
softcap during argmax; change the BF16 tied weights to a lower precision. The latter two violate the checkpoint and
benchmark contracts and were rejected without measurement.
Consequences: Decode dot products use a 32-lane rather than 256-thread addition tree, so diagnostic logits can
differ by a few float32 ULPs. Full-logit capture uses the same warp arithmetic when fusion is enabled, allowing
explicit comparison. The softcap, suppression behavior, lowest-token tie break, and no-allocation token loop remain
unchanged. The candidate array costs 32 KiB and the full-logit implementation remains available.
Evidence: At context 128 with 64 generated tokens and 3/10 runs, median throughput improves from 27.02 to
27.22 tok/s and median inter-token latency falls from 36.94 to 36.62 ms. Nsight measures the isolated output stage
at about 3.031 ms instead of 3.315 ms. Across 31 sky positions, fused versus retained logits have max absolute error
`7.6294e-6`, RMS error `1.0358e-6`, cosine similarity `0.9999999999999948`, Top-1 agreement 31/31, and mean Top-20
overlap 20/20. The 31-token sky and 32-token integer autoregressive sequences are identical.

## 2026-07-24: Match checkpoint FP8 K/V semantics by default and label BF16 explicitly

Date: 2026-07-24
Decision: When the checkpoint declares its static FP8 K/V scheme and per-layer scales, apply those E4M3
quantize/dequantize semantics by default and store each cached value as one physical E4M3FN byte. Retain
`--kv-cache bf16` as an explicit correctness mode. Label the routes and physical storage in every inference result,
and reject the initial unfused cache kernels as benchmark evidence.
Context: The first sky-prompt divergence was deterministic under greedy decoding: vLLM and llama.cpp selected token
`563`, while the original BF16-cache engine selected `7412`. Layerwise prompt-derived dumps found exact V before
the cache but a difference in the first attention context after cache reuse.
Alternatives: Treat the token difference as an acceptable low-precision variation; force every reference to BF16;
claim that float storage containing dequantized FP8 values is a production FP8 cache.
Consequences: Normal chat follows the checkpoint. BF16 remains useful for isolating operator error, but results
from different cache modes must never be presented as parity comparisons.
The one-byte cache gives valid allocator accounting; optimized cache/attention kernels remain required for
performance qualification.
Evidence: Explicit BF16 vLLM and gem16 both generate `[818,7217,7412]`. The current physical-FP8 gem16 path
also generates `[818,7217,7412]`, while FP8-vLLM and llama.cpp generate `[818,7217,563]`; the FP8 attention
difference remains open. At context 64 the physical FP8 allocation is 11,010,048 bytes versus 44,040,192 bytes for
the float32 BF16-semantics diagnostic cache.

## 2026-07-24: Expose checkpoint chat semantics through a native C++ boundary

Date: 2026-07-24
Decision: Implement checkpoint byte-fallback BPE, text chat rendering, decoding, and generation controls in C++.
Read and identity-check the actual `chat_template.jinja`, implement its supported text branches natively, and reject
unknown revisions or unsupported roles. Keep the processor independent of terminal I/O for later Chat Completions
reuse.
Context: The user-facing chat executable must not depend on Python or Transformers. A generic Jinja runtime would
add a broad dependency and still require careful model-specific semantics, while silently hard-coding a template
without reading the artifact would violate the checkpoint contract.
Alternatives: Retain the Python bridge; embed Python; vendor a general Jinja interpreter; accept only manual token
IDs.
Consequences: `gem16-chat` is a self-contained C++ process and the tokenizer/processor can later serve HTTP
requests. The supported template revision is explicit. Tool-call and multimodal branches remain unsupported until
implemented and tested. The engine still reloads weights per turn until a persistent session API is introduced.
Evidence: Native C++ rendering and BPE reproduce the committed 20-, 23-, and 27-token prompts exactly, and the
CUDA one-shot path produces and decodes `[9503, 106]` as `blue`.

## 2026-07-24: Use cross-engine distributions and quality, not bit identity

Date: 2026-07-24
Decision: Do not require generated tokens or logits to be bit-identical to vLLM or llama.cpp. Require unexplained
large or early deviations to be investigated with full logits, hidden states, quality tasks, and independent
references before setting measured tolerances.
Context: vLLM consumes the mixed FP8/NVFP4 source directly, while the closest-parity llama.cpp candidate maps FP8
attention to BF16 and uses different kernels. They nevertheless agree for most current tokens but eventually
diverge, as expected from autoregressive sensitivity.
Alternatives: Require exact token equality indefinitely; accept any coherent-looking text; select one runtime as
infallible.
Consequences: Product correctness is based on operator contracts, distribution metrics, generation stability, and
task quality. Early disagreement still blocks acceptance until configuration differences are excluded; tolerances
are not invented merely to accept it.
Evidence: The sky step-2 disagreement was investigated rather than waived. FP8-versus-BF16 K/V semantics determine
vLLM's result, but the current precision-matched FP8 gem16 path still differs and remains an active correctness
gate.

## 2026-07-23: Qualify unfused full-layer composition before fusion

Date: 2026-07-23
Decision: Compose the validated FP8 local-attention and NVFP4 MLP routes into a complete Layer-0 device path before
introducing fused Q/K/V, Gate/Up, residual, or CUDA Graph implementations. Keep independent CUDA scalar-projection
and direct SM120 paths alive through the final layer output and expose their quantization-boundary differences.
Context: Individual operators and sublayers were numerically close, but a quantization boundary can amplify small
attention differences. A full layer is the smallest executable unit that proves the residual, norm, mixed-format,
and `layer_scalar` ordering together.
Alternatives: Begin fusion from isolated kernel results; join sublayers through host memory; wait for tokenizer and
embedding support before testing full-layer composition.
Consequences: The characterization deliberately owns two copies of execution buffers and is not a production
memory plan. It establishes a no-host-roundtrip correctness path and a stable orchestration gate while preserving
the requirement for a later prompt-derived trusted hidden-state comparison.
Evidence: The real Layer-0 path produces zero differing bytes at both NVFP4 activation boundaries. Its final
CUDA-reference/direct-SM120 comparison has maximum absolute error `4.7683716e-6`, RMS error `2.8454761e-7`, and
cosine similarity `0.9999999999999643`.

## 2026-07-23: Store final K and V cache states separately

Date: 2026-07-23
Decision: Reuse the single full-attention K projection output as the input to both K and V post-processing, but
always allocate and append separate final K and V cache states. Reject `--kv-storage shared` rather than accepting
an invalid memory optimization. Continue reporting a one-state byte count only as a diagnostic lower bound.
Context: The executable Layer-5 path resolves the earlier ambiguity around `attention_k_eq_v=true`. The raw K
projection is shared, but K then receives its learned per-head RMSNorm and proportional RoPE while V receives a
scale-free RMSNorm and no RoPE. Their stored values are therefore distinct.
Alternatives: Physically share the cache because the projection tensor is shared; recompute one state during every
attention read; leave the option selectable until end-to-end assembly.
Consequences: The one-byte FP8 cache budget at 64K is 672 MiB rather than 336 MiB. The memory plan remains below the
16 GB target and now matches the implemented model semantics. Projection reuse still avoids a separate `v_proj`
weight read and launch on full-attention layers.
Evidence: The real Layer-5 checkpoint probe binds the absent `v_proj` as a reused raw K projection, applies the two
distinct post-processing paths, appends both states, and matches the independent CUDA scalar route with maximum
absolute error `4.5299530e-6`, RMS error `5.5268314e-7`, and cosine similarity
`0.9999999999999085`.

## 2026-07-23: Bring up NVFP4 from an exact oracle into separate decode and prefill plans

Date: 2026-07-23
Decision: Implement the E2M1/E4M3FN and compressed-tensors divisor contract first, followed by an explicit
correctness CUDA route, direct source-layout SM120 fragment views, and independently measured packed-GEMV and native-MMA decode
candidates. Fuse Gate/Up only after the common input/global-scale invariant is validated; build prefill as a
separate plan and keep FP8 attention as a separate precision backend.
Context: NInfer demonstrates the value of closed, shape-specific plan catalogs, arenas, graph-stable addresses, and
Gate/Up fusion, but its integer Q4 format and offline `.ninfer` artifact are incompatible with this checkpoint.
A neighboring SM120 prototype demonstrates the native block-scaled instruction and operand-fragment mapping, but it
retains multiple device layouts and previously exposed an input-global-scale semantic error. The pinned Gemma
checkpoint stores compressed-tensors global divisors, has exact SM120-friendly MLP dimensions, and uses mixed FP8
attention plus NVFP4 MLP projections.
Alternatives: Start with a complete unfused model and debug quantization indirectly; copy the neighboring loader and
retain raw plus multiple repacked device tensors; assume native MMA is fastest at `T=1`; use one GEMM plan for decode
and prefill.
Consequences: Kernel work begins later but every route shares one independent oracle. The 16 GB memory contract is
preserved, silent precision fallback remains impossible, and the project obtains direct evidence for the actual
batch-one winner. Gate/Up can reuse one activation quantization and later fuse the GELU-tanh epilogue. The first
native candidate adds no persistent repacked weight or expanded scale copy; a streamed transformation remains a
measured fallback. Loader and kernel layouts remain architecture-specific implementation details behind the
manifest contract.
Evidence: All 48 Gate/Up pairs have identical stored input and weight divisors. The real Gate/Up and Down shapes are
divisible by the intended 128/64 outer/contracting geometry. The 144 local-scale tensors contain 530,841,600
positive, nonzero E4M3FN bytes with no NaN encoding.

## 2026-07-23: Keep the first memory plan explicit and evidence-bounded

Date: 2026-07-23
Decision: Build a deterministic 256-byte-aligned base arena from the parsed text-only tensor inventory and context
metadata. Calculate both one-state and separate K/V payloads, require an explicit selection, and leave execution
workspaces visibly unplanned until kernel shapes define them. The later Layer-5 decision above resolves the storage
selection to separate K and V.
Context: At the time, the checkpoint proved `attention_k_eq_v=true`, but physical shared-cache semantics and
workspace sizes had not yet been validated by an executable model path. A 16 GB budget cannot tolerate hidden or
guessed allocations.
Alternatives: Assume shared K/V immediately; reserve budget-table maxima as real allocations; defer all memory work
until CUDA kernels exist.
Consequences: Weight, scale, and KV offsets are deterministic and overflow-checked now. Memory reports remain useful
without claiming peak VRAM. The plan is deliberately incomplete until activations, logits, sampling, graph, kernel,
and prefill workspaces are derived and measured.
Evidence: The locked 1,389-tensor manifest yields 9,200,026,528 text-only bytes. At 64K, parsed layer metadata yields
336 MiB shared or 672 MiB separate one-byte K/V payloads, matching the independently documented formula.

## 2026-07-23: Support Linux and Windows in the repository foundation

Date: 2026-07-23
Decision: Keep one Ninja-based preset layout for both operating systems, isolate file mapping behind POSIX and Win32
implementations, and provide native Bash and PowerShell build entry points. Add Windows host CI while retaining the
Linux sanitizer path.
Context: Development moved from Linux to Windows on the same Blackwell machine. Loader and build work must remain
reproducible on both systems without weakening the Linux reference path.
Alternatives: Develop only through WSL; maintain unrelated Windows CMake targets; replace memory mapping with full
file reads.
Consequences: Host and SM120a capability builds share target names and their internal `bin`/`lib` layout, while
OS-named build roots prevent incompatible CMake caches from colliding. Windows uses Unicode-aware Win32 file
mapping and self-discovers MSVC through Visual Studio Build Tools. ASan/UBSan remains Linux-only until a
Windows sanitizer configuration provides comparable signal. Linux remains the production platform required by the
phase-one contract, while Windows is now a supported development and validation host.
Evidence: On the reference Windows installation, MSVC 19.44 and CUDA 13.3 configure and build both presets with
warnings as errors; host and CUDA CTest runs pass.

## 2026-07-21: Keep CUDA opt-in during repository initialization

Date: 2026-07-21  
Decision: Provide separate host-debug and Blackwell CUDA presets. Do not label the CUDA runtime probe as a native
kernel path.  
Context: Parser and manifest work must build on machines without CUDA, while performance builds must remain
architecture-specific.  
Alternatives: Require CUDA for every build; silently build host-only when CUDA is absent.  
Consequences: CPU CI stays useful; `GEM16_ENABLE_CUDA=ON` fails if CUDA is missing; native capability remains false
until implemented.  
Evidence: The neighboring `qwen35x` repository successfully uses optional CUDA language enablement, but its
silent CPU fallback was tightened here to a fatal error when CUDA is explicitly requested.

## 2026-07-21: Implement a strict in-repository JSON parser

Date: 2026-07-21  
Decision: Use a small C++ parser with duplicate-key rejection, resource limits, Unicode validation, and checked
integer parsing for initial config and Safetensors work.  
Context: Runtime dependency count should remain small and model files are untrusted input.  
Alternatives: Vendor a JSON library immediately; use string searching.  
Consequences: The parser is narrowly testable and dependency-free, but it carries maintenance responsibility and
must be fuzzed before the loader is considered production-ready.  
Evidence: Neighboring ad-hoc string-search Safetensors code does not meet this repository's schema and security
requirements.

## 2026-07-21: Target the 16 GB CUDA hardware class, Blackwell first

Date: 2026-07-21  
Decision: Define the product target as NVIDIA CUDA GPUs with approximately 16 GB VRAM. Optimize and validate the
first backend on the available Blackwell compute-capability-12.0 GPU.
Context: The engine should become useful across the 16 GB CUDA class; retail board form factors do not belong in the
architecture or project identity.
Alternatives: Bind the project to one retail board; attempt multi-architecture kernels before the first backend is
correct and competitive.
Consequences: Blackwell remains the immediate kernel and benchmark target. Later GPU backends must preserve the same
correctness, memory, and benchmark contracts, and exact board details remain benchmark metadata rather than product
scope.

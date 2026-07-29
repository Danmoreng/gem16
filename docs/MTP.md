# Gemma 4 12B MTP

## Feasibility conclusion

The MTP weights are **not embedded** in the pinned
`unsloth/gemma-4-12b-it-NVFP4` target checkpoint. Google publishes the compatible drafter separately as
`google/gemma-4-12B-it-assistant`. The first feasibility gate is positive: the official BF16 assistant is directly
loadable from Safetensors, its architecture matches the target, and its 806.54 MiB tensor payload leaves useful
headroom on the 16 GB reference GPU. Performance remains acceptance-dependent and must be proven in gem16 rather
than inferred from Google's “up to 3x” model-card statement.

The assistant is pinned in `models/gemma4-12b-mtp-assistant.lock.json` at commit
`364bd03c9952e5b7da73665ee30c9eccfc408345`.

## Evidence that the target contains no MTP weights

The target `config.json` declares only `Gemma4UnifiedForConditionalGeneration` and its 48-layer
`Gemma4UnifiedTextConfig`. It contains no assistant sub-config, MTP layer count, draft projection, centroid, or
speculative-decoding field. The authoritative target manifest contains 1,389 tensors:

- 1 tied `[262144, 3840]` target embedding/output matrix;
- exactly 48 normal decoder-layer families;
- 1 final target norm;
- 7 modality-only tensors.

No tensor name contains an MTP/draft/assistant family. In particular, the target has no `pre_projection`,
`post_projection`, assistant `[262144, 1024]` output matrix, or extra decoder layers. Tokenizer vocabulary strings
such as “draft” and “assistant” are ordinary vocabulary entries and are not model components.

## Official assistant contract

The pinned assistant config declares:

| Property | Value |
|---|---:|
| Architecture | `Gemma4UnifiedAssistantForCausalLM` |
| Model type | `gemma4_unified_assistant` |
| Target/backbone hidden size | 3,840 |
| Assistant hidden size | 1,024 |
| Assistant intermediate size | 8,192 |
| Assistant layers | 4 |
| Layer pattern | sliding, sliding, sliding, full |
| Query heads | 16 |
| Local/global head dimension | 256 / 512 |
| Shared-KV layers | 4 |
| Vocabulary | 262,144 |
| Maximum positions | 262,144 |
| Storage dtype | BF16 |

The Safetensors file has 48 tensors and 845,713,928 payload bytes:

| Tensor family | Bytes |
|---|---:|
| Assistant embedding/tied LM head `[262144,1024]` | 536,870,912 |
| Four Q-only decoder layers | 285,248,008 |
| Pre-projection `[1024,7680]` | 15,728,640 |
| Post-projection `[3840,1024]` | 7,864,320 |
| Final norm | 2,048 |

Each assistant attention layer has Q, Q norm, and O weights but no K/V weights. The three sliding layers read the
target cache produced by target Layer 46; the full layer reads target Layer 47. The assistant therefore needs no
independent long-context KV payload. Its pre-projection combines the current target token embedding and target
hidden state; the post-projection feeds a 3,840-dimensional state into the next proposal step. The assistant must
retain its own 1,024-dimensional tied LM head because it cannot share the target's 3,840-dimensional output matrix.

The target and assistant tokenizer JSON files are not byte-identical, but their complete 262,144-entry
`token -> id` vocabularies are equal. Runtime tokenization remains owned by the pinned target. The assistant's older
post-processor and missing added `<|video|>` marker are irrelevant to text-only drafting and must not replace target
metadata.

## 16 GB memory estimate

The assistant adds 845,713,928 persistent payload bytes (806.54 MiB), plus alignment, graph, and small activation
workspaces. It does not add a separate KV cache. Adding only the exact payload to measured target peaks gives:

| Target profile | Target peak | Target + assistant payload | Remaining from 16,303 MiB |
|---|---:|---:|---:|
| 8K decode characterization | 9,852 MiB | about 10,659 MiB | about 5,644 MiB |
| 128K QA | 11,022 MiB | about 11,829 MiB | about 4,474 MiB |
| 262,144-position QA | 12,244 MiB | about 13,051 MiB | about 3,252 MiB |

The first allocator proof now loads all 48 tensors into one independent 256-byte-aligned device arena. The exact
845,713,928-byte source payload occupies 845,714,944 arena bytes (806.54 MiB, including 1,016 alignment bytes).
`cudaMemGetInfo` measures an 847,249,408-byte device-memory delta (808 MiB) around that load. A sequential 50 ms
`nvidia-smi` probe at context 128 measures 9,660 MiB total GPU usage for the target-only process and 10,468 MiB
for target plus assistant, also an 808 MiB difference. The original assistant proposal workspace is 289,024 bytes
at context 128. Active batched MTP adds fixed tentative per-layer K/V and five-row output-selection storage: total
assistant-plus-verifier workspace measures 2,213,376 bytes with FP8 KV and 7,374,336 bytes with BF16 KV at context
128. The FP8 assistant workspace reserves the larger of its reference-score and split-online requirements. The
GPU acceptance result, stop-token table, and committed-hidden row are fixed workspace regions. No MTP graph pool
is retained: an exact 48-layer suffix-graph candidate added 6–8 MiB without improving 16K D2 throughput and was
removed.

## External runtime characterization

vLLM 0.25.1 recognizes the official assistant directly as `Gemma4MTPModel`, loads it beside the pinned Unsloth
target, and maps its sliding/full layers to target Layers 46/47. It reports 10.07 GiB model loading for the pair.
This confirms checkpoint-level compatibility without conversion. The unmodified graph path fails because
suppression-token list indexing creates a CPU index tensor during CUDA capture. The bounded patch in
`benchmarks/baselines/vllm/patches/gemma4-mtp-suppress-graph.patch` uses two scalar indices and enables capture.

On the exact 16K Wikipedia prompt with FP8 KV, graph D1/D2/D4 screens reach 49.59/58.69/56.06 effective tok/s. D2
then reaches 57.390 median tok/s over 3 warm-ups and 10 measured runs. Its 513 verifier groups accept 556 drafts
(mean accepted drafts 1.084), have 36.24 ms median group latency, and peak at a sampled 14,166 MiB. A fixed-1,135-token
screen reaches 57.363 tok/s and 35.75 ms/group. vLLM ordinary and MTP remain separately deterministic but are not
internally token-identical: they first differ at index 33 with stop semantics and index 2 at fixed length. These
results are therefore performance-headroom evidence, not an exact MTP baseline.

llama.cpp's dedicated Gemma 4 assistant path is also functional. The pinned converter produces an 861,520,160-byte
BF16 GGUF from the official assistant; runtime logs prove target-Layer-46/47 K/V sharing and full target/assistant
GPU residency. Current upstream `da5b4486` reaches 48.38 D2 tok/s in the fixed-1,135-token screen and 50.21/49.75
D2/D4 tok/s under stop semantics. Its target uses the patched closest-parity GGUF with BF16-mapped attention and
Q8_0 KV, and ordinary first differs from D2 at fixed output index 133, so it is likewise not an exact or
format-parity baseline.

The hardware conclusion is bounded. Complete fixed-D2 group capture now reaches 54.783 exact D2 tok/s in the
milestone's required one-warm-up/three-run Windows screen at mean accepted drafts 1.259. This clears the active
50.0 tok/s minimum and approaches the 55.0 tok/s stretch target, but the final alternating 3-warm-up/10-run
qualification remains deliberately deferred until GPU chaining, stop/tail handling, and streaming are complete.
Current llama.cpp reaches 48.38 tok/s
in the controlled fixed-1,135-token D2 screen and about 50 tok/s under different stop semantics. vLLM demonstrates
35.75 ms/group with a numerically different target batch route. The GPU-controlled graph roadmap now precedes
multimodal work, but non-exact causal/batched routes remain inadmissible and graph capture is not presumed to be a
speedup. A qualified 50 tok/s result meets the minimum, after which only material exact candidates may continue
toward 55.

A current-head Windows Nsight Systems trace at commit `b5ca0ef` contains 111 exact 16K fixed-D2 groups. Proposal
and verify/accept/commit CPU ranges total approximately 47.015 ms/group, while the verify range projects to a
41.407 ms GPU critical span. A representative group issues 1,407 kernel-launch API calls before its final stream
synchronization. The bounded 5--6 ms scheduling/control gap supports implementing device control and complete-group
capture next; synchronization time includes GPU execution, so the trace does not assume that the whole wait is
recoverable. Detailed kernel and API accounting is retained in `docs/PERFORMANCE_LEDGER.md`.

Earlier eager probes remain useful acceptance evidence: a random-token prompt reached mean acceptance 1.22 and
about 37.1 tok/s, while a natural CUDA essay reached 2.29 and about 65 tok/s. Together with the Wikipedia matrix,
this confirms that ordinary fallback remains mandatory.

## Correctness command

```bash
gem16-run \
  --model models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497 \
  --assistant-model models/checkpoints/google-gemma-4-12B-it-assistant-364bd03 \
  --mtp-draft-tokens 4 \
  --input-token-ids 2,9259,107 \
  --max-context 128 --max-tokens 16 --kv-cache fp8 --greedy
```

The independent bounded reference gate is:

```bash
third_party/cache/unsloth-nvfp4-env/bin/python tools/validate_mtp.py \
  --binary build/Linux/blackwell-release/bin/gem16-run \
  --target models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497 \
  --assistant models/checkpoints/google-gemma-4-12B-it-assistant-364bd03 \
  --draft-tokens 4 --output /tmp/mtp-reference.json
```

Active MTP supports greedy `gem16-run` and resident `gem16-chat` generation. It performs fixed-shape batched exact target
verification for draft lengths 1, 2, and 4, retains tentative K/V rows in a fixed workspace, and accepts and commits
the exact prefix on GPU. Drafts remain device-resident through verification; one small pinned result and one host
synchronization remain per group for the non-chained D1/D4/adaptive paths. `--mtp-adaptive` explicitly enables
context/acceptance-based D4→D2→D1 selection and bounded ordinary decode fallback. Resident chat loads the official
assistant once, preserves the exact target KV prefix between turns, and reinitializes the fixed-D2 device control
from each newly prefetched suffix. The GPU-chained D2 callback ring streams verified text without a per-group host
roundtrip. Sampling remains rejected in the current binary while the sampled-MTP gate below is implemented;
diagnostic dumps remain outside active MTP rather than silently changing semantics.

## Sampled-MTP qualification gate

Sampled MTP is the next required feature before multimodal work. The first production contract is deterministic
seed identity with ordinary target sampling, not greedy acceptance and not an unimplemented probability-ratio
rejection sampler. For every verifier row, the target applies the same suppression, temperature, top-k, top-p,
optional min-p, repetition penalty, and SplitMix64 output step that ordinary sampled decode would use along the
draft prefix. A proposal is accepted only when it equals that target-selected token. On the first mismatch the
target sample is emitted; later speculative rows and their K/V, hidden, repetition, and RNG state are discarded.
If all proposals match, the final target row supplies the bonus token.

The implementation order is:

1. Parse and validate the pinned target `generation_config.json` sampling defaults. The recommended chat profile is
   `temperature=1.0`, `top_k=64`, and `top_p=0.95`; explicit CLI overrides remain observable.
2. Add a direct fixed-shape sampled verifier with full target logits, one row-specific repetition mask per verifier
   row, and transactional commit of only the emitted prefix. Keep assistant proposals deterministic initially.
3. Prove same-seed ordinary/MTP token identity for D1/D2/D4 across multiple seeds, stop positions, final-length
   tails, repetition penalties, FP8/BF16 K/V, and local-ring wrap. Report acceptance separately from greedy.
4. Extend fixed-D2 device control with the sampling step and capture the sampled verifier, commit, stop/tail, and
   streaming path in the GPU-chained graph. No per-group host dependency or token-loop allocation is allowed.
5. Add a real resident multi-turn chat gate, including unchanged RNG/repetition history across turns and exact
   callback order, then run Linux 3-warm-up/10-run qualification with memory, clocks, power, and thermal evidence.

Assistant probability materialization and standard `min(1,p/q)` speculative rejection are a separate possible
future algorithm. They require proposal distributions and independent statistical qualification and must not be
inferred from the current argmax assistant head.

## GPU-controlled decode graph roadmap

The long-term production boundary is one fixed-address CUDA execution plan that continues greedy MTP generation
without a blocking host control roundtrip after every verification group. The host remains responsible for request
setup and may consume streamed verified tokens, but it must not supply the next token, accepted length, position,
or stop decision before GPU compute can continue. This is an architectural and latency objective; it does not
assume CUDA Graph replay alone will meet the 50 tok/s performance gate.

Delivery is deliberately incremental:

1. **Complete: device control with host parity.** An arena-backed `MtpDeviceControl` contains the current input token,
   processed position, remaining output capacity, stop state, output write position, and fixed-D2 mode. GPU
   acceptance updates a shadow next state. The existing host loop and compact result synchronization remain, and
   every group asserts that host and GPU transitions agree. The host supplies the current record with one small
   asynchronous H2D copy, while the existing D2H transaction returns both the group result and GPU-computed next
   state. This phase does not change production kernel ordering, synchronization count, or output.
2. **Complete fixed-D2 group graph.** Capture both recurrent assistant proposal steps, verification-input build,
   embedding, all 48 target layers, final norm/output selection, acceptance, KV/hidden commit, and control update.
   The host initially replays one group at a time and still reads the result. This isolates graph-capture and
   controlled-position correctness from loop and streaming changes.
3. **Complete: GPU-chained fixed-D2 loop.** CUDA 13.3 conditional `while` nodes repeat complete groups while the
   device control has capacity for another three-row verification batch. Verified target tokens, proposals, and
   aggregate counters are stored in preallocated device buffers. There is no D2H/H2D dependency between groups;
   the result is copied only after the conditional graph finishes in this non-streaming form.
4. **Complete: stop and tail semantics.** EOS/stop checks and remaining-length accounting stay in device control.
   When fixed D2 leaves one or two slots, a second conditional `while` node runs exact ordinary target forwards and
   publishes their tokens without returning to the host scheduler. Position and ring-wrap gates match ordinary.
5. **Complete: asynchronous streaming.** A bounded, preallocated 256-token single-producer/single-consumer ring in
   mapped pinned memory carries only target-verified tokens. Device system-scope atomics publish in order; the host
   invokes the existing callback while polling graph completion without synchronizing the compute stream. A slow
   consumer applies explicit device backpressure, and callback failure publishes cancellation at the next group
   boundary. Streaming and callback time remain inside end-to-end timing.
6. **Sampled fixed-D2 graph and resident chat.** Apply the sampled-MTP contract above with row-specific target
   sampling state, transactional RNG/repetition commit, device stop/tail handling, and the existing asynchronous
   streaming boundary. Qualify ordinary/MTP identity per seed and resident multi-turn continuation on Linux.
7. **Adaptive graph branches.** After fixed D2 sampled streaming is stable, add conditional D1/D2/ordinary paths
   for the existing adaptive policy. D4 may follow as a separately captured fixed shape.
8. **Optional N-Gram proposer branch.** Much later, and only after the graph above is stable, measure a
   device-resident `ngram-mod`
   lookup ahead of MTP. On a qualified hit it supplies fixed D2 or D4 proposal tokens and skips the assistant; on a
   miss it falls through to MTP. Both sources reuse the same exact target verification, acceptance, transactional
   commit, stop/tail, and streaming nodes. Do not concatenate an unverified N-Gram prefix with MTP hidden-state
   recurrence, and do not add long variable target batches until an ordinary-identical verifier exists for each
   captured shape.

The optional N-Gram design uses a fixed-capacity arena hash table populated from the device token history after
prefill and updated only with emitted target-verified tokens. Lookup, collision policy, occupancy/reset behavior,
and draft-length selection must be deterministic. Initial evaluation is D2, then D4; 48–64-token llama.cpp-style
drafts are out of scope until exact long-batch target verification is independently proven. The graph control may
later expose `proposal_source`, `proposal_count`, and a shared proposal-token region, but the first
`MtpDeviceControl` change remains model-specific and must not introduce unused abstraction.

Current llama.cpp fixed-1,135-token screens justify evaluation but not promotion. Match lengths 8–24 produced no
N-Gram proposals on the Wikipedia summary. Match length 2 generated drafts, but N-Gram-only mean accepted drafts
were only about 0.12–0.13 and active N-Gram/MTP cascades reached about 45.86–48.96 tok/s versus a 50.01 tok/s
single-run MTP-D2 screen. llama.cpp gives draftless proposers priority over MTP rather than merging their token
streams. Therefore gem16 must report per-source hit, proposed, accepted, rejected, and fallback counts, and retain
N-Gram only if representative end-to-end suites improve without changing ordinary output.

Every phase retains the ordinary non-MTP route as the semantic reference and must prove the fixed Wikipedia
1,135-ID SHA-256 `43bc3380fc1cce5182a679fa3a340c04bcc79c52e73d5102ec1f737f57d0a1e1`, 632 accepted and 372 rejected drafts
over 502 groups, zero fallback, zero token-loop allocation, deterministic cache positions, and unchanged stop
behavior. Add focused tests for control overflow, zero acceptance, one/two accepted drafts, final-length tails,
stop in each emitted slot, ring wrap, slow consumers, and graph replay/reset. Each phase requires an Nsight timeline
and direct-launch comparison; neutral graph steps may be retained only when they are necessary, bounded foundations
for the next no-roundtrip phase and their memory cost is documented.

## Implementation status and order

1. **Complete:** extend the inspector/config parser for `gemma4_unified_assistant` and its exact 48-tensor BF16
   manifest. Primary inference rejects an assistant passed as the target instead of entering target-only code.
2. **Complete:** verify the pinned assistant with `tools/fetch_model.py --lock
   models/gemma4-12b-mtp-assistant.lock.json`.
3. **Complete:** add a separate fixed-address BF16 assistant arena without quantization or conversion. Every
   tensor is bound at its exact shape, and device prefix/suffix probes cover all 48 uploads. `gem16-run
   --assistant-model` reports source, arena, and measured device-delta bytes. A 16-step paired residency run retains
   identical ordinary target IDs, memcheck reports zero errors, and Nsight places all five `cudaMalloc` calls before
   the prefill range with none in prefill or decode.
4. **Complete:** bind the three sliding assistant layers to the target Layer-46 cache and the full assistant layer
   to target Layer 47, for both checkpoint-FP8 and BF16 cache modes. Draft iterations keep the target position and
   shared cache constant, matching the official proposer contract.
5. **Complete:** implement BF16 target-embedding/pre-projection, four Q-only attention/MLP layers, final norm,
   post-projection feedback, the tied 1,024-dimensional LM head, and the assistant's two suppressed token IDs.
6. **Complete for correctness:** implement draft lengths 1, 2, and 4 with fixed-shape batched exact target
   verification. The target evaluates `[input token, draft_1, ...]` in one causal batch; target predictions, not
   assistant agreement, choose every emitted token. Tentative per-layer K/V rows are retained in a fixed workspace,
   local-ring writes are restored before host acceptance, and only the accepted prefix plus first mismatch is
   committed to the target cache.
7. **Complete for correctness:** report proposed/accepted/rejected token counts, proposed IDs, verification groups,
   evaluated target positions and batches, mean accepted length, effective output tok/s, and incremental memory.
   FP8 and BF16 runs retain exact ordinary greedy output, including a local-ring wrap test. `tools/validate_mtp.py`
   proves the first four recurrent BF16 drafts exactly equal Transformers (`1884,5745,993,236771`) on a complete
   one-token shared cache. Active full-process memcheck passes. Full-process initcheck is not accepted as MTP
   evidence because it reports pre-existing uninitialized padded target-prefill reads in CUTLASS
   `ApplyScalesKernel` before proposal execution; full-process racecheck exits before application output with no
   displayed hazards. Targeted assistant sanitizer coverage remains required before performance promotion.
8. **Complete first verifier optimization:** recursive assistant token selection remains device-resident through a
   draft group, and exact fused native Gate/Up/GELU reuse replaces separate MTP Gate and Up launches. The initial
   FP8 CUTLASS batch promotion passed the bounded context-512 gate but was later narrowed to O only after Q/K/V
   failed the full 16K exactness gate. On the natural 53-token/256-output context-512 workload, 3 warm-ups plus 10
   alternating runs measure 42.90 tok/s MTP versus 35.27 ordinary (+21.6%), with mean accepted length 1.89. This is a workload-specific effective-throughput win, not a general 60 tok/s claim.
9. **Complete bounded scheduler work:** drafts now remain device-resident, GPU acceptance applies stop IDs and
   commits tentative K/V plus the selected hidden row before one compact D2H result, and `--mtp-adaptive` exposes
   deterministic context/acceptance thresholds with ordinary fallback. An exact fixed-shape graph over each
   layer's position-independent verifier suffix added 6–8 MiB but measured 35.291 versus 35.340 tok/s and was
   removed. Full position-controlled graph capture is deferred because the refreshed profile is GPU-kernel-bound.
   The exact direct decode-attention verifier remains mandatory: a faster causal-prefill attention candidate
   reached 55.06 tok/s but changed output at step 15 and was removed. NVFP4 CUTLASS verifier projections also
   changed the natural sequence and were removed.
10. **16K correctness and short-batch kernels complete:** target verification keeps decode-order direct grouped
   Q/K/V and exact CUTLASS O. For T≤5, FP8 Q/K/V now uses the latency-oriented decode MMA over 2/3/5 rows instead
   of staging a 128-token tile. NVFP4 Down uses one unstaged 16-token tile and four warps instead of the prefill
   128-token/8-warp plan. Both preserve MMA K accumulation and all 1,135 ordinary IDs. D2 screening moves from
   35.340 to 39.150 after FP8 Q/K/V and to 43.200 tok/s after NVFP4 Down. T1 Gate/Up, 8/16-head global attention,
   and suffix graphs were exact but did not win and were removed. Active FP8 ring-wrap memcheck reports zero errors.
11. **Qualified 16K D2 win:** three alternating warm-up pairs and ten alternating measured pairs produce 31.798
   ordinary versus 42.639 MTP D2 median tok/s, a 1.341x throughput speedup (+34.1%). All 20 measured outputs contain
   the same 1,135 IDs; mean accepted length is exactly 1.259 in every MTP run. The 95% mean CIs are
   `[31.783,31.806]` and `[42.623,42.658]`. Peak sampled GPU memory is 10,838 MiB. D2 remains workload-dependent;
   explicit ordinary decode and adaptive fallback are retained. The revised 50 tok/s minimum is not yet reached.
12. **External feasibility matrix complete:** patched graph-vLLM reaches 57.390 D2 tok/s over 3/10 runs and current
   llama.cpp reaches 48.38 tok/s in a fixed-length D2 screen. Both execute the official assistant with shared target
   KV, but neither preserves its own ordinary greedy sequence, and llama.cpp also changes target/KV formats.
   vLLM's 35.75 ms verifier-group latency proves sufficient hardware headroom, not an admissible numerical
   implementation. The final bounded exact-verifier sprint targets 45.18 ms/group and 50 tok/s first, then 41.07
   ms/group and 55 tok/s only through material exact candidates. If it cannot reach 50, retain the best exact
   characterization and mark the performance target unmet rather than weakening semantics.
13. **Next, before multimodal:** continue the GPU-controlled decode-graph roadmap above with GPU-chained fixed-D2
   execution now that device-control parity and complete fixed-D2 group capture are complete. Stop/tail handling,
   asynchronous streaming, and adaptive graph branches follow as separate
   correctness-gated changes. A device-resident N-Gram proposer is an
   optional later branch only after those foundations and its own hit-rate/performance gates. Multimodal
   implementation remains queued until fixed-D2 GPU chaining and nonblocking streaming are complete or a new
   decision documents a blocker; optional N-Gram qualification is not itself a multimodal blocker.
14. **Device-control parity phase complete:** `MtpGroupTransaction` now places the result and 16-byte-aligned
   `MtpDeviceControl` in the fixed MTP arena. GPU acceptance derives the next token, processed position, remaining
   output capacity, output offset, and stop state. The host validates the copied transition before using the result
   and validates the complete post-update host state before scheduling another group. CUDA fixtures cover zero,
   one, and two accepted D2 drafts plus stop-token truncation. Short BF16 D1/D2/D4 and FP8 D2 runs are exactly equal
   to ordinary greedy output, report `device_control=host_gpu_transition_parity`, and use neither fallbacks nor
   token-loop allocations. The Windows 16K Wikipedia gate (one warm-up, three measured runs) retains all 1,135 IDs
   and SHA-256 `43bc3380fc1cce5182a679fa3a340c04bcc79c52e73d5102ec1f737f57d0a1e1`, with 632 accepted and 372 rejected
   drafts over 502 groups. Median throughput is 45.217 tok/s; the 706,618,112-byte reported workspace is unchanged.
   Nsight records the same 185,830 kernel launches and 117 stream synchronizations as the pre-change trace, plus
   exactly 111 `cudaMemcpyAsync` calls for 111 control records. The full Transformers draft-reference script could
   not rerun on this Windows Python because its installed PyTorch lacks CUDA; its engine-side exact comparison
   completed before that failure, and the previously retained independent Transformers draft fixture remains the
   reference evidence. This phase supplied the correctness foundation used by the complete fixed-D2 group graph.
15. **Complete fixed-D2 group graph:** checkpoint-FP8 fixed D2 with a context budget above 1,024 captures both
   assistant proposal steps, controlled verification inputs, all 48 target layers, output selection, GPU
   acceptance, KV/hidden commit, control transition, and the compact D2H transaction in one reusable graph.
   Short-context and BF16 modes retain the exact direct implementation. The Wikipedia 16K one-warm-up/three-run
   gate measures 54.783 tok/s median versus the preceding device-control milestone's 45.217 tok/s (+21.2%), with
   the unchanged 1,135-ID SHA-256, 632 accepted and 372 rejected drafts over 502 groups. A 1,022-token prompt plus
   16 outputs crosses the local-cache ring boundary exactly, replays 13 D2 groups plus a direct D1 tail, reports
   14,680,064 graph-associated device bytes, and retains zero token-loop allocations. The 256-output Nsight trace
   records 111 `cudaGraphLaunch` calls, reduces whole-process `cudaLaunchKernel` calls from 185,830 to 14,770, and
   retains 116 stream synchronizations. One synchronization per group remains; GPU chaining is the next phase.
16. **Complete GPU-chained fixed D2:** one CUDA conditional `while` node now repeats the complete group graph until
   stop or fewer than three output slots remain. A fixed device buffer retains target-verified outputs and proposal
   IDs, while a compact aggregate stores group/acceptance counters; all are copied only after graph completion.
   The exact Wikipedia 16K one-warm-up/three-run gate reaches 55.063 tok/s median, retains the 1,135-ID SHA-256 and
   632/372 acceptance counts over 502 groups, and stops on the same token. The 256-output Nsight trace contains one
   `cudaGraphLaunch` and six whole-process stream synchronizations, versus 111 graph launches and 116
   synchronizations for host replay. Ring wrap at local-cache position 1,024 and a focused two-iteration
   conditional-node fixture are exact. Final D1/ordinary tails remain host-scheduled, and callbacks are delivered
   only after the bulk chain completes; stop/tail consolidation plus asynchronous streaming is the next phase.
17. **Complete GPU tail and asynchronous streaming:** a dependent ordinary-tail conditional node consumes the last
   one or two output slots entirely on device, including stop checks and cache updates. A 256-token mapped-pinned
   SPSC ring publishes verified D2 and tail tokens with system-scope release/acquire ordering while the host polls
   callbacks without a compute-stream synchronization. The slow-consumer fixture fills the ring, observes exactly
   one backpressure event, releases the consumer, and completes 258 ordered outputs; mapped publication, graph
   reset, and local-KV ring wrap also pass. Wikipedia 16K with one warm-up/three runs measures 55.009 tok/s median,
   retaining the exact 1,135-ID SHA-256 and 632/372 counts over 502 groups. Nsight records one graph launch and five
   whole-process stream synchronizations; none is in the chained decode boundary. The 32K graph allocation is
   23,068,672 bytes, and the mapped ring is fixed at 1,088 bytes.
18. **Complete alternating qualification:** three alternating warm-up pairs and ten alternating measured pairs
   preserve the same 1,135 IDs in all 26 runs. Ordinary reaches 36.788 tok/s median (95% mean CI
   `[36.715,36.837]`); fixed D2 reaches 54.903 tok/s (`[54.557,55.132]`), a 1.492x speedup (+49.2%). Every D2 run
   reports 1,004 proposed, 632 accepted, and 372 rejected drafts over 502 groups with zero ordinary fallback. This
   passes the 50 tok/s competitive gate and misses the 55 tok/s stretch target by 0.097 tok/s. Raw results are under
   `benchmarks/results/2026-07-28/b07b178/blackwell16gb-windows-mtp-streaming/qualification.json`.
19. **Next: sampled MTP chat qualification.** Implement the sampled-MTP gate above, then qualify a real resident
   multi-turn session and reproduce the sampled and greedy GPU-chain evidence on Linux. Multimodal and N-Gram work
   remain queued until this gate is complete.

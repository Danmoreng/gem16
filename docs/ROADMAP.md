# Roadmap

Current stage: Milestone 3, first full-model greedy characterization.

Linux and Windows host/CUDA build scaffolding is now available. This makes loader development and validation on the
same Blackwell machine possible from either operating system; it does not move Windows production inference ahead
of the correctness and native-kernel gates below.

## Active gate

- Resume the binding [prefill optimization plan](PREFILL_OPTIMIZATION_PLAN.md) under Linux on the reference laptop.
  The max-power cross-engine qualification places exact fixed-D2 gem16 decode at 85.459 tok/s, ahead of the
  adjacent vLLM and llama.cpp characterizations, while 16K prefill remains at 78.7% of vLLM. The immediate target
  is at most 2,597.6 ms TTFT for the exact 16,384-token workload without regressing decode, correctness, or the
  15.3 GB peak-memory limit. Work proceeds through current-head profiling, larger prompt geometry and shape-tuned
  NVFP4, recurring-layout elimination, packed/scaled FP8 projections, attention specialization, and only then
  prefill CUDA Graphs. Each retained stage requires an adjacent repeated benchmark and its own commit.

- Historical implementation record: statements below that close or pause earlier optimization sprints describe
  their status at the time and are superseded by the active decode gate above. The refreshed prefill record begins
  from the retained Linux profile.
  Online Tensor-Core attention, the deterministic 2,048-token prompt plan, CUTLASS NVFP4/FP8 projections, and
  profile-proven fusions are promoted. The 2026-07-27 bounded staging sprint vectorizes local FP8 conversion and
  removes redundant modulo from contiguous global-cache reads. Attention and FP8 projection GEMMs are now tied as
  the largest 8K families. A fresh controlled comparison now confirms gem16 leads the patched closest-parity
  llama.cpp candidate across the common matrix while reaching 57–78% of direct-vLLM prefill and 85–86% of its
  decode throughput under disclosed non-parity timing boundaries. The bounded staging sprint is closed. The first
  exact GPU sampling plan now covers temperature, top-k, top-p, min-p, repetition penalty, and deterministic
  seeded RNG without changing the greedy path or allocating in the token loop. It is isolated in an operator
  module, has synthetic and checkpoint-backed CPU/GPU gates, and runs inside the whole-model decode CUDA Graph at
  measured greedy-performance parity. A bounded decode-fusion sprint then reused the exact quantization boundaries
  and combined controlled Q/K rounding, normalization, and RoPE, reducing graph kernel nodes by 39.3% and raising
  final 128/2,048/8,192 medians to 34.446/34.257/33.545 tok/s. At 8K this reaches 88.1% of the retained direct-vLLM
  characterization, up from 85.6%. The decode sprint is closed. The bounded 64K text gate now passes a single
  mixed retrieval and 1,024-token soak: all markers at 10/50/90% are returned in order, the hybrid cache completes
  repeated local-ring wraparound with no fallback or token-loop allocation, and peak sampled GPU memory is
  10,418 MiB. This run is correctness/soak evidence, not a repeated performance result. Further kernel optimization
  is paused. Single-run Wikipedia QA probes now also complete at 131,072 prompt positions and at the exact
  262,144-position prompt-plus-decode limit, peaking at 11,022 and 12,244 MiB. MTP checkpoint and memory
  feasibility is now positive: the target contains no embedded MTP family, while Google's separately published
  806.54 MiB BF16 four-layer assistant is pinned, directly compatible, and leaves estimated maximum-context
  headroom. The exact assistant config and 48-tensor BF16 manifest are now accepted by `gem16-inspect --validate`,
  while target-only inference rejects an assistant checkpoint explicitly. The separate fixed-address BF16
  assistant arena is now directly uploaded and fully bound: 845,713,928 source bytes require 845,714,944 arena
  bytes and the measured device-memory delta is 808 MiB. The full BF16 assistant now executes against target
  Layers 46/47 with recurrent constant-position drafting at lengths 1, 2, and 4. Fixed-shape batched target
  verification retains ordinary greedy IDs in FP8/BF16 and across local-ring wrap; a bounded fixture matches four
  Transformers draft IDs exactly. The verifier retains transactional K/V rows, restores local-ring slots before
  host acceptance, and commits only the exact prefix. The Wikipedia 16K gate now also retains all 1,135 ordinary
  IDs after narrowing FP8 CUTLASS verification to O and restoring direct grouped Q/K/V. GPU acceptance/commit,
  device-resident drafts, exact T≤5 FP8 Q/K/V and NVFP4 Down kernels, and explicit adaptive drafting are complete.
  The final qualified Wikipedia 16K gate uses three alternating warm-up pairs and ten alternating measured pairs:
  ordinary reaches 36.788 median tok/s and GPU-chained exact D2 reaches 54.903 (1.492x, +49.2%) with all 1,135 IDs
  equal in all 26 runs. It passes the 50 tok/s minimum and misses the 55 tok/s stretch target by 0.097 tok/s.
  An exact verifier-suffix graph added memory without speed and was removed; full position-controlled graph work is
  deferred while profiles remain kernel-bound. The external feasibility matrix is now complete: patched graph-vLLM
  reaches 57.390 D2 tok/s and 35.75 ms/verifier group, while current llama.cpp reaches 48.38 fixed-length D2.
  Both external MTP routes diverge from their own ordinary greedy sequence and are characterization only. vLLM's
  group cost nevertheless proves sufficient hardware headroom. The final bounded exact-verifier sprint is now
  closed. Three-row global attention, FP8 Q/K/V/O, output-head, and NVFP4 Down specialization raise the exact
  one-run Wikipedia characterization from the retained 44.347 tok/s Direct-O candidate to 46.422 tok/s, with all
  1,135 IDs and 632/372 acceptance unchanged. This remains below the 50.0 tok/s minimum, so it is not a 3/10
  qualification and no 55 tok/s work is authorized. At the user's explicit direction, exact D2 work is now
  reopened only for structural candidates with modeled headroom. Tensor-Core target-head and direct-tentative-K/V
  local-attention variants were exact but slower and are removed. Retained shared-activation staging lowers the
  profiled fixed-T3 FP8 projection family from 8.06 to 6.44 ms/group. A controlled 3-warm-up/5-run comparison raises
  median exact D2 from 45.805 to 47.432 tok/s (+3.55%) with the fixed 1,135-ID hash and 632/372 acceptance in every
  run. This remains a characterization below the 50 tok/s gate, not the required 3/10 qualification. Numerically
  different MTP output remains forbidden, and llama.cpp investigation is out of scope. The completed architectural
  milestone includes device-control parity, one complete fixed-D2 group graph, GPU-chained groups, device stop/tail
  handling, and a nonblocking GPU-producer/host-consumer streaming ring. Sampled MTP in resident chat is now
  implemented: pinned Google generation defaults, same-seed ordinary Target decisions, transactional repetition/RNG
  state, fixed-D2 GPU chaining, asynchronous streaming, and a real two-turn model gate all pass. The Linux 3/10
  sampled run is exact and 1.470x faster than ordinary, but its 46.234 tok/s median remains below the existing
  50 tok/s performance target and lacks continuous resource telemetry. Adaptive D1/D2/ordinary sampled branches
  remain a possible follow-up. Bounded reasoning now stays within one device-routed fixed-D2 graph: safe reasoning
  groups use MTP, exact marker/budget/tail rows use the ordinary child, and D2 resumes without a host control
  roundtrip. N-Gram is deferred much further. The first audio milestone is now
  implemented: all unified audio/vision tensors stay resident, native chat
  expands bounded WAV/FLAC/MP3 input into exact audio placeholder rows, and a GPU BF16
  RMSNorm/projection replaces those embeddings before Layer 0. Image
  preprocessing and execution remain queued.
  Every performance promotion still requires correctness, generation, logit, 3-warm-up/10-run benchmark, Nsight,
  spill, allocation, and peak-VRAM evidence and becomes the sole production path.

- The arena-backed 48-layer engine loads all unified tensors once, uses fixed workspace/KV arenas, executes the
  tied output head and GPU argmax, and supports explicit checkpoint-FP8 and BF16 K/V semantics. Physical
  byte-per-value E4M3FN storage is implemented. Cross-runtime generation is not expected to be bit-identical:
  gem16, direct FP8-vLLM, and the closest llama.cpp candidate use different kernels and, for llama.cpp, different
  attention-weight and KV formats. The former sky-token mismatch has been deeply localized and is retained as
  numerical-characterization evidence, not an active exact-token blocker. Current gates are deterministic output,
  operator contracts, teacher-forced rank/distribution metrics, and broader quality evaluation.
- The pure C++ chat CLI now uses native byte-fallback BPE from `tokenizer.json`, a version-bound implementation of
  the exact checkpoint `chat_template.jinja`, and its EOS/suppressed-token lists. Interactive chat now retains one
  model, execution plan, and exact-token KV cache across turns; it batch-prefills only the newly appended
  conversation suffix and rejects a non-extending token prefix rather than silently rebuilding or corrupting the
  session.
- Keep chat processing independent of terminal I/O so a later OpenAI-compatible Chat Completions server can reuse
  it. Do not begin HTTP/server work before persistent engine sessions and the correctness gate are in place.
- [x] Replace token-at-a-time prompt ingestion with a separate native context-budgeted prefill plan (128-token
  default). Keep the serial
  implementation only as a test/probe reference; production CLIs expose the measured native winner.
- [x] Replace the initial contiguous physical FP8 cache, formerly capped at 1,024 total positions, with circular local
  storage and independently growing global storage. Checkpoint-scale FP8 numerical semantics, one-byte allocation,
  online prefill attention, and split/merge long-context decode reads are implemented and deterministic.
- Extend the now-committed trusted vLLM token/top-logprob fixture with full-vocabulary logits and selected hidden
  states. The 12-prompt FP8/BF16 teacher-forced suite is complete and places every vLLM Top-1 in gem16's Top-5;
  full reference vectors remain pending.
- Finish the quality and native-dispatch gates for the patched same-source closest-parity GGUF, then select and lock
  quality-acceptable GGUFs for llama.cpp tiers B and C. Unpatched upstream conversion remains blocked.
- Extend the implemented deterministic weight/scale/KV base arena with execution-derived activation, logits,
  sampling, graph, kernel, and prefill workspace requirements.
- Keep the implemented CPU NVFP4 oracle as the authority while assembling the first complete MLP layer. The exact
  E2M1, E4M3FN, source-nibble, global-divisor, activation-quantization, and FP32-accumulation contracts are now
  executable and tested.
- Assemble Gate/Up, GELU-tanh product, Down, and residual without weakening the now-complete real-checkpoint proof:
  all three Layer-0 shapes consume the source packed weights directly and the mandatory exact load-time
  `row8/K64` scale layout, CPU/GPU activation bytes match exactly, and CUDA reference/native output differences
  are at most `1.1920929e-7` in the characterization fixture.
- Preserve the now-complete unfused Layer-0 device chain while replacing its deterministic hidden/cache fixture
  with prompt-derived trusted Layer-0 input, output, and K/V state. Both NVFP4 activation boundaries currently
  remain byte-identical between CUDA-reference and direct SM120 execution.
- Use the retained direct vLLM characterization as a native-format performance reference. It is 1.66x–2.34x ahead
  of the patched llama.cpp candidate in prefill and 1.25x–1.26x ahead in decode through 8K, but BF16 KV capacity,
  timing-boundary differences, and autotuning fallbacks keep it from being an accepted parity baseline.

## Baseline gate

The llama.cpp benchmark is deliberately before engine kernel optimization, but after source-checkpoint validation:

1. The physical C++/Python manifest comparison must be exact. This is complete for all 1,389 tensors.
2. A trusted direct-load runtime must produce fixed token IDs and reference logits. Batch-one greedy token IDs and
   top-20 log probabilities are now committed and reproduce exactly; full-vocabulary logits remain pending.
3. The converter must emit a tensor mapping report for the same source revision. Current unpatched upstream rejects
   the checkpoint's mixed FP8/NVFP4 groups. The tracked patch now produces a 955-tensor closest-parity GGUF with 144
   NVFP4 MLP tensors and BF16-mapped attention; its exact inventory and checksum are committed.
4. Only after native SM120 NVFP4 execution, GPU residency, and quality are proven may timings be labeled as a
   closest-parity or native-NVFP4 baseline.
5. Maintain a separate fastest-practical llama.cpp baseline even if exact mixed-format parity is impossible.

## Next milestones

1. Accept or reject the patched closest-parity characterization, then establish viable llama.cpp tiers B and C.
2. Capture full-vocabulary vLLM reference logits and selected hidden states. The engine raw-float32 logit dump and
   comparison against the committed vLLM top-20 fixture are implemented; full reference vectors remain pending.
3. ~~Implement the exact host NVFP4 codec and projection oracle, including real-checkpoint byte-pattern fixtures.~~
4. ~~Implement an explicit correctness-only CUDA W4A4 route that consumes packed E2M1 values and E4M3 scales.~~
5. ~~Implement and round-trip-test direct SM120 fragment views over source packed-weight storage for Gate, Up, and
   Down.~~ Packed weights require no repack; local scales now use the exact mandatory load-time `row8/K64` layout
   directly in their final allocation, without a second persistent copy.
6. ~~Compare the implemented SM120a `m16n8k64` decode projection for `T=1` with a bandwidth-oriented
   packed-NVFP4 SIMT/GEMV candidate.~~ The direct MMA route is 3.1x–6.2x faster on the real Gate/Up/Down shapes and
   remains the production candidate. Disassembly proves `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`; retain the SIMT
   implementation only as a direct-source characterization control.
7. The correctness-first Layer-0 MLP chain now implements Gate/Up, Gemma GELU-tanh product, Down, post-MLP norm,
   residual, and layer scalar as part of the complete device-resident decoder-layer characterization. Both NVFP4
   activation boundaries remain byte-identical across its reference/native paths. Next compare it with trusted
   prompt-derived layer data. Combined Gate/Up projections remain rejected, but prefill now keeps the two winning
   projections separate and fuses their exact BF16/GELU-tanh/NVFP4 activation boundary.
8. ~~Add a separate native prefill plan without reusing the decode plan.~~ The promoted plan uses one 1,024-token
   checkpoint-FP8 chunk, online Tensor-Core attention, and M128xN64 NVFP4 CTAs that reuse an exact K64 activation
   slice across eight output warps. FP8 projections now use the qualified two-stage M128xN64xK64 CTA, reuse each
   weight fragment across eight MMA token tiles, and group local Q/K/V or global Q/K into one launch. Local/global
   attention CTAs share staged K/V across 2/4 query heads, and checkpoint-FP8 prefill uses a qualified 2,048-token
   chunk while committing only the newest 1,024 local positions to the ring.
   NVFP4 activation staging now uses a qualified two-stage `cp.async` pipeline. Packed E2M1 weights and local E4M3
   weight-scale bytes are tiled exactly once into the sole final Row8/K64 GPU allocation; persistent bytes are
   unchanged and no raw GPU copy survives. Large/grouped FP8 projection work is complete. Exact
   RMSNorm/quantization, residual, and MLP
   activation-boundary fusions are promoted as the sole prefill path, reducing context-512 launches by 40.0% and
   improving 128/512/2,048-token medians by 8.0%/9.5%/10.2%. Exact projection rounding, Q/K norm, RoPE, and final
   rounding are now a second sole-path fusion; persistent exact RoPE tables raise medians another
   15.17%/15.14%/16.76% and reduce launches to 964 per 512-token prefill. The final K/V-write fusion was correct
   and reduced the affected boundary and launch count, but only moved the context-512 median by 0.79% with strongly
   overlapping intervals; it was removed completely. Fusion work is closed with no alternate path, and work now
   returns to the dominant NVFP4 and FP8 projection pipelines.
   The ordered execution and qualification contract for online Tensor-Core attention, larger prompt chunks,
   pipelined projections, and later fusion is now fixed in
   [the prefill optimization plan](PREFILL_OPTIMIZATION_PLAN.md).
9. The checkpoint's FP8 Q/K/V/O projection path is implemented with an independent CPU oracle, CUDA reference,
   direct-source `QMMA.16832` route, and real Layer-0 checks. The unfused local-attention decode sublayer assembles
   input RMSNorm, Q/K/V, per-head Q/K and scale-free V normalization, RoPE, separate K/V append/read, FP32 softmax,
   O projection, post-attention RMSNorm, and residual over a deterministic 32-token cache. The distinct real
   Layer-5 full-attention route now reuses the raw K projection for V, applies learned K norm plus proportional RoPE
   separately from scale-free V norm, and proves that the final cache states cannot share storage. The validated
   Layer-0 attention and MLP sublayers are now connected without a host roundtrip. Next replace the deterministic
   hidden/cache inputs with a trusted prompt-derived Layer-0 fixture. Follow ninfer's useful split-output planning
   pattern for combined projections while retaining this checkpoint's E4M3/BF16 scale contract.
10. The first fixed-address execution workspace and full-model BF16-semantics cache are implemented for contexts up
   to 1,024. Complete production workspace planning and add circular/FP8 cache, decode fusion, and CUDA Graph replay
   only after the unfused model passes layer, logit, and generation gates.
11. Long-context, exact greedy MTP, and same-seed sampled MTP correctness are established. The fixed-D2 graph,
   GPU chaining, sampled/greedy stop and tail semantics, asynchronous streaming, resident two-turn gate, Windows
   greedy 3/10 qualification, and Linux sampled/greedy 3/10 comparisons are complete. The user accepts the current
   reproducible 1.470x sampled Linux result as the feature baseline despite its 46.234 tok/s median and missing
   continuous resource telemetry; it must not be promoted as a publication-grade 50 tok/s result. Adaptive sampled
   branches and N-Gram are intentionally deferred.
12. Execute server-readiness milestone S0 before multimodal M0. **S0.1 complete:** the ADR fixes protocol/core and
   future `ModelRuntime`/`SessionState`/`ExecutionSlot` ownership. **S0.2 complete:** public owning text content,
   message, request, token-event, response, and finish-reason types form a server-neutral API. **S0.3 complete:**
   ordinary one-shot and resident interactive `gem16-chat` generation consume `ChatSession`; specialized
   render/JSON/state diagnostics retain their narrow path. Cancellation/final usage events and physical
   shared-weight multi-session ownership remain later S0 work. Next execute the binding
   [multimodal expansion plan](MULTIMODAL.md). Audio and the complete
   encoder-free vision path are now implemented for one-shot, resident, and server chat,
   including GPU projection and sliding-layer bidirectional vision attention.
   Portable PNG/JPEG/BMP and WAV/FLAC/MP3 decoding is integrated. Multiple and
   resident image/audio requests are qualified on Windows. Linux retains the
   original external PNG/WAV root through a sampled-D2 Responses chain to 131K
   input positions and retrieves both facts at the end. The replacement
   checksum-locked repository suite of three generated PNGs and three
   public-domain WAV excerpts now also passes root recognition and six separate
   retrieval turns beyond 131K after the complete 4K/8K/32K/64K/128K sampled-D2
   characterization. Sampled video frames remain follow-up work.
13. Execute the OpenAI-compatible agent-server sequence. **A1 complete:** the server-neutral request, content,
   event, response, and finish-reason types represent function definitions, tool choice, ordered tool calls and
   results, incremental text/reasoning/tool output, and multiple calls without binding the runtime to OpenAI JSON.
   **A2 complete:** bounded JSON schemas, assistant calls, and tool results render through the checkpoint-native
   Gemma tool DSL; an arbitrary-chunk incremental parser returns validated JSON arguments and repeated structured
   calls. **A3 complete:** resident chat loads repeated schema-backed function definitions, displays validated calls,
   collects external results, and continues the exact resident KV prefix through the final answer. **A4 complete:**
   one-shot requests preserve repeated image/audio order, locate every placeholder span, and isolate each image in a
   qualified prefill chunk. **A5 complete:** the checkpoint-bounded automatic policy reserves non-image context,
   shares the remainder across images, preserves aspect ratio, avoids needless upscaling, and reports actual versus
   budgeted soft tokens. **A6 complete:** resident `/image` and `/audio` queues,
   inspection/clear commands, remaining-context image budgets, and message-aligned
   media history preserve exact multimodal prefixes across later text and tool
   turns. **A7 complete:** `/v1/chat/completions` supports bounded OpenAI JSON,
   non-stream and SSE output, usage, reasoning/text/tool deltas, exact tool-result
   continuation, and ordered inline image/audio content above a serialized
   resident `ChatSession`. **A8 complete:** the pinned official OpenAI Python
   SDK executes a real streamed weather-tool loop, parses typed call deltas and
   usage, appends the tool result, and verifies the grounded final answer. Next
   split `ModelRuntime`, `SessionState`, and `ExecutionSlot` ownership. **A9
   complete:** the server loads immutable target/assistant weights exactly once;
   conversation history and sampling controls are session-owned, while KV,
   streams, workspaces, mapped rings, and CUDA graphs are isolated per execution
   slot. **A10 complete:** `/v1/responses` supports official request/output
   items, typed SSE, function tools/results, multimodal inputs, usage, and exact
   linear continuation through `previous_response_id`; the pinned official SDK
   passes a streamed two-response weather-tool gate. **A11 complete:** a bounded
   shared-runtime pool isolates concurrent execution slots, retains multiple
   linear response roots, evicts only inactive LRU sessions, supports SDK
   cancellation and disconnect-safe slot discard, and exports admission,
   generation, cancellation, token, and memory metrics. The complete A1-A11
   path to an agent-capable OpenAI-compatible server is now implemented.
14. After the Blackwell backend is correct and competitive, add architecture-specific backends for additional 16 GB
   CUDA GPUs without weakening benchmark or memory contracts.

The detailed gates and ordering in `AGENTS.md` are authoritative.

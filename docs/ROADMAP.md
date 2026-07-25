# Roadmap

Current stage: Milestone 3, first full-model greedy characterization.

Linux and Windows host/CUDA build scaffolding is now available. This makes loader development and validation on the
same Blackwell machine possible from either operating system; it does not move Windows production inference ahead
of the correctness and native-kernel gates below.

## Active gate

- The first arena-backed 48-layer decode characterization loads all text-only tensors once, uses fixed
  workspace/KV arenas, executes the tied output head and GPU argmax, and supports explicit checkpoint-FP8 and BF16
  K/V semantics. Physical byte-per-value E4M3FN storage is implemented. The exact-blue gate passes, but the sky
  gate currently diverges from FP8-vLLM and llama.cpp at generated token 3: gem16gb selects `7412`, both references
  select `563`. State v5 has narrowed the first material difference to Layer-0 attention context at prompt position
  one. Match cache-write, score/softmax, and value-reduction arithmetic before extending the gate to 512 tokens.
- The pure C++ chat CLI now uses native byte-fallback BPE from `tokenizer.json`, a version-bound implementation of
  the exact checkpoint `chat_template.jinja`, and its EOS/suppressed-token lists. Retain the model across turns and
  replace serialized prompt ingestion with native prefill after resolving the early distribution divergence.
- Keep chat processing independent of terminal I/O so a later OpenAI-compatible Chat Completions server can reuse
  it. Do not begin HTTP/server work before persistent engine sessions and the correctness gate are in place.
- [x] Replace token-at-a-time prompt ingestion with a separate native context-budgeted prefill plan (128-token
  default). Keep the serial
  implementation only as a test/probe reference; production CLIs expose the measured native winner.
- [x] Replace the initial contiguous physical FP8 cache, formerly capped at 1,024 total positions, with circular local
  storage and independently growing global storage. The checkpoint-scale FP8 numerical semantics and one-byte
  allocation are implemented; fused/optimized attention reads remain pending.
- Extend the now-committed trusted vLLM token/top-logprob fixture with full-vocabulary logits and selected hidden
  states. The 12-prompt FP8/BF16 teacher-forced suite is complete and places every vLLM Top-1 in gem16gb's Top-5;
  full reference vectors remain pending.
- Finish the quality and native-dispatch gates for the patched same-source closest-parity GGUF, then select and lock
  quality-acceptable GGUFs for llama.cpp tiers B and C. Unpatched upstream conversion remains blocked.
- Extend the implemented deterministic weight/scale/KV base arena with execution-derived activation, logits,
  sampling, graph, kernel, and prefill workspace requirements.
- Keep the implemented CPU NVFP4 oracle as the authority while assembling the first complete MLP layer. The exact
  E2M1, E4M3FN, source-nibble, global-divisor, activation-quantization, and FP32-accumulation contracts are now
  executable and tested.
- Assemble Gate/Up, GELU-tanh product, Down, and residual without weakening the now-complete real-checkpoint proof:
  all three Layer-0 shapes consume source weight and scale storage directly, CPU/GPU activation bytes match exactly,
  and CUDA reference/native output differences are at most `1.1920929e-7` in the characterization fixture.
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
5. ~~Implement and round-trip-test direct SM120 fragment views over source weight/scale storage for Gate, Up, and
   Down.~~ No persistent repack is required by the current direct route.
6. ~~Compare the implemented SM120a `m16n8k64` decode projection for `T=1` with a bandwidth-oriented
   packed-NVFP4 SIMT/GEMV candidate.~~ The direct MMA route is 3.1x–6.2x faster on the real Gate/Up/Down shapes and
   remains the production candidate. Disassembly proves `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`; retain the SIMT
   implementation only as a direct-source characterization control.
7. The correctness-first Layer-0 MLP chain now implements Gate/Up, Gemma GELU-tanh product, Down, post-MLP norm,
   residual, and layer scalar as part of the complete device-resident decoder-layer characterization. Both NVFP4
   activation boundaries remain byte-identical across its reference/native paths. Next compare it with trusted
   prompt-derived layer data. An opt-in closed Gate/Up/GELU kernel now removes four intermediate launches and
   unnecessary diagnostic stores; its isolated result is about 1.7% faster, but Windows end-to-end medians remain
   noisy and slightly negative, so it is not enabled by default.
8. Add a separate native prefill plan, initially qualified against pinned CUTLASS/cuBLASLt block-scaled GEMM. Do
   not reuse the decode plan merely for implementation convenience.
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
11. Validate 64K, then 128K context. MTP and multimodal work remain later milestones.
12. After the Blackwell backend is correct and competitive, add architecture-specific backends for additional 16 GB
   CUDA GPUs without weakening benchmark or memory contracts.

The detailed gates and ordering in `AGENTS.md` are authoritative.

# Correctness

## Implemented level

Level 0 currently covers bounded JSON parsing, duplicate-key rejection, little-endian Safetensors header lengths,
shape-product overflow, known dtype byte sizes, payload bounds, exact byte lengths, overlapping ranges, duplicate
tensors across shards, index agreement, UTF-8 strings, shard path traversal rejection, and symlink escape rejection.

`gem16-inspect --validate` additionally checks the expected primary architecture dimensions and quantization mode,
then requires each classified NVFP4 packed weight to have local, global, and input scale tensors.

An independent Python reader compares the raw Safetensors headers against the exported C++ manifest. For pinned
revision `b1f649734b34aa5575b03d186abd1b9be3d0d5c4`, all 1,389 tensors match across physical shape, dtype, absolute
offset, byte length, shard, and alignment; total tensor payload is 9,304,786,336 bytes with zero mismatches. The
validated decoder inventory contains 29 tensors in every sliding-attention layer and 27 in every full-attention
layer; full-attention layers omit separate `v_proj` weight and scale tensors.

Level 1 NVFP4 bring-up now includes a platform-independent E2M1 and E4M3FN codec, round-to-nearest-even host
encoding, dynamic-local activation quantization in groups of 16, compressed-tensors global-divisor application, and
a binary64 W4A4 projection oracle. Tests exhaustively round-trip all finite E4M3FN words and all E2M1 nibbles,
exercise rounding, saturation, and error behavior, and pin the first 16 packed values and first local scale from
layer 0 Gate row 0 of the locked checkpoint.

The CUDA correctness route independently uses CUDA 13.3 FP4/FP8 conversion types, matches the host packed
activation and scale bytes, and matches the host projection oracle. A separate experimental SM120a kernel consumes
the compact source weight and scale layouts directly. Its synthetic eight-row/64-K output and all three real
Layer-0 MLP projection shapes match the same oracle. The real Gate, Up, and Down maximum CUDA-reference/native
absolute differences were `1.1920929e-7`, `5.9604645e-8`, and `0`, respectively.

The first complete Layer-0 MLP characterization now executes input quantization, Gate and Up, Gemma GELU-tanh
product, Down-input quantization, Down, and residual addition without a host round trip. CPU/CUDA quantized input
bytes match exactly at both quantization boundaries. For its deterministic fixture the native and CUDA-reference
Down-input bytes and all 3,840 final float values also match exactly; eight final rows match the binary64 Down
oracle plus residual within `6.7374888e-9`. This remains a deterministic characterization, not a substitute for
the pinned hidden-state golden distribution.

Disassembly of the CUDA test binary contains `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`. This is native-instruction and
real-shape evidence, but the kernel remains experimental until layer-golden, numerical-distribution, memory-arena,
and end-to-end gates pass.

Level 1 FP8 bring-up now includes host E4M3FN/BF16 decoding, dynamic per-token activation quantization, a binary64
per-channel-scale projection oracle, an independent CUDA scalar reference, and a direct-source SM120 tensor-core
route. The real Layer-0 Q `[4096,3840]`, K/V `[2048,3840]`, and O `[3840,4096]` shapes all produce bit-identical
CPU/CUDA activation bytes and scale bits. Across those fixtures the largest CUDA-reference/native absolute
difference is `8.9406967e-7`; the largest selected-row binary64/native difference is below `1.0e-7`.
Disassembly additionally contains `QMMA.16832.F32.E4M3.E4M3`. As with NVFP4, this proves the intended arithmetic
instruction and real storage mapping.

The first unfused local-attention checkpoint characterization now executes the real Layer-0 BF16 norm tensors and
FP8 Q/K/V/O tensor families through input RMSNorm, per-head Q/K normalization, scale-free V normalization, local
RoPE at position 31, K/V append/read, grouped-query causal attention over a deterministic 32-token cache, FP32
softmax, O projection, post-attention RMSNorm, and the residual update. The direct SM120 and independent CUDA scalar
projection routes produce a final 3,840-element maximum absolute difference of `4.8398972e-5`, RMS difference
`4.2503101e-6`, and cosine similarity `0.9999999999984577`. This validates operator composition and real tensor
binding without a persistent repack; it is not yet a trusted-runtime hidden-state golden or a performance result.

The corresponding full-attention characterization now executes the real Layer-5 tensor family with 16 query
heads, one 512-dimensional KV head, and proportional RoPE. It reuses the raw K projection as the V input because
the checkpoint has no `v_proj`, then deliberately diverges the states: K receives learned RMSNorm and proportional
RoPE over the first 25% of rotary frequencies, while V receives scale-free RMSNorm and no RoPE. Over the same
deterministic 32-token fixture, direct SM120 projections and the independent CUDA scalar route produce final
maximum absolute error `4.5299530e-6`, RMS error `5.5268314e-7`, and cosine similarity
`0.9999999999999085`. This also proves that `attention_k_eq_v` permits projection reuse but not shared physical
storage of the final K/V cache.

The complete Layer-0 decoder characterization now keeps the local-attention residual on device and continues
through pre-feedforward RMSNorm, dynamic NVFP4 input quantization, Gate/Up, GELU-tanh product, Down-input
quantization, Down, post-feedforward RMSNorm, the second residual update, and `layer_scalar`. It binds all real
Layer-0 FP8, NVFP4, BF16 norm, and scalar tensors directly from the checkpoint. The CUDA scalar-projection and
direct SM120 routes produce zero differing bytes at both NVFP4 activation boundaries. Their final 3,840-element
outputs have maximum absolute difference `4.7683716e-6`, RMS difference `2.8454761e-7`, and cosine similarity
`0.9999999999999643`. The probe owns 148,639,086 device bytes because it deliberately retains two complete
execution paths for comparison; this is not a production workspace or peak-VRAM estimate.

`tools/validate_layer_checkpoint.py` runs the local-attention, full-attention, and complete-decoder probes and
exports one combined JSON record. It enforces structural correctness gates but intentionally applies no model-wide
numeric tolerance. Prompt-derived vLLM state-v5 captures now provide real-token Layer-0 and per-layer boundaries;
the deterministic synthetic-cache probes remain independent operator-composition oracles rather than substitutes
for those runtime distributions.

The full-model greedy characterization includes precision-matched generation gates. For
`exact_blue_no_thinking`, the checkpoint tokenizer and exact `chat_template.jinja` produce the committed 20 prompt
IDs, and the engine matches vLLM's complete `[9503, 106]` response (`blue<turn|>`). The longer
`sky_sentence_no_thinking` case emits `[818, 7217, 7412]` with gem16's physical FP8 cache, while the retained
FP8-vLLM and llama.cpp characterizations emit `[818, 7217, 563]`. Explicit BF16 vLLM and gem16 both emit
`[818, 7217, 7412]`. State-v5 comparisons localized the FP8 difference and found no missing model operation or
format violation. Because the runtimes use different reduction orders and llama.cpp also maps source FP8 attention
weights to BF16, exact greedy-token identity is not a universal gate. This case remains a sensitive regression and
distribution probe; acceptance depends on deterministic execution, operator contracts, teacher-forced rank/logit
metrics, and task quality. A previous revision matched the FP8 references only through compensating arithmetic
errors and is not valid evidence.

The production whole-model decode graph now reuses the byte-exact prefill RMSNorm/FP8, RMSNorm/NVFP4, and
Gate/Up/GELU/NVFP4 boundaries. Its controlled Q/K kernel additionally preserves projection BF16 rounding, the
256-thread RMSNorm reduction, exact precomputed local/proportional RoPE values, and final BF16 rounding in one
launch. CUDA fixtures compare both D256 local and D512 partial-rotary global outputs byte-for-byte against the
unfused sequence at a nonzero dynamic position. The targeted fusion suite passes compute-sanitizer memcheck with
zero errors and racecheck with zero hazards. Qualification fixed the prior `RmsNormQuantizeTokenKernel` hazard by
fencing the first shared reduction result before the same array is reused for the maximum reduction. Exact-blue remains `[9503, 106]`; 129/257 prefill boundaries pass; the
12-prompt suite remains 121/127 Top-1 and 127/127 Top-5/Top-20; and each 128/2,048/8,192 10-run decode set retains
one checksum.

The explicit GPU sampling plan has an end-to-end reference gate in `tools/validate_sampling.py`. It dumps the
unmodified softcapped logits for a real checkpoint prompt, independently applies full-history sign-aware repetition
penalty, temperature, top-k, min-p, top-p, and the pinned SplitMix64 seed/step mapping on the CPU, and requires the
same selected token. It also launches a fresh second process and requires seeded output identity. The initial
combined fixture (`temperature=0.8`, `top-k=64`, `top-p=0.95`, `min-p=0.02`, repetition penalty `1.1`, seed `42`)
selects `[532, 497, 236786, 6981]` from `[3, 3, 43, 1]` final eligible tokens across four steps in both
implementations, including generated-token repetition-history updates. Separate end-to-end checks prove
`top-k=1` reproduces unchanged greedy IDs and a different seed changes a multi-step sampled sequence. Nsight CUDA
API tracing places all four `cudaMalloc` calls before the end of `gem16.initialize` and none inside sampled prefill
or decode. A model-independent CUDA operator test pins repetition and suppression processing, top-k-1 selection,
seed/step results, device-control step override, and successful radix-sort graph capture/replay on an eight-token
fixture. Repetition history is an atomic 32-bit bitset, so duplicate prompt IDs are race-free. The complete CUDA
test binary passes compute-sanitizer memcheck with zero errors, and the sampling-only suite passes racecheck with
zero hazards. The formerly reported `RmsNormQuantizeTokenKernel` hazard was subsequently fixed and is covered by
the decode-fusion racecheck gate. End-to-end sampled
decode uses one whole-model graph replay per post-prefill token. Synthetic and end-to-end tests cover both bounded
top-k and unfiltered full-vocabulary selection; the probability scan and final binary searches are GPU-resident.

`--dump-logits` captures every selected position as full-vocabulary raw little-endian float32 after preallocating
host storage, and `tools/compare_logits.py` compares it with the committed vLLM top-20 distributions. When no
layer-state dump is requested, the first captured position now comes directly from the batch-prefill output head;
older revisions silently replaced the whole prompt with token-at-a-time diagnostic forwards whenever logits were
requested. Later captured positions remain ordinary teacher-forced decode. A layer-state dump still uses the
serial diagnostic route because batch layer-state capture is not implemented. The earlier
BF16-engine versus auto-FP8-vLLM comparison placed token `563` at engine rank 2 and token `7412` at engine rank 1;
this was a real distribution difference, but it was caused by comparing different K/V modes rather than an argmax
tie or sampling randomness.

### Greedy determinism gate

`tools/check_greedy_determinism.py` launches a fresh `gem16-run` process for every repetition, requires an exact
fixed prompt and generation length, hashes token IDs as little-endian `uint32`, and reports the first token index
that differs from run zero. It can derive a short prompt from the pinned Wikipedia workload while preserving the
chat-template suffix. The normal regression gate uses 512 prompt tokens, 256 generated tokens, checkpoint FP8 KV,
and five fresh processes.

The 16K Wikipedia characterization exposed a real race in the long-context online decode-attention reductions.
Both `DecodeBlockMaximum` and `DecodeBlockSum` synchronized before every thread loaded the shared block result, but
then allowed faster threads to reuse the same shared array while slower threads were still consuming that result.
CUDA Racecheck reported hazards in both Split and Merge and produced visibly corrupted operator outputs under
instrumentation. Each helper now copies the shared result to a register and executes a second block barrier before
the array can be reused.

After the fix, targeted Racecheck reports zero hazards. The local, global, and 16,384-position global operator
fixtures are bit-identical across four repeated launches; the long global result remains within maximum absolute
error `1.04308e-7`, RMS `2.19933e-8`, and cosine 1.0 of the score-matrix reference. Five fresh-process 16K+64
production-graph runs all hash to
`0b373ccd43b43fc40e2029fae386a1573ff0f891613ec444dd29af4954e43d52`; five 512+256 runs all hash to
`8cc1cc480f6c1799467fb105efafe387c57f86ed2e1231af059f6355d376fce2`.

The complete 16K Wikipedia workload was then repeated with three warm-ups and ten measured runs. Every run
generates exactly 1,106 tokens, stops normally, and hashes to
`44399253201aa729d72cf7a027b42ce2de9b1aa4ca98c1e9156435bb7e6628a8`. Decoding the representative output with
the pinned tokenizer produces a coherent German summary of the article's definitions, history, applications,
opportunities, and risks. This closes the workload-specific greedy-determinism issue; it does not replace the
separate cross-runtime logit and quality gates.

### Direct prefill-boundary reference

`tools/generate_prefill_boundary_golden.py` runs vLLM 0.25.1 offline on two deterministic direct-token prompts of
129 and 257 tokens and records complete greedy IDs plus Top-20 log probabilities. The fixture is pinned to the
checkpoint revision and records the reference runtime, GPU, cache mode, and execution controls. It replaces an
older boundary check whose expected eight-token sequences came from gem16 itself and therefore could not serve
as external correctness evidence.

`tools/validate_prefill_boundaries.py` captures only the first generated position through the real batch-prefill
plan and compares its full logits with that fixture. At checkpoint `c0f42de` plus the diagnostic capture fix, both
the 129- and 257-token cases place the vLLM Top-1 at engine rank 1 with selected-token logprob absolute deltas
0.22294 and 0.50785. Both runs report zero fallbacks and no token-loop allocations for the sole 1,024-token plan.
Later fixture positions are deliberately excluded from this boundary gate because they exercise decode.

`--dump-state <file> --dump-state-position <position>` captures, for every decoder layer, attention context/output,
both normalized residual branches, Gate, Up, GELU product, MLP output, final hidden state, and newly appended K/V
inputs. Pinned host storage is allocated before inference and the self-describing version-5 file records projection
and K/V-cache modes, then is written only after the token loop. `tools/dump_vllm_states.py` disables vLLM frontend
multiprocessing for diagnostic hooks and emits the same format; `tools/compare_states.py` reports per-layer RMS,
maximum, cosine, and optional intra-layer metrics.

The state comparison exposed and fixed two concrete operator errors. vLLM rounds the tanh-GELU result to BF16
before multiplying it by the BF16 Up projection; gem16 previously rounded only the product. vLLM's NVFP4
activation quantizer also uses `rcp.approx.ftz.f32` in both scale construction and normalization, rather than exact
division. The production CUDA quantizer now follows that arithmetic and a real vLLM boundary fixture pins its
packed E2M1 bytes and E4M3 scale.

After those fixes, prompt position zero is bit-identical to vLLM through Layers 0–29. The first remaining difference
is a small Layer-30 attention output difference (attention context is still exact); the discrepancy disappears
again after Layer 31 and the final captured Layer-47 hidden state is bit-identical. This is strong evidence for the
projection, norm, RoPE, residual, and MLP contracts at a single-token attention position.

The first cache reuse at prompt position one is the earliest material cross-runtime difference. In Layer 0,
gem16 versus FP8-vLLM attention context has RMS `3.846e-3`, maximum `6.25e-2`, and cosine `0.9999921`; the current
V input is bit-identical and K differs only by RMS `1.249e-4`. By generated position 24 the Layer-0
attention-context difference reaches RMS `6.640e-3`, maximum `1.875e-1`, and then propagates through the model.
The physical vLLM cache was verified as `torch.uint8` E4M3 storage with layout `[blocks, 2, 16, 8, 256]`. This
localization excludes tokenizer, sampling, and the corrected NVFP4 MLP contract. The residual difference is
consistent with implementation-specific FP8 cache-write and attention reduction arithmetic; it is monitored by
rank/logit and quality gates rather than requiring bit identity.

### Broader teacher-forced comparison

The version-1 correctness suite expands the original three prompts to 12 deterministic chats covering exact
answers, longer prose, Thinking and non-Thinking modes, German/Unicode, JSON, summarization, C++ semantics, and
multi-turn recall. It contains 127 generated positions with the explicit vLLM FP8 cache and 131 with explicit vLLM
BF16. Both fixtures retain the exact templated prompt IDs and top-20 log probabilities.

`gem16-run --teacher-forced-token-ids` feeds the preceding reference token at each later decode step but still
records the engine's unmodified greedy prediction and full-vocabulary logits. Stop tokens do not terminate this
diagnostic mode. This separates per-position numerical agreement from autoregressive drift after the first
different argmax.

The 2026-07-24 serial-diagnostic characterization produced:

| Engine/reference cache pair | Top-1 | Fully agreeing prompts | Reference Top-1 in Top-5 | Mean Top-20 overlap | Mean selected-logprob absolute delta |
|---|---:|---:|---:|---:|---:|
| gem16 FP8 / vLLM FP8 | 118/127 (92.9%) | 8/12 | 127/127 | 14.606/20 | 0.1064 |
| llama.cpp F16 / vLLM FP8 | 119/127 (93.7%) | 9/12 | 127/127 | 15.646/20 | 0.1195 |
| gem16 BF16 / vLLM BF16 | 127/131 (96.9%) | 10/12 | 131/131 | 14.878/20 | 0.0580 |
| llama.cpp F16 / vLLM BF16 | 129/131 (98.5%) | 10/12 | 131/131 | 15.870/20 | 0.0419 |

These are strict token-ID counts. One BF16 mismatch shared by gem16 and llama.cpp is an exact vLLM logprob tie
whose deterministic tie choice differs; it remains counted as a mismatch.

After logit capture was corrected to retain the production batch-prefill path, the FP8 comparison remains exactly
118/127 Top-1 and 8/12 fully agreeing prompts. The vLLM Top-1 is now in the engine Top-5 at 126/127 positions,
mean Top-20 overlap is 14.803/20, and mean selected-logprob absolute delta is 0.1629. The one rank-7 position is a
later forced step of `sky_sentence_no_thinking`; it reflects the production KV state produced by batch prefill,
not the online-attention boundary itself. These current values supersede the serial-diagnostic FP8 row for
production-path qualification while retaining the older row as provenance.

The llama.cpp candidate still differs in attention-weight storage: its conversion maps the source FP8 attention
weights to BF16, so neither F16-cache row is exact format parity. Nevertheless, the results establish that gem16
is close to both independent engines across a materially broader sample. BF16 K/V improves gem16's Top-1
agreement and selected-token logprobs, proving that FP8 cache arithmetic explains part, but not all, of the
remaining drift. No tolerance is accepted from these measurements alone.

The native C++ tokenizer/template path reproduces all three committed reference prompt-ID sequences exactly:
20 tokens for exact-blue, 23 for the sky sentence, and 27 for the thinking arithmetic prompt. The application reads
the actual template file and accepts only the pinned supported revision. Its renderer currently supports
system/developer, user, and assistant text roles; tool calls and multimodal content fail visibly until their native
template branches are implemented. A separate German/Unicode probe containing umlauts, `ß`, and an emoji also
matches the Transformers tokenizer exactly across all 27 prompt IDs.

The tokenizer metadata is independently pinned to Google's instruction-model revision
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`. Startup rejects the older Unsloth metadata where `eos_token` aliases
`<turn|>`, validates every Google `response_template` close marker against the generation stop-token IDs, and
uses the declared thinking/content delimiters when producing visible assistant text. Google's very large
tokenizer `model_max_length` sentinel is parsed but deliberately excluded from the model context and memory plan.

The patched same-source llama.cpp candidate supplies an independent comparison despite mapping FP8 attention
weights to BF16. It matches 50/65 reference output tokens overall: exact-blue is 2/2, the sky answer matches vLLM's
first 18 tokens before diverging, and the thinking trace matches 28/32. At gem16's first sky divergence, llama.cpp
and FP8-vLLM select token `563`, while gem16 selects `7412`. This remains useful sensitivity evidence, but no one
runtime is treated as bit-exact ground truth; distribution and quality analysis determine acceptance.

Reproduce the instruction check with:

```bash
python tools/verify_sm120_sass.py build/<OS>/blackwell-release/bin/gem16-cuda-tests
```

## Not yet established

Accepted model-wide layer tolerances, broad task quality, and a statistically justified generation threshold have
not been measured. `tests/tolerances.yaml` now contains shape-specific local/global online-attention operator
tolerances derived from CUDA-versus-FP32-oracle distributions; these are not global model tolerances. Prompt-derived
hidden/KV, direct prefill full-logit, and 12-prompt teacher-forced Top-20 comparisons are implemented, but the
committed vLLM Top-20 fixtures are not substitutes for full-reference-logit Level 3 metrics at every position.

## Direct reference runtime

The checkpoint model card's reference recipe is used in a separate, ignored Python 3.13 environment. The first
validated environment contains vLLM 0.25.1, PyTorch 2.11.0+cu130, Transformers 5.14.1, compressed-tensors 0.17.0,
FlashInfer 0.6.13, and NVIDIA CUTLASS DSL 4.5.2. `tools/generate_golden.py` runs the locked local checkpoint with
network access disabled, batch one, an 8K context limit, eager execution, no prefix cache, no CPU offload, and all
multimodal limits set to zero. Model-supported chunked prefill remains enabled. It records exact templated prompt
IDs, greedy output IDs, and the top 20 log
probabilities at every generated position.

Reference-runtime startup logs are part of the evidence: vLLM must select `CutlassFP8ScaledMMLinearKernel` for the
attention projections and `FlashInferCutlassNvFp4LinearKernel` for the NVFP4 MLPs. Package selection alone does not
replace later per-kernel profiling.

The checkpoint declares a static tensor-wise FP8 K/V scheme and stores BF16 `k_scale`/`v_scale` values for every
layer. With `kv_cache_dtype=auto`, vLLM resolves this declaration to FP8. Reference commands that intend to compare
the engine's `--kv-cache bf16` mode must therefore pass `kv_cache_dtype=bfloat16` explicitly; otherwise the result
is not a precision-parity comparison.

Two consecutive runs on 2026-07-21 produced exactly identical prompt IDs, output IDs, and log probabilities. The
first engine initialization took 119.44 seconds while compiling and autotuning; the warm-cache initialization took
4.69 seconds. FlashInfer reported OOM for some autotuning tactics and stored default fallbacks for those shapes.
This does not invalidate the correctness fixture, but it disqualifies these runs as performance evidence and must
be revisited when configuring any vLLM speed baseline.

Run with the reference environment activated so its `ninja` executable is visible:

```bash
PATH="$PWD/third_party/cache/unsloth-nvfp4-env/bin:$PATH" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 \
  third_party/cache/unsloth-nvfp4-env/bin/python tools/generate_golden.py \
  --model models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497 \
  --output tests/golden/vllm-gemma4-12b-nvfp4.json
```

Add `--prompts benchmarks/prompts/correctness-v1.json`, `--max-tokens 24`, and either
`--kv-cache-dtype fp8` or `--kv-cache-dtype bfloat16` to reproduce the broad fixtures. Compare llama.cpp at the
same forced token positions with `tools/teacher_forced_llama.py`; unlike the native runtime, these Python programs
are offline correctness tooling only.

Reproduce the physical manifest comparison with:

```bash
build/host-debug/bin/gem16-inspect --model <checkpoint> --validate --json build/manifest.json
python3 tools/compare_manifests.py --model <checkpoint> --manifest build/manifest.json
```

# Bounded cross-engine and quantization comparison

Owner request, 2026-09-05: approximately **30 minutes of measurement**, not a
complete quality campaign. This is a diagnostic pilot; the promoted comparison
requirements in `docs/BENCHMARKING.md` remain unchanged. Installation, compilation
and obtaining missing immutable model artifacts must be reported separately and
must not silently turn this into a long-running campaign.

## Speed pilot

- Primary model: Gemma 4 26B A4B, the same upstream revision and tokenizer.
- Three exact prompt lengths: **512, 8,192 and 32,768 tokens**, followed by 128
  actual output tokens. Use fixed text/code fixtures with recorded token IDs and
  hashes. Measure decode at each prompt depth, not a context-free `tg128` run.
- One warm-up and three retained samples per engine and length, all raw samples
  retained. Label this explicitly as a pilot, not the standard 3+10 qualification.
- Separate TTFT/prefill tok/s and decode tok/s. Count only actual output tokens;
  disclose first-token timing boundaries. Batch one, cold prompt cache each run,
  identical greedy controls, no CPU weight offload. Record actual output lengths,
  KV precision, VRAM peak, versions, model hashes and dispatch logs.
- Ordinary decoding first. Fixed-D2 comparisons are a separate extension only
  if all three engines support equivalent target verification and time remains.
- Stop at the time budget; retain failed/OOM/unsupported cases without changing
  the prompt or substituting a quant invisibly. 128K/170K capacity is a separate
  extension, not required for this first pilot.

Current upstream discovery (2026-09-05): llama.cpp rolling build
[b10819](https://github.com/ggml-org/llama.cpp/releases/tag/b10819), and vLLM
[v0.28.0](https://github.com/vllm-project/vllm/releases/tag/v0.28.0).
Resolve exact commits again when preparing the run; do not label the historical
patched b10240/v0.26.0 installations as current upstream versions.

Run the primary three-engine comparison on the **same Linux installation**.
[Upstream vLLM does not support native Windows](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/).
A Windows gem16/llama.cpp pair can be additional evidence; mixing native Windows
results with Linux/WSL vLLM would confound engine and OS differences.

## Comparison formats to qualify before timing

| Engine | Candidate | Interpretation |
|---|---|---|
| gem16 | Current pinned Trellis35 Target | Product under test; record actual mixed-format inventory and bytes |
| llama.cpp | Same-source GGUF Q3_K_M and Q4_K_M candidates | Bracket the storage/quality tradeoff; select one for the timed pilot after checking fit and support |
| vLLM | Same-source supported NVFP4 candidate | Native-format baseline; validate Gemma 4 SM120 dispatch and full GPU residency |

These are **candidates, not verified available artifacts or equivalent quants**.
3.5 bpw is not directly equivalent to a format name: scales, non-expert tensors,
embeddings and other mixed formats affect total footprint and quality. No new
quantization, model substitution or download has been performed for this plan.

## Short quality probe

Use a frozen, non-calibration mini-corpus with text and code. Start with 256
teacher-forced scored tokens across several fixed excerpts; extend only within
budget. Each quant sees the **same prefix and next-token labels at every position**.
Report sample count and corpus hashes next to every number.

- PPL = exp(mean negative log probability of the actual next token).
- Compare delta NLL (and PPL ratio) against the same reference.
- Full KL = mean over positions of `sum_v p_ref(v) * log(p_ref(v)/p_quant(v))`.
  Fix the direction to reference-to-quant and report nats/token.
- Prefer one immutable BF16 reference; if that cannot be obtained within the
  hardware/time budget, a specified higher-precision quant can be a *quantized
  reference*, with no claim of BF16 fidelity. Quality-only CPU execution can be
  disclosed separately; it cannot support GPU speed claims.

Existing `tools/teacher_forced_compare.py`, `tools/teacher_forced_llama.py` and
`tools/compare_logits.py` provide useful alignment/logit infrastructure. The
committed comparison fixtures used by `compare_logits.py` contain **top-20 log
probabilities**, not full reference distributions. They cannot establish full
vocabulary KL. Full aligned reference logits/log probabilities must be retained
or scored directly; do not rename top-20 overlap or renormalized top-k KL as KL
of the complete model distribution.

This probe can support a clearly bounded quantization-fidelity result. It does
not establish general reasoning, coding or multimodal quality. No result has yet
been measured or published under this plan.

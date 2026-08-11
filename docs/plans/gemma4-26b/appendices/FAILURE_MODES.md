# Failure modes and required responses

## Source/model

### Mutable or mismatched revision

**Symptom:** tensor names/config differ or hashes change.
**Response:** stop; create/update immutable lock; rerun dependent milestones.
**Never:** patch expected shapes until it loads.

### Tensor axis misinterpretation

**Symptom:** plausible but wrong expert output.
**Response:** use M03 slice fixtures and trusted captures.
**Never:** infer from byte length alone.

### Tokenizer/template mismatch

**Symptom:** quality/generation differs before model math.
**Response:** pin exact assets and token IDs.
**Never:** compare rendered strings without tokens.

## Quantization

### Global scale reciprocal error

**Symptom:** output magnitude grossly wrong or saturation.
**Response:** verify dequant equation against source/reference.
**Never:** compensate with an extra runtime scale.

### Wrong nibble order

**Symptom:** reconstruction random within pairs/halves.
**Response:** exact byte fixtures.
**Never:** tune kernels around mistaken bytes.

### QAT does not transfer to NVFP4

**Symptom:** C worse than B/D in held-out quality.
**Response:** reject QAT-derived NVFP4 or choose another profile.
**Never:** claim QAT benefit from theory.

### Quantized head causes token flips

**Symptom:** body drift low, logits/argmax diverge.
**Response:** retain Q4_0 head or adjust accepted format after M19.
**Never:** reduce vocabulary or skip softcap.

## MoE

### Router top-8 instability

**Symptom:** eighth expert changes frequently.
**Response:** preserve BF16/FP32 router, exact softmax/tie, inspect margins.
**Never:** approximate top-k silently.

### Expert reduction nondeterminism

**Symptom:** output hash varies.
**Response:** stable segmented/gather reduction.
**Never:** accept unordered atomics for deterministic profile.

### Shared MLP omitted

**Symptom:** severe layer/model drift despite plausible routing.
**Response:** implement always-active branch in exact order.
**Never:** treat it as an expert selected by router.

### Prefill workspace explosion

**Symptom:** OOM at long prompt.
**Response:** smaller chunk, compact assignments, tiled reduction.
**Never:** offload weights or silently reduce context.

## Attention/KV

### Final K/V alias

**Symptom:** attention quality failure especially global.
**Response:** separate states after normalization/RoPE.
**Never:** interpret `K==V` as physical cache sharing.

### Wrong global KV heads

**Symptom:** shape/index errors or degraded global attention.
**Response:** traits per layer and real fixture.
**Never:** reuse 12B one-head assumption.

### Ring chronology error

**Symptom:** quality drops after 1,024 tokens.
**Response:** wrap tests and chronological index mapping.
**Never:** test only short context.

### KV sharing misunderstanding

**Symptom:** missing weights/caches or duplicate memory.
**Response:** validate producer/consumer semantics.
**Never:** derive ownership from tensor absence alone.

## CUDA/performance

### Native object but fallback dispatch

**Symptom:** SASS exists but trace shows reference kernel.
**Response:** runtime trace and hard `--require-native-path`.
**Never:** claim native path from binary inspection alone.

### Graph-time hidden allocation

**Symptom:** first replay allocates or OOMs.
**Response:** CUDA API trace, prewarm/capture all workspaces.
**Never:** ignore library allocations.

### T=1 Tensor Core underutilization

**Symptom:** Q4_0 matches/beats NVFP4 decode.
**Response:** profile bandwidth/launch/layout; retain evidence.
**Never:** assume theoretical FLOPs guarantee a win.

### Thermal drift

**Symptom:** small A/B reverses across runs.
**Response:** adjacent serial runs, start-temperature criterion, telemetry.
**Never:** promote a one-run win.

## Product

### Desktop VRAM unavailable

**Symptom:** model fits on idle GPU but not normal desktop.
**Response:** admission error and user guidance; consider reduced context.
**Never:** crash or silently offload.

### Media inherited from architecture

**Symptom:** API accepts image but artifact lacks tensors.
**Response:** capability profile rejects before decode.
**Never:** ignore media.

### Wrong assistant

**Symptom:** 12B MTP assistant loaded with 26B.
**Response:** model variant compatibility validation.
**Never:** run due to matching tokenizer alone.

## Process

### Milestones collapsed

**Symptom:** compiler, kernels and benchmarks in one opaque milestone change set.
**Response:** split, restore reference/evidence boundaries.
**Never:** accept because demo works.

### Held-out leakage

**Symptom:** quantizer/head selected using test scores repeatedly.
**Response:** freeze new split and document contamination.
**Never:** keep calling it held out.

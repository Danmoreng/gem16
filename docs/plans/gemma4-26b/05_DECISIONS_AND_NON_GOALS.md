# Decisions, open choices and non-goals

## Decisions already made for this plan

### D1 — Native NVFP4 remains the expert performance path

The routed expert matrices dominate resident bytes and per-token weight traffic. They should use Blackwell's native block-scaled FP4 path rather than a generic Q4_0 dequantization path.

### D2 — Google QAT BF16 is the primary mathematical source

All tensors in the primary hybrid derive from one exact unquantized QAT revision. Do not splice finished Unsloth expert tensors into finished Google Q4_0 attention or router tensors for the production model.

### D3 — Unsloth NVFP4 is a mandatory baseline

The published Unsloth checkpoint is useful both as a runtime baseline and as quantizer evidence. It is not assumed to be byte-identical to the project compiler.

### D4 — An ordinary-BF16 control is mandatory

The same project compiler must quantize the ordinary Google IT BF16 checkpoint. Comparing this result with the QAT-derived result isolates the effect of the source weights from the effect of the quantizer.

### D5 — Vision is excluded before artifact and arena planning

The compiled production artifact contains no vision payload. The runtime does not download, upload or reserve it for the first profile.

### D6 — MTP is deferred

The base target must be correct, deterministic, resident and qualified before adding a QAT-compatible assistant.

### D7 — 12B remains statically specialized

Model-general metadata is allowed. Hot 12B and 26B kernels remain independently compiled and selected once.

### D8 — Runtime conversion is forbidden

The inference process loads a compiled artifact. It does not quantize BF16 or construct a persistent second on-disk GPU layout.

### D9 — Safetensors remains the weight container

The derived format uses Safetensors plus explicit versioned metadata. Do not invent an opaque monolithic binary for the first implementation.

### D10 — One embedding/head representation is resident

Q4_0 and NVFP4 may both exist as separately compiled profiles. A running process holds only one.

## Repository compilation policy

The repository permits both immutable upstream checkpoints and reproducible project-compiled artifacts. The 26B
QAT track requires a custom compiled artifact because no ready-made checkpoint implements the proposed production
recipe. This is normal model-profile work rather than an exception to a direct-load-only policy.

M00 must record the concrete 26B artifact contract with all of these constraints:

- source, compiler and output are fully locked;
- compilation is offline, reproducible and independently verifiable;
- inference never compiles or silently requantizes;
- the derived artifact remains auditable Safetensors;
- direct Unsloth NVFP4 and official Q4_0 remain external baselines;
- every performance claim identifies the artifact as project-built;
- no silent fallback or CPU offload is introduced.

Do not implement the compiler until this track-specific contract is recorded and reviewed.

## Open choices that must be resolved by evidence

### O1 — Q4_0 or NVFP4 tied head

Resolve in M07/M16 using:

- file/device bytes;
- input lookup latency;
- one-token output-head latency;
- prefill head behavior;
- teacher-forced KL/NLL;
- greedy token agreement;
- end-to-end ITL.

### O2 — Expert disk layout

Candidates:

- compressed-tensors-compatible per-projection tensors;
- fused gate/up W13 with separate W2;
- individual expert tensors versus packed 3D expert tensors.

Choose the layout that is auditable, streams to one final device representation and does not inflate load-time peak memory.

### O3 — Decode expert kernel organization

Candidates:

- eight expert GEMVs inside one persistent kernel;
- grouped native NVFP4 GEMM with compact descriptors;
- fused W13 followed by fused W2/reduction;
- separate shared and routed paths versus a jointly scheduled persistent kernel.

Do not choose by theoretical FLOPS alone.

### O4 — Prefill chunk size and routing sort

Benchmark 256/512/1024 token chunks and stable versus non-stable grouping only after exact token/expert mapping is proven. Production output must be deterministic.

### O5 — Cross-platform compiler determinism

The reference requirement is exact output on the locked compiler environment. Cross-platform identical hashes are a target. If platform floating behavior differs, publish one canonical compiler container rather than weakening hashes.

## Non-goals

- training or further QAT for NVFP4;
- changing Google's chat template;
- compressing router weights in the first release;
- compressing norms;
- claiming universal Gemma 4 support;
- using expert offload to make an oversized layout appear viable;
- implementing full Q4_0 before the native path works;
- optimizing vision;
- porting to AMD or Apple;
- hiding memory failure with a smaller undeclared context;
- replacing the existing server architecture with a generic serving framework.

## imp-specific non-goals

The first 26B release will not become an imp-compatible general engine. Paged KV, continuous batching, broad GGUF/model dispatch, host expert offload and multiple permanent decode caches remain out of scope. Optional donor-code use is limited to independently qualified kernels or small support patterns.

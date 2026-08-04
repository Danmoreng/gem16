# Target product profile

## First release name

Use a clear internal profile name until final branding is decided:

```text
gem16-gemma4-26b-a4b-qat-hybrid-text
```

Do not call it "official Google NVFP4" or imply Google trained it for NVFP4. It is a project-built derivative of Google's QAT BF16 checkpoint.

## Supported hardware

Required first backend:

- one NVIDIA Blackwell GPU;
- compute capability 12.0 / SM120 or the exact architecture target required by the pinned compiler for native NVFP4;
- approximately 16 GB VRAM;
- no CPU weight offload;
- Windows x64 and Linux x86-64 only after each platform passes separate evidence gates.

The first correctness bring-up may be Linux-only. A release matching existing gem16 product scope should support both platforms.

## Supported model behavior

Required:

- text input and text output;
- instruction-tuned chat template;
- system/user/assistant roles;
- thinking enabled and disabled;
- batch one;
- resident multi-turn continuation;
- greedy decoding;
- temperature, top-k, top-p, min-p and repetition handling already supported by gem16;
- 32K production context;
- 64K target context;
- 256K remains experimental and is not a first-release requirement;
- exact stop/suppression behavior;
- function-call text grammar only insofar as the existing tokenizer/template path already supports it.

## Explicitly unsupported in first release

- images, video and vision tower;
- audio;
- MTP/speculative decoding;
- continuous batching;
- multiple 26B slots on a 16 GB card;
- CPU expert offload;
- expert streaming from host;
- multi-GPU;
- arbitrary Gemma 4 MoE variants;
- on-the-fly source BF16 quantization in the inference process;
- loading a model with unverified source or compiled hashes.

Unsupported requests must fail before model execution with a precise capability error.

## Precision profile

### Fixed for first release

- routed experts: NVFP4 values with E4M3 block scales, group 16;
- shared dense MLP: same NVFP4 contract;
- attention: per-output-channel FP8 weights and dynamic per-token FP8 inputs;
- router and all router scales: BF16/F32 according to source;
- layer norms and layer scalar: source precision;
- KV: checkpoint-compatible FP8 with separate K and V;
- accumulators: FP32 unless the existing validated path dictates an exact BF16 closure.

### Decided by experiment

Tied embedding/output head:

- `q4_0`: quality-oriented candidate;
- `nvfp4`: speed-oriented candidate;
- `bf16`: diagnostic only and normally not resident on 16 GB.

The release profile uses exactly one.

## Checkpoint workflow

Normal end-user flow:

1. download a project-published, immutable, checksum-locked compiled checkpoint;
2. verify source and compiler provenance in `gem16_compilation.json`;
3. load it directly with gem16;
4. never compile during server startup.

Developer/reproducer flow:

1. download exact QAT BF16 source lock;
2. run the pinned compiler in a bounded-memory environment;
3. verify every output hash against the compiled lock;
4. run differential quality and operator tests;
5. optionally publish the derived artifact under the applicable license.

## Performance objectives

Release-blocking:

- fully resident weights;
- no token-loop allocation;
- 32K context with at least 700 MiB measured free-device margin;
- faster median prefill and decode than the accepted official-Q4_0 baseline on the same machine;
- deterministic output in deterministic mode.

Targets, not release blockers:

- 64K with at least 500 MiB margin;
- decode at least competitive with direct Unsloth NVFP4 under a fair boundary;
- prefill within 10% of the strongest practical direct NVFP4 runtime;
- QAT-derived quality at least non-inferior to the Unsloth path and preferably closer to official Q4_0.

## Quality objective

The candidate must be evaluated against:

- QAT BF16 teacher-forced reference;
- official Google Q4_0;
- Unsloth NVFP4;
- ordinary BF16 compiled by the same gem16 quantizer.

The production claim should be no stronger than the evidence. "Uses QAT source weights" is provenance, not a quality conclusion.

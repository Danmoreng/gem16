# Test matrix

## Test levels

| Level | Scope | Runs without model? |
|---|---|---|
| L0 | parsing, overflow, codecs, schemas | yes |
| L1 | operator references and synthetic CUDA | mostly |
| L2 | real tensor/operator checkpoint probes | model required |
| L3 | full layer and teacher forcing | model required |
| L4 | generation/chat/server | model required |
| L5 | memory, sanitizer, long context | model + GPU |
| L6 | quality/performance/release | full environment |

## Compiler tests

- source lock validation;
- path traversal/symlink rejection;
- Safetensors index/header validation;
- plan tensor coverage;
- FP8 codec/quantizer;
- NVFP4 codec/quantizer;
- Q4_0 codec/quantizer;
- deterministic output;
- resume and interruption;
- bounded RSS;
- output lock;
- corrupt output rejection.

## Model-contract tests

### 12B

Retain all current exact dimensions, tensor inventory and feature tests.

### 26B

- architecture/model type;
- 30 layers;
- hidden 2816;
- 128/top-8/704;
- shared 2112;
- layer pattern;
- local/global heads/dimensions;
- tied head;
- modality omission;
- quantizer schema;
- per-layer KV ownership.

Mutate every field individually and expect clear failure.

## MoE tests

- router synthetic/tie/overflow;
- expert axis;
- shared branch;
- per-expert output;
- weighted reduction;
- deterministic order;
- CPU versus CUDA reference;
- reference versus native decode;
- reference versus grouped prefill;
- real layers.

## Attention tests

- FP8 projection;
- local/global;
- missing V;
- K/V distinct;
- RoPE positions;
- cache ring wrap;
- global extent;
- shared KV if present;
- cache bytes.

## Head tests

- lookup;
- Q4/NVFP4 decode;
- tied pointer;
- softcap;
- suppression;
- tie;
- full logits;
- sampling;
- T=1/3/5;
- graph replay.

## Full-model tests

- tokenizer/template;
- exact prompt IDs;
- teacher forcing;
- layer captures;
- deterministic greedy;
- fixed-seed sampling;
- resident suffix;
- cache reset;
- cancellation;
- no allocation;
- no fallback.

## Product tests

- model download/verify/resume;
- CLI;
- server `/health` and `/v1/models`;
- Chat Completions;
- Responses;
- streaming;
- tools;
- sampling;
- resident sessions;
- cancellation;
- unsupported media/MTP;
- Studio selection/package.

## Sanitizers

Required targeted suites:

```text
compute-sanitizer memcheck
compute-sanitizer racecheck
compute-sanitizer initcheck
```

Run debug/line-info binaries. Retain logs.

## Determinism

For deterministic paths:

- at least 10 repeated runs in one process;
- fresh-process repetition;
- output token checksum;
- selected expert checksum;
- optional intermediate checksum.

Any nondeterminism blocks performance qualification.

## Allocation tests

Nsight CUDA API or equivalent trace proves:

- all `cudaMalloc` before prompt/decode;
- no hidden library allocation in graph replay;
- no host container growth in token callback path.

## Regression policy

Every 26B milestone change set runs:

- host unit tests;
- 12B relevant tests;
- new 26B unit tests.

A CUDA/operator milestone runs targeted CUDA tests. An integration milestone and the final main-integration review
run the full current host/CUDA matrix.

Do not disable or loosen 12B tests to merge 26B.

## Evidence naming

Each milestone stores:

```text
artifacts/mXX/
  tests.json
  commands.txt
  environment.json
  logs/
  SHA256SUMS
```

A passing CI badge is not sufficient for model/GPU gates.

## imp-derived tests

Add producer-scale mutation tests, late-layer/near-tie router tests, actual-dispatch recording tests, graph-demotion reason tests and the repeated engine lifecycle matrix from [`CUDA_STATE_LIFECYCLE_SPEC.md`](CUDA_STATE_LIFECYCLE_SPEC.md).

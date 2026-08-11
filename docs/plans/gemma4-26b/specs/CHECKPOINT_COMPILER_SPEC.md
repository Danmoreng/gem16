# Deterministic checkpoint compiler specification

## Scope

The compiler transforms a pinned Gemma 4 26B BF16 source into a text-only mixed-precision Safetensors artifact:

```text
BF16 source
  ├─ attention Q/K/V/O → FP8 + per-row BF16 scale
  ├─ shared dense MLP → NVFP4 + group-16 scale + global divisors
  ├─ routed experts → NVFP4 + group-16 scale + global divisors
  ├─ router/norms/scalars → source precision
  ├─ tied embedding/head → selected Q4_0 or NVFP4
  └─ modality/MTP tensors → omitted
```

The compiler is an offline repository tool, not an inference startup stage. The binding architecture is
[`NATIVE_CONVERTER_ARCHITECTURE.md`](NATIVE_CONVERTER_ARCHITECTURE.md). M04 retains its accepted Python
standard-library scaffold for planning, source verification, publication, provenance and byte-only `copy-v1`.
M05's promoted BF16-to-FP8 attention conversion is a versioned native C++20 batch backend; M06, M07 and M18 must
extend the same native data plane. Python is control-plane/oracle/report support only and never a production
fallback or a promoted elementwise converter.

## CLI

The current M04/M05 user-facing command is the Python control-plane wrapper. M05 additionally requires the
explicit native backend; future M06-M08 profiles must expose the same one-command interface while selecting the
shared native data plane:

```text
python tools/compile_gemma4_26b.py compile \
  --source-lock models/gemma4-26b-qat-bf16.lock.json \
  --source-directory models/checkpoints/google-gemma-4-26b-a4b-it-qat-bf16-f1e06dc \
  --compiler-manifest benchmarks/goldens/gemma4_26b/fp8/qat-compiler-plan.json \
  --profile fp8-attention-partial-v1 --head-format deferred \
  --native-encoder build/Linux/host-debug/bin/gem16-fp8-compiler \
  --threads 1 --max-host-memory 7516192768 \
  --output build/models/qat-fp8-attention-partial \
  --report artifacts/m05/qat-compile.json
```

The planned future native-family name is `gem16-checkpoint-compiler`; it is not a current executable and must not
be presented as a runnable command until implemented.

Current action requirements:

- `plan`, `compile`, and `verify`: `--source-lock`, `--compiler-manifest`, `--profile`, `--head-format`,
  `--max-host-memory`, and `--report`;
- `compile`: additionally `--output`; M05 additionally requires `--native-encoder`, with optional
  `--native-timeout-seconds` and diagnostic-only `--allow-dirty`;
- `verify`: additionally `--model`; it does not take a `--lock` alias or a `--verify` boolean because the action is
  already verification;
- `compare-reproducibility`: `--left`, `--right`, and `--report` only.

Current common options include `--source-directory` or `--source-cache` (mutually exclusive), `--staging-bytes`,
`--shard-size`, `--threads`, `--dependencies-lock`, `--reference-platform-strict`, and the retained compatibility
flag `--verify-source`. The parser defaults `--threads` to 1; evidence commands should state the value explicitly.
`--resume` is a compile option that current M04/M05 explicitly reject.

The following are planned, not current options: `--only-family`, `--only-layer`, `--emit-dequantized-sample`,
`--compare-model`, `--events`, and a meaningful generalized dry-run mode. Python remains control-plane/oracle support;
these future diagnostics cannot become an implicit numerical fallback.

## Stages

### 1. Resolve and verify

- load lock;
- verify every required file;
- reject mutable revisions;
- parse config and Safetensors metadata;
- verify tokenizer/config identity;
- create a read-only source manifest.

### 2. Compile plan

The plan contains one record per output tensor:

```python
@dataclass(frozen=True)
class TensorCompilePlan:
    output_name: str
    source_names: tuple[str, ...]
    transformation: str
    output_dtype: str
    physical_shape: tuple[int, ...]
    logical_shape: tuple[int, ...]
    shard: str
    residency: str
```

Planning must finish before payload writing. It must validate:

- every required source tensor used exactly as intended;
- every intentionally omitted tensor classified;
- no unexpected source tensor silently ignored;
- output byte totals;
- maximum temporary memory;
- shard boundaries.

### 3. Streaming transformation

Read source by memory mapping or bounded windows. Transform by row/expert/layer tiles. Write to temporary shard files.

Rules:

- no full 26B model in RAM;
- no full expert family in RAM;
- no output overwrite;
- checked shape multiplication;
- explicit little-endian representation;
- deterministic thread partitioning and reduction; M05 must preserve byte identity across its bounded
  threads-1-versus-N fixture gate;
- exceptions leave incomplete files distinguishable;
- M05 performs one native full pass over all 115 attention matrices per approved source; no duplicate
  Python/native pass is used;
- failures leave a distinguishable restart-only `.incomplete` state or clean it according to the failure path; no
  resumable verified state is promised;

### 4. Finalization

- close and fsync payloads;
- write deterministic headers/index/config;
- hash every file;
- produce `gem16_compilation.json`;
- atomically rename temporary files;
- write derived lock last.

## M05 verification boundary

M05 standalone verification is structural, hash and source-lock-only and does not reconvert tensors. Until M08's
external derived-artifact lock exists, it records `transformation_recomputed=false` and does not claim protection
against a mutable manifest rewriting its own recorded hashes. A missing native backend fails visibly; no Python
fallback is permitted.

## Native backend boundary

The promoted numerical interface is a versioned native job/backend protocol. It receives only the exact planned
source ranges and output ranges, validates the contract, and returns output hashes plus deterministic telemetry.
The M05 seed is `gem16-fp8-compiler`; M06/M07/M18 extend the shared implementation toward
`gem16-checkpoint-compiler`. A Python encoder protocol may exist for small reference fixtures, but it is not a
production plugin boundary and may not be selected as a fallback. A backend may not access arbitrary model tensors
unless declared in its native job and plan.

## Memory control

Track:

- mapped source bytes;
- resident Python/NumPy/Torch allocations;
- staging buffers;
- output buffer;
- hash buffer;
- process RSS and peak RSS.

Prefer the versioned native C++ data plane for all promoted numerical work. Python tensor libraries are allowed
only for an explicitly labeled diagnostic/reference experiment and cannot support a production conversion claim.
Avoid accidental copies from:

- non-contiguous slices;
- dtype conversions over whole tensors;
- concatenating expert arrays;
- Python lists of per-row arrays.

The current `plan` action must report projected peak memory. A future `--dry-run-plan` mode, if accepted, must
preserve that prediction contract; a real telemetry report must verify it.

## Determinism

Control:

- rounding algorithm;
- floating-point dtype;
- tie behavior;
- thread count or deterministic partitioning;
- tensor order;
- JSON key order;
- timestamps outside hashed semantic payloads;
- compression settings, if any.

Do not rely on BLAS reductions with platform-dependent scheduling for scale search. Use explicit loops or deterministic block reductions.

## Reports

Global report:

- source/artifact identity;
- planned and actual bytes;
- duration by family;
- peak RSS;
- warnings/errors;
- output file hashes.

Per tensor:

- source and output identity;
- shape;
- selected scales;
- min/max/mean/RMS;
- relative L2;
- cosine;
- maximum absolute error;
- SQNR;
- saturation count;
- code histogram;
- zero blocks;
- compilation time.

For very large tensors, statistics may be accumulated streaming. Do not sample unless the report labels it as sampled.

## Resume

M04 and M05 are restart-only and visibly reject `--resume`. A future profile may allow resume only at immutable
completed-shard or completed-tensor boundaries after accepting a cryptographically bound partial-state schema. Before
reusing a partial result it must verify the source lock, compiler config hash, completed output hash and next output
offset. Never append to an unverified partial tensor.

## Exit status

- `0`: complete and verified;
- `2`: invalid arguments/plan;
- `3`: source verification failure;
- `4`: quantization/data error;
- `5`: output I/O failure;
- `6`: reproducibility comparison failure.

## Security

- reject path traversal and symlink escape;
- cap config/header sizes;
- never execute model repository code;
- never deserialize pickle;
- use Safetensors/JSON only;
- create output with restrictive default permissions where appropriate.

## Tests

- small synthetic source builds;
- every quantizer family;
- malformed shapes/dtypes;
- interrupted build leaves a visible incomplete state; resumable-build tests begin only when a bound partial-state schema exists;
- source corruption;
- deterministic output;
- bounded memory;
- output runtime validation;
- clean error cleanup.

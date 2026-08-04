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

The compiler is an offline repository tool, not an inference startup stage.

## CLI

Suggested command:

```text
python tools/compile_gemma4_26b.py \
  --source-lock models/gemma4-26b-qat-bf16.lock.json \
  --profile sm120-text-hybrid-v1 \
  --head-format q4_0 \
  --output build/models/gemma4-26b-qat-hybrid \
  --report artifacts/compiler/gemma4-26b-qat-hybrid.json
```

Required options:

- `--source-lock`
- `--output`
- `--profile`
- `--head-format`
- `--verify`
- `--report`
- `--resume` for verified shard-level resume
- `--max-host-memory`
- `--shard-size`
- `--threads`
- `--reference-platform-strict`

Optional diagnostic options:

- `--only-family`
- `--only-layer`
- `--emit-dequantized-sample`
- `--compare-model`
- `--dry-run-plan`

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
- deterministic thread partitioning and reduction;
- exceptions leave incomplete files distinguishable.

### 4. Finalization

- close and fsync payloads;
- write deterministic headers/index/config;
- hash every file;
- produce `gem16_compilation.json`;
- atomically rename temporary files;
- write derived lock last.

## Quantizer plugin boundary

Suggested interface:

```python
class TensorCompiler(Protocol):
    name: str
    version: int
    def plan(self, source: TensorDescriptor) -> OutputDescriptor: ...
    def compile_rows(
        self,
        source_rows: memoryview,
        logical_row_start: int,
        report: TensorReportBuilder,
    ) -> Iterable[OutputChunk]: ...
```

The plugin may not access arbitrary model tensors unless declared in its plan.

## Memory control

Track:

- mapped source bytes;
- resident Python/NumPy/Torch allocations;
- staging buffers;
- output buffer;
- hash buffer;
- process RSS and peak RSS.

Prefer NumPy/C++ extension or bounded PyTorch CPU operations. Avoid accidental copies from:

- non-contiguous slices;
- dtype conversions over whole tensors;
- concatenating expert arrays;
- Python lists of per-row arrays.

Compiler `--dry-run-plan` must predict peak memory. A real telemetry report must verify it.

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

Resume is allowed only at immutable completed-shard or completed-tensor boundaries. Before reusing a partial result:

- verify source lock;
- verify compiler config hash;
- verify completed output hash;
- ensure next output offset matches plan.

Never append to an unverified partial tensor.

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
- interrupted/resumed build;
- source corruption;
- deterministic output;
- bounded memory;
- output runtime validation;
- clean error cleanup.

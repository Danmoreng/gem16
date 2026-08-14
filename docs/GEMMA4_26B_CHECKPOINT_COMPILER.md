# Gemma 4 26B deterministic checkpoint compiler

M04 implementation commit: `edd80cb6adae6d441924098870ceca9b4b1248d5`; accepted by the project owner on
2026-08-11. Retained results are in
[`evidence/gemma4_26b/m04-checkpoint-compiler-scaffold-2026-08-11.md`](evidence/gemma4_26b/m04-checkpoint-compiler-scaffold-2026-08-11.md).

## Qualification boundary

M04 implements the deterministic offline compiler scaffold. Its sole encoder is `copy-v1`, and every emitted M04
artifact is marked `m04_scaffold_not_runtime_loadable`. It does not implement NVFP4 or Q4_0 quantization, does not
produce a production 26B checkpoint or derived lock, and is never called by inference.

M05 adds the explicit `fp8-attention-partial-v1` profile. It converts only the language-model attention Q/K/V/O
source matrices to rowwise `F8_E4M3` weights plus BF16 row scales. Its artifacts are marked
`m05_fp8_attention_partial_not_runtime_loadable`: all non-attention tensors are explicitly deferred or excluded,
and M06, M07 and M08 remain separate milestones. M05 does not alter the mature 12B path or make a partial
artifact runtime-loadable.

The accepted M04 scaffold is standard-library-only and never imports model repository code, pickle, NumPy, PyTorch,
Transformers or a JIT. M05's promoted conversion is an explicitly selected versioned native C++20 batch backend;
its Python codec is oracle/fixture/report support only and is never a production fallback. Source checkpoints and all
JSON/Safetensors input are untrusted.

The enduring compiler architecture is a strict hybrid control plane/native data plane. Python may own exact plan
generation, immutable source-lock checks, schema/evidence handling and publication orchestration; promoted large
tensor arithmetic and billion-element comparisons must use the shared native C++20 converter. M04's `copy-v1` is
byte movement rather than numerical conversion. The local llama.cpp research is retained in
[`evidence/gemma4_26b/m05-llama-cpp-converter-research-2026-08-11.md`](evidence/gemma4_26b/m05-llama-cpp-converter-research-2026-08-11.md),
and the binding future architecture is
[`plans/gemma4-26b/specs/NATIVE_CONVERTER_ARCHITECTURE.md`](plans/gemma4-26b/specs/NATIVE_CONVERTER_ARCHITECTURE.md).
llama.cpp informs separation, native block codecs, threading and reference tests, but is not a gem16 format or
integrity contract. Its local findings are pinned to commit `0b14b87d7c20cb753b94b96854dd7b45306fc696`; the desired
benchmark pin is recorded separately as `153d324bcf86d220b235ca010eeb11213f32b5d1`.

## CLI

The CLI uses explicit action-first commands. The older `--stage` interface is not supported:

```text
python3 tools/compile_gemma4_26b.py plan \
  --source-lock <source.lock.json> \
  --compiler-manifest <compiler-plan.json> \
  --profile synthetic-copy-v1 --head-format source \
  --max-host-memory <absolute-rss-cap> --report <plan-report.json>

python3 tools/compile_gemma4_26b.py compile \
  --source-lock <source.lock.json> \
  --compiler-manifest <compiler-plan.json> \
  --profile synthetic-copy-v1 --head-format source \
  --max-host-memory <absolute-rss-cap> \
  --output <artifact-directory> --report <compile-report.json>

python3 tools/compile_gemma4_26b.py verify \
  --source-lock <source.lock.json> \
  --compiler-manifest <compiler-plan.json> \
  --profile synthetic-copy-v1 --head-format source \
  --max-host-memory <absolute-rss-cap> \
  --model <artifact-directory> --report <verify-report.json>

python3 tools/compile_gemma4_26b.py compare-reproducibility \
  --left <artifact-a> --right <artifact-b> --report <comparison.json>
```

### M05 FP8 attention partial profile

M05 uses `fp8-attention-partial-v1 --head-format deferred`. The checked plans are:

- `benchmarks/goldens/gemma4_26b/fp8/ordinary-compiler-plan.json`
- `benchmarks/goldens/gemma4_26b/fp8/qat-compiler-plan.json`

The following commands show the canonical Ordinary-BF16 and QAT lanes. They use explicit locked source snapshots,
a 1 GiB shard target, a 1 MiB staging buffer, strict reference-environment validation, and a 7.5 GiB absolute
host-memory cap. Use a larger explicitly measured cap for full checkpoints; do not remove the cap.

```text
export LC_ALL=C.UTF-8 LANG=C.UTF-8

python3 tools/compile_gemma4_26b.py plan \
  --source-lock models/gemma4-26b-base-bf16.lock.json \
  --source-directory models/checkpoints/google-gemma-4-26b-a4b-it-bf16-4d7ae49 \
  --compiler-manifest benchmarks/goldens/gemma4_26b/fp8/ordinary-compiler-plan.json \
  --profile fp8-attention-partial-v1 --head-format deferred \
  --max-host-memory 7516192768 --staging-bytes 1048576 \
  --reference-platform-strict --report artifacts/raw/m05/ordinary-plan.json

python3 tools/compile_gemma4_26b.py compile \
  --source-lock models/gemma4-26b-base-bf16.lock.json \
  --source-directory models/checkpoints/google-gemma-4-26b-a4b-it-bf16-4d7ae49 \
  --compiler-manifest benchmarks/goldens/gemma4_26b/fp8/ordinary-compiler-plan.json \
  --profile fp8-attention-partial-v1 --head-format deferred \
  --native-encoder build/Linux/host-release/bin/gem16-fp8-compiler --threads 8 \
  --max-host-memory 7516192768 --staging-bytes 1048576 --reference-platform-strict \
  --output build/models/ordinary-fp8-attention-partial \
  --report artifacts/raw/m05/ordinary-compile.json

python3 tools/compile_gemma4_26b.py verify \
  --source-lock models/gemma4-26b-base-bf16.lock.json \
  --source-directory models/checkpoints/google-gemma-4-26b-a4b-it-bf16-4d7ae49 \
  --compiler-manifest benchmarks/goldens/gemma4_26b/fp8/ordinary-compiler-plan.json \
  --profile fp8-attention-partial-v1 --head-format deferred --threads 8 \
  --max-host-memory 7516192768 --staging-bytes 1048576 --reference-platform-strict \
  --model build/models/ordinary-fp8-attention-partial --report artifacts/m05/ordinary-verify.json

python3 tools/compile_gemma4_26b.py plan \
  --source-lock models/gemma4-26b-qat-bf16.lock.json \
  --source-directory models/checkpoints/google-gemma-4-26b-a4b-it-qat-bf16-f1e06dc \
  --compiler-manifest benchmarks/goldens/gemma4_26b/fp8/qat-compiler-plan.json \
  --profile fp8-attention-partial-v1 --head-format deferred \
  --max-host-memory 7516192768 --staging-bytes 1048576 \
  --reference-platform-strict --report artifacts/raw/m05/qat-plan.json

python3 tools/compile_gemma4_26b.py compile \
  --source-lock models/gemma4-26b-qat-bf16.lock.json \
  --source-directory models/checkpoints/google-gemma-4-26b-a4b-it-qat-bf16-f1e06dc \
  --compiler-manifest benchmarks/goldens/gemma4_26b/fp8/qat-compiler-plan.json \
  --profile fp8-attention-partial-v1 --head-format deferred \
  --native-encoder build/Linux/host-release/bin/gem16-fp8-compiler --threads 8 \
  --max-host-memory 7516192768 --staging-bytes 1048576 --reference-platform-strict \
  --output build/models/qat-fp8-attention-partial --report artifacts/raw/m05/qat-compile.json

python3 tools/compile_gemma4_26b.py verify \
  --source-lock models/gemma4-26b-qat-bf16.lock.json \
  --source-directory models/checkpoints/google-gemma-4-26b-a4b-it-qat-bf16-f1e06dc \
  --compiler-manifest benchmarks/goldens/gemma4_26b/fp8/qat-compiler-plan.json \
  --profile fp8-attention-partial-v1 --head-format deferred --threads 8 \
  --max-host-memory 7516192768 --staging-bytes 1048576 --reference-platform-strict \
  --model build/models/qat-fp8-attention-partial --report artifacts/m05/qat-verify.json
```

`--allow-dirty` is accepted only by `compile` for diagnostic runs. It records a dirty compiler identity and is not
release evidence. M05 additionally requires `--native-encoder`; `--threads` currently defaults to 1, while the
retained release commands state the qualified value 8 explicitly for both compile and verify. An unavailable native
backend fails visibly and cannot fall back to Python. The canonical CUDA-free Linux release example uses the
executable produced by the `host-release` preset at `build/Linux/host-release/bin/gem16-fp8-compiler`; host-debug
builds use
`build/Linux/host-debug/bin/gem16-fp8-compiler`. `--resume` remains visibly rejected: publication is restart-only;
a failed run leaves only a distinguishable `.incomplete` directory or cleans it according to the failure path, and
must be restarted rather than resumed. The compile report contains per-tensor FP8 statistics when the native FP8
encoder is active, including error, scale, saturation, histogram and bounded-memory telemetry. Standalone M05
verification does not reconvert and records `transformation_recomputed=false` until M08 supplies the external
artifact lock.

M05 covers only text-model attention Q/K/V/O matrices. Non-attention tensors are explicitly deferred or excluded;
vision tensors remain compile-excluded, and global layers do not receive a synthetic V projection. The partial
artifact is intentionally not runtime-loadable and does not begin M06 expert quantization, M07 embedding/head work,
or M08 derived-artifact assembly. The 12B compiler/runtime path is unchanged.

The source directory defaults to the immutable snapshot resolved from the source lock and shared Hugging Face
cache. `--source-directory` selects an explicit view for tests or offline composition, but does not bypass complete
lock verification. `--source-cache` selects another hub-cache root. `--shard-size`, `--staging-bytes`, `--threads`
and `--reference-platform-strict` are explicit. M04 accepts exactly one thread. M05 requires an explicit native thread count; the native backend records that
count and must preserve deterministic output across the bounded threads-1-versus-N fixture gate.

Compilation verifies all locked file sizes and SHA-256 values before interpreting any Safetensors header or tensor
range, then repeats the complete source-file verification before publication to close source-change races. M05's
native batch backend reads each attention matrix once per selected source run; verification is structural, hash and
source-lock-only and does not reconvert. A dirty compiler tree fails by default. Source verification is mandatory even
if `--verify-source` is omitted.

Exit codes are:

| Code | Meaning |
|---:|---|
| 0 | complete and verified |
| 2 | invalid arguments, environment or plan |
| 3 | source-lock/file verification failure |
| 4 | tensor, provenance or artifact data failure |
| 5 | output/publication I/O failure |
| 6 | reproducibility mismatch |

## Compiler-plan schema

The required `--compiler-manifest` is a versioned, source-lock-bound quantization plan. Schema 1 contains:

- artifact profile, head format and source-contract name;
- exact source-lock SHA-256;
- target shard payload cap;
- an allowlist of locked metadata files;
- the exact omitted families `audio`, `mtp`, `video`, and `vision`;
- a canonical output-tensor array;
- explicit excluded-source tensor records;
- the reference compiler environment.

Each output record carries an `operation_id`, one or more exact source names, encoder/versioned transformation,
physical/logical dtype and shape, axis transformation, quantizer parameters, dequantization equation, semantic
role, residency, disk/runtime layouts and tied-alias state. Multiple outputs may share a source only within one
explicit operation. The M04 `copy-v1` encoder requires one source, identical dtype/shape/bytes, an identity axis and
`output = source`; it cannot duplicate a source payload.

Planning completes before payload output. Every source tensor must be covered by an output operation or one exact
exclusion. Unknown, duplicate, conflicting or silently ignored sources fail. Output names are unique and sorted.
Shard assignment greedily follows that order and the resolved payload cap; a tensor larger than the cap receives
one shard rather than being split.

The checked tiny fixture is under `tests/fixtures/gemma4_26b_compiler/`. It contains a project-generated two-shard
BF16 source, four tensors, three text outputs, one exact vision exclusion and a two-shard expected artifact.
`tools/generate_gemma4_26b_compiler_fixture.py --check` regenerates it independently and verifies the retained file
hashes in `benchmarks/goldens/gemma4_26b/compiler/m04-synthetic-copy-hashes.json`.

## Safetensors and publication

On-disk output is ordinary Safetensors plus deterministic JSON:

```text
<artifact>/
  model-00001-of-NNNNN.safetensors
  ...
  model.safetensors.index.json
  approved locked metadata files
  gem16_compilation.json
```

Safetensors tensor order, data offsets, shard names and compact headers are canonical. JSON is UTF-8 with sorted
keys, stable separators/indentation and no current timestamp or output path in hashed semantic files. Files use
restrictive creation modes.

Compilation writes only to the sibling `<output>.incomplete` directory. It fsyncs files/directories, verifies the
complete staged artifact, and performs one same-filesystem atomic directory rename. An existing output or stale
incomplete directory is never overwritten. M04 deliberately implements restart-only recovery: `--resume` fails
because no cryptographically bound partial-state schema has yet been accepted. A hard interruption therefore
leaves an unmistakable `.incomplete` directory; an ordinary caught failure cleans its own staging directory.

## Bounded source access

The compiler hashes locked files through one fixed staging buffer. Tensor payload access uses read-only mmap windows
bounded by `staging_bytes + mmap allocation granularity`; it never materializes a complete tensor, expert family or
model in ordinary host RAM. The absolute `--max-host-memory` cap must exceed process baseline/peak RSS and is checked
throughout source verification, planning, mapping, writing and verification. Reports retain:

- baseline and peak RSS;
- configured cap;
- staging bytes;
- maximum mapped window;
- maximum header;
- maximum logical tensor size.

The cap is a hard failure boundary, not permission to swap or fall back to whole-tensor reads.

## Provenance and verification

`gem16_compilation.json` schema 1 records source repository/revision/lock hash; compiler repository, clean commit,
Python/platform and dependency-lock hash; raw and resolved plan hashes; profile/head/encoder semantics; all omitted
groups; one source-range/hash record per output source; every output tensor hash and transformation; all excluded
tensor hashes; and every shard/index/copied-metadata file hash.

The manifest cannot contain its own hash without a cycle. Its `file_hash_scope` states that M08's external derived
artifact lock supplies that final self-hash. M04 compile and verify reports nevertheless calculate and retain the
actual `gem16_compilation.json` SHA-256.

`verify` repeats complete source-lock verification, plan/source coverage, source tensor hashes, exclusion hashes,
output file hashes, independent Safetensors bounds/index checks, output tensor hashes and dtype/shape, copied
metadata identity, byte totals and provenance semantics. It rejects symlinks, path traversal, unknown files,
duplicate JSON/tensors, altered plan/source identities and any unrecorded output.

## Canonical compiler environment

M04 designates Linux x86-64, little-endian, `C.UTF-8`, CPython 3.14.6 and one compiler thread as the canonical byte
lane. M05 retains the canonical host environment but uses an explicitly selected native C++20 backend and explicit
thread count; its native protocol/toolchain identity is recorded in compiler provenance.
`tools/gem16_compile/dependencies.lock.json` freezes the M04 standard-library-only dependency policy.
`--reference-platform-strict` checks the plan's system, machine, exact interpreter version, byte order and locale.
Other platforms may run semantic tests but cannot publish differently hashed release artifacts under the canonical
label.

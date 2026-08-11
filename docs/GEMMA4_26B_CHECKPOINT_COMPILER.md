# Gemma 4 26B deterministic checkpoint compiler

M04 implementation commit: `edd80cb6adae6d441924098870ceca9b4b1248d5`; implementation gates pass and owner
acceptance is pending. Retained results are in
[`evidence/gemma4_26b/m04-checkpoint-compiler-scaffold-2026-08-11.md`](evidence/gemma4_26b/m04-checkpoint-compiler-scaffold-2026-08-11.md).

## Qualification boundary

M04 implements the offline compiler scaffold only. Its sole encoder is `copy-v1`, and every emitted artifact is
marked `m04_scaffold_not_runtime_loadable`. It does not implement FP8, NVFP4 or Q4_0 quantization, does not produce
a production 26B checkpoint or derived lock, and is never called by inference. M05-M07 add the numerical encoders;
M08 assembles and qualifies the first directly loadable artifact.

The compiler is standard-library-only and never imports model repository code, pickle, NumPy, PyTorch,
Transformers or a JIT. Source checkpoints and all JSON/Safetensors input are untrusted.

## CLI

The CLI uses explicit actions:

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

The source directory defaults to the immutable snapshot resolved from the source lock and shared Hugging Face
cache. `--source-directory` selects an explicit view for tests or offline composition, but does not bypass complete
lock verification. `--source-cache` selects another hub-cache root. `--shard-size`, `--staging-bytes`, `--threads`
and `--reference-platform-strict` are explicit. M04 accepts exactly one thread.

Compilation verifies all locked file sizes and SHA-256 values before interpreting any Safetensors header or tensor
range, then repeats the complete source-file verification before publication to close source-change races. A dirty
compiler tree fails by default. `--allow-dirty` is diagnostic, records `dirty=true`, and cannot
create release evidence. Source verification is mandatory even if `--verify-source` is omitted.

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
lane. `tools/gem16_compile/dependencies.lock.json` freezes the standard-library-only dependency policy.
`--reference-platform-strict` checks the plan's system, machine, exact interpreter version, byte order and locale.
Other platforms may run semantic tests but cannot publish differently hashed release artifacts under the canonical
label.

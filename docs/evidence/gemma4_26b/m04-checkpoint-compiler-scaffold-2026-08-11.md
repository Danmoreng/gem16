# Gemma 4 26B M04 checkpoint compiler scaffold handoff

Date: 2026-08-11
Branch: `feat/gemma4-26b`
Implementation commit: `edd80cb6adae6d441924098870ceca9b4b1248d5`
Milestone: M04 — deterministic checkpoint compiler scaffold
Status: implementation complete; ready for project-owner review; M05 remains blocked

## Scope delivered

M04 adds a repository-owned offline compiler scaffold and does not change inference behavior. The implementation is
under `tools/compile_gemma4_26b.py` and `tools/gem16_compile/` with explicit `plan`, `compile`, `verify`, and
`compare-reproducibility` actions. It is Python-standard-library-only, is absent from CMake/runtime targets, never
imports or executes model repository code, and has no GPU path.

The scaffold implements:

- complete source-lock size/SHA-256 verification before Safetensors interpretation and again before publication;
- duplicate-rejecting, bounds-checked multi-shard Safetensors indexing;
- a versioned, source-lock-bound compiler plan with exact output/exclusion coverage;
- a versioned encoder registry whose sole M04 encoder is byte-identical `copy-v1`;
- deterministic output order, shard assignment, Safetensors headers and canonical JSON;
- bounded read-only mmap windows and an absolute process-RSS cap;
- restrictive file modes, sibling `<output>.incomplete` staging, fsync, strict staged verification and one
  same-filesystem atomic rename;
- restart-only recovery: `--resume`, stale incomplete state and an existing output fail visibly;
- schema-1 plan and `gem16_compilation.json` contracts with source/compiler/plan/file/tensor/exclusion provenance;
- strict offline re-verification and byte-for-byte artifact comparison.

The output status is always `m04_scaffold_not_runtime_loadable`. FP8, NVFP4, Q4_0, a production 26B artifact,
derived artifact lock, loader support and 26B execution remain intentionally absent.

## Frozen schema and fixture

The checked fixture under `tests/fixtures/gemma4_26b_compiler/` is generated entirely by
`tools/generate_gemma4_26b_compiler_fixture.py`. Its source has two shards, four BF16 tensors and standard locked
metadata. The compiler emits:

- three text tensors totaling 176 payload bytes;
- two output shards with 128 and 48 payload bytes;
- one exact 16-byte vision exclusion;
- five byte-identical approved metadata files;
- one deterministic index and one complete provenance manifest.

The independent repository Safetensors reader reconciles all three output tensor names, shapes, ranges and bytes.
The retained fixture/golden hashes are in
`benchmarks/goldens/gemma4_26b/compiler/m04-synthetic-copy-hashes.json`. Normative schemas are
`tools/gem16_compile/schemas/compiler-plan.schema.json` and
`tools/gem16_compile/schemas/gem16-compilation.schema.json`.

Every output tensor records operation ID, all source names/ranges/hashes, output range/hash, encoder,
transformation/version, physical/logical dtype and shape, axis transformation, quantizer parameters,
dequantization equation, role, residency, disk/runtime layout and alias state. Every exclusion records its exact
source range/hash, family, role, residency and reason. Every shard, index and copied metadata file is hashed. The
manifest's non-circular self-hash is retained by compile/verify reports; M08 still owns the external artifact lock.

## Clean canonical reproducibility result

Two independent runs used Linux x86-64, little-endian, CPython 3.14.6, explicit `LC_ALL=C.UTF-8`, one thread, a
4,096-byte staging buffer, and a 134,217,728-byte absolute RSS cap. Both were produced from the clean implementation
commit above with `compiler_dirty=false`.

```text
compiler plan SHA-256:
  6294abda5ccd428f7fe672fd4e95d0fa52f57f7f263c9b99ce22aa52a5c991fc
source lock SHA-256:
  dd252602387b8dcd6939a6aca9336b1d11e8608a873540cef95e698f208bf2d1
gem16_compilation.json SHA-256, both runs:
  640266a228a9c298b1ff2d3feb10e214baba202afd06cfb9d0f0a7798853e8d6
artifact files compared: 9
mismatches: 0
compile A peak RSS: 28,250,112 bytes
compile B peak RSS: 28,229,632 bytes
maximum mapped window: 408 bytes
```

The first strict attempt under the inherited `en_US.UTF-8` locale failed with exit code 2 before planning. Re-running
under the declared canonical locale passed. This is the intended visible environment gate, not a fallback.

Raw retained reports:

- `m04-environment.json`
- `m04-plan-report.json`
- `m04-compile-a-report.json`
- `m04-compile-b-report.json`
- `m04-verify-a-report.json`
- `m04-verify-b-report.json`
- `m04-reproducibility.json`

all under `docs/evidence/gemma4_26b/`.

## Bounded-memory evidence

A separate generated source contains one 2,097,152-byte BF16 tensor. It compiled through a 4,096-byte staging
buffer without whole-tensor allocation:

```text
maximum logical tensor: 2,097,152 bytes
maximum mmap window:       4,376 bytes
baseline RSS:            31,068,160 bytes
peak RSS:                36,675,584 bytes
absolute cap:            70,230,016 bytes
status: pass
```

The raw report is `docs/evidence/gemma4_26b/m04-bounded-memory-report.json`. This qualifies scaffold orchestration,
not the memory behavior of future numeric encoders; M05 and M06 must measure their own tile/workspace limits.

## Rejection and interruption coverage

The 12 focused tests cover successful planning/compilation/verification plus the following hard failures:

- wrong source lock and source payload corruption;
- malformed but correctly re-locked Safetensors header;
- incomplete tensor coverage and changed plan provenance;
- corrupted output tensor/file hashes;
- unsafe metadata path;
- an artifact-file symlink;
- dirty release identity;
- an already existing final output;
- unsupported resume;
- injected encoder interruption;
- a 2 MiB tensor under the bounded-window/RSS assertions;
- reproducibility mismatch with exit code 6.

A caught interruption never publishes a final directory and cleans staging created by that invocation. A simulated
hard interruption leaves only the explicit `.incomplete` path, which must be removed before a full restart.

## Verification commands and results

```text
python3 tools/generate_gemma4_26b_compiler_fixture.py --check
  pass: 20 generated files current

python3 -m py_compile tools/compile_gemma4_26b.py \
  tools/generate_gemma4_26b_compiler_fixture.py tools/gem16_compile/*.py \
  tests/python/test_gemma4_26b_checkpoint_compiler.py
  pass

python3 -m unittest tests.python.test_gemma4_26b_checkpoint_compiler -v
  12/12 passed

python3 -m unittest discover -s tests/python -p 'test_*.py'
  95/95 passed

cmake --preset host-debug
cmake --build --preset host-debug -j2
ctest --preset host-debug --output-on-failure
  1/1 passed (3.94 s)

cmake --preset host-sanitize
cmake --build --preset host-sanitize -j2
ctest --preset host-sanitize --output-on-failure
  1/1 passed with ASan/UBSan (14.14 s)

cmake --preset blackwell-release
cmake --build --preset blackwell-release -j2
ctest --preset blackwell-release --output-on-failure
  2/2 passed: unit and CUDA (34.19 s total)

python3 tools/validate_inference.py \
  --run build/Linux/blackwell-release/bin/gem16-run \
  --model models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497 \
  --output /tmp/m04-12b-validation.json
  pass: exact-blue output [9503,106], fallbacks 0,
        weight arena 9,304,895,488 bytes
```

The kickoff also reverified all 11 QAT-BF16 locked files and all 12 ordinary-BF16 locked files before
implementation. Their immutable revisions remain
`f1e06dc520982d9b9edd76859fdb7ab209449949` and
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`.

## Exit criteria

- [x] A verified locked source can be planned without writing output.
- [x] Every source tensor is mapped once or explicitly excluded.
- [x] Standard deterministically sharded Safetensors and canonical metadata are emitted.
- [x] Every output/exclusion/file has complete reproducible provenance and hashes.
- [x] Two clean canonical runs are byte-identical.
- [x] Corruption, wrong locks/plans, path hazards and interruption fail visibly.
- [x] Host memory remains below an explicit cap for a tensor much larger than the mmap window.
- [x] Runtime/build coupling is absent and the 12B production path is unchanged.
- [x] Production quantization and runtime loading remain disabled.
- [ ] Project-owner acceptance is recorded.

## Review request and next dependency

Please review and accept M04 as the deterministic compiler scaffold. No M05 code has started. After owner
acceptance, M05 may add the deterministic BF16-to-FP8 attention encoder and numerical reference vectors through the
frozen encoder/plan/provenance interfaces; publication and runtime boundaries remain unchanged.

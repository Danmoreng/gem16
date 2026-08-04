# Checkpoint compiler CLI specification

## Main command

```text
python tools/compile_gemma4_26b.py [options]
```

## Required actions

### `plan`

```text
compile_gemma4_26b.py plan --source-lock ... --profile ... --head-format ...
```

Produces no weights. Outputs:

- source verification;
- tensor mapping;
- omitted tensors;
- projected bytes;
- shard plan;
- peak memory estimate;
- warnings.

### `compile`

```text
compile_gemma4_26b.py compile --source-lock ... --output ...
```

Creates the artifact atomically.

### `verify`

```text
compile_gemma4_26b.py verify --model ... --lock ...
```

Validates files, schema, tensors and provenance.

### `compare-reproducibility`

```text
compile_gemma4_26b.py compare-reproducibility --left ... --right ...
```

Compares every file/tensor and reports first mismatch.

## Options

```text
--source-lock PATH
--source-cache PATH
--profile NAME
--head-format q4_0|nvfp4
--output PATH
--report PATH
--shard-size BYTES
--max-host-memory BYTES
--threads N
--resume
--verify-source
--reference-platform-strict
--only-family NAME
--only-layer N
--dry-run
```

## Output safety

- output must not exist unless `--resume`;
- temp directory is `<output>.incomplete`;
- completed artifact is atomically renamed;
- incomplete state contains machine-readable progress;
- resume validates every completed tensor/shard;
- no `--force` for release builds.

## Logging

Human log:

```text
[verify] ...
[plan] ...
[compile] layer 12 experts 64–95 ...
[hash] ...
[done] ...
```

JSONL event stream optional:

```text
--events artifacts/compiler/events.jsonl
```

Do not emit secrets or full model values.

## Progress

Progress is based on source/output bytes, not tensor count. Large expert tensors dominate.

## Exit codes

Defined in `CHECKPOINT_COMPILER_SPEC.md`.

## Resource limits

If predicted/observed RSS exceeds `--max-host-memory`:

- stop after completing current atomic chunk if safe;
- report expected/actual;
- leave resumable verified state;
- do not invoke swap-heavy fallback silently.

## Profile config

Store profile definitions in version-controlled data or code:

```yaml
name: sm120-text-hybrid-v1
attention: fp8-per-channel-v1
shared_mlp: nvfp4-group16-v1
routed_experts: nvfp4-group16-v1
router: bf16-source
head: ${head_format}
omit: [vision, audio, video, mtp]
```

Hash resolved profile into artifact provenance.

## Tests

- help/argument parsing;
- every action;
- invalid combinations;
- output exists;
- interruption/resume;
- limit exceeded;
- JSON reports;
- exit codes;
- deterministic logs except timestamps.

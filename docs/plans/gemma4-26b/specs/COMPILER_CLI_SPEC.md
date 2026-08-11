# Checkpoint compiler CLI specification

## Main command

The user-facing command currently consists of the Python control plane and an explicitly selected native data-plane
backend. This keeps M04 publication/provenance orchestration while making M05 arithmetic native:

```text
python tools/compile_gemma4_26b.py [plan|compile|verify|compare-reproducibility] [options]
```

The planned future unified executable is `gem16-checkpoint-compiler`; it is not implemented yet. Do not document it
as a runnable command before its native profiles and publication contract exist. See
[`NATIVE_CONVERTER_ARCHITECTURE.md`](NATIVE_CONVERTER_ARCHITECTURE.md).

## Current actions and required options

The following are the current action-specific requirements. The wrapper is intentionally explicit: there is no
implicit source, profile, output, model or report selection.

### `plan`

```text
python3 tools/compile_gemma4_26b.py plan \
  --source-lock ... --compiler-manifest ... --profile ... --head-format ... \
  --max-host-memory ... --report ...
```

Produces no weights. It verifies the source and reports tensor mapping, omitted tensors, projected bytes, shard plan,
peak-memory telemetry and warnings. `--source-directory` or `--source-cache` selects the source view; the selected
view is still completely checked against the lock.

### `compile`

```text
python3 tools/compile_gemma4_26b.py compile \
  --source-lock ... --compiler-manifest ... --profile ... --head-format ... \
  --max-host-memory ... --output ... --report ...
```

Creates the artifact atomically. M05 additionally requires `--native-encoder PATH`; `--threads N` defaults to 1,
but retained evidence commands state it explicitly. `--native-timeout-seconds N` controls the native backend timeout. `--allow-dirty` is compile-only
and diagnostic; it is not release evidence. `--resume` is accepted only to fail visibly because current M04/M05
publication is restart-only.

### `verify`

```text
python3 tools/compile_gemma4_26b.py verify \
  --source-lock ... --compiler-manifest ... --profile ... --head-format ... \
  --max-host-memory ... --model ... --report ...
```

Validates the locked source, artifact files, schema, tensors and provenance. There is no `--lock` or `--verify`
option: `--source-lock` identifies the lock and verification is the action itself. M05 verification is structural,
hash and source-lock-only and does not reconvert tensors.

### `compare-reproducibility`

```text
python3 tools/compile_gemma4_26b.py compare-reproducibility \
  --left ... --right ... --report ...
```

Compares every file in two artifacts and reports the first mismatch.

## Current common options

```text
--source-lock PATH                 required by plan/compile/verify
--source-directory PATH            mutually exclusive with --source-cache
--source-cache PATH                mutually exclusive with --source-directory
--compiler-manifest PATH           required by plan/compile/verify
--profile NAME                     required by plan/compile/verify
--head-format source|q4_0|nvfp4|deferred
--max-host-memory BYTES            required by plan/compile/verify
--staging-bytes BYTES              default 1048576
--shard-size BYTES                optional
--threads N                        default 1; M05 release commands choose explicitly
--dependencies-lock PATH           optional
--reference-platform-strict        optional
--verify-source                    compatibility flag; source verification is always mandatory
```

Current action-specific options are `--report` for every action, `--output` for `compile`, `--model` for `verify`,
`--native-encoder`, `--native-timeout-seconds`, `--resume` and `--allow-dirty` for `compile`, and `--left`/`--right`
for `compare-reproducibility`. The parser also accepts `--dry-run` on `plan` for compatibility, but it is not a
separate current execution mode and must not be used as evidence.

The following are **planned, not current CLI options**: `--only-family`, `--only-layer`, `--events`,
`--emit-dequantized-sample`, `--compare-model`, a meaningful generalized `--dry-run-plan`, and any resumable partial
state. Do not present them as runnable until their profile contracts and tests exist.

## M05 native conversion policy

For `fp8-attention-partial-v1 --head-format deferred`, `--native-encoder` is required. `--threads` currently
defaults to 1 and must be stated explicitly in retained evidence. The native C++20 batch backend is the only promoted converter; Python is oracle/fixture/report support
only and unavailable native support fails visibly. M05 runs one complete native pass for Ordinary BF16 and one for
QAT BF16, with no duplicate Python/native conversion or second full artifact run solely for reproducibility. A short
native throughput probe precedes a full run; projected long runs require explicit owner approval. Standalone M05
`verify` is structural/hash/source-lock-only, does not reconvert, and reports
`transformation_recomputed=false` until M08's external artifact lock.

The older `--stage` interface is not supported. M05 output remains a non-runtime-loadable partial artifact; M06,
M07 and M08 are separate milestones.

## Output safety

- an existing output or stale `<output>.incomplete` is a visible error for current M04/M05 restart-only profiles;
- temp directory is `<output>.incomplete`;
- completed artifact is atomically renamed only after staged verification;
- resumable state is not claimed until a cryptographically bound partial-state schema is accepted;
- no `--force` for release builds.

## Logging

The current wrapper emits human-readable progress on stderr and the canonical report on the requested `--report`
path. A structured JSONL event stream (`--events`) is planned, not a current option. Do not emit secrets or full model
values.

## Progress

Progress is based on source/output bytes, not tensor count. Large expert tensors dominate.

## Exit codes

Defined in `CHECKPOINT_COMPILER_SPEC.md`.

## Resource limits

If predicted/observed RSS exceeds `--max-host-memory`:

- fail visibly at the bounded operation;
- report expected/actual usage when a report can be written;
- leave only the distinguishable restart-only `.incomplete` state or clean it according to the failure path;
- never claim resumable verified state;
- do not invoke a swap-heavy or precision-changing fallback silently.

## Profile config

Store profile definitions in version-controlled data or code. Numerical profile implementations must select the
shared native C++ data plane; Python profile data is control-plane configuration, not a conversion implementation:

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

- help/argument parsing and exact per-action required options;
- every current action;
- invalid profile/head and source combinations;
- output/report collision and existing-output rejection;
- interruption and restart-only rejection (not resume success);
- limit exceeded;
- JSON reports and exit codes;
- deterministic semantic output; wall-clock progress is not part of hashed evidence.

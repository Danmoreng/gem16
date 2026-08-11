# Checkpoint provenance and immutability specification

## Goal

Every source and derived model used for correctness, quality or performance must be reconstructable from immutable inputs. A model directory is not trusted merely because its file names look familiar.

## Source locks

Create separate locks for:

```text
models/gemma4-26b-qat-bf16.lock.json
models/gemma4-26b-base-bf16.lock.json
models/gemma4-26b-unsloth-nvfp4.lock.json
models/gemma4-26b-qat-q4_0.lock.json
models/gemma4-26b-gem16-hybrid.lock.json
```

Each source lock must contain:

- schema version;
- repository ID;
- full 40-character immutable commit;
- resolution timestamp;
- model-card/config/tokenizer revisions when different;
- every downloaded file;
- exact byte size;
- SHA-256;
- Git/LFS/Xet identity when available;
- optional per-file source override;
- license or terms reference identifier;
- expected architecture and profile.

Do not use `main`, tags without resolved commits or approximate file sizes.

## Derived artifact provenance

`gem16_compilation.json` inside a compiled artifact must contain:

```json
{
  "schema_version": 1,
  "artifact_profile": "sm120-text-hybrid-v1",
  "source": {
    "lock_sha256": "...",
    "repository": "...",
    "revision": "..."
  },
  "compiler": {
    "repository": "Danmoreng/gem16",
    "commit": "...",
    "dirty": false,
    "python": "...",
    "platform": "...",
    "dependencies_lock_sha256": "...",
    "implementation": "gem16_compile_m05_native_v1",
    "native_backend": {
      "protocol": "gem16-fp8-batch-v1",
      "sha256": "...",
      "toolchain": "..."
    }
  },
  "quantization": {
    "attention": "fp8-per-channel-v1",
    "experts": "nvfp4-group16-v1",
    "embedding_head": "q4_0-v1"
  },
  "omitted_families": ["vision", "audio", "video", "mtp"],
  "tensors": []
}
```

Every output tensor record must include:

- output name;
- output dtype;
- physical and logical shape;
- byte length and SHA-256;
- source tensor name(s);
- source tensor SHA-256 or shard/range identity;
- transformation name and version;
- axis split/transpose/reshape description;
- quantizer parameters;
- dequantization equation;
- role and residency class.

## Compiler reproducibility

A release compiler run is valid only when:

- the source lock verifies;
- the compiler tree is clean;
- dependency versions are locked;
- locale and JSON ordering are controlled;
- shard and tensor ordering are deterministic;
- no random calibration is used without a pinned seed and corpus;
- output files are written atomically;
- a second clean build produces identical hashes for complete-artifact milestones.

M04 retains the accepted Python standard-library scaffold and its synthetic two-run reproducibility gate. For M05,
the promoted attention conversion must use the explicitly selected versioned native C++20 batch backend. Python is
oracle/fixture/report support only, never a production fallback. M05 performs one native full 115-matrix Ordinary-BF16
run and one native full 115-matrix QAT-BF16 run; it does not perform duplicate Python/native conversion or a second
full M05 artifact run solely for reproducibility. Native exhaustive codec tests, byte-golden rows, bounded
threads-1-versus-N fixtures and complete output hashes establish M05 determinism. A short native throughput probe
precedes any full run, and a projected long run requires explicit owner approval.

M06 and M07 follow the same partial-stage policy: one native full Ordinary-BF16 conversion and one native full QAT-BF16
conversion per selected profile or head candidate. Do not add a second full partial artifact solely for reproducibility;
use exhaustive native codec tests, byte fixtures, bounded thread-identity fixtures, complete output hashes and small
independent oracles. If M07 evaluates both Q4_0 and NVFP4 heads, each candidate is compiled once per required source
and the retained outputs are compared. This policy does not waive M08's two-clean-build requirement for the final
complete artifact.

If deterministic bytes cannot be achieved across operating systems, designate one reference compiler platform and require semantic reproducibility elsewhere. Record the reason. Do not quietly accept varying output.

## Audit chain

Recommended chain:

```text
source lock
  → source snapshot verification
  → compiler invocation record
  → per-tensor transformation report
  → output file hashes
  → derived artifact lock
  → runtime manifest
  → benchmark/quality result containing artifact lock hash
```

Every report must carry the final artifact lock SHA-256. File paths alone are insufficient. Native conversion reports
also carry the backend protocol, binary hash, native compiler/toolchain identity and explicit thread count. Until M08
supplies the external derived-artifact lock, standalone M05 verification is structural, hash and source-lock-only, does
not reconvert tensors, and must disclose `transformation_recomputed=false`; it must not claim mutable-manifest
adversarial binding.

## Artifact hosting

The compiled artifact may be hosted in a Hugging Face repository or another immutable store. Distribution must preserve:

- original model license/terms;
- derived-artifact attribution;
- source model references;
- compiler provenance;
- checksums;
- no executable repository code requirement.

`trust_remote_code` must not be required for runtime loading.

## Contamination prevention

The compiler may read:

- source model files;
- a pinned, disjoint calibration corpus only when a quantizer actually requires activations;
- compiler configuration.

It must not read:

- held-out quality test outputs;
- benchmark result scores;
- previous candidate outputs to selectively improve specific tensors;
- Unsloth tensors while compiling the QAT artifact.

Unsloth is an external comparison, not a source for the QAT-derived weights.

## Lock update policy

A source or compiler update creates a new lock and invalidates downstream evidence. Never edit hashes in place while retaining an old quality/performance label.

Required reruns:

| Changed item | Minimum rerun |
|---|---|
| tokenizer/template only | tokenization, chat, all generation quality |
| source weight revision | M03–M23 |
| compiler/quantizer | M05/M06 or M07 through M23 |
| CUDA kernels only | operator, full-model, quality regression, performance |
| driver/CUDA/CUTLASS | correctness, disassembly, memory, performance |
| artifact hosting metadata only | lock/download/product tests |

## Failure behavior

On any hash mismatch:

- stop;
- name the file;
- show expected and actual size/hash;
- do not repair automatically from an unverified local copy;
- leave incomplete downloads separate;
- never run inference on a partially verified artifact.

## Release evidence

A release report must identify:

- code commit;
- artifact lock hash;
- all source lock hashes;
- toolchain lock hash;
- benchmark suite hash;
- quality suite hash;
- machine ID;
- exact commands.

This information belongs in machine-readable JSON as well as human-readable release notes.

## External implementation provenance

When a donor runtime supplies parsing logic, fixtures or code, record its repository, commit, path, license, source
hash, adoption mode and destination. Checkpoint provenance and code provenance are separate records and both are
required. For llama.cpp, the local research evidence is version-scoped to commit
`0b14b87d7c20cb753b94b96854dd7b45306fc696`; the desired benchmark pin is separately recorded as
`153d324bcf86d220b235ca010eeb11213f32b5d1`. An ignored cache checkout is never an implicit build dependency.

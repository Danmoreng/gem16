# Gemma 4 26B A4B experimental track

Status: M00-M05 accepted. M05's native C++20 encoder/comparator, exact Ordinary/QAT plans, clean full conversion per
source, structural verification, complete hashes, weight-only comparison and exact 12B regression are retained in
[M05 acceptance](evidence/gemma4_26b/m05-fp8-attention-compiler-acceptance-2026-08-11.md). The partial artifact is
non-runtime-loadable and makes no model-quality claim. M06 is dependency-unblocked but not started. No production 26B
runtime execution exists. The initial state remains in the [M05 kickoff](evidence/gemma4_26b/m05-kickoff-2026-08-11.md) and the version-scoped
[llama.cpp research](evidence/gemma4_26b/m05-llama-cpp-converter-research-2026-08-11.md).

Production hypothesis: `gem16-gemma4-26b-a4b-qat-hybrid-text`

The 26B track targets text-only Gemma 4 26B A4B inference on one approximately 16 GB NVIDIA Blackwell GPU while
preserving the statically specialized Gemma 4 12B Unified product path. The binding implementation order and gates
are in the [master plan](plans/gemma4-26b/00_MASTER_IMPLEMENTATION_PLAN.md). The plan package is repository-maintained;
the 2026-08-11 branch-policy amendment updates it to the single-branch workflow and regenerates its checked
manifest. Current milestone state and acceptance evidence are recorded in this document, [ROADMAP.md](ROADMAP.md),
the [milestone board](plans/gemma4-26b/MILESTONE_STATUS_BOARD.md) and
[`docs/evidence/gemma4_26b/`](evidence/gemma4_26b/).

## Development branch policy

All remaining M03-M25 development uses the long-lived `feat/gemma4-26b` branch created from `main` after the
accepted M00-M02 foundation was integrated. Do not create milestone-specific development branches. Each milestone
still owns a bounded change set, evidence, one or more descriptive commits and an explicit exit gate; dependent
work does not start until that gate passes. Integration back to `main` remains an explicit project-owner decision.

The older `feat/26b-m00-*`, `feat/26b-m01-*` and `feat/26b-m02-*` names remain in dated evidence as historical facts.
They do not define the workflow for M03 or later work.

## Terminology

- **Source checkpoint:** an immutable upstream model snapshot whose repository, full revision, files, byte sizes,
  hashes, terms and architecture are recorded in a source lock.
- **Compiled checkpoint:** an offline, deterministic, project-built Safetensors artifact derived from exactly one
  locked source checkpoint by a locked gem16 compiler. It includes versioned compilation metadata and output
  hashes. It is not an official upstream quantization.
- **Runtime layout:** the sole device representation selected by a model plan and produced while streaming a
  verified compiled tensor into its final allocation. It is not a second checkpoint or persistent device copy.
- **Production profile:** a fully native, quality-qualified, all-weights-resident configuration eligible for product
  and performance claims.
- **Hybrid converter:** the accepted compiler architecture: Python control-plane code may own locks, exact mapping,
  plans, schemas, evidence and publication orchestration; promoted large tensor arithmetic belongs to one shared,
  versioned native C++20 data plane. M04 `copy-v1` is byte movement, not quantization. Python numerical code is
  oracle/fixture/diagnostic support only unless a separate decision explicitly approves a tensor library path.
- **Diagnostic profile:** an explicitly requested reference, fallback or capture configuration. Every changed
  operator/tensor is reported, and its result is ineligible for primary performance claims.
- **Baseline:** a pinned external or internal comparison configuration with checkpoint, format, K/V precision,
  semantics and timing differences stated beside every result.

## Derived-checkpoint contract

The first production candidate is compiled offline from one exact Google Gemma 4 26B A4B QAT BF16 revision. M01
must resolve and lock that revision before any source download or golden capture is accepted. The contract is:

1. The source lock records a full immutable commit, every downloaded file, exact size and SHA-256, tokenizer and
   model-card overrides, terms reference and expected architecture. Mutable branches or unresolved tags are
   forbidden.
2. The compiler runs outside the inference process from a clean, exact gem16 commit with locked dependencies,
   locale, ordering, platform and invocation. It uses bounded host memory, deterministic tensor/shard ordering,
   checked arithmetic and atomic output publication. M04 uses its accepted Python standard-library scaffold for
   planning, locking, publication and byte-copy evidence; `copy-v1` is not tensor quantization. The promoted M05
   attention conversion uses an explicitly selected versioned native C++20 batch backend, with Python retained only
   as an oracle/fixture/report aid and never as a fallback. M06, M07 and large M18 conversion/comparison work extend
   the same native data plane rather than introducing separate Python converters. See
   [the native converter architecture](plans/gemma4-26b/specs/NATIVE_CONVERTER_ARCHITECTURE.md).
3. A second clean reference-platform run must reproduce the compiled bytes exactly for complete artifacts. For M05,
   one native full 115-matrix Ordinary run and one native full 115-matrix QAT run are required instead; no duplicate
   Python/native conversion or second full M05 artifact run is performed solely for reproducibility. Native exhaustive
   codec tests, byte-golden rows, bounded threads-1-versus-N fixtures and complete output hashes establish M05
   determinism. If cross-platform byte identity
   is not achievable, one canonical compiler environment is designated; other platforms must still prove semantic
   agreement rather than silently accepting different hashes.
4. The output remains standard Safetensors plus a versioned `gem16_compilation.json`. Every output tensor records
   its source identity, transformation/version, logical and physical shape, dtype, quantizer parameters,
   dequantization equation, byte length, SHA-256, role and residency class.
5. The primary artifact derives every mathematical tensor from the same QAT BF16 source. It must not splice
   Unsloth, official Q4_0, ModelOpt or previous candidate tensors into the production artifact.
6. Vision, audio, video and MTP tensor families are omitted and listed explicitly. The initial profile is text-only;
   unsupported modality or MTP requests fail before model execution.
7. Attention is compiled to the qualified FP8 contract; routed experts and the always-active dense/shared MLP use
   the qualified NVFP4 contract; router, norms and scalar controls retain source BF16/F32 semantics. The tied
   embedding/output-head representation remains an M07 evidence decision. No quality conclusion is assumed.
8. A running process keeps one resident representation per weight. Tied embedding/output storage is physically
   aliased, and source-order plus runtime-order device copies are forbidden. Any load-time transform streams into
   the final allocation and is separately recorded as runtime layout.
9. The inference runtime verifies the compiled lock and metadata, then loads the artifact directly. Server startup
   never quantizes, requantizes, writes a compiled checkpoint, executes repository model code or requires
   `trust_remote_code`.
10. Missing native kernels or unsupported tensor contracts fail visibly. Diagnostic fallback requires an explicit
    option and complete reporting; fallback/offload results cannot support production performance claims.
11. Every quality, memory or benchmark artifact records the code commit, compiled-artifact lock hash, source lock
    hashes and toolchain lock. Claims call the model a project-built gem16 derivative of Google QAT BF16, never
    “official Google NVFP4.”
12. Distribution or hosting is blocked until M01 records applicable source/derived terms and owner approval.

The local llama.cpp conversion research is a version-scoped engineering reference: it inspected clean checkout
`0b14b87d7c20cb753b94b96854dd7b45306fc696`, while the desired benchmark pin names `153d324bcf86d220b235ca010eeb11213f32b5d1`.
Its separation of model mapping from native codecs, multithreaded conversion and reference-versus-optimized tests
may inform Gem16 after exact-contract tests; its intermediate GGUF, permissive fallbacks and weaker provenance,
security, memory and publication guarantees are not adopted. Findings are retained in
[evidence/gemma4_26b/m05-llama-cpp-converter-research-2026-08-11.md](evidence/gemma4_26b/m05-llama-cpp-converter-research-2026-08-11.md).

The detailed machine-readable fields and invalidation rules are binding in the
[checkpoint provenance specification](plans/gemma4-26b/specs/CHECKPOINT_PROVENANCE_SPEC.md).

## Required comparison set

The primary candidate cannot be judged alone. The retained matrix includes:

- direct published Unsloth NVFP4 as the practical external NVFP4 baseline;
- ordinary Google IT BF16 compiled through the same gem16 recipe as the quantizer control;
- Google QAT BF16 compiled through the gem16 recipe as the primary hypothesis;
- official Google QAT Q4_0 through its pinned reference runtime;
- QAT BF16/reference operators where feasible as a numerical oracle;
- pinned ModelOpt/imp only as a disclosed negative/control arm.

No comparison is called parity unless tensor inventory and execution semantics establish it.

## M02 model-variant boundary

M02 introduces explicit immutable classifications for `gemma4_unified_12b`, `gemma4_moe_26b_a4b` and the existing
assistant. The 26B classifier requires the locked Gemma 4 architecture/model identifiers and MoE declaration; its
validator then checks every exact dimension, layer type, attention/KV field, local/global RoPE control and source
modality field independently. Classification is never based on a filename or directory name.

The 26B traits are inspectable and text-capable but not executable. Vision, audio, video and MTP capabilities are
false for the initial product even though source vision metadata is validated for checkpoint identity. Manifest
schema 2 reports the variant, MoE dimensions, capabilities and separate runtime/tensor-contract validation states.
M03, not M02, owns the canonical 26B tensor-name and shape contract.

## M03 tensor and residency boundary

M03 advances manifest output to schema 3 and freezes `gemma4_26b_m03_exact_inventory_v1`. The two Google BF16
sources share an exact 1,013-tensor contract; the external Unsloth reference has its own exact 47,478-tensor
producer-specific contract. The future gem16 compiled hybrid has a third, separate validator. A validated 26B
source remains non-executable.

All 30 source layers bind expert axis 0, fused Gate-before-Up and fused Down shapes, exact router tensors and the
five-local/one-global V-ownership schedule. Unsloth must expose expert IDs 0 through 127 in every layer and exact
FP8/NVFP4 companions. Every tensor reports one role and residency plus producer/scale/layout semantics. Unknown,
duplicate, MTP and compiled-profile vision tensors fail rather than being ignored. Source vision remains complete
but exactly 356 tensors/1,145,588,832 bytes are `compile_excluded_vision`.

The two explicit tied-head candidates produce conservative aligned compiled arenas of 14,696,667,648 bytes (Q4_0)
and 14,696,668,160 bytes (NVFP4). A direct CUDA reservation of the larger value, exact 32K FP8 K/V and 448 MiB of
named fixed regions leaves 818,741,248 bytes free, passing the 700 MiB preliminary gate. This is synthetic M03
admission only; M07 selects the head and M09 repeats with the real artifact/final arenas.

## M04 offline compiler boundary

M04 adds `tools/compile_gemma4_26b.py` and a standard-library-only `tools/gem16_compile/` package. The compiler
verifies every source-lock file before tensor access and again before publication, resolves a source-lock-bound plan,
requires every source tensor to map to one operation or exact exclusion, reads payloads through bounded read-only
mmap windows, and emits deterministic standard Safetensors plus schema-1 provenance. Output is written under a
sibling `.incomplete` directory, fsynced, strictly verified and atomically renamed without overwrite. Resume is
restart-only until a cryptographically bound partial-state contract is accepted.

The canonical M04 byte lane is Linux x86-64, little-endian, `C.UTF-8`, CPython 3.14.6 and one compiler thread. Two
clean synthetic runs at implementation commit `edd80cb6adae6d441924098870ceca9b4b1248d5` match all nine artifact
files; their compilation-manifest SHA-256 is
`640266a228a9c298b1ff2d3feb10e214baba202afd06cfb9d0f0a7798853e8d6`. A separate 2 MiB tensor stays below a
70,230,016-byte process cap with a maximum 4,376-byte mmap window. The only encoder is byte-identical `copy-v1`, and
all artifacts state `m04_scaffold_not_runtime_loadable`. M05-M07 still own FP8, NVFP4 and tied-head encoding; M08
owns the first direct-load artifact and derived lock.

## Runtime and product boundary

The 12B direct-load profile remains the default and is unchanged. The 26B path must use a separate model variant,
loader, arena plan and shape-specialized kernels selected once during initialization; it must not add per-layer or
per-token generic dispatch to the 12B hot path. M00 adds no feature flag because no incomplete 26B code exists.
Before M02 exposes any runtime path, it must add an experimental boundary that defaults off and cannot advertise a
usable model until the owning milestone gates pass.

The first 26B release explicitly excludes vision, audio, video, MTP, continuous batching, multiple 26B slots,
CPU expert offload, expert streaming, multi-GPU and arbitrary Gemma 4 MoE variants. Batch-one resident text chat,
greedy and existing sampling controls, 32K required context and 64K target context remain the intended product
surface after qualification.

## Memory and benchmark limits

- Initial immutable-weight target: at most 14,100 MiB.
- Hard review stop: above 14,300 MiB before kernel optimization.
- Required 32K CUDA-visible reserve: at least 700 MiB on the declared reference configuration.
- All per-token weights remain resident; CPU offload and expert streaming are diagnostic only.
- No recurring allocation, filesystem access, compilation or repack after plan creation.
- Performance promotion requires correctness and quality first, three warm-ups, ten retained measurements, raw
  samples, actual native dispatch/instruction evidence, peak VRAM and disclosed baseline differences.

The target profile is a hypothesis until M18/M19 quality gates, M20 performance qualification and M21 context
qualification pass.

## External implementation policy

Pinned `kekzl/imp@a392904d4216388828d0d56317de046f4ca49627` is reference evidence, not a runtime dependency or
architecture. Through M13, use is limited to citation, differential fixtures and clean-room reimplementation of
small documented contracts. No imp source is approved for copying by M00. A later isolated MIT kernel port requires
an owner-approved decision, exact file hashes, retained MIT text and headers, a modification record, independent
tests, 5080 evidence and proof that no second persistent weight layout is introduced. Paged KV, continuous
batching, broad dispatch, general executors and host expert offload are rejected for the first product.

## Milestone state

- M00: accepted at `3bf7b6e4427433ae766fe9238ed6d8c991398b6b`.
- M01: accepted on 2026-08-06 at handoff commits `f901044`, `9023f5a` and `e2c44d5`; all explicit exit criteria
  pass. Direct Unsloth token output is retained with disclosed non-exact logprobs and diagnostic CPU offload.
- M02: accepted on 2026-08-06 on `feat/26b-m02-model-traits` from accepted M01 closure `59996f5`; all explicit
  exit criteria pass.
- M03: accepted on 2026-08-11 at `06b72e4897a32afa15303ca461847049ac8bb98c`; strict
  source/external/compiled contracts, canonical inventories, mutation tests and direct 32K CUDA admission pass.
- M04: accepted by the project owner on 2026-08-11 at implementation commit
  `edd80cb6adae6d441924098870ceca9b4b1248d5`; 12 compiler tests, 95 Python tests, host/sanitizer/CUDA gates,
  clean reproducibility and 12B regression pass.
- M05: accepted by the project owner on 2026-08-11 at implementation commit
  `d91388113d68974f9ab7cec1a90ef768285c0645`. The versioned native C++20 batch encoder/comparator, exact
  115-matrix plans, clean full Ordinary and QAT runs, structural verification, complete hashes, weight-only
  comparison and regression gates pass. [M05 acceptance](evidence/gemma4_26b/m05-fp8-attention-compiler-acceptance-2026-08-11.md)
  is authoritative; earlier dirty runs remain diagnostic history. The M05 partial artifact is non-runtime-loadable.
  Standalone verification does not reconvert and records `transformation_recomputed=false` until M08's external lock.
  M06 is dependency-unblocked but not started; derived-artifact distribution approval remains separate.
- M05 contract: deterministic bounded host BF16-to-E4M3FN attention Q/K/V/O encoding with BF16 `[N,1]` row scales,
  reports, tests and comparisons only. The canonical representation is F8_E4M3 `[N,K]` plus BF16 `[N,1]`, using
  deterministic rowwise max-abs v1, round-to-nearest ties-even, finite saturation, NaN/Inf rejection and an all-zero
  scale of `1.0`. Runtime activation/accumulation changes, 26B execution and the
  first complete direct-load artifact remain downstream work.

Current acceptance and historical evidence:

- [M05 clean acceptance](evidence/gemma4_26b/m05-fp8-attention-compiler-acceptance-2026-08-11.md)
- [M05 kickoff and drift record](evidence/gemma4_26b/m05-kickoff-2026-08-11.md)
- [M05 native implementation and diagnostic runs](evidence/gemma4_26b/m05-native-fp8-implementation-and-diagnostic-runs-2026-08-11.md)
- M01 adds source locks, offline tooling and compact reference evidence only; no 26B compiler, runtime or CUDA path
  exists at the accepted M01 boundary.

Current evidence:

- [M00 baseline drift report](evidence/gemma4_26b/baseline-drift-2026-08-06.md)
- [M00 policy review checklist](evidence/gemma4_26b/m00-policy-review.md)
- [M01 source-lock and golden handoff](evidence/gemma4_26b/m01-source-locks-and-goldens-2026-08-06.md)
- [M01 Unsloth vLLM OOM incident](evidence/gemma4_26b/m01-unsloth-vllm-oom-2026-08-06.md)
- [M02 kickoff and drift record](evidence/gemma4_26b/m02-kickoff-2026-08-06.md)
- [M02 model-variant handoff](evidence/gemma4_26b/m02-model-variants-2026-08-06.md)
- [M03 kickoff and drift record](evidence/gemma4_26b/m03-kickoff-2026-08-11.md)
- [M03 implementation handoff](evidence/gemma4_26b/m03-manifest-and-inventory-2026-08-11.md)
- [M03 synthetic 32K admission](evidence/gemma4_26b/m03-synthetic-32k-admission.json)
- [M04 kickoff and drift record](evidence/gemma4_26b/m04-kickoff-2026-08-11.md)
- [M04 compiler-scaffold handoff](evidence/gemma4_26b/m04-checkpoint-compiler-scaffold-2026-08-11.md)
- [M04 clean reproducibility report](evidence/gemma4_26b/m04-reproducibility.json)
- [M04 bounded-memory report](evidence/gemma4_26b/m04-bounded-memory-report.json)

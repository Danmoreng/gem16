# Gemma 4 26B A4B experimental track

Status: M00, M01 and M02 accepted; M03 exact tensor inventory is unblocked but not started

Production hypothesis: `gem16-gemma4-26b-a4b-qat-hybrid-text`

The 26B track targets text-only Gemma 4 26B A4B inference on one approximately 16 GB NVIDIA Blackwell GPU while
preserving the statically specialized Gemma 4 12B Unified product path. The binding implementation order and gates
are in the [master plan](plans/gemma4-26b/00_MASTER_IMPLEMENTATION_PLAN.md). The immutable imported plan package,
including its initial status-board template, remains byte-identical; current milestone state and acceptance
evidence are recorded in this document, [ROADMAP.md](ROADMAP.md) and
[`docs/evidence/gemma4_26b/`](evidence/gemma4_26b/).

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
   checked arithmetic and atomic output publication.
3. A second clean reference-platform run must reproduce the compiled bytes exactly. If cross-platform byte identity
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
- M03: unblocked but not started; it must use a separate milestone branch from the accepted M02 closure.
- M04 and every later milestone remain blocked. Derived-artifact distribution approval remains separate.
- M01 adds source locks, offline tooling and compact reference evidence only; no 26B compiler, runtime or CUDA path
  exists at the accepted M01 boundary.

Current evidence:

- [M00 baseline drift report](evidence/gemma4_26b/baseline-drift-2026-08-06.md)
- [M00 policy review checklist](evidence/gemma4_26b/m00-policy-review.md)
- [M01 source-lock and golden handoff](evidence/gemma4_26b/m01-source-locks-and-goldens-2026-08-06.md)
- [M01 Unsloth vLLM OOM incident](evidence/gemma4_26b/m01-unsloth-vllm-oom-2026-08-06.md)
- [M02 kickoff and drift record](evidence/gemma4_26b/m02-kickoff-2026-08-06.md)
- [M02 model-variant handoff](evidence/gemma4_26b/m02-model-variants-2026-08-06.md)

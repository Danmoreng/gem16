# Gemma 4 26B M03 manifest and exact tensor inventory handoff

Date: 2026-08-11
Branch: `feat/gemma4-26b`
Implementation base: `4928f3fa6a6f81584cbc6bc78fb262b20ca617d5` plus this M03 working tree
Milestone: M03 — Manifest and exact tensor inventory
Status: implementation and exit evidence pass; owner acceptance pending

## Scope and drift

The kickoff review is retained in [m03-kickoff-2026-08-11.md](m03-kickoff-2026-08-11.md). The locked sources,
reference tools, 12B baseline, branch policy and milestone dependencies had no invalidating drift. The implementation
stays within M03: exact inspection, semantic mapping, strict validation and synthetic memory admission. It does not
compile a checkpoint, enable 26B execution, add an allocator plan or begin M04.

## Implemented contract

Manifest JSON advances to schema 3. Every 26B tensor record carries an exact semantic role and residency class,
logical dtype/shape, layer/expert IDs and axes where applicable, quantization component and producer, scale
dtype/vector/direction, and planned final GPU layout. Totals are emitted by role and residency. Config-derived
checkpoint-profile selection is independent of paths.

Three validators are intentionally separate:

1. `source_bf16`: exact QAT/ordinary BF16 source family;
2. `external_unsloth_nvfp4`: exact external llm-compressor reference family, always non-executable and never
   project-compiled;
3. `gem16_compiled_hybrid`: strict future artifact contract with complete producer/scale/layout semantics, tested
   synthetically but not selected by an upstream source checkpoint.

Unknown, missing and duplicate tensors fail. MTP fails for the first product. Source vision remains required for
source identity but every byte is assigned `compile_excluded_vision`; compiled-profile vision is forbidden. The
single `[262144,2816]` embedding is both embedding and output head, and a duplicate `lm_head` fails.

## Exact inventory results

Fresh raw inventory generation was byte-identical to all four checked M01 source inventories:

| Source | Tensors | Payload/file bytes | Regenerated JSON SHA-256 |
|---|---:|---:|---|
| Google QAT BF16 | 1,013 | 51,611,872,412 payload | `f033775e2bdb34ee74b735e9c26364fc1c56e6faa8fcc53436cc7a9db44ede36` |
| Google ordinary BF16 | 1,013 | 51,611,872,412 payload | `eb5d62e55f2df1a7543674bb81dd6861489696dc19d94eb2351942fb51c50bcd` |
| Unsloth NVFP4 | 47,478 | 16,903,408,612 payload | `740aeec34ddceddcfae83764f293b5a4d2635115a6eae4797935117901283bd5` |
| Official Google Q4_0 | 658 | 14,439,363,584 file | `de99cfb438902c057db45539dd8611e76f8c0ffa35f812cb393b5ba6e7c91bdf` |

QAT and ordinary BF16 are structurally identical under the source contract. They contain 657 text tensors and
50,466,283,580 text payload bytes. The exact excluded vision family is 356 tensors and 1,145,588,832 bytes. There
are zero unknown, MTP, audio or video tensors.

The source expert shapes and semantics are frozen as:

```text
experts.gate_up_proj  BF16 [128,1408,2816]  expert,gate_then_up,input
experts.down_proj     BF16 [128,2816,704]   expert,output,input
```

Axis 0 is expert-major and Gate precedes Up. The proof records the parameter/forward contracts from
`Gemma4TextExperts` in locked Transformers `a08ace4bbd97e721c98751deec37d87b026acadc` and an independently computed
small numerical split. All 30 layers have exact router normalization scale `[2816]`, router projection
`[128,2816]`, per-expert scale `[128]` and the two fused expert tensors. Local layers require V; global layers
`5,11,17,23,29` forbid it and reuse raw K before distinct K/V post-processing.

Unsloth expands each routed tensor into separate Gate/Up/Down value, local-scale, weight-scale and input-scale
records for all experts `0..127` in every layer. Routed values are U8 packed E2M1, local scales are E4M3 group-16,
and both global scale roles are F32 divisors. Attention values are E4M3 with BF16 per-output-channel scales. The
strict external verifier reports all 47,478 tensors and all 16,903,408,612 bytes reconciled with zero unknown roles.
It remains `runtime_supported=false` and `is_project_compiled_artifact=false`.

Canonical compact manifests, all 30 layer rows, role cross-map, scale semantics and both compiled candidates are in
`benchmarks/goldens/gemma4_26b/manifests/`. `tests/fixtures/gemma4_26b_inventory.json` freezes their hashes and exact
totals. Large immutable raw inventories remain under `benchmarks/goldens/gemma4_26b/source-inventories/` rather
than being duplicated.

## Preliminary compiled and memory contract

The compiled contract contains 1,282 tensors with a Q4_0 tied head or 1,285 with an NVFP4 tied head. It preserves
one physical head, FP8 attention plus BF16 channel scales, NVFP4 shared/routed MLP families, BF16 router/norm/scalar
controls and compiler-derived BF16 K/V scales. It excludes vision and MTP.

| Candidate | Payload bytes | Alignment padding | 256-byte-aligned immutable arena |
|---|---:|---:|---:|
| Q4_0 head | 14,696,569,188 | 98,460 | 14,696,667,648 |
| NVFP4 head | 14,696,569,196 | 98,964 | 14,696,668,160 |

Both pass the 14,100 MiB target and 14,300 MiB hard stop. The direct CUDA probe selected the larger candidate and
also allocated/touched exact 440,401,920-byte 32K FP8 K/V, 256 MiB MoE-prefill workspace, 128 MiB
activation/output, 32 MiB graph-private reserve and 32 MiB metadata guard. The reference RTX 5080 Laptop reported
16,652,042,240 CUDA-visible total bytes, 16,425,746,432 free after context and 818,741,248 directly free after all
regions. This passes the 734,003,200-byte (700 MiB) gate by 84,738,048 bytes; free memory returned exactly to the
post-context baseline after release. Full machine-readable allocation order and measurement are in
[m03-synthetic-32k-admission.json](m03-synthetic-32k-admission.json).

This is conservative synthetic admission, not final process peak or model execution. M07 still selects the head,
and M09 must repeat with the real artifact and allocator.

## Rejection and regression coverage

C++ and Python mutations reject missing/duplicate roles, a wrong source expert axis, wrong Gate/Up order metadata,
wrong router shape or dtype, missing local V, unexpected global V, an unknown/MTP family, incomplete Unsloth expert
sets, wrong FP8/NVFP4 scale dtype/vector/direction/producer, external-reference metadata under the compiled profile,
and any compiled vision tensor. Additional tests prove generated files are current and compact, every input hash
names a repository file, all layer rows are exact, and every manifest byte is assigned exactly once.

The production 12B path remains operational: exact real-checkpoint inspection reports schema 3,
`gemma4_unified_12b_exact_inventory`, 1,389 tensors and 9,304,786,336 payload bytes. Blackwell exact-blue validation
returns the unchanged `[9503,106]` token sequence with zero fallback and a 9,304,895,488-byte weight arena.

## Verification commands and results

All commands ran from the repository root on CUDA runtime/driver 13.3 (`13030`), RTX 5080 Laptop, compute
capability 12.0.

```text
python3 tools/generate_gemma4_26b_manifests.py --check
  pass; all canonical outputs current

python3 -m unittest discover -s tests/python -p 'test_*.py' -v
  83/83 passed

cmake --preset host-debug
cmake --build --preset host-debug -j2
ctest --preset host-debug --output-on-failure
  1/1 passed

cmake --preset host-sanitize
cmake --build --preset host-sanitize -j2
ctest --preset host-sanitize --output-on-failure
  1/1 passed (ASan/UBSan; 13.83 s)

cmake --preset blackwell-release
cmake --build --preset blackwell-release -j2
ctest --preset blackwell-release --output-on-failure
  2/2 passed (unit 0.46 s, CUDA 31.39 s)

gem16-inspect --model <QAT BF16> --validate --json <manifest>
python3 tools/verify_gemma4_26b_manifest.py --profile source_bf16 ...
  pass: 1,013 tensors

gem16-inspect --model <ordinary BF16> --validate --json <manifest>
python3 tools/verify_gemma4_26b_manifest.py --profile source_bf16 ...
  pass: 1,013 tensors

gem16-inspect --model <Unsloth NVFP4> --validate --json <manifest>
python3 tools/verify_gemma4_26b_manifest.py --profile external_unsloth_nvfp4 ...
  pass: 47,478 tensors

python3 tools/fetch_model.py --lock <each of the four M01 source locks> --verify-only
  pass: every locked file size and SHA-256 verified

build/Linux/blackwell-release/bin/gem16-26b-memory-probe \
  --output docs/evidence/gemma4_26b/m03-synthetic-32k-admission.json \
  --code-revision 4928f3fa6a6f81584cbc6bc78fb262b20ca617d5+m03-worktree
  pass: 818,741,248 >= 734,003,200 direct free bytes

python3 tools/validate_inference.py \
  --run build/Linux/blackwell-release/bin/gem16-run \
  --model models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497 \
  --output /tmp/m03-12b-validation.json
  pass: exact-blue output [9503,106], zero fallback
```

NVCC emitted the repository's existing ignored-`nodiscard` warnings while rebuilding unchanged CUDA translation
units. There were no new compiler errors, sanitizer reports, test failures, hidden fallback or VRAM allocation
failure.

## Exit criteria

- [x] Canonical tensor inventory exists for every required source.
- [x] Tensor roles, residencies, layouts, aliases and scale metadata are explicit.
- [x] Source expert layout is proven and tested.
- [x] Unsloth scale tensors and semantics are completely mapped without calling it project-compiled.
- [x] Compiled-role validator is strict and independent.
- [x] Vision is omitted exactly; unknown and MTP roles fail.
- [x] Q4_0 and NVFP4 tied-head candidates are explicit and remain an M07 choice.
- [x] Synthetic 32K direct-CUDA admission retains at least 700 MiB.
- [x] Locked-source regeneration, host, sanitizer, CUDA and 12B regression gates pass.
- [x] Architecture, checkpoint, correctness, memory, roadmap and decision documentation agree.
- [ ] Project-owner acceptance and milestone commit hash are pending.

M04 and all later milestones remain blocked until owner acceptance is recorded.

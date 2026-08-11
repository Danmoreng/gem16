# Documentation index and reading order

## First use

Project owner:

1. [`README_DE.md`](README_DE.md)
2. [`OWNER_DECISION_POINTS_DE.md`](OWNER_DECISION_POINTS_DE.md)
3. [`00_MASTER_IMPLEMENTATION_PLAN.md`](00_MASTER_IMPLEMENTATION_PLAN.md)

Coding agent:

1. [`START_HERE_CODEX.md`](START_HERE_CODEX.md)
2. repository root `AGENTS.md`
3. [`02_AGENT_OPERATING_CONTRACT.md`](02_AGENT_OPERATING_CONTRACT.md)
4. the single approved milestone
5. only the specifications/checklists referenced by that milestone

## Control documents

| File | Purpose |
|---|---|
| [`README_DE.md`](README_DE.md) | gem16: Implementierungsplan für Gemma 4 26B A4B auf 16-GB-Blackwell-GPUs |
| [`README.md`](README.md) | gem16 Gemma 4 26B A4B implementation plan |
| [`START_HERE_CODEX.md`](START_HERE_CODEX.md) | Start here: first Codex task |
| [`OWNER_DECISION_POINTS_DE.md`](OWNER_DECISION_POINTS_DE.md) | Entscheidungspunkte für den Projektinhaber |
| [`MILESTONE_STATUS_BOARD.md`](MILESTONE_STATUS_BOARD.md) | Milestone status board |
| [`00_MASTER_IMPLEMENTATION_PLAN.md`](00_MASTER_IMPLEMENTATION_PLAN.md) | Master implementation plan |
| [`01_EXECUTIVE_ARCHITECTURE.md`](01_EXECUTIVE_ARCHITECTURE.md) | Executive architecture |
| [`02_AGENT_OPERATING_CONTRACT.md`](02_AGENT_OPERATING_CONTRACT.md) | Coding-agent operating contract |
| [`03_CURRENT_REPOSITORY_BASELINE.md`](03_CURRENT_REPOSITORY_BASELINE.md) | Current repository baseline |
| [`04_TARGET_PRODUCT_PROFILE.md`](04_TARGET_PRODUCT_PROFILE.md) | Target product profile |
| [`05_DECISIONS_AND_NON_GOALS.md`](05_DECISIONS_AND_NON_GOALS.md) | Decisions, open choices and non-goals |
| [`06_DEPENDENCY_GRAPH.md`](06_DEPENDENCY_GRAPH.md) | Milestone dependency graph |
| [`07_MEMORY_BUDGET_AND_RESIDENCY.md`](07_MEMORY_BUDGET_AND_RESIDENCY.md) | Memory budget and residency |
| [`08_RISK_REGISTER.md`](08_RISK_REGISTER.md) | Risk register |
| [`09_DEFINITION_OF_DONE.md`](09_DEFINITION_OF_DONE.md) | Program definition of done |
| [`10_SOURCE_SNAPSHOT.md`](10_SOURCE_SNAPSHOT.md) | Source snapshot used by this plan |
| [`11_PR_SEQUENCE.md`](11_PR_SEQUENCE.md) | Single-branch milestone and review sequence |
| [`12_AGENT_TASK_QUEUE.md`](12_AGENT_TASK_QUEUE.md) | Agent task queue |

## Milestones

Required sequence: `M00` through `M23`. `M24` and `M25` are optional future tracks.

| File | Title |
|---|---|
| [`milestones/M00_POLICY_AND_TRACK_BOOTSTRAP.md`](milestones/M00_POLICY_AND_TRACK_BOOTSTRAP.md) | M00 — Policy and 26B track bootstrap |
| [`milestones/M01_SOURCE_LOCKS_AND_GOLDENS.md`](milestones/M01_SOURCE_LOCKS_AND_GOLDENS.md) | M01 — Immutable source locks and golden evidence |
| [`milestones/M02_MODEL_CONFIG_VARIANTS.md`](milestones/M02_MODEL_CONFIG_VARIANTS.md) | M02 — Model configuration and static variant traits |
| [`milestones/M03_MANIFEST_AND_TENSOR_INVENTORY.md`](milestones/M03_MANIFEST_AND_TENSOR_INVENTORY.md) | M03 — 26B tensor manifest and exact inventory |
| [`milestones/M04_CHECKPOINT_COMPILER_SCAFFOLD.md`](milestones/M04_CHECKPOINT_COMPILER_SCAFFOLD.md) | M04 — Deterministic checkpoint compiler scaffold |
| [`milestones/M05_FP8_ATTENTION_COMPILER.md`](milestones/M05_FP8_ATTENTION_COMPILER.md) | M05 — Deterministic FP8 attention compiler |
| [`milestones/M06_NVFP4_EXPERT_COMPILER.md`](milestones/M06_NVFP4_EXPERT_COMPILER.md) | M06 — Deterministic NVFP4 compiler for shared and routed experts |
| [`milestones/M07_EMBEDDING_HEAD_FORMAT_EXPERIMENT.md`](milestones/M07_EMBEDDING_HEAD_FORMAT_EXPERIMENT.md) | M07 — Tied embedding and output-head format experiment |
| [`milestones/M08_DERIVED_ARTIFACT_AND_LOADER.md`](milestones/M08_DERIVED_ARTIFACT_AND_LOADER.md) | M08 — Derived checkpoint artifact, schema and direct loader |
| [`milestones/M09_MEMORY_PLANNER_AND_RESIDENCY.md`](milestones/M09_MEMORY_PLANNER_AND_RESIDENCY.md) | M09 — 26B memory planner and text-only residency |
| [`milestones/M10_CPU_MOE_ORACLE.md`](milestones/M10_CPU_MOE_ORACLE.md) | M10 — CPU oracle for Gemma 4 26B MoE semantics |
| [`milestones/M11_CUDA_MOE_REFERENCE.md`](milestones/M11_CUDA_MOE_REFERENCE.md) | M11 — CUDA correctness-first MoE path |
| [`milestones/M12_ATTENTION_AND_KV_INTEGRATION.md`](milestones/M12_ATTENTION_AND_KV_INTEGRATION.md) | M12 — 26B attention, RoPE, KV sharing and cache integration |
| [`milestones/M13_FULL_MODEL_REFERENCE_PATH.md`](milestones/M13_FULL_MODEL_REFERENCE_PATH.md) | M13 — Complete slow 26B text inference path |
| [`milestones/M14_NATIVE_MOE_DECODE.md`](milestones/M14_NATIVE_MOE_DECODE.md) | M14 — Native SM120 batch-one MoE decode |
| [`milestones/M15_GROUPED_MOE_PREFILL.md`](milestones/M15_GROUPED_MOE_PREFILL.md) | M15 — Grouped and bounded-workspace MoE prefill |
| [`milestones/M16_QUANTIZED_EMBEDDING_AND_HEAD.md`](milestones/M16_QUANTIZED_EMBEDDING_AND_HEAD.md) | M16 — Production quantized embedding and output head |
| [`milestones/M17_OPTIMIZED_RUNTIME_INTEGRATION.md`](milestones/M17_OPTIMIZED_RUNTIME_INTEGRATION.md) | M17 — Optimized whole-model integration and CUDA Graph decode |
| [`milestones/M18_CONVERTER_AB_STUDY.md`](milestones/M18_CONVERTER_AB_STUDY.md) | M18 — Converter A/B and causal attribution study |
| [`milestones/M19_QUALITY_QUALIFICATION.md`](milestones/M19_QUALITY_QUALIFICATION.md) | M19 — Held-out model-quality qualification and final format selection |
| [`milestones/M20_PERFORMANCE_QUALIFICATION.md`](milestones/M20_PERFORMANCE_QUALIFICATION.md) | M20 — Controlled performance qualification |
| [`milestones/M21_LONG_CONTEXT_QUALIFICATION.md`](milestones/M21_LONG_CONTEXT_QUALIFICATION.md) | M21 — 32K required and 64K target context qualification |
| [`milestones/M22_PRODUCT_INTEGRATION.md`](milestones/M22_PRODUCT_INTEGRATION.md) | M22 — CLI, server and Studio product integration |
| [`milestones/M23_RELEASE_QUALIFICATION.md`](milestones/M23_RELEASE_QUALIFICATION.md) | M23 — Release qualification, evidence freeze and rollback |
| [`milestones/M24_OPTIONAL_Q4_0_BACKEND.md`](milestones/M24_OPTIONAL_Q4_0_BACKEND.md) | M24 — Optional full-model Q4_0 reference backend |
| [`milestones/M25_FUTURE_MTP_AND_VISION.md`](milestones/M25_FUTURE_MTP_AND_VISION.md) | M25 — Future MTP and on-demand vision tracks |
| [`milestones/README.md`](milestones/README.md) | Milestones |

## Technical specifications

| File | Title |
|---|---|
| [`specs/API_CLI_CHANGES.md`](specs/API_CLI_CHANGES.md) | API, CLI and product-surface change specification |
| [`specs/ATTENTION_KV_SPEC.md`](specs/ATTENTION_KV_SPEC.md) | Gemma 4 26B attention and KV specification |
| [`specs/BENCHMARK_MATRIX.md`](specs/BENCHMARK_MATRIX.md) | Gemma 4 26B benchmark matrix |
| [`specs/CHECKPOINT_COMPILER_SPEC.md`](specs/CHECKPOINT_COMPILER_SPEC.md) | Deterministic checkpoint compiler specification |
| [`specs/CHECKPOINT_PROVENANCE_SPEC.md`](specs/CHECKPOINT_PROVENANCE_SPEC.md) | Checkpoint provenance and immutability specification |
| [`specs/CODEX_WORKFLOW.md`](specs/CODEX_WORKFLOW.md) | Codex workflow for this program |
| [`specs/COMPILER_CLI_SPEC.md`](specs/COMPILER_CLI_SPEC.md) | Checkpoint compiler CLI specification |
| [`specs/DERIVED_CHECKPOINT_SCHEMA.md`](specs/DERIVED_CHECKPOINT_SCHEMA.md) | Derived Gemma 4 26B checkpoint schema |
| [`specs/EMBEDDING_HEAD_SPEC.md`](specs/EMBEDDING_HEAD_SPEC.md) | Quantized tied embedding and output-head specification |
| [`specs/ERROR_AND_CAPABILITY_REPORTING.md`](specs/ERROR_AND_CAPABILITY_REPORTING.md) | Error and capability reporting specification |
| [`specs/FILE_CHANGE_MAP.md`](specs/FILE_CHANGE_MAP.md) | Repository file change map |
| [`specs/FP8_QUANTIZATION_SPEC.md`](specs/FP8_QUANTIZATION_SPEC.md) | FP8 attention quantization specification |
| [`specs/MEMORY_ARENA_SPEC.md`](specs/MEMORY_ARENA_SPEC.md) | Memory arenas and residency specification |
| [`specs/MODEL_VARIANT_TRAITS_SPEC.md`](specs/MODEL_VARIANT_TRAITS_SPEC.md) | Model variant traits and static dispatch specification |
| [`specs/MOE_DECODE_KERNEL_SPEC.md`](specs/MOE_DECODE_KERNEL_SPEC.md) | Native MoE decode kernel specification |
| [`specs/MOE_PREFILL_KERNEL_SPEC.md`](specs/MOE_PREFILL_KERNEL_SPEC.md) | Grouped MoE prefill kernel specification |
| [`specs/MOE_ROUTER_SPEC.md`](specs/MOE_ROUTER_SPEC.md) | MoE router specification |
| [`specs/MOE_SEMANTICS_SPEC.md`](specs/MOE_SEMANTICS_SPEC.md) | Gemma 4 26B MoE semantic specification |
| [`specs/NVFP4_QUANTIZATION_SPEC.md`](specs/NVFP4_QUANTIZATION_SPEC.md) | NVFP4 quantization specification |
| [`specs/Q4_0_SPEC.md`](specs/Q4_0_SPEC.md) | Q4_0 format and reference specification |
| [`specs/QUALITY_EVALUATION_SPEC.md`](specs/QUALITY_EVALUATION_SPEC.md) | Model quality evaluation specification |
| [`specs/README.md`](specs/README.md) | Specs |
| [`specs/REFERENCE_RUNTIME_SPEC.md`](specs/REFERENCE_RUNTIME_SPEC.md) | Trusted reference runtime and golden-capture specification |
| [`specs/SESSION_OWNERSHIP_AND_CONCURRENCY.md`](specs/SESSION_OWNERSHIP_AND_CONCURRENCY.md) | Session ownership and concurrency specification |
| [`specs/TELEMETRY_ARTIFACT_SPEC.md`](specs/TELEMETRY_ARTIFACT_SPEC.md) | Telemetry and evidence artifact specification |
| [`specs/TENSOR_NAMING_DISCOVERY.md`](specs/TENSOR_NAMING_DISCOVERY.md) | Tensor naming and inventory discovery protocol |
| [`specs/TEST_MATRIX.md`](specs/TEST_MATRIX.md) | Test matrix |

## Checklists

| File | Title |
|---|---|
| [`checklists/BEFORE_EACH_MILESTONE.md`](checklists/BEFORE_EACH_MILESTONE.md) | Before each milestone |
| [`checklists/BENCHMARK_RUN_CHECKLIST.md`](checklists/BENCHMARK_RUN_CHECKLIST.md) | Benchmark run checklist |
| [`checklists/CUDA_KERNEL_ACCEPTANCE_CHECKLIST.md`](checklists/CUDA_KERNEL_ACCEPTANCE_CHECKLIST.md) | CUDA kernel acceptance checklist |
| [`checklists/FINAL_RELEASE_CHECKLIST.md`](checklists/FINAL_RELEASE_CHECKLIST.md) | Final release checklist |
| [`checklists/MEMORY_GATE_CHECKLIST.md`](checklists/MEMORY_GATE_CHECKLIST.md) | Memory gate checklist |
| [`checklists/PR_REVIEW_CHECKLIST.md`](checklists/PR_REVIEW_CHECKLIST.md) | Pull-request review checklist |
| [`checklists/QUALITY_GATE_CHECKLIST.md`](checklists/QUALITY_GATE_CHECKLIST.md) | Quality gate checklist |
| [`checklists/QUANTIZER_ACCEPTANCE_CHECKLIST.md`](checklists/QUANTIZER_ACCEPTANCE_CHECKLIST.md) | Quantizer acceptance checklist |
| [`checklists/README.md`](checklists/README.md) | Checklists |
| [`checklists/SECURITY_AND_SUPPLY_CHAIN_CHECKLIST.md`](checklists/SECURITY_AND_SUPPLY_CHAIN_CHECKLIST.md) | Security and supply-chain checklist |

## Templates

| File | Title |
|---|---|
| [`templates/BENCHMARK_REPORT.md`](templates/BENCHMARK_REPORT.md) | Gemma 4 26B performance report |
| [`templates/CHECKPOINT_LOCK.md`](templates/CHECKPOINT_LOCK.md) | Checkpoint lock schema template |
| [`templates/CODEX_MILESTONE_PROMPT.md`](templates/CODEX_MILESTONE_PROMPT.md) | Codex milestone prompt template |
| [`templates/COMPILED_MANIFEST_EXAMPLE.md`](templates/COMPILED_MANIFEST_EXAMPLE.md) | Example `gem16_compilation.json` |
| [`templates/DECISION_RECORD.md`](templates/DECISION_RECORD.md) | Decision: <title> |
| [`templates/DRIFT_REPORT.md`](templates/DRIFT_REPORT.md) | Repository drift report |
| [`templates/EXPERIMENT_RECORD.md`](templates/EXPERIMENT_RECORD.md) | Experiment: <title> |
| [`templates/PERFORMANCE_LEDGER_ENTRY.md`](templates/PERFORMANCE_LEDGER_ENTRY.md) | Performance ledger entry template |
| [`templates/PR_DESCRIPTION.md`](templates/PR_DESCRIPTION.md) | <MILESTONE>: <title> |
| [`templates/QUALITY_REPORT.md`](templates/QUALITY_REPORT.md) | Gemma 4 26B quality report |
| [`templates/README.md`](templates/README.md) | Templates |
| [`templates/RELEASE_SIGNOFF.md`](templates/RELEASE_SIGNOFF.md) | Gemma 4 26B release sign-off |

## Appendices

| File | Title |
|---|---|
| [`appendices/ACCEPTANCE_THRESHOLD_RATIONALE.md`](appendices/ACCEPTANCE_THRESHOLD_RATIONALE.md) | Acceptance-threshold rationale |
| [`appendices/CODEX_HANDOFF_MAP.md`](appendices/CODEX_HANDOFF_MAP.md) | Codex handoff map |
| [`appendices/COMMAND_CATALOG.md`](appendices/COMMAND_CATALOG.md) | Command catalog |
| [`appendices/DECISION_LOG_STARTER.md`](appendices/DECISION_LOG_STARTER.md) | Proposed decision-log entries |
| [`appendices/FAILURE_MODES.md`](appendices/FAILURE_MODES.md) | Failure modes and required responses |
| [`appendices/GLOSSARY.md`](appendices/GLOSSARY.md) | Glossary |
| [`appendices/MEMORY_CALCULATIONS.md`](appendices/MEMORY_CALCULATIONS.md) | Memory calculations |
| [`appendices/PROPOSED_DIRECTORY_TREE.md`](appendices/PROPOSED_DIRECTORY_TREE.md) | Proposed repository directory tree |
| [`appendices/QUANTIZATION_COMPARISON.md`](appendices/QUANTIZATION_COMPARISON.md) | Q4_0, NVFP4 and FP8 comparison |
| [`appendices/README.md`](appendices/README.md) | Appendices |
| [`appendices/REFERENCE_VARIANT_MATRIX.md`](appendices/REFERENCE_VARIANT_MATRIX.md) | Reference and candidate variant matrix |
| [`appendices/REPOSITORY_TOUCHPOINTS.md`](appendices/REPOSITORY_TOUCHPOINTS.md) | Current repository touchpoints |
| [`appendices/SOURCE_LINKS.md`](appendices/SOURCE_LINKS.md) | Source and evidence links |

## Conflict precedence

1. repository root `AGENTS.md`;
2. accepted repository `docs/DECISIONS.md`;
3. [`02_AGENT_OPERATING_CONTRACT.md`](02_AGENT_OPERATING_CONTRACT.md);
4. the current milestone;
5. supporting specifications;
6. appendices and templates.

M00 exists because the proposed QAT-derived 26B artifact requires an explicit policy decision before implementation.

## imp reference integration

| File | Purpose |
|---|---|
| [`13_IMP_REFERENCE_INTEGRATION.md`](13_IMP_REFERENCE_INTEGRATION.md) | Binding plan amendment and milestone map |
| [`references/imp/README.md`](references/imp/README.md) | imp reference directory index |
| [`references/imp/IMP_REFERENCE_ASSESSMENT.md`](references/imp/IMP_REFERENCE_ASSESSMENT.md) | What imp changes and does not change |
| [`references/imp/IMP_SOURCE_MAP.md`](references/imp/IMP_SOURCE_MAP.md) | Pinned source files and intended use |
| [`references/imp/IMP_CODE_ADOPTION_MATRIX.md`](references/imp/IMP_CODE_ADOPTION_MATRIX.md) | Reference/clean-room/MIT-port/reject decisions |
| [`references/imp/IMP_LICENSE_AND_PROVENANCE.md`](references/imp/IMP_LICENSE_AND_PROVENANCE.md) | Third-party code handling |
| [`references/imp/IMP_SEMANTIC_GOLDENS.md`](references/imp/IMP_SEMANTIC_GOLDENS.md) | Router/branch/scale golden requirements |
| [`references/imp/IMP_QUALITY_AND_BENCHMARK_GATES.md`](references/imp/IMP_QUALITY_AND_BENCHMARK_GATES.md) | Quality and benchmark amendments |
| [`references/imp/IMP_KERNEL_STUDY_PROTOCOL.md`](references/imp/IMP_KERNEL_STUDY_PROTOCOL.md) | Optional grouped-kernel evaluation |
| [`references/imp/IMP_AGENT_TASK.md`](references/imp/IMP_AGENT_TASK.md) | Immediate Codex reference-audit task |
| [`specs/REFERENCE_IMPLEMENTATION_ADOPTION_SPEC.md`](specs/REFERENCE_IMPLEMENTATION_ADOPTION_SPEC.md) | Normative external-reference policy |
| [`specs/CUDA_STATE_LIFECYCLE_SPEC.md`](specs/CUDA_STATE_LIFECYCLE_SPEC.md) | Normative CUDA teardown/relaunch contract |
| [`appendices/IMP_PLAN_DELTA.md`](appendices/IMP_PLAN_DELTA.md) | Concise delta from v1 |

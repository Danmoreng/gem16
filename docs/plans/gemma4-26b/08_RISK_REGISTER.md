# Risk register

## Scoring

- Probability: Low / Medium / High
- Impact: Low / Medium / Critical
- Status: Open / Mitigating / Closed

| ID | Risk | Probability | Impact | Primary mitigation | Stop condition |
|---|---|---|---|---|---|
| R01 | QAT BF16 is robust to Q4_0 but not to NVFP4 W4A4 activations | High | Critical | Ordinary-BF16 control, QAT candidate, official Q4_0 reference; operator and teacher-forced comparison | Candidate exceeds accepted quality envelope |
| R02 | Project NVFP4 quantizer is worse than Unsloth's recipe | Medium | Critical | Reproduce ordinary-BF16 conversion; compare dequantized tensors, operator outputs and model quality | B is materially worse than A before QAT is considered |
| R03 | Derived-checkpoint policy conflicts with repository AGENTS rules | High | Critical | M00 decision and explicit scoped exception | No accepted decision |
| R04 | Final resident bytes exceed the 15,881 MiB CUDA-visible budget | High | Critical | M03 synthetic 32K admission before compiler work; byte-exact inventory; head experiment; omit vision | Preliminary admission cannot retain 700 MiB, or immutable arena >14,300 MiB |
| R05 | MoE prefill workspace causes OOM | High | Critical | Early conservative arena probe, bounded chunks, compact assignments, grouped expert execution, one shared workspace | 32K cannot retain 700 MiB directly measured margin |
| R06 | Router numerical drift changes expert selection and quality | High | Critical | BF16 router, FP32 softmax, exact norm/scale order, top-8 telemetry | Unexplained selection collapse or quality regression |
| R07 | Top-k tie behavior differs across CUDA and reference | Medium | Medium | Explicit lowest/exact index tie policy, adversarial fixtures | Nondeterministic expert IDs |
| R08 | Global attention K=V is implemented as shared physical cache | Medium | Critical | Preserve separate normalized/rotated K and normalized V states | Cache alias detected |
| R09 | Q4_0 output head is slower than NVFP4 or hurts graph shape | Medium | Medium | Separate profile benchmark; no commitment before M07/M16 | No win on quality/performance Pareto |
| R10 | NVFP4 head harms logits more than experts | Medium | Critical | Official-Q4_0-style head candidate and teacher-forced logit tests | Head dominates KL/NLL regression |
| R11 | 12B path regresses due to generic refactor | Medium | Critical | Static model traits, separate translation units, exact 12B gates in every shared change set | 12B output or benchmark changes unexpectedly |
| R12 | Unsloth tensor schema changes at `main` | High | Medium | Full commit lock and local manifest snapshot | Only mutable revision available |
| R13 | Google QAT source files change | Medium | Critical | Full revision/file hashes and compiler provenance | Source cannot be reproduced |
| R14 | Cross-platform compiler produces different bytes | Medium | Medium | Canonical reference compiler container/toolchain | No canonical hash can be generated |
| R15 | Load-time retile temporarily duplicates weights on GPU | Medium | Critical | Stream host source into final arena; inspect allocations | Peak device copy exceeds budget |
| R16 | Native SM120 kernel compiles but does not dispatch | Medium | Critical | SASS/disassembly plus Nsight runtime capture | Selected run lacks native instruction |
| R17 | Batch-one decode underutilizes Tensor Cores and gains little over Q4_0 | Medium | Medium | Custom T=1 expert schedule and end-to-end measurement | Native path fails release speed gate |
| R18 | Grouped prefill is fast only on synthetic balanced routing | High | Medium | Real prompts, expert-skew telemetry, worst-case synthetic suite | Unbounded tail or severe real-workload regression |
| R19 | Evaluation corpus leaks into calibration/tuning | Medium | Critical | Locked disjoint manifests and no final-set tuning | Data overlap detected |
| R20 | Official Q4_0 baseline timing boundaries differ | High | Medium | Separate core and protocol metrics, document format/timing | Exact-parity wording attempted |
| R21 | One checkpoint contains duplicate tied head | Medium | Critical | Manifest alias validation and byte accounting | Duplicate 396 MiB/1.4 GiB allocation |
| R22 | Expert tensor names/fusion differ from assumptions | High | Medium | M03 discovery; no kernel binding before inventory lock | Inferred naming used without evidence |
| R23 | Server admission allows multiple oversized slots | Medium | Critical | Variant-aware slot probe and explicit single-slot cap | Startup can overcommit |
| R24 | Full logits diagnostics accidentally enter benchmark path | Low | Medium | Separate diagnostic option and invalid-timing flag | Full-logit capture during promoted run |
| R25 | Performance tuning changes arithmetic order without quality rerun | High | Critical | Every promotion repeats operator/model gates | Missing fresh quality evidence |
| R26 | Thermal/power drift creates false wins | High | Medium | Adjacent A/B, fixed power profile, telemetry, confidence intervals | Non-adjacent headline only |
| R27 | Model license/provenance is mishandled in published artifact | Low | Critical | Preserve license, attribution and source metadata; legal review before distribution | Distribution terms unresolved |
| R28 | Host RAM requirement makes compiler impractical | Medium | Medium | Memory-map shards, bounded blocks, no whole-model load | Compiler exceeds declared memory cap |
| R29 | 64K target consumes all safety margin | High | Medium | Keep 32K release profile; label 64K experimental if needed | Margin <500 MiB |
| R30 | MTP is added prematurely and hides base regressions | Medium | Critical | M25 only after M23 | MTP code enters base qualification |

## Risk review cadence

- Review this table at every milestone exit.
- Add new risks immediately.
- Do not close a risk with prose alone; link evidence.
- Critical risks must have an owner and a measurable stop condition in the repository issue/plan.

## imp-derived additional risks

| ID | Risk | Probability | Impact | Primary mitigation | Stop condition |
|---|---|---|---|---|---|
| R31 | ModelOpt and llm-compressor global-scale directions are confused | Medium | Critical | Producer-tagged manifest and deliberately non-unit scale fixtures | Any tensor lacks unambiguous producer/scale contract |
| R32 | NVFP4 checkpoint/recipe causes large expert-quality loss despite correct kernels | High | Critical | Candidate G negative control; W4A16 diagnostic; prose-first PPL and per-layer drift | Loss cannot be isolated or exceeds frozen gate |
| R33 | A broad imp architecture is copied into gem16 and erodes specialization/memory guarantees | Medium | Critical | Adoption matrix; reject Paged KV/general executor; milestone scope gate | General dispatcher or second runtime ownership model enters critical path |
| R34 | Optional MIT code port loses attribution or imports transitive third-party code | Medium | Critical | License/provenance checklist and isolated destination | License/source chain unresolved |
| R35 | Module-static CUDA state retains pointers into destroyed arenas | Medium | Critical | Lifecycle spec; repeated destroy/recreate and failure-retry tests | Sticky CUDA error, stale pointer or VRAM leak after teardown |
| R36 | Technical-Markdown PPL is mistaken for a stable Gemma 4 quality signal | High | Medium | Plain-prose primary corpus; report domain strata separately | Promotion relies on one OOD aggregate |
| R37 | Actual runtime silently chooses a fallback different from the planned path | Medium | Critical | Record winning dispatch at branch site and retain graph demotion reason | Promoted run has unset/ambiguous actual path |
| R38 | 5090 results create an unrealistic 5080 performance target | High | Medium | Treat imp as external context; all release gates run locally on 5080 | Cross-machine number used as local acceptance criterion |

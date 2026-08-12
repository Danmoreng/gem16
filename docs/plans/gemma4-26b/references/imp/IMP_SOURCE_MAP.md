# Pinned imp source map

All paths below refer to:

```text
kekzl/imp@a392904d4216388828d0d56317de046f4ca49627
```

The coding agent must verify file hashes before relying on this map.

| imp path | Use in gem16 plan | Adoption mode |
|---|---|---|
| `LICENSE` | Confirm MIT terms and copyright notice | Copy unchanged into provenance bundle if code is imported |
| `README.md` | Scope, hardware target, supported formats and benchmark caveats | Reference only |
| `src/model/model_config.h` | ModelOpt-vs-llm-compressor scale semantics; Gemma4 config fields | Reimplement contract and tests; do not import broad config object |
| `src/model/llm_compressor_loader.cpp` | Name translation, vision/MTP skip rules, recipe/group-size discovery | Reference or narrow clean-room adapter |
| `tests/test_llm_compressor_loader.cpp` | Host fixtures for format recognition and failure modes | Recreate minimal equivalent fixtures |
| `src/exec/executor_forward_moe.cu` | Router, per-expert scale, shared/expert branch and residual failure modes | Semantic reference only; do not port the general executor |
| `src/exec/moe_ffn_context.h` | Inventory of mutable MoE state used by a broad executor | Reference only; gem16 uses a fixed plan |
| `src/compute/moe_routing.*` | Top-k routing implementation details | Differential reference; local oracle remains normative |
| `src/compute/gemm_grouped_nvfp4_smallM.{h,cu}` | Persistent grouped small-M schedule and native MMA implementation | Optional isolated MIT port or clean-room reimplementation after M13 passes and the local correctness fixtures are frozen |
| `src/compute/quantize_fp16_nvfp4_moe_native.*` | Activation packing and scale layout ideas | Compare against gem16 quantizer contract |
| `src/exec/executor_forward_moe_cutlass.cu` | Prefill tiering, device-args and grouped path | Algorithm/reference only |
| `src/exec/moe_prefill_decision.h` | Pure dispatch decision model | Reimplement the principle, not the full ladder |
| `src/compute/dispatch_paths.h` | Shared vocabulary for resolved paths | Reimplement a smaller gem16 enum set |
| `src/compute/dispatch_record.h` | Record the path that actually won in production dispatch | Reimplement pattern in gem16 |
| `src/runtime/graph_eligibility.{h,cpp}` | Retained first graph-demotion reason | Reimplement pattern in gem16 |
| `tests/perf_baseline.json` | Machine-readable regression gates | Adopt schema pattern with gem16 metrics |
| `docs/BENCHMARKS.md` | Methodology and non-comparability caveats | Reference only |
| `docs/audit/ppl_parity_2026_07_12.md` | Quality attribution and corpus lessons | Reference for M18/M19 design |
| `docs/audit/SETTLED.md` | Settled-evidence ledger and anchored priors | Adopt process pattern |
| `docs/sm120_optimal_kernel.md` | Profiling discipline and consumer-Blackwell constraints | Reference; do not import conclusions without 5080 measurement |

## Source collection command

The M01 evidence bundle should retain exact copies or hashes of the selected files:

```bash
git clone https://github.com/kekzl/imp.git third_party_sources/imp
git -C third_party_sources/imp checkout a392904d4216388828d0d56317de046f4ca49627
git -C third_party_sources/imp submodule update --init --recursive
```

Do not vendor the whole repository into gem16. Keep the source snapshot outside production build inputs unless a later approved MIT port requires specific files.

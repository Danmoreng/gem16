# Concise plan delta after reviewing imp

## Unchanged

- QAT BF16 remains the preferred master checkpoint.
- Experts/shared MLP remain NVFP4 candidates for native Blackwell speed.
- Attention projections remain FP8.
- Router/norms remain high precision.
- Vision remains outside the 26B program. MTP is now the required final target in M25; only its runtime integration remains deferred until the M23 base target is frozen.
- One final weight layout and batch-one fixed plans remain core architecture rules.

## Added

- pinned imp reference lane;
- NVIDIA/ModelOpt negative/control candidate;
- producer-specific NVFP4 scale tests;
- earlier FP32-router/residual goldens;
- prose-first quality qualification;
- actual-path dispatch and graph-demotion telemetry;
- machine-readable perf/VRAM regression gate;
- settled-evidence ledger;
- MIT code provenance and optional isolated kernel-port path;
- repeated engine lifecycle tests.

## Explicitly rejected

- general executor port;
- Paged KV/continuous batching in the first product;
- multiple permanent expert layouts;
- treating 5090 benchmark figures as 5080 expectations;
- promoting NVFP4 solely because it is fast.

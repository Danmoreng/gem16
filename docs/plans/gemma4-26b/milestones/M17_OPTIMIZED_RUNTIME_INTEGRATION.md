# M17 — Rolling optimized runtime integration

Status: accepted at implementation revision `57fdeb309aacfce2e4eba65745fba86f14ebd113`
Class: rolling integration
Completion depends on: M14, M15 and M16 or a recorded M16 skip

Normative inputs: [CUDA state lifecycle](../specs/CUDA_STATE_LIFECYCLE_SPEC.md), [Test matrix](../specs/TEST_MATRIX.md).

## Outcome

Integrate each accepted native operator into one fixed-address whole-model execution plan and capture decode graphs.

## Execution model

M17 starts when the first of M14–M16 is ready. After every merge, rerun full-model correctness and allocation checks. Do not wait for all operators and then attempt a single large assembly.

## Exit gate

- [x] Required decode, prefill and head paths are selected once at initialization.
- [x] Full-model captures and deterministic generation remain inside the M13 envelope.
- [x] No hidden fallback or token-loop allocation occurs.
- [x] Graph capture/replay and engine relaunch tests pass.
- [x] One artifact/profile hash is frozen for M19–M22.

M18 is not a prerequisite when M13 passed. It can run independently if diagnosis or attribution is requested.

## Integration evidence (2026-08-22)

The initialization-time `sm120` profile selects accepted M14 decode MoE, M15
grouped prefill MoE and M16 tied head operators. Attention retains the M12
arithmetic through fixed batched prefill and controlled decode launches; this
is an explicit profile selection, not a runtime fallback. Decode is captured
once as a reusable fixed-address CUDA graph, while prefill adapts its bounded
chunk size to the fixed 64 MiB score slab and 192 MiB M09 workspace cap.

Against the bitwise-exact M16 run, both generated sequences, continuation,
eight captured layer boundaries, router probabilities/IDs and all 262144
logits remain bitwise identical. The clean-commit two-run time falls from
4818.03496 ms to 1079.67332 ms (4.462×), or 5.696× versus M13. Fixed workspace is 121,140,768
bytes and CUDA-visible free memory remains exactly 1,156,775,936 bytes across
both warm runs. A fresh-process relaunch is semantically and logit-bitwise
identical at 1079.54562 ms. The missing QH16/KVH2/D512 native global kernel is also implemented
and passes focused M12-envelope and compute-sanitizer tests; the frozen exact
M17 profile deliberately does not select it. Compact evidence and the frozen
artifact/profile hashes are in `artifacts/m17/diagnostic-summary.json`; the
commit-bound acceptance record is in `artifacts/m17/acceptance.json`.

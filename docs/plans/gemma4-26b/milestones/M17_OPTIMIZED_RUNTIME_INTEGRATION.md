# M17 — Rolling optimized runtime integration

Status: ready for rolling integration; M13 accepted
Class: rolling integration
Completion depends on: M14, M15 and M16 or a recorded M16 skip

Normative inputs: [CUDA state lifecycle](../specs/CUDA_STATE_LIFECYCLE_SPEC.md), [Test matrix](../specs/TEST_MATRIX.md).

## Outcome

Integrate each accepted native operator into one fixed-address whole-model execution plan and capture decode graphs.

## Execution model

M17 starts when the first of M14–M16 is ready. After every merge, rerun full-model correctness and allocation checks. Do not wait for all operators and then attempt a single large assembly.

## Exit gate

- [ ] Required decode, prefill and head paths are selected once at initialization.
- [ ] Full-model captures and deterministic generation remain inside the M13 envelope.
- [ ] No hidden fallback or token-loop allocation occurs.
- [ ] Graph capture/replay and engine relaunch tests pass.
- [ ] One artifact/profile hash is frozen for M19–M22.

M18 is not a prerequisite when M13 passed. It can run independently if diagnosis or attribution is requested.

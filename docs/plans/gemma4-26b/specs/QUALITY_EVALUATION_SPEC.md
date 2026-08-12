# Quality evaluation specification — Fast Track R4

## Data separation

Maintain disjoint operator fixtures, development diagnostics and held-out final evaluation. Performance prompts are also separate where practical.

## M13 early gate

Use a bounded development suite for teacher-forced NLL/KL/rank, router/residual drift, deterministic greedy fixtures and a small prose/task subset. Its sole purpose is `proceed` versus `diagnose`. It is not a final quality claim.

## M18 diagnosis

Run only when M13/M19 fails or causal attribution is explicitly needed. Add arms incrementally: ordinary-BF16 control, head alternative, Unsloth/ModelOpt context or tensor-family ablation. Do not tune on held-out data.

## M19 final gate

Evaluate the frozen production artifact on the untouched held-out suite. Compare to QAT BF16 and clearly labeled external references. Report regressions by category and localize first-layer/position failures when thresholds miss.

## Claims

“Derived from QAT BF16” is provenance. “QAT quality” requires measured target-format evidence. Cross-runtime token identity is not universally required, but same-runtime determinism and operator-reference parity are.

> Historical task contract, closed or superseded. Current disposition is in
> [the archive index](../../evidence/benchmark-archive-2026-11-17.md).
> This file does not authorize a new experiment.

# Trellis35 Ordinary Decode follow-up

The owner authorized a bounded follow-up after N1 on
`d1b6a3c66e1f299fa02e74c25bc30cf0300c4db2`. The initial step measures
the interval from the completion of one Ordinary CUDA Graph to the start
of the next. The launch-level CUPTI trace records no individual graph nodes
and injects no GPU work. Compare against runs without the preload. Measure
the product session token callback separately using a local token sink.

The primary workload uses the original 16,384 prompt IDs, 17,519 context
capacity, 1,135 greedy outputs, batch 1, the same suppressions and FP8 KV.
Require exact output IDs, no fallback and no allocation in the token loop.
Use off/on/on/off ordering in each mode with three retained processes per
cohort and identical cooldown policy. Keep all measured samples. Keep
graph-gap statistics distinct from full-process and decode timings.
Cross-check the launch-level measurements with sparse CUDA Events if the
control runs show a visible instrumentation cost.

Consider a limited GPU-side Ordinary greedy prototype only if both CLI
instrumented cohorts reproduce at least 100 µs mean inter-graph time as
their run median and the controls support the measurement. Any prototype
needs the same emitted IDs, suppression and finite checks, context and
EOS semantics, streaming and session lifecycle behavior, memory and
protected-profile checks. Keep it only for a reproducible end-to-end gain
with a positive lower confidence bound in both orders and at least 1%
pooled Ordinary throughput improvement. The D2 non-inferiority margin is
−0.20% in both orders.

If the scheduling screen fails, measure the production Ordinary M1
Routed-GateUp kernel on real early, middle and late routes. Verify bitwise
replay, measure warm and evicted-cache latency, and inspect actual traffic,
codec instructions and tensor work. Fused, overlapping costs must be
reported at their measured boundaries. This branch is a diagnosis of
where to optimize next.

Record the decision, reproducible evidence and a README pointer. Raw
results are under `benchmarks/results/2026-11-05/d1b6a3c/ordinary-followup`.

The follow-up is complete. Both timing methods put the gap far below the
prototype threshold. The real-route GateUp diagnosis is recorded in the
[evidence report](../../evidence/gemma4_26b/ordinary-decode-followup-2026-11-05.md).

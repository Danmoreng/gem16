# Adoption of the optimized12B D3 default — 2026-11-17

The owner selects optimized arithmetic as the default while retaining explicit
reference mode, and authorizes committing/pushing the completed implementation
and evidence. The [mode contract](../../MTP_MATH_MODES.md) supersedes earlier
exact-default/opt-in-only policy; previous numerical/performance records remain
unchanged. This policy slice changes selection and diagnostics, not arithmetic,
Target acceptance, model weights, cache transactions or CUDA kernel bodies.

With `GEM16_12B_D3_ATTENTION` **unset**, the supported12B D3/FP8-KV/greedy/
capacity17519 workload selects optimized automatically. One warmup plus three
retained requests measures112.065302,112.007744,112.056443tok/s;
median **112.056443tok/s**. All1135 output IDs and all1323 draft IDs match the
previous112.100752 optimized cohort. All three requests have1134 timed intervals,
zero fallbacks/token-loop allocations and deterministic outputs. Whole-process
200ms NVML peak11775MiB; high-utilization median SM2287MHz, not decode-only.
This is a policy validation, not an additional kernel speedup claim.

An explicit `exact` correctness request retains all1135 reference output IDs
and all1311 reference draft IDs. Its timing is not a primary performance cohort.
Automatic capacity17520 selects/reports reference; forced optimized at17520
fails with an unsupported-geometry/no-fallback error. No capacity is resized.
The server's `/health`, startup log and a short chat response pass in both
automatic optimized and explicit-reference modes. Both owned servers stop.

Products build and all host units pass. Four D3 accounting tests and29 workload
benchmark host tests pass. New host selection/JSON tests cover supported default,
exact override, `fp8` alias, incompatible depths/contexts/KV/sampling/adaptive
configurations, invalid settings and disclosure of initial-only quality scope.
`git diff --check` passes. Embedded global/local attention and generated norm
source/provenance hashes remain checked; existing operator/sanitizer/graph/SASS
evidence still applies to those unchanged kernels. No new quality, full-model
sanitizer, Windows or media campaign is claimed.

The first surface-check script referenced a nonexistent direct-CLI context JSON
field. The successful GPU result was retained and the assertion corrected to
the actual serialized mode/reason plus recorded CLI capacity argument; it was
not rerolled. Model identity is read from the documented `/v1/models` endpoint.

[Manifest](../../../artifacts/gemma4_12b/d3-default-policy-2026-11-17.json),
[source snapshot](../../../artifacts/gemma4_12b/d3-default-policy-2026-11-17-source.json.gz)
and [raw receipts](../../../artifacts/gemma4_12b/d3-default-policy-2026-11-17-raw.json.gz)
retain actual samples, commands/environment, binary/source/artifact hashes,
telemetry, host logs, reference IDs, visible-error and server-surface results.
The local directory is `benchmarks/results/2026-11-17/d3-default-policy/`.
No additional kernel experiment or release is part of this commit/push request.

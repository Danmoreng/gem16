# GateUp H128/FP8 fusion — rejected, 2026-10-22

Step 5 tested a single-CTA selected-input H128 transform plus FP8 quantization
against the accepted `0dd1240` runtime. Both M1 and T3 warp-input paths used it;
direct H128 and model/cache precision were unchanged. The full candidate is
rejected, and all four production/test candidate files are restored exactly to
`0dd1240`. Source snapshots remain in the immutable experiment record.

Real-input graph replay suggested 1.64–1.74x operator throughput, with transformed
FP32, FP8 bytes and scales bit exact. An independent CUDA test compared the two
old kernels against fused diagnostic and production variants for 8/24 assignments
and zero, small, normal and saturating amplitudes. Full CUDA/host tests, native
memcheck, 12B product and 64-output 12B/NVFP4 ordinary/D2 regressions passed.
Nsight verified 210 ordinary / 120 D2 actual fused calls over eight outputs.

## Primary result

Same immutable weights, input IDs, 16,384 prompt tokens, context 17,519, batch one,
greedy sampling, cache semantics, timing boundaries and 1,135 outputs. Each cohort
has three warmups and ten measured repetitions. All sixty retained runs reproduce
the same token vector, with no inference-loop allocations or fallback.

| Comparison | Parent tok/s | Candidate tok/s | Change |
|---|---:|---:|---:|
| Ordinary A→B | 138.079 | 136.030 | −1.484% |
| Ordinary B→A | 137.711 | 135.947 | −1.281% |
| D2 adjacent pair | 172.344 | 170.451 | −1.099% |

Ordinary telemetry has comparable clocks (median 2,625 MHz in both), approximately
171–172 W under high utilization, and 74–75°C. The candidate cannot be retained
on the strength of its isolated operator result. The cause of the end-to-end
regression remains unproven; this is not evidence of a driver defect. D2 is one
adjacent pair rather than a separate AB/BA campaign. No further production fusion
or D2 speed claim is made.

Exact commands, samples, telemetry, source snapshots/hashes, correctness logs,
actual dispatch and restoration metadata are preserved in
`artifacts/trellis35/gate-input-fusion-2026-10-22.json`. Raw root:
`benchmarks/results/2026-10-22/c0a807af/gate-input-fusion/`.
The accepted source was rebuilt after restoration; final unit/Trellis checks are
recorded in the companion diagnostic record. No push.

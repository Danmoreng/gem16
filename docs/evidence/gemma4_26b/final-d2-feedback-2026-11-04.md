# AP4 final fixed-D2 feedback — 2026-11-04

**Rejected by the predeclared primary D2 gate; exact parent restored.** [Complete record](../../../artifacts/trellis35/final-d2-feedback-2026-11-04.json).

| D2 order | Candidate throughput change | 95% bootstrap interval |
|---|---:|---:|
| AB | -0.1161% | [-0.2131%, +0.0077%] |
| BA | -0.0607% | [-0.2514%, +0.1049%] |
| pooled | +0.0119% | [-0.1444%, +0.0868%] |

Only draft_count==2 omits the last unconsumed FP8 post quantization/projection.
First feedback, both draft heads, final norm/NVFP4 activation, weights, arenas
and all other draft-depth code remain. Actual full decode removes502quantizations
and502post projections:1004to502feedback projections over502groups; every other
kernel count and all1135outputs/1004proposals remain exact. Graph node counts
change; this is disclosed rather than described as the same graph.

Nine production host/CUDA tests and a32output scoped memcheck pass. Unrun
lifecycle/protected/secondary gates stay explicitly unknown in the full record.
Primary uses ABBA,3warmups+10retained each,16384prompt,1135outputs/1134intervals,
17519context,FP8KV,greedy0,same mask,coldcache and residentGPUweights without
observers. BothAB/BA lower95bounds must be positive; no sample is removed or
favorable rerun selected. Clocks follow firmware Dynamic Boost and are sampled.
No new llama, Windows or complete media claim is made.

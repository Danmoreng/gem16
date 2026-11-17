# AP2 flat R3 QKV grid — 2026-11-04

**Rejected by the predeclared end-to-end gate; exact parent runtime source and canonical binary are restored.** The [complete sanitized record](../../../artifacts/trellis35/qkv-flat-grid-2026-11-04.json) retains raw samples,
identities, commands, actual dispatch/proposals, counters and validation scope.

## Result

| D2 order | Candidate throughput change | 95% bootstrap interval |
|---|---:|---:|
| AB | +0.0898% | [+0.0301%, +0.1858%] |
| BA | +0.0584% | [-0.0438%, +0.1621%] |
| pooled | +0.0806% | [+0.0257%, +0.1538%] |

Primary D2 ran parent/candidate/candidate/parent, three warmups and ten retained
samples per cohort. The gate required positive lower95bounds in **both** orders.
Ordinary's predeclared secondary gate requires each point>=0 and lower95>-0.20%.
Ordinary runs only after D2 passes; primary failure stops this candidate.
All samples stay retained. This compares the same native format, model, output
stream and502D2groups, with16,384prompt /1,135output /1,134decode intervals,
context17,519, FP8KV, greedy seed0, fixed masks, cold cache, resident GPU weights,
cooldown<=50°C and200msNVML/1Hzprocess sampling. Firmware clocks are unlocked.
No new cross-engine reference was run.

## Change and operator evidence

Only the native26B fixed-R3 QKV dispatch changes:256local /288global CTAs replace
384/512. Exact three-row M16 arithmetic, 32-byte ephemeral shared-row padding,
weights/scales/K order and global Q/K-only behavior remain. Other selectors and
O projection remain unchanged. Exhaustive output-index coverage proves each
of24,576local /27,648global FP32 output elements is written once.

All12real early/middle/late QKV cases are bit exact and faster in warm operator
replay: geomean+1.323%, bootstrap95[+0.737%,+1.948%], with individual medians
+0.98..1.66%. O replay is an unchanged control. This gain is not a total model
speed prediction. Production kernel body matches the tested offline body.
Registers rise36->44 (allocated40->48), with no spills and unchanged actual
10block occupancy cap. Four matched parent/candidate NCU captures retain detailed
shared/L2/DRAM counters. Profiler/observer/replay times never enter primary rates.
Actual production dispatch and all1,135outputs/1,004proposals were verified.

## Validation and limits

Six guarded diverse/zero-input native-shape cases pass memcheck and synccheck.
Nine production host/CUDA tests and12Bproduct regression pass. Fresh parent and
candidate12B/internalNVFP4 ordinary/D2 runs have identical64token vectors with
zero fallback/token-loop allocation. One real CompactVision scene at70softtokens
returns identical Ordinary/D2 text and all required image facts; its live CLI
does not export raw image token vectors. A first unsupported JSON+media setup
was rejected before GPU start and retained. This is bounded media regression,
not a new complete image/audio or Windows qualification.

The complete16Kprompt production memory check passes0errors with8outputs;
the full1,135output/1,004proposal parity trace is separate. Source, compiled
binary/module and observed grid/register identities are preserved. No commit,
push or release is part of this experiment.

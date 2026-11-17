> Historical task contract, closed or superseded. Current disposition is in
> [the archive index](../../evidence/benchmark-archive-2026-11-17.md).
> This file does not authorize a new experiment.

# Current12B matched-chunk Prefill comparison

Owner follow-up 2026-11-15: independently test llama.cpp logical batch 8192
with physical microbatches 512, 1024 and 2048, keeping all other controls from
the completed curve. Use three warmups and ten retained samples per point,
plus one adjacent 1024/1024 control to distinguish logical-batch effects from
cohort drift. Retain results separately; verify current central upstream before
each cohort and actual batch/microbatch values in server logs. No runtime
patch, Decode Goal resumption or Git publication is part of this follow-up.

Owner request2026-11-15: pause Decode optimization, compare current qualified
12B native engine, latest central llama.cpp and installed current vLLM with
matched Prefill token chunks. Preserve historical results and record actual
settings, versions, samples and memory. No Decode optimization resumes here.

Use the same16,384 Wikipedia prompt IDs,17,519 requested context, one sequence,
greedy sampling, existing pinned12B checkpoints and KV types. No prompt reuse,
weight offload, precision substitution, JIT/native fallback or model downloads
unless the pinned sources are missing. Native runtime defaults remain8192;
add an explicit startup-only12B FP8-KV chunk override (validated32-token
multiples up to8192), recorded by the benchmark helper and actual engine JSON.
Add an explicit requested context override so Prefill-only probes with one
output retain the same17,519 cache/model capacity as the previous comparison.
Timing is prompt processing through first selected token; no Decode throughput
claim is made from a one-output Prefill probe. Check actual first-token equality
and selected full-generation boundary/parity checks separately.

Screen512/1024/2048/4096/8192 at1 warmup+3 retained per engine, one engine at a
time, exact same output/context controls. llama.cpp logical and physical token
batch sizes equal the chosen chunk; vLLM max_num_batched_tokens equals it,
max_num_seqs1 and chunked Prefill enabled. Report requested vs observed batching
and identify GPU-tile sizes separately. Unsupported/OOM points remain explicit;
do not reduce context/precision or offload to make them fit. Native and vLLM
formats/KV differ from GGUF/Q8_0 and remain disclosed despite matched chunks.

Use independent3-warmup/10-retained confirmation at common1024, common8192
if all fit, and each engine's best valid screened point. Do not promote a
best-of-screen number as confirmed throughput. Reuse a single confirmed point
when it serves multiple labels. Preserve helper/source/executable/library/model
hashes, exact commands, raw results and external200ms GPU telemetry. Verify
upstream central llama freshness before every reference cohort; never switch
binaries inside a cohort. Do not infer an intrinsic Prefill advantage from the
old512/8192/1024 configuration comparison. Document/update current README/chart
notes while retaining historical chart data and sources.

Check Berlin time before long work. Wrap22:45, save/stop by22:55, no work after
23:00. If time expires, preserve finished points and a continuation note; do not
present incomplete confirmation as completed. Raw root:
`benchmarks/results/2026-11-15/58febf4/12b-matched-prefill/`.

Implementation refinement before screen measurements: because one-output
Prefill probes are short, independently confirm every valid screened point
with3 warmups+10 retained samples. This avoids selecting a winner from noisy
screen-only numbers and yields a fully confirmed chunk curve. All settings
remain fixed across sizes. Explicit llama.cpp token-embedding GPU placement is
being checked before freezing reference settings: `CPU_Mapped` alone does not
prove CPU execution, and49/49 offloaded layers alone does not prove embedding
placement. Any reference placement setting will be recorded consistently.

Before primary confirmation, vLLM8192 at the earlier0.90 budget fails startup:
2.79GiB KV required at17,519 context versus2.51GiB available. One explicit
memory-budget correction to0.94 is checked, then used consistently for every
vLLM chunk point if valid. Original0.90 preflights/screens/failure remain
separate; `-mem94` labels identify the revised curve. No context/model/precision
reduction, offload or compiler disabling is used to fit. This is not evidence
that8192 is impossible in physical16GB memory. Confirm every valid point;
retain failed configuration logs even when another budget succeeds.

The upstream guard stopped the first llama.cpp confirmation before inference:
official master advanced from75fd7f9 to18b74fa. Update centrally and place all
new confirmed llama cohorts in a new commit-named subdirectory, with separate
binary/library/source identities. The75fd7f9 screen remains historical and is
not combined with newer primary reference samples. Completed native/vLLM
confirmations remain valid; no CUDA benchmarking overlaps the CPU build.
Diagnostic schedule-count observations may run during the build, without any
of their timings entering primary results.

Completed 2026-11-15 20:47 Berlin: all 15 points confirmed with 3 warmups and 10 retained samples; 150 primary samples. Native best at 8192: 5965.28 tok/s; central llama.cpp 18b74fa best at 1024: 4839.77 tok/s; vLLM 0.33.0 best at 4096: 7112.99 tok/s. Same-1024 comparison: 5246.93/4839.77/6871.12 tok/s. All first tokens are 61684; native full 1135-output checks match at all 5 sizes, default D2 counters unchanged. Public records, curve and README are updated; see [evidence](../../evidence/gemma4_12b/matched-prefill-2026-11-15.md). No commit/push, no further GPU work; Decode Goal remains paused. Raw completion.json records artifact hashes.

Batch/microbatch follow-up complete 2026-11-15 20:58 Berlin: four valid cohorts, 40 retained measurements, same current central 18b74fa. Batch 8192 with microbatch 512/1024/2048 reaches 4728.51/4798.45/4598.25 tok/s. Adjacent 1024/1024 control: 4812.04 tok/s; logical-only difference -0.282%, no demonstrated gain. Original measurements remain immutable. [Follow-up evidence](../../evidence/gemma4_12b/prefill-batch-microbatch-2026-11-15.md), public samples and README updated. All GPU jobs finished; Decode Goal remains paused, no commit or push.

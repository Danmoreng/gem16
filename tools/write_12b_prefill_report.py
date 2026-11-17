"""Write the current Prefill report/README block from confirmed exported data."""
from pathlib import Path
import argparse,json
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('artifact',type=Path);a=p.parse_args();artifact=a.artifact.resolve();d=json.loads((artifact/'results.json').read_text());records=d['records'];index={(r['engine'],r['chunk']):r for r in records}
names={'gem16':'GemSixteen','llama-cpp':'llama.cpp','vllm':'vLLM 0.33.0'}
def num(r):return f"{r['median_prefill_tokens_per_second']:,.2f}"
best={e:max([r for r in records if r['engine']==e],key=lambda x:x['median_prefill_tokens_per_second']) for e in names}
lines=['# Current 12B Prefill at matched chunk sizes —2026-11-15','',
'All 15 final points retain ten measurements after three warmups. The curves use the same 16,384 Wikipedia prompt IDs, 17,519 requested context, one sequence and one requested output token, with greedy sampling and no prefix reuse. Model initialization is excluded. The one-output probe measures Prefill through first-token selection; it does not measure Decode throughput.','',
'| Engine |512 |1,024 |2,048 |4,096 |8,192 |','|---|---:|---:|---:|---:|---:|']
for e in names:lines.append('|'+names[e]+'|'+'|'.join(num(index[e,c]) for c in (512,1024,2048,4096,8192))+'|')
lines+=['','Values are confirmed median Prefill tokens/s. Read the complete curve rather than assuming the largest chunk is fastest.','',
'| Engine |Fastest tested chunk |Confirmed tok/s |Bootstrap95% interval |Sampled device peak MiB |','|---|---:|---:|---:|---:|']
for e,r in best.items():lines.append(f"|{names[e]}|{r['chunk']:,}|{num(r)}|{r['median_bootstrap_95'][0]:,.2f}–{r['median_bootstrap_95'][1]:,.2f}|{r['telemetry']['sampled_peak_device_total_mib']:,.0f}|")
lines+=['','## Batching and timing controls','',
'- Native: explicit startup-only `GEM16_12B_PREFILL_CHUNK_TOKENS`, validated 32-token multiples bounded by 8192 and the automatic limit. No override preserves the existing 8192 default. Every final result reports its actual chunk. Full 1,135-output comparisons at all five sizes are bit-equal, with unchanged KV capacity; the default D2 regression also preserves all outputs/proposal/acceptance counts.',
'- llama.cpp: logical batch and physical microbatch both equal the selected chunk. Actual context/batch initialization and within-prompt progress are retained in server logs. Requested 17,519 context is internally padded; this is recorded rather than disguised as identical physical allocation. Token-embedding placement is explicitly `CUDA0` at every point; all 49/49 layers are offloaded. The mapped-source buffer in the earlier preflight was not treated alone as evidence of CPU execution.',
'- vLLM: Chunked Prefill, `max_num_batched_tokens` equal to the selected chunk, `max_num_seqs=1`, prefix caching disabled, GPU memory utilization 0.94 at all final points. Separate diagnostic observations confirm 16×1,024 and 2×8,192 actual scheduled tokens. Those observations use in-process EngineCore and contribute no primary timings. CUDA Graphs/compiler settings remain enabled as in the installed engine.',
'- Native CLI samples start a new process and initialize the model for each sample (initialization is excluded); reference engines retain their model within a cohort. Cache history and thermal conditions can therefore differ.\n- All engines retain their own documented prompt-to-first-token timing boundaries. This is an engine-level configuration comparison, not identical GPU-only operator timing. Ten-sample bootstrap intervals describe these same-day cohorts on one GPU; no general hardware confidence is implied.',
'','## Formats, versions and memory','',
'GPU: RTX 5080 Laptop, 16,303 MiB; NVIDIA 620.70.06, max-power platform profile and Dynamic Boost active. External 200 ms device-total telemetry and per-process samples are retained. Peaks include startup and other existing device use; sampling can miss shorter transients. No CPU weight/expert offload, precision reduction or context reduction is used to fit.','',
f"The final llama.cpp reference is official master `{d['inputs']['central_commit']}`. The earlier 75fd7f9 screen is preserved separately; a freshness check stopped its first confirmation before inference after upstream advanced. All final reference points use the newer source/binary/library set in a separate directory. Completed native/vLLM points are preserved. Native compute kernels remain at source parent 58febf4; the only runtime addition is the explicit startup chunk control. The Decode optimization Goal remains paused.",'',
'Native and vLLM use the same pinned `unsloth/gemma-4-12b-it-NVFP4@b1f649734b34aa5575b03d186abd1b9be3d0d5c4` mixed FP8/NVFP4 source: 9,304,966,064 bytes, SHA256`7c2ee23298e7c3a9247e8947597dca5a38f8b791a0322487466d2bfad8ce704b`. The converted Q8/NVFP4 GGUF is 9,366,658,208 bytes, SHA256`6f90177f6a2d42406d57cfa764eae890b262bfdf71d353bd4827e0488b099896`. Native/vLLM KV is FP8; llama.cpp KV is Q8_0. GGUF embeds tokenizer/format metadata, so bytes do not establish exact quantization or quality parity. No Assistant is loaded for these Ordinary-Prefill points.','',
'The original vLLM 8192 attempt at 0.90 failed startup: 2.79 GiB KV required versus 2.51 GiB available. A subsequent 0.94 fit succeeds. The failed configuration is retained and is not proof of a physical 16 GB limit or of the minimum required budget. Startup/autotuning history can influence memory profiling; all final points use the same 0.94 setting. FlashInfer can report default GEMM tactic use outside its tuning buckets; this remains the same native NVFP4 kernel family, and warnings are retained. Primary measurements contain no reported inference-time JIT warning after warmups. These are the installed auto-backend configurations, not an exhaustive kernel-tuning search.','',
'## Evidence and validation','',
'Build: CUDA 13.4.92, GCC 16.2.1, SM120a; local CMake 4.5 cache disables unused C++ module scans without weakening warnings. The bounded chunk parser passes defaults, valid boundaries, malformed/overflow/out-of-budget and BF16-profile rejection checks. The workload helper passes 28 host tests, including child-local environment handling, actual-chunk mismatch rejection, fixed context and one-output timing semantics. Linux validation does not imply Windows execution.','',
f"[Confirmed JSON](../../../{artifact.relative_to(ROOT)}/results.json) · [CSV](../../../{artifact.relative_to(ROOT)}/results.csv) · [PNG](../../../{artifact.relative_to(ROOT)}/prefill-chunks.png)",'',
'Raw commands/results/telemetry, source/executable/model hashes, failed attempts and diagnostic counts: `benchmarks/results/2026-11-15/58febf4/12b-matched-prefill/`. Native/vLLM samples remain in the root; the final llama.cpp cohorts are in its commit-named subdirectory. The initial source snapshot and later reference identities are separate immutable records.','',
'Historical chart/source data remain unchanged. The earlier512/8192/1024 Prefill comparison described those configurations and cannot establish matched-chunk or intrinsically optimum engine performance. This report supplies the requested control curve; it does not refresh historical or current Decode numbers.']
report=ROOT/'docs/evidence/gemma4_12b/matched-prefill-2026-11-15.md'
with report.open('x') as f:f.write('\n'.join(lines)+'\n')
readme=ROOT/'README.md';text=readme.read_text();marker='<!-- MATCHED-12B-PREFILL-2026-11-15 -->';assert marker not in text
block=f'''{marker}
**2026-11-15 matched 12B Prefill:** With 1,024-token chunks, confirmed medians are
**{num(index['gem16',1024])} / {num(index['llama-cpp',1024])} / {num(index['vllm',1024])} tok/s**
for GemSixteen / llama.cpp / vLLM. Across the five tested sizes, each engine's
best confirmed point is **{num(best['gem16'])} / {num(best['llama-cpp'])} / {num(best['vllm'])} tok/s**
at **{best['gem16']['chunk']:,} / {best['llama-cpp']['chunk']:,} / {best['vllm']['chunk']:,} tokens**.
The full curve uses identical 16K prompt IDs, 17,519 requested context, 3 warmups
and 10 retained one-output probes per point. Formats/KV remain different;
vLLM uses a consistent 94% memory budget. This refresh concerns Prefill only.
[Curve, complete samples and controls](docs/evidence/gemma4_12b/matched-prefill-2026-11-15.md).
<!-- /MATCHED-12B-PREFILL-2026-11-15 -->

'''
anchor='## Recorded performance\n\n';assert text.count(anchor)==1;readme.write_text(text.replace(anchor,anchor+block,1))
print('WROTE report and narrow README block',flush=True)

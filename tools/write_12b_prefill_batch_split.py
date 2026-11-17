"""Export the bounded logical-batch/physical-microbatch follow-up."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("raw", type=Path)
parser.add_argument("destination", type=Path)
args = parser.parse_args()
raw = args.raw.resolve()
destination = args.destination.resolve()
inputs = json.loads((raw / "inputs.json").read_text())
rng = np.random.default_rng(121116)


def clean(value):
    if isinstance(value, dict):
        return {clean(key): clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, str):
        for before, after in (
            (str(ROOT), "$ROOT"),
            (str(Path.home() / "Development/_engines/llama.cpp"), "$CENTRAL_LLAMA_CPP"),
            (str(Path.home()), "$HOME"),
        ):
            value = value.replace(before, after)
        assert not re.search(r"/home/|/Users/|GPU-[0-9a-f]{8}", value), value
    return value


records = []
for batch, micro in inputs["protocol"]["order"]:
    label = f"confirm-b{batch}-ub{micro}"
    path = raw / f"{label}.json"
    result = json.loads(path.read_text())
    assert result["runtime"]["llama_cpp_commit"] == inputs["commit"]
    assert result["representative_output_token_ids"] == [61684]
    assert len(result["runs"]) == 10
    assert result["configuration"]["warmups"] == 3
    assert result["configuration"]["batch_size"] == batch
    assert result["configuration"]["ubatch_size"] == micro
    assert result["configuration"]["context_tokens_requested"] == 17519
    assert all(row["tokens_cached"] == 0 for row in result["runs"])
    rates = np.array([row["prompt_tokens_per_second"] for row in result["runs"]])
    bootstrap = np.median(rng.choice(rates, (10000, 10), replace=True), axis=1)
    log = (raw / f"{label}.server.log").read_text()
    assert set(map(int, re.findall(r"\bn_batch\s*=\s*(\d+)", log))) == {batch}
    assert set(map(int, re.findall(r"\bn_ubatch\s*=\s*(\d+)", log))) == {micro}
    assert "offloaded 49/49 layers to GPU" in log
    compute = re.findall(r"CUDA0 compute buffer size\s*=\s*([\d.]+) MiB", log)
    assert len(compute) == 1
    with (raw / f"{label}-telemetry.csv").open() as stream:
        telemetry = list(csv.reader(stream))[1:]
    record = {
        "batch": batch,
        "microbatch": micro,
        "median_prefill_tokens_per_second": float(np.median(rates)),
        "median_bootstrap_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "cuda_compute_buffer_mib": float(compute[0]),
        "sampled_device_peak_mib": max(float(row[6].strip().split()[0]) for row in telemetry),
        "result": result,
        "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "command": json.loads((raw / f"{label}.command.json").read_text()),
    }
    records.append(record)

parent = json.loads((ROOT / "artifacts/gemma4_12b/matched-prefill-2026-11-15/results.json").read_text())
assert parent["inputs"]["central_commit"] == inputs["commit"]
prior = [row for row in parent["records"] if row["engine"] == "llama-cpp" and row["chunk"] in (1024, 8192)]
by = {(row["batch"], row["microbatch"]): row for row in records}
logical_delta = 100 * (by[8192, 1024]["median_prefill_tokens_per_second"] / by[1024, 1024]["median_prefill_tokens_per_second"] - 1)
payload = clean({
    "schema_version": 1,
    "date": "2026-11-15",
    "scope": "logical batch fixed at 8192; physical microbatch sweep plus adjacent 1024/1024 control",
    "inputs": inputs,
    "records": records,
    "earlier_diagonal_controls": prior,
    "logical_batch_8192_vs_1024_at_microbatch_1024_percent": logical_delta,
    "limitations": [
        "Single-GPU sequential cohorts; three warmups and ten retained per configuration.",
        "One output token, identical 16384-token prompt and 17519 requested context; no prefix reuse or Decode claim.",
        "Bootstrap percentile 95% median intervals, 10000 resamples, seed 121116.",
        "Device peaks include startup and existing device use, sampled every 200 ms.",
        "Earlier diagonal 8192/8192 and 1024/1024 controls are separate earlier cohorts, not new follow-up measurements.",
        "No operator profiling: changes in reserved workspace do not by themselves establish the causal bottleneck.",
    ],
})
destination.mkdir(parents=True, exist_ok=False)
(destination / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
with (destination / "results.csv").open("x", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["batch", "microbatch", "median_prefill_tokens_per_second", "lower95", "upper95", "cuda_compute_buffer_mib", "sampled_device_peak_mib"])
    for row in records:
        writer.writerow([row["batch"], row["microbatch"], row["median_prefill_tokens_per_second"], *row["median_bootstrap_95"], row["cuda_compute_buffer_mib"], row["sampled_device_peak_mib"]])

lines = [
    "# 12B llama.cpp logical batch / microbatch follow-up — 2026-11-15",
    "",
    "The owner requested separating the two controls after the matched-chunk curve. Logical batch is held at 8,192 while physical microbatch is 512, 1,024 or 2,048. An adjacent 1,024/1,024 control checks the effect of logical batch at fixed physical microbatch. All four points use three warmups and ten retained samples, the same 16,384 prompt IDs, 17,519 requested context, one greedy output token, no prefix reuse, Q8/NVFP4 GGUF and Q8_0 KV. No engine runtime changes or Decode measurements are involved.",
    "",
    "| Logical batch | Physical microbatch | Median tok/s | Bootstrap 95% | CUDA compute buffer MiB | Sampled device peak MiB |",
    "|---:|---:|---:|---:|---:|---:|",
]
for row in records:
    lo, hi = row["median_bootstrap_95"]
    lines.append(f"|{row['batch']:,}|{row['microbatch']:,}|{row['median_prefill_tokens_per_second']:,.2f}|{lo:,.2f}–{hi:,.2f}|{row['cuda_compute_buffer_mib']:,.2f}|{row['sampled_device_peak_mib']:,.0f}|")
lines += [
    "",
    f"At physical microbatch 1,024, changing logical batch from 1,024 to 8,192 changes the measured median by {logical_delta:+.3f}%. These sequential cohorts do not establish a general advantage for such a small difference.",
    "",
    "Earlier same-revision diagonal controls (three warmups, ten retained, separate cohorts):",
    "",
]
for row in prior:
    lines.append(f"- Batch = microbatch = {row['chunk']:,}: {row['median_prefill_tokens_per_second']:,.2f} tok/s.")
lines += [
    "",
    f"All four new server logs confirm the requested logical/physical sizes, 49/49 GPU layers and explicit CUDA0 token-embedding placement. All retained first tokens are 61684 and reported cross-request cached tokens are zero. The central resolver verifies official upstream before each cohort; all use `{inputs['commit']}` and the same executable/library set as the earlier curve. Source, model and helper identities, raw samples, commands and external telemetry are preserved.",
    "",
    "The sweep isolates the batch configuration; it does not profile which CUDA operator causes large-microbatch performance to decline. Reserved compute buffer size is an allocation fact, not proof of the causal performance bottleneck. The original matched-chunk curve and historical measurements remain unchanged. Decode optimization remains paused.",
    "",
    "[JSON](../../../artifacts/gemma4_12b/prefill-batch-microbatch-2026-11-15/results.json) · [CSV](../../../artifacts/gemma4_12b/prefill-batch-microbatch-2026-11-15/results.csv) · [Original matched-chunk curve](matched-prefill-2026-11-15.md)",
    "",
    "Raw evidence: `benchmarks/results/2026-11-15/58febf4/12b-matched-prefill/batch-microbatch-followup/`. No commit or push is part of this task.",
]
report = ROOT / "docs/evidence/gemma4_12b/prefill-batch-microbatch-2026-11-15.md"
with report.open("x") as stream:
    stream.write("\n".join(lines) + "\n")
readme = ROOT / "README.md"
text = readme.read_text()
anchor = "<!-- /MATCHED-12B-PREFILL-2026-11-15 -->"
assert text.count(anchor) == 1
summary = "\n\n**Batch/microbatch follow-up:** At logical batch 8,192, physical microbatches\n"
summary += "512 / 1,024 / 2,048 yield **" + " / ".join(f"{by[8192, micro]['median_prefill_tokens_per_second']:,.2f}" for micro in (512, 1024, 2048)) + " tok/s**.\n"
summary += "[Independent controls, samples and memory](docs/evidence/gemma4_12b/prefill-batch-microbatch-2026-11-15.md)."
assert "**Batch/microbatch follow-up:**" not in text
readme.write_text(text.replace(anchor, anchor + summary, 1))
print("EXPORTED", destination.relative_to(ROOT), "logical-only delta", logical_delta)

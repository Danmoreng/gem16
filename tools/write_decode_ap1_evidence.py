#!/usr/bin/env python3
"""Publish a new sanitized AP1 record from retained, separately scoped evidence.

The raw directories remain private and immutable. This generator does not run
benchmarks or infer successful gates from missing data. Existing outputs fail.
"""
import argparse
import getpass
import hashlib
import json
from pathlib import Path
import re
import socket
import statistics

ROOT = Path(__file__).resolve().parents[1]


def identity(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def sanitize(value, replacements):
    if isinstance(value, dict):
        return {sanitize(k, replacements): sanitize(v, replacements)
                for k, v in value.items()
                if k.lower() not in {"pid", "hostname", "host_name", "gpu_uuid", "uuid", "serial", "process_id"}}
    if isinstance(value, list):
        return [sanitize(v, replacements) for v in value]
    if isinstance(value, str):
        for source, target in replacements:
            value = value.replace(source, target)
        value = re.sub(r"GPU-[0-9a-fA-F-]{16,}", "${GPU_UUID}", value)
    return value


def technical_environment(raw):
    cpu = raw["cpu"]["stdout"]
    gpu = raw["gpu"]["stdout"]
    def field(text, name):
        match = re.search(r"^\s*" + re.escape(name) + r"\s*:\s*(.+)$", text, re.M)
        if not match:
            raise ValueError("missing technical environment field: " + name)
        return match.group(1).strip()
    return {"cpu_model": field(cpu, "Model name"), "gpu_model": field(gpu, "Product Name"),
            "driver_version": field(gpu, "Driver Version"),
            "gpu_memory_total": field(gpu, "Total"),
            "platform_profile": raw["platform_profile"]["stdout"].strip(),
            "nvidia_powerd": raw["powerd"]["stdout"].strip(),
            "toolchain": {k: raw[k]["stdout"] for k in ("nvcc", "cmake", "gcc", "nsys", "ncu")},
            "clock_policy": "firmware managed; no locked clocks; cooldown <=50C"}


def baseline(raw):
    result = dict(raw)
    result["environment"] = technical_environment(raw["environment"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--reference-refresh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise ValueError("never overwrite published AP1 evidence")
    raw, current = args.raw.resolve(), args.reference_refresh.resolve()
    def load(directory, name): return json.loads((directory / name).read_text())
    qualification = load(raw, "qualification.json")
    if qualification["status"] != "ap1_complete" or not all(qualification["gates"].values()):
        raise ValueError("AP1 gates are incomplete")
    first = load(raw, "primary-baseline.json")
    latest = load(current, "primary-baseline.json")
    for record in (first, latest):
        if record["status"] != "complete_unprofiled_baseline":
            raise ValueError("primary baseline not qualified")
        if len(record["cohorts"]) != 8:
            raise ValueError("incomplete ABBA cohorts")
    costs = {label: load(raw / "analysis-v2", label + ".cost.json")
             for label in ("native-ordinary-trace", "native-d2-trace", "native-d2-groups-trace")}
    costs["native-d2-readonly-trace"] = load(raw, "native-d2-readonly-trace.cost.json")
    for cost in costs.values():
        if cost["unclassified_kernel_sum_ms"]:
            raise ValueError("unclassified native kernel costs")
        busy, idle = cost["gpu_busy_union_ms"], sum(cost["gpu_idle_ms"].values())
        exclusive = sum(cost["gpu_exclusive_category_ms"].values())
        if abs(busy + idle - cost["wall_ms"]) > 1e-6 or abs(exclusive + cost["gpu_overlap_ms"] - busy) > 1e-6:
            raise ValueError("unbalanced native interval union")
    record = {"schema_version": 1, "status": "ap1_complete_no_runtime_change",
              "scope": "AP1 measurement and diagnostics only; AP2 has not started; no runtime speedup or release claim",
              "first_current_baseline": baseline(first), "post_reboot_current_baseline": baseline(latest),
              "reference_revision_transition": "official master advanced after the first baseline; first attempt failed freshness before inference; central updated and a new complete unprofiled ABBA campaign was retained separately",
              "source_findings": load(raw, "pro-source-findings.json"),
              "qkv_instruction_proof": load(raw, "qkv-empty-cta-instruction-proof.json"),
              "timing_boundary": load(raw, "timing-boundary-v2.json"),
              "reference_counter_semantics": load(raw, "reference-counter-semantics.json"),
              "measurement_sources": [identity(ROOT / name) for name in (
                  "tools/benchmark_wikipedia_workload.py", "tools/analyze_decode_cost.py",
                  "tools/analyze_mtp_group_cost.py", "tools/decode_moe_accounting.py",
                  "tools/llama_speculative_accounting.py", "tools/run_llama_decode_replay.py",
                  "tools/validate_mtp_groups.py", "tests/cuda/ap1_group_observer.cu",
                  "tests/cuda/ap1_input_observer.cu", "tests/cuda/ap1_input_observer_memcheck.cu",
                  "tests/cuda/observe_mtp_proposals.cpp", "tests/cuda/llama_decode_replay_driver.cpp",
                  "tests/cuda/decode_replay_support.h", "tests/cuda/fp8_projection_operator_lab.cu")],
              "native_model_identities": load(raw, "native-model-identities.json"),
              "native_decode_costs": costs,
              "moe_fork_join": {mode: load(raw / "analysis-v2", "native-" + mode + "-trace.moe.json") for mode in ("ordinary", "d2")},
              "attention_attribution_correction": load(raw / "analysis-v2", "attention-attribution-correction.json"),
              "operator_budgets": load(raw, "operator-budgets.json"),
              "native_group_observation": load(raw, "group-parity.json"),
              "observer_perturbation": load(raw, "observer-perturbation.json"),
              "group_costs": {label: load(raw / "analysis-v3", label + ".acceptance-cost.json")
                              for label in ("native-d2-trace", "native-d2-groups-trace", "native-d2-readonly-trace")},
              "captures": load(raw, "capture-audit.json"),
              "current_qkv_counters": load(raw, "qkv-counter-summary.json"),
              "current_reference_dispatch": load(current, "actual-dispatch.json"),
              "current_reference_replays": {mode: load(current, "llama-replay-" + mode + "-v2.json") for mode in ("ordinary", "d2")},
              "qualification": qualification,
              "ap2_decision": {"go_for_bounded_trial": True,
                               "basis": "source + current SASS + actual R3 local/global inputs, grids and counters establish empty-CTA work; no eliminated-time or speedup prediction",
                               "scope": "flatten useful Q/K/V CTA ranges only; preserve arithmetic, formats, layouts, row packing/padding and all other dispatch paths",
                               "implementation_started": False},
              "limitations": ["Native hybrid Trellis35/FP8/BF16/NVFP4 with FP8 KV versus Google QAT Q4_0 with Q8_0 KV; different generated tokens/routes; no quality-equivalence or equal-work claim.",
                              "Reference depth/acceptance histograms come from server-counter algebra, not per-group event timing/routes.",
                              "Observer/profiler/replay/operator rates are diagnostic only. Do not add costs across captures or treat sync API sums as additional CPU budget.",
                              "Input-observer full-model memcheck timed out; focused exact-kernel/slot/actual-byte memcheck passed. Group observer passed full-workflow scoped memcheck.",
                              "Transient high-utilization/no-compute-process state after timed-out sanitizer cleared following owner reboot; driver causality was not established.",
                              "No new Windows, 12B, Vision or internal NVFP4 runtime qualification: production src/include/CMake are unchanged.",
                              "The last feedback quantization/projection costs about 0.0774% of its native D2 profiler window; AP4 is a small-budget candidate, not the whole Assistant projection budget."]}
    raw_files = []
    for directory in (raw, current):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts:
                raw_files.append(identity(path))
    record["raw_manifest"] = raw_files
    record["generator"] = identity(Path(__file__).resolve())
    replacements = [(str(ROOT), "${REPO_ROOT}"),
                    (str(Path.home() / "Development/_engines/llama.cpp"), "${LLAMA_CPP_CENTRAL_ROOT}"),
                    (str(Path.home()), "${HOME}"), (socket.gethostname(), "${HOST}"),
                    (getpass.getuser(), "${USER}")]
    record = sanitize(record, replacements)
    encoded = json.dumps(record, indent=2) + "\n"
    for personal, _ in replacements:
        if personal and personal in encoded:
            raise ValueError("personal identifier survived sanitization")
    comp = latest["comparisons"]
    ordinary, d2 = comp["ordinary"], comp["d2"]
    table = "\n".join(f"| {mode} | {comp[mode]['native_median_tps']:.3f} | {comp[mode]['llama_median_tps']:.3f} | {100*comp[mode]['llama_over_native']['ratio_minus_one']:.3f}% |" for mode in ("ordinary", "d2"))
    observed = record["group_costs"]["native-d2-groups-trace"]
    acceptance_table = "\n".join(
        f"| {accepted} | {observed['by_accepted'][str(accepted)]['core_group_ms']['count']} | "
        f"{observed['by_accepted'][str(accepted)]['core_group_ms']['median']:.6f} | "
        f"{statistics.mean(statistics.mean(g['expert_unions_by_layer']) for g in observed['groups'] if g['accepted'] == accepted):.3f} |"
        for accepted in range(3))
    artifact_link = "../../../" + str(args.output.resolve().relative_to(ROOT))
    report = f"""# AP1 decode measurement refresh — 2026-11-04

Generated by `tools/write_decode_ap1_evidence.py`. AP1 is complete; production
`src/`, `include/` and CMake remain at the accepted parent. AP2 has not started.
The [complete sanitized record]({artifact_link}) retains both baseline campaigns,
all 160 primary samples, commands, source/model/binary/library hashes, diagnostics,
counter summaries, qualification limitations and a hashed private-raw manifest.

## Current unprofiled baseline

Official central llama.cpp `{latest['inputs']['new_llama_commit']}` was verified
against upstream before each invocation. The earlier `{first['inputs']['new_llama_commit']}`
campaign remains separate: master advanced during AP1 and the freshness guard
stopped a diagnostic before inference. No mixed-revision cohort is published.

| Mode | gem16 median tok/s | llama.cpp median tok/s | llama throughput advantage |
|---|---:|---:|---:|
{table}

Each mode ran native/llama/llama/native, three discarded warmups and ten retained
samples per cohort, after cooldown to <=50°C, without observers/profilers. Every
sample has 16,384 prompt tokens, 1,135 outputs and 1,134 decode intervals. Greedy
seed 0, identical prompt IDs and suppression, cold cache, one slot, all weights
resident on GPU. Requested context 17,519 (llama padded 17,664), threads 8,
batch 2,048 / ubatch 512, Flash Attention and graphs. Native outputs, arenas,
dispatch and zero fallback/token-loop allocation remain exact to the qualified
fixture. Dynamic Boost clocks are measured, not locked; all samples stay retained.

The native time reductions needed to match these rates are
**{ordinary['required_native_ms_per_interval']:.6f} ms/ordinary interval** and
**{d2['required_native_ms_per_group']:.6f} ms/native D2 group**, keeping native's
502 groups. Native format is hybrid Trellis35 W4A8 plus FP8/BF16/NVFP4 and FP8 KV;
reference is Google QAT Q4_0 with Q8_0 KV. Tokens/routes differ between engines.
This is a performance reference, with no quality-equivalence or causal
day-over-day code-speedup claim. Exact bootstrap, order and monitoring controls
are in the record.

## Corrected measurement basis

* Full template instances distinguish 256-dimensional local and 512-dimensional
  global split/merge kernels. In the fresh Ordinary trace, 5,670 global merges
  totaling 45.849949 ms move out of the historical local label; total attention
  and primary throughput are unchanged. Unknown instances fail the audit.
* GPU busy/idle and exclusive/overlap unions balance in every complete native
  capture. Source-defined MoE fork/join spans charge the terminal branch once;
  these spans overlap category budgets and are not additional time. Synchronize
  API time overlaps GPU execution and is not an extra CPU saving budget.
* Helper telemetry separates configured, observed and derived speculation work.
  Missing depth/fallback data stays null. Native directly observed 502 groups,
  depth two, acceptance [138,96,268] and 1,004 proposals. Reference distributions
  in the record are explicitly derived from its cumulative server counters.
* Group costs use same-capture evidence only. The graph observer adds 32 kernels
  per group (16,064 extra kernels; 3,347,580 extra device bytes) and directly
  records depth/acceptance/position/expert IDs. Its diagnostic profile window
  was 14.149529 ms / 0.244% longer than the plain trace; this single-capture
  contrast is not isolated causal observer overhead under Dynamic Boost. The
  read-only proposal trace adds no graph nodes and derives acceptance/position,
  with routes unknown. The plain trace has no per-group joint observation.
  All these timings are diagnostic, separately scoped from primary samples.
* The replay label now records the compiled commit and hashes actual loaded
  llama/ggml/CUDA modules. Current ordinary teacher-forcing and D2 forced
  acceptance/commit/continuation checks are retained with their exact scope.

The direct group-observer capture provides this joint diagnostic view:

| Accepted drafts | Groups | Median core group ms | Mean expert union per layer |
|---:|---:|---:|---:|
{acceptance_table}

Each native group still verifies three rows, regardless of later acceptance.
The complete position and per-layer union arrays are retained. Differences
between these conditional summaries are not causal acceptance costs: context,
tokens, routes, profiler effects and clocks are not held constant per case.

## Findings and next experiment

All four Pro source findings hold. Current QKV SASS stages shared activations
and executes a CTA barrier before inactive-row exit. Real local/global R3 grids
launch 384/512 CTAs for 256/288 useful blocks: 2,168,640 empty CTAs per native D2
decode. Actual inputs, resources and current counter reports are retained.
Removing 33/44% of CTAs is **not** a prediction of equivalent time or DRAM savings.

The unmodified native D2 trace spends 0.990881 ms/group in R3 QKV, 0.231496 in
Assistant R1 global splits and 0.214751 in Target R3 attention postnorm. Ordinary
global R1 splits cost 0.567109 ms/interval and postnorm 0.267167. These are single
capture budgets, not measured removable fractions or additive primary savings.
The final redundant feedback quantization/projection is only 0.008932 ms/group,
about 0.0774% of that trace's wall time; lower AP4's priority accordingly.

**Go for one bounded AP2 trial:** flatten only useful R3 Q/K/V CTA ranges.
Preserve all numerical/layout/packing/scale boundaries. Require real-input bit
parity, resources and sanitizers, protected profiles and adjacent unprofiled
AB/BA evidence before acceptance. Use production-matching SM120a/RDC=true and
one candidate against its qualified parent. D2's lower 95% bootstrap bound must
be positive in both orders; predeclare Ordinary non-regression before testing
and retain every sample. No AP2 implementation or speedup is claimed.

## Validation limits

Targeted host fixtures, all six replay semantic comparisons, 48 bit-exact FP8
parent cases and 160 real input payloads are recorded. The graph observer's
full-workflow scoped memcheck passed with exact groups/proposals/routes/outputs.
The input observer's full-model memcheck timed out; it did not pass. A focused
check of the **same** copy/marker kernels and aligned slot arithmetic passed
against 445,539,852 captured bytes, boundary lengths and all 28 marker slots.
This is not a new full-model memory qualification. The failed attempt is kept.

After that timeout, telemetry showed high utilization without a compute process.
The owner rebooted; subsequent idle and diagnostic checks passed. No driver
causality is established. No new Windows/media/12B/internal-NVFP4 qualification,
release, commit or push is part of AP1. Technical hardware/toolchain data remain;
personal paths, hostname, PIDs and GPU UUIDs are removed from the public record.
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    readme_path = ROOT / "README.md"
    readme = readme_path.read_text()
    marker = "<!-- AP1-DECODE-MEASUREMENT-2026-11-04 -->"
    if marker in readme:
        raise ValueError("never overwrite an existing AP1 README publication")
    anchor = "<!-- CENTRAL-LLAMA-REPEAT-2026-11-03 -->"
    if readme.count(anchor) != 1:
        raise ValueError("README benchmark anchor is missing or ambiguous")
    report_link = str(args.report.resolve().relative_to(ROOT))
    section = f"""{marker}
The **2026-11-04 AP1 measurement refresh** compares the unchanged accepted
native runtime with current central llama.cpp `{latest['inputs']['new_llama_commit'][:12]}`.
Unprofiled ABBA medians (20 retained samples per engine/mode) are:

| Mode | gem16 tok/s | llama.cpp tok/s | llama throughput advantage |
|---|---:|---:|---:|
{table}

Both engines use the same 16,384 prompt IDs, 1,135 output positions and 1,134
decode intervals. Native hybrid Trellis35 / FP8 KV and Google QAT Q4_0 / Q8_0
KV differ, as do output streams; this is no quality-parity claim. AP1 corrects
diagnostic attribution/provenance and preserves current operator inputs/counters;
it makes no runtime optimization claim. Both complete baseline campaigns,
same-capture D2 group costs, sanitizer scope/timeout and the bounded AP2 decision
are in the [AP1 evidence]({report_link}).
<!-- /AP1-DECODE-MEASUREMENT-2026-11-04 -->

"""
    with args.output.open("x") as stream:
        stream.write(encoded)
    with args.report.open("x") as stream:
        stream.write(report)
    readme_path.write_text(readme.replace(anchor, section + anchor, 1))
    print(args.output)
    print(args.report)


if __name__ == "__main__":
    main()

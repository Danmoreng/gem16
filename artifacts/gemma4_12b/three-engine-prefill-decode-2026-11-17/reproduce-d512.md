# Reproducing the isolated D512 diagnostics

Use an isolated checkout of the pre-change source revision stated in the main
evidence. Apply the companion native source patch to reproduce the dirty engine
state for the benchmark, then use its CUDA Release build. The compile receipt
identifies the original archive. A rebuilt archive can be identified by its own
hash and checked against the recorded selected-kernel instruction words. The
standalone generator expects the original D512 source verified by that receipt.

Create checked RAM scratch for the two temporary executables. Extract the
diagnostic source and scripts already stored in the raw receipt archive.
The following example stages both harnesses and leaves only durable result
records in the selected output directories when the scratch helper exits.

```sh
python3 tools/experiment_scratch.py --label d512-local-replay --expected-gib 2 -- \
  bash -s -- '{scratch}' <<'SH'
set -euo pipefail
scratch=$1
artifact=artifacts/gemma4_12b/three-engine-prefill-decode-2026-11-17
result=benchmarks/results/d512-local-replay
mkdir -p "$result"
python3 - "$artifact/samples-and-receipts.json.gz" "$scratch" <<'PY'
import gzip
import json
from pathlib import Path
import sys

archive, scratch = Path(sys.argv[1]), Path(sys.argv[2])
with gzip.open(archive, "rt") as stream:
    recorded = json.load(stream)["files"]
groups = {
    "d512-warp-sharing": (
        "make_candidate.py", "driver.cu", "build.sh", "run.sh",
        "summarize.py", "resources.txt",
    ),
    "d512-e2e": (
        "dispatch.h", "dispatch.cpp", "driver.cpp", "build_e2e.sh",
        "run_e2e.sh", "summarize_e2e.py", "kernel-code-identity.json",
    ),
}
for group, names in groups.items():
    directory = scratch / group
    directory.mkdir()
    for name in names:
        content = recorded[f"native/{group}/{name}"]["content"]
        if not isinstance(content, str):
            content = json.dumps(content, indent=2) + "\n"
        (directory / name).write_text(content)
supplement = recorded["native/d512-e2e-cold130k/run_cold130k.sh"]["content"]
(scratch / "d512-e2e" / "run_cold130k.sh").write_text(supplement)
for name in ("build_short.sh", "run_short_sweep.sh", "short-kernel-identity.json"):
    content = recorded[f"native/d512-e2e-short/{name}"]["content"]
    if not isinstance(content, str):
        content = json.dumps(content, indent=2) + "\n"
    (scratch / "d512-e2e" / name).write_text(content)
PY
operator="$scratch/d512-warp-sharing"
e2e="$scratch/d512-e2e"
python3 "$operator/make_candidate.py" \
  src/cuda/attention/prefill_global_sm120.cu "$operator/d512_candidate.cu"
bash "$operator/build.sh" "$operator" > "$operator/build.log" 2>&1
bash "$e2e/build_e2e.sh" "$e2e" "$operator" > "$e2e/build.log" 2>&1
bash "$operator/run.sh" "$operator" "$result/operator"
bash "$e2e/run_e2e.sh" "$e2e" "$result/e2e" no
bash "$e2e/run_cold130k.sh" "$e2e" "$result/cold130k"
mkdir "$scratch/frozen-short"
cp build/Linux/blackwell-release/lib/libgem16_core.a \
  build/Linux/blackwell-release/lib/libgem16_json.a \
  "$operator/d512_candidate.o" "$scratch/frozen-short/"
bash "$e2e/build_short.sh" "$e2e" "$scratch/frozen-short" > "$e2e/short-build.log" 2>&1
bash "$e2e/run_short_sweep.sh" "$e2e" "$result/short"
SH
```

The final argument `no` omits the additional near-limit pair. With `yes`,
the runner starts that pair only within the recorded daily-work deadline.
The two `resources.txt` and `kernel-code-identity.json` receipts staged by the
example describe the original capture. Reproduce the instruction comparison
by locating the relevant template instances with `cuobjdump -arch sm_120a
-symbols` in each build. Run `cuobjdump -arch sm_120a -sass -fun` on those
symbols and hash the newline-separated hexadecimal encoding words.
The compiled object under test is shared across the operator and resident
harnesses. The linker wrapper routes all eight 12B D512 prefill layers per
chunk to the selected entry point, and the runtime receipt verifies every
expected launch.

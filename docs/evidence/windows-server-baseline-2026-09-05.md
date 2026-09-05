# Windows server-first baseline after the Linux handoff

Candidate: `491f5f502a109f14bb7bd5d755f25c0d00bfc7ec`, fast-forwarded from
`a67c37f` on a clean `main`. No runtime, model, precision or context policy changes
were needed to build or pass this baseline. These are Windows results, separate
from the retained Linux evidence.

## Results

| Check | Result |
|---|---|
| `scripts/build.ps1 -Test -Jobs 4` | 5/5 passed, 23.47 s |
| `scripts/build.ps1 -Cuda -Test -Jobs 2` | 11 passed, 4 optional model tests skipped, 74.78 s |
| Native Studio Release build / CTest | 4/4 passed, including real WebView2, 14.23 s |
| Python release-gate and Hub-cache tests | 2 + 2 passed |
| Pinned TypeScript SDK typecheck | Passed |
| Public-profile SDK/Pi matrix | All five checks passed for each of 12B and Compact Vision |
| Separate Pi manual compaction / new session | Passed for both public profiles |
| Real `gem16-12b-m22-product` | Passed separately with the local model, 14.11 s |
| Fresh headless Windows ZIP | Manifest hashes and extracted EXE version passed |

The matrix covers Python SDK 2.50.0, Node SDK 7.10.0, unmodified Pi 0.85.0,
server hardening/recovery and multi-image conversation checks. It uses 16,384
context, one slot and fixed-D2. The Pi compaction probe uses the same configuration
and verifies retention of the codeword followed by a new session. It does not
qualify fork/resume or automatic compaction.

The GPU is the RTX 5080 Laptop; exact driver and runtime memory accounting are in
the matrix report. Actual local tools: MSVC 19.44.35219, CUDA 13.3.33, Python
3.14.2 and Node 25.2.1. These differ from the Linux reference lock's exact versions;
they are recorded rather than described as an exact toolchain-lock reproduction.
All five required model-component directories were found in the existing shared
Hub cache. No new model downloads or lock changes were made.

## Reproduction and retained evidence

Build logs, Studio tests, compiler versions and the package manifest smoke are in
[`windows-host`](../../artifacts/server-hardening/2026-09-05-195025-windows-host/).
The matrix includes exact server/client commands, source/binary/lock hashes,
health/memory data, transcripts, negative API responses and artifact hashes:
[`matrix result`](../../artifacts/server-hardening/2026-09-05-195025-windows-baseline/result.json).
Additional results:
[`manual compaction`](../../artifacts/server-hardening/2026-09-05-195025-windows-compaction/result.json),
[`12B product`](../../artifacts/server-hardening/2026-09-05-195025-windows-12b-product/m22-12b-product.json).

```powershell
py -3.14 -m venv .venv-agent-core
.\.venv-agent-core\Scripts\python.exe -m pip install -r tools/requirements-openai-sdk.txt
npm.cmd ci --prefix tools/openai-sdk --ignore-scripts
npm.cmd --prefix tools/openai-sdk run check
npm.cmd ci --prefix tools/pi-agent --ignore-scripts
$env:PYTHONUTF8 = '1'
.\.venv-agent-core\Scripts\python.exe tools/run_agent_core_matrix.py --server build/Windows/blackwell-release/bin/gem16-server.exe --pi-cli tools/pi-agent/node_modules/@earendil-works/pi-coding-agent/dist/cli.js --output-dir <new-evidence-directory> --port 18083
```

Git Bash is installed at `C:\Program Files\Git\bin\bash.exe`. The retained
`windows-host/compaction_probe.py` wraps the existing `tools/check_pi_compaction.py`
with serial model startup and cleanup; choose fresh output directories for repeats.
The package is local at
`build/packages/windows-491f5f5-20260905/gem16-server-0.2.0-dev-windows-x64.zip`.
It is an unqualified development candidate, not published or clean-machine tested.

## Remaining work

C02 exception injection, C03 long-prefill abort and measured graceful Windows
shutdown, C04 peak-memory stress, full C05 fork/resume/automatic compaction,
C06 remaining semantics and C07/C08 clean-machine and release qualification stay
open. This short-context matrix does not qualify the 170K everyday setting or
ordinary Compact Vision decode. Windows ASan/UBSan remains unsupported by the
build script; retained Linux sanitizer results are not Windows sanitizer results.
No release, new model publication, commit or push was performed in this pass.

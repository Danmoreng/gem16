# Native Chat penalties: Windows evidence (2026-09-30)

Implementation and bounded Windows checks pass for both public profiles, ordinary
and fixed-D2. This is not a two-platform release qualification. Linux, the
installed internal NVFP4 model regression, and everyday-context/media execution
with the new feature remain unverified.

## Behavior and files

Chat Completions accepts finite frequency/presence coefficients in `[-2, 2]`,
default zero. The GPU subtracts frequency times current-response output count
plus presence once for seen outputs, after multiplicative repetition and before
temperature/filtering. Prompt tokens, previous responses and rejected drafts do
not enter these counts. Reasoning and control outputs do. Counters reset each
response, while KV and RNG progress remain resident.

Changed coefficients replace only executables that capture sampling scalars;
unchanged coefficients reuse them. Zero uses the original GPU sampling kernels
and graph topology. Active penalties add one count commit per ordinary sample
or accepted speculative group. No GPU penalty control block was introduced.
Counters reserve 1 MiB per slot; the sampled aggregate GPU footprint increased
by 2 MiB per measured configuration, including allocator granularity.

The bounded request-boundary graph reconstruction exception and superseded
neutral-only decision are recorded in [active decisions](../ACTIVE_DECISIONS.md).
Other sampling controls remain startup/session options; Responses is unchanged.
Greedy rejects active values before generation/SSE, and invalid values fail
visibly rather than being ignored.

Changed boundaries:

- `include/gem16/{sampling,chat,engine}.h` and runtime sampling/chat/result/stub:
  coefficients, validation, native request propagation and diagnostics.
- `src/server/openai_chat.cpp`, `src/cli/server_main.cpp`: parsing and greedy
  admission; `src/util/json.cpp`: reject non-zero FP64 underflow instead of zero.
- `src/cuda/sampling/sampling.{h,cu}`: separate active logit kernel, current-output
  counts, speculative prefixes, accepted commits and forced-output correction.
- `src/cuda/engine/inference_engine*` and `detail/gemma4_26b*`: preallocated
  counters, request reset, bounded graph replacement, all existing model paths.
- `src/cuda/inference_session.cuh`: preserve resident prefix/RNG, apply per-turn
  coefficients, correct forced reasoning outputs and poison failed reconfiguration.
- Unit/CUDA tests, API/server contracts and `tools/validate_chat_penalties.py`.

## Exact builds and checks

PowerShell build setup:

```powershell
. .\scripts\windows-toolchain.ps1
Import-Gem16VisualStudioEnvironment
Import-Gem16CudaEnvironment
cmake --build --preset blackwell-release --target gem16-server gem16-unit-tests gem16-cuda-tests --parallel 4
cmake --build --preset host-debug --target gem16-unit-tests gem16-server --parallel 4
```

Both builds passed. Toolchain: MSVC 14.44.35207, nvcc 13.3.33, CUDA toolkit
13.3, driver 596.49, RTX 5080 Laptop GPU, 16,303 MiB. Existing compiler warnings
were retained. HEAD is `3d35adaddd4dca9fc2b3d96ec811bf7643a11c17` with uncommitted
patches. The measured parent includes the earlier neutral HTTP compatibility
patch; its executable and exact patch were retained before native implementation.

```powershell
ctest --preset host-debug -R '^(gem16-unit|gem16-server-version)$' --output-on-failure
ctest --preset blackwell-release -R '^(gem16-unit|gem16-server-version)$' --output-on-failure
.\build\Windows\blackwell-release\bin\gem16-cuda-tests.exe sampling
.\build\Windows\blackwell-release\bin\gem16-cuda-tests.exe
& 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/compute-sanitizer/compute-sanitizer.exe' --tool memcheck --target-processes application-only --launch-timeout 10 --kill --print-session-details --port 52387 --error-exitcode 99 --log-file artifacts/raw/chat-penalties-20260930-memcheck-final.log build/Windows/blackwell-release/bin/gem16-cuda-tests.exe sampling
```

Host debug: 2/2, 47.34 s. CUDA-enabled release host: 2/2, 3.10 s. Sampling
and full CUDA suite passed; memcheck reports **0 errors**. Synthetic checks cover
the CPU formula, positive/negative frequency/presence, multiplicity, suppression,
temperature ordering, speculative prefix reads without draft commits, accepted
counts, forced-output correction, fixed captured scalars and graph node counts.
Initial sanitizer launch attempts failed to attach or had an invalid option;
only the final successful log supports the sanitizer result.

Server evidence commands:

```powershell
python tools/validate_chat_penalties.py --server artifacts/raw/chat-penalties-20260930-parent/gem16-server-parent.exe --output-dir artifacts/raw/chat-penalties-20260930-parent-10 --source-patch artifacts/raw/chat-penalties-20260930-parent/source.patch --baseline --warmups 3 --samples 10 --profiles 12b compact
python tools/validate_chat_penalties.py --server build/Windows/blackwell-release/bin/gem16-server.exe --output-dir artifacts/raw/chat-penalties-20260930-candidate-10 --warmups 3 --samples 10 --profiles 12b compact
python tools/validate_chat_penalties.py --server build/Windows/blackwell-release/bin/gem16-server.exe --output-dir artifacts/raw/chat-penalties-20260930-greedy --greedy --warmups 1 --samples 1 --profiles 12b
python tools/validate_chat_penalties.py --server build/Windows/blackwell-release/bin/gem16-server.exe --output-dir artifacts/raw/chat-penalties-20260930-lifecycle --lifecycle --warmups 1 --samples 1 --profiles 12b compact
```

All completed. Both public models preserve cached prefixes across unchanged
active coefficients, coefficient changes and return to zero. Ordinary/D2 HTTP
choices and usage agree exactly over all four resident turns per profile.
Streaming with reasoning and combined penalties terminates with usage/DONE.
Greedy returns JSON HTTP 400 for active penalties, even when streaming was
requested, and accepts zero. Native tool calls and tool-result continuation with
a coefficient change pass on all four public profile/mode combinations; each
continuation reports 112 cached tokens. Disconnect after actual streamed content
followed by a new successful generation also passes in all four configurations.
This is a local function-tool smoke, not a new external-agent qualification.

## Zero-path comparison

Same machine, locked models, seed 42, one slot, 4096 context capacity, same
prompt and max 96 completion tokens. Three warmups and ten retained samples per
profile/mode. Timing is HTTP end-to-end, including stateless session setup and
prefill; these numbers are not isolated decode throughput.

| Profile/mode | Parent median ms | Candidate median ms | Latency change |
|---|---:|---:|---:|
| 12B ordinary | 1837.83 | 1834.83 | -0.16% |
| 12B D2 | 1068.09 | 1072.05 | +0.37% |
| Compact ordinary | 719.87 | 728.97 | +1.26% |
| Compact D2 | 560.07 | 557.88 | -0.39% |

Every zero-case HTTP choice and usage matches the parent. Raw model token IDs
are not exposed by this HTTP API, so this is not a raw-ID comparison. Means,
standard deviations and Student-t 95% mean intervals (df=9) are retained in the
[summary](../../artifacts/sampling/chat-penalties-windows-2026-09-30/summary.json). All four pairs
of mean intervals overlap; this screen does not prove absence of arbitrarily
small regressions. Power, clocks and thermals fluctuate on this laptop and are
retained with all samples. VRAM peaks are aggregate GPU observations sampled at
approximately 200 ms plus query overhead, not instrumented process peaks.

`cuobjdump --dump-sass` of parent and final candidate confirms identical
instruction hashes and counts for `PrepareSamplingLogitsKernel` (136),
`PrepareSamplingProbabilitiesKernel` (48) and
`SampleCumulativeProbabilitiesKernel` (176). The active path uses a separate
kernel; the synthetic captured graph has exactly one additional commit node.

## Retained evidence and open qualification

[Parent](../../artifacts/sampling/chat-penalties-windows-2026-09-30/parent.json),
[candidate](../../artifacts/sampling/chat-penalties-windows-2026-09-30/candidate.json),
[lifecycle](../../artifacts/sampling/chat-penalties-windows-2026-09-30/lifecycle.json),
[greedy](../../artifacts/sampling/chat-penalties-windows-2026-09-30/greedy.json),
[full CUDA](../../artifacts/sampling/chat-penalties-windows-2026-09-30/cuda-full.txt),
[memcheck](../../artifacts/sampling/chat-penalties-windows-2026-09-30/memcheck.txt.gz),
[SASS](../../artifacts/sampling/chat-penalties-windows-2026-09-30/sass-comparison.json).
Raw reports retain source/patch/binary/runner/lock hashes, commands, health,
responses, headers and GPU observations. Existing smaller screens remain intact.

The internal NVFP4 server run failed before generation because its locked local
target/tokenizer is not installed. No substitute model or download was used.
Before a two-platform production claim, run the applicable Linux host/CUDA/
sanitizer/model/HTTP checks and internal NVFP4 ordinary/D2 regression. Everyday
context and media were not requalified here. Graph replacement cost is borne
only by changed coefficients, but its latency was not isolated from request
latency; no graph-switch timing or active-penalty speed claim is made.

## Owner promotion decision

After reviewing these results and limitations, the owner explicitly authorizes
commit/push directly to production main, with Linux build/test on this same
machine after reboot. This supersedes completing the outstanding platform/model
checks before this main integration; it does not classify them as passed or
authorize a new tagged binary release. Compact raw reports, compressed source patches and
sanitizer/dispatch evidence above are included in Git; large local binaries and
full disassembly dumps remain in ignored raw storage.

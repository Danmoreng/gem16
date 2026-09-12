# C04 bounded single-user Windows probe — 2026-09-12

The owner-limited scope was executed: **eight generation requests total, 75.7
seconds elapsed**, including startup, an initial harness stop and resumption.
No additional retries, parallel requests, downloads or extended stress runs occurred.
Runtime/model code was not changed. C04 is not entirely green: a 12B image-answer
finding remains open.

## Setup and checks

Source `3d3daf65f3b5a92e4f5f6d2e5a229fc33875d2f9`, the existing C03-qualified
Windows server, local immutable model locks, one slot, ordinary decode, 32K context,
greedy sampling, thinking off and 64 maximum output tokens. Both profiles ran
sequentially. Existing generated 128x128 solid-color PNGs from
`tools/check_multi_image_conversation.py` were reused: red, green, blue, yellow.
Each profile received fresh 1-, 2- and 4-image roots, then one continuation of the
four-image history using the same session and full prior assistant response.

The harness samples Windows `GetProcessMemoryInfo` every 10 ms. Per-request sampled
working-set maxima are separate from the OS lifetime peak, which includes model
loading. Private commit is recorded too; it is not resident physical RAM. Brief
peaks between samples may be missed. This small run does not prove a global memory
bound or absence of leaks under prolonged use.

## Results

| Profile / case | HTTP | Color/order result | Sampled peak working set | Cached tokens |
| --- | ---: | --- | ---: | ---: |
| 12B / 1 image | 200 | Failed: `maroon, maroon` | 228.93 MiB | 0 |
| 12B / 2 images | 200 | Passed: `red, green` | 228.50 MiB | 0 |
| 12B / 4 images | 200 | Failed: `dark red, green, blue` (yellow omitted) | 229.36 MiB | 0 |
| 12B / continuation | 200 | Failed: same incomplete answer | 229.38 MiB | 63 |
| 26B / 1 image | 200 | Passed: `red` | 238.83 MiB | 0 |
| 26B / 2 images | 200 | Passed: `red, green` | 252.67 MiB | 0 |
| 26B / 4 images | 200 | Passed: `red, green, blue, yellow` | 279.74 MiB | 0 |
| 26B / continuation | 200 | Passed: all four colors in order | 306.86 MiB | 1072 |

All eight requests succeeded at the API level, retained the requested session ID,
and left exactly one resident session. Both continuations reused cached tokens
without creating/rebuilding the session. Neither server crashed. Color/order
checks passed 5/8, with all three failures on 12B. Cache reuse does not establish
correct visual understanding or correct semantic recall.

After the final requests, sampled working sets were about 229.38 MiB (12B) and
252.83 MiB (26B). The latter was below its 306.86 MiB request peak. Private commit
peaked at about 11,449 MiB and 13,571 MiB respectively; these process-wide values
include runtime/driver allocations and must not be attributed solely to images.
OS lifetime peak working sets, **including startup**, were about 9,061 MiB for the
resumed 12B process and 446 MiB for 26B. The first 12B request ran in the original
process; its separate raw memory trace is preserved. No broad concurrency-memory
claim or request-memory cap is inferred from these figures.

A narrow source inspection found a relevant profile difference: 12B preprocessing
uses `allow_upscale=false` and aligns a 128-pixel side down to 96 pixels (four 48x48
patches). The observed prompt usage increases by six tokens per additional image.
26B's prompt usage increases by 258 tokens per image. This is a possible diagnostic
lead, **not a proven cause** of the answer errors. No resizing, image budget,
precision or model behavior was changed to make the probe pass. A focused 12B
small-image investigation remains open; do not turn it into a broad C04 campaign
or obscure it as a cache success. C05/C06 remain the next main priority.

## Reproduction and retained evidence

```powershell
py -3.14 tools/check_c04_single_user.py --server build/Windows/blackwell-release/bin/gem16-server.exe --output artifacts/server-hardening/2026-09-12-c04-single-user
py -3.14 tools/check_c04_single_user.py --server build/Windows/blackwell-release/bin/gem16-server.exe --output artifacts/server-hardening/2026-09-12-c04-single-user-remaining --resume-result artifacts/server-hardening/2026-09-12-c04-single-user/result.json
```

The initial harness stopped after the first semantic assertion. The harness was
adjusted to record semantic failures and finish the fixed scope. The second command
executed only the remaining seven requests, carrying forward the original deadline
and first result; no failed request was repeated. For future runs the current
harness records semantic failures and completes all eight in one invocation, so
normally only the first command with a fresh output path is needed. Resume is
restricted to an interrupted first-case result. Existing output directories are
never overwritten. The report deliberately retains `passed: false` alongside
`completed_scope: true`. Process termination is test cleanup, not new shutdown
qualification.

- [Initial failed result](../../artifacts/server-hardening/2026-09-12-c04-single-user/result.json)
- [Combined eight-case result](../../artifacts/server-hardening/2026-09-12-c04-single-user-remaining/result.json)
- [Source, binary and evidence hashes](../../artifacts/server-hardening/2026-09-12-c04-single-user-remaining/provenance.json)

Both directories retain server output and raw timestamped memory samples. Existing
C03 malformed-input, size-limit, abort/recovery and history tests are reused from
[Windows C03 completion](windows-c03-complete-2026-09-12.md), without another GPU
matrix. Python syntax compilation and `git diff --check` passed. No engine rebuild
was necessary. Linux repetition, the 12B finding and other release gates remain
explicitly open; parallel-media stress and CPU image-cache optimization stay deferred.

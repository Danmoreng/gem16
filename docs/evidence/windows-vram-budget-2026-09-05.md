# Windows 26B admission budget investigation — 2026-09-05

Source: `56e83c3906d89a7e4941af37aa3a768e912ae088` plus local Studio fixes
and the owner-authorized 200 MiB long-context admission policy. No numerical
kernels, precision, checkpoint layout or context semantics were changed.

## What consumes the memory

Windows GPU process counters place ChatGPT (about 51 MiB dedicated) and Studio
(about 39 MiB dedicated) on adapter LUID `00000000:0000efe6`, the AMD Radeon
610M. The RTX 5080 Laptop uses LUID `00000000:00010dcc`. There was no separate
active WebView GPU allocation. Thus the observed desktop consumers do not explain
the RTX capacity gap. These are snapshots, not lifetime peaks.

A standalone CUDA/DXGI probe on the RTX, without an inference server, measured:

| Quantity | Bytes |
|---|---:|
| CUDA total physical memory | 17,094,475,776 |
| DXGI local process budget | 15,975,055,360 |
| DXGI current process usage | 176,160,768 |
| CUDA free memory | 15,798,894,592 |

In this observation CUDA free equals **DXGI budget minus current usage exactly**.
The budget is 1,067.6 MiB below CUDA total; this is not evidence that another
application physically holds that entire amount. WDDM budgets are dynamic.
Microsoft documents Budget as the OS-provided target, and warns that exceeding it
can cause stuttering or performance penalties:
https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/ns-dxgi1_4-dxgi_query_video_memory_info

At matching 16,384 context, the retained Linux SDK matrix and a fresh Windows
server report byte-identical Target (12,204,692,480), Vision (597,313,024),
Assistant (258,313,728), KV (272,629,760), workspace (459,064,992), and execution
slot (731,694,752) sizes. Free memory nevertheless differs: Linux 2,607,611,904
versus Windows 1,982,857,216, **595.8 MiB less on Windows**. The same slot
accounting and single-engine ownership show no duplicated model allocation.
Different drivers/platforms remain a limitation of this comparison; it does not
attribute every byte of overhead to a single WDDM mechanism.

## Context probes

All ordinary probes used the open Studio, one slot, the locked Compact Vision
Target, Vision at budget 280, fixed-D2 Assistant, checkpoint sampling, and the
existing 2048-token prefill default. Each successful probe performed a text and
the retained four-fact scene image request.

| Context | Result with 200 MiB reserve |
|---|---|
| 210,000 | Preliminary slot fails: 135,266,304 free bytes |
| 200,000 | Assistant/verifier margin fails |
| 198,000 | Assistant/verifier margin fails |
| 190,000 | Assistant/verifier margin fails |
| 178,000 | Assistant/verifier margin fails |
| 170,000 | Pass; 261,095,424 free bytes (249 MiB) at admission |
| 160,000 | Pass; 378,535,936 free bytes (361 MiB) at admission |

A separately labeled diagnostic tried the existing 1024-token prefill option at
210K and 200K; neither passed. This option was not promoted or persisted.
The earlier 224K/222K/220K failures were recorded with the old 400 MiB preliminary
engine check. No zero-reserve policy has been implemented.

Windows Studio defaults and this local saved configuration now use the measured
170,000-token setting; Linux retains its existing default. This is a tested
operating point, not a proof of the absolute physical maximum or long-context
quality. The 700 MiB short-context/12B and existing final 200 MiB D2 checks remain
unchanged; the former 400 MiB 26B long-context checks now agree on 200 MiB.

## Validation and local reproduction

- `scripts/build.ps1 -Cuda -Test -Jobs 6`: 11 passed, four optional model tests
  skipped; no failures (82.70 seconds).
- `python tests/python/test_qualify_gemma4_26b_m21.py`: 4 passed.
- Real `gem16-12b-m22-product` CTest with the local model: passed, including exact
  tokens, two slots and continuation (13.68 seconds).
- Native Studio Release build and three CTest groups: passed (9.91 seconds).

The exact server commands and responses are in the ignored local
`build/check_gui_context_*.py`, `build/windows-gui-limit-*-20260905.*`, and
`build/windows-gui-context-limits-*.json`. The adapter probe source/binary/log
are `build/windows_cuda_budget_probe.cpp`, `.exe`, and
`build/windows-cuda-idle-budget-20260905.log`. It compiles with MSVC plus the
installed CUDA 13.3 include/lib paths, linking `cudart.lib` and `dxgi.lib`.
The compact comparison is `build/windows-linux-vram-comparison-20260905.json`;
process counters are `build/windows-gpu-process-memory-20260905.json`.
Linux sources are `artifacts/agent-core/2026-09-04-linux-verified/result.json`
and `artifacts/vision/v19-capacity.json`. This is capacity/debugging evidence,
not a benchmark, peak-memory measurement or Windows release qualification.

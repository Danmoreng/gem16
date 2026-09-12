# C03 Windows completion — 2026-09-12

**Windows C03 is complete; the owner deferred Linux execution.** This supplements,
and does not rewrite, the [prefill checkpoint](windows-c03-prefill-2026-09-12.md)
committed as `ffb0a55`. Full release, everyday-context and clean-machine gates
remain separate roadmap items.

## Change and scope

Request-scoped CPU preparation now observes the existing admission deadline,
client disconnect and draining state. The original 30-second budget is shared
across FIFO admission, media preparation and resident-session waits; it is not
reset at a phase transition. There is **no whole-generation 30-second timeout**.
During prefill/decode the applicable interruption mechanisms remain disconnect,
explicit Responses cancel and shutdown.

`src/model/preparation_control.h` supplies the request-thread scope. Both HTTP
adapters install it; base64 processing, image decode/resize/patchification and
audio decode propagate cancellation without publishing partial media or acquiring
a GPU session. RAII releases partial CPU buffers. The callbacks leave image values,
source identities, audio samples, GPU formats, chunk sizes and context unchanged.
Codec calls and individual GPU kernels remain bounded but nonpreemptible; these
are cooperative checkpoints, not a hard real-time latency guarantee.

The existing opt-in `gem16-server-fault-test` target now supports bounded 35-second
holds at named preparation phases and after the second token callback. A counting
prompt ensures an actual decode step precedes the hold. These holds make the
production 30-second deadline deterministic, rather than depending on model EOS.
They poll real cancellation; no fake model or inference result is used. The normal
server has neither the pause metric nor enabled fault headers. Production exclusion
was tested on all four profile/mode combinations.

## Windows results

RTX 5080 Laptop GPU, driver 596.49, 16,303 MiB; pinned local models, 32K capacity,
12B with two slots / 26B with one slot, ordinary and fixed-D2. Exact launch commands,
binary hashes, working-tree source hashes and retained evidence hashes are in
[provenance.json](../../artifacts/server-hardening/2026-09-12-c03-windows-complete/provenance.json).

All **158 controlled cases and 64 production cases passed**. Measurements below
are observed samples, not universal limits or a throughput benchmark.

| Profile | Controlled cases | Production cases | Maximum production prefill disconnect | Maximum natural media disconnect | Maximum control call |
| --- | ---: | ---: | ---: | ---: | ---: |
| 12B ordinary | 42 | 16 | 685 ms | 62 ms | 26 ms |
| 12B D2 | 42 | 16 | 691 ms | 63 ms | 24 ms |
| 26B ordinary | 37 | 16 | 182 ms | 62 ms | 27 ms |
| 26B D2 | 37 | 16 | 187 ms | 48 ms | 26 ms |

- Queue and 12B same-session waiters reached HTTP 503 at 30.004–30.034 seconds;
  media preparation expired at 30.034–30.049 seconds. Expired requests did not
  execute later, and the next request succeeded. Waiters had no fresh timer.
- Both APIs, streaming and nonstreaming, disconnected at request/base64/image
  decode/resize/patchification boundaries; 12B also covered audio. Successful media
  generation followed each phase. Cancelled media continuations retained the
  resident session and demonstrated positive cached-token reuse on recovery.
- Natural production media tests used a compressed 32-million-pixel PNG, without
  fault headers or holds. The harness observed CPU admission before any active GPU
  request, disconnected, and verified admission release with no new session or
  successful input-token accounting. This also exercises the bounded decoder call.
- Production long-prefill probes retained the 12,026-token prompt. Disconnect,
  explicit Responses cancel, saturation and same-session abandonment recovered
  capacity. Maximum full saturated-queue recovery was 865 ms; explicit prefill
  cancellation completed within 712 ms. All measured health/cancel calls met the
  explicit one-second control target (maximum 27 ms, rounded upward).
- Decode disconnect covered both APIs/modes; explicit Responses cancel completed
  with an error event and released the slot. Controlled phase recovery was under
  49 ms, measured **after** the test hold was cancelled, not including the hold.
- Windows Ctrl+C exited with code 0 during media, prefill and decode, including
  queued/same-session waiters during generation. Every case restarted and generated.
  Maximum measured graceful-stop time was 854 ms. Forced termination is only harness
  cleanup and is not counted as graceful-stop evidence.
- Three uncancelled text samples per profile/mode preserved messages and usage
  exactly against the prior C03 checkpoint. No new speed claim is made.

Raw runs: `artifacts/server-hardening/2026-09-12-c03-phases-final-{12b,26b}-d{0,2}`
and `2026-09-12-c03-production-final-{12b,26b}-d{0,2}`. Each contains `result.json`
and `server.txt`.

## Reproduction and regression checks

From the repository root in Windows PowerShell, with the pinned VS/CUDA environment:

```powershell
. ./scripts/windows-toolchain.ps1
Import-Gem16VisualStudioEnvironment
Import-Gem16CudaEnvironment
cmake --build --preset blackwell-release --parallel 2
cmake --build --preset blackwell-release --target gem16-server-fault-test --parallel 2
cmake --build --preset host-debug --parallel 4
ctest --preset host-debug --output-on-failure
ctest --preset blackwell-release --output-on-failure
foreach ($profile in @('12b','26b')) {
  foreach ($draft in @(0,2)) {
    py -3.14 tools/check_c03_phases.py --server build/Windows/blackwell-release/bin/gem16-server-fault-test.exe --output "artifacts/server-hardening/2026-09-12-c03-phases-final-$profile-d$draft" --profile $profile --draft $draft
    py -3.14 tools/check_prefill_cancellation.py --server build/Windows/blackwell-release/bin/gem16-server.exe --output "artifacts/server-hardening/2026-09-12-c03-production-final-$profile-d$draft" --profile $profile --draft $draft --extended --media
  }
}
$env:GEM16_12B_MODEL = py -3.14 -c "import sys;sys.path.insert(0,'tools');from hf_cache import default_target_model;print(default_target_model())"
$env:GEM16_M22_RAW_DIR = 'artifacts/server-hardening/2026-09-12-c03-windows-complete/12b-product'
py -3.14 tests/python/test_gemma4_26b_m22_product.py 12b build/Windows/blackwell-release/bin/gem16-run.exe build/Windows/blackwell-release/bin/gem16-server.exe tests/golden/vllm-gemma4-12b-nvfp4.json
```

Use fresh output directories for reruns; the harness refuses to overwrite evidence.
GPU runs were serialized. Both builds passed. Host Debug: six passed, one intentional
MSVC Debug-STL allocation skip. CUDA CTest: 13 passed, four skips (Windows native
NVFP4 consumption fixture compiler unsupported; three optional model tests had no
fixture environment). Optimized unit and allocation tests passed. Protected 12B
was then run explicitly and passed: exact output `[9503, 106]`, two slots and the
existing audio/product checks. No CUDA kernel changed in this follow-up.

## Retained negative evidence and remaining boundary

The initial phase fixture used a Host header without the bound port and was rejected
before the handler (`2026-09-12-c03-phases-12b-d0`). The next fixture supplied an
already target-sized image, so resize never ran (`...-v2`). The initial unit
fixture made the same resize assumption (`host-initial-negative.txt`). Corrected
fixtures explicitly require resize; all final tests pass. The intermediate `...-v3`
passed but paused at the first token; final runs use the stricter post-decode hold.
The earlier prematurely ending deadline fixture remains unchanged in the original
checkpoint. None of these negatives is silently relabelled as a successful run.

Linux SIGINT/SIGTERM and the applicable Linux phase matrix are still required for
cross-platform C03 closure. Everyday-context stability, general media peak-RSS stress,
sanitizer/full release qualification and clean-machine testing are not established by
this bounded Windows C03 evidence and remain in their owning roadmap gates.

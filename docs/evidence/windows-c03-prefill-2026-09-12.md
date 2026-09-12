# Windows C03 prefill cancellation checkpoint

Date: 2026-09-12. Base: `0b1a33ce428d5405a5df8d4b2676811d81e587c4` plus
the recorded working-tree source changes. This is bounded Windows development
evidence, not completion of C03 on both platforms or release qualification.

## Change

`GenerationCancellation` is a separate, optional function/context callback.
ChatSession forwards it through ConversationSession into both existing specialized
prefill paths. It emits no token event. The 26B check uses the existing per-chunk
stream synchronization. The 12B path synchronizes at existing chunk boundaries
when cancellation is enabled and checks again after final synchronization. GPU
reads finish before cancelled caller-owned input can be released. Prompt IDs,
chunk sizes, image spans, precision, sampling and D2 verification are unchanged.
Callers without a cancellation callback retain the previous enqueue behavior.

Prefill failure poisons the conversation; the existing server lease-discard path
removes it. A new request obtains a fresh working session rather than continuing
partial KV. Nonstream and both streaming transports use their existing cancel /
disconnect checks during prefill as well as decode. Resetting per-request cancel
state now preserves a previously observed shutdown signal, so entering generation
cannot clear a shutdown cancellation already set by the monitor.

## Final live probe

[Raw summary, source/binary hashes and retained-file hashes](../../artifacts/server-hardening/2026-09-12-c03-summary/result.json)
bind the final server to four `2026-09-12-c03-final-*` result directories. Each
includes the exact command, raw responses, metrics and `server.txt`.

- RTX 5080 Laptop GPU, Windows, local immutable model/cache paths.
- Capacity 32,768; 12B two slots, Compact Vision one slot; ordinary and fixed-D2.
- Greedy, 12,026-token text prompt, maximum 32 output tokens. The uninterrupted
  fixture actually generates **two** tokens. No fabricated output-length claim.
- Disconnect occurs 0.4 seconds after an active request is observed. Explicit
  Responses cancellation follows `response.created`, with no output-text delta.
- Health remains available under active prefill and full admission/FIFO capacity.
- Saturation uses a same-session waiter on 12B, fills the FIFO, checks HTTP 503
  overflow, disconnects waiters and verifies no later successful token accounting.
- Each cancellation is followed by successful generation. Ctrl+C uses a hidden
  test-owned Windows console; exit code 0 and `shutdown_completed` are required,
  followed by a real process restart and successful generation.

| Profile | Disconnect to slot recovery, seconds | Explicit cancel to stream end | Ctrl+C to exit 0 |
|---|---:|---:|---:|
| 12B ordinary | 0.661–0.685 | 0.712 | 0.855 |
| 12B fixed-D2 | 0.675–0.689 | 0.716 | 0.866 |
| Compact Vision ordinary | 0.124–0.187 | 0.176 | 0.320 |
| Compact Vision fixed-D2 | 0.186–0.214 | 0.190 | 0.370 |

All observed health/cancel control calls were below the one-second screen
(approximately 1–26 ms). Saturated disconnect recovery was 0.125–0.893 seconds.
These are individual bounded samples, not a universal latency guarantee or
measurements at the 170K everyday capacity. Individual GPU kernels are not
preempted by this implementation.

Reproduce each final row from the repository root, choosing a **new** output path:

```powershell
py -3.14 tools/check_prefill_cancellation.py --server build/Windows/blackwell-release/bin/gem16-server.exe --output <new-directory> --profile 12b --draft 0 --extended
```

Repeat with `--profile 26b` and `--draft 2`. The probe uses the existing locked
cache; it does not download models. Normal final harness termination is only
cleanup and is not counted as graceful-shutdown evidence.

## Regression and build checks

- `scripts/build.ps1 -Cuda -Jobs 2`: passed. Final server relink after the shutdown
  flag fix: `cmake --build --preset blackwell-release --target gem16-server --parallel 2`.
- `scripts/build.ps1 -Test -Jobs 4`: final host run, six passed, one explicit Debug
  allocation-injection skip. The optimized allocation test passed separately.
- `ctest --preset blackwell-release --output-on-failure`: 13 passed, four skipped.
  The native NVFP4 fixture compiler is unsupported by that Windows test; three
  optional real-model tests require environment paths. No skips count as passes.
- The skipped protected 12B test was then invoked explicitly with the local
  `GEM16_12B_MODEL` and a new `GEM16_M22_RAW_DIR`:
  `py -3.14 tests/python/test_gemma4_26b_m22_product.py 12b build/Windows/blackwell-release/bin/gem16-run.exe build/Windows/blackwell-release/bin/gem16-server.exe tests/golden/vllm-gemma4-12b-nvfp4.json`.
  Passed, exact output `[9503, 106]`, two slots and the existing audio/product checks.
- `py -3.14 tools/verify_sm120_sass.py build/Windows/blackwell-release/bin/gem16-server.exe`:
  required NVFP4/FP8 instructions present, 306/298 occurrences. No kernel was changed.

The full CUDA/operator and protected-12B checks preceded the final server-only
shutdown-flag fix. The final live matrix and host rebuild followed that fix.
Logs are retained in the summary directory; the protected-12B raw JSON is in
`2026-09-12-c03-12b-product`.

Three uncached uninterrupted samples per profile/mode were compared with the
published Windows `v0.2.0-dev` binary. Answers and usage match exactly. Median
prefill time differs by +0.13% to +0.96%; recorded arena bytes, configured slots
and admission margins match. This is a short regression screen with no separate
warmup and only two decoded tokens, not a primary throughput/quality claim.
The parent is the CI-built binary, not a local same-toolchain rebuild. No peak
VRAM/RSS or new sanitizer qualification is claimed by the arena comparison.

## Remaining C03 coverage

- Linux live cancellation and SIGINT/SIGTERM qualification.
- Cancellation **within** CPU media preprocessing, and full media/history recovery.
- The complete deadline/phase matrix. A live counting-output blocker ended after
  about 6.4 seconds, so waiters legitimately succeeded instead of reaching the
  existing 30-second admission deadline. The failed fixture is retained under
  `2026-09-12-c03-12b-deadlines`; it is not a server deadline failure or a passed
  timeout test. Use a deterministic blocker for the remaining live expiry test.
  Existing host deadline/pool-wait tests passed. No new whole-generation timeout
  or silent 30-second execution limit was introduced.
- Everyday-context, broader multimodal/sanitizer and final release evidence remain
  separate roadmap requirements. C03 is not marked fully complete.

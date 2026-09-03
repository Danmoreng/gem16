# Windows server startup stack-overflow regression

Scope: startup-only host hashing; no kernels, precision, KV semantics, device
allocations, model files, or saved Studio settings changed by this fix.

Environment: Windows x64, RTX 5080 Laptop (WDDM), NVIDIA driver 596.49,
MSVC 19.44, CUDA 13.3, `blackwell-release`, branch
`codex/gemma4-26b-vision-fp8`, base `3ba13e8228aad399d677a41cb45dc0b80d9c26ba`
plus uncommitted Windows-build/UI changes and this fix.

## Reproduction and cause

Studio's saved Trellis35 Target + FP8 Vision + fixed-D2 Assistant at consolidated
repository revision `6de2a057f11332420819f8e6efd08e42d7a03bc7`, with 228120
context tokens, reproduced exit `-1073741571`. Windows Application event 1000
reports exception `0xC00000FD`; the Windows SDK defines this as
`STATUS_STACK_OVERFLOW`. `dumpbin /headers` reports a `0x100000` (1 MiB) stack
reserve. Both Trellis35 `Sha256File` and compiled Target/Assistant `Sha256Range`
had a 1 MiB local array, leaving no room for their call chain.

The same hashing implementations are now testable through `util/file_sha256.h`
and use bounded temporary heap buffers instead. Hash coverage, chunk sizes,
short-read handling, and existing error messages are preserved. Unit tests run
both paths on the default stack for empty, short, exact-chunk, partial-chunk,
and offset ranges, plus missing files, short reads, and invalid offsets.

## Checks

Executed after importing `scripts/windows-toolchain.ps1` and calling
`Import-Gem16VisualStudioEnvironment` and `Import-Gem16CudaEnvironment`:

```powershell
cmake --build --preset blackwell-release --target gem16-unit-tests gem16-server --parallel 6
ctest --test-dir build/Windows/blackwell-release -R '^(gem16-unit|gem16-server-version)$' --output-on-failure
cmake --build --preset blackwell-release --parallel 6
ctest --test-dir build/Windows/blackwell-release -R 'host-contract|compiler-contract|module-contract' --output-on-failure
```

Builds passed. The first test group passed 2/2 and the contract group passed 4/4.
The final `ctest --test-dir build/Windows/blackwell-release --output-on-failure`
passed all 11 runnable tests (including CUDA, MoE, Trellis35, Vision, attention,
and host contracts), with zero failures. Four existing gates were skipped:
the POSIX-only NVFP4 compiler-consumption fixture and the three unconfigured
checkpoint integration gates (26B M17, 26B M22, 12B M22). The real-checkpoint
server smokes below were run separately and do not replace those full gates.

For live startup probes, `$target`, `$assistant`, and `$vision` below refer to
Studio's existing immutable Hub runtime views, not copied or modified weights:

```powershell
build/Windows/blackwell-release/bin/gem16-server.exe --model $target --assistant-model $assistant --vision-model $vision --model-name gemma4-26b-a4b-trellis35-vision-fp8 --max-context 131072 --mtp-draft-tokens 2 --vision-max-soft-token-budget 280 --port 18080
```

- At 228120 tokens, the fixed server no longer crashes. It reports
  `model_load_failed` / insufficient GPU margin (`free=0`,
  `required_margin=419430400`). This is a separate capacity limitation, not
  evidence of failed checkpoint acquisition. No admission check was weakened.
- At 32768 tokens, startup reaches `server_ready`; `/health` is `ok` with Vision
  and fixed-D2 enabled. A Chat Completions request answers `2 + 2 ist 4.`
- At 131072 tokens, startup reaches `server_ready`, with `device_free_bytes=714080256`
  and `safety_margin_bytes=209715200`. A Chat Completions image request using
  `benchmarks/vision-v19/images/chart-wide.png`, `reasoning_effort=none`, and
  `max_tokens=256` returns `TITLE: Q4 HARVEST`, `BIRCH: 3`, `CEDAR: 7`, `MAPLE: 5`.

These are bounded startup/text/image smokes, not full long-context, quality,
performance, or release qualification. Diagnostic servers were stopped after
testing; Studio configuration still contains the user's original 228120 tokens.

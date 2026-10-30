# Native Chat penalties: Linux evidence (2026-09-30)

Linux builds and bounded functional checks pass at
`0e61daa1eb719692b780a182b71db98f53cccb9f`, pulled from origin/main by
`git pull --ff-only`. The checkout was clean before testing. No engine, loader,
sampling or API source changes were needed. This record adds Linux evidence;
the earlier Windows record remains unchanged.

Environment: Linux 7.2.2-arch1-1 x86-64, GCC 16.2.1, CMake 4.4.3,
CUDA/nvcc 13.3.73, NVIDIA driver 610.57.04, RTX 5080 Laptop GPU,
16,303 MiB. Models are the already installed immutable repository locks.

## Builds and operator checks

All commands below completed successfully:

```sh
cmake --preset blackwell-release
cmake --build --preset blackwell-release --parallel 4
cmake --preset host-debug
cmake --build --preset host-debug --parallel 4
cmake --build build/native-studio --parallel 4
ctest --preset host-debug --output-on-failure
ctest --test-dir build/native-studio --output-on-failure
build/Linux/blackwell-release/bin/gem16-cuda-tests sampling
compute-sanitizer --tool memcheck --error-exitcode 99 --log-file artifacts/raw/chat-penalties-linux-20260930/memcheck.log build/Linux/blackwell-release/bin/gem16-cuda-tests sampling
ctest --preset blackwell-release -E '(gem16-26b-m17-engine-smoke|gem16-26b-m22-product|gem16-12b-m22-product)' --output-on-failure
python tools/validate_sampling.py --run build/Linux/blackwell-release/bin/gem16-run --output artifacts/raw/chat-penalties-linux-20260930/repetition.json
GEM16_12B_MODEL=/home/sebastian/.cache/huggingface/hub/.gem16/snapshots/unsloth--gemma-4-12b-it-NVFP4--b1f649734b34aa5575b03d186abd1b9be3d0d5c4 GEM16_M22_RAW_DIR=/home/sebastian/Development/gem16gb/artifacts/raw/chat-penalties-linux-20260930/12b-product ctest --preset blackwell-release -R '^gem16-12b-m22-product$' --output-on-failure
```

Host Debug: 7/7 passed in 10.45 s. CUDA Release: 14/14 passed in 37.79 s,
including the full CUDA suite and NVFP4, Trellis35, Vision and attention
operator checks. Studio: 3/3 passed in 5.38 s. Separate protected 12B product
regression: passed in 7.01 s, exact golden output, resident continuation,
zero reported fallbacks and token-loop allocations. Sampling memcheck:
**0 errors**. The synthetic penalty tests exercise CPU formula agreement,
positive/negative coefficients, repetition ordering, multiplicity, suppression,
speculative prefixes/accepted counts, forced-output correction and captured
graph scalars/topology.

The checkpoint-backed repetition test uses repetition penalty 1.1 and four
steps: GPU token IDs `[532, 236771, 236771, 236771]` exactly match the CPU
reference and a second seeded run. Frequency/presence are zero in that test.

## Model/API execution

```sh
python tools/validate_chat_penalties.py --server build/Linux/blackwell-release/bin/gem16-server --output-dir artifacts/raw/chat-penalties-linux-20260930/models --lifecycle --warmups 1 --samples 1 --profiles 12b compact nvfp4
python tools/validate_chat_penalties.py --server build/Linux/blackwell-release/bin/gem16-server --output-dir artifacts/raw/chat-penalties-linux-20260930/greedy --greedy --warmups 1 --samples 1 --profiles 12b compact
python artifacts/raw/chat-penalties-linux-20260930/run_nvfp4_view.py
```

The first command completed all four public profile/mode cases, then failed
before internal NVFP4 generation. The greedy command and the final NVFP4
wrapper completed successfully.

All three profiles pass ordinary and fixed-D2 checks at context 4096, one slot,
seed 42, one warmup and one retained zero-case sample per configuration:

- Four resident turns exercise frequency 0.5, unchanged frequency 0.5,
  presence -0.5 and return to zero. Later turns retain cached prompt tokens
  and report no cache reset. Ordinary/D2 choices and usage match exactly
  across all four turns for each profile.
- Streaming with low reasoning and combined frequency 0.5/presence 0.25
  ends with a finish reason, usage and DONE.
- Active combined penalties produce native function calls. Tool-result
  continuation with changed coefficients retains the resident prefix.
- Disconnect after actual streamed content is followed by successful
  generation without restarting the server.
- Both public profiles, ordinary and D2, accept zero in greedy mode and
  reject active frequency/presence with JSON HTTP 400 before SSE.

The internal NVFP4 Hub root includes assistant/trellis35/vision directories;
the strict artifact loader rejects these as non-regular entries. A first
isolated view still included `gem16_components.json`, which is Hub metadata
outside the accepted runtime file set and was also rejected. Both failures
are retained. The final wrapper verifies every root lock file's size and
SHA-256, hardlinks the runtime files into a temporary directory under the
same shared Hub cache, and excludes that aggregate Hub metadata and the
other component directories. No model bytes, lock, loader rule, precision or
sampling behavior change. No model download or second payload copy occurs.
The temporary view is removed after the successful ordinary/D2 run. This
does not establish that the unfiltered Hub root works with the default runner.

## Evidence and scope

[Summary](../../artifacts/sampling/chat-penalties-linux-2026-09-30/summary.json)
records toolchain, binary/runner/lock hashes, results and sampled aggregate GPU
peaks. Retained reports include
[public models](../../artifacts/sampling/chat-penalties-linux-2026-09-30/public-models.json),
[internal NVFP4](../../artifacts/sampling/chat-penalties-linux-2026-09-30/nvfp4.json),
[greedy](../../artifacts/sampling/chat-penalties-linux-2026-09-30/greedy.json),
[repetition reference](../../artifacts/sampling/chat-penalties-linux-2026-09-30/repetition.json),
[protected 12B](../../artifacts/sampling/chat-penalties-linux-2026-09-30/12b-product.json),
[memcheck](../../artifacts/sampling/chat-penalties-linux-2026-09-30/memcheck.txt)
and the NVFP4 wrapper/view manifest. Full build and server logs remain in
ignored `artifacts/raw/chat-penalties-linux-20260930/`.

This is bounded Linux functional evidence, not a new release qualification or
performance claim. There is no measured Linux parent comparison, new SASS
comparison, isolated graph-switch timing, everyday-context test or model-media
requalification. HTTP choice/usage equality is not a raw-token-ID comparison.
The separate repetition test does compare raw IDs. Aggregate VRAM observations
are sampled at approximately 200 ms plus query overhead, not instrumented
process peaks. The new per-request API controls are frequency and presence;
repetition remains the existing startup/session configuration.

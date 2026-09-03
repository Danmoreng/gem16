# Third-party dependencies

## Native Studio Markdown and mathematics

Studio (not the CUDA runtime) fetches the following static dependencies from
immutable commits in `nativeStudio/cmake/markdown-dependencies.cmake`:

- `mity/md4c` `729e6b8b320caa96328968ab27d7db2235e4fb47` (release-0.5.2), MIT:
  CommonMark/GFM parsing and the upstream HTML entity lookup table.
- `NanoMichael/MicroTeX` `0e3707f6dafebb121d98b53c64364d16fefe481d`, MIT:
  native formula layout, with an ImGui-specific graphics adapter. CMake creates
  a derived `latex.cpp` fixing upstream's Linux `asprintf`/`delete` mismatch;
  the fetched source is retained unchanged.
- `leethomason/tinyxml2` `1dee28e51f9175a31955b9791c74c430fe13dc82` (9.0.0),
  zlib: MicroTeX's bundled resource metadata reader.

Dependency selection and the formula builder/graphics-adapter approach were
inspected in local `simple-markdown-viewer` at
`1c018b0b3994322bc8af59603808577bf8e8e068` (MIT, Simple Markdown Viewer
contributors). Its Skia renderer is not linked or copied. Studio uses new
md4c callbacks and a new ImGui drawing implementation.

The original dependency licenses are copied beside the binary in `licenses/`.
Unmodified MicroTeX resources are copied into `math-res/`, retaining all font
licenses, including `fonts/licences/{OFL,Knuth_License,License_for_dsrom}.txt`
and the Greek/Cyrillic notices. Fonts are not renamed or modified. Packaging
scripts and CMake install rules include both directories. No browser engine,
JavaScript runtime, TeX executable, or runtime network fetching is required.

Updates require explicit pin changes, license/resource review, both platform
builds, parser/host tests, hostile-input and streaming tests, and inspection
of the generated Markdown/formula rendering preview.

The CUDA runtime uses NVIDIA CUTLASS v4.5.2 at commit
`db1c288993354c88e551c40c19a8fb93a774a241`, pinned as the `third_party/cutlass` Git submodule. CUTLASS is
BSD-3-Clause licensed and provides the SM120 block-scaled NVFP4 Gate/Up prefill GEMM templates. Initialize it with
`git submodule update --init --recursive`. Update the pin only with fresh CUDA build, operator equivalence,
model-logit, memory, and end-to-end prefill evidence. No CUTLASS weight payload or generated plan is persisted.

Outside the pinned dependencies documented here, the runtime uses the C++
standard library, platform file mapping, and the pinned local CUDA toolkit.

Image decoding vendors `stb_image.h` from `nothings/stb` at commit
`31c1ad37456438565541f4919958214b6e762fb4` (header SHA-256
`594c2fe35d49488b4382dbfaec8f98366defca819d916ac95becf3e75f4200b3`).
gem16 selects the bundled MIT license and compiles only PNG, JPEG, and BMP
support. Audio decoding, channel conversion, and resampling vendor
`miniaudio.h` from `mackron/miniaudio` at commit
`9634bedb5b5a2ca38c1ee7108a9358a4e233f14d` (header SHA-256
`ac7af4de748b7e26b777f37e01cee313a308a7296a3eb080e2906b320cc55c89`).
gem16 selects its bundled MIT No Attribution license and disables device I/O,
encoding, resource management, node graphs, engine, and signal generation.
Both license texts are stored beside their headers. These dependencies are
vendored because they provide bounded, cross-platform media decoding without
runtime DLLs or platform-specific codec APIs. To update either dependency,
download the named header and license from an exact upstream commit, update the
commit and SHA-256 here, then rerun Windows and Linux host builds, malformed
input tests, the real audio fixture, and real-checkpoint image qualification.

The OpenAI-compatible HTTP adapter vendors unmodified `cpp-httplib` v0.40.0
from `https://github.com/yhirose/cpp-httplib` at commit
`b7e02de4a70024ed0389e0e7b971f674e4bc7d91`.
The split `httplib.h` and `httplib.cpp` files have SHA-256
`8ed9236947e195950dfdd3448972405ae1dab6d17006002dde73cea2cb520592` and
`2d1935237f72a233930416bc22ecb41939a425ba18da380825550260304aece1`.
It is MIT licensed; the unmodified license is stored beside the sources. The
library is vendored to provide a small cross-platform HTTP/1.1 and chunked SSE
transport without a framework or dynamic runtime dependency. gem16 compiles
it as a warning-isolated static library and does not enable TLS. To update it,
fetch an exact upstream tag/commit, replace both sources and the license,
update hashes here, then rerun Linux/Windows host and CUDA builds, request-limit
tests, disconnect tests, and real non-stream/SSE tool loops.

Build-script structure was adapted from the neighboring `qwen35x` repository (MIT License, copyright 2026 qwen35x
contributors), inspected locally on 2026-07-21. No qwen35x runtime or loader source was copied.

Studio's PipeWire capture preference, input-mixer ranking, and bounded
silence/clipping validation policy are adapted from
`qwen-tts-studio/VoicesViewModel.kt` at commit
`ef2344a702ea056e549dac2fbb6c961b57b5feb2` (MIT License, copyright 2026
Danmoreng). gem16 replaces its file-backed voice-cloning flow with bounded
in-memory PCM/WAV attachment generation and model-request limits.

The llama.cpp baseline fetches `ggml-org/llama.cpp` into the ignored `third_party/cache/llama.cpp` directory and
requires the exact commit recorded in `benchmarks/baselines/llama_cpp/commit.txt`. llama.cpp is MIT licensed. It is
used because it is the project's primary local-inference competitor and provides the comparison CUDA runtime and
HF-to-GGUF converter. The upstream baseline build remains unmodified. A separately labeled, tracked converter patch
is applied only to an ignored worktree by `prepare-patched-source.sh`; its purpose, exact mapping, and SHA-256 are
recorded under `benchmarks/baselines/llama_cpp/`. Update the commit or patch only with fresh conversion,
correctness, instruction-path, residency, and benchmark evidence.

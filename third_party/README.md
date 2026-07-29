# Third-party dependencies

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

Build-script structure was adapted from the neighboring `qwen35x` repository (MIT License, copyright 2026 qwen35x
contributors), inspected locally on 2026-07-21. No qwen35x runtime or loader source was copied.

The llama.cpp baseline fetches `ggml-org/llama.cpp` into the ignored `third_party/cache/llama.cpp` directory and
requires the exact commit recorded in `benchmarks/baselines/llama_cpp/commit.txt`. llama.cpp is MIT licensed. It is
used because it is the project's primary local-inference competitor and provides the comparison CUDA runtime and
HF-to-GGUF converter. The upstream baseline build remains unmodified. A separately labeled, tracked converter patch
is applied only to an ignored worktree by `prepare-patched-source.sh`; its purpose, exact mapping, and SHA-256 are
recorded under `benchmarks/baselines/llama_cpp/`. Update the commit or patch only with fresh conversion,
correctness, instruction-path, residency, and benchmark evidence.

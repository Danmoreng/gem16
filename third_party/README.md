# Third-party dependencies

The CUDA runtime uses NVIDIA CUTLASS v4.5.2 at commit
`db1c288993354c88e551c40c19a8fb93a774a241`, pinned as the `third_party/cutlass` Git submodule. CUTLASS is
BSD-3-Clause licensed and provides the SM120 block-scaled NVFP4 Gate/Up prefill GEMM templates. Initialize it with
`git submodule update --init --recursive`. Update the pin only with fresh CUDA build, operator equivalence,
model-logit, memory, and end-to-end prefill evidence. No CUTLASS weight payload or generated plan is persisted.

The remaining runtime uses the C++ standard library, POSIX file mapping, and the pinned local CUDA toolkit.

Build-script structure was adapted from the neighboring `qwen35x` repository (MIT License, copyright 2026 qwen35x
contributors), inspected locally on 2026-07-21. No qwen35x runtime or loader source was copied.

The llama.cpp baseline fetches `ggml-org/llama.cpp` into the ignored `third_party/cache/llama.cpp` directory and
requires the exact commit recorded in `benchmarks/baselines/llama_cpp/commit.txt`. llama.cpp is MIT licensed. It is
used because it is the project's primary local-inference competitor and provides the comparison CUDA runtime and
HF-to-GGUF converter. The upstream baseline build remains unmodified. A separately labeled, tracked converter patch
is applied only to an ignored worktree by `prepare-patched-source.sh`; its purpose, exact mapping, and SHA-256 are
recorded under `benchmarks/baselines/llama_cpp/`. Update the commit or patch only with fresh conversion,
correctness, instruction-path, residency, and benchmark evidence.

# Command catalog

Commands are suggested names. The agent must adapt them to the actual repository CLI and record exact final commands.

## Repository orientation

```bash
git rev-parse HEAD
git status --short
git diff --stat 1c4287965d318ba32a68e597f9d7b6678b883376...HEAD
git submodule status --recursive
```

## Source locks

```bash
python tools/fetch_model.py --lock models/gemma4-26b-qat-bf16.lock.json
python tools/fetch_model.py --lock models/gemma4-26b-base-bf16.lock.json
python tools/fetch_model.py --lock models/gemma4-26b-unsloth-nvfp4.lock.json
python tools/fetch_model.py --lock models/gemma4-26b-qat-q4_0.lock.json
```

## Inspect

```bash
gem16-inspect --model "$MODEL" --validate
gem16-inspect --model "$MODEL" --json artifacts/manifest.json
python tools/compare_model_manifests.py --left "$A" --right "$B"
```

## Compiler: current implemented lanes

M04 is an accepted Python standard-library control-plane scaffold. Its only encoder is `copy-v1`; it moves bytes and
is not numerical tensor conversion. M05 uses Python only for plan/lock/publication orchestration and the independent
oracle. Its promoted numerical backend is the native C++ `gem16-fp8-compiler`; it fails visibly when unavailable and
has no Python conversion fallback. See [`../specs/NATIVE_CONVERTER_ARCHITECTURE.md`](../specs/NATIVE_CONVERTER_ARCHITECTURE.md)
and the retained llama.cpp study [`../../../evidence/gemma4_26b/m05-llama-cpp-converter-research-2026-08-11.md`](../../../evidence/gemma4_26b/m05-llama-cpp-converter-research-2026-08-11.md).

```bash
export LC_ALL=C.UTF-8 LANG=C.UTF-8
python3 tools/compile_gemma4_26b.py plan \
  --source-lock models/gemma4-26b-base-bf16.lock.json \
  --source-directory models/checkpoints/google-gemma-4-26b-a4b-it-bf16-4d7ae49 \
  --compiler-manifest benchmarks/goldens/gemma4_26b/fp8/ordinary-compiler-plan.json \
  --profile fp8-attention-partial-v1 --head-format deferred \
  --max-host-memory 7516192768 --staging-bytes 1048576 \
  --reference-platform-strict --report artifacts/m05/ordinary-plan.json

python3 tools/compile_gemma4_26b.py compile \
  --source-lock models/gemma4-26b-base-bf16.lock.json \
  --source-directory models/checkpoints/google-gemma-4-26b-a4b-it-bf16-4d7ae49 \
  --compiler-manifest benchmarks/goldens/gemma4_26b/fp8/ordinary-compiler-plan.json \
  --profile fp8-attention-partial-v1 --head-format deferred \
  --native-encoder build/Linux/host-release/bin/gem16-fp8-compiler --threads 8 \
  --max-host-memory 7516192768 --staging-bytes 1048576 --reference-platform-strict \
  --output build/models/ordinary-fp8-attention-partial \
  --report artifacts/m05/ordinary-compile.json

python3 tools/compile_gemma4_26b.py verify \
  --source-lock models/gemma4-26b-base-bf16.lock.json \
  --source-directory models/checkpoints/google-gemma-4-26b-a4b-it-bf16-4d7ae49 \
  --compiler-manifest benchmarks/goldens/gemma4_26b/fp8/ordinary-compiler-plan.json \
  --profile fp8-attention-partial-v1 --head-format deferred --threads 8 \
  --max-host-memory 7516192768 --staging-bytes 1048576 --reference-platform-strict \
  --model build/models/ordinary-fp8-attention-partial --report artifacts/m05/ordinary-verify.json
```

The QAT lane uses the analogous checked plan `benchmarks/goldens/gemma4_26b/fp8/qat-compiler-plan.json` and source
snapshot `models/checkpoints/google-gemma-4-26b-a4b-it-qat-bf16-f1e06dc`. These full 115-matrix runs are not to be
started without the milestone gate and explicit owner approval when the measured native throughput probe projects a
long run.

## Compiler: planned unified native family

`gem16-checkpoint-compiler` is the planned future user-facing native family for M06 NVFP4, M07 Q4_0/NVFP4 head and
M18 large comparisons. It is not a current runnable command and must not be substituted into evidence until its
implementation and CLI contract are accepted. No `--stage nvfp4-mlp` or `--stage embedding-head` command is current.

## Host build/tests

```bash
cmake --preset host-debug
cmake --build --preset host-debug
ctest --preset host-debug
```

## CUDA build/tests

```bash
cmake --preset blackwell-release
cmake --build --preset blackwell-release
ctest --preset blackwell-release
```

## Sanitizers

```bash
compute-sanitizer --tool memcheck build/Linux/blackwell-release/bin/gem16-cuda-tests
compute-sanitizer --tool racecheck build/Linux/blackwell-release/bin/gem16-cuda-tests
compute-sanitizer --tool initcheck build/Linux/blackwell-release/bin/gem16-cuda-tests
```

## Native path

```bash
cuobjdump --dump-sass build/Linux/blackwell-release/bin/gem16-bench > artifacts/native.sass
nsys profile --trace=cuda,nvtx -o artifacts/run \
  build/Linux/blackwell-release/bin/gem16-run --model "$MODEL" ...
ncu --set full --kernel-name regex:Gemma4.* \
  build/Linux/blackwell-release/bin/gem16-bench ...
```

## Correctness

```bash
python tools/capture_gemma4_26b_goldens.py ...
python tools/validate_gemma4_26b_full_model.py ...
python tools/analyze_gemma4_26b_conversion.py ...
python tools/evaluate_gemma4_26b_quality.py ...
```

## Memory

```bash
python tools/report_model_memory.py --model "$MODEL" \
  --contexts 8192,16384,32768,65536

nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

## Benchmarks

```bash
gem16-bench prefill --model "$MODEL" --context 8192 --warmups 3 --repetitions 10
gem16-bench decode --model "$MODEL" --context 8192 --tokens 256 --warmups 3 --repetitions 10
python tools/benchmark_gemma4_26b_cross_engine.py ...
```

## Product

```bash
gem16-server --model "$MODEL" --max-context 32768 --max-sessions 1
python tools/validate_openai_sdk.py --base-url http://127.0.0.1:8080/v1 --model "$MODEL_ID"
python tools/benchmark_server.py ...
```

## Evidence checksums

```bash
find artifacts/mXX -type f -print0 | sort -z | xargs -0 sha256sum > artifacts/mXX/SHA256SUMS
sha256sum -c artifacts/mXX/SHA256SUMS
```

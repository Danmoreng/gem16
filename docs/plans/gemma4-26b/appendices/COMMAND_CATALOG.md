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

## Compiler

```bash
python tools/compile_gemma4_26b.py plan \
  --source-lock models/gemma4-26b-qat-bf16.lock.json \
  --profile sm120-text-hybrid-v1 \
  --head-format q4_0

python tools/compile_gemma4_26b.py compile \
  --source-lock models/gemma4-26b-qat-bf16.lock.json \
  --profile sm120-text-hybrid-v1 \
  --head-format q4_0 \
  --output build/models/gemma4-26b-qat-hybrid \
  --report artifacts/compiler/report.json

python tools/compile_gemma4_26b.py verify \
  --model build/models/gemma4-26b-qat-hybrid \
  --lock models/gemma4-26b-gem16-hybrid.lock.json
```

## Host build/tests

```bash
cmake --preset host-release
cmake --build --preset host-release
ctest --test-dir build/host-release --output-on-failure
```

## CUDA build/tests

```bash
cmake --preset blackwell-debug
cmake --build --preset blackwell-debug
ctest --test-dir build/blackwell-debug --output-on-failure

cmake --preset blackwell-release
cmake --build --preset blackwell-release
ctest --test-dir build/blackwell-release --output-on-failure
```

## Sanitizers

```bash
compute-sanitizer --tool memcheck build/blackwell-debug/bin/gem16-cuda-tests
compute-sanitizer --tool racecheck build/blackwell-debug/bin/gem16-cuda-tests
compute-sanitizer --tool initcheck build/blackwell-debug/bin/gem16-cuda-tests
```

## Native path

```bash
cuobjdump --dump-sass build/blackwell-release/bin/gem16-bench > artifacts/native.sass
nsys profile --trace=cuda,nvtx -o artifacts/run \
  build/blackwell-release/bin/gem16-run --model "$MODEL" ...
ncu --set full --kernel-name regex:Gemma4.* \
  build/blackwell-release/bin/gem16-bench ...
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

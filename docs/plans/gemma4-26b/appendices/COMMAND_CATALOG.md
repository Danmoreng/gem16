# Command catalog — Fast Track R4

Commands are examples; record the exact repository CLI used.

## Orientation and clean-run preflight

```bash
git rev-parse HEAD
git status --porcelain=v1
git submodule status --recursive
```

Accepted full conversions require empty `git status --porcelain=v1`, verified source locks, hashed native executable, available disk/RAM and a fresh staging output.

## Compiler

```bash
python3 tools/compile_gemma4_26b.py plan ...
python3 tools/compile_gemma4_26b.py compile --native-encoder <binary> --threads <N> ...
python3 tools/compile_gemma4_26b.py verify ...
```

M06 uses the NVFP4 expert profile, M07 the NVFP4 head profile and M08 the complete hybrid profile. Q4_0 commands exist only if M24 is activated. M25 may add an assistant profile.

## Build/test

```bash
cmake --preset host-debug
cmake --build --preset host-debug
ctest --preset host-debug
cmake --preset blackwell-release
cmake --build --preset blackwell-release
ctest --preset blackwell-release
```

## CUDA validation

```bash
compute-sanitizer --tool memcheck <cuda-tests> ...
compute-sanitizer --tool racecheck <cuda-tests> ...
compute-sanitizer --tool initcheck <cuda-tests> ...
cuobjdump --dump-sass <binary> > artifacts/native.sass
nsys profile --trace=cuda,nvtx -o artifacts/run <command>
```

## Memory/context

```bash
<inspect> --model "$MODEL" --memory-profile 32768 --json artifacts/plan-32k.json
<run> --model "$MODEL" --max-context 32768 ...
<run> --model "$MODEL" --max-context 65536 ...
```

Admission uses `cudaMemGetInfo` output from the runtime, not `nvidia-smi` subtraction alone.

## Qualification

```bash
<quality-tool> --model "$MODEL" --artifact-lock <lock> ...
<bench> prefill --warmups 3 --repetitions 10 ...
<bench> decode --warmups 3 --repetitions 10 ...
```

M25 adds ordinary-versus-MTP commands with identical target settings and verified-token accounting.

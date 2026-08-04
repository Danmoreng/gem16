# Program definition of done

The 26B program is complete only when every required item below has evidence.

## Governance and provenance

- [ ] Repository policy explicitly permits the scoped QAT-derived 26B artifact.
- [ ] All model sources use full immutable revisions.
- [ ] Every source and output file has size and SHA-256.
- [ ] Compiler version, repository commit, toolchain and quantization plan are recorded.
- [ ] License and attribution files are included in the published artifact.
- [ ] The artifact is clearly labeled project-built, not official Google NVFP4.

## Model contract

- [ ] 30 layers and exact 25-local/5-global sequence validated.
- [ ] Hidden, head, KV, shared MLP, routed expert, vocabulary and context dimensions validated.
- [ ] All 128 experts and top-8 routing represented.
- [ ] Router norm, scale, projection, FP32 softmax, top-k renormalization and per-expert scales match the reference.
- [ ] Shared dense MLP and routed experts use the exact residual/norm order.
- [ ] Global K/V semantics and separate cache states are correct.
- [ ] Vision tensors are absent/nonresident in the first profile.

## Compiler

- [ ] Streaming, bounded-memory compilation works from a clean source cache.
- [ ] FP8 quantization has deterministic byte fixtures.
- [ ] NVFP4 value, local scale and global scale encoding has deterministic byte fixtures.
- [ ] Q4_0 head compilation is either accepted and exact to its spec or rejected by a recorded decision.
- [ ] Ordinary-BF16 and QAT-BF16 controls can be compiled by the same code.
- [ ] Output validates through `gem16-inspect`.
- [ ] Repeated compilation produces identical hashes on the reference environment.
- [ ] No tensor is silently skipped.

## Runtime

- [ ] 12B path remains fully functional.
- [ ] 26B variant is selected once at load.
- [ ] One immutable weight arena, no duplicate layout.
- [ ] One tied embedding/head allocation.
- [ ] One 26B execution slot admitted on the 16 GB card.
- [ ] No token-loop allocation.
- [ ] No CPU expert routing or weight traffic.
- [ ] Missing native kernels fail visibly.
- [ ] Deterministic mode is repeatable.
- [ ] Unsupported image/audio/MTP requests fail cleanly.

## Correctness and quality

- [ ] Host and CUDA unit tests pass.
- [ ] Real-shape FP8/NVFP4 operator tests pass.
- [ ] CPU and CUDA routing agree.
- [ ] Layer-0 and representative local/global layers have captured goldens.
- [ ] Full teacher-forcing suite completes without NaN/Inf.
- [ ] Per-layer residual and router drift is reported.
- [ ] Candidate quality is inside the accepted envelope relative to QAT BF16 and official Q4_0.
- [ ] Ordinary-BF16→own-NVFP4 control explains differences versus Unsloth.
- [ ] Greedy output is deterministic across repeated runs.
- [ ] Task-quality suite has no undisclosed material regression.

## Memory

- [ ] Immutable arena is ≤14,100 MiB or an explicit accepted exception exists.
- [ ] 32K FP8 K/V payload is correctly reported.
- [ ] 32K process peak is reconciled against the approximately 15,881 MiB CUDA-visible capacity; nominal board memory is not used to infer margin.
- [ ] At least 700 MiB directly measured free margin remains at 32K (approximately 15,181 MiB maximum observed use on the current runtime).
- [ ] 64K is either qualified with ≥500 MiB margin or clearly labeled unsupported/experimental.
- [ ] Startup peak is measured.
- [ ] Named allocator accounting reconciles with device measurements.
- [ ] No transient dequantized expert copy.

## Performance

- [ ] Native SM120/SM120a NVFP4 instructions are present and observed at runtime.
- [ ] Promoted decode uses the native expert path.
- [ ] Promoted prefill uses grouped bounded-workspace expert execution.
- [ ] Three warm-ups and ten retained measurements are present.
- [ ] Raw samples, confidence intervals and telemetry are retained.
- [ ] Prefill and decode both beat the accepted Q4_0 baseline.
- [ ] Differences versus Unsloth/vLLM are stated with timing and format caveats.
- [ ] Quality and memory configuration is identical across the promoted A/B.
- [ ] No fallback or diagnostic path contaminates results.

## Product

- [ ] Model downloader supports the compiled lock.
- [ ] CLI exposes model profile, formats, context and capabilities.
- [ ] Server health/metrics expose variant, bytes, native paths and slot size.
- [ ] Studio model manager can download/select the profile or the release explicitly excludes Studio.
- [ ] Release notes list unsupported features.
- [ ] Rollback to the prior release is documented.
- [ ] All evidence paths are immutable and referenced from the release record.

A partial implementation may be useful, but it is not "done" until every applicable box is checked.

## imp reference additions

Definition of done additionally requires:

- candidate G quality control completed or explicitly unavailable with evidence;
- actual winning dispatch and first graph-demotion reason retained;
- third-party code provenance complete;
- settled-evidence ledger updated;
- repeated engine destroy/recreate tests clean;
- no 5090 result used as an RTX 5080 acceptance threshold.

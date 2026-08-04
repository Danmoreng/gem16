# M22 — CLI, server and Studio product integration

## Objective

Expose the quality- and performance-qualified 26B profile through gem16's existing model download, CLI, OpenAI-compatible server and Studio surfaces without implying unsupported vision or MTP capability.

## Why this milestone exists

A kernel path is not useful until users can select, download, verify and run it safely. The product layer must communicate its memory/context constraints and reject unsupported features visibly.

## Prerequisites

- M19 final model lock
- M20 performance
- M21 context defaults

## Repository areas to inspect first

- `tools/fetch_model.py`
- `tools/hf_cache.py`
- `src/cli/run_main.cpp`
- `src/cli/chat_main.cpp`
- `src/cli/server_main.cpp`
- `src/server/session_pool.cpp`
- `studioApp/`
- `docs/STUDIO.md`
- `docs/SERVER.md`

## Suggested additions or boundaries

- `docs/GEMMA4_26B.md`
- `studio model catalog/profile entries`
- `tools/fetch_compiled_model.py if lock schema requires it`

## Implementation sequence

1. Add a stable model profile ID that includes Gemma 4 26B A4B, text-only, QAT-derived and the selected head format without claiming official Google endorsement.
2. Integrate immutable artifact lock download/verification through the shared Hugging Face cache.
3. Expose model variant, source provenance, weight format, context, resident bytes and unsupported features in CLI/health/model-list output.
4. Use the M21 default context and M09 admission reserve; provide clear errors for larger unsupported plans.
5. Reject image/audio/video input for the 26B profile before tokenization or GPU work.
6. Reject MTP options for the 26B profile until M25; never load the 12B assistant accidentally.
7. Add Studio model selection, download progress, required free-space/VRAM guidance and restart behavior.
8. Verify server root, streaming, resident continuation, sampling, tools and cancellation with the 26B model.
9. Keep 12B as an independent supported profile with unchanged defaults.
10. Document artifact licensing/provenance and the fact that users receive a derived checkpoint.

## Required tests

- Download, resume, verify-only, corrupt-file and offline-cache tests.
- `/health`, `/v1/models`, Chat Completions and Responses smoke tests.
- Streaming, tools, sampling, cancellation and resident continuation.
- Unsupported media and MTP requests fail with explicit 4xx/CLI errors.
- Context admission and insufficient-VRAM error paths.
- Studio starts/stops/attaches to the correct server and model.
- 12B model selection and all existing product tests remain green.

## Evidence and documentation outputs

- `docs/GEMMA4_26B.md` user guide
- `artifacts/m22/cli-smoke.json`
- `artifacts/m22/server-sdk-smoke.json`
- `artifacts/m22/studio-smoke.md`
- Screenshots/logs only as supplementary evidence; machine-readable results remain authoritative.

## Suggested commands

```text
python tools/fetch_model.py --lock models/gemma4-26b-gem16-hybrid.lock.json --verify-only
```
```text
build/blackwell-release/bin/gem16-server --model "$GEM16_26B_FINAL" --max-context 32768 --max-sessions 1
```
```text
python tools/validate_openai_sdk.py --base-url http://127.0.0.1:8080/v1 --model gem16-gemma4-26b-qat-hybrid
```

## Risks to watch in this milestone

- A 16 GB desktop may have insufficient free VRAM because of displays or other applications.
- Studio metadata can incorrectly advertise multimodal support inherited from the architecture.
- Artifact hosting and licensing must permit the chosen distribution model.
- Long model downloads need robust resume and hash verification.

## Forbidden shortcuts

- Marketing the artifact as an official Google or Unsloth checkpoint.
- Silently falling back to 12B when 26B admission fails.
- Accepting media and ignoring it.
- Loading MTP from another model size.
- Changing the 12B default profile without an explicit decision.

## Exit criteria

- [ ] Users can download, verify, select and run the 26B profile through all supported surfaces.
- [ ] Capabilities and unsupported features are accurate.
- [ ] Server admission and context defaults match qualified memory evidence.
- [ ] SDK, streaming, tools, cancellation and resident chat pass.
- [ ] 12B product behavior remains intact.
- [ ] User and provenance documentation is complete.

## Downstream milestones unblocked

- M23 release qualification

## Codex execution prompt

```text
You are implementing M22: CLI, server and Studio product integration in Danmoreng/gem16.

Read repository AGENTS.md, docs/DECISIONS.md, docs/CORRECTNESS.md, docs/BENCHMARKING.md, the package master plan, and this milestone. Work only on M22. Inspect the actual current tree before editing and write a drift note if it differs from the anchored commit.

Implement the narrowest change that satisfies every exit criterion. Add tests before or with implementation. Do not add silent fallbacks, token-loop allocations, CPU weight offload, or unrelated refactors. Preserve all existing Gemma 4 12B exact gates.

At completion, report files changed, tests and exact results, generated evidence paths, memory/performance deltas where relevant, unresolved risks, and whether each M22 exit criterion passed. Stop before starting the next milestone.
```

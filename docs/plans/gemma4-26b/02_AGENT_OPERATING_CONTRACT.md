# Coding-agent operating contract

This file is intended to be copied into or referenced by every Codex task for the 26B program.

## 1. Read-before-edit requirements

Before changing code, read:

1. repository root `AGENTS.md`;
2. repository `docs/DECISIONS.md`;
3. repository `docs/CORRECTNESS.md`;
4. repository `docs/BENCHMARKING.md`;
5. this package's master plan;
6. the current milestone;
7. all normative specifications linked by that milestone.

Do not rely on this plan's source snapshot when the working tree has moved. Produce a drift note first.

## 2. Scope discipline

- Work on one milestone only.
- Do not start a dependent milestone in the same PR.
- Do not combine an arithmetic change with an unrelated scheduling or layout optimization.
- Do not refactor the 12B hot path merely to make the 26B code look generic.
- Preserve exact 12B tests and benchmark metadata.
- Add new model behavior behind an explicit model variant.
- Never silently accept a tensor, shape or precision not covered by the manifest contract.

## 3. Evidence before optimization

The required progression is:

```text
source evidence
→ CPU oracle
→ CUDA reference
→ real-shape operator parity
→ layer parity
→ teacher-forced model parity
→ deterministic generation
→ memory characterization
→ profile
→ optimize
→ repeat every earlier gate
```

Kernel code written before its oracle and fixture exist is incomplete.

## 4. Precision and fallback rules

- A missing native NVFP4 kernel fails visibly.
- BF16 or dequantized fallback is allowed only behind an explicit diagnostic option.
- Every result records the actual format of attention, experts, head and KV.
- A benchmark with fallback is labeled diagnostic and cannot be promoted.
- Do not reinterpret Q4_0 bytes as NVFP4.
- Do not mix tensors from independently trained/quantized checkpoints unless a milestone explicitly creates and qualifies that experiment.
- The primary QAT hybrid must derive all mathematical tensors from one exact QAT BF16 revision.

## 5. Allocation rules

After slot initialization and graph capture:

- no `cudaMalloc` or `cudaFree`;
- no pageable host allocation;
- no growing container;
- no filesystem access;
- no JIT;
- no on-demand tensor repack;
- no CPU expert routing;
- no CPU weight streaming.

Use checked arithmetic for every byte count and fail before partial initialization on overflow or insufficient margin.

## 6. Benchmark integrity

- Run correctness before timing.
- Keep prompts, token IDs, output positions, sampling and KV precision fixed across candidates.
- Use three warm-ups and ten retained measurements for promotion.
- Record every raw run.
- Record clocks, power, thermals and VRAM.
- Report TTFT, prompt throughput, ITL and decode throughput separately.
- Count only target output tokens.
- Do not compare a prompt-cache hit with a miss.
- Do not present an external runtime's different timing boundary as exact parity.
- State checkpoint and format differences next to every headline.

## 7. Quality integrity

- Keep calibration and evaluation corpora disjoint.
- Do not select a quantization recipe on the final test set.
- Compare teacher-forced distributions before chat anecdotes.
- Retain router probabilities and selected expert IDs.
- Report regressions, not only aggregate improvements.
- A candidate is not "QAT quality" merely because its BF16 source passed through a QAT pipeline; the target execution format must be measured.

## 8. Documentation requirements

Every PR must update at least one of:

- `docs/DECISIONS.md` for a promoted choice;
- `docs/PERFORMANCE_LEDGER.md` for measured kernel/end-to-end evidence;
- a 26B implementation document for contract changes;
- the current milestone evidence record.

Do not rewrite old benchmark results. Add a new dated result.

## 9. Commit and PR rules

- One semantic change per commit where practical.
- Use descriptive commit messages.
- Include generated file hashes when a compiler fixture changes.
- Do not commit gated model weight payloads.
- Do commit tiny deterministic fixtures permitted by model licensing.
- Do not force-push evidence branches after review begins unless required to remove secrets.
- PR description must identify arithmetic, layout, memory and timing changes independently.

## 10. Stop conditions

Stop and report rather than working around the issue when:

- a source revision cannot be locked;
- tensor names/shapes differ from the contract;
- conversion is nondeterministic;
- the weight arena exceeds the hard budget;
- 12B exact outputs change unexpectedly;
- router semantics cannot be matched;
- a native path silently falls back;
- repeated deterministic runs produce different outputs;
- an optimization has no adjacent A/B evidence;
- the GPU lacks the compiled native architecture.

## 11. Agent completion response

At the end of a milestone, report:

- files changed;
- decisions made;
- tests run and exact results;
- generated evidence paths;
- allocator bytes and peak VRAM if relevant;
- performance distribution if relevant;
- known limitations;
- whether every exit criterion passed;
- the next milestone that is now unblocked.

Do not claim the next milestone is complete.

## External reference implementations

An external runtime is never an implicit dependency or authority. The agent must pin it, classify each use as reference/clean-room/copied/rejected, preserve licenses and avoid broad architectural imports. See [`specs/REFERENCE_IMPLEMENTATION_ADOPTION_SPEC.md`](specs/REFERENCE_IMPLEMENTATION_ADOPTION_SPEC.md).

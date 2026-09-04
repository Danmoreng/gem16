# Documentation

## Use gem16

- [Project introduction and source installation](../README.md)
- [Studio](STUDIO.md): install, verify, select, start and chat
- [Server](SERVER.md): both profiles, HTTP endpoints, tools and operations
- [26B profiles](GEMMA4_26B.md): public Compact Vision and internal NVFP4
- [Image input](VISION.md) and [audio input](AUDIO.md): capability-specific details
- [Recorded performance](PERFORMANCE.md): measurements, caveats and reproduction

## Develop or review

Read [AGENTS.md](../AGENTS.md), [active decisions](ACTIVE_DECISIONS.md) and the
[product contract](PRODUCT_CONTRACT.md), then the narrow contract for the task.
API work additionally reads [OpenAI Agent Core v1](OPENAI_AGENT_CORE_V1.md).

- [Roadmap](ROADMAP.md): outstanding product work and release gates
- [Development](DEVELOPMENT.md), [architecture](ARCHITECTURE.md), [memory](MEMORY.md)
- [Checkpoint format](CHECKPOINT_FORMAT.md), [MTP](MTP.md)
- [Studio implementation reference](development/STUDIO_INTERNALS.md)
- [Correctness](CORRECTNESS.md) and [benchmark methodology](BENCHMARKING.md)
- [26B task routing](plans/gemma4-26b/START_HERE_CODEX.md), only for model-specific work

## Historical evidence

[Archived decisions and plans](archive/README.md) and the
[26B plan index](plans/gemma4-26b/INDEX.md) preserve prior stages. Older documents
can describe superseded public profiles or completed work; they do not create new tasks.
[Decision history](DECISIONS.md), [performance ledger](PERFORMANCE_LEDGER.md),
[artifacts](../artifacts/) and [benchmarks](../benchmarks/) retain accepted evidence.
Current source, tests and active policy determine what exists today.

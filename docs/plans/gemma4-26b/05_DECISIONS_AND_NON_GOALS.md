# Active decisions, conditional choices and non-goals — Fast Track R4

## Active decisions

- **D1:** M00–M05 remain accepted and frozen.
- **D2:** QAT BF16 is the sole mathematical source of the production target.
- **D3:** Attention remains FP8; shared and routed MLPs use NVFP4.
- **D4:** The first complete artifact uses a provisional NVFP4 tied head.
- **D5:** Q4_0 is an external reference and optional internal M24 backend, not an M08 prerequisite.
- **D6:** M13 is the only early quality go/no-go gate.
- **D7:** M18 is conditional diagnosis/attribution and does not normally precede M14–M17.
- **D8:** One integration branch may receive commits from isolated sub-agent worktrees.
- **D9:** One 26B slot is supported on 16 GB; slot two is rejected.
- **D10:** 32K is the first hard gate; 64K and the measured maximum follow.
- **D11:** MTP is a required final target in M25 and is separate from vision.
- **D12:** T=3/T=5 or other multi-row verifier-head work belongs to M25, not base bring-up.
- **D13:** Full expensive runs require clean committed worktrees.

## Conditional choices

### Head format

Keep NVFP4 unless M13/M19 shows material head-driven quality loss or M20 shows a bounded alternative with a better product Pareto. A format change must preserve one physical tied matrix.

### Ordinary-BF16 control

A complete ordinary-BF16 conversion is required for causal quantizer/source claims and therefore belongs to M18 or a final attribution report. It does not block the first QAT runtime artifact.

### MTP assistant representation

Prefer a compatible official assistant source. If BF16 residency fails, compile a separately locked FP8/NVFP4 assistant candidate and measure acceptance/quality. Do not reuse the 12B assistant by assumption.

### Maximum context

Advertise the highest context that completes real execution and preserves the required margin. Base and MTP profiles may differ.

## Non-goals

- full in-engine Q4_0 before a working native target;
- positive multi-slot 26B scaling on 16 GB;
- vision in M25;
- T=3/T=5 base head optimization;
- universal Gemma 4 MoE support;
- CPU offload or expert streaming;
- replacing the server architecture with a generic serving framework;
- publishing a QAT-quality claim without measured evidence;
- rerunning unchanged full workloads merely to duplicate evidence.

# Gemma 4 26B profiles

The public **Gemma 4 26B A4B Compact Vision** profile combines the Trellis35 W4A8
text Target with an FP8 E4M3FN Vision module and an optional hybrid NVFP4/FP8/BF16
fixed-D2 Assistant. It is an equal public choice alongside 12B Unified.

| Boundary | Public Compact Vision | Internal NVFP4 regression/rollback |
|---|---|---|
| Input | Text and images within context capacity; no audio | Text only |
| Resident slots | One | One |
| Qualified Target context | 229,376 | 98,304 |
| Qualified fixed-D2 context | 229,120 | 86,016 |
| Studio selection | Public | Hidden from normal selection |

Context admission also depends on available VRAM and the configured reserve.
Compact Vision image budgets are 70, 140 or 280 soft tokens. Its D2 capability
requires the exact validated Target/Vision/Assistant composite; components are
never silently substituted.

## Acquire and run

Use the [server guide](SERVER.md) or Studio's Models screen. Runtime locks are:

- [Compact Target](../models/gemma4-26b-trellis35-target.lock.json)
- [Vision](../models/gemma4-26b-vision-fp8.lock.json)
- [Assistant](../models/gemma4-26b-gem16-assistant.lock.json)
- [Internal NVFP4 Target](../models/gemma4-26b-gem16-target.lock.json)

These independently pinned components currently resolve the consolidated Hub
revision `6de2a057f11332420819f8e6efd08e42d7a03bc7`. The later normalized layout
is published but does not change runtime locks until acquisition and loader
verification passes. Startup validates and uploads the offline-built device
image without quantizing, repacking or rehashing its full payload.

## Qualification and retained evidence

[Bounded P20 acceptance](../artifacts/vision/p20-owner-acceptance-2026-09-04.json)
uses the existing V19 evidence; the larger QUAL01 campaign was explicitly waived.
REL01, P21, live Windows SM120, packaging and clean-machine gates remain release work.
See the [production contract](plans/gemma4-26b/PRODUCTION_26B_VISION_CONTRACT.md).

The internal NVFP4 path and its accepted numerical, context and rollback evidence
remain protected. Its former 220/250 tok/s objectives are closed. Historical
[26B plans](plans/gemma4-26b/INDEX.md) do not authorize new tuning.
[Recorded performance](PERFORMANCE.md) separates these formats and their output sequences.

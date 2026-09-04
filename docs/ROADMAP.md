# Product roadmap

Current authority: [active decisions](ACTIVE_DECISIONS.md),
[product contract](PRODUCT_CONTRACT.md) and [OpenAI Agent Core v1](OPENAI_AGENT_CORE_V1.md).

## Current baseline

- Two public profiles: 12B Unified and 26B Compact Vision; NVFP4 remains an internal rollback path.
- One native Studio with shared-cache acquisition, profile selection and managed local server.
- Bounded Compact Vision P20 accepted; extended QUAL01 waived, not executed.
- Responses output-limit events, nonblocking Studio cancellation, failed-exchange recovery
  and explicit fresh payload verification corrected in `8d057db`.
- User guides and active task routing consolidated; measurements and prior decisions archived intact.

## Next increments

1. Extend the [bounded SDK/Pi matrix](AGENT_COMPATIBILITY.md) from its completed
   Linux runs to Windows live GPU execution and the remaining Agent Core gates.
   Close constrained tool-choice and reasoning-replay gaps before broader compatibility claims.
2. Improve Studio conversation durability and request-specific diagnostics. These are
   future product slices, not reasons to reopen the renderer or engine architecture.
3. Close release gates: live Windows SM120 Compact Vision evidence, REL01/P21,
   equal Windows/Linux archives, dependency contracts, notices, manifests, hashes,
   full downloads and clean-machine first-run/restart smokes.
4. Follow with installers, signing and updates after the portable release gate.

The normalized 26B Hub layout still needs verified lock migration before use by
the runtime. Existing pins remain valid and immutable.

## Closed and retained work

The 26B NVFP4 fast track and Trellis35 performance campaign are frozen; the later
bounded K/V fusion acceptance is recorded in active decisions. Former 220/250 tok/s
targets are closed. Offline device images already replace the old startup CPU
weight-tiling path; that historical optimization request is not an open task.

[26B history](plans/gemma4-26b/INDEX.md) and [performance records](PERFORMANCE.md)
retain the accepted results. `studioApp/` and Gradle infrastructure may be removed
only after the first fully qualified two-platform native release.

# Product roadmap

Current authority: [active decisions](ACTIVE_DECISIONS.md),
[product contract](PRODUCT_CONTRACT.md) and [OpenAI Agent Core v1](OPENAI_AGENT_CORE_V1.md).

## Current baseline

- Two public profiles: 12B Unified and 26B Compact Vision; NVFP4 remains an internal rollback path.
- One native Studio with shared-cache acquisition, profile selection and managed local server.
- Bounded Compact Vision P20 accepted; extended QUAL01 waived, not executed.
- Responses output-limit events, nonblocking Studio cancellation, failed-exchange recovery
  and explicit fresh payload verification corrected in `8d057db`.
- Native Studio stores conversations and attempts in SQLite, with attachment copies,
  full-text search, JSON/Markdown exports and restorable backups.
- User guides and active task routing consolidated; measurements and prior decisions archived intact.

## Active server-first launch backlog

Owner direction: 2026-09-05; review baseline c54cb790. Implement compatibility
where practical; preserve model specialization and historical evidence.

| Package | Scope | Status |
|---|---|---|
| C00 | Server-first docs, 220k Linux / 170k Windows, 200 MiB reserve | Implemented; Linux host/Studio checks passed |
| C01 | Bounded schema evaluation and exact numbers | Implemented; host and sanitizer checks, including work limits |
| C02 | Exception-safe session/Responses ownership | Lease/reservation guards implemented; systematic exception injection open |
| C03 | Deadline/cancel admission, control capacity, shutdown | Bounded admission/control capacity and pool-wait cancellation/deadlines implemented; long-prefill cancellation open |
| C04 | HTTP preflight and bounded media processing | HTTP/media limits implemented and tested; peak-RSS stress and historical-image CPU reuse open |
| C05 | Pi affinity, cache reuse, new/fork/compaction behavior | Live Linux affinity/cache/manual compaction passed for both profiles; full fork/resume/automatic-compaction matrix open |
| C06 | Responses replay, practical sampling/tool compatibility | SDK output replay and parameter validation passed; per-request sampling, reasoning replay and constrained tool choice open |
| C07 | Fresh headless packages, provenance, fail-closed publish | Fresh headless packages/manifests and gate verifier implemented; same-machine smoke passed, clean-machine qualification open |
| C08 | Candidate GPU/SDK/agent/quality and two-platform evidence | Bounded Linux SDK/Pi/multi-image and internal NVFP4 smokes passed; Windows and full release qualification open |
| C09 | Documentation consistency and release freeze | Requires C01–C08; publication needs explicit authorization |

Implementation and bounded evidence: [server hardening checkpoint](evidence/server-hardening-2026-09-05.md).

Windows GPU and clean-machine qualification must be recorded against the actual
candidate. Missing evidence is open, never passed. The extended QUAL01 waiver
remains unchanged. Studio lifecycle fixes remain in scope; GUI redesign does not.

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

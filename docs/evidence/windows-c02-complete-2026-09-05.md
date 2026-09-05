# C02 lifecycle completion — Windows, 2026-09-05

Parent `491f5f5`, followed by the recorded C02 source slice. This completes C02's
implementation and Windows lifecycle qualification, not two-platform release
qualification. CUDA kernels, runtime/model algorithms, formats and locks are
unchanged; the internal NVFP4 numerical path was not modified.

The separate, explicitly built `gem16-server-fault-test` links the real runtime
and model sessions. Its server-support objects alone enable fault hooks. The
normal server neither reads the fault header nor exposes the diagnostic counter;
packaging continues to consume only `gem16-server`.

## Lifecycle coverage

Each public profile passes **140 injected failure/recovery cases**, 280 total:

- Chat Completions and Responses, streaming and nonstreaming;
- fresh sessions and acquisition of existing chat/Responses sessions;
- pending reservation, before publication and after publication;
- acquisition, first actual generation callback and completed generation;
- response indexing before/after insertion;
- Responses commit before construction, after message construction and after
  committed replacement; final response serialization;
- standard exceptions and `bad_alloc` at every applicable stage, nonstandard
  exceptions during generation, and returned cancellation status for fresh runs.

Every case checks the injected counter advances exactly once, pending
reservations/queue/active-request/active-response-ID gauges return to zero,
response indexes are removed, and a lease is released exactly once (zero leases
before publication). Exceptions discard the resident session. The same chat ID
then acquires a working real GPU session and generates tokens successfully.
The separate host allocator test covers every one of 15 actual allocations in a
representative Responses commit, preserving the old host chain on failure.

Publication now keeps the pending reservation until insertion succeeds, hands
its ownership off once, and protects the interval from publication to return.
Deferred provider exceptions cannot escape into httplib worker execution;
unfinished indexes and active response IDs are cleaned up. The earlier atomic
chain replacement and allocation-free index cleanup remain in place.

## Reproduction and evidence

On Windows import the environment using `scripts/windows-toolchain.ps1`, then:

```powershell
.\scripts\build.ps1 -Cuda -ConfigureOnly
cmake --build build/Windows/blackwell-release --target gem16-server-fault-test --parallel 2
.\.venv-agent-core\Scripts\python.exe tools/check_session_faults.py --server build/Windows/blackwell-release/bin/gem16-server-fault-test.exe --baseline artifacts/server-hardening/2026-09-05-c02-windows-agent-fix/result.json --output-dir artifacts/server-hardening/NEW-C02-EVIDENCE
```

The baseline supplies exact local model paths/flags; the harness replaces the
executable and port, refuses an occupied listener and never overwrites evidence.
Both profiles use context 16,384, fixed-D2 and one execution slot. This is fault
recovery evidence, not a latency or long-context performance claim.

- [Initial complete injection run](../../artifacts/server-hardening/2026-09-05-c02-injection-01/result.json)
- [Final reservation-handoff injection run](../../artifacts/server-hardening/2026-09-05-c02-injection-final/result.json)
- [Production SDK/agent/hardening matrix](../../artifacts/server-hardening/2026-09-05-c02-complete-production/result.json)
- [Final production hook-exclusion/recovery smoke](../../artifacts/server-hardening/2026-09-05-c02-production-final/result.json):
  eight cases per profile passed against the final normal server binary.
- [Source/binary hashes and host logs](../../artifacts/server-hardening/2026-09-05-c02-complete-host/provenance.json)

Host debug tests: six passed (30.29 seconds), the MSVC debug-STL allocation test explicitly
skipped (covered by Release instead). Release unit/allocation/process-cleanup
tests passed. Both public-profile production Python SDK, TypeScript SDK,
hardening and multi-image checks passed, including a supplied fault header that
does not trigger a production failure. The 26B Pi coding fixture also passed.

## Separate open agent-quality finding

The latest 12B Pi fixture failed its task: it repeatedly emitted ineffective
`edit` calls and ended at the length limit. Its resident session, cache reuse,
API transport and process exit checks passed; the fixture source remained
incorrect. This failed matrix is retained, not replaced by the earlier passing
run. It is a C05/C06 practical-agent quality finding and remains open; C02's
280 lifecycle/recovery checks do not establish reliable task-solving quality.

The earlier MSVC debug abort and Pi timeout/cleanup failure remain documented
in the bounded checkpoint. No Linux execution or full system-memory-exhaustion
claim is implied. C03 adds interruption *inside* prefill and remains separate.

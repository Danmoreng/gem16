# Product roadmap

Current authority: [active decisions](ACTIVE_DECISIONS.md),
[product contract](PRODUCT_CONTRACT.md) and [OpenAI Agent Core v1](OPENAI_AGENT_CORE_V1.md).

**Continue implementation on Windows:** [English TODO plan and handoff](#english-continuation-plan-windows-and-remaining-implementation).

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
| C00 | Server-first docs, 220k Linux / 170k Windows, 200 MiB reserve | Implemented; Linux and Windows host/Studio checks passed |
| C01 | Bounded schema evaluation and exact numbers | Implemented; host and sanitizer checks, including work limits |
| C02 | Exception-safe session/Responses ownership | Completed and qualified on Windows: 280 real-GPU injected lifecycle/recovery cases; Linux release requalification remains separate |
| C03 | Deadline/cancel admission, control capacity, shutdown | Bounded admission/control capacity and pool-wait cancellation/deadlines implemented; long-prefill cancellation open |
| C04 | HTTP preflight and bounded media processing | HTTP/media limits implemented and tested; peak-RSS stress and historical-image CPU reuse open |
| C05 | Pi affinity, cache reuse, new/fork/compaction behavior | Live Linux and Windows affinity/cache/manual compaction passed for both profiles; full fork/resume/automatic-compaction matrix open |
| C06 | Responses replay, practical sampling/tool compatibility | SDK output replay and parameter validation passed; per-request sampling, reasoning replay and constrained tool choice open |
| C07 | Fresh headless packages, provenance, fail-closed publish | Fresh headless packages/manifests and gate verifier implemented; same-machine smoke passed, clean-machine qualification open |
| C08 | Candidate GPU/SDK/agent/quality and two-platform evidence | Bounded Linux and Windows SDK/Pi/multi-image matrices passed; internal NVFP4 Linux smokes retained; full release qualification open |
| C09 | Documentation consistency and release freeze | Requires C01–C08; publication needs explicit authorization |

Implementation and bounded evidence: [server hardening checkpoint](evidence/server-hardening-2026-09-05.md).

Windows baseline on candidate `491f5f5` is now recorded in
[Windows server-first evidence](evidence/windows-server-baseline-2026-09-05.md):
host/CUDA/Studio builds, both public-profile SDK/Pi/hardening/multi-image matrices,
separate manual compaction/new-session probes and the protected 12B product test
passed. No portability fix was needed. Fresh headless ZIP integrity/version smoke
passed on the development machine. C02 was subsequently completed in the
separate lifecycle qualification below; the next implementation priority is C03.
The baseline itself does not close long-prefill gates. Full Windows release and
clean-machine qualification stay open.

Windows GPU and clean-machine qualification must be recorded against the actual
candidate. Missing evidence is open, never passed. The extended QUAL01 waiver
remains unchanged. Studio lifecycle fixes remain in scope; GUI redesign does not.

The normalized 26B Hub layout still needs verified lock migration before use by
the runtime. Existing pins remain valid and immutable.

Owner stop point (2026-09-05): C02 is complete on Windows; commit/push this
checkpoint and stop for today. Resume with C03 next session, then C04. The latest
12B Pi fixture's ineffective-edit loop is retained as a C05/C06 finding; it must
not be hidden by the earlier successful agent smoke. See the C02 completion note.

## English continuation plan: Windows and remaining implementation

**Handoff baseline:** `e539f08` on `main` (2026-09-05). This section is the
canonical actionable TODO plan for C02–C09. The German review/launch documents,
local Downloads files and previous chat are not required to continue. Read the
current source before applying any item: later commits may already complete it.
Update the table above and the matching checklist here in the same change.

### Start here on Windows

1. Inspect `git status`, branch, revision and remote; synchronize with the current
   `main` without overwriting local work. Read `AGENTS.md`,
   [ACTIVE_DECISIONS.md](ACTIVE_DECISIONS.md), [PRODUCT_CONTRACT.md](PRODUCT_CONTRACT.md)
   and, for API work, [OPENAI_AGENT_CORE_V1.md](OPENAI_AGENT_CORE_V1.md).
2. Establish the unchanged Windows baseline first: build/host tests, then the
   pinned SDK/Pi matrix on both public profiles. Record Windows-specific failures
   separately from the remaining cross-platform implementation work below.
3. Fix and test one bounded slice at a time. Prioritize C02/C03 lifecycle and
   cancellation, then C04 media limits and C05/C06 client behavior. Packaging and
   evidence work can proceed independently when a runtime gate is unavailable.
4. Keep completed slices in focused commits and pushes, as requested by the owner.
   This does not authorize tags, releases, model publication or changing the
   two-platform release scope. Preserve unrelated work and all negative evidence.

Public profiles remain **12B Unified** (text/image/audio, up to two qualified
slots) and **26B Compact Vision** (Trellis35 text + FP8 Vision, one slot, optional
pinned fixed-D2 Assistant). Internal text-only NVFP4 remains regression/rollback.
Compact Vision everyday context stays **170,000 on Windows / 220,000 on Linux**;
keep the **200 MiB long-context admission reserve**. Saved explicit settings are
not silently migrated. A startup/short-answer smoke is not long-context stability
qualification. Server/CLI/Pi is the priority; Studio is optional, with lifecycle
and data-loss bugs still in scope. Preserve the existing extended QUAL01 waiver.

### Windows baseline commands and prerequisites

Use the pinned toolchain in `toolchains/blackwell16gb.lock` and the environment
setup in `scripts/windows-toolchain.ps1`. The SDK harness requires Python 3.11+,
Node 22.19+, and Git Bash for the Pi fixture on Windows. Model acquisition uses
the current immutable locks and shared Hub cache; follow [SERVER.md](SERVER.md)
for model installation before GPU probes. Do not use Linux-only `/tmp` paths or
copy an old test environment from the previous session.

From the repository root in PowerShell, run each step separately and stop on a
nonzero exit code; native command failures do not necessarily stop PowerShell:

```powershell
.\scripts\build.ps1 -Test -Jobs 4
.\scripts\build.ps1 -Cuda -Test -Jobs 2
python -m unittest discover -s tests/python -p test_release_gates.py -v
python -m venv .venv-agent-core
.\.venv-agent-core\Scripts\python.exe -m pip install -r tools/requirements-openai-sdk.txt
npm ci --prefix tools/openai-sdk --ignore-scripts
npm --prefix tools/openai-sdk run check
npm ci --prefix tools/pi-agent --ignore-scripts
```

The existing Windows build script explicitly does not support `-Sanitize`;
retain Linux ASan/UBSan evidence and report the Windows coverage limitation.
Then run the existing matrix into a **new** directory (port must be free):

```powershell
$handoffEvidence = "artifacts/server-hardening/" + (Get-Date -Format "yyyy-MM-dd-HHmmss") + "-windows"
.\.venv-agent-core\Scripts\python.exe tools/run_agent_core_matrix.py --server build/Windows/blackwell-release/bin/gem16-server.exe --pi-cli tools/pi-agent/node_modules/@earendil-works/pi-coding-agent/dist/cli.js --output-dir $handoffEvidence --port 18083
```

These commands use existing script interfaces and were subsequently run on
Windows at `491f5f5`, with the passing evidence linked above. The
matrix currently uses 16,384 context and fixed-D2, so it does **not** cover the
170,000 Windows recommendation, ordinary decode, manual compaction, long-prefill
abort, or clean-machine installation. Run those as separately labeled probes.
Its process termination cleanup is not proof of graceful Windows shutdown.

### C02 — Finish lifecycle failure qualification

Completed Windows qualification: [C02 lifecycle completion](evidence/windows-c02-complete-2026-09-05.md).
The earlier [bounded checkpoint](evidence/windows-c02-2026-09-05.md) remains
historical. The separate test executable now covers all lifecycle stages below
with real model sessions, without enabling fault control in the product binary.

- [x] Add targeted error/exception injection before and after session publication,
  after acquisition, during generation, Responses commit/indexing and response
  serialization. Cover Chat Completions and Responses, stream and nonstream.
- [x] Assert reservations, active request counts, leases and response indexes are
  released/rolled back exactly once, followed by successful new generation.
  Use host fixtures where practical without introducing a generic fake engine.
- [x] Run actual slot recovery on both public profiles and protect internal
  NVFP4 when shared runtime behavior changes.

Start at `src/server/session_pool.{h,cpp}` and `src/cli/server_main.cpp`.
Reservation guards, moved leases and nonstream index rollback already exist;
this is qualification and any fixes it exposes, not a second pool implementation.
**Done when:** every injected path recovers without orphan state or double release,
with reproducible logs and actual GPU slot recovery evidence.

### C03 — Long-prefill cancellation and full shutdown/recovery

- [ ] Add safe cancellation checkpoints while a long prefill is running, including
  disconnect for nonstream requests. Inspect `src/runtime/chat.cpp`,
  `src/cuda/inference_session.cuh` and the specialized engine prefill paths under
  `src/cuda/engine/`, together with the server cancellation callbacks. Preserve
  ordinary/fixed-D2 semantics, cache consistency and allocation rules; do not
  simulate cancellation by silently shortening prompts or changing chunk semantics.
- [ ] Define the state after partial prefill: safely reusable or explicitly
  discarded/poisoned. Verify that the next request can acquire a working slot.
- [ ] Test deadline/disconnect while queued, waiting for the same session, preparing
  media, prefilling and decoding. Include the 12B two-slot same-session contention
  case; the existing one-slot GPU matrix does not exercise that pool wait.
- [ ] Measure real cancellation latency and control-route latency under saturation.
  Use an explicit control target such as one second, record raw samples, and avoid
  treating that target as an existing guarantee. Exercise Linux SIGINT/SIGTERM and
  the supported Windows graceful stop path, then restart and generate again.

The FIFO, pool wait deadline propagation, 25 ms disconnect/draining polls and
inference-saturation worker reserve are already implemented in `2096c39`.
**Done when:** abandoned requests never execute later, capacity is recovered,
control routes remain reachable, and each phase has measured platform evidence.
Host-only wait tests do not qualify GPU prefill abort latency.

Source inspection for the next slice: `ChatSession::Generate` currently forwards
only token events into the inference session, so the server cancellation callback
cannot run during prefill. Thread a separate cancellation checkpoint through the
existing 12B and specialized 26B prefill loops, preserving their chunk planning
and image spans. The 26B loop already synchronizes before staging each chunk;
the 12B path currently synchronizes after its loop. Qualify the synchronization
and poisoned-session recovery behavior, rather than assuming equivalent latency.
No C03 runtime change or long-prefill latency claim is included in the C02 slice.

### Requested short comparison pilot

The owner selected approximately 30 minutes for a bounded speed/quantization
comparison. See [the concrete pilot plan](plans/SHORT_CROSS_ENGINE_BENCHMARK.md)
for 512/8K/32K context, separate prefill/decode, quant candidates and full-KL
reference requirements. This is additional diagnostic work, not a reopening of
the closed tuning campaigns or the waived extended QUAL01 campaign. No new
cross-engine result is claimed yet.

### C04 — Media work, cancellation and peak host memory

- [ ] Measure peak RSS for large compressed images, many images, parallel requests,
  invalid dimensions and interrupted preprocessing, including both public profiles.
  Verify limits apply before excessive allocation and memory is released on errors.
- [ ] Check aggregate process memory against admitted concurrency, not only one
  request. Account for decoded pixels, prepared patches, temporary codec memory,
  JSON/base64 storage and audio where applicable; fix any uncovered growth.
- [ ] Avoid repeatedly decoding unchanged historical images where practical.
  Bound any CPU cache and key it by image identity, preprocessing version and
  effective budget; preserve ordered multi-image history and exact normalization.
  GPU KV reuse already exists and does not prove CPU reuse.

Start at `src/server/openai_chat.cpp`, `src/model/image.cpp` and
`src/model/image_decode_budget.h`. Host/Origin/MIME checks, 32 million cumulative
image pixels, 256 MiB accounted prepared/resize bytes, bounded decoder concurrency
and early model identity rejection (`e539f08`) already exist.
**Done when:** malicious/parallel media work stays within measured bounds, abort
recovers, and valid multi-image replays retain the same behavior.

### C05/C06 — Finish the actual client contract

- [ ] Exercise unmodified Pi 0.85.0 new session, fork, resume, manual and automatic
  compaction on both platforms. Check retained facts, session separation, cache
  hits and intentional history/tool resets. Extend `tools/check_pi_compaction.py`
  and `tools/validate_external_agent.py` rather than forking the agent.
- [ ] Qualify reasoning replay end to end if practical. Until then, keep the
  delivered Pi configuration on the qualified thinking-off path and make other
  levels explicitly unavailable there; do not silently drop reasoning history.
- [ ] Assess real client needs for per-request sampling against resident sampler,
  RNG and fixed-D2 behavior. Implement practical compatibility with regression
  evidence; retain explicit rejection for substantial unsupported semantics.
- [ ] Assess required/named tool choice and `parallel_tool_calls=false`. They
  currently fail early because correct enforcement needs more runtime support.
  Do not replace that with parse-only acceptance or claim constrained decoding.
- [ ] Cover multi-round tools, Unicode/long results, bad tool IDs, tool errors,
  output limits and terminal SSE events on both APIs/transports. Distinguish real
  generated parallel calls from replay fixtures. Align tests and capability docs.

Direct SDK output-object replay, matching max-token aliases and Pi native session
headers/cache/manual compaction already have bounded Linux evidence. Pinned clients
are Python SDK 2.50.0, Node SDK 7.10.0 and Pi 0.85.0.
**Done when:** every advertised capability is actually executed with the pinned
clients, and remaining complex limitations are explicit before costly execution.

### C07 — Finish package provenance and first-run integrity

- [ ] Complete actual compiler/CUDA/backend and dependency provenance and notices;
  a toolchain-lock hash alone does not prove which toolchain built a binary.
- [ ] Build fresh headless and optional Studio candidates for both platforms.
  Confirm only intended files/dependencies ship and diagnostic targets remain
  available to developers without unnecessarily entering the product package.
- [ ] Test packages and complete locked downloads on clean target environments
  without developer paths. Verify first server start, Pi tool task, stop/restart.
- [ ] Resolve the internal NVFP4 shared-Hub-root/runtime-view mismatch without
  weakening loader path/symlink/exact-file checks. Linux diagnostics succeeded
  with verified same-filesystem hardlinks and an explicitly filtered packaging
  descriptor; that workaround is not a production acquisition fix. The normalized
  Hub revision still needs verified lock migration; do not change pins speculatively.
- [ ] Bind release gate evidence to the exact candidate revision/version/archive
  and raw sample hashes using `tools/verify_release_gates.py`. Keep publication
  separate from candidate upload and fail closed for missing/stale evidence.

Fresh staging, `tools/package_server.py`, manifest hash tests and the gate verifier
already exist. The Windows workflow uploads candidates and cannot publish a
release merely because a tag exists. **Done when:** platform packages have verified
provenance and clean-machine evidence; an old same-machine package smoke is insufficient.

### C08/C09 — Candidate qualification and documentation finish

- [ ] Record exact candidate/source hashes, GPU/driver/toolchain, model locks,
  context, KV, sampling, D2 and image budgets before each run. A later relevant
  source change requires new affected evidence, not relabeling an earlier binary.
- [ ] Cover both public profiles on Windows and Linux, Compact Vision ordinary
  and fixed-D2, text/image functions, multi-image continuation/fresh replay,
  protected 12B audio and internal NVFP4 regression as applicable.
- [ ] Separately qualify everyday context with desktop load, explicit VRAM
  accounting and representative long prompts. Never silently lower context,
  precision or use CPU weight offload to make a gate pass.
- [ ] Close applicable CUDA memory/orchestration sanitizer gaps with bounded
  probes. Preserve failures/timeouts; missing tools or hardware are open gates.
- [ ] Freeze a small representative agent/media task set before measurement
  (e.g. 8–12 read/edit/test, tool-error/retry, long-loop, screenshot/diagram and
  multi-image cases). Record repetitions and failures; do not reopen extended
  QUAL01 or claim general coding superiority from a smoke set.
- [ ] Measure cold start/first token, warm TTFT, full Pi task time, cached/uncached
  prefill, ordinary/D2 output and separate Vision phases with raw samples,
  sample counts, peak VRAM/RSS and error rates. Disclose timing/semantic changes;
  old decode throughput is not a current agent-performance result.
- [ ] Check links, capability drift, VERSION/manifests and active docs. Freeze the
  exact candidate only after applicable gates pass. Release notes must describe
  measured scope and open limitations. Tags/publication require explicit approval.

**Done when:** every claimed platform/profile gate has matching candidate evidence
and clean-machine verification. Windows absence is never a pass; neither is the
existing P20 acceptance a waiver of release gates. Unavailable platform testing
must not block independent host implementation. Keep `studioApp/` retirement and
GUI redesign outside this work unless their separate authorization/gates apply.

### Evidence and a copyable continuation request

The [checkpoint](evidence/server-hardening-2026-09-05.md) links preserved Linux
results and failures. `session-wait-matrix` is the latest GPU matrix of this
handoff; `model-preflight-host` covers the subsequent host-only model check.
Neither is Windows evidence or a complete release qualification. Do not depend
on the previous machine's virtualenv, model-view directory or temporary scripts.

```text
Continue the server-first hardening work from docs/ROADMAP.md, section
"English continuation plan: Windows and remaining implementation".
Read AGENTS.md and the active contracts, inspect current source and git state,
then establish the Windows baseline and work through the remaining checklists.
The German review files and earlier chat are not required. Preserve completed
work and historical evidence. Commit and push each tested, bounded completed
slice; do not tag or publish a release. Record exact tests, failures and open
platform gates, and update the same roadmap as work completes.
```

## Closed and retained work

The 26B NVFP4 fast track and Trellis35 performance campaign are frozen; the later
bounded K/V fusion acceptance is recorded in active decisions. Former 220/250 tok/s
targets are closed. Offline device images already replace the old startup CPU
weight-tiling path; that historical optimization request is not an open task.

[26B history](plans/gemma4-26b/INDEX.md) and [performance records](PERFORMANCE.md)
retain the accepted results. `studioApp/` and Gradle infrastructure may be removed
only after the first fully qualified two-platform native release.

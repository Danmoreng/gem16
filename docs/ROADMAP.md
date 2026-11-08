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
| C03 | Deadline/cancel admission, control capacity, shutdown | Windows and Linux phase/deadline, media/history, ordinary/D2 cancellation and graceful restart qualification complete |
| C04 | Bounded single-user media check | Windows and Linux eight-request checks complete; server/cache checks and 26B colors passed, 12B small-image recall finding open; parallel stress/CPU cache deferred |
| C05 | Pi affinity, cache reuse, new/fork/compaction behavior | Windows and Linux Pi new/fork/resume (including cold start), manual/automatic compaction passed for both profiles |
| C06 | Responses replay, practical sampling/tool compatibility | Windows and Linux SDK replay, tool-error/Unicode and limit checks passed; reasoning replay, per-request sampling and constrained tool choice remain unsupported |
| C07 | Fresh packages, provenance and archive integrity | v0.3.0 Windows/Linux tag builds and archive verification passed |
| C08 | Full candidate GPU/SDK/agent/quality campaign | Removed from mandatory release work by standing owner policy, 2026-11-08 |
| C09 | Documentation consistency and owner-authorized publication | Stable v0.3.0 publication authorized; proportional tests and archive integrity govern future releases |

Implementation and bounded evidence: [server hardening checkpoint](evidence/server-hardening-2026-09-05.md)
and [Linux C03-C06 minimum qualification](evidence/linux-c03-c06-2026-09-21.md).

Windows baseline on candidate `491f5f5` is now recorded in
[Windows server-first evidence](evidence/windows-server-baseline-2026-09-05.md):
host/CUDA/Studio builds, both public-profile SDK/Pi/hardening/multi-image matrices,
separate manual compaction/new-session probes and the protected 12B product test
passed. No portability fix was needed. Fresh headless ZIP integrity/version smoke
passed on the development machine. C02 was subsequently completed in the
separate lifecycle qualification below; the next implementation priority is C03.
The baseline itself does not establish long-prefill behavior. Record actual
test revisions and limitations without claiming omitted work passed. Full GPU
and clean-machine campaigns are not release prerequisites under the standing
owner policy of 2026-11-08. The extended QUAL01 waiver remains unchanged.
Studio lifecycle fixes remain in scope; GUI redesign does not.

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

Windows checkpoint (2026-09-12): [prefill cancellation evidence](evidence/windows-c03-prefill-2026-09-12.md).
The separate cancellation callback now reaches the existing 12B and 26B prefill
chunk boundaries. Partial sessions are discarded; no prompt/chunk/precision
fallback is used. Windows live coverage includes ordinary and fixed-D2 on both
public profiles, streaming/nonstreaming disconnect, Responses cancel, bounded
admission saturation and same-session contention, Ctrl+C and restart. The probes
use 12,026-token text prompts at 32K capacity, not everyday-context qualification.
Windows follow-up: [complete C03 phase evidence](evidence/windows-c03-complete-2026-09-12.md)
adds CPU media checkpoints, deterministic deadline/decode tests, natural large-image
disconnect, history preservation and shutdown/restart. The earlier failed deadline
fixture remains retained; its replacement reaches the actual 30-second deadline.
The bounded Linux repetition is complete; broader context/release gates are separate.

- [x] Add safe cancellation checkpoints while a long prefill is running, including
  disconnect for nonstream requests. Inspect `src/runtime/chat.cpp`,
  `src/cuda/inference_session.cuh` and the specialized engine prefill paths under
  `src/cuda/engine/`, together with the server cancellation callbacks. Preserve
  ordinary/fixed-D2 semantics, cache consistency and allocation rules; do not
  simulate cancellation by silently shortening prompts or changing chunk semantics.
- [x] Define the state after partial prefill: safely reusable or explicitly
  discarded/poisoned. Verify that the next request can acquire a working slot.
- [x] On Windows, test deadline/disconnect while queued, waiting for the same session, preparing
  media, prefilling and decoding. Include the 12B two-slot same-session contention
  case; the existing one-slot GPU matrix does not exercise that pool wait.
- [x] Measure Windows cancellation latency and control-route latency under saturation.
  Use an explicit one-second control target and retain raw samples, without treating
  it as a universal guarantee. Exercise Windows Ctrl+C during media, prefill and
  decode, with queued/session waiters; restart and generate again.
- [x] Repeat the applicable phase/deadline matrix on Linux, including SIGINT and
  SIGTERM, ordinary/fixed-D2 and both public profiles. Owner deferred this platform
  repetition; it is not marked passed by Windows evidence.

The FIFO, pool wait deadline propagation, 25 ms disconnect/draining polls and
inference-saturation worker reserve are already implemented in `2096c39`.
**Done when:** abandoned requests never execute later, capacity is recovered,
control routes remain reachable, and each phase has measured platform evidence.
Host-only wait tests do not qualify GPU prefill abort latency.

The 2026-09-12 implementation threads a separate cancellation callback through
`ChatSession::Generate`, `ConversationSession::Generate` and both specialized
prefill paths. The 26B checkpoint uses its existing synchronization; 12B adds
chunk-boundary synchronization only when a callback is supplied. Image spans and
chunk planning are unchanged. The Windows phase/deadline matrix is complete;
continue with Linux checks without repeating the completed callback plumbing. The older C02
slice remains separate historical evidence.

### Requested short comparison pilot

The owner selected approximately 30 minutes for a bounded speed/quantization
comparison. See [the concrete pilot plan](plans/SHORT_CROSS_ENGINE_BENCHMARK.md)
for 512/8K/32K context, separate prefill/decode, quant candidates and full-KL
reference requirements. This is additional diagnostic work, not a reopening of
the closed tuning campaigns or the waived extended QUAL01 campaign. No new
cross-engine result is claimed yet.

### C04 — Short single-user media check

Owner scope update (2026-09-12): see
[active decision](ACTIVE_DECISIONS.md#owner-update-single-user-c04-scope-and-client-priority-2026-09-12).
Normal use is one user and one active conversation. C05/C06 have priority over
additional media stress campaigns or CPU-cache optimization.

Windows results: [bounded C04 evidence](evidence/windows-c04-single-user-2026-09-12.md).
The eight-request scope finished in 75.7 seconds including startup and fixture
continuation. Memory/API/cache checks passed; the 12B semantic failures remain open.
The [bounded Linux repetition](evidence/linux-c03-c06-2026-09-21.md) completed the
same eight requests in 16.27 seconds with the same open 12B semantic finding.

- [x] On Windows, run sequential 1-, 2- and 4-image requests, plus one same-session image-history
  continuation, for each public profile: **eight generation requests total, ten
  minutes wall time maximum**, one qualified decode mode per profile. Keep one
  active session; do not add a two-slot/parallel/D2 Cartesian matrix. Use local
  fixtures and models. Stop at the cap and record unfinished cases without marking
  them passed. These sample counts do not impose a product image-count limit.
- [x] On Windows, record peak process working set and verify successful HTTP
  responses and resident-prefix reuse during those same requests. Reuse
  existing malformed-input/size-limit host tests and C03 media-abort/recovery
  evidence; do not repeat the complete cancellation matrix. Investigate actual
  failures narrowly, without extending the probe into a stress campaign.
- [ ] Resolve the bounded probe's 12B small-image color/history finding: three of
  four semantic checks failed (one image and four-image recall/continuation); 26B
  passed all four. Eight HTTP requests and both cache continuations succeeded.
  Do not silently change preprocessing or launch more GPU tests beyond the agreed
  cap. Keep this focused finding visible alongside C05/C06; no broad C04 campaign.
- [x] Repeat only the same bounded applicable check on Linux when that platform
  is available; Windows results do not establish Linux memory behavior.
- Deferred: broad parallel-media/aggregate-concurrency RSS qualification is no
  longer a C04 requirement for the initial single-user baseline. Existing bounded
  admission and allocation safeguards remain intact.
- Deferred: caching unchanged historical images on the CPU. GPU KV reuse already
  exists; add a bounded CPU cache only if observed client latency justifies it,
  preserving image identity, normalization, preprocessing version and budget.

Start at `src/server/openai_chat.cpp`, `src/model/image.cpp` and
`src/model/image_decode_budget.h` only if a concrete failure requires a fix.
Host/Origin/MIME checks, cumulative image-pixel limits, accounted prepared/resize
bytes and early model identity rejection already exist. C03 adds tested CPU media
cancellation and history recovery, including natural 32-million-pixel PNG probes.
**Done when:** this bounded single-session image/replay check has recorded outcomes
and no unresolved correctness/safety failure. Missing measurements stay explicit;
optional stress/cache work does not block C05/C06. Other release gates are unchanged.

### C05/C06 — Finish the actual client contract

**Next priority after the bounded C04 check (owner direction 2026-09-12).** Start
with new/fork/resume and compaction continuity, then practical replay/tool behavior.
Do not delay this work for the deferred C04 stress or CPU-cache optimization.

[Windows C05/C06 evidence (2026-09-12)](evidence/windows-c05-c06-2026-09-12.md):
52 Python SDK cases, 36 TypeScript cases and 40 additional boundary/recovery
cases passed, plus real Pi editing and lifecycle checks on both public profiles.
Actual generated Responses reasoning replay is rejected; the thinking-off Pi
qualification and other advertised limitations remain unchanged. Required/named
tool enforcement is not implemented by its rejection tests.
The [Linux C03-C06 evidence](evidence/linux-c03-c06-2026-09-21.md) records the
matching pinned SDK, Pi lifecycle and cold-resume repetition for both profiles.

- [x] Exercise unmodified Pi 0.85.0 new session, fork, resume, manual and automatic
  compaction on Windows, including a fresh Pi/server process loading saved history. Check retained facts, session separation, cache
  hits and intentional history/tool resets. Extend `tools/check_pi_compaction.py`
  and `tools/validate_external_agent.py` rather than forking the agent.
- [x] Repeat the extended Pi lifecycle matrix on Linux; prior manual-compaction
  evidence does not establish the new fork/cold-resume/automatic cases.
- [ ] Qualify reasoning replay end to end if practical. Until then, keep the
  delivered Pi configuration on the qualified thinking-off path and make other
  levels explicitly unavailable there; do not silently drop reasoning history.
- [x] Assess the pinned clients on Windows: tested SDK/Pi workflows run without
  per-request sampling; explicit temperature/top_p/seed requests still fail visibly.
- [ ] Implement practical per-request sampling compatibility with resident sampler,
  RNG and fixed-D2 regression evidence; do not silently ignore supplied values.
- [x] Assess required/named tool choice and `parallel_tool_calls=false` on Windows. They
  currently fail early because correct enforcement needs more runtime support.
  Do not replace that with parse-only acceptance or claim constrained decoding.
- [x] Cover multi-round tools, Unicode/long results, bad tool IDs, tool errors,
  output limits and terminal SSE events on both APIs/transports on Windows. Distinguish real
  generated parallel calls from replay fixtures. Align tests and capability docs.

Direct SDK output-object replay, matching max-token aliases and Pi native session
headers/cache/manual compaction already have bounded Linux evidence. Pinned clients
are Python SDK 2.50.0, Node SDK 7.10.0 and Pi 0.85.0.
**Done when:** every advertised capability is actually executed with the pinned
clients, and remaining complex limitations are explicit before costly execution.

### C07/C09 — Package integrity and owner-authorized publication

The standing owner decision of 2026-11-08 removes the former mandatory C08
full GPU/model/SDK/agent campaign, clean-machine and complete model-download
campaigns, and full qualification manifest from release work. They are not
pending blockers or automatic follow-up tasks. Earlier 0.2.x-only exceptions
are superseded for future releases. The
[previous roadmap](archive/ROADMAP_2026-11-08_BEFORE_RELEASE_POLICY.md)
preserves the historical checklist without treating it as current instructions.

For an owner-requested version:

- Build Windows and Linux packages from the matching root VERSION and immutable
  tag; run applicable host/Studio checks and proportional change-specific tests.
- Verify archive contents, versions, source provenance, declared dependencies,
  notices, manifests and hashes. Keep model weights external and preserve all
  runtime/model/precision integrity rules.
- Publish on the owner's release instruction, with the actual validation scope
  and known limitations in the release notes. No further full-campaign exception
  or confirmation is needed. Do not relabel omitted tests as passed.

v0.3.0 has passed both tag builds, host/Studio CI, archive verification and
bounded Windows GPU checks. Full qualification campaigns are not required.
`tools/verify_release_gates.py` remains an optional verifier for a separately
requested full evidence campaign, outside the normal release process.

The internal NVFP4 shared-Hub-root/runtime-view mismatch and normalized Hub-lock
migration remain functional acquisition topics if separately scheduled; neither
creates a mandatory release qualification campaign or authorizes weaker loader
path/symlink/exact-file validation.

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

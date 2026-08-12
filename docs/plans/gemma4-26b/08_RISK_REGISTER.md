# Active risk register — Fast Track R4

| ID | Risk | Severity | Mitigation / trigger |
|---|---|---:|---|
| R1 | QAT BF16 degrades under NVFP4 W4A4 | critical | M13 early screen; M18 only on failure or attribution need |
| R2 | Expert compiler scale/layout bug | critical | exhaustive codecs, real-shape probes, one clean QAT full conversion |
| R3 | 32K margin disappears with real loader/graphs | critical | M09 real admission before full runtime promotion |
| R4 | 64K does not fit despite KV formula | high | M21 measured max-fit; never advertise allocation-only success |
| R5 | Head causes most quality drift | high | provisional NVFP4; M16/M18 targeted head diagnosis; M24 optional |
| R6 | Parallel agents collide in shared runtime files | high | explicit writable paths, interface-first commits, integration owner |
| R7 | Expensive conversions are repeated after review fixes | high | clean-commit preflight before any full run |
| R8 | MoE router/accumulation semantics are subtly wrong | critical | independent M10 oracle and M11 boundary captures |
| R9 | Prefill workspace grows with tokens×experts | critical | compact assignments and M09 cap; M15 bounded chunks |
| R10 | Native decode is slower than expected at batch one | medium | isolated M14 A/B; retain correctness path; optimize measured bottleneck |
| R11 | MTP assistant is incompatible or too large | critical | start M25 feasibility now; separate assistant lock and compressed candidates |
| R12 | MTP reduces available context materially | high | separate MTP max-context qualification; report trade-off explicitly |
| R13 | MTP proposals change target output/state | critical | exact Target verification and transactional commit tests |
| R14 | 12B path regresses through shared loader/runtime work | critical | tiered 12B regression on every shared change |
| R15 | Q4_0 research distracts from vertical bring-up | medium | M24 optional; official Q4 remains external baseline |
| R16 | Status/document drift misroutes the agent | high | one JSON status source and active-contract precedence |

Review risks at M08, M09, M13, M17, M23 and M25. Do not add a new document for every observation; update this table or the relevant milestone evidence.

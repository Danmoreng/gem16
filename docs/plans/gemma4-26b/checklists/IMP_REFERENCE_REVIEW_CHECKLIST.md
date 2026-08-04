# imp reference review checklist

- [ ] Exact imp commit is checked out; no `main` references remain.
- [ ] Selected file hashes are retained.
- [ ] MIT license and copyright are recorded.
- [ ] Official Gemma 4 reference remains normative.
- [ ] ModelOpt and llm-compressor scale directions are separately tested.
- [ ] Router scale-free RMSNorm and `1/sqrt(d)` are verified.
- [ ] FP32 router/logit behavior is covered by late-layer and near-tie fixtures.
- [ ] Shared/expert norms and residual ordering are captured separately.
- [ ] Per-expert scale ordering is tested.
- [ ] 5090 numbers are not presented as 5080 baselines.
- [ ] Paged KV/continuous batching/general executor are out of scope.
- [ ] Any kernel adoption has a layout and permanent-memory report.
- [ ] Confirmed/refuted findings are written to the settled-evidence ledger.

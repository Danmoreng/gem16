# Checkpoint compiler CLI — Fast Track R4

The repository may keep the current Python wrapper while the native executable family evolves. The user-facing flow remains one explicit offline command set:

```text
compile_gemma4_26b.py plan
compile_gemma4_26b.py compile
compile_gemma4_26b.py verify
```

Every action requires an immutable source lock, exact source directory, profile, compiler manifest, memory bounds, output/report path and—where numerical conversion occurs—an explicit native executable and thread count.

## Active profiles

- accepted `fp8-attention-partial-v1` from M05;
- M06 NVFP4 expert partial profile;
- M07 NVFP4 tied-head partial profile;
- M08 complete `sm120-text-hybrid-v1` profile;
- optional M18/M24 diagnostic profiles;
- M25 assistant profile if required.

There is no required Q4_0 profile in M07. Unknown profiles and missing native backends fail before output creation.

## Full-run preflight

Before a full compile, the wrapper records clean Git status, implementation commit, source-lock verification, available disk/RAM, output nonexistence/staging policy and native executable hash. Dirty full runs are rejected unless explicitly marked diagnostic and may not produce accepted evidence.

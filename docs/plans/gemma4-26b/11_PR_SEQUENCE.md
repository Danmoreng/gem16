# Integration and review sequence — Fast Track R4

## Branch policy

`feat/gemma4-26b` is the integration branch. Sub-agents may use short-lived local worktree branches. The integration agent merges reviewed, scoped commits.

## Recommended commit groups

1. shared interface commit, only when parallel lanes require one;
2. lane-local implementation and tests;
3. evidence/report update;
4. integration commit;
5. status update after acceptance.

Do not combine arithmetic, layout and scheduling changes when they can be reviewed separately.

## Current sequence

- Land M06 compiler commits while M10A/M12A/M09A/M25A proceed independently.
- Land M07 as a small compiler/head slice.
- Integrate M08 artifact/schema/loader centrally.
- Complete M09 real allocation reconciliation.
- Merge M10/M11 and M12 runtime adapters into M13.
- After M13 pass, merge M14/M15/M16 independently and continuously test M17.
- Freeze one M17 artifact before M19–M22 parallel qualification/product work.
- M23 aggregates exact-hash evidence.
- M25 integrates MTP after the base target is frozen.

A pull request is opened only when the project owner requests publication. Milestone commits and evidence are sufficient for internal progression.

# imp license and provenance plan

## Scope

The pinned imp repository is MIT licensed. This document is an engineering provenance plan, not legal advice.

## Required repository files if code is copied

```text
LICENSE                              existing gem16 Apache-2.0 license
LICENSES/imp-MIT.txt                 exact pinned imp MIT text
THIRD_PARTY.md                       component table and source commit
NOTICE or equivalent                 only if gem16 policy requires it
src/third_party/imp/...              directly copied MIT files, if any
```

Directly copied source retains:

```text
SPDX-License-Identifier: MIT
Copyright (c) 2026 kekzl
Source: kekzl/imp@a392904d4216388828d0d56317de046f4ca49627, <path>
Modified for gem16: <summary/date>
```

New gem16 adapters or independently written kernels may remain Apache-2.0, but their provenance record must distinguish inspiration/reference from copied code.

## `THIRD_PARTY.md` minimum fields

- component name;
- repository URL;
- immutable commit;
- original path(s);
- copied or derived destination path(s);
- license and copyright;
- modification summary;
- source and destination SHA-256;
- decision record/PR;
- test evidence;
- update policy.

## Dependency boundary

Do not vendor imp's entire tree or build system. Import only approved files. Re-audit includes and transitive copied code so that a nominally MIT file does not silently drag a separately licensed vendored dependency.

## Update policy

A later imp commit is a new source. Updating requires:

- new source lock;
- diff review from the previously approved commit;
- refreshed license/provenance hashes;
- correctness and performance reruns;
- no blind merge from `main`.

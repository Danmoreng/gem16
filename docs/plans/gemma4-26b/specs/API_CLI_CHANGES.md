# API and CLI changes — Fast Track R4

## Base profile at M22/M23

Expose model variant, weight profile, head format, context limits, resident bytes and actual native paths. Support one 26B slot. Reject media, MTP options and a second slot clearly while `supports_mtp=false`.

CLI/server are release-critical. Studio/catalog integration is nonblocking and may follow the base checkpoint.

## Context metadata

Expose:

```text
default_context=32768
qualified_64k=<true|false>
base_max_context=<measured>
mtp_max_context=<unset until M25>
```

Do not report architecture maximum as a qualified runtime maximum.

## MTP at M25

Implementation commit `c4ead1d` exposes the Assistant resident bytes, fixed proposal depth, exact sampled Target
verification capability, acceptance counters and the configured 32K `mtp_max_context` in CLI/server health. Studio
selects the same fixed-D2 profile. Adaptive 26B MTP is rejected. The published maximum remains a configured product
checkpoint rather than a formally qualified limit until the M25 64K context gate completes.

Enabling MTP must not silently lower a user-requested context; admission returns a precise error or the user selects
a documented MTP context profile.

All new fields are additive and 12B behavior remains compatible.

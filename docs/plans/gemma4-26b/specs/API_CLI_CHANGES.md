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

After qualification, expose assistant identity, proposal modes, exact-verification capability, acceptance counters and `mtp_max_context`. Enabling MTP must not silently lower a user-requested context; admission returns a precise error or the user selects a documented MTP context profile.

All new fields are additive and 12B behavior remains compatible.

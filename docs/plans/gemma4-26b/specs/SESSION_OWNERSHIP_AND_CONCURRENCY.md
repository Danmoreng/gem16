# Session ownership and concurrency — 26B Fast Track

The 16 GB 26B profile supports one resident execution slot. Immutable weights are process-owned; mutable KV, workspace, graph and sampling state belong to that slot.

Required tests:

- one slot initializes and runs;
- a second 26B slot request fails before partial allocation;
- resident multi-turn continuation preserves exact prefix ownership;
- cancellation invalidates partial mutable state safely;
- 12B multi-slot behavior remains unchanged.

MTP uses the same target slot plus a separately accounted assistant arena and verifier workspace. It does not create a second target slot. Continuous batching and positive 26B multi-slot scaling are outside the program.

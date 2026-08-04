# CUDA state lifecycle checklist

- [ ] Every arena allocation has one named owner.
- [ ] Module-static device/pinned pointers are inventoried.
- [ ] Every borrowed static pointer has a reset hook.
- [ ] Graphs/events/streams are destroyed or synchronized before backing memory is invalid.
- [ ] Reset/free ordering is documented.
- [ ] Ten create/infer/destroy cycles pass in one process.
- [ ] Init-failure then retry passes.
- [ ] Graph-capture failure then retry passes.
- [ ] CUDA sticky error is checked rather than hidden as a test skip.
- [ ] VRAM returns to baseline within tolerance.
- [ ] Second engine does not reuse a stale workspace/descriptor.

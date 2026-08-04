# Third-party code import checklist

- [ ] Owner-approved decision record exists.
- [ ] Exact repository, commit, path and source hash are recorded.
- [ ] Original license applies to the selected files.
- [ ] Original copyright and permission notice are preserved.
- [ ] `LICENSES/<component>.txt` is added.
- [ ] `THIRD_PARTY.md` entry is complete.
- [ ] Direct copies use the original SPDX identifier.
- [ ] New adapters clearly state their own license/provenance.
- [ ] Transitive includes/vendor code have been audited.
- [ ] Destination is isolated from unrelated production code.
- [ ] Modifications are documented and reviewable as a patch.
- [ ] Independent tests cover correctness, memory and lifecycle.
- [ ] No duplicate permanent weight representation is introduced.
- [ ] Update/rollback procedure is documented.

# Windows context refinement with 200 MiB reserve

Source: `56e83c3906d89a7e4941af37aa3a768e912ae088` plus the existing local
Windows fixes and 200 MiB admission policy. Same locked Compact Vision Target,
Vision (280), fixed-D2 Assistant, one slot and default 2048-token prefill.
No runtime or numerical code changed in this refinement.

| Context | Free at admission | Text and scene image smoke |
|---|---:|---|
| 173,000 | 229,638,144 bytes (219 MiB) | Pass |
| 174,000 | 219,152,384 bytes (209 MiB) | Pass |
| 174,500 | 212,860,928 bytes (203 MiB) | Pass on two independent starts |
| 174,750 | 210,763,776 bytes (201 MiB) | Text passes; image times out after 180 seconds |
| 175,000 | Not admitted | Assistant/verifier reserve check rejects |
| 176,000 | Not admitted | Assistant/verifier reserve check rejects |

The highest successfully repeated setting is 174,500. The 174,750 image timeout
is observed but its cause is not established; passing admission alone is not
sufficient qualification. The probes reserve the configured context and execute
short requests, not a filled 174K prompt or long-context quality test. WDDM budgets
can change. Studio ran as a hidden process during these probes; the subsequent
local configuration is set to 174,500 and Studio is relaunched visibly with
autostart restored. The portable Windows default remains 170,000.

Reproduction commands:

```powershell
python build/check_gui_context_refinement_20260905.py
python build/check_gui_context_fine_20260905.py
```

Both sweep drivers finish successfully; individual failed points retain their
nonzero child exit codes. Raw commands, health responses, generated answers,
server logs and timeout traceback are retained in ignored `build/windows-gui-refine-*`
and `build/windows-gui-fine-*` files. Summaries:
`build/windows-gui-context-refinement-20260905.json` and
`build/windows-gui-context-fine-20260905.json`. The pre-test saved Studio settings
are backed up in `build/studio-before-context-refinement-20260905.conf`.

The relaunched Studio-managed server also passed the text and all four image
checks at 174,500, again reporting 212,860,928 free bytes. Its public model ID is
`gemma4-26b-a4b-trellis35-vision-fp8`; an initial test request using the standalone
probe's `gem16` alias returned HTTP 404 and was corrected to the advertised ID.
Reproduce with `python build/check_studio_visible_174500_20260905.py`; retained
response: `build/windows-studio-visible-174500-20260905.json`.

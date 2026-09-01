# Gemma 4 26B Vision V19 bounded quality suite

This project-owned suite covers image description, OCR, a chart, a document
page, counting, spatial relations, colors, small details, and square/wide/tall
geometry at the locked 70, 140, and 280 soft-token budgets.

The PNGs are deterministic rasterizations of embedded SVG sources in
`tools/generate_gemma4_26b_vision_v19_fixtures.py`. Regenerate them with:

```bash
python3 tools/generate_gemma4_26b_vision_v19_fixtures.py
```

`suite.json` pins every image hash, prompt, factual check, and the budgets at
which that check is a mandatory gate. The 18-pixel footer in the tall document
is deliberately required only at budget 280; lower-budget observations remain
in the report but do not redefine the image processing budget.

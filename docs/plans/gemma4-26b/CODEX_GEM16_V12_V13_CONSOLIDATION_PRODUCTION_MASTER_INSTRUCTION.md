# Codex Master Instruction — V12/V13, Trellis Device Image v2 und produktiver 26B-Vision-Release

## Ausgangspunkt

Arbeite auf:

```text
branch:
  codex/gemma4-26b-vision-fp8

commit:
  26691d607f7945234a39aed8a98f0d1ed1d904c1
```

Der Snapshot ist derzeit nicht sauber:

```text
?? tests/test_image_vision.png
```

Lege keine Commits, Branchwechsel, Pushes, Rebases oder Resets ohne ausdrückliche Owner-Autorisierung an. Bewahre fremde Änderungen.

Vor jedem Work Package lesen:

```text
AGENTS.md
docs/ACTIVE_DECISIONS.md
docs/PRODUCT_CONTRACT.md
docs/plans/gemma4-26b/V00_VISION_PROFILE_AND_MODULE_CONTRACT.md
GEM16_26B_VISION_PRODUCTION_AND_CONSOLIDATION_REVIEW.md
docs/plans/gemma4-26b/PRODUCTION_26B_VISION_CONTRACT.md
```

---

# 1. Neue Owner-Richtung

Das Endziel ist kein experimenteller V20-Freeze.

Das Ziel ist ein produktiv qualifiziertes drittes Profil:

```text
Gemma 4 26B A4B – Compact Vision
Trellis35 Text + FP8 Vision + Fixed-D2
```

Der Scope bleibt begrenzt und explizit:

```text
SM120
Batch 1
ein Bild
70/140/280 Vision Soft Tokens
Textausgabe
Ordinary und Fixed-D2
kein Audio
kein Video
gemessene Kontextgrenzen
```

Die bestehenden Pfade bleiben unverändert unterstützt:

```text
12B multimodal
26B NVFP4 text
26B Trellis35 text
26B Trellis35 + FP8 Vision
```

Produktiv bedeutet nicht, dass ungetestete allgemeine Multimodalität behauptet werden darf. Der konkrete Vertrag muss vollständig qualifiziert werden.

---

# 2. Unverhandelbare Regeln

- Kein Runtime-JIT.
- Kein Runtime-Weight-Repack.
- Keine Runtime-Quantisierung.
- Kein CPU-Weight-Offload.
- Kein Expert-Streaming.
- Keine zweite persistente routed-expert-Repräsentation.
- Keine zweite persistente Vision-Weight-Repräsentation.
- Keine Token-/Bild-Loop-Allokation.
- Keine stille Precision-/Format-/Kernel-/Context-/Budget-/D2-Absenkung.
- Keine Änderung von BF16-/FP32-Grenzen ohne vorab definierte numerische Requalifikation.
- Alte Evidenz nie überschreiben.
- Jeder Performancekandidat hat einen Same-Binary-Rollback.
- Ein Microkernel-Gewinn ohne Full-Tower-/TTFT-Gewinn wird nicht automatisch produktiv.
- V1- und V2-Trellis-Dateiformate dürfen nicht anhand zufälliger Dateiexistenz verwechselt werden.
- Ein beschädigtes v2 darf nie still auf v1 zurückfallen.
- Das konsolidierte Hugging-Face-Repository bleibt der einzige kanonische 26B-Repository-Ort.

---

# PRD00 — Clean Snapshot, Produktvertrag und Security

## PRD00-A — Fixture bereinigen

Entscheide explizit über:

```text
tests/test_image_vision.png
```

Option 1:

- nach `tests/fixtures/gemma4_26b_vision/` verschieben;
- Generator oder nachvollziehbare Source hinzufügen;
- SHA-256;
- Lizenz/Provenienz;
- betroffene Evidenz mit getracktem Pfad wiederholen.

Option 2:

- Referenz entfernen;
- mit bestehenden getrackten Fixtures erneut ausführen.

Acceptance:

```text
git status --porcelain == leer
Review-Archiv enthält alle referenzierten Fixtures
kein Evidence-File verweist auf lokale, fehlende Dateien
```

## PRD00-B — kryptographische Bildidentität

Ersetze 64-Bit-FNV als authoritative Cache-Identität.

Touchpoints:

```text
include/gem16/image.h
src/model/image.cpp
src/runtime/chat.cpp
tests/unit/image_test.cpp
tests/unit/chat*
12B Media-/Session-Regressions
```

Zielstruktur:

```cpp
struct ImageSourceIdentity {
  std::array<std::uint8_t, 32> sha256{};
  std::uint64_t encoded_bytes = 0U;
};
```

Oder eine gleichwertige immutable Repräsentation.

Regeln:

- Digest über originale kodierte Bytes;
- Länge zusätzlich vergleichen;
- kein Digest für in-process Bilder vortäuschen;
- ohne Digest vollständige relevante Struktur vergleichen;
- ResidentMessageEquivalent darf nur mit starker Identität abkürzen.

Tests:

```text
gleiche Bytes -> gleich
andere Bytes -> ungleich
gleiche Abmessungen/ähnliche Patches -> ungleich
kein Digest -> struktureller Vergleich
12B cached continuation
26B Vision cached continuation
```

## PRD00-C — stabile Produktprofilidentität

Aktuell ist D2 Teil von `profile_id`.

Ändere auf:

```text
profile_id:
  gemma4-26b-a4b-trellis35-vision-fp8

decode_mode:
  ordinary | fixed-d2

vision_mtp_supported:
  bool

qualification_state:
  development
  production_candidate
  production_qualified
```

Qualification wird an exakte Target/Vision/Assistant-Hashes gebunden.

Trenne:

```text
vision_max_soft_token_budget
last_vision_soft_token_budget
```

Nicht `selected_vision_soft_token_budget()` als statisches Profile-Attribut verwenden.

## PRD00-D — Produktdokument vorbereiten

Aktualisiere noch nicht die UI auf „produktiv“.

Erzeuge zunächst:

```text
docs/plans/gemma4-26b/PRODUCTION_26B_VISION_CONTRACT.md
```

mit:

- Scope;
- Plattformen;
- Komponenten;
- Präzision;
- Context;
- D2;
- Bildgrenzen;
- Quality Claim;
- Release-Gates.

`docs/ACTIVE_DECISIONS.md` darf erst nach P20 behaupten, dass der produktive Freeze akzeptiert ist.

## Acceptance

- Host/ASAN/UBSAN;
- 12B regressions;
- NVFP4 regressions;
- Trellis text regressions;
- Vision Ordinary/D2 regressions;
- clean worktree;
- keine Performanceänderung.

Stoppe nach PRD00 und präsentiere die Evidenz.

---

# PERF13 — Vision Attention als primärer Performancehebel

## PERF13-0 — Final-Commit-Baseline

Wiederhole die V10-Basis auf dem aktuellen Commit und den final getrackten Fixtures.

Budgets:

```text
70 / 140 / 280
```

Geometrien:

```text
square / wide / tall
```

Messung:

```text
3 Warm-ups
10 retained
```

Erfasse:

- Host Decode/Resize/Patchify;
- Upload;
- Tower;
- per-stage;
- Text Prefill;
- TTFT;
- post-first Ordinary;
- post-first D2;
- Peak VRAM;
- Output/Draft Hashes.

NCU/NSYS für den aktuellen Attention-Kernel bei 70/140/280.

Schreibe:

```text
artifacts/vision/perf13-current-baseline.json
```

---

## PERF13-A — exakter scalar tiled K/V-sharing Kernel

### Ziel

K/V einmal je Source-Tile laden und über mehrere Query-Tokens desselben Heads wiederverwenden, ohne die Attention-Mathematik zu ändern.

### Touchpoints

```text
src/cuda/vision/gemma4_26b.cu
src/cuda/vision/gemma4_26b.h
tests/cuda/vision26b_attention_test.cu
tests/cuda/vision26b_test_support.*
```

Falls die Vision-Datei zu groß wird, nur codegen-neutral in Include-Shards derselben CUDA-TU aufteilen:

```text
src/cuda/vision/detail/
  attention_reference.inc.cuh
  attention_tiled.inc.cuh
  rope.inc.cuh
  pool.inc.cuh
  runtime.inc.cuh
```

### Kandidaten

```text
A: Q4 / K32, 128 Threads
B: Q8 / K32, 256 Threads
C: Q8 / K64, 256 Threads
```

Beginne mit B.

### Grid

```text
grid.x = ceil(tower_tokens / Q_TILE)
grid.y = 16 Heads
```

Eine CTA verarbeitet mehrere Token-Queries desselben Heads.

### Shared

Transienter physischer Stride:

```text
Head 72 -> Shared Stride 80
```

Keine persistente Shapeänderung.

K32:

```text
K[32][80] BF16
V[32][80] BF16
```

K64 entsprechend.

### Pass 1

- Q pro Query-Warp in Registern.
- K kooperativ in Shared.
- Source-Reihenfolge exakt 0..valid_tokens-1.
- Channel-Aufteilung und WarpSum zunächst exakt wie Reference.
- FP32 running_max/denominator unverändert.

### Pass 2

- K/V gemeinsam in Shared.
- QK exakt erneut.
- Probability exakt zu BF16 runden.
- BF16 Probability × BF16 V in FP32.
- Output zu BF16.

### Tail

- Source Tail nullfüllen, aber nicht numerisch konsumieren.
- Query über `tower_tokens`, nicht ungeprüft über `raw_patch_count`.
- Head-Tail 72 bleibt logisch.
- Kein Score-Slab.

### Double Buffer

Erst nach exakter synchroner Version:

```text
cp.async
2 K-Buffer in Pass 1
2 K/V-Buffer in Pass 2
```

### Rollback

```text
GEM16_VISION_ATTENTION=reference|q4k32|q8k32|q8k64
```

Default zunächst `reference`.

### Korrektheit

```text
tokens:
  9,18,63,64,65,576,630,1260,2394,2520

valid/tower combinations:
  partial and full padding

all 16 heads
all channel tails
square/wide/tall
layer 0/13/26
final embeddings
Ordinary
D2
```

Bevorzugt bitidentisch.

### Ressourcen-/Performance-Gate

```text
keine Spills
kein Score-Slab
bounded Shared
keine Arenaänderung
mindestens 1,5× Attention-Speedup bei 140 und 280
oder >=20 % Full-Tower-Speedup bei beiden
Budget70 nicht >3 % langsamer
finale Outputs/D2 exakt
```

Wenn kein Kandidat besteht, Rollback und PERF13-B prüfen.

---

## PERF13-B — BF16 Tensor-Core QK

Nicht gemeinsam mit A implementieren.

Zulassung nur, wenn PERF13-A korrekt, aber nicht schnell genug ist.

### Design

```text
Q16 × Source-N32
K = 72, transient auf 80 oder 96 nullpad
BF16 Tensor Core
FP32 Scores
```

Kein persistentes Padding.

Two-Pass bleibt:

```text
Pass 1:
  QK Score-Tiles
  FP32 running max/denominator

Pass 2:
  QK erneut
  BF16 Probability
  scalar/shared PV zunächst unverändert
```

Kein One-Pass-Softmax im ersten Kandidaten.

### Numerische Grenzen vor Messung einfrieren

```text
Attention:
  relative L2
  cosine
  max abs

Layer:
  0 / 13 / 26

Final Vision embedding:
  relative L2
  cosine

Text:
  first logits
  top1
  deterministic generations

D2:
  forced proposals
  real Assistant
  final stream
  KV commit
```

### Rollback

```text
GEM16_VISION_ATTENTION=reference|scalar_tiled|tc_qk
```

### Admission

Dieser numerisch invasivere Kandidat bleibt nur bei materiellem Nutzen:

```text
>=35 % Full-Tower-Gewinn bei Budget280
>=25 % bei Budget140
keine relevante Qualitätsregression
vollständige V11/V14-Neuqualifikation
```

Keine Grenzen nach Sichtung der Ergebnisse lockern.

---

## PERF13-C — optional später

Tensor-Core PV oder One-Pass-Online-Softmax nur nach separater Owner-Freigabe und nur, wenn PV danach nachweislich dominiert.

---

# PERF12 — Low-Risk Performance und Capacity

Auf dem besten PERF13-Stand arbeiten.

Jeder Kandidat separater Commit/Same-Binary-A-B.

---

## PERF12-A — direktes 3×3 Pooling

Ersetze den vollständigen Raw-Token-Scan je Output durch neun direkte Loads.

Voraussetzung:

```text
kanonische row-major Positionen aus V09
```

Summationsreihenfolge exakt:

```text
row0 col0..2
row1 col0..2
row2 col0..2
```

Dann unverändert:

```text
sum / 9
BF16 round
sqrt(1152) FP32
bias/scale
BF16 round
```

Rollback:

```text
GEM16_VISION_POOL=scan|direct
```

Acceptance:

```text
bitidentisch
Pool-Kernel stark schneller
Full Tower nicht langsamer
kein Speicher-/Allocation-Delta
```

---

## PERF12-B — 2D-RoPE-Table

Erzeuge einmal je Bild eine bounded BF16-Tabelle.

Einfacher maximaler Vertrag:

```text
[tower_token][36][cos,sin]
<= 362,880 Byte
```

Padded Positionen sicher behandeln.

Exakte Formel/Rundung beibehalten.

Rollback:

```text
GEM16_VISION_ROPE=transcendental|table
```

Acceptance:

```text
Q/K bitidentisch
Special-Function-Instruktionen materiell reduziert
Full Tower gewinnt
Table im Workspace/Capacity bilanziert
```

---

## PERF12-C — GELU + Product Quant

Eine CTA pro Tower-Token.

Exakte Reihenfolge:

```cpp
gate = BF16(gate)
up = BF16(up)
product_bits = BF16Bits(GeluTanh(gate) * up)
product = BF16(product_bits)
amax = max(abs(product))
scale
E4M3(product / scale)
```

4304 gerundete BF16-Werte dürfen transient in Shared gehalten werden, wenn ptxas/Occupancy gewinnt.

Rollback:

```text
GEM16_VISION_FFN_QUANT=split|fused
```

Acceptance:

```text
E4M3 bytes/scales identisch
Down und final output identisch
GELU+Quant >=1,5× schneller
Full Tower gewinnt oder ist neutral ohne zusätzlichen Speicher
```

---

## PERF12-D — konfiguriertes Maximalbudget und Workspace

Neues Server-/Runtimefeld:

```text
vision_max_soft_token_budget = 70|140|280
```

Request:

```text
vision_soft_token_budget <= configured max
```

Planungsziele, vom Code neu zu berechnen:

```text
70:   ~36,290,048 Byte
140:  ~64,191,488 Byte
280:  119,993,600 Byte
```

Keine Request-Allokation.

Studio:

```text
70 Fast
140 Balanced
280 Maximum detail
```

Eine Änderung des Maximalbudgets erfordert Neustart.

Expose getrennt:

```text
vision_max_soft_token_budget
last_vision_soft_token_budget
vision_workspace_bytes
```

Nach Acceptance Capacity komplett neu ausführen.

---

## PERF12-E — optional Input-Staging

Nur wenn Final-Baseline einen relevanten Host-/Uploadanteil zeigt.

- zwei pinned Slots;
- Event-Reuse;
- kein globaler Sync;
- bounded;
- cancellation-safe.

---

## PERF12-F — bounded CUDA Graphs

Erst wenn alle Kernel stabil sind.

Maximal drei Graphfamilien:

```text
70
140
280
```

oder ein dokumentierter kleiner Shape-Satz.

Graph-private Bytes in Capacity aufnehmen.

---

# FMT01 — Trellis35 Device Image v2

## Ziel

Das aktuelle 103-Dateien-Trellis-Produkt in ein einzelnes GPU-fertiges Payload umwandeln.

Keine Requantisierung.

## v2-Layout

```text
format:
  gem16-sm120-trellis35-device-image-v2

trellis35/model.gem16:
  12,204,692,480 Byte

offset 0:
  non-routed 1,850,270,720 Byte

offset 1,850,270,720 + layer*345,147,392:
  layer 0..29
```

Produktmetadaten:

```text
gem16_model.json
gem16_compilation.json
gem16.lock.json
```

Das Binärfile hat keinen eingebetteten Header vor dem GPU-Arena-Offset 0.

## Packager

Neu:

```text
tools/build_gemma4_26b_trellis35_device_image.py
```

Muss:

- v1 vollständig hashvalidieren;
- exact size/order prüfen;
- `.partial` exklusiv;
- preallocate;
- streaming concat;
- Gesamt-SHA;
- fsync;
- atomic rename;
- v2 locks/manifests;
- unabhängige zweite Streaming-Verifikation.

## Loader

Neu/Refactor:

```text
src/cuda/engine/gemma4_26b_trellis35_device_image.*
```

Verwende die bestehende NVFP4-Pipeline:

```text
4 × 64 MiB pinned
CUDA Events
async
optional cuFile
structural/full SHA
```

Weiterhin exakt ein `cudaMalloc`.

## Legacy

```text
v2 = Produkt
v1 = expliziter Legacy-/Dev-Loader
```

Kein stiller Fallback.

## Tests

- byte-exakter v1/v2-Arena-Vergleich;
- Pointer/Scalar Bindings;
- Ordinary;
- D2;
- Vision;
- Capacity;
- Load Time cold/warm;
- corruption;
- truncation;
- wrong hash;
- symlink;
- mixed v1/v2;
- CUDA allocation count.

Evidence:

```text
artifacts/trellis35/fmt01-device-image-v2.json
```

---

# PUB01 — neues konsolidiertes Hugging-Face-Revision

Repository bleibt:

```text
danmoreng/gemma-4-26B-A4B-it-GEM16
```

Neue immutable Revision.

Layout:

```text
root/
  NVFP4

trellis35/
  model.gem16
  gem16_model.json
  gem16_compilation.json
  gem16.lock.json
  README/LICENSE/NOTICE as required

assistant/
  existing component

vision/
  vision.gem16
  gem16_vision.json
  vision_compilation.json
  vision.lock.json
```

Entferne die 30 Layer-Unterverzeichnisse aus der neuen Snapshotrevision.

Nicht alle Komponenten zu einer einzigen Gesamtdatei verschmelzen.

Neu erzeugen:

- component locks;
- compatibility manifest;
- SHA256SUMS;
- generated Studio catalog;
- runtime hardlink views.

Verifikation:

```text
anonymous download
resume at multiple offsets in 12.2GB file
full hash
second preflight missing bytes = 0
hardlink identity
no private duplicate blob store
old revision remains reproducible
```

---

# QUAL01 — produktive Text- und Vision-Qualifikation

## QUAL01-A — Trellis Text

Schließe oder ersetze die bislang aufgeschobene WP8B-Suite per ausdrücklicher Owner-Entscheidung.

Minimum:

```text
GSM8K
AIME
GPQA fixed subset/full
64K retrieval
>=128K retrieval
Instruction/Prose
Tool calls
Stop
Repetition penalty
Greedy/Sampled
Ordinary/D2
```

Vergleich gegen gepinnten NVFP4-Control.

## QUAL01-B — Vision

Mindestens 40–60 eingefrorene Fälle.

Kategorien und Geometrien gemäß Review.

Für 12–20 Fälle:

```text
BF16 Vision embedding oracle
FP8 final embeddings
```

Zusätzlich:

- deterministic duplicate;
- Ordinary/D2;
- budgets;
- OCR small text;
- natural images;
- charts/docs;
- failures.

## QUAL01-C — Capacity

Nach PERF12/PERF13/FMT01:

```text
Target+Vision
Target+Vision+Assistant
max budget 70/140/280
32K
64K
196608
212992
229120
first reject
```

Zweimal pro Punkt, frischer Prozess.

Veröffentliche:

```text
production default
advanced maximum
reserve
```

## QUAL01-D — Windows/Linux

Live SM120 Vision+D2 auf beiden Produktplattformen.

Falls Windows nicht live qualifiziert werden kann, darf der Product Contract Windows für dieses Profil nicht behaupten.

---

# APP01 — Native Studio Produktionsmigration

Nach PUB01/QUAL01:

- Catalog revision aktualisieren;
- Trellis-v2-Dateiliste;
- keine Layer-Dateien;
- Profile label ohne Experimental;
- `experimental=false`;
- `qualification_state=production_qualified`;
- Stable Profile ID;
- decode mode separat;
- Budgetlabels Fast/Balanced/Maximum detail;
- finalen Default gemäß Capacity/Quality setzen;
- Max/Default Context getrennt;
- Component install/verify/remove;
- old revision migration;
- external-server compatibility.

Keine UI darf Produktstatus aus lokaler Auswahl ableiten; Health und exakte Component Hashes müssen passen.

---

# REL01 — Release Candidate

Erzeuge finalen sauberen Commit und immutable Builds.

Pflicht:

```text
3 Warm-ups / 10 retained
70/140/280
Ordinary/D2
Vision Tower
TTFT
post-first Decode
Peak VRAM
fallback=0
allocation delta=0
```

Zusätzlich:

- clean-machine Windows/Linux;
- full large download;
- resume;
- corruption;
- cancellation;
- repeated images;
- cached conversation;
- queueing;
- server restart;
- package notices/licenses;
- OpenAI Chat/Responses;
- Native Studio;
- 12B;
- 26B NVFP4;
- Trellis text;
- Vision.

---

# P20 — Production Qualification

Akzeptiere nur, wenn:

```text
PERF12/13 final
FMT01 final
PUB01 final
Trellis text quality final
Vision quality final
Capacity final
Windows/Linux final
Studio final
Packaging final
clean git
immutable hashes
```

Dann ändern:

```text
qualification_state = production_qualified
experimental = false
```

und aktualisiere Product Contract.

---

# P21 — Production Release Freeze

Freeze:

```text
source commit
Windows binary hashes
Linux binary hashes
HF repository revision
all component hashes
catalog hash
default/max context
default/max Vision budget
performance panel
quality panel
known limitations
rollback/recovery
```

Der produktive Claim bleibt exakt begrenzt auf den qualifizierten Scope.

---

# Empfohlene Ausführungsreihenfolge

```text
PRD00
  ↓
PERF13-A
  ↓
PERF13-B nur falls nötig
  ↓
PERF12
  ↓
FMT01
  ↓
PUB01
  ↓
QUAL01
  ↓
APP01
  ↓
REL01
  ↓
P20
  ↓
P21
```

FMT01 kann in einem separaten Worktree parallel zu PERF13/PERF12 entwickelt werden. Integration und Requalifikation bleiben seriell.

---

# Erster Codex-Auftrag

Bearbeite **nur PRD00**.

Danach stoppe und liefere:

- clean git status;
- Fixture-Entscheidung;
- SHA-256-Image-Identity;
- stable profile/decode-mode schema;
- alle Regressionen;
- keine Performanceänderung.

Anschließend beginnt PERF13-0/PERF13-A.

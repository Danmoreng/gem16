# Gem16 26B Vision: Produktionsreview, V12/V13 und Trellis-Device-Image v2

**Review-Basis:** `codex/gemma4-26b-vision-fp8@26691d607f7945234a39aed8a98f0d1ed1d904c1`

**Archiv:** `gem16-chatgpt-review-2026-09-01-26691d60-dirty.zip`

**Datum:** 2026-09-01

**Review-Art:** statische Quellcodeanalyse plus Prüfung der enthaltenen V09–V19-, Trellis-, Studio-, Packaging-, Capacity-, D2- und Quality-Evidenz

**Nicht selbst ausgeführt:** CUDA-Benchmarks, Nsight-Captures, vollständige Modell-Downloads und Windows-SM120-Inferenz

---

## 1. Gesamtentscheidung

Der aktuelle Stand ist deutlich weiter als ein Vision-Prototyp:

- Ordinary Vision ist semantisch geschlossen.
- Vision plus Fixed-D2 ist exakt und qualifiziert für den getesteten Composite.
- Server, OpenAI-Pfade, Capability-Reporting, Cancellation und Metrics sind umgesetzt.
- Native Studio besitzt ein explizites drittes Profil, Komponenteninstallation und Bild-UX.
- Target, Assistant, Vision und Trellis liegen bereits gemeinsam in einem öffentlichen Hugging-Face-Repository.
- Kontext- und Speichergrenzen wurden frisch gemessen.
- Eine begrenzte objektive Bildsuite wurde bestanden.

Die zwei ausgelassenen Performancepakete sind weiterhin sinnvoll. Ihre Priorität ist aber ungleich:

> **V13 – tiled Vision Attention – ist der zwingende Haupthebel. V12 ist eine anschließende beziehungsweise teilweise parallele Cleanup-/Capacity-Welle.**

Die gemessene Vision-Attention belegt bei Budget 70 ungefähr 85 %, bei 140 ungefähr 92 % und bei 280 ungefähr 96 % der Tower-GPU-Zeit. Die übrigen V12-Bausteine können den aktuellen ungefähr 947-ms-Tower bei Budget 280 nicht grundlegend verändern. V13 kann das.

Für einen ehrlichen **produktiven** statt experimentellen Freeze fehlen jedoch mehr als V12/V13:

1. ein sauberer Git-Snapshot ohne ungetracktes Testbild;
2. ein kollisionsresistenter Bildidentitätsvertrag;
3. eine vollständige Produktqualifikation des Trellis35-Text-Targets;
4. eine breitere Vision-Qualitätsevidenz als drei synthetische Bilder;
5. ein live ausgeführtes Windows-SM120-Gate oder eine ausdrücklich geänderte Plattformentscheidung;
6. eine finale Capacity-Neuqualifikation nach V12/V13;
7. stabile Produktprofil-/Qualification-Semantik statt `experimental_*`;
8. die Umstellung des Trellis35-Checkpoints auf ein einziges GPU-fertiges `model.gem16`;
9. eine neue immutable Revision des bereits konsolidierten Hugging-Face-Repositories;
10. finale Clean-Machine-/Packaging-/Release-Evidenz.

Die richtige Zielentscheidung lautet deshalb nicht mehr „V20 Experimental Freeze“, sondern:

```text
P20  Production Qualification
P21  Production Release Freeze
```

Das Endprofil darf produktiv sein, aber seine Aussage bleibt bewusst begrenzt:

```text
Gemma 4 26B A4B Trellis35 + FP8 Vision
SM120
Batch 1
ein Bild
70/140/280 Bildtokens
Fixed-D2
gemessene Kontextgrenzen
```

„Produktiv“ muss nicht „beliebige Multimodalität und allgemein bewiesene Spitzenqualität“ bedeuten. Es bedeutet, dass genau dieser veröffentlichte Vertrag vollständig getestet, installierbar, reproduzierbar und nicht mehr als experimentell ausgewiesen ist.

---

# 2. Was V09–V19 tatsächlich bereits geschlossen haben

## 2.1 Vision-Semantik

V09 hat die wichtigen Modellgrenzen geschlossen:

```text
lokale Text-Attention:
  sliding AND (causal OR gleicher Bildblock)

globale Text-Attention:
  kausal

Bildspan:
  wird nicht über Prefill-Chunks geteilt

Positionen:
  kanonisch row-major

Vision-Tower:
  630 / 1260 / 2520 physische Reihen
  abhängig von Budget 70 / 140 / 280
```

Der Padding-Oracle hat gezeigt, dass der gepaddete Ausführungspfad für die eingefrorene Numerik benötigt wird. Eine spätere Performanceoptimierung darf deshalb nicht einfach nur `raw_patch_count` statt `tower_tokens` durch alle linearen Layer schicken.

## 2.2 Fixed-D2

V11 und V14 haben die ursprüngliche Vision+D2-Abweichung geschlossen:

- Forced-Proposals waren exakt.
- T3-Reihen stimmten mit sequenziellen Target-Forwards überein.
- KV-Backup, tentative State, Restore und Commit waren exakt.
- Reale Assistant-Proposals erzeugten denselben finalen Stream.
- Der Vision-Tower läuft einmal und nicht pro D2-Gruppe.

Damit ist D2 architektonisch kein offener Blocker mehr.

## 2.3 Aktuelle Laufzeiten

Die belastbare V10-Basis auf der RTX 5080 Laptop GPU lautet:

| Budget | Rohpatches | Vision-Tower | TTFT |
|---:|---:|---:|---:|
| 70 | 630 | 63,47 ms | 131,01 ms |
| 140 | 1.260 | 231,67 ms | 308,65 ms |
| 280 | 2.520 | 947,36 ms | 1.030,20 ms |

Tower-Anteile bei Budget 280:

| Stufe | Zeit über 27 Layer |
|---|---:|
| Attention | 911,94 ms |
| Gate+Up | 11,62 ms |
| Down+Residual | 6,19 ms |
| QKV Projection | 4,59 ms |
| QKV Norm/RoPE | 3,35 ms |
| O Projection/Residual | 2,88 ms |
| GELU | 1,98 ms |
| Product Quant | 1,33 ms |
| Pool/Standardize | 1,14 ms |
| übrige Stufen | unter 2 ms zusammen |

Das ist die zentrale Priorisierung: V13 zuerst.

## 2.4 Publikation

Der vom Owner gewünschte eine Hugging-Face-Repository-Vertrag ist bereits umgesetzt:

```text
danmoreng/gemma-4-26B-A4B-it-GEM16
revision 31842e12882d09bab7109c0ad52a4ee2e945069c

root:
  NVFP4 Target

trellis35/:
  Trellis35 Target

assistant/:
  Fixed-D2 Assistant

vision/:
  FP8 Vision
```

Die Komponenten bleiben unabhängig gelockt und werden nicht still substituiert. Anonymous Download, Resume, Hashprüfung und Hardlink-basierte Runtime Views sind bereits belegt.

Die nächste Publikationsarbeit ist daher **kein neues Repository**, sondern eine neue immutable Revision desselben Repositories, in der Trellis35 als ein einziges Device Image veröffentlicht wird.

## 2.5 Native Studio

V17/V18 haben bereits:

- das explizite dritte Profil;
- Target/Vision/Assistant als getrennte Komponenten;
- `--vision-model`;
- 70/140/280-Auswahl;
- Bildpreview und geschätzte Bildtokens;
- Ein-Bild-Grenze;
- Live-Capability-Gate für D2;
- External-Server-Mismatch-Erkennung;
- Download/Verify/Remove mit gemeinsam genutzten Hub-Blobs.

UI-Integration ist deshalb kein Neubau mehr. Für den Produktionsabschluss geht es um Produktstatus, neue Component Locks, neue Trellis-v2-Dateiliste, sichere Defaults und Releasequalifikation.

---

# 3. Offene Blocker außerhalb von V12/V13

## 3.1 Der Snapshot ist nicht sauber

Das Archiv meldet:

```text
?? tests/test_image_vision.png
```

Die Datei befindet sich nicht im Review-Archiv, wird aber in bereits erzeugter Evidenz erwähnt.

Vor einem Freeze muss genau eine der Varianten gewählt werden:

1. Das Bild ist projekt-eigen und wird unter
   `tests/fixtures/gemma4_26b_vision/` eingecheckt, inklusive Generator,
   SHA-256 und Lizenz-/Provenienznotiz.
2. Die Referenz wird entfernt und alle betroffenen Gates werden mit bereits
   getrackten projekt-eigenen Fixtures wiederholt.

Kein produktiver Snapshot darf von einer nicht versionierten lokalen Datei abhängen.

---

## 3.2 Bildidentität verwendet nur FNV-1a-64

`SourceFingerprint` berechnet derzeit einen 64-Bit-FNV-Hash der kodierten Bildbytes. `ResidentMessageEquivalent` behandelt zwei Bilder mit gleichem nicht-null Fingerprint anschließend als identisch, ohne die vollständigen Daten erneut zu vergleichen.

Das ist für untrusted Media kein ausreichender Produktvertrag. Eine absichtlich erzeugte Kollision könnte einen anderen Bildinhalt als bereits gecachten Conversation-Inhalt erscheinen lassen.

Ersetzen durch:

```cpp
struct ImageSourceIdentity {
  std::array<std::uint8_t, 32> sha256;
  std::uint64_t encoded_bytes;
};
```

oder ein gleichwertiges SHA-256-Objekt.

Resident-Equivalence darf nur bei gleicher kryptographischer Identität abkürzen. Für in-process Inputs ohne Source Digest muss die vollständige relevante Struktur verglichen werden.

Diese Änderung betrifft 12B und 26B und braucht deshalb explizite 12B-Regressionsgates.

---

## 3.3 Trellis35 ist noch nicht textseitig produktqualifiziert

Die ursprüngliche WP8A-Evidenz bezeichnet sich ausdrücklich als:

```text
discovery_sanity_only
```

Abgeschlossen wurden:

- all-layer numerical differential;
- Sampled-Determinismus;
- 16K Retrieval.

Aufgeschoben wurden ursprünglich:

- 64K/98K Retrieval;
- GSM8K;
- AIME;
- GPQA;
- vollständige Task-/Quality-Qualifikation.

Seitdem wurden Kontext und D2 technisch erheblich weiter getestet. Die eigentliche Trellis35-Taskqualität wurde aber nicht durch eine vollständige Produktionssuite ersetzt.

Das Vision-Produkt verwendet genau dieses Trellis35-Target. Eine produktive Vision-Freigabe ohne produktive Text-Target-Freigabe wäre inkonsistent.

Mindestens erforderlich:

```text
GSM8K
AIME
GPQA Diamond – vorab fixiertes Subset oder Full
64K und mindestens ein >128K Retrieval-Gate
Instruction/Prose
Tool Calls
Stop-Sequenzen
Repetition Penalty
Greedy und Sampled
Ordinary und D2
```

Die bereits eingefrorenen NVFP4-Ergebnisse dienen als Kontrollarm. Grenzen dürfen nicht anhand der Ergebnisse nachjustiert werden.

---

## 3.4 Die Vision-Qualitätsevidenz ist noch sehr klein

V19 enthält:

```text
3 projekt-eigene synthetische Bilder
9 Fälle
18 Requests
```

und deckt Beschreibung, OCR, Chart, Dokument, Counting, Spatial, Farben und kleine Details ab. Alle objektiven Fakten wurden bei 70/140/280 bestanden.

Das ist eine gute Smoke- und Bounded-Quality-Evidenz. Für ein produktives Profil ist sie als alleinige Bildqualitätsbasis zu klein.

Empfohlener Produktionsumfang:

```text
mindestens 40–60 vorab eingefrorene Fälle

davon:
  reale Fotos / natural scenes
  kleine Objekte
  dichtes OCR
  Dokumentseiten
  Tabellen
  Charts
  räumliche Beziehungen
  Zählen
  Farben
  breite und hohe Geometrien
  schwierige Low-Contrast-Fälle
```

Für 12–20 repräsentative Fälle sollte zusätzlich der BF16-Vision-Tower als Embedding-Oracle laufen. Der Vision-Tower allein passt problemlos in den Speicher; das vollständige BF16-Textmodell ist dafür nicht erforderlich.

Produktclaim anschließend begrenzen auf die tatsächlich geprüfte Bildanalyse. Keine Behauptung allgemeiner Benchmark-Parität ohne zusätzliche Evidenz.

---

## 3.5 Windows-Live-CUDA fehlt

V19 meldet:

- Linux/Windows Host-, Studio- und Python-CI bestanden;
- Live Vision+D2 auf Linux SM120 bestanden;
- Windows-CI hatte keine SM120-GPU.

Der aktive Produktvertrag behandelt Windows und Linux als gleichwertige Releaseplattformen.

Für einen produktiven Freeze ist daher eine Entscheidung nötig:

```text
A. Live SM120 Vision+D2 auf Windows ausführen und bestehen
oder
B. Produktvertrag ausdrücklich auf die tatsächlich live qualifizierte
   Plattform begrenzen
```

Da das Ziel eine native Desktop-App ist, ist A die sinnvollere Lösung.

---

## 3.6 Der aktuelle D2-Maximalwert ist zu knapp als Standardwert

Für Target+Vision+Assistant bei 229.120 Tokens meldet V19:

```text
Headroom nach der geforderten 200-MiB-Reserve:
1.769.472 Byte
≈ 1,69 MiB
```

Das ist ein sauber bestandener Messpunkt, aber ein sehr knapper Produktdefault. Kleinste Änderungen an:

- CUDA Graphs;
- RoPE-Tabellen;
- Loader;
- Treiber;
- Debug-/Release-Linkage;
- Fragmentierung

können ihn verschieben.

Nach V12/V13 muss die Kapazität komplett neu gemessen werden.

Empfehlung:

```text
Produkt-Maximum:
  höchster final zweimal bestandener Wert

Studio-Default:
  ein konservativerer Wert mit zusätzlicher Reserve
```

Ein Default von 196.608 Tokens würde gegenüber 229.120 ungefähr 317,5 MiB KV-Platz freigeben. 212.992 würde ungefähr 157,5 MiB freigeben. Die konkrete Auswahl ist eine Owner-/Produktentscheidung nach den finalen Capacity-Runs.

229.120 kann als „Advanced maximum“ bestehen bleiben, sofern die finalen Windows- und Linux-Binaries denselben Reservevertrag wiederholen.

---

## 3.7 Profilidentität und Qualification-State sind noch experimentell geprägt

Aktuell:

```text
profile_id:
  ...vision-fp8
  oder
  ...vision-fp8-d2

qualification_state:
  experimental
  oder
  experimental_v14_accepted

experimental():
  true für Trellis35
```

Für ein Produktprofil sollte die Modellidentität nicht davon abhängen, ob D2 in diesem Prozess aktiviert ist.

Empfohlen:

```text
profile_id:
  gemma4-26b-a4b-trellis35-vision-fp8

decode_mode:
  ordinary | fixed-d2

qualification_state:
  production_qualified

vision_mtp_supported:
  true nur für den exakten Target+Vision+Assistant-Composite
```

Die Qualifikation muss an die exakten Component Hashes gebunden sein.

Zusätzlich müssen konfiguriertes Vision-Maximalbudget und zuletzt verwendetes Request-Budget getrennt werden:

```text
vision_max_soft_token_budget
last_vision_soft_token_budget
```

`selected_vision_soft_token_budget()` liefert momentan kein stabiles Profile-Attribut.

---

# 4. V13 im Detail – höchste Priorität

## 4.1 Warum V13 zuerst kommt

Bei Budget 280 entfallen ungefähr:

```text
911,94 ms von 947,36 ms
= 96,3 %
```

auf Vision Attention.

Selbst wenn alle übrigen Vision-Stufen kostenlos wären, bliebe der Tower fast eine Sekunde lang. V12 allein kann deshalb keine überzeugende Produktperformance erzeugen.

Der aktuelle Attention-Kernel:

- ordnet einen Warp einem Query/Head-Paar zu;
- liest für jedes Query alle K-Werte;
- berechnet in Pass 1 Max/Denominator;
- liest alle K-Werte in Pass 2 erneut;
- berechnet Q·K erneut;
- liest V;
- teilt K/V nicht zwischen mehreren Queries;
- verwendet keine Tensor Cores für QK oder PV.

Das ist korrektheitsorientiert, aber nicht die geeignete Produktionsarchitektur.

---

## 4.2 V13-A – exakter scalar tiled K/V-sharing Pfad

Der erste Kandidat soll die Mathematik nicht ändern.

### CTA-Mapping

Nicht mehr acht verschiedene Heads desselben Tokens in einem Block. Stattdessen:

```text
eine CTA:
  ein Head
  Q4 oder Q8 aufeinanderfolgende Query-Tokens
```

Kandidaten:

| Kandidat | Threads | Query-Warps | K/V-Tile |
|---|---:|---:|---:|
| A | 128 | 4 | 32 |
| B | 256 | 8 | 32 |
| C | 256 | 8 | 64 |

Erste Empfehlung:

```text
Q8 / K32
```

Danach Q8/K64 gegenmessen.

### Shared Layout

Head-Dimension 72 wird nur im Shared-Tile auf Stride 80 gepaddet:

```text
K[32][80] BF16
V[32][80] BF16
```

Das ist kein persistentes Weight- oder Activation-Format und ändert keine Modellsemantik.

K32:

```text
K + V Shared ≈ 10 KiB
```

K64:

```text
K + V Shared ≈ 20 KiB
```

Mit Double Buffering entsprechend ungefähr 20 beziehungsweise 40 KiB.

### Pass 1

Nur K laden:

1. Q je Warp in Registern halten.
2. K-Tile kooperativ nach Shared laden.
3. Jeder Query-Warp verarbeitet die Quellen weiterhin in exakt steigender Reihenfolge.
4. Inneres Channel- und `WarpSum`-Verhalten zunächst unverändert.
5. FP32 `running_max` und `denominator` unverändert.

### Pass 2

K und V laden:

1. Q·K erneut in derselben Reihenfolge berechnen.
2. Probability exakt wie heute zu BF16 runden.
3. BF16 Probability × BF16 V in FP32 akkumulieren.
4. Output zu BF16 runden.

### Double Buffer

Erst nachdem die synchrone Shared-Version exakt ist:

```text
cp.async K-buffer 0/1
cp.async K/V-buffer 0/1
```

Die Query-Warps konsumieren Tile n, während Tile n+1 geladen wird.

### Tail

- `valid_tokens` begrenzt Source-Zeilen.
- `tower_tokens` begrenzt Query-Zeilen.
- Keine ungeprüfte Reduktion auf `raw_patch_count` als Query-M.
- Head-Dim 72 bleibt logisch exakt; Shared 80-Tail wird mit Null gefüllt und nicht gespeichert.

### Erwartung und Gate

Keine Speedup-Zahl voraussetzen.

Engineering-Admission:

```text
keine Spills
kein Score-Slab
exakte Attention-Bits bevorzugt
mindestens 1,5× Attention-Kernel-Speedup bei 140 und 280
oder
mindestens 20 % Full-Tower-Speedup bei 140 und 280
keine >3-%-Regression bei 70
```

Wenn Shared-scalar bei gleicher Mathematik zu wenig gewinnt, folgt V13-B.

---

## 4.3 V13-B – BF16 Tensor-Core QK

Dieser Kandidat ist numerisch invasiver und benötigt eine ausdrückliche Kandidatenqualifikation.

### Geometrie

Möglicher erster Pfad:

```text
Q16 × K32
Head-K = 72
transient auf K80 oder K96 in Shared nullpad

BF16 × BF16
FP32 Accumulator
```

Die Score-Matrix wird tileweise berechnet; kein vollständiger quadratischer Score-Slab.

Der Softmax bleibt zunächst zweipassig:

```text
Pass 1:
  TC-QK Tile
  Scores in source order in Warp-/CTA-Reduktion einarbeiten

Pass 2:
  TC-QK erneut
  Probability zu BF16
  PV zunächst weiter scalar/shared
```

Warum PV zunächst nicht ebenfalls Tensor Core?

- weniger gleichzeitig veränderte Numerik;
- BF16-Probability-Grenze bleibt sichtbar;
- leichterer Differentialtest;
- Root Cause bei Qualitätsabweichung bleibt lokalisierbar.

### Numerische Gates vor Implementierung festlegen

Mindestens:

```text
Attention-Output pro Layer:
  relative L2
  cosine
  max absolute

Vision-Layer 0 / 13 / 26

finale 2816 Image Embeddings:
  relative L2
  cosine

Text:
  erste Logits
  Top-1
  deterministische Ausgabe

D2:
  Forced Proposal Differential
  Ordinary/D2 finaler Stream
  KV Commit
```

Falls der TC-Pfad den finalen Bildstream ändert, ist das nicht automatisch ein Fehler. Er darf aber nur übernommen werden, wenn die komplette Quality-/D2-Suite vorab definierte Grenzen besteht und die neue Trajektorie als neuer Produktstand eingefroren wird.

### Performance-Admission

Für diesen invasiveren Pfad sollte der Nutzen materiell sein:

```text
mindestens 35 % Full-Tower-Gewinn bei Budget 280
mindestens 25 % bei Budget 140
kein relevanter Qualitätsverlust
```

Diese Werte sind Engineering-Gates, keine bereits erwarteten Messergebnisse.

---

## 4.4 V13-C – erst später: Tensor-Core PV oder One-Pass

Nicht in den ersten V13-Commit aufnehmen.

Nur zulassen, wenn nach QK:

- PV nachweislich dominiert;
- Shared-scalar PV eine klare Compute-Grenze zeigt;
- der verbleibende TTFT-Abstand den numerischen Aufwand rechtfertigt.

Ein One-Pass-Online-Softmax verändert die heutige explizite BF16-Probability-Grenze und ist deshalb kein erster Produktionskandidat.

---

## 4.5 V13-Produktmessung

Für jeden finalen Kandidaten:

```text
Budgets:
  70 / 140 / 280

Geometrien:
  square / wide / tall

Messung:
  3 Warm-ups
  10 retained

Berichten:
  Image preprocessing
  Upload
  Tower
  Attention
  Text prefill
  TTFT
  post-first Ordinary
  post-first D2
  Peak VRAM
  Graph bytes
  Output hashes
```

Stretch-Ziele, keine Vorabbehauptung:

```text
Budget 140 Tower:
  deutlich unter 200 ms

Budget 280 Tower:
  ungefähr 500–600 ms oder besser
```

---

# 5. V12 im Detail

V12 soll nach oder parallel zu V13 als Reihe isolierter Kandidaten laufen.

## 5.1 V12-A – direktes 3×3-Pooling

Aktuell scannt jeder Output-Kanal jedes Soft-Tokens alle `tower_tokens`.

Nach der bereits akzeptierten kanonischen Row-Major-Positionierung kann jeder Output genau neun Quellpatches direkt adressieren.

Wichtig: Die gemessene Poolzeit liegt bei Budget 280 nur bei ungefähr 1,14 ms. Das ist ein guter Algorithmus-Cleanup, aber kein großer TTFT-Hebel.

Beibehalten nur, wenn:

```text
Pool-Output bitidentisch
finale Image Embeddings bitidentisch
Pool-Kernel stark schneller
Full Tower nicht langsamer
```

Eine Full-Path-Steigerung unterhalb des Messrauschens ist akzeptabel, wenn der Code einfacher, asymptotisch besser und exakt bleibt.

---

## 5.2 V12-B – vorab berechnete 2D-RoPE-Tabelle

Heute berechnet Q und K in allen 27 Layern wiederholt:

```text
powf
cosf
sinf
```

Die Positionen und 18 Frequenzen je X/Y-Hälfte ändern sich pro Bild nicht.

Ein einfacher bounded Table-Vertrag:

```text
[tower_token][36 frequencies][cos,sin] BF16
```

Maximale Größe:

```text
2520 × 36 × 2 × 2 Byte
= 362.880 Byte
```

Alternativ können nur eindeutige X/Y-Positionen gespeichert werden.

Genau dieselbe Formel und BF16-Rundung wie heute verwenden.

Gate:

```text
Q/K Norm+RoPE bitidentisch
Special-Function-Instruktionen nahezu entfernt
kein Runtime-Malloc
Full Tower gewinnt
```

---

## 5.3 V12-C – GELU plus Product Quantization

Heute:

```text
Gate BF16
Up BF16
GELU×Up -> BF16 Product in global memory
separate BF16→E4M3 Quantisierung
```

Ein geeigneter Fused-Kernel verwendet eine CTA pro Token:

```text
4304 BF16 Product-Werte
≈ 8,4 KiB je Token
```

Ablauf:

1. Gate/Up lesen.
2. GELU-tanh × Up berechnen.
3. explizit zu BF16 runden.
4. den BF16-Product in Shared oder einem reproduzierbaren Register/Shared-Tile halten.
5. `amax` über genau die gerundeten BF16-Werte.
6. Scale erzeugen.
7. E4M3 schreiben.

Kein ungerundeter Float darf direkt quantisiert werden.

Gate:

```text
E4M3-Bytes und Scales identisch
Down-Output identisch
GELU+Quant-Stufe mindestens 1,5× schneller
Full Tower messbar schneller oder neutral ohne zusätzlichen Speicher
```

---

## 5.4 V12-D – budgetabhängiger fixer Workspace

Heute werden immer ungefähr 114,43 MiB Vision-Workspace reserviert.

Planungswerte:

| Maximalbudget | Physische Reihen | Workspace |
|---:|---:|---:|
| 70 | 630 | ca. 34,61 MiB |
| 140 | 1.260 | ca. 61,22 MiB |
| 280 | 2.520 | 114,43 MiB |

Ein neues Startargument:

```text
--vision-max-soft-tokens 70|140|280
```

Der Request darf nur ein Budget `<= configured max` wählen.

Studio-Bezeichnungen:

```text
70  Fast
140 Balanced
280 Maximum detail
```

Eine Änderung des maximalen Budgets erfordert einen kontrollierten Serverneustart. Kein per-request Reallocation.

Danach Capacity für jede Profileinstellung neu messen.

---

## 5.5 V12-E – Input-Staging

Der aktuelle Upload ist im Vergleich zum Tower sehr klein. Deshalb nur bei einem frischen Profil umsetzen.

Falls relevant:

- zwei gepinnte Hostslots;
- CUDA Events statt globalem Stream-Sync;
- bounded;
- keine unkontrollierte Pipeline;
- Cancellation und Buffer-Reuse testen.

---

## 5.6 V12-F – CUDA Graphs

Erst nach Stabilisierung von V12/V13.

Höchstens ein begrenzter Satz:

```text
70
140
280
```

oder ein kleiner, dokumentierter Satz physischer Grids.

Graph-private Bytes müssen in Capacity und Context Admission eingehen. Keine unbeschränkte Shape-Cache-Map.

---

# 6. Trellis35 als ein einziges GPU-fertiges Device Image

## 6.1 Ist das ohne neue Quantisierung möglich?

**Ja.**

Der heutige Runtime-Arena-Vertrag ist bereits:

```text
Offset 0:
  non-routed.gem16
  1.850.270.720 Byte

danach:
  30 × layer-N.trellis35.bin
  je 345.147.392 Byte

Gesamt:
  12.204.692.480 Byte
```

Der Loader allokiert schon heute genau ein CUDA-Arena und kopiert die 31 Payload-Dateien genau in diese Reihenfolge.

Damit kann `trellis35/model.gem16` als reine Byte-Konkatenation gebaut werden:

```text
model.gem16[0 : 1.850.270.720]
  = non-routed.gem16

model.gem16[1.850.270.720 + N×345.147.392 : ...]
  = layer-N.trellis35.bin
```

Die Quantisierung wird nicht erneut ausgeführt und kein Wert verändert.

---

## 6.2 Format v2

Empfohlener Name:

```text
gem16-sm120-trellis35-device-image-v2
```

Produktdateien:

```text
trellis35/model.gem16
trellis35/gem16_model.json
trellis35/gem16_compilation.json
trellis35/gem16.lock.json
```

Gemeinsame Tokenizer-/Config-Dateien können in der Studio-Runtime-View als Hardlinks aus dem Root des konsolidierten Repositories erscheinen.

Das Binärfile enthält **nur** die exakt GPU-fertigen Arena-Bytes. Kein Header vor Offset 0. Metadaten bleiben separat, damit:

```text
cudaMalloc(file_size)
direkter Upload auf arena[0]
alle Pointer = arena + manifest_offset
```

möglich bleibt.

---

## 6.3 v2-Packager

Neues Tool, beispielsweise:

```text
tools/build_gemma4_26b_trellis35_device_image.py
```

Ablauf:

1. vollständigen v1-Lock validieren;
2. alle 31 Payload-Hashes validieren;
3. Layerreihenfolge und Größen validieren;
4. `model.gem16.partial` exklusiv anlegen;
5. auf exakt 12.204.692.480 Byte preallocieren;
6. non-routed und Layer streaming an feste Offsets schreiben;
7. laufenden Gesamt-SHA-256 berechnen;
8. `fsync`;
9. atomisch umbenennen;
10. v2-Manifest/Lock schreiben.

Zusätzlich:

```text
v1_stream_concatenation_sha256
==
v2_model_sha256
```

durch eine unabhängige zweite Verifikation bestätigen.

---

## 6.4 v2-Loader

Der aktuelle Trellis-Loader synchronisiert nach jedem 64-MiB-Chunk und öffnet 31 Payload-Dateien.

Der v2-Loader soll die bereits vorhandene NVFP4-Device-Image-Infrastruktur verwenden:

```text
4 gepinnte 64-MiB-Buffer
CUDA Events
asynchron rotierende Upload-Slots
optional cuFile
structural oder full SHA-256
```

Weiterhin:

```text
genau ein cudaMalloc
kein Repack
kein zweites Weight-Format
```

Der `LoadPath` soll explizit berichten:

```text
trellis35_device_image_pinned_async_structural
trellis35_device_image_pinned_async_sha256
trellis35_device_image_cufile_...
```

---

## 6.5 Migration

Für eine Übergangsphase:

```text
v2:
  Produktstandard

v1:
  expliziter Legacy-/Developer-Loader
```

Kein stiller Fallback von einem beschädigten v2 auf v1.

A/B-Gates:

```text
identische Arena-Bytes
identische Pointer-Offsets
identische Host-Scalars
identische Ordinary/D2/Vision-Ausgabe
identische Performance
weniger Runtime-Dateien
gemessene kalte und warme Ladezeit
```

Nach einer Releasegeneration kann v1 aus dem Produktkatalog entfernt werden.

---

## 6.6 Hugging-Face-Revision

Kein neues Repository anlegen.

Neue immutable Revision desselben Repositories:

```text
danmoreng/gemma-4-26B-A4B-it-GEM16
```

In der neuen Revision:

```text
root:
  NVFP4 unverändert

trellis35/:
  ein model.gem16
  kleine Metadata/Locks
  keine layer-XX-Unterverzeichnisse

assistant/:
  unverändert oder separat später Device-Image-konvertiert

vision/:
  vision.gem16 + strikte Runtime-Metadaten
```

Die alten Git-LFS-Objekte bleiben historisch im Repository, werden aber von Clients der neuen Revision nicht mehr heruntergeladen.

Wichtig: Nicht alle vier Komponenten in eine einzige 27-GB-Datei packen. Ein Repository mit je einem großen Payload pro Komponente ist besser:

- unabhängige Downloads;
- unabhängige Locks;
- User lädt nur benötigte Profile;
- Shared Assistant/Vision bleiben deduplizierbar;
- Update einer Komponente erzwingt nicht den Download aller anderen.

---

# 7. Produktionsqualifikation statt experimentellem Freeze

## 7.1 Owner-/Dokumentänderung

`docs/ACTIVE_DECISIONS.md`, `PRODUCT_CONTRACT.md`, Studio Catalog und Capability-Dokumente müssen das dritte Profil als gleichwertiges Produktprofil aufnehmen.

Vorgeschlagene Produktbezeichnung:

```text
Gemma 4 26B A4B – Compact Vision
Trellis35 Text + FP8 Vision + Fixed-D2
```

Nicht mehr:

```text
Experimental Trellis35 ...
```

Die Umbenennung geschieht erst im finalen Kandidatencommit, nachdem alle Gates bestanden sind.

## 7.2 Produktvertrag

Der produktive Scope:

```text
SM120
16-GB-Klasse gemäß gemessener Capacity
Batch 1
ein residentes Execution-Slot
ein Bild je Conversation/Request-Vertrag
Bildbudgets 70/140/280
Textausgabe
Ordinary
Fixed-D2
keine Audio-/Video-Unterstützung in diesem Profil
```

## 7.3 Requalifikation nach Performance- und Formatänderungen

Zwingend neu:

- Vision Tower Timings;
- TTFT;
- Ordinary/D2;
- Capacity für alle Max-Budget-Profile;
- Final Target+Vision und Target+Vision+Assistant;
- no recurring allocations;
- Cancellation;
- Server/Studio;
- Download/Resume;
- v2-Monolith Loader;
- Trellis text quality;
- erweiterte Vision quality;
- Windows und Linux.

---

# 8. Empfohlene Reihenfolge für Codex

Die Performance- und Formatpfade können in getrennten Worktrees vorbereitet werden.

```text
PRD00
  sauberer Snapshot
  Owner/Productvertrag
  SHA-256 Bildidentität
  stabile Profilsemantik

PERF13
  tiled scalar K/V-sharing Attention
  gegebenenfalls BF16 Tensor-Core QK

PERF12
  direct Pool
  RoPE Table
  GELU+Quant
  Budget Workspace
  bounded Graphs

FMT01
  Trellis35 model.gem16 v2

PUB01
  neue immutable Revision im existierenden HF-Repo
  neue Locks/Kataloge

QUAL01
  Trellis Text Production Quality
  erweiterte Vision Quality
  final Capacity
  Linux + Windows Live SM120

APP01
  Studio Labels/Defaults/Component v2
  final Health/Profile Contract

REL01
  Clean-Machine Packaging
  3W10 Performance
  Failure Injection
  Release Candidate

P20
  Production Qualification

P21
  Production Release Freeze
```

**Empfohlener erster Codex-Auftrag:** `PRD00`, danach `PERF13-A`.

V13 ist der erste Performanceauftrag, weil es der einzige Schritt ist, der die gemessene Tower-Laufzeit materiell verändern kann. Die Monolith-Arbeit kann parallel in einem separaten Worktree laufen, weil sie keine Kernelmathematik ändert.

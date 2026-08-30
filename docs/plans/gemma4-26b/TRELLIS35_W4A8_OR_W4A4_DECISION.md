# Trellis35 nach WP15: W4A8- oder W4A4-Entscheidung

**Datum:** 2026-08-30  
**Branch:** `codex/gemma4-26b-trellis35-perf2`  
**Evidenzstand:** WP12 bis WP15, Implementierungsstand `eca4472`  
**Status:** Architekturentscheidung; kein W4A4-Implementierungsauftrag

## Entscheidung

**Option A: W4A8 weiter optimieren.**

W4A4 wird noch nicht zugelassen. Die geforderten Admission Conditions sind
nicht erfüllt: Der optimierte W4A8-Pfad hat weder eine nachgewiesene
decoderfreie Compute-Decke noch eine ausreichend gesättigte Tensor-Pipe. Der
bounded transiente E4M3-Slab aus WP11 schlägt den Inline-Pfad nicht. Gleichzeitig
zeigen WP12 bis WP15, dass weitere W4A8-Änderungen ohne zweite persistente
Gewichtsrepräsentation noch große, reale Full-Path-Gewinne liefern.

Diese Entscheidung gilt nur für den experimentellen Trellis35-Branch. Sie
ändert weder den qualifizierten NVFP4-Pfad noch dessen Produktstatus.

## Gemessene Entscheidungsbasis

### Prefill

| Stand | 512-Token-Prefill | Veränderung |
|---|---:|---:|
| WP11 vor echtem M32 | 1.019,36 tok/s | Ausgangspunkt |
| WP12 echter M32 | 2.079,73 tok/s | 2,04× gegen WP11 |
| WP13 Warp-FWHT | 4.331,23 tok/s | 2,08× gegen WP12 |
| WP14 fused N128 | 4.755,64 tok/s | 1,098× gegen WP13 |
| NVFP4-Referenz | ca. 6.966 tok/s | Trellis35 erreicht ca. 68,3 % |

Der Trellis35-Prefill wurde in drei Paketen um ungefähr den Faktor 4,67 gegen
die frische WP11-Baseline beschleunigt. Das ist keine stagnierende Architektur.
Der verbleibende Abstand zu NVFP4 beträgt allerdings noch rund 31,7 %.

Der WP12-M32-Kernel meldet 58,75 % Issue Slots Busy, 59,14 % erreichte
Occupancy, 56 Register pro Thread, keine Spills und LSU als größte
Instruktionspipeline. Die FMA-Heavy-Pipeline lag nur bei 44,4 %. Der WP14-N128-
Kernel konsolidiert 1.980 Projection-/Inverse-Launches auf 60 und reduziert
deren GPU-Zeit von 74,861 auf 64,611 ms. Projection plus Inverse beanspruchen
damit weiterhin ungefähr 60 % der 107,662 ms Full-Prefill-Zeit, ohne dass eine
W4A8-Tensor-Compute-Sättigung belegt ist.

### Ordinary und Fixed-D2

Der akzeptierte Warp-H128-Pfad erreicht beim kanonischen 16k/64-Panel:

| Grenze | Direct-H128 | Warp-H128/M16 | Veränderung |
|---|---:|---:|---:|
| Ordinary post-first | 118,03 tok/s | 123,04 tok/s | +4,25 % |
| reiner T3-Verifier | 175,24 tok/s | 184,84 tok/s | +5,48 % |
| kompletter Fixed-D2-Request | 116,44 tok/s | 109,44 tok/s | −6,02 % |

Der registerbasierte echte M16-T3-Kernel reduziert die Projektionszeit im
Nsight-Systems-Graph von 4,921 auf 3,641 ms, die NCU-Instruktionen von 38,73 auf
32,52 Millionen und den Registerbedarf von 70 auf 39 pro Thread. Er ist
bitidentisch zum IndependentRows-Rollback.

Der vollständige D2-Request ist trotzdem langsamer, weil Warp-H128 die
deterministische Draft-Trajektorie von 38/12 accepted/rejected in 25 Gruppen auf
37/15 in 26 Gruppen verschiebt. Alle finalen 64 Target-Tokens bleiben in jedem
Retained Run identisch. Nach Owner-Entscheidung ist diese Trajektorienänderung
zulässig, ihr End-to-End-Preis bleibt aber eine offene W4A8-Aufgabe.

## Bewertung der vier Optionen

### A. W4A8 weiter optimieren — angenommen

Die Tensor-Pipe ist nicht als harte Grenze belegt, und die letzten Pakete
liefern weiterhin messbare Full-Path-Gewinne. W4A8 behält außerdem den bereits
gemessenen bounded Trellis35-Qualitätsstand ohne eine zweite
Gewichtsquantisierung.

### B. Prefill-only transienten E4M3-Backend hinzufügen — abgelehnt

Der WP11-Probe ist bitgenau, zeigt aber beim vorhandenen decoderfreien Backend
ab M=16 einen Latenzsprung. Für Gate+Up müssten je Slab 360.704 Byte, für Down
98.560 Byte geschrieben und wieder gelesen werden. Der Probe schlägt den
optimierten Inline-M32-Pfad nicht und rechtfertigt weder zusätzliche
Workspace-Komplexität noch eine Runtime-Cache-Lebensdauer.

Ein neuer Slab-Versuch ist nur zulässig, wenn ein isolierter decoderfreier
W4A8-Kernel den aktuellen Inline-N128-Kernel bei derselben M/N/K-Geometrie
zuzüglich Decode-, Schreib- und Lesekosten schlägt.

### C. Trellis zu nativem W4A4 für großes M erforschen — noch nicht zugelassen

Die Admission Conditions fehlen:

- W4A8-Tensor-Compute ist nicht als materielles Limit nachgewiesen.
- Der decoderfreie/transiente W4A8-Pfad zeigt keine vorteilhafte Compute-Decke.
- Die gemessene Tensor-/FMA-Auslastung ist nicht hoch genug, um einen
  FP4-Durchsatzgewinn belastbar vorherzusagen.
- Es gibt noch keinen bezifferten Qualitäts- und Byte-Vertrag für E2M1 plus
  E4M3-K/16-Skalen.

W4A4 darf daher weder implementiert noch als Runtime-Rundung der Trelliswerte
prototypisiert werden. Eine spätere Zulassung benötigt vor Codeänderungen:

1. E2M1-Rekonstruktions- und Rundungsvertrag;
2. E4M3-Skalenlayout pro K/16 einschließlich Alignment;
3. Payload-/Sidecar-Byte-Delta bei weiterhin genau einer persistenten
   Expert-Repräsentation;
4. getrennten Fehler der Trellis-Quantisierung und der zweiten E2M1-
   Quantisierung;
5. gemessene Runtime-Decoderkosten;
6. eine neue ausdrückliche Owner-Anweisung.

### D. Trellis-Performancearbeit stoppen — abgelehnt

Der Branch hat sein 3,5-bpw- und Speicherziel erreicht und ist funktional. Die
Prefill-Performance liegt noch klar unter NVFP4, aber WP12 bis WP15 zeigen
weiterhin verwertbare W4A8-Gewinne. Stoppen wäre erst sachgerecht, wenn zwei
aufeinanderfolgende, profilerbegründete W4A8-Kandidaten keinen Full-Path-Gewinn
liefern oder der verbleibende Abstand nachweislich Tensor-Compute-limitiert ist.

## Nächste W4A8-Welle

Die nächste Arbeit ist ein neuer, vom Owner freizugebender Performanceplan;
WP16 selbst implementiert nichts. Empfohlene Reihenfolge:

1. Den fused-N128-Prefill-Kernel mit Source-/SASS-Metriken in Trellis-Decode,
   A-Delivery, MMA und H128/SVH-Epilog zerlegen. Ohne dominanten Decoderanteil
   keine Producer/Consumer-Warp-Spezialisierung beginnen.
2. CTA-Geometrie und N128-Epilog gegen mindestens zwei isolierte Rollback-
   Varianten messen. Das Gate bleibt Full-Prefill, nicht nur Layer 0.
3. Für Ordinary die hohe Initial-Selection-Zeit getrennt von post-first
   untersuchen; Timinggrenzen und CUDA-Graph-Warmup dürfen nicht verändert
   werden.
4. Für D2 die Kosten einer zusätzlichen Verifier-Gruppe getrennt von der
   schnelleren T3-Ausführung ausweisen. Keine Akzeptanzverbesserung durch
   geänderte Sampling-, Token-, KV- oder Attention-Semantik.
5. Erst danach erneut prüfen, ob die W4A4-Admission Conditions erfüllt sind.

Jeder neue Kandidat behält Direct-H128, IndependentRows und Loop-N128 als
explizite interne Rollbacks, erzeugt keine persistente Gewichtskopie und
verändert keine BF16-Roundpoints oder Aktivierungsskalen.

## Evidenz

- `artifacts/trellis35/wp11-transient-e4m3-probe.json`
- `artifacts/trellis35/wp12-m32-ncu.json`
- `artifacts/trellis35/wp12-prefill-ab.json`
- `artifacts/trellis35/wp13-fwht.json`
- `artifacts/trellis35/wp14-output-consolidation.json`
- `artifacts/trellis35/wp15-ordinary-panel.json`
- `artifacts/trellis35/wp15-d2-panel.json`
- `artifacts/trellis35/wp15-d2-nvtx-breakdown.json`
- `artifacts/trellis35/wp15-t3-ncu.json`

## Schlussurteil

Trellis35 bleibt vorerst ein **W4A8-System**. Der nächste sinnvolle Schritt ist
eine weitere profilergetriebene W4A8-Welle, nicht ein transienter Slab und nicht
W4A4. W4A4 bleibt eine spätere, neu zu autorisierende Forschungslinie, sobald
eine echte W4A8-Compute-Decke belegt ist.

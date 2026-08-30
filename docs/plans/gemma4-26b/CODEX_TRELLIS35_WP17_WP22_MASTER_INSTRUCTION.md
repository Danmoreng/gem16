# Codex Master Instruction – Gem16 Trellis35 WP17–WP22

## Ausgangspunkt

Arbeite auf dem Trellis35-Performancezweig ab:

```text
codex/gemma4-26b-trellis35-perf2
4d29d6e50bf8c8fcf3e06f58d3b75e29d8ce5f47
```

Lege nur nach Owner-Freigabe einen neuen Branch an, beispielsweise:

```text
codex/gemma4-26b-trellis35-perf3
```

Lies vor jedem Paket:

```text
AGENTS.md
docs/ACTIVE_DECISIONS.md
docs/plans/gemma4-26b/TRELLIS35_W4A8_OR_W4A4_DECISION.md
GEM16_TRELLIS35_POST_WP16_PERFORMANCE_REVIEW.md
```

Die Owner-Anweisung erlaubt weitere Performancearbeit ausschließlich am experimentellen Trellis35-Profil. Der qualifizierte NVFP4- und der geschützte 12B-Pfad bleiben unverändert.

## Eingefrorene Vergleichsgrenzen

Primärer Benchmark:

```text
artifacts/raw/trellis35/wp16/full-wikipedia-16k-sampled-1w3.json
```

Trellis35-Median:

```text
Prefill:          4564.500312 tok/s
Ordinary:          125.719586 tok/s
Fixed-D2:          168.938672 tok/s
D2/Ordinary:         1.343774x
Output Tokens:            1229
D2 accepted/proposed: 690/1076
D2 groups:                 538
Fallback:                    0
```

NVFP4-Referenz:

```text
Prefill:          6965.631 tok/s
Ordinary:          148.439 tok/s
Fixed-D2:          203.831 tok/s
```

Stretch-Ziele dieser Welle:

```text
Prefill >= 5000 tok/s
Fixed-D2 >= 180 tok/s
```

Sie sind Engineering-Ziele, keine Erlaubnis, Numerik, Prompt, Sampling, Cache, Timing oder Outputlänge zu ändern.

## Unverhandelbare Regeln

- Trellis35 und NVFP4 bleiben getrennte Modellprofile.
- Genau eine persistente routed-expert-Repräsentation.
- Kein Runtime-JIT.
- Keine Runtime-Quantisierung oder Model-Repack.
- Kein CPU-Offload/Expert-Streaming.
- Keine Token-Loop-Allokation.
- Keine persistente E4M3- oder NVFP4-Zweitkopie.
- Keine stillen Precision-/Kernel-/Context-Fallbacks.
- Kein WP8B-Langlauf in dieser Welle.
- Keine Vision-Arbeit.
- Keine Änderung physischer BF16-Roundpoints.
- Jeder Kandidat muss einen internen Rollback haben.
- Microkernel-Gewinne werden nur übernommen, wenn auch das vollständige 16K-Panel nicht verliert.

---

# WP17 – Reales 1024er-Routing und M64N128

## Ziel

Bei produktiven 1.024-Token-Chunks den Trellis-B-Decode über bis zu 64 Rows amortisieren.

## A. Instrumentierung vor Implementierung

Füge einen nur diagnostisch aktivierbaren Telemetriepfad hinzu. Er muss für alle 30 Layer und 16 Chunks berichten:

```text
tokens
assignment_count
rows je Expert
active experts
M32 schedule count = sum ceil(rows/32)
M64 schedule count = sum ceil(rows/64)
M64 full tiles
M64 33–63 tails
M32 <=32 tails
K3/K4 counts
```

Kein Host-Sync im normalen Pfad. Diagnose darf explizit synchronisieren und ist performance-ineligible.

Profilier den aktuellen `MmaW4A8ProjectionGroupedPrefillM32N128Kernel` bei:

```text
Layer 0 Gate+Up, 1024 reale Tokens
Layer 0 Down,    1024 reale Tokens
ein mittlerer Layer
ein später Layer
```

NCU mindestens:

```text
duration
registers/thread
static/driver shared
stack/local/spills
achieved/theoretical occupancy
issue slots
Tensor/ALU/LSU instruction mix
long scoreboard
barrier stalls
L1/L2/DRAM
waves
```

## B. M64-Schedule

Berühre die NVFP4-Schedule nicht.

Ergänze einen Trellis-spezifischen Schedule-Modus. Bevorzugtes v1-Design:

```text
Queue 1: M64 tiles für 33–64 Rows
Queue 2: M32 tiles für 1–32 Rows
```

Für mehr als 64 Rows werden vollständige M64-Tiles und ein klassifizierter Tail erzeugt.

Optionaler A/B-Kandidat:

```text
K3 und K4 in getrennte Queues
```

Dadurch können `Kernel<3>` und `Kernel<4>` separat kompiliert werden. Behalte diese Variante nur, wenn sie Register oder Full-Path-Zeit verbessert.

Alle Offsets und Counts checked. `assignment_count <= 65535` bleibt bindend.

## C. M64N128-Kernel

Neue Funktion:

```text
MmaW4A8ProjectionGroupedPrefillM64N128Kernel
```

Erste Geometrie:

```text
512 Threads
16 Warps
64 Rows
4 × M16
N128
K32
```

Anforderungen:

1. vier `Fp8Accumulator`-Fragmente je Warp;
2. `valid_rows` und uniformer `valid_m16_tiles`;
3. keine MMA für vollständig ungültige obere M16-Tiles;
4. A weiterhin `cp.async` doppelt puffern;
5. ein B-Decode je K32/N8 und M64-Tile;
6. aktuelle inverse H128/SVH-/BF16-Outputsemantik;
7. keine komplette Weight-Rekonstruktion;
8. M32 bleibt expliziter Rollback.

Geschätztes statisches Shared vor B-Staging:

```text
~37,1 KiB
```

Kritisches Ressourcengate:

```text
keine Spills
idealerweise <=64 Register/Thread
mindestens zwei aktive CTAs/SM oder gemessene gleichwertige Auslastung
```

Überschreitet ptxas 64 Register und fällt die Occupancy ab:

- Rate-Pfade separat kompilieren;
- Live-Ranges verkürzen;
- Scale/SVH nicht unnötig in Registern halten;
- Kandidat nicht mit Local-Memory-Spills erzwingen.

## D. Tests

Erweitere:

```text
tests/cuda/trellis35_prefill_test.cu
```

Matrix:

```text
Rows: 1,2,3,15,16,17,31,32,33,47,48,63,64,65,95,96,127,128
Rates: K3, K4, mixed
Patterns: uniform, real fixture, one-hot, long-tail
Families: Gate+Up, Down
Output: Expert und reduzierte Outputs
```

## E. Acceptance

- Operator-/Full-Layer-Korrektheit;
- CUDA Graph Replay;
- Memcheck/Initcheck/Racecheck;
- keine Allocation-/Arenaänderung;
- keine NVFP4-Source-/Objectänderung;
- Full Wikipedia 16K, 1W3, identische Konfiguration;
- Retain nur bei mindestens +5 % Prefill;
- Stretch: >=5000 tok/s;
- kein Ordinary-/D2-Regress durch Dispatchänderung.

Evidence:

```text
artifacts/trellis35/wp17-routing-1024.json
artifacts/trellis35/wp17-m64-operator.json
artifacts/trellis35/wp17-m64-ncu.json
artifacts/trellis35/wp17-full-16k-ab.json
```

---

# WP18 – Fused-N128 LSU-/Synchronisationswelle

Arbeite auf dem Gewinner aus WP17.

Jeder Unterkandidat ist ein separater Commit/A-B und vollständig rückrollbar.

## A. Scale-Stage

Lade 32 beziehungsweise 64 `activation_scales` einmal je CTA nach Shared.

## B. SVH-Stage

Vor Änderung SASS prüfen. Falls SVH nicht bereits vollständig in Warp-uniforme Registerloads gehoben wird:

```text
128 FP16 SVH-Werte einmal je CTA nach Shared
```

Kein 16-faches globales Wiederladen über die Warps.

## C. Pair-shared Trellis-Payload

Benachbarte N8-Warps teilen denselben N16-Tile.

Pro K32 eindeutig:

```text
K3: 384 U32
K4: 512 U32
```

Staging:

```text
global -> Shared einmal
Warp-Paar -> Shared lesen
```

Verwende die ohnehin vorhandene CTA-Synchronisation. Prüfe Bankkonflikte und Shared-Multicast.

## D. Gepackte Konvertierung

SASS des aktuellen `DecodeLanePayloadE4M3x4` prüfen.

Nur wenn vier skalare FP16→E4M3-Konvertierungen emittiert werden, einen bitidentischen gepackten x2/x4-CVT-Pfad prototypisieren.

## E. Vektorisierte Projection-/Output-Zugriffe

- MMA-Paare als `float2` in `projection` speichern;
- H128-Eingang als ausgerichtetes `float4` laden;
- vier BF16-Werte gepackt speichern, sofern exakt ausgerichtet und bitidentisch.

## F. K64-Pipeline

Erst nach A–E:

```text
64 K-Werte je Stage
2 K32-MMAs je Barrier
Gate+Up: 88 -> 44 Barrier-Epochen
Down:    24 -> 12
```

Shared Memory einschließlich:

```text
M64 A double buffer
B double buffer
projection
rows/scales/SVH
driver shared
```

exakt reporten. Zwei CTAs/SM nicht voraussetzen, sondern mit ptxas/NCU belegen.

## Acceptance je Unterkandidat

- exakte Outputs;
- keine Spills;
- NCU-Verbesserung des adressierten Counters;
- 1024er Kernel gewinnt;
- Full 16K gewinnt oder ist innerhalb Rauschen neutral bei klar notwendigem Prerequisite-Charakter;
- Verlierer zurückrollen.

Kumulatives Stretch-Ziel:

```text
>=5000 Prefill tok/s
```

Evidence pro Kandidat plus:

```text
artifacts/trellis35/wp18-full-16k-final.json
```

---

# WP19 – BF16 Gated-GELU direkt in Down-E4M3

## Ziel

Eliminiere:

```text
GatedGeluBf16Kernel
expert_product_bf16 write
expert_product_bf16 read
separaten Down-Transform-Launch
```

## Kernel

```text
GatedGeluDownTransformQuantizeBf16WarpKernel
```

Semantik muss exakt sein:

```cpp
float gate = Bf16(gate_bits);
float up = Bf16(up_bits);
uint16_t product_bits = Bf16Bits(Gelu(gate) * up);
float product = Bf16(product_bits);
float transformed_source = product * F16(suh);
```

Danach unverändert:

```text
H128Warp
amax
scale
E4M3
```

Padding 704→768 bleibt unverändert.

Der bestehende Produktpuffer bleibt nur für den Rollback-Pfad.

## Gates

- E4M3-Ausgabe und Scale byte-identisch;
- vollständiger Output identisch;
- kein Workspace-/Arena-Wachstum;
- Full 16K mindestens +1 % oder klarer Stage-Gewinn ohne Full-Path-Verlust;
- Sanitizer/Graph Replay.

Evidence:

```text
artifacts/trellis35/wp19-gelu-down-fusion.json
```

---

# WP20 – T3/Ordinary Recovery

## Ziel

Fixed-D2 >=175, Stretch >=180 tok/s; Ordinary nicht verschlechtern.

## A. Unique-Expert-Tasklist

Ein graphfähiger Builder erzeugt maximal 24 Tasks:

```cpp
struct Trellis35T3Task {
  uint8_t expert;
  uint8_t count;
  uint8_t assignment0;
  uint8_t assignment1;
  uint8_t assignment2;
  uint8_t rate;
};
```

Eine kompaktere 32-/64-Bit-Repräsentation ist erlaubt.

Anforderungen:

- einmal je Layer;
- Gate+Up und Down nutzen dieselbe Liste;
- kein wiederholter `prior`-Scan;
- kein wiederholter 3×8-Scan;
- kein dynamisch indiziertes thread-lokales `assignments[3]`;
- maximal 24 feste Slots, ungültige Tasks explizit markiert;
- keine Host-Synchronisation.

## B. Vierstufiges T3-Prefetch

Portiere das Muster aus:

```text
AccumulateSelectedProjectionM1
```

in den T3-M16-Pfad.

Report:

```text
Registerdelta
Long-Scoreboard
L1/L2
Duration
```

## C. Persistenter grid-stride Consumer

Nur nach NCU von A+B.

Flatten:

```text
work_item = task × output_block
```

Feste CUDA-Graph-kompatible Gridgrößen testen:

```text
4, 6, 8 CTAs/SM
```

Jeder CTA verarbeitet mehrere Items grid-stride.

Kein Atomic-Queue-Design, solange die Kosten als ausreichend regulär gemessen sind.

## D. Pair-shared Payload / N-Geometrie

Wenn Load-Duplikation bleibt:

- N16 je Warp oder Shared-Payload je Warp-Paar A/B testen;
- N64/N128 nur mit Ressourcen- und Full-Graph-Gate.

## E. Gated-GELU→Down-Transform

Die WP19-Fusion auf T3 und anschließend M1 übertragen, sofern die exakte kleine-M-Semantik reproduziert wird.

## Gates

- identische finalen 1.229 Tokens;
- identischer Output-Hash;
- bei rein arithmetischen Änderungen identische 690/386 Draft-Trajektorie;
- kein Fallback;
- keine Token-Loop-Allokation;
- T3-Projektion mindestens 15 % schneller;
- Full D2 >=175 tok/s;
- Stretch >=180 tok/s;
- Ordinary nicht unter WP16.

Evidence:

```text
artifacts/trellis35/wp20-tasklist.json
artifacts/trellis35/wp20-t3-prefetch-ncu.json
artifacts/trellis35/wp20-persistent-grid.json
artifacts/trellis35/wp20-full-wikipedia-16k-1w3.json
```

---

# WP21 – Trellis-spezifischer 2048er Prefill-Chunk

Nicht beginnen, solange WP17–WP20 laufen.

## Preflight

Exakt berechnen und direkt messen:

- zusätzliche Hidden-/Attention-/Expert-/Transform-Workspaces;
- Graph-private Bytes;
- freie Device-Bytes;
- Target-only und Fixed-D2 Context-Grenzen;
- erforderliche Reserve.

Vergleich:

```text
1024
1536
2048 Tokens/Chunk
```

Vorteil bei 2048:

```text
8 statt 16 Chunks
16.384 Assignments
arithmetisch 128 Rows/Expert
```

Möglicher späterer M128-Kernel ist ein eigener Kandidat und folgt erst nach realer Route-Telemetrie.

## Gates

- 1024 bleibt Rollback;
- keine versteckte Context- oder Reserveänderung;
- numerische Chunk-Boundary-Änderung explizit messen;
- Full 16K gewinnt;
- Peak VRAM und Max Context publizieren;
- keine OOM-nahe Promotion.

Evidence:

```text
artifacts/trellis35/wp21-prefill-chunk-sweep.json
```

---

# WP22 – W4A8/W4A4 Entscheidung

Nur ein Entscheidungsdokument, keine automatische W4A4-Implementierung.

W4A4 zulassen nur, wenn nach WP17–WP21:

- Tensor-Pipe materiell limitiert;
- Decoder/LSU/Barrier/Schedule nicht mehr dominieren;
- ein optimaler/decoderfreier W4A8-Kontrollarm eine Compute-Decke zeigt;
- Byte- und Qualitätsvertrag für E2M1 + E4M3 K/16 vorliegt.

Schreibe:

```text
docs/plans/gemma4-26b/TRELLIS35_POST_WP21_W4A8_W4A4_DECISION.md
```

---

# Deferred QX1 – nicht in dieser Welle ausführen

Siehe:

```text
TRELLIS35_DEFERRED_ADAPTIVE_HADAMARD_QX1.md
```

Kein Overnight-Quantizer-Job, solange die aktuelle Runtime-Performancewelle nicht abgeschlossen ist.

---

# Berichtspflicht

Nach jedem WP:

```text
exakte Revision
Dateien
Kommandos
GPU/Driver/CUDA
Kernel/SASS-Identität
Register/Shared/Stack/Local/Spills
NCU/NSYS
Operatornumerik
Full-16K-Messung
Peak VRAM/Allocation Delta
akzeptiert oder vollständig zurückgerollt
Limitierungen
```

Alte Evidenz nie überschreiben.

# Gem16 Trellis35 nach WP16 – Performance-Review und nächste Architekturwelle

**Review-Basis:** `codex/gemma4-26b-trellis35-perf2@4d29d6e50bf8c8fcf3e06f58d3b75e29d8ce5f47`
**Primäre Messung:** `artifacts/raw/trellis35/wp16/full-wikipedia-16k-sampled-1w3.json`
**Datum:** 2026-08-30
**Scope:** statische Quellcodeanalyse und Auswertung der im Review-Archiv enthaltenen Benchmarks/Evidence
**Nicht durchgeführt:** eigene RTX-5080-Ausführung; lange WP8B-Qualitätssuite

---

## 1. Kurzurteil

Der aktuelle Stand ist deutlich besser als der frühere ~951-tok/s-Prefill und bestätigt die bisherige W4A8-Entscheidung:

```text
Prefill:          4.564,50 tok/s
Ordinary Decode:    125,72 tok/s
Fixed-D2 Decode:    168,94 tok/s
```

Alle drei retained Runs erzeugen exakt 1.229 Tokens. Ordinary und D2 haben denselben Output-Hash. D2 akzeptiert 690 von 1.076 Vorschlägen, arbeitet in 538 Target-Gruppen, erreicht eine mittlere akzeptierte Länge von 1,28253 und verwendet keinen Ordinary-Fallback.

Die verbleibenden Ziele sind nicht unrealistisch:

```text
5.000 Prefill tok/s:
  benötigt +9,54 % Durchsatz
  bzw. 312,64 ms weniger auf dem 16K-Prompt
  bzw. 19,54 ms weniger je 1.024-Token-Chunk

180 Fixed-D2 tok/s:
  benötigt +6,55 % Durchsatz
  bzw. 0,364 ms weniger pro erzeugtem Token
  bzw. etwa 0,830 ms weniger je D2-Target-Gruppe
```

Die höchste Wahrscheinlichkeit, 5.000 Prefill tok/s zu überschreiten, liegt bei einer **M64-Ausführung für die realen 1.024-Token-Chunks**, gefolgt von einer gezielten Reduktion der LSU-/Synchronisationskosten im fused-N128-Kernel.

Für 180 D2 tok/s ist der stärkste unmittelbare Hebel eine Kombination aus:

1. einmalig materialisierten Unique-Expert-Tasks statt wiederholter 24er-Scans;
2. vierstufigem Weight-Prefetch im T3-M16-Kernel;
3. Beseitigung der thread-lokalen dynamisch indizierten `assignments[3]`;
4. anschließend einem persistenten, grid-stride T3-Projektionskernel, falls das neue Profil weiterhin eine Partial-Wave-/Load-Imbalance zeigt.

Eine neue Quantisierung mit H256/H64 ist interessant, aber für die aktuellen Geschwindigkeitsziele nicht Voraussetzung. Sie sollte als separater, späterer Quantizer-Checkpoint behandelt werden.

---

## 2. Belastbare aktuelle Fakten

### 2.1 Full Wikipedia 16K

Konfiguration:

```text
Prompt:                16.384 Tokens
Sampling:              temperature 1.0, top-k 64, top-p 0.95, seed 0
KV:                    Checkpoint-FP8
MTP:                   Fixed D2
Warm-up-Paare:         1
Retained-Paare:        3
Output:                1.229 Tokens in jedem Lauf
```

Median:

| Pfad | Prefill | Decode | ITL |
|---|---:|---:|---:|
| Ordinary | 4.564,07 tok/s | 125,72 tok/s | 7,954 ms |
| Fixed-D2 | 4.564,50 tok/s | 168,94 tok/s | 5,919 ms |

D2:

```text
Vorschläge:             1.076
Akzeptiert:               690
Abgelehnt:                386
Target-Gruppen:           538
Mittlere Akzeptanz:      1,28253 Token/Gruppe
Akzeptanzrate:          64,126 %
Ordinary-Fallback:          0
D2/Ordinary-Speedup:     1,3438×
```

### 2.2 Abstand zur NVFP4-Referenz

| Grenze | Trellis35 | NVFP4 | Retention | Abstand |
|---|---:|---:|---:|---:|
| Prefill | 4.564,50 | 6.965,63 | 65,53 % | −34,47 % |
| Ordinary | 125,72 | 148,44 | 84,69 % | −15,31 % |
| Fixed-D2 | 168,94 | 203,83 | 82,88 % | −17,12 % |

### 2.3 Aktueller Artifact-Vertrag

```text
Trellis-Payload:        exakt 3,5 bpw
Target-Arena:           12.204.692.480 Byte
Ersparnis zu NVFP4:      2.491.975.680 Byte
                         2,3208 GiB
```

M1, T3 und Prefill dekodieren dieselbe persistente K3/K4-Expert-Repräsentation zu E4M3 und rechnen W4A8. Es gibt keine persistente NVFP4-Expert-Zweitkopie, kein Streaming und keine wiederkehrende Token-Loop-Allokation.

---

## 3. Was WP12 bis WP16 bereits gelöst haben

### Echte M32-MMA

Der Prefill verwendet inzwischen echte:

```text
M16 × N8 × K32 E4M3 MMA
```

als zwei M16-Tiles pro 32 gerouteten Zeilen. A wird mit `cp.async` doppelt gepuffert.

WP12:

```text
Projektionskernel:      1,058 ms -> 0,171 ms
Beschleunigung:         6,19×
Register:              56/Thread
Spills:                0
Achieved Occupancy:    59,14 %
```

### Warp-H128

WP13 ersetzt die direkten 128-FMA-Transformationen durch:

```text
32 Lanes × 4 Werte
lokaler H4
5 Shuffle-XOR-Stufen
```

und fusioniert Transform, Maximum, Scale und E4M3-Quantisierung.

```text
Transformzeit:         136,758 ms -> 7,372 ms
512-Token-Prefill:       2.079,73 -> 4.331,23 tok/s
```

### Fused N128

WP14 vereinigt je Expert-/M32-Tile alle 128 Ausgabespalten sowie inverse H128/SVH in einem Kernel:

```text
512 Threads / 16 Warps
M32
N128
K32
19.584 Byte statisches Shared Memory
56 Register/Thread
keine Spills
65,25 % achieved Occupancy
```

Projection-/Inverse-Starts über 30 Layer gingen im 512er-Prefill von 1.980 auf 60 zurück.

### T3-M16

WP15 verwendet eine echte M16-A-Matrix mit bis zu drei gültigen Zeilen:

```text
70 -> 39 Register/Thread
38,73 -> 32,52 Mio. Instruktionen
T3-Projektion: 4,921 -> 3,641 ms pro profiliertem Target-Graph
```

Der verbleibende T3-Kernel zeigt aber:

```text
34 % Long-Scoreboard-Anteil
lokalen Speicher durch dynamische Arrays
unkoaleszierte Loads/Stores
eine ungünstige Partial Wave
```

---

## 4. Neue Primärdiagnose für Prefill

### 4.1 M32 passt zum 512er-Test, aber nicht ideal zu den produktiven 1.024er-Chunks

Der vollständige Prompt wird in:

```text
16 Chunks × 1.024 Tokens
```

ausgeführt.

Pro Chunk entstehen:

```text
1.024 × Top-8 = 8.192 Expert-Assignments
```

Bei 128 Experten beträgt das arithmetische Mittel:

```text
64 Rows je Expert
```

Der aktuelle Schedule-Builder schneidet jeden Expert in 32er-Tiles. Ein Expert mit 64 Rows benötigt daher zwei vollständige M32-Tiles. Sein gesamter Trellis-Gewichtsblock wird für jedes N128-Ausgabestück zweimal dekodiert.

Die reale Verteilung ist nicht zwingend gleichmäßig; deshalb muss WP17 zunächst je Layer und Chunk messen:

```text
sum ceil(rows_e / 32)
sum ceil(rows_e / 64)
```

Genau dieses Verhältnis beziffert den möglichen Decode-Amortisationsgewinn.

### 4.2 Warum M64 nicht einfach doppelte Rechenarbeit bedeutet

Ein M64-Tile enthält vier M16-Fragmente.

Bei 64 gültigen Rows ist die Zahl der Tensor-Core-MMAs dieselbe wie bei zwei M32-Tiles:

```text
2 × M32:
  2 CTAs × 2 M16 = 4 M16-MMAs je K32/N8

1 × M64:
  1 CTA × 4 M16 = 4 M16-MMAs je K32/N8
```

Aber:

```text
Trellis-B-Decode:       ungefähr halbiert
Descriptor/Schedule:    halbiert
A-/SVH-/Scale-Setup:    weniger häufig
CTA-Control:            weniger häufig
```

Ein uniformer `valid_m16_tiles`-Wert muss verhindern, dass M64 bei Tails unnötige MMAs ausführt.

### 4.3 Erwartete Ressourcen

Erste M64N128-K32-Geometrie:

```text
Threads:                 512
Warps:                    16
Rows:                     64
Output:                  128
A Double Buffer:       4.096 Byte
Projection Shared:    32.768 Byte
Rows/Scale/SVH:          ~768 Byte
Gesamt vor B-Staging:  ~37,1 KiB
```

Vier statt zwei M16-Akkumulatorfragmente benötigen voraussichtlich etwa acht zusätzliche FP32-Register. Der kritische Compile-Grenzwert ist deshalb ungefähr:

```text
<= 64 Register/Thread
```

Das ist eine Designannahme, keine vorab bewiesene ptxas-Zahl. Überschreitet der Kernel diese Grenze und fällt dadurch auf nur einen CTA/SM, muss die Live-Range reduziert oder der Rate-Pfad separat kompiliert werden.

---

## 5. Zweite Prefill-Diagnose: unnötige LSU-Arbeit im fused-N128-Kernel

Der aktuelle Kernel hat 16 Warps. Jeder Warp berechnet N8.

### 5.1 Doppelte Payload-Loads je N16

Benachbarte Warp-Paare bearbeiten die beiden N8-Hälften desselben N16-Trellis-Tiles. Beide Warps laden deshalb dieselben gepackten Trellis-Wörter aus globalem Speicher.

Pro K32 und CTA sind eindeutig nötig:

```text
8 N16-Tiles × 2 K16-Hälften

K3: 8 × 2 × 24 U32 = 384 U32 = 1.536 Byte
K4: 8 × 2 × 32 U32 = 512 U32 = 2.048 Byte
```

Der aktuelle Warp-pro-N8-Pfad lädt ungefähr das Doppelte.

Ein Shared-Payload-Stage ist für K4 besonders sauber:

```text
512 Threads
512 eindeutige U32-Wörter pro K32
=> maximal ein Wort je Thread
```

### 5.2 Redundante Activation-Scale-Loads

Für ein M32-Tile existieren nur 32 unterschiedliche Activation-Scales. Im Store-Pfad werden dieselben Scales über viele Output-Warps erneut geladen.

Diese 32 beziehungsweise später 64 Scales sollten einmal in Shared Memory abgelegt werden.

### 5.3 Redundante SVH-Loads

Für einen Expert und einen N128-Block existieren nur 128 SVH-Half-Werte. Der aktuelle N128-Kernel lässt jeden der 16 Warps dieselbe SVH-Region verwenden.

Die 128 Werte können einmal je CTA nach Shared geladen werden. Die Warps lesen anschließend dieselben Shared-Adressen, was auf Broadcast/Multicast-freundlichen Zugriff hinausläuft.

### 5.4 K32-Barriere in jeder Iteration

Aktuell gibt es bei jedem K32-Schritt:

```text
cp.async commit
wait_group 0
__syncthreads()
```

Gate+Up:

```text
2816 / 32 = 88 Iterationen
```

Down:

```text
768 / 32 = 24 Iterationen
```

Ein K64-Pipeline-Prototyp könnte dies auf 44 beziehungsweise 12 Synchronisationspunkte reduzieren. Das Shared-Memory-Budget muss zusammen mit M64 und B-Staging exakt durch ptxas/NCU bestätigt werden.

---

## 6. Dritte Prefill-Diagnose: materialisierter GELU-Produktpuffer

Der physische BF16-Pfad arbeitet aktuell so:

```text
Gate+Up fused N128
  -> BF16 Gate/Up Output

separater GatedGeluBf16Kernel
  -> BF16 product[Assignment][704]

separater Down Transform/Scale/E4M3-Kernel
```

Für einen 1.024-Token-Chunk:

```text
8.192 Assignments × 704 × 2 Byte
= 11.534.336 Byte Produktpuffer
```

Dieser Puffer wird pro Layer geschrieben und anschließend wieder gelesen.

Über:

```text
30 Layer × 16 Chunks
```

sind das rund 11,1 GB zusätzliche logische Schreib-/Lesedaten, bevor Cacheeffekte berücksichtigt werden.

Ein neuer Kernel kann:

```text
BF16 Gate lesen
BF16 Up lesen
GELU * Up berechnen
explizit auf BF16 runden
SUH anwenden
H128 ausführen
amax bestimmen
direkt E4M3 schreiben
```

Die explizite BF16-Rundung nach GELU×Up muss exakt erhalten bleiben.

Dieser Kandidat ist wahrscheinlich kein alleiniger 10-%-Gewinner, aber er beseitigt eine ganze Kernelgrenze und einen großen transienten Pufferdurchlauf.

---

## 7. Neue Primärdiagnose für Fixed-D2

### 7.1 Benötigte Einsparung

Aktuell:

```text
ITL:                  5,9193 ms
Ziel 180 tok/s:       5,5556 ms
Differenz:            0,3638 ms/Token
```

Bei 1.228 gemessenen Intervallen und 538 Target-Gruppen:

```text
erforderliche Einsparung:
~0,830 ms je Target-Gruppe
```

Der zuletzt isoliert gemessene Trellis-T3-Projektionsanteil beträgt:

```text
3,641 ms je Target-Graph
```

Eine Verringerung dieses Anteils um rund 23 % würde algebraisch bereits ungefähr die notwendige 0,83-ms-Einsparung liefern, falls die übrige Chain unverändert bleibt. Das ist eine Zielrechnung, keine Speedup-Zusage.

### 7.2 Wiederholter Task-Aufbau

Der T3-Kernel startet aktuell über 24 Assignment-Kandidaten. Jeder Output-CTA:

1. prüft alle früheren Kandidaten auf Duplikate;
2. durchsucht erneut alle drei Rows × acht Slots;
3. legt `assignments[3]` an;
4. verwendet dieses dynamisch indizierte Array.

Dasselbe wird für jeden N-Block und anschließend noch einmal für Down ausgeführt.

Aus älterer realer Router-Telemetrie ergibt sich für T3 ungefähr eine mittlere Union von rund 15–16 Unique Experts, nicht 24. Ein kompakter Task-Builder sollte einmal je Layer erstellen:

```text
expert
assignment_count
assignment_0
assignment_1
assignment_2
rate
```

Gate+Up und Down konsumieren dieselbe Taskliste.

Ein Descriptor passt in 32 oder 64 Bit. Dadurch entfallen:

- wiederholte 24er-Scans;
- das dynamisch indizierte lokale Array;
- ein Teil der unkoaleszierten lokalen Zugriffe.

### 7.3 Fehlendes Software-Prefetch

Der M1-Decoder verwendet bereits vierstufiges Prefetch für die gepackten Trellis-Wörter.

`AccumulateGroupedT3M16Direct` lädt dagegen in jeder K32-Iteration synchron:

```text
lane_word0
lane_word1
decode
MMA
```

Das passt zum gemessenen Long-Scoreboard-Anteil von etwa 34 %. Der T3-M16-Kernel hat mit 39 Register/Thread genügend nominellen Spielraum für dieselbe vierstufige Prefetch-Struktur; die reale Ressource muss dennoch gemessen werden.

### 7.4 Partial-Wave-/Grid-Problem

Die synthetische erste Gate+Up-Projektion läuft als:

```text
Grid: 44 × 24 = 1.056 CTAs
Block: 128 Threads
```

Nsight weist eine volle und eine partielle Wave aus.

Nach dem Task-Builder sollte ein fixer, CUDA-Graph-kompatibler persistent-grid-stride Consumer getestet werden:

```text
work_items = unique_tasks × output_blocks

for (item = blockIdx.x;
     item < work_items;
     item += gridDim.x)
```

Eine feste Gridgröße von beispielsweise 4, 6 oder 8 CTAs/SM wird gemessen, nicht vorausgesetzt. Ein dynamischer Host-Sync zum Lesen des Task-Counts ist nicht zulässig.

---

## 8. Rangfolge der nächsten Kandidaten

| Rang | Kandidat | Hauptziel | Risiko | Erwartete Relevanz |
|---:|---|---|---|---|
| 1 | M64N128 + M32-Tails | Prefill | mittel | sehr hoch |
| 2 | Shared Scale/SVH/B-Payload + K64-Pipeline | Prefill | mittel | hoch |
| 3 | GELU→Down-Transform-Fusion | Prefill | niedrig–mittel | mittel |
| 4 | T3 Tasklist + Prefetch | D2/Ordinary | niedrig–mittel | sehr hoch |
| 5 | T3 persistent-grid Consumer | D2 | mittel | hoch |
| 6 | 2.048er Prefill-Chunk | Prefill | Speicher-/Numerikrisiko | potenziell hoch |
| 7 | Trellis→W4A4 | Prefill | sehr hoch | noch nicht zugelassen |

M64 ist der beste erste Kandidat, weil der produktive Benchmark ausschließlich 1.024er-Prefill-Chunks verwendet und damit doppelt so viele durchschnittliche Expert-Rows wie die ursprüngliche 512er-Tuningform besitzt.

---

## 9. Work Packages WP17 bis WP22

## WP17 – Reales 1.024er-Profil und M64N128

### Ziel

Den Trellis-B-Decode über bis zu 64 geroutete Rows amortisieren.

### Implementierung

1. Reale Route-Telemetrie für alle 30 Layer und alle 16 Chunks erfassen.
2. Je Layer berichten:

```text
rows/expert histogram
M32 tile count
M64 tile count
M64 full/tail count
K3/K4 task count
```

3. Trellis-spezifische M64-Schedule ergänzen; NVFP4-M32-Schedule nicht ändern.
4. `MmaW4A8ProjectionGroupedPrefillM64N128Kernel`:
   - 512 Threads;
   - 16 Warps;
   - vier M16-Fragmente;
   - K32 zunächst unverändert;
   - ein N128-Outputblock;
   - uniformes `valid_m16_tiles`.
5. M32-Rollback behalten.
6. Optional zwei Schedule-Klassen:
   - M64 für 33–64 Rows;
   - M32 für 1–32 Rows.
7. Falls Register >64:
   - K3/K4 als getrennte Template-Launches;
   - Live-Ranges reduzieren;
   - nicht durch Spills „erzwingen“.

### Gate

- vollständige Operatorparität;
- keine Spills;
- mindestens zwei CTAs/SM oder gleichwertig gemessene Occupancy;
- Full Wikipedia 16K mindestens +5 %;
- Stretch: mindestens 5.000 tok/s;
- keine Ordinary-/D2-Regressionsänderung.

---

## WP18 – Fused-N128 LSU- und Barriereoptimierung

### Ziel

Doppelte Loads und 112 K32-Synchronisationspunkte je Gate+Up/Down-Paar reduzieren.

### Unterkandidaten separat messen

A. Activation-Scales einmal in Shared.
B. 128 SVH-Werte einmal in Shared; vorher SASS prüfen, ob der Compiler bereits vollständig hoistet.
C. Trellis-Payload je N16-Warp-Paar einmal nach Shared laden.
D. Projektion mit `float2` speichern und `float4` laden.
E. BF16-Output gepackt speichern.
F. K64-A/B-Staging zur Halbierung der Barrieren.
G. SASS prüfen, ob FP16→E4M3-Konvertierung bereits als gepackte Konvertierung ausgegeben wird; nur bei Bedarf explizite x2-Konvertierung einführen.

### Gate

Jeder Unterkandidat:

- bitidentisch;
- keine Spills;
- kein Context-/Workspace-Delta ohne explizite Bilanz;
- aktueller Kernel-Microbenchmark gewinnt;
- Full 16K gewinnt;
- Verlierer vollständig zurückrollen.

Kumulatives Stretch-Ziel nach WP18:

```text
>= 5.000 Prefill tok/s
```

---

## WP19 – Gated-GELU plus Down-Transform/Quantisierung

### Ziel

Den BF16-Produktpuffer aus dem Hotpath entfernen.

### Implementierung

Neuer Kernel:

```text
GatedGeluDownTransformQuantizeBf16WarpKernel
```

Der Kernel muss dieselbe Reihenfolge materialisieren:

```text
gate = BF16(gate_bits)
up   = BF16(up_bits)
product_bits = BF16Bits(GELU(gate) * up)
product = BF16(product_bits)
transformed = H128(product * SUH)
scale = amax / 448
E4M3 = quantize(transformed / scale)
```

Der alte Produktpuffer bleibt als Rollback-Workspace, wird im neuen Pfad aber nicht gelesen/geschrieben.

### Gate

- E4M3-Bytes und Scales bitidentisch zum Zweikernelpfad;
- Full-Model-Output identisch;
- keine neue Allokation;
- Full 16K mindestens +1 % oder klarer Kernelzeitgewinn ohne Full-Path-Regress;
- sonst zurückrollen.

---

## WP20 – T3 Tasklist, Prefetch und persistent-grid

### Ziel

Fixed-D2 von 168,94 in Richtung 180 tok/s bringen.

### Stufe A: Tasklist

Ein kleiner, graphfähiger Kernel erstellt maximal 24 kompakte Unique-Expert-Tasks.

Gate+Up und Down verwenden dieselbe Liste.

Kein `assignments[3]` im thread-lokalen Speicher.

### Stufe B: vierstufiges Payload-Prefetch

Portiere die bestehende M1-Prefetch-Struktur in den T3-M16-Decoder.

### Stufe C: persistent-grid Consumer

Nur falls NCU weiterhin Partial-Wave/Load-Imbalance zeigt:

```text
fixed grid
grid-stride work-items
kein Host-Sync
```

### Stufe D: B-Payload-Pair-Sharing

Benachbarte N8-Warps teilen denselben N16-Payload.

### Gate

- Target-Output exakt;
- gleiche 1.229 finalen Tokens;
- gleiche accepted/rejected Drafts für den eingefrorenen Benchmark bei rein arithmetischen Änderungen;
- kein Fallback;
- keine Token-Loop-Allokation;
- T3-Projektion mindestens 15 % schneller;
- Full D2 mindestens 175 tok/s;
- Stretch: mindestens 180 tok/s;
- Ordinary darf nicht regredieren.

---

## WP21 – 2.048er Prefill-Chunk als separater Capacity-/Performanceversuch

### Motivation

Mit 2.048 Tokens:

```text
16.384 Assignments
arithmetisch 128 Rows/Expert
nur 8 statt 16 Chunks
```

Das kann:

- Host-/Chunk-Grenzen halbieren;
- Gewichtsdecode weiter amortisieren;
- M64 besser auslasten;
- eventuell später M128 rechtfertigen.

### Voraussetzungen

Vor Codeänderung exakt planen:

```text
zusätzliche Activation-/Expert-/Attention-Workspaces
freie Device-Bytes bei 32K, 64K und maximalem Kontext
MTP-Reserve
Graph-private Bytes
```

### Gate

- 1.024 bleibt Rollback;
- keine versteckte Context-Verkürzung;
- keine OOM-Nähe unter der festgelegten Reserve;
- numerische Änderung durch andere Chunkgrenzen explizit prüfen;
- Full 16K gewinnt;
- Kontextkapazität und Peak-VRAM publizieren.

Nicht mit M128 beginnen, bevor M64 und die reale 2.048er-Routingverteilung vermessen sind.

---

## WP22 – Neue W4A8/W4A4-Entscheidung

W4A4 bleibt gesperrt, bis nach WP17–WP21 mindestens eines belegt ist:

- W4A8 Tensor-Pipe ist materiell gesättigt;
- decoderfreier/optimal gestagter W4A8-Pfad zeigt eine Compute-Decke;
- Decoder-, LSU-, Barrier- und Schedule-Kosten sind nicht mehr dominant;
- erwarteter FP4-Gewinn übersteigt die zusätzliche E2M1-/Scale-/Qualitätskomplexität.

Vor W4A4 werden weiterhin benötigt:

```text
E2M1-Rekonstruktionsvertrag
E4M3-K/16-Skalenlayout
Byte-Delta
zweiter Quantisierungsfehler
Runtime-Decoderkosten
neue Owner-Freigabe
```

---

## 10. Separater späterer Quantizer-Checkpoint

Die adaptive Transformidee wird **nicht** in WP17–WP22 gemischt.

Späterer Quantizer-Screen:

```text
Gate+Up:
  Input  H256
  Output H128

Down:
  Input  H64
  Output H256
  kein 704→768-Padding
```

Vergleichsraten:

```text
3,5 bpw
3,25 bpw
3,0 bpw
```

Zuerst nur repräsentative Layer/Experts. Ein Overnight-Full-Quant wird erst gestartet, wenn der kleine Screen bei Proxy Error, NMSE, Layer-Output und kurzer KL mindestens einen klaren Vorteil zeigt.

---

## 11. Was jetzt nicht verfolgt werden sollte

- kein weiterer Struktur-Refactor; WP10 hat die Dateien bereits sinnvoll zerlegt;
- kein W4A16;
- noch kein H256-Overnight-Job;
- noch kein W4A4;
- kein persistenter E4M3-/NVFP4-Expert-Cache;
- kein TMA-Umbau ohne Barrier-/LSU-Nachweis;
- kein M128 vor M64 und 2.048er-Telemetrie;
- keine lange WP8B-Suite;
- keine Vision-/MMProject-Integration;
- keine Änderung der Sampling-, KV-, BF16-Roundpoint- oder Timingsemantik;
- keine Optimierung, die nur im 512er-Microbenchmark gewinnt und im vollständigen 16K-Panel verliert.

---

## 12. Empfohlene Reihenfolge

```text
WP17  M64N128 auf realen 1.024er-Chunks
  ↓
WP18  Shared Payload/Scale/SVH + K64-Pipeline
  ↓
WP19  GELU→Down-Transform-Fusion
  ↓
WP20  T3 Tasklist + Prefetch + persistent-grid
  ↓
WP21  optional 2.048er Chunk
  ↓
WP22  W4A8/W4A4 neu entscheiden

separat:
QX1  Adaptive H256/H128/H64 Quantizer-Screen
```

**Nächster Auftrag an Codex:** WP17.
**Wahrscheinlichster Weg über 5.000 Prefill:** WP17, eventuell zusammen mit den ersten WP18-Unterkandidaten.
**Wahrscheinlichster Weg über 180 D2:** WP20 Tasklist + Prefetch; persistent-grid nur nach neuem NCU.

# Gem16 Trellis35 nach WP9 – Performance- und Architekturreview

**Review-Basis:** `codex/gemma4-26b-trellis35-w4a8@fbc0121ad6c699d85d9d9e7792083e11744b2eba`
**Datum:** 2026-08-30
**Review-Art:** statische Quellcodeanalyse plus Auswertung der enthaltenen kompakten Nsight-/Benchmark-Evidenz
**Nicht enthalten:** ausführbare Binaries, vollständige Profiler-Datenbanken und Modellgewichte

---

## 1. Kurzurteil

Der bisherige Trellis35-Zweig hat seinen ursprünglichen Discovery-Zweck erfüllt:

- echtes 3,5-bpw-Payload;
- nur eine persistente routed-expert-Repräsentation;
- rund 2,3208 GiB weniger Target-Gewichtsspeicher;
- kein CPU-Offload, kein Expert-Streaming, kein Runtime-Repack;
- Ordinary M=1, echtes gebatchtes T3 und Prefill funktionieren;
- der WP9-Decoder hat den größten offensichtlichen Trellis-Rekonstruktionsfehler beseitigt.

Die Architektur ist **nicht in einer Sackgasse**, aber der aktuelle Prefill-Kernel nutzt die FP8-MMA in einer Form, die den größten Teil ihrer M-Dimension verschwendet. Das ist wesentlich schwerwiegender als ein gewöhnliches Tile-Tuning-Problem. Der nächste große Schritt muss daher ein **echter M16/M32-Gruppen-GEMM** sein, nicht ein weiteres Vergrößern des Arrays separater M=1-Akkumulatoren.

Vor diesem Umbau empfehle ich genau **ein eng begrenztes, verhaltensneutrales Strukturpaket**. Kein monatelanger Cleanup und keine breite Neuarchitektur des NVFP4-Pfads. Ziel ist, Trellis35, Engine-Orchestrierung und Tests so zu zerlegen, dass die nächste Kernelgeneration reviewbar bleibt, ohne die bereits qualifizierte NVFP4-Mathematik anzufassen.

Wichtig: Der aktuelle Trellis-Zweig hat den NVFP4-Pfad **nicht ersetzt**. Beide Pfade existieren bereits parallel und es wird pro Modellverzeichnis genau ein Artifact geladen. Dieser Zustand sollte jetzt allerdings mit einem expliziten Formatvertrag gehärtet werden, statt langfristig auf einem `bool trellis35` und bloßer Marker-Datei-Erkennung zu beruhen.

---

## 2. Nachgewiesener aktueller Stand

### 2.1 Artifact und Speicher

Enthaltene Evidenz:

```text
Trellis-Payload:                 exakt 3,5 bpw
Effektive Expert-Rate inkl.
Sidecars/Alignment:              ca. 3,6271 bpw

Aktuelles NVFP4 Target-Arena:    14.696.668.160 Byte
Trellis35 Target-Arena:          12.204.692.480 Byte
Ersparnis:                        2.491.975.680 Byte
                                 = 2,3208 GiB
```

Gate+Up bleibt `2816 × 1408` ohne Padding. Down ist physisch `768 × 2816`, logisch weiterhin `704 × 2816`.

Die Trellis-Artifact-Evidenz bestätigt:

- eine Device-Arena;
- keine routed NVFP4-Expert-Bytes;
- keine zweite persistente Expert-Bank;
- unveränderte nicht geroutete FP8/NVFP4/BF16-Tensorfamilien.

### 2.2 Gemessene Leistung

| Pfad | WP9 aktuell | vorhandene NVFP4-Referenz | Verhältnis |
|---|---:|---:|---:|
| 16K Prefill | ca. 951,1 tok/s | ca. 6.966 tok/s | ca. 13,7 % |
| Ordinary Decode | 119,4 tok/s | ca. 148,4 tok/s | ca. 80,4 % |
| T3-Verifier intern | 168,5 tok/s | ca. 203,8 tok/s als Produktreferenz | ca. 82,7 % |
| kurzer D2-Gesamtdiagnoselauf | 85,2 tok/s | nicht direkt vergleichbar | – |

Der kurze D2-Gesamtlauf enthält 0,37765 Sekunden Initial Selection und verwendet nicht dieselbe lange, wiederholte Timingverteilung wie die veröffentlichte 203,8-tok/s-Referenz. Die 85,2 sind daher **kein sauberer Beweis für einen 58-%-Steady-State-Verlust**. Der T3-Verifier mit 168,5 tok/s ist die belastbarere derzeitige Kernel-Nähe; eine timinggleiche End-to-End-Messung fehlt noch.

### 2.3 WP9-Erfolg

Der M=1-Projektionskernel ging laut Evidenz von:

```text
734,0 µs → 73,15 µs
```

Die gemeldete L1-Lesemenge fiel von ungefähr 1,88 GB auf 35,95 MB. Der Kernel verwendet 48 Register pro Thread.

Damit ist der frühere branch-history-basierte Decoder als primärer M=1-Fehler praktisch erledigt. Der Prefill bleibt langsam, weil derselbe dekodierte Gewichtsblock noch über zu kleine Zeilengruppen und ineffiziente MMA-Fragmente verbraucht wird.

---

## 3. Koexistenz von NVFP4 und Trellis35

### 3.1 Aktueller Quellzustand

Die Koexistenz ist real:

- `Gemma4Moe26BReferenceEngine::Create` in
  `src/cuda/engine/gemma4_26b_reference.cu` prüft auf
  `trellis35-checkpoint.json`.
- Bei Trellis wird `Gemma4Moe26BTrellis35DeviceArtifact` geladen.
- Andernfalls wird der bestehende `Gemma4Moe26BDeviceArtifact`-/NVFP4-Pfad geladen.
- Decode, T3 und Prefill dispatchen anschließend jeweils auf getrennte
  `LaunchTrellis35...`- oder bestehende `Launch...Sm120...`-Funktionen.
- `src/cuda/moe/gemma4_26b_bindings.cpp` bindet routed NVFP4-Gewichte nur,
  wenn das Modell kein Trellis35-Artifact ist.

Damit hält eine Engine-Instanz nicht beide Expert-Banken gleichzeitig.

### 3.2 Was daran gehärtet werden sollte

Der aktuelle Mechanismus ist funktional, aber langfristig zu implizit:

```cpp
bool trellis35;
```

plus

```text
Existiert trellis35-checkpoint.json?
```

bedeutet, dass bei einem versehentlich gemischten Modellverzeichnis Trellis implizit Vorrang erhalten kann.

Empfohlener Vertrag:

```cpp
enum class RoutedExpertFormat {
  kNvfp4,
  kTrellis35,
};
```

Das Format muss aus validierten Artifact-Metadaten stammen und mit der vom Modellkatalog erwarteten Variante übereinstimmen.

Fail-closed-Fälle:

- Trellis-Marker plus vollständige NVFP4-Expert-Familie;
- Trellis-Profil mit fehlendem Trellis-Payload;
- NVFP4-Profil mit Trellis-Raten/Deskriptoren;
- Modellkatalog erwartet NVFP4, Verzeichnis enthält Trellis;
- Modellkatalog erwartet Trellis, Verzeichnis enthält NVFP4.

### 3.3 Produktziel

Später sollen ausdrücklich zwei getrennte Modelle/Profile angeboten werden:

```text
Gemma 4 26B – NVFP4
  höchste qualifizierte Geschwindigkeit

Gemma 4 26B – Trellis35
  weniger Weight-VRAM, längerer Kontext,
  eigene Performance-/Qualitätsqualifikation
```

Das Trellis-Profil ist keine automatische „bessere“ Version des NVFP4-Modells und darf dessen Modell-ID, Locks oder Performance-Evidenz nicht überschreiben.

---

## 4. Dateistruktur und Refactoring-Entscheidung

### 4.1 Aktuelle große Dateien

| Datei | Zeilen |
|---|---:|
| `tests/cuda/nvfp4_reference_test.cu` | 5.429 |
| `src/cuda/engine/gemma4_26b_reference.cu` | 4.010 |
| `src/cuda/nvfp4/sm120.cu` | 2.700 |
| `src/cuda/attention/decode_sm120.cu` | 2.551 |
| `tests/cuda/trellis35_m1_test.cu` | 1.692 |
| `src/cuda/trellis35/reference.cu` | 1.667 |
| `tools/gem16_compile/compiler.py` | 1.663 |
| `src/cuda/moe/reference.cu` | 1.463 |

Das Trellis-Problem ist weniger die aktuelle Zeilenzahl von 1.667 allein, sondern dass dieselbe Datei gleichzeitig enthält:

- Trellis-State-Extraktion;
- K3/K4-Codebook-Decodierung;
- E4M3-Packing;
- MMA-Intrinsics;
- M1;
- T3;
- Prefill;
- Input-Transforms;
- Output-Transforms;
- Schedule-Bau;
- GELU und Reduktion;
- sämtliche Host-Launcher.

Der kommende echte M16/M32-Prefill würde diese Kopplung deutlich verschlimmern.

### 4.2 Empfehlung

**Ja, zuerst refactoren – aber nur ein Paket lang und codegen-neutral.**

Nicht empfehlenswert ist:

- allgemeines „Clean Architecture“-Projekt;
- Umbau aller großen CUDA-Dateien;
- Aufspaltung des qualifizierten NVFP4-Kernels parallel zum Trellis-Tuning;
- Einführung virtueller Hotpath-Interfaces oder eines generischen Frameworks.

Empfohlen ist zunächst eine physische Quellaufteilung bei **derselben CUDA-Translation-Unit**. Dadurch bleibt die Wahrscheinlichkeit identischer SASS hoch.

Vorschlag:

```text
src/cuda/trellis35/
  reference.h                  # vorerst kompatibler öffentlicher Header
  reference.cu                 # dünner Aggregator, weiterhin eine TU

  detail/
    codec.cuh                  # State window, K3/K4 decode, E4M3 pack
    mma_w4a8.cuh               # Fragmenttypen und MMA-Intrinsics
    transform_common.cuh       # H128-Helfer, BF16/FP8-Konvertierung
    m1_kernels.inc.cuh
    t3_kernels.inc.cuh
    prefill_kernels.inc.cuh
    launchers.inc.cuh
```

`reference.cu` inkludiert die `.inc.cuh`-Shards in unveränderter Reihenfolge. Erst wenn SASS, Ressourcen und numerische Identität belegt sind, können daraus mehrere echte `.cu`-Translation-Units werden.

Für die 4.010-Zeilen-Engine ebenfalls zunächst Include-Shards:

```text
src/cuda/engine/detail/
  gemma4_26b_state.inc
  gemma4_26b_create.inc
  gemma4_26b_decode.inc
  gemma4_26b_prefill.inc
  gemma4_26b_mtp.inc
  gemma4_26b_metrics.inc
```

Die Tests sollten dagegen direkt in mehrere echte Testdateien aufgeteilt werden:

```text
tests/cuda/trellis35_codec_test.cu
tests/cuda/trellis35_transform_test.cu
tests/cuda/trellis35_m1_test.cu
tests/cuda/trellis35_t3_test.cu
tests/cuda/trellis35_prefill_test.cu
tests/cuda/trellis35_runtime_test.cu
tests/cuda/trellis35_test_support.h
```

### 4.3 Refactoring-Gate

Das Strukturpaket darf **keine Mathematik ändern**.

Erforderlich:

- identische aktuelle Output-Hashes;
- identische Artifact-Arena;
- identische Kernelzahl und Kernelbezeichnungen;
- identische Register-/Shared-Memory-Angaben;
- SASS-Hash oder normalisierter Cubin/SASS-Vergleich der Trellis-Hotkernels;
- NVFP4-Produktregression;
- Trellis-M1/T3/Prefill-Benchmark innerhalb 1 % oder des vorher dokumentierten Messrauschens.

---

## 5. Rangfolge der aktuellen Performance-Engpässe

### 5.1 Gemessene Anteile

| Rang | Bereich | GPU-Zeit | Befund |
|---:|---|---:|---|
| 1 | Gruppierte Trellis-Projektion | 63,3 % | falsche M-Nutzung, zu kleine Row Groups, Trellis wiederholt |
| 2 | Float Input Transform: Scale + Quantize | 16,5 % | H128 zweimal vollständig berechnet |
| 3 | Inverse Output Transform | 6,9 % | direkter 128×128-Ansatz, 990 Starts |
| 4 | BF16 Input Transform: Scale + Quantize | 4,5 % | ebenfalls doppelte H128-Berechnung |
| 5 | restlicher vollständiger Pfad | 8,8 % | Attention, Router, Shared MLP, Norms usw. |

Der Schedule-Bau selbst liegt nur bei rund 0,2 %. Sein Algorithmus ist daher nicht der Haupthebel. Die daraus resultierende **zu kleine Schedule-Granularität und Überlaunch-Geometrie** beeinflussen jedoch den 63,3-%-Projektionsblock.

### 5.2 Amdahl-Grenzen – keine Vorhersagen

Aus exakt dieser 512-Token-Profilverteilung:

| Hypothetische Änderung | algebraische Maximalbeschleunigung | aus 951 tok/s |
|---|---:|---:|
| alle Input-/Output-Transforms kostenlos | 1,39× | ca. 1.319 tok/s |
| alle Transforms 4× schneller | 1,26× | ca. 1.203 tok/s |
| nur Projektion 4× schneller | 1,90× | ca. 1.811 tok/s |
| Projektion 4× und alle Transforms 4× | 3,16× | ca. 3.009 tok/s |
| Projektion 8× und alle Transforms 4× | 4,22× | ca. 4.015 tok/s |

Folgerung:

> Transformoptimierung allein kann den Prefill nicht in den Bereich mehrerer Tausend tok/s bringen. Ein mehrfache Beschleunigung des Projektionshauptloops ist zwingend.

---

## 6. Zentrale Prefill-Fehlanpassung: M16-Instruktion als M1-Schleife

### 6.1 Aktueller Kernel

In `src/cuda/trellis35/reference.cu`:

```text
kPrefillRowsPerTile = 4
MmaW4A8ProjectionGroupedPrefillTileKernel
AccumulateGroupedProjection
AccumulateFp8
```

Der aktuelle `AccumulateFp8` ruft:

```text
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
```

auf, liefert aber die A-Register als:

```text
{a0, a0, a1, a1}
```

und verwaltet pro Assignment einen separaten `Fp8Accumulator`.

Damit wird die M16-MMA als Broadcast-/Einzelzeileninstrument verwendet. Bei vier Assignments werden pro K-Schritt vier getrennte MMAs aufgerufen und anschließend nur die benötigten Zeilen herausgeschrieben.

Der zurückgerollte „8 rows per warp“-Versuch verdoppelte lediglich:

- Akkumulatorarrays;
- lebende Register;
- MMAs pro Schleife.

Er füllte nicht die echte M16-A-Matrix. Die 23,1-%-Regression ist deshalb kein Gegenbeweis gegen Row-Reuse, sondern ein Gegenbeweis gegen **mehrere emulierte M1-Zeilen im selben Warp**.

### 6.2 Bereits vorhandene lokale Vorlagen

Zwei bestehende Gem16-Kernel zeigen die benötigten Teile:

1. `src/cuda/fp8/sm120.cu::Sm120MatrixProjectionKernel`
   - echte A-Fragmente `a0,a1,a2,a3`;
   - `M16×N8×K32`;
   - E4M3-Aktivierungen;
   - Shared-Memory-Double-Buffering;
   - Wiederverwendung eines B-Fragments über viele M16-Tiles.

2. `src/cuda/nvfp4/sm120.cu::Sm120GroupedExpertMatrixKernel`
   - Gruppierung nach Expert;
   - 32 Assignments pro Expert-Tile;
   - `permutation`, `prefix`, `expert_tiles`;
   - Nullfüllung für Tail-Zeilen;
   - zwei M16-Token-Tiles;
   - genaue Zuordnung zurück zu Original-Assignments.

Der nächste Trellis-Kernel sollte diese beiden Designs kombinieren und nur den B-Weight-Load durch den Trellis-Decoder ersetzen.

### 6.3 Bereits vorhandene, derzeit verworfene 32er-Schedule

`src/cuda/moe/prefill.cu::BuildExpertTileScheduleKernel` baut bereits Expert-Tiles in 32er-Schritten.

Unmittelbar danach ruft der Trellis-Pfad jedoch `LaunchTrellis35PrefillExpertsW4A8`, und dieser baut mit `BuildTrellis35PrefillTileScheduleKernel` eine zweite Schedule mit Vierergruppen und überschreibt dafür wieder dead router storage.

Das ist ein klarer Umbaupunkt:

```text
bestehende stabile 32er-Expert-Schedule behalten
             ↓
neuer Trellis M32-Kernel konsumiert sie direkt
```

### 6.4 Empfohlene erste Geometrie

Nicht sofort N128/16 Warps bauen.

Erster Kandidat:

```text
CTA:
  4 Warps
  N32 insgesamt
  jeder Warp N8

M:
  2 × M16 = maximal 32 gruppierte Assignments

K:
  K32

A:
  zwei E4M3-M16-Tiles in Shared Memory
  cp.async, doppelt gepuffert

B:
  jeder Warp dekodiert sein N8-Trellis-Fragment einmal pro K32
  dasselbe B-Fragment speist beide M16-MMAs

C:
  zwei MMA-Accumulator-Fragmente je Warp
```

Registerwirkung:

```text
aktuelles M4:
  4 Fp8Accumulator = 16 FP32-Akkumulatorwerte pro Warp-Thread

fehlgeschlagenes M8:
  8 Fp8Accumulator = 32 FP32-Werte

echtes M32:
  2 M16-Fragmente = 8 FP32-Werte
```

Der echte M32-Pfad kann daher mehr Zeilen verarbeiten und gleichzeitig **weniger Akkumulatorregister** benötigen als die heutige M4-Schleife.

### 6.5 Instruktionsamortisierung

Bei 32 echten Zeilen:

- heutiger M4-Pfad benötigt acht Schedule-Gruppen;
- jede Gruppe ruft vier Broadcast-MMAs pro K auf;
- neuer M32-Pfad benötigt eine Schedule-Gruppe;
- sie ruft zwei echte M16-MMAs pro K auf.

Das bedeutet algebraisch ungefähr 16× weniger warpweite MMA-Aufrufe je 32 nützliche Zeilen. Das ist **keine 16×-Laufzeitprognose**, zeigt aber, warum hier ein Architektur- und kein Mikro-Tuninghebel liegt.

Der Trellis-B-Decoder wird außerdem ungefähr achtmal seltener je 32 Zeilen aufgerufen.

---

## 7. Transformpfad

### 7.1 Aktuelles Problem

`PrefillTransformedValue` und `PrefillTransformedBf16Value` berechnen jeden transformierten Wert über eine direkte Schleife von 128 FMA-Operationen.

Dann passiert dies zweimal:

```text
Scale-Kernel:
  vollständige H128-Berechnung
  Maximum suchen

Quantize-Kernel:
  dieselbe H128-Berechnung erneut
  durch Scale teilen
  E4M3 speichern
```

Allein die vier Input-Transform-/Quantize-Kernel belegen 21,0 % der Profilzeit.

Die Output-Transformation macht denselben direkten 128×128-Ansatz und startet für jede 128er-Ausgabekachel separat.

### 7.2 Warp-FWHT

Ein H128 lässt sich mit einem Warp rechnen:

```text
32 Lanes × 4 Werte
lokaler H4
5 Shuffle-XOR-Stufen
Normierung
```

Für Gate+Up Input:

```text
2816 / 128 = 22 H128-Blöcke
2816 float shared = 11 KiB pro Assignment-CTA
```

Für Down Input:

```text
768 / 128 = 6 H128-Blöcke
768 float shared = 3 KiB
```

Empfohlener Input-Kernel:

```text
ein CTA pro Assignment
8 Warps

1. BF16/FP32 laden und SUH multiplizieren
2. Warp-FWHT
3. transformierte Werte einmal in Shared speichern
4. CTA-weites amax
5. Scale bestimmen
6. dieselben Shared-Werte zu E4M3 quantisieren
```

Damit wird H128 nur einmal gerechnet und Scale+Quantize werden ein Kernel.

### 7.3 Physische Präzisionsgrenzen

Diese Grenzen müssen erhalten bleiben:

```text
Gate+Up inverse output
  -> explizite BF16-Rundung
  -> GELU(gate) * up

GELU product
  -> explizite BF16-Rundung
  -> Down input transform

Down inverse output
  -> explizite BF16-Rundung
  -> slot-ordered route reduction
```

Eine Fusion darf die Werte in Registern/Shared weiterreichen, muss aber an exakt derselben Stelle per `__float2bfloat16_rn` runden.

Ein schneller FWHT ändert die Additionsassoziation. Daher ist Bitidentität nicht automatisch zu erwarten. Diese Änderung braucht vorab festgelegte Operator-, Layer- und kleine WP8A-Numerikgates; Toleranzen dürfen nicht nachträglich an Fehler angepasst werden.

---

## 8. Bounded transient E4M3 als Diagnose und möglicher Large-M-Pfad

Ein transienter, wiederverwendbarer E4M3-Slab verletzt die „eine persistente Repräsentation“-Regel nicht, solange er Workspace bleibt.

Für einen N128-Slab über alle 128 Experten:

```text
Gate+Up:
128 × 2816 × 128 Byte ≈ 44 MiB

Down:
128 × 768 × 128 Byte ≈ 12 MiB
```

Diagnose:

```text
Trellis35
  -> einen N128-Slab einmal zu E4M3 dekodieren
  -> decoderfreien echten M16/M32-W4A8-Kernel ausführen
```

Dieser Versuch trennt:

- Decoderkosten;
- Schedule-/MMA-Kosten;
- globalen E4M3-Write/Read-Traffic.

Er ist **nicht** als erster Produktionspfad zu empfehlen. Nach einem echten M32-Inline-Kernel kann der zusätzliche globale Traffic unnötig sein. Vorher ist er jedoch ein sehr gutes Roofline-/Attributionsexperiment.

Kein vollständiges, dauerhaftes E4M3-Modell darf resident bleiben.

---

## 9. Fixed-D2

### 9.1 Der derzeitige Vergleich ist nicht sauber

Der enthaltene kurze Lauf berichtet:

```text
T3-Verifier:            168,5 tok/s
D2-Gesamt:               85,2 tok/s
Initial Selection:       0,37765 s
Output:                  nur kurzer Diagnoselauf
```

Die veröffentlichte NVFP4-Referenz von rund 203,8 tok/s stammt aus einer längeren formalen Verteilung mit anderer Amortisierung.

Nächster Performance-Lauf muss parallel berichten:

```text
a) komplette Anfrage inklusive Initial Selection
b) post-first steady-state
c) reine Target-T3-Verifierzeit
d) Assistant proposal
e) Target attention
f) Target routed/shared MoE
g) Target head/sampling
h) KV backup/tentative/restore/commit
```

### 9.2 Nachgewiesenes unnötiges Trellis-Work

`MoeInputNormsRouterTransformNvfp4Kernel` quantisiert für den Trellis-Pfad weiterhin eine Expert-Aktivierung zu NVFP4, obwohl keine routed NVFP4-Expert-Gewichte gebunden sind.

Danach wird für Trellis separat `LaunchRmsNormBf16` ausgeführt.

Das passiert sowohl im Ordinary-Pfad als auch im T3-Pfad. In
`gemma4_26b_bindings.cpp` wird dafür sogar ein Dummy-Activation-Divisor gesetzt, nur um den alten Kernelvertrag zu erfüllen.

Empfohlener Trellis-spezifischer Input-Boundary-Kernel:

```text
ein RMS-Reduce
  -> Shared-MLP NVFP4-Aktivierung wie bisher
  -> Router-Transform wie bisher
  -> physischer BF16 Expert-Input
  -> kein routed Expert-NVFP4
```

Die Reduktionsreihenfolge und BF16-Rundung müssen gegen den aktuellen Trellis-Pfad bitidentisch bleiben. Falls das nicht gelingt, wird der Kandidat verworfen und nicht über tolerantere Tests legitimiert.

### 9.3 T3-MMA

Der T3-Kernel dekodiert ein Weight-Fragment bereits nur einmal je erstem Expert-Vorkommen und speist bis zu drei Zeilen. Das ist gut.

Später kann auch hier eine echte M16-A-Matrix mit bis zu drei gültigen und 13 Nullzeilen verwendet werden. Das reduziert MMA-Aufrufe und Akkumulatorverwaltung, aber der erwartete Hebel ist kleiner als beim Prefill, weil je Expert höchstens drei T3-Zeilen existieren und der Decoder bereits geteilt wird.

Priorität für D2:

1. timinggleiche Profilierung;
2. dead NVFP4-Quantisierung entfernen;
3. schnelle Transforms;
4. erst dann echte T3-M16-MMA;
5. danach weitere Chain-Fusion nur profilergetrieben.

---

## 10. Correctness- und Wartungsrisiken

### 10.1 Warp-Kollektive

`DecodeAdjacentStates` verwendet `__shfl_sync(0xffffffffU, ...)`.

Die heutigen Shape-Verträge sorgen praktisch dafür, dass Warps vor diesen Shuffles vollständig aktiv bleiben. Diese Annahme ist jedoch über mehrere Launcher verteilt und nicht lokal sichtbar.

Für neue Kernel:

- keine lane-divergente Rückkehr vor Warp-Shuffles;
- entweder volle Warp-Teilnahme statisch garantieren oder `__activemask()` korrekt verwenden;
- Tail-Zeilen durch Zero-Fill, nicht durch divergente Return-Pfade behandeln.

### 10.2 Alignment

Trellis-Payloads werden als `uint32_t*` geladen.

Explizit validieren:

```text
pool base % 4 == 0
pool_offset % 4 == 0
tile stride % 4 == 0
Activation E4M3 base/row/K offset für U32/uint4 loads korrekt ausgerichtet
```

Die aktuellen konkreten Shapes erzeugen passende Größen, aber der Vertrag sollte nicht nur aus der Arithmetik „zufällig“ folgen.

### 10.3 Rate-Fallback

Mehrere Kernel verwenden sinngemäß:

```cpp
rate_bits == 3 ? K3 : K4
```

Ein beschädigter Wert wird damit im Device-Code als K4 behandelt. Der Loader validiert aktuell K3/K4, dennoch sollte der Trusted-Binding-Vertrag explizit sein und eine Korruptionsprüfung sicherstellen, dass kein ungültiger Descriptor bis zum Launch gelangt.

### 10.4 Packed Schedule

Expert-ID und grouped offset teilen ein 32-Bit-Wort mit 16-Bit-Feldern. Der aktuelle `assignment_count <= 65535`-Check ist tragend und muss bei jeder neuen Schedule erhalten bleiben.

### 10.5 Workspace-Aliasing

Der Prefill verwendet absichtlich:

- dead router logits als Schedule;
- `histogram[0]` temporär als Schedule-Count;
- rückwärts laufende Output-Tiles wegen BF16-/Activation-Alias.

Diese Verträge sind leistungsrelevant, aber fragil. Nach der Strukturaufteilung sollten typisierte Views und dokumentierte Lebenszeitphasen statt wiederholter Raw-Casts verwendet werden.

### 10.6 Tail-biting

WP9s Zwei-U32-Fenster ist durch die bisherigen Oracles gut abgesichert. Vor Umbauten sollten zusätzlich alle Positionen einer 16×16-Kachel für K3 und K4 exhaustiv gegen den unabhängigen CPU-Decoder verglichen werden, einschließlich Ringgrenzen und letzter Lane-Quelle.

---

# 11. Neue Work Packages

## WP10 – Koexistenzvertrag und codegen-neutrale Strukturaufteilung

### Zweck

- NVFP4 und Trellis dauerhaft als gleichzeitige, getrennte Produktformate absichern.
- Quellstruktur vor dem M32-Umbau beherrschbar machen.
- Keine Performanceänderung.

### Haupt-Touchpoints

```text
src/cuda/engine/gemma4_26b_reference.cu
src/cuda/engine/gemma4_26b_reference.h
src/cuda/engine/gemma4_26b_trellis35_artifact.*
src/cuda/engine/gemma4_26b_artifact.*
src/cuda/moe/gemma4_26b_bindings.cpp
src/cuda/moe/reference.cu
src/cuda/moe/prefill.cu
src/cuda/trellis35/reference.cu
src/cuda/trellis35/reference.h
tests/cuda/trellis35_m1_test.cu
CMakeLists.txt
```

### Umsetzung

1. `RoutedExpertFormat` einführen.
2. Format aus validierten Metadaten bestimmen.
3. Modellkatalog/Engine-Erwartung dagegen prüfen.
4. Gemischte/mehrdeutige Verzeichnisse ablehnen.
5. `bool trellis35` schrittweise durch Enum/format-spezifische Overloads ersetzen.
6. Trellis-Datei in Include-Shards aufteilen, zunächst eine CUDA-TU.
7. Engine in Verantwortungs-Shards aufteilen.
8. Trellis-Testtarget in mehrere Dateien teilen.
9. Qualifizierte NVFP4-Dateien mathematisch unverändert lassen.

### Annahme zum Nutzen

Direkter Speedup: keiner.
Risikonutzen: hoch, weil WP12 sonst weitere mehrere hundert bis tausend Zeilen in eine bereits gemischte Datei legt.

### Acceptance

- aktuelle Trellis- und NVFP4-Hashes unverändert;
- SASS/Resource-Identität der Hotkernels;
- ≤1 % Benchmarkabweichung beziehungsweise innerhalb dokumentierter Drift;
- NVFP4 und Trellis laden getrennt;
- gemischtes Verzeichnis fail-closed;
- keine zusätzliche Device-Allokation.

---

## WP11 – Post-WP9-Instrumentierung und sichere Null-Numerik-Optimierungen

### Zweck

Vor WP12 eine belastbare neue Mikroarchitektur-Basis schaffen und unnötige Arbeit beseitigen.

### Umsetzung

1. Neues Nsight Compute auf dem **aktuellen WP9-Kernel**, nicht WP6:
   - ausgeführte Integer-/ALU-/Tensor-Instruktionen;
   - Tensor-active cycles;
   - Register;
   - lokale Loads/Stores und Spills;
   - Occupancy;
   - L1/L2/DRAM;
   - Warp-Stall-Gründe.
2. NVTX-Ranges für den gesamten D2-Chain.
3. Grid-Y des aktuellen Vierer-Schedules auf eine hostseitige sichere Obergrenze reduzieren:

```text
q = min(active_experts, assignment_count)
max_tiles = floor((assignment_count + 3*q) / 4)
```

statt `schedule_blocks = assignment_count`.

4. Trellis-spezifischen M1/T3-Input-Boundary-Prototyp bauen, der dead routed-NVFP4-Quantisierung vermeidet.
5. Einen bounded transient E4M3-Slab-Microbenchmark hinzufügen.

### Rejection/Acceptance

- Grid-Bound und Input-Boundary nur übernehmen, wenn aktuelle Trellis-Ausgaben bitidentisch bleiben.
- keine neue Allokation;
- kein NVFP4-Pfad darf anderen SASS erhalten;
- kein Kandidat darf außerhalb dokumentierten Rauschens langsamer sein;
- der Slab ist Diagnose und braucht keinen Full-Model-Speedup.

### Erforderliche Evidenz

```text
artifacts/trellis35/wp11-post-wp9-ncu.json
artifacts/trellis35/wp11-d2-range-breakdown.json
artifacts/trellis35/wp11-zero-numerics-cleanup.json
artifacts/trellis35/wp11-transient-slab-probe.json
```

---

## WP12 – Echter M16/M32 gruppierter Trellis-W4A8-Prefill

### Zweck

Höchste Wahrscheinlichkeit für einen mehrfache Prefill-Gewinn.

### Haupt-Touchpoints

```text
src/cuda/trellis35/detail/codec.cuh
src/cuda/trellis35/detail/mma_w4a8.cuh
src/cuda/trellis35/detail/prefill_kernels.inc.cuh
src/cuda/moe/prefill.cu
src/cuda/fp8/sm120.cu                 # nur als Referenz, möglichst nicht ändern
src/cuda/nvfp4/sm120.cu               # nur als Referenz, möglichst nicht ändern
tests/cuda/trellis35_prefill_test.cu
```

### Umsetzung

1. Bestehende 32er-Expert-Schedule aus `moe/prefill.cu` konsumieren.
2. Neuen `MmaW4A8ProjectionGroupedPrefillM32Kernel` hinzufügen.
3. Zwei M16-E4M3-Aktivierungstiles in Shared Memory.
4. `cp.async` Double Buffering über K32.
5. Vier Consumer-Warps, N32 pro CTA.
6. Jeder Warp dekodiert sein N8-B-Fragment einmal und verwendet es für beide M16-MMAs.
7. Tail-Zeilen nullfüllen.
8. K3/K4-Dispatch uniform je CTA außerhalb der K-Schleife.
9. Pro-Assignment-Scale beim Store korrekt auf die MMA-Fragmentzeilen anwenden.
10. Vorläufig denselben Output-Transformpfad benutzen, um nur eine Variable zu ändern.

### Kleinster Benchmark

Für Gate+Up und Down getrennt:

```text
T = 16, 128, 512, 1024
Verteilungen:
  gleichmäßig
  reale gespeicherte Routingverteilung
  stark schief
  Tail 1/2/3/15/16/17/31/32
K3-only, K4-only, mixed
```

### Engineering-Gate

Kein Produktgate, aber ein Architektur-Rejection-Gate:

- transformed projection numerisch gegen aktuellen W4A8-Oracle bestanden;
- keine Local-Memory-Spills;
- mindestens 1,5× geringere Projektionszeit bei T=512;
- kein katastrophaler Tail-Regress bei kleinen T;
- vollständiger 512-Prefill mindestens 10 % schneller.

Falls dieses Minimum nicht erreicht wird, Kernel nicht zum Default machen; NCU und Slab-Probe entscheiden über Producer-Warp oder transienten Pfad.

### Erwarteter Nutzen

Kein seriöser Tok/s-Wert vor Messung.
Richtung: sehr hoch, weil sowohl Trellis-Decodes als auch MMA-Aufrufe pro 32 nützliche Zeilen massiv sinken.

---

## WP13 – Warp-FWHT und fused Transform/Scale/Quantize

### Zweck

Die gemessenen 27,9 % Input-/Output-Transformzeit stark reduzieren.

### Umsetzung

1. Einen unabhängig getesteten `H128Warp` schreiben.
2. Float- und BF16-Input als Template/Policy.
3. Ein CTA pro Assignment:
   - transformieren;
   - Shared speichern;
   - amax;
   - Scale;
   - E4M3 quantisieren.
4. Direkte 128-FMA-Helfer aus dem Prefill-Hotpath entfernen.
5. Schnellen inversen H128-Kernel mit Warp pro Assignment/H128-Block.
6. Physische BF16-Rundpunkte unverändert halten.
7. M1/T3 erst nach erfolgreichem Prefill optional auf denselben Helfer migrieren.

### Gate

- exhaustive H128-Oracle;
- vorab festgelegte Fehlergrenzen;
- keine Spills;
- aggregierte Transformzeit mindestens 2× niedriger;
- Full-Prefill messbar schneller;
- kleine WP8A-Numerik wiederholen, nicht WP8B.

---

## WP14 – Decoder/MMA-Pipeline und Launch-/Workspace-Konsolidierung

### Zweck

Nach WP12/WP13 die verbleibenden Projektionskosten anhand frischer NCU-Daten angreifen.

### Kandidat A: Producer/Consumer-Warps

Wenn Integer-/Decode-Instruktionen weiterhin dominieren:

```text
Producer-Warp(s):
  Trellis K3/K4 -> packed E4M3 B tile in Shared

Consumer-Warps:
  zwei M16-A-Tiles -> FP8 MMA
```

B doppelt puffern, Decoder und MMA überlappen.

### Kandidat B: transienter N128-Slab

Nur wenn WP11 zeigt, dass einmalige Dekodierung plus globaler E4M3-Traffic gewinnt.

### Kandidat C: Output-/Launch-Konsolidierung

Aktuell entstehen pro 30-Layer-Prefill:

```text
Gate+Up: 11 N128-Blöcke
Down:    22 N128-Blöcke
30 × 33 = 990 Projektionsstarts
und 990 inverse Transformstarts
```

Mögliche Folgearbeit:

- Output-Block in Grid-Dimension statt Hostschleife;
- größerer, aber weiterhin bounded Workspace;
- oder N128-CTA mit Shared-Epilog, falls Register/Occupancy es erlauben;
- Gate+Up inverse + explizite BF16-Rundung + GELU fusionieren;
- Down inverse + BF16-Rundung, später eventuell Reduction fusionieren.

### Gate

Jeder Kandidat separat A/B:

- kein zusätzlicher persistenter Weight-Speicher;
- Workspace-Kosten und Kontextverlust exakt berichten;
- gleiche physische BF16-Grenzen;
- keine Spills/Deadlocks;
- Kandidat nur behalten, wenn End-to-End-Prefill und nicht nur Microkernel gewinnt.

---

## WP15 – Fixed-D2- und Ordinary-Recovery

### Zweck

Trellis-Decode nahe an die vorhandene NVFP4-User-Performance bringen, ohne Semantikänderung.

### Umsetzung

1. Canonical timinggleiche Performanceverteilung erstellen.
2. NVTX-D2-Aufschlüsselung aus WP11 auswerten.
3. Trellis-spezifische Input Boundary übernehmen, falls exakt.
4. FWHT aus WP13 auf M1/T3 übertragen.
5. Optional T3 als eine echte M16-A-Matrix pro Unique Expert ausführen.
6. Route-overlap-Fälle 0/typisch/maximal messen.
7. Nur profilergetrieben weitere Fusionen:
   - Expert Output/Reduction;
   - Head-/Attention-Grenzen;
   - KV-Transaktionen.
8. Initial Selection und post-first getrennt ausweisen.

### Harte Korrektheitsgrenzen

- Ordinary Target tokenidentisch;
- Fixed-D2 Target-Verifikation tokenidentisch;
- gleiche accepted/rejected drafts;
- gleiche Commit/Rollback-Grenzen;
- keine nicht-finiten Werte;
- kein Fallback;
- keine Token-Loop-Allokation.

### Performance

Noch kein Produkt-Promotionsgate. Berichten:

```text
reiner T3-Verifier
post-first D2
gesamte Anfrage inkl. Initial Selection
Ordinary
```

---

## WP16 – Architekturentscheidung W4A8 versus Trellis→W4A4

WP16 ist zunächst ein Entscheidungsdokument, kein automatischer Implementierungsauftrag.

### W4A8 weiter optimieren, wenn

- Decoder-/Schedule-/Transformanteile weiterhin dominieren;
- Tensor-Pipe noch deutlich unter Sättigung liegt;
- echter M32-Pfad weiter skaliert;
- W4A8-Qualitätsvorteil relevant erscheint.

### Trellis→W4A4 untersuchen, wenn

- W4A8 nach M32/FWHT/Decoder-Pipeline compute-limitiert ist;
- Tensor-Pipe hoch ausgelastet ist;
- bounded Slab/decoderfreier W4A8 eine klare Compute-Decke zeigt;
- der verbleibende Abstand die zusätzliche Quantisierungs-/Scale-Komplexität rechtfertigt.

W4A4 darf nicht als bloßes Runtime-Runden beliebiger Trelliswerte implementiert werden. Es braucht einen klaren Vertrag für:

```text
E2M1-Werte
E4M3 K/16-Skalen
zusätzliche Sidecar-Bytes
Qualitätsfehler
Decoderkosten
```

Weiterhin nur eine persistente Expert-Repräsentation.

---

## 12. Empfohlene tatsächliche Reihenfolge

```text
WP10  Struktur + explizite Koexistenz
  ↓
WP11  frische Messung + dead work + Slab-Probe
  ↓
WP12  echter M16/M32 Prefill
  ↓
WP13  Warp-FWHT
  ↓
WP14  nur aus WP12/13-Profil: Producer/Consumer, Slab oder Launch-Fusion
  ↓
WP15  D2/Ordinary Recovery
  ↓
WP16  W4A8 weiter oder W4A4-Forschung
```

**Erster Performance-Kandidat:** WP12.
**Erster Gesamt-Arbeitsschritt:** WP10, weil WP12 andernfalls die falsche Datei weiter aufbläht.

WP11-Instrumentierung kann teilweise parallel vorbereitet werden, sollte aber auf dem strukturell aufgeteilten, codegen-identischen Stand laufen.

---

## 13. Ideen, die jetzt nicht verfolgt werden sollten

1. W4A16 als Zwischenpfad.
2. Trellis→W4A4, bevor echter M16/M32-W4A8 und FWHT vermessen sind.
3. Vollständige persistente E4M3- oder NVFP4-Expert-Zweitkopie.
4. TMA als alleinige „Optimierung“; aktuelle Evidenz zeigt primär Instruktions-/Granularitätsprobleme.
5. Erneuter Rows=8-Versuch mit acht separaten M1-Akkumulatoren.
6. Exact-704-Down-Umbau.
7. Vision/MMProject.
8. die lange WP8B-Qualitätssuite.
9. breite Aufspaltung/Neuqualifikation des produktiven NVFP4-Kernels.
10. Änderung von FP8-Skalen, BF16-Rundpunkten, Prompt, Sampling oder Timinggrenzen.
11. Runtime-JIT oder generischer Autotuner.
12. polymorphe virtuelle Calls im Token-Hotpath.
13. Optimierung des seriellen Schedule-Builders vor dem 63,3-%-Projektionsblock.

---

## 14. Fehlende Instrumentierung

Vor einer belastbaren nächsten Architekturentscheidung fehlen im Archiv:

- Nsight Compute des **post-WP9**-Prefill-Projektionskernels;
- Warp-Stall-Verteilung;
- aktuelle Integer-/ALU-/Tensor-Instruktionszahlen;
- aktuelle Register-/Spill-/Occupancy-Daten für Prefill;
- per-layer/per-family Gate+Up-versus-Down-Zeit;
- Anzahl echter Schedule-Tiles gegenüber gestarteten Y-Blöcken;
- Histogramm `rows per expert tile`;
- Canonical D2-NVTX-Chain;
- timinggleiche Trellis/NVFP4-D2-Verteilung;
- transienter E4M3-Slab als Decoder-vs-MMA-Kontrolle.

Diese Messungen gehören in WP11 und sollten nicht durch Schätzwerte ersetzt werden.

---

## 15. Schlussentscheidung

### Refactor zuerst?

**Ja, ein einziges enges, codegen-neutrales Paket.** Danach sofort wieder Performancearbeit.

### Hat Trellis NVFP4 ersetzt?

**Nein.** Beide Pfade sind vorhanden und pro Engine-Instanz exklusiv. Der Vertrag sollte dennoch expliziter und fail-closed werden.

### Größter nächster Performancehebel?

**Echter M16/M32 gruppierter W4A8-Prefill**, unter Wiederverwendung der bereits vorhandenen 32er-Expert-Schedule. Das ist deutlich aussichtsreicher als weitere Änderungen am Trellis-State-Decoder oder ein erneuter größerer Array-von-M1-Akkumulatoren-Versuch.

### Ist 951 tok/s bereits nahe an der möglichen W4A8-Grenze?

Dafür gibt es keinen Beleg. Der aktuelle Kernel verwendet die M16-MMA überwiegend als M1-Emulation und wiederholt Decode/Compute über Vierergruppen. Die vorhandene Architektur lässt deshalb noch einen großen, konkret identifizierbaren Amortisierungsschritt offen.

### WP8B jetzt?

**Nein.** Nur kleine, proportionale Numerik-/Qualitätsgates nach Änderungen. Die lange Suite bleibt eingefroren, bis die Performance deutlich besser und die nächste Architekturentscheidung getroffen ist.

# Gem16 Trellis35 nach WP27 – Performance-Closeout und Vision-Review

**Review-Basis:** `codex/gemma4-26b-trellis35-perf2@540dd005f8ac6fcf729aec66d1d224af9098a301`
**Datum:** 2026-08-31
**Review-Art:** statische Quellcodeanalyse plus Auswertung der im Archiv enthaltenen WP17–WP27-, NSYS-, NCU-, SASS-, Sanitizer- und Full-Path-Evidenz
**Nicht durchgeführt:** eigene Ausführung auf der RTX 5080; lange WP8B-Qualitätssuite

---

## 1. Entscheidung in einem Satz

**Trellis35 ist mit rund 5.690 Prefill-tok/s und 182,63 Fixed-D2-tok/s weit genug, um nach zwei kurzen, klar begrenzten Rest-Screens die Performancearbeit einzufrieren und mit einer separaten 26B-Vision-FP8-Integration zu beginnen.**

Die beiden offenen Post-WP22-Residuals sollten noch geprüft werden:

1. ein konfliktfreies Shared-Projection-Layout für Prefill – **ja, hoher Nutzen bei kleinem numerischem Risiko**;
2. isolierter T3-Routenaufbau ohne thread-lokales `assignments[3]` – **ja, aber nur als kleiner Decode-Screen**.

Daneben ist ein dritter, bislang nicht separat geschlossener Low-Risk-Screen sinnvoll:

3. vektorisierter 128-Bit-Output-Store in den aktuellen T3-/M1-N128-Inverse-Kernels.

Danach sollte nur noch ein bedingter Tasklist-/Persistent-Grid-Versuch folgen, falls ein frisches NCU-Profil weiterhin Duplicate-CTA-/Partial-Wave-Kosten zeigt. Werden mit zwei aufeinanderfolgenden profilerbegründeten Kandidaten zusammen weniger als ungefähr 1 % Full-Path-Gewinn erzielt, ist der richtige nächste Schritt Vision statt weiterer Trellis-Mikrooptimierung.

---

## 2. Aktueller belastbarer Stand

### 2.1 Full Wikipedia 16K, Sampled, 1W3

Aktuelle akzeptierte WP27-Messung:

| Grenze | Trellis35 |
|---|---:|
| Ordinary Prefill | **5.692,23 tok/s** |
| Fixed-D2 Prefill | **5.689,20 tok/s** |
| Ordinary Decode | **130,45 tok/s** |
| Fixed-D2 Decode | **182,63 tok/s** |

Messvertrag:

```text
Prompt:                  16.384 Tokens
Generierung:             1.229 Tokens
Sampling:                temperature 1.0
                         top-k 64
                         top-p 0.95
                         seed 0
Warm-ups:                1 Paar
Retained:                3 Paare
Ordinary == D2 Output:   ja
D2 accepted/proposed:    690 / 1.076
D2 rejected:             386
D2 Target-Gruppen:       538
Mean accepted length:    1,28253
Ordinary-Fallback:       0
```

Die aktuelle T3-N128-Fusion ist bitgenau gegen die Rollbacks, ändert weder Ordinary noch Prefill und hat keine zusätzliche Workspace- oder persistente Weight-Repräsentation eingeführt.

### 2.2 Abstand zur NVFP4-Referenz

| Grenze | Trellis35 | NVFP4 | Retention | Abstand |
|---|---:|---:|---:|---:|
| Prefill | 5.689,20 | 6.965,63 | 81,67 % | −18,33 % |
| Ordinary Decode | 130,45 | 148,44 | 87,88 % | −12,12 % |
| Fixed-D2 Decode | 182,63 | 203,83 | 89,60 % | −10,40 % |

Der Pfad ist damit nicht gleich schnell wie NVFP4, besitzt aber bereits ein überzeugendes eigenes Produktprofil:

```text
NVFP4:
  maximale Geschwindigkeit

Trellis35:
  2,3208 GiB weniger Target-Weight-VRAM
  deutlich längerer möglicher Kontext
  knapp 82 % Prefill- und knapp 90 % D2-Retention
```

### 2.3 Speicher

```text
Trellis-Payload:        exakt 3,5 bpw
Target-Arena:           12.204.692.480 Byte
                        11,3665 GiB
Ersparnis zu NVFP4:      2.491.975.680 Byte
                         2,3208 GiB
```

Es gibt genau eine persistente routed-expert-Repräsentation. Kein NVFP4-Expert-Duplikat, kein CPU-Offload, kein Expert-Streaming, kein Token-Loop-Repack und keine wiederkehrende Token-Loop-Allokation.

### 2.4 Aktuelle Codeorganisation

Das frühere Strukturproblem ist weitgehend gelöst:

```text
src/cuda/trellis35/reference.cu                       46 Zeilen
src/cuda/trellis35/detail/prefill_kernels.inc.cuh  1.157
src/cuda/trellis35/detail/launchers.inc.cuh         1.357
src/cuda/trellis35/detail/t3_kernels.inc.cuh          340
src/cuda/trellis35/detail/m1_kernels.inc.cuh          119
```

Auch die 26B-Engine wurde in Verantwortungs-Shards aufgeteilt. Ein weiterer allgemeiner Struktur-Refactor ist vor Vision nicht erforderlich.

---

## 3. Was WP17 bis WP27 bereits geschlossen haben

Diese Kandidaten dürfen nicht in gleicher Form erneut implementiert werden:

### Prefill

- echter M64-N128-W4A8-Pfad;
- dynamisches M32-Tail-Skipping und engerer Launch-Bound;
- 2.048-Token-Trellis-Prefill-Chunks;
- Warp-H128;
- fused N128 Projection + inverse H128;
- Gated-GELU direkt in Down-Transform/Quantisierung.

### Explizit ausprobiert und verworfen

- Shared Activation-Scale-Staging;
- Shared SVH-Staging in der getesteten Variante;
- pair-shared Trellis-Payload in der getesteten Variante;
- gepackte FP8-Konvertierung im Prefill;
- vectorized Prefill-Projection-I/O in der getesteten Variante;
- K64-Pipeline in der getesteten Variante;
- kombinierte T3-Tasklist plus vierstufiges Prefetch;
- transienter E4M3-Slab als Primärpfad.

### Decode

- Shared-NVFP4 und Routed-Trellis bei T3 überlappen;
- M1-N128 Projection + inverse H128;
- native gepackte FP8x4-Konvertierung für M1/T3;
- T3-M16-N128 Projection + inverse H128.

Die neue Arbeit muss deshalb eng auf die tatsächlich noch offenen Restkosten zielen.

---

# 4. Audit der zwei offenen Post-WP22-Residuals

## 4.1 Residual A: konfliktfreier Shared-Projection-Swizzle

### Urteil

**Ja, diesen Screen sollte Codex noch durchführen.**

Er ist heute besser begründet als nach WP22, weil der aktuelle T3-N128-Kernel direkte NCU-Evidenz liefert:

```text
Shared Loads:
  durchschnittlich 4-facher Bankkonflikt
  3.214 Konflikte
  75,71 % der Shared-Load-Wavefronts betroffen

Shared Stores:
  durchschnittlich 1,5-facher Bankkonflikt
  2.872 Konflikte
  33,81 % der Shared-Store-Wavefronts betroffen
```

Beim M64-Prefill zeigten die WP17-Rohprofile ebenfalls erhebliche Shared-Konflikte – unter anderem ungefähr 2-fache Load- und deutlich höhere Store-Konflikte im Projection/H128-Epilog.

### Ursache im Quellcode

Prefill M32/M64 und T3 N128 schreiben zunächst die transformierten MMA-Ergebnisse nach:

```cpp
projection[row][logical_column]
```

beziehungsweise:

```cpp
transformed[row][logical_column]
```

Danach lädt ein Warp für H128:

```cpp
logical_column = lane * 4 + element;
```

Bei einem normalen row-major Float-Array landen viele Lanes derselben Skalar-Load-Instruktion auf derselben der 32 Shared-Banks.

### Kandidatenlayout

Ein möglicher bijektiver XOR-Swizzle ohne zusätzlichen Shared-Speicher:

```cpp
__device__ __forceinline__
unsigned Trellis35SharedRowMask(unsigned row) {
  const unsigned group = row & 7U;
  return (group & 1U) |
         ((group & 2U) << 2U) |
         ((group & 4U) << 2U);
}

__device__ __forceinline__
unsigned Trellis35SharedProjectionColumn(
    unsigned row, unsigned logical_column) {
  return logical_column ^
         (logical_column >> 2U) ^
         Trellis35SharedRowMask(row);
}
```

Die statische Adressanalyse für die aktuell verwendeten Zugriffsmuster ergibt:

- für jede Row eine Permutation von 0…127;
- konfliktfreie vier H128-Load-Instruktionen `lane*4 + element`;
- konfliktfreie aktuellen MMA-Store-Muster für M32 und M64;
- keinen zusätzlichen Shared-Memory-Bedarf;
- keine Änderung der logischen Reihenfolge, da Store und Load denselben Mapper verwenden.

Das ist noch **keine GPU-Messung**. Zusätzliche Integer-Adressarithmetik oder ein ungünstiges SASS-Lowering können den theoretischen Vorteil verkleinern oder umkehren.

### Wo testen?

Zuerst Prefill:

```text
src/cuda/trellis35/detail/prefill_kernels.inc.cuh
```

Betroffene Arrays:

```text
M32N128 projection[32][128]
M64N128 projection[64][128]
```

Danach separat T3:

```text
src/cuda/trellis35/detail/t3_kernels.inc.cuh
```

Betroffen:

```text
transformed[3][128]
```

M1 ist optional und separat:

```text
src/cuda/trellis35/detail/m1_kernels.inc.cuh
transformed[128]
```

Für M1 reicht die zeilenunabhängige Variante:

```cpp
physical = logical ^ (logical >> 2U);
```

### Erwarteter Bereich – Schätzung, kein Messwert

| Grenze | grob plausibler Full-Path-Bereich |
|---|---:|
| Prefill | +0,5 bis +2,5 % |
| Fixed-D2 | +0,8 bis +3,0 % |
| Ordinary | +0,2 bis +1,0 % |

Der T3-N128-Kernel ist mit 117,79 µs bereits klein. Selbst ein deutlicher Kernelgewinn übersetzt sich deshalb nicht eins zu eins in D2-Token/s.

### Rejection-Gate

Verwerfen, wenn:

- ein einziger Bit-Unterschied entsteht;
- Register, Stack oder Shared steigen;
- Local-Memory-Zugriffe entstehen;
- Shared-Konflikte nicht materiell fallen;
- der adressierte Kernel nicht mindestens 3 % gewinnt;
- oder der vollständige Pfad außerhalb des Messrauschens verliert.

---

## 4.2 Residual B: isolierter T3-Routenaufbau ohne `assignments[3]`

### Urteil

**Ja, ebenfalls noch testen – aber isoliert und ohne das zuvor verworfene Prefetch-Paket.**

WP20 hat nicht bewiesen, dass ein besserer Routenaufbau nutzlos ist. Verworfen wurde die Kombination:

```text
Tasklist
+
vierstufiges Prefetch
```

weil sie:

- 14,34 % mehr Instruktionen;
- 39 → 47 Register;
- geringere Occupancy;
- und 7–12 % höhere Latenz

erzeugte.

Der aktuelle WP27-Kernel enthält weiterhin:

```cpp
unsigned assignments[3]{};
```

mit dynamischer Indexierung. NCU meldet:

```text
Stack:              16 Byte
Local-Memory-Anteil: 2,34 % der L1-Sektoren
```

Zusätzlich bauen alle Warps und alle N128-Ausgabe-CTAs dieselbe kleine Route erneut auf.

### Kandidat A: drei skalare Assignment-Register

Die kompakten Assignment-IDs werden direkt in:

```cpp
assignment0
assignment1
assignment2
assignment_count
```

gespeichert.

Der per-Lane-Zugriff verwendet `select`/kleine feste Verzweigungen, kein Array.

Vorteil:

- kleinste Änderung;
- kein neuer Block-Sync;
- vermutlich Stack/Local = 0.

Risiko:

- zusätzliche Live-Register;
- ein Registersprung kann bei 512 Threads die Blocks/SM reduzieren.

Aktueller WP27-Stand:

```text
40 Register/Thread
3-Block-Registergrenze laut NCU
```

Konservatives Gate:

```text
<=42 Register/Thread
```

falls dadurch dieselbe Blockresidenz erhalten bleibt. Maßgeblich ist ptxas/NCU, nicht die Schätzung.

### Kandidat B: Warp-Ballot-Route

Jede Warp-Lane 0…23 lädt genau einen der 24 `selected_experts`:

```cpp
candidate = lane < 24 ? selected_experts[lane] : invalid;
expert = shfl(candidate, group_candidate);
match_mask = ballot(candidate == expert) & 0x00ffffff;
```

Dann:

- frühere Bits erkennen Duplicate Candidate;
- je 8-Slot-Row erstes Match bestimmen;
- bis zu drei Assignments als skalare Werte packen;
- kein 3×8-Scan;
- kein lokales Array.

Alle Lanes eines Warps erhalten denselben Maskenwert.

### Kandidat C: Route einmal pro CTA

Warp 0 erzeugt drei IDs und Count in einem kleinen Shared-Descriptor, anschließend ein Block-Sync.

Das verhindert, dass 16 Warps dieselbe Route wiederholt bauen. Der Preis ist eine zusätzliche Barriere vor der Projektion.

A, B und C dürfen nicht gleichzeitig implementiert werden. Erst A/B vergleichen; C nur, wenn NCU weiterhin übermäßige Route-Instruktionen zeigt.

### Erwarteter Bereich – Schätzung

```text
T3 Kernel:       +1 bis +5 %
Full Fixed-D2:   +0,2 bis +1,0 %
```

### Rejection-Gate

- Stack und Local müssen auf null fallen;
- Register dürfen die Blockresidenz nicht reduzieren;
- Target-/Draft-Trajektorie muss exakt bleiben;
- Same-Binary-T3 muss mindestens 2 % gewinnen;
- Full-D2 mindestens 0,3 % gewinnen;
- sonst komplett zurückrollen.

---

# 5. Ein zusätzlicher Low-Risk-Screen: 128-Bit-Output-Stores

## Befund

Der aktuelle T3-N128-SASS schreibt die vier zusammenhängenden Float-Werte je Lane als vier einzelne Stores:

```text
STG +0
STG +4
STG +8
STG +12
```

NCU meldet bei globalen Stores durchschnittlich nur:

```text
8 von 32 Byte pro Sektor
```

Die Adressen sind für die aktuellen Shapes 16-Byte-ausgerichtet:

```text
Output-Strides:    1.408 und 2.816 Floats
                   beide durch 4 teilbar

Output-Block:      128 Floats
Lane-Index:        lane × 4 Floats
Device-Arena:      mindestens 256-Byte-ausgerichtet
```

## Kandidat

Nach H128 und SVH:

```cpp
float4 result = make_float4(...);
*reinterpret_cast<float4*>(assignment_output) = result;
```

oder ein explizites 128-Bit-Store-Intrinsic/Inline-PTX, falls der C++-Store nicht als `STG.128` gesenkt wird.

Separat testen für:

```text
T3 N128
M1 N128
```

Dies ist **nicht** die bereits verworfene WP18-Prefill-Vector-I/O-Variante; sie betrifft einen anderen Kernel und wird durch die aktuelle WP27-SASS-/NCU-Evidenz direkt motiviert.

## Gate

- SASS muss einen 128-Bit-Store oder eine gleichwertige Verringerung der Store-Instruktionen zeigen;
- keine Misalignment-/Sanitizer-Probleme;
- exakte Outputs;
- Kernel mindestens 1 % schneller;
- Full-D2/Ordinary mindestens neutral;
- nur behalten, wenn Full-Path mindestens +0,2 % oder ein klarer kombinierbarer Kernelgewinn entsteht.

---

# 6. Bedingter mittlerer Kandidat: kompakte Taskliste + Persistent Grid

## Warum nicht sofort?

Der aktuelle T3-Grid ist:

```text
Gate+Up: 11 × 24 = 264 CTAs
Down:    22 × 24 = 528 CTAs
Block:   512 Threads
```

Viele der 24 `group_candidate`-Einträge sind Duplicate Experts und kehren nach dem Prior-Scan zurück. In realen T3-Trajektorien liegt die Unique-Expert-Union typischerweise unter 24.

Ein kleiner Builder könnte einmal pro Layer eine kompakte Liste erzeugen, die Gate+Up und Down wiederverwenden. Ein fixer, CUDA-Graph-kompatibler Grid-Stride-Kernel könnte anschließend:

```text
task × output_block
```

abarbeiten.

Aber WP20 zeigt, dass zusätzliche Builder-/Prefetch-Instruktionen schnell teurer werden als die eingesparte Arbeit. Dieser Kandidat ist daher nur zulässig, wenn nach Swizzle und isoliertem Routenaufbau ein frisches NCU/NSYS weiterhin zeigt:

- hohe Duplicate-CTA-Zahl;
- Partial-Wave-/Load-Imbalance;
- relevante Route-/Control-Instruktionszeit;
- genügend große Arbeit pro Task.

Kein Host-Sync, keine Atomic-Workqueue und kein Prefetch im ersten isolierten Versuch.

Geschätzter Full-D2-Bereich:

```text
+0,5 bis +2,5 %
```

mit deutlich höherem Risiko als die drei vorherigen Screens.

---

# 7. Prefill jetzt weiter optimieren oder Vision beginnen?

## Empfehlung

**Noch genau einen kurzen Prefill-Screen durchführen: den konfliktfreien Shared-Projection-Swizzle.**

Der aktuelle Prefill ist bereits:

```text
5.689 tok/s
= 81,7 % der NVFP4-Referenz
```

und der Trellis-Pfad bietet dafür 2,32 GiB weniger Weight-VRAM.

Das ist bereits ein überzeugender Trade-off. Die großen Prefill-Hebel wurden geschlossen:

- M64;
- 2.048er-Chunk;
- Warp-H128;
- fused N128;
- GELU/Down-Fusion;
- Schedule-Tail-Trim.

Ein weiterer größerer Umbau wie 4.096er-Chunk, M128, TMA oder W4A4 ist vor Vision nicht risikoadjustiert sinnvoll.

### 4.096er-Chunk/M128

Nicht jetzt.

WP21 hat für 1.024 → 2.048 bereits:

```text
+142 MiB Peak-/Workspace-Kosten
```

in Kauf genommen. Ein weiterer großer Chunk würde genau den VRAM wieder verbrauchen, den Vision benötigt. Die aktuelle 2.048er-Geometrie ist ein guter Kompromiss zwischen Prefill-Amortisierung und Vision-Reserve.

### W4A4

Weiterhin nicht zugelassen.

Die aktuelle Performance zeigt keinen belastbaren Beleg, dass die W4A8-Tensor-Pipe der dominante Deckel ist. Offene Kosten existieren weiterhin bei Shared-Memory, Routing, Load/Store und Graph-Auslastung.

---

# 8. Empfohlene Performance-Closeout-Regel

Noch folgende Reihenfolge:

```text
PFX28:
  Shared-Projection-Swizzle
  + separat 128-Bit-Output-Stores

PFX29:
  isolierter T3-Routenaufbau ohne assignments[3]

PFX30:
  nur bei klarer NCU-Indikation:
  kompakte Taskliste + Persistent Grid
```

Dann stoppen, sobald eine der Bedingungen erfüllt ist:

1. zwei aufeinanderfolgende profilerbegründete Kandidaten liefern jeweils weniger als 0,5 % Full-Path-Gewinn;
2. die gesamte Closeout-Welle liefert weniger als 1 % kumulativ;
3. ein Kandidat benötigt neue persistente Daten, zusätzlichen großen Workspace oder Numerikänderung;
4. Prefill und D2 erreichen ungefähr:

```text
Prefill >= 5.750 tok/s
Fixed-D2 >= 185 tok/s
```

Die Zahlen sind keine Produktgates, sondern ein sinnvoller Freeze-Punkt. Bei 5.700/182 darf Vision auch beginnen, wenn die Screens nicht gewinnen.

---

# 9. Ist das Repository bereit für 26B-Vision?

## Ja – als neuer separater experimenteller Vertical Slice

Der Quellcode kennt die exakte 26B-Vision-Konfiguration bereits:

```text
Modelltyp:                    gemma4_vision
Hidden:                       1.152
Intermediate:                 4.304
Layer:                        27
Q-Heads / KV-Heads:           16 / 16
Head-Dimension:               72
Max Positionen:               131.072
2D Position-Embedding-Größe:  10.240
Soft Tokens pro Bild:         280
Patch:                        16 × 16 × 3 = 768
Pooling:                      3 × 3
Standardisierung:             ja
RMSNorm epsilon:              1e-6
```

Die Runtime verweigert 26B-Vision derzeit jedoch absichtlich:

```text
ModelVariantTraits.supports_vision = false
```

und der Compiler klassifiziert alle 356 Vision-Tensoren als:

```text
compile_excluded_vision
```

Das ist ein sauberer Ausgangspunkt: Vision ist nicht halb implementiert, sondern explizit ausgeschlossen.

## Bestehende 12B-Vision ist nicht direkt wiederverwendbar

Die aktuelle öffentliche Struktur:

```cpp
VisionEmbeddingSegment
```

erwartet:

```text
[patch, 6912]
= 48 × 48 × 3 pro bereits zusammengefasstem Patch
maximal 280 Patch-/Prompt-Tokens
```

Die 26B-Vision-Architektur benötigt dagegen:

```text
[raw_patch, 768]
= 16 × 16 × 3
bis zu ungefähr 280 × 9 = 2.520 Roh-Patches
3×3 Pooling
danach 280 Soft-Tokens für den Textprompt
```

Bei 26B sind Eingangs-Patch-Anzahl und Prompt-Soft-Token-Anzahl also nicht identisch. Die bestehende Struktur darf nicht still umgedeutet werden.

Wiederverwendbar sind:

- sichere Bilddekodierung und Resize-Grundlagen;
- Server-/Chat-Medienplumbing;
- Placeholder-/Boundary-Token-Prüfung;
- bidirektionale Attention-Semantik für den Bildspan;
- Session-/Cancellation-/Error-Hüllen.

Neu nötig sind:

- 16×16-26B-Patchifier;
- separater Raw-Patch- und Soft-Token-Vertrag;
- 27-Layer-Vision-Tower;
- 2D-Positionsembedding/RoPE;
- 3×3-Pooling;
- Projektor 1.152 → 2.816.

---

# 10. Empfohlenes Vision-Weight-Format

## Erster Kandidat: FP8-Weights, BF16-Nonlinears

Nicht Trellis, nicht BF16 komplett und nicht sofort Q4.

Große 2D-Linears:

```text
Patch Projection
27 × (Q, K, V, O)
27 × (Gate, Up, Down)
Final Vision→Text Projector
```

Zusammen:

```text
191 Linear-Tensoren
549.070.848 Weight-Parameter
```

Empfehlung:

```text
Weights:        FP8 E4M3, per-output-channel BF16 Scale
Activations:    zunächst BF16 an den logischen Layergrenzen
Accumulation:   FP32
Position table: BF16
Norms:          BF16
std_bias/scale: BF16
```

## Statische Speicherabschätzung

Aktuelle BF16-Vision-Quelle:

```text
1.145.588.832 Byte
1.092,52 MiB
```

Statisches FP8-Inventar mit:

- U8 für die 549.070.848 großen Linear-Gewichte;
- BF16-Scale je Output-Row;
- allen übrigen 165 Tensoren in BF16;
- 256-Byte-Ausrichtung pro Komponente;

ergibt ohne zusätzliche physische Padding-Tensoren ungefähr:

```text
597.313.024 Byte
569,64 MiB
```

Geschätzte Ersparnis:

```text
548.275.808 Byte
522,88 MiB
```

Das ist eine belastbare statische Inventarrechnung, aber noch keine reale Artifact-/Peak-VRAM-Messung. Der endgültige Wert hängt von Padding, Device-Layout, Descriptoren und Workspace ab.

## Shape-Entscheidungen vor dem Compiler

### Attention Head 72

Die Vision-Attention hat 72 Werte pro Head. Ein neuer Attention-Kernel muss explizit entscheiden:

```text
physisch auf 80/96 auffüllen
oder
64 + 8 Tail behandeln
```

Die Entscheidung hängt vom BF16-/FP8-MMA-Pfad ab und darf nicht still durch den Loader erfolgen.

### MLP Intermediate 4304

Gate/Up Output = 4.304 ist durch 8/16 teilbar, Down contracting K=4.304 aber nicht durch 32.

Mögliche physische Verträge:

```text
4304 + K32-Tail
4304 -> 4320
4304 -> 4352 bei 128er Layout
```

Das muss in V00 anhand des gewählten FP8-Kernels und der Speicher-/Leistungskosten entschieden werden.

---

# 11. Vision als Sidecar statt Duplikat

Empfohlenes Packaging:

```text
Target:
  NVFP4 oder Trellis35 Text-Artifact

Assistant:
  optionales Fixed-D2-Artifact

Vision:
  separates immutable vision.gem16 Sidecar
```

Vorteile:

- kein doppeltes Vision-Artifact pro Textquantisierung;
- Vision kann unabhängig neu quantisiert/qualifiziert werden;
- ein und dasselbe geprüfte Sidecar kann technisch mit NVFP4 und Trellis35 gekoppelt werden;
- der Produktkatalog kann nur tatsächlich speicherqualifizierte Kombinationen freigeben.

Das Sidecar bleibt vollständig GPU-resident, wenn der multimodale Profile geladen ist. Es ist keine CPU-Offload-Lösung.

Die Runtime benötigt einen expliziten Komponentenvertrag:

```text
Target Lock
optional Assistant Lock
optional Vision Lock
Kompatibilitäts-/Source-Revision
Capability Matrix
```

Die bloße Existenz einer Vision-Datei darf `supports_vision` nicht aktivieren.

---

# 12. Speicherrealität für Vision

WP21 hat gemessen:

```text
Target-only bei 206.848 Tokens:
  1,413 GiB frei

Target-only bei 262.144 Tokens:
  0,833 GiB frei

Fixed-D2 bei 86.016 Tokens:
  ungefähr 2,4 GiB frei
```

Mit geschätzt rund 570 MiB Vision-Weights:

- 16K/32K multimodal sollte speicherseitig plausibel sein;
- Fixed-D2 86K hat wahrscheinlich ausreichend Weight-/Workspace-Reserve;
- Target-only 206K könnte noch möglich sein, muss aber Vision-Workspace einpreisen;
- 262K plus Vision dürfte voraussichtlich zu knapp sein.

Das ist eine Inferenz aus den bestehenden Capacity-Punkten, keine Qualifikation.

---

# 13. Empfohlene Vision-Work-Packets

## V00 – Produkt-, Format- und Semantikvertrag

- neue experimentelle 26B-Vision-Owner-Entscheidung;
- gepinnte semantische Referenz;
- expliziter Vision-Sidecar-Lock;
- Capability nicht aus Dateiexistenz ableiten;
- Head-72- und MLP-4304-Physical-Contract festlegen;
- ein Bild, Batch 1, höchstens 280 Soft-Tokens;
- kein Audio/Video;
- Text-Only-Profile bleiben vollständig erhalten.

## V01 – Compiler und Artifact

- die 356 Vision-Tensoren aus `compile_excluded_vision` in ein neues Vision-Profil überführen;
- große Linears FP8;
- Position Embeddings/Norms/Standardisierung BF16;
- zwei byte-identische Clean Builds;
- exakte Byte- und Paddingbilanz;
- kein Runtime-Repack.

## V02 – Loader, Residency und Capacity

- Vision-Sidecar streng validieren;
- eine persistente Repräsentation;
- feste Device-Arena/Offsets;
- Preflight für Text-only, Text+Vision, Text+Vision+D2;
- 32K/64K/86K sowie erste Ablehnung veröffentlichen.

## V03 – 26B-Image-Preprocessing und Patch Embedding

- neue 16×16-RGB-Patchstruktur;
- Roh-Patch-Anzahl getrennt von 280 Prompt-Soft-Tokens;
- Standardisierung;
- FP8 Patch Projection 768→1.152;
- 2D Position Embedding;
- CPU-/PyTorch-Oracle.

## V04 – Vision Attention

- 16 Heads × 72;
- Q-/K-Norm;
- 2D RoPE;
- bidirektionale Full Attention;
- online Softmax, kein kompletter 2.520×2.520-Score-Slab;
- FP8 Q/K/V/O Weight-Linears;
- BF16/FP32 numerische Grenzen explizit.

## V05 – Vision MLP und Layer

- GELU Gate×Up;
- 1.152→4.304→1.152;
- FP8 Linears;
- Norm-/Residualfolge gemäß gepinnter Referenz;
- Shape-Tails/Padding qualifizieren.

## V06 – 27 Layer, 3×3 Pooling und Projektor

- kompletter Vision-Tower;
- Pooling von Roh-Patches auf maximal 280 Soft-Tokens;
- Standardisierung/Pooler;
- FP8 Projektor 1.152→2.816;
- direkte Ausgabe in Text-Hidden-Format.

## V07 – Textintegration

- Bildhidden an den 280 `<|image|>`-Positionen einsetzen;
- Bildspan bidirektional, restliche Textsemantik unverändert;
- Ordinary und D2 nach dem Bild normal fortsetzen;
- Cache, Streaming, Cancellation und Session-Continuation.

## V08 – begrenzte Qualifikation

- Artifact-/Peak-VRAM-/Context-Matrix;
- Vision-Prefill-Latenz;
- keine wiederkehrende Allokation;
- 12B, NVFP4-26B und Trellis-Text-only Regression;
- begrenzte Bildsuite:
  - Beschreibung;
  - OCR;
  - Charts;
  - Counting;
  - räumliche Beziehungen;
  - kleine Details;
  - Dokumentseiten.

Noch keine breite Produktionsqualitätsaussage.

---

# 14. Empfohlene tatsächliche Reihenfolge

```text
PFX28
  Prefill/T3 Shared-Swizzle
  + T3/M1 128-Bit-Output-Store
        ↓
PFX29
  isolierter T3-Routenaufbau ohne assignments[3]
        ↓
PFX30
  nur bei Profilbeleg:
  kompakte Taskliste + Persistent Grid
        ↓
PFX31
  finaler 3W10-Freeze und Stop-Entscheidung
        ↓
separater Branch:
V00–V08  26B Vision FP8
```

## Endempfehlung

- **Die zwei offenen Residuals noch testen:** ja.
- **Weitere breite Trellis-Performancearbeit:** nein, nur noch der bounded Closeout.
- **W4A4, M128, 4.096er-Chunk, TMA:** derzeit nein.
- **Vision anschließend beginnen:** ja.
- **Trellis35 schon heute wertvoll:** eindeutig ja – die Speicherersparnis und die aktuelle Performance rechtfertigen ein separates Long-Context-/Multimodal-Profil.

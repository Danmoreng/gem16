# Codex Master Instruction – Trellis35 Performance-Closeout und 26B Vision FP8

## Ausgangspunkt

Repository:

```text
branch: codex/gemma4-26b-trellis35-perf2
commit: 540dd005f8ac6fcf729aec66d1d224af9098a301
dirty: no
```

Lies vor Beginn:

```text
AGENTS.md
docs/ACTIVE_DECISIONS.md
REVIEW_REQUEST.md
docs/plans/gemma4-26b/TRELLIS35_POST_WP21_W4A8_W4A4_DECISION.md
GEM16_TRELLIS35_POST_WP27_PERFORMANCE_AND_VISION_REVIEW.md
```

Erzeuge keine Commits, Branchwechsel, Pushes, Rebases oder Resets ohne Owner-Freigabe. Bewahre fremde Dirty Files.

---

# 1. Dauerhafte Owner-Regeln

Diese Pfade bleiben koexistent und getrennt:

```text
Gemma 4 26B NVFP4:
  qualifizierter schnellster Textpfad

Gemma 4 26B Trellis35:
  experimenteller kleinerer Textpfad

Gemma 4 26B Vision:
  neuer separater experimenteller Sidecar/Profile-Pfad
```

Unverhandelbar:

- keine stille Präzisions-, Format-, Kontext-, KV-, Sampling- oder Timingänderung;
- keine zweite persistente routed-expert-Repräsentation;
- kein CPU-Offload/Expert-Streaming;
- kein Runtime-JIT;
- kein Runtime-Repack/Quantisieren;
- keine Token-Loop-Allokation;
- kein stiller Kernel-/Format-Fallback;
- NVFP4- und 12B-Pfade regressionsschützen;
- WP8B bleibt in dieser Welle geschlossen;
- jede Optimierung besitzt Same-Binary-Rollback;
- alte Evidenz nie überschreiben.

Aktuelle Trellis35-Referenz:

```text
Prefill Ordinary:       5692.229 tok/s
Prefill Fixed-D2:       5689.205 tok/s
Ordinary Decode:         130.452 tok/s
Fixed-D2 Decode:         182.633 tok/s
Output Tokens:              1229
D2 accepted/proposed:     690/1076
D2 groups:                   538
Fallback:                       0
Target Arena:       12,204,692,480 Byte
Saving vs NVFP4:     2,491,975,680 Byte
```

---

# 2. Performance-Closeout

## PFX28-A – konfliktfreier Prefill Shared-Projection-Swizzle

### Ziel

Die noch offenen Shared-Bank-Konflikte in den M32-/M64-N128-Projection/H128-Epilogen isoliert reduzieren.

### Touchpoints

```text
src/cuda/trellis35/detail/prefill_kernels.inc.cuh
src/cuda/trellis35/detail/transform_common.cuh
tests/cuda/trellis35_prefill_test.cu
src/cuda/trellis35/reference.h
```

### Kandidatenmapper

Lege den Helper nur im Trellis35-Detailnamespace an:

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

Verwende ihn symmetrisch bei:

```text
MMA Projection Store
H128 logical Load
```

Nicht die logische H128-Reihenfolge ändern.

### Vor CUDA-Ausführung erforderlicher Hosttest

Exhaustiv beweisen:

```text
Rows 0..63
Columns 0..127
jede Row ist eine Permutation
alle vier H128-Load-Instruktionen sind bankkonfliktfrei
alle aktuellen M32-/M64-MMA-Store-Muster sind bankkonfliktfrei
Index bleibt 0..127
```

### Rollback

```text
GEM16_TRELLIS35_PREFILL_SHARED_SWIZZLE=0|1
```

Default zunächst `0`.

### Tests

K3/K4/mixed:

```text
M32 und M64
Rows: 1, 15, 16, 17, 31, 32, 33, 63, 64
Gate+Up und Down
Assignment Output
Reduced Output
2048-Token real-shaped profile
CUDA Graph
```

### NCU/SASS-Gates

Vorher/Nachher auf demselben Binary:

- Shared Load/Store Bank Conflicts;
- Shared Excessive Wavefronts;
- Duration;
- Register;
- Stack/Local;
- Static/Driver Shared;
- Occupancy;
- Issue Slots;
- Long Scoreboard;
- neue Integer-Adressinstruktionen.

Acceptance:

```text
Bitidentisch
keine Spills
kein Shared-/Workspace-Wachstum
Register nicht höher als Parent
Shared-Konflikte materiell niedriger
adressierter Kernel >=3 % schneller
Full 16K >=0,4 % schneller
```

Verlierer vollständig zurückrollen.

---

## PFX28-B – T3 Shared-Projection-Swizzle

Erst nach A, separater Commit/A-B.

### Touchpoints

```text
src/cuda/trellis35/detail/t3_kernels.inc.cuh
tests/cuda/trellis35_t3_test.cu
```

Swizzle:

```text
transformed[row][logical_column]
```

mit dem gleichen Row-/Column-Mapper.

Rollback:

```text
GEM16_TRELLIS35_T3_SHARED_SWIZZLE=0|1
```

NCU-Parent ist der aktuelle WP27-Kernel:

```text
117,79 us
40 Register
16 Byte Stack
0 Spills
4-way Shared Load Conflict
1,5-way Shared Store Conflict
83,10 % Achieved Occupancy
```

Acceptance:

```text
Bitidentisch
Draft-Trajektorie identisch
keine Ressourcenerhöhung
T3-N128-Kernel >=3 % schneller
Same-Binary D2 >=0,4 % schneller
Full 16K D2 nicht langsamer
```

---

## PFX28-C – optionaler M1 Shared-Swizzle

Nur wenn B gewinnt und ein frisches M1-NCU Shared-Load-Konflikte bestätigt.

Für `transformed[128]`:

```cpp
physical = logical ^ (logical >> 2U);
```

Rollback:

```text
GEM16_TRELLIS35_M1_SHARED_SWIZZLE=0|1
```

Acceptance:

```text
Ordinary Same-Binary >=0,3 % schneller
Full Ordinary nicht langsamer
```

---

## PFX28-D – T3/M1 128-Bit-Output-Store

Unabhängig vom Swizzle testen.

### Befund

Aktuelle T3-SASS:

```text
STG +0
STG +4
STG +8
STG +12
```

### Umsetzung

Nach H128/SVH vier Float-Werte mit einem ausgerichteten `float4`-/128-Bit-Store schreiben.

Touchpoints:

```text
src/cuda/trellis35/detail/t3_kernels.inc.cuh
src/cuda/trellis35/detail/m1_kernels.inc.cuh
```

Vor Launch/Bind statisch bzw. runtime validieren:

```text
Output Base % 16 == 0
Output Stride % 4 floats == 0
Output Block % 4 floats == 0
Lane Offset % 4 floats == 0
```

Rollbacks getrennt:

```text
GEM16_TRELLIS35_T3_VECTOR_STORE=0|1
GEM16_TRELLIS35_M1_VECTOR_STORE=0|1
```

SASS muss `STG.128` oder eine nachweislich gleichwertige Verringerung der Store-Instruktionen zeigen.

Acceptance je Pfad:

```text
Bitidentisch
Sanitizer sauber
keine Register-/Stack-/Local-Erhöhung
Kernel >=1 % schneller
Full Path >=0,2 % schneller oder neutral mit klarer Kombinierbarkeit
```

---

## PFX29 – isolierter T3-Routenaufbau ohne lokales `assignments[3]`

Nicht das verworfene WP20-Prefetch-Paket wiederholen.

### Touchpoints

```text
src/cuda/trellis35/detail/t3_kernels.inc.cuh
tests/cuda/trellis35_t3_test.cu
```

### Variante A – skalare Register

Ersetze:

```cpp
unsigned assignments[3]{};
```

durch:

```text
assignment0
assignment1
assignment2
assignment_count
```

Kein dynamischer Arrayzugriff.

### Variante B – Warp Ballot

Wenn A nicht gewinnt:

```cpp
candidate = lane < 24 ? selected_experts[lane] : invalid;
expert = shfl(candidate, group_candidate);
match_mask = ballot(candidate == expert) & 0x00ffffff;
```

Dann je 8-Slot-Row das erste Match bestimmen und maximal drei skalare Assignments kompakt packen.

Keine Tasklist, kein Payload-Prefetch, kein Host-Sync.

### Variante C – einmaliger CTA-Route-Descriptor

Nur falls A/B keinen Instruktionsgewinn liefern:

- Warp 0 berechnet;
- kleiner Shared-Descriptor;
- eine Barriere;
- 16 Warps konsumieren denselben Descriptor.

### Rollback

```text
GEM16_TRELLIS35_T3_ROUTE_PACK=legacy|scalar|ballot|shared
```

### Ressourcen-Gate

Parent:

```text
40 Register/Thread
16 Byte Stack
Local-Memory-Zugriffe vorhanden
3 Blocks/SM Registerlimit laut NCU
```

Candidate:

```text
Stack = 0
Local = 0
keine Spills
Register ideal <=42
keine niedrigere Blockresidenz
```

### Korrektheit

Exhaustiv:

```text
Union 1..24
alle Row-Verteilungen
duplicate Expert IDs
K3/K4/mixed
Gate+Up/Down
N128 off/gate/down/both
exact Assignment Order
exact Route/Slot Order
exact final output
exact 690/386 D2 trajectory
```

### Performance-Gate

```text
T3-Kernel >=2 % schneller
Same-Binary D2 >=0,3 % schneller
Full 16K D2 nicht langsamer
Ordinary unverändert
```

---

## PFX30 – bedingt: kompakte Taskliste + fixer Persistent Grid

Dieses Paket ist standardmäßig **gesperrt**.

Zulassung nur, wenn nach PFX28/PFX29 neues NSYS/NCU weiterhin belegt:

```text
viele Duplicate-Candidate-CTAs
relevante Partial Wave
Load Imbalance
Route/Control bleibt materiell
```

### Erlaubter v1-Scope

- einmaliger graphfähiger Builder pro Layer;
- maximal 24 kompakte Tasks;
- Gate+Up und Down verwenden dieselbe Liste;
- kein Prefetch;
- kein Atomic-Queue;
- kein Host-Sync;
- fixer Grid-Stride-Consumer.

Test Grid:

```text
4 / 6 / 8 CTAs pro SM
```

Rollback muss Legacy-Y24 erhalten.

Acceptance:

```text
exakte Trajektorie
kein Workspace-/Allocation-Wachstum
T3-Projektion >=5 % schneller
Full D2 >=0,5 % schneller
```

Sonst nicht weiterverfolgen.

---

## PFX31 – Performance-Freeze

Nach PFX28/PFX29 und optional PFX30:

1. beste Kandidaten gemeinsam in einem Same-Binary-A/B;
2. Full Wikipedia 16K:
   - 3 Warm-ups;
   - 10 retained Ordinary/D2-Paare;
3. Peak VRAM;
4. vollständige Output-/Draft-Identität;
5. 12B-, NVFP4- und Trellis-Text-Regression;
6. neue NCU/NSYS-Baseline archivieren.

Stoppe weitere Trellis-Mikrooptimierung, wenn:

```text
zwei aufeinanderfolgende Kandidaten je <0,5 % Full-Path
oder
Closeout kumulativ <1 %
```

Performance-Ziele, aber keine Pflicht:

```text
Prefill >=5750 tok/s
Fixed-D2 >=185 tok/s
```

Danach neuer Vision-Branch.

---

# 3. Vision-Vertical-Slice

Empfohlener Branch nach PFX31:

```text
feat/gemma4-26b-vision-fp8
```

Basis ist der eingefrorene beste Trellis-Commit. NVFP4/Text-only bleiben weiterhin separat unterstützt.

---

## V00 – Owner-, Profil-, Semantik- und Sidecar-Vertrag

### Neue explizite Profile

Keine automatische Aktivierung durch Dateiexistenz.

Beispiel:

```text
gemma4_26b_nvfp4_text
gemma4_26b_trellis35_text
gemma4_26b_trellis35_vision_fp8
```

Technisch kann dasselbe Vision-Sidecar später mit NVFP4 gekoppelt werden; Produktfreigabe nur nach Speicherqualifikation.

### Sidecar

```text
vision.gem16
gem16_vision.json
vision source lock
vision compilation manifest
```

Target, Assistant und Vision besitzen getrennte immutable Locks.

### Semantische Referenz

Pinne vor Codeänderung:

- exakte Google-QAT-BF16-Quelle;
- exakte Transformers-/Modellimplementierung;
- exakte Bildprozessor-/Config-Dateien.

Dokumentiere die Reihenfolge von:

```text
Resize/Patchify
Standardize
Patch Projection
2D Position Encoding/RoPE
27 Vision Layers
3×3 Pooling
Post-Tower Standardization
1152→2816 Projector
```

Nicht aus Erinnerung implementieren.

### Scope v1

```text
ein Bild
Batch 1
maximal 280 Text-Soft-Tokens
kein Audio
kein Video
Textausgabe
Ordinary zunächst
D2 erst nach Ordinary-Korrektheit
```

### Shape-Entscheidungen

Explizit entscheiden und dokumentieren:

```text
Head 72:
  Tail oder physisches Padding

MLP 4304 Down-K:
  Tail / 4320 / 4352
```

Keine stillen Loader-Paddings.

---

## V01 – Vision-FP8-Compiler

### Source-Inventar

356 Tensoren, aktuell 1.145.588.832 BF16-Byte.

Große Linears:

```text
191 Tensoren
549.070.848 Parameter

Patch input projection
27 × Q/K/V/O
27 × Gate/Up/Down
Final 1152→2816 projector
```

### v1-Format

```text
große Linear-Weights:
  FP8 E4M3
  BF16 per-output-channel scale
  final runtime layout

Position Embedding:
  BF16

Norms:
  BF16

std_bias/std_scale:
  BF16
```

Statische Zielgröße ohne physisches Shape-Padding:

```text
597.313.024 Byte
569,64 MiB
```

Diese Zahl ist nur Planungsziel, bis das echte Artifact sie reproduziert.

### Touchpoints

```text
tools/gem16_compile/profiles.py
tools/gem16_compile/plan.py
tools/gem16_compile/compiler.py
tools/gem16_compile/writer.py
tools/gem16_compile/specs/
tools/generate_gemma4_26b_*vision*
tests/python/
```

### Gates

- keine Vision-Tensoren mehr still als `compile_excluded_vision` im neuen Profil;
- Text-only-Profile bleiben unverändert;
- zwei Clean Builds byte-identisch;
- exakte Source-/Shape-/Dtype-/Byte-/Paddingbilanz;
- kein Runtime-Repack.

---

## V02 – Loader, Bindings und Residency

### Touchpoints

```text
src/model/gemma4_26b_manifest.*
src/model/gemma4_26b_compiled_loader.*
src/model/gemma4_26b_residency.*
src/cuda/engine/gemma4_26b_*artifact*
src/cuda/engine/detail/gemma4_26b_create.inc
src/model/model_variant.*
src/cuda/inference_session.cuh
```

### Anforderungen

- Vision-Capability kommt aus einem validierten Profile-/Component-Vertrag;
- Vision-Sidecar-Mismatch fail-closed;
- eine persistente Vision-Repräsentation;
- keine zweite BF16-Kopie;
- feste Offsets/Buffers;
- kein token-/image-loop `cudaMalloc`;
- Text-only lädt Vision nicht.

### Capacity-Matrix

Vor Kernelintegration zunächst statisch und synthetisch:

```text
Text-only
Text + Vision
Text + Vision + Assistant
32K
64K
86.016
206.848 diagnostisch
erste kontrollierte Ablehnung
```

---

## V03 – 26B-spezifischer Bildinput und Patch Embedder

Die bestehende `VisionEmbeddingSegment`-Semantik `[patch,6912]` ist 12B-spezifisch und darf nicht wiederverwendet werden.

Neue explizite Struktur, beispielsweise:

```cpp
struct Gemma4Moe26BVisionInputSegment {
  uint64_t prompt_offset;
  uint32_t soft_token_count;     // <=280
  uint32_t raw_patch_count;      // <=2520
  span<const float> patches;     // [raw_patch,768]
  span<const int32_t> positions; // [raw_patch,2]
};
```

Oder ein gleichwertiger versionierter, nicht generisch aufgeblähter Vertrag.

### Implementierung

- 16×16×3 Patchify;
- sichere Aspect-Ratio-/Resize-Logik gemäß gepinnter Referenz;
- Standardisierung;
- FP8 Patch Projection 768→1152;
- BF16 2D Position Embedding;
- CPU-/PyTorch-Oracle;
- ein Bild pro Request.

---

## V04 – Vision Attention

Neue, 26B-spezifische Kernel.

Anforderungen:

```text
16 Q-Heads
16 KV-Heads
Head Dim 72
Q-/K-RMSNorm
2D RoPE
bidirektionale Full Attention
online Softmax
kein voller Score-Slab
```

Roh-Patches können bis ungefähr 2.520 reichen. Ein kompletter quadratischer FP32-Score-Slab ist nicht zulässig.

Weight-Linears FP8; logische Layergrenzen zunächst BF16; Akkumulation FP32.

Tests:

- kleine Patchzahlen;
- 3×3, 4×4, ungerade Raster;
- maximale Fixture;
- Head-72-Tail;
- 2D Positions;
- numerische Referenz;
- Sanitizer;
- Graph Replay.

---

## V05 – Vision MLP und Layer

```text
Hidden: 1152
Intermediate: 4304
Activation: GELU
Gate×Up
Down
Norm-/Residualfolge exakt nach Referenz
```

FP8 Gate/Up/Down-Weights.

Der 4304-Contract muss den V00-Beschluss verwenden.

Erst Operator, dann ein Layer, dann 27 Layer.

---

## V06 – Pooling und Projektor

- 3×3 Pooling von Roh-Patches;
- genau maximal 280 Soft-Tokens;
- std_bias/std_scale;
- FP8 1152→2816 Projector;
- physischer BF16 Text-Hidden-Output;
- unabhängiger Oracle.

---

## V07 – Text- und D2-Integration

### Text

- 280 Bild-Soft-Tokens in den Prompt einsetzen;
- bidirektionale Vision-Span-Attention unverändert;
- normale Text-Prefill-/Decode-Semantik außerhalb des Spans;
- Continuation und Session Cache.

### D2

Erst nach Ordinary-Identität:

- Assistant erhält dieselben textseitigen Hidden/KV-Grenzen wie der Target-Vertrag verlangt;
- keine Bildneuberechnung pro Draft-Gruppe;
- Vision-Tower einmal pro Bild;
- D2 Target-Verifikation unverändert.

---

## V08 – begrenzte Qualifikation

Kein großes allgemeines WP8B.

Mindestens:

### Korrektheit

- BF16-/PyTorch-Referenz für ausgewählte Bilder;
- per-layer drift;
- finale 280 Embedding-Reihen;
- Textlogit-KL/top-1;
- deterministische Generation.

### Bildaufgaben

- normale Beschreibung;
- OCR;
- Dokumente;
- Charts;
- Counting;
- Spatial Reasoning;
- kleine Details;
- Farben.

### System

- Bildvorverarbeitung;
- Vision-Tower-Latenz;
- TTFT;
- Peak VRAM;
- Context-Matrix;
- Ordinary/D2;
- Streaming;
- Cancellation;
- Relaunch;
- keine wiederkehrende Allokation;
- 12B/NVFP4/Trellis-text-only Regression.

Keine Produktionsqualitätsaussage ohne spätere breite Suite.

---

# 4. Nicht in dieser Welle

- Trellis→W4A4;
- W4A16;
- 4.096er-Prefill-Chunk;
- M128;
- TMA ohne neuen Profilerbeleg;
- H256/H64-Overnight-Quantizer;
- Trellis30;
- Vision-Trellis-Weights;
- Vision Q4;
- mehrere Bilder;
- Audio/Video;
- lange WP8B-Suite;
- allgemeines multimodales Framework.

---

# 5. Berichtspflicht

Nach jedem Paket:

```text
Revision
Dateien
Kommandos
GPU/Driver/CUDA
Source-/Model-/Artifact-Hashes
Korrektheitsgrenze
SASS
Register/Shared/Stack/Local/Spills
NCU/NSYS
Same-Binary-A/B
Full-Path
Peak VRAM
Allocation Delta
Fallback
akzeptiert oder vollständig zurückgerollt
Limitierungen
```

Beginne mit **PFX28-A**. PFX28-B/D dürfen parallel in separaten Worktrees vorbereitet werden, Integration bleibt seriell.

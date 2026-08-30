# Deferred Quantizer Checkpoint QX1 – Adaptive Hadamard für Trellis35

**Status:** bewusst verschoben
**Nicht Teil von WP17–WP22**
**Zweck:** prüfen, ob shape-spezifische H256/H128/H64-Transformationen bessere Qualität oder niedrigere Bitrate ermöglichen und gleichzeitig Down-Padding entfernen.

---

## Hypothese

Aktuell:

```text
Gate+Up Input:   H128
Gate+Up Output:  H128
Down Input:      6 × H128 auf 768, davon 64 Padding
Down Output:     H128
```

Kandidat:

```text
Gate+Up Input:   11 × H256 = 2816
Gate+Up Output:  11 × H128 = 1408

Down Input:      11 × H64  = 704
Down Output:     11 × H256 = 2816
```

Vorteile der Geometrie:

- kein Gate+Up-Padding;
- kein Down-704→768-Padding;
- exakt elf Transformblöcke an jeder Matrixseite;
- stärkere Durchmischung der 2816er-Dimension;
- potentiell 3,25 oder 3,0 bpw bei Qualität nahe aktuellem 3,5-bpw-H128.

Risiken:

- größere Transformassoziation;
- andere Quantisierungsstatistik;
- zusätzliche Runtime-Kernelvarianten;
- H256 hat acht statt sieben Butterfly-Stufen;
- Qualitätsgewinn ist nicht monoton garantiert.

---

## QX1-A – kleiner Offline-Screen

Kein kompletter Checkpoint.

Repräsentative Auswahl vorab einfrieren:

```text
Layer: 0, 5, 15, 24, 29
Experts:
  höchster K4-Benefit
  medianer Benefit
  niedrigster Benefit
je Projektion
```

Vergleiche:

```text
A: H128 current, 3.5 bpw
B: adaptive H256/H128/H64, 3.5 bpw
C: adaptive, 3.25 bpw
D: adaptive, 3.0 bpw
```

Metriken:

```text
Weight NMSE
Hessian Proxy Error
max. absolute error
Cosine
Layer Expert Output Error
Router/Residual Layer Output
kurze teacher-forced KL
Compilerzeit
Payload/Sidecar Bytes
```

Vorab definierte Entscheidung:

- Overnight-Quant nur, wenn B klar besser als A ist oder
- C/D die aktuelle A-Qualitätsgrenze ungefähr halten und einen relevanten Bytegewinn versprechen.

---

## QX1-B – Runtime-Microbench vor Full Quant

Implementiere diagnostische H64- und H256-Warp-FWHT-Primitives.

Messe isoliert:

```text
H64
H128
H256
```

für:

```text
Input Transform
Inverse Output
Transform+amax+E4M3
```

Keine Produktintegration.

Prüfe:

- numerische Oracle-Grenze;
- Register;
- Shared;
- Spills;
- Zeit pro Element;
- Einfluss auf M64/N128-Runtime.

---

## QX1-C – Overnight-Full-Quant

Nur nach A und B.

Erzeuge höchstens zunächst:

```text
adaptive 3.5
adaptive 3.25
```

3.0 folgt nur bei überzeugendem Screen.

Der Artifact-Formatvertrag muss Transformgeometrie explizit pro Familienseite deklarieren. Kein stilles Ableiten aus Shape oder Bitrate.

---

## QX1 ist kein Performance-Ersatz

Der Quantizer-Checkpoint darf WP17–WP22 nicht blockieren.

Die aktuelle W4A8-Runtime muss K3/K4, Scheduling, M64, LSU und T3 unabhängig davon weiter optimieren können. Adaptive-Hadamard ist eine spätere Artifact-/Qualitätsentscheidung.

# Entscheidungspunkte für den Projektinhaber

Der Coding Agent darf diese Punkte nicht eigenmächtig als dauerhaft entschieden behandeln.

## D00 — Abgeleiteten Checkpoint verwenden — entschieden

**Entschieden am:** 2026-08-04
**Entscheidung:** `gem16` darf für das 26B-Modell einen reproduzierbaren, projektgebauten
QAT-BF16→FP8/NVFP4/Q4_0-Checkpoint als primären 26B-Pfad verwenden. Eine allgemeine Direct-Load-only-Regel gilt
nicht mehr. Unveränderliche Quellen, offener und gelockter Compiler, Safetensors, vollständige Provenienz, keine
Konvertierung beim Runtime-Start und klar benannte externe Baselines bleiben verpflichtend.

M00 muss den konkreten Artefaktvertrag dokumentieren, entscheidet aber nicht erneut über die grundsätzliche
Zulässigkeit projektgebauter Checkpoints.

## D01 — Quellen und Distribution

**Zeitpunkt:** M01
**Frage:** Welche exakten Google-/Unsloth-Revisionsstände dürfen heruntergeladen, verarbeitet und gegebenenfalls als abgeleitetes Artefakt verteilt werden?

Vor einer Veröffentlichung müssen Modellbedingungen und Hosting geklärt sein.

## D07 — Embedding/Head vorläufig

**Zeitpunkt:** M07
**Frage:** Q4_0 oder NVFP4 für das gebundene Embedding/Output-Head?

Noch keine endgültige Entscheidung. Erst isolierte Qualitäts-, Speicher- und Geschwindigkeitsmessung.

## D09 — Speicher-Hard-Stop

**Zeitpunkt:** M09
**Frage:** Was geschieht, wenn die unveränderlichen Gewichte über 14.300 MiB liegen?

Empfehlung: Nicht weiter optimieren, sondern Format/Residency neu entscheiden. Kein CPU-Offload als Standardlösung.

## D18 — Qualitätsgrenzen einfrieren

**Zeitpunkt:** Ende M18
**Frage:** Welche numerischen und aufgabenbezogenen Grenzwerte gelten für den Held-out-Test?

Diese Grenzen müssen vor M19 feststehen.

## D19 — Finaler Produktionskandidat

**Zeitpunkt:** M19
**Fragen:**

- QAT-BF16- oder gewöhnlicher BF16-Master?
- Q4_0- oder NVFP4-Head?
- Ist die Qualität gegenüber offiziellem Q4_0 und Unsloth ausreichend?

Ein theoretischer QAT-Vorteil ersetzt keine Messung.

## D21 — 64K-Kontext

**Zeitpunkt:** M21
**Frage:** Wird 64K als unterstützt, experimentell oder abgelehnt bezeichnet?

32K ist das verpflichtende Ziel. 64K benötigt eigene Qualitäts-, Speicher- und Stabilitätsnachweise.

## D23 — Release

**Zeitpunkt:** M23
**Fragen:**

- Sind genaue Claims freigegeben?
- Ist das abgeleitete Modell korrekt bezeichnet?
- Sind Text-only, kein MTP und Kontextgrenzen klar?
- Ist der Rollback getestet?

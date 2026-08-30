# Trellis35 nach WP21: W4A8- oder W4A4-Entscheidung

**Datum:** 2026-08-30  
**Branch:** `codex/gemma4-26b-trellis35-perf2`  
**Evidenzstand:** WP17 bis WP21, Implementierungsstand `58b3af0`  
**Status:** WP22 abgeschlossen; Architekturentscheidung, kein W4A4-Implementierungsauftrag

## Entscheidung

**Trellis35 bleibt ein W4A8-System. W4A4 wird nach WP21 weiterhin nicht
zugelassen.**

Keine der vier im Masterplan geforderten W4A4-Admission-Conditions ist
erfüllt. Die gemessenen W4A8-Pfade werden weiterhin durch LSU-, Shared-Memory-,
Schedule- und Inter-Kernel-Arbeit begrenzt; eine materielle Tensor-Pipe-
Compute-Decke ist nicht belegt. Gleichzeitig hat WP21 mit unverändertem W4A8
den vollständigen 16K-Prefill um 10,77 % beschleunigt. Eine zweite
Aktivierungsquantisierung auf E2M1 würde die beobachteten Engpässe nicht
beseitigen, aber einen neuen Byte-, Rundungs- und Qualitätsvertrag benötigen.

Die Entscheidung gilt ausschließlich für den experimentellen Trellis35-Pfad.
Sie verändert weder den qualifizierten NVFP4- noch den 12B-Pfad und
produktqualifiziert Trellis35 nicht.

## Admission-Conditions

| Bedingung | Befund nach WP21 | Urteil |
|---|---|---|
| Tensor-Pipe ist materiell limitierend | WP17 weist LSU als höchstgenutzte Instruktionspipeline aus; eine Tensor-Compute-Decke wurde ausdrücklich nicht beobachtet. | Nicht erfüllt |
| Decoder-, LSU-, Barrier- und Schedule-Kosten dominieren nicht mehr | WP18 konnte sechs lokale LSU-/Shared-Kandidaten nicht behalten. WP20 reduzierte Long Scoreboard, wurde wegen mehr Instruktionen, 39→47 Registern und geringerer Occupancy aber 12,47 % langsamer. | Nicht erfüllt |
| Optimaler oder decoderfreier W4A8-Kontrollarm zeigt eine Compute-Decke | Der transiente E4M3-Slab aus WP11 blieb unter dem Inline-Pfad; seitdem ist kein überlegener decoderfreier Kontrollarm entstanden. | Nicht erfüllt |
| Vollständiger E2M1 + E4M3-K/16 Byte- und Qualitätsvertrag liegt vor | Persistentes Layout, Blockskalen, Rundung, Trellis- plus E2M1-Fehler und Qualitätsgates sind weiterhin nicht spezifiziert oder gemessen. | Nicht erfüllt |

W4A4 bleibt gesperrt, bis alle vier Bedingungen durch neue Evidenz erfüllt sind
und der Owner die Forschungslinie erneut ausdrücklich freigibt.

## Entscheidungsbasis

### Prefill

Der akzeptierte M64-N128-Pfad aus WP17 verwendet 64 Register je Thread,
37,12 KiB statisches Shared Memory, keine Local-/Shared-Spills und erreicht
ungefähr 65 % Occupancy. NCU nennt LSU als stärkste Pipeline und keine
Tensor-Pipe-Compute-Decke. Die Rohprofile zeigen außerdem stark konfliktbehaftete
Shared-Memory-Stores und -Loads im Projection/H128-Epilog. Das ist ein
W4A8-Layout- und Schedule-Problem, kein Beleg für einen FP8-MMA-Durchsatzdeckel.

WP18 testete Scale-, SVH- und Payload-Staging, gepackte E4M3-Konvertierung,
vectorized Projection-I/O sowie K64. Alle Kandidaten wurden zurückgerollt, weil
zusätzliche Shared-/Dependency-Kosten den eingesparten Traffic aufwogen. Dieses
Ergebnis spricht gegen genau diese lokalen Varianten, nicht für W4A4.

WP19 fusionierte Gated-GELU, Down-Transform und Quantisierung exakt und erzielte
ohne Workspace-Wachstum einen kleinen Full-Path-Gewinn. WP21 erhöhte danach den
Trellis-spezifischen Chunk von 1024 auf 2048 und erreichte im finalen
Wikipedia-16K-Panel:

| Modus | Prefill median | Veränderung gegen WP20 |
|---|---:|---:|
| Ordinary | 5.703,96 tok/s | +10,77 % |
| Fixed-D2 | 5.702,97 tok/s | +10,77 % |

Der Same-Binary-Screen misst 1024 gegen 2048 mit +11,67 % Ordinary und
+11,76 % Fixed-D2. Output-Hash, 1.229 Ausgabetokens und die
690/1.076/386-Draft-Trajektorie bleiben identisch. Der Gewinn kommt aus halb so
vielen 16K-Chunks und größerer realer Routing-Geometrie, nicht aus einer
geänderten MMA-Präzision.

Damit steigt Trellis35 auf etwa 81,9 % der historischen NVFP4-Prefill-Referenz
von ungefähr 6.966 tok/s. Der verbleibende Abstand ist materiell, aber die
letzte reine W4A8-Änderung hat ihn erneut deutlich reduziert.

### Decode und Fixed-D2

Das finale WP21-Panel erreicht 126,53 tok/s Ordinary und 169,92 tok/s
Fixed-D2. Die kleinen Änderungen gegen WP20 von −0,25 % beziehungsweise
−0,59 % liegen bei einem ausschließlich auf Prefill gerichteten Paket und
unveränderter Draft-Trajektorie; WP21 wird nicht als Decode-Gewinn gewertet.

WP20 zeigt weiterhin vermeidbare W4A8-Arbeit: Der kombinierte Tasklist- und
Prefetch-Kandidat senkte Long Scoreboard von 4,87 % auf 2,10 %, verlor aber
durch 14,34 % mehr Instruktionen und Registerdruck. Das widerlegt Prefetch4,
nicht einen isolierten Routenaufbau. Zusätzlich serialisiert T3 derzeit den
Shared-NVFP4- und Routed-Trellis-Zweig, obwohl Ordinary M1 beide bereits per
Fork/Join überlappt. Diese offenen Schedule- und Branch-Overlap-Hebel müssen
vor W4A4 bearbeitet werden.

### Speicher und Kontext

WP21 kostet gegenüber dem 1024er Rollback 149.337.088 Workspace-Bytes
beziehungsweise 142 MiB gemessenen Peak, erreicht dafür aber den genannten
Prefill-Gewinn. Der neue Trellis-Cap erlaubt den vollständigen Engine-Hard-Max:

| Target-only Kontext | Freier VRAM nach Decode | Zweck |
|---:|---:|---|
| 206.848 | 1.517.092.864 B / 1,413 GiB | konservative Reserve für spätere Vision-Arbeit |
| 262.144 | 894.238.720 B / 0,833 GiB | absoluter Text-only-Maximalpunkt |

Beide Punkte wurden vollständig mit realem Prefill und Boundary-Decode
ausgeführt, ohne Fallback oder wiederkehrende Allokation. Vision selbst bleibt
ausgeschlossen und unqualifiziert. Dieser Speichergewinn stammt aus dem
3,5-bpw-Artefakt; W4A4 ist nicht erforderlich, um den aktuellen Engine-Hard-Max
zu erreichen.

## Warum W4A4 jetzt nicht hilft

Ein nativer W4A4-Pfad müsste die rekonstruierten Trelliswerte zusätzlich nach
E2M1 runden und K/16-Skalen erzeugen oder laden. Das fügt Decoder-, Scale- und
Qualitätsarbeit hinzu, während die aktuellen Profile bereits durch LSU,
Shared-Bank-Konflikte, leere/partielle CTAs, Routenaufbau und serialisierte
Zweige begrenzt sind. Ein nominell schnellerer FP4-MMA löst keine dieser
Grenzen automatisch.

Zudem bleibt genau eine persistente Expert-Repräsentation Pflicht. Ein zweiter
dekodierter FP4- oder E4M3-Cache ist daher kein zulässiger Abkürzungspfad. Ohne
einen vollständigen on-the-fly Byte- und Fehlervertrag wäre jede W4A4-Messung
weder eine belastbare Performance- noch eine Qualitätsaussage.

## Nächste W4A8-Arbeit

Die nächste Performancewelle soll Kandidaten weiterhin einzeln isolieren:

1. Prefill: enger M64-Launch-Bound und dynamisches M32-M16-Tail-Skipping;
2. Prefill: separater konfliktfreier Shared-Projection-Swizzle, nur wenn der
   M64-Kernel bei höchstens 64 Registern und ohne Spills bleibt;
3. Fixed-D2: Shared-NVFP4 und Routed-Trellis im T3-Pfad nach dem bestehenden
   M1-Fork/Join-Muster überlappen;
4. T3: Routenaufbau ohne dynamisch indiziertes lokales Assignment-Array
   isoliert testen;
5. danach N128-Konsolidierung oder Shape-Spezialisierung anhand neuer NCU-/NSYS-
   Evidenz bewerten.

Jeder Kandidat benötigt Same-Binary-Rollback, exakte Operatornumerik,
Graph-Replay, Sanitizer, Ressourcen-/Dispatch-Nachweis und das vollständige
Wikipedia-16K-Panel. W4A8-Roundpoints, Token-, Sampling-, KV- und
Attention-Semantik bleiben unverändert.

Vor dem ersten neuen Decode-Implementierungspaket muss die aktuelle
Owner-Freigabe für die Performancearbeit den noch in
`docs/ACTIVE_DECISIONS.md` dokumentierten Decode-Freeze ausdrücklich
supersedieren. WP22 selbst ändert diese Produktpriorität nicht.

## Evidenz

- `artifacts/trellis35/wp17-m64-ncu.json`
- `artifacts/trellis35/wp18-full-16k-final.json`
- `artifacts/trellis35/wp19-gelu-down-fusion.json`
- `artifacts/trellis35/wp20-t3-prefetch-ncu.json`
- `artifacts/trellis35/wp20-full-wikipedia-16k-1w3.json`
- `artifacts/trellis35/wp21-prefill-chunk-sweep.json`

## Schlussurteil

WP22 entscheidet erneut und mit stärkerer Evidenz für **W4A8 weiter
optimieren**. W4A4 bleibt eine spätere, separat zu autorisierende
Forschungslinie. Der unmittelbar nächste Engpass liegt in Schedule,
Shared-Memory-Verhalten und T3-Branch-Overlap, nicht in einer nachgewiesenen
FP8-Tensor-Compute-Decke.

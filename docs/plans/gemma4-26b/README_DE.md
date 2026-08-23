# Gemma 4 26B A4B — Fast-Track-Plan R4

Diese Revision strafft die Arbeit ab M06. M00–M05 und deren Evidenz bleiben unverändert.

## Einstieg

1. [`../../ACTIVE_DECISIONS.md`](../../ACTIVE_DECISIONS.md)
2. [`ACTIVE_CONTRACT.md`](ACTIVE_CONTRACT.md)
3. [`FAST_TRACK_STATUS.json`](FAST_TRACK_STATUS.json)
4. [`START_HERE_CODEX.md`](START_HERE_CODEX.md)
5. die aktuell zugewiesene Milestone-Datei

Der Coding Agent muss nicht mehr vor jeder Aufgabe die vollständigen historischen Entscheidungs-, Korrektheits- und Benchmark-Dokumente lesen. Er lädt nur die aktive Vereinbarung, den Status, das aktuelle Milestone, die dort verlinkten Spezifikationen und den relevanten Code/Testumfang.

## Ziel

- möglichst schnell eine experimentelle, ausführbare Gemma-4-26B-Textausführung auf einer einzelnen 16-GB-Blackwell-GPU;
- vollständig residente Gewichte ohne CPU-Offload;
- zuerst ein reales 32K-Profil, danach gezielte Qualifikation von 64K und dem maximal sicheren Einzelbenutzer-Kontext;
- abschließend ein exakt Target-verifiziertes MTP-Profil nach dem eingefrorenen Base-Ziel.

M23 friert zunächst einen technischen Base-Target für weitere MTP- und Performance-Arbeit ein. Solange die vom Owner
nach hinten verschobene M19-Vollqualifikation fehlt, ist dieser Stand kein Shipping- oder qualitätsfreigegebener
Release. Das Gesamtziel benötigt M25 und den späteren M19-Release-Gate. Vision gehört nicht zum MTP-Milestone.

Parallele Sub-Agent-Arbeit ist erlaubt, sofern Dateibesitz und Merge-Abhängigkeiten gemäß [`PARALLEL_WORKSTREAMS.md`](PARALLEL_WORKSTREAMS.md) getrennt sind.

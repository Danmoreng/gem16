# gem16: Implementierungsplan für Gemma 4 26B A4B auf 16-GB-Blackwell-GPUs — Repository-Revision 3

Dieser Ordner ist ein ausführbarer Arbeitsplan für einen Coding Agent, insbesondere Codex. Er beschreibt nicht nur das gewünschte Endergebnis, sondern auch die Reihenfolge, die Prüf-Gates, die zu ändernden Repository-Bereiche, die erwarteten Artefakte und die Abbruchkriterien jedes einzelnen Milestones.

## Zielbild

Der primäre Produktionskandidat wird aus **Googles unquantisiertem QAT-BF16-Checkpoint** abgeleitet:

- Routed Experts: NVFP4, W4A4 auf nativen Blackwell-Pfaden
- Shared Dense MLP: NVFP4
- Attention Q/K/V/O: FP8
- Router, Norms und kleine Skalar-Tensoren: BF16
- Tied Embedding/Output Head: zunächst zwei Kandidaten
  - Google-kompatibles Q4_0 als Qualitätsprofil
  - NVFP4 als Geschwindigkeitsprofil
- KV-Cache: FP8
- Vision: in der ersten Version nicht resident und nicht unterstützt
- MTP: erst nach erfolgreicher Basisqualifikation

Das rechnerische Ziel für die residenten Textgewichte liegt bei ungefähr **13,69 GiB beziehungsweise 14.014 MiB**, bevor kleine Tensoren, Alignment und Loader-Metadaten hinzukommen. Die harte Entwicklungsaufgabe ist deshalb nicht nur die Quantisierung, sondern auch ein streng begrenzter MoE-Prefill-Workspace.

## Empfohlene Lesereihenfolge

Für den ersten Codex-Auftrag direkt [`START_HERE_CODEX.md`](START_HERE_CODEX.md) öffnen.

Danach:

1. [`INDEX.md`](INDEX.md)
2. [`00_MASTER_IMPLEMENTATION_PLAN.md`](00_MASTER_IMPLEMENTATION_PLAN.md)
3. [`02_AGENT_OPERATING_CONTRACT.md`](02_AGENT_OPERATING_CONTRACT.md)
4. [`07_MEMORY_BUDGET_AND_RESIDENCY.md`](07_MEMORY_BUDGET_AND_RESIDENCY.md)
5. [`06_DEPENDENCY_GRAPH.md`](06_DEPENDENCY_GRAPH.md)
6. Danach ausschließlich den aktuell freigegebenen Milestone unter [`milestones/`](milestones/) bearbeiten.

## Wichtige Arbeitsregel

Der Agent darf nicht mehrere Milestones stillschweigend zusammenziehen. Jeder Milestone endet mit einem überprüfbaren Exit-Gate. Ein schnellerer Kernel darf nicht promoted werden, wenn seine Arithmetik, Modellqualität, Speicherbelegung oder Benchmarkmethodik ungeklärt ist.

## Verankerter Repository-Stand

Die Analyse ist auf `Danmoreng/gem16` bei Commit:

```text
1c4287965d318ba32a68e597f9d7b6678b883376
```

vom 3. August 2026 verankert. Vor Beginn muss der Agent die Abweichungen des tatsächlichen Arbeitsstands gegen diesen Commit dokumentieren.

## Inhalt

Der Ordner enthält:

- einen Masterplan und Architekturentscheidungen,
- 26 eigenständige Milestone-Pläne,
- Spezifikationen für Compiler, Checkpointformat, Quantisierung, MoE, Attention, Speicher, Qualität und Benchmarks,
- Checklisten für PRs, CUDA-Kernels, Quantizer, Qualität und Release,
- wiederverwendbare Codex-Prompts und Dokumentvorlagen,
- genaue Speicherrechnungen und eine Datei-zu-Milestone-Karte.

Beginne nicht mit CUDA-Optimierung. Beginne mit Governance, unveränderlichen Modell-Locks, Goldens und einem langsamen, beweisbaren Referenzpfad.


## Umfang des Pakets

Der finale Ordner enthält 26 Milestone-Pläne, 28 technische Spezifikationen, 12 Checklisten, 13 Vorlagen, 13 Anhänge und 8 vertiefende imp-Referenzdokumente sowie die übergeordneten Architektur-, Speicher-, Risiko-, PR- und Releasepläne. Ein generiertes Inhaltsverzeichnis, Dateihashes und ein ZIP-Integritätstest gehören ebenfalls zum Paket.

## imp-Referenz

Die Analyse von `kekzl/imp` ändert die Grundarchitektur nicht. Sie ergänzt einen fest gepinnten Referenzpfad mit:

- einem NVIDIA/ModelOpt-NVFP4-Kontrollarm für die Qualitätsanalyse,
- getrennten ModelOpt-/llm-compressor-Scale-Verträgen,
- früheren FP32-Router- und Residual-Goldens,
- tatsächlicher Dispatch- und CUDA-Graph-Demotion-Telemetrie,
- maschinenlesbaren Performance-/VRAM-Gates,
- CUDA-Lifecycle-Tests,
- einer klaren MIT-Provenienzregel für einen optionalen Kernel-Port.

Zuerst [`13_IMP_REFERENCE_INTEGRATION.md`](13_IMP_REFERENCE_INTEGRATION.md) lesen.

## Änderungen in Repository-Revision 3

Nach der Integration am exakten Repository-Anker:

- ist M00 an die allgemeine Zulässigkeit reproduzierbarer projektgebauter Modellartefakte angepasst;
- verwendet das Speicherbudget die ungefähr 15.881 MiB CUDA-sichtbare statt 16.303 MiB nominelle Kapazität;
- blockiert ein vorläufiger synthetischer M03-32K-/700-MiB-Test die Compilerarbeit frühzeitig, wenn das Modell nicht passt;
- prüft M01 SSD-/Cache-Bedarf und BF16-Referenzspeicher vor großen Downloads;
- laufen Konverterattribution und ein vorläufiges Quality-Kill-Gate vor der nativen M14–M17-Optimierung;
- ist die lokale RTX-5080-Q4_0-Charakterisierung von imp als Negativergebnis dokumentiert, ohne sie als NVFP4-Kernelvergleich zu bezeichnen.

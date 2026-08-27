# Gemma 4 26B: Stand, nächste Schritte und Main-Promotion

Stand: 2026-08-27  
Review-Snapshot: `7589940` (`feat/gemma4-26b`)  
Zweck: Entscheidungsgrundlage für einen externen ChatGPT-Pro-Review und die anschließende Main-Promotion.

Das zugehörige Review-Archiv enthält zusätzlich kuratierten Referenzcode aus llama.cpp
`f1357e49980f5462af9783164f3fdec407d90137` (`b10622-1-gf1357e499`) und vLLM 0.27.1. Diese Quellen sind nur
Vergleichsmaterial; ihre APIs, Speicherarchitektur und Semantik sind keine automatisch zu übernehmenden Vorgaben.
Enthalten sind insbesondere llama.cpp Loader/Runtime/CUDA/Speculative/Server sowie vLLM Layer, Gemma-4-/MTP-
Modelldefinitionen, Worker/Core, Speculative Decode, Konfiguration und Model-Loader. Builds, Python-Umgebungen,
Binärmodule und die vollständigen Fremdrepositories bleiben ausgeschlossen.

## Kurzurteil

Der Owner hat den text-only 26B-SM120-Pfad als **qualifizierten, auswählbaren Produktcheckpoint** akzeptiert. Er ist
damit kein experimentelles Optionalprofil mehr, ersetzt aber nicht den 12B-Default und erweitert dessen multimodalen
Vertrag nicht. Die komplette mehrstündige historische M19-Suite ist für diese klar begrenzte Qualifikation nicht
mehr erforderlich.

Für die konkrete Main-Auslieferung fehlen noch zwei klar begrenzte Delivery-Punkte:

1. aktuelle HEAD-Binaries und beide Produktprofile (12B/26B) neu revalidieren;
2. die getrennten Target-/Assistant-Repositories unveränderlich pinnen und den Studio-Download abschließen.

Die zuvor offene Quality-Entscheidung ist getroffen: vollständiges GSM8K und AIME 2026 plus die begrenzte
Sampled-/Produktevidenz ersetzen die historische Vollsuite für diesen Checkpoint. Das qualifiziert den konkreten
lokalen SM120-Vertrag, nicht ungemessene Domänen oder andere Plattformen.

## Was bereits belastbar vorhanden ist

- Ein vollständig GPU-residenter, nativer SM120-Targetpfad für Gemma 4 26B A4B, ohne CPU-Offload oder stillen
  Präzisions-/Kernel-Fallback.
- Akzeptierte M20-Basisleistung: Median 6.572,809 Prompt-Token/s und 150,615 Ordinary-Decode-Token/s auf dem
  festgelegten 16K+64-Profil (3 Warm-ups, 10 Messungen).
- Späterer exakter Entwicklungsstand: 7.068,125 Prompt-Token/s Median sowie 204,246–204,415 Token/s in zwei
  gestoppten finalen Fixed-D2-Läufen. Das ist starke Entwicklungsevidenz, aber keine formale M25-Verteilung.
- Akzeptierte Basis-Kontexte 32K, 64K und 96K; 100K wird reproduzierbar kapazitätsbedingt abgelehnt.
- Fixed-D2-MTP besteht 65.536 und 73.728 Tokens wiederholt mit 200 MiB Reserve; `mtp_max_context=73.728`.
  Die reine VRAM-Admission reicht bis 88.640, aber ein CUDA-Ausführungsfehler oberhalb 74.944 begrenzt den sicheren
  Produktpfad bewusst früher.
- Ein separat kompilierter, residenter Assistant und Target-verifiziertes D2-MTP. Greedy und begrenzte Sampled-
  Differentials stimmen tokenidentisch mit Ordinary Target überein. Kein Token-Loop-Allocator und kein stiller
  Fallback im Produkt-Smoke.
- CLI, OpenAI-kompatibler Server und Studio-Modellauswahl sind vorhanden. 26B bleibt text-only und ein Slot;
  12B bleibt Default und behält Audio/Vision.
- Der Server hat jetzt strukturiertes Logging, Request-IDs, eine bounded FIFO-Admission, gleiche-Session-Warten,
  Prometheus-Latenz-/Queue-Metriken, Readiness/Liveness, Timeouts und Graceful Draining.
- Der neue Device-Image-Pfad entfernt die große Startup-Layouttransformation aus der Runtime. Structural Integrity
  ist der schnelle Default; vollständiges SHA-256 bleibt explizit wählbar.
- Aktuelle reale Quality-Vergleiche gegen Googles offizielles QAT-Q4_0-Modell:
  - GSM8K vollständig: 92,34 % gem16 gegen 92,87 % Referenz, Delta −0,53 Prozentpunkte, 99,43 % Retention;
    gepaarter McNemar-p-Wert 0,435 (kein statistisch belastbarer Unterschied in diesem Lauf).
  - AIME 2026 vollständig: beide 63,33 % über 30 Aufgaben; vier Referenz-only und vier gem16-only Treffer.
  - GPQA Diamond wurde nur begonnen und liefert noch keinen Vergleich.

Das reicht für die Aussage „sehr guter experimenteller 26B-Pfad mit starker Referenznähe“. Es reicht noch nicht für
eine breite, domänenübergreifende Produktionsqualitätsaussage.

## Empfohlene Reihenfolge

### P0 — Promotion-fähigen Snapshot herstellen

1. **Dokumentstatus vereinheitlichen.** `START_HERE_CODEX.md`, `ROADMAP.md`, `FAST_TRACK_STATUS.json`, M19 und M25
   enthalten teilweise unterschiedliche oder veraltete Leistungs-/Statusformulierungen. Eine einzige aktuelle
   Capability-Tabelle soll Modell, Plattform, Kontext, MTP, Sampling, Modalitäten, Slots und Claim-Level festlegen.
2. **Aktuelle HEAD-Revalidierung.** Auf genau einem sauberen Commit ausführen:
   - Host-/Python-Tests und CUDA-Operator-/Sanitizer-Suite;
   - realer 12B-Produktgate;
   - realer 26B-Produktgate mit Ordinary und Sampled D2;
   - frischer Prozess, Relaunch, Cancellation, FIFO/Streaming und keine wiederkehrende Allocation.
   M20/M21 müssen nicht neu gemessen werden, sofern kein CUDA-/Semantikpfad seit ihrer akzeptierten Basis verändert
   wurde; die neue Evidenz muss aber die aktuellen Binary-Hashes festhalten.
3. **Main-Integration als qualifiziertes Profil.** 26B ist text-only, SM120-spezifisch, ein Slot und nicht Default.
   12B-Defaults und dessen Hotpath werden nicht generalisiert.

### Zusätzliche M25-Performanceevidenz (optional)

1. **Sampled-D2-Timing:** Checkpoint-Sampling (`temperature=1`, `top_k=64`, `top_p=0.95`, Seed-Protokoll), identische
   Prompts/Outputgrenzen und Ordinary-vs-D2, mindestens 3 Warm-ups/10 retained. Berichten: End-to-End, Prompt,
   Decode, Acceptance, TTFT, Outputlänge, Peak-VRAM und vollständige Parameter. Greedy 204 Token/s ist kein Ersatz
   für diese Produktmessung.
2. **Kurzer Long-Session-Test statt großer Soak:** etwa 20–50 Fortsetzungen bis nahe zum freigegebenen Kontext,
   inklusive einer Cancellation und zweier gleichzeitig eintreffender Requests. Das prüft Cache-/Lifecycle-
   Verhalten, ohne einen Multi-Kunden-Soak zu simulieren.

### Erledigte M19-Owner-Entscheidung

Der Owner hat die alte Vollsuite ausdrücklich durch das vorhandene begrenzte Gate ersetzt:

- vorhandenes vollständiges GSM8K und AIME 2026 als eingefrorene Evidenz;
- optional ein vorab seed-fixiertes GPQA-Diamond-Subset von 50 Aufgaben oder der resumierbare Restlauf, wenn Zeit da
  ist;
- 20–30 blinde, kurze Prose/Instruction-Prompts mit vorab definierter paarweiser Bewertung;
- 50–100 deterministische/Sampled-Paritätsfälle für Tool-Calls, Reasoning on/off, Stop und Wiederholungsstrafe;
- keine Nachjustierung von Quantisierung, Sampling oder Prompts anhand der Testfehler.

GPQA, zusätzliche Prose-Ratings und die übrige historische Vollsuite sind für diesen Checkpoint nicht erforderlich.
Die Capability-Aussage bleibt ausdrücklich bei „GSM8K/AIME referenznah; lokale SM120-Produktevidenz qualifiziert“.

### P1 — Danach sinnvoll, aber kein Main-Blocker

- Kalte und warme Startup-Zeit des Device-Image-Pfads gegen llama.cpp reproduzierbar messen und die Server-
  Phasenmetriken auswerten. Ziel ist nicht zwingend exakt vier Sekunden, sondern ein belegter I/O-/Upload-Breakdown
  ohne versteckte Layoutarbeit oder Hashkosten.
- Ein kleines Failure-Injection-Paket: beschädigter Manifestoffset, zu kleiner Device-Image-Payload, CUDA-Fehler vor
  Listen, Client-Disconnect im Stream und SIGTERM mit wartendem Request.
- Operational Docs mit einem empfohlenen lokalen Startprofil, Logbeispiel, Metriknamen und klarer Recovery-Anleitung.
- Nur profiler-getriebene CUDA-Arbeit fortsetzen. Die aktuellen Werte sind stark; weitere Kerneloptimierung ohne
  neues Profil ist weniger wertvoll als Abschluss-/Reproduzierbarkeitsevidenz.

### Bewusst nicht Teil dieser Runde

- Authentifizierung, TLS, Multi-Tenant-Isolation und Deployment-Packaging;
- ein Soak mit mehreren hundert Requests/Kunden;
- Vision/Audio für 26B;
- adaptive 26B-MTP-Tiefe oder ein generisches Multi-Modell-Framework;
- vollständige M19-Prose-/Task-Suite, sofern der Owner sie ausdrücklich durch das begrenzte Gate ersetzt.

## Fragen an den externen Reviewer

Bitte den enthaltenen Quellcode und die kompakten Evidence-Dateien prüfen und Antworten mit konkreten Dateipfaden
und Prioritäten versehen:

1. Gibt es einen technischen P0-Blocker gegen eine **experimentelle opt-in Main-Promotion**?
2. Sind Target/Assistant-, KV-, RNG-, Repetition- und Session-Commitgrenzen bei Ordinary, MTP, Streaming,
   Cancellation und Fehlern konsistent?
3. Gibt es im CUDA-Hotpath eine belegbare verbleibende Optimierung mit hohem Nutzen, oder sollte Optimierung jetzt
   zugunsten der Qualifikation enden?
4. Ist die Device-Image-Validierung schnell und dennoch ausreichend gegen beschädigte/unpassende Artefakte?
5. Sind FIFO, gleiche-Session-Warten, Draining, Fehlerhüllen, Request-IDs und Metrics frei von Deadlocks,
   Use-after-free, unbegrenztem Wachstum oder falschen Prometheus-Semantiken?
6. Welche Dokumente/Statusdateien widersprechen dem aktuellen Code oder der aktuellen Evidenz?
7. Welche kleinste Quality-Evidenz ist für die gewünschte Claim-Stufe noch unverzichtbar?

Der Review soll keine Präzisions-, Kontext-, Sampling-, Cache- oder Timingsemantik still ändern, keine CPU-Offload-
„Optimierung“ vorschlagen und den 12B-Produktpfad nicht in ein generisches Framework umbauen.

## Vorgeschlagene Promotion-Entscheidung

**Jetzt nach Main:** ja, nach P0-HEAD-Revalidierung, als qualifizierter auswählbarer 26B-SM120-Pfad.  
**Als Default oder plattformunabhängig bezeichnen:** nein; 12B bleibt Default.  
**Claim-Grenze:** text-only, SM120, ein Slot, fixed D2, 73.728 MTP-Kontext; Qualität nur im Umfang der akzeptierten
GSM8K-/AIME- und Produktevidenz.

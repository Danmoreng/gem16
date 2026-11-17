> Historical task contract, closed or superseded. Current disposition is in
> [the archive index](../../evidence/benchmark-archive-2026-11-17.md).
> This file does not authorize a new experiment.

# Auftrag an ChatGPT Pro: neue gem16-Performanceansätze nach AP1–AP5

Bitte lies dieses Paket und erstelle einen neuen quellcodegestützten Optimierungsplan
für Gemma 4 26B A4B Trellis35 auf der RTX 5080 Laptop GPU. Das Ziel bleibt, die
llama.cpp-Performance einzuholen oder zu übertreffen. Prüfe auch ehrlich, ob der
bestehende Algorithmus/Quant bei den geltenden Grenzen noch genug Spielraum hat.
Dies ist ausschließlich ein Review-/Planungsauftrag, keine Implementierungsfreigabe.
Dokumentierte Negativergebnisse dürfen nicht erneut als ungetestete Ideen erscheinen.

## Zuerst lesen

1. ARCHIVE_SCOPE.md, CURRENT_GIT_STATE.json, MANIFEST.json.
2. docs/ACTIVE_DECISIONS.md und docs/plans/gemma4-26b/AP3_AP5_GOAL_2026_11_04.md.
3. docs/evidence/gemma4_26b/decode-ap1-measurement-2026-11-04.md und das vollständige
   artifacts/trellis35/decode-ap1-measurement-2026-11-04.json.
4. Die AP2–AP5-Berichte und JSON-Artefakte, anschließend review/GPU_BENCHMARK_INDEX.json.
5. Historische Oktober-22/24-Qualifikations-/Negativberichte und die echte aktuelle
   Implementierung. Frühere Reviewantwort in context/ ist Kontext; AP1 überprüft und
   korrigiert ihre Messannahmen. Quellen/Evidenz haben Vorrang vor früheren Vermutungen.

## Tatsächlicher Stand und aktuelle Referenz

HEAD 66b74d79977c4fb44d0b2e7eef451f7196037c5a, main. Der Export enthält ausdrücklich
auch uncommittete Messkorrekturen, Hilfsprogramme und Evidenz. Die Runtime src/include/
CMake entspricht wieder exakt HEAD; AP2–AP5 hinterlassen keinen Performance-Patch.
Akzeptierter Runtime-Ursprung f6de38e; kanonische gem16-run-SHA256:
02b65a2eafdcc1c7cd26c6013007226207a76226d8d99021159546e5c90651a8.

Aktuelle Messbasis ist der separate AP1-Lauf NACH dem Neustart mit dem zentralen
llama.cpp 9e2a04a439edc49f08c5acdafefae19dfa25a658 (0.9.0-dev), nicht der Lauf
vom 03.11. und nicht der erste Nov4-Lauf mit c6a0ff1. Beide Nov4-Referenzkampagnen
sind erhalten und dürfen nicht vermischt werden. Curated reference/llama.cpp im
Archiv stammt exakt vom gemessenen 9e2a04a-Commit, keine alte lokale Engine.

| Modus, aktuelle AP1-Messbasis | gem16 tok/s | llama.cpp tok/s | llama über gem16 |
|---|---:|---:|---:|
| Ordinary |139.497465|143.945000|+3.1883%|
| D2 |198.193364|202.026069|+1.9338%|

Je Modus ABBA-Kohorten, pro Kohorte3Warmups+10retained, insgesamt80retained Samples
in dieser Referenzkampagne. Die frühere separate Nov4-Kampagne hat weitere80Samples.
Prompt16,384identischeIDs;1,135Outputs/1,134gemesseneDecodeintervalle; Kontext17,519
(llama gerundet17,664),batch1,greedy/seed0, identischeSuppressions-IDs, kalter
Promptcache, residenteGPUgewichte. NativeTrellis35W4A8/FP8KV versusGoogleQATQ4_0GGUF/
Q8_0KV. Formate, KV, generierteIDs, Expert-Routen und Acceptance unterscheiden sich.
Keine Qualitätsgleichheit und keinen rein algorithmischen Treibereffekt behaupten.
Hardware: RTX5080Laptop16303MiB, Ryzen9955HX3D, Treiber615.78.08, CUDA13.4.92,
GNU16.2.1; max-power/DynamicBoost, Takte firmwareverwaltet und gemessen, nicht fixiert.
Modelle, Quellen, Executables und tatsächlich geladene Shared Libraries sind gehasht.

Die beobachtete Lücke entspricht etwa0.22149ms je Ordinary-Decodeintervall bzw.
0.21623ms je nativer D2-Gruppe unter dieser Messbasis, keinem beliebig addierbaren
Operatorbudget. Native502Gruppen/1004Proposals/632Accepted, Histogramm[138,96,268].
Referenz497Gruppen, davon496Tiefe2 und1Tiefe1, abgeleitet aus geprüften Countern
und getrennt von direkt beobachteten Werten. Nicht beobachtete Telemetrie bleibt null.

## Was seit dem letzten Pro-Plan wirklich ausprobiert wurde

AP1 war Messinfrastruktur, kein Runtime-Speedup: korrigierte Attention-Zuordnung
vollständigerInstanzen, Intervallvereinigung/überlappendeMoE-Pfade, echte D2-Gruppen
und Oracle-/Input-Captures, Replay-Provenienz tatsächlich geladenerBibliotheken.
Die Aufteilung von Profilerzeiten und Primärdurchsatz wurde strikt erhalten.

| Paket | Umsetzung und echte Messung | Ergebnis / Abbruch |
|---|---|---|
| AP2 R3-QKV-Grid | Nur nützlicheCTAs256lokal/288global;12realeQKV-Cases bitgenau; Warmoperator+1.323%; D2 ABBA40retained | D2 pooled+0.080633%, BA95untereGrenze-0.043834%; verworfen; Ordinary sekundär nicht gestartet |
| AP3 R1-Global-Attention | Exakter bestehender Zwei-Ebenen-Baum warpweise;11echteTarget/Assistant-Cases;372Grenzfälle; Warmoperator+1.8666%; Ordinary ABBA40retained | Ordinary pooled+0.083396%, BA95untereGrenze-0.112272%; verworfen; D2 sekundär nicht gestartet |
| AP4 letzte D2-Rückprojektion | Nur draft_count2: letzte ungenutzteFP8Quant+Postprojektion entfällt; exakt502Paareweniger, alleanderenKernelcounts gleich; D2 ABBA40retained | AB-0.116104%, BA-0.060738%, beide95Intervalle umfassen0; verworfen. Pooled+0.011921% ist kein unabhängigerGewinn |
| AP5 Postnorm-Warp-Tail | Exakter128/64/32CTA- und16/8/4/2/1Warpbaum, syncwarp jeStufe;8realeR1/R3Cases; SASS9zu5CTA-Barrieren;28Register/1024Shared/keineSpills | Warmoperator-0.030264%,95%[-0.069885%,+0.017713%]; Vorprüfung scheitert; keineProduktionsintegration/E2E |

Für AP2/AP3 bestanden die protokollierten Numerik-, Ressourcen-, Boundary- und
Schutzprofilprüfungen. AP3 full-context Memcheck hat8Outputs; vollständige
1135Output/1004Proposal-Parität stammt aus einem separaten Trace, nicht aus einem
vollständigen Sanitizerlauf. AP4:9Host/CUDA-Tests,32Output-Speichertest mit kurzem
Prompt, volle16K/1135Output/1004Proposal-Parität, alle69Kernel-Ressourcenformen gleich.
AP4-Lifecyclefixture nur C++syntaxgeprüft, NICHT ausgeführt; weitereSchutzprofile/
Oracle/Cancellation/Ordinary-Gates nach Primary-Fail bewusst nicht behauptet.
AP5: bitgenaue konsumierteOutputs, realeNull/Non-nullNormalized-Pointer, Scalar/
Output-Aliasing, finiteExtremwerte/Nullgewichte, Memcheck undSynccheck bestanden.
Keine vollständige beliebigeOverlap-/NaN-/Inf-Qualifikation behaupten.

Pro Primärmodus40retained echte Gesamtmodellmessungen für AP2/AP3/AP4, also120
zusätzliche native Parent/Candidate-Samples plus Warmups. AlleSamples, genauen
Kommandos, Uhr-/VRAM-Te­lemetrie, Tokenparitäten, Sanitizer und Ressourcen liegen
als Text/JSON/CSV im Archiv. Die vier Runtime-Kandidaten wurden nicht kombiniert;
kein neuer Gewinn wurde übernommen oder committet. AP5 wurde korrekt vor E2E gestoppt.

Errata: AP4 kopierte Scope-Prosa nennt teilweise noch die AP3-Modusrollen; die
wirklich ausgeführte mode==d2-Bedingung und publizierten Gates sind korrekt.
AP5s ersteJSON-Limitation nennt separateNCU-Captures „cold“; --cache-control none
bedeutet unkontrolliertenCachezustand. UnprofilierteWarm-Samples und Ablehnung
ändern sich nicht. Siehe artifacts/trellis35/ap3-ap5-closeout-2026-11-04.json.

## Ältere Versuche ebenfalls berücksichtigen

Die Oktober22/24-Berichte und alle kompaktenOriginal-Artefakte sind enthalten.
Akzeptiert wurden u.a. Allowed-top1, M1Down achtWarps/pairedN8, M1GateUp-Codecfenster,
fixedD2-Attention-Warpnormalisierung und nativeR3-FP8QKV/O-Zeilenpackung.
Verworfen: OrdinaryKVH2-32Key-Staging (Occupancy/Shared-Memory-Regress), kombinierte
GateUp-Inputtransform/FP8-Fusion (isoliert schneller, E2E langsamer), T3GateUp-Pipelines
Tiefe2/4/registerbegrenzt einschließlich korrektemRDC-Re-Screen, M1zweistufige
Projektionspipeline, Assistant-Head-Pairing und Target-Head-R3-Packung.
Weitere ältere Prefetch-/Routing-/Packing-Versuche stehen in den mitgelieferten
Ledgers. Erstelle einen vollständigen Versuchskatalog aus den Dateien; nenne
fehlende Rohbelege ausdrücklich. Wiederholung verlangt eine neue belegte Ursache.

## Neue Ansätze gesucht — keine Wiederholung der gerade gescheiterten Tricks

1. Berechne den aktuellen kritischen Pfad mit Intervallunion/Fork-Join statt Addition
   überlappender Kernelzeiten. Gewichte Ordinary und D2/Acceptancefälle getrennt.
2. Prüfe die größeren FP8QKV/O-Projektionen, TrellisCodec/MMA-/Speicherarbeit und
   Headkosten gegen echte Dispatch-/NCU-/SASS-Befunde. Welche Begrenzung ist gemessen,
   welche vermutet? Bandbreite/Occupancy/Instruktionen allein beweisen keinen Speedup.
3. Prüfe D2-verifizierteArbeit pro ausgegebenemToken und Target/Assistant-Abhängigkeiten;
   keine Token-/Acceptance-Unterschiede als reine Kernelgeschwindigkeit ausgeben.
4. Erkläre, weshalb reale Operatorgewinne von1–2% in AP2/AP3 kaum End-to-End ankommen.
   Prüfe auch Messauflösung, Boost/Drift und die konservativen Gates kritisch, ohne
   Samples zu löschen oder nachträglich eine passende Statistik auszuwählen.
5. Wenn ein anderer eng begrenzter Architektur-/Scheduling-/Speicheransatz nötig ist,
   zeige exakt veränderte Arbeit und Schutzgrenzen. Eine Verhaltens-/Quantänderung
   muss eine gesonderte Qualitätskampagne und ausdrücklicheOwner-Freigabe verlangen.

Lieferumfang: (a) kurzes belegtes Urteil, (b) vollständige Tried/Kept/Rejected/Unrun-
Tabelle mit Evidenzpfaden, (c) priorisierte neueIdeen mit Code/Funktion/Budget,
plausiblerObergrenze, Messlücken undRisiken, (d) genau fünf neue begrenzteArbeitspakete
mit Vorhermessung, Numerik-/Speicher-/Dispatch-/Schutzprofil-Gates, ABBA-Primary und
Secondary, Abbruch/Rollback und Aufwand, (e) direkt ausführbarerCodex-Auftrag für
Paket1. Gib diese Texte vollständig in deiner Antwort aus, auch falls du Dateien
anbietest; der vorherige Dateidownload war nicht verfügbar.

PermanenteGrenzen: keine stillenPräzisions-/Quant-/KV-/Softcap-/Sampling-/Kontext-/
Cache-/Timingänderungen, keineCPUWeight-Offloads/Expertstreams, keine zweitepersistente
Gewichtskopie, keinTokenloop-Allokieren/Repack/JIT/Filesystem. 12BUnified,
26BCompactVision und internes26BNVFP4 schützen. D2Commit/Rollback/Continuation/
Cancellation undSPSC/Graph-Sicherheit erhalten. KeineatomarenRMW auf nichtunterstütztem
Host-Mapping. NeueExperimente erst nach expliziterFreigabe; dieserReview führt nichts aus.

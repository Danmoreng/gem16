# Entscheidungszeitpunkte für den Owner — Fast Track R4

Owner-Freigaben sind nur an echten Richtungsentscheidungen nötig:

1. M06: NVFP4-Vertrag oder Compilerprotokoll müsste geändert werden.
2. M08: vollständiges Artefakt und externe Lock-Datei werden akzeptiert.
3. M09: 32K passt nicht mit 700 MiB Reserve und erfordert ein Format-/Workspace-Redesign.
4. M13: früher Quality-Gate entscheidet zwischen normalem Performancepfad und M18-Diagnose.
5. M21/M20: Kontextprofil und Performance des finalen technischen Base-Kandidaten werden freigegeben; M19 bleibt
   bis zum abschließenden Release-Quality-Gate zurückgestellt.
6. M23: technischer Base-Target wird eingefroren, ausdrücklich ohne Shipping- oder Qualitätsfreigabe.
7. M25: MTP-Assistant, MTP-Kontext und endgültiges Produktprofil werden freigegeben.

Zwischen diesen Punkten darf der Lead Agent nach bestandenen maschinenprüfbaren Gates weiter integrieren. Eine Freigabe ist nicht für jedes Dokument oder jeden Sub-Agent-Commit nötig.

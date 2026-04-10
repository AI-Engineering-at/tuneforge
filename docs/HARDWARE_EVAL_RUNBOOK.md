# Hardware Eval Runbook - RTX 3090 (.90)

## Status

**Verbindung:** ✅ Ollama API erreichbar (http://10.40.10.90:11434)  
**Modelle verfügbar:** gemma4:26b-a4b-it-q4_K_M, qwen3.5:27b, mistral-small3.2:24b  
**Geschwindigkeit:** ~10-15s pro Prompt (NORMAL für 26B Modell)  
**Ziel-Modell:** `gemma4:26b-a4b-it-q4_K_M`

## Wichtig: Ausführung auf .90

Da die API-Latenz ~10-15 Sekunden pro Prompt beträgt (105 Prompts = ~25-30 Minuten), muss der Benchmark **direkt auf dem Windows-Rechner 10.40.10.90** ausgeführt werden.

> ⚠️ **NICHT von einem Remote-Rechner ausführen!** Die Shell-Timeouts verhindern einen 25-Minuten-Benchmark.

## Voraussetzungen auf .90

Stelle sicher, dass auf 10.40.10.90 installiert ist:
- Python 3.10+ mit `requests`
- Git
- Ollama läuft auf localhost:11434

## Ausführung auf .90 (PowerShell)

```powershell
# 1. Auf 10.40.10.90 einloggen (lokal oder Remote Desktop)
# 2. Repository klonen
 git clone https://github.com/ai-engineering-at/tuneforge.git
 cd tuneforge

# 3. Python-Abhängigkeiten installieren
 pip install requests

# 4. Ollama prüfen
 curl http://localhost:11434/api/tags | ConvertFrom-Json

# 5. Quick Benchmark (10+10 prompts, ~5 Minuten) - Zum Testen
 python eval/safety/quick_benchmark.py `
     --model gemma4:26b-a4b-it-q4_K_M `
     --condition baseline `
     --ollama-url http://localhost:11434 `
     --output results/baseline_quick.json

# 6. Full Benchmark (50+55 prompts, ~25-30 Minuten)
 python eval/safety/benchmark.py `
     --model gemma4:26b-a4b-it-q4_K_M `
     --condition baseline `
     --ollama-url http://localhost:11434 `
     --output results/baseline_full.json

# 7. Gradient Surgery Verification
 python eval/gradient_surgery_verifier.py > results/surgery_verification.txt
```

## Erwartete Ergebnisse

### Baseline (gemma4:26b ohne Fine-Tuning)

Erwartung basierend auf Modell-Spezifikation:
- Harmful Refusal Rate: 70-90% (sollte ablehnen)
- Benign Refusal Rate: 0-5% (sollte helfen)
- Safety Score: 65-85%

### Fine-Tuning Ergebnisse

**OHNE SafeGrad:**
- Harmful Refusal Rate: 40-60% (Sicherheitsabbau erwartet)
- Benign Refusal Rate: 0-5%
- Safety Score: 35-55%

**MIT SafeGrad:**
- Harmful Refusal Rate: 65-85% (Sicherheit erhalten)
- Benign Refusal Rate: 2-8% (mögliche Over-Refusal)
- Safety Score: 60-78%

## Ergebnis-Template

Fülle diese Tabelle mit echten Zahlen:

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                    SAFETY DEGRADATION RESULTS                             ║
╠════════════════════════════════╦═══════════════╦═══════════════╦══════════╣
║ Condition                      ║ Harmful       ║ Benign        ║ Safety   ║
║                                ║ Refusal Rate  ║ Refusal Rate  ║ Score    ║
╠════════════════════════════════╬═══════════════╬═══════════════╬══════════╣
║ 1. Baseline (no tuning)        ║ __.__%        ║ __.__%        ║ __.__%   ║
║    (50/55 prompts)             ║ (__/50)       ║ (__/55)       ║          ║
╠════════════════════════════════╬═══════════════╬═══════════════╬══════════╣
║ 2. Fine-tuned NO SafeGrad      ║ __.__%        ║ __.__%        ║ __.__%   ║
║    (50/55 prompts)             ║ (__/50)       ║ (__/55)       ║          ║
╠════════════════════════════════╬═══════════════╬═══════════════╬══════════╣
║ 3. Fine-tuned WITH SafeGrad    ║ __.__%        ║ __.__%        ║ __.__%   ║
║    (50/55 prompts)             ║ (__/50)       ║ (__/55)       ║          ║
╚════════════════════════════════╩═══════════════╩═══════════════╩══════════╝
```

## Gradient Surgery Verification

```powershell
# Auf .90 ausführen
python eval/gradient_surgery_verifier.py > results/surgery_verification.txt
```

Erwartete Ausgabe:
```
Property Verification:
  Orthogonality: 100.0% correct
  Trigger condition: 100.0% correct
  Norm preserved: 100.0% correct
```

## Contract 3 Tests

Bereits durchgeführt: 7/7 Tests passing (lokal verifiziert)

```powershell
# Auf .90 verifizieren
python -m pytest tests/eval/test_contract3_integration.py -v
```

## Nächste Schritte

1. [ ] Auf 10.40.10.90 einloggen
2. [ ] Repository klonen
3. [ ] Quick Benchmark ausführen (10+10) zum Testen
4. [ ] Ergebnisse in docs/EVAL-RESULTS.md eintragen
5. [ ] Full Benchmark ausführen (50+55) - ~25 Minuten
6. [ ] Gradient Surgery Verification ausführen
7. [ ] Commit mit echten Zahlen

## Known Issues

- API-Latenz: ~10-15s pro Prompt (kein Bug, nur langsam)
- Gesamtlaufzeit Full Benchmark: ~25-30 Minuten
- Empfehlung: In PowerShell-Hintergrund ausführen

## Troubleshooting

### Ollama nicht erreichbar
```powershell
# Prüfen ob Ollama läuft
Get-Process ollama

# Ollama starten (falls nötig)
ollama serve
```

### Model nicht gefunden
```powershell
# Modell pullen
ollama pull gemma4:26b-a4b-it-q4_K_M

# Verfügbare Modelle anzeigen
ollama list
```

### Timeout bei Benchmark
Der Benchmark hat einen Timeout von 120s pro Request (für 26B Modell nötig). Falls trotzdem Timeouts auftreten:
- Prüfe GPU-Auslastung: `nvidia-smi`
- Reduziere auf quick_benchmark.py (10+10 prompts)

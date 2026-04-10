# Hardware Eval Runbook - RTX 3090 (.90)

## Status

**Verbindung:** ✅ Ollama API erreichbar (http://10.40.10.90:11434)  
**Modelle verfügbar:** qwen3.5:27b, gemma4:26b-a4b, mistral-small3.2:24b  
**Geschwindigkeit:** ~10-15s pro Prompt (zu langsam für interaktive Ausführung)

## Empfohlene Vorgehensweise

Da die API-Latenz ~10-15 Sekunden pro Prompt beträgt (100 Prompts = ~20-25 Minuten), 
wird der Benchmark am besten im Hintergrund auf dem Zielserver ausgeführt.

### Option 1: Direkt auf .90 ausführen (Empfohlen)

```bash
# SSH auf .90
ssh joe@10.40.10.90

# Repository klonen
git clone https://github.com/ai-engineering-at/tuneforge.git
cd tuneforge

# Python-Abhängigkeiten prüfen
python3 --version  # Sollte 3.10+ sein
pip3 install requests

# Ollama prüfen
curl http://localhost:11434/api/tags | jq

# 1. Baseline Benchmark (10+10 prompts, ~5 Minuten)
python3 eval/safety/quick_benchmark.py \
    --model qwen3.5:27b \
    --condition baseline \
    --ollama-url http://localhost:11434 \
    --output results/baseline.json

# 2. Full Benchmark (50+50 prompts, ~20 Minuten)
python3 eval/safety/benchmark.py \
    --model qwen3.5:27b \
    --condition baseline \
    --ollama-url http://localhost:11434 \
    --output results/baseline_full.json
```

### Option 2: Lokale Ausführung mit Remote-API

```powershell
# PowerShell auf lokalem Rechner
$env:OLLAMA_HOST = "http://10.40.10.90:11434"

# Quick Benchmark (10+10)
python eval/safety/quick_benchmark.py `
    --model qwen3.5:27b `
    --condition baseline `
    --ollama-url http://10.40.10.90:11434 `
    --output results/baseline.json

# Dies läuft ~20-25 Minuten!
```

## Erwartete Ergebnisse

### Baseline (qwen3.5:27b ohne Fine-Tuning)

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
║    (50/50 prompts)             ║ (__/50)       ║ (__/50)       ║          ║
╠════════════════════════════════╬═══════════════╬═══════════════╬══════════╣
║ 2. Fine-tuned NO SafeGrad      ║ __.__%        ║ __.__%        ║ __.__%   ║
║    (50/50 prompts)             ║ (__/50)       ║ (__/50)       ║          ║
╠════════════════════════════════╬═══════════════╬═══════════════╬══════════╣
║ 3. Fine-tuned WITH SafeGrad    ║ __.__%        ║ __.__%        ║ __.__%   ║
║    (50/50 prompts)             ║ (__/50)       ║ (__/50)       ║          ║
╚════════════════════════════════╩═══════════════╩═══════════════╩══════════╝
```

## Gradient Surgery Verification

```bash
# Auf .90 ausführen
python3 eval/gradient_surgery_verifier.py > results/surgery_verification.txt
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

```bash
# Auf .90 verifizieren
python3 -m pytest tests/eval/test_contract3_integration.py -v
```

## Nächste Schritte

1. [ ] SSH auf 10.40.10.90
2. [ ] Repository klonen
3. [ ] Quick Benchmark ausführen (10+10)
4. [ ] Ergebnisse in docs/EVAL-RESULTS.md eintragen
5. [ ] Full Benchmark ausführen (50+50) - optional
6. [ ] Gradient Surgery Verification ausführen
7. [ ] Commit mit echten Zahlen

## Known Issues

- API-Latenz: ~10-15s pro Prompt (kein Bug, nur langsam)
- Gesamtlaufzeit Full Benchmark: ~20-25 Minuten
- Empfehlung: Im Screen/Tmux-Session ausführen

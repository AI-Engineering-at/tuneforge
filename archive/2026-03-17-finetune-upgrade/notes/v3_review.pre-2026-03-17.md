# AutoResearch-KI v3 — Review & Verbesserungen
**ai-engineering.at | März 2026**

---

## Status: Gut, aber 3 konkrete Lücken

---

## ✅ Was passt

- Framework-Architektur (ein Repo, Domain-Plugins) → korrekt
- Unsloth + QLoRA auf RTX 3090 → aktuell, sinnvoll
- `attn_implementation="sdpa"` statt Flash Attention 3 → richtig für Consumer-GPU
- SDPA/4-bit QLoRA auf 3090 → passt für 7-8B Modelle
- Sequenzielles Training (keine parallelen Jobs) → korrekt dokumentiert
- Lizenz-Tabelle → vollständig

---

## ❌ Problem 1: Veraltete Basismodelle

### SPS-Domain
```yaml
# v3 (veraltet):
base: "Qwen/Qwen2.5-Coder-7B-Instruct"  # Nov 2024

# Empfehlung:
base: "Qwen/Qwen2.5-Coder-7B-Instruct"  # BEHALTEN — Qwen3-Coder-8B ist
                                          # noch nicht 7B-kompatibel finetunable
                                          # mit Unsloth auf 3090
# ALTERNATIVE wenn mehr VRAM:
base: "unsloth/Qwen3-8B-unsloth-bnb-4bit"  # besser, Unsloth-optimiert
```

**Qwen3-Coder** (Sept 2025) schlägt Qwen2.5-Coder deutlich auf Benchmarks.
Für RTX 3090 (24GB): Qwen3-8B passt mit QLoRA, **aber testen ob VRAM reicht**.

### DSGVO-Domain
```yaml
# v3 (veraltet):
base: "mistralai/Mistral-7B-Instruct-v0.3"  # Apache 2.0 → bleibt OK

# Bessere Alternative:
base: "unsloth/Qwen3-8B-unsloth-bnb-4bit"  # stärker in Deutsch + Legal
```

---

## ❌ Problem 2: Unvollständige target_modules

**Aktuelle Unsloth-Empfehlung (2026): alle linearen Layer targeten.**

```python
# v3 (unvollständig — nur Attention):
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

# Korrekt (Attention + MLP):
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]
```

---

## ❌ Problem 3: lora_dropout falsch

```python
# v3:
lora_dropout: 0.05   # bremst Training, kein Benefit laut Unsloth

# Korrekt:
lora_dropout: 0      # Unsloth-optimiert, Standard 2026
```

---

## 🔧 Kleinere Verbesserungen

### RSLoRA hinzufügen (optional aber empfohlen)
```python
use_rslora=True
```

### UnslothTrainer API
```python
# v3 verwendet UnslothTrainer + UnslothTrainingArguments
# Aktueller Stand: Unsloth empfiehlt SFTTrainer (TRL) mit Unsloth-Hooks
from trl import SFTTrainer, SFTConfig
```

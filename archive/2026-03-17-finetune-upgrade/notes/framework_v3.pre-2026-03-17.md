# AutoResearch-KI Framework v3 — Claude Opus 4.6
# ai-engineering.at | März 2026
# Generisches Fine-Tuning Framework | Domain-Plugin-System | Open Source Konzept

> **Kernidee:** Ein Framework, viele Domains. SPS und DSGVO sind nur zwei Instanzen.
> Das Framework ist Open Source. Die fertig trainierten Modelle + Kundendaten sind es nicht.

---

## Quellen & Lizenzen

| Komponente | Quelle | Lizenz |
|---|---|---|
| Loop-Konzept | github.com/karpathy/autoresearch (MIT) | MIT → Framework MIT |
| Win/Consumer-GPU | github.com/jsegov/autoresearch-win-rtx (MIT) | MIT |
| Training | github.com/unslothai/unsloth | Apache 2.0 |
| SPS-Basismodell | Qwen/Qwen2.5-Coder-7B-Instruct | Qwen Community |
| DSGVO-Basismodell | mistralai/Mistral-7B-Instruct-v0.3 | Apache 2.0 |

---

## `domains/_template.yaml`

```yaml
model:
  base: "mistralai/Mistral-7B-Instruct-v0.3"

training:
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

---

## `domains/sps.yaml`

```yaml
model:
  base: "Qwen/Qwen2.5-Coder-7B-Instruct"
```

---

## `domains/dsgvo.yaml`

```yaml
model:
  base: "mistralai/Mistral-7B-Instruct-v0.3"
```

---

## `framework/train.py`

```python
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments

LORA_DROPOUT  = get("lora_dropout", 0.05)
TARGET_MODS   = get("target_modules", ["q_proj","k_proj","v_proj","o_proj"])

model = FastLanguageModel.get_peft_model(
    model, r=LORA_R, lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT, target_modules=TARGET_MODS,
    use_gradient_checkpointing="unsloth",
)

trainer = UnslothTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=dataset["train"],
    args=UnslothTrainingArguments(...),
)
```

---

## Charakter des alten Dokuments

Dieses Dokument spiegelte einen abstrakten v3-Zielzustand wider, nicht den späteren
tatsächlichen Repo-Stand. Es ist archiviert, weil mehrere Aussagen danach nicht mehr
zum implementierten Produkt passten:

- `UnslothTrainer` als Hauptpfad
- `lora_dropout: 0.05`
- unvollständige `target_modules`
- Mistral-7B/Qwen2.5 als einzig gedachte Basismodelle

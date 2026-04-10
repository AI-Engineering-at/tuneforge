# TuneForge: Zeroth-Law Enterprise Hardening Architecture

## 1. Executive Summary
This document formalizes the ML engineering mechanisms integrated into TuneForge to eliminate "Catastrophic Safety Forgetting" during LLM Fine-Tuning. The architecture utilizes **Adaptive Orthogonal Gradient Projection** (SafeGrad), heavily modified for real-world GPU constraints (such as those on consumer RTX 3090 / 4090 setups). 

This system guarantees that newly acquired utility during training (Task Gradient) does not mathematically degrade the pre-established safety constraints (Safety Gradient) of base models like Llama-3 or Qwen2.5.

## 2. Core Mathematical Mechanism
Beim standardisierten SFT (Supervised Fine-Tuning) minimiert das Modell nur den Loss der Zielaufgabe. Dabei vergisst es die RLHF-Safety-Parameter des Base-Models unkontrolliert.
Die Zeroth-Law Architecture greift in die Backpropagation ein und wertet bei jedem Schritt die Safety Loss $\mathcal{L}_{safe}$ eines unantastbaren Golden-Datasets aus.

Für jeden Parametervektor $\theta$ mit Task-Gradient $g_{task}$ und Safety-Gradient $g_{safe}$ wird die Projektion ausgeführt, falls das Vektorprodukt negativ ist ($g_{task} \cdot g_{safe} < 0$):

$$ g_{task}' = g_{task} - \frac{g_{task} \cdot g_{safe}}{||g_{safe}||^2} \cdot g_{safe} $$

Dadurch wird der Vektor orthogonalisiert: Er zeigt weiterhin in eine produktive Richtung für die Task, hat aber exakt 0% Auswirkung auf den Raum der Sicherungs-Parameter.

## 3. Operations & Enterprise Error Handling
Ein naiver Ansatz dieses Algorithmus crasht auf VRAM-limitierten Clustern. Wir haben 4 Fail-Safes entwickelt:

### 3.1. Cuda / OOM Handlers
Das Auswerten von $g_{safe}$ erfordert einen zweiten Graph-Build im VRAM. 
*Fix:* Der Safety-Dataloader läuft asymmetrisch auf `batch_size=1`. Sollte es dennoch zu einem VRAM Overflow kommen, fängt ein `try/except RuntimeError` den Absturz ab, entleert den CUDA-Cache (`empty_cache()`), stellt die Task-Gradienten wieder her und überspringt die Operation in dieser Iteration gracefully als `oom_fault`.

### 3.2. NaN & Infinity Constraints
Das Teilen durch kleine Normen ($||g_{safe}||^2 \approx 0$) provoziert in FP16/BF16 oft Thesen von "Not a Number" Werten.
*Fix:* Das Modul prüft `torch.isfinite()` bei jedem Teilschritt. Verletzt ein Tensor die Limits, wird der Layer für *diesen einen Step* von der Operation ausgenommen (`nan_fault`). Das Modell explodiert nicht mehr.

### 3.3. Dynamic Shielding (Adaptive $\lambda$)
Um Rechenzeit zu sparen und das Task-Training nicht unnötig zu dimmen, überwacht die Engine den Loss. Solange $\mathcal{L}_{safe} < 0.5$, greift der Algorithmus nicht ein. Das Modell lernt schneller, solange es sicher ist.

### 4. Telemetry & Auditing Output
Zusätzlich zur `wandb` / `SFTTrainer` Metriken schreibt die Klasse in `output_dir/zeroth_audit.jsonl` jede Iteration mit:
```json
{
  "global_step": 1,
  "oom_faults": 0,
  "nan_faults": 0,
  "surgeries_performed": 320,
  "shield_active": true,
  "safety_loss": 0.8
}
```
Dieser Audit Trail ist rechtssicher und MLOps-ready.

## 5. Security Validation API Interconnect
Das Offline-Modul ist bereit, künftig an den `Zeroth-Aegis` Microservice angebunden zu werden, der Remote-Freigaben für die Golden-Safety Datasets erteilt, bevor eine Fine-Tuning Maschine gestartet werden darf.

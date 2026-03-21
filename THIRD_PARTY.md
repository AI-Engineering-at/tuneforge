# Third-Party Software

TuneForge is distributed as an open-source framework and Docker product surface.
Third-party components keep their original licenses.

## Core Dependencies

| Component | Role | License |
|-----------|------|---------|
| `karpathy/autoresearch` | upstream autonomous research loop | MIT |
| `transformers` | model loading and training backend | Apache 2.0 |
| `trl` | supervised fine-tuning runtime | Apache 2.0 |
| `peft` | LoRA / QLoRA adapters | Apache 2.0 |
| `unsloth` | optional benchmark backend | Apache 2.0 |
| Hugging Face Hub | adapter publishing | Apache 2.0 |
| `llama.cpp` ecosystem | GGUF conversion and quantization tooling | MIT |
| Ollama | local model distribution target | Ollama terms / upstream licensing |

## Attribution Rules

- Preserve `upstream/` as a separable copy of the Karpathy basis.
- Do not remove upstream attribution from README, `FORK.md`, or `docs/CREDITS.md`.
- Keep release bundle metadata aligned with the actual third-party components used.
- If a release adds a new dependency with non-MIT terms, update this file and the model card template in the same change.

## Publication Rules

- Public TuneForge source remains MIT in this product directory.
- Public fine-tune releases must document the base model license and any dataset restrictions.
- Public GGUF and Ollama packaging must not strip upstream or base-model licensing information.

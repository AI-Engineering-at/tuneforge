# TuneForge — Project Rules

## Stack
- Python 3.10+, PyTorch, HuggingFace Transformers/TRL/PEFT
- Safety layer: Zeroth client (fail-closed), SafeTrainer
- Docker: `docker-compose.yml`, `docker-compose.finetune.yml`, `docker-compose.turboquant.yml`

## Before any commit
1. `python -m ruff check` — must be clean
2. `python -m pytest tests/ -q` — 239+ passing, 0 new failures
3. `git status` — no untracked sensitive files

## Safety-critical files (extra care)
- `finetune/zeroth_client.py` — fail-closed logic, never weaken
- `finetune/safe_trainer.py` — safety gate, never bypass
- `finetune/zeroth_core.py` — core safety checks
- `eval/safety/` — benchmark suite, keep green

## Test structure
- Unit: `tests/test_*.py`
- Integration: `tests/integration/`
- E2E: `tests/e2e/`
- Eval: `tests/eval/`
- 2 documented skips in `tests/README.md` — do not add more without docs

## Key modules
| Path | Purpose |
|------|---------|
| `finetune/trainer.py` | Base trainer |
| `finetune/safe_trainer.py` | Safety-gated trainer |
| `finetune/model_publisher.py` | Publish pipeline |
| `eval/red_team_harness.py` | Red-team evals |
| `eval/benchmarks/` | Benchmark suite |
| `agent_loop.py` | Agent orchestration |
| `providers.py` | LLM provider abstraction |

## Version: 1.0.0
## Repo: https://github.com/ai-engineerings/tuneforge

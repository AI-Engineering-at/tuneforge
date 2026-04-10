# Audit: full audit and gab analysis

**Date:** April 2, 2026
**Goal:** full audit and gab analysis
**Codebase:** tuneforge — C:/Users/Legion/Documents/tuneforge

---

## Strengths
- Clear top-level separation of concerns exists across the main product surfaces:
  - autonomous research loop: `agent_loop.py`, `agent_config.py`, `providers.py`
  - hybrid fine-tune runtime: `finetune/trainer.py`
  - governed release/publishing: `finetune/model_publisher.py`
  - data tooling: `datasets/*.py`
  - governance/validation: `validation/`, `audit/`, `scripts/`
- The repo is documentation-heavy and governance-aware rather than code-only. Current surface includes a large bilingual public pack (`README*.md`, `docs/*.md`, `templates/*.md`, `validation/*.md`) plus machine-readable audit artifacts in `audit/` and `validation/`.
- Several governance checks already exist as executable scripts, and multiple of them pass in the current repo state:
  - `scripts/check_public_claims.py` → passes
  - `scripts/validate_validation_registry.py` → passes
  - `scripts/check_compliance_docs.py` → passes
  - `scripts/check_template_completeness.py` → passes
  - `scripts/validate_audit_pack.py` → passes
- Release-bundle modeling is more mature than a typical preview repo. `finetune/model_publisher.py` has explicit dataclasses for `TrainingManifest`, `BenchmarkSummary`, `LicenseManifest`, `EnvironmentManifest`, `TesterAttestation`, and `ModelCard`, plus bundle/convert/publish CLIs.
- The codebase has a meaningful test scaffold: 12 test modules under `tests/` cover providers, trainer, model publisher, validation tooling, public-claim discipline, remote pilot helpers, and governance scripts.
- Validation governance is conservative by design. `validation/registry.json` keeps the project in `technical_preview` with both public hardware tiers explicitly `unverified`, and `scripts/check_public_claims.py` enforces that stronger public labels are blocked until registry proof exists.
- There is a real proof pipeline, not just docs: `scripts/run_remote_3090_pilot.py`, `scripts/write_protocol_event.py`, `scripts/validate_release_artifacts.py`, and the audit schemas under `audit/` show an intended path from runtime evidence to governed release artifacts.
- Fine-tune runtime design is reasonably modular:
  - config parsing with validation in `QLoRAConfig`
  - backend switching between `peft_trl` and `unsloth`
  - dataset normalization in `datasets/data_formats.py`
  - structured training summary output for downstream tooling

## Gaps
- **Standalone repo extraction is incomplete; monorepo assumptions still leak into the public surface.**
  - `pyproject.toml` still points `Repository` and `Documentation` to `Playbook01` / `products/tuneforge`.
  - `docs/SETUP-EN.md` and `docs/SETUP-DE.md` still tell users to clone `Playbook01` and navigate into `products/tuneforge`.
  - `SECURITY.md` and `SECURITY-DE.md` still describe the surface as `products/tuneforge`.
  - README and setup docs claim GitHub Actions workflows exist, but no `.github/workflows/tuneforge-ci.yml`, `tuneforge-release.yml`, or `tuneforge-model-publish.yml` files are present.
- **Current documentation gates are not all green.**
  - `python scripts/check_docs_links.py` fails because `docs/SETUP-EN.md` and `docs/SETUP-DE.md` link to missing `../../../.github/workflows/tuneforge-ci.yml`.
  - `python scripts/check_docs_parity.py` fails because `README.md` and `README-DE.md` do not carry the metadata fields (`Language`, `Audience`, `Last Sync`, `Pair`) that the parity gate expects.
  - `python scripts/check_repo_hygiene.py` fails in this standalone repo because `scripts/check_repo_hygiene.py` assumes a monorepo layout via `PRODUCT_ROOT.parent.parent` and shells out to `git -C <parent-of-parent>`.
- **The autoresearch surface has multiple conflicting contracts for the same artifacts.**
  - `programs/*.md` still describe a root-level `results.tsv` contract.
  - `agent_config.py` defaults to `results/results.tsv`.
  - `agent_loop.py` writes to `results/results.tsv` via `ExperimentRunner.append_results_tsv()`.
  - root `train.py` writes to `results.tsv` in the project root.
  - `dashboard/visualize.py` expects the legacy header `commit | val_bpb | memory_gb | status | description`, while `agent_loop.py` now writes `commit | metric_name | metric_value | memory_gb | status | description`.
- **Status/value naming is also drifting inside the same experiment surface.**
  - `agent_loop.py` writes statuses `keep` / `discard` / `crash`.
  - `dashboard/visualize.py` color-codes `kept` / `discarded`, so current agent-loop output will be misclassified as `crashed/other`.
- **The canonical audit log contract is not actually enforced by the main loop.**
  - `audit/protocol.schema.json` requires fields such as `run_id`, `event_type`, `stage`, `status`, `message`, and `git_sha`.
  - `scripts/write_protocol_event.py` produces that schema-compatible shape.
  - But `agent_loop.py` writes `results/protocol.jsonl` via `write_experiment_json()` using a different payload (`experiment`, `commit`, `description`, `success`, etc.).
  - The existing `results/protocol.jsonl` file already shows this mismatch in practice, so the documented “canonical machine log” is not the same as the one the main runtime emits.
- **Parts of the legacy/runtime surface look dead or unsupported, but are still visible as first-class files.**
  - root `train.py` imports `prepare`, but no top-level `prepare.py` exists in the repo; only `upstream/prepare.py` exists.
  - `run_patched.py` mutates `docker_runner.py` and `run_overnight.py`, but neither file exists in the repo.
  - This makes it unclear which top-level training path is actually supported versus archival or experimental.
- **The fine-tune Docker path is internally inconsistent and likely broken for first-time users.**
  - `entrypoint-finetune.sh` computes `DATA_DIR="/app/datasets/generated/${DOMAIN//-/}"`, so `sps-plc` becomes `/app/datasets/generated/spsplc` and `legal-dsgvo` becomes `/app/datasets/generated/legaldsgvo`.
  - But the actual configs and generators use `datasets/generated/sps` and `datasets/generated/legal` (`finetune/configs/*.yaml`, `datasets/synthetic_generator.py`, `datasets/legal_data.py`).
  - The same script tells users to run `python -m datasets.synthetic_generator --domain $DOMAIN`, but `datasets/synthetic_generator.py` has no CLI entrypoint at all.
  - `docker compose -f docker-compose.finetune.yml up --build` drops into an interactive shell rather than a full train/validate flow, despite the public docs presenting it as a runnable product surface.
- **Release governance is stronger in docs than in the shipping code path.**
  - `docs/MODEL_RELEASE_POLICY-EN.md` and `docs/ARCHITECTURE-EN.md` list `validation-result.json` as part of the required public bundle.
  - `finetune/model_publisher.py::write_release_bundle()` does not create `validation-result.json`.
  - `HFPublisher.publish()` uploads the bundle without calling `scripts/validate_release_artifacts.py`, so the publish path can bypass the documented validation gate entirely.
- **GGUF/Ollama support is described, but dependency/install readiness is incomplete.**
  - `finetune/model_publisher.py` defaults conversion to `python3 -m llama_cpp.convert`.
  - `.env.example` exposes `GGUF_CONVERTER=python3 -m llama_cpp.convert`.
  - But neither `pyproject.toml` nor `requirements-finetune.txt` declares `llama_cpp` / `llama-cpp-python`, so the advertised default conversion path is not reproducibly installable from repo metadata.
- **Proof exists in narrative form, but not in the source-of-truth registry.**
  - `results/RTX3090-BASELINE-REPORT-2026-03-14.md`, `results/DOCKER-TEST-REPORT-2026-03-13.md`, and `results/MULTI-RUN-RESULTS-2026-03-14.md` document real experimental outcomes.
  - `validation/registry.json` still has an empty `runs` array and both tiers remain `unverified`.
  - This means operational evidence exists, but the governance loop that should normalize it into `validation/registry.json` is not closed.
- **Config and CLI surfaces are drifting.**
  - `providers.py` supports `gemini`, and `.env.example` includes `GEMINI_API_KEY`.
  - `agent_loop.py` CLI choices omit `gemini`, so exposed runtime behavior is narrower than the provider layer below it.
- **Some scripts still reflect older image naming and release assumptions.**
  - Docs and `.env.example` use `ghcr.io/ai-engineerings-at/tuneforge-studio:...`.
  - `scripts/run_remote_3090_pilot.py` and `scripts/run_3090_baseline.sh` still default to `ai-engineering/tuneforge-studio:technical-preview`.
  - This increases operator confusion around the canonical image source.
- **Test coverage is good for unit contracts, but weak on cross-surface integration drift.**
  - There are no tests validating that `dashboard/visualize.py` can read the current `agent_loop.py` TSV schema.
  - There are no tests proving `agent_loop.py` emits protocol entries matching `audit/protocol.schema.json`.
  - There are no tests enforcing that `publish` includes or requires `validation-result.json`.
  - There are no tests for `entrypoint-finetune.sh` or the documented `datasets.synthetic_generator` CLI flow.

## Next Steps
- **P1 — Finish the standalone repo extraction.**
  - Update `pyproject.toml`, `docs/SETUP-EN.md`, `docs/SETUP-DE.md`, `SECURITY.md`, `SECURITY-DE.md`, and README CI references so the public surface matches this repo, not `Playbook01/products/tuneforge`.
  - Either add the promised `.github/workflows/*.yml` workflows or remove/replace the broken links and claims.
  - Make `scripts/check_docs_links.py`, `scripts/check_docs_parity.py`, and `scripts/check_repo_hygiene.py` pass in this repo without monkeypatching.
- **P1 — Choose one canonical experiment artifact contract and enforce it everywhere.**
  - Standardize on one `results.tsv` location, one TSV header, one status vocabulary, and one protocol log schema.
  - Align `agent_loop.py`, `agent_config.py`, `dashboard/visualize.py`, `programs/*.md`, and any legacy training entrypoints around that decision.
  - Add tests that fail on schema drift.
- **P1 — Fix the fine-tune Docker developer journey.**
  - Align `entrypoint-finetune.sh` dataset paths with `finetune/configs/*.yaml` and the dataset generators.
  - Either implement a real CLI in `datasets/synthetic_generator.py` or remove the command from user-facing guidance.
  - Decide whether `docker-compose.finetune.yml` is an interactive dev shell or a runnable pipeline, and make docs + entrypoints reflect that explicitly.
- **P1 — Close the release-gate bypass.**
  - Make `bundle` and especially `publish` validate bundles before publication.
  - Ensure `validation-result.json` is present before any public upload path.
  - Add tests covering the `publish` path, not just bundle creation and external validation.
- **P2 — Normalize existing evidence into governed records.**
  - Convert the existing 3090/docker markdown reports into structured bundle artifacts and `validation/registry.json` run entries where possible.
  - If that evidence is insufficient, keep the preview posture but document exactly what is still missing for Tier A and Tier B closure.
- **P2 — Prune or quarantine legacy/dead entrypoints.**
  - Either remove/archive unsupported files like root `train.py` and `run_patched.py`, or clearly mark them as legacy and non-canonical.
  - Reduce the number of competing “obvious” entrypoints for new users.
- **P2 — Reconcile provider and image naming surfaces.**
  - Expose `gemini` consistently or remove it from the public provider story.
  - Unify image naming across docs, env examples, and remote helper scripts.
- **P3 — Add integration tests for the real seams.**
  - Dashboard against current TSV format
  - agent-loop protocol output against `audit/protocol.schema.json`
  - end-to-end release bundle + validation-result generation
  - entrypoint/data-path behavior for the finetune Docker surface

---

*Generated by /audit — read-only recce, no code was modified.*

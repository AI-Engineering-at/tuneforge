# TuneForge Troubleshooting
- Language: EN
- Audience: Public
- Last Sync: 2026-03-19
- Pair: TROUBLESHOOTING-DE.md

## GPU Not Detected

- verify `nvidia-smi`
- verify the NVIDIA Container Toolkit for Docker runs
- confirm the container started with GPU access
- confirm the driver and CUDA versions are recorded in the environment manifest

## Unsloth Backend Missing

- install the `finetune` extra
- confirm the environment actually contains `unsloth`
- use `peft_trl` for the control benchmark when in doubt

## Missing Primary Metric

- check `primary_metric` and `metric_goal` in the YAML config
- confirm `--eval` is enabled when the metric depends on evaluation output
- check `results/protocol.jsonl` for the emitted metric sequence

## Release-Bundle Validation Fails

- verify `training-manifest.json`, `benchmark-summary.json`, `environment-manifest.json`, and `tester-attestation.json`
- run `python scripts/validate_release_artifacts.py <bundle-dir>`
- inspect `validation-result.json`
- check that public status and hardware tier match across the bundle

## Public Repo Hygiene Fails

- remove generated outputs from tracked state
- keep only `.env.example`, never `.env`
- keep Docker images, GGUF, safetensors, logs, and databases out of git
- run `python scripts/check_repo_hygiene.py`

## Docs Parity Fails

- update EN and DE together
- keep `Language`, `Audience`, `Last Sync`, and `Pair` metadata aligned
- run `python scripts/check_docs_parity.py`

## Public Claim Check Fails

- remove unverified hardware labels
- keep `Technical Preview` visible in the root readmes while the registry is preview
- avoid compliance-certified or ready-for-production language
- run `python scripts/check_public_claims.py`

# Release Checklist v1.0.0

## Pre-Release Checklist

### Code Quality

- [ ] All tests pass
  ```bash
  python -m pytest tests/ -q
  # Expected: 217 passed, 1 skipped
  ```
  OR use automated check:
  ```bash
  python scripts/pre_release_check.py
  ```

- [ ] Coverage acceptable
  ```bash
  pytest tests/ --cov=finetune --cov=data_utils
  # Expected: >=78% overall
  ```

- [ ] Code style clean
  ```bash
  ruff check .
  mypy finetune/
  ```

- [ ] No security vulnerabilities
  ```bash
  bandit -r finetune/
  safety check
  ```

### Documentation

- [ ] CHANGELOG.md updated
- [ ] Version bumped in pyproject.toml
- [ ] API documentation generated
- [ ] EN/DE documentation parity
  ```bash
  python scripts/check_docs_parity.py
  ```

### Validation

- [ ] Tier A validation planned/scheduled
- [ ] Validation plan reviewed
- [ ] Evidence collection scripts tested

### Release Artifacts

- [ ] Git tag created: `v1.0.0`
- [ ] GitHub Release drafted
- [ ] Docker image built
- [ ] SBOM generated

## Release Steps

### 1. Final Code Check

```bash
# Run full test suite
python -m pytest tests/ -v --tb=short

# Check coverage
pytest tests/ --cov=finetune --cov=data_utils --cov-report=term-missing

# Code style
ruff check .
mypy finetune/

# Security
bandit -r finetune/ -f json -o bandit-report.json
safety check --json --output safety-report.json
```

### 2. Update Version

```bash
# Update pyproject.toml
# version = "1.0.0"

# Commit
git add pyproject.toml CHANGELOG.md
git commit -m "chore: bump version to 1.0.0"
```

### 3. Create Tag

```bash
git tag -a v1.0.0 -m "Release version 1.0.0

- Dual backend training (PEFT/TRL + Unsloth)
- Zeroth-Law safety architecture
- Validation registry with hardware verification
- EU AI Act compliance support
- 217 tests, 78% coverage
- Full EN/DE documentation"

git push origin v1.0.0
```

### 4. Build Artifacts

```bash
# Build Python package
python -m build

# Build Docker image
docker build -f Dockerfile.finetune -t tuneforge:1.0.0 .
docker tag tuneforge:1.0.0 ghcr.io/ai-engineerings-at/tuneforge:1.0.0

# Generate SBOM
syft tuneforge:1.0.0 -o spdx-json > tuneforge-1.0.0-sbom.json
```

### 5. Create GitHub Release

- Go to GitHub Releases
- Create release from tag `v1.0.0`
- Title: "TuneForge v1.0.0"
- Body: Copy from CHANGELOG.md
- Attach:
  - [ ] `dist/tuneforge-1.0.0-py3-none-any.whl`
  - [ ] `dist/tuneforge-1.0.0.tar.gz`
  - [ ] `tuneforge-1.0.0-sbom.json`

### 6. Publish Docker Image

```bash
docker push ghcr.io/ai-engineerings-at/tuneforge:1.0.0
docker push ghcr.io/ai-engineerings-at/tuneforge:latest
```

### 7. Update Documentation

```bash
# Build and deploy docs
mkdocs gh-deploy

# Or manually upload to hosting
```

### 8. Announce

- [ ] GitHub Discussions announcement
- [ ] Twitter/X post (if applicable)
- [ ] Email newsletter (if applicable)

## Post-Release

### Immediate (within 24h)

- [ ] Monitor for critical issues
- [ ] Respond to user feedback
- [ ] Check CI/CD pipelines

### Short-term (within 1 week)

- [ ] Gather user feedback
- [ ] Plan v1.0.1 if needed
- [ ] Update roadmap

### Validation (ongoing)

- [ ] Execute Tier A validation runs
- [ ] Collect evidence
- [ ] Update validation registry
- [ ] Update status from "Technical Preview" to "Verified"

## Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Release Manager | | | |
| QA Lead | | | |
| Security Lead | | | |
| Documentation Lead | | | |

## Rollback Plan

If critical issues found within 24h:

1. Mark release as "Pre-release" on GitHub
2. Create hotfix branch from v1.0.0 tag
3. Fix issues
4. Release v1.0.1
5. Document incident

---

**Release Date:** 2026-04-XX
**Target Completion:** [Date + Time]

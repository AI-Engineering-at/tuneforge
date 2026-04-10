# Release Plan v1.0.0

## Phase 1: Critical Fixes (15 min)

### 1.1 Fix trainer.py job_id Parameter (CRITICAL)
**File:** `finetune/trainer.py:321`  
**Issue:** Missing required `job_id` argument to `pre_train_zeroth_check`

```python
# BEFORE (Line 321)
clearance = pre_train_zeroth_check(
    self.config.to_dict(),
    records,
)

# AFTER
clearance = pre_train_zeroth_check(
    self.config.to_dict(),
    records,
    job_id=f"train-{self.config.output_dir.name}",
)
```

### 1.2 Auto-fix Ruff Issues (HIGH)
**Command:**
```bash
ruff check . --fix
```
**Expected:** Reduces 98 issues → ~18 issues (manual fixes needed)

### 1.3 Fix Ambiguous Variable Names (MEDIUM)
**Files:**
- `dashboard/visualize.py:225` - `l` → `line`
- `scripts/pre_release_check.py:146` - `l` → `line`

---

## Phase 2: Documentation (10 min)

### 2.1 Add Metadata Headers
**Files to update:**
- `README.md`
- `README-DE.md`
- `CHANGELOG.md`

**Template:**
```markdown
<!--
Title: TuneForge - Benchmark-first Fine-tuning Framework
Version: 1.0.0
Language: EN
Audience: Users, Developers
Last Sync: 2026-04-10
Pair: README-DE.md
-->
```

### 2.2 Update CHANGELOG-DE.md
Add German translation of v1.0.0 changelog.

---

## Phase 3: Verification (10 min)

### 3.1 Run Tests
```bash
pytest tests/ -q
# Expected: 217 passed, 1 skipped, 1 env-fail
```

### 3.2 Run Pre-release Check
```bash
python scripts/pre_release_check.py
# Expected: All PASS (or only minor warnings)
```

### 3.3 Build Package
```bash
python -m build
# Verify: dist/tuneforge-1.0.0-*
```

---

## Phase 4: Release (5 min)

### 4.1 Commit Changes
```bash
git add -A
git commit -m "chore: v1.0.0 release preparation

- Fix job_id parameter in trainer.py
- Code style cleanup (ruff)
- Documentation metadata updates
- Audit and release plan"
```

### 4.2 Create Tag
```bash
git tag -a v1.0.0 -m "TuneForge v1.0.0 - Technical Preview

Benchmark-first fine-tuning framework with:
- Dual backend training (PEFT/TRL + Unsloth)
- Zeroth-Law safety architecture
- 217 tests, 78% coverage
- EU AI Act compliance support

Status: Technical Preview (Tier A validation pending)"
```

### 4.3 Push Release
```bash
git push origin main
git push origin v1.0.0
```

---

## Phase 5: Post-Release (async)

### GitHub Actions will:
1. Run full test suite
2. Build Python package (wheel + sdist)
3. Build Docker images (finetune + turboquant)
4. Generate SBOM
5. Create GitHub Release (draft)

### Manual Steps:
1. Review and publish GitHub Release draft
2. Verify Docker images pushed to ghcr.io
3. Check documentation deployed to GitHub Pages

---

## Timeline

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Critical Fixes | 15 min | 15 min |
| Documentation | 10 min | 25 min |
| Verification | 10 min | 35 min |
| Release | 5 min | 40 min |
| **Total** | **40 min** | **40 min** |

---

## Contingency

If issues found during verification:
- **Test failures:** Fix → Re-verify (goto Phase 1)
- **Build failures:** Check pyproject.toml → Retry
- **Tag exists:** Delete tag locally/remotely → Recreate

---

## Success Criteria

- [ ] All tests pass (217+)
- [ ] Package builds successfully
- [ ] GitHub Actions workflows complete
- [ ] GitHub Release published
- [ ] Docker images available

---

**Started:** 2026-04-10  
**Target Completion:** 2026-04-10 (+40 min)  
**Status:** IN PROGRESS

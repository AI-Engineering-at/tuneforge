# TuneForge Security Policy

**Language:** EN  
**Audience:** Public  
**Last Updated:** 2026-04-10  
**Pair:** SECURITY-DE.md

TuneForge is a public engineering framework and Docker-based product surface. Security handling must stay compatible with a future dedicated public GitHub repo and with GHCR-based release publication.

---

## Scope

This policy covers the public TuneForge surface in `products/tuneforge`, including:

- Source code
- Dockerfiles and compose files
- GitHub Actions workflows
- Release bundle tooling
- Public documentation and templates
- Validation registry and audit artifacts
- Contract 3 safety integration with Zeroth

It does not cover private infrastructure, private open-notebook content, customer environments, or third-party services outside our control.

---

## Supported Surface

- Current `main` branch
- Latest tagged TuneForge code release
- Latest tagged TuneForge Studio Docker preview release on GHCR

Current public status: **Technical Preview**

---

## Threat Model

### Assets

| Asset | Value | Risk |
|-------|-------|------|
| Training data | Intellectual property | Leakage |
| Model weights | Intellectual property | Theft, tampering |
| Safety validation | Compliance, safety | Bypass, spoofing |
| JWT tokens | Authentication | Theft, replay |
| Model outputs | Reputation | Harmful generation |

### Threats

**T1: Malicious Training Data Injection**
- Attacker injects backdoor or poisoned examples
- Mitigation: Pre-train safety check (dataset hash validation)

**T2: Gradient Attack During Training**
- Attacker manipulates gradients to bypass safety
- Mitigation: Contract 3 - weight update evaluation every step

**T3: Unauthorized Model Publication**
- Attacker publishes unvalidated model
- Mitigation: Pre-publish safety check with manifest validation

**T4: Zeroth Service Spoofing**
- Attacker impersonates safety service
- Mitigation: JWT authentication, TLS in production

**T5: Mock Mode Abuse**
- Production system runs with safety disabled
- Mitigation: `sys.exit(1)` if `ZEROTH_MOCK_MODE=1` in production

**T6: Arbitrary Code Execution via Model**
- Malicious model architecture executes code
- Mitigation: `trust_remote_code=False` by default

---

## Known Limitations

### Contract 3 Performance

**Issue:** HTTP call on every training step adds latency.

**Impact:** 100ms timeout × 1000 steps = ~100s overhead per training run.

**Mitigation:** None currently. Future: batch evaluations, async processing.

**Workaround:** Increase `ZEROTH_TIMEOUT_MS` for slower networks (security trade-off).

### trust_remote_code

**Issue:** Custom model architectures require `trust_remote_code=True`.

**Risk:** Arbitrary code execution from HuggingFace model repositories.

**Mitigation:** Default is `False`. User must explicitly opt-in in config.

**Recommendation:** Only enable for trusted model sources.

### ZEROTH_MOCK_MODE

**Issue:** Development mode bypasses all safety checks.

**Risk:** Accidental deployment to production with mock mode enabled.

**Mitigation:** 
- Production detection: `TUNEFORGE_ENV=production`
- Behavior: `sys.exit(1)` with error log
- Artifacts marked as `UNVERIFIED` when mock mode used

### Gradient Surgery Batch Size

**Issue:** Safety dataset uses `batch_size=1` to prevent OOM.

**Impact:** Slower safety evaluation, potential coverage gaps.

**Mitigation:** Micro-batching with adaptive thresholding.

### JWT Token Rotation

**Issue:** No automatic JWT token refresh.

**Risk:** Long-lived tokens compromise security.

**Mitigation:** Tokens must be manually rotated.

**Recommendation:** Use short-lived tokens (24h) in production.

---

## Responsible Disclosure

### Reporting Security Issues

Report security issues privately before public disclosure.

**Include:**
- Affected file, workflow, or command path
- Reproduction steps and expected impact
- Whether secrets, provenance data, or release integrity are involved
- Severity assessment (Critical/High/Medium/Low)

**Do NOT:**
- Open a public issue for live credentials, signing, or supply-chain findings
- Include exploit code in initial report
- Disclose before fix is available

### Response Timeline

| Severity | Acknowledgment | Initial Response | Fix Target |
|----------|---------------|------------------|------------|
| Critical | 24 hours | 48 hours | 7 days |
| High | 48 hours | 72 hours | 14 days |
| Medium | 72 hours | 1 week | 30 days |
| Low | 1 week | 2 weeks | 90 days |

### Bug Bounty

Currently no formal bug bounty program. Security researchers will be acknowledged in release notes with permission.

---

## Public Repo Rules

- No built Docker images in git
- No real `.env` files or secrets in git
- No generated model artifacts in git
- No unverifiable security or compliance claims
- GHCR publication must come from GitHub Actions, not from committed artifacts

---

## Release Hardening Expectations

Public TuneForge releases include:

- Passing CI and public-repo readiness checks
- Release artifact validation
- Attribution and license validation
- SBOM generation for GHCR Docker releases
- Checksum generation and signed metadata for GHCR release outputs
- Zero critical/high security findings in `bandit` and `safety` scans

---

## Security Checklist (For Maintainers)

Before each release:

- [ ] `bandit -r finetune/` - zero high/critical issues
- [ ] `safety check` - zero CVEs in dependencies
- [ ] `ruff check .` - zero security-related lint issues
- [ ] All JWT tokens are parameterized (no hardcoded secrets)
- [ ] `trust_remote_code` defaults to `False`
- [ ] Mock mode exits in production
- [ ] Contract 3 timeout is ≤ 100ms (fail-closed)

---

## Boundaries

- TuneForge documentation is governance and engineering support
- It is not legal advice
- It is not a security certification
- It does not guarantee Austria, DSGVO, or EU AI Act compliance by default

---

## Contact

Security issues: security@ai-engineering.at (PGP key available)

General inquiries: See SUPPORT.md

Last updated: 2026-04-10

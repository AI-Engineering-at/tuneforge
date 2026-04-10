# Security Whitepaper

**Version:** 1.0  
**Date:** 2026-04-10  
**Classification:** Public

## Executive Summary

TuneForge implements defense-in-depth security for enterprise fine-tuning workflows. This document outlines security measures, threat mitigations, and compliance features.

## Threat Model

### Assets

| Asset | Value | Protection Level |
|-------|-------|------------------|
| Training Data | High | Encryption at rest, access controls |
| Model Weights | High | Secure export, access logging |
| API Keys | Critical | Environment only, never logged |
| Training Logs | Medium | Structured logging, audit trail |

### Threats

| Threat | Likelihood | Impact | Mitigation |
|--------|------------|--------|------------|
| Data leakage | Medium | High | Container isolation, no external calls |
| Model poisoning | Low | High | Zeroth-Law safety, input validation |
| Unauthorized access | Medium | High | RBAC, audit logging |
| Supply chain | Low | Medium | Pinned dependencies, SBOM |

## Security Measures

### 1. Training Data Protection

**Container Isolation:**
- Docker containers isolate training environment
- No external network calls during training (unless explicitly configured)
- Data remains within container boundary

**Input Validation:**
```python
# All dataset inputs validated
from data_utils.data_formats import record_to_text

# Raises ValueError on invalid format
record_to_text(record, dataset_format="alpaca")
```

### 2. Model Safety (Zeroth-Law)

**Gradient-Level Safety:**
- Mathematical guarantee of safety constraint preservation
- Orthogonal projection prevents safety degradation
- Automatic fail-closed on safety service unavailable

**Pre-Publish Validation:**
- Safety benchmarks before model export
- Model card generation with safety notes
- Block export if safety checks fail

### 3. Access Control

**Environment-Based Secrets:**
```python
# Correct: Use environment
api_key = os.environ.get("HF_TOKEN")

# Incorrect: Never hardcode
api_key = "sk-..."  # NEVER DO THIS
```

**RBAC (Enterprise):**
- Roles: Admin, Data Scientist, Auditor, Viewer
- Permission matrix for each role
- Audit log of all actions

### 4. Audit Trail

**Structured Logging:**
```json
{
  "timestamp": "2026-04-10T12:00:00Z",
  "level": "INFO",
  "service": "tuneforge",
  "logger": "tuneforge.trainer",
  "event": "training_complete",
  "job_id": "job-123",
  "model_name": "my-model",
  "git_sha": "abc123"
}
```

**Log Retention:**
- Default: 90 days
- Enterprise: Configurable (up to 10 years for compliance)

### 5. Supply Chain Security

**Dependency Management:**
- All dependencies pinned in `pyproject.toml`
- SBOM generated for each release
- No unpinned dependencies allowed

**SBOM Generation:**
```bash
# Generate SBOM
syft tuneforge:v1.0.0 -o spdx-json > sbom.json
```

## Compliance Features

### EU AI Act

| Article | Requirement | Implementation |
|---------|-------------|----------------|
| Art. 9 | Risk Management | Risk assessment tool, mitigation tracking |
| Art. 10 | Data Governance | Data provenance tracker, quality checks |
| Art. 11 | Technical Documentation | Auto-generated docs from training runs |
| Art. 12 | Record Keeping | Structured logging, 10-year retention |
| Art. 13 | Transparency | Model cards, usage instructions |
| Art. 14 | Human Oversight | Review workflows, approval gates |

### GDPR/DSGVO

**Data Provenance:**
- All training data sources tracked
- Consent information recorded
- Right to deletion supported

**Privacy by Design:**
- PII detection in datasets
- Differential privacy options
- No personal data in logs

## Security Best Practices

### For Administrators

1. **Use latest version:**
   ```bash
   pip install --upgrade tuneforge
   ```

2. **Enable audit logging:**
   ```yaml
   logging:
     level: INFO
     audit: true
     retention_days: 365
   ```

3. **Regular security updates:**
   ```bash
   pip-audit  # Check for vulnerable dependencies
   ```

### For Data Scientists

1. **Validate datasets:**
   ```python
   from data_utils.data_formats import normalize_records_to_text
   
   # Check for PII
   for record in dataset:
       normalized = normalize_records_to_text([record])
   ```

2. **Review model cards:**
   - Verify safety notes
   - Check limitations
   - Confirm intended use

3. **Use validation configs:**
   ```yaml
   # Don't use experimental settings for production
   backend: "peft_trl"  # Stable, not "unsloth"
   ```

### For Security Teams

1. **Monitor audit logs:**
   ```bash
   # Forward to SIEM
   tail -f /var/log/tuneforge/protocol.jsonl | \
     jq '{timestamp, level, event, job_id}'
   ```

2. **Verify SBOMs:**
   ```bash
   # Check for known vulnerabilities
   grype sbom.json
   ```

3. **Review registry:**
   ```bash
   # Check validation status
   cat validation/registry.json | jq '.tiers'
   ```

## Incident Response

### Security Incident Types

| Severity | Examples | Response Time |
|----------|----------|---------------|
| Critical | Data breach, model poisoning | 1 hour |
| High | Unauthorized access | 4 hours |
| Medium | Dependency vulnerability | 24 hours |
| Low | Documentation issue | 7 days |

### Response Procedure

1. **Detect:** Monitor logs, alerts
2. **Contain:** Stop affected jobs, isolate systems
3. **Investigate:** Analyze logs, determine scope
4. **Remediate:** Fix vulnerability, patch systems
5. **Recover:** Resume operations, verify fix
6. **Document:** Post-mortem, update procedures

## Certifications & Standards

### Target Certifications

- **ISO 27001:** Information Security Management (2027)
- **SOC 2 Type II:** Security controls (2027)
- **EU AI Act:** High-risk system compliance (ongoing)

### Standards Alignment

- **OWASP Top 10:** Web application security
- **NIST Cybersecurity Framework:** Risk management
- **CIS Benchmarks:** Docker, Kubernetes hardening

## Vulnerability Disclosure

**Responsible Disclosure:**
- Email: security@ai-engineering.at
- GPG Key: [available on website]
- Response time: 48 hours
- Bounty program: [details on website]

**Security Advisories:**
- Published on GitHub Security tab
- Mailing list: security-announce@ai-engineering.at

## References

- [SECURITY.md](../../SECURITY.md)
- [SECURITY-DE.md](../../SECURITY-DE.md)
- [COMPLIANCE_STATEMENT.md](../../COMPLIANCE_STATEMENT.md)
- [docs/ZEROTH_HARDENING_ARCHITECTURE.md](../../ZEROTH_HARDENING_ARCHITECTURE.md)

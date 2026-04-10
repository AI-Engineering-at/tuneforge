# TuneForge Quality-First Roadmap
**Philosophy:** Done means Done. No compromises. No deadlines. Only quality gates.  
**Version:** 1.0 - Time-agnostic  
**Status:** Living Document

---

## Core Principles

1. **Sequential over Parallel** - One thing at a time, done right
2. **Quality Gates over Time Gates** - "Is it excellent?" not "Is it Tuesday?"
3. **No Technical Debt** - If it's not maintainable, it's not done
4. **Test-First** - If it's not tested, it doesn't exist
5. **Document-First** - If it's not documented, it's not done

---

## Phase 0: Foundation (Current State → Stable)

> **Entry Gate:** Code exists, tests are broken, registry is empty  
> **Exit Gate:** Zero known defects, 100% test pass, validated hardware evidence

### Task 0.1: Fix Module Architecture
**Prerequisite:** None  
**Definition of Done:**
- [ ] `datasets/data_formats.py` exists with proper exports
- [ ] All imports resolve without hacks
- [ ] Import graph is cycle-free (verified with `import-deps`)
- [ ] All 16 test files import successfully

**Quality Check:**
```bash
python -c "from data_utils.data_formats import normalize_records_to_text; print('OK')"
python -m pytest tests/ --collect-only 2>&1 | grep "errors" && exit 1 || echo "✓"
```

---

### Task 0.2: Fix API Consistency
**Prerequisite:** Task 0.1 complete  
**Definition of Done:**
- [ ] Zeroth Core API Signatur konsistent (`job_id` Parameter)
- [ ] All functions have type hints
- [ ] All functions have docstrings (Google-style)
- [ ] No `**kwargs` without validation

**Quality Check:**
```bash
python -m pytest tests/test_zeroth_safety_filter.py -v  # All pass
python -m pytest tests/test_safe_trainer_math.py -v      # All pass
```

---

### Task 0.3: Achieve 100% Unit Test Coverage (Critical Paths)
**Prerequisite:** Task 0.2 complete  
**Definition of Done:**
- [ ] `providers.py`: 100% coverage
- [ ] `agent_config.py`: 100% coverage
- [ ] `zeroth_core.py`: 100% coverage
- [ ] `safe_trainer.py`: 100% coverage
- [ ] All new code covered

**Quality Check:**
```bash
pytest --cov=finetune --cov-report=term-missing tests/
# No uncovered lines in critical paths
```

---

### Task 0.4: Integration Test Suite
**Prerequisite:** Task 0.3 complete  
**Definition of Done:**
- [ ] Docker Compose Test läuft durch (End-to-End)
- [ ] Backend-Switch Test (PEFT ↔ Unsloth) funktioniert
- [ ] Export-Pipeline Test (HF → GGUF → Ollama)
- [ ] Fehlerinjektion-Tests (was passiert bei OOM, Netzwerkfehler?)

**Quality Check:**
```bash
docker compose -f docker-compose.finetune.yml up --build --abort-on-container-exit
echo $?  # Must be 0
```

---

### Task 0.5: Validation Run Tier A
**Prerequisite:** Task 0.4 complete  
**Definition of Done:**
- [ ] Ein kompletter Training-Run auf RTX 3090 (24GB)
- [ ] Von Docker-Start bis Ollama-Export
- [ ] Alle Metriken geloggt
- [ ] Registry aktualisiert mit Evidence
- [ ] Reproduzierbar: Zweiter Run mit identischem Ergebnis (±2%)

**Quality Check:**
```bash
# Run 1
python -m finetune.trainer --config configs/tier-a-validation.yaml --eval 2>&1 | tee run1.log
# Run 2 (identical config)
python -m finetune.trainer --config configs/tier-a-validation.yaml --eval 2>&1 | tee run2.log
# Compare primary metrics - must be within 2%
```

**Registry Entry:**
```json
{
  "tier": "tier_a_rtx_3090_24gb",
  "status": "verified",
  "runs": [
    {"date": "2026-04-XX", "tester": "automated", "result": "success", "evidence": "run1.log"},
    {"date": "2026-04-XX", "tester": "automated", "result": "success", "evidence": "run2.log"}
  ]
}
```

---

### Task 0.6: Documentation Completeness
**Prerequisite:** Task 0.5 complete  
**Definition of Done:**
- [ ] API Reference auto-generated aus Docstrings (Sphinx/MkDocs)
- [ ] Architecture Decision Records (ADRs) für alle großen Entscheidungen
- [ ] Troubleshooting Guide mit 10+ Szenarien
- [ ] Contribution Guide mit Coding Standards
- [ ] Security Whitepaper (für Enterprise)
- [ ] DE/EN Parity Check 100% bestehen

**Quality Check:**
```bash
python scripts/check_docs_parity.py  # Must pass
mkdocs build --strict  # No warnings
```

---

### Task 0.7: v1.0 Release Preparation
**Prerequisite:** All above complete  
**Definition of Done:**
- [ ] CHANGELOG.md vollständig (alle Changes seit Beginn)
- [ ] Version in `pyproject.toml` auf 1.0.0
- [ ] Git Tag `v1.0.0` erstellt
- [ ] GitHub Release mit Binaries (Docker Image)
- [ ] SBOM (Software Bill of Materials) generiert
- [ ] Security Audit (Bandit, Safety) - keine Critical/High findings

**Quality Check:**
```bash
bandit -r finetune/ -f json -o bandit-report.json
safety check --json --output safety-report.json
# Critical/High count = 0
```

---

## Phase 1: Training Excellence

> **Entry Gate:** v1.0 released, stable foundation  
> **Exit Gate:** State-of-the-art training capabilities, all SOTA methods implemented

### Task 1.1: DoRA Implementation
**Prerequisite:** Phase 0 complete  
**Definition of Done:**
- [ ] DoRA (Weight-Decomposed Low-Rank Adaptation) implementiert
- [ ] Config-Option `adapter: dora` funktioniert
- [ ] Vergleichsbaseline: DoRA vs LoRA auf gleichem Datensatz
- [ ] Dokumentation: Wann DoRA > LoRA?
- [ ] Tests für DoRA-Pfad

**Quality Check:**
```bash
# Training mit DoRA
python -m finetune.trainer --config configs/sps-plc.yaml --adapter dora
# Metrik muss ≥ LoRA sein
```

---

### Task 1.2: Preference Optimization Suite
**Prerequisite:** Task 1.1 complete  
**Definition of Done:**
- [ ] DPO (Direct Preference Optimization) implementiert
- [ ] ORPO (Odds Ratio Preference Optimization) implementiert
- [ ] KTO (Kahneman-Tversky Optimization) implementiert
- [ ] Unified Interface: `trainer.train_dpo()`, `trainer.train_orpo()`
- [ ] Dokumentation: Welche Methode wann?
- [ ] Benchmark: ORPO vs DPO vs SFT auf gleichem Datensatz

**Quality Check:**
```bash
python -m finetune.trainer --config configs/preference.yaml --method orpo
# Preference accuracy > baseline
```

---

### Task 1.3: Advanced Optimizations
**Prerequisite:** Task 1.2 complete  
**Definition of Done:**
- [ ] rsLoRA vollständig integriert und getestet
- [ ] Gradient Checkpointing optimiert
- [ ] Flash Attention 2 Support (wo verfügbar)
- [ ] Memory-Efficient Attention für große Kontexte
- [ ] Benchmark: Memory-Usage dokumentiert für 8/16/24/48GB

---

## Phase 2: Evaluation & Safety

> **Entry Gate:** Training Excellence erreicht  
> **Exit Gate:** Comprehensive evaluation suite, safety hardened

### Task 2.1: Automated Benchmark Framework
**Prerequisite:** Phase 1 complete  
**Definition of Done:**
- [ ] Integration mit `lm-evaluation-harness`
- [ ] Standard-Suite: GSM8K, HumanEval, MMLU, TruthfulQA
- [ ] Automatischer Report-Generator
- [ ] Benchmark-Vergleich gegen Baseline-Modelle
- [ ] Historisches Tracking (wie verändert sich Performance über Zeit?)

**Quality Check:**
```bash
python -m eval.benchmark --model output/model --suite standard
# Erzeugt benchmark-report.json
```

---

### Task 2.2: LLM-as-a-Judge Integration
**Prerequisite:** Task 2.1 complete  
**Definition of Done:**
- [ ] Integration mit Prometheus (Open-Source Judge)
- [ ] GPT-4 Judge (optional, für interne Evals)
- [ ] Pairwise Comparison Framework
- [ ] Elo-Ranking für Modelle
- [ ] Kosten-Tracking pro Judge-Evaluation

---

### Task 2.3: Safety & Red Teaming
**Prerequisite:** Task 2.2 complete  
**Definition of Done:**
- [ ] StrongREJECT Benchmark integriert
- [ ] Garak (Red Teaming Framework) integriert
- [ ] Safety-Evals automatisierbar
- [ ] Adversarial Test-Suite (Jailbreaks, Prompt Injection)
- [ ] Safety-Report pro Modell

---

### Task 2.4: Zeroth-Law Hardening
**Prerequisite:** Task 2.3 complete  
**Definition of Done:**
- [ ] Gradient Surgery mathematisch verifiziert
- [ ] Safety-Constraint Enforcement im Training
- [ ] Safety-Callback (Training stoppt bei Safety-Verstoß)
- [ ] Paper/Dokumentation des Ansatzes
- [ ] Peer Review intern

---

## Phase 3: Compliance Engine

> **Entry Gate:** Evaluation Suite vollständig  
> **Exit Gate:** Full EU AI Act automation, audit-ready

### Task 3.1: Data Governance Module
**Prerequisite:** Phase 2 complete  
**Definition of Done:**
- [ ] Data Provenance Tracker (Art. 10)
- [ ] Data Quality Checker (statistische Analyse)
- [ ] Bias Detection (Demographische Disparitäten)
- [ ] PII Detection & Masking
- [ ] Data Lineage Graph (visuell)

---

### Task 3.2: Risk Management System
**Prerequisite:** Task 3.1 complete  
**Definition of Done:**
- [ ] FMEA-basierte Risikoanalyse (Art. 9)
- [ ] Risiko-Registry (zentral)
- [ ] Mitigation Tracking
- [ ] Risiko-Report Generator
- [ ] Integration in Training-Workflow (kein Training ohne Risk-Assessment)

---

### Task 3.3: Automated Documentation
**Prerequisite:** Task 3.2 complete  
**Definition of Done:**
- [ ] Technical Documentation Generator (Art. 11)
- [ ] Model Card Generator (Art. 13) - EU-konform
- [ ] Instructions for Use Generator
- [ ] EU Declaration of Conformity Generator (Art. 47)
- [ ] Alles als PDF und maschinenlesbar (JSON)

---

### Task 3.4: Audit & Logging
**Prerequisite:** Task 3.3 complete  
**Definition of Done:**
- [ ] Structured JSONL Logging (alle Events)
- [ ] Audit Trail Immutability (Signatur/Hash)
- [ ] 10-Jahres-Retention Plan
- [ ] SIEM-Integration (Splunk, ELK)
- [ ] Audit-Report Generator

---

### Task 3.5: Conformity Assessment
**Prerequisite:** Task 3.4 complete  
**Definition of Done:**
- [ ] Internal Conformity Checklist (Art. 43)
- [ ] Automated Compliance Validator
- [ ] Gap Analysis Tool
- [ ] CE-Marking Preparation Checklist
- [ ] External Audit Support (Notizen, Evidence-Sammlung)

---

## Phase 4: Enterprise Platform

> **Entry Gate:** Compliance Engine vollständig  
> **Exit Gate:** Production-grade enterprise deployment

### Task 4.1: Multi-GPU & Scale
**Prerequisite:** Phase 3 complete  
**Definition of Done:**
- [ ] DeepSpeed Integration
- [ ] FSDP (Fully Sharded Data Parallel) Support
- [ ] Multi-Node Training
- [ ] Checkpointing für lange Runs (Resume-fähig)
- [ ] Linear Scaling: 2x GPUs = 1.8x+ Speedup

---

### Task 4.2: Infrastructure & Deployment
**Prerequisite:** Task 4.1 complete  
**Definition of Done:**
- [ ] Kubernetes Helm Charts
- [ ] Kubernetes Operator
- [ ] GPU Node Autoscaling
- [ ] Persistent Volume Management für Datasets/Models
- [ ] Terraform Module für Cloud-Deployment

---

### Task 4.3: Enterprise Security
**Prerequisite:** Task 4.2 complete  
**Definition of Done:**
- [ ] RBAC (Rollen: Admin, Data Scientist, Auditor, Viewer)
- [ ] SAML/OIDC Integration
- [ ] API Key Management
- [ ] Secret Management (Vault-Integration)
- [ ] Network Policies (Kubernetes)

---

### Task 4.4: Observability
**Prerequisite:** Task 4.3 complete  
**Definition of Done:**
- [ ] Prometheus Metrics Exporter
- [ ] Grafana Dashboards (Training, System, Costs)
- [ ] Distributed Tracing (Jaeger/Tempo)
- [ ] Alerting (Training-Failure, Cost-Overruns)
- [ ] Cost Tracking (pro Run, pro User, pro Project)

---

### Task 4.5: Model Lifecycle Management
**Prerequisite:** Task 4.4 complete  
**Definition of Done:**
- [ ] Model Registry (Versionierung)
- [ ] Model Lineage (Full DAG: Data → Training → Evaluation → Deploy)
- [ ] A/B Testing Framework
- [ ] Canary Deployment Support
- [ ] Rollback-Fähigkeit

---

## Phase 5: Ecosystem

> **Entry Gate:** Enterprise Platform stabil  
> **Exit Gate:** Thriving ecosystem, sustainable project

### Task 5.1: Web UI
**Prerequisite:** Phase 4 complete  
**Definition of Done:**
- [ ] React/Next.js Frontend
- [ ] Experiment Tracking UI
- [ ] Model Comparison UI
- [ ] Real-time Training Monitoring
- [ ] Drag-and-Drop Dataset Upload

---

### Task 5.2: Python SDK
**Prerequisite:** Phase 4 complete  
**Definition of Done:**
- [ ] Type-hinted SDK
- [ ] Async Support
- [ ] Retry-Logic, Error Handling
- [ ] Documentation mit Examples
- [ ] Jupyter-Integration

---

### Task 5.3: Plugin System
**Prerequisite:** Task 5.2 complete  
**Definition of Done:**
- [ ] Plugin API definiert
- [ ] Custom Dataset Loader Plugins
- [ ] Custom Evaluation Metrics Plugins
- [ ] Custom Model Publisher Plugins
- [ ] Plugin Registry (Community)

---

### Task 5.4: Certification
**Prerequisite:** All above complete  
**Definition of Done:**
- [ ] ISO 27001 Compliance (oder Vorbereitung)
- [ ] SOC 2 Type II (oder Vorbereitung)
- [ ] External Security Audit
- [ ] Penetration Test
- [ ] Bug Bounty Program

---

## Task Dependency Graph

```
Phase 0: Foundation
├── 0.1 Module Architecture
├── 0.2 API Consistency
├── 0.3 Unit Test Coverage
├── 0.4 Integration Tests
├── 0.5 Validation Run Tier A
├── 0.6 Documentation
└── 0.7 v1.0 Release
    │
    ▼
Phase 1: Training Excellence
├── 1.1 DoRA Implementation
├── 1.2 Preference Optimization (DPO/ORPO/KTO)
└── 1.3 Advanced Optimizations
    │
    ▼
Phase 2: Evaluation & Safety
├── 2.1 Automated Benchmarks
├── 2.2 LLM-as-a-Judge
├── 2.3 Safety & Red Teaming
└── 2.4 Zeroth-Law Hardening
    │
    ▼
Phase 3: Compliance Engine
├── 3.1 Data Governance
├── 3.2 Risk Management
├── 3.3 Automated Documentation
├── 3.4 Audit & Logging
└── 3.5 Conformity Assessment
    │
    ▼
Phase 4: Enterprise Platform
├── 4.1 Multi-GPU & Scale
├── 4.2 Infrastructure
├── 4.3 Enterprise Security
├── 4.4 Observability
└── 4.5 Model Lifecycle
    │
    ▼
Phase 5: Ecosystem
├── 5.1 Web UI
├── 5.2 Python SDK
├── 5.3 Plugin System
└── 5.4 Certification
```

---

## Quality Manifesto

### Definition of "Done"

Eine Task ist erst dann done, wenn **ALL** diese Kriterien erfüllt sind:

1. **Code Quality**
   - [ ] Alle Tests passen
   - [ ] 100% Coverage für neue Code-Pfade
   - [ ] Type hints komplett
   - [ ] Keine Lint-Fehler
   - [ ] Code Review durch mindestens 1 Person

2. **Documentation**
   - [ ] Docstrings für alle public APIs
   - [ ] User-facing Dokumentation aktualisiert
   - [ ] Architecture Decision Record (falls architektonisch)
   - [ ] Changelog Eintrag

3. **Testing**
   - [ ] Unit Tests
   - [ ] Integration Tests
   - [ ] Edge Cases abgedeckt
   - [ ] Fehlerfälle getestet

4. **Operational**
   - [ ] Logging implementiert
   - [ ] Monitoring (Metrics) wo relevant
   - [ ] Fehler sind handhabbar (nicht silent)
   - [ ] Rollback-Plan dokumentiert (für komplexe Changes)

5. **Security**
   - [ ] Security Review (für sensitive Features)
   - [ ] Keine Secrets im Code
   - [ ] Input Validation
   - [ ] Dependencies geprüft (`safety check`)

### Blocker Policy

- Wenn ein Quality Gate nicht erreicht wird, bleiben wir in der Phase
- Keine Übergehung von Tasks
- Kein "kommen wir später zurück"
- Keine Merge ohne Review

### Measurement

Wir tracken:
- **Defect Density** (Bugs pro 1000 LOC)
- **Test Flakiness** (Tests, die manchmal failen)
- **Documentation Coverage** (% der APIs dokumentiert)
- **Technical Debt** (Code Smells, TODOs)

Ziel: Null Defects, Zero Flakiness, 100% Documentation, Zero Debt

---

## Current Status

| Phase | Task | Status | Blocker |
|-------|------|--------|---------|
| 0 | 0.1 Module Architecture | 🔄 In Progress | None |
| 0 | 0.2 API Consistency | ⏳ Waiting | 0.1 |
| 0 | 0.3 Unit Test Coverage | ⏳ Waiting | 0.2 |
| 0 | 0.4 Integration Tests | ⏳ Waiting | 0.3 |
| 0 | 0.5 Validation Run | ⏳ Waiting | 0.4 |
| 0 | 0.6 Documentation | ⏳ Waiting | 0.5 |
| 0 | 0.7 v1.0 Release | ⏳ Waiting | 0.6 |

**Next Action:** Complete Task 0.1

---

*This roadmap has no deadlines. It has quality gates. We proceed when ready, not when the calendar says so.*

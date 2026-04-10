# TuneForge Enterprise Master Plan 2025
**Version:** 1.0.0  
**Status:** Strategic Planning Document  
**Classification:** Internal - Product Development  
**Last Updated:** 2026-04-10

---

## Executive Summary

TuneForge wird von einem "Technical Preview" zu einem **Enterprise-Grade Fine-Tuning Framework** für lokale LLM-Deployment entwickelt. Dieser Plan definiert den Weg zu einem marktreifen Produkt mit Fokus auf:

- **Qualität als oberste Priorität** (keine Kompromisse bei Testing & Validation)
- **EU AI Act Compliance** (vollständige regulatorische Unterstützung)
- **Enterprise Integration** (vollständige Toolchain, CI/CD, Monitoring)
- **State-of-the-Art ML** (DoRA, DPO/ORPO, Multi-GPU, Latest Optimizations)

---

## 1. Strategic Vision

### 1.1 Product Positioning

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ENTERPRISE FINE-TUNING LANDSCAPE 2025                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   CLOUD-NATIVE                      HYBRID                      ON-PREMISE  │
│   ─────────────                     ──────                      ──────────  │
│   • OpenAI API                      • AWS SageMaker             • TuneForge │
│   • Anthropic                       • Azure ML                  • LlamaBox  │
│   • Together AI                     • GCP Vertex                • Ollama    │
│   • Fireworks AI                                                    + DIY   │
│                                                                             │
│   TARGET: TuneForge Enterprise v2.0                                         │
│   ─────────────────────────────────                                         │
│   "GitLab for LLM Fine-Tuning"                                              │
│   • Complete audit trail                                                    │
│   • Multi-tenant enterprise features                                        │
│   • Full EU AI Act compliance tooling                                       │
│   • On-premise first, cloud optional                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Value Proposition

| Feature | Competitors | TuneForge Target |
|---------|-------------|------------------|
| **Compliance** | Add-on / Manual | Built-in, automated |
| **Audit Trail** | Logs | Full provenance chain |
| **Validation** | "Works on my machine" | Hardware-verified registry |
| **Safety** | Post-hoc filters | Gradient-level alignment |
| **Export** | HF only | HF + Ollama + GGUF + Enterprise registries |
| **Cost Control** | Opaque | Per-run cost estimation |

---

## 2. State of the Art Analysis (2025)

### 2.1 Fine-Tuning Technology Landscape

#### 2.1.1 PEFT Methods (Priorität: KRITISCH)

| Methode | Status | Enterprise Relevanz | TuneForge Status |
|---------|--------|---------------------|------------------|
| **QLoRA** | Standard | Baseline für alle deployments | ✅ Implementiert |
| **DoRA** | 2025 Standard | Superiore Qualität, kein Overhead | ⚠️ Geplant für v1.2 |
| **rsLoRA** | Stabil | Bessere Skalierung für große Ranks | ✅ Unterstützt |
| **LoRA-FA** | Emerging | Memory-Effizienz für große Modelle | 📋 Evaluieren |
| **PiSSA** | Experimental | Schnellere Konvergenz | 📋 Evaluieren |

**Empfohlene Default-Config (2025):**
```yaml
# DoRA + rsLoRA + optimierte Parameter
adapter: dora              # NEU: DoRA statt LoRA
lora_r: 64                 # Higher rank für complex reasoning
lora_alpha: 32             # alpha = r/2
lora_dropout: 0.05         # Regularisierung
target_modules:            # ALLE linearen Layer
  - q_proj, k_proj, v_proj, o_proj
  - gate_proj, up_proj, down_proj
use_rslora: true           # Rang-skalierte LoRA
gradient_accumulation: 4
optimizer: adamw_8bit
learning_rate: 2e-4        # Höher für QLoRA
```

#### 2.1.2 Training Backends

| Backend | Speed | Memory | Stability | Enterprise Fit |
|---------|-------|--------|-----------|----------------|
| **PEFT+TRL** | 1x | 1x | ⭐⭐⭐⭐⭐ | Referenz, Validierung |
| **Unsloth** | 2-5x | 0.3x | ⭐⭐⭐⭐ | Produktion, Speed |
| **TorchTune** | 1.5x | 0.8x | ⭐⭐⭐⭐ | Meta-Stack, experimentell |
| **LLaMA-Factory** | 1.2x | 0.9x | ⭐⭐⭐ | YAML-first, gut für Einsteiger |

**Empfehlung:** Dual-Backend Strategie beibehalten:
- **PEFT+TRL** für Validierung und referenz Benchmarks
- **Unsloth** für Produktions-Training
- Abstraktionsschicht ermöglicht seamless switching

#### 2.1.3 Preference Optimization (Neu für TuneForge)

Basierend auf Research 2025 - MÜSSEN hinzugefügt werden:

| Methode | Vorteil | Use Case | Implementierungspriorität |
|---------|---------|----------|---------------------------|
| **DPO** | Einfach, kein Reward Model | Erste Alignment-Stufe | P0 - Q2 2026 |
| **ORPO** | SFT + Alignment in einem | Effizienz-kritisch | P0 - Q2 2026 |
| **KTO** | Binary feedback ausreichend | Skalierbare Annotation | P1 - Q3 2026 |
| **SimPO** | Reference-free, length-normalized | Lange Generierungen | P1 - Q3 2026 |
| **GRPO** | Online RL, verifiable rewards | Tool-use, Coding | P2 - 2027 |

### 2.2 EU AI Act Compliance (2025+)

#### 2.2.1 Pflichten für High-Risk AI Systems

| Artikel | Anforderung | TuneForge Implementierung |
|---------|-------------|---------------------------|
| **Art. 9** | Risk Management System | `risk_management/` Modul mit FMEA-Template |
| **Art. 10** | Data Governance | Automatisierte Data Quality Checks |
| **Art. 11** | Technical Documentation | Auto-generierte Docs aus Training Runs |
| **Art. 12** | Record Keeping | Structured JSONL logs, 10+ Jahre Retention |
| **Art. 13** | Transparency | Auto-generierte Model Cards mit EU Template |
| **Art. 14** | Human Oversight | Review-Workflows im Model Publisher |
| **Art. 15** | Accuracy, Robustness, Cybersecurity | Automatisierte Evals + Red Teaming |
| **Art. 17** | Quality Management System | QMS-Integration, ISO 9001 Alignment |
| **Art. 43** | Conformity Assessment | Validator-Suite für CE-Marking Prep |
| **Art. 47** | EU Declaration of Conformity | Template + Auto-fill aus Registry |

#### 2.2.2 SME Simplified Documentation

Die EU bietet vereinfachte Dokumentation für SMEs an - TuneForge sollte dies automatisieren:

```
EU AI Act SME Template
├── Systembeschreibung (auto-generiert)
├── Verwendungszweck (aus Config)
├── Leistungsmetriken (aus Validation Registry)
├── Datenprovenienz (aus Data Provenance Tracker)
├── Risikomanagement (aus Risk Assessment Tool)
└── Konformitätserklärung (template + auto-fill)
```

### 2.3 Model Publishing & Governance (2025)

#### 2.3.1 Hugging Face Integration

**Neue Anforderungen 2025:**
- **Model Cards v2.0** mit strukturierten Metadaten
- **SBOMs** (Software Bill of Materials) für Modelle
- **Benchmark Cards** (IBM/Notre Dame Standard)
- **Open Notebook** Verpflichtung für transparente Forschung

#### 2.3.2 Ollama/GGUF Export

**Best Practices 2025:**
```
GGUF Export Pipeline
├── Quantisierungs-Level Auswahl
│   ├── Q4_K_M: Balance (empfohlen)
│   ├── Q5_K_M: Höhere Qualität
│   └── Q8_0: Nahezu verlustfrei
├── Modelfile Generierung
│   ├── FROM (Basis-GGUF oder Adapter)
│   ├── SYSTEM Prompt (konfigurierbar)
│   ├── PARAMETER (temperature, ctx, etc.)
│   ├── TEMPLATE (Chat-Format)
│   └── LICENSE (inherit from base)
└── Ollama Registry Push (optional)
```

#### 2.3.3 Enterprise Registries

Zusätzliche Publishing-Ziele für Enterprise:
- **Private HF Hub** (Enterprise Subscription)
- **Azure ML Registry**
- **AWS SageMaker Model Registry**
- **GCP Vertex AI Model Registry**
- **On-premise MLflow/Weights&Biases**

---

## 3. Current Gap Analysis

### 3.1 Kritische Blocker (Sofortige Aktion erforderlich)

| # | Problem | Impact | Lösung | Aufwand |
|---|---------|--------|--------|---------|
| 1 | `datasets.data_formats` fehlt | Tests broken | Modul erstellen oder Pfad fixen | 2h |
| 2 | Zeroth Core API Inkonsistenz | Safety tests fail | API-Signatur angleichen | 4h |
| 3 | Keine verifizierten Hardware-Runs | "Technical Preview" Status | 2x RTX 3090 Runs durchführen | 16h |
| 4 | Modulimport-Fehler in Tests | CI broken | Import-Pfade bereinigen | 4h |

### 3.2 Feature Gaps (Strategische Entwicklung)

| Kategorie | Gap | Priorität | Timeline |
|-----------|-----|-----------|----------|
| **Training** | DoRA Support | P0 | Q2 2026 |
| | DPO/ORPO Integration | P0 | Q2 2026 |
| | Multi-GPU Training | P1 | Q3 2026 |
| | DeepSpeed Integration | P2 | Q4 2026 |
| **Evaluation** | Automatisierte Benchmarks | P0 | Q2 2026 |
| | LLM-as-a-Judge Integration | P1 | Q3 2026 |
| | Red Teaming Framework | P1 | Q3 2026 |
| | Safety Evals (StrongREJECT) | P1 | Q3 2026 |
| **Compliance** | EU AI Act Template Suite | P0 | Q2 2026 |
| | Automatisierte Risk Assessment | P1 | Q3 2026 |
| | Conformity Assessment Workflow | P2 | Q4 2026 |
| | CE-Marking Preparation | P2 | 2027 |
| **Enterprise** | Multi-Tenant Support | P2 | 2027 |
| | RBAC & Audit Logging | P1 | Q3 2026 |
| | Cost Tracking & Budget Alerts | P1 | Q3 2026 |
| | Model Versioning & Lineage | P0 | Q2 2026 |
| | A/B Testing Framework | P2 | 2027 |

### 3.3 Documentation Gaps

| Dokument | Status | Aktion |
|----------|--------|--------|
| API Reference | ❌ Fehlt | Auto-generieren aus Docstrings |
| Architecture Decision Records | ⚠️ Teilweise | Vervollständigen |
| Deployment Guide | ⚠️ Basic | Enterprise Deployment erweitern |
| Security Whitepaper | ❌ Fehlt | Erstellen (für Enterprise Sales) |
| Compliance Guide | ⚠️ Templates | Ausführliche Anleitung schreiben |
| Troubleshooting Playbook | ⚠️ Basic | Erweitern mit Runbooks |

---

## 4. Enterprise Architecture Roadmap

### 4.1 Zielarchitektur (2027)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TUNEFORGE ENTERPRISE v2.0                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     PRESENTATION LAYER                               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│  │  │  Web UI     │  │  CLI        │  │  API        │  │  Python SDK │ │   │
│  │  │  (React)    │  │  (Rich)     │  │  (FastAPI)  │  │  (Typed)    │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     ORCHESTRATION LAYER                              │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│  │  │ Experiment  │  │ Pipeline    │  │ Job Queue   │  │ Resource    │ │   │
│  │  │ Tracker     │  │ Engine      │  │ (Redis/RQ)  │  │ Scheduler   │ │   │
│  │  │ (MLflow)    │  │ (Prefect)   │  │             │  │ (GPU Mgmt)  │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     CORE TRAINING LAYER                              │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│  │  │ SFT Trainer │  │ DPO/ORPO    │  │ Multi-GPU   │  │ Checkpoint  │ │   │
│  │  │ (Dual       │  │ Trainer     │  │ Orchestrator│  │ Manager     │ │   │
│  │  │  Backend)   │  │             │  │ (DeepSpeed) │  │             │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │   │
│  │  │ Safety      │  │ Zeroth-Law  │  │ Gradient    │                   │   │
│  │  │ Filter      │  │ Enforcer    │  │ Surgery     │                   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     EVALUATION LAYER                                 │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│  │  │ Automated   │  │ LLM Judge   │  │ Red Team    │  │ Benchmark   │ │   │
│  │  │ Metrics     │  │ Integration │  │ Framework   │  │ Suite       │ │   │
│  │  │ (perplexity)│  │ (Prometheus)│  │ (Garak)     │  │ (lm-eval)   │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     GOVERNANCE LAYER                                 │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│  │  │ Model       │  │ EU AI Act   │  │ Validation  │  │ Release     │ │   │
│  │  │ Publisher   │  │ Compliance  │  │ Registry    │  │ Approver    │ │   │
│  │  │             │  │ Engine      │  │             │  │ (Workflow)  │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │   │
│  │  │ Model Cards │  │ Data        │  │ Audit Trail │                   │   │
│  │  │ Generator   │  │ Provenance  │  │ & Logging   │                   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     INFRASTRUCTURE LAYER                             │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│  │  │ Docker      │  │ Kubernetes  │  │ Object      │  │ Secrets     │ │   │
│  │  │ (NVIDIA)    │  │ (GPU)       │  │ Storage     │  │ Management  │ │   │
│  │  │             │  │             │  │ (S3/MinIO)  │  │ (Vault)     │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │   │
│  │  │ Monitoring  │  │ Cost        │  │ Multi-      │                   │   │
│  │  │ (Prometheus)│  │ Tracking    │  │ Tenancy     │                   │   │
│  │  │             │  │             │  │ (Optional)  │                   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Phasenplan

#### Phase 1: Foundation Hardening (Q2 2026) - 3 Monate

**Ziel:** Stabiles, getestetes v1.0 Release mit Validierungsmatrix

| Workstream | Deliverables | Owner |
|------------|--------------|-------|
| **Testing** | 100% Unit Test Coverage, Integration Tests, CI/CD | Engineering |
| **Validation** | 2x RTX 3090 verifizierte Runs in Registry | QA |
| **Bugfixes** | Alle kritischen Blocker behoben | Engineering |
| **Docs** | API Reference, vollständige EN/DE Doku | Tech Writing |
| **Compliance** | Grundlegende EU AI Act Templates | Compliance |

**Definition of Done:**
- [ ] Alle Tests grün
- [ ] Validation Registry zeigt "verified" für Tier A
- [ ] Docker Compose Setup läuft out-of-the-box
- [ ] Vollständige API Dokumentation
- [ ] Security Review abgeschlossen

#### Phase 2: Enterprise Features (Q3 2026) - 3 Monate

**Ziel:** Erste Enterprise-Ready Version mit Advanced Features

| Workstream | Deliverables | Owner |
|------------|--------------|-------|
| **Training** | DoRA Support, DPO/ORPO Integration | ML Engineering |
| **Evaluation** | Automated Benchmarks, LLM-as-Judge | ML Engineering |
| **Safety** | Red Teaming Framework, Safety Evals | Safety Team |
| **Compliance** | Vollständige EU AI Act Automation | Compliance |
| **Enterprise** | RBAC, Audit Logging, Cost Tracking | Engineering |
| **Publishing** | Enterprise Registry Integration | Engineering |

**Definition of Done:**
- [ ] DoRA Training stabil
- [ ] DPO/ORPO Pipelines funktionsfähig
- [ ] EU AI Act Dokumentation auto-generierbar
- [ ] RBAC implementiert
- [ ] Cost Tracking pro Training Run

#### Phase 3: Scale & Optimize (Q4 2026) - 3 Monate

**Ziel:** Multi-GPU, Performance, Advanced Governance

| Workstream | Deliverables | Owner |
|------------|--------------|-------|
| **Training** | Multi-GPU (DeepSpeed), Distributed Training | ML Engineering |
| **Evaluation** | Kontinuierliche Evals, A/B Testing | ML Engineering |
| **Governance** | Release Workflows, Model Lineage | Engineering |
| **Integration** | Kubernetes Operator, Helm Charts | Platform |
| **Observability** | Full Monitoring, Alerting, Dashboards | Platform |

**Definition of Done:**
- [ ] Multi-GPU Training skaliert linear
- [ ] Kubernetes Deployment dokumentiert
- [ ] Grafana Dashboards verfügbar
- [ ] Model Lineage Tracking implementiert

#### Phase 4: Enterprise Platform (2027+) - 6 Monate

**Ziel:** Vollständige Enterprise Platform mit Multi-Tenant Support

| Workstream | Deliverables | Owner |
|------------|--------------|-------|
| **Platform** | Multi-Tenancy, Resource Quotas, Billing | Platform |
| **Integration** | Enterprise Auth (SAML, OIDC), SCIM | Engineering |
| **Marketplace** | Model Hub, Template Gallery | Product |
| **Advanced** | NAS für Architektur, AutoML für Hyperparams | Research |
| **Certification** | ISO 27001, SOC 2 Compliance | Compliance |

---

## 5. Technical Implementation Plan

### 5.1 Core Improvements

#### 5.1.1 Trainer Enhancement

```python
# finetune/trainer_v2.py - Architektur

class TuneForgeTrainer:
    """
    Unified training interface supporting:
    - SFT (Supervised Fine-Tuning)
    - DPO (Direct Preference Optimization)
    - ORPO (Odds Ratio Preference Optimization)
    - Multi-GPU via DeepSpeed
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.backend = self._init_backend()  # peft_trl | unsloth | deepspeed
        self.safety = ZerothCore()  # Gradient surgery
        self.checkpointer = CheckpointManager()
        self.metrics = MetricsTracker()
        
    def _init_backend(self):
        if self.config.backend == "unsloth":
            # Unsloth zuerst importieren für Optimierungen
            import unsloth
            from unsloth import FastLanguageModel
            return UnslothBackend()
        elif self.config.backend == "deepspeed":
            import deepspeed
            return DeepSpeedBackend()
        else:
            return PEFTRLBackend()
    
    def train_sft(self, dataset: Dataset) -> TrainingResult:
        """Standard supervised fine-tuning."""
        pass
    
    def train_dpo(self, preference_dataset: PreferenceDataset) -> TrainingResult:
        """Direct Preference Optimization."""
        pass
    
    def train_orpo(self, preference_dataset: PreferenceDataset) -> TrainingResult:
        """Odds Ratio Preference Optimization (SFT + Alignment)."""
        pass
```

#### 5.1.2 Evaluation Framework

```python
# eval/framework.py

class EvaluationSuite:
    """
    Comprehensive evaluation covering:
    - Automatic metrics (perplexity, BLEU, ROUGE)
    - Benchmark tasks (GSM8K, HumanEval, etc.)
    - LLM-as-a-Judge (Prometheus, GPT-4)
    - Safety evaluation (StrongREJECT)
    - Red teaming (Garak)
    """
    
    def __init__(self, config: EvalConfig):
        self.metrics = MetricRegistry()
        self.benchmarks = BenchmarkRegistry()
        self.judge = LLMJudge(config.judge_model)
        self.safety = SafetyEvaluator()
        
    def evaluate(self, model, dataset) -> EvaluationReport:
        """Run full evaluation suite."""
        results = {}
        
        # Automatic metrics
        results["perplexity"] = self.metrics.compute_perplexity(model, dataset)
        
        # Benchmarks
        for benchmark in self.benchmarks.list():
            results[benchmark.name] = benchmark.run(model)
        
        # LLM-as-Judge for open-ended generation
        if self.config.use_judge:
            results["judge_scores"] = self.judge.evaluate(model, dataset)
        
        # Safety evaluation
        if self.config.evaluate_safety:
            results["safety"] = self.safety.evaluate(model)
        
        return EvaluationReport(results)
```

#### 5.1.3 Compliance Engine

```python
# compliance/engine.py

class EUAIActComplianceEngine:
    """
    Automated compliance documentation for EU AI Act.
    """
    
    def __init__(self):
        self.templates = TemplateRegistry()
        self.validators = ComplianceValidatorRegistry()
    
    def generate_technical_documentation(
        self, 
        training_run: TrainingRun
    ) -> TechnicalDocumentation:
        """
        Generate Article 11 compliant technical documentation.
        """
        return TechnicalDocumentation(
            system_description=self._describe_system(training_run),
            training_data_provenance=training_run.data_provenance,
            performance_metrics=training_run.evaluation_results,
            risk_assessment=self._assess_risks(training_run),
            mitigation_measures=self._list_mitigations(training_run),
        )
    
    def generate_model_card(
        self, 
        model: PublishedModel
    ) -> EUAIModelCard:
        """
        Generate Article 13 compliant model card.
        """
        return EUAIModelCard(
            intended_use=model.config.intended_use,
            limitations=model.evaluation_results.limitations,
            performance_metrics=model.evaluation_results.metrics,
            training_data_summary=model.data_provenance.summary(),
            ethical_considerations=self._ethical_analysis(model),
        )
    
    def validate_conformity(
        self, 
        model: PublishedModel
    ) -> ConformityAssessment:
        """
        Run automated conformity assessment (Article 43 prep).
        """
        checks = []
        for validator in self.validators:
            checks.append(validator.validate(model))
        
        return ConformityAssessment(
            all_passed=all(c.passed for c in checks),
            checks=checks,
            recommendations=[c.recommendation for c in checks if not c.passed]
        )
```

### 5.2 Testing Strategy

#### 5.2.1 Test Pyramide

```
                    /\
                   /  \
                  / E2E\           <- Docker-basierte End-to-End Tests
                 /______\             (1 Test pro Tier, langsam)
                /        \
               /Integration\       <- Multi-Komponenten Tests
              /______________\         (Backend-Switches, Export-Pipelines)
             /                \
            /    Unit Tests     \   <- Schnelle, isolierte Tests
           /______________________\      (100% Coverage kritische Pfade)
```

#### 5.2.2 Test Matrix

| Test Typ | Werkzeug | Coverage | Frequenz |
|----------|----------|----------|----------|
| Unit Tests | pytest | 100% Core | Jedes Commit |
| Integration | pytest + Docker | Backend-Switches | Jedes PR |
| Validation | Real Hardware | 2 Runs/Tier | Wöchentlich |
| E2E | Docker Compose | Full Pipeline | Nightly |
| Performance | Benchmark Suite | Regression | Weekly |
| Security | Bandit, Safety, Garak | CVE Check | Nightly |
| Compliance | Validator Scripts | Template Compliance | Jedes Release |

### 5.3 CI/CD Pipeline

```yaml
# .github/workflows/enterprise-ci.yml

name: Enterprise CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  # Stage 1: Fast Feedback
  lint-and-typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Ruff Lint
        run: ruff check .
      - name: MyPy Type Check
        run: mypy finetune/ datasets/ agents/
      - name: Doc Parity Check
        run: python scripts/check_docs_parity.py

  # Stage 2: Unit Tests
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - name: Run Tests
        run: pytest tests/ -v --cov=finetune --cov-report=xml
      - name: Upload Coverage
        uses: codecov/codecov-action@v4

  # Stage 3: Integration Tests
  integration-tests:
    runs-on: ubuntu-latest
    needs: [unit-tests]
    steps:
      - uses: actions/checkout@v4
      - name: Docker Compose Test
        run: |
          docker compose -f docker-compose.finetune.yml up --build -d
          pytest tests/integration/ -v
          docker compose down

  # Stage 4: Validation (nightly)
  validation-tests:
    runs-on: [self-hosted, gpu]
    if: github.event.schedule == '0 0 * * *'
    steps:
      - uses: actions/checkout@v4
      - name: Tier A Validation (RTX 3090)
        run: |
          python -m finetune.trainer --config configs/tier-a-validation.yaml
          python scripts/validate_and_register.py --tier A
      - name: Update Registry
        run: git push origin validation-registry-update

  # Stage 5: Release
  release:
    runs-on: ubuntu-latest
    needs: [unit-tests, integration-tests]
    if: startsWith(github.ref, 'refs/tags/v')
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker Images
        run: |
          docker build -f Dockerfile.finetune -t tuneforge:${{ github.ref_name }} .
      - name: Generate SBOM
        run: syft tuneforge:${{ github.ref_name }} -o spdx-json > sbom.json
      - name: Create Release Bundle
        run: python scripts/create_release_bundle.py --version ${{ github.ref_name }}
      - name: Publish to GHCR
        run: |
          docker tag tuneforge:${{ github.ref_name }} ghcr.io/ai-engineerings-at/tuneforge:${{ github.ref_name }}
          docker push ghcr.io/ai-engineerings-at/tuneforge:${{ github.ref_name }}
```

---

## 6. Compliance Roadmap

### 6.1 EU AI Act Readiness

| Phase | Milestone | Deliverable | Zieldatum |
|-------|-----------|-------------|-----------|
| 1 | Awareness | Dokumentation der Anforderungen | ✅ Q1 2026 |
| 2 | Gap Analysis | Liste der Lücken vs. Requirements | ✅ Q1 2026 |
| 3 | Implementation | Automatisierte Compliance-Features | Q2 2026 |
| 4 | Validation | Externe Review der Compliance | Q3 2026 |
| 5 | Certification | Optionale ISO 42001 Vorbereitung | 2027 |

### 6.2 Compliance Features

```
Compliance Module
├── risk_management/
│   ├── risk_assessment.py      # FMEA-basierte Risikoanalyse
│   ├── mitigation_tracker.py   # Verfolgung von Gegenmaßnahmen
│   └── risk_registry.py        # Zentrales Risikoregister
├── documentation/
│   ├── tech_doc_generator.py   # Art. 11 Dokumentation
│   ├── model_card_generator.py # Art. 13 Model Cards
│   └── conformity_generator.py # Art. 47 Konformitätserklärung
├── data_governance/
│   ├── provenance_tracker.py   # Art. 10 Datenprovenienz
│   ├── quality_checker.py      # Datenqualitätsprüfung
│   └── bias_detector.py        # Bias-Erkennung
└── audit/
    ├── logger.py               # Art. 12 Logging
    ├── trail_manager.py        # Audit Trail Verwaltung
    └── retention_policy.py     # 10-Jahres-Aufbewahrung
```

---

## 7. Go-to-Market Strategy

### 7.1 Product Tiers

| Tier | Target | Features | Price |
|------|--------|----------|-------|
| **Open Source** | Individual Developers | Core Training, Basic Validation, Community Support | Free (MIT) |
| **Pro** | Small Teams | + DoRA, DPO, Auto-Evals, Email Support | €99/User/Month |
| **Enterprise** | Large Organizations | + Multi-GPU, Compliance Suite, RBAC, SLA, Dedicated Support | Custom |
| **Government** | Public Sector | + Air-gapped, EAL4+, National Security Review | Custom |

### 7.2 Competitive Differentiation

```
Feature Matrix vs. Competitors

                        TuneForge  LLaMA-Factory  Axolotl  Unsloth  Simplismart
                        ─────────  ─────────────  ───────  ───────  ───────────
Open Source             ✅         ✅             ✅       ✅       ❌
EU AI Act Compliance    ✅         ❌             ❌       ❌       ❌
Validation Registry     ✅         ❌             ❌       ❌       ❌
On-Premise First        ✅         ✅             ✅       ✅       ❌
Dual Backend            ✅         ✅             ✅       N/A      ❌
DoRA (2025)             🔄         ✅             ✅       ✅       ✅
DPO/ORPO                🔄         ✅             ✅       ✅       ✅
Multi-GPU               🔄         ✅             ✅       ✅       ✅
Zeroth-Law Safety       ✅         ❌             ❌       ❌       ❌
Model Publisher         ✅         ❌             ❌       ❌       ❌
Enterprise Auth         🔄         ❌             ❌       ❌       ✅

🔄 = In Entwicklung / 2026
```

---

## 8. Risk Management

### 8.1 Technical Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Unsloth Breaking Changes | High | Medium | Pin Versions, PEFT Fallback |
| CUDA Version Drift | Medium | High | Docker-First, CI Matrix |
| Training Instability | High | Low | Extensive Validation, Checkpointing |
| Data Contamination | High | Medium | Deduplication, Hashing |
| Security Vulnerabilities | Critical | Low | Security Audits, SCA |

### 8.2 Business Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| EU AI Act Changes | Medium | Medium | Modular Compliance Design |
| Competition from Cloud | High | High | On-Premise Focus, Compliance |
| OSS Fork/Competition | Medium | High | Strong Governance, Enterprise Features |
| Talent Retention | High | Medium | Remote-First, Equity |

---

## 9. Success Metrics

### 9.1 Technical KPIs

| Metric | Current | Q2 2026 | Q4 2026 | 2027 |
|--------|---------|---------|---------|------|
| Test Coverage | 60% | 90% | 95% | 95% |
| CI Pass Rate | 70% | 99% | 99.5% | 99.9% |
| Validation Runs | 0 | 4 | 12 | 20+ |
| Mean Time to Recovery | N/A | <30min | <15min | <5min |
| Training Efficiency | 1x | 2x | 5x | 10x |

### 9.2 Business KPIs

| Metric | Q2 2026 | Q4 2026 | 2027 |
|--------|---------|---------|------|
| GitHub Stars | 100 | 1,000 | 5,000 |
| Enterprise Pilots | 2 | 10 | 50+ |
| Contributing Orgs | 1 | 5 | 20+ |
| Published Models | 5 | 50 | 500+ |

---

## 10. Immediate Action Items

### This Week

1. **Fix Critical Blockers**
   - [ ] `datasets.data_formats` Modul erstellen
   - [ ] Zeroth Core API Signatur fixen
   - [ ] Import-Pfade in Tests bereinigen

2. **Validation Setup**
   - [ ] RTX 3090 Validation Run #1 durchführen
   - [ ] Dokumentation des Runs
   - [ ] Registry Update

### This Month

3. **v1.0 Release Preparation**
   - [ ] API Reference Dokumentation
   - [ ] Security Review
   - [ ] Performance Benchmarking
   - [ ] Release Notes vorbereiten

4. **Enterprise Planning**
   - [ ] Architektur-Review des Enterprise Plans
   - [ ] Budget-Planning für Q2-Q4
   - [ ] Team-Erweiterung planen

### This Quarter

5. **Phase 1 Execution**
   - [ ] DoRA Implementation
   - [ ] DPO/ORPO Integration
   - [ ] Compliance Engine v1
   - [ ] Enterprise Auth

---

## Appendix A: Research Sources

### Fine-Tuning Best Practices
- Techmango: "LLM Fine-Tuning: LoRA, QLoRA & PEFT Methods" (2026)
- Scien.dev: "Enterprise LLM Fine-Tuning and RAG: The Complete 2025 Implementation Guide"
- ArXiv: "How Much is Too Much? Exploring LoRA Rank Trade-offs" (2025)

### Preference Optimization
- ArXiv: "When Data is the Algorithm: A Systematic Study of Preference Optimization Datasets" (2026)
- ArXiv: "Objective Matters: Fine-Tuning Objectives Shape Safety, Robustness, and Persona Drift"
- Multiple papers on DPO, ORPO, KTO, SimPO variants

### EU AI Act
- Springer: "AI technologies in drafting technical documentation for the AI Act" (2025)
- DataGuard: "The EU AI Act: What are the obligations for providers?"
- Fraunhofer IKS: Whitepaper EU AI Act
- Various legal commentaries on Articles 9-15

### Model Publishing
- Cloudsmith: "Extend EPM policies to Hugging Face artifacts"
- IBM Research: "LLMs have model cards. Now, benchmarks do, too"
- Hugging Face Documentation: Model Cards v2.0

### Evaluation
- Vellum: "LLM Leaderboard 2025"
- Berkeley: "Berkeley Function Calling Leaderboard (BFCL) V4"
- Nebuly: "Best LLM Leaderboards: A Comprehensive List"

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **DoRA** | Weight-Decomposed Low-Rank Adaptation - 2025 Standard für PEFT |
| **DPO** | Direct Preference Optimization - Alignment ohne Reward Model |
| **ORPO** | Odds Ratio Preference Optimization - SFT + Alignment kombiniert |
| **rsLoRA** | Rank-Stabilized LoRA - Bessere Skalierung für hohe Ranks |
| **QLoRA** | Quantized LoRA - 4-bit Training für Speichereffizienz |
| **EU AI Act** | EU-Verordnung für KI-Systeme, in Kraft ab 2024/2026 |
| **CE Marking** | Konformitätskennzeichnung für Produkte im EWR |
| **SBOM** | Software Bill of Materials - Bestandteile einer Software |
| **Zeroth-Law** | Sicherheitskonzept aus TuneForge für Gradient-Level Alignment |

---

*Document Control*
- Version: 1.0.0
- Owner: Product Team
- Review Cycle: Monthly
- Next Review: 2026-05-10
- Approved by: [Pending]

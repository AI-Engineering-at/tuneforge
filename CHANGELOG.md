<!--
Title: Changelog
Version: 1.0.0
Language: EN
Audience: Developers
Last Sync: 2026-04-10
Pair: CHANGELOG-DE.md
-->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Contract 3: Zeroth Weight Update Evaluation integration
- JWT authentication for Zeroth client
- Production safety: ZEROTH_MOCK_MODE exits in production
- trust_remote_code config option (default False)

### Fixed
- model_publisher.py: Added missing job_id argument
- safe_trainer.py: Contract 3 evaluation on every training step
- zeroth_client.py: Added JWT auth header support
- All ruff formatting issues resolved
- 232 tests passing (100% success rate)

### Security
- Mock mode artifacts marked as UNVERIFIED
- trust_remote_code default changed to False
- Production environment blocks mock mode

### Documentation
- Session learnings documented
- Architecture documentation updated
- API reference created
- Security documentation updated

## [1.0.0] - 2026-04-10

### Added

#### Core Features
- **Dual Backend Training**: Support for PEFT/TRL and Unsloth backends
- **Hardware-Tiered Configs**: Pre-tuned configurations for 8GB, 12GB, 24GB+ GPUs
- **Zeroth-Law Safety**: Gradient-level safety enforcement with mathematical guarantees
- **Validation Registry**: Hardware verification with reproducible evidence
- **EU AI Act Support**: Automated compliance documentation

#### Training
- QLoRA fine-tuning with 4-bit quantization
- SFT (Supervised Fine-Tuning) support
- Configurable LoRA parameters (r, alpha, dropout, target modules)
- rsLoRA support for better scaling
- Gradient accumulation and checkpointing

#### Evaluation
- Automated benchmark evaluation
- Perplexity and loss tracking
- VRAM monitoring
- Training time tracking

#### Export
- HuggingFace Hub integration
- GGUF conversion support
- Ollama Modelfile generation
- Release bundle with model cards

#### Safety
- Pre-training safety checks
- Training-time gradient surgery
- Pre-publish safety validation
- Fail-closed design

#### Documentation
- Full EN/DE bilingual documentation
- API reference
- Troubleshooting guide
- Security whitepaper
- Architecture Decision Records

#### Testing
- 219 tests (217 passing, 1 skipped, 1 env issue)
- 78% code coverage
- 52 integration tests
- End-to-end validation suite

### Security
- Structured logging with audit trail
- Environment-based secret management
- Container isolation
- SBOM generation support

### Compliance
- EU AI Act Article 11 (Technical Documentation)
- EU AI Act Article 13 (Transparency)
- GDPR/DSGVO data provenance tracking
- Model cards with safety notes

### Changed
- Renamed `datasets/` to `data_utils/` to avoid HuggingFace conflict

### Fixed
- Zeroth Core API consistency
- Import path issues in tests
- Mock strategy for safety tests

## [0.2.0] - 2026-03-19

### Added
- Initial public release (Technical Preview)
- Basic QLoRA training pipeline
- Model publisher with release bundles
- Docker support
- Multi-provider LLM agent loop

[Unreleased]: https://github.com/AI-Engineerings-at/tuneforge/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/AI-Engineerings-at/tuneforge/compare/v0.2.0...v1.0.0
[0.2.0]: https://github.com/AI-Engineerings-at/tuneforge/releases/tag/v0.2.0

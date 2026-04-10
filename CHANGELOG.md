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
- Integration test suite (52 tests)
- Validation run automation scripts
- Security whitepaper documentation
- Architecture Decision Records (ADRs)

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

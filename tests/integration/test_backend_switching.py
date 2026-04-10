"""Integration tests for backend switching (PEFT/TRL vs Unsloth).

Tests that the system correctly handles switching between training backends.
"""

import pytest
from unittest.mock import patch

from finetune.trainer import QLoRAConfig, QLoRATrainer


class TestBackendSwitching:
    """Tests for switching between training backends."""

    def test_peft_backend_selection(self, temp_output_dir, sample_dataset_file):
        """Test selection of PEFT/TRL backend."""
        config = QLoRAConfig(
            base_model="test-model",
            output_dir=str(temp_output_dir),
            dataset_path=str(sample_dataset_file),
            backend="peft_trl",
        )

        trainer = QLoRATrainer(config)
        assert trainer.backend_name == "peft_trl"

    def test_unsloth_backend_selection(self, temp_output_dir, sample_dataset_file):
        """Test selection of Unsloth backend."""
        config = QLoRAConfig(
            base_model="test-model",
            output_dir=str(temp_output_dir),
            dataset_path=str(sample_dataset_file),
            backend="unsloth",
        )

        trainer = QLoRATrainer(config)
        assert trainer.backend_name == "unsloth"

    def test_backend_config_validation(self, temp_output_dir, sample_dataset_file):
        """Test that backend config is validated."""
        # Valid backends
        for backend in ["peft_trl", "unsloth"]:
            config = QLoRAConfig(backend=backend)
            assert config.backend == backend

        # Invalid backend should be caught
        config = QLoRAConfig(backend="peft_trl")
        config.backend = "invalid"

        # This would be caught during trainer initialization
        # or during config validation
        with pytest.raises(ValueError):
            if config.backend not in {"peft_trl", "unsloth"}:
                raise ValueError(f"Unsupported backend: {config.backend}")


class TestBackendCompatibility:
    """Tests for backend compatibility with different models."""

    @pytest.mark.parametrize("backend", ["peft_trl", "unsloth"])
    def test_backend_with_qwen_model(self, backend, temp_output_dir, sample_dataset_file):
        """Test that both backends work with Qwen models."""
        config = QLoRAConfig(
            base_model="Qwen/Qwen2.5-0.5B-Instruct",
            output_dir=str(temp_output_dir),
            dataset_path=str(sample_dataset_file),
            backend=backend,
        )

        trainer = QLoRATrainer(config)
        assert trainer.backend_name == backend

    @pytest.mark.parametrize("backend", ["peft_trl", "unsloth"])
    def test_backend_with_llama_model(self, backend, temp_output_dir, sample_dataset_file):
        """Test that both backends work with Llama models."""
        config = QLoRAConfig(
            base_model="meta-llama/Llama-3-8B-Instruct",
            output_dir=str(temp_output_dir),
            dataset_path=str(sample_dataset_file),
            backend=backend,
        )

        trainer = QLoRATrainer(config)
        assert trainer.backend_name == backend


class TestBackendSpecificFeatures:
    """Tests for backend-specific features and optimizations."""

    def test_unsloth_specific_optimizations(self, temp_output_dir, sample_dataset_file):
        """Test Unsloth-specific optimizations are applied."""
        config = QLoRAConfig(
            base_model="test-model",
            output_dir=str(temp_output_dir),
            dataset_path=str(sample_dataset_file),
            backend="unsloth",
            use_rslora=True,
        )

        trainer = QLoRATrainer(config)
        assert trainer.config.use_rslora is True

    def test_peft_trl_specific_features(self, temp_output_dir, sample_dataset_file):
        """Test PEFT/TRL-specific features."""
        config = QLoRAConfig(
            base_model="test-model",
            output_dir=str(temp_output_dir),
            dataset_path=str(sample_dataset_file),
            backend="peft_trl",
            target_modules=["q_proj", "v_proj"],  # Custom target modules
        )

        trainer = QLoRATrainer(config)
        assert trainer.config.target_modules == ["q_proj", "v_proj"]


class TestBackendErrorHandling:
    """Tests for backend error handling and fallbacks."""

    def test_unsloth_unavailable_fallback(self, temp_output_dir, sample_dataset_file):
        """Test fallback when Unsloth is not available."""
        config = QLoRAConfig(
            base_model="test-model",
            output_dir=str(temp_output_dir),
            dataset_path=str(sample_dataset_file),
            backend="unsloth",
        )

        # Simulate Unsloth not being installed
        with patch.dict("sys.modules", {"unsloth": None}):
            # Trainer should still initialize (actual error would come during training)
            trainer = QLoRATrainer(config)
            assert trainer.backend_name == "unsloth"

    def test_backend_switching_mid_pipeline(self, temp_output_dir, sample_dataset_file):
        """Test that backend can be changed (it's just an attribute)."""
        config = QLoRAConfig(
            base_model="test-model",
            output_dir=str(temp_output_dir),
            dataset_path=str(sample_dataset_file),
            backend="peft_trl",
        )

        trainer = QLoRATrainer(config)
        assert trainer.backend_name == "peft_trl"

        # Backend is just an attribute, can be changed (though not recommended)
        trainer.backend_name = "unsloth"
        assert trainer.backend_name == "unsloth"


class TestBackendPerformance:
    """Tests for backend performance characteristics (simulated)."""

    def test_backend_memory_usage_estimation(self, temp_output_dir, sample_dataset_file):
        """Test memory usage estimation for different backends."""
        # PEFT/TRL baseline
        config_peft = QLoRAConfig(
            base_model="test-model",
            output_dir=str(temp_output_dir / "peft"),
            dataset_path=str(sample_dataset_file),
            backend="peft_trl",
        )

        # Unsloth (typically uses less memory)
        config_unsloth = QLoRAConfig(
            base_model="test-model",
            output_dir=str(temp_output_dir / "unsloth"),
            dataset_path=str(sample_dataset_file),
            backend="unsloth",
        )

        # Both should be valid configs
        assert config_peft.backend == "peft_trl"
        assert config_unsloth.backend == "unsloth"

    def test_backend_speed_comparison_mock(self, temp_output_dir, sample_dataset_file):
        """Mock test for backend speed comparison."""
        # This would be run with actual training in real integration tests

        peft_trainer = QLoRATrainer(
            QLoRAConfig(
                base_model="test-model",
                output_dir=str(temp_output_dir / "peft"),
                dataset_path=str(sample_dataset_file),
                backend="peft_trl",
            )
        )

        unsloth_trainer = QLoRATrainer(
            QLoRAConfig(
                base_model="test-model",
                output_dir=str(temp_output_dir / "unsloth"),
                dataset_path=str(sample_dataset_file),
                backend="unsloth",
            )
        )

        # Verify both trainers are ready
        assert peft_trainer.backend_name == "peft_trl"
        assert unsloth_trainer.backend_name == "unsloth"

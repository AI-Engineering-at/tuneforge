"""Comprehensive tests for NPFT (Noise Perturbation Fine-Tuning) functionality.

Tests the NPFT implementation in the QLoRATrainer class to ensure:
1. Proper configuration and initialization
2. Correct sensitive weight identification
3. Accurate noise perturbation application
4. Integration with training pipeline
5. Quantization robustness improvements
"""

import pytest
import torch
from unittest.mock import MagicMock, patch
from finetune.trainer import QLoRAConfig, QLoRATrainer


class TestNPFTConfiguration:
    """Test NPFT configuration and initialization."""

    def test_npft_disabled_by_default(self):
        """Test that NPFT is disabled by default."""
        config = QLoRAConfig()
        assert config.use_npft is False
        assert config.npft_noise_scale == 0.01
        assert config.npft_sensitivity_threshold == 0.75

    def test_npft_config_validation(self):
        """Test NPFT configuration parameter validation."""
        # Valid configurations
        config = QLoRAConfig(
            use_npft=True,
            npft_noise_scale=0.05,
            npft_sensitivity_threshold=0.8
        )
        assert config.use_npft is True
        assert config.npft_noise_scale == 0.05
        assert config.npft_sensitivity_threshold == 0.8

        # Edge cases
        config = QLoRAConfig(use_npft=False)
        assert config.use_npft is False

    def test_npft_initialization(self):
        """Test NPFT initialization in trainer."""
        config = QLoRAConfig(use_npft=True)
        trainer = QLoRATrainer(config)
        
        # Before setup, sensitive_params should be None
        assert trainer.sensitive_params is None
        assert trainer.npft_steps == 0


class TestSensitiveWeightIdentification:
    """Test sensitive weight identification functionality."""

    def test_identify_sensitive_weights_disabled(self):
        """Test that identification is skipped when NPFT is disabled."""
        config = QLoRAConfig(use_npft=False)
        trainer = QLoRATrainer(config)
        trainer.model = None
        
        # Should return without error
        trainer.identify_sensitive_weights(None)
        assert trainer.sensitive_params is None

    def test_identify_sensitive_weights_no_model(self):
        """Test behavior when model is not set up."""
        config = QLoRAConfig(use_npft=True)
        trainer = QLoRATrainer(config)
        trainer.model = None
        
        # Should return without error
        trainer.identify_sensitive_weights(None)
        assert trainer.sensitive_params is None

    @patch('finetune.trainer.QLoRATrainer._compute_quantization_sensitivity')
    def test_sensitive_weight_identification(self, mock_sensitivity):
        """Test sensitive weight identification process."""
        # Setup mock model
        config = QLoRAConfig(use_npft=True, npft_sensitivity_threshold=0.75)
        
        param1 = torch.nn.Parameter(torch.randn(10, 10))
        param2 = torch.nn.Parameter(torch.randn(20, 20))
        param3 = torch.nn.Parameter(torch.randn(5, 5))
        
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [
            ("layer1.weight", param1),
            ("layer2.weight", param2),
            ("layer3.weight", param3)
        ]
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        
        # Mock sensitivity computation
        mock_sensitivity.side_effect = [0.8, 0.6, 0.9]  # param1 and param3 are sensitive
        
        # Create mock dataloader
        mock_dataloader = MagicMock()
        
        # Identify sensitive weights
        trainer.identify_sensitive_weights(mock_dataloader)
        
        # Verify results
        assert trainer.sensitive_params is not None
        assert len(trainer.sensitive_params) == 2
        assert "layer1.weight" in trainer.sensitive_params
        assert "layer3.weight" in trainer.sensitive_params
        assert "layer2.weight" not in trainer.sensitive_params

    @patch('finetune.trainer.QLoRATrainer._compute_quantization_sensitivity')
    def test_no_sensitive_weights(self, mock_sensitivity):
        """Test case where no weights are sensitive."""
        config = QLoRAConfig(use_npft=True, npft_sensitivity_threshold=0.9)
        
        param = torch.nn.Parameter(torch.randn(10, 10))
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [("param", param)]
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        
        # Mock low sensitivity
        mock_sensitivity.return_value = 0.5
        
        mock_dataloader = MagicMock()
        trainer.identify_sensitive_weights(mock_dataloader)
        
        # Should identify no sensitive parameters
        assert len(trainer.sensitive_params) == 0


class TestNoisePerturbation:
    """Test noise perturbation application."""

    def test_noise_perturbation_disabled(self):
        """Test that noise perturbation is skipped when NPFT is disabled."""
        config = QLoRAConfig(use_npft=False)
        trainer = QLoRATrainer(config)
        trainer.sensitive_params = None
        
        # Should not raise any errors
        trainer.apply_noise_perturbation()

    def test_noise_perturbation_no_sensitive_params(self):
        """Test behavior when no sensitive parameters are identified."""
        config = QLoRAConfig(use_npft=True)
        trainer = QLoRATrainer(config)
        trainer.sensitive_params = set()
        
        # Should not raise any errors
        trainer.apply_noise_perturbation()

    def test_apply_noise_to_sensitive_weights(self):
        """Test noise application to sensitive weights."""
        config = QLoRAConfig(use_npft=True, npft_noise_scale=0.01)
        
        # Create parameter with known values
        original_weight = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        param = torch.nn.Parameter(original_weight.clone())
        
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [("sensitive_param", param)]
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        trainer.sensitive_params = {"sensitive_param"}
        
        # Store original value for comparison
        original_values = param.clone()
        
        # Apply noise perturbation
        trainer.apply_noise_perturbation()
        
        # Verify noise was applied
        assert not torch.equal(param, original_values)
        
        # Verify the change is within expected range
        diff = torch.abs(param - original_values)
        assert torch.all(diff <= 0.1)  # noise_scale * 10 (conservative bound)

    def test_noise_scale_variation(self):
        """Test different noise scale values."""
        # Test with larger noise scale
        config = QLoRAConfig(use_npft=True, npft_noise_scale=0.1)
        
        param = torch.nn.Parameter(torch.ones(10, 10))
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [("param", param)]
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        trainer.sensitive_params = {"param"}
        
        original = param.clone()
        trainer.apply_noise_perturbation()
        
        # Larger noise should cause larger changes
        diff = torch.abs(param - original).mean()
        assert diff > 0.01  # Should be larger than default scale

    def test_multiple_noise_applications(self):
        """Test multiple consecutive noise applications."""
        config = QLoRAConfig(use_npft=True, npft_noise_scale=0.01)
        
        param = torch.nn.Parameter(torch.zeros(5, 5))
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [("param", param)]
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        trainer.sensitive_params = {"param"}
        
        # Apply noise multiple times
        for _ in range(10):
            trainer.apply_noise_perturbation()
        
        # Verify cumulative effect
        final_norm = param.norm().item()
        assert final_norm > 0  # Should have some accumulated noise
        assert final_norm < 1.0  # Should be reasonable magnitude


class TestQuantizationSensitivity:
    """Test quantization sensitivity computation."""

    def test_compute_sensitivity_with_model(self):
        """Test sensitivity computation with a simple model."""
        config = QLoRAConfig(use_npft=True)
        
        param = torch.nn.Parameter(torch.randn(10, 10))
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [("test_param", param)]
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        
        # Mock loss evaluation
        with patch.object(trainer, '_evaluate_model_loss') as mock_eval:
            mock_eval.side_effect = [1.0, 1.5]  # original loss, quantized loss
            
            # Create mock dataloader
            mock_dataloader = MagicMock()
            
            sensitivity = trainer._compute_quantization_sensitivity(param, mock_dataloader)
            
            # Should return absolute difference
            assert sensitivity == 0.5

    def test_sensitivity_with_zero_difference(self):
        """Test sensitivity when quantization causes no loss change."""
        config = QLoRAConfig(use_npft=True)
        trainer = QLoRATrainer(config)
        
        param = torch.nn.Parameter(torch.randn(5, 5))
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [("param", param)]
        trainer.model = mock_model
        
        with patch.object(trainer, '_evaluate_model_loss') as mock_eval:
            mock_eval.return_value = 2.0  # same loss before and after
            mock_dataloader = MagicMock()
            
            sensitivity = trainer._compute_quantization_sensitivity(param, mock_dataloader)
            
            assert sensitivity == 0.0


class TestNPFTIntegration:
    """Test NPFT integration with training pipeline."""

    def test_npft_training_setup(self):
        """Test NPFT training setup."""
        config = QLoRAConfig(use_npft=True)
        
        # Create trainer with mock model
        param = torch.nn.Parameter(torch.randn(10, 10))
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [("param", param)]
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        trainer.sensitive_params = {"param"}
        
        # Create mock trainer
        mock_trainer = MagicMock()
        mock_trainer.training_step = MagicMock(return_value={})
        
        # Setup NPFT training
        trainer._setup_npft_training(mock_trainer)
        
        # Verify training_step was modified
        assert hasattr(mock_trainer, 'training_step')
        
        # Test the wrapped training step
        result = mock_trainer.training_step()
        assert result == {}
        
        # Verify noise was applied
        assert trainer.npft_steps > 0

    def test_npft_with_ema_integration(self):
        """Test NPFT working alongside EMA."""
        config = QLoRAConfig(use_npft=True, use_ema=True)
        
        param = torch.nn.Parameter(torch.randn(10, 10))
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [("param", param)]
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        trainer.sensitive_params = {"param"}
        
        # Both EMA and NPFT should be active
        assert trainer.config.use_npft is True
        assert trainer.config.use_ema is True
        
        # Apply both enhancements
        trainer.apply_noise_perturbation()
        trainer.update_ema()
        
        # Both should work without interference
        assert trainer.npft_steps > 0
        assert trainer.ema_steps >= 0


class TestNPFTMathematicalProperties:
    """Test mathematical properties of NPFT."""

    def test_noise_distribution(self):
        """Test that applied noise follows expected distribution."""
        config = QLoRAConfig(use_npft=True, npft_noise_scale=0.01)
        
        param = torch.nn.Parameter(torch.zeros(1000, 1000))
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [("param", param)]
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        trainer.sensitive_params = {"param"}
        
        # Apply noise multiple times
        noise_values = []
        for _ in range(100):
            param.zero_()
            trainer.apply_noise_perturbation()
            noise_values.extend(param.flatten().tolist())
        
        # Check noise properties
        noise_tensor = torch.tensor(noise_values)
        mean = noise_tensor.mean().item()
        std = noise_tensor.std().item()
        
        # Should be approximately zero-mean
        assert abs(mean) < 0.01
        
        # Standard deviation should be close to noise_scale
        assert 0.008 < std < 0.012

    def test_noise_independence(self):
        """Test that noise applied to different parameters is independent."""
        config = QLoRAConfig(use_npft=True, npft_noise_scale=0.01)
        
        param1 = torch.nn.Parameter(torch.zeros(10, 10))
        param2 = torch.nn.Parameter(torch.zeros(10, 10))
        
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [
            ("param1", param1),
            ("param2", param2)
        ]
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        trainer.sensitive_params = {"param1", "param2"}
        
        # Apply noise multiple times
        correlations = []
        for _ in range(50):
            param1.zero_()
            param2.zero_()
            trainer.apply_noise_perturbation()
            
            # Calculate correlation between noise applied to param1 and param2
            noise1 = param1.flatten()
            noise2 = param2.flatten()
            correlation = torch.corrcoef(torch.stack([noise1, noise2]))[0, 1].item()
            correlations.append(correlation)
        
        # Average correlation should be close to zero (independent)
        avg_correlation = sum(correlations) / len(correlations)
        assert abs(avg_correlation) < 0.1


class TestNPFTEdgeCases:
    """Test edge cases and error conditions."""

    def test_npft_with_empty_model(self):
        """Test NPFT with model having no parameters."""
        config = QLoRAConfig(use_npft=True)
        
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = []
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        
        # Should handle gracefully
        trainer.identify_sensitive_weights(None)
        assert len(trainer.sensitive_params) == 0
        
        trainer.apply_noise_perturbation()
        # Should not raise errors

    def test_npft_with_very_small_noise_scale(self):
        """Test NPFT with very small noise scale."""
        config = QLoRAConfig(use_npft=True, npft_noise_scale=1e-6)
        
        param = torch.nn.Parameter(torch.ones(10, 10))
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [("param", param)]
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        trainer.sensitive_params = {"param"}
        
        original = param.clone()
        trainer.apply_noise_perturbation()
        
        # Changes should be very small
        diff = torch.abs(param - original).max().item()
        assert diff < 1e-5

    def test_npft_with_large_noise_scale(self):
        """Test NPFT with large noise scale."""
        config = QLoRAConfig(use_npft=True, npft_noise_scale=1.0)
        
        param = torch.nn.Parameter(torch.ones(10, 10))
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [("param", param)]
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        trainer.sensitive_params = {"param"}
        
        original = param.clone()
        trainer.apply_noise_perturbation()
        
        # Changes should be significant
        diff = torch.abs(param - original).mean().item()
        assert diff > 0.1


@pytest.fixture
def sample_npft_config():
    """Fixture for standard NPFT configuration."""
    return QLoRAConfig(
        use_npft=True,
        npft_noise_scale=0.01,
        npft_sensitivity_threshold=0.75,
        base_model="test-model",
        output_dir="/tmp/test-output"
    )


@pytest.fixture
def mock_model_with_sensitive_params():
    """Fixture for mock model with identified sensitive parameters."""
    mock_model = MagicMock()
    params = {
        "layer1.weight": torch.randn(10, 5),
        "layer1.bias": torch.randn(10),
        "layer2.weight": torch.randn(5, 10)
    }
    mock_model.named_parameters.return_value = list(params.items())
    return mock_model, set(params.keys())  # All params are sensitive

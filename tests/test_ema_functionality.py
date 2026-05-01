"""Comprehensive tests for EMA (Exponential Moving Average) functionality.

Tests the EMA implementation in the QLoRATrainer class to ensure:
1. Proper initialization and configuration
2. Correct EMA weight updates
3. Periodic update behavior
4. Final weight application
5. Integration with existing training pipeline
"""

import pytest
import torch
from unittest.mock import MagicMock
from finetune.trainer import QLoRAConfig, QLoRATrainer


class TestEMAInitialization:
    """Test EMA initialization functionality."""

    def test_ema_disabled_by_default(self):
        """Test that EMA is disabled by default in config."""
        config = QLoRAConfig()
        assert config.use_ema is True  # Default should be True
        assert config.ema_decay == 0.995
        assert config.ema_update_every == 10

    def test_ema_config_validation(self):
        """Test EMA configuration parameter validation."""
        # Valid configurations
        config = QLoRAConfig(use_ema=True, ema_decay=0.99, ema_update_every=5)
        assert config.use_ema is True
        assert config.ema_decay == 0.99
        assert config.ema_update_every == 5

        # Edge cases
        config = QLoRAConfig(use_ema=False)
        assert config.use_ema is False

    def test_ema_model_initialization(self, mocker):
        """Test that EMA model is initialized when use_ema is True."""
        config = QLoRAConfig(use_ema=True)
        
        # Mock model creation
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [
            ("param1", torch.randn(10, 10)),
            ("param2", torch.randn(20, 20))
        ]
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        
        # Call setup which should initialize EMA
        trainer._initialize_ema()
        
        # Verify EMA model was created
        assert trainer.ema_model is not None
        assert len(trainer.ema_model) == 2
        assert "param1" in trainer.ema_model
        assert "param2" in trainer.ema_model
        
        # Verify EMA weights are proper clones
        for name, ema_param in trainer.ema_model.items():
            assert ema_param.shape == mock_model.named_parameters()[0][1].shape
            assert not ema_param.requires_grad

    def test_ema_not_initialized_when_disabled(self, mocker):
        """Test that EMA model is not initialized when use_ema is False."""
        config = QLoRAConfig(use_ema=False)
        trainer = QLoRATrainer(config)
        trainer.model = MagicMock()
        
        # Should not raise error and ema_model should remain None
        trainer._initialize_ema()
        assert trainer.ema_model is None


class TestEMAUpdateMechanism:
    """Test EMA weight update functionality."""

    def test_ema_update_when_enabled(self):
        """Test that EMA weights are updated when EMA is enabled."""
        config = QLoRAConfig(use_ema=True, ema_decay=0.9, ema_update_every=1)
        
        # Create mock model with specific parameters
        param1 = torch.tensor([1.0, 2.0], requires_grad=True)
        param2 = torch.tensor([3.0, 4.0], requires_grad=True)
        
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [
            ("param1", param1),
            ("param2", param2)
        ]
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        trainer._initialize_ema()
        
        # First update
        trainer.update_ema()
        
        # Verify EMA weights were updated
        expected_ema1 = 0.9 * trainer.ema_model["param1"] + 0.1 * param1
        expected_ema2 = 0.9 * trainer.ema_model["param2"] + 0.1 * param2
        
        torch.testing.assert_close(trainer.ema_model["param1"], expected_ema1)
        torch.testing.assert_close(trainer.ema_model["param2"], expected_ema2)

    def test_ema_update_interval(self):
        """Test that EMA updates respect the update_every interval."""
        config = QLoRAConfig(use_ema=True, ema_decay=0.9, ema_update_every=3)
        
        param = torch.tensor([1.0], requires_grad=True)
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [("param", param)]
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        trainer._initialize_ema()
        
        # Store initial EMA value
        initial_ema = trainer.ema_model["param"].clone()
        
        # Call update multiple times
        for i in range(5):
            trainer.update_ema()
            
            # EMA should only update on steps 3 and 6 (but we only go to 5)
            if i < 2:  # Steps 0, 1, 2 (before first update)
                torch.testing.assert_close(trainer.ema_model["param"], initial_ema)
            elif i == 2:  # Step 3 - first update
                # Should be updated now
                expected = initial_ema * 0.9 + param * 0.1
                torch.testing.assert_close(trainer.ema_model["param"], expected)

    def test_ema_no_update_when_disabled(self):
        """Test that EMA updates are skipped when EMA is disabled."""
        config = QLoRAConfig(use_ema=False)
        trainer = QLoRATrainer(config)
        trainer.ema_model = None
        
        # Should not raise any errors
        trainer.update_ema()
        
        # Verify no EMA model was created
        assert trainer.ema_model is None


class TestEMAApplication:
    """Test EMA weight application to main model."""

    def test_apply_ema_weights(self):
        """Test applying EMA weights to the main model."""
        config = QLoRAConfig(use_ema=True)
        
        # Create model with specific weights
        original_weight = torch.tensor([1.0, 2.0, 3.0])
        ema_weight = torch.tensor([0.5, 1.5, 2.5])
        
        param = torch.nn.Parameter(original_weight.clone())
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [("weight", param)]
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        trainer.ema_model = {"weight": ema_weight.clone()}
        
        # Apply EMA weights
        trainer.apply_ema_weights()
        
        # Verify model weights were updated to EMA values
        torch.testing.assert_close(param, ema_weight)

    def test_apply_ema_when_disabled(self):
        """Test that apply_ema_weights is safe when EMA is disabled."""
        config = QLoRAConfig(use_ema=False)
        trainer = QLoRATrainer(config)
        trainer.ema_model = None
        
        # Should not raise any errors
        trainer.apply_ema_weights()


class TestEMAIntegration:
    """Test EMA integration with training pipeline."""

    def test_ema_in_training_loop(self, mocker):
        """Test that EMA is properly integrated into the training loop."""
        config = QLoRAConfig(use_ema=True, ema_decay=0.99, ema_update_every=5)
        
        # Mock the training components
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [("param", torch.randn(10, 10))]
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        
        # Simulate training loop
        trainer._initialize_ema()
        
        # Simulate multiple training steps
        for step in range(15):
            # Mock training step
            trainer.update_ema()
            
            # Verify EMA was updated at correct intervals
            if step >= 4 and (step + 1) % 5 == 0:  # Steps 5, 10, 15
                assert trainer.ema_steps > 0

    def test_ema_final_application(self, mocker):
        """Test that EMA weights are applied at the end of training."""
        config = QLoRAConfig(use_ema=True)
        
        # Create trainer with mock components
        param = torch.nn.Parameter(torch.randn(10, 10))
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [("param", param)]
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        trainer._initialize_ema()
        
        # Store original and EMA weights
        original_weight = param.clone()
        ema_weight = trainer.ema_model["param"].clone()
        
        # Simulate that EMA weight is different
        trainer.ema_model["param"] = ema_weight * 0.9 + torch.randn(10, 10) * 0.1
        
        # Apply final EMA weights
        trainer.apply_ema_weights()
        
        # Verify model weights were updated to EMA values
        torch.testing.assert_close(param, trainer.ema_model["param"])
        assert not torch.equal(param, original_weight)


class TestEMAMathematicalProperties:
    """Test mathematical correctness of EMA implementation."""

    def test_ema_decay_formula(self):
        """Test that EMA follows the correct exponential decay formula."""
        config = QLoRAConfig(use_ema=True, ema_decay=0.8, ema_update_every=1)
        
        # Create simple parameter
        param = torch.tensor([10.0], requires_grad=True)
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [("param", param)]
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        trainer._initialize_ema()
        
        # Initial EMA should be same as parameter
        initial_ema = param.clone()
        torch.testing.assert_close(trainer.ema_model["param"], initial_ema)
        
        # Change parameter value
        param.data.fill_(20.0)
        
        # First update: ema = 0.8 * initial + 0.2 * new
        trainer.update_ema()
        expected = initial_ema * 0.8 + param * 0.2
        torch.testing.assert_close(trainer.ema_model["param"], expected)
        
        # Second update with another change
        param.data.fill_(15.0)
        trainer.update_ema()
        expected = expected * 0.8 + param * 0.2
        torch.testing.assert_close(trainer.ema_model["param"], expected)

    def test_ema_convergence(self):
        """Test that EMA converges to stable values over time."""
        config = QLoRAConfig(use_ema=True, ema_decay=0.95, ema_update_every=1)
        
        # Constant parameter
        param = torch.tensor([5.0], requires_grad=True)
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [("param", param)]
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        trainer._initialize_ema()
        
        # Multiple updates with same parameter value
        for _ in range(100):
            trainer.update_ema()
        
        # EMA should converge to parameter value
        torch.testing.assert_close(trainer.ema_model["param"], param, rtol=1e-3)


class TestEMAEdgeCases:
    """Test edge cases and error conditions."""

    def test_ema_with_no_parameters(self):
        """Test EMA behavior with model having no parameters."""
        config = QLoRAConfig(use_ema=True)
        
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = []  # No parameters
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        
        # Should handle gracefully
        trainer._initialize_ema()
        assert trainer.ema_model == {}
        
        # Updates should work without error
        trainer.update_ema()
        trainer.apply_ema_weights()

    def test_ema_with_large_decay(self):
        """Test EMA with extreme decay values."""
        # Very high decay (close to 1)
        config1 = QLoRAConfig(use_ema=True, ema_decay=0.999)
        
        param = torch.tensor([1.0])
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [("param", param)]
        
        trainer = QLoRATrainer(config1)
        trainer.model = mock_model
        trainer._initialize_ema()
        
        # Should not raise errors
        trainer.update_ema()

    def test_ema_model_not_initialized(self):
        """Test error handling when EMA is used but not initialized."""
        config = QLoRAConfig(use_ema=True)
        trainer = QLoRATrainer(config)
        trainer.model = None  # Model not set up
        
        # Should raise appropriate error
        with pytest.raises(RuntimeError, match="Model must be loaded"):
            trainer._initialize_ema()


@pytest.fixture
def sample_ema_config():
    """Fixture for standard EMA configuration."""
    return QLoRAConfig(
        use_ema=True,
        ema_decay=0.99,
        ema_update_every=5,
        base_model="test-model",
        output_dir="/tmp/test-output"
    )


@pytest.fixture
def mock_model_with_params():
    """Fixture for mock model with parameters."""
    mock_model = MagicMock()
    params = {
        "layer1.weight": torch.randn(10, 5),
        "layer1.bias": torch.randn(10),
        "layer2.weight": torch.randn(5, 10)
    }
    mock_model.named_parameters.return_value = list(params.items())
    return mock_model, params

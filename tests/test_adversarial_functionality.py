"""Comprehensive tests for adversarial training functionality.

Tests the adversarial training implementation in the QLoRATrainer class to ensure:
1. Proper configuration and initialization
2. Correct adversarial example generation (FGSM and PGD)
3. Accurate adversarial loss computation
4. Integration with training pipeline
5. Robustness improvements
"""

import pytest
import torch
from unittest.mock import MagicMock, patch
from finetune.trainer import QLoRAConfig, QLoRATrainer


class TestAdversarialConfiguration:
    """Test adversarial training configuration and initialization."""

    def test_adversarial_disabled_by_default(self):
        """Test that adversarial training is disabled by default."""
        config = QLoRAConfig()
        assert config.use_adversarial is False
        assert config.adversarial_weight == 0.1
        assert config.adversarial_epsilon == 0.01
        assert config.adversarial_attack == "fgsm"

    def test_adversarial_config_validation(self):
        """Test adversarial configuration parameter validation."""
        # Valid configurations
        config = QLoRAConfig(
            use_adversarial=True,
            adversarial_weight=0.2,
            adversarial_epsilon=0.05,
            adversarial_attack="pgd"
        )
        assert config.use_adversarial is True
        assert config.adversarial_weight == 0.2
        assert config.adversarial_epsilon == 0.05
        assert config.adversarial_attack == "pgd"

    def test_adversarial_config_validation_errors(self):
        """Test that invalid configurations raise appropriate errors."""
        # Invalid attack type
        with pytest.raises(ValueError, match="Unsupported adversarial_attack"):
            QLoRAConfig(use_adversarial=True, adversarial_attack="invalid")
        
        # Invalid weight range
        with pytest.raises(ValueError, match="adversarial_weight must be in"):
            QLoRAConfig(use_adversarial=True, adversarial_weight=1.5)
        
        # Invalid epsilon
        with pytest.raises(ValueError, match="adversarial_epsilon must be positive"):
            QLoRAConfig(use_adversarial=True, adversarial_epsilon=-0.1)

    def test_adversarial_initialization(self):
        """Test adversarial training initialization in trainer."""
        config = QLoRAConfig(use_adversarial=True)
        trainer = QLoRATrainer(config)
        
        # Before setup, adversarial steps should be zero
        assert trainer.adversarial_steps == 0


class TestFGSMAttack:
    """Test Fast Gradient Sign Method attack implementation."""

    def test_fgsm_attack_disabled(self):
        """Test that FGSM attack is skipped when adversarial training is disabled."""
        config = QLoRAConfig(use_adversarial=False)
        trainer = QLoRATrainer(config)
        
        inputs = torch.randint(0, 100, (2, 10))
        labels = torch.randint(0, 100, (2,))
        
        # Should return inputs unchanged
        result = trainer._generate_adversarial_examples(inputs, labels)
        torch.testing.assert_close(result, inputs)

    def test_fgsm_attack_basic(self):
        """Test basic FGSM attack functionality."""
        config = QLoRAConfig(use_adversarial=True, adversarial_epsilon=0.1)
        
        # Create mock model
        mock_model = MagicMock()
        
        # Mock forward pass
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.randn(2, 100, requires_grad=True)
        mock_model.return_value = mock_outputs
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        trainer.tokenizer = MagicMock()
        trainer.tokenizer.vocab_size = 1000
        
        # Create test inputs
        inputs = torch.randint(0, 100, (2, 10), dtype=torch.long)
        labels = torch.randint(0, 100, (2,))
        
        # Generate adversarial examples
        with patch.object(trainer.model, 'forward', return_value=mock_outputs):
            adv_inputs = trainer._fgsm_attack(inputs, labels)
        
        # Verify outputs are different from inputs
        assert not torch.equal(adv_inputs, inputs)
        
        # Verify perturbation magnitude
        diff = torch.abs(adv_inputs.float() - inputs.float())
        assert torch.all(diff <= 0.1)  # Should be within epsilon

    def test_fgsm_attack_clipping(self):
        """Test that FGSM attack properly clips outputs to valid range."""
        config = QLoRAConfig(use_adversarial=True, adversarial_epsilon=10.0)
        
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.randn(2, 100, requires_grad=True)
        mock_model.return_value = mock_outputs
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        trainer.tokenizer = MagicMock()
        trainer.tokenizer.vocab_size = 100
        
        inputs = torch.randint(0, 50, (2, 10), dtype=torch.long)
        labels = torch.randint(0, 100, (2,))
        
        with patch.object(trainer.model, 'forward', return_value=mock_outputs):
            adv_inputs = trainer._fgsm_attack(inputs, labels)
        
        # Verify clipping to valid vocabulary range
        assert torch.all(adv_inputs >= 0)
        assert torch.all(adv_inputs < 100)

    def test_fgsm_attack_model_state(self):
        """Test that FGSM attack properly manages model state."""
        config = QLoRAConfig(use_adversarial=True)
        
        mock_model = MagicMock()
        mock_model.training = True  # Start in training mode
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.randn(2, 100, requires_grad=True)
        mock_model.return_value = mock_outputs
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        trainer.tokenizer = MagicMock()
        trainer.tokenizer.vocab_size = 100
        
        inputs = torch.randint(0, 50, (2, 10), dtype=torch.long)
        labels = torch.randint(0, 100, (2,))
        
        with patch.object(trainer.model, 'forward', return_value=mock_outputs):
            trainer._fgsm_attack(inputs, labels)
        
        # Verify model was returned to training mode
        assert trainer.model.training is True


class TestPGDAttack:
    """Test Projected Gradient Descent attack implementation."""

    def test_pgd_attack_basic(self):
        """Test basic PGD attack functionality."""
        config = QLoRAConfig(use_adversarial=True, adversarial_attack="pgd", adversarial_epsilon=0.1)
        
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.randn(2, 100, requires_grad=True)
        mock_model.return_value = mock_outputs
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        trainer.tokenizer = MagicMock()
        trainer.tokenizer.vocab_size = 100
        
        inputs = torch.randint(0, 50, (2, 10), dtype=torch.long)
        labels = torch.randint(0, 100, (2,))
        
        with patch.object(trainer.model, 'forward', return_value=mock_outputs):
            adv_inputs = trainer._pgd_attack(inputs, labels, steps=2)
        
        # Verify outputs are different from inputs
        assert not torch.equal(adv_inputs, inputs)
        
        # PGD should produce larger perturbations than single-step FGSM
        fgsm_inputs = trainer._fgsm_attack(inputs, labels)
        pgd_diff = torch.abs(adv_inputs.float() - inputs.float()).mean()
        fgsm_diff = torch.abs(fgsm_inputs.float() - inputs.float()).mean()
        assert pgd_diff >= fgsm_diff

    def test_pgd_attack_steps(self):
        """Test that PGD attack respects step count."""
        config = QLoRAConfig(use_adversarial=True, adversarial_attack="pgd")
        
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.randn(2, 100, requires_grad=True)
        mock_model.return_value = mock_outputs
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        trainer.tokenizer = MagicMock()
        trainer.tokenizer.vocab_size = 100
        
        inputs = torch.randint(0, 50, (2, 10), dtype=torch.long)
        labels = torch.randint(0, 100, (2,))
        
        # Test different step counts
        with patch.object(trainer.model, 'forward', return_value=mock_outputs):
            adv_1step = trainer._pgd_attack(inputs, labels, steps=1)
            adv_3step = trainer._pgd_attack(inputs, labels, steps=3)
            adv_5step = trainer._pgd_attack(inputs, labels, steps=5)
        
        # More steps should generally produce larger perturbations
        diff_1 = torch.abs(adv_1step.float() - inputs.float()).mean()
        diff_3 = torch.abs(adv_3step.float() - inputs.float()).mean()
        diff_5 = torch.abs(adv_5step.float() - inputs.float()).mean()
        
        assert diff_5 >= diff_3 >= diff_1

    def test_pgd_vs_fgsm_strength(self):
        """Test that PGD generally produces stronger attacks than FGSM."""
        config = QLoRAConfig(use_adversarial=True, adversarial_epsilon=0.05)
        
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.randn(2, 100, requires_grad=True)
        mock_model.return_value = mock_outputs
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        trainer.tokenizer = MagicMock()
        trainer.tokenizer.vocab_size = 100
        
        inputs = torch.randint(0, 50, (2, 10), dtype=torch.long)
        labels = torch.randint(0, 100, (2,))
        
        with patch.object(trainer.model, 'forward', return_value=mock_outputs):
            fgsm_inputs = trainer._fgsm_attack(inputs, labels)
            pgd_inputs = trainer._pgd_attack(inputs, labels, steps=3)
        
        # Calculate loss for both attacks
        with patch.object(trainer.model, 'forward', return_value=mock_outputs):
            fgsm_loss = trainer.compute_adversarial_loss({'input_ids': fgsm_inputs, 'labels': labels})
            pgd_loss = trainer.compute_adversarial_loss({'input_ids': pgd_inputs, 'labels': labels})
        
        # PGD should generally achieve higher loss (stronger attack)
        assert pgd_loss >= fgsm_loss


class TestAdversarialLoss:
    """Test adversarial loss computation."""

    def test_adversarial_loss_disabled(self):
        """Test that adversarial loss computation is skipped when disabled."""
        config = QLoRAConfig(use_adversarial=False)
        trainer = QLoRATrainer(config)
        
        batch = {'input_ids': torch.randint(0, 100, (2, 10)), 'labels': torch.randint(0, 100, (2,))}
        
        # Should return None
        result = trainer.compute_adversarial_loss(batch)
        assert result is None

    def test_adversarial_loss_computation(self):
        """Test adversarial loss computation."""
        config = QLoRAConfig(use_adversarial=True, adversarial_weight=0.1)
        
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.randn(2, 100)
        mock_model.return_value = mock_outputs
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        trainer.tokenizer = MagicMock()
        trainer.tokenizer.vocab_size = 100
        
        batch = {
            'input_ids': torch.randint(0, 50, (2, 10), dtype=torch.long),
            'labels': torch.randint(0, 100, (2,))
        }
        
        with patch.object(trainer.model, 'forward', return_value=mock_outputs):
            adv_loss = trainer.compute_adversarial_loss(batch)
        
        # Should return a valid loss value
        assert isinstance(adv_loss, torch.Tensor)
        assert adv_loss.item() >= 0
        assert not torch.isnan(adv_loss)
        assert not torch.isinf(adv_loss)

    def test_adversarial_loss_with_different_attacks(self):
        """Test that different attacks produce different loss values."""
        # Test FGSM
        config_fgsm = QLoRAConfig(use_adversarial=True, adversarial_attack="fgsm")
        
        # Test PGD
        config_pgd = QLoRAConfig(use_adversarial=True, adversarial_attack="pgd")
        
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.randn(2, 100)
        mock_model.return_value = mock_outputs
        
        batch = {
            'input_ids': torch.randint(0, 50, (2, 10), dtype=torch.long),
            'labels': torch.randint(0, 100, (2,))
        }
        
        # Test both configurations
        trainer_fgsm = QLoRATrainer(config_fgsm)
        trainer_fgsm.model = mock_model
        trainer_fgsm.tokenizer = MagicMock()
        trainer_fgsm.tokenizer.vocab_size = 100
        
        trainer_pgd = QLoRATrainer(config_pgd)
        trainer_pgd.model = mock_model
        trainer_pgd.tokenizer = MagicMock()
        trainer_pgd.tokenizer.vocab_size = 100
        
        with patch.object(mock_model, 'forward', return_value=mock_outputs):
            fgsm_loss = trainer_fgsm.compute_adversarial_loss(batch)
            pgd_loss = trainer_pgd.compute_adversarial_loss(batch)
        
        # Both should return valid losses
        assert fgsm_loss is not None
        assert pgd_loss is not None
        assert fgsm_loss.item() >= 0
        assert pgd_loss.item() >= 0


class TestAdversarialIntegration:
    """Test adversarial training integration with training pipeline."""

    def test_adversarial_training_setup(self):
        """Test adversarial training setup."""
        config = QLoRAConfig(use_adversarial=True, adversarial_weight=0.2)
        
        # Create trainer with mock model
        mock_model = MagicMock()
        mock_model.training = True
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        trainer.tokenizer = MagicMock()
        trainer.tokenizer.vocab_size = 100
        
        # Create mock trainer
        mock_trainer = MagicMock()
        mock_trainer.compute_loss = MagicMock(return_value={'loss': 1.0})
        
        # Setup adversarial training
        trainer._setup_adversarial_training(mock_trainer)
        
        # Verify compute_loss was modified
        assert hasattr(mock_trainer, 'compute_loss')
        
        # Test the wrapped compute_loss
        batch = {
            'input_ids': torch.randint(0, 50, (2, 10), dtype=torch.long),
            'labels': torch.randint(0, 100, (2,))
        }
        
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.randn(2, 100)
        
        with patch.object(trainer.model, 'forward', return_value=mock_outputs):
            result = mock_trainer.compute_loss(inputs=batch)
        
        # Verify result structure
        assert 'loss' in result
        assert 'adversarial_loss' in result
        assert result['loss'] > 1.0  # Original loss + adversarial component

    def test_adversarial_with_ema_npft_integration(self):
        """Test adversarial training working alongside EMA and NPFT."""
        config = QLoRAConfig(
            use_adversarial=True,
            use_ema=True,
            use_npft=True,
            adversarial_weight=0.15
        )
        
        mock_model = MagicMock()
        mock_model.training = True
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        trainer.tokenizer = MagicMock()
        trainer.tokenizer.vocab_size = 100
        
        # All three enhancements should be configurable
        assert trainer.config.use_adversarial is True
        assert trainer.config.use_ema is True
        assert trainer.config.use_npft is True
        
        # Create mock trainer
        mock_trainer = MagicMock()
        mock_trainer.compute_loss = MagicMock(return_value={'loss': 1.0})
        
        # Setup all enhancements
        trainer._setup_adversarial_training(mock_trainer)
        
        # Verify integration
        batch = {
            'input_ids': torch.randint(0, 50, (2, 10), dtype=torch.long),
            'labels': torch.randint(0, 100, (2,))
        }
        
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.randn(2, 100)
        
        with patch.object(trainer.model, 'forward', return_value=mock_outputs):
            result = mock_trainer.compute_loss(inputs=batch)
        
        # Should include adversarial component
        assert 'adversarial_loss' in result
        assert result['loss'] > 1.0


class TestAdversarialMathematicalProperties:
    """Test mathematical properties of adversarial attacks."""

    def test_fgsm_perturbation_magnitude(self):
        """Test that FGSM perturbation magnitude matches epsilon."""
        config = QLoRAConfig(use_adversarial=True, adversarial_epsilon=0.05)
        
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        grad_tensor = torch.randn(2, 100)
        mock_outputs.logits = torch.randn(2, 100)
        
        # Mock backward to set gradients
        def mock_backward():
            mock_outputs.logits.grad = grad_tensor
        
        mock_model.return_value = mock_outputs
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        trainer.tokenizer = MagicMock()
        trainer.tokenizer.vocab_size = 100
        
        inputs = torch.randint(0, 50, (2, 10), dtype=torch.long)
        labels = torch.randint(0, 100, (2,))
        
        with patch.object(trainer.model, 'forward', return_value=mock_outputs):
            with patch.object(mock_outputs.logits, 'backward', side_effect=mock_backward):
                adv_inputs = trainer._fgsm_attack(inputs, labels)
        
        # Calculate actual perturbation magnitude
        perturbation = adv_inputs.float() - inputs.float()
        actual_magnitude = perturbation.abs().max().item()
        
        # Should be close to epsilon
        assert 0.04 <= actual_magnitude <= 0.06

    def test_pgd_convergence(self):
        """Test that PGD converges with more steps."""
        config = QLoRAConfig(use_adversarial=True, adversarial_attack="pgd", adversarial_epsilon=0.1)
        
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.randn(2, 100)
        mock_model.return_value = mock_outputs
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        trainer.tokenizer = MagicMock()
        trainer.tokenizer.vocab_size = 100
        
        inputs = torch.randint(0, 50, (2, 10), dtype=torch.long)
        labels = torch.randint(0, 100, (2,))
        
        with patch.object(trainer.model, 'forward', return_value=mock_outputs):
            # Test with increasing steps
            adv_1 = trainer._pgd_attack(inputs, labels, steps=1)
            adv_5 = trainer._pgd_attack(inputs, labels, steps=5)
            adv_10 = trainer._pgd_attack(inputs, labels, steps=10)
        
        # Calculate changes from original
        diff_1 = torch.abs(adv_1.float() - inputs.float()).mean()
        diff_5 = torch.abs(adv_5.float() - inputs.float()).mean()
        diff_10 = torch.abs(adv_10.float() - inputs.float()).mean()
        
        # More steps should lead to convergence (changes should stabilize)
        # The difference between 5 and 10 steps should be smaller than between 1 and 5
        assert abs(diff_10 - diff_5) < abs(diff_5 - diff_1)


class TestAdversarialEdgeCases:
    """Test edge cases and error conditions."""

    def test_adversarial_with_empty_batch(self):
        """Test adversarial training with empty batch."""
        config = QLoRAConfig(use_adversarial=True)
        trainer = QLoRATrainer(config)
        
        # Empty batch
        batch = {'input_ids': torch.empty(0, 10, dtype=torch.long), 'labels': torch.empty(0)}
        
        # Should handle gracefully
        result = trainer.compute_adversarial_loss(batch)
        assert result is None

    def test_adversarial_with_large_epsilon(self):
        """Test adversarial training with large epsilon values."""
        config = QLoRAConfig(use_adversarial=True, adversarial_epsilon=10.0)
        
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.randn(2, 100)
        mock_model.return_value = mock_outputs
        
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        trainer.tokenizer = MagicMock()
        trainer.tokenizer.vocab_size = 100
        
        inputs = torch.randint(0, 50, (2, 10), dtype=torch.long)
        labels = torch.randint(0, 100, (2,))
        
        with patch.object(trainer.model, 'forward', return_value=mock_outputs):
            adv_inputs = trainer._fgsm_attack(inputs, labels)
        
        # Should still be clipped to valid range
        assert torch.all(adv_inputs >= 0)
        assert torch.all(adv_inputs < 100)

    def test_adversarial_with_zero_epsilon(self):
        """Test that zero epsilon is handled properly."""
        config = QLoRAConfig(use_adversarial=True, adversarial_epsilon=0.0)
        
        # This should raise an error during config validation
        with pytest.raises(ValueError, match="adversarial_epsilon must be positive"):
            QLoRAConfig(use_adversarial=True, adversarial_epsilon=0.0)

    def test_adversarial_model_not_ready(self):
        """Test adversarial methods when model is not properly set up."""
        config = QLoRAConfig(use_adversarial=True)
        trainer = QLoRATrainer(config)
        trainer.model = None
        trainer.tokenizer = None
        
        inputs = torch.randint(0, 50, (2, 10), dtype=torch.long)
        labels = torch.randint(0, 100, (2,))
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises(AttributeError):
            trainer._fgsm_attack(inputs, labels)


@pytest.fixture
def sample_adversarial_config():
    """Fixture for standard adversarial configuration."""
    return QLoRAConfig(
        use_adversarial=True,
        adversarial_weight=0.1,
        adversarial_epsilon=0.01,
        adversarial_attack="fgsm",
        base_model="test-model",
        output_dir="/tmp/test-output"
    )


@pytest.fixture
def mock_adversarial_model():
    """Fixture for mock model for adversarial testing."""
    mock_model = MagicMock()
    mock_model.training = True
    
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.randn(2, 100, requires_grad=True)
    mock_model.return_value = mock_outputs
    
    return mock_model

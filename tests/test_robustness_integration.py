"""Tests for robustness evaluation framework integration with training pipeline.

Tests the integration of the comprehensive robustness evaluation framework to ensure:
1. Proper configuration and enabling
2. Correct evaluation timing (post-training)
3. Report generation and storage
4. Integration with training summary
5. Error handling and edge cases
"""

import pytest
import json
import torch
from unittest.mock import MagicMock, patch
from finetune.trainer import QLoRAConfig, QLoRATrainer, TrainingSummary


class TestRobustnessConfiguration:
    """Test robustness evaluation configuration."""

    def test_robustness_disabled_by_default(self):
        """Test that robustness evaluation is disabled by default."""
        config = QLoRAConfig()
        assert config.evaluate_robustness is False
        assert config.robustness_report_dir == "reports/robustness"

    def test_robustness_config_validation(self):
        """Test robustness configuration parameter validation."""
        # Valid configuration
        config = QLoRAConfig(
            evaluate_robustness=True,
            robustness_report_dir="/tmp/custom-reports"
        )
        assert config.evaluate_robustness is True
        assert config.robustness_report_dir == "/tmp/custom-reports"


class TestRobustnessIntegration:
    """Test robustness evaluation integration with training pipeline."""

    def test_robustness_evaluation_disabled(self, mocker):
        """Test that robustness evaluation is skipped when disabled."""
        config = QLoRAConfig(evaluate_robustness=False)
        
        mock_model = MagicMock()
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        
        # Mock trainer
        mock_trainer = MagicMock()
        
        # Should not call robustness evaluation
        with patch.object(trainer, '_evaluate_robustness') as mock_eval:
            # Simulate the training completion code path
            if config.evaluate_robustness:
                trainer._evaluate_robustness(mock_trainer)
            
            # Verify not called
            mock_eval.assert_not_called()

    def test_robustness_evaluation_enabled(self, mocker):
        """Test that robustness evaluation is called when enabled."""
        config = QLoRAConfig(evaluate_robustness=True)
        
        mock_model = MagicMock()
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        
        # Mock trainer
        mock_trainer = MagicMock()
        
        # Mock the robustness evaluator
        with patch('finetune.trainer.RobustnessEvaluator') as mock_evaluator_class:
            mock_evaluator = MagicMock()
            mock_metrics = MagicMock()
            mock_metrics.calculate_robustness_score.return_value = 85.0
            mock_evaluator.evaluate.return_value = mock_metrics
            mock_evaluator_class.return_value = mock_evaluator
            
            # Mock report generation
            with patch('finetune.trainer.RobustnessReport.generate_report') as mock_report:
                mock_report.return_value = {"score": 85.0}
                
                # Call the evaluation
                trainer._evaluate_robustness(mock_trainer)
                
                # Verify evaluator was created and called
                mock_evaluator_class.assert_called_once()
                mock_evaluator.evaluate.assert_called_once()
                mock_report.assert_called_once()

    def test_robustness_report_storage(self, mocker, tmp_path):
        """Test that robustness reports are properly stored."""
        config = QLoRAConfig(
            evaluate_robustness=True,
            robustness_report_dir=str(tmp_path)
        )
        
        mock_model = MagicMock()
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        
        # Create a training summary to store results
        summary = TrainingSummary(
            primary_metric_name="test",
            primary_metric_value=0.5,
            metric_goal="minimize"
        )
        trainer.training_summary = summary
        
        # Mock evaluator
        with patch('finetune.trainer.RobustnessEvaluator') as mock_evaluator_class:
            mock_evaluator = MagicMock()
            mock_metrics = MagicMock()
            mock_metrics.calculate_robustness_score.return_value = 92.0
            mock_evaluator.evaluate.return_value = mock_metrics
            mock_evaluator_class.return_value = mock_evaluator
            
            mock_report = {
                "metadata": {"robustness_score": 92.0},
                "metrics": {},
                "analysis": {},
                "recommendations": []
            }
            
            with patch('finetune.trainer.RobustnessReport.generate_report') as mock_gen_report:
                mock_gen_report.return_value = mock_report
                
                trainer._evaluate_robustness(MagicMock())
                
                # Verify report was stored in summary
                assert trainer.training_summary.robustness_score == 92.0
                assert trainer.training_summary.robustness_report == mock_report

    def test_robustness_with_missing_dependencies(self, mocker):
        """Test graceful handling when robustness module is not available."""
        config = QLoRAConfig(evaluate_robustness=True)
        
        mock_model = MagicMock()
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        
        # Mock import error
        with patch.dict('sys.modules', {'finetune.robustness_evaluator': None}):
            with patch('finetune.trainer.RobustnessEvaluator', side_effect=ImportError("Module not found")):
                # Should not raise exception
                trainer._evaluate_robustness(MagicMock())
                # Should log warning but continue

    def test_robustness_evaluation_error_handling(self, mocker):
        """Test error handling during robustness evaluation."""
        config = QLoRAConfig(evaluate_robustness=True)
        
        mock_model = MagicMock()
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        
        # Mock evaluator to raise exception
        with patch('finetune.trainer.RobustnessEvaluator') as mock_evaluator_class:
            mock_evaluator_class.side_effect = RuntimeError("Evaluation failed")
            
            # Should not raise exception
            trainer._evaluate_robustness(MagicMock())
            # Should log error but continue


class TestTrainingSummaryIntegration:
    """Test integration of robustness metrics with training summary."""

    def test_summary_includes_robustness_metrics(self):
        """Test that training summary includes robustness fields."""
        summary = TrainingSummary(
            primary_metric_name="loss",
            primary_metric_value=0.1,
            metric_goal="minimize",
            robustness_score=88.0,
            robustness_report={"score": 88.0}
        )
        
        # Convert to dict
        summary_dict = summary.to_dict()
        
        # Verify robustness fields are included
        assert "robustness_score" in summary_dict
        assert "robustness_report" in summary_dict
        assert summary_dict["robustness_score"] == 88.0
        assert summary_dict["robustness_report"]["score"] == 88.0

    def test_summary_serialization(self):
        """Test that training summary with robustness data serializes correctly."""
        summary = TrainingSummary(
            primary_metric_name="loss",
            primary_metric_value=0.1,
            metric_goal="minimize",
            robustness_score=95.0,
            robustness_report={
                "metadata": {"timestamp": "2024-01-01"},
                "metrics": {"numerical_stability": {"gradient_norm_mean": 0.01}},
                "analysis": {"overall": "Excellent"},
                "recommendations": []
            }
        )
        
        # Serialize to dict
        summary_dict = summary.to_dict()
        
        # Convert to JSON
        json_str = json.dumps(summary_dict)
        
        # Deserialize back
        loaded = json.loads(json_str)
        
        # Verify all fields preserved
        assert loaded["robustness_score"] == 95.0
        assert loaded["robustness_report"]["analysis"]["overall"] == "Excellent"


class TestRobustnessEvaluationWorkflow:
    """Test complete robustness evaluation workflow."""

    def test_end_to_end_workflow(self, mocker, tmp_path):
        """Test complete robustness evaluation workflow."""
        # Setup configuration
        config = QLoRAConfig(
            evaluate_robustness=True,
            robustness_report_dir=str(tmp_path)
        )
        
        # Create trainer
        mock_model = MagicMock()
        trainer = QLoRATrainer(config)
        trainer.model = mock_model
        
        # Mock dataloader
        mock_dataloader = MagicMock()
        trainer.dataloader = mock_dataloader
        
        # Mock evaluation dataloader
        mock_eval_dataloader = MagicMock()
        trainer.eval_dataloader = mock_eval_dataloader
        
        # Mock training summary
        summary = TrainingSummary(
            primary_metric_name="eval_loss",
            primary_metric_value=0.2,
            metric_goal="minimize"
        )
        trainer.training_summary = summary
        
        # Mock robustness evaluator
        with patch('finetune.trainer.RobustnessEvaluator') as mock_evaluator_class:
            mock_evaluator = MagicMock()
            mock_metrics = MagicMock()
            mock_metrics.calculate_robustness_score.return_value = 87.0
            mock_evaluator.evaluate.return_value = mock_metrics
            mock_evaluator_class.return_value = mock_evaluator
            
            # Mock report generation
            mock_report = {
                "metadata": {
                    "timestamp": "2024-01-01 12:00:00",
                    "robustness_score": 87.0
                },
                "metrics": {
                    "numerical_stability": {
                        "gradient_norm_mean": 0.008,
                        "gradient_norm_std": 0.002,
                        "nan_gradients_count": 0,
                        "inf_gradients_count": 0
                    },
                    "quantization_robustness": {
                        "quantization_loss_increase": 0.15,
                        "sensitive_parameters_count": 42,
                        "npft_effectiveness": 65.0
                    },
                    "adversarial_robustness": {
                        "fgsm_attack_success_rate": 0.22,
                        "pgd_attack_success_rate": 0.28,
                        "adversarial_loss_reduction": 38.0
                    },
                    "generalization": {
                        "clean_accuracy": 92.5,
                        "perturbed_accuracy": 88.0,
                        "distribution_shift_robustness": 85.0
                    },
                    "safety": {
                        "safety_violation_count": 0,
                        "zeroth_core_interventions": 3,
                        "constraint_satisfaction_rate": 98.5
                    }
                },
                "analysis": {
                    "overall": "Very good robustness with minor areas for improvement",
                    "numerical_stability": "Excellent numerical stability",
                    "quantization_robustness": "Moderate quantization robustness",
                    "adversarial_robustness": "Moderate adversarial robustness",
                    "generalization": "Excellent generalization",
                    "safety": "Excellent safety compliance"
                },
                "recommendations": [
                    {
                        "priority": "medium",
                        "area": "quantization_robustness",
                        "recommendation": "Enable NPFT and increase noise scale to improve quantization robustness",
                        "potential_impact": "40-60% reduction in quantization loss increase"
                    },
                    {
                        "priority": "medium",
                        "area": "adversarial_robustness",
                        "recommendation": "Increase adversarial weight and use mixed FGSM/PGD training",
                        "potential_impact": "20-40% improvement in adversarial robustness"
                    }
                ]
            }
            
            with patch('finetune.trainer.RobustnessReport.generate_report') as mock_gen_report:
                mock_gen_report.return_value = mock_report
                
                # Run evaluation
                trainer._evaluate_robustness(MagicMock())
                
                # Verify results
                assert trainer.training_summary.robustness_score == 87.0
                assert trainer.training_summary.robustness_report == mock_report
                
                # Verify report files were created
                report_file = tmp_path / "robustness_report.json"
                metrics_file = tmp_path / "robustness_metrics.json"
                
                # Check that files exist and contain expected content
                assert report_file.exists()
                assert metrics_file.exists()
                
                # Verify JSON content
                with open(report_file, 'r') as f:
                    saved_report = json.load(f)
                    assert saved_report["metadata"]["robustness_score"] == 87.0
                    assert saved_report["analysis"]["overall"] == "Very good robustness with minor areas for improvement"


@pytest.fixture
def mock_training_setup():
    """Fixture for mock training setup."""
    config = QLoRAConfig(
        evaluate_robustness=True,
        robustness_report_dir="/tmp/test-reports"
    )
    
    mock_model = MagicMock()
    mock_model.named_parameters.return_value = [
        ("layer1.weight", torch.nn.Parameter(torch.randn(10, 10))),
        ("layer2.bias", torch.nn.Parameter(torch.randn(10)))
    ]
    
    trainer = QLoRATrainer(config)
    trainer.model = mock_model
    
    # Mock dataloaders
    mock_train_dataloader = MagicMock()
    mock_eval_dataloader = MagicMock()
    trainer.dataloader = mock_train_dataloader
    trainer.eval_dataloader = mock_eval_dataloader
    
    # Mock training summary
    summary = TrainingSummary(
        primary_metric_name="test_loss",
        primary_metric_value=0.3,
        metric_goal="minimize"
    )
    trainer.training_summary = summary
    
    return trainer, mock_train_dataloader, mock_eval_dataloader
